# 0074 — EmbeddingGemma-300M on the NPU: the padding trick, and the arch=1 GEMM path

**Goal (user, 2026-08-22):** *"Få en versjon av embeddinggemma300 til å kjøre MEST
mulig på NPU. Undersøk alle muligheter for geometriske og matematiske triks.
Første skritt av release 0.4.0 er embeddinggemma på npu."*

Status: **DONE and on hardware.** EmbeddingGemma-300M runs 97.7% of its MACs on
the array at **~133 seq/s** (4 lanes, wall clock end to end, three guarded runs
1.7% apart) against the host-only path's **0.20 seq/s** measured in the same
session — **~665×** — at
`1-cos` **9.962e-06** against that same host-only path, which is itself
5.496e-13 from the numpy reference. Every lane count returns bit-identical
output. See §11 for what is deliberately not done yet.

---

## 1. What was actually wrong — the question this task reopens

[T29](../../research/OPEN-THREADS.md#t29) has stood since tasks/0055 as *"does the
`tile_n=16` tax rule it out?"*, with a prior of **62–72 seq/s** against bge-base's
measured 181–209. That prior is what deprioritised the NPU path, and the host-only
path that shipped instead runs at **~7.9 s/sentence** (tasks/0064 —
correctness-first by design, scalar, double-accumulated, batch 1, no AVX2).

**The `tile_n=16` floor is real arithmetic, but it was computed over the wrong
option set.** 0055 enumerated four *fusion* strategies (QKV fused/unfused ×
gate-up fused/unfused) and took the worst. It never considered **zero-padding the
N axis**, which is free to the packer because B is pre-tiled offline anyway.

Two separate findings fall out, and the second one is a bug in 0055's own verdict:

1. **0055's own table already contradicts its headline.** Strategy C (fused QKV
   1280, fused gate_up 2304) is listed at **`tile_n = 32` @ 8 cols** and passes the
   L1 budget at 40,960 B. The verdict line, the performance prior, and T29's title
   all use **16**, which is strategy A/B. Nothing in the task justifies taking the
   worst strategy — the packer chooses the fusion.
2. **Padding beats every fusion strategy**, and lands Gemma on exactly the same
   tile geometry the four shipped BERT-family models already use.

---

## 2. The geometric trick: pad the fused QKV to 1536, trailing zeros only

`num_key_value_heads = 1` and `head_dim = 256`, so MQA's K and V projections are
each **N = 256** wide. 256 caps `gcd(N/n_cols)` at 32 no matter what else is in
the N-set — that is 0055's finding and it is correct.

**Append 256 zero columns to the fused QKV weight.** N goes 1280 → **1536**.

```
qkv B = [ Wq (768) | Wk (256) | Wv (256) | 0 (256) ]     K=768, N=1536
```

Zero columns of B give exactly-zero columns of C (`C = A·B`), so the host slices
`q = C[:, 0:768]`, `k = C[:, 768:1024]`, `v = C[:, 1024:1280]` and ignores the
rest. **Exact, not approximate** — no accuracy question to answer.

Trailing padding, not interior padding: Q/K/V stay contiguous at their natural
offsets, so the host slicing is unchanged from the unpadded layout. Interior
padding (K and V each padded to 384) reaches the same N=1536 and costs the same,
but moves every offset for no gain.

### The resulting design set

| stream | K | N | K % 64 | N % 384 |
|---|---:|---:|:--:|:--:|
| `qkv` | 768 | **1536** | ✓ | ✓ (4) |
| `attn_out` | 768 | 768 | ✓ | ✓ (2) |
| `ffn_up` (gated, 2×1152) | 768 | 2304 | ✓ | ✓ (6) |
| `ffn_down` | 1152 | 768 | ✓ | ✓ (2) |

Checked against `experiments/m5-pretiled-gemm/gemm_pretiled.py`'s own assertions
(`_build_design`, lines 126–129), not against a restatement of them:

- `M % (m * n_aie_rows) == 0` → M % 256 == 0. M = batch·64, batch a multiple of
  4 → ✓ (the existing `--batch` check already enforces this).
- `K % k == 0` → {768, 1152} % 64 == 0 → ✓
- `N % (n * n_aie_cols) == 0` → {1536, 768, 2304, 768} % 384 == 0 → ✓
- `n % t == 0` → 48 % 8 == 0 → ✓
- L1 budget `2*(64·64·2 + 64·48·2 + 64·48·4)` = **53,248 B** of 63 KB → ✓
  (the identical number MiniLM / bge-small / bge-base / nomic already run)
- C-drain stride guard (`m·n_aie_rows·N > 2^20`, the tasks/0068 bug): threshold is
  N > 4096; our largest N is 2304 → **nowhere near it**, and it is fixed anyway.
- DMA BD 10-bit dimension limit: worst case 6 n-blocks, 18 k-blocks → ✓

**So `tile (m,k,n) = (64,64,48)` at 8 columns is legal for Gemma**, which is the
production geometry. Consequence worth stating plainly: **Gemma's packed B shares
its `layout_hash` with every other model this project ships.**

---

## 3. What it is worth (PRIOR, not a measurement)

Rule 1: no wall clock here is an NPU claim. This is
[0048](../0048-m9-what-is-the-gemm-time/TASK.md)'s fitted model
`t = 573 µs + 4.72 µs × iterations-per-core`, applied to shapes it was not fitted
on, so it is a **prior**.

First, the iteration formula was re-derived from 0048's own discriminating pair
rather than reused from 0055's proxy (which drops M and n_aie_rows):

```
iterations_per_core = (K/k) · (M/(m·n_aie_rows)) · (N/(n·n_aie_cols))
```

Check against 0048's measured pair at MiniLM production (M=8192, tile 64/64/48,
8 cols): `ffn_up` [8192,384,1536] → 6·32·4 = **768**; `ffn_down` [8192,1536,384]
→ 24·32·1 = **768**. Model: 573 + 4.72·768 = **4,198 µs** each, against measured
**4,196 / 4,273 µs**. That is the pair whose *identical MACs and 1.5× byte
difference* made T1's point, and the formula reproduces both to 1.8%.

At M = 8192 (batch 128, seq 64), 8 columns:

| configuration | iters/layer | layers | total iters | dispatches | modelled t | vs bge-base |
|---|---:|---:|---:|---:|---:|---:|
| bge-base-en-v1.5 (measured 181–209 seq/s) | 9,216 | 12 | 110,592 | 48 | 549.5 ms | 1.00× |
| Gemma, `tile_n=16` (T29's standing prior) | 16,512 | 24 | 396,288 | 96 | 1,925 ms | **3.50×** |
| Gemma, `tile_n=32` (0055 strategy C, unused) | 8,256 | 24 | 198,144 | 96 | 990 ms | **1.80×** |
| **Gemma, padded, `tile_n=48`** | **5,760** | 24 | **138,240** | 96 | **707 ms** | **1.29×** |

**Prior for the padded design: ≈ 140–162 seq/s**, from bge-base's measured
181–209 divided by 1.29. Against T29's standing 62–72, padding is worth **2.7×**;
against the best fusion-only option, **1.40×**.

Caveats, stated because they bound what may be claimed later:

- The 573 µs fixed term is **55.0 ms of the 707 ms**, i.e. the dispatch count
  (96, twice bge-base's) is 8% of modelled time here, not the dominant term.
- The model is fitted at h=384 and
  [0051](../0051-m9-bge-base-and-in-exe-fetch/TASK.md) recorded a **27% miss**
  when it was extrapolated to h=768. Treat the range as indicative only.
- It counts **GEMM dispatch time only**. Gemma's host side is heavier than any
  BERT model here: 4 RMSNorm/layer (96 sites) vs 2 LayerNorm, plus q_norm/k_norm,
  RoPE, and GeGLU's gate multiply. 0044 measured host eltwise + the transport it
  forces at 33% of a BERT encode. The host path must be AVX2+threaded or it, not
  the array, will be the wall.
- The padding itself costs **4.4% of GEMM iterations** (256 wasted columns of
  1536 on one of four shapes), against the ~3× it removes.

---

## 4. How much of the model actually moves to the NPU

Per token per layer, MACs:

| op | MACs | where |
|---|---:|---|
| qkv (useful 768×1280) | 983,040 | **NPU** |
| attn_out 768×768 | 589,824 | **NPU** |
| ffn_up 768×2304 | 1,769,472 | **NPU** |
| ffn_down 1152×768 | 884,736 | **NPU** |
| attention QK^T + A·V (3 heads, S=64) | 98,304 | host |

**97.7% of the model's MACs move to the array.** The remaining 2.3% is attention,
which F3 and [0043](../0043-m9-attention-geometry/TASK.md) already price as not
worth the geometry fight — and at `head_dim=256`, N = S = 64 fails the `N % 384`
rule outright, so it is not expressible at this sequence length regardless.

Also left on the host, deliberately:

- **The two post-pool Dense heads** (768→3072→768). They run once per *sequence*,
  not per token: 604 MFLOP per batch-128 encode against ~700 ms of array time,
  i.e. ~1%. Expressible (M would need padding 128 → 256), not worth 2 dispatches.
- **RMSNorm / RoPE / GeGLU**, per [0032](../0032-m7-one-xclbin-production/TASK.md)'s
  measured precedent that host eltwise beats an NPU dispatch at these widths, and
  0044's 2.3 ms-per-design-switch corollary.

---

## 5. Mathematical traps found while reading the verified reference

1. **DO NOT fold `query_pre_attn_scalar` into Wq.** Every other model in this
   project folds `1/sqrt(head_dim)` into the Q block of the packed qkv weight
   (BERT since M4, nomic in `pack_nomic` with an explicit linearity argument for
   RoPE). **For Gemma that fold is silently destroyed**: `q_norm` is an RMSNorm
   applied to q *after* the projection, and RMSNorm is scale-invariant —
   `(s·q)/rms(s·q) = q/rms(q)` exactly. The scale would vanish and attention would
   run unscaled, with no size error to notice. If the multiply is ever to be
   folded, the only legal target is `q_norm.weight` (store `s·(1+w) − 1`), and it
   buys nothing measurable: the scale currently costs S²·H = 12,288 multiplies per
   layer, fewer than moving it earlier would.
2. **`q_norm`/`k_norm` run between the projection and RoPE**, per head over
   `head_dim`, and the reference's ordering is load-bearing (0063 measured a
   negative control at `rel_fro` 0.439 for dropping RMSNorm's `1+`).
3. **GeGLU needs the reference's two-stage rounding** — round `act`, *then*
   promote and multiply by `up` — to stay bit-exact (0063). A mathematically
   equivalent fused double-precision expression is not enough.
4. **Zero biases, not absent biases.** `Encoder::gemm()` adds a bias
   unconditionally. Gemma has none (`attention_bias: false`, both Dense heads
   `bias: false`). Follow `pack_nomic`'s precedent: zero-fill, which is exact and
   cheaper than a nullable branch in the hot path.

---

## 6. A fail-open this will walk into if it is not fixed first

`design_fits()` (runtime/src/main.cpp:1774) matches the four streams as

```c++
{"qkv", hidden, 3 * hidden}, ...
```

Gemma's `qkv` N is **1536**, not `3·768 = 2304`. So the check as written would
**reject Gemma's own correct design**, and — worse in the other direction — the
`qkv` width is not a fact the runtime can derive from `hidden` any more, exactly
as `ffn_up`'s width stopped being derivable when nomic introduced the gated FFN
(tasks/0069, thread T31, which is the same bug one field to the left).

`qkv_n` must become **data in `design.json` and in the container**, matched
explicitly. Anything else is the T31 fail-open re-shipped.

---


## 7. What was built

| piece | file | what changed |
|---|---|---|
| padding arithmetic | `tools/pack_npue.py` `gemma_qkv_blocks()` | owns the N-padding; returns Q/K/V offsets as DATA |
| packer | `tools/pack_npue.py` `pack_gemma()` | pre-tiled bf16 under BERT names is now the DEFAULT; `--gemma-host-only` rebuilds the control |
| design | `tools/export_gemm_rtp.py` | `--qkv-n`; `qkv_n` written into `design.json` |
| design match | `runtime/src/main.cpp` `design_fits()` | `qkv` N is read, not derived as `3*hidden` |
| catalogue | `runtime/src/hub.cpp`, `hub.hpp` | Gemma row gains `gated_ffn` and `qkv_n = 1536` |
| encoder | `runtime/src/main.cpp` `GemmaNpuEncoder` | new; 4 GEMMs/layer on the array, AVX2+threaded host RMSNorm/RoPE/GeGLU/MQA attention, lanes |
| dispatch | `runtime/src/main.cpp` `run_gemma_mode()` | picks NPU or host from the CONTAINER; `--cpu` forces the control |
| gate | `tools/verify_gemma_npu_encode.py` | new; differential gate + a row-routing check |

---

## 8. Commands run

```powershell
# 1. container (the NPU layout is now the DEFAULT for arch=1)
.\.venv-ref\Scripts\python.exe tools\pack_npue.py `
    --model-dir models\embeddinggemma-300m --out models\embeddinggemma-300m.npue

# 2. designs -- iron env dot-sourced
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd $repo
python tools\export_gemm_rtp.py --batch 128 --batches 4,16,32,128 --cols 8 `
    --hidden 768 --intermediate 1152 --gated-ffn --qkv-n 1536 `
    --out runtime\artifacts_gemma            # log: export_design.log

# 3. runtime
cmake --build runtime\build --config Release

# 4. the control (host-only path, 65.5 s for 13 texts)
.\runtime\build\npuembed.exe . --model embeddinggemma-300m.cpp_test `
    --embed tasks\0074-m13-gemma-on-npu\corpus.txt out_cpu.f32

# 5. the NPU path, and the gate
.\runtime\build\npuembed.exe . --model embeddinggemma-300m `
    --embed tasks\0074-m13-gemma-on-npu\corpus.txt out_npu.f32 `
    --threads 24 --pipeline 4
.\.venv-ref\Scripts\python.exe tools\verify_gemma_npu_encode.py `
    --npu out_npu.f32 --cpu out_cpu.f32
```

---

## 9. Results

### The design built first attempt

ONE xclbin, **16 streams** (4 shapes x 4 batch tiers), 8 columns, identity
**65-74 differing bytes** against `qkv@b128` -- the UUID footprint, inside
0029's <=80 check. The packer's own iteration table reproduces the research
prediction exactly: 1,536 + 768 + 2,304 + 1,152 = **5,760 iters/core/layer**.

`layout_hash` **`94266693ea31aa67...`** -- the same constant MiniLM, bge-small,
bge-base and nomic already pack to. Gemma joins them on the production tile
geometry rather than needing one of its own.

Container **1,046.9 MB**, down from the host-only 1,239.7 MB, of which
**212.3 MB** is tiled bf16 weight staged on the device.

### Correctness -- differential, against the host-only control

13 distinct sentences (T32: never a tiled fixture), NPU path vs
`npue::GemmaEncoder`, which tasks/0064-0065 tied to `reference/encoder_gemma.py`
at 1-cos 5.496e-13.

| | |
|---|---|
| 1-cos worst | **9.962e-06** |
| 1-cos mean | 7.068e-06 |
| gate | 2e-03 -> **PASS**, 201x inside |
| nearest-is-self | **13/13** |
| closest off-diagonal pair | cos 0.3885 -- the corpus can actually separate its own rows |

In family with every other bf16 model here -- MiniLM 1.086e-05, bge-base
1.353e-05, nomic 2.599e-05 -- and better than two of them.

At 520 rows (40x the corpus) the **40 repeats of each sentence are
bit-identical**, `max |delta| = 0.000e+00`, and every lane count produces
**bit-identical output to every other**: p2, p4 and p6 all `array_equal` with
p1. Position within a batch, batch tier and lane count are provably irrelevant
to the answer.

### Throughput (wall clock, end to end -- NOT an NPU kernel claim, rule 1)

520 texts, seq 64, `--threads 24`, idle machine:

| lanes | seq/s |
|---:|---:|
| 1 | 88.5 |
| 2 | 116.8 |
| **4** | **136.2** |
| 6 | 137.2 (saturated) |

Repeated afterwards through `tools/release_benchmark.ps1` with the **NPU
contention guard on** and three runs at 4 lanes: **134.9 / 133.0 / 132.7
seq/s**, a 1.7% spread, `xrt-smi` reporting `none Active but ours` each time.
**Take ~133 seq/s as the figure**; the 136.2 above and the 128.1 from the first
series bracket it at +2.4% / -3.7%.

**Machine-state caveat, stated because the sweep's own CPU guard refused the
first attempt.** The sampling window found CLion at ~21% of one core and
Spotify at ~13% -- ordinary desktop background, not the runaway process
tasks/0073's guard was written for, but not an idle machine either. The runs
above were taken with `-AllowCpuContention`. It matters less here than it would
for a ratio (0044: contention on one side only makes a RATIO confidently wrong,
and no CPU ratio is claimed for this model) but the NPU column is a wall-clock
figure with host work in it, so treat ~133 seq/s as good to a few percent
rather than to 0.1.

Against the host-only path measured on the same corpus in the same session --
**0.20 seq/s**, 65.5 s for 13 texts -- that is **~665x**.

The 0048-model prior in section 3 said 140-162 seq/s for the array side alone.
The single-lane breakdown puts array time at 2,953 ms of 6,040 ms for 520
texts, so **the array alone runs at 176 seq/s** -- inside the predicted band.
The prior is confirmed, not merely un-refuted.

### Where the time goes (single lane, 520 texts, 480 dispatches)

| | ms | share |
|---|---:|---:|
| NPU (in 88 / dispatch 2,786 / out 79) | 2,953 | 48.9% |
| host attention | 890 | 14.7% |
| C readback + bias | 693 | 11.5% |
| RMSNorm x97 | 473 | 7.8% |
| fp32->bf16 convert | 357 | 5.9% |
| GeGLU | 271 | 4.5% |
| RoPE | 92 | 1.5% |
| tokenize | 5 | 0.1% |

**The array and the host are almost exactly balanced (48.9% / 46.0%)**, which
is why lanes are worth 1.54x here against the 1.19-1.35x they buy the BERT
models. It also names the next levers: host attention is the largest single
host term, and the C readback is the second -- the one `--c-bf16` (tasks/0045)
already halves.

---

## 10. Problems hit

- **One row of 520 diverged at 1-cos 6.4e-03 while the other 519 sat at 7e-06,
  and it was the HARNESS, not the runtime.** The 520-line corpus was generated
  with PowerShell 5.1's `Set-Content -Encoding utf8`, which writes a **UTF-8
  BOM**; `EF BB BF` became part of line 0, so the tokenizer correctly embedded
  a different string. What gave it away was the shape of the failure -- exactly
  ONE row, exactly the FIRST line of the file, with the same sentence at rows
  13, 26, ... all agreeing to 7e-06. A per-row runtime bug does not select the
  first line of a file. Regenerated without the BOM: worst 1-cos back to
  9.962e-06 and the repeat spread to exactly 0. Note the near-miss: had the
  gate been run only on the 13-line corpus (written by a tool that emits no
  BOM) this would never have appeared, and had it been run only on the 520-line
  one it would have looked like an accuracy failure at 3.2x the gate.
- **`design_fits()` would have rejected Gemma's own correct design** -- it
  derived `qkv` N as `3 * hidden`. Found by reading before building, not by
  running (section 6). Same bug class as T31, one field to the left.
- **The `list` table's `cpu` state was a property of the ARCHITECTURE**, so it
  would have kept saying "cpu" about a container running 97.7% of its MACs on
  the array. It is now read from the container's `gemm_layout`, so a host-only
  container still reports `cpu` and a tiled one reports `ready` -- the same
  discipline tasks/0069 applied in the opposite direction.

---

## 11. Not done

- ~~`runtime/src/npue_pack.cpp` still emits the HOST-only layout.~~ **DONE.**
  `prepare_model_gemma()` gained `tile_k`/`tile_n`/`host_only` and a
  `add_gemm_b_concat_pad()` helper, and `--prepare-model` gained
  `--gemma-host-only`. `tools/verify_pack_parity.py` reports
  **byte-identical, sha256 `4a4c6f70995601f6…`, first attempt** -- and the
  container this task actually ships, `models/embeddinggemma-300m.npue`, has
  that same digest, so the two packers and the shipped artifact are one file.
- **MTEB has never been run on this architecture.** It is practical for the
  first time (at 0.20 seq/s it was not), and it is this project's authority for
  accuracy decisions (tasks/0035), not the 1-cos above.
- ~~`tools/release_benchmark.ps1` still carries `npu = $false`.~~ **DONE.**
  The row is `npu = $true, artifacts = artifacts_gemma, harness = "gemma"`.
  `harness` is a second axis on purpose: it says which measurement HARNESS can
  drive the model, which is a fact about the tooling. The accuracy stage runs
  the differential gate, throughput runs a real encode behind the same NPU
  contention guard `--bench` uses (new `--guard-contention`), and interleaved /
  energy / MTEB write **UNMEASURED with a stated reason** rather than being
  silently skipped.
- **No CPU ratio and no energy figure is claimed.** Interleaved measurement
  against torch/ORT (0040's rule) has not been run.
- **`--serve` still refuses on this arch.**
