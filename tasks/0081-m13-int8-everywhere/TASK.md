# 0081 — int8 across the catalogue, `tile_n = 64`, and the number that was wrong

**Goal (user, 2026-08-22):** *"Jeg tror vi gjør noe feil, det er ingen grunn til at dette ikke
skal gi større fordeler. Men jeg klarer ikke helt å si hva. … det som må løses er large. …
Les en ekstra gang gjennom dokumentene for å se om det er noe vi overser. … Målet nå er int8
på alle, og deretter research på mer speed."*

The instinct was right twice over. Re-reading CLAUDE.md's geometry section found a tile width
we had been leaving on the table for bge-large; re-reading the benchmark output found a
**reporting bug that overstated the biggest unexplained cost by 2×**. And the honest answer to
*"why not bigger?"* turns out to be structural: **69.6% of a bge-large encode is host work**,
so the array was never going to deliver more than it has.

---

## 1. What we were overlooking: `tile_n = 64` is an int8-only width

CLAUDE.md's own geometry section says it, and it reads as a closed question:

> `tile_n` is **48** for hidden 384/768, **32** for bge-large, whose N ∈ {1024, 3072, 4096}
> makes 48 illegal and **whose 64 would need 65,536 B of a 63 KB budget**.

That is exactly right — **for bf16**. Trap 3's budget is
`2·(m·k·in + k·n·in + m·n·out) < 64512`, and `in` halves for int8:

| tile | bf16 | int8 | N ∈ {1024, 3072, 4096} all divide `n·8`? |
|---|---:|---:|---|
| (64, 64, 32) — shipped | 40,960 ok | 28,672 ok | yes |
| (64, 64, 48) | 53,248 ok | 38,912 ok | **no** |
| **(64, 64, 64)** | **65,536 OVER** | **49,152 ok** | **yes — no padding needed** |

**bge-large has been running at half the tile width it can afford**, because the geometry was
inherited from a budget that stopped applying when the datapath changed. And unlike
[`0074`](../0074-m13-gemma-on-npu/TASK.md)'s Gemma padding, this costs nothing at all: 1024,
3072 and 4096 are all multiples of 64·8 = 512.

`tile_n` is the term that governs **A re-streaming** — A is fetched `N/(n·cols)` times — which
[`0080`](../0080-m13-int8-traffic-bound/TASK.md) showed is what the int8 datapath is bound by.
Measured on bge-large's four production shapes, M=8192, 8 columns, int8 with bf16 C:

| shape | A re-streams 32 → 64 | traffic MB | µs at 32 | µs at 64 |
|---|---|---:|---:|---:|
| qkv | 12 → 6 | 251.7 → 201.3 | 8687 | 6492 |
| attn_out | 4 → 2 | 83.9 → 67.1 | 4823 | 3380 |
| ffn_up | 16 → 8 | 335.5 → 268.4 | 11096 | 8450 |
| ffn_down | 4 → 2 | 285.2 → 218.1 | 11321 | 7970 |
| **sum** | | | **35,927** | **26,292 = 1.366×** |

**Bit-exact** against exact int32 accumulation followed by one RNE rounding (|acc|max 913,800,
well inside int32). Needed one new entry point, `narrow_4096_i32_bf16` — `m·n = 4096` had never
been reachable.

**Only bge-large benefits.** 64 needs every N divisible by 512, and hidden 384's {1152, 384,
1536} and hidden 768's {2304, 768, 3072} do not. This is the first time a model's tile
geometry has been decided by its *own* widths rather than by one project-wide default.

---

## 2. The number that was wrong: `everything else` was double-counting

The benchmark's residual bucket was computed as `wall − npu − attn`, which **does not subtract
the three host eltwise lines it prints directly above it.** GELU, softmax and LayerNorm were
counted twice. On bge-large that inflated the unexplained bucket from 12.9% to **24.4%** —
making it look like the single largest host cost and pointing optimisation at a phantom.

Fixed. This is the same failure mode as tasks/0042's `tile (64, 32)` banner and 0078's
layout-hash line: a status line reporting the intention rather than the value.

---

## 3. The honest answer to *"why not bigger?"*

bge-large, int8, `tile_n = 64`, bf16 C, single lane, 2 encodes of 128 sequences:

| | ms | % |
|---|---:|---:|
| dispatch + wait | 432.44 | **30.4%** |
| read out + dequantise | 260.28 | 18.3% |
| host attention (QK^T, A·V) | 207.96 | 14.6% |
| everything else (residual adds, pooling) | 183.50 | 12.9% |
| quantise A | 140.76 | 9.9% |
| host GELU | 108.79 | 7.6% |
| host LayerNorm | 37.00 | 2.6% |
| sync to/from device | 34.34 | 2.4% |
| host softmax | 19.39 | 1.4% |

**The array is 30.4% of the encode. Even making the NPU infinitely fast caps the whole thing
at 1.44×.** That is the answer, and no amount of kernel work changes it.

**And 69.6% is host work that is almost entirely memory traffic, not arithmetic.** At batch
128 / seq 64 / hidden 1024 one activation tensor is 33.5 MB and the FFN intermediate is
134 MB, and the host walks them one pass at a time:

```
GEMM -> dequantise C -> add residual -> LayerNorm -> quantise -> GEMM
        (read+write)    (2 reads,1 wr)  (2 passes)   (read+write)
```

Four to six separate streaming passes over the same tensor between two GEMMs, ≈17 GB of host
memory traffic per encode. `cpu 14710 ms / wall 1465 ms = 10.04 cores busy` — the host is
already threaded and already saturated; it is bandwidth, not cores.

**So the next lever is fusion on the host side, and it is worth far more than anything left on
the array.** See §6 — and [`0082`](../0082-m13-fused-ffn-epilogue/TASK.md) went and did it,
for **1.39–1.48×** on top of everything measured here.

---

## 4. int8 across the catalogue — all six, five passing

| model | arch | int8 `1-cos` | gate | notes |
|---|---|---:|---|---|
| `all-MiniLM-L6-v2` | 0 | **1.161e-03** | PASS | MTEB mean −0.04 (0080) |
| `bge-small-en-v1.5` | 0 | **6.385e-04** | PASS | reuses MiniLM's design set |
| `bge-base-en-v1.5` | 0 | **1.778e-03** | PASS | tight |
| `bge-large-en-v1.5` | 0 | **2.968e-03** | **1-cos FAIL, MTEB PASS** | §5 |
| `nomic-embed-text-v1.5` | **2** | **1.098e-03** | PASS | needed its own oracle |
| `embeddinggemma-300m` | **1** | **1.902e-03** † | PASS | needed a runtime path too |

† EmbeddingGemma has no HuggingFace golden fixture (CLAUDE.md: `arch=1`'s gate is
differential), so this is `tools/verify_gemma_npu_encode.py` against the **bf16 NPU** encode
of the same container — both sides through the same encoder, which isolates quantisation
exactly. `nearest-is-self 64/64`. Tight against the 2e-03 gate, and worth saying that the
corpus is `0061`'s **byte-fallback stress corpus** — deliberately hard tokenisation, chosen
because it was the only one on disk with 64 distinct sentences.

**The corpus mattered, and the verifier caught it.** `corpus_520.txt` — the obvious choice,
and what 0074 used — turns out to be **13 distinct sentences repeated 40×** to reach a batch
size. Every routing check against it is vacuous, and `verify_gemma_npu_encode.py` refused to
certify rather than printing PASS: *"a corpus that cannot separate its own rows cannot detect
a routing bug"*. That assertion was written in 0074 and earned its keep here.

Note the shape of the error across the four BERT models: it tracks **width** more than depth.
bge-small is 12 layers at hidden 384 and lands at 6.4e-04; bge-large is 24 layers at hidden
1024 and lands 4.6× worse. Wider weight matrices have more per-column outlier range for
SmoothQuant to move, and moving it costs the activations.

### The oracle's GEMM count is not the packer's, and the difference is fusion

Each architecture's calibration oracle runs the model as the **checkpoint** stores it; the
packer **concatenates** operands along N so the array sees four GEMMs per layer whatever the
architecture. The two therefore disagree, differently per arch:

| arch | oracle sites/layer | packer | what fuses |
|---|---:|---:|---|
| 0 (BERT) | 4 | 4 | qkv already fused upstream |
| **2 (nomic)** | **5** | 4 | `fc11` (up) + `fc12` (gate) → `ffn_up` |
| **1 (Gemma)** | **7** | 4 | `q`+`k`+`v`(+pad) → `qkv`, `gate`+`up` → `ffn_up` |

Left unhandled the keys shift and every factor from that point lands on the **next** tensor —
it packs cleanly and is wrong from layer 0, the same class of bug as
[`0078`](../0078-m13-int8-accuracy/TASK.md) §4b's 400× blowup, which was caught only because
the number was too implausible to accept. **Here an assert written before it was needed
caught both**, refusing rather than shifting silently.

Merging is exact: a fused group's members all consume the **same** activation, so `amax` is
identical across them and the fused operand's per-input-channel weight maximum is the
elementwise max of the members'. Taking a max also makes the result independent of visit
order.

Gemma needed one more exclusion: its `dense2`/`dense3` post-pool heads are wide enough to
pass the "is this an NPU GEMM" filter but the packer deliberately keeps them on the host (they
run once per *sequence*, not per token). They are last in encode order, so they are excluded
by position — 170 sites seen, 168 wanted.

### nomic (arch = 2) needed its own calibration oracle

`calibrate_smoothing` ran `reference/encoder.py` — BERT's forward pass — unconditionally. For
nomic that would have produced statistics for a model with absolute positions and a plain
GELU FFN, describing activations the array never sees. Now arch-selected, using
`reference/encoder_nomic.py` and `reference/encoder_gemma.py`.

### EmbeddingGemma needed a *runtime* path, not just a packer one

`GemmaNpuEncoder` is a **separate encoder** from `Encoder` (arch=1 needs RMSNorm ×4, MQA,
per-layer RoPE and GeGLU). It handled `c_elem_bytes` 2 and 4 and had no notion of quantised
operands at all — no per-row absmax on the way in, no `wscale`/`asmooth` on the way out.

Rather than copy ~120 lines of hand-vectorised quantisation into it, the quantise and
dequantise passes were **extracted into free functions both encoders call**
(`quantise_a_int8`, `dequantise_c`). The duplicate would have drifted: the reciprocal hoist
inside the quantiser was a 63 ms fix found once (0080), and a second copy would not have had
it. **Regression check after the extraction** — all four gates reproduce their numbers
exactly: MiniLM int8 int32-C 1.178e-03, MiniLM int8 bf16-C 1.161e-03, nomic int8 1.098e-03,
MiniLM bf16 1.086e-05.

Gemma's padded qkv turned out to need nothing special: `gemma_qkv_blocks()` appends genuinely
all-zero columns, and `add_gemm_b_int8` already keeps a zero column's scale at 1 rather than
dividing by zero — so a padded column stays exactly zero through quantisation, which is what
the host slicing by offset assumes.

### Three status lines that reported the intention rather than the value

All found while packing, all the same shape as tasks/0042's `tile (64, 32)`:

* `pack_nomic` printed the **bf16** layout hash over int8 tensors — dtype is part of the
  layout, so the hash was simply of a different thing.
* `pack_gemma` had its own copy of the same line, with the same bug.
* The Gemma banner printed `staged N MB of tiled **bf16** weights` the moment arch=1 got an
  int8 path. Now read off `a_elem_bytes`.

---

## 5. bge-large: better, still failing, and MTEB is the decider

| container | α | `tile_n` | worst `1-cos` |
|---|---:|---:|---:|
| shipped before this task | 0.3 | 32 | 4.475e-03 |
| 0079's best sweep point | 0.5 | 32 | 3.127e-03 |
| **this task** | **0.5** | **64** | **2.968e-03** |

The **best bge-large int8 number yet**, and still 1.48× outside the 2e-03 gate. Worth being
precise about what changed: α=0.3 → 0.5 is the accuracy move; `tile_n` 32 → 64 is a *speed*
move that happens to be numerically free (bit-exact accumulation either way), and the residual
1.05× improvement between the last two rows is the bf16-C narrowing from 0080, not the tile.

**MTEB settles it: PASS.** Five tasks, both sides one session, against the same checkpoint:

| task | CPU (fp32) | NPU (int8) | delta |
|---|---:|---:|---:|
| STSBenchmark | 87.52 | 87.53 | **+0.01** |
| SICK-R | 81.68 | 81.67 | −0.01 |
| STS12 | 79.05 | 78.87 | −0.18 |
| Banking77Classification | 84.82 | 84.80 | −0.02 |
| TwentyNewsgroupsClustering | 51.10 | 51.07 | −0.03 |
| **mean** | | | **−0.05** |

Gate is `|mean| ≤ 0.5` and no task worse than −0.5. **PASS**, worst task −0.18.

**So bge-large int8 fails `1-cos` at 2.968e-03 and is indistinguishable on the measure that
describes what embeddings are for.** That is [`0035`](../0035-m8-mteb-gate/TASK.md)'s whole
point, playing out for the third time — bfp16 failed `1-cos` at 3.47e-03, MiniLM int8 sat at
1.4e-03 and cost 0.03 MTEB points, and now a model 1.48× outside the fidelity gate loses
0.05. **`1-cos` is a fidelity check on the arithmetic; it is not a quality gate**, and this
task is the sharpest example the project has produced of the difference.

Worth stating what does *not* follow: this is not a licence to ignore `1-cos`. It caught every
real bug in 0077–0081 — the BOM row, the double prefix, the harness bugs — precisely because
it is sensitive to things MTEB averages away. The two measure different things and the project
needs both.

---

## 6. Where the speed research goes next, in priority order

From §3, and this is the whole point of the breakdown:

1. **Fuse the GEMM epilogue chain on the host.** Between two GEMMs the host does dequantise →
   residual → LayerNorm → quantise as four separate streaming passes. They can be **one**: a
   row is 4 KB and fits L1, LN already reduces over exactly that row, and it can write *two*
   outputs (fp32 for the residual, int8 for the next GEMM). One read and two writes instead of
   four of each. This is 0079 §3's idea generalised from LN alone to the whole chain, and §3
   prices it at a far larger share than that task assumed.
2. **Re-examine "eltwise on the host" itself.** [`0032`](../0032-m7-one-xclbin-production/TASK.md)
   measured every eltwise kernel as faster *and* more accurate on the host — **when each NPU
   dispatch cost a design switch**. There is now ONE xclbin in ONE hw_context, the switch is
   gone, and the array sits idle for 70% of an encode. `epilogue="gelu"` already exists in
   `gemm_pretiled.py`; fusing GELU into `ffn_up`'s epilogue would delete a dequantise, a GELU
   and a quantise pass over a **134 MB** tensor per layer.
3. **Host attention, 14.6%** — untouched by everything so far and now the third-largest item.
4. **Lane overlap** — int8 lanes buy 1.11× against bf16's 1.43%, because a smaller NPU share
   leaves less host work to hide behind it.

---

## 7. Commands

```powershell
. C:\dev\mlir-aie\iron_env.ps1

# the int8-only tile width, bge-large
python tools\export_gemm_rtp.py --int8 --c-bf16 --out runtime\artifacts_int8c_large_n64 `
    --hidden 1024 --intermediate 4096 --cols 8 --batches 128 -n 64
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\bge-large-en-v1.5 `
    --out models\bge-large-en-v1.5.int8n64.npue --int8 --tile-n 64 --smooth-alpha 0.5

# the rest of the catalogue
python tools\export_gemm_rtp.py --int8 --c-bf16 --out runtime\artifacts_int8c_base `
    --hidden 768 --intermediate 3072 --cols 8 --batches 128
python tools\export_gemm_rtp.py --int8 --c-bf16 --out runtime\artifacts_int8c_nomic `
    --hidden 768 --intermediate 3072 --gated-ffn --cols 8 --batches 128
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\bge-small-en-v1.5 `
    --out models\bge-small-en-v1.5.int8.npue --int8
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\bge-base-en-v1.5 `
    --out models\bge-base-en-v1.5.int8.npue --int8
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\nomic-embed-text-v1.5 `
    --out models\nomic-embed-text-v1.5.int8.npue --int8

# where the time actually goes
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5.int8n64 `
    --artifacts artifacts_int8c_large_n64 --threads 24 --bench 2
```
