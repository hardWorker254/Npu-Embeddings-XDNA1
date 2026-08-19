# 0042 — M9 Phase D1: bge-large, and the width prediction under test

**Goal.** Run `BAAI/bge-large-en-v1.5` — **hidden 1024, 24 layers, head_dim 64,
ffn 4096** — and test the claim [`0027`](../0027-m7-width-hypothesis/TASK.md)
made from a synthetic sweep: *the elementwise share falls as 1/h, so the NPU's
relative position should improve with model width.* Until now that had never
been checked against a real model.

Plan: `C:\Users\vegar\.claude\plans\ja-la-oss-kj-re-shimmering-dewdrop.md`.

---

## Headline: the prediction holds, and depth is the opposing force

**bge-large runs on the NPU on the first attempt** — `1-cos` **8.432e-06**
against HuggingFace, 604 MB of weights staged, 24 layers, head_dim 64 through
the templated attention path from [`0038`](../0038-m9-model-driven-runtime/TASK.md).

Interleaved against both CPU baselines, same statistic on every side
([`0040`](../0040-m9-honest-cpu-baseline/TASK.md)'s protocol):

| model | h | layers | NPU | torch | ORT | **NPU / strongest CPU** |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L6 | 384 | 6 | 877.0 | 489.4 | 234.3 | **1.792×** |
| bge-small | 384 | 12 | 444.6 | 290.0 | 134.0 | **1.533×** |
| **bge-large** | **1024** | **24** | **52.8** | **25.1** | **11.4** | **2.106×** |

Read as a two-factor experiment, because bge-small holds width fixed and
doubles depth while bge-large moves both:

- **Width helps us.** 384 → 1024 takes the ratio to 2.106×, the direction
  `0027` predicted, now on a real model instead of a synthetic sweep.
- **Depth hurts us.** Same width, twice the layers: **1.792 → 1.533**
  (0.856×). Our cost is per dispatch — 12 layers is 96 dispatches instead of
  48 — and the CPU has no such term.

Extrapolating the depth factor alone, 24 layers at h=384 would sit near 1.31×.
bge-large measures 2.106×, so **width bought about 1.6× against a 2.667×
increase in h** — the effect is real and it is smaller than a naive 1/h reading
of `0027` would suggest. Two reasons it should be: softmax scales with
`heads · seq²` rather than with `h` (16 heads, not 12), and the host
elementwise implementation is not the aie kernel `0027` measured.

**Per core it is stronger still.** 52.8 seq/s on ~6.0 cores against torch's
25.1 on twelve — **4.2× better per core**, up from 3.2× at MiniLM.

---

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

.venv-ref\Scripts\python.exe reference\fetch_model.py `
    --model BAAI/bge-large-en-v1.5 --layers 24
.venv-ref\Scripts\python.exe reference\make_goldens.py `
    --model-dir models\bge-large-en-v1.5 --taps

# does the machine even hold it? -- BEFORE building four xclbins at h=1024
runtime\build\npuembed.exe .. --model all-MiniLM-L6-v2 `
    --probe-bo artifacts_b128il/gemm_rtp 134.2 8

runtime\build\npuembed.exe .. --prepare-model ..\models\bge-large-en-v1.5 `
    ..\models\bge-large-en-v1.5.npue --tile-n 32     # from runtime\

cd C:\dev\mlir-aie; . .\iron_env.ps1; cd <repo>
python tools\export_gemm_rtp.py --hidden 1024 -n 32 --batch 128 `
    --out runtime\artifacts_large

python tools\export_validation.py --model bge-large-en-v1.5
.venv-ref\Scripts\python.exe tools\verify_npue.py --model bge-large-en-v1.5
runtime\build\npuembed.exe .. --model bge-large-en-v1.5 `
    --artifacts artifacts_large --threads 24 --pipeline 2
.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\compare_three.py `
    --model bge-large-en-v1.5 --artifacts artifacts_large --rounds 4
```

---

## Memory was not the blocker, and now we know the ceiling

The plan's own correction said to probe the ~800 MB XRT footprint **first**,
before spending a four-xclbin build. Measured budget at batch 128, two lanes:
**1023 MB** of XRT buffers (604 MB staged weights + 209 MB of A/B/C per lane)
plus ~738 MB of host scratch.

`--probe-bo` (new) allocates through the same path the encoder uses and
**touches** each buffer, because an untouched allocation may not have committed
pages and would report a ceiling that does not exist:

| probe | result |
|---|---|
| 8 × 134.2 MB = **1073.6 MB** | allocated in **0.10 s** |
| 60 × 134.2 MB = **8052.0 MB** | allocated in **1.76 s** |
| one **3000 MB** buffer | allocated in 0.41 s |

No ceiling anywhere near what bge-large needs. The concern is retired, and the
probe stays in the tree for the next width.

---

## `tile_n` stopped being a constant

`tile_n = 48` is illegal here. The design asserts `N % (tile_n · n_cols) == 0`
and bge-large's N are {1024, 3072, 4096}:

| N | legal `tile_n` at 8 columns |
|---:|---|
| 1024 | 8, 16, **32**, 64 |
| 3072 | 8, 16, 24, **32**, 48, 64 |
| 4096 | 8, 16, **32**, 64 |

**32** is the largest legal value that fits L1: the Stationary-C budget
`2(m·k·in + k·n·in + m·n·out)` gives **40,960 B** at (64, 64, 32), against
**65,536 B** at n=64 — over the 63 KB limit (`CLAUDE.md` trap 3).

The plan called for changing it in three places at once. Instead it became a
**parameter**, because a global default would have made MiniLM's existing
container incompatible with any newly built design:

- `tools/pack_npue.py` already had `--tile-n`.
- The C++ packer carried the layout JSON **and a frozen sha256** for tile_n 48,
  which made a second tile size unexpressible without editing the packer. Now
  `npue::gemm_b_layout(tile_k, tile_n)` builds both — insertion-ordered JSON
  for the file, key-sorted for the hash, matching `tools/npue.py` exactly.
  **Verified by reproducing the frozen constant**: tile_n 48 computes
  `94266693ea31aa67…`, byte for byte the value that had been hardcoded, and
  tile_n 32 agrees with Python at `f2ab7b0d310c935e…`.
- `tools/verify_pack_parity.py` gained `--tile-n`. Both packers are
  byte-identical at 48 **and** at 32, for MiniLM and for bge-large.

Losing the frozen hash loses nothing: the load-bearing guard is the runtime
comparing the design's `b_layout_hash` against the container's, and that check
is untouched.

---

## The failure worth keeping: a banner that printed the intention, not the value

After wiring `--tile-n` the packer printed

```
  layout     tile (64, 32), hash f2ab7b0d310c935e...
```

and then failed with `operand [1024,1024] does not tile evenly by (64, 48)`.
The flag reached the *layout descriptor* and not the `prepare_model` call,
which still passed literal `64, 48`. **The banner reported what I meant, and
the code did something else** — the same shape as every fail-open in this
project, arriving this time through a status line.

It was caught only because the shapes made 48 illegal. At a width where both
values happened to be legal it would have written a container tiled at 48 and
labelled 32, and the layout-hash guard would have rejected it later with a
message pointing at the wrong thing.

The error message itself was also useless — `operand does not tile evenly`,
naming nothing. It now prints the shape, the tile and both remainders, which is
what turned a guess into a two-minute diagnosis.

---

## Four more constants that only fitted one model

Found by running the tools against h=1024 rather than by reading them:

1. **`HIDDEN = 384`** in `verify_embed_e2e.py`, `verify_endpoint.py` and
   `npu_encoder.py`. Now read from the container and used to **check** the byte
   count — an inferred width would reshape a truncated file into a plausible
   array.
2. **`check_fold_cost` forwarded `--model-dir` but not `--tile-n`**, so the
   unfolded control was built at a tile size the model cannot use. Exactly the
   bug [`0039`](../0039-m9-bge-small/TASK.md) fixed, one layer deeper, and the
   fix is the same: take it from the container, which knows.
3. **The default output name** for `--prepare-model` was
   `<dir>/all-MiniLM-L6-v2.npue` regardless of `<dir>`.
4. **`make_goldens.py` died on `UnicodeEncodeError`** twenty minutes into a
   golden run. The corpus is deliberately non-ASCII and Windows redirects
   stdout as cp1252, so *reporting* on the data killed the run that produced
   it. `sys.stdout.reconfigure(encoding="utf-8")` in the three tools that print
   corpus text.

---

## Results

| check | bge-large |
|---|---|
| checkpoint vs `docs/04-model` | OK — 24 layers, hidden 1024, 16 heads × 64, ffn 4096 |
| oracle vs HF **and** sentence-transformers | 29 comparisons within `rel_fro` 2e-05 |
| pooling | **CLS**, asserted against sentence-transformers at 8.196e-08 |
| both packers, tile_n 32 | **byte-identical** (`001d29c0…`) |
| `.npue` verify, bf16 activations | `1-cos` **1.355e-05**, fold free (1.000×) |
| NPU end to end vs HF golden | `1-cos` **8.432e-06** |
| two pipeline lanes | bitwise identical on 8,388,608 floats |
| end to end vs sentence-transformers | **PASS** |
| throughput, batch 128, 2 lanes | **52.8 seq/s** on ~6.0 cores |
| NPU busy share | **79.1%** of wall (192 dispatches per group) |

The design set is **one xclbin, 4 streams** at h=1024 — the one-context
architecture carries to a model 2.7× wider with no change.

---

## Carry forward

- `artifacts_large` has a single batch tier (128). MiniLM's has four, so
  small requests are right-sized there and padded here. Export the tiers before
  quoting a small-batch number for bge-large.
- The NPU is busy **79.1%** of wall at h=1024 against 55.3% at h=384. Host work
  is no longer the co-bottleneck it was; the array is.
- `bench_bo_mode.py` should be re-run here — [`0041`](../0041-m9-bo-allocation-flavour/TASK.md)
  found allocation flavour irrelevant at MiniLM's 175 MB, and this stages
  604 MB, where TLB pressure could differ.
- D2 (attention on the array) now has its precondition: **head_dim 64**.
- **The boundary golden is 26.1 MB**, the largest tracked file in the repo by
  4x, because 24 layers of `[4, 64, 1024]` taps is simply that much data.
  `reference/README.md`'s rule (boundary committed, taps gitignored) is
  followed, and it is kept for uniformity -- `check_reference.py` expects the
  same tensor set for every model. It is worth a decision before the next
  public sync, which already reports files over 0.5 MB.
