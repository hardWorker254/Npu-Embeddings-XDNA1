# 0043 — M9 Phase D2: can attention run on the array?

**Question.** `head_dim = 64` on bge-large removes the tiling blocker that has
kept attention on the host since M5. The plan required the **geometry table
measured first**, and the architecture decided from it rather than from a
build: a design that expresses attention *and* the projections costs one
switch-free context, but only if its tile size does not cost more than
attention gains.

Depends on [`0042`](../0042-m9-bge-large/TASK.md).

---

## The plan's candidate geometry does not exist

The plan proposed `(m=16, k=64, n=8, cols=8)` as the geometry that could serve
all six operators with zero switches. It **does not compile**:

```
matmul_bf16_f32.cc:321: static_assert(n % (2 * t) == 0);
  note: expression evaluates to '8 == 0'
```

The AIE microkernel constrains the tile independently of the dataflow, and the
plan's analysis had only considered the dataflow:

| | constraint | with bf16 `mac (r,s,t) = (4,8,8)` |
|---|---|---|
| microkernel | `m % (2r) == 0` | **m multiple of 8** |
| microkernel | `k % s == 0` | **k multiple of 8** |
| microkernel | `n % (2t) == 0` | **n multiple of 16** |
| design | `M % (m · rows) == 0` | rows = 4 |
| design | `N % (n · cols) == 0` | |
| design | `K % k == 0` | |

Attention's per-head GEMM is `[64,64] × [64,64]` — M = K = N = **64** — so
`N % (n · cols) == 0` forces `n · cols` to divide 64, while the microkernel
forces `n ≥ 16`. **Therefore `cols ≤ 4`.**

### The structural result

**A design that can express attention can use at most half the array.** That
is not a tuning outcome; it follows from seq = 64 and the microkernel's
16-wide n. Enumerating everything legal for attention *and* the bge-large
projections, within the 63 KB L1 budget:

| m | k | n | cols | L1 | output tile |
|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 64 | 1 | 28,672 B | 1024 |
| 16 | 64 | 32 | 2 | 16,384 B | 512 |
| **16** | **64** | **16** | **4** | **10,240 B** | **256** |
| 8 | 64 | 16 | 4 | 7,168 B | 128 |

against production's `(64, 64, 32)` at **8 columns** with a **2048**-element
output tile. The best unified candidate gives up **half the columns and 8× the
tile**.

So the plan's "24× more per-tile overhead" was the right worry pointed at the
wrong geometry: the real cost is 8× on tiles *and* 2× on columns, and the
geometry it was computed for cannot be built.

---

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# A -- production
python tools\export_gemm_rtp.py --hidden 1024 -n 32 --batch 128 `
    --out runtime\artifacts_large
# B -- mid
python tools\export_gemm_rtp.py --hidden 1024 -m 32 -k 64 -n 16 --batch 128 `
    --out runtime\artifacts_large_m32
# C -- small tile, FULL width (isolates tile size from column count)
python tools\export_gemm_rtp.py --hidden 1024 -m 16 -k 64 -n 16 --cols 8 `
    --batch 128 --out runtime\artifacts_large_m16c8
# D -- small tile, half width: the only unified candidate
python tools\export_gemm_rtp.py --hidden 1024 -m 16 -k 64 -n 16 --cols 4 `
    --batch 128 --out runtime\artifacts_large_m16c4

# the n=16 sets need a container tiled to match
runtime\build\npuembed.exe .. --prepare-model ..\models\bge-large-en-v1.5 `
    ..\models\bge-large-n16.npue --tile-n 16
python tools\export_validation.py --model bge-large-n16
```

`C` exists to separate the two costs: it has the unified geometry's tile at the
production column count, so `A → C` is the tile-size penalty alone and
`C → D` is the column-count penalty alone.

---

## Results

*(filled in below)*

---

## A container is not a checkpoint

`bge-large-n16.npue` and `bge-large-en-v1.5.npue` are the **same weights**
packed at different tile sizes, and they must share goldens. They could not:
`export_validation.py` derived the golden filename from the **container name**,
so the n16 container went looking for `bge-large-n16_l24_s64_*`.

Goldens belong to a checkpoint; a container is one packing of it. The lookup
now scans `reference/goldens/*_boundary.safetensors` and matches on
**`source_sha256`**, failing if the number of matches is not exactly one.
Content, not name — the same rule `CLAUDE.md` trap 7c already states for build
artifacts, arriving here from the other direction.
