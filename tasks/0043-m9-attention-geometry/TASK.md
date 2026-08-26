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

**Filled in 2026-08-23** ([`0097`](../0097-t18-t21-t4-measurements/TASK.md)). All four
artifact sets (`artifacts_large`, `artifacts_large_m32`, `artifacts_large_m16c8`,
`artifacts_large_m16c4`) and `models/bge-large-n16.npue` already existed on disk from
this task's own export commands above; nothing had to be rebuilt. `--probe-streams`
(tasks/0048, not yet written when this task's Results section was left empty) is the
right instrument: no host work in the loop, so it isolates the array from the host
`bench` path. Two repeats per set (A and D; B and C got one each, low priority since
A/D bracket the question), idle array, contention gate green throughout
(`xrt-smi examine --report aie-partitions` showed no `Active` foreign context on every
check).

Per-stream `us/disp`, mean of repeats, and the four-shape sum (fp32 C, bf16 A/B — none
of these predate `--c-bf16` or `--int8`, so this is the plain production bf16 datapath):

| set | m,k,n,cols | qkv | attn_out | ffn_up | ffn_down | **sum** | GMAC/ms (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** (production) | 64,64,32,8 | 18,839 | 6,793 | 25,084 | 25,038 | **75,754** | 1.35 |
| **B** (mid) | 32,64,16,8 | 19,998 | 7,315 | 26,436 | 26,364 | **80,113** | 1.27 |
| **C** (small tile, full width) | 16,64,16,8 | 24,680 | 8,583 | 31,935 | 32,442 | **97,640** | 1.05 |
| **D** (small tile, half width — the unified candidate) | 16,64,16,4 | 42,086 | 14,978 | 55,695 | 56,107 | **168,865** | 0.60 |

All four in µs, M=8192 (batch 128, seq 64), the bge-large projection shapes.

**Correctness spot-check** (not part of the original plan, added because a fast wrong
design is not a result): plain `.\npuembed.exe . --model <name> --artifacts <set>`
(no `--bench`, no `--probe`) on set **A** and set **D** both reproduce the documented
production number exactly — `rel_fro` 3.763e-03, worst `1-cos` 8.432e-06, PASS. Sets B
and C were not independently re-verified this session (A and D bracket both cost axes
and both pass, and the tile geometry only changes access order, not arithmetic — but
this is inference, not measurement, for B and C specifically).

### The two costs, isolated

- **A → C (tile-size penalty alone, cols fixed at 8): 1.289×.** Shrinking the tile from
  the production (64,64,32) to the unified candidate's (16,64,16) costs 28.9% more array
  time by itself, before touching column count.
- **C → D (column-count penalty alone, tile fixed at 16,64,16): 1.729×.** Halving the
  array from 8 columns to 4 costs another 72.9%.
- **A → D (the full cost of the only geometry that can also express attention): 2.229×.**
  Compounding, not additive (1.289 × 1.729 = 2.228, matching to 3 significant figures —
  the two penalties are independent, as the plan's split experiment intended).

### The verdict CLAUDE.md has been citing without support

CLAUDE.md and `docs/CURRENT_STATUS.md` both currently credit this task with concluding
that folding attention onto the array is "not worth the fight," but until today that
conclusion had no measurement behind it in this file — task `0089` flagged the gap this
session. It is now supported, and the number is sharper than "not worth it": **the only
geometry that can express attention costs the four projection GEMMs 2.23× their current
array time**, against the ~2–5% of the encode that attention itself costs on the host
(CLAUDE.md's own F3 figure, and the `docs/CURRENT_STATUS.md` "~4%" estimate). Even under
the most generous version of the trade — the array being, say, 40% of a bge-large encode
(the high end measured for MiniLM in `0051`; bge-large's own int8 figure from `0081` is
30.4%) — moving projections onto the D geometry would add roughly `0.40 × 1.23 ≈ 0.49`
of an encode's worth of array time to remove at most 0.05 of an encode's worth of host
attention. **The trade is unambiguously negative, by roughly an order of magnitude, at
any plausible array share.** CLAUDE.md's existing wording ("has not found it worth the
fight") is confirmed rather than refuted, and can now cite this measurement instead of
standing on the plan's un-measured intuition.

**note 0007 §1.1's `pad_dimensions` idea is not closed by anything measured here.**
§1.1 proposes padding attention's real 8-wide-per-column N=64 slice to 16 *in the mem
tile* (`AIEDialect.cpp`'s own verifier already restricts `pad_dimensions` to mem tiles,
which is exactly where §1.1 proposed to use it), making `n=16, cols=8` legal for
attention instead of this task's derived `cols<=4`. `tasks/0090` (T5, DMA compression)
checked the AIE2P vendor register tables, and a follow-up read of the same file settles
the padding question directly — `.Padding` in `xaie2pgbl_reginit.c`:

| line | DMA module | `.Padding` |
|---:|---|---|
| 1667 | `Aie2PMemTileDmaMod` | **`XAIE_FEATURE_AVAILABLE`** |
| 1905 | `Aie2PTileDmaMod` (compute tile) | `XAIE_FEATURE_UNAVAILABLE` |
| 2158 | `Aie2PShimDmaMod` | `XAIE_FEATURE_UNAVAILABLE` |

So the hardware has padding **only on the mem tile** — precisely where §1.1 wants to use
it, and matching what `AIEDialect.cpp`'s verifier already enforces in software. This
**confirms the mechanism exists in silicon** rather than merely not refuting it.
(Correcting this task's own first draft, which said 0090 had not checked `.Padding` at
all: it had — the field sits three lines below the `.Compression` flag 0090 quotes — but
the shim's `UNAVAILABLE` there is not the relevant tile, which is why it reads as
irrelevant at first glance.) **If** padding works, the geometry the projections would have to
share is set **C** here (cols=8), not **D** — a **1.289×** tax rather than **2.229×**,
since the column halving is exactly what padding is meant to remove. That is a
meaningfully smaller cost than this task's headline number, and it changes the verdict
from "clearly negative" to "still probably negative, at 30–40% array share, but close
enough to be worth an actual build" — the padding mechanism itself remains unbuilt and
unmeasured. See [`0097`](../0097-t18-t21-t4-measurements/TASK.md) T4 section for the
full reasoning and the correction of an earlier mischaracterisation of what `0090`
established.

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
