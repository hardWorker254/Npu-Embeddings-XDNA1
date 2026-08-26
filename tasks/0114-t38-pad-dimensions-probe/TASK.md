# 0114 — T38 answered: mem-tile padding routes and works at attention's geometry, on all 8 columns

- **Date** 2026-08-25
- **Milestone** research (T38)
- **Status** done — T38 **ANSWERED**. The mechanism works: 8 → 16 padding on
  the mem tile, **exact on all 8 columns**, the un-pad is an ordinary strided
  mem-tile read, and a compute tile consumes the padded stream. So **0043's
  `cols ≤ 4` wall is removable**. Whether to spend it on folding attention is
  now a *sequence-length* question, and 0113 changed its sign.

## Goal

[T38](../../research/CLOSED-THREADS.md#t38) says, in as many words, that this
one is worth building rather than arguing:

> **What is not known**: whether it actually routes and works in an IRON design
> at this geometry, and whether 1.289× on every projection GEMM is worth
> attention's ~2–5% (F3). Probably still negative — but "probably" is a
> different verdict from 0043's, and this one is worth an actual build rather
> than an argument.

## Context

[`0043`](../0043-m9-attention-geometry/TASK.md) established the wall
structurally: attention's per-head GEMM is `[64,64] × [64,64]`, the whole-array
design needs `N % (n · cols) == 0`, and the bf16 microkernel needs `n % 16 == 0`
— so `n · cols` must divide 64 with `n ≥ 16`, hence **`cols ≤ 4`**.
[`0097`](../0097-t18-t21-t4-measurements/TASK.md) then measured the price:
**2.229×** on the projections, of which **1.729× is the column halving alone**.

[note 0007](../../research/notes/0007-unused-iron-surface.md) §1.1 observed that
`N = 64` is not given — the DMA can pad the per-column 8-wide slice to 16 — and
warned exactly where a first attempt would fail:

> padding is one-directional … **C comes back 16-wide per column and the output
> side needs a strided `dims_to_stream` that takes 8 of every 16.** That is an
> ordinary access pattern, not a second padding feature, but it is a second
> thing to get right and it is where a first attempt will fail.

Two hardware facts already supported it, neither of them a design:
`xaie2pgbl_reginit.c:1667` gives `Aie2PMemTileDmaMod .Padding = AVAILABLE`
where the compute tile (1905) and shim (2158) do not, and `AIEDialect.cpp`'s
`DMABDOp::verify()` restricts `pad_dimensions` to mem tiles, requires an n-d
access pattern, and requires the inner-most pad to land on 32-bit word
boundaries.

## What was built

`experiments/m9-attn-geometry/pad_dimensions_probe.py` and its kernel
`kernels/scale_bf16.cc`. Three stages, because they fail differently:

| stage | topology | question |
|---|---|---|
| `pad` | shim → mem tile (**pad 8→16**) → shim, drained 16 wide | does padding work at this geometry, and on 8 columns? |
| `unpad` | shim (16 wide) → mem tile (**strided read, 8 of 16**) → shim | is the un-pad really just an access pattern? |
| `compute` | shim → mem tile (pad 8→16) → **CORE** → mem tile → shim | can a compute tile consume a padded stream? |

Geometry is attention's own: `ROWS = 64` (the K of the slice), `N_REAL = 8`
(what a column owns at `N=64` over 8 columns), `N_PAD = 16` (the microkernel's
minimum), bf16, `pad_value = 0` — zero columns of B give exactly-zero columns
of C, so the padding is *exact*, not approximate.

The `compute` kernel is deliberately trivial — `y = 2x` — so a mismatch can only
be about data movement. Doubling rather than copying, because a copy cannot
distinguish "the kernel ran" from "the buffer already held the right bytes",
and `2 × 0` is still 0 so the padded half stays checkable.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m9-attn-geometry

python pad_dimensions_probe.py --stage pad     --cols 1
python pad_dimensions_probe.py --stage unpad   --cols 1
python pad_dimensions_probe.py --stage pad     --cols 8
python pad_dimensions_probe.py --stage unpad   --cols 8
python pad_dimensions_probe.py --stage compute
```

## Result — all four checks pass, exactly

```
PAD    cols=1: real MATCH, padded half all-0 YES (max |pad| = 0.000e+00)
UNPAD  cols=1: EXACT  max|diff| = 0.000e+00
PAD    cols=8: real MATCH, padded half all-0 YES (max |pad| = 0.000e+00)
UNPAD  cols=8: EXACT  max|diff| = 0.000e+00
compute over a padded stream: EXACT  max|diff| = 0.000e+00
  padded half after compute: max|.| = 0.000e+00
```

**Every number is exact, not within a tolerance**, which is the right gate here:
this is data movement plus a doubling, and there is no rounding anywhere in it.

So, point by point:

1. **`pad_dimensions` routes on all 8 columns.** This is the one that matters —
   0043's restriction was that a design expressing attention could use at most
   half the array, and the padded form uses all of it. Eight mem tiles padding
   concurrently, eight shim drains, no routing failure.
2. **The padded half is exactly the pad value.** Not "small" — bit-zero.
3. **The un-pad is an ordinary access pattern**, exactly as note 0007 said:
   `dims_to_stream=[(64, 16), (8, 1)]` on the mem tile's outbound leg. No
   second feature, no special support.
4. **A compute tile can consume a padded stream**, which is a distinct question
   from whether a mem tile can make one — the verifier forbids padding *on* a
   compute tile, so the core reads a buffer that was padded upstream.

## Problems hit — both were predicted, in writing, before the probe ran

**1. The un-pad failed first, exactly where note 0007 said it would.**
The first version put both halves on one hop: pad 8→16 on the mem tile, then
try to take 8 of every 16 back with a strided tap on the *host drain*. It
compiled, ran, and returned **MISMATCH, max|diff| = 2.121e+00**.

The reason is worth keeping: **a shim DMA consumes the stream in order and
cannot skip.** A drain tap describes where bytes *land* in host memory, not
which stream elements to keep — so a 512-element tap against a 1024-element
stream takes the first 512 (rows 0–31 of the padded data), which is what the
mismatch was. Selecting 8 of 16 is a strided **read on the mem tile**, a
different BD on a different hop. That is also the real design's topology: pad on
the way *in* to the cores, un-pad on the way *out* of them. Restructuring into
two separate designs made both exact.

**2. CLAUDE.md trap 7d, hit live.** The second variant was served the first
variant's binary:

```
RuntimeError: Tensor argument 'Y' has 512 elements but the kernel was
compiled for 1024 elements.
```

`iron.jit`'s cache key derives from the call's arguments plus
`(device, full_elf)` and **never inspects the generator's globals or closure** —
and `cols`/`full_width` were closure variables. The `purge()` marker did not
save it either, because a purge matches the markers of the build it is *about
to make*, not of the stale one it needs to remove. Fixed the way trap 7d
prescribes and [`0110`](../0110-refuse-silent-truncation/TASK.md) verified for
`M`: make them `CompileTime[int]` kwargs, so they enter the key and each variant
is its own build. This is the **seventh** instance of the "stale binary fails
open" class in this project — and the first where it failed *loudly*, because
the argument sizes disagreed. Had both variants been the same size it would
have returned a wrong answer silently.

**3. Two kernel-compile errors, both about bf16 widening.**
`(*it++).to_vector<float>()` does not compile (the iterator yields
`aie::vector<bfloat16,16>`, which has no such member), and neither does the
implicit `aie::vector<float,16> x = *it++` (no such conversion exists). The
idiom that works is the one [`0091`](../0091-t7-t8-gelu-poly/TASK.md) (T7)
established for exactly this: `aie::accum<accfloat,16>::from_vector`, which
compiles to `vlda.conv.fp32.bf16`, a load-with-conversion.

## What this changes — and the verdict is now sequence-dependent

**The blocker is gone.** 0043's `cols ≤ 4` is not a property of the hardware; it
is a property of a design that declines to pad. With padding, attention's
geometry is expressible at `n = 16, cols = 8`, and the 1.729× column halving
0097 measured stops being forced.

**But whether to spend it on folding attention is a different question, and
[`0113`](../0113-t40-seq512-close/TASK.md) changed its sign this same day.**
T38 inherited its cost/benefit from F3's "attention is 2–5% of the work" — and
0113 measured that this is a **seq-64 number**:

| | seq 64 | seq 256 | seq 512 |
|---|---:|---:|---:|
| array share of wall | **76.3%** | 62.0% | **46.6%** |
| host attention share of wall | 16.5% | 33.6% | **52.1%** |

At seq 64 the array is the bottleneck, so **moving attention onto it is the
wrong direction** — F3's conclusion holds and 0043's verdict was right. At seq
512 the host is the bottleneck and attention is more than half the wall clock,
so the direction reverses.

**A model estimate, labelled as one** (nomic, hidden 768, intermediate 3072,
gated). Per token, projections + FFN are ≈9.4 M MACs and attention is
`2 · seq · hidden`:

| seq | attention as share of MACs | array if folded | host attention removed |
|---:|---:|---:|---:|
| 64 | ~1.0% | 1569 → ~1585 ms | −339 ms (host not the bottleneck) |
| 512 | **~7.7%** | 1537 → **~1656 ms** | **−1721 ms** |

At 512 that would take the lane's host work from ~2475 ms to ~755 ms and leave
the array at ~1656 ms as the new bottleneck — **wall 3300 ms → roughly
1700–2000 ms, i.e. 1.65×–1.95×.** That is a model, not a measurement: it prices
the MACs and the removed host work, and it prices **nothing** for softmax on the
array, for the per-head data movement, or for the design switch that a separate
attention shape would reintroduce. note 0007 §3.4's warning still stands —
whisper-xdna built fused attention, it was correct, and it still lost.

So: **T38's mechanism question is answered YES; its value question is answered
"no at seq 64, plausibly large above ~470", and the successor is a design task,
not a question.** Filed as [T42](../../research/OPEN-THREADS.md#t42).

## Artifacts

* `experiments/m9-attn-geometry/pad_dimensions_probe.py` — the probe, three
  stages, with the failed first form documented in `build_transport`'s docstring
* `experiments/m9-attn-geometry/kernels/scale_bf16.cc` — the trivial kernel

## Next

Nothing further on T38. [T42](../../research/OPEN-THREADS.md#t42) carries the
design question this unblocked, with the numbers above as its starting price.
