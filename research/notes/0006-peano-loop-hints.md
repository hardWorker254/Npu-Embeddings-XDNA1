# 0006 — which loop hints survive Peano, and which are decoration

*Written 2026-08-18, prompted by AMD's `aie-kernel-opt` skill
(`.claude/skills/aie-kernel-opt/SKILL.md`, Apache-2.0 WITH LLVM-exception).
Verified against our own installed toolchain rather than taken on trust.*

---

## The short version

`CLAUDE.md` trap 5b says **`chess_*` pragmas are silently ignored by Peano**.
True, and we acted on it: our kernels use the portable `AIE_*` wrappers from
`aie_kernels/aie_kernel_utils.h` instead.

**One of the two wrappers we picked is also empty under Peano.**

| macro | `__chess__` | **`__AIECC__` (Peano)** | in our kernels |
|---|---|---|---|
| `AIE_PREPARE_FOR_PIPELINING` | `[[chess::prepare_for_pipelining]]` | **nothing** | **8 uses** |
| `AIE_LOOP_MIN_ITERATION_COUNT(n)` | `[[chess::min_loop_count(n)]]` | `clang loop min_iteration_count(n)` | 8 uses |
| `AIE_LOOP_MAX_ITERATION_COUNT(n)` | `[[chess::max_loop_count(n)]]` | `clang loop max_iteration_count(n)` | **0** |
| `AIE_LOOP_RANGE(a, b)` | both chess counts | both clang pragmas | **0** |
| `AIE_LOOP_UNROLL_FULL` | `[[chess::unroll_loop()]]` | `clang loop unroll(full)` | **0** |
| `AIE_LOOP_UNROLL(x)` | `[[chess::unroll_loop(x)]]` | `clang loop unroll_count(x)` | **0** |

So every `AIE_PREPARE_FOR_PIPELINING` we have written compiles to nothing.
It is the same failure trap 5b describes, wearing the portable name — code
that *looks* tuned and is not, second occurrence.

The min-count hint is real and does bind, so our kernels are not hint-free.
What we have never used is the **upper** bound or either unroll.

## How it was verified, rather than read

The header (`C:\dev\mlir-aie\aie_kernels\aie_kernel_utils.h`) has three
branches: `__chess__`, `__AIECC__`, and an else that defines everything to
nothing. Reading it only tells you what each branch contains; it does not tell
you which branch Peano takes. A probe does:

```c
// probe_macros.cc
#include "aie_kernel_utils.h"
#if defined(__chess__)
#error BRANCH_IS_CHESS
#elif defined(__AIECC__)
#error BRANCH_IS_AIECC
#else
#error BRANCH_IS_FALLBACK_EVERYTHING_EMPTY
#endif
```

```shell
clang++ -O2 -std=c++20 -I C:/dev/mlir-aie/aie_kernels \
  --target=aie2p-none-unknown-elf -S probe_macros.cc
#  -> error: BRANCH_IS_AIECC
```

`__AIECC__` is defined by the **aie2p clang driver itself**, with no `-D` on
the command line — which is worth knowing, because it also means `-U__AIENGINE__`
on the command line does not stick against the driver's own builtins.

## What is worth trying, and what is not

The skill's headline for this lever is **−47% on a 3×3 conv** from changing
`AIE_LOOP_RANGE(3,3)` to `AIE_LOOP_UNROLL_FULL`, and it is specific about
*why*: the loop body `switch`ed on the loop variable, so a trip-count hint
left a runtime branch in place while a full unroll removed it.

**Our kernels do not have that shape.** `gelu_poly.cc`'s hot loop is
straight-line Horner over four independent chains with no index-dependent
branching, so the mechanism the −47% came from is absent. Adding
`AIE_LOOP_UNROLL_FULL` there is a guess, not an inference.

What is a genuine inference: an **upper** iteration bound is missing
everywhere, and the skill's lever #2 says the hints only bind when the trip
count is a known constant. That is also the mechanism behind
[`0029`](../../tasks/0029-m7-one-xclbin-probe/TASK.md)'s open risk — `rtp=True`
makes the GEMM's loop bounds *runtime values*, and
[`0030`](../../tasks/0030-m7-expert-review-tests/TASK.md) measured the cost at
**+1.6%**. That number now has an explanation rather than just a
size: runtime bounds are exactly what stops these pragmas folding.

**Not acted on yet**, because the eltwise kernels this affects run on the
**host** in production ([`0031`](../../tasks/0031-m7-eltwise-ilp/TASK.md)
onward) and the GEMM microkernel is mlir-aie's, not ours. The finding is
recorded so that whoever brings eltwise back onto the array — the fused-design
direction — starts from real hints instead of decoration.

## The other thing worth taking from that skill

> AIE cores don't stall (no cache, no dynamic scheduling, no branch
> misprediction), so the hot loop's instruction count ÷ core clock ≈ its
> execution time.

Our `docs/05-measurement` already has static instruction counting as
**Signal 2, a cross-check**. The skill's claim is stronger: for a
compute-bound kernel it can **replace** hardware timing.

That matters here more than it does for most projects, because
`CLAUDE.md` trap 7 says **the production 8-column design cannot be traced** —
adding one trace flow exhausts routing. A method that needs only the `.o` is
not blocked by that.

Its distinction is the important part, and it is one we had not drawn:
counting instructions in *emitted assembly* is not the same as the
source-level paper-compute modelling the same document warns has mispredicted
by 5–300×. The first counts what runs; the second guesses what the compiler
did.
