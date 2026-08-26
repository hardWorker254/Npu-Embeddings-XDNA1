# 0098 — T26: reading the kernel source finds a rounding-mode dead-code-elimination in the emulated-bf16 matmul

- **Date** 2026-08-23
- **Milestone** post-M13 (research thread, `research/OPEN-THREADS.md` T26)
- **Status** done — **new mechanism found, source-level confirmed, NOT yet
  confirmed on hardware.** T26 stays OPEN; a specific ablation is proposed.

## Goal

T26 asks why bfp16-emulated GEMM + bf16-C transport measures 6.6x **more**
accurate than bfp16 + fp32-C (`1-cos` 2.395e-03 -> 3.615e-04,
[`0052`](../0052-m10-research-night/TASK.md) S6). [`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md)
and [`0056`](../0056-m10-t26-rounding-and-chain-probe/TASK.md) refuted two
hypotheses (k-block-boundary re-quantisation; floor-vs-conv_even asymmetry in
the *host* narrowing path) and confirmed a mechanism class (bf16-C narrows its
raw accumulator once, early, on-core, before bias-add; this compounds
monotonically across a chained-GEMM probe, ratio 0.99 -> 1.35 -> 1.46 -> 1.67
by stage 4). [`0096`](../0096-t26-numerical-mechanism/TASK.md) built a
calibrated numpy/fp64 model of "narrow position applied to independent
per-block bfp16 noise" and it **failed its own control** -- the model's ratio
stays pinned at 0.99-1.00 across every stage, mantissa calibration (3-9 bits)
and real-weight test, against hardware's climb to 1.67x. 0096's own
conclusion: the missing ingredient must be a hardware/kernel-level numerical
detail a host bit-formula model cannot express, and the way to find it is to
read the kernel source directly -- the same method that found trap 2b's
floor-rounding bias by reading `aie_api/aie.hpp`, not by modelling.

This task does that reading. It does **not** touch the NPU (a sibling agent
had exclusive hardware access for timing work during this task; all evidence
below comes from source, headers, and objdump artifacts already checked into
the repo from 0053's 2026-08-20 run).

## What was read

1. `research/OPEN-THREADS.md` T26 entry in full (lines 570-703, all four
   PROBED FURTHER updates from 0052/0053/0056/0096).
2. `tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md`,
   `tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md`,
   `tasks/0096-t26-numerical-mechanism/TASK.md` in full.
3. `experiments/m5-eltwise/kernels/narrow_f32_bf16.cc` (111 lines, in full) --
   the bf16-C epilogue kernel.
4. `experiments/m5-pretiled-gemm/gemm_pretiled.py` lines 95-518 -- the IRON
   DSL that builds both `core_fn` variants (fp32-C: accumulate directly into
   the C ObjectFifo object; bf16-C: accumulate into a dedicated core-local
   `Buffer`, then call `narrow()`).
5. `C:\dev\mlir-aie\python\iron\kernels\linalg.py` lines 1-202 (`kernels.mm`,
   the `emulate_bf16_mmul_with_bfp16` flag and its `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
   compile flag).
6. `C:\dev\mlir-aie\aie_kernels\aie2p\mm.cc` lines 78-232, 456-489 -- the
   matmul kernel template, including the `#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`
   block (lines 89-101, 226-229) that wraps the whole k-loop in
   `aie::swap_rounding(aie::rounding_mode::conv_even)` /
   `aie::set_rounding(saved_rounding)`, with a comment explaining why: the
   bf16->bfp16 conversion inside the MAC follows the core rounding mode, and
   the AIE default (`floor`) would bias every converted element low.
7. `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\detail\aie2p\mmul_bf16_bf16.hpp`
   in full (212 lines) -- the `mmul_bf16_bf16<4,8,8,...>` and `<8,8,8,...>`
   specialisations selected by `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`.
   Both call `accum<accfloat,N>(a)` / `accum<accfloat,N>(b)` (a plain fp32-width
   accumulator load, see item 9) then `::to_v64bfp16ebs8(acc)` -- the actual
   bf16->bfp16 quantisation -- then `::mac_8x8_8x8T_conf(tmp_a, tmp_b, ...)`.
   Also confirms the emulated 8x8x8 path uses **one** accumulator register
   (`C_block<...,1>`) against plain bf16's **two** (`C_block<...,2>`, the
   `#else` branch) -- consistent with CLAUDE.md trap 3b (aie2p has 5
   accumulator registers, not 8) but common to both fp32-C and bf16-C builds,
   not a differentiator between them.
8. `C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\lib\clang\21\include\aie2p\aie2p_srs.h`
   lines 1280-1378 -- the actual intrinsic definitions. **This is the load-bearing
   read.** `to_v64bfp16ebs8(v64accfloat a)` (line 1308, the plain form called
   by `mmul_bf16_bf16.hpp`) does **not** touch the rounding-mode register at
   all -- it is a bare `__builtin_aie2p_v64accfloat_to_v64bfp16ebs8` call, so
   it runs under whatever the core's *ambient* rounding-mode control register
   holds at the moment it executes. Only the sibling `to_v64bfp16ebs8_conf(a, rnd)`
   (line 1331) does the save/set/convert/restore dance
   (`get_rnd()`/`set_rnd(rnd)`/`...`/`set_rnd(prev_rnd)`) -- and
   `mmul_bf16_bf16.hpp` never calls the `_conf` form; it calls the plain one.
   The same ambient-vs-`_conf` split holds for every other conversion in the
   file (`ssrs`/`lsrs`/`to_v32int8`/etc., lines 1-475), so this is a
   systematic API convention, not a one-off.
9. `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\detail\aie2p\accum_native_types.hpp`
   lines 167-172 -- `accum_storage<AccumClass::FP, 32, N>` confirms `accfloat`
   is a **32-bit-per-lane** accumulator format (`Bits=32` in the FP class,
   backed by `v8/16/32/64accfloat`), i.e. plain IEEE-754 single-precision
   width per element, not an extended-precision register. Loading an exact
   fp32 (or bf16-widened-to-fp32) value into an `accfloat` accumulator and
   reading it back is lossless at the bit-width level; it is not itself a
   source of extra error and is identical machinery in both the emulated
   matmul (item 7) and `narrow_f32_bf16.cc`'s epilogue (item 3). Answers the
   task brief's opening question directly: **`aie::accum<accfloat,N>` is a
   normal 32-bit float accumulator, not a wider intermediate.**
10. `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\tile.hpp` line 62 and
    `detail/aie2\tile.hpp` lines 88-98 -- `aie::tile::set_rounding`/`get_rounding`
    (what `aie::set_rounding`/`aie::swap_rounding` in item 6 actually call)
    lower to exactly `::set_rnd((unsigned)mode)` / `::get_rnd()` -- the
    **same** low-level pair `to_v64bfp16ebs8_conf` uses internally. Confirms
    both call sites (mm.cc's function-level swap_rounding, and the intrinsic's
    own `_conf` variants) target the identical hardware register; there is no
    separate "matmul rounding register" vs "narrow rounding register".
11. `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\aie.hpp` lines
    6963 and 6979 -- "The rounding mode is set to the default
    `aie::rounding_mode::floor`", the same citation CLAUDE.md trap 2b already
    uses. Cited here as the basis for "an AIE core that has never had its
    rounding mode touched runs under `floor`."
12. **The objdump artifacts 0053 already produced and checked in**
    (`experiments/m5-pretiled-gemm/artifacts/objdump_{fp32C,bf16C}_rtp_*.txt`,
    16 core wrappers + matmul + narrow, from the M=256/K=384/N=192/4-column
    T26 probe, `rtp=True` both sides). Re-diffed independently rather than
    trusting 0053's prose summary, per the brief's instruction ("0053 may have
    looked at less than it thought").

## What was found

### Claim 1 (0053's wrapper-diff claim): SURVIVES an independent re-diff

`diff objdump_fp32C_rtp_main_core_0_2.txt objdump_bf16C_rtp_main_core_0_2.txt`
(full diff saved as `diff_main_core_0_2.txt` in this task folder, 384 diff
lines). Read every hunk, not just the summary. The k-block loop body itself
(`.LBB0_3`..`.LBB0_5`, the `acq`/`vlda`/`jl 333c4d33_matmul_bf16_f32`
sequence) is **structurally identical** between the two builds -- same
instruction opcodes, same operand shapes, same relocation targets for
`A_L2L1_*_cons_buff_*`/`B_L2L1_*`. The only differences are: (a) which
physical buffer the `movxm`-loaded pointer targets --
`C_L1L2_0_0_buff_0/1` (fp32-C, the C fifo object itself) vs `cacc_0_0`
(bf16-C, the dedicated `Buffer`) -- confirming the register-allocation-driven
reordering of the *other* pointer loads at offsets 0x46/0x50/0x5c is exactly
that: reordering caused by the extra `acc` parameter changing argument
layout, not a new operation; and (b) one new basic block (`.LBB0_6` in the
bf16-C build) between the k-loop's end and the `rel #0x36` release, which
loads `p1` from `[sp,#-120]` (the acquired C-fifo slot) and `p0` from the
`cacc_0_0` relocation, then `jl narrow_3072_f32_bf16`. **This matches 0053's
characterisation exactly** -- accumulator address plus one extra call after
the loop, nothing inside it.

`matmul_bf16_f32_333c4d33.o` disassembly (`objdump_{fp32C,bf16C}_rtp_matmul_bf16_f32_333c4d33.txt`):
re-diffed with the header line normalised out; **zero remaining differences**
(only the printed cache-directory path differs). Byte-identical, independently
reconfirmed.

### Claim 2 (new, this task): the emulated matmul's own rounding-mode fix-up is DEAD CODE in both builds

`mm.cc` (item 6) explicitly wraps the whole k-loop's worth of `mac()` calls
in `aie::swap_rounding(conv_even)` / `aie::set_rounding(saved)`, with a
comment naming exactly the failure mode this is meant to prevent: the
bf16->bfp16 conversion "follows the core rounding mode... default `floor`...
biased low... accumulates over the K reduction." **The compiled object
does not contain this instruction.** `grep -in "rnd|round" objdump_fp32C_rtp_matmul_bf16_f32_333c4d33.txt`
returns **zero hits** across all 309 lines, including the section headers
for `matmul_scalar_bf16_f32`, `zero_f32`, and `zero_scalar_f32` also in that
file (`crrnd_evidence.txt` in this task folder). This is not a "the
emulated path wasn't compiled in" false negative -- the same disassembly
contains `vconv.bfp16ebs8.fp32` (the actual bf16->bfp16 quantisation, at
0x186, 0x196, 0x1a6... -- the bfp16 emulation code path is unambiguously
present) and `vlda.conv.fp32.bf16` (the accfloat-widening load), i.e. this
**is** the `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` build, and the
rounding-mode set/restore that should surround it is simply absent.

Tracing why: `mmul_bf16_bf16.hpp`'s `mac()`/`mul()` (item 7) call the
**plain** `to_v64bfp16ebs8(acc)`, not `to_v64bfp16ebs8_conf(acc, rnd)` (item
8). The plain form is a bare intrinsic with no rounding-mode save/restore of
its own -- it is meant to run inside a caller-established `swap_rounding`
window, exactly as `mm.cc`'s comment says. But if LLVM's backend does not
model `set_rnd`/`get_rnd`'s register write/read as something the
`to_v64bfp16ebs8` builtin depends on (i.e. the builtin is modelled as a pure
function of its explicit vector operand, with no implicit dependency on the
ambient rounding-control register), then `aie::swap_rounding(conv_even)`
followed eventually by `aie::set_rounding(saved)` with nothing recognisably
"in between" is exactly the shape of a redundant store-pair a standard
dead-store-elimination pass removes. **This account is inferred, not proven
by an LLVM-IR-level trace** (see Problems, below) -- but the two directly
observed facts (source asks for the set/restore; compiled object has
neither) are not in question; they were read from files under
`ironenv/Lib/site-packages`, and the toolchain's own competence at emitting
this exact instruction is independently proven by `narrow_f32_bf16.o`, which
**does** contain `mov crrnd, #0xc` (three occurrences, one per tile-size
entry point, `crrnd_evidence.txt`) from the *same* Peano/aie2p_srs.h
toolchain, compiling the *same* `aie::set_rounding(conv_even)` API call
verbatim from `narrow_f32_bf16.cc` line 54.

**Consequence, if this account is right**: the emulated matmul's bf16->bfp16
quantisation of every A/B tile, every k-block, every core, runs under
whatever rounding mode is *already sitting in that core's control register*
at dispatch time -- not the `conv_even` the source asks for. `narrow_f32_bf16.cc`
(item 3, its own header comment, "the first kernel in the project written
with the mode set") is the **only** kernel in this codepath that actually
emits a working `mov crrnd`, and it never restores it (there is no matching
`set_rounding(floor)` anywhere in `narrow_f32_bf16.cc` -- confirmed by
re-reading all 111 lines).

### Why this would produce exactly what T26 has measured

- **A core that has never run `narrow()`** (any fp32-C core, for its entire
  lifetime; any bf16-C core, before its *first* output tile completes) has
  whatever rounding mode the hardware reset/context-load leaves in the
  control register. Per item 11 (the same citation trap 2b already
  established from the same file), that default is `floor` -- systematically
  biased low, compounding additively over every k-block of the K reduction.
- **A bf16-C core, from its second output tile onward**, inherits the
  `conv_even` (0xC) that its *own* first `narrow()` call wrote and never
  undid. Every subsequent `matmul()` call's bfp16 quantisation on that core
  now runs unbiased, for free, as a side effect of a completely different
  kernel's leftover state.
- **This reproduces every qualitative feature T26 has measured that 0096's
  host model could not**:
  - *Tied at stage 1* (0053's isolated single-GEMM probe, 0056's chain-probe
    stage 1, ratio 0.99-1.00): the very first tile any core computes is
    `floor` in both builds, because bf16-C's own poisoning `narrow()` call
    has not happened yet when that first tile's k-loop runs.
  - *Monotonic divergence with chain length* (0056, 0.99 -> 1.35 -> 1.46 ->
    1.67 by stage 4): from stage 2 on, bf16-C's core is stuck at `conv_even`
    (unbiased, no further degradation), while fp32-C's core is stuck at
    `floor` (biased, and that bias keeps compounding with every additional
    stage) -- a gap that should widen roughly linearly with additional biased
    reductions, matching the observed trend qualitatively.
  - *0053's split-mode shrinkage* (bf16-C's error at K=64-per-dispatch,
    host-summed, falls from 1.073e-2 at 6 blocks to 9.878e-3 at 24 blocks --
    the one sub-question 0096 explicitly could not reproduce): the *first*
    of those 6 or 24 independent dispatches is `floor` (before that core's
    first `narrow()`), every dispatch after it is `conv_even`; more
    dispatches dilutes the one bad (`floor`) sample against a growing
    majority of good (`conv_even`) ones -- shrinking average error exactly as
    block count grows.
  - *Why 0096's model could not find this*: a per-block, magnitude-calibrated
    noise model has no way to represent "a control register's value depends
    on which kernel last ran on this physical core, and persists across
    otherwise-independent dispatches." That is precisely the "hardware/kernel-level
    numerical detail a host bit-formula model cannot express" 0096's own
    conclusion named as the next thing to look for.

This account does **not** need to explain 0096 Part 2d's small, opposite-signed
effect on today's shipping plain-bf16 (non-emulated) path -- plain bf16 never
calls `to_v64bfp16ebs8`/`mac_8x8_8x8T_conf` at all (item 7's `#else` branch
uses `mac_4x8_8x8_bf16` directly on bf16 vectors, no bfp16 intermediate), so
this specific mechanism is inert there by construction; that effect's cause
remains open and unrelated.

## What was ruled out

- **0053's "the wrappers differ only in accumulator address and one narrow
  call" claim**: not refuted. It survives an independent re-diff of the raw
  disassembly (Claim 1). What it did not capture (because 0053 was not looking
  for it) is that the *matmul kernel itself*, despite being byte-identical
  between the two builds, contains a latent behaviour whose *effect* depends
  on a piece of core state the wrapper-level diff cannot see: which kernel
  last touched `crrnd` on that specific core.
- **An extended-precision accumulator register** (the task brief's opening
  hypothesis to check first): ruled out by item 9. `accfloat` is a plain
  32-bit-per-lane format on aie2p; there is no hidden extra mantissa being
  carried through `accum<accfloat,N>` that a narrower path would lose and a
  wider path would keep.
- **A second, independent rounding-mode control register for narrow vs
  matmul**: ruled out by item 10 -- both go through the identical
  `set_rnd`/`get_rnd` pair.

## Problems hit

1. **Could not confirm the DCE mechanism at the LLVM-IR level.** I traced the
   *effect* (source asks for a rounding-mode fix-up; compiled object has
   none) and a *plausible* cause (the plain `to_v64bfp16ebs8` intrinsic is
   very likely not modelled by LLVM as depending on `crrnd`, so the
   surrounding save/restore looks dead to the optimiser), but did not get a
   Peano `-S -emit-llvm` dump to prove the elimination happens where I think
   it does, or at what optimisation level. This would need a local
   recompile of `mm.cc` in isolation (no NPU involved, should be safe to do
   in a follow-up session) rather than pure header reading.
2. **Did not find/regenerate a JIT-cache pair matching 0053's exact probe
   shape.** `C:\Users\vegar\.npu\cache` currently holds ~136 directories, all
   dated 2026-08-21 through 2026-08-23 (i.e. 0053's 2026-08-20 cache entries
   have since been evicted/purged by later sessions' builds). Per this task's
   explicit "do not touch the NPU" constraint, did not rebuild the probe to
   regenerate a fresh pair. Used the objdump artifacts 0053 already extracted
   and checked into `experiments/m5-pretiled-gemm/artifacts/` instead, which
   was sufficient for both claims above since the question was about static
   code content, not fresh numbers.
3. Did not attempt to independently verify the **hardware reset value** of
   `crrnd` (item 11's citation is a software-API-default statement in
   `aie_api/aie.hpp`, not a datasheet register-reset-value spec) -- flagged
   explicitly as an assumption the proposed ablation (below) would settle
   directly regardless of the documentation's precise wording.

## Artifacts

- `diff_main_core_0_2.txt` -- full unified diff of the two builds' `main_core_0_2.o`
  disassembly (0053's checked-in artifacts, re-diffed this session).
- `crrnd_evidence.txt` -- the three greps this task's central claim rests on:
  zero `rnd`/`round` hits in the matmul+zero object; three `mov crrnd, #0xc`
  hits in the narrow object; the `vconv.bfp16ebs8.fp32`/`vlda.conv.fp32.bf16`
  lines proving the emulated code path was in fact compiled into the object
  with no rounding fix-up around it.
- Source files read only, not modified: `experiments/m5-eltwise/kernels/narrow_f32_bf16.cc`,
  `experiments/m5-pretiled-gemm/gemm_pretiled.py`,
  `C:\dev\mlir-aie\python\iron\kernels\linalg.py`,
  `C:\dev\mlir-aie\aie_kernels\aie2p\mm.cc`,
  `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\detail\aie2p\mmul_bf16_bf16.hpp`,
  `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\detail\aie2p\accum_native_types.hpp`,
  `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\tile.hpp`,
  `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\detail\aie2\tile.hpp`,
  `C:\dev\mlir-aie\third_party\aie_api\include\aie_api\aie.hpp`,
  `C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\lib\clang\21\include\aie2p\aie2p_srs.h`.

## Next

**The decisive test is on hardware, and it is sharp.** This hypothesis makes
a prediction no other T26 hypothesis makes: **priming a core's rounding-mode
register to `conv_even` before a fp32-C dispatch, with no other change to
the fp32-C design (no narrow, no bf16 transport), should close most of the
6.6x gap by itself** -- because the fix would be acting purely on the
matmul's *internal* A/B bfp16 quantisation, not on C's transport precision at
all. Concretely, two independent ablations, either sufficient:

1. **Warm-up-poison ablation.** Before running the fp32-C GEMM dispatch,
   run one throwaway `c_bf16=True` dispatch on the *same* cores/hw_context
   (same xclbin, `rtp=True`, so no context switch — CLAUDE.md trap 7b) whose
   only purpose is to execute a `narrow()` call and leave `crrnd` at
   `conv_even`, then discard its output and run the real fp32-C measurement.
   If the fp32-C error drops to bf16-C's level, the mechanism is confirmed
   directly.
2. **Explicit-`_conf` ablation** (more invasive, but conclusive): recompile
   `mmul_bf16_bf16.hpp`'s emulated `mac()`/`mul()` to call
   `to_v64bfp16ebs8_conf(a, RND_CONV_EVEN)` instead of the ambient form (a
   one-line-per-call change, forces the correct rounding regardless of core
   history) and re-run 0053's or 0056's exact probe. If this alone closes the
   fp32-C vs bf16-C gap without touching the C-transport dtype at all, the
   mechanism is confirmed and the fix belongs in `mm.cc`/`mmul_bf16_bf16.hpp`
   upstream (a real, shippable correctness fix, independent of T23's transport
   decision).

Neither ablation was run this task (no hardware access, per the task
brief). **T26 stays OPEN.**

## Proposed register update

Append to T26's entry in `research/OPEN-THREADS.md`, after the existing 0096
paragraph, before the "Closed" table:

> **PROBED FURTHER 2026-08-23** ([`0098`](TASK.md)):
> reading the kernel source directly (0096's own prescribed next step) found
> a concrete, previously-unnoticed asymmetry that no host model could see.
> `aie_kernels/aie2p/mm.cc`'s emulated-bf16 matmul explicitly wraps its
> k-loop in `aie::swap_rounding(conv_even)`/`aie::set_rounding(saved)` so the
> MAC's internal bf16->bfp16 quantisation of A/B tiles is unbiased -- but the
> **compiled object contains no such instruction**: `grep -in "rnd|round"`
> across the full disassembled matmul/zero object returns zero hits, even
> though the same disassembly clearly shows the emulated bfp16 code path
> (`vconv.bfp16ebs8.fp32`) was compiled in. Tracing why: `mmul_bf16_bf16.hpp`
> calls `to_v64bfp16ebs8(acc)` (aie2p_srs.h line 1308), the **ambient**-mode
> intrinsic, not `to_v64bfp16ebs8_conf(acc, rnd)` (line 1331), the one that
> actually saves/sets/restores the rounding register -- so the source's own
> save/restore pair around it is, on the read evidence, dead code the
> compiler removes (root LLVM-IR cause not proven; see 0098 Problems). The
> same toolchain unambiguously CAN emit this instruction --
> `narrow_f32_bf16.o` (the bf16-C epilogue) contains three `mov crrnd, #0xc`,
> one per tile-size entry point -- and never restores it. Net effect: **every
> A/B-tile bfp16 quantisation inside the emulated matmul runs under whatever
> rounding mode is already sitting in that physical core's control register**,
> not the `conv_even` the source asks for. A core that has never executed
> `narrow()` (every fp32-C core, always; a bf16-C core, before its first
> output tile) is stuck at AIE's default `floor` (aie_api/aie.hpp lines 6963,
> 6979 -- the same citation trap 2b uses) -- systematically biased low,
> compounding over the K reduction. A bf16-C core, from its second output
> tile onward, inherits the `conv_even` its own prior `narrow()` call left
> behind and never undoes -- unbiased, for free, as a side effect of an
> unrelated kernel. This single mechanism reproduces every qualitative
> feature measured so far that 0096's model could not: parity at stage 1 (no
> core has poisoned itself yet), monotonic divergence with chain length
> (fp32-C's bias keeps compounding; bf16-C's does not), and 0053's
> split-mode shrinkage (more dispatches dilutes the one `floor`-quantised
> first dispatch against a growing majority of `conv_even` ones) -- the one
> sub-question 0096 explicitly could not reproduce. **Not yet confirmed on
> hardware.** Two ablations are proposed and neither has been run: (1) warm
> up a fp32-C core with a throwaway `narrow()`-calling dispatch before the
> real measurement, predicting most of the 6.6x gap closes without touching
> C's transport dtype at all; (2) patch `mmul_bf16_bf16.hpp` to call
> `to_v64bfp16ebs8_conf` instead of the ambient form, predicting this alone
> (no C-dtype change) closes the gap and is a real upstream correctness fix.
> **T26 remains OPEN** pending either ablation; this is the strongest
> evidenced lead so far, not a confirmed answer.

`tasks/README.md` index row:

```
| [0098](0098-t26-kernel-source/TASK.md) | T26 -- reading `aie_kernels/aie2p/mm.cc` and `aie2p_srs.h` finds the emulated matmul's own rounding-mode fix-up is dead code: the compiled object never sets `crrnd`, so its bf16->bfp16 quantisation runs under whatever mode a PRIOR kernel left on that core -- `narrow_f32_bf16`'s un-restored `conv_even` write is the leading candidate for the whole 6.6x anomaly. Two hardware ablations proposed, neither run; T26 stays OPEN | research (T26) | done |
```
