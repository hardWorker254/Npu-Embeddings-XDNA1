# 0099 — T26: the warm-up-poison ablation — NOT CONFIRMED, and the reason why is itself the finding

- **Date** 2026-08-23
- **Milestone** post-M13 (research thread, `research/OPEN-THREADS.md` T26)
- **Status** done — **negative result for 0098's specific mechanism, and a
  bigger, unresolved finding underneath it**: today's freshly-compiled
  emulated-matmul kernel object is NOT the one 0098 examined. It correctly
  contains the `crrnd` save/set/restore triple `mm.cc`'s source asks for,
  and (independently) today's unmodified re-run of `t26_chain_probe.py`
  does not reproduce the original 0.99->1.67x divergence either. T26 stays
  OPEN, but the open question has shifted.

## Goal

Test 0098's hypothesis directly on hardware: **warm up an fp32-C core with a
throwaway `narrow()`-calling dispatch before the real measurement, and see
whether most of the 6.6x accuracy gap (bfp16+bf16-C vs bfp16+fp32-C,
[`0052`](../0052-m10-research-night/TASK.md) S6, `1-cos` 2.395e-03 ->
3.615e-04) closes without touching C's transport dtype at all.** If it does,
[`0098`](../0098-t26-kernel-source/TASK.md)'s account (the emulated matmul's
`aie::swap_rounding(conv_even)`/`set_rounding(saved)` around its k-loop is
dead code because `mmul_bf16_bf16.hpp` calls the ambient-mode
`to_v64bfp16ebs8` instead of the `_conf` form, so every core's bfp16
quantisation runs under whatever `crrnd` a *prior* kernel left behind) is
confirmed, and T26 would be the first of five hypotheses to close.

## Context

- [`0052`](../0052-m10-research-night/TASK.md) measured bfp16+bf16-C beating
  bfp16+fp32-C 6.6x on production MiniLM.
- [`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md) refuted k-block-boundary
  re-quantisation and showed the matmul kernel object is byte-identical
  between the two builds; found the anomaly does NOT appear on an isolated
  single GEMM (near-tie at full K).
- [`0056`](../0056-m10-t26-rounding-and-chain-probe/TASK.md) refuted a
  floor-vs-conv_even asymmetry in the *host* narrowing path, then built
  `t26_chain_probe.py` (4 chained GEMM stages, one constant shape, only 2
  device builds) and measured genuine, monotonic compounding: `1-cos` ratio
  (fp32C/bf16C) 0.99 -> 1.35 -> 1.46 -> 1.67 across stages 1-4.
- [`0096`](../0096-t26-numerical-mechanism/TASK.md) built a calibrated
  numpy/fp64 model of "narrow position applied to independent per-block
  bfp16 noise" and it failed its own control (ratio pinned at 0.99-1.00 at
  every stage, every mantissa calibration) -- ruling that class of mechanism
  out and concluding the missing ingredient is a hardware/kernel-level detail
  a host model cannot express.
- [`0098`](../0098-t26-kernel-source/TASK.md) read the kernel source and the
  checked-in objdump artifacts from 0053's 2026-08-20 run
  (`experiments/m5-pretiled-gemm/artifacts/objdump_{fp32C,bf16C}_rtp_*.txt`)
  and found: `mm.cc` wraps the emulated k-loop in
  `aie::swap_rounding(conv_even)`/`set_rounding(saved)`, but a grep for
  `rnd`/`round` across the disassembled matmul object found **zero hits**,
  while `narrow_f32_bf16.o` (the bf16-C epilogue) has three `mov crrnd,
  #0xc` and never restores. Traced the likely cause to
  `mmul_bf16_bf16.hpp` calling the ambient-mode `to_v64bfp16ebs8` intrinsic
  instead of the `_conf` form. Proposed exactly this ablation, unrun.

## What was done

### 1. Design decision: not the literal ablation 0098 sketched

0098's "Next" section proposed warming up with a `c_bf16=True` dispatch on
"the same xclbin", then switching to the `c_bf16=False` design to measure.
Read `experiments/m5-pretiled-gemm/gemm_pretiled.py`'s two `rtp` core_fn
branches (lines ~404-470 before this task's edits) and confirmed **fp32-C
and bf16-C are different compiled objects**, not the same xclbin -- the
bf16-C wrapper has an extra basic block and a different accumulator address
(0053's own finding, independently re-confirmed in 0098). Warming up on one
xclbin and measuring on another crosses a full context switch (CLAUDE.md
trap 7b), and it is genuinely unknown whether `crrnd` (a per-core hardware
control register) survives that switch. A negative result from *that*
design would be ambiguous between "mechanism wrong" and "crrnd resets on
context switch, mechanism untested."

So this task adds a **third** `gemm_pretiled.py` worker variant, `poison`
(`rtp=True, c_bf16=False, poison=True`): structurally the plain fp32-C
worker -- the real output goes straight into the C ObjectFifo, C's
transport dtype never changes -- plus one extra, discarded call **after**
each dispatch's real output is released: `narrow()` a scratch accumulator
into a scratch tile connected to no ObjectFifo (never DMA'd, never read).
This makes "cold" and "warmed" fp32-C literally the same xclbin, the same
hw_context, dispatched twice in the same process -- byte-for-byte identical
machine code both times. There is no cross-design core-matching question to
answer, because there is only ever one design involved for the
cold-vs-warm comparison; "did the warm-up land on the right cores" is true
by construction, not by inspection.

Implementation (`gemm_pretiled.py`, `_build_design`/`pretiled_array`):
- Added `poison: CompileTime[bool] = False` end to end.
- Asserted mutual exclusion with `c_bf16` and `epilogue` (both own the
  post-K step).
- Split the narrow-kernel construction: `c_bf16`'s narrow kernel still
  matches the real tile (`m*n`, required -- it narrows the real output).
  `poison`'s narrow kernel uses a **separate, smaller** entry point
  (see Problems #1) sized independently of the real tile, since nothing
  about its *values* matters, only that the instruction executes.
- New `elif poison:` branch in the `rtp` core_fn dispatch: same body as the
  plain worker, plus `poison_narrow(poison_acc, poison_out)` called once,
  right after `out_c.release(1)` -- after this dispatch's own real output,
  so it can only poison a *later* dispatch on this core, never its own.
  No `zero()` on the scratch accumulator (Problems #1 explains why that is
  fine, not sloppy).

### 2. Cache-identity fix (the marker trap, seventh near-miss)

`poison=True` changes no tensor memref shape in the generated MLIR -- C's
transport dtype is fp32 in both the plain and the poison design -- so
`t26_probe.py`'s existing M/K/N/dtype marker cannot tell a poison build
apart from a plain rtp=True fp32-C build at the same shape. Added an
explicit, unique marker (the `poison_acc_0_0` buffer name, confirmed by
`grep -c` against the actual `aie.mlir` before trusting it -- see Problems
#2) used as a REQUIRE-marker for poison lookups and an EXCLUDE-marker for
non-poison lookups, so `purge()`/`find_cache()` can never confuse the two.

### 3. NPU contention check

```
& "C:\Windows\System32\AMD\xrt-smi.exe" examine --report aie-partitions
```
Five `WorkloadsSessionHost.exe` hw_contexts listed, all **Idle**, 0
`Submissions` in flight (this is Windows/Ryzen-AI-SW background driver
housekeeping, not another agent's workload -- no `python.exe` process held
a context). Ran throughout the session with no other workload observed.

### 4. Prebuild + verify (cache identity by CONTENTS, trap 7c)

`python t26_ablation.py --prebuild` purges, force-builds both designs
(`poison` and `bf16c`) at M=256,K=384,N=192, tile 64x64x48, cols=4, then:
- reads `aie.mlir` from the resolved cache dir and asserts
  `"poison_acc_0_0" in text` matches the expected flag exactly (True for
  `poison`, False for `bf16c`) -- **this is the identity check**, not the
  directory's mtime.
- objdumps every `*.o` in that cache dir and greps for
  `crrnd|\brnd\b|round`.

**This is where the session's actual finding is.** The `poison` build's
`matmul_bf16_f32_333c4d33.o` -- same kernel-object filename 0098's checked-in
artifacts used -- disassembles to **3 rnd/round/crrnd matches**, not zero:

```
36:  mova r19, #0x180; movx r18, #0x6;  mov r22, crrnd     <- SAVE ambient mode
4c:  ...                                mov crrnd, #0xc    <- SET conv_even
37c: mov crrnd, r22                                        <- RESTORE
```

This is exactly the save/set/restore triple `mm.cc`'s source asks for and
0098 said the compiled object was missing. Independently re-grepped 0098's
own checked-in artifact
(`experiments/m5-pretiled-gemm/artifacts/objdump_fp32C_rtp_matmul_bf16_f32_333c4d33.txt`,
309 lines) with a clean, unambiguous `grep -c "crrnd"` (no regex-alternation
risk): **zero hits, confirmed** -- so 0098's finding on *that specific
2026-08-20 build* was real, not a grep-syntax artifact on either side. But
the object compiled **today**, from the identical unmodified source tree
(`C:\dev\mlir-aie` untouched, read-only per CLAUDE.md) and the identical
installed package (`mlir-aie 1.4.2.dev16+g7e00b57`, checked with `pip show
mlir_aie` -- same version string CLAUDE.md already records, so this is not
a toolchain upgrade), **is a different compiled object with the fix-up
present**. Reproduced on **three independent purge+rebuild cycles** in this
session (first prebuild attempt failed on an unrelated L1 overflow before
reaching objdump, Problems #1; the two successful rebuilds after the fix,
plus an explicit third rebuild-and-diff, all agree: crrnd save/set/restore
present, byte-stable across the three).

Given this, checked whether the underlying *anomaly* (not just 0098's
proposed mechanism for it) still reproduces at all today, using the
**unmodified** existing probe (per the task brief: reuse the harness, don't
build a new one):

```powershell
python t26_chain_probe.py --out artifacts\t26_ablation\t26_chain_probe_rerun.json
```

```
stage   fp32C rel_fro   bf16C rel_fro  ratio (fp32C/bf16C)
    1    9.552452e-03    9.703122e-03                0.984
    2    1.346220e-02    1.359894e-02                0.990
    3    1.650642e-02    1.663694e-02                0.992
    4    1.927234e-02    1.940127e-02                0.993
```

**This does not reproduce 0056's measurement at all.** 0056 measured this
exact probe (same shape, same script, unmodified since) climbing 0.99 ->
1.35 -> 1.46 -> 1.67. Today's rerun is **flat at 0.98-0.99** -- fp32-C is
marginally *better* than bf16-C throughout, the opposite direction from
both 0056's original finding and 0052's production anomaly. This is
internally consistent with the disassembly finding: if today's matmul
kernel already saves/sets/restores `conv_even` around its own k-loop,
independent of ambient state, there is no core-history-dependent bias left
for either 0098's mechanism or 0056's structural narrow-position hypothesis
to act on.

### 5. The ablation itself

`python t26_ablation.py --mode {cold,warm,bf16c,plain_check} --seed N --out
...` (M=256,K=384,N=192, tile 64x64x48, cols=4 -- 0053's isolated-GEMM
shape). Each invocation is a **separate process** (a fresh `hw_context`),
on the stated assumption that opening a new context resets per-core control
state -- this assumption is exactly what `plain_check` cross-checks (see
Result).

- `cold`: one dispatch, the first thing this process's context ever runs.
  Measured directly.
- `warm`: one throwaway dispatch (independent random discard data, `seed +
  999_000`) on the `poison` design, then a second dispatch (same process,
  same context, no rebuild) with the real data. Second dispatch measured.
- `bf16c`: same warm-up-then-measure protocol, on the unmodified `c_bf16`
  design -- 0053 already showed a bf16-C core's *own first* dispatch is
  tied with cold fp32-C (its own narrow hasn't run yet), so measuring
  bf16-C at its *own* first dispatch would not be the right operating
  point; production's regime (many chained GEMMs) is closer to "already
  settled," which this protocol reaches in one extra throwaway dispatch.
- `plain_check`: one dispatch of the **original, unmodified** fp32-C design
  (`rtp=False`, no poison plumbing at all, narrow() does not exist anywhere
  in its compiled code) -- a cross-check that the poison plumbing itself
  introduces no side effect, and that "fresh process" really does give a
  clean baseline.

Ran 5 seeds (0-4) per mode, 20 process invocations total.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm

# contention check
& "C:\Windows\System32\AMD\xrt-smi.exe" examine --report aie-partitions

# prebuild + cache-identity + disassembly verification (run 3x this session)
python t26_ablation.py --prebuild

# cross-check: does the ORIGINAL anomaly still reproduce today, unmodified script
python t26_chain_probe.py --out artifacts\t26_ablation\t26_chain_probe_rerun.json

# the ablation: 5 seeds x 4 modes, one process per invocation
foreach ($seed in 0..4) {
  foreach ($mode in "cold","warm","bf16c","plain_check") {
    python t26_ablation.py --mode $mode --seed $seed --out "artifacts\t26_ablation\${mode}_${seed}.json"
  }
}
```

## Result

Mean `rel_fro` over 5 seeds (M=256,K=384,N=192, tile 64x64x48, cols=4):

| condition | mean rel_fro | vs cold |
|---|---:|---:|
| **cold fp32-C** (poison design, 1st dispatch, never touched narrow) | 9.44211e-03 | 1.000x |
| **warmed fp32-C** (poison design, 2nd dispatch, warmed by its own throwaway narrow) | 9.44211e-03 | **1.000x -- bit-identical to cold, per seed, to full float64 precision** |
| **bf16-C** (unmodified design, 2nd dispatch, warmed by its own real narrow) | 9.59035e-03 | 1.016x (**worse**, not better) |
| plain_check (original unmodified fp32-C, `rtp=False`) | 9.44211e-03 | matches cold exactly, every seed |

Per-seed: `cold_{seed}.json` and `warm_{seed}.json` agree to every printed
digit for all 5 seeds (e.g. seed 0: both `0.009431754404363746`). `bf16c`
sits consistently ~1.5-2% *worse* than fp32-C at every seed, not better --
matching 0053's original "isolated single GEMM, statistically tied, a hair
worse for bf16-C" finding almost exactly, and the opposite sign from
production's 6.6x.

**Verdict: the ablation does NOT confirm 0098's mechanism.** The predicted
effect ("most of the 6.6x gap closes") produced a clean, exact null instead
-- not a small effect, a *zero* effect: warming a core with a throwaway
`narrow()` call changes nothing measurable about its subsequent matmul
output, at every seed tested.

This null is not a mystery sitting next to the disassembly finding -- it is
**exactly what the disassembly finding predicts**. If the compiled matmul
kernel already saves the ambient rounding mode, sets `conv_even`, runs the
whole k-loop, and restores the saved mode -- all inside its own call, every
time -- then no amount of *prior* core history can matter, because the
kernel never depends on it. `cold == warm` bit-for-bit is the correct,
mechanical consequence of "the `crrnd` fix-up is present and self-contained
in the compiled code I am running," not evidence the mechanism is
impossible in general.

**What this task cannot settle**: why the compiled object 0098 examined
(2026-08-20, checked-in, independently re-confirmed here to genuinely lack
the fix-up) differs from the object built here, today, from the identical
source tree and package version. Considered and set aside as out of budget
for this task:
- **Toolchain update**: ruled out -- `pip show mlir_aie` reports
  `1.4.2.dev16+g7e00b57`, the exact string CLAUDE.md already records; no
  `update-mlir-aie` run is logged between 0098 and this task.
- **Stale/reused kernel-object cache serving an old .o**: considered,
  because `~/.npu/cache` is per-design-dir but kernel `.o` files are
  filenames shared across dirs by content-hash-like naming
  (`matmul_bf16_f32_333c4d33.o`). Checked whether any surviving cache dir
  still held the old, buggy 2026-08-20 object -- it does not; that cache
  dir was already evicted (0098's own Problems #2 already noted this).
  Cannot rule this class of explanation in or out further without a kernel
  cache location this task did not locate.
- **Peano/LLVM non-determinism** (e.g. iteration order over an internal
  hash-keyed data structure affecting a dead-code-elimination decision) is
  the leading remaining candidate but is **not verified** -- three rebuilds
  *today* were stable (crrnd present all three times), which rules out
  "flip-flops on every compile" but not "flipped once, non-deterministically,
  sometime between 2026-08-20 and today and is now stable in the other
  state."

This is flagged prominently in the register update below because **it
matters beyond T26**: 0053's "the matmul kernel object is byte-for-byte
identical between the fp32-C and bf16-C builds" claim -- load-bearing for
ruling out the k-block-boundary hypothesis -- was itself established by
diffing two objdumps from a single build session. If Peano's codegen for
this kernel can differ between otherwise-identical builds at all, that
specific byte-identity claim was true *for that session*, and this task's
own prebuild independently reproduced it (`bf16c` and `poison` builds
today objdump identically to each other) -- but the general assumption
"same source + same flags + same package version -> same object, always"
should not be relied on without a dedicated determinism check.

## Problems hit

1. **First `--prebuild` attempt: L1 overflow.** Sizing the poison scratch
   pair (`poison_acc`, `poison_out`) at the real tile's `m*n=3072` elements
   (mirroring `c_bf16`'s accumulator) pushed the design to 75,016 B of the
   63 KB (64,512 B) L1 budget (CLAUDE.md trap 3):
   ```
   error: 'aie.tile' op allocated buffers exceeded available memory
     poison_acc_3_3  : 0x6D00-0x9CFF (12288 bytes)
     ...
     poison_out_3_3  : 0x10D00-0x124FF (6144 bytes)
   ```
   Fix: nothing about the poison call's *values* matters, only that the
   instruction executes -- so its scratch pair does not need to match the
   real tile size at all. Re-sized to the smallest legal `narrow_*_bf16.cc`
   entry point (1024 elements, 6,144 B total) via a **separate**
   `poison_narrow_kernel` object (distinct from `c_bf16`'s, which still
   must match the real tile). This also meant dropping the `zero()` call on
   `poison_acc` before narrowing it (its `zero_kernel` is shaped for the
   *real* m*n and would not match the smaller scratch buffer) -- narrowing
   whatever bits are already sitting in an uninitialised `Buffer` is exactly
   as good a throwaway as narrowing zeros, since the output is never read.
   Second attempt built clean at ~62.7 KB.

2. **Cache-marker false negative.** Copied `t26_probe.py`'s second marker,
   `f"<size = {TK}, stride = {TN}>"`, verbatim. `grep -c` against the actual
   `aie.mlir` for this shape returned **0** -- the literal substring does
   not appear (this shape's `(s,t)` sub-tile factoring renders differently
   from whatever shape that marker was calibrated against). Would have
   caused `find_cache` to raise "0 candidates" on every lookup, which it
   did, loudly, rather than silently matching the wrong thing -- but still
   fixed by dropping that marker rather than debugging its exact format,
   since the `aie.runtime_sequence` signature line plus the poison-specific
   `poison_acc_0_0` marker are already sufficient and independently
   `grep -c`-confirmed present (see `crrnd_evidence_0099.txt`).

3. **The disassembly discrepancy itself (detailed in Result)** is really a
   "problem hit" in the sense that it invalidated the ablation's original
   premise mid-task -- there turned out to be no gap to close. Not treated
   as a blocker: the task brief explicitly anticipates a non-confirming
   outcome ("If it does NOT confirm... Say so plainly") and this is not
   confounded evidence, it is a clean explanation for a clean null.

4. **Local-workaround test not run.** The brief also asked to test whether
   explicitly calling `aie::set_rounding(conv_even)` inside the kernel
   wrapper (where the compiler cannot elide it) restores correct behaviour
   "if it is cheap." Given today's compiled kernel already contains the
   correct save/set/restore, there is currently no incorrect behaviour to
   fix, so this test has no baseline to improve on and was not run. It
   remains a well-defined, cheap follow-up **if** a future rebuild is ever
   caught reproducing 0098's crrnd-absent state again.

## Artifacts

All under `experiments/m5-pretiled-gemm/`:
- `gemm_pretiled.py` -- modified, checked in: added the `poison` worker
  variant end to end (`_build_design`, `pretiled_array`). Existing
  `c_bf16`/plain/`gelu` paths untouched (verified: `plain_check`'s numbers
  match `cold`'s exactly, i.e. the plain fp32-C path's behaviour did not
  change).
- `t26_ablation.py` -- new, checked in. The ablation harness (`--prebuild`,
  `--mode {cold,warm,bf16c,plain_check}`).
- `artifacts/t26_ablation/`:
  - `prebuild_verify.json` -- cache-identity + objdump summary.
  - `objdump_poison_matmul_bf16_f32_333c4d33.txt`,
    `objdump_bf16c_matmul_bf16_f32_333c4d33.txt`,
    `objdump_poison_matmul_rebuild3.txt` -- full disassembly, three
    independent builds, all agreeing.
  - `objdump_poison_narrow_1024_f32_bf16.txt`,
    `objdump_bf16c_narrow_3072_f32_bf16.txt`.
  - `crrnd_evidence_0099.txt` -- the four greps Result's central claim
    rests on, including the re-confirmation of 0098's original artifact.
  - `t26_chain_probe_rerun.json` -- today's unmodified rerun of 0056's
    chain probe (flat ~0.98-0.99, does not reproduce 0056's 0.99->1.67).
  - `cold_{0..4}.json`, `warm_{0..4}.json`, `bf16c_{0..4}.json`,
    `plain_check_{0..4}.json` -- all 20 trial results, raw.
  - `ablation_summary.json` -- the aggregated means in the Result table.
- 0098's own artifacts (`experiments/m5-pretiled-gemm/artifacts/objdump_fp32C_rtp_matmul_bf16_f32_333c4d33.txt`
  etc.) -- read only, re-grepped, not modified.

## Next

The immediate ablation question is answered (does not confirm). The bigger,
newly-opened question -- **why did today's build of the identical kernel
source, at the identical package version, differ from 0098's 2026-08-20
build, and does the underlying 0052/0056 production anomaly still
reproduce at the full MTEB/e2e level today?** -- is not. Two concrete
follow-ups, neither attempted here (out of this task's budget):
1. A determinism check: rebuild the same kernel N times across fresh
   processes (this task only got 3 same-session samples, all agreeing) and
   see whether it ever reverts to the crrnd-absent form. If it does, that
   is Peano/LLVM non-determinism affecting a *correctness*-relevant
   instruction, not just a performance one -- worth its own thread.
2. Re-run 0052's actual production-scale measurement (not just the isolated
   GEMM / synthetic chain probe) to see whether the 6.6x anomaly itself
   still reproduces today, now that the chain probe's synthetic version of
   it does not.

## Proposed register update

**T26 verdict: still OPEN**, but reframed. 0098's specific mechanism
(ambient-mode DCE in `mmul_bf16_bf16.hpp`) is **refuted as a currently
active cause** -- not because the reasoning was wrong (0098's read of the
2026-08-20 build is independently re-confirmed here, byte for byte), but
because the compiled artifact it describes is not the one produced by an
identical rebuild today.

Suggested replacement text for T26's entry (append after the existing 0098
paragraph, before the "Closed" table):

> **PROBED FURTHER 2026-08-23** ([`0099`](TASK.md)):
> the warm-up-poison ablation 0098 proposed was run on hardware -- a new
> `gemm_pretiled.py` worker variant (`poison`) that is byte-for-byte the
> plain fp32-C worker plus one throwaway, discarded `narrow()` call after
> each dispatch's real output is released, so "cold" and "warmed" fp32-C are
> literally the same compiled xclbin dispatched twice in the same process,
> avoiding the cross-context ambiguity 0098's literal proposal would have
> had. **Result: cold and warmed fp32-C are bit-identical, per seed, across
> 5 seeds** (`rel_fro` 9.44211e-03 both, to full float64 precision) -- not a
> small effect, an exact null -- and bf16-C (measured at its own
> already-warmed operating point, for a fair comparison) is **1.6% worse**,
> not 6.6x better. **The ablation does not confirm 0098's mechanism.**
>
> But the reason is not a refutation of the reasoning -- it is that **the
> premise no longer holds on a fresh build**. Objdumping today's freshly
> purge-and-rebuilt `matmul_bf16_f32_333c4d33.o` (same kernel, same shape,
> same flags 0098 examined) shows the `crrnd` save/set/restore triple
> `mm.cc`'s source asks for **IS present** (`mov r22, crrnd` /
> `mov crrnd, #0xc` / `mov crrnd, r22`, three independent purge+rebuild
> cycles this session, all agreeing) -- where 0098's checked-in artifact
> from the 2026-08-20 build (independently re-confirmed here with a clean,
> unambiguous `grep -c "crrnd"`, zero hits) genuinely lacked it. Same
> unmodified source tree (`C:\dev\mlir-aie`, read-only), same installed
> package version (`mlir-aie 1.4.2.dev16+g7e00b57`, matching CLAUDE.md) --
> no toolchain update is logged between the two. **Consistent with this**:
> an unmodified re-run of `t26_chain_probe.py` (0056's exact script, same
> shape) today measures the fp32C/bf16C ratio **flat at 0.98-0.99 across
> all 4 stages**, not climbing to 0056's measured 1.67x. If the matmul
> kernel now self-contains a correct, history-independent rounding fix-up,
> there is no core-history-dependent bias left for 0098's mechanism (or
> 0056's structural narrow-position hypothesis) to act on -- exactly what
> both the ablation's null and the chain-probe's non-reproduction show.
>
> **This raises a question bigger than T26**: why would an identical
> source tree and package version produce a different compiled kernel
> object on two separate days? Not settled here (toolchain update ruled
> out; stale kernel-object cache serving an old `.o` considered but not
> confirmed or ruled out; Peano/LLVM codegen non-determinism is the leading
> remaining candidate, unverified). If codegen for this kernel is not fully
> deterministic given identical inputs, **0053's "the matmul kernel object
> is byte-for-byte identical between the fp32-C and bf16-C builds" claim**
> -- load-bearing for ruling out the k-block-boundary hypothesis -- held
> *for that session* (independently reproduced again this session: today's
> `poison` and `bf16c` builds objdump identically to each other) but should
> not be assumed to hold in general without a dedicated determinism check.
> **T26 remains OPEN.** What is now needed is not another mechanism
> hypothesis but (1) a determinism check on this exact kernel across
> several independent fresh-process rebuilds, and (2) a re-run of 0052's
> actual production-scale (not synthetic-probe) measurement, to establish
> whether the 6.6x anomaly itself still reproduces today at all.

**T23 consequent note.** No change to T23's PASS/FAIL gate table -- 0052's
numbers stand as a historical measurement. But T23's framing should now
also note: the 6.6x accuracy gain's own reproducibility is in question, not
just its mechanism -- a re-run of the numbers T23 is gated on is advisable
before shipping `--c-bf16` on that basis, given today's synthetic chain
probe (same script, same shape as 0056's) does not reproduce the effect at
all.

`tasks/README.md` index row:

```
| [0099](0099-t26-rounding-ablation/TASK.md) | T26 -- ran 0098's proposed warm-up-poison ablation on hardware via a new same-xclbin `poison` worker variant: cold and warmed fp32-C are BIT-IDENTICAL (not a small effect, an exact null), refuting 0098's specific mechanism. But a fresh rebuild of the exact kernel 0098 examined now correctly contains the `crrnd` save/set/restore triple that 0098's checked-in build (independently re-confirmed) lacked -- and an unmodified re-run of `t26_chain_probe.py` no longer reproduces 0056's 0.99->1.67 divergence either. Root cause of the build difference not settled (toolchain version unchanged); raises a determinism question bigger than T26. T26 stays OPEN, reframed | research (T26) | done |
```
