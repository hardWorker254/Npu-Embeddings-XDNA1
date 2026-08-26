# 0101 — T23: re-measuring the bfp16 datapath's accuracy on mlir-aie 1.4.2

- **Date** 2026-08-23
- **Milestone** post-M13 (research thread, `research/OPEN-THREADS.md` T23)
- **Status** done — all six cells measured (3 configs x 2 models), both gates,
  on hardware, same session. **Corrected mid-task**: the first pass computed
  MTEB deltas by hand across separate `--sides npu`-only invocations, which
  is exactly the fail-open `run_mteb.py`'s own M8 gate guard exists to
  prevent (its "NO GATE" message says so explicitly). The coordinator caught
  this before it was accepted. All four decisive MTEB cells were re-run
  properly with `--sides cpu,npu` in one invocation each — see "Problems
  hit" #1 and Result §2 for the full correction; the real gate results
  confirmed the hand-computed estimates to within 0.02 points on every cell,
  so the verdict is unchanged, but it now rests on actual gate output rather
  than arithmetic performed outside the tool the project uses to gate on.

## Goal

T23's entire accuracy case for the bfp16-emulated MMAC datapath (`tasks/0035`,
`0052`, `0053`) was measured on mlir-aie **1.3.4**, which — per T26's closure
today (`tasks/0099`) — never emitted the `crrnd` rounding-mode control the
emulated matmul's own source (`mm.cc`) asks for. 1.4.2 does emit it (the
1.3.4 -> 1.4.2 migration landed **the same day** as the artifacts that
carried the anomaly, `42da31d` at 16:05:34 vs `efd8c76` at 12:15:45). So every
number T23 currently cites is toolchain-stale. This task re-measures the
bfp16 datapath's accuracy from scratch on today's toolchain and states
whether the datapath decision should change.

## Context

- `research/OPEN-THREADS.md` T23 — the standing entry, citing `0049`
  (2.9x/1.74-2.20x speed), `0052` (bfp16+fp32C FAILS 1-cos at 2.395e-03,
  bfp16+bf16C PASSES at 3.615e-04 — a 6.6x gap with no known mechanism at
  the time), `0053` (bge-base bfp16+bf16C MTEB passes).
- `research/CLOSED-THREADS.md` T26, closed today by `tasks/0099`: the 6.6x
  gap does not reproduce on a fresh 1.4.2 rebuild (bit-identical
  cold-vs-warmed fp32-C; bf16-C measured 1.6% *worse*, not 6.6x better), and
  a fresh objdump of the exact kernel config hash `333c4d33` that 0098's
  2026-08-20 build lacked the `crrnd` triple in now contains it (0 -> 3
  hits). **Conclusion travelling into this task**: the old FAIL for plain
  bfp16+fp32-C may have been the `floor`-bias bug, not a datapath accuracy
  ceiling — worth re-measuring directly rather than assuming.
- `tasks/0094` (T32) changed the golden gate **today**: it now compares all
  128 rows (previously 4) with per-copy-rotated golden content. Confirmed
  there to leave the *healthy* production numbers unchanged to the digit,
  but a *degraded* path's number is not directly comparable to a pre-0094
  figure. This matters below because none of T23's own reference numbers
  were taken with the current gate.
- `tasks/0035` established MTEB (5 tasks, seq 64, delta vs the same
  checkpoint on CPU) as the accuracy **authority** for a datapath decision,
  over the `1-cos` fidelity number.

## What was done

1. **Verified the NPU was uncontended** before starting and again at the
   end: `xrt-smi examine --report aie-partitions` — five `WorkloadsSessionHost.exe`
   hw_contexts, all `Idle`, 0 in-flight submissions, no `python.exe`/
   `npuembed.exe` holding a context (background driver housekeeping, not a
   competing workload). Every `npuembed.exe` run below also printed its own
   contention guard: `npu exclusive -- 7 hw_context(s), none Active but ours`.

2. **Control run (production plain bf16, fp32-C), both models** — must
   reproduce the currently published number or stop. It did, to the digit,
   against `tasks/0094`'s post-fix numbers (the only ones comparable — see
   Result §4):

   | model | `rel_fro` (this run) | `rel_fro` (0094) | worst 1-cos (128 rows) |
   |---|---:|---:|---:|
   | `all-MiniLM-L6-v2` (`artifacts_b128il`) | 4.473e-03 | 4.473e-03 | 1.086e-05 |
   | `bge-base-en-v1.5` (`artifacts_base`) | 4.297e-03 | 4.297e-03 | 1.353e-05 |

   **Bit-identical to the digit both times this run was repeated** (once at
   the start, once again at the end to bracket the session — see
   `gate_minilm_control.txt` / `gate_base_control.txt`, and the earlier
   in-session prints in `gate_minilm_bfp_fp32c.txt`'s sibling log, all
   agreeing). Harness confirmed correct before touching bfp16.

3. **Built four bfp16 artifact sets**, one per (model x C-transport-dtype)
   cell, matching production's own build recipe exactly (`tasks/0032`,
   `0051`) plus the research flag:

   ```powershell
   cd C:\dev\mlir-aie
   . .\iron_env.ps1
   cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

   python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --emulate-bfp16 --out runtime\artifacts_minilm_bfp_fp32c
   python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --emulate-bfp16 --c-bf16 --out runtime\artifacts_minilm_bfp_cbf16
   python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --hidden 768 --emulate-bfp16 --out runtime\artifacts_base_bfp_fp32c
   python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --hidden 768 --emulate-bfp16 --c-bf16 --out runtime\artifacts_base_bfp_cbf16
   ```

   All four passed the script's own xclbin-identity-mod-UUID check across
   all 16 streams (4 shapes x 4 batch tiers) — the same one-context-per-model
   invariant production relies on. No `--int8`; unrelated to this task.

4. **Verified the emulation is genuinely on, from the built object, not the
   flag** (trap 1) — AIE `llvm-objdump`, not the MSVC one on PATH:

   ```powershell
   C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\bin\llvm-objdump.exe -d `
       "<cache-dir>\matmul_bf16_f32_333c4d33.o"
   ```

   All four builds' matmul kernel objects disassemble to the **same config
   hash `333c4d33`** (consistent with 0098/0099 — the kernel object does not
   depend on M/N/hidden, only the tile shape) and to **20 hits of
   `vconv.bfp16ebs8.fp32`** (the bfp16 datapath instruction) and **3 hits of
   `crrnd`** (the save/set/restore triple `mm.cc`'s source asks for — 0 in
   0098's stale 2026-08-20 artifact, 3 in every rebuild this session and in
   0099's). The two bf16-C builds' `narrow_3072_f32_bf16.o` epilogue also
   shows 3 `crrnd` hits (sets `conv_even`, matches 0098's finding). **These
   are fixed-toolchain numbers, not assumed** — every one of the four builds
   was independently objdumped this session.

5. **Golden gate**, all four bfp16 builds, both models — `.\build\npuembed.exe
   .. --model <model> --artifacts <artifacts> --threads 16`.

6. **`--bench 5`** on all six configs (2 controls + 4 bfp16) for the array
   GEMM time and end-to-end wall clock, labelled per rule 1.

7. **MTEB, 5 tasks, first pass — WRONG, see Problems hit #1** —
   `experiments/m8-npu-vs-cpu/run_mteb.py`, `.venv-ref`. Ran `--sides cpu,npu`
   once per model (the control cell), then `--sides npu` for the two bfp16
   cells per model, and computed the delta table **by hand** against the
   control run's CPU column from a *different invocation*:

   ```powershell
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides cpu,npu --model all-MiniLM-L6-v2 --artifacts artifacts_b128il `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_minilm_control.json
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides npu --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp_fp32c `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_minilm_bfp_fp32c.json
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides npu --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp_cbf16 `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_minilm_bfp_cbf16.json
   # same pattern for bge-base-en-v1.5 / artifacts_base / artifacts_base_bfp_fp32c / artifacts_base_bfp_cbf16
   ```

   Every one of these four `--sides npu` runs prints `NO GATE -- this run
   has no delta to gate on: cpu side(s) not run` and its JSON's `per_side`
   has no `cpu` key at all — the tool's own fail-open guard, working exactly
   as designed, and initially misread as "no gate verdict to report" rather
   than "this is not a valid comparison, stop and re-run." See Problems hit
   #1 for the correction.

   All runs used the tool's default `--pipeline 2` (0052's bfp16 MTEB run
   used `--pipeline 1`; 0033 established lanes are bit-identical, and using
   one consistent value across all six cells here matters more than matching
   0052's choice).

8. **A borderline MTEB result triggered a determinism check** (Result §2):
   `bge-base` bfp16+fp32-C's worst task (`TwentyNewsgroupsClustering`) landed
   at −0.578, just past the gate's −0.5 threshold — a verdict-flipping
   number, so it was re-run standalone rather than accepted on one sample:

   ```powershell
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides npu --tasks TwentyNewsgroupsClustering --model bge-base-en-v1.5 `
       --artifacts artifacts_base_bfp_fp32c `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_base_bfp_fp32c_clustering_repeat.json
   ```

   Result: **`0.4999837157304225` both times, bit-identical to the last
   printed digit** (`mteb_base_bfp_fp32c.json` vs
   `mteb_base_bfp_fp32c_clustering_repeat.json`). NPU dispatch is
   deterministic given fixed weights and input, so this is a real,
   reproducible number, not noise from clustering's own randomness. This
   check is real and stands — but see Problems hit #1: it answers "is the
   NPU-side number stable," not "is this a valid gate result," which is a
   different question the hand-computed table could not answer on its own.

9. **The correction**: re-ran all four bfp16 MTEB cells properly, `--sides
   cpu,npu` in one invocation each, so the CPU and NPU sides being compared
   come from the same session against the same data:

   ```powershell
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides cpu,npu --model bge-base-en-v1.5 --artifacts artifacts_base_bfp_fp32c `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_base_bfp_fp32c_realgate.json
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides cpu,npu --model bge-base-en-v1.5 --artifacts artifacts_base_bfp_cbf16 `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_base_bfp_cbf16_realgate.json
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides cpu,npu --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp_fp32c `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_minilm_bfp_fp32c_realgate.json
   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
       --sides cpu,npu --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp_cbf16 `
       --out tasks\0101-t23-bfp16-gates-on-1.4.2\mteb_minilm_bfp_cbf16_realgate.json
   ```

   Confirmed from each log that the gate block actually printed per-task
   CPU/NPU/delta numbers and a `PASS`/`FAIL` line (not the `NO GATE`
   paragraph) before trusting any of them — see Result §2.

## Result

### 1. Golden gate — all four bfp16 builds PASS, on both models

`rel_fro` / worst `1-cos`, all 128 rows, per-copy-rotated golden (0094's
gate):

| model | config | `rel_fro` | worst 1-cos | verdict |
|---|---|---:|---:|---|
| MiniLM | control (bf16, fp32-C) | 4.473e-03 | 1.086e-05 | PASS |
| MiniLM | **bfp16, fp32-C** | 2.347e-02 | **3.368e-04** | **PASS** |
| MiniLM | bfp16, bf16-C | 2.430e-02 | 3.406e-04 | PASS |
| bge-base | control (bf16, fp32-C) | 4.297e-03 | 1.353e-05 | PASS |
| bge-base | **bfp16, fp32-C** | 1.928e-02 | **2.174e-04** | **PASS** |
| bge-base | bfp16, bf16-C | 1.941e-02 | 2.284e-04 | PASS |

**Plain bfp16 + fp32-C now passes the golden gate on both models** — MiniLM
at 3.368e-04 (vs the 2e-03 tolerance) and bge-base at 2.174e-04, both roughly
**7-10x inside the tolerance**, not the ~20%-over-tolerance FAIL 0035/0052
measured at 2.395e-03 pre-migration. This is the headline the task was
launched to check, and it holds on both geometries measured.

**The 6.6x fp32-C/bf16-C gap is gone, and the sign has flipped slightly**:
MiniLM 3.368e-04 (fp32-C) vs 3.406e-04 (bf16-C) — fp32-C is marginally
*better*, not 6.6x worse. bge-base: 2.174e-04 vs 2.284e-04 — same direction.
This matches T26's closure (`0099`) almost exactly: the ablation there found
bf16-C 1.6% worse than fp32-C on an isolated GEMM once the `crrnd` fix is
present; here, on the full production encoder, bf16-C is 1.1-5.1% worse on
`1-cos`. **The old 6.6x accuracy case for shipping bf16-C alongside bfp16 no
longer exists on this toolchain.**

**One number here has no 1.3.4 precedent at all**: bge-base bfp16+fp32-C's
golden-gate `1-cos` was never measured pre-migration (0052/0053 only ran
bge-base at bfp16+**bf16**-C). It is a genuinely new measurement, not a
re-measurement.

### 2. MTEB — the golden gate's PASS does not settle it; bge-base fp32-C FAILS the MTEB gate

**2a. First pass — NOT gate results, kept visible per rule 3b, do not use
these to decide anything.** Delta (points, NPU main score minus CPU main
score) computed by hand from the control run's CPU column against each
bfp16 run's separate `--sides npu`-only invocation:

| model | config | mean delta (hand-computed, cross-session) | worst task delta | "gate" |
|---|---|---:|---:|---|
| MiniLM | bfp16, fp32-C | +0.124 | −0.051 | *(estimate)* |
| MiniLM | bfp16, bf16-C | +0.122 | −0.071 | *(estimate)* |
| bge-base | bfp16, fp32-C | −0.137 | −0.578 | *(estimate)* |
| bge-base | bfp16, bf16-C | −0.064 | −0.193 | *(estimate)* |

This is **not a gate result** — every one of the four source runs printed
`NO GATE -- this run has no delta to gate on: cpu side(s) not run`, and
`run_mteb.py` refuses to compute this table itself specifically because a
CPU baseline from a different invocation is not guaranteed to be the same
comparison (the tool's own comment: *"A caller glancing at the output saw a
completed run with no FAIL in it"* — precisely what nearly happened here).
Kept in the log, clearly labelled, because it is what actually happened
first and rule 3b says the wrong turn is part of the record — not because
it is evidence.

**2b. The real M8 gate, `--sides cpu,npu` in one invocation each — this is
the actual result:**

| model | config | mean delta | worst task delta | **gate result (tool's own PASS/FAIL line)** |
|---|---|---:|---:|---|
| MiniLM | control | +0.03 | −0.01 | PASS |
| MiniLM | **bfp16, fp32-C** | **+0.12** | **−0.05** | **PASS** |
| MiniLM | bfp16, bf16-C | +0.12 | −0.07 | PASS |
| bge-base | control | +0.02 | −0.00 | PASS |
| bge-base | **bfp16, fp32-C** | **−0.14** | **−0.58 (TwentyNewsgroupsClustering)** | **FAIL** |
| bge-base | bfp16, bf16-C | −0.06 | −0.19 | PASS |

Every row above is read directly from the tool's own `================ M8
GATE ================` block and its `PASS`/`FAIL -- gate is |mean| <= 0.5
points AND no task worse than -0.5` line (`mteb_*_realgate.log`) — not
derived. **The real gate confirms the hand-computed estimate on every cell,
within 0.02 points**: bge-base bfp16+fp32-C genuinely FAILS (mean −0.14,
worst −0.58, one task past the −0.5 line), the other three bfp16 cells
genuinely PASS. The verdict that follows is unchanged from the first pass,
but it now rests on the tool's own gate output rather than arithmetic
performed outside it — the distinction the coordinator's correction was
about, and it mattered: a CPU baseline from another session moving by a
fraction of a point could have flipped the −0.578/−0.58 verdict, and only a
same-session run rules that out.

**This is the finding that changes the recommendation.** Plain bfp16+fp32-C
passes the golden gate comfortably on both models, and passes MTEB cleanly
on MiniLM — but on bge-base it **fails the MTEB gate**, entirely on one task
(`TwentyNewsgroupsClustering`, a k-means clustering task), at −0.58 against
the −0.5 threshold. The other four tasks on that same run are all within
±0.06 points. **bf16-C does not have this problem**: same model, same task,
−0.19 — comfortably inside the gate. The NPU-side number was additionally
verified not to be sampling noise on its own (Result unchanged from the
original determinism check): the clustering task's NPU score reproduced
**bit-identically** (`0.4999837157304225` both times, full double
precision) across two independent `--sides npu` invocations — NPU dispatch
is deterministic, so the FAIL is not run-to-run variance on the NPU side,
and the real-gate re-run additionally confirms the CPU-side comparison
itself, done properly, still fails.

Per rule "report both and do not collapse them": the golden gate and MTEB
**disagree in verdict** for bge-base bfp16+fp32-C — 1-cos PASSES
comfortably (2.174e-04, ~9x inside tolerance) while MTEB FAILS on one task.
This is the same divergence class 0045/int8 already established as a real,
not contradictory, outcome — and per 0035, MTEB is the authority.

### 3. Speed — secondary, labelled by provenance

**Array GEMM time** (`wait (hardware)`, `--bench 5`, single invocation per
config, not repeated across separate runs — flagged, see Residual risk):

| model | config | ms/encode (wait hw) | speedup vs plain bf16 |
|---|---|---:|---:|
| MiniLM | control | 14.528 | 1.00x |
| MiniLM | bfp16, fp32-C | 8.322 | **1.75x** |
| MiniLM | bfp16, bf16-C | 7.128 | **2.04x** |
| bge-base | control | 103.336 | 1.00x |
| bge-base | bfp16, fp32-C | 49.818 | **2.08x** |
| bge-base | bfp16, bf16-C | 45.062 | **2.29x** |

These land close to 0052's MiniLM figures (1.74x / 2.20x measured
pre-migration) — the array-side speedup is not something this task expected
to move, and it did not, within measurement noise.

**End-to-end wall clock** (explicitly labelled as such, per rule 1 — single
5-encode `--bench` invocation, host-shared machine but confirmed uncontended
by `xrt-smi`; not averaged across repeated invocations):

| model | config | seq/s | vs control |
|---|---|---:|---:|
| MiniLM | control | 696.5 | — |
| MiniLM | bfp16, fp32-C | 834.2 | +19.8% |
| MiniLM | bfp16, bf16-C | 920.2 | +32.1% |
| bge-base | control | 129.2 | — |
| bge-base | bfp16, fp32-C | 186.7 | +44.5% |
| bge-base | bfp16, bf16-C | 202.7 | +56.9% |

### 4. The 128-row vs 4-row comparability trap — how it actually bites here

`tasks/0094` (today) changed the golden gate to compare all 128 rows with
per-copy-rotated content, and confirmed the *healthy* production number is
identical to the digit under both the old and new gate. That means the
**control** comparisons above (§1, §2's control rows) are safe. But every
1.3.4-era bfp16 number in T23's register entry (2.395e-03 FAIL, 3.615e-04
PASS, 3.470e-03 FAIL) was measured against the **old 4-row gate**, and this
task's bfp16 numbers are measured against the **new 128-row gate** — an
apples-to-oranges comparison in row count, on top of being apples-to-oranges
in toolchain. Stated plainly: **the historical FAILs and PASSes may not be
directly comparable to today's numbers on row coverage alone**, independent
of the toolchain question. This task cannot separate "toolchain fixed it"
from "toolchain fixed it, partially offset/compounded by row coverage
change" using 1-cos alone — which is exactly why MTEB (unaffected by the
golden-gate's internal row-counting logic; it embeds real, distinct MTEB
corpora, not the 4-sentence golden fixture) is the number this task's
verdict is actually built on, per rule 3's instruction and 0035's precedent.

## Problems hit

1. **The M8 gate never actually ran for any bfp16 configuration on the first
   pass — this was the coordinator's catch, not something found
   internally.** To save CPU-side MTEB time (bge-base's CPU side alone runs
   ~13 min), the first pass ran `--sides cpu,npu` once per model and
   `--sides npu` for the two bfp16 cells, then computed a delta table by
   hand against the control run's CPU numbers. `run_mteb.py` has an explicit
   guard against exactly this: a `--sides npu`-only run prints `NO GATE --
   this run has no delta to gate on: cpu side(s) not run` and writes no `cpu`
   key into `per_side`, and the tool's own comment names the failure mode by
   history — *"A caller glancing at the output saw a completed run with no
   FAIL in it"* — i.e. this guard exists because the exact mistake made here
   shipped once before. The hand-computed numbers were real and, as it turned
   out, accurate (Result §2b confirms them within 0.02 points), but they were
   not a gate verdict, and the load-bearing claim of the whole task — bge-base
   bfp16+fp32-C failing at −0.578 against a −0.5 line — is exactly the kind of
   margin a CPU baseline from a different session could plausibly move.
   **Fixed**: re-ran all four bfp16 cells with `--sides cpu,npu` in one
   invocation each (`*_realgate.json`/`.log`), and confirmed from each log
   that the real `PASS`/`FAIL` gate block printed (not the `NO GATE`
   paragraph) before trusting any number from it. See Result §2 for both the
   original (mislabelled) table, kept per rule 3b, and the corrected one.
2. **A background build silently stalled the task once** before this
   session's numbers were taken: the first `export_gemm_rtp.py` build was
   launched with PowerShell `run_in_background`, and PowerShell's `*>`
   redirect does not flush to the log file until the process exits — so
   polling the log file showed 0 bytes for the entire build, and passively
   waiting for an async completion notification did not reliably resume the
   task. Fixed by switching to **foreground invocations with generous
   timeouts** (the remaining three builds, all four golden-gate runs, and
   the four `--bench` runs), and for the two genuinely long MTEB runs
   (bge-base control ~14 min, bge-base bfp16 fp32-C ~10 min), a **bounded
   polling loop** (`until grep -q "wrote tasks" "$OUTFILE"; do sleep 20;
   done`) rather than an unbounded background wait. Recorded here because it
   cost real time and the fix (foreground + generous timeout, or an
   explicit poll loop with a concrete exit condition) is the reusable
   lesson, not the specific stall.
3. **The borderline MTEB clustering result** (bge-base bfp16+fp32-C,
   −0.578) initially looked like it could be noise given the register's own
   note that clustering has ±0.25 spread **across different models**
   (0052 §3) — re-read carefully, that spread is model-to-model, not
   run-to-run, so it did not directly bear on whether *this* number was
   stable. Resolved by an actual repeat measurement (Result §2) rather than
   assuming either way; it reproduced bit-for-bit. (This addressed NPU-side
   stability, not the separate cross-session-CPU-baseline problem #1 above —
   the two are different risks to the same number, and both needed closing.)
4. **No int8 confusion, no cache-marker collisions, no L1 overflows** — the
   build recipe here is exactly the one `tasks/0032`/`0051` already
   established for production artifact sets, just with `--emulate-bfp16`
   /`--c-bf16` added, so none of `export_gemm_rtp.py`'s known fail-open
   classes (int8/bf16 marker collision, tier ambiguity) were freshly
   triggered.

## Artifacts

All in `tasks/0101-t23-bfp16-gates-on-1.4.2/`:

- `gate_minilm_control.txt`, `gate_base_control.txt` — control golden gate,
  re-run and saved this session (bit-identical to 0094's recorded numbers).
- `gate_minilm_bfp_fp32c.txt`, `gate_minilm_bfp_cbf16.txt`,
  `gate_base_bfp_fp32c.txt`, `gate_base_bfp_cbf16.txt` — the four bfp16
  golden-gate runs.
- `bench_minilm_control.txt`, `bench_base_control.txt`,
  `bench_minilm_bfp_fp32c.txt`, `bench_minilm_bfp_cbf16.txt`,
  `bench_base_bfp_fp32c.txt`, `bench_base_bfp_cbf16.txt` — `--bench 5` raw
  output for all six configs.
- `objdump_minilm_fp32c_matmul.txt`, `objdump_minilm_cbf16_matmul.txt`,
  `objdump_base_fp32c_matmul.txt`, `objdump_base_cbf16_matmul.txt` — AIE
  objdump of `matmul_bf16_f32_333c4d33.o` from each of the four builds' JIT
  cache directories (20x `vconv.bfp16ebs8.fp32`, 3x `crrnd`, every one).
- `objdump_minilm_cbf16_narrow.txt`, `objdump_base_cbf16_narrow.txt` — AIE
  objdump of the bf16-C epilogue's `narrow_3072_f32_bf16.o` (3x `crrnd`,
  matching 0098).
- `mteb_minilm_control.json`/`.log`, `mteb_base_control.json`/`.log` —
  `--sides cpu,npu` MTEB runs (also this task's MTEB-side control check).
- `mteb_minilm_bfp_fp32c.json`/`.log`, `mteb_minilm_bfp_cbf16.json`/`.log`,
  `mteb_base_bfp_fp32c.json`/`.log`, `mteb_base_bfp_cbf16.json`/`.log` —
  the four **first-pass, `--sides npu`-only** bfp16 MTEB runs. Kept per rule
  3b (Problems hit #1); each one's own log shows the `NO GATE` message —
  **do not read these as gate results**, use the `_realgate` files instead.
- `mteb_base_bfp_fp32c_clustering_repeat.json`/`.log` — the determinism
  check on the borderline clustering result (NPU-side reproducibility only;
  does not by itself establish a gate verdict — see Problems hit #1 and #3).
- `mteb_base_bfp_fp32c_realgate.json`/`.log`,
  `mteb_base_bfp_cbf16_realgate.json`/`.log`,
  `mteb_minilm_bfp_fp32c_realgate.json`/`.log`,
  `mteb_minilm_bfp_cbf16_realgate.json`/`.log` — **the actual M8 gate
  results**, `--sides cpu,npu` in one invocation each, each log showing the
  real `PASS`/`FAIL -- gate is |mean| <= 0.5 points AND no task worse than
  -0.5` line. These are what Result §2b and the recommendation are built on.

Artifact sets built this session (in `runtime/`, not checked in, reusable
for follow-up work): `artifacts_minilm_bfp_fp32c`, `artifacts_minilm_bfp_cbf16`,
`artifacts_base_bfp_fp32c`, `artifacts_base_bfp_cbf16`.

## Residual risk / what this task did not settle

- **The methodological mistake in Problems hit #1 is corrected, not merely
  noted** — all four decisive MTEB numbers now come from real, same-session
  `--sides cpu,npu` gate runs, not hand arithmetic across sessions. Flagged
  here anyway because it is the kind of error that would not have been
  caught without a second reader: the hand-computed numbers were internally
  consistent, printed cleanly, and *looked* like a gate table.
- **Only 5 MTEB tasks, on 2 of the project's 6 shipped models.** The failing
  task is one k-means clustering task on bge-base; whether bge-small,
  bge-large, nomic, or EmbeddingGemma would show the same or a different
  failure mode is unmeasured.
- **Speed figures are single `--bench 5` invocations per config**, not
  repeated across multiple separate invocations the way rule 1's "never a
  single run" ideally asks for hardware timing claims. The array-GEMM-time
  ratios landing close to 0052's independently-measured pre-migration
  figures is corroborating evidence they are not a fluke, but they were not
  independently re-run this session.
- **Cross-process/cross-day kernel-object determinism is still the open
  question T26/0099 left behind**, not this task's to close: all four
  builds *this session* agreed on `crrnd`=3 (a within-session determinism
  data point), but 0099 already showed a cross-day divergence (0 vs 3 hits,
  same source, same package version) that nobody has explained. If that
  recurs, these numbers would need re-verifying against a fresh objdump
  before being trusted again — the objdump evidence in this task's artifacts
  is what to check against.
- **Golden-gate numbers here are not directly comparable to any pre-0094
  measurement** (Result §4) — stated as a limitation, not resolved, because
  MTEB (unaffected by the golden gate's internal logic) is what this task's
  recommendation is actually built on.

## Proposed register update

**T23 verdict: re-measured on 1.4.2 — mixed, not a clean reopen.** Plain
bfp16+fp32-C's golden-gate FAIL (0035/0052, pre-migration, `1-cos`
2.395e-03) does not reproduce: on today's toolchain it PASSES comfortably on
both measured models (MiniLM 3.368e-04, bge-base 2.174e-04, both ~7-10x
inside the 2e-03 tolerance). But **MTEB — the project's stated authority for
this decision (0035) — fails it anyway, on bge-base specifically**: worst
task delta −0.58 (`TwentyNewsgroupsClustering`), past the −0.5 gate. This is
a real gate result (`--sides cpu,npu` in one invocation, the tool's own
`FAIL` line, not a hand-computed estimate — an earlier hand-computed pass at
this task got caught and corrected, see the task log's Problems hit #1), and
the NPU-side score was independently confirmed deterministic across two
separate runs. bfp16+bf16-C does not have this problem (bge-base worst task
−0.19, comfortably inside, also a real gate PASS) and
also runs modestly faster (2.04-2.29x array-GEMM speedup vs 1.75-2.08x for
fp32-C). **Separately, T26's closure (0099) is confirmed at the
whole-encoder level, not just the isolated-GEMM level this task ran on**:
the old 6.6x fp32-C-vs-bf16-C accuracy gap that motivated shipping bf16-C
alongside bfp16 is gone on 1.4.2 — the two are now within 1-5% of each other
on `1-cos`, with fp32-C marginally *ahead* in 3 of 4 measured cells, not 6.6x
behind.

Suggested replacement/addition to T23's `OPEN-THREADS.md` entry (append
after the "bge-base MTEB landed 2026-08-20" paragraph):

> **RE-MEASURED 2026-08-23 on mlir-aie 1.4.2** ([`0101`](TASK.md)),
> because T26's closure the same day (`0099`) showed every number above was
> measured on 1.3.4, which never emitted the rounding-mode control the
> emulated matmul's own source asks for. Rebuilt all four bfp16 artifact
> sets fresh (both C-transport dtypes x MiniLM/bge-base), confirmed via AIE
> objdump that emulation is genuinely active (`vconv.bfp16ebs8.fp32`, 20
> hits) and the `crrnd` save/set/restore triple is present (3 hits, where
> 0098's stale 2026-08-20 build had 0) in every one of the four builds.
>
> **The golden-gate FAIL is gone**: plain bfp16+fp32-C now PASSES on both
> models (MiniLM `1-cos` 3.368e-04, bge-base 2.174e-04 — both ~7-10x inside
> the 2e-03 tolerance, against the old 2.395e-03 FAIL). **But MTEB — this
> project's stated authority for a datapath decision (0035) — FAILS
> bfp16+fp32-C on bge-base anyway**: worst task delta −0.58
> (`TwentyNewsgroupsClustering`, k-means clustering), past the −0.5 gate —
> a real `--sides cpu,npu` gate run, the tool's own `FAIL` line, and the
> NPU-side score additionally confirmed deterministic across two separate
> runs. MiniLM's bfp16+fp32-C passes MTEB cleanly (mean +0.12, worst
> −0.05). **bfp16+bf16-C passes MTEB cleanly on both models** (bge-base
> worst −0.19) and the old
> 6.6x accuracy edge that motivated pairing bf16-C with bfp16 is **gone** —
> the two C-transport dtypes are now within 1-5% of each other on `1-cos`,
> consistent with T26's finding that the 6.6x was a 1.3.4 `floor`-bias
> artifact, not a real property of the datapath. Array-GEMM speedups
> reproduce close to 0052's pre-migration figures (1.75-2.08x fp32-C,
> 2.04-2.29x bf16-C).
>
> **Reading**: the 1.3.4->1.4.2 migration did not simply "fix" bfp16, it
> **changed which configuration is safe**. Plain bfp16+fp32-C is not
> recommendable as-is despite passing 1-cos — MTEB catches a real,
> reproducible regression on at least one task on at least one model that
> 1-cos does not see, the same divergence class 0045/T20(int8) already
> established as real rather than contradictory. bfp16+bf16-C clears both
> gates on both measured models and is also the faster of the two — if this
> datapath is reopened, bf16-C should travel with it, not for the old
> (refuted) 6.6x accuracy reason but because it is the config that actually
> passes. Not measured: bge-small, bge-large, nomic, EmbeddingGemma. Decision
> remains the user's per the 0045 precedent — this is evidence, not a
> changed default.

`tasks/README.md` index row:

```
| [0101](0101-t23-bfp16-gates-on-1.4.2/TASK.md) | **T23 re-measured on mlir-aie 1.4.2 — mixed verdict, not a clean reopen.** All six cells measured (2 models x 3 configs, both gates, objdump-verified emulation+`crrnd` on every build). Plain bfp16+fp32-C's old golden-gate FAIL (0035/0052, `1-cos` 2.395e-03) does NOT reproduce -- passes comfortably on both models now (3.368e-04 MiniLM, 2.174e-04 bge-base). But MTEB (the project's stated authority, 0035) FAILS it anyway on bge-base: worst task -0.58 (TwentyNewsgroupsClustering), past the -0.5 gate -- a real `--sides cpu,npu` gate run's own FAIL line, after an earlier hand-computed cross-session estimate of the same number was caught and corrected mid-task (rule 3b, kept in the log) -- while bfp16+bf16-C passes MTEB cleanly on both models, also a real gate run. The old 6.6x fp32-C-vs-bf16-C accuracy gap that motivated bf16-C (T26) is gone at the whole-encoder level too, confirming 0099's isolated-GEMM finding: the two C dtypes are now within 1-5% of each other on 1-cos. Recommendation: bfp16+bf16-C is the only config that clears both gates on both measured models; plain bfp16+fp32-C is not safe to adopt on 1-cos alone. Decision remains the user's | research (T23) | done |
```
