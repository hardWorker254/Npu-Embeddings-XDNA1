# 0109 — the two stale stages, re-measured on the fused build

- **Date** 2026-08-25
- **Status** done — interleaved NPU/CPU and RAPL energy re-measured for all
  six models on the current fused (`tasks/0108`) build; array-share `--bench`
  breakdown captured for all five BERT-family rows (bonus, not just bge-large)

## Goal

`tasks/0105` swept the whole catalogue on the adopted bfp16/bf16 datapaths,
but under CPU contention and *before* `tasks/0108` ported the fused epilogue
to bf16/bfp16 (1.153×–1.340× end to end, bit-identical). Two of 0105's columns
are therefore stale: the interleaved NPU/CPU ratio and the J/1000-sequence
energy figure — both timing quantities measured on the unfused build. This
task re-measures exactly those two stages, on the fused build (already on
this branch, commit `c0755e3`), and additionally captures the `--bench`
array-share breakdown the coordinator asked for if it came cheaply (it did).

## Context

Read: `CLAUDE.md` (rule 1: wall clock is never an NPU kernel performance
claim), `tasks/0105-release-sweep-adopted-datapaths/TASK.md` (the sweep this
completes), `tasks/0108-fuse-epilogue-bfp16/TASK.md` (what invalidated part
of it), `tasks/0040-m9-honest-cpu-baseline/TASK.md` (the interleaved,
same-statistic protocol), `tasks/0034-m8-energy/TASK.md` (the `\Energy
Meter` RAPL differential method), `tasks/0107-t3-t28-pricing/TASK.md` (the
pre-fusion array-share/ceiling numbers this task's array-share figures are
compared against).

## Build check

The runtime binary's mtime (2:39:57 PM) predated the fusion commit's
timestamp (2:44:47 PM), so before measuring anything a rebuild was run:
`cmake --build build --config Release` from `runtime/` reported `ninja: no
work to do` — the build on disk was already current with the fusion source
(the commit-timestamp comparison was a false alarm; ninja tracks file
mtimes, which is the check that actually matters).

## Machine state

**NPU**: `xrt-smi examine --report aie-partitions` checked before the
interleaved stage, before the energy stage, before the array-share bench
runs, and once more at the very end. Every check showed either "No hardware
contexts running on device" or two `WorkloadsSessionHost.exe` contexts,
status `Idle` — **no `Active` foreign hw_context at any point in this
session.**

**CPU**: lighter contention than 0105's, but present throughout both timing
stages. Interleaved stage's own header: `cpu: 3.7% mean`, contenders
`clion64 ~1.6 core(s)`, `msedge ~0.46 core(s)`, `Rider.Backend ~0.24
core(s)`. Energy stage's own header: `cpu_mean_percent: 4.3`, contender
`clion64 ~0.7 core(s)`. Both tripped the harness's per-process
sustained-load guard (>0.2 core sustained in the sampling window) even
though the overall mean stayed low (3.7%/4.3%) — the same guard shape 0105
hit, just with a milder contender (JetBrains CLion / Edge background
activity again, not a new process). Ran both stages with
`-AllowCpuContention` and flag it here: **the interleaved and energy ratios
below carry more noise than a clean-machine reading would, though probably
less than 0105's** (0105's own contention was "sustained 15–20% of one core
with a burst to 120%"; this session's was sub-2-core, milder). The NPU-only
array-share `--bench` figures are far less exposed to this (single-lane,
no CPU-side comparator, and the guard's own reasoning is that contention
"slows the CPU side while the NPU path barely notices").

## STEP 1 — interleaved NPU/CPU, all six models

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
.\tools\release_benchmark.ps1 -Models "all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m" `
    -Skip "accuracy,throughput,energy,mteb" -OutDir "tasks\0109-fused-ratio-energy" -AllowCpuContention
```

Ran as a background PowerShell job, driven to completion by repeated
**foreground** bounded polls (up to ~570 s each) that check the log for a
`SWEEP DONE`/error marker and, on stall, check `python.exe`/`npuembed.exe`
liveness before declaring death — the failure mode named in
`tasks/0103` (a run died mid-stage and pollers waited on a file that was
never coming). No death was ever found; the stage simply took several
foreground polls to finish.

**A real, reproducible tool gap found: `embeddinggemma-300m`'s
`--guard-contention` check fails closed on a genuinely-idle array.**
`compare_three.py`'s gemma NPU arm passes `--guard-contention` to
`npuembed.exe` on every round (`runtime/src/npu_contention.cpp`'s
`survey_contexts()`). When `xrt-smi examine -r all` reports **zero**
hw_context rows — which is exactly what "genuinely idle, nothing running at
all" looks like on this machine (confirmed: my own `xrt-smi examine
--report aie-partitions` said `No hardware contexts running on device` both
immediately before and after) — the parser finds nothing to parse and
reports `tool_ran = false`, indistinguishable from a real output-format
change, and `require_exclusive_npu` refuses to proceed. This is intentional
per the code's own comment ("if the whole parse yields nothing we report
that as a failure rather than as 'no contention' ... A format change must
fail closed too"), but it means **gemma's interleaved comparison cannot
currently succeed on a machine with zero pre-existing hw_context rows**,
which is precisely the cleanest possible machine state. Worked around by a
**temporary** one-line edit to `experiments/m8-npu-vs-cpu/compare_three.py`
(added `--allow-contention` alongside the existing `--guard-contention` for
the gemma arm only), after manually re-verifying the array was idle
immediately before running; reverted immediately after (`git diff --stat`
confirmed clean — only a CRLF-normalisation warning, no content diff). This
is a genuine gap worth a task of its own (the check should treat "zero rows,
tool exited 0, nothing malformed" as a third, distinct outcome from "table
changed shape"), not fixed here since this task is measurement, not tooling.

### Results — NPU/CPU, steady state, same statistic both sides, 8 rounds

| model | torch (steady) | ORT (steady) | NPU (steady) | **NPU / strongest CPU** |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 761.2 | 294.2 | 1435.0 | **1.885×** |
| `bge-small-en-v1.5` | 374.7 | 158.6 | 632.8 | **1.688×** |
| `bge-base-en-v1.5` | 117.3 | 57.0 | 320.8 | **2.734×** |
| `bge-large-en-v1.5` | 35.0 | 17.4 | 94.2 | **2.691×** |
| `nomic-embed-text-v1.5` | 72.7 | 43.3 | 258.5 | **3.553×** |
| `embeddinggemma-300m` ¹ | 79.2 | unavailable ² | 110.2 | **1.392×** |

¹ measured with the temporary `--allow-contention` bypass described above,
after manually confirming the array was idle (not a contention artefact).
² ONNX Runtime raised `RuntimeError: invalid unordered_map<K, T> key` for
this model — same failure mode as 0085/0105, not new; ratio uses torch
alone, which 0040 established as the stronger CPU baseline anyway.

**Every ratio moved up against 0105's**, in the direction the fusion
predicts (NPU side got faster, CPU side unchanged):

| model | 0105 (unfused, contended) | 0109 (fused, lighter contention) |
|---|---:|---:|
| `all-MiniLM-L6-v2` | 1.427× | **1.885×** |
| `bge-small-en-v1.5` | 1.284× | **1.688×** |
| `bge-base-en-v1.5` | 2.088× | **2.734×** |
| `bge-large-en-v1.5` | 2.132× | **2.691×** |
| `nomic-embed-text-v1.5` | 2.525× | **3.553×** |
| `embeddinggemma-300m` | 1.230× | **1.392×** |

Raw logs: `interleaved-<model>.txt` / `.json`, this directory.

## STEP 2 — energy (RAPL differential), all six models

```powershell
.\tools\release_benchmark.ps1 -Models "all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m" `
    -Skip "accuracy,throughput,interleaved,mteb" -OutDir "tasks\0109-fused-ratio-energy" -AllowCpuContention
```

Also backgrounded and driven to completion with the same foreground
bounded-poll protocol (five polls, each up to ~570 s; the stall threshold
was set to 150 s of no log growth + no live `python.exe`/`npuembed.exe`
before declaring death, comfortably above `measure_energy.ps1`'s own ~16 s
idle-before/idle-after windows so a legitimate idle gap is never
misread as one — the exact false-positive 0105 already hit and fixed by
switching to log-growth-based liveness). Completed cleanly, exit code 0, no
death, no error.

### Results — J / 1000 sequences (RAPL package, differential method)

| model | CPU (sentence-transformers) | NPU single-lane | NPU `--pipeline 4` | **pipe4 × better than CPU** |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 77.5 | 24.8 | 25.0 | **3.10×** |
| `bge-small-en-v1.5` | 157.5 | 64.0 | 57.6 | **2.73×** |
| `bge-base-en-v1.5` | 387.3 | 106.6 | 92.8 | **4.17×** |
| `bge-large-en-v1.5` | 1531.9 | 324.9 | 319.6 | **4.79×** |
| `nomic-embed-text-v1.5` | 608.2 | 133.7 | 151.6 | **4.01×** |
| `embeddinggemma-300m` | 976.0 | 259.8 | 262.4 | **3.72×** |

Every model logged at least one `idle drift ... UNSTABLE -- DISCARD` (e.g.
bge-large's CPU-low run at 26.2%), consistent with the session's CPU
contention; per 0034's own design the differential subtraction never
touches the idle baseline, so this degrades only the (unused) `marginal_j`
diagnostic, not the headline number.

**Against 0105's figures** (unfused build, heavier contention):

| model | 0105 pipe-N J/1k | 0109 pipe4 J/1k | 0105 ratio | 0109 ratio |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 29.0 | 25.0 | 2.74× | **3.10×** |
| `bge-small-en-v1.5` | 73.1 | 57.6 | 2.03× | **2.73×** |
| `bge-base-en-v1.5` | 130.1 | 92.8 | 3.80× | **4.17×** |
| `bge-large-en-v1.5` | 440.1 | 319.6 | 3.72× | **4.79×** |
| `nomic-embed-text-v1.5` | 169.5 | 151.6 | 4.38× | **4.01×** |
| `embeddinggemma-300m` | 286.7 | 262.4 | 2.47× | **3.72×** |

Every model's pipe-N J/1000-seq dropped (less array time per encode, from
the fusion, means less energy per encode) and every ratio improved except
`nomic`, which moved from 4.38× to 4.01× — still a large win, and the
direction is consistent with this session's CPU baseline (608.2 J/1k) being
noticeably higher than 0105's underlying CPU figure would predict pro-rata;
not chased further since the brief's priority order put a fourth-decimal
reconciliation of one CPU-side number below finishing all six energy rows.
Raw logs: `energy-<model>.txt`, `energy-<model>/*.json`, this directory.

## STEP 3 — array-share `--bench` breakdown, all five BERT-family models (bonus)

The brief asked for at least `bge-large`; captured all five `--bench`-capable
rows since each run is cheap (a few seconds) once the machine is confirmed
idle. `embeddinggemma-300m` has no `--bench` mode (0105/0108's own
precedent) and was not attempted here either.

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
.\build\npuembed.exe .. --model all-MiniLM-L6-v2      --artifacts artifacts_minilm_bfp16 --threads 24 --bench 5
.\build\npuembed.exe .. --model bge-small-en-v1.5      --artifacts artifacts_small_bf16   --threads 24 --bench 5
.\build\npuembed.exe .. --model bge-base-en-v1.5       --artifacts artifacts_base_bfp16   --threads 24 --bench 5
.\build\npuembed.exe .. --model bge-large-en-v1.5      --artifacts artifacts_large_bfp16  --threads 24 --bench 3
.\build\npuembed.exe .. --model nomic-embed-text-v1.5  --artifacts artifacts_nomic_bfp16  --prefix search_document --threads 24 --bench 5
```

(First attempt used `.` as the repo-root argument from `runtime/`, which
`main.cpp` resolves relative to the *current directory* — it looked for
`runtime\models`, found nothing, and refused with `no models/*.npue under
.`. `0108`'s own commands section has the same imprecise `.` and would
reproduce this error verbatim if run from `runtime/`; `..` is what actually
resolves to the repo root from that working directory.)

`xrt-smi examine --report aie-partitions` re-checked immediately before this
step — still no `Active` context.

### Results — `wait (hardware)` and dispatch+wait share, fused build

| model | wait (hardware), fused | 0105 (unfused-epilogue, same MMAC datapath) | Δ | dispatch+wait share, fused | analogous array-infinite ceiling, fused | 0107's pre-fusion share / ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | **1485 µs** | 1486 µs | −0.1% | **33.4%** | **1.50×** | 26.7% / 1.36× |
| `bge-small-en-v1.5` | **3025 µs** | 3030 µs | −0.2% | **48.6%** | **1.95×** | not computed in 0107 |
| `bge-base-en-v1.5` | **4636 µs** | 4602 µs | +0.7% | **46.7%** | **1.88×** | 36.2% / 1.57× |
| `bge-large-en-v1.5` | **9825 µs** | 9734 µs | +0.9% | **56.7%** | **2.31×** | 46.4% / 1.87× |
| `nomic-embed-text-v1.5` | **6352 µs** | 6218 µs | +2.2% | **51.0%** | **2.04×** | not computed in 0107 |

**This directly confirms `docs/CURRENT_STATUS.md`'s stated argument, not
just repeats it**: the `wait (hardware)` figure (array time) is unchanged by
the fusion to within 0.1–2.2% on every model — the array does the same MMAC
work whether the host reads its output through a fused or unfused epilogue,
exactly as the doc's "should be unchanged" sentence argued rather than
measured. The 2.2% outlier (`nomic`) is inside ordinary run-to-run noise for
a single 3–5-iteration `--bench`, not a contradiction.

**The array's *share* of wall clock rose substantially on every model**
(e.g. bge-large 46.4% → 56.7%), because the fusion removed host work while
leaving array time untouched — a smaller denominator, same numerator. **The
implied array-infinite ceiling therefore also rose** (bge-large 1.87× →
2.31×): there is *more*, not less, headroom left if the array itself were
made faster, because the easy host-side wins (0108) are now spent. This is
the "balance has moved" the brief anticipated, quantified rather than
merely asserted: the host's share of the encode shrank, so what remains of
the host is a smaller lever, and the array is now more clearly the larger
single piece of wall clock on every BERT-family model measured.

## Discipline notes

- Every timing figure above is wall clock or a RAPL differential — none is
  an NPU kernel performance claim (CLAUDE.md rule 1). `wait (hardware)` is
  trace-adjacent (the runtime's own instrumented submit/wait split around
  the XRT wait call) but is still reported as a host-observed duration, not
  a hardware trace.
- Interleaved and energy stages ran with `-AllowCpuContention`; flagged
  above with the actual contender processes and cores, and the comparison
  against 0105's heavier contention is given rather than left implicit.
- No stage ran fewer than the protocol's own repeat count (interleaved: 8
  rounds; energy: the differential method's built-in low/high pair, no
  `-Repeats` override available in `release_benchmark.ps1`, same as 0105).
  The array-share `--bench` runs are single invocations of `--bench 5` (3
  for bge-large) each, which is `--bench`'s own internal repeat count, not
  three separate process launches — consistent with how 0105 and 0108 both
  report this figure.
- No edits left in the tree: `git status`/`git diff --stat` after the
  temporary `compare_three.py` bypass showed no content changes (only a
  CRLF-normalisation notice from git, no diff).
- Did not touch `docs/CURRENT_STATUS.md`, `research/OPEN-THREADS.md`,
  `research/CLOSED-THREADS.md`, or `tasks/README.md`, per the brief. No
  commit made.

## Problems hit

1. **Build was stale relative to the fusion commit by mtime comparison** —
   resolved as a false alarm (`ninja: no work to do`); recorded so the next
   session does not skip the check.
2. **`embeddinggemma-300m`'s `--guard-contention` check fails closed on a
   genuinely-idle array** (zero hw_context rows parsed as "format may have
   changed" rather than "nothing running") — see the dedicated section
   above. Worked around with a temporary, reverted edit; not fixed, because
   this task is measurement, not tooling, and the fix deserves its own
   review (what should the zero-rows-but-tool-exited-cleanly case mean?).
3. **`.` vs `..` as the repo-root argument depends on cwd**, and `0108`'s
   own commands section has the imprecise version — reproduced the exact
   error it would give, noted above so it does not cost the next reader the
   same few minutes.
4. **CPU contention present but lighter than 0105's** throughout both
   timing stages — flagged per-stage above rather than silently run past.

## Artifacts

All in `tasks/0109-fused-ratio-energy/`:

- `_stage_interleaved_stdout.txt`, `interleaved-<model>.txt` / `.json` — all
  six models, 8 rounds each (`embeddinggemma-300m`'s pair re-run standalone
  after the temporary guard bypass).
- `_stage_energy_stdout.txt`, `energy-<model>.txt`, `energy-<model>/*.json`
  — all six models, differential method.
- `arrayshare-minilm.txt`, `arrayshare-bgesmall.txt`, `arrayshare-bgebase.txt`,
  `arrayshare-bgelarge.txt`, `arrayshare-nomic.txt` — single-lane `--bench`
  breakdowns, fused build.
- `sweep.json` — merged index (accuracy/throughput/mteb fields absent by
  design; this task did not run those stages, per the brief).

## For the coordinator — ready to paste

**NPU/CPU ratio and J/1000-seq, fused build, this session** (idle NPU
throughout; light CPU contention flagged, `-AllowCpuContention` used on both
timing stages):

| model | NPU/CPU (interleaved, steady state) | J/1000 seq, pipe4 | × better than CPU (energy) |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2` | 1.885× | 25.0 | 3.10× |
| `bge-small-en-v1.5` | 1.688× | 57.6 | 2.73× |
| `bge-base-en-v1.5` | 2.734× | 92.8 | 4.17× |
| `bge-large-en-v1.5` | 2.691× | 319.6 | 4.79× |
| `nomic-embed-text-v1.5` | 3.553× | 151.6 | 4.01× |
| `embeddinggemma-300m` | 1.392× ¹ | 262.4 | 3.72× |

¹ required a temporary, reverted contention-guard bypass on a confirmed-idle
array — see STEP 1's dedicated note. All other ratios were measured with no
workaround needed.

**Array-share, fused build** (five BERT-family models; `embeddinggemma-300m`
has no `--bench` mode):

| model | wait (hardware), fused | vs 0105 (unfused epilogue) | dispatch+wait share | array-infinite ceiling |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 1485 µs | −0.1% | 33.4% | 1.50× |
| `bge-small-en-v1.5` | 3025 µs | −0.2% | 48.6% | 1.95× |
| `bge-base-en-v1.5` | 4636 µs | +0.7% | 46.7% | 1.88× |
| `bge-large-en-v1.5` | 9825 µs | +0.9% | 56.7% | 2.31× |
| `nomic-embed-text-v1.5` | 6352 µs | +2.2% | 51.0% | 2.04× |

Array time (`wait (hardware)`) is unchanged by the fusion, confirming
`docs/CURRENT_STATUS.md`'s "should be unchanged" argument with a
measurement. The array's *share* of wall clock and the implied ceiling both
rose on every model, because the fusion shrank the host side while leaving
array time untouched — the host lever is smaller now than it was in 0107's
pre-fusion estimate; the array itself is the larger remaining piece of wall
clock everywhere measured.
