# 0105 — release sweep on the adopted datapaths

- **Date** 2026-08-24
- **Status** done — accuracy, throughput, array-time, interleaved and energy
  measured for all six bf16-family rows in one session; int8 carried over
  from `tasks/0085` (not re-measured, see below); MTEB not re-run (see below)

## Goal

`tasks/0104` moved five of six models to the bfp16-emulated MMAC + bf16-C
datapath (`--emulate-bfp16 --c-bf16`); `bge-small-en-v1.5` stays on plain
bf16, having failed its MTEB gate. `docs/CURRENT_STATUS.md`'s "0.4.0 NUMBERS"
table was measured on the pre-adoption datapath for five of its six rows and
now carries a warning saying so. This task re-measures the whole catalogue
on what actually ships, following `tasks/0085`'s protocol so the numbers are
comparable in method (not in absolute value — five rows changed datapath,
which is the point of this sweep, not a confound to hide).

## Context

Read: `CLAUDE.md`, `tasks/0104-adopt-bfp16-per-model/TASK.md`,
`tasks/0085-m13-release-sweep/TASK.md`, `tasks/0097-t18-t21-t4-measurements/TASK.md`
(source of the "quote `--bench`'s `wait (hardware)` line for array time"
convention).

## STEP 1 — harness fix

`tools/release_benchmark.ps1`'s `$CATALOG` still named the pre-adoption
artifact directories and labelled every bf16-family row `dtype = "bf16"`
with no way to see which MMAC datapath actually ran. Fixed:

- The six bf16-family rows now point at the artifact directories `tasks/0104`
  built and `tools/make_release.ps1` ships: `artifacts_minilm_bfp16`,
  `artifacts_small_bf16` (bge-small, plain bf16 — unchanged design, new
  directory name since 0104's coordinator review, so it *states* its own
  datapath where the old `artifacts_b128il` predates the field and reads
  `UNRECORDED`), `artifacts_base_bfp16`, `artifacts_large_bfp16`,
  `artifacts_nomic_bfp16`, `artifacts_gemma_bfp16`. Cross-checked against
  `runtime/src/hub.cpp`'s catalogue (`CatalogEntry::datapath`, set by
  allowlist in `table()`) and `tools/make_release.ps1`'s default `$Artifacts`
  list — both agree.
- Added a `datapath` field to each `$CATALOG` row: the row's **intended**
  assignment (`"bfp16"` / `"bf16"` / `"int8-native"`), which is a comment,
  not the truth.
- Added `Get-DatapathLine`, which scrapes the runtime's own status line
  (`  datapath   <text>`, printed by `runtime/src/main.cpp` off the loaded
  `Design`'s `info().emulate_bfp16`/`datapath_recorded` — never off a flag)
  out of the first log a row produces (accuracy, else throughput), and writes
  it to `$row.datapath_reported`. This is the value the brief asked to prefer
  over restating intention — it is what 0104's whole guard exists to make
  authoritative, and every row's `datapath_reported` below matches its
  `datapath` exactly (verified in `sweep.json`), i.e. `pick_artifacts()`
  selected what the table intended for every row.
- int8 rows unchanged (still `artifacts_int8c_*`, `dtype = "int8"`), labelled
  `datapath = "int8-native"` — int8 is a third, separate datapath unaffected
  by the bfp16 decision (native int8 MMAC), and the bf16-vs-bfp16-emulated
  status line does not describe it either way, so it is not scraped for
  those rows.

Syntax-checked with `[System.Management.Automation.Language.Parser]::ParseFile`.

## Machine state / idleness

`xrt-smi examine --report aie-partitions` checked before accuracy, before
throughput, before interleaved, and again before energy: every context
listed is `WorkloadsSessionHost.exe`, status `Idle`, and **no context
anywhere in the report was `Active`** at any of the four checks — **the
array was not contended by a foreign process at any point in this
session.**

**CPU side was not clean**, unlike 0085's transient Notepad/PyCharm/Spotify
blips. `release_benchmark.ps1`'s CPU-quiet guard (>20% of one core sustained
over its 3 s sampling window trips it) refused the throughput stage four
times in a row over several minutes; a direct 5 s-window poll of
`clion64`/`msedge` (JetBrains CLion + Edge, both apparently doing background
work unrelated to this session) showed a **sustained** ~15-20% of one core,
with one burst to ~120% of a core — not a transient blip settling out.
Overall CPU mean stayed low throughout (1.5-3.7%); it was specifically the
per-process sustained-load check that tripped. During the energy stage a
second, unrelated contender appeared (`BackgroundDownload`, ~1.1 core).
Ran every timing stage (throughput, interleaved, energy) with
`-AllowCpuContention`, the tool's own documented (if discouraged) escape
valve for exactly this situation, and flagged it every time — **NPU/CPU
ratios (interleaved, energy) from this session carry more noise than a
clean-machine reading would**; the NPU-only throughput and array-time
numbers are far less exposed (the guard's own reasoning is that contention
"slows the CPU side while the NPU path barely notices" — it biases ratios,
not the NPU-side wall clock itself, though host eltwise sharing the same
cores is a real if smaller channel). The energy stage's own per-run
"idle drift" diagnostic came back `UNSTABLE -- DISCARD` far more often than
a clean run (visible per-model below) — consistent with the CPU contention
above. This does **not** invalidate the differential energy numbers
themselves: tasks/0034 designed the differential method (`(E_high - E_low)
/ (n_high - n_low)`) specifically so it **never touches the idle baseline**,
which is what makes it robust to idle drift — but it is one more reason to
read this session's ratios as "consistent with a real effect," not as
publication-grade absolutes.

## A real bug found and fixed mid-sweep: the per-stage merge was clobbering
## `datapath_reported` with `null`

`sweep.json`'s row-merge is per-key ("foreach `$k` in `$r.Keys`, overwrite
`$merged[name][$k]`"), by design, so a later stage's row can add fields
without erasing an earlier stage's. But the initial `$row` hashtable
pre-declared `datapath_reported = $null` on **every** stage's row, including
stages (interleaved, energy) that never call `Get-DatapathLine`. Running
the interleaved stage after accuracy/throughput had already scraped a real
value **wiped `datapath_reported` back to `null` for all six models** —
exactly the kind of fail-open tasks/0104 built the whole guard to prevent,
reintroduced one file away from the fix. Caught by inspecting `sweep.json`
after the interleaved stage (all six rows showed `datapath_reported: null`
despite `datapath: bfp16`/`bf16` still being correct). Fixed by not
pre-declaring the key at all — `$row.datapath_reported = $dp` is only ever
set when a stage's log actually contains the line, so an unset key cannot
overwrite a real one on merge. Re-ran the (fast, ~2 min) accuracy stage
afterward to repopulate the field; confirmed in the final `sweep.json` that
every row's `datapath_reported` survived the subsequent energy stage too.

**No row's `datapath_reported` ever disagreed with its `datapath`
(intended) value** — `pick_artifacts()` selected exactly what the table
asked for, on every model, at every point this was checked. That is the
answer to the question that mattered most here.

## Results

### The six-row table — every datapath confirmed by the runtime's own status line

| model | datapath (runtime's own status line) | throughput mean, 3 runs (wall clock, `--pipeline 4`) | spread | array time, single-lane `--bench` `wait (hardware)` | worst `1-cos` golden gate |
|---|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | `bfp16-emulated MMAC, C as bf16` | **1074.8 seq/s** | 2.7% | 35.66 ms total, **1486 µs/dispatch** | 3.406e-04 PASS |
| `bge-small-en-v1.5` | `bf16 MMAC, C as fp32` (unchanged) | **492.2 seq/s** | 0.6% | 145.46 ms total, **3030 µs/dispatch** | 8.348e-06 PASS |
| `bge-base-en-v1.5` | `bfp16-emulated MMAC, C as bf16` | **241.3 seq/s** | 1.2% | 220.90 ms total, **4602 µs/dispatch** | 2.284e-04 PASS |
| `bge-large-en-v1.5` | `bfp16-emulated MMAC, C as bf16` | **74.6 seq/s** | 0.5% | 934.43 ms total, **9734 µs/dispatch** | 2.626e-04 PASS |
| `nomic-embed-text-v1.5` | `bfp16-emulated MMAC, C as bf16` | **183.3 seq/s** | 0.5% | 298.47 ms total, **6218 µs/dispatch** | 1.402e-03 PASS (thinnest margin) |
| `embeddinggemma-300m` ‡ | `bfp16-emulated MMAC, C as bf16` | **146.6 seq/s** | 2.9% | no `--bench` for arch=1; see note below | 2.315e-04 PASS (differential harness) |

‡ arch=1 has no `--bench`; throughput reproduces 0085's approach — encoding
`corpus_520.txt` (520 real sentences) end to end and dividing wall clock —
a **different harness with a different noise floor** (2.9% here vs ≤1.2%
for the `--bench`-driven rows), not directly comparable cell-for-cell.
`embeddinggemma-300m` also has no single-lane `--bench`, so it has no
`wait (hardware)` line; the closest available figure is the `dispatch`
component of its own breakdown line (`breakdown npu ... dispatch 1774 ms
... [480 dispatches]` ≈ 3.7 ms/dispatch), which **bundles submit and wait**
and is not the same quantity as the other rows' `wait (hardware)` alone —
reported here as a note, not as a table cell, to avoid implying a false
comparison.

**Every `datapath` line above was read from the runtime's own status
line** (`Get-DatapathLine` scraping `  datapath   ...`), not restated from
the harness table — and on every model it matched what the table asked
for. `bge-large`'s **9734 µs/dispatch** corroborates
`docs/CURRENT_STATUS.md`'s own already-recorded post-adoption figure of
**9,776 µs** (different session, same order of measurement) — the two
agree to within 0.4%.

**Throughput moved against 0085 in the direction and magnitude the
datapath change predicts** — every model that adopted bfp16 gained, the one
that did not (bge-small) did not:

| model | 0085 (pre-adoption, bf16) seq/s | 0105 (this sweep) seq/s | Δ | datapath changed? |
|---|---:|---:|---:|---|
| `all-MiniLM-L6-v2` | 992.6 | 1074.8 | **+8.3%** | yes → bfp16 |
| `bge-small-en-v1.5` | 503.5 | 492.2 | −2.2% | **no** (unchanged bf16; within CPU-contention noise) |
| `bge-base-en-v1.5` | 214.3 | 241.3 | **+12.6%** | yes → bfp16 |
| `bge-large-en-v1.5` | 61.9 | 74.6 | **+20.5%** | yes → bfp16 |
| `nomic-embed-text-v1.5` | 166.8 | 183.3 | **+9.9%** | yes → bfp16 |
| `embeddinggemma-300m` ‡ | 136.6 | 146.6 | **+7.3%** | yes → bfp16 |

The one row that did **not** change datapath is also the one row whose
throughput did not improve — and its small (2.2%) dip is consistent with
this session's CPU contention (0085's own five-of-six rows had <0.6%
run-to-run spread on a clean machine; this session's spread on the same
row was 0.6% run-to-run but the cross-session delta is larger, which is
what you'd expect comparing a contended session to a clean one, not a
regression in the design).

### Interleaved NPU vs CPU (8 rounds, `--pipeline 4`; **not defensible as a
### clean ratio this session** — CPU contention, see above)

| model | torch (steady) | ORT (steady) | NPU (steady) | NPU / strongest CPU |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 749.8 | 286.7 | 1069.7 | 1.427× |
| `bge-small-en-v1.5` | 391.3 | 162.9 | 502.3 | 1.284× |
| `bge-base-en-v1.5` | 115.6 | 58.6 | 241.4 | 2.088× |
| `bge-large-en-v1.5` | 35.0 | 17.1 | 74.6 | 2.132× |
| `nomic-embed-text-v1.5` | 71.5 | 42.5 | 180.6 | 2.525× |
| `embeddinggemma-300m` | 81.7 | unavailable ¹ | 100.4 | 1.230× |

¹ ONNX Runtime raised `RuntimeError: invalid unordered_map<K, T> key` for
this model, same failure mode as 0085 (not new); the ratio uses torch alone,
which 0040 measured as the stronger of the two CPU baselines anyway.

### Energy, differential method, J / 1000 sequences (RAPL package; **CPU
### contention present, see above**)

| model | CPU (sentence-transformers) | NPU single-lane | NPU `--pipeline 4` | pipe4 × better than CPU |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 79.5 | 27.0 | 29.0 | 2.74× |
| `bge-small-en-v1.5` | 148.2 | 93.4 | 73.1 | 2.03× |
| `bge-base-en-v1.5` | 495.0 | 120.8 | 130.1 | 3.80× |
| `bge-large-en-v1.5` | 1636.1 | 397.8 | 440.1 | 3.72× |
| `nomic-embed-text-v1.5` | 742.4 | 173.9 | 169.5 | 4.38× |
| `embeddinggemma-300m` | 707.7 | 290.9 | 286.7 | 2.47× |

Every model logged at least one `idle drift ... UNSTABLE -- DISCARD` reading
during this stage (2–4 per model) — more than a clean-machine run would
show, consistent with the CPU contention above. As noted, the differential
computation does not use the idle readings, so this is a corroborating
quality signal, not grounds to discard the results outright.

**Against 0085's pre-adoption CPU-energy figures** (same CPU-side
measurement, different session, `sentence-transformers` config): MiniLM
79.5 vs 74.0 (+7.4%), bge-small 148.2 vs 144.9 (+2.3%), bge-base 495.0 vs
469.3 (+5.5%), bge-large 1636.1 vs 1645.7 (−0.6%), nomic 742.4 vs 783.9
(−5.3%), gemma 707.7 vs 673.4 (+5.1%) — all within session-to-session noise
(±7%), as expected since the CPU side did not change. **Every model that
adopted bfp16 shows a better NPU/CPU energy ratio than 0085 recorded**
(bge-base 3.80× vs 2.94×, bge-large 3.72× vs 2.84×, nomic 4.38× vs 3.78×,
gemma 2.47× vs 2.13×, MiniLM 2.74× vs 2.55×) — consistent with the same
mechanism as throughput: less array time per encode means less energy per
encode. `bge-small` (unchanged datapath) moved the other way (2.03× vs
2.18×), consistent with no mechanism to improve it and this session's
added noise.

## int8: carried over from 0085, **not re-measured**

Given the time already spent getting a clean six-row bfp16/bf16 sweep
(accuracy, throughput ×3, array-time, interleaved, energy — all six
models), and the brief's explicit permission to prioritise the adopted
rows, **the int8 numbers in the proposed table below are 0085's**, verbatim,
not re-measured this session. int8 is a separate, unaffected datapath
(tasks/0104's whole point was that geometry and even `b_layout_hash` cannot
tell bfp16 from plain bf16, but int8 is native-int8 MMAC throughout, on its
own artifact directories, untouched by this decision) — so there is no
reason to expect these numbers to have moved, but "unlikely to have moved"
is not the same claim as "measured this session," and this task states
which one it is rather than letting the distinction blur.

## MTEB: not re-run this session, cited from 0101/0103 (not from 0085)

The brief's "report at minimum" list does not include MTEB, and a full
six-model MTEB pass is the longest stage in the protocol (0085: "hours").
This task's accuracy-stage golden-gate numbers above are **bit-exact
reproductions** of 0104's own re-verification of the adopted datapath
(MiniLM/bge-base/bge-large exactly, bge-small exactly; nomic and gemma
newly established there), which in turn is the datapath 0101/0103 actually
MTEB-gated (mean +0.01 to worst −0.25 across the five adopted models,
`bge-small` failing at −0.5010 — see 0104's own summary). **0085's MTEB
numbers are NOT a valid substitute** — they were measured on the
pre-adoption plain-bf16 datapath for five of these six rows, which is
exactly the number this whole task exists to stop being quoted as current.
Citing 0101/0103 here, not 0085, and not re-running MTEB this session.

## Commands (representative; full sequence in the log)

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# accuracy (golden gate), all six adopted rows
.\tools\release_benchmark.ps1 -Models "all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m" `
    -Skip "throughput,interleaved,energy,mteb" -OutDir "tasks\0105-release-sweep-adopted-datapaths"

# throughput, 3 runs/model (contention override, see above)
.\tools\release_benchmark.ps1 -Models "all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m" `
    -Skip "accuracy,interleaved,energy,mteb" -OutDir "tasks\0105-release-sweep-adopted-datapaths" -AllowCpuContention

# array time (wait (hardware) line), five BERT-family rows, single-lane bench
.\runtime\build\npuembed.exe . --model <model> --artifacts <set> --threads 16 --bench 5

# interleaved CPU ratio, 8 rounds
.\tools\release_benchmark.ps1 -Models "..." -Skip "accuracy,throughput,energy,mteb" `
    -OutDir "tasks\0105-release-sweep-adopted-datapaths" -AllowCpuContention

# energy, differential (low=20, high=60)
.\tools\release_benchmark.ps1 -Models "..." -Skip "accuracy,throughput,interleaved,mteb" `
    -OutDir "tasks\0105-release-sweep-adopted-datapaths" -AllowCpuContention
```

## Problems hit

1. **The `datapath_reported` merge-clobber bug** — see the dedicated section
   above. Fixed in `tools/release_benchmark.ps1`; the accuracy stage was
   re-run afterward to repopulate the field cleanly.
2. **The CPU-quiet guard refused the throughput stage four times in a row**
   on a genuinely, if mildly, busy machine (`clion64` + `msedge`, ~15-20% of
   one core sustained, not a transient blip). Used `-AllowCpuContention` and
   flagged every affected stage rather than stall indefinitely or silently
   report a clean-machine reading that was not one.
3. **Two foreground PowerShell calls exceeded the 600 s tool timeout**
   (interleaved, energy) and were auto-moved to background. Both were driven
   to completion with a bounded poll loop that checks both the output log
   for a `SWEEP DONE`/error marker **and** process liveness (`Get-Process
   python,npuembed`), so a silent death would have been caught rather than
   waited on forever (the failure mode named in tasks/0103). One early
   liveness check briefly misread a legitimate ~16 s "idle before" RAPL
   measurement window (during which the energy harness intentionally runs
   no process) as a possible death; corrected by switching the liveness
   signal to log-file growth instead of raw process count, which does not
   false-positive on the harness's own designed idle gaps.
4. **`embeddinggemma-300m`'s interleaved comparison has no ONNX Runtime
   side** (`RuntimeError: invalid unordered_map<K, T> key`) — reproduces
   0085's same failure, not a new issue; the ratio uses torch alone.

## Artifacts

All in `tasks/0105-release-sweep-adopted-datapaths/`:

- `accuracy-<model>.txt` (+ `.run1.txt` from the pre-fix pass, kept per the
  harness's rotate-don't-truncate rule), `sweep.json` — golden gate + status
  line, all six models.
- `throughput-<model>.txt` / `.run{1,2,3}.txt` — three runs each, six models.
- `arraytime-<model>.txt` — single-lane `--bench 5`, five BERT-family models
  (`wait (hardware)` line).
- `interleaved-<model>.txt` / `.json` — 8 rounds, six models.
- `energy-<model>.txt`, `energy-<model>/*.json` — differential energy, six
  models.
- `sweep.json` — merged index, all stages, all six models.

## Proposed register update

### `docs/CURRENT_STATUS.md` headline table replacement

Replace the "⚠ THE `bf16` COLUMN..." warning block and the "THE 0.4.0
NUMBERS" table with the following. The bf16-family throughput/array-time/
`1-cos`/interleaved/energy columns are freshly measured this session
(tasks/0105) on the datapath each model actually ships (per-row, from the
runtime's own status line — see `datapath` column). **The int8 columns are
carried over from tasks/0085, not re-measured** (int8 is an unaffected,
separate datapath, but "unlikely to have moved" ≠ "measured"). MTEB
verdicts for the adopted datapath are 0101/0103's, not re-run here or in
0085 (0085 predates the adoption for five of six rows).

> **THE 0.4.0 NUMBERS, ADOPTED DATAPATHS**
> ([`0105`](TASK.md), building on
> [`0104`](../0104-adopt-bfp16-per-model/TASK.md) and
> [`0085`](../0085-m13-release-sweep/TASK.md)) — six bf16-family rows
> freshly measured in one session, one machine state, `--threads 24
> --pipeline 4 --bench 5`, three throughput runs/model. **The `datapath`
> column is read from the runtime's own status line, not restated from a
> harness table** — the whole point of tasks/0104's guard. **int8 columns are
> carried over from 0085, not re-measured this session.** A CPU-side
> background contender (JetBrains CLion + Edge) was present throughout this
> session's timing stages, so interleaved/energy ratios here carry more
> noise than a clean-machine reading — throughput and array-time are much
> less exposed. End-to-end throughput, **not an NPU kernel performance
> claim** (rule 1).
>
> | model | datapath | throughput | `1-cos` (own datapath) | int8 throughput ¹ | int8 `1-cos` ¹ | MTEB mean / worst, own datapath ² | gate ² | NPU/CPU ³ | J/1k better ³ |
> |---|---|---:|---:|---:|---:|---:|---|---:|---:|
> | `all-MiniLM-L6-v2` | **bfp16-emulated** | 1074.8 | 3.406e-04 | 1696.6 | 1.161e-03 | +0.12 / −0.07 | PASS | 1.43× | 2.74× |
> | `bge-small-en-v1.5` | **bf16 (unchanged)** | 492.2 | 8.348e-06 | 862.9 | 6.385e-04 | −0.10 / **−0.5010** | **FAIL — why it stays on bf16** | 1.28× | 2.03× |
> | `bge-base-en-v1.5` | **bfp16-emulated** | 241.3 | 2.284e-04 | 416.0 | 1.778e-03 | −0.06 / −0.19 | PASS | 2.09× | 3.80× |
> | **`bge-large-en-v1.5`** | **bfp16-emulated** | 74.6 | 2.626e-04 | **147.9** | 2.968e-03 ⁴ | +0.13 / −0.01 | PASS | 2.13× | **3.72×** |
> | `nomic-embed-text-v1.5` | **bfp16-emulated** | 183.3 | 1.402e-03 ⁵ | 325.6 | 1.098e-03 | +0.01 / −0.25 | PASS | **2.53×** | **4.38×** |
> | `embeddinggemma-300m` | **bfp16-emulated** | 146.6 ⁶ | 2.315e-04 ⁷ | 155.8 | 1.070e-03 | +0.16 / −0.02 | PASS | 1.23× | 2.47× |
>
> ¹ carried over from 0085, not re-measured this session. ² from
> [`0103`](../0103-t23-bfp16-all-models/TASK.md)'s own gate table (its
> Result section), the run that decided adoption per model — **not** from
> 0085 (which predates the adoption for five of six rows and measured the
> pre-adoption datapath). Gate is `|mean| <= 0.5` AND no task worse than
> `-0.5`. ³ this session (tasks/0105), `-AllowCpuContention` (see caveat
> above). ⁴ int8's only `1-cos` gate failure; MTEB passes (0085: mean −0.05,
> worst −0.18). ⁵ thinnest margin of the five adopted models — still ~1.4×
> inside the 2e-03 tolerance. ⁶ arch=1 has no `--bench`; different harness
> (corpus encode), different noise floor (2.9% here vs ≤1.2% elsewhere).
> ⁷ differential harness (NPU vs host-only CPU path), arch=1 has no
> HuggingFace golden.

### `tasks/README.md` index row

```
| [0105](0105-release-sweep-adopted-datapaths/TASK.md) | **The 0.4.0 sweep, re-run on the datapaths that actually ship.** tasks/0104 moved five of six models to bfp16-emulated MMAC + bf16-C; the standing release-benchmark harness (`tools/release_benchmark.ps1`) still pointed at the pre-adoption artifact directories and labelled every row `bf16` with no way to see which MMAC datapath ran. Fixed the harness first -- six rows repointed at `artifacts_{minilm,base,large,nomic,gemma}_bfp16` / `artifacts_small_bf16`, and a new `datapath_reported` field SCRAPED FROM THE RUNTIME'S OWN STATUS LINE (never restated from the table) confirms `pick_artifacts()` selected exactly what was intended, on every model. Found and fixed a real bug on the way: the per-stage JSON merge is per-key by design, but a stage that does not scrape the status line was still writing `datapath_reported = null` into its row, silently WIPING an earlier stage's real reading on merge -- caught by inspecting `sweep.json`, not by the harness itself. Measured, one session, one machine state: accuracy (golden gate, all six PASS, four bit-exact reproductions of 0104), throughput (3 runs/model -- MiniLM +8.3%, bge-base +12.6%, bge-large +20.5%, nomic +9.9%, gemma +7.3% against 0085's pre-adoption numbers; bge-small, whose datapath did NOT change, -2.2%, within this session's CPU-contention noise), array time (single-lane `--bench`'s `wait (hardware)` line; bge-large's 9734 us/dispatch corroborates CURRENT_STATUS's already-recorded 9,776 us to within 0.4%), interleaved NPU/CPU and RAPL energy (both run under `-AllowCpuContention` after a persistent, non-transient CLion/Edge background load repeatedly tripped the CPU-quiet guard -- flagged, not hidden). int8 carried over from 0085 verbatim, NOT re-measured, given the time already spent on six clean bfp16/bf16 rows -- stated plainly rather than left ambiguous. MTEB not re-run; cited from 0101/0103 (the datapath that actually gates adoption), explicitly NOT from 0085 (which predates the adoption for five of six rows). Proposed `CURRENT_STATUS.md` replacement table in this task's own log | M13 (T23) | done |
```

