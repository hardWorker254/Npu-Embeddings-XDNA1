# 0073 — M13: the 0.3.0 whole-catalogue benchmark sweep

**Date:** 2026-08-21
**Status:** DONE — all four stages, all five NPU models

## Goal

Measure **every model in the catalogue**, in one session, on one machine state,
with one protocol — and make that a property of the release build rather than of
whoever remembers to re-run things.

## Context

The user's standing instruction: *"Benchmark it! All releases should have
updated benchmarks, for new and old models!"*

The numbers this repo shipped before today were a patchwork. Throughput from one
night, the interleaved CPU ratio from another, energy from
[`0034`](../0034-m8-energy/TASK.md) (MiniLM only), MTEB for two of five models.
Every individual figure was honest when it was taken. The **table** was not
comparable to itself: different lane defaults (2 vs 4), different mlir-aie
versions, different machine states. A row-by-row patchwork misleads even when no
row lies.

Coverage before this task, and the holes:

| model | seq/s @4 lanes | interleaved ratio | energy | MTEB |
|---|---|---|---|---|
| MiniLM-L6 | 962.6 | 1.792× | 44.0 J/1000 | ✅ 0035 |
| bge-small | — (a 2-lane 445) | 1.533× | — | ❌ |
| bge-base | 209.1 | 1.633× | — | ✅ 0053 |
| bge-large | — (a 2-lane 52.8) | 2.106× (mixed provenance) | — | ❌ |
| embeddinggemma-300m | ~0.13 (host) | n/a | — | ❌ |
| **nomic** | — | — | — | — |

## What was done

### The harness could not measure the catalogue it was about to sweep

Before a single number could be taken, five things had to be fixed. Committed
separately (`f124f31`, `30b63d6`) because they are bugs, not measurements.

- **`energy_compare.ps1` had no `-Model` parameter at all**, and every
  `npuembed` invocation in it omitted `--model` — required since
  [`0038`](../0038-m9-model-driven-runtime/TASK.md) made selection explicit once
  a second model existed. **The script has been unrunnable since then and nobody
  noticed**, because the one energy figure on record was taken in 0034 when
  there was only one model to measure. That is the entire explanation for why
  energy exists for MiniLM and nothing else. It also measured **2 lanes** while
  the production default moved to 4 in [`0052`](../0052-m10-research-night/TASK.md),
  so its figure described a configuration we no longer ship.
- **`npu_encoder.py` could not load nomic at all** — it builds
  `emb = word + pos + typ` and `pick()`s the position table unconditionally;
  arch=2 has none.
- **`npu_encoder.py` mislabelled every non-MiniLM MTEB result.**
  `mteb_model_meta` hardcoded `name="NpuEmbeddings/all-MiniLM-L6-v2-npu"` while
  `revision` was already parameterised. A fail-open that mislabels rather than
  breaks; it survived because only two models had ever been run.
- **`run_mteb.py --out` was one constant path**, so a six-model sweep would have
  left only the last model's numbers. Fourth instance of the bug class
  [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md) named.
- **`run_mteb.py --sides npu` produced no verdict at all** — a table of
  `-- -- --`, no gate block, `"pass": null`. It now says loudly that a delta
  gate has nothing to measure with one side.

The **task prefix** is now read from the container in all three harnesses and
applied to **every** side. It is part of the work — it lengthens the sequence —
so a throughput or energy figure that omits it on one side measures different
sequences, and an MTEB delta that omits it measures the prefix rather than the
datapath.

### Machine state

`no battery device reported` — recorded **by tool**
(`Get-CimInstance Win32_Battery` returning an empty set), not by a hand-rolled
check. [`0040`](../0040-m9-honest-cpu-baseline/TASK.md) reported "ON BATTERY"
for a machine with no battery because the `else` branch fired on an absent data
source.

NPU contention checked before starting: four `WorkloadsSessionHost.exe` hardware
contexts are resident, all with status **`Idle`**. The `--bench` guard refuses
only on a foreign **Active** context, and it passed on every run below. **No run
used `--allow-contention`.**

### Stage 1 — accuracy and throughput (CONTAMINATED — see Stage 2d for the real numbers)

> Everything in this section was measured while a stray `find` held one core
> (Stage 2c). The **accuracy** figures are unaffected — they are arithmetic, not
> timing — and reproduce exactly in the clean run. The **throughput** figures
> are all slightly low and are superseded by
> [Stage 2d](#stage-2d--the-clean-run-these-are-the-numbers). Kept because the
> comparison between the two is what shows the contamination's shape.

Accuracy first, because a throughput number for a model that computes the wrong
thing is worthless. All five reproduce their recorded figures **exactly**:

| model | `rel_fro` vs HF golden | |
|---|---:|---|
| all-MiniLM-L6-v2 | 4.473e-03 | PASS |
| bge-small-en-v1.5 | 3.789e-03 | PASS |
| bge-base-en-v1.5 | 4.297e-03 | PASS |
| bge-large-en-v1.5 | 3.763e-03 | PASS |
| nomic-embed-text-v1.5 | 6.119e-03 | PASS |

Throughput, `--threads 24 --pipeline 4 --bench 5`, three runs each, **end-to-end
throughput and not an NPU kernel performance claim** (CLAUDE.md rule 1):

| model | runs (seq/s) | mean | spread | previous record |
|---|---|---:|---:|---|
| all-MiniLM-L6-v2 | 970.6 / 943.2 / 949.3 | **954.4** | 2.9% | 962.6 (0052) |
| bge-small-en-v1.5 | 480.8 / 463.4 / 454.3 | **466.2** | 5.7% | 445 @ 2 lanes |
| bge-base-en-v1.5 | 201.8 / 203.4 / 202.9 | **202.7** | 0.8% | 209.1 (0052) |
| bge-large-en-v1.5 | 59.1 / 59.8 / 58.7 | **59.2** | 1.9% | 52.8 @ 2 lanes |
| **nomic-embed-text-v1.5** | 156.2 / 153.8 / 154.3 | **154.8** | 1.6% | — (first) |

**On the nomic prior — and a correction I had to make to my own scoring.**
[`0069`](../0069-m13-nomic-arch2-container/TASK.md) predicted **150–160 seq/s**
before measuring, from one argument: nomic does ~1.33× bge-base's per-layer GEMM
work, because the gated `ffn_up` emits both halves, at the same depth.

Against these contaminated numbers that looked like a **1.6% hit** (202.7 / 1.33
= 152.4 vs a measured 154.8), and it was written up as one. Against the clean
run it is 211.0 / 1.33 = **158.6 vs a measured 165.5** — the prior was **4%
low**, a near miss rather than a hit.

**A prediction scored against a number that later turns out to be wrong has not
been tested**, and the first scoring is left here rather than quietly replaced,
because congratulating yourself on a contaminated measurement is precisely the
failure mode this project keeps writing guards against.

What survives: [`0051`](../0051-m9-bge-base-and-in-exe-fetch/TASK.md) recorded a
**27% miss** on the same kind of prediction, diagnosed as 0048's iteration fit
being extrapolated across a width doubling it was never tested at. This one was
made *at* h=768 and changed only the FFN width, and lands within 4%. The
iteration-count model ([T1](../../research/OPEN-THREADS.md)) extrapolates
usefully within a width and badly across one — supported now by a near miss as
well as a large one.

### Stage 2 — interleaved CPU ratio (DONE, two rows re-measuring)

`compare_three.py`, 8 rounds, round-robin NPU / torch / ORT in one process,
steady state = mean of the last four (the choice that favours the CPU, since
torch ramps). All five in one session.

| model | NPU | torch | ORT | ratio | on record | |
|---|---:|---:|---:|---:|---:|---|
| all-MiniLM-L6-v2 | 871.8 | 477.4 | 228.4 | **1.826×** | 1.792× (0040) | reproduces |
| bge-small-en-v1.5 | 383.4 | 204.4 | 85.8 | 1.876× | 1.533× (0040) | **unstable** |
| bge-base-en-v1.5 | 188.4 | 76.7 | 39.0 | 2.456× | 1.633× (0053) | **torch 31% low** |
| bge-large-en-v1.5 | 55.1 | 26.4 | 12.2 | **2.083×** | 2.106× (0042) | reproduces |
| **nomic-embed-text-v1.5** | 148.9 | 53.8 | 30.2 | **2.767×** | — | first |

**nomic has the best ratio in the catalogue.** That is consistent with
[`0027`](../0027-m7-width-hypothesis/TASK.md)'s 1 : 3h argument rather than
surprising: it is wide (768), and its gated FFN puts *even more* of the work
into GEMMs — where the array wins — instead of the elementwise glue, where it
does not. Two things worth stating because they could have gone the other way:
the ONNX export **traced** the custom architecture, so all three sides are real
rather than two; and the task prefix reached **all three sides**, which the log
records as `task prefix (all three sides): 'search_document: '`.

**Two rows are not publishable yet.** MiniLM and bge-large reproduce their
recorded ratios closely (1.826 vs 1.792, 2.083 vs 2.106). bge-small and bge-base
do not, and for different reasons:

- **bge-small is unstable within the run.** torch spans **113.0 – 335.6 seq/s**
  across eight rounds and ORT 41.5 – 143.9, while its NPU side holds to
  359.5 – 431.4 (1.2×). A "mean of the last four" over a 3× swing is not a
  statistic.
- **bge-base is stable and simply low.** torch 64.5 – 78.9, a tight spread, but
  76.7 against the 111.2 recorded in [`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md)
  — 31% down with no run-to-run noise to explain it.

Two of five reproducing and two not is a question, not a result, so both are
being re-measured before either number leaves this file.

**The protocol is working, and that is why every side dropped.** Against their
isolated stage-1 measurements: bge-small's NPU 466.2 → 383.4 (−18%), bge-base's
202.7 → 188.4 (−7%). During an interleaved run torch and ORT saturate all twelve
cores between the NPU rounds, so the NPU rounds execute on a loaded machine.
That is the comparison being fair, not a regression.

**A structural limit the sweep exposes about itself:** models run back to back,
so later ones are measured on a machine that has been hot for half an hour.
[`0040`](../0040-m9-honest-cpu-baseline/TASK.md)'s rule — interleaving cancels
drift that hits both sides — holds *within* a model and says nothing about the
machine state *between* models. A ratio from one sweep is comparable within a
model and only loosely across them. Recorded rather than silently fixed:
randomised order or a cooldown changes what "one session" means, and that is the
next sweep's decision.

bge-small and bge-base look like large improvements. **They are not, and the
ratio is not the quantity that moved.** Every side is slower than its own
isolated measurement:

| model | NPU isolated (stage 1) | NPU during interleave | |
|---|---:|---:|---|
| bge-small | 466.2 | 383.4 | −18% |
| bge-base | 202.7 | 188.4 | −7% |

That direction is expected and is the protocol working: during an interleaved
run, torch and ORT saturate all twelve cores between the NPU rounds, so the NPU
rounds execute on a loaded machine. What is *not* fine is the spread.

**bge-small's CPU sides swing by a factor of three within one measurement** —
torch **113.0 – 335.6** seq/s across eight rounds, ORT 41.5 – 143.9 — while its
NPU side holds to 359.5 – 431.4 (1.2×). bge-base, measured minutes later, is
stable on every side (torch 64.5 – 78.9). A "steady state = mean of the last
four rounds" statistic is not meaningful over a 3× swing, so **bge-small's
1.876× is not a number this release should quote** without a re-measurement on a
settled machine.

[`0040`](../0040-m9-honest-cpu-baseline/TASK.md) established that the ratio is
the defensible quantity because interleaving cancels drift that hits both sides.
That holds. It does **not** cover a single side swinging 3× inside one run,
which is a different failure and one this sweep is the first thing to expose —
because it is the first thing to measure five models back to back.

**A second, structural point the sweep surfaces about itself:** models are
measured sequentially, so the later ones run on a machine that has been hot for
half an hour. Interleaving protects the ratio *within* a model; it does nothing
about the machine state *between* models. Comparing two models' ratios from one
sweep therefore carries a bias the per-model protocol does not remove. Recorded
rather than fixed — the fix (randomised order, or a cooldown between models) is
cheap but changes what "one session" means, and that is a decision for the next
sweep rather than a silent edit to this one.

### Stage 2b — the re-measurement, and what a CPU ratio is worth here

bge-small, same machine, ~40 minutes later:

| run | torch (steady) | torch range | NPU | ratio |
|---|---:|---|---:|---:|
| 1 | 204.4 | 113.0 – 335.6 | 383.4 | 1.876× |
| **2** | **336.7** | **258.1 – 369.2** | 442.8 | **1.315×** |
| 0040's record | 290.0 | — | 445 | 1.533× |

The two runs are **43% apart**, and run 1's own spread (113–336) brackets run 2's
mean. So neither is "the" number and the record sits between them. torch on this
model, on this machine, is simply not reproducible to better than that in an
8-round window; the NPU side, by contrast, is tight in both runs (359–431, then
435–449).

**What this means for what the release may claim.** Ranking the four BERT models
by how well their ratio reproduced:

| model | reproduces? | evidence |
|---|---|---|
| all-MiniLM-L6-v2 | **yes**, 1.9% | 1.826× vs 1.792× recorded |
| bge-large-en-v1.5 | **yes**, 1.1% | 2.083× vs 2.106× recorded |
| bge-small-en-v1.5 | **no**, 43% | 1.876× then 1.315×, same hour |
| bge-base-en-v1.5 | **within today**, 1.1% | 2.456× then 2.429× — but 1.633× on record |

bge-base turned out to be the informative case. It is **stable today**
(2.456× and 2.429×, torch 76.7 then 78.4) and simply disagrees with the record.
The gap is entirely on the CPU side:

| | recorded (0051/0053) | today | |
|---|---:|---:|---|
| NPU | 209.1 | 202.7 | −3% |
| torch | 111.2 | 78.4 | **−30%** |

Same `torch 2.10.0+cpu`, same `transformers 5.15.0`, same twelve threads —
checked, because a library upgrade would have been the tidy explanation and it
is not available. No cause identified; the remaining candidates are machine
state (thermal, power plan, background load), which is unfalsifiable after the
fact.

**Which is the point.** A cross-session comparison of CPU numbers is not
reliable, and that is precisely the failure this sweep was built to stop. The
release quotes today's session because both sides of it were measured together;
it does not claim bge-base got 50% better.

So: **NPU throughput is the reproducible quantity** — 0.8–5.7% spread across
three runs per model, and every model's accuracy figure reproduced to the
recorded digit. **The CPU ratio is not**, on two of four models, and no amount
of interleaving fixes a side that swings 43% between runs.
[`0040`](../0040-m9-honest-cpu-baseline/TASK.md) chose the ratio over the
absolute because interleaving cancels drift hitting both sides — correct, and it
is why MiniLM and bge-large reproduce. It does not cover a CPU side that is
independently unstable.

**The release therefore leads with NPU throughput and reports ratios with their
measured run-to-run variation**, naming which reproduce and which do not, rather
than choosing the flattering number. bge-small's 1.876× would have been the
flattering one, and it is the one being discarded.

### Stage 2c — THE CAUSE: a stray process of my own ate a core for 2.5 hours

Everything above about "unexplained CPU-side slowness" has one explanation, and
it is not subtle. The user suggested the CPU had been busy. Checking properly:

```
   Id Name CPU_s StartTime
   -- ---- ----- ---------
28456 find  9561  8/21/2026 8:46:53 PM

delta over 1.5s: 1.48 cpu-seconds     <- STILL RUNNING, ~1 full core
```

A `find` launched by an earlier command in this session — searching for a file
that a `grep` would have found — had been burning **one full core continuously
since 20:46**, straight through every interleaved measurement taken between
22:45 and 23:30.

**The effect is exactly what the data showed.** One core of twelve is 8% of the
machine, but torch runs twelve threads and contends directly, while the NPU path
offloads the arithmetic to the array and only needs its host threads. So the CPU
sides came out ~30% low with wild variance, the NPU sides barely moved, and
**the ratio looked like a large improvement.**

That is the same shape as
[`0044`](../0044-m9-optimisation-sweep/TASK.md)'s ninth fail-open — contention
that hits only ONE side makes the ratio *confidently wrong* rather than merely
noisy — and 0044's fix was a guard against a foreign process holding an NPU
context. **Nothing checked the CPU side.** The reasoning transfers completely
and nobody had transferred it.

**Two lessons, one methodological and one about cumulative counters.**
`Get-Process | Sort-Object CPU` shows *lifetime* CPU time, so a process that
finished an hour ago still tops the list and a process burning a core right now
looks the same as one that burned it yesterday. Only the **delta over a sampling
window** distinguishes them, which is why the earlier "top processes by CPU"
check did not flag it.

Fixed: `release_benchmark.ps1` now has a **CPU contention guard** — mean load
over a sampling window, plus a delta-based scan for any process holding >20% of
a core, excluding the sweep's own children. It refuses by default and
`-AllowCpuContention` says in the artifact that no ratio from that run is
defensible. `sweep.json` records the measured load either way, so a reader never
has to assume the machine was idle.

**Every interleaved number in Stage 2 and 2b is therefore contaminated and is
being re-measured.** They are kept above rather than deleted, because the
*pattern* they show — CPU side wrecked, NPU side steady — is what identified the
cause.

### Stage 2d — THE CLEAN RUN. These are the numbers.

Everything above is diagnosis. With the stray `find` killed and the machine
verified quiet (1.4% mean load, delta-checked), the whole sweep was re-run.

**Throughput, isolated**, three runs each — note the spreads against the
contaminated run's:

| model | runs (seq/s) | mean | spread | contaminated |
|---|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 958.0 / 934.4 / 961.4 | **951.3** | 2.8% | 954.4 (2.9%) |
| bge-small-en-v1.5 | 494.1 / 495.0 / 493.7 | **494.3** | **0.3%** | 466.2 (5.7%) |
| bge-base-en-v1.5 | 210.4 / 211.2 / 211.5 | **211.0** | 0.5% | 202.7 (0.8%) |
| bge-large-en-v1.5 | 60.8 / 60.8 / 60.8 | **60.8** | **0.0%** | 59.2 (1.9%) |
| nomic-embed-text-v1.5 | 165.9 / 165.5 / 165.2 | **165.5** | 0.4% | 154.8 (1.6%) |

bge-large returns three identical figures. bge-small's spread collapses from
5.7% to 0.3%.

**Interleaved**, 8 rounds, steady = last four:

| model | NPU | torch | ORT | ratio | on record | |
|---|---:|---:|---:|---:|---:|---|
| all-MiniLM-L6-v2 | 942.2 | 508.7 | 245.2 | **1.852×** | 1.792× | +3.3% |
| bge-small-en-v1.5 | 494.0 | 309.1 | 134.2 | **1.598×** | 1.533× | +4.2% |
| bge-base-en-v1.5 | 210.5 | 84.5 | 41.3 | **2.491×** | 1.633× | **+53%** |
| bge-large-en-v1.5 | 60.7 | 24.6 | 11.2 | **2.471×** | 2.106× | +17% |
| nomic-embed-text-v1.5 | 164.8 | 73.0 | 44.0 | **2.258×** | — | first |

MiniLM and bge-small now reproduce within 4%. **bge-base does not, and the
contamination does not explain it** — its contaminated and clean runs agree
(2.456× and 2.491×), so the `find` barely touched it. Its torch side is 84.5
here against 111.2 in [`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md),
both with tight spreads, same library versions. **Unexplained.**

Worth noting *which* models the contamination hit: bge-small −15% and nomic
−18% on their ratios, bge-base and bge-large essentially untouched. One extra
busy core disrupts a fast, scheduling-sensitive workload far more than a slow,
compute-saturated one.

### Stage 2e — the result this whole sweep was not looking for

Put the isolated and interleaved NPU columns side by side:

| model | isolated | with torch+ORT saturating 12 cores | |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 951.3 | 942.2 | −0.9% |
| bge-small-en-v1.5 | 494.3 | 494.0 | −0.1% |
| bge-base-en-v1.5 | 211.0 | 210.5 | −0.2% |
| bge-large-en-v1.5 | 60.8 | 60.7 | −0.2% |
| nomic-embed-text-v1.5 | 165.5 | 164.8 | −0.4% |

**Under 1.5% on every model, and under 0.5% on four of five.** The project's
stated goal — get the work off the CPU cores — has never before been measured
*as such*. Every previous number was a throughput ratio, which answers "is it
faster" rather than "did the work actually move". This answers the second, and
it is a stronger result than the first.

It also reframes the ratio. A ratio of 1.85× on a *contended* machine is not the
same claim as 1.85× on an idle one, and the contended figure is the one that
describes real use: something else is always running.

### Stage 3 — energy (DONE)

Differential method, package RAPL, `-Low 20 -High 60`:

| model | CPU J/1000 | NPU 1 lane | NPU 4 lanes | best |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 112.4 | 31.2 | **30.7** | **3.65×** |
| bge-small-en-v1.5 | 195.6 | 72.2 | **65.3** | **3.00×** |
| bge-base-en-v1.5 | 687.6 | **176.6** | 206.4 | **3.33×** |
| bge-large-en-v1.5 | 1932.1 | **550.5** | 589.1 | **3.28×** |
| nomic-embed-text-v1.5 | 786.0 | 222.8 | **209.6** | **3.75×** |

**Not comparable to [`0034`](../0034-m8-energy/TASK.md)'s 1.94×.** That was two
lanes; production is four, and MiniLM's NPU side went 44.0 → 30.7 J/1000 largely
because of it. Same method, different configuration — which is exactly the kind
of cross-session comparison this sweep exists to stop making.

Note that **more lanes are not always better for energy**: bge-base and
bge-large are cheaper per sequence at one lane than at four, while the three
smaller models are cheaper at four. Not investigated.

nomic tops the energy table (3.75×) while sitting third on the throughput ratio
(2.26×). Energy is power × time and the array runs at markedly lower power, so a
model can be middling on one and best on the other — worth knowing before
quoting either as "the" number.

### Stage 4 — MTEB

Five tasks, both sides in one session, same checkpoint, 64-token truncation on
both. Gate: `|mean| ≤ 0.5` **and** no single task worse than −0.5.

| model | mean Δ | worst | | |
|---|---:|---:|---|---|
| all-MiniLM-L6-v2 | **+0.03** | −0.01 | PASS | 0035 recorded +0.04 |
| bge-small-en-v1.5 | **−0.03** | −0.14 | PASS | **first ever** |
| bge-base-en-v1.5 | **+0.02** | −0.00 | PASS | 0053 recorded +0.023 |
| bge-large-en-v1.5 | **+0.05** | −0.00 | PASS | **first ever** |
| nomic-embed-text-v1.5 | **+0.09** | −0.03 | PASS | **first ever** |
| embeddinggemma-300m | — | | | **deliberately unmeasured** |

**Three of five had never been through this gate.** That is the whole reason the
harness had to be repaired before the sweep could begin: `npu_encoder.py` could
not load nomic at all, and it filed every non-MiniLM result under MiniLM's name.
A tool used on one model for a year does not stay correct for six.

The two with history reproduce closely (+0.03 vs +0.04, +0.02 vs +0.023). The
deltas are tight enough to be uninteresting, which is the point: bf16 multiplies
with fp32 accumulation, and the pre-tiled weight layout, cost **nothing
measurable downstream**. `1 − cos` cannot establish that — it measures fidelity
to a reference, and a systematic distortion can be small in cosine and still
hurt retrieval. Several individual tasks land within ±0.01 points, and
bge-large's SICK-R and STS12 are identical to two decimals.

The one consistent outlier is **TwentyNewsgroupsClustering, always positive**
(+0.17, +0.25 on the two largest models). Clustering scores are the noisiest of
the five and this is well inside the gate, so it is recorded rather than
explained.

`embeddinggemma-300m` is **not measured, and that is a decision rather than a
gap**: at ~7.9 s/sentence on the host path a full run is impractical. Stating it
beats a blank cell that reads as an oversight.

### Stage 4b — nomic FAILED, and the failure was in the harness

First run: **mean −0.68, worst −4.54**, well outside the gate. The number alone
would have read as "the new architecture costs accuracy". The *pattern* said
otherwise:

| task | delta | |
|---|---:|---|
| STSBenchmark | +0.02 | cosine |
| SICK-R | +0.01 | cosine |
| STS12 | −0.03 | cosine |
| **Banking77Classification** | **−4.54** | logistic regression on raw features |
| **TwentyNewsgroupsClustering** | **+1.15** | k-means on raw features |

The three cosine tasks agree to ±0.03 — in family with every other model. Only
the two tasks that consume the embeddings as **raw features** move, and they
move in opposite directions, which is not what a precision loss looks like.

**Cause: the two sides had different vector scales.** The runtime always
L2-normalises (`g_l2_normalize`, hardcoded true). Four of the five models
normalise on the sentence-transformers side too — their `modules.json` ends in a
`Normalize` module — so the sides matched **by luck**, and nobody had to notice.
nomic's pipeline is `Transformer + Pooling` only and `encode()` returns vectors
of norm ≈20.9.

Cosine cannot see a scale difference. A logistic-regression classifier and a
k-means clustering can see nothing else.

Fixed by normalising the CPU side unconditionally in `run_mteb.py` — a no-op for
the four that already did it. **The failing numbers are kept above**, because the
pattern in them is what identified the cause; the number on its own would have
sent us looking at the SwiGLU kernel.

**And it is not only a harness fact.** A user who runs nomic through
sentence-transformers and through this runtime gets vectors of *different scale*
for the same text. Cosine agrees. A downstream classifier trained on one will
not transfer to the other. That belongs in the model's documentation, not just
in a test fixture.

### Stage 4 — MTEB (PENDING)

## Commands, in order

```powershell
# Stage 1
.\tools\release_benchmark.ps1 -Skip interleaved,energy,mteb -Bench 5

# Stage 2
.\tools\release_benchmark.ps1 -Skip accuracy,throughput,energy,mteb -Rounds 8
```

## Problems hit

**`2>&1` on a native command aborted the sweep on its first interleaved run.**
Symptom: `NativeCommandError` from `python.exe` with no message, killing the
whole script. Cause: CLAUDE.md's own PowerShell warning — Windows PowerShell 5.1
wraps each stderr line of a native command in an `ErrorRecord`, and under
`$ErrorActionPreference = "Stop"` that aborts on a program which merely printed
to stderr and returned 0. The sweep's most important line — the applied task
prefix — goes to stderr, so it could not simply be dropped. Fixed with an
`Invoke-Logged` helper that lets **cmd.exe** do the redirection, so PowerShell
only ever sees a file, and which returns the real exit code. The
`energy_compare.ps1` call keeps `2>&1` because it is a PowerShell script, not a
native executable, where the trap does not apply.

## Artifacts

`sweep.json` (index), `accuracy-<model>.txt`, `throughput-<model>.txt`,
`interleaved-<model>.txt`, `energy-<model>/`, `mteb-<model>.json`.

## Next

Fill stages 2–4, then `tools/make_release.ps1` reads `sweep.json` directly — it
now **refuses to assemble a release without it**, which is what makes "every
release ships fresh whole-catalogue benchmarks" mechanical rather than
remembered.
