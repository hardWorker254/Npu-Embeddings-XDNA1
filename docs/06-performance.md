# Performance — what was measured, and what the numbers are worth

This document holds the results and, more importantly, the **caveats**. The
README carries a short table; everything about how much to trust it is here.

For *how* to measure on this hardware — traces, static instruction counts, the
rule that wall clock is never an NPU performance claim — see
[`05-measurement/`](05-measurement/README.md). That document is doctrine; this
one is results.

---

## What the workload actually is

Sentence embedding is not a chatbot. It is a fixed, small model applied to a
large number of short inputs, usually as a background job behind a search index
or a RAG pipeline. Nobody watches it. What matters is that it finishes, and
that it does not take the machine hostage while it does.

So the goal here is **not** to beat the CPU in wall clock. It is to get that
work **off the CPU cores** without losing throughput or quality — leaving the
cores for whatever the user is actually doing. A result of "the same speed on a
third of the cores, at a third of the energy" is a win for this workload even
though it is a draw on the headline number.

## How these numbers were taken

Every figure in the table comes from one sweep
(`tools/release_benchmark.ps1`), which measures the whole catalogue in **one
session, one machine state, one protocol**. That constraint exists because
earlier releases quoted numbers gathered across different sessions at different
pipeline depths — each honest when taken, and collectively not comparable to
itself.

Three guards run before anything is measured:

- **NPU contention.** `--bench` refuses if any foreign process holds an
  `Active` hardware context. A resident NPU context hits only our side, so it
  makes the ratio *confidently wrong* rather than merely noisy
  ([`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) measured 221.4 seq/s
  against a true 691.0 that way).
- **CPU contention.** The mirror of the above, and it exists because we were
  bitten by it: a stray `find` burned one full core for two and a half hours
  straight through a measurement series
  ([`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)). torch runs twelve
  threads and contends directly; the NPU path offloads the arithmetic and
  barely notices — so the CPU side collapsed, the NPU side held, and the ratio
  looked like a large improvement. The guard samples the **delta** over a
  window, because cumulative CPU time cannot tell a process burning a core now
  from one that finished an hour ago.
- **Machine power state**, recorded by tool rather than by a hand-rolled check
  — [`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) once reported
  "ON BATTERY" for a machine with no battery, because an absent data source
  fell through to the negative branch.

The CPU comparison is **interleaved**: NPU, torch and ONNX Runtime round-robin
inside one process, same statistic on every side, steady state taken as the
mean of the last half of the rounds. torch ramps — measured going 469 → 686
seq/s *within* a single run — so an all-rounds mean measures how cold the
machine was as much as how fast the code is, and taking the steady state is the
choice that favours the CPU.

## Results

Ryzen AI 9 HX 370, batch 128, sequence 64, `--threads 24 --pipeline 4`.
One session, 2026-08-22 ([`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)).

### Throughput, isolated

Three runs per model, mean, end-to-end. **Not an NPU kernel performance
claim** — this is the whole pipeline including host work.

| model | seq/s | spread over 3 runs |
|---|---:|---:|
| all-MiniLM-L6-v2 | **951** | 2.8% |
| bge-small-en-v1.5 | **494** | 0.3% |
| bge-base-en-v1.5 | **211** | 0.5% |
| bge-large-en-v1.5 | **60.8** | 0.0% |
| nomic-embed-text-v1.5 | **166** | 0.4% |

### Against the CPU, interleaved

All three encoders round-robin in one process, steady state = mean of the last
four of eight rounds.

| model | **NPU** | torch | ONNX Runtime | **NPU / best CPU** |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | **942** | 509 | 245 | **1.85×** |
| bge-small-en-v1.5 | **494** | 309 | 134 | **1.60×** |
| bge-base-en-v1.5 | **211** | 84.5 | 41.3 | **2.49×** |
| bge-large-en-v1.5 | **60.7** | 24.6 | 11.2 | **2.47×** |
| nomic-embed-text-v1.5 | **165** | 73.0 | 44.0 | **2.26×** |

ONNX Runtime is measured because `prior-art.md` prescribes it as *the* CPU
baseline. It is roughly **half** of torch on every model — attention never
fuses in the exported graph — so torch is the harder opponent and the ratio is
taken against it. That prescription is refuted here, and the refutation is kept
rather than the prescription quietly dropped.

### The result that actually matters

Compare the two tables. The NPU column is **the same in both**, to within 1.5%
on every model, even though the second was measured with torch and ONNX Runtime
saturating all twelve cores in between:

| model | isolated | with the CPU busy | |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 951 | 942 | −0.9% |
| bge-small-en-v1.5 | 494 | 494 | −0.1% |
| bge-base-en-v1.5 | 211 | 211 | −0.2% |
| bge-large-en-v1.5 | 60.8 | 60.7 | −0.2% |
| nomic-embed-text-v1.5 | 166 | 165 | −0.4% |

That is the point of the project stated as a measurement. The throughput ratio
is the number people ask for; **this** is the one that says the work moved off
the cores.

### Energy

Differential method — every configuration measured at two encode counts and
subtracted, so startup, model load and weight staging cancel exactly. Package
RAPL, which covers the CPU cores *and* the NPU block, so both sides are read by
the same instrument and its systematic errors cancel in the comparison.

| model | CPU J/1000 seq | NPU J/1000 seq | |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 112 | **30.7** | **3.65×** better |
| bge-small-en-v1.5 | 196 | **65.3** | **3.00×** |
| bge-base-en-v1.5 | 688 | **206** | **3.33×** |
| bge-large-en-v1.5 | 1932 | **589** | **3.28×** |
| nomic-embed-text-v1.5 | 786 | **210** | **3.75×** |

**These are not comparable to the 1.94× on record from
[`0034`](../tasks/0034-m8-energy/TASK.md).** That was measured at two encode
lanes; the production default is now four, and MiniLM's NPU side went 44.0 →
30.7 J/1000 largely because of it. Same method, different configuration.

## How much these numbers are worth

This is the part that matters more than the table.

**NPU throughput is reproducible.** Across three runs per model the spread is
small, and every model's accuracy figure reproduces to the recorded digit —
the same number, not merely one inside the tolerance.

**The CPU ratio is less reproducible than it looks.** Two independent effects:

1. *Within a session*, a contended CPU wrecks the CPU side while leaving the
   NPU side intact. Guarded against now, but it is worth knowing that the
   failure mode inflates our number rather than depressing it.
2. *Between sessions*, the CPU absolute moves by more than the difference
   anyone is arguing about. bge-base's torch side measured **111.2 seq/s** in
   one session and **84.5** in another, both with tight spreads (±2%), on the
   same machine, same torch and transformers versions. That is a 24% gap with
   no noise to explain it — most likely thermal state, since the second
   measurement came after hours of continuous benchmarking, but that is a
   hypothesis and not a finding.

So: treat a ratio here as **indicative to about ±20%**, and treat the NPU
column as solid. If you want a number you can lean on, it is the NPU
throughput and the accuracy — not the ratio.

**And a limit of the sweep itself.** Models are measured back to back, so the
later ones run on a machine that has been hot for a while. Interleaving cancels
drift that hits both sides *within* a model; it does nothing about the machine
state *between* models. A ratio from one sweep is comparable within a model and
only loosely across them. Randomising the order or inserting cooldowns would
fix it and would change what "one session" means — it has not been done.

## Why width helps and depth hurts

Our cost is **per dispatch**; the CPU has no such term. So a model with twice
the layers costs us twice the dispatches and costs the CPU only twice the
arithmetic.

Against that, a BERT layer has `4h` elementwise elements and `12h²` MACs — a
ratio of **1 : 3h** — so the elementwise share, which we run on the host, falls
as `1/h`. Wider models put proportionally more work where the array wins.
[`0027`](../tasks/0027-m7-width-hypothesis/TASK.md) predicted this from a
synthetic sweep before real models confirmed it.

nomic-embed-text-v1.5 is the clean illustration: same width and depth as
bge-base, but its gated SwiGLU feed-forward puts *even more* of the work into
GEMMs rather than glue.

## Accuracy

Throughput is uninteresting if the vectors are different vectors. Every model
is gated three ways before it ships:

- against a from-scratch **numpy oracle** at every layer boundary, which is
  itself gated against two independent HuggingFace implementations before
  either is trusted;
- **end to end**, text in and vectors out, against `sentence-transformers`,
  with a top-k nearest-neighbour overlap check as well as `1 − cos`;
- on **MTEB**, five tasks, both sides run on the same checkpoint at the same
  sequence length in one session — because `1 − cos` measures fidelity to a
  reference and MTEB measures whether the embeddings are still *good*.

The `1 − cos` gate is 2e-03 and every shipping model sits two orders of
magnitude inside it.

### End to end — text in, vectors out

Run against the binary that ships, not an earlier one. Worst `1 − cos` over a
mixed corpus (ASCII, Norwegian, CJK, emoji, empty, whitespace-only, a
>100-character word), plus a top-10 nearest-neighbour overlap check, because a
similarity that is close on average can still reorder results.

| model | worst `1 − cos` | top-10 overlap | |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 2.644e-05 | 1.0000 | PASS |
| bge-small-en-v1.5 | 3.022e-05 | 1.0000 | PASS |
| bge-base-en-v1.5 | 2.613e-05 | 1.0000 | PASS |
| bge-large-en-v1.5 | 3.801e-04 | 1.0000 | PASS |
| nomic-embed-text-v1.5 | 2.401e-05 | 1.0000 | PASS |

nomic's run exercises the real `--prefix` flag: the runtime applies
`search_document: ` itself, and only the reference side is prefixed by the
harness.

### MTEB

Five tasks — STSBenchmark, SICK-R, STS12, Banking77Classification,
TwentyNewsgroupsClustering — with **both sides run in the same session**, same
checkpoint, same 64-token truncation. Absolute scores are therefore below the
published seq-256 numbers by construction; **the claim is the delta**, and a
gate on a delta cannot be gamed by picking a favourable task set.

Gate: `|mean| ≤ 0.5` points **and** no single task worse than −0.5.

| model | mean Δ | worst task | | |
|---|---:|---:|---|---|
| all-MiniLM-L6-v2 | **+0.03** | −0.01 | PASS | (0035 recorded +0.04) |
| bge-small-en-v1.5 | **−0.03** | −0.14 | PASS | **first ever run** |
| bge-base-en-v1.5 | **+0.02** | −0.00 | PASS | (0053 recorded +0.023) |
| bge-large-en-v1.5 | **+0.05** | −0.00 | PASS | **first ever run** |
| nomic-embed-text-v1.5 | **+0.09** | −0.03 | PASS | **first ever run** |

MiniLM and bge-base reproduce their recorded deltas closely. Three of these five
had **never had an MTEB run at all** before this sweep — which is why the harness
needed fixing first: it could not load nomic, and it filed every non-MiniLM
result under MiniLM's name.

`embeddinggemma-300m` is **not measured, deliberately.** At ~7.9 s/sentence on
the host path a full run is impractical, and stating that is better than leaving
a blank cell that reads as an oversight.

What the deltas mean: bf16 multiplies with fp32 accumulation, and the pre-tiled
weight layout, cost **nothing measurable downstream**. That is the question
`1 − cos` cannot answer — it measures fidelity to a reference, and a systematic
distortion could be small in cosine and still hurt retrieval.

**A gate that failed to catch something, kept here on purpose.** The on-device
golden check tiles one batch-4 corpus to fill a batch-128 design, so every
"different" row is identical content — which means a bug that reads the *wrong
row* is invisible to it. It passed a real threaded data race in
[`0070`](../tasks/0070-m13-nomic-runtime/TASK.md); a 13-distinct-sentence
end-to-end run caught it at `1 − cos` 0.44. Filed as
[T32](../research/OPEN-THREADS.md) with three cheap fixes, none done yet.

## What this does not do

- **Long inputs.** Sequence length is 64 tokens, fixed when the designs are
  compiled. Longer means re-exporting.
- **Training, or anything but encoder inference.**
- **Other NPUs.** XDNA2 / Strix Point only, and developed on one machine.
- **Attention on the array**, for the models where `head_dim` is 32 — it does
  not tile. It is ~5% of the FLOPs, so it has never been the bottleneck.
- **Beat a GPU.** Not the comparison; the point is the cores you get back.
