# 0137 — The gte quality gates: English MTEB (gating), multilingual STS (evidence), tail baseline, and a multilingual semantic corpus

**Date**: 2026-08-27
**Goal**: the adoption evidence for `gte-multilingual-base` on the bfp16
datapath, per the user's 2026-08-27 decision: (1) the SAME English MTEB
protocol that gated the six shipped models ([`0103`](../0103-t23-bfp16-all-models/TASK.md));
(2) a small multilingual STS set, run and reported as **evidence, not gate**
in 0.5.0 (it becomes a gate in 0.6.0 once this baseline exists); (3) the T51
tail-gate baseline ([`0132`](../0132-t51-tail-gate/TASK.md) instrument);
(4) a multilingual extension of the semantic corpus. The coordinator takes
the adoption decision from these numbers; this task changes no runtime code
and no catalogue.

## The non-negotiable premise: the fp32 baseline must be REPAIRED

[`0134`](../0134-gte-oracle/TASK.md) landmines 1+2 and
[`0136`](../0136-gte-runtime/TASK.md) problem 1: transformers v5 materialises
this checkpoint's remote-code `persistent=False` buffers (rotary `inv_freq`
AND `embeddings.position_ids`) as uninitialised meta-device memory, and
honours the checkpoint's `torch_dtype: float16`. Any sentence-transformers /
MTEB run of gte on this env without `reference/make_goldens_gte.py::repair_rotary`
plus an explicit fp32 load is **silently position-scrambled** (measured
9.0e-02 max-abs on normalized CLS in 0136) — a baseline that could fake a
pass or a fail. Every fp32 reference in this task carries both repairs, and
each harness asserts the loaded dtype rather than trusting the flag:

* `experiments/m8-npu-vs-cpu/run_mteb.py` — the CPU side now detects the gte
  checkpoint, loads `model_kwargs={"torch_dtype": float32}`, applies
  `repair_rotary(m[0].auto_model)`, refuses on a non-fp32 parameter dtype,
  and prints `[cpu] gte rotary+position_ids repaired, fp32 verified` (the
  line is in this task's stored logs).
* `tools/make_tail_reference.py` — new `ref_gte()` branch for
  `arch == "gte_new_rope_geglu"`: AutoModel + trust_remote_code + fp32 +
  `repair_rotary`, CLS pooling + L2 normalize, no prompt.

**Pre-flight, before any MTEB minute was spent** ([`probe_bridges.py`
transcript below]): on 0136's 8 stored texts,

* (a) the MTEB NPU bridge (`npu_encoder.py`'s `--encode-file` path, HF
  tokenizer + emb_sum) is **bit-identical** (max|d| 0.0) to the shipped
  CLI's own `--embed` output for gte — so the harness measures the same
  datapath 0136 validated, and the shipped `embed` subcommand's output is
  itself bit-identical to 0136's stored `gte_out.f32`;
* (b) the repaired CPU side reproduces 0136's known per-text 1-cos table to
  the fourth digit (2.558e-04 … 3.682e-04, worst 3.682e-04). An unrepaired
  baseline reads ~9e-02 — 250× the signal being measured — so this probe
  discriminates repaired from broken before anything gates on it.

## Deliverable 1 — English MTEB gate (the shipped protocol)

Same five tasks, same `--sides cpu,npu` in ONE invocation (0103: the gate
only issues a verdict when the delta is same-session), same seq 64 both
sides, same threshold: |mean| ≤ 0.5 points AND no task worse than −0.5.

```powershell
.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
    --sides cpu,npu --out tasks\0137-gte-gates\mteb_gte_bfp16.json
```

(`--artifacts` is explicit because gte is not in the catalogue yet — 0136
problem 5: a locally-packed model defaults to plain bf16 artifacts, and the
only fitting design set is the bfp16 one. The runtime's status line, scraped
into every artifact here, confirms `bfp16-emulated MMAC, C as bf16`.)

**Verdict: PASS.** Real gate block, issued same-session (both sides in one
invocation, log + JSON in this directory):

| task | CPU (fp32, repaired) | NPU (bfp16) | delta (points) |
|---|---:|---:|---:|
| STSBenchmark | 86.46 | 86.44 | -0.02 |
| SICK-R | 79.34 | 79.37 | +0.03 |
| STS12 | 77.51 | 77.50 | -0.01 |
| Banking77Classification | 80.94 | 80.89 | -0.06 |
| TwentyNewsgroupsClustering | 49.67 | 50.01 | **+0.34** |
| **MEAN** | | | **+0.06** |
| **worst single task** | | | **-0.06** |

`PASS -- gate is |mean| <= 0.5 points AND no task worse than -0.5` (the
harness's own verdict line). Placed against 0103's table this is the
cleanest verdict in the catalogue: the worst task (-0.06) is 8x inside the
line, and `TwentyNewsgroupsClustering` -- the task the gate is effectively
decided on, worst cell in all ten of 0101/0103's runs -- is **positive**
(+0.34). Scores are "gte at seq 64" on both sides by construction; only the
delta is the claim.

## Deliverable 2 — multilingual STS (EVIDENCE, NOT GATE; baseline for 0.6.0)

**EVIDENCE, NOT GATE — baseline for 0.6.0.** Two multilingual STS tasks from
the installed mteb 2.19.5, chosen for coverage plus modest runtime: STS17
(11 subsets, in-language ar/es/ko/en + cross-lingual en-ar/en-de/en-tr/
es-en/fr-en/it-en/nl-en, 5,346 pairs) and STS22.v2 (18 subsets, news text,
in-language + cross-lingual across ar/de/en/es/fr/it/pl/ru/tr/zh, 3,958
pairs). Same paired protocol as the gate — `--sides cpu,npu` in one
invocation, repaired fp32 CPU side, seq 64 both sides. The harness prints
its PASS line because it always gates; **for 0.5.0 this set is recorded as
evidence only**, per the user's decision — it becomes a gate in 0.6.0
against exactly this baseline.

```powershell
.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
    --sides cpu,npu --tasks STS17,STS22.v2 `
    --out tasks\0137-gte-gates\mteb_gte_multilingual.json
```

| task | CPU (fp32, repaired) | NPU (bfp16) | delta (points) |
|---|---:|---:|---:|
| STS17 (11-subset mean) | 82.59 | 82.59 | +0.00 |
| STS22.v2 (18-subset mean) | 60.33 | 60.31 | -0.02 |
| **MEAN** | | | **-0.01** |
| **worst task** | | | **-0.02** |

**Per-language, from the JSON's new `per_subset` block** (run_mteb.py was
extended this task to record per-subset scores — score_of() collapses a
multilingual task's subsets into one mean, the right shape for the M8 gate
and the wrong shape for a per-language baseline). All 29 subsets sit within
**[-0.30, +0.20] points**; the five worst: zh-en -0.30, es-en (STS17)
-0.18, pl-en -0.12, ru -0.12, ar -0.11. No language family degrades
preferentially — the largest deltas are cross-lingual pairs, the same
subsets with the lowest absolute scores, where rank metrics are noisiest.
Full 29-row table in `mteb_gte_multilingual.json`; absolute scores are
"gte at seq 64" and below published seq-256+ numbers by construction.

## Deliverable 3 — tail-gate baseline (T51 instrument)

Reference: `reference/tail/gte-multilingual-base.{f32,json}` from the
extended `make_tail_reference.py` — repaired fp32 AutoModel, CLS pooling,
L2 normalize, no prompt, the shared 224-text corpus (180 single words + 36
semantic-corpus sentences + 8 phrases, 0129's byte for byte — see the
corpus note under deliverable 4).

```powershell
.venv-ref\Scripts\python tools\make_tail_reference.py --models gte-multilingual-base
python tools\verify_tail.py --models gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
    --write-baseline --out tasks\0137-gte-gates\tail_gte_baseline.json
python tools\verify_tail.py --models gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
    --out tasks\0137-gte-gates\tail_gte_gate.json     # gate against the recorded ceiling: PASS
```

| model | datapath | median | p90 | p99 | max | max/med | ceiling |
|---|---|---:|---:|---:|---:|---:|---:|
| gte-multilingual-base | bfp16-emulated MMAC, C as bf16 | 3.419e-04 | 4.299e-04 | 5.160e-04 | 7.361e-04 | **2.2×** | 2.000e-03 (floor; 2×p99 = 1.03e-03 < 2e-03) |

Per group: words n=180 median 3.54e-04 / max 7.36e-04; sentences n=36 median
2.93e-04 / max 3.72e-04; phrases n=8 median 3.47e-04 / max 4.51e-04.

**No bge-large-style tail.** max/median 2.2× against bge-large's 100.2×, and
the worst single input (a single word, 7.36e-04) is still 2.7× under the
2e-03 floor. This is consistent with 0129's still-standing discriminator
list (depth 24 / width 1024 / learned representations — gte is 12 layers ×
768 wide, bge-base's geometry, and lands in bge-base's decade: bge-base
baselined at p99 3.711e-04 in 0132, gte at 5.160e-04). Measured, not
assumed, per the brief.

## Deliverable 4 — multilingual semantic-corpus extension (v6)

**Design decision: lang-tagged clusters, gated per model.** Three new
clusters — one Norwegian, one German, one Spanish (topics: knitting, chess,
flamenco dance) — each with the standard `.a`/`.b`/`.c` discipline and the
80–95-char budget, each sentence carrying a `"lang"` field. A new top-level
`multilingual_models` list (currently only `gte-multilingual-base`) tells
`tools/verify_semantics.py` which models score the non-English clusters;
everything else sees EXACTLY the pre-v6 36 sentences. `make_tail_reference.py`
filters to `lang == "en"` for the same reason: its corpus is 0129's byte for
byte, and the cross-model tail comparison stays a comparison of models.

**Why gate per model rather than gate everyone:** measured first, decided
second. A scratch corpus listing MiniLM and bge-base as multilingual shows
both PASS 30/30 with all 15 probes hitting (`semantic_probe_english_on_ml.json`,
stored here) — the `.a`/`.b` lexical pairs share almost every surface
token, so an English-only model passes on shared-subword luck. That makes
gating the six on these clusters meaningless in both directions: a pass
claims nothing (bag-of-subwords suffices) and a fail would not be a
regression (the model never claimed the language). The corpus header records
this reasoning; the topics were checked against 0121 §3's four failure
classes (no season/time tokens near the weather cluster, no city names the
flight cluster already uses, no recipe words near cooking, no first-person
errand frame in the `.c` tier, and — cross-lingually — no cognate of an
existing cluster's head noun, e.g. no Spanish "banco").

**Verification, corpus v6:**

| run | result |
|---|---|
| six shipped models, default corpus (`semantic_gate_six_v6.json`) | **all PASS**, gate 24/24 each, probes 12/12 ×5 + 10/12 (bge-small, its two known misses from 0121) — bit-for-bit the v5 shape, so the six are provably unaffected |
| gte, full 15 clusters (`semantic_gate_gte_v6.json`) | **PASS** — gate 30/30, probe 15/15, control fails-as-required, margin +0.087, datapath `bfp16-emulated MMAC, C as bf16` |

```powershell
python tools\verify_semantics.py --out tasks\0137-gte-gates\semantic_gate_six_v6.json
python tools\verify_semantics.py --models gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
    --out tasks\0137-gte-gates\semantic_gate_gte_v6.json
```

(`--out` always named — 0136 problem 2 is the precedent for what the default
path does to the canonical 0121 report. That file is untouched here; whether
the canonical sweep is re-run on v6 is the coordinator's call with the
adoption decision.)

## Problems hit

1. **The first English-gate attempt died silently in
   `TwentyNewsgroupsClustering`, CPU side** — process gone, no traceback,
   no JSON, log ends mid-cell (`mteb_gte_bfp16.attempt1.log`), last
   observed at ~2.4 GB resident. This is the exact failure shape 0103
   problem 1 recorded on the same task (bge-large control, ~1.85 GB), now
   seen a second time on a smaller model. The relaunch with nothing else
   running completed normally (clustering CPU cell 584 s), and its CPU
   scores reproduced attempt 1's four completed cells **bit-identically at
   print precision** (86.46/79.34/77.51/80.94), so attempt 2 is a clean
   run of record, not a divergent one. Cause still not established;
   memory pressure remains the leading suspect (attempt 1 shared the
   machine with this session's semantic-gate and probe processes).
2. **The multilingual run was stopped ~1 min in and relaunched** after
   realising the artifact would collapse each task's subsets to one mean —
   useless as the per-language 0.6.0 baseline. `run_mteb.py` now stores
   `per_subset` per task per side; the aborted attempt's log is
   `mteb_gte_multilingual.attempt1.log`. Nothing was measured and nothing
   reused from the aborted attempt.
3. **`verify_tail.py`'s default `--out` is tasks/0132's canonical report
   and `verify_semantics.py`'s is tasks/0121's** — both runs here named
   `--out` explicitly into this directory (0136 problem 2 is the precedent
   for what the default does: it clobbered the six-model report with a
   one-model one). The canonical files are untouched by this task.
4. **`probe_bridges.py` first ran against scratchpad-relative inputs** and
   failed on rerun from the task directory; the stored copy reads its
   inputs (`gte_in.txt`, `gte_cli_out.f32`) beside itself and its
   transcript is `probe_bridges.out`.

## Artifacts in this directory

`mteb_gte_bfp16.{json,log}` (the gate) and `mteb_gte_bfp16.attempt1.log`
(the silent death, kept per rule 3b), `mteb_gte_multilingual.{json,log}`
(the evidence set, `per_subset` blocks inside the JSON) and
`mteb_gte_multilingual.attempt1.log` (the aborted no-per-subset attempt),
`tail_gte_baseline.json` / `tail_gte_gate.json`,
`semantic_gate_six_v6.json` / `semantic_gate_gte_v6.json`,
`probe_bridges.py` + `probe_bridges.out` with its inputs `gte_in.txt`
(0136's texts) and `gte_cli_out.f32` (the shipped CLI's output, bit-equal
to 0136's stored `gte_out.f32`). The tail reference lives at
`reference/tail/gte-multilingual-base.{f32,json}` (the baseline ceiling is
inside the .json, written by `--write-baseline`; its recorded
`semantic_corpus_version` is 5 because it was generated before the v6 bump
-- the texts are identical either way, since `build_corpus()` now filters
to `lang == "en"`).

## What is NOT done here

The adoption decision, the catalogue entry, `hub.cpp`/`npue_pack.cpp`,
register/docs updates — the coordinator owns all of these. The multilingual
STS numbers are a recorded baseline; promoting them to a gate is 0.6.0
scope, per the user's decision.
