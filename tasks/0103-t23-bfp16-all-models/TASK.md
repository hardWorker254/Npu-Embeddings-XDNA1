# 0103 — bfp16 across the whole catalogue: five of six pass, and the one failure misses by 0.001

- **Date** 2026-08-23
- **Milestone** M13 (T23)
- **Status** done — all six models have a real MTEB gate verdict on the
  adoption candidate (bfp16 + bf16-C). Secondary cells (fp32-C on four of the
  six) were deliberately not run; see "What was not done".

## Goal

[`0101`](../0101-t23-bfp16-gates-on-1.4.2/TASK.md) re-measured T23 on mlir-aie
1.4.2 for MiniLM and bge-base after [T26](../../research/CLOSED-THREADS.md#t26)
showed the thread's whole accuracy case was 1.3.4-era. The user then set the
adoption rule and the scope:

> *"Vi kjører alle. Men i utgangspunktet kjører vi når mteb er pass."*

So: **measure the remaining four models, and adopt bfp16 per model wherever
MTEB passes.** `1-cos` does not gate adoption; it stays as a fidelity check.

## Context

`--emulate-bfp16` was retired in [`0035`](../0035-m8-mteb-gate/TASK.md) at
`1-cos` 3.470e-03 — on 1.3.4, where the emulated matmul emitted no `crrnd`
rounding control, so its bfp16 quantisation of A and B ran under the default
`floor` and was systematically biased low. 1.4.2 emits the control. 0101 found
the retirement is overturned on `1-cos` and that the historical 6.6× fp32-C /
bf16-C gap is gone.

## What was done

Every MTEB run used `--sides cpu,npu` **in one invocation**, and every log was
checked for a real `PASS`/`FAIL` block rather than the `NO GATE` paragraph.
That check is not a formality: 0101's first pass reported four verdicts the
gate never issued, because the deltas had been hand-computed against a CPU
baseline from a separate invocation.

Order was chosen so a partial result would still answer the question:
bge-large first (widest model, the one the decision leans on), then nomic
(a different architecture), then bge-small, then EmbeddingGemma. Per model the
**bf16-C** config ran first, since that is the adoption candidate.

The plain-bf16 MTEB control per model was **dropped after bge-large**, on
purpose: the gate compares CPU against NPU inside one invocation, so every
bfp16 run already carries its own same-session CPU baseline, and 0101 had
already validated the harness on two models. The cheap `1-cos` control was
kept. (The bge-large control run died mid-`TwentyNewsgroupsClustering` before
this decision was made — see Problems.)

## Commands

```powershell
. C:\dev\mlir-aie\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# one per model; --sides cpu,npu in ONE invocation is what makes the gate issue a verdict
.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model bge-large-en-v1.5 --artifacts artifacts_large_bfp_cbf16 `
    --sides cpu,npu --out tasks\0103-t23-bfp16-all-models\mteb_large_bfp_cbf16.json

.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp_cbf16 `
    --sides cpu,npu --out tasks\0103-t23-bfp16-all-models\mteb_nomic_bfp_cbf16.json

# bge-small shares MiniLM's geometry (hidden 384), so it is served by MiniLM's design
.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model bge-small-en-v1.5 --artifacts artifacts_minilm_bfp_cbf16 `
    --sides cpu,npu --out tasks\0103-t23-bfp16-all-models\mteb_small_bfp_cbf16.json
#   ... and again to _repeat.json, to test the knife-edge failure

.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model embeddinggemma-300m --artifacts artifacts_gemma_bfp_cbf16 `
    --sides cpu,npu --out tasks\0103-t23-bfp16-all-models\mteb_gemma_bfp_cbf16.json
```

## Result — bfp16 + bf16-C, the adoption candidate, real gate verdicts

| model | arch | MTEB mean | MTEB worst | **gate** | adopt? |
|---|---|---:|---:|---|---|
| `all-MiniLM-L6-v2` † | 0 | +0.12 | −0.07 | **PASS** | **yes** |
| `bge-small-en-v1.5` | 0 | −0.10 | **−0.5010** | **FAIL** | **no** |
| `bge-base-en-v1.5` † | 0 | −0.06 | −0.19 | **PASS** | **yes** |
| `bge-large-en-v1.5` | 0 | +0.13 | −0.01 | **PASS** | **yes** |
| `nomic-embed-text-v1.5` | 2 | +0.01 | −0.25 | **PASS** | **yes** |
| `embeddinggemma-300m` | 1 | +0.16 | −0.02 | **PASS** | **yes** |

† from [`0101`](../0101-t23-bfp16-gates-on-1.4.2/TASK.md), same method.

**Five of six. Three architectures — 0, 1 and 2 — all pass**, so this is not a
BERT-only result.

### bge-small fails by 0.001 points, and it is not noise

```
TwentyNewsgroupsClustering          48.63    48.12    -0.50
worst single task: -0.50 points
FAIL -- gate is |mean| <= 0.5 points AND no task worse than -0.5
```

Exact: **−0.5010**, against a −0.5 line. Re-run end to end, it reproduces
**bit-identically** — `cpu 0.4862581899`, `npu 0.4812477913` on both runs. Both
sides are deterministic, so this is a stable fact about the model and the
datapath, not a marginal reading. The other four tasks are −0.04, −0.04, −0.00
and +0.07.

**An initial reading of this as "a coin flip against a bright line" was wrong
and is corrected here.** The wide swings on this task (+0.68 to −0.58) are
variance *between models*, not noise *within* a measurement.

### The gate is effectively a test on one task

Across all ten cells measured in 0101 and here, `TwentyNewsgroupsClustering` is
the worst task **every single time**, while the other four stay inside ±0.12:

| model | clustering Δ | other four, range |
|---|---:|---|
| MiniLM | +0.68 | −0.07 … +0.05 |
| bge-small | **−0.50** | −0.04 … +0.07 |
| bge-base (bf16-C) | −0.19 | −0.06 … +0.02 |
| bge-base (fp32-C) | **−0.58** | −0.06 … +0.01 |
| bge-large | +0.62 | −0.01 … +0.02 |
| nomic | −0.25 | +0.04 … +0.12 |
| gemma | +0.89 | −0.02 … −0.02 |

So the `-0.5` bright line is, in practice, a test on clustering. That is
defensible — clustering amplifies small distance changes into hard group
assignments, so it *should* be the sensitive one — but it means the gate
measures one task's sensitivity rather than an average quality loss, and which
models pass is decided there. Recorded as a property of the gate, not as a
caveat on these measurements, since each is exactly reproducible.

### bge-small is the model MTEB likes least, on both quantised datapaths

This is not a bfp16 quirk. On **int8**, `docs/CURRENT_STATUS.md` already
records that bge-small has the **best `1-cos` of any int8 row (6.385e-04) and
the worst MTEB (−0.09)**. Here it is the only bfp16 MTEB failure. Whatever
makes bge-small sensitive is a property of the model, and it shows up under two
unrelated quantisation schemes.

### `1-cos` (fidelity check, does not gate adoption)

| bge-large | `rel_fro` | worst `1-cos` |
|---|---:|---:|
| control (plain bf16) | 3.763e-03 | 8.432e-06 |
| bfp16 + fp32-C | 1.910e-02 | 2.273e-04 |
| bfp16 + bf16-C | 1.973e-02 | 2.626e-04 |

The control matches `docs/CURRENT_STATUS.md` exactly. Both emulated builds were
verified from the artifact rather than from the flag — **20 `vconv.bfp16ebs8.fp32`
ops and the 3-instruction `crrnd` triple** in each matmul object, so the
emulation is genuinely on (trap 1 would make it a silent no-op) and these are
demonstrably fixed-toolchain numbers.

**bfp16 does not degrade with width the way int8 does.** int8 goes 6.385e-04
(bge-small) → 2.968e-03 (bge-large), 4.6× worse and the catalogue's only int8
`1-cos` failure. bfp16 is flat: 3.368e-04 / 2.174e-04 / 2.273e-04 across
MiniLM, bge-base and bge-large. Plausible mechanism, not verified here: int8
quantises per tensor, so one outlier drags the whole scale and SmoothQuant has
more outlier range to move as matrices widen; bfp16 keeps a shared exponent per
8-element block, so an outlier only affects its own block.

### Speed — bge-large, labelled by provenance

Array GEMM time from `--bench`'s `wait (hardware)` line, which
[`0097`](../0097-t18-t21-t4-measurements/TASK.md) established is the number to
quote:

| | per dispatch | vs control |
|---|---:|---:|
| control (plain bf16) | 19,099 µs | — |
| bfp16 + fp32-C | 10,458 µs | **1.83×** |
| bfp16 + bf16-C | 9,776 µs | **1.95×** |

The array share of wall clock falls from 43.3% to 28.8%, which is the other way
of reading the same number.

## Problems hit

1. **The bge-large plain-bf16 MTEB control died mid-run**, in
   `TwentyNewsgroupsClustering` — no verdict, no JSON, process gone, at ~1.85 GB
   resident. Four `until grep -q "wrote tasks"` polling shells and a Monitor
   were left waiting on a file that would never be written. Cause not
   established; memory pressure on the heaviest cell in the sweep is the leading
   suspect (bge-large is 24 layers at hidden 1024, and this task alone took 750 s
   on nomic). **Not retried**, because the control was already judged
   unnecessary — every bfp16 run carries its own same-session CPU baseline.
   Recorded because a run that dies silently mid-sweep is exactly how a
   half-result gets reported as a whole one.
2. **The measuring agent stopped four times waiting on background work** that
   nothing was going to wake it for. The remaining cells were driven directly
   with foreground launches plus a bounded wait loop that also exits when the
   process disappears — which is what caught (1).
3. A `head` on a freshly-launched log reported "No such file", which looked like
   a failed launch and was a race against file creation. Verified by `ls` and a
   short foreground probe before concluding anything.

## What was not done

- **`fp32-C` on bge-small, bge-large, nomic and gemma.** Only the adoption
  candidate was measured on those four. bge-large's fp32-C has a `1-cos` and a
  speed figure here but no MTEB verdict. Under the user's rule this does not
  block a decision, but it means the fp32-C column of the catalogue is
  incomplete — and 0101 already showed fp32-C failing where bf16-C passes
  (bge-base), so it cannot be assumed equivalent.
- **No repeat runs except bge-small.** Each cell is one run. That is sound
  because both sides proved deterministic, but it means no within-cell variance
  figure exists.
- **Nothing was adopted and no default changed.** Per the 0045 precedent the
  datapath decision is the user's; this task supplies the evidence.

## Artifacts

`mteb_*.json` / `mteb_*.log` per cell (each carrying its own gate block),
`gate_large_*.txt`, `bench_large_*.txt`, `objdump_large_*` — all in this
directory.
