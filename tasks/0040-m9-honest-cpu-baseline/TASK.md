# 0040 — M9 Phase C: an adversarial CPU baseline, and what it found

**Goal.** We published *"faster than the CPU"* against a comparator we chose.
`research/prior-art.md` prescribes **ONNX Runtime CPU** as *the* primary
baseline, and we had never measured it. Worse, the CPU was measured
**best-of-5** while our NPU figures are **means** — an asymmetry that flatters
the CPU by whatever its spread happens to be. Better that we find this than a
reader does.

Plan: `C:\Users\vegar\.claude\plans\ja-la-oss-kj-re-shimmering-dewdrop.md`.

---

## Headline

**ONNX Runtime is not the stronger opponent — it is half the speed of torch**,
and the premise of this phase is refuted on this model and this machine.
`sentence-transformers`/torch remains the CPU baseline to beat, so the
comparator we had already chosen was the *strong* one, not the convenient one.

Interleaved, same statistic on every side, steady state, batch 128, seq 64:

| | MiniLM-L6 | bge-small |
|---|---:|---:|
| **NPU** | **877.0** | **444.6** |
| torch (sentence-transformers) | 489.4 | 290.0 |
| ONNX Runtime CPU | 234.3 | 134.0 |
| **NPU / strongest CPU** | **1.792×** | **1.533×** |

**And the CPU baseline is not reproducible while the NPU one is.** In a single
12-round interleaved run torch ranged 465.2–596.5 seq/s (**1.28×**) and across
runs this session it measured 710, 662.9, 518.5, 580.3 and 489.4. The NPU
ranged 826.5–898.2 (**1.09×**) and sat at 865–885 in every run. **Any ratio
between numbers taken minutes apart is measuring the machine's mood.**

---

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\bench_cpu_ort.py
.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\bench_cpu_baseline.py
.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\compare_three.py `
    --model all-MiniLM-L6-v2 --rounds 12
.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\compare_three.py `
    --model bge-small-en-v1.5 --rounds 8
```

Machine state (protocol item 2): **mains**, `NoSystemBattery`, **Balanced**
power plan, background load ≈ 4.5 of 72 core-seconds (~6%: Docker, CLion,
OneDrive). torch 12 threads, ORT `intra_op_num_threads` 12, NPU 24 host threads
/ 2 lanes, `artifacts_b128il`.

---

## ONNX Runtime, and the effort made to make it fast

ORT verifies correct first — **max abs 1.714e-07, worst 1-cos 2.312e-08**
against sentence-transformers. *A faster baseline that computes something else
is not a baseline*, so `bench_cpu_ort.py` refuses to print a timing until that
passes.

Then three tuning attempts, so the conclusion is not "we configured it badly":

| attempt | batch-128 mean |
|---|---:|
| ORT default threads | 301.2 |
| `intra_op` 4 / 8 / **12** / 16 / 24 | 234.6 / 281.0 / **309.1** / 285.0 / 201.5 |
| + `onnxruntime.transformers` BERT fusion | 254.7 |
| + export with `attn_implementation="eager"` | 291.1 |

**Attention never fused.** The optimiser reports `BiasGelu: 6`,
`LayerNormalization: 1`, `SkipLayerNormalization: 12` and no `Attention`, with
either attention implementation. That fusion is where the win would have been.
torch reaches ~490 with SDPA and oneDNN doing the same job inside its own
kernels.

**Conclusion: for a 384-wide BERT at seq 64 on this machine, ORT CPU is
~0.48× of torch.** `prior-art.md`'s prescription does not hold here. It is
recorded as a baseline anyway — it is what an ONNX-based deployment would
actually get.

---

## Two export traps, both of which fail loudly only by luck

**`torch.onnx.export` silently ignored `dynamic_axes`.** torch 2.10 defaults to
the **dynamo** exporter, which takes `dynamic_shapes` and disregards
`dynamic_axes`. Batch was baked in as 1, ORT's optimiser folded a `Reshape` to
`{seq, hidden}` around it, and inference failed at **run time** with
`Input shape:{4,64,384}, requested shape:{64,384}`. Fixed with `dynamo=False`,
**plus a check that the batch axis is symbolic in the written graph** — the
failure was loud this time, but only because ORT happened to insert that
reshape; a graph that merely produced wrong results for batch > 1 is the same
bug with no exception.

**Positional arguments no longer line up with `BertModel.forward`.** Passing
`(ids, mask, tt)` positionally now lands one of them on `use_cache`
(`TypeError: got multiple values for argument 'use_cache'`). Exported through
an explicit wrapper with named arguments, which also pins the graph's input
order.

---

## The statistic asymmetry, fixed

`bench_cpu_baseline.py` took **best-of-5**; every NPU figure we publish is a
**mean**. Both benches now report best, mean and median with the spread, and
`seconds`/`seq_per_s` keep their old meaning so previously published numbers
stay traceable.

The effect is smaller than feared and **not** what explains the change in the
published number:

| batch | best | mean | spread |
|---:|---:|---:|---:|
| 4 | 255.8 | 227.7 | 30.8% |
| 32 | 475.9 | 462.0 | 8.6% |
| 128 | **536.3** | **518.5** | 8.0% |

Best-of-5 buys the CPU ~3.4% at batch 128. The published **710** is 1.37×
above today's best-of-9 of 536.3, so **the statistic is not the explanation** —
the machine is. Background load is ~6%, which does not cover it either. The
honest position is that **today's absolute CPU number is not the one published
in `0033`, the gap is not fully explained, and the defensible quantity is the
interleaved ratio.**

---

## The failure worth keeping: I reported "ON BATTERY" for a machine with no battery

The protocol's item 2 requires recording mains vs battery, because
[Rösti](https://arxiv.org/abs/2504.03083) measured **145 → 255 GFLOP/s on mains against
95 → 111 on battery**. A wrong answer there invalidates every ratio under it.

My hand-rolled check was `Win32_Battery | BatteryStatus -eq 2`. On a machine
with **no battery** `Win32_Battery` returns **nothing**, the comparison never
ran, and the `else` branch printed `ON BATTERY (status )` — with an empty
status, which is the only reason it looked wrong. I nearly built the whole
Phase C conclusion on "the CPU is slow because we are on battery".

`[System.Windows.Forms.SystemInformation]::PowerStatus` reports
`PowerLineStatus: Online`, `BatteryChargeStatus: NoSystemBattery`. That check
is now **inside `compare_three.py`** and written into every result file, so it
cannot be re-derived by hand and re-derived wrong.

**Rule this adds:** an absent data source is not a negative reading. A check
whose "no" branch also fires when it could not measure is a fail-open, and this
is the ninth in this project.

---

## Why interleaving is now mandatory, and steady state with it

The protocol says same input, same sequence length, same batch. It did not say
**interleaved**, and it must: the three encoders now run round-robin in one
process so that whatever the machine is doing, it is doing it to all three.

It must also say **steady state**. torch ramps — 469 → 686 seq/s over five
rounds in one run — while the NPU does not. `compare_three.py` reports the mean
of the second half as well as of everything, and the ratio uses the second
half, which is the choice that **favours the CPU**.

---

## bge-small degrades with depth more than the CPU does

| | MiniLM (6 layers) | bge-small (12) | ratio |
|---|---:|---:|---:|
| NPU | 877.0 | 444.6 | **0.507×** |
| torch | 489.4 | 290.0 | 0.593× |

Twice the layers costs us **almost exactly half**, because our cost is per
dispatch and 12 layers is 96 dispatches instead of 48. The CPU loses only 41%.
That is finding **F1** showing up in a second model: **the fusion lever is
worth more to us than to the CPU**, and the gap between 0.507 and 0.593 is the
size of the prize.

---

## Carry forward

- `docs/CURRENT_STATUS.md` and `README.md` quote `1.17× of 710`. That comparison
  was best-of-5 CPU against mean NPU and is superseded by the interleaved
  1.792×. Both need updating with the protocol change, not just the number.
- The protocol in `docs/05-measurement/` needs **interleaved** and **steady
  state** written into it.
- Phase D (bge-large) should use `compare_three.py` from the start.
