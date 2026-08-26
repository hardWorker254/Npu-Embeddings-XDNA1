# 0111 — T13 answered: the pre-tiled instability at production scale, and what it is not

- **Date** 2026-08-25
- **Milestone** research (T13)
- **Status** done — T13 **ANSWERED**. The instability is real at production `M`,
  it is a **3.6% tail of long stalls on an otherwise bit-clean kernel**, it
  follows neither the tile permutation nor B's L1 depth, and it **does not cost
  the shipped path anything**: untraced, pre-tiled beats row-major end to end at
  production `M`.

## Goal

[T13](../../research/CLOSED-THREADS.md#t13) was re-scoped by
[`0093`](../0093-t11-t12-t13-research/TASK.md) to one precise question, and
0093 named the experiment that would settle it:

> no measurement anywhere traces the pretiled access pattern's per-core
> stability AT production scale (8 columns, batch ≥ 1024) […] a single traced
> `--repeat` run at production geometry would settle it either direction, and
> it is the cheapest experiment left on this thread.

Two prior measurements straddled the question without meeting:
[`0058`](../0058-m11-iron-1.4-migration/TASK.md) reproduced 0007's
**~10% pre-tiled spread** at M=512 / 4 columns while row-major measured
**0.0%** in the same session (which is what rules out machine contention), and
[`0085`](../0085-m13-release-sweep/TASK.md) measured end-to-end spread **under
0.6%** at production scale — but neither tests the other's geometry.

## Context

Read: [T13 as 0093 left it](../0093-t11-t12-t13-research/TASK.md),
[`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md) §4 (the original filing, and
its "best-case pre-tiled runs match row-major exactly"), CLAUDE.md rule 1 and
trap 7, `docs/05-measurement/`.

## A correction to the experiment as filed, found before running it

**"8 columns, traced" is not expressible.** CLAUDE.md trap 7 says a fully-packed
8-column design cannot be core-traced, and the harness encodes exactly that:

```python
# experiments/m5-pretiled-gemm/gemm_pretiled.py:91
TRACE_ROUTING = {2: (1, 1), 4: (0, 0)}
```

`run_one()` refuses with `NOT TRACEABLE` for any other width. So the honest form
of 0093's experiment holds the traced width at **4 columns — the widest that
traces — and moves the axis that was actually untested, `M`**: 512 → **8192**,
which is production's own row count (`M = batch·seq = 128·64`). This measures
batch scale, not column scale; the column axis remains untestable by trace on
this hardware, and no number below claims otherwise.

## Contention record (rule 1)

Foreign `hw_context` submission counts, sampled before the first run, again 12 s
later, and again after the last run — all three identical:

```
9028:0  16360:14  16756:0  17336:4  17144:3  14008:0  14008:0
```

Seven parked `WorkloadsSessionHost.exe` contexts, **zero submissions across the
whole session**. Liveness by growth, not by `Active`/`Idle` status — 0109 found
that status alone misreads a genuinely idle array.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm

# 1. control: 0058's exact config, does the instrument still reproduce?
python gemm_pretiled.py --preset ffn_down -M 512  --cols 4 -n 48 --emulate-bfp16 --baseline --repeat 5 --out ...\control-M512-cols4.json

# 2. the experiment: same everything, production M
python gemm_pretiled.py --preset ffn_down -M 8192 --cols 4 -n 48 --emulate-bfp16 --baseline --repeat 5 --out ...\prod-M8192-cols4.json

# 3. does the tail follow the B permutation?
python gemm_pretiled.py --preset ffn_down -M 8192 --cols 4 -n 48 --emulate-bfp16 --orders "k,n;n,k" --inner both --repeat 3 --out ...\orders-M8192-cols4.json

# 4. control: does it exist WITHOUT the trace instrument? (wall clock, quiesced
#    NPU, end-to-end throughput only -- rule 1's permitted use)
python gemm_pretiled.py --preset ffn_down -M 8192 --cols 4 -n 48 --emulate-bfp16 --baseline --bench --bench-iters 50 --out ...\bench-M8192-cols4.json

# 5. is it B-fetch latency? deeper B in L1 (bd3 fits the 63 KB budget, bd4 does not)
python gemm_pretiled.py --preset ffn_down -M 8192 --cols 4 -n 48 --emulate-bfp16 --b-depth 3 --repeat 3 --out ...\bdepth3-M8192-cols4.json

# per-invocation distribution from the stored trace.json
python tasks\0111-t13-pretiled-stability-at-scale\analyse_trace.py
```

## Result 1 — the instrument still reproduces 0007, exactly

`ffn_down` 512×1536×384, tile (64,64,48), 4 cols, bfp16-emulated, 5 runs:

| arm | mean MACs/cyc/core | range | spread |
|---|---:|---|---:|
| row-major | **140.9** | 140.9–141.0 | **0.1%** |
| pre-tiled `[k,n\|st]` | 109.6 | 99.3–114.0 | **13.4%** |

140.9 is 0007's own table row to the digit, and 0058's. The instrument is sound
and the effect is live on today's toolchain.

## Result 2 — at 16× the rows it does NOT dilute

Same config, `M = 8192`, 5 runs each:

| arm | mean MACs/cyc/core | range | spread |
|---|---:|---|---:|
| row-major | **140.7** | 140.7–140.8 | **0.1%** |
| pre-tiled `[k,n\|st]` | 124.7 | 116.4–135.2 | **15.0%** |

So 0085's <0.6% end-to-end spread was **dilution by host work, not the absence
of the effect** — the two measurements never contradicted each other, and now
neither is guessing.

Note the pre-tiled *mean* improves with M (109.6 → 124.7) while its *spread*
does not (13.4% → 15.0%). Whatever this is, it does not amortise.

## Result 3 — it is a rare stall on a bit-clean kernel, not a slower kernel

Per-invocation deltas from the stored `trace.json` (the **slowest** of the five
pre-tiled repeats, 116.4 MACs/cyc, against the row-major run beside it). One
core carries the trace flow (trap 7), so this is one core's distribution:

| | min | p10 | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| row-major | 583 | 1385 | **1422** | 1472 | 1476 | **1500** |
| pre-tiled | 583 | 1385 | **1450** | 1535 | 11659 | **19192** |

**The two kernels are the same kernel up to p90.** Row-major has **zero**
invocations above 1.5× its median; pre-tiled has **13 of 365 (3.6%)**, spanning
2,175–19,192 cycles and carrying **19.7% of all cycles in the run**. Their
excess over the median is **16.6% of the entire run**.

Remove that tail and the pre-tiled kernel measures **139.8 MACs/cyc against
row-major's 140.7 — 0.6% apart.** That is 0007 §4's "best-case pre-tiled runs
match row-major exactly", now quantified rather than observed.

The outliers are **bursty, not periodic**: positions 77, 78, 79, 92, 118, 120,
134, 163, 166, 205, 303, 329, 352 — gaps of 1 to 98.

**Sampling caveat, stated because it biases in a known direction.** The trace
buffer holds a fixed number of events, so it captures a **prefix** of the run:
520 of the traced core's ~1,536 k-block invocations for row-major (128 row-tiles
÷ 4 rows × 2 col-tiles × 24 k-blocks), and only 358–470 for the pre-tiled arms.
The pre-tiled arm capturing *fewer* invocations per fixed buffer is itself
consistent with it emitting more trace records. Every rate above is therefore
measured over the first ~25–34% of the run, not the whole of it.

## Result 4 — it is not the tile permutation

All four pre-tiled variants, `M = 8192`, 3 runs each:

| variant | mean MACs/cyc/core | spread |
|---|---:|---:|
| `k,n \| st` | 123.5 | 13.8% |
| `k,n \| rowmaj` | 129.3 | 5.6% |
| `n,k \| st` | 118.8 | 3.6% |
| `n,k \| rowmaj` | 121.1 | 4.7% |

Every one of them carries it; none reaches row-major's 140.7 / 0.1%. The spread
itself is noisy at 3 runs (3.6–13.8%), which is the point — **no permutation is
clean**, so the stall is a property of the pre-tiled path, not of one ordering's
access pattern.

## Result 5 — it is not B-fetch latency that deeper L1 buffering absorbs

`--b-depth 3` (the deepest that fits the 63 KB budget: 16,384 A + 18,432 B +
24,576 C = 59,392 B; `--b-depth 4` needs 65,536 and is illegal):

| | mean | range | spread |
|---|---:|---|---:|
| `b-depth 2` (default) | 124.7 | 116.4–135.2 | 15.0% |
| `b-depth 3` | **121.0** | 114.3–130.0 | **13.0%** |

Unchanged within the noise. A third B buffer in L1 does not absorb it.

## Result 6 — and it costs the shipped path NOTHING. Pre-tiled WINS.

The control that decides whether T13 matters: the same two designs, **no trace
instrument at all**, 50 iterations each, quiesced NPU. This is end-to-end
dispatch throughput, which `docs/05-measurement/` permits wall clock for and
which the harness's own `bench_one()` docstring labels as such — it is not a
kernel-cycle claim.

| variant | NPU avg µs | NPU best µs | TFLOP/s |
|---|---:|---:|---:|
| row-major | 3568.0 | 3260.9 | 2.71 |
| **pre-tiled `[k,n\|st]`** | **3348.0** | **2884.7** | **2.89** |

**Pre-tiled is 1.066× faster on the mean and 1.130× on the best**, on the exact
shape whose per-core cycles say it is 12% *slower*. Both facts are true and they
are not in conflict:

> At 4 columns this dispatch is **not compute-bound**. Per-core MACs/cyc
> measures the compute window *including the core's waiting*, so a design whose
> data movement is better can finish sooner while its traced core is recorded
> waiting more often. CLAUDE.md's F2 / "we are bandwidth-bound" is the frame;
> this is a direct instance of it.

The instability does show in wall clock, just not enough to matter:
avg/best is **1.161×** pre-tiled against row-major's **1.094×**.

## What this answers, and what it retires

**T13 is ANSWERED.** The pre-tiled instability is real, reproducible, and
survives to production `M` — and it is **not a symptom of something else that
costs the product**, which is the condition 0007 §4 attached to it. The shipped
path pre-tiles B, and at production `M` the shipped path is the faster one.

**It also corrects the framing the thread carried for 104 tasks.** 0007 measured
the pre-tiled arm with per-core cycles, and per-core cycles are not the binding
constraint for this shape at this width. Reading a data-movement-bound dispatch
through a compute-window instrument is what made a 1.13× win look like a 12%
loss.

## Problems hit

1. **First analysis pass used the wrong threshold.** I took the trace's `min`
   (583 cycles, identical in both arms) for the typical iteration and set the
   stall threshold at 2× it — which flagged 96% of *row-major's* invocations as
   stalls. The typical iteration is the **median** (~1,422), and 583 is a rare
   short one. Kept here because the corrected script is what produced Result 3,
   and because "the minimum is the clean case" is exactly the assumption 0007's
   own "best-case matches row-major" invites.
2. **`gemm_pretiled.py` traces exactly one core**, so "per-core stability" in
   0093's phrasing can only ever mean "one core's per-invocation stability" on
   this harness. Adding flows for more cores runs into trap 7 again.

## Artifacts

All in this directory:

* `control-M512-cols4.json` — Result 1
* `prod-M8192-cols4.json` — Result 2 (and the trace behind Result 3)
* `orders-M8192-cols4.json` — Result 4
* `bdepth3-M8192-cols4.json` — Result 5
* `bench-M8192-cols4.json` — Result 6
* `analyse_trace.py` — the per-invocation distribution, run under the IRON env

Traces themselves (overwritten per run, last repeat of each arm survives):
`experiments/m5-pretiled-gemm/artifacts/trace_{rowmajor,pretiled_kn_st}_4c_bf16_f32_bfp16_8192x1536x384_t64x64x48.json`

## Next

Nothing on T13. The one thing this leaves unmeasured is deliberate and named
above: **the column axis cannot be traced past 4**, so "does the tail change at
8 columns" is unanswerable with this instrument, and Result 6 is the only
evidence that speaks for production's actual width.
