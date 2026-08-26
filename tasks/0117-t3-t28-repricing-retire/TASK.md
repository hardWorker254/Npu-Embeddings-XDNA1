# 0117 — T3/T28 retired: re-priced on the post-fusion build, the ceiling is now below what 0108 already banked

- **Date** 2026-08-26
- **Milestone** research (T3, T28)
- **Status** done — T3 and T28 **RETIRED**. Best case on the build that ships
  today is **1.095× (bge-large) to 1.160× (MiniLM)**, against the **1.223×–1.265×
  [`0108`](../0108-fuse-epilogue-bfp16/TASK.md) already measured and shipped**
  for a pure host rewrite. Even an absurd upper bound that credits the relay
  with the *entire* transport bucket lands at 1.24× on bge-large. The mechanism
  is proven and stays documented; the value has been collected by cheaper means.

## Goal

The register asked for exactly this and named the input:

> **Update 2026-08-25 ([`0108`](../0108-fuse-epilogue-bfp16/TASK.md))**: […]
> what it is parked *against* has changed […] **T3 should be re-priced against
> the post-fusion breakdown, not 0107's pre-fusion one.**

[`0107`](../0107-t3-t28-pricing/TASK.md) priced the relay at **19.7–25.9%**
addressable and **1.25×–1.35×** best case, and recommended porting the host
fusion first. 0108 did that and got 1.153–1.340×.
[`0109`](../0109-fused-ratio-energy/TASK.md) then measured that array time did
not move while its *share* rose on every model — which is why the register kept
T3/T28 open rather than closing them: the array was the larger piece again.

This task takes the measurement 0107 could not.

## Context

Read: [`0107`](../0107-t3-t28-pricing/TASK.md) (the pre-fusion pricing and its
stated assumptions), [`0108`](../0108-fuse-epilogue-bfp16/TASK.md) (what the
host fusion actually removed), [`0109`](../0109-fused-ratio-energy/TASK.md)
(array share after it), [`0092`](../0092-t28-relay-bf16-output/TASK.md) (the
relay proven correct at production tile width), [`0054`](../0054-m10-phase-fusion-pipeline/TASK.md)
(the mechanism, and five wrong designs on the way to it).

## Contention record

Both benches bracketed by `xrt-smi examine -r all` samples reading
`9028:0 16360:26 16756:0 17336:4 17144:5 14008:0 14008:0` before and after —
**zero foreign submissions**. The runtime's own `npu exclusive` line appears in
each run.

## Commands

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2  --artifacts artifacts_minilm_bfp16 --threads 24 --pipeline 1 --bench 5
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5 --artifacts artifacts_large_bfp16  --threads 24 --pipeline 1 --bench 5
```

Single lane on purpose: lanes overlap host and array work, and the question is
what the relay would *remove*, not what overlap hides.

## The post-fusion breakdown, measured

| | MiniLM | bge-large |
|---|---:|---:|
| wall | 111.35 ms | 1652.69 ms |
| **wait (hardware)** — the array | **35.64 ms (32.0%)** | **935.07 ms (56.6%)** |
| read out + bias | 22.08 ms (19.8%) | 242.75 ms (14.7%) |
| bf16 convert (both ways) | 2.27 ms (2.0%) | 24.98 ms (1.5%) |
| sync to + from device | 10.21 ms (9.2%) | 55.66 ms (3.4%) |
| host attention | 22.33 ms (20.1%) | 198.08 ms (12.0%) |
| host layernorm | 12.38 ms (11.1%) | 163.27 ms (9.9%) |
| host softmax | 3.08 ms (2.8%) | 18.82 ms (1.1%) |

bge-large's 56.6% array share reproduces 0109's 56.7% to within 0.1 points, in
a separate session, which is the check that these two runs are describing the
same machine state 0109 described.

## The re-pricing

The relay keeps the FFN intermediate on device: `ffn_up`'s C never reaches the
host, GELU runs on-core, and `ffn_down`'s A never goes back. So what it can
remove is the `ffn_up`→`ffn_down` leg's share of **read out + bias**, **bf16
convert** and **sync** — nothing else. Post-0108 the host GELU is no longer a
bucket of its own; it lives *inside* the fused read-out pass, which is precisely
why the target shrank.

**Split by N**, the same assumption 0107 stated and for the same reason — the C
tensor is `[M, N]`, so its bytes are proportional to N. For BERT the four widths
are `3:1:4:1`, so `ffn_up` is **4/9 = 44.4%**, and this is invariant across the
catalogue rather than fitted (MiniLM 1536/3456, bge-large 4096/9216 — both
exactly 44.4%).

| | MiniLM | bge-large |
|---|---:|---:|
| transport bucket (readout + convert + sync) | 34.56 ms | 323.39 ms |
| × 44.4% = relay-addressable | **15.35 ms** | **143.6 ms** |
| as a share of wall | **13.8%** | **8.7%** |
| **best case, zero rebuild cost** | **1.160×** | **1.095×** |

Against 0107's **1.25×–1.35%**: roughly halved, and for a legible reason — 0108
collected the host-side half of it.

**And the verdict does not rest on the 44.4% assumption.** Credit the relay with
the *entire* transport bucket — every byte of readout, convert and sync for all
four shapes, which it demonstrably cannot remove — and the ceiling is:

| | MiniLM | bge-large |
|---|---:|---:|
| absurd upper bound (100% of transport) | 1.450× | **1.242×** |

**On bge-large even that bound is barely above the 1.223× 0108 already
delivered**, on hardware, for a C++ rewrite with no array work at all. The
realistic figure is well below it, and three costs are priced at zero above:

1. **Array time goes up.** GELU moves onto the cores. 0109 measured array time
   as the larger piece on bge-large (56.6%); adding work to it is the wrong
   direction on exactly the model where the relay looks best.
2. **The relay's matmul is hand-written and slow** —
   [`0092`](../0092-t28-relay-bf16-output/TASK.md) says so, and
   [`0091`](../0091-t7-t8-gelu-poly/TASK.md) measured that class of gap.
   `kernels.mm()`'s MMAC operand order composing with a join's `dims_to_stream`
   is [T28](../../research/CLOSED-THREADS.md#t28) item (a), still unexplored.
3. **A separate design shape can reintroduce switches**, priced by
   [note 0004](../../research/notes/0004-context-switch-cost.md) at ~25 µs +
   7.2 µs per lock.

## Verdict

**RETIRE both.** Not because the mechanism failed — it is built and correct at
production tile width ([`0092`](../0092-t28-relay-bf16-output/TASK.md), rel_fro
2.510e-03) and the pipeline it needs works end to end
([`0054`](../0054-m10-phase-fusion-pipeline/TASK.md)) — but because **the value
was collected by something cheaper**, and what is left is under 1.16× best case
with three unpriced costs against it. This is the register's own precedent:
[T2](../../research/CLOSED-THREADS.md#t2), [T9](../../research/CLOSED-THREADS.md#t9)
and [T12](../../research/CLOSED-THREADS.md#t12) all retired on a measured "not
worth it", and rule 3b keeps the reasoning rather than the conclusion.

**T3's own 33%-of-the-encode figure is now three datapath generations old** —
it predates `--emulate-bfp16`, `--c-bf16` and 0108's fusion, each of which took
a piece of what it was counting. It should not be quoted again.

**The trigger for reopening, stated so a future session does not have to
re-derive it**: the relay becomes interesting again only if the **host** side
grows back to dominate. Two things would do that, and one is already measured —
[`0113`](../0113-t40-seq512-close/TASK.md) found host share at **75%** at seq
512, where attention alone exceeds the array. But at seq 512 the host work that
dominates is *attention*, not the FFN leg this relay addresses, so the lever
that matters there is [T42](../../research/OPEN-THREADS.md#t42), not this one.
The other would be a much faster array (a further datapath change), which
nothing currently proposes.

## Problems hit

Nothing broke. One thing worth naming: the honest version of this pricing needed
**both ends of the catalogue**, because the ordering is not obvious —
bge-large has the *largest* array share (56.6%) and the *smallest* relay-
addressable share (8.7%), while MiniLM is the reverse. Pricing on one model
would have given a number that reads as general and is not.

## Artifacts

The two `--bench 5 --pipeline 1` outputs quoted above, reproducible with the
commands given; no new files.

## Next

The register drops to **two** live threads: [T42](../../research/OPEN-THREADS.md#t42)
(attention on the array for long-sequence designs, filed with a trigger) and
nothing else. Everything else is closed with a pointer.
