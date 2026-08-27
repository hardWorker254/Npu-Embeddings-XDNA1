# 0130 — T45 item 1: the roof is a measurement now — 45.5 GB/s on the hardware timebase, and the wall clock was telling the truth

**Date**: 2026-08-27
**Goal**: T45's remaining item — one traced dispatch on the bfp16 datapath at
production tile width, converting the ~44 GB/s roof from an inference over
host-observed aggregates into a number on the array's own clock, and testing
whether the `--bench`/probe y-axis (host wall clock) was measuring the
machine or the kernel.

## The run

`gemm_pretiled.py`'s existing traced path (the 0049 instrument), production
tile `(64,64,48)`, bfp16 emulation, bge-base's `ffn_up` shape at production
M, **4 columns** (trap 7: tracing caps at 4; `TRACE_ROUTING` picks
`(trace_col=0, egress=0)`):

```powershell
. C:\dev\mlir-aie\iron_env.ps1
cd experiments\m5-pretiled-gemm
python gemm_pretiled.py --preset ffn_up -M 8192 -K 768 -N 3072 --cols 4 --emulate-bfp16
#   cols= 4 pretiled[k,n|st] cores=16 n=316  avg=1658.4 cyc  per-core=118.5 MACs/cyc (46.3%)
#   relfro=3.17e-04 PASS
python gemm_pretiled.py --preset ffn_up -M 8192 -K 768 -N 3072 --cols 4 --emulate-bfp16 --bench --bench-iters 50
#   pretiled[k,n|st]  npu avg 10006.2 us   npu best 9537.8 us   3.86 TFLOP/s
```

`trace.txt` is non-empty (rule: asserted before belief); `trace.txt`,
`trace.json` and [`analyze_trace.py`](analyze_trace.py) are stored here.

## The measurement

The 256 KB trace buffer captures ~350 of the traced core's 6,144 kernel
invocations — a steady-state window. What it yields *exactly* is the
**pitch** between kernel starts on the 1.808 GHz core clock:

```
python tasks\0130-t45-traced-roof\analyze_trace.py
  kernel starts: 350  pitch median 2956 cyc  mean 2932.5  p10 1473  p90 4294
  duty cycle 56.6%   effective 67.0 MACs/cyc/core
  implied dispatch 9.97 ms (hardware timebase)
  traffic 453.0 MB  ->  45.5 GB/s
```

* **Window vs pitch**: the kernel computes for 1,658 cycles and the next
  start comes 2,932 cycles later — a **56.6% duty cycle**. Within the
  window the core runs at 118.5 MACs/cyc (46.3% of the 256 MMAC peak;
  0049's traced-in-isolation ceiling is 146.7, so the window itself is 24%
  slower under streaming); across the pitch the effective rate is **67.0
  MACs/cyc/core**.
* **The dispatch on the hardware timebase**: 6,144 invocations × 2,932.5
  cycles = **9.97 ms**. The design is symmetric — every core runs the same
  schedule — which is what licenses extrapolating one core's steady-state
  pitch to the dispatch.
* **The wall clock agrees to 0.4%**: 50-iteration `--bench` of the same
  design measures 10.006 ms average (9.538 best). So on this dispatch, the
  host-observed y-axis every `--bench` and probe number rides on was NOT
  measuring machine business — T45's central doubt, resolved by
  cross-instrument agreement rather than assumption.
* **The roof**: 453.0 MB of DRAM traffic in 9.97 ms = **45.5 GB/s on the
  array's own clock** (45.3 GB/s by wall). Now four independent
  instruments agree: 0120's aggregate inference (~44), 0125's per-dispatch
  probes (43.8–46.7 on the large shapes), 0128's M-sweep marginal rate
  (44.0–45.5), and this trace (45.5).
* **The core-side signature**: within the captured span, `LOCK_STALL`
  events cover ~53% of the timeline and `INSTR_VECTOR` a minority — the
  core spends roughly half its time waiting on operand locks. The bfp16
  datapath is **traffic-bound on the array itself**, measured, not
  inferred. (Rough B/E-span accounting; the precise per-bucket anatomy at
  this operating point is T46-shaped follow-on work, not claimed here.)

## Caveats, stated

* Measured at **4 columns** (trap 7). The 8-column roof remains inferred
  from consistency (0125/0128) — but the four instruments now bracket it
  tightly, and 0128's marginal rate is an 8-column measurement of the same
  number on the validated y-axis.
* One shape, one datapath. The pitch p10/p90 spread (1,473–4,294) shows
  the schedule is not metronomic; the mean is what integrates.
* 0128's *fixed-cost shape-dependence* (nomic 278 vs base 180 µs) is NOT
  explained by this task and moves to T46's ledger as a residual.

## Closure

**T45 is ANSWERED** — both items done (0128 + this). Closure condition (per
0123's preamble rule): the 45.5 GB/s figure is a 4-column, one-shape
measurement whose extrapolation to 8 columns rests on 0128's marginal-rate
agreement; if a future design's probe numbers stop clustering at ~44–46,
the roof must be re-traced, not quoted.
