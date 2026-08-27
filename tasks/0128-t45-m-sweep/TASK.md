# 0128 — T45 item 2: the M-sweep intercept — the ~44 GB/s roof survives separating the fixed cost, and the fixed cost is not one number

**Date**: 2026-08-27
**Goal**: T45's second measurement — separate the fixed per-dispatch cost from
the marginal rate on the same y-axis the roofline uses, via an M-sweep. The
three prior fits disagreed (150 µs in `0010`, 573 µs in `0048`, 627 µs in
`0080`), and 0120's roofline carries no fixed-cost term at all.

## The instrument

`--probe-streams` now probes **all four batch tiers**, not just the top one —
a one-line change (removing the `st.batch != batch` filter) that turns the
probe into an M-sweep over 256 / 1024 / 2048 / 8192 rows on identical
geometry, 16 points per design set spanning **~0.7 → 604 MB** of dispatch
traffic. Byte accounting is 0124's corrected one. Timing-only: the buffers
hold whatever is staged, and a dispatch reads the same bytes regardless of
their values.

Rebuild: same command as [`0124`](../0124-t47-t50-runtime-fixes/TASK.md).
Sweep command per set (full transcript in [`raw.txt`](raw.txt)):

```powershell
runtime\build\npuembed.exe . --model <model> --artifacts <set> --probe-streams
```

Machine state: CPU 1% before, 6% after (mild activity near the end —
recorded, not hidden); resident `WorkloadsSessionHost.exe` contexts Idle as
in [`0125`](../0125-t47-gbs-audit/TASK.md).

## The fit

Least-squares `t = t0 + MB / B` over all 16 (op, tier) points per set:

| set | t0 (µs) | marginal BW (GB/s) | R² | n |
|---|---:|---:|---:|---:|
| `base_bfp16` (h=768) | **180.0** | **44.0** | 0.979 | 16 |
| `nomic_bfp16` (h=768) | **277.6** | **45.5** | 0.992 | 16 |
| `minilm_bfp16` (h=384) | 74.5 | 36.8 | 0.985 | 16 |
| `int8c_mini` (h=384, i8) | 85.3 | 29.6 | 0.971 | 16 |
| `large_bfp16` | — | — | — | 4 (its design set carries only the top tier; no sweep possible without a re-export) |

Fit script inline in the session; the regression is two parameters against
sixteen points, computed with stdlib only.

## What it says

1. **The ~44 GB/s roof survives the intercept.** On the two hidden-768 bfp16
   sets — where the shapes are wide enough to be clearly traffic-dominated —
   the *marginal* rate is 44.0 / 45.5 GB/s with the fixed cost separated
   out, at R² 0.98–0.99. 0120 §3c inferred ~44 from four aggregate points
   with no intercept; this measures the same number *with* one, on 32
   points. Still host-observed — T45 item 1 (the traced dispatch) remains
   the conversion to a hardware measurement.
2. **The fixed cost is real but not one number: 74–278 µs across sets.**
   It lands near `0010`'s 150 µs, not near `0048`/`0080`'s ~600 µs (whose
   fits ran through different code paths and shapes). And nomic vs bge-base
   — the *same* geometry class — differ 180 vs 278 µs, which a
   shape-independent per-dispatch constant cannot explain. The honest
   conclusion: **the two-parameter model is incomplete**, and what the
   residual structure is (a compute floor on small tiers? a per-shape
   term? instruction-stream length?) is exactly what the traced dispatch
   disambiguates. Do not quote a single "the fixed cost is X µs" from this
   task.
3. **The narrow shapes read a lower marginal rate** (MiniLM 36.8, its int8
   twin 29.6): K=384 dispatches may simply not reach the streaming steady
   state the wide shapes do. Consistent with 0125's per-dispatch
   observation that int8 implied-GB/s sits below bfp16 on identical shapes.
4. For [T48](../../research/OPEN-THREADS.md#t48)'s gate 2, the operative
   numbers are: roof ≈ **44–45.5 GB/s** (h=768 bfp16), fixed cost
   **≈ 180–278 µs** on those sets. A B-reuse re-price that halves B's bytes
   must charge the unchanged t0 on every dispatch.

## Problems hit

* `artifacts_large_bfp16`'s design.json carries only the batch-128 tier, so
  bge-large gets no sweep without a re-export — recorded as the same gap
  0120 hit with gemma's missing `--bench`. The h=768 sets cover the roof
  question; bge-large's tier gap only matters if its 32-wide tile is
  suspected of a different roof (0125's per-dispatch numbers say it is not:
  43.8–44.3 on the big shapes).
* First fit script ran in the wrong working directory — `raw.txt` not
  found; rerun from repo root.

## Status

**T45 stays OPEN**: item 2 (this task) is done; item 1 — one traced bfp16
dispatch at production tile width, 4 columns — is not, and it is the item
that converts the y-axis itself. The thread annotation carries this table.
