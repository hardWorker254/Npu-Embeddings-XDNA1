# 0054 — M10: phase fusion, Del B (T28) — production-shape epilogue survey,
and the first working cross-core GEMM->mem-tile->GELU pipeline

- **Date** 2026-08-20
- **Milestone** M10 (Del B of the approved research plan, following 0053's Del A)
- **Status** done (B1 complete; B2 proves the core mechanism at small scale,
  a full production-scale ffn_up->GELU->ffn_down chain is NOT attempted —
  see Next)

## Goal

Execute Del B ("Kveld 2+: fasefusjon") of
`~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`: build and measure
toward the pipelined block-fusion architecture T28 names as the largest
unclaimed host-side lever, in `experiments/` only — no change to the shipped
production path (`runtime/src/main.cpp`'s `Encoder::run()`, `hub.cpp`, or any
`.npue` container contents).

1. **B1** — extend 0030's single-shape `epilogue="gelu"` proof
   (rel_fro 3.167e-04 at MiniLM's ffn_up shape only) to all three distinct
   ffn_up geometries the four shipped models actually use.
2. **B2** — a genuinely cross-core pipeline: GEMM output flowing through a
   mem tile directly into a GELU-consuming core, in ONE dispatch, with the
   intermediate never touching host DRAM — the mechanism note 0005 §4 names
   as the real win (AMD 15→3, STEEL 22.8×, ARIES adjacent-tile handoff),
   distinct from B1's same-core epilogue fusion.
3. Note where `xrt::runlist`/`disable_synchronization`+`delegate_tile` would
   help if a concrete use turns up (none did — see Next).

## Context

[`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md) (Del A) filed
[T28](../../research/OPEN-THREADS.md#t28--true-phase-fusion-0030-4-the-pipelined-block-fusion-design--open--filed-2026-08-20-registry-gap)
and left Del B for this session. `experiments/m5-pretiled-gemm/gemm_pretiled.py`
already carries `rtp=True`, `epilogue="gelu"` and `c_bf16` as proven,
individually-tested mechanisms (note 0005 §1/§2, tasks 0030/0045), but
`epilogue="gelu"` had only ever been measured at ONE shape
(`experiments/m7-switch-cost/gelu_fusion_probe.py`, M=1024/K=384→448/N=1536,
2 columns) and no cross-core pipeline had ever been attempted at all.

## What was done

### B1 — `epilogue="gelu"` at all three production ffn_up shapes

Wrote `experiments/m5-pretiled-gemm/epilogue_gelu_survey.py`, generalising
`gelu_fusion_probe.py`'s pattern (K-augmented bias trick, tile_b pre-tiling,
bf16-precision reference) over shape and `tile_n`. The four shipped models
reduce to **three distinct GEMM geometries** for ffn_up (K=h, N=4h):
MiniLM-L6 and bge-small-en-v1.5 share one (h=384, tile_n=48 — literally
0030's own shape), bge-base is a second (h=768, tile_n=48, per 0051), and
bge-large is a third (h=1024, **tile_n=32**, per 0042 — 48 is illegal at
N=4096). The epilogue kernel entry point `gelu_epilogue_3072_f32` was
hardcoded to `m*n=3072` (true for tile_n=48), so bge-large's tile
(64,64,32) needed a new one:

- Added `gelu_epilogue_2048_f32` to `experiments/m5-eltwise/kernels/gelu_poly.cc`
  (same body, `n=2048`) — purely additive, same file production's
  `gelu_poly_bf16`/`gelu_poly_bf16_4k` already live in.
- Generalised `gemm_pretiled.py`'s epilogue-kernel selection (was
  `assert m*n == 3072`) to a `{3072: ..., 2048: ...}` lookup that fails
  loudly on an unrecognised tile size instead of silently picking the wrong
  entry point.

Measured, per shape: (1) accuracy — fused `gelu(A@B+bias)` vs an exact-erf
reference at bf16-quantised, fp32-accumulate precision (the datapath's own
numbers, not a device read-back — CLAUDE.md trap 6c); (2) dispatch latency
(fused K-augmented+epilogue GEMM vs the plain unaugmented GEMM alone, same
M/N/cols) via `aie.utils.benchmark.run_iters`, explicitly labelled
wall-clock-derived (host timer around `kernel.wait()`), never presented as a
hardware trace; (3) a genuine hardware trace (per-core cycles) for the
MiniLM/bge-small shape, fused vs plain, at 4 columns (the traceable width —
CLAUDE.md trap 7).

Every build purges JIT cache candidates matching its exact ordered
`aie.runtime_sequence` signature before compiling — 0053's own bug (matching
tile dims instead of per-call shape) and 0045's marker collision (two shapes,
same three sizes) are both instances of the class this guards against.

### B2 — a genuine cross-core GEMM->mem-tile->GELU pipeline

**Architecture chosen.** A full ffn_up→GELU→ffn_down THREE-stage chain
(the plan's stretch goal) needs ffn_down's GEMM to consume ffn_up's ENTIRE
N=4h output as its K-reduction input — which spans every GEMM column's
N-slice, i.e. a genuine many-to-many cross-column regather, not a simple
hop. That is a materially harder problem than this session's budget allows
(see Next). Built instead: the smallest design that proves the actual
mechanism T28 names — GEMM core output flowing through a mem tile straight
into a GELU-consuming core, ONE dispatch, intermediate never DMA'd to host
DRAM. One column, all 4 compute rows: rows 2-3 do GEMM (tile 64×64×48, no K
augmentation — this probe tests ROUTING, not fusion accuracy again), rows
4-5 do GELU. `rt.sequence()`'s signature is exactly `(A, B, Y)` — **no C
tensor at all** — which is the proof-by-construction that the intermediate
never leaves the array, not an inference from timing.

New files: `experiments/m5-pretiled-gemm/pipeline_gemm_gelu_probe.py` (the
design), `pipeline_diag_gemm_only.py` (the diagnostic that found the real
bug — see Problems), `pipeline_bench.py` (dispatch-latency comparison).
`gelu_poly.cc` gained two more entry points: `gelu_epilogue_3072_f32_io`
(separate in/out args — the same-core epilogue kernel is in-place and
cannot be reused across a core boundary) and `identity_copy_3072_f32` (a
pure-copy diagnostic kernel used to isolate routing bugs from GELU-math
bugs — see Problems #4).

This took five wrong designs before it ran correctly. Each is recorded
below because each is a genuine, previously-undocumented IRON trap.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm

# B1
python epilogue_gelu_survey.py --shapes minilm_bge_small --no-trace --iters 5 --out artifacts\epilogue_smoke.json
python epilogue_gelu_survey.py --iters 30 --out artifacts\epilogue_gelu_survey.json

# B2 (each preceded by rm -rf ~/.npu/cache/* during debugging -- see Problems #5)
python pipeline_diag_gemm_only.py
python pipeline_gemm_gelu_probe.py --identity
python pipeline_gemm_gelu_probe.py --out artifacts\pipeline_probe.json
python pipeline_gemm_gelu_probe.py --trace --out artifacts\pipeline_probe_traced.json
python pipeline_bench.py
```

## Result

### B1 — accuracy (all three shapes PASS)

| shape | h | tile_n | K_real→K_aug | N | rel_fro | tolerance |
|---|---:|---:|---|---:|---:|---|
| MiniLM/bge-small | 384 | 48 | 384→448 | 1536 | **3.167e-04** | PASS (identical to 0030 — control reproduced) |
| bge-base | 768 | 48 | 768→832 | 3072 | **2.102e-04** | PASS |
| bge-large | 1024 | 32 | 1024→1088 | 4096 | **1.764e-04** | PASS |

The MiniLM number reproducing 0030's `3.167e-04` **exactly** is this
session's control with a known-correct value (M5's rule). Accuracy actually
*improves* slightly with h, consistent with more of the tile mass sitting in
the K-loop (unaffected by the epilogue) as h grows.

### B1 — dispatch latency (wall-clock-derived; NOT a hardware trace)

Fused (K-augmented GEMM + epilogue) vs plain (unaugmented GEMM alone, no
GELU at all) — this is a LOWER-bound comparison, not "fused vs the real
separate-dispatch-plus-host-round-trip path" (0044/0045 already carry that
number: ~70 ms/encode at batch 128 for the readback+convert+sync this
epilogue removes; not re-measured here):

| shape | fused npu avg (µs) | plain npu avg (µs) | ratio |
|---|---:|---:|---:|
| MiniLM/bge-small | 2732.2 | 1021.0 | **2.68×** |
| bge-base | 6377.2 | 3822.1 | **1.67×** |
| bge-large | 10790.8 | 6699.4 | **1.61×** |

The ratio **shrinks as h grows** even though the absolute K-augmentation
overhead is the SAME +64 (a smaller fraction of a bigger K: 16.7% at
MiniLM's K=384, 8.3% at bge-base's K=768, 6.25% at bge-large's K=1024) —
consistent with a roughly FIXED per-dispatch cost that a bigger design
amortises better, though the mechanism was not isolated further this
session.

### B1 — hardware trace, MiniLM shape, 4 columns (per-core cycles)

Raw `get_cycles_summary` windows include the "zero-kernel" artifact 0048/0049
already documented (sub-1000-cycle windows that deflate a blind mean) —
filtered at >2000 cycles, matching that established methodology:

| | fused (n=40 raw, 35 filtered) | plain (n=157 raw, 134 filtered) | ratio |
|---|---:|---:|---:|
| filtered mean cycles | 8066.3 | 7708.5 | **1.046×** |

**The per-core COMPUTE overhead of K-augmentation + the GELU epilogue is
only ~5%**, sharply smaller than the 2.68× dispatch-latency gap above for
the same shape — meaning most of that dispatch-latency gap is NOT array
compute. This echoes 0045's finding (the bf16-narrow epilogue "disappears
into the DMA shadow") but the magnitude discrepancy here (~5% traced vs
~168% dispatch-latency) is large enough that it deserves a dedicated look,
not asserted as understood. The raw window-count mismatch (40 vs 157
invocations for what should be the same `n_tiles_per_core=32` loop trip
count) is itself unexplained and consistent with 0049's separately-recorded
warning that trace packet counts are not always reliable at this trace
size/design — **flagged, not resolved**.

Full JSON: `experiments/m5-pretiled-gemm/artifacts/epilogue_gelu_survey.json`,
log: `epilogue_gelu_survey.log`.

### B2 — the pipeline works: NO C tensor, correct numbers

Final, working design (`pipeline_gemm_gelu_probe.py`, after the fixes in
Problems below), one column, 2 GEMM rows -> mem tile -> 2 GELU rows,
M=128/K=64/N=48, no K augmentation:

| variant | rel_fro | tolerance | note |
|---|---:|---|---|
| `--identity` (pure copy instead of GELU — isolates routing from math) | **3.550e-08** | PASS | essentially exact |
| real GELU (`gelu_epilogue_3072_f32_io`) | **9.052e-04** | PASS (1.5e-2 bar) | same polynomial as B1's, ~2.9x B1's number at a different seed/shape, still far inside tolerance |

**`rt.sequence(A_ty, B_ty, Y_ty)` — three buffers. No C.** The GEMM output
never has an `rt.fill`/`rt.drain` touching host memory; it is produced by
`C_outs[j]`, forwarded through the mem tile (`ObjectFifo.forward()`) directly
into the GELU core's consumer fifo, and only the GELU'd result (`Y`) reaches
L3. This is the proof the plan asked for, by construction rather than
inference.

### B2 — dispatch latency (context, not a production claim)

`pipeline_bench.py`, wall-clock-derived (`run_iters`), same M/K/N for both:

| design | npu avg (µs) | npu min (µs) |
|---|---:|---:|
| pipeline (GEMM->mem-tile->GELU, 1 dispatch) | 212.1 | 187.0 |
| GEMM-only (same shape, 1 dispatch, no GELU at all) | 194.9 | 158.6 |
| **ratio** | | **1.088×** |

The GELU stage adds only ~9% to a bare GEMM dispatch at this tiny scale —
directionally consistent with B1's finding that the epilogue's own compute
is cheap and mostly DMA-shadowed. Not a production-scale number (2 rows,
one tile) and not claimed as one.

### B2 — tracing attempted, did not yield cycle numbers

Tracing BOTH a GEMM core and a GELU core failed to route:
`'aie.packet_rules' op packet switched source DMA0 cannot match another
connect or masterset operation`. Tracing the GEMM core ALONE hit the
identical error. Tracing the GELU core alone compiled and ran (still PASS,
rel_fro unchanged) but produced an empty cycle list (`get_cycles_summary`
found no event0/event1 pairs). This design's mem-tile routing (A split + B
forward + a forward PER GEMM->GELU pipe + a final Y join, all sharing
column 0's shim) apparently leaves less routing headroom than a plain GEMM
design's, so CLAUDE.md trap 7 ("adding one trace flow exhausts routing")
bites at a smaller design than usual here. **No hardware-trace cycle count
exists for the pipelined design** — the correctness result stands on its
own merits (a real reference comparison, not a trace), but any FUTURE
performance claim about this specific design needs its own tracing
investigation first, per rule 1.

## Problems hit

Recorded in the order found, because each one changes what the next
attempt tried.

1. **`ObjectFifo` cannot be BOTH a join-destination and a split/forward-
   source.** First design: GEMM cores `.join()` into one mem-tile buffer,
   then `.split()` that SAME buffer out to the GELU cores. Compile error:
   `'aie.objectfifo' op objectfifo cannot be in more than one
   ObjectFifoLinkOp`. Confirmed by reading `ObjectFifoLink.__init__`
   directly (`dataflow/objectfifo.py`): "An ObjectFifoLink may only have > 1
   of either sources or destinations, but not both" — so an N:M
   fan-in-then-fan-out through one buffer is not expressible at all, not
   just misconfigured. **Fix**: since this probe has `N_GEMM_ROWS ==
   N_GELU_ROWS == 2`, pair them 1:1 — two independent point-to-point chains
   instead of one shared merge/split buffer.
2. **A bare point-to-point `ObjectFifo` (no `.join()`/`.split()`/
   `.forward()` at all) compiles but the kernel HANGS** at runtime
   (`ERT_CMD_STATE_TIMEOUT`). No stuck `hw_context` afterward (`xrt-smi
   examine -r all` clean both times). **Fix attempted**: wrap the hop in an
   explicit `.forward()` (the SAME primitive B's shim→L2→core broadcast
   already uses, just fed by a core instead of a shim) — did NOT fix the
   hang by itself (see #3); recorded because it's still probably the
   architecturally correct form and the eventual working design uses it.
3. **The hang reproduced in a design with NO cross-core hop at all.**
   Suspecting the 2-GEMM-row geometry itself (production always uses all 4
   rows), wrote `pipeline_diag_gemm_only.py` — 2 rows only, straight
   `join()`+`rt.drain()` to host, no GELU, no forward. **Also hung.**
   Widened to 4 rows (matching gemm_pretiled.py's own row count exactly):
   **also hung**, ruling out row count as the cause. This redirected the
   search from "the pipeline mechanism is broken" to "something in this
   from-scratch script's basic plumbing is broken" — a materially different,
   and much more tractable, hypothesis.
4. **Root cause: a hand-built `TensorAccessPattern` for a plain full-region
   copy compiles cleanly but hangs the hardware.** `a_tap =
   TensorAccessPattern((M*K,), 0, [M,K], [K,1])` — a straightforward 2D
   row-major descriptor — passed every static check (this is NOT the
   earlier `[size],[1]` single-dim repeat-count-8191 compile error, a
   DIFFERENT bug caught earlier and fixed by using 2 dims) and still hung at
   runtime with no diagnostic at all. **Fix**: replace with
   `TensorTiler2D.simple_tiler(tensor_dims)[0]` — a tile whose size defaults
   to the whole tensor, i.e. the *same logical access pattern*, built through
   the library helper instead of by hand. That alone took the diagnostic
   from HANG to `rel_fro=3.888e-08` in one change. **Not chased to a root
   cause inside the helper's own generated AP** — recorded as a hard rule
   for future work: never hand-build a `TensorAccessPattern` for a plain
   full-region copy, even though the constructor accepts it and the compiler
   accepts the result. This is a new, previously-undocumented trap; it does
   not appear in CLAUDE.md's existing trap list and probably should once
   this note graduates.
5. **With routing fixed, results were WRONG but finite** (rel_fro 1.333,
   1.346 for the real-GELU and `--identity` pipeline respectively) — a data
   problem, not a hang. The `--identity` (pure-copy) variant reproducing
   the SAME magnitude of error as the real-GELU variant isolated the bug to
   *routing*, not GELU math. **Cause**: `Y_mem` (the final GELU-cores-join
   -to-host buffer) needs the identical `dims_to_stream` unscrambling
   formula `gemm_pretiled.py`'s own `C_l2l3_fifos` carries for exactly this
   reason — a mem-tile join of multiple producer tiles does not simply
   concatenate them byte-for-byte; the transform is required to undo
   whatever interleaving the join DMA actually does. Copied verbatim from
   `_build_design`; `pipeline_diag_gemm_only.py`'s own C-join needed the
   same fix (`rel_fro` 1.3-ish → 3.888e-08 there too).
6. **The dims_to_stream fix APPEARED to do nothing on first retest** —
   identical wrong numbers (1.346e+00 to 3 sig figs) before and after
   editing the source. **Cause**: `purge()`'s cache marker matched only the
   top-level `aie.runtime_sequence` signature and the epilogue kernel's
   symbol name, neither of which changes when an `ObjectFifo`'s
   `dims_to_stream` metadata changes elsewhere in the same design — so the
   JIT silently reused a stale cache entry compiled from the WRONG (pre-fix)
   source. This is the same fail-open class 0030's fifth fail-open and
   0053's `markers_for` bug both are (a purge marker specific enough to
   match the wrong thing, or not specific enough to catch a real change) —
   a THIRD, structurally distinct instance: this time the marker was too
   coarse for something entirely internal to the MLIR body. **Fix**: a full
   `rm -rf ~/.npu/cache/*` before the retest, which then showed the correct
   number (`rel_fro=3.550e-08` for `--identity`). Not fixed IN the scripts'
   own `purge()` functions this session — a real, minor gap for whoever next
   edits a design's internal `ObjectFifo` metadata without changing its I/O
   signature or kernel symbol names.
7. **Tracing exhausts routing at a SMALLER design than trap 7 usually
   implies** — see the B2 tracing result above. Not a new mechanism, but a
   new data point: trap 7 was previously calibrated on plain GEMM/eltwise
   designs at 2/4/8 columns; this pipeline hits the SAME wall with far fewer
   columns and far fewer total DMA streams, because its per-column stream
   COUNT (not width) is what's tight (5 mem-tile hops sharing one shim
   column: A split, B forward, 2 GEMM->GELU forwards, Y join).

## Artifacts

- `experiments/m5-pretiled-gemm/epilogue_gelu_survey.py` (new)
- `experiments/m5-pretiled-gemm/pipeline_gemm_gelu_probe.py` (new)
- `experiments/m5-pretiled-gemm/pipeline_diag_gemm_only.py` (new — the
  diagnostic that found problem #4; kept, not deleted, per tasks/README's
  "failures are the valuable part")
- `experiments/m5-pretiled-gemm/pipeline_bench.py` (new)
- `experiments/m5-eltwise/kernels/gelu_poly.cc` (added
  `gelu_epilogue_2048_f32`, `gelu_epilogue_3072_f32_io`,
  `identity_copy_3072_f32` — all additive)
- `experiments/m5-pretiled-gemm/gemm_pretiled.py` (generalised the epilogue
  entry-point lookup from a hardcoded `assert m*n==3072` to a dict; no
  behaviour change for existing callers)
- `experiments/m5-pretiled-gemm/artifacts/epilogue_gelu_survey.{json,log}`,
  `epilogue_smoke.json`, `pipeline_probe.json`, `pipeline_probe_traced.json`,
  `mlir_epi_minilm_bge_small_{fused,plain}.mlir`, `mlir_pipeline_probe.mlir`,
  and the (gitignored, `trace_*.json`/`.txt` pattern) trace files themselves
  — present on disk, not tracked, per the repo's existing `.gitignore`.
- **Not touched**: `runtime/src/main.cpp`, `runtime/src/hub.cpp`,
  `tools/pack_npue.py`, `runtime/src/npue_pack.cpp`, and no `.npue`
  container for any shipped model changed. Nothing here is wired into
  production.

## Next

**`research/OPEN-THREADS.md` T28 updated, left OPEN** (not ANSWERED — the
plan explicitly said not to close it unless the pipelined form is proven
end to end with numbers, and only a 2-op slice of it is):

- **Proven this session**: the pipelined-through-a-mem-tile mechanism is
  REAL and WORKS — a GEMM core's output tile reaches a different core's
  GELU computation with zero host DRAM round trip, in one dispatch, at
  correct numerical accuracy (9.052e-04), with the I/O signature itself as
  the proof (no C tensor). Five real IRON traps found and fixed on the way
  (recorded above), the most valuable being #4 (hand-built
  `TensorAccessPattern` hangs hardware silently) and #1
  (`ObjectFifoLink`'s one-link-per-object, no-N:M restriction).
- **NOT attempted**: the full production-scale ffn_up→GELU→ffn_down THREE-
  stage chain. The blocker is architectural, not a matter of more time on
  the same approach: ffn_down's K-reduction needs ffn_up's ENTIRE N=4h
  output, which is split across every GEMM column's own N-slice — feeding
  it into ffn_down's cores is a many-to-many regather across columns, not a
  single mem-tile hop like this session's 1:1 pairing. Whether that regather
  is expressible at all (given #1's N:M restriction) is an open question a
  future session should answer BEFORE attempting the build, not during it.
- **NOT attempted**: any dispatch-count-driven use of `xrt::runlist` (T9) or
  `disable_synchronization`+`delegate_tile` (note 0007 §1.5) — this
  session's designs are single-dispatch probes, so neither had a concrete
  use. They remain relevant the moment a multi-block pipelined design with
  many dispatches exists.
- **T3** (device-resident intermediates, ~33% of the encode per 0044) is
  now supported by a WORKING small-scale demonstration of its prerequisite
  mechanism, not just a priced argument — worth noting in T3 itself.
- **Open, not investigated further**: B1's trace/dispatch-latency
  discrepancy (§B1 hardware trace result above) and B2's routing-exhausts-
  tracing finding (#7) are both real, both flagged, neither explained.
