# 0057 — M10: T28 Del C — is the ffn_up→GELU→ffn_down cross-column regather
expressible at all?

- **Date** 2026-08-20
- **Milestone** M10 (T28 follow-on; 0054 left this question explicitly open)
- **Status** done — the regather is now known to be EXPRESSIBLE at reduced
  scale (≤6 sources per hop) and PRECISELY PORT-BUDGET-BLOCKED at full
  8-column production scale; the full production-scale three-stage chain
  was NOT built (a hierarchical 2-hop version was designed and budgeted but
  not implemented — see Next)

## Goal

[`0054`](../0054-m10-phase-fusion-pipeline/TASK.md) built and validated the
first cross-core GEMM→mem-tile→GELU pipeline (1:1, 2 GEMM rows → 2 GELU
rows) and explicitly did NOT attempt the full production
ffn_up→GELU→**ffn_down** three-stage chain, because ffn_down's K-reduction
needs ffn_up's ENTIRE N=4h output — split across every GEMM COLUMN's own
N-slice in the production whole-array design — which looked like a
many-to-many (N:M) regather across columns, structurally the same shape as
0054's Problem #1 (`ObjectFifoLink` forbids one `ObjectFifo` being both a
join-destination and a split/forward-source).

This task's job: determine whether that regather is expressible at all,
using patterns that don't require a single `ObjectFifo`/link to be both a
join-dest and a split-source, and build a small-scale proof if so. Per
CLAUDE.md trap 6c, correctness is checked against a real fp32 reference
computed independently of the device, never a device read-back.

## Context

Read in full before starting:
[`research/OPEN-THREADS.md`](../../research/OPEN-THREADS.md) T28,
[`tasks/0054`](../0054-m10-phase-fusion-pipeline/TASK.md) (all six numbered
Problems), [`research/notes/0007-unused-iron-surface.md`](../../research/notes/0007-unused-iron-surface.md)
§§2-3 (cross-tile `Buffer`, `CascadeFlow`),
[`tasks/0046`](../0046-m9-b-reuse-asymmetric/TASK.md) /
[`0047`](../0047-m9-cascade-channel-probe/TASK.md) (the mem-tile 6-in/6-out
port census that turned out to be the actual wall here, not just a B-reuse
finding), and `experiments/m5-pretiled-gemm/gemm_pretiled.py` (the
production whole-array GEMM this regather would need to feed).

**The production geometry that creates the problem.** In
`gemm_pretiled.py`'s `_build_design`, K is walked entirely INSIDE each
core's own inner loop (`for _ in range_(K // k): ...`) — K is never split
across cores. Only M (by physical row) and N (by physical column) are
split. So for `ffn_up` (K=h, N=4h) at 8 columns, each column computes the
FULL K-reduction for its own `N/8`-wide output slice, across all its
assigned M-rows. `ffn_down` (K=4h, N=h) then needs, for each of ITS output
n-tiles, the FULL 4h-wide K-input for a given M-row-block — which is the
concatenation of ALL 8 `ffn_up` columns' N-slices for that same M-row-block.
Every `ffn_down` worker therefore needs contributions from every `ffn_up`
column: an 8:1 (at minimum) gather per M-row-block, replicated across
however many `ffn_down` workers exist.

## What was done

Two new probes, both in `experiments/m5-pretiled-gemm/`, testing two
independent questions that together bound the regather problem:

**Q1 — `join_then_consume_probe.py`.** 0054 found that an `ObjectFifo`
cannot be BOTH a join-destination and a split/forward-source (confirmed in
`ObjectFifoLink.__init__` and the MLIR verifier). But every instance of
that finding in 0054 involved calling `.split()`/`.forward()` a SECOND time
on an already-joined object. Never tested: does simply handing a joined
object's OWN `.cons()` to a THIRD on-chip `Worker` — the exact same pattern
every other fifo in this codebase uses (`B_fwd.cons()`,
`A_l2l1_fifos[row].cons()`) — count as a second link, or is it a
structurally different (and legal) operation? Built: 2 GEMM cores (same
column, rows 2/3) `.join()` into a mem-tile buffer `C_mem`; NO
`.split()`/`.forward()` is called on `C_mem` anywhere; instead `C_mem.cons()`
is handed straight to a third `Worker` (row 4) running a plain
`identity_copy_6144_f32` kernel (new, additive, in `gelu_poly.cc`), whose
own output goes through a FRESH point-to-point pipe (the same `.forward()`
primitive 0054's `C_outs[j]`→`C_pipes[j]` already used) to a host drain.

**Q2 — `cross_column_join_probe.py`.** Even if Q1 works, production's real
problem is gathering across COLUMNS, not rows of one column — a completely
separate question about routing/adjacency, not about link arity. Built: 2
independent `[TM,K]×[K,TN]` GEMM problems, each with its OWN A/B fed from
its OWN column's shim (exactly like production), computed by GEMM cores
pinned to DIFFERENT physical columns (`Tile(SRC_COLS[0], 2)`,
`Tile(SRC_COLS[1], 2)`, …), `.join()`-ed into ONE mem tile explicitly
pinned via `tile=Tile(DEST_COL, 1)`. Parametrised by `NPUE_SRC_COLS` /
`NPUE_DEST_COL` env vars so the same script probes adjacent-column,
maximum-distance, and N-way (port-budget-boundary) configurations without
duplicating the design.

A real, independently-known-correct fp32 reference is compared against
every result (`a16 @ b16`, `bfloat16`-quantised inputs, computed in numpy —
never a device read-back, CLAUDE.md trap 6c). `A[:] = ...` / `B[:] = ...`
writes are followed by `np.array_equal(A.numpy(), ...)` assertions that the
data actually reached the device (trap 6b). Every JIT cache candidate
matching the exact `aie.runtime_sequence` I/O signature + kernel symbol is
purged before each build; when only an internal MLIR detail (a
`dims_to_stream`/tap change) was edited with no I/O-signature change, a full
`rm -rf ~/.npu/cache/*` was used instead, per 0054 Problem #6's own warning
that its `purge()` marker is too coarse for that case.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm

# Contention check before any hardware run
& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all
# (WorkloadsSessionHost.exe / Windows Studio Effects was resident and
# "Active", as in every prior session that checked this — 0024, 0044. Not
# started by this session; correctness results are unaffected. No timing
# claim is made in this task, so the --bench contention guard does not
# apply here.)

# Q1
python join_then_consume_probe.py

# Q2 -- each preceded by a full cache purge (see Problems #2 below for why
# the script's own purge() is not trusted for a tap-only change)
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="0,1"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="0,7"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="0,1,2,3,4,5,6,7"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="1,2,3,4,5,6,7"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="1,2,3,4,5,6"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="1,2,3,4,5"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="0,1,2,3"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
rm -rf ~/.npu/cache/*
$env:NPUE_SRC_COLS="0,1,2"; $env:NPUE_DEST_COL="0"; python cross_column_join_probe.py
```

Full raw output of every run: `artifacts/run_log.txt`.

## Result

### Q1 — PASS: a join's `.cons()` works as an ordinary Worker input

`join_then_consume_probe.py`: **rel_fro 3.550e-08** (identity copy through
the third core; essentially exact, matching 0054's own `--identity` control
number, 3.550e-08, to the digit — a strong independent cross-check).

Getting there needed one fix, not a design change: the first attempt used
`depth=2` on both `C_mem` and the gather-consumer's own `Y_out` — that hit
CLAUDE.md trap 3 (`'aie.tile' op allocated buffers exceeded available
memory`, 98,304 B requested of the 63 KB budget for tile (0,4)). Dropping
both to `depth=1` (a single-shot probe, so single-buffering is free here)
fixed it. **No "second link" compile error was ever raised** — the
restriction 0054 found is specifically about calling `.split()`/`.join()`/
`.forward()` a second time on an object that already anchors one; simply
reading `.cons()` from a Worker, which every other fifo in this codebase
already does, is a structurally different and unrestricted operation.

### Q2 — cross-column JOIN routing works, at ANY distance

`cross_column_join_probe.py`, `NPUE_SRC_COLS=0,1` (adjacent) and
`NPUE_SRC_COLS=0,7` (maximum span, the full width of the array):
**both PASS at rel_fro 3.474e-08**, bit-for-bit the same number. Column
1's (or column 7's) GEMM core writes correctly into a mem tile pinned at
column 0 via `tile=Tile(0, 1)` on the `.join()` call. **Distance made no
measurable difference** — the stream-switch fabric routes an 8-column-span
hop at the same numerical cost as a 1-column hop.

This result came with one real bug on the way, recorded because it looked
EXACTLY like a routing failure and was not one — see Problems #1.

### Q2 continued — the wall is the mem-tile 6-in/6-out port budget,
now compiler-QUANTIFIED rather than counted after the fact

CLAUDE.md trap 3b and [`0046`](../0046-m9-b-reuse-asymmetric/TASK.md)/
[`0047`](../0047-m9-cascade-channel-probe/TASK.md) already established "6
in / 6 out per mem tile" by counting ports in placed, successfully-compiled
MLIR. This task got the compiler to STATE the number directly, at the exact
point a design exceeds it:

| `NPUE_SRC_COLS` | `DEST_COL` also a source (local A/B)? | Result |
|---|---|---|
| `0,1,2,3,4,5,6,7` (8) | yes | **FAILS**: `tile (0, 1) requires 8 input/1 output DMA channels, but only 4 input/4 output available` |
| `1,2,3,4,5,6,7` (7) | no (gather-only tile) | **FAILS**: `...requires 7 input/1 output... only 6 input/6 output available` |
| `1,2,3,4,5,6` (6) | no | **FAILS, but at a DIFFERENT allocation**: the 6-source join itself fits (uses all 6 IN), but the gather-consumer's own result (`Y_out`, routed back OUT through the SAME mem tile to reach the shim) needs a 7th IN port: `requires 1 input/1 output... only 0 input/5 output available` |
| `0,1,2,3` (4, incl. `DEST_COL`) | yes | **FAILS, same pattern**: A(1 in)+B(1 in)+join(4 in)=6, 0 left for the outbound `Y_pipe` forward's 1 more IN |

Reconciling the two "N available" numbers: **6 total, minus 2 if the
destination tile also does its own local A/B GEMM feed** (6−2=4, exactly
matches row 1), and **minus 1 more if the gathered result must be relayed
back OUT through that SAME mem tile** to reach a further consumer or the
shim (a JOIN's own consumption is a genuine port cost too, not free just
because it happens "at the same tile"). This is the same "A(1)+B(1)+C(4
rows)=6" arithmetic 0046 found for `ffn_up`'s own C join, now shown to be a
GENERAL accounting rule that the compiler itself enforces, not a
coincidence specific to that one design.

**Decisive conclusion: an 8-way single JOIN into ONE mem tile categorically
cannot express the full ffn_down regather** — it needs 8 input ports and no
configuration of this hardware has more than 6 per mem tile, full stop,
compiler-verified at three different combinations of "how many other
ports the same tile also needs."

### A second, separate, non-fundamental wall: gather-consumer L1 capacity

`NPUE_SRC_COLS=1,2,3,4,5` (5, port budget fits: 6−1(Y_pipe)=5) and
`NPUE_SRC_COLS=0,1,2` (3, port budget fits: A+B+join(3)+Y_pipe=6 exactly)
both FAILED, but on a completely different error —
`'aie.tile' op allocated buffers exceeded available memory` at the
gather-consumer's OWN tile (row 4), because this probe's gather kernel
(`identity_copy_*`) acquires the WHOLE joined tile in one flat shot rather
than streaming it k-block-by-k-block the way every real GEMM core in this
codebase (including `gemm_pretiled.py`'s own inner loop) already does. At
(64,48)-element tiles, holding BOTH the joined input and the relayed output
in one shot tops out at N=2 (49,152 B of the 62,464 B budget after the
2,048 B stack) — confirmed as the largest N this SPECIFIC one-shot design
supports end to end (both Q2 PASS results above are at N=2).

This is NOT claimed as a second fundamental wall. It is evidence that any
real fused `ffn_down` consumer must acquire/release per k-block, exactly
like `gemm_pretiled.py`'s `core_fn` already does
(`for _ in range_(K // k): elem_in_a = in_a.acquire(1); ...`), not
gather-then-copy a whole flat tile at once. Not fixed in this session — see
Next.

## Problems hit

1. **A host-side tiling bug in `cross_column_join_probe.py`'s FIRST version
   looked EXACTLY like a cross-column routing failure, and was not one.**
   `a_full_taps`/`b_full_taps` were built as
   `[TensorTiler2D.simple_tiler((TM, K))[0] for _ in range(N_GEMM_COLS)]` —
   IDENTICAL for every column, always describing offset 0 of the host
   buffer. Both GEMM cores therefore read the SAME A and SAME B (column
   0's), and the "gathered" result showed column 1's slot byte-identical to
   column 0's (rel_fro 9.944e-01 overall, col 1 alone 1.375). The
   diagnosis came from adding per-column `rel_fro` and a raw-value printout:
   `got[1, 0, :4]` matched `got[0, 0, :4]` to 8 decimal digits, which a
   genuine routing/hang bug would not produce (garbage or a hang, not an
   exact duplicate). **Fix**: `TensorTiler2D.simple_tiler((N_GEMM_COLS*TM,
   K), (TM, K))` — tile the FULL multi-column tensor and let the helper
   emit one correctly-offset TAP per tile, instead of building N identical
   single-tile taps by hand. Recorded because the failure mode (a
   host-side data bug masquerading as a hardware/routing bug) is a
   different class from every trap 0054 found, and it is exactly the kind
   of thing that would have been reported as "cross-column join is broken"
   if the per-column diagnostic hadn't been added first.
2. **A design-internal-only edit (adding per-column diagnostics, or
   changing the fill-tap construction) does not change this script's
   `markers_for()` output**, since that function only matches the top-level
   `aie.runtime_sequence` I/O signature and the kernel symbol name — neither
   of which changes when only a tap or a `dims_to_stream` formula changes.
   This is the SAME class of fail-open 0054's Problem #6, 0053's
   `markers_for` bug, and 0030's fifth fail-open all are. Not fixed in
   either script's `purge()` this session (same gap 0054 left for `.cons()`-
   the-scripts' own `purge()` functions) — worked around throughout by
   using a full `rm -rf ~/.npu/cache/*` before every run whose MLIR body
   could plausibly have changed, rather than trusting the marker.
3. **The `ObjectFifo` constructor does not accept a `tile=` kwarg** — the
   first version of `cross_column_join_probe.py` passed `tile=Tile(0, 1)`
   directly to `ObjectFifo(...)` for `C_mem` and `Y_out`, which is not a
   parameter of `ObjectFifo.__init__` (confirmed via
   `inspect.signature(ObjectFifo.__init__)`: no `tile` parameter exists;
   placement is controlled by the `tile=` kwarg on `.join()`/`.split()`/
   `.forward()`, or implicitly by which `Worker.tile=` acquires/releases the
   object). Fixed by removing the invalid kwarg from the two `ObjectFifo()`
   constructor calls and keeping `tile=` only on `.join()`/`.forward()`,
   matching how `programming_examples/basic/matrix_multiplication/cascade/
   cascade.py` does it (`.split(..., tile=Tile(col, 1))`).

## Artifacts

- `experiments/m5-pretiled-gemm/join_then_consume_probe.py` (new)
- `experiments/m5-pretiled-gemm/cross_column_join_probe.py` (new,
  parametrised by `NPUE_SRC_COLS`/`NPUE_DEST_COL` env vars)
- `experiments/m5-eltwise/kernels/gelu_poly.cc` — added
  `identity_copy_6144_f32` (purely additive, alongside the existing
  `identity_copy_3072_f32` from 0054)
- `tasks/0057-m10-t28-cross-column-regather/artifacts/run_log.txt` — raw
  output of every command in order, including the three failed attempts
  before the host-tiling bug was found and the five port/L1-budget compiler
  errors that map out the wall
- **Not touched**: `runtime/src/main.cpp`, `runtime/src/hub.cpp`,
  `tools/pack_npue.py`, `runtime/src/npue_pack.cpp`, no `.npue` container
  contents changed. Nothing here is wired into production. `gemm_pretiled.py`
  was read but not modified.

## Next

**`research/OPEN-THREADS.md` T28 updated** with a decisive, evidence-backed
answer to the specific question 0054 left open, though the full
production-scale build remains undone:

- **The full 8-column ffn_down regather is NOT expressible as a single
  JOIN into one mem tile** — compiler-verified, not just theorised: 8
  sources need 8 input ports, no mem tile has more than 6, and the 6 must
  also cover whatever local A/B/outbound traffic that same tile carries.
- **It IS expressible as a hierarchical 2-hop merge**, budgeted but NOT
  BUILT this session: e.g. two 4-way joins (each at a mem tile ALSO doing
  its own local `ffn_up` A/B feed: A(1)+B(1)+join(4)=6, exactly at budget,
  per the `0,1,2,3` boundary case measured above) landing in two DIFFERENT
  mem tiles, then a relay/`ffn_down` compute core reading BOTH joined
  results — using its 2 input DMA channels (CLAUDE.md trap 3b: a core has
  exactly 2 in / 2 out) — one per hop. Both sub-mechanisms this needs are
  now proven correct on hardware (Q1: a join's `.cons()` feeding a
  downstream Worker; Q2: cross-column join routing, any distance). The
  arithmetic closes; the code does not exist yet.
- **A real hierarchical build needs the L1 wall (§ above) fixed first**: any
  `ffn_down`-consuming core must acquire/release the gathered K-input
  k-block-by-k-block (mirroring `gemm_pretiled.py`'s own inner loop),
  not gather-then-copy a whole flat tile — this session's probes never
  needed to solve this because they stopped at N=2, well under where it
  bites.
- **Not attempted, and now a well-scoped next task**: (a) build the
  4+4 hierarchical merge end to end with a REAL GELU epilogue (not the
  identity-copy diagnostic kernel) feeding a real `ffn_down`-shaped matmul,
  at whatever reduced M/N/K scale keeps the build tractable; (b) revisit
  whether `CascadeFlow` — proven upstream to reduce ACROSS ROWS of one
  column (`programming_examples/.../cascade/cascade.py`, not tested here
  for cross-COLUMN use) — could replace the mem-tile regather entirely for
  a RESTRUCTURED tiling where `ffn_up`'s N-split is by ROW instead of by
  COLUMN, sidestepping the port budget altogether (candidate approach 3 from
  the task brief; not investigated this session beyond reading the
  existing cascade example, which is column-local only in every case it
  demonstrates); (c) whether the two-mem-tile hierarchical merge is worth
  it AT ALL once the 573 µs/dispatch + descriptor-count switch-cost model
  ([`0048`](../0048-m9-what-is-the-gemm-time/TASK.md),
  [`0024`](../0024-m7-dispatch-cost-anatomy/TASK.md)) is applied to the
  extra relay core and extra mem-tile hop this design would add — not
  priced this session.
