# 0062 — M11: T28 — the hierarchical 2-hop merge, built with real compute

- **Date** 2026-08-20
- **Milestone** M11 (T28 follow-on; 0057 designed but did not build the
  hierarchical merge; this task builds it)
- **Status** done — PASS on hardware, first attempt after fixing one new
  build trap; reproducible across seeds and deterministic on repeat

## Goal

[`0054`](../0054-m10-phase-fusion-pipeline/TASK.md) proved a 1:1
GEMM→mem-tile→GELU pipeline. [`0057`](../0057-m10-t28-cross-column-regather/TASK.md)
proved the two sub-mechanisms a full `ffn_up`→GELU→`ffn_down` chain needs
(a join's own `.cons()` feeding a third on-chip Worker with no second link;
cross-column join routing at any distance) and quantified, with the
compiler's own error text, exactly why an 8-way single JOIN cannot express
`ffn_down`'s full regather (no mem tile has more than 6 ports), leaving a
**hierarchical 2-hop merge** as the untested but arithmetically-sound
answer. Neither prior session built that hierarchical form, and both used
only an `identity_copy_*` diagnostic kernel — no real GELU, no real second
matmul.

This task's job, per the brief: build the hierarchical merge with (1) a
REAL GELU epilogue (not identity) at every producer, (2) the hierarchical
2-hop merge itself, with each merge mem tile ALSO carrying local A/B feed
for its own column's GEMM core, (3) a relay/`ffn_down`-consuming core that
acquires/releases its two gathered hops one at a time (not a one-shot flat
gather, which 0057 found tops out around N=2 at full tile size), (4) wired
into a real second-stage matmul, not a copy-through, and (5) correctness
checked against an independent numpy/fp64 reference, never a device
read-back (CLAUDE.md trap 6c).

## Context

Read in full before starting: [`research/OPEN-THREADS.md`](../../research/OPEN-THREADS.md)
T28, [`tasks/0054`](../0054-m10-phase-fusion-pipeline/TASK.md),
[`tasks/0057`](../0057-m10-t28-cross-column-regather/TASK.md),
[`research/notes/0008-iron-1.4-migration.md`](../../research/notes/0008-iron-1.4-migration.md)
(the toolchain moved to mlir-aie main / past v1.4.1 earlier tonight; the
`Runtime(seq_fn, fn_args)` callback API, `TaskGroup`, and `.prod(tile=)`/
`.cons(tile=)` shim pinning are all as documented there — this task is
written entirely against the new API, no migration needed).

`experiments/m5-pretiled-gemm/gemm_pretiled.py`'s own K-loop
(`for _ in range_(K // k): elem_in_a = in_a.acquire(1); ...`) is the
acquire/release discipline item (3) needed to mirror.

## What was built

**The scale, and why.** A REAL second-stage matmul — unlike the prior
sessions' identity-copy diagnostic — needs L1 for weights and an
accumulator on the relay core, on top of the gathered hop buffers
themselves. 0057's own boundary case (group=4, i.e. `NPUE_SRC_COLS=0,1,2,3`)
already used the ENTIRE one-shot gather budget (49,152 B of ~62,464
available after the stack) with NOTHING in it but a copy kernel; adding a
weight matrix, an accumulator, and an output buffer on top would overflow
regardless of anything else. This task uses **GROUP=2** (2 producer
columns joined per merge mem tile, not 4) and narrows the merge/gather data
to **bf16** (not fp32, which 0057's probes used) specifically to make room
for those three extra things — see the L1 table below. The relay's own
`ffn_down`-shaped output width is also reduced, **N_DOWN=16** (not
production's 48), chosen only so one output row is exactly one 16-lane fp32
vector in the hand-written kernel — a code-simplicity choice, not a hard
constraint.

Full shape: `TM=64, TK=64, TN=48` (native tile — the SAME (64,64,48)/48
production uses at h=384), `GROUP=2`, `N_HOPS=2` (4 producer columns total,
`K_total = 4*TN = 192`), `N_DOWN=16`.

**Architecture:**

```
col0 GEMM+GELU(bf16) --\
                         +--> C_mem_g0 (mem tile @ col0, ALSO local A/B feed
col1 GEMM+GELU(bf16) --/      for col0's own GEMM core)  --\
                                                              \
col2 GEMM+GELU(bf16) --\                                      +--> relay core
                         +--> C_mem_g1 (mem tile @ col2, ALSO  /    @ Tile(0,4)
col3 GEMM+GELU(bf16) --/      local A/B feed for col2)     --/     (2 in / 1 out)
```

Each producer column (0,1,2,3, all row 2): a real `kernels.mm()` bf16×bf16
→fp32 GEMM `(64,64)×(64,48)`, then a REAL GELU epilogue that narrows
straight to bf16 on the SAME core — `gelu_epilogue_3072_f32_to_bf16`
(new, in `gelu_poly.cc`), combining `gelu_poly_f32_epilogue`'s Horner math
with `narrow_f32_bf16.cc`'s RNE-narrow-on-store, `aie::set_rounding
(conv_even)` (trap 2b: the AIE default `floor` is a systematic downward
bias). This is what 0054/0057's `identity_copy_*` kernels never did.

Two merge mem tiles, each a `.join()` of GROUP=2 producer columns' GELU'd
bf16 tiles into ONE combined object, `dims_to_stream` set to the SAME
join-undo formula `gemm_pretiled.py`'s `C_l2l3_fifos` and
`pipeline_gemm_gelu_probe.py`'s `Y_mem` already use for `(64,48)` tiles at
these `(r,t)=(8,8)` — proven (0054 Problem #5) to undo whatever
byte-interleaving artifact the JOIN DMA itself introduces, independent of
tile content. Each mem tile is ALSO pinned (`tile=Tile(dest_col,1)`) as the
SAME mem tile that column 0 (resp. column 2)'s own local A/B feed already
lives on — the "mem tile does double duty" topology the brief asked for,
at GROUP=2 rather than 0057's tightest GROUP=4 boundary case (see L1 table
for why).

The relay core (`Tile(0,4)`) reads BOTH merge mem tiles directly via
`.cons()` — Q1's mechanism (0057), NOT `.forward()`, which is what let
GROUP=2 avoid the extra output-port cost 0057's 4-source case hit when it
ALSO needed an outbound relay hop. It streams: zero the accumulator once,
then **acquire hop0 → partial-matmul-accumulate → release hop0 → acquire
hop1 → partial-matmul-accumulate → release hop1** — never holding both
hops' buffers, both weight slices, the accumulator and the output buffer
all at production scale simultaneously. This is item (3), the mechanism
0057 flagged as "the new mechanism task 0057 didn't need and you do."

**Why the relay's own matmul is hand-written, not `kernels.mm()`.**
`kernels.mm()`'s A operand needs the MMAC intrinsic's `(r,s)` sub-tile
order — the transform `A_l2l1_fifos`' `dims_to_stream` applies on every
OTHER GEMM in this codebase. Composing that with the join's OWN
`dims_to_stream` (which undoes the join DMA's interleaving, not the MMAC's)
on top of a GELU+narrow chain is a genuinely different, unexplored
question this session did not chase (see "What was not attempted" below).
Instead `ffn_down_hop_matmul_g2_64x48x16` (new, hand-written, vectorised
via `aie::broadcast`+`aie::mul`/`aie::add` — not the 1,617×-slower scalar
form CLAUDE.md trap 5 warns about, just not MMAC-accelerated) operates
directly on the join's own block-concatenated, per-block-plain-row-major
output — established correct-without-permutation by
`pipeline_gemm_gelu_probe.py`'s own `Y_mem` result (an elementwise GELU
kernel, straight-line indexing, reproduces the exact-erf reference, so a
GEMM core's own accumulator tile is plain row-major from the KERNEL's own
point of view; nothing needs undoing at THAT boundary). **No performance
claim is made for this kernel** (CLAUDE.md rule 1); this task is about
mechanism and correctness, and the design is far too small to trace
meaningfully anyway (trap 7).

**L1 budget** (relay core, the binding one):

| buffer | bytes |
|---|---:|
| hop0 (bf16, 2×64×48, depth 1) | 12,288 |
| hop1 (bf16, 2×64×48, depth 1) | 12,288 |
| weight0 (bf16, 2×48×16, resident `Buffer`) | 3,072 |
| weight1 (bf16, 2×48×16, resident `Buffer`) | 3,072 |
| accumulator (fp32, 64×16, resident `Buffer`) | 4,096 |
| output (fp32, 64×16, depth 1) | 4,096 |
| **total** | **38,912** |

Comfortably inside the ~62,464 B budget after the stack (trap 3). Producer
cores: `A_l1`(8,192) + `B_l1`(6,144) + `acc`(12,288) + `out_c`(6,144) =
32,768 B, likewise comfortable.

**Correctness reference.** Independent numpy/fp64 computation
(`gelu_exact` via `math.erf`, never a device read-back — trap 6c):
`up_j = A@B_j` per column, `gelu_j = GELU_exact(up_j)`, quantised to bf16
(`.astype(bfloat16)`, matching the device's `conv_even`-rounded narrow),
then `acc_ref = Σ_j gelu_bf16_j @ w_j` where `w_j` is the SAME
fixed-seed (`WEIGHT_SEED=42`) weight the design bakes into the relay
core's resident `Buffer`s at compile time — since the weight is a
compile-time constant with no runtime fill, the ONLY way to check it is to
regenerate it from the same seed on the host, which is what the reference
does. `A[:] = ...` / `B[:] = ...` are followed by `np.array_equal` asserts
against the intended values (trap 6b) before the design ever runs.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm

# contention guard
& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all
# (WorkloadsSessionHost.exe resident, all contexts Idle -- same background
# process every prior session in this repo has noted; not started by this
# session. No timing claim is made anywhere in this task, so the guard does
# not block anything here.)
```
```bash
rm -rf ~/.npu/cache/*
```
```powershell
python hierarchical_merge_ffn_probe.py                # seed 11 (default)
python hierarchical_merge_ffn_probe.py --seed 7        # different data
python hierarchical_merge_ffn_probe.py --seed 11        # repeat, determinism check
```

## Result

**First hardware attempt after fixing the one build trap below: PASS.**

| run | rel_fro | worst abs | verdict |
|---|---:|---:|---|
| seed 11 (default) | 1.726e-03 | 9.222e-04 | PASS |
| seed 7 | 1.783e-03 | 9.978e-04 | PASS |
| seed 11 (repeat) | 1.726e-03 (bit-identical to the first) | 9.222e-04 | PASS, deterministic |

Tolerance was set at 3e-2 rel_fro going in (two chained bf16 narrows plus a
non-MMAC second matmul, an untested error budget) — the achieved ~1.7e-3
sits **~17× inside** that bound, and is the right order of magnitude for
"two independent bf16-narrow-and-requantise steps" against this project's
own established single-narrow floor (`1-cos` ~1e-5 to ~1e-4 for a single
bf16 GEMM/epilogue elsewhere in this codebase; two compounded narrows
landing an order of magnitude higher is unsurprising and not itself
alarming). The seed-11 repeat being bit-identical rules out a flaky-pass
explanation.

**This proves, on hardware, all four things item (1)-(4) of the brief
asked for, combined in one design for the first time**: real (non-identity)
GELU at every producer; the hierarchical 2-hop merge, each mem tile ALSO
serving its own column's local A/B feed; a relay core reading two separate
mem tiles through its two input channels via direct `.cons()` (not
`.forward()`); genuine hop-by-hop acquire/release (not a one-shot flat
gather); and a real second-stage matmul (not a copy-through) producing a
correct, non-trivial `ffn_down`-shaped output — all checked against an
independent reference that never touches the device's own output.

## Problems hit

1. **NEW TRAP: pulling more than one `ExternalFunction` entry point from
   the SAME multi-symbol `.cc` file, within ONE design, duplicate-symbol-
   fails the link.** First attempt put all three relay kernels
   (`zero_f32_1024`, `ffn_down_hop_matmul_g2_64x48x16`, `copy_f32_1024`) in
   one file (`ffn_down_relay_g2.cc`), following `gelu_poly.cc`'s and
   `narrow_f32_bf16.cc`'s own precedent of holding several entry points in
   one file. It failed to link:
   ```
   ld.lld: error: duplicate symbol: zero_f32_1024
   >>> defined at zero_f32_1024.cc: ...zero_f32_1024.o:(zero_f32_1024)
   >>> defined at ffn_down_hop_matmul_g2_64x48x16.cc:
       ...ffn_down_hop_matmul_g2_64x48x16.o:(.text.zero_f32_1024+0x0)
   ```
   (and five more, every pair of the three symbols against each other).
   Reading the object names in the error (`zero_f32_1024.o`,
   `ffn_down_hop_matmul_g2_64x48x16.o`, `copy_f32_1024.o` — one per
   REQUESTED entry point, not per source file) shows each requested symbol
   name compiles the WHOLE source file into ITS OWN object; there is no
   per-symbol extraction. Linking three such objects, each containing all
   three functions, links three copies of every function.
   **No prior build in this codebase had ever triggered this**: every
   existing multi-entry-point file (`gelu_poly.cc`, `narrow_f32_bf16.cc`)
   is used by designs that each pick exactly ONE symbol from it per build
   (`gemm_pretiled.py`'s epilogue/narrow selection is always a single
   dict lookup). This design is the first to need THREE symbols from what
   would have been one file in the same design.
   **Fix**: split into three single-symbol files —
   `ffn_down_zero.cc`, `ffn_down_hop_matmul_g2.cc`, `ffn_down_copy_out.cc`
   — matching the (previously implicit, now explicit) rule this trap
   reveals: **one `.cc` file may hold several entry points textually, but a
   single DESIGN may only ever pull ONE of them.** Worth adding to
   CLAUDE.md's trap list if this pattern (multiple small kernels feeding
   one core) recurs.
2. Everything else — the hierarchical join topology, the local-A/B-feed
   co-location, the direct `.cons()` relay read, the hop-by-hop
   acquire/release, the GELU-to-bf16 narrow, the hand-written second-stage
   matmul, the reference computation — worked on the FIRST hardware
   attempt after fixing problem #1. No second problem was found. (This is
   itself worth recording precisely, per the task brief's own expectation
   that this would need several wrong designs the way 0054 needed five:
   it did not, this time — the design was built directly on TWO
   already-proven mechanisms (0054's pipeline, 0057's Q1/Q2) rather than
   discovering a new one, and the scope reductions made deliberately in
   response to 0057's own L1 finding (GROUP=2, bf16 gather, hand-written
   relay matmul) were each chosen specifically to avoid re-hitting a
   wall 0057 had already characterised, not stumbled into.)

## What was not attempted

- **Composing `kernels.mm()`'s MMAC `(r,s)` sub-tile order with a join's
  own `dims_to_stream` join-undo transform**, which would let the relay
  stage use the fast MMAC-accelerated matmul instead of the hand-written
  one. This is a real, still-open IRON question (does a SECOND
  `dims_from_stream` layered on a join's `.cons()` compose correctly with
  the base object's own `dims_to_stream`? Untested in this or any prior
  session). Left open — see T28 update below.
- **GROUP=4** (the tightest port-budget boundary case 0057 measured
  directly) with a REAL second-stage compute: confirmed by this session's
  own L1 arithmetic to be infeasible for ANY design that also needs
  weight+accumulator+output resident on the relay core (49,152 B alone for
  a one-shot GROUP=4 gather, before anything else), not attempted on
  hardware since the byte count already rules it out. A version that
  streams GROUP=4 in SMALLER sub-chunks (rather than one-shot per hop)
  might still fit — not investigated; this session's GROUP=2, two-hop
  streaming design already demonstrates the acquire/release mechanism the
  brief asked for.
- **Full production scale** (8 columns, tile_n=48 output, K=1536 for a
  real `ffn_down`): this probe's K_total=192 and N_DOWN=16 are both a
  fraction of production's K=1536/N=384. The L1 arithmetic above shows
  WHY doubling the group size or restoring N_DOWN to 48 would overflow a
  relay core built this way; scaling to full production would need EITHER
  more than 2 hops (impossible — a core has exactly 2 input channels, trap
  3b) or a fundamentally different relay strategy (e.g. streaming smaller
  N_DOWN chunks per hop, or a THIRD tier of hierarchy). Not designed or
  estimated this session.
- No hardware trace was attempted (this design is smaller and has more
  mem-tile hops than `pipeline_gemm_gelu_probe.py`'s own design, which
  already exhausted trace routing at ONE flow per trap 7 — no reason to
  expect this one would trace any better, and no performance claim is made
  here that a trace would support anyway).

## Artifacts

- `experiments/m5-pretiled-gemm/hierarchical_merge_ffn_probe.py` (new)
- `experiments/m5-eltwise/kernels/gelu_poly.cc` — added
  `gelu_epilogue_3072_f32_to_bf16` (purely additive)
- `experiments/m5-eltwise/kernels/ffn_down_zero.cc` (new)
- `experiments/m5-eltwise/kernels/ffn_down_hop_matmul_g2.cc` (new)
- `experiments/m5-eltwise/kernels/ffn_down_copy_out.cc` (new)
- `experiments/m5-eltwise/kernels/ffn_down_relay_g2.cc` — created then
  DELETED this session (superseded by the three files above, Problem #1);
  not present in the final tree.
- `tasks/0062-m11-t28-hierarchical-merge/artifacts/run_log.txt` — raw
  output of the failing first attempt (Problem #1's exact linker errors)
  and every command after the fix.
- **Not touched**: `runtime/src/main.cpp`, `runtime/src/hub.cpp`,
  `tools/pack_npue.py`, `runtime/src/npue_pack.cpp`, no `.npue` container
  contents changed. Nothing here is wired into production.

## Next

**`research/OPEN-THREADS.md` T28 updated** (see that file): the
hierarchical 2-hop merge is no longer just arithmetically budgeted — it is
BUILT and PASSES on hardware, with real GELU, real hierarchical routing
with co-located local-A/B-feed mem tiles, real hop-by-hop streaming
acquire/release, and a real (if reduced-scale, non-MMAC) second-stage
matmul, all checked against an independent reference. The remaining gap to
a production-scale three-stage `ffn_up`→GELU→`ffn_down` chain is
(a) whether `kernels.mm()`'s MMAC operand order composes with a join's own
`dims_to_stream` (unexplored — would let the relay stage run at full
MMAC speed), and (b) a relay strategy for K/N wider than this session's
GROUP=2/N_DOWN=16 reduced scale, since a core's 2-input-channel ceiling
means widening past 2 hops needs a DIFFERENT structural idea, not just
bigger buffers.
