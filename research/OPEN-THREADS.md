# Open threads

Every question this project has written down and not answered, in one place,
with a status.

**Why this exists.** [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) found
that the expert review's **§6b** had been *"deferred with cause"*, then
explicitly unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
and then sat untouched for two more tasks — because nothing re-reads a deferral
when its cause expires. [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md)
Part 3 found the same shape again: [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)
wrote down the *correct* hypothesis for a mystery and left it unchased for **28
tasks**.

A `TASK.md` is a diary entry — it is written once and never revisited. That is
the right property for a diary and the wrong one for an open question. This file
is the register that gets revisited.

**Rules.** A thread is added the moment a task says "untested", "not measured",
"deferred" or "open question". It leaves only by being **ANSWERED** (with a
pointer to where), **RETIRED** (with a reason), or **SUPERSEDED**. Nothing is
removed silently, and a stale "open" in an old task is not evidence that the
thread is still open — this file is.

Status: **OPEN** · **ANSWERED** · **RETIRED** · **BLOCKED**

---

## Live, ordered by what they would change

### T30 — The C-drain guard is half-wired, and every shipped model is one step from it · **ANSWERED 2026-08-21 (found and fixed same day)**
**A design built with `N > 4096` COMPILES and returns the wrong answer.**
[`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §6, found while asking
only "does nomic's N=6144 `ffn_up` compile?".

`gemm_pretiled.py::_build_design` guards the C-drain DMA stride
(`if m * n_aie_rows * N > 2**20: tb_n_rows = 1`, tasks/0030). `C_tiles` uses the
guarded value; the fill loop's `current_tb_n_rows` computes its own from
`tb_max_n_rows // 2` and never reads it. When the guard fires, drains cover 1
row-block while fills stream 2, and most of the output is stale. Measured
`rel_fro` **7.074e-01**, 28/32 row-bands with max abs error > 1.0 — invariant
across cols 4/8, tile_n 16/32/48 and M 1024/8192, which is what ruled out a
hardware limit (BD size, stride range, DMA channels and L1 all checked clean).

**The boundary is exact and the near-miss is the point.** The guard fires at
`N > 2**20/(m·n_aie_rows)` = **N > 4096** at m=64, rows=4. Measured: N=3840 PASS
(3.291e-07), **N=4096 PASS (3.283e-07) — bge-large's real production shape**,
N=4224 FAIL (7.083e-01), N=6144 FAIL. `64·4·4096 = 1,048,576` is **exactly 2^20**
and the test is a strict `>`, so **the guard has never fired in a shipped
design**. The tasks/0030 fix has never executed in production and half of it is
wrong. The file's own comment records the near-miss — *"measured: N=4096 at
exactly 2^20 builds"* — without drawing the conclusion, because measuring **at**
a boundary never exercises what is beyond it.

**Also corrects a doc:** `docs/CURRENT_STATUS.md`'s "known walls" still lists
`hidden >= 1536` as an open *build* wall. It is not a build wall. It builds and
returns wrong numbers, which is strictly worse.

**FIXED** in 0068 §6b. The whole row-block walk is now driven by the guarded
`tb_n_rows` (`tb_step = 2 * tb_n_rows`; `row_base = tb*tb_step + pingpong*tb_n_rows`;
`current_tb_n_rows = min([tb_n_rows, ...])`), which at the historical unguarded
`tb_n_rows = 2` reduces algebraically to the originals — provably a no-op below
the threshold. Four gates: **N=6144 7.076e-01 → 3.283e-07 PASS**; the three
below-threshold shapes reproduce (8.168e-07, 3.288e-07, 3.284e-07); the M=256
single-row-block tail still builds correct (3.289e-07); and fixed-vs-unfixed
xclbins under the same toolchain differ by **74 bytes**, inside the 0029 UUID
budget.

The fallback is therefore NOT needed, but stays on record: split a gated
`ffn_up` into two N=3072 dispatches over the existing bge-base design set
(6144 = 2×3072; `64·4·3072 = 786,432 < 2^20`), at 12 extra dispatches per encode.

**Left behind, newly written down:** the shipped `artifacts_base` xclbin
**cannot be reproduced byte-for-byte from today's toolchain** — a scratch rebuild
with the *unmodified* generator still differs by 3,457 bytes, from the mlir-aie
1.3.4 → 1.4.x upgrade (tasks/0058). tasks/0059 established this does not affect
correctness (a shipped `.xclbin` is a static binary XRT loads regardless of what
built it), but "regenerating the release produces a different binary than the one
on disk" is a real property of the repo that had not been recorded.

**Standing consequence:** any future model wider than `N = 4096` per GEMM would
have been silently wrong before this fix, and the failure mode left no trace. See
also
[T31](#t31--design_fits-matches-on-k-alone-and-nomic-breaks-it--open-filed-2026-08-21).

### T31 — `design_fits()` matches on K alone, and nomic breaks it · **ANSWERED 2026-08-21 (filed and fixed same day)**
`pick_artifacts()` picks a design set by asking whether `hidden` appears as a
`"K"` in `design.json` (`main.cpp:1363`). nomic's K set is `{768, 3072}` —
**identical to bge-base's** — but its gated `ffn_up` is `N=6144` where
bge-base's is `N=3072`. So the check passes and hands nomic a design that
computes half the FFN width, silently.
[`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §7 verified this
against the real `artifacts_base/gemm_rtp/design.json`: three of four streams
match exactly, `ffn_up` does not.

The predicate is weaker than its own comment, which names precisely this danger
(*"a design built for another width has the same filenames and loads fine — it
would simply compute the wrong thing"*). It has been sound so far only because
every shipped model has `N = 4·hidden`.

**FIXED** in [`0069`](../tasks/0069-m13-nomic-arch2-container/TASK.md).
`design_fits()` now matches the **streams' `(op, K, N)`** against the
container's `hidden`, `intermediate` and `gated_ffn`, requiring every op to be
present and *every* occurrence of it to match. It reuses `parse_streams()`,
which every `design.json` has carried since 0032, so the hole closes on design
sets exported long before the geometry keys existed — **no re-export needed**.
`export_gemm_rtp.py` still writes explicit `hidden`/`intermediate`/`gated_ffn`
keys, because a design that states its own geometry beats one inferred from its
streams.

Falsification test, which is the part that matters — a predicate that only ever
says "yes" proves nothing: with `artifacts_nomic` moved aside and
`artifacts_base` still on disk, nomic reports **`no design`** while bge-base
stays **`ready`**. The four shipping models each still resolve to the set they
resolved to before, and their validation encodes reproduce their recorded
figures exactly (bge-base 4.297e-03, the number in 0051).

**It also closed a fail-open nobody had filed:** `print_catalog()` called
`pick_artifacts()` unconditionally, so `embeddinggemma-300m` reported **`ready`**
whenever any hidden-768 design happened to be present — despite arch=1 having no
NPU kernel at all and running entirely on the host. There is now a `cpu` state.

**And it exposed a third one, in the making.** With nomic's container packed and
its design built, `list` said `ready` — true about designs, wrong about
outcomes. `Encoder::run()` is a BERT forward pass, and arch=2 reuses BERT's
tensor names and shapes deliberately, so it would have read every tensor, run
the wrong model, and returned embeddings nothing downstream could question.
`set_model_shape()` now refuses on any architecture this build has no encoder
for (exit 2), written as a **whitelist of what is implemented** rather than a
blacklist — the pre-existing arch=1 diversion names the one arch it redirects,
which is exactly the mechanism by which anything new falls through to BERT.
`encoder_implemented()` is the single source both the `list` table and the
dispatch refusal read, so they cannot drift.

### T33 — `l2_normalize` is written to the container and never read, and it costs nomic 4.5 MTEB points · **OPEN, filed 2026-08-22**
**The runtime always L2-normalises. For one shipping model that is measurably
the wrong choice on classification tasks.**

`pack_npue.py` writes `"l2_normalize": true` into every container.
`main.cpp:90` hardcodes `g_l2_normalize = true` and **never reads the key** — the
eighth fail-open's shape, a literal that should have been data, except here it
has been harmless because every model wanted `true`.

nomic-embed-text-v1.5 is the first where it is not obviously right. Its
sentence-transformers pipeline is `Transformer + Pooling` with **no `Normalize`
module** — `encode()` returns vectors of norm ≈20.9 — while all four BERT models
end in `Normalize`. Measured in [`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)
Stage 4b, on the reference side alone:

| Banking77Classification | score |
|---|---:|
| nomic, **unnormalised** (what sentence-transformers gives you) | **83.77** |
| nomic, **normalised** (what this runtime gives you) | **79.23** |

**4.5 points, on a logistic-regression task.** STS is unaffected — cosine cannot
see a scale change — and clustering moves a little (38.29 → 38.96, the other
direction). So this is not a precision question; it is a question about which
geometry downstream code wants.

Two separate things to decide:

1. **Should `l2_normalize` be read from the container** rather than hardcoded?
   Cheap, and it turns a silent assumption into a stated one. The container
   already carries the key.
2. **Should nomic's container say `false`?** Less obvious. nomic's own
   documented usage calls `F.normalize`, and the Matryoshka recipe normalises
   after truncation — so `true` matches the model card while `false` matches
   what sentence-transformers actually returns. The model is ambiguous, not us.

Until decided, note the user-facing consequence plainly: **running nomic through
sentence-transformers and through this runtime gives vectors of different scale
for the same text.** Cosine agrees; a classifier trained on one does not
transfer to the other.

**How this surfaced is the useful part.** nomic *failed* its MTEB gate at
mean −0.68 / worst −4.54, and the number alone reads as "the new architecture
costs accuracy". The pattern said otherwise: the three cosine tasks agreed to
±0.03 while the two raw-feature tasks moved in **opposite** directions, which is
not the signature of a precision loss. The harness was comparing normalised NPU
vectors against unnormalised CPU ones. Fixed there; the underlying question is
this thread.

### T32 — The golden gate is structurally blind to cross-row corruption · **OPEN, filed 2026-08-21**
**Every "different" row in the validation batch is identical content, so any bug
that reads the wrong ROW is invisible to it.**

`--model X --artifacts Y` is this project's primary on-device accuracy gate, and
it runs the M3 goldens, which are **batch 4**. For a larger design the runtime
tiles those four sequences to fill it (`reps4 = batch / 4`, so **32×** at batch
128 — `main.cpp:2855-2870`). The code is honest that it "makes no accuracy claim
beyond batch 4". What nobody wrote down is the consequence: **a row-indexing or
cross-row-aliasing bug reads identical data whichever copy it lands on**, so it
cannot change the answer.

Demonstrated, not theorised. [`0070`](../tasks/0070-m13-nomic-runtime/TASK.md)
shipped a threaded in-place `swiglu_cpu()` whose write range genuinely raced
another row's read range. The golden gate **PASSED**. A 13-distinct-sentence
end-to-end run failed at worst `1-cos` **0.44**, with rows immediately after a
thread-chunk boundary wrong by rel err up to 1.23 — wrong sign and magnitude,
not bf16 noise — while every other row matched the oracle to ~1e-3.

So the gate that this project leans on hardest has a whole bug class it cannot
see, and the gate that caught it (`verify_embed_e2e.py`) is a separate tool that
is not run on every change.

Cheap fixes, in increasing order of value:
1. **Make the tiled copies distinguishable.** Even permuting the four sentences
   per tile, or scaling copy *i* by a known factor and dividing it back out,
   turns "identical content" into "content that must match its own row".
2. **Generate goldens at a batch that is not 4** — at least one fixture with as
   many distinct sequences as the design's batch, so the tiling disappears.
3. **Run `verify_embed_e2e.py` as part of the standard gate**, since distinct
   texts is exactly the property it has and the golden check lacks.

Until then, treat a golden-gate PASS as evidence about arithmetic and **not**
about indexing, and say so wherever the two get conflated.

### T1 — What *is* the GEMM's 3 ms? · **ANSWERED 2026-08-19**
**The count of tile iterations, not bytes.**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md): `ffn_up` and
`ffn_down` have identical MACs and differ **1.50× in bytes**, and measure
**4,196 vs 4,273 µs** — 1.8% apart, in the wrong direction. `GMAC/ms` is flat at
1.08–1.15; `GB/s` spreads 17.7–27.0. Reproduced on the `--c-bf16` set. Fit:
`t = 573 µs + 4.72 µs × iterations`, ≤2.3% residual, against 0010's traffic
model being 50% out on the discriminating pair.

Left a successor, [T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19),
answered the same day — the "4.3×" compared production against the
bfp16-emulated datapath's trace. Note therefore that this thread's own framing
("compute-shaped, but not compute") is half-superseded: it IS compute, on the
fp32 vector datapath.

### T2 — Claim B-reuse via cascade · **RETIRED 2026-08-19**
**Retired by [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19).** B-reuse
removes *bytes*; bytes are not the constraint. The 1.26–1.68× that
[`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) priced was priced with
the model 0048 refutes, and the cascade milestone
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) scoped existed only to
free channels *for* B-reuse.

The channel census from [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) and
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) keeps its value — it is
about what the array can express, not about B.

### T3 — Device-resident intermediates (expert review §6b) · **OPEN**
Raised in [note 0005](notes/0005-expert-review-tests.md) §6b, deferred *with
cause*, unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
re-priced by [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 4 at
**33% of the encode** — against §6b's original ~70 ms estimate, which never
counted the C readback. Needs the one-operator-per-core design that 0032's
16 KB program-memory wall dictates. **Activates inside the same architecture
as [T28](#t28--true-phase-fusion-0030-4-the-pipelined-block-fusion-design--open--filed-2026-08-20-registry-gap)
and does not have its own separate build-out.** **Update 2026-08-20
([`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md))**: the
prerequisite mechanism — a tile crossing from one core to another through a
mem tile with NO host DRAM round trip, in one dispatch — is no longer just
priced, it is **built and measured at small scale** (rel_fro 9.052e-04,
correct). T3's own 33%-of-the-encode number is still unclaimed at
production scale.

### T28 — True phase fusion (0030 §4, the pipelined block-fusion design) · **OPEN** · filed 2026-08-20, registry gap
[`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) §4 and
[note 0005](notes/0005-expert-review-tests.md) proved every individual
mechanism the pipelined post-attention/FFN fusion design needs (RTP-unified
GEMM shapes, a K-augmented GELU epilogue, 8-column eltwise via split/join,
heterogeneous workers all exist in IRON) and then priced the *naive* build:
spatially partitioning one static design across the 8 columns (some for
GEMM, some for eltwise) is **roughly a wash at batch 128** — squeezing
eltwise from 8 columns to 4 to make room costs **≈+75 ms/encode** (GELU +28,
LayerNorm +14, softmax +32) against the **≈+60 ms** the eliminated design
switches save. The winning form is not spatial partition but **pipeline**:
GEMM columns streaming into eltwise columns through the mem tiles, one
dispatch per layer block, so no column idles while another works — what
AMD's 15→3 dispatch reduction, STEEL's 22.8×, and ARIES' adjacent-tile
handoff all independently measured (CLAUDE.md F1). [T3](#t3--device-resident-intermediates-expert-review-6b--open)
(device-resident intermediates, ~33% of the encode per 0044) and §6a
(micro-batch pipelining) both activate inside this SAME architecture and are
not separate builds.

**Update 2026-08-20 ([`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md),
Del B): the pipeline mechanism itself now WORKS, at small scale — a real
2-op chain, not just priced.** A GEMM core's output tile now reaches a
DIFFERENT core's GELU computation through a mem tile, in ONE dispatch, with
**no C tensor at all** in the design's I/O signature (`rt.sequence(A, B,
Y)`) — the intermediate provably never leaves the array. Correct to
rel_fro **9.052e-04** (a pure-routing `--identity` variant that swaps GELU
for a copy measures **3.550e-08**, isolating that the routing itself is
exact). Dispatch-latency cost of the extra GELU stage over a bare GEMM at
the same tiny scale: **1.088×** (wall-clock-derived, not a trace — no trace
exists for this design, see below). Getting here took five wrong designs,
each a genuine new IRON trap, none previously documented in this repo:
(1) an `ObjectFifo` cannot be BOTH a join-destination and a split/forward-
source — confirmed in the library source, `ObjectFifoLink` explicitly
forbids N:M fan-in-then-fan-out through one buffer, not just this
particular attempt at it; (2) a bare point-to-point `ObjectFifo` with no
`.forward()` compiles but **hangs the hardware** at runtime with no
diagnostic; (3) **a hand-built `TensorAccessPattern` for a plain
full-region copy compiles cleanly and HANGS THE HARDWARE** — root-caused via
a from-scratch diagnostic that reproduced the hang even with NO cross-core
routing at all (ruling out row count and the pipeline mechanism itself),
fixed by using `TensorTiler2D.simple_tiler(dims)[0]` for the identical
logical access pattern instead; (4) a mem-tile JOIN of multiple producer
tiles needs an explicit `dims_to_stream` unscrambling formula even on this
tiny 2-tile case (gives a WRONG but FINITE result if omitted — not a
crash); (5) that fix appeared to do nothing on first retest because the
JIT's cache purge marker was too coarse to notice a `dims_to_stream`-only
source change, silently serving a stale pre-fix binary — a THIRD, distinct
instance of the marker-specificity fail-open class 0030 and 0053 already
found two of. Full account: [`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md)
Problems #1-6.

**NOT attempted, and the reason is architectural, not time**: the full
production ffn_up→GELU→**ffn_down** THREE-stage chain. ffn_down's
K-reduction needs ffn_up's ENTIRE N=4h output, which is split across every
GEMM COLUMN's own N-slice — feeding it to ffn_down is a many-to-many
regather across columns, not the 1:1 single-mem-tile hop this session
built. Whether that regather is even expressible given finding (1) above
(no N:M through one buffer) is itself the open question a future session
needs to answer before attempting the build. `xrt::runlist`
([T9](#t9--xrtrunlist--open--09-today)) and
`disable_synchronization`+`delegate_tile` remain unused — this session's
designs are single-dispatch probes with no concrete use for either yet.

**Upstream check, 2026-08-20 (no build, research only)**: read
`verifyObjectFifoLinks()` in `AIEObjectFifoStatefulTransform.cpp` at both our
checked-out commit (`ed23bba`, 2026-07-01, part of installed mlir-aie 1.3.4)
and the current GitHub `main` (past released v1.4.1, 2026-08-11) — **byte-for-
byte identical**. The one-`ObjectFifoLinkOp`-per-`ObjectFifo` rule (finding
(1) above) is a deliberate, tracked structural invariant, not something a
newer mlir-aie relaxes; a future session should not expect an upgrade to open
the N:M regather. What upstream DOES have that we don't: **`--aie-objectfifo-
liveness`** (PRs #3257/#3312, landed 2026-07-09/12), a new opt-in static pass
that turns one class of "compiles clean, hangs hardware, zero diagnostic"
bugs — coupled cyclic multicast under-buffering — into a compile-time error.
Whether it would have caught findings (2)/(3) above (plain point-to-point /
`TensorAccessPattern` hangs, no cycle involved) is unconfirmed; the PR
description scopes it to cyclic coupled multicasts specifically and lists
plain acquire-in-loop exhaustion as explicitly out of scope. **Not pursued**:
our installed mlir-aie is a built/installed 1.3.4, not a git checkout on a
branch — getting to 1.4.1 means sourcing or building a new wheel, a real
infrastructure undertaking CLAUDE.md already flags as high-risk ("a pip
install accident must not break the toolchain that took the most work to get
running"), and out of scope for a research session. Recorded so nobody
re-checks this before a deliberate toolchain-upgrade decision.

**Update 2026-08-20 ([`0057`](../tasks/0057-m10-t28-cross-column-regather/TASK.md)):
the cross-column regather question 0054 left open is now ANSWERED — it is
expressible at reduced scale and PRECISELY port-budget-blocked at
production (8-column) scale, with the exact numbers coming from the
compiler itself, not a count taken after the fact.**

Two things were tested, independently, both against a real numpy reference
(never a device read-back, trap 6c):

- **A join's own `.cons()` works as an ordinary Worker input for further
  on-chip compute — no second link.** 0054's restriction (an `ObjectFifo`
  cannot be both a join-dest and a split/forward-source) turns out to be
  specifically about calling `.split()`/`.forward()`/`.join()` a SECOND
  time on an already-linked object — every one of 0054's failed attempts
  did exactly that. Simply handing the join's OWN `.cons()` to a third
  `Worker`, the same pattern `B_fwd.cons()`/`A_l2l1_fifos[row].cons()`
  already use everywhere in this codebase, is a different, unrestricted
  operation: **PASS, rel_fro 3.550e-08**, matching 0054's own `--identity`
  control number to the digit.
- **Cross-column JOIN routing works, at ANY distance across the array.**
  Two independent `[TM,K]x[K,TN]` GEMMs, each fed from ITS OWN column's
  shim, computed by cores in DIFFERENT physical columns, `.join()`-ed into
  ONE mem tile pinned at a third column: **PASS at rel_fro 3.474e-08 for
  both an adjacent-column pair (0->1) and the maximum-distance pair spanning
  the WHOLE array (0->7) -- identical number, distance was free.** (One
  probe-construction bug on the way looked exactly like a routing failure
  and was not: both columns' host-side fill taps pointed at buffer offset
  0, so both GEMM cores silently computed the SAME problem -- diagnosed by
  per-column rel_fro showing column 1's output byte-identical to column
  0's, not garbage, and fixed by tiling the full multi-column tensor
  instead of building N identical single-tile taps by hand.)

**The wall, quantified by the compiler at the exact point it bites:** an
8-source join needs 8 mem-tile input ports; no mem tile has more than 6
(CLAUDE.md trap 3b / [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) /
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md), now compiler-stated
rather than counted from placed MLIR): `"tile (0, 1) requires 8 input/1
output DMA channels, but only 4 input/4 output available"` when the
destination tile also does its own local `ffn_up` A/B feed (6-2=4, exact),
`"...requires 7 input/1 output... only 6 input/6 output available"` for a
dedicated gather-only tile (the bare 6-port ceiling), and a THIRD distinct
failure at exactly 6 sources -- the join itself fits, but relaying the
gathered result back OUT through the SAME mem tile needs a 7th port
(`"only 0 input/5 output available"`). **A single JOIN cannot express the
full 8-column ffn_down regather in one hop, full stop, on this hardware.**

**What this means for T28: the full production-scale three-stage chain is
NOT proven inexpressible -- it is proven to need a HIERARCHICAL 2-hop merge**
(e.g. two 4-way joins, each exactly at the 6-port budget when its tile also
carries local A/B traffic, landing in two different mem tiles, read by a
relay/`ffn_down` core using its 2 input DMA channels, one per hop) -- an
architecture whose two load-bearing sub-mechanisms are now BOTH proven
correct on hardware, but which was not itself built this session. A
separate, non-fundamental L1 wall showed up in the small-scale probes (a
gather-consumer acquiring a whole flat joined tile at once tops out around
N=2 at this tile size) and is not a blocker -- it just means a real
`ffn_down` consumer must stream k-blocks, exactly like every other GEMM
core in this codebase already does, not gather-then-copy in one piece.
Full account, including every raw compiler error: [`0057`](../tasks/0057-m10-t28-cross-column-regather/TASK.md).

**BUILT 2026-08-20 ([`0062`](../tasks/0062-m11-t28-hierarchical-merge/TASK.md)):
the hierarchical 2-hop merge itself is no longer just budgeted -- it is
built, and PASSES on hardware, with REAL compute at every stage, not the
`identity_copy_*` diagnostic 0054/0057 both used.** 4 GEMM+GELU producer
columns (a real `kernels.mm()` GEMM, then a real GELU epilogue narrowing
straight to bf16 on the SAME core, new kernel
`gelu_epilogue_3072_f32_to_bf16`) feed two merge mem tiles (`.join()`,
GROUP=2 columns each, each mem tile ALSO carrying its own column's local
A/B feed -- the double-duty topology this thread named), read by ONE relay
core over its two input channels via direct `.cons()` (Q1's mechanism, not
`.forward()`), hop-by-hop: acquire hop0, partial-matmul-accumulate,
release; acquire hop1, partial-matmul-accumulate, release -- the
acquire/release-per-hop discipline 0057 flagged as untested. The relay's
own second-stage matmul is a hand-written (not `kernels.mm()`) vectorised
kernel producing a real, non-trivial `ffn_down`-shaped output. Checked
against an independent fp64 reference (never a device read-back, trap 6c):
**rel_fro 1.726e-03**, reproduced at a different seed (1.783e-03) and
bit-identically on repeat -- 17x inside the 3e-2 tolerance set going in.

**Scale reduction, and why**: GROUP=2 (not 0057's tightest GROUP=4
boundary case) and the merge/gather data narrowed to bf16 (not fp32) -- a
REAL second-stage matmul needs L1 for a resident weight and accumulator on
top of the gathered hop buffers, and 0057's own GROUP=4 one-shot gather
ALONE already used the entire one-shot L1 budget with nothing but a copy
kernel in it. `N_DOWN=16` (not production's 48) is a code-simplicity
choice only. Full L1 arithmetic and the port topology are in the task log.

**One new build trap found**: pulling more than one `ExternalFunction`
entry point from the SAME multi-symbol `.cc` file within ONE design
duplicate-symbol-fails the link -- every requested entry point compiles
the WHOLE file, with no per-symbol extraction, so N requested symbols from
one file link N copies of everything in it. No prior design in this
codebase had ever asked for more than one symbol per file per build
(`gelu_poly.cc` and `narrow_f32_bf16.cc` both hold several, but every
existing caller picks exactly one). Fixed by splitting the three new relay
kernels into three single-symbol files. Full error text and the fix are in
the task log.

**Still not attempted, and this is where the thread's remaining gap now
sits, precisely**: (a) whether `kernels.mm()`'s MMAC operand order composes
with a join's own `dims_to_stream` join-undo transform -- unexplored, and
the reason this session's relay matmul is hand-written rather than
MMAC-accelerated; (b) any relay design at GROUP=4 or N_DOWN=48
(production scale) -- the L1 arithmetic in the task log shows why a
one-shot version of either overflows, and no streamed/chunked alternative
was designed or estimated; (c) the third tier a full 8-column, K=1536
regather would need, since a relay core's 2-input-channel ceiling (trap
3b) is fixed regardless of how the first two tiers are built. T28 stays
**OPEN**: the hierarchical merge mechanism is now proven correct with real
compute, not just arithmetically sound, but full production scale is
neither built nor decisively ruled out.

### T29 — EmbeddingGemma-300M: does the tile_n=16 tax rule it out? · **OPEN** · gated fetch verified with a real token; tokenizer-table generation is now C++ too, no gap left in the fetch path
**2026-08-21 update (tasks/0067)**: **the one gap tasks/0066 left open is
closed — `gemma_tokenizer.bin` now has a C++ generator.** A from-scratch,
dependency-free JSON DOM parser (`runtime/include/json_min.hpp` +
`runtime/src/json_min.cpp`) backs a line-by-line port of
`tools/gen_gemma_tokenizer_table.py` (`runtime/include/gemma_tokenizer_gen.hpp`
+ `runtime/src/gemma_tokenizer_gen.cpp`), every validation guard preserved.
`prepare_model_gemma()` now generates-and-caches the table instead of only
reading one. Verified against the real, genuinely Python-generated reference
(not self-reference, CLAUDE.md traps 6b/6c): **byte-identical, sha256
`c7a03c2c35ffc2a16b5513bb11c3d04e4a19c84acb9254b36765510acbf5bc81`, both
exactly 9,020,206 bytes** — confirmed twice, once from a standalone verifier
and once from the real `npuembed.exe --prepare-model` with the cached file
deliberately removed. A from-scratch clone with only `HF_TOKEN`/`--token` set
can now produce a fully working `.npue` with no manual Python step. The
tile_n=16 question this thread is named for remains genuinely open and
unaffected by this update — it only concerns a future NPU-kernel version of
this model, and the shipped host-only path still needs none of that
machinery.

**2026-08-20 update (tasks/0066)**: **the gated HuggingFace fetch was tested
with a real `HF_TOKEN`, for the first time.** `hub.cpp`'s `table()` now
carries a real, verified catalogue row for `embeddinggemma-300m` /
`google/embeddinggemma-300m`, `sha256` **`cbf5a78393b6a033e0b8a63a57549964
f7ed5c6fbeb4ba0694214f36123f2fd2`** — pinned by downloading the OFFICIAL
gated repository and confirming it is **byte-identical** (same sha256, same
`config.json`) to the `unsloth/embeddinggemma-300m` mirror every prior task
verified against, so nothing already on record needed re-checking. Two
integration bugs (neither in the fetch/pack logic itself) found by testing
against a genuinely FRESH root rather than the already-populated checkpoint
directory: (1) the `embed <model>` subcommand's post-fetch dispatch was
BERT-only (`pick_artifacts()` unconditionally, throwing "no NPU design for
hidden 768" before Gemma's own dispatch ever got a chance) — fixed with an
arch-aware short-circuit right after `ensure_model()`; (2) **`gemma_tokenizer
.bin` has no way to be produced by a token-only fetch** — its generator,
`tools/gen_gemma_tokenizer_table.py`, is Python-only and this project's
shipped product is C++-only at runtime (CLAUDE.md rule 5), so a from-scratch
clone cannot self-produce it. Worked around for THIS test by copying the
already-generated table; **not fixed** — either port the generator to C++,
or ship the table (vocab-derived, checkpoint-independent) as a release/repo
asset. Full end-to-end correctness confirmed **bit-identical** against the
already-verified reference container on the same input, once the table was
present.

**2026-08-20 update (tasks/0064)**: **EmbeddingGemma-300M now runs end to end
in production C++, entirely on the host** — the arch=1 integration C3/C4
scoped for. `tools/pack_npue.py` gained `pack_gemma()` (dispatched from the
checkpoint's own `config.json["model_type"]`, never guessed): every GEMM
operand is stored PLAIN (F32, row-major, no tiling), because there is no NPU
kernel for this arch and nothing here ever becomes a DMA descriptor — which
means **the tile_n=16 tax this thread is named for DOES NOT APPLY to the
path that actually exists today**. It remains the right question for a
*future* NPU-kernel version of this model (unanswered, unchanged by this
task), but the integration that shipped needed none of the tile_n/L1-budget/
DMA-BD machinery at all. 317/317 packed tensors round-trip bit-exact against
the source checkpoint (which is confirmed ALL-F32 on disk, 314 tensors,
correcting an assumption that it shipped bf16). A new `GemmaEncoder`
(`runtime/include/gemma_encode.hpp`) runs the full 24-layer forward pass
(double-accumulated host GEMMs, `0061`'s tokenizer, `0063`'s RMSNorm/RoPE/
GeGLU kernels, MQA collapsed to direct K/V reuse since
`num_key_value_heads=1`) and is wired into `main.cpp`'s production
`Encoder::run()` as a genuinely separate early-dispatch path (not a branch
inside it — `Encoder` needs seven NPU `Design&` this arch has none of).
Checked against `reference/encoder_gemma.py` on two independent 4-sentence
corpora, real HuggingFace tokenization: worst `1-cos` **4.969e-12** and
**5.496e-13** — tighter than any other model's gate in this project (no bf16
rounding anywhere in this path). `npuembed.exe --model embeddinggemma-300m
--embed` reproduces the standalone verification CLI bit-for-bit; the BERT
path's MiniLM golden check is unaffected (`1-cos` 1.086e-05, identical to
its recorded history) after rebuild. `--serve` on this arch refuses loudly
(no HTTP endpoint built) instead of silently doing nothing.
**Not done**: the C++ packer mirror (`npue_pack.cpp` has no arch=1 support —
`--prepare-model` on a Gemma checkpoint only works through Python today);
`hub.cpp` gained a `gated` field and fail-closed `HF_TOKEN` bearer-auth
support but deliberately NO catalogue row for EmbeddingGemma (no session has
ever held `HF_TOKEN` to pin a verified sha256, and the auto-pack step is
still BERT-only C++); no batching/AVX2/threading (~7.9 s/sentence,
unoptimised by design — correctness was this task's stated priority); no
MTEB run on this arch at all yet. Full detail:
[`0064`](../tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md).

**2026-08-20 update (tasks/0065): the C++ packer mirror gap CLOSED, fully —
`--prepare-model` on a Gemma checkpoint now works through C++ too, and it is
proven byte-identical to `pack_npue.py`, not just structurally similar.**
`runtime/src/npue_pack.cpp` gained `prepare_model_gemma()`, a direct port of
`pack_gemma()`: `Writer::write()` grew one new `arch` parameter (default 0,
BERT call site unchanged), a new `add_gemm_b_host()` sits next to the
existing tiled `add_gemm_b()` and is less code than it (transpose + raw F32
copy, no tiling, no `layout_hash`), and `main.cpp`'s `--prepare-model` CLI
gained an early `config.json["model_type"]` dispatch mirroring Python's
`main()`. **`tools/verify_pack_parity.py`, extended to detect the arch by
`model_type` instead of assuming every checkpoint has `vocab.txt`, reports
byte-identical `sha256` for both packers' 1,239.65 MB / 317-tensor output** —
confirmed a second way by a direct `sha256sum` on both files outside the
harness, and the BERT path stays byte-identical too (regression check). The
correctness gate was re-run against the FRESH C++-packed container rather
than assumed from parity: worst `1-cos` **5.496e-13** / **4.969e-12** on
tasks/0064's own two corpora, identical to the digit already on record, with
the raw encode output file `cmp`-matching tasks/0064's bit-for-bit. The one
real risk in the port — C++ reformatting `rms_norm_eps`/`rope_theta`/
`rope_local_base_freq` and disagreeing with Python's float-to-string in the
last digit — was sidestepped by copying the config.json literal substring
verbatim (checked correct for this checkpoint's exact text by a direct
`json.dumps(json.loads(x))` round-trip, not assumed in general — a future
checkpoint with non-canonical float formatting would FAIL parity loudly
rather than silently packing wrong bytes). What tasks/0064 left NOT done
(HF_TOKEN-gated fetch + `hub.cpp` catalogue row, MTEB, CPU speed work) is
unchanged and was out of this task's scope. Full detail:
[`0065`](../tasks/0065-m12-embeddinggemma-cpp-packer/TASK.md).

**2026-08-20 update (tasks/0055/0061/0063, superseded in relevance by 0064
above but kept for the history)**: the user decided to proceed ("Vi kjører
Gemma") — the
~62-72 seq/s prior below is accepted, not a blocker. **C2 (tokenizer) is done
and independently verified**, [`0061`](../tasks/0061-m12-embeddinggemma-tokenizer/TASK.md):
**1,925/1,925 sequence comparisons byte-identical to HuggingFace** (base
corpus 210/210 + a 300-codepoint byte-fallback stress corpus 1,715/1,715),
across five task-prefix configurations, `max_len` 64. **Load-bearing
correction to this thread's own framing and to the plan**: this checkpoint's
`tokenizer.json` declares `model.type == "BPE"`, not Unigram — the plan and
`0055`'s "~600-900 LOC SentencePiece Unigram" estimate both assumed the wrong
algorithm family (reasonable by analogy with other sentence-embedding
tokenizers; wrong for this one). It is the standard Gemma/Llama-family
SentencePiece BPE (metaspace normalizer, no pre-split word boundaries —
confirmed the whole prefixed+normalized text is ONE BPE input, not
per-word — byte-fallback via `<0xXX>` vocabulary entries, 514,906 merge
rules). `tools/gen_gemma_tokenizer_table.py` and `runtime/src/
tokenizer_gemma.cpp` implement BPE, not Viterbi/Unigram search — a materially
different core algorithm than what was asked for, caught only by reading
`tokenizer.json` before writing code. Task-prefix table (14 rows) read
verbatim from the checkpoint's own `config_sentence_transformers.json`;
project default is `"document"` (`"title: none | text: "`) — a decision,
since the checkpoint's own `default_prompt_name` is `null` (no prefix by
default under the standard sentence-transformers API). **Not yet wired into
`main.cpp`/`hub.cpp` — standalone only**, by design (0061's explicit scope).

**C3 (RMSNorm/RoPE/GeGLU host kernels) is done and independently verified**,
[`0063`](../tasks/0063-m12-embeddinggemma-kernels/TASK.md): `runtime/src/
gemma_kernels.cpp`, checked against real tapped intermediates from `0055`'s
checkpoint run — **36/36 records PASS**. Per-head `q_norm`/`k_norm` RMSNorm
and all 4 GeGLU cases are **bit-exact** (`rel_fro` 0.0); full-hidden RMSNorm
0.0–2.75e-11; RoPE 1.7e-8–4.3e-8 (the float32 ULP floor, confirmed against
numpy's own `cos`/`sin`). One `rms_norm_cpu` (double-precision reduction,
the Gemma `x/rms*(1+w)` form — NOT the more common `x/rms*w`) serves both
the full-hidden and per-head uses. GeGLU needed the reference's exact
two-stage float32 rounding (round `act`, THEN promote and multiply by `up`)
to reach bit-exact from ~1e-8 — a mathematically-equivalent single fused
double-precision expression was NOT enough, a real finding about matching
reference rounding order, not just reference formulas. **A negative control
run against the same real tensors** (drop RMSNorm's `1+`; swap RoPE's
per-layer theta; use the existing BERT `gelu_cpu`'s exact-erf form instead
of Gemma's tanh-approximation) measures `rel_fro` 0.439 / 0.757 / 3.26e-4 —
4-6 orders of magnitude worse than the real result, proving the test
discriminates rather than just measuring noise. **Not yet wired into
`main.cpp`/`hub.cpp`/the packer — standalone only**, same discipline as C2.
**Next slice if continuing**: the `arch=1` runtime branch in `Encoder::run()`
that actually calls these kernels in sequence, plus the `hub.cpp` catalogue
entry (with `HF_TOKEN`-gated fetch per the plan's decision) and the packer
changes (both `pack_npue.py` and `npue_pack.cpp`, held byte-identical) to
get MQA-fused QKV weights and the container's `arch=1` field into a real
`.npue` — none of which touch the NPU directly, since Gemma's host-side ops
have no NPU kernel yet either (matching this project's own "eltwise lives on
the host" precedent for the BERT models).

[`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md), Del C's C1 spike
(plan: `~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`). The numpy
reference encoder (`reference/encoder_gemma.py`) validates against HuggingFace
and sentence-transformers to **1-cos 1.065e-07 / 2.110e-08** — as tight as
M3's MiniLM oracle, despite 24 RMSNorm/RoPE/GeGLU/MQA layers against MiniLM's
6 LayerNorm/GELU ones. The architecture question the plan flagged is answered
with real numbers, not the plan's estimate: MQA's K/V width (1 KV head ×
head_dim 256 = 256) floors `tile_n` at **16 at 8 columns / 32 at 4 columns**
regardless of whether QKV or gate/up are fused offline — confirmed across all
four fusion combinations, matching the plan's prior exactly. L1 budget passes
comfortably at both (28,672 B and 40,960 B of 63 KB); the DMA BD 1024-dim
limit is nowhere close (worst case 80 n-blocks). **The real cost is
iteration count, per [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19)'s
own model**: a per-shape iteration count comparison against bge-base
(same hidden=768, tile_n=48, 12 layers) gives Gemma **3.58× more total
GEMM-iteration-proxy per encode** (24 layers × 516 vs 12 × 288), of which
isolating tile_n alone (hypothetical illegal tile_n=48 for Gemma) accounts
for **~3×** and GeGLU's genuinely lighter per-layer FFN width (2×1152 vs
bge-base's 1×3072) claws back roughly 0.6× — so this is a geometry tax, not
an inherent-FLOPs one. Feeding both shape sets through
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s own fit
(`t = 573 µs + 4.72 µs × iterations`, summed per shape then per layer) gives
a **2.90× dispatch-time ratio** (the fixed 573 µs/dispatch term dilutes the
raw 3.58× iteration ratio slightly) — a **PRIOR, not a measurement** (no
hardware trace exists for this model): roughly **62-72 seq/s**, against
bge-base's measured 181.2-209.1. This ignores new host-side work this model
needs that BERT-style models do not (4× RMSNorm/layer, RoPE, GeGLU's
elementwise gate multiply — all host per the plan's C3 sketch), so it is
likely optimistic, not pessimistic. **Not answered**: whether ~62-72 seq/s
(or worse) is worth the C2-C4 cost (new SentencePiece tokenizer ~600-900 LOC,
two packer rewrites, `arch=1` runtime branch, MQA-aware weight fusion) — a
product decision for the user, laid out in
[`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md)'s go/no-go section,
not concluded here.

### T4 — Finish [`0043`](../tasks/0043-m9-attention-geometry/TASK.md) · **OPEN**
Its Results section is empty. The four artifact sets are exported and the
commands are recorded; only the measurements are missing.
[note 0007](notes/0007-unused-iron-surface.md) §1.1 additionally reopens its
`cols ≤ 4` conclusion via mem-tile `pad_dimensions`, and §3.4 supplies
whisper-xdna's warning that fused attention was built elsewhere, was correct,
and still lost.

### T5 — Does DMA compression pay on real weights? · **OPEN** · cheap, decisive
[note 0007](notes/0007-unused-iron-surface.md) §2. Lossless, unused by
mlir-aie's own passes for the compute-tile path, and the only published ratio is
**1.39× on `arange`**. One run of `basic/dma_compression`'s `cmp_only` with a
real `.npue` tile answers it permanently.

### T6 — Does `aie_stream=` free a DMA channel? · **OPEN** · one probe
[`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) closed B-reuse on channel
exhaustion. `aie_stream=(end, port)` makes a producer wire-only with no L1
buffer ([note 0007](notes/0007-unused-iron-surface.md) §1.6); whether it also
costs no channel is unknown and directly relevant.

### T7 — `gelu_poly.cc` narrows through an emulated fp32 multiply · **OPEN** · free
[`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) measured
`aie::mul(v, 1.0f).to_vector<bfloat16>()` at **34 `vmul.f`/`vadd.f` per 64
elements** against **0** for `accum::from_vector`. `gelu_poly.cc` uses the
expensive form in two places, and
[`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md) called that kernel "at the
machine's fp32 vector limit" — a limit measured with avoidable emulated ops in
it. Dormant while eltwise runs on the host.

### T8 — `gelu_poly` degree 8 → 7 · **OPEN** · ~9% of that kernel
[note 0007](notes/0007-unused-iron-surface.md) §3.2: degree 8 sits at 3.6e-04
against a 2.465e-03 bf16 floor, degree 7 at 5.9e-04 — still 4.2× inside.
One Horner step of eight. Dormant with T7.

### T9 — `xrt::runlist` · **OPEN** · 0.9% today
[note 0007](notes/0007-unused-iron-surface.md) §3.3. Present in our XRT, unused
by our runtime, and worth having the moment dispatch count rises and dispatch
size falls — i.e. after T3 or T4.

### T10 — `aie::exp2` never measured · **OPEN** · small
[`0020`](../tasks/0020-m5-layernorm-kernel/TASK.md). Superseded in practice by
our own `exp2_poly` ([`0030`](../tasks/0030-m7-expert-review-tests/TASK.md)),
but the comparison was never made.

### T11 — Is the hw_context partition width settable at creation? · **OPEN** · low value
[note 0004](notes/0004-context-switch-cost.md) §1 and
[`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md). Decided in the
driver's create-hwctx call, not at XRT level. Low value now that production is
one xclbin at a fixed width.

### T12 — Larger L2 megatiles · **OPEN** · re-pointed by T1
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §3, citing
[2602.06063](https://arxiv.org/abs/2602.06063)'s 5.9 → 13.7 TOPS from megatile size alone.
Was filed as a *bandwidth* lever, which T1 has now retired. If the megatile
result is real it must be acting through **iteration count** (bigger tiles) —
i.e. it is [T17](#t17--bigger-l1-tiles-which-need-more-l1-than-a-core-has--open--the-successor-lever)
in L2 clothing, and the paper is worth re-reading with that question.

### T13 — Explain the pre-tiled instability · **OPEN** · "or stop caring"
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §4: best-case pre-tiled
runs match row-major exactly, so it is an intermittent stall, not a ceiling.
Pre-tiling was refuted as a lever, so this only matters if it is a symptom of
something else.

### T14 — Per-core: operand prep or accumulator dependency? · **ANSWERED 2026-08-19**
**Neither — the datapath choice.** Folded into T16 and answered with it
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): the plain-bf16
inner loop is near-perfectly packed (a `vmac.f` in essentially every VLIW
bundle, operand shuffles dual-issued alongside), so operand prep costs ~nothing
and the 32-MACs/cycle ceiling is the fp32 vector datapath itself. 0003's static
prediction (~28) was right all along.

### T15 — `tasks/README.md` has no rows for 0038–0043 · **OPEN** · documentation debt
Noted in [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md). Deliberately not
back-filled from `CLAUDE.md` summaries, because an index entry written from a
summary rather than from the task is the kind of second-hand claim this repo
does not keep. Needs reading the six tasks.

### T16 — Why is a k-block iteration 4.3× the arithmetic in it? · **ANSWERED 2026-08-19**
**It is not — the 145-MACs/cycle baseline was the bfp16-EMULATED datapath.**
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md): every row of 0007's
"148.9–149.9 traced" artifact carries `emulate_bfp16: true` and the 1.04e-02
error signature; M2's 137–142 are the same path, and M2's own plain-bf16 figure
was always **25.0**. The production datapath traced today at 4 columns:
**7,813-cycle k-block window of which 6,144 = exactly 768 MMAC steps × 8
`vmac.f` — the fp32 datapath at its hard 32 MACs/cycle limit** — plus 1,669
non-vector in-window and an 84-cycle gap. LOCK/STREAM_STALL ≈ 0. At the
documented 1.808 GHz that is 4.37 µs, 93% of 0048's fitted 4.72 µs (0048's
"5,900 cycles" used an implicit wrong clock). **The GEMM is compute-bound on
the fp32 datapath at ~100% code quality; only the 22% non-vector share is
amortisable by any tile-geometry lever.** Under emulation the same design is
the opposite — 39% vector busy, lock-stall gaps, DMA-bound — which is what
M2's "starved, not slow" actually described, and why bytes measured free in
0048.

### T17 — Bigger L1 tiles, which need more L1 than a core has · **OPEN** · re-priced DOWN by T16
Was "the successor lever" under the assumption that per-iteration cost is fixed.
[T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19)
measured it: **78% of the iteration is `vmac.f` compute that scales with
m·k·n**, so bigger tiles amortise only the ~1,753 non-vector cycles — **≤1.29×
with all overhead gone, ~1.06× for (64,64,64)**. The L1 wall itself is
unchanged: `(64,64,48)` costs 53,248 B of 63 KB, every legal larger geometry
overflows, and cross-tile `Buffer`
([note 0007](notes/0007-unused-iron-surface.md) §1.2) remains the one way past
it — now priced as a small lever, not the lever.

### T18 — `--probe-streams` and `--bench` disagree by up to 10% on array time · **OPEN**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md). Mean per-dispatch:
probe 3,328 µs vs bench 3,028 µs (fp32 C); 3,038 vs 2,999 (bf16 C). Both on an
idle array with the gate green. The probe dispatches back to back with no buffer
syncs; bench syncs around each dispatch. Until it is explained, **neither number
should be quoted as "the" array time** — though 0048's conclusion is a
within-run comparison and does not depend on it.

### T19 — Stationary-B single buffering, to make `k` bigger · **ANSWERED 2026-08-20: negative, measured**
**Built and traced in [`0052`](../tasks/0052-m10-research-night/TASK.md) §8:
k=96 with B single-buffered is 5.2% WORSE per MAC than the shipping
(64,64,48).** The compute scales perfectly (vector cycles exactly 1.5×, still
at the 32 MACs/cyc limit) and the overhead amortises as predicted — but
single-buffering exposes B's L2→L1 fill, and the inter-window gap grows
84 → 1,193 cycles, roughly double what the amortisation saves. The thread's
own risk note called this exactly. `--b-depth` stays in `gemm_pretiled.py`
for reuse. Only the ATB form (shrink A's M, keep everything double-buffered,
[2511.16041](https://arxiv.org/abs/2511.16041)) remains open on this front, under T17's
≤1.29× ceiling. Original pricing follows as history.
Falls straight out of [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19):
iterations go as `1/(m·k·n)`, so raising `k` cuts them directly. **`k` is already
maximal under today's budget** — everything is double-buffered, giving
`4mk + 4kn + 8mn`, and at `(64, ?, 48)`:

| k | today (all double-buffered) | Stationary-B (`B` single) | 384 % k | 1536 % k |
|---:|---:|---:|---:|---:|
| **64** | 53,248 **OK** | 47,104 OK | yes | yes |
| 88 | 64,000 OK | 55,552 OK | no | no |
| **96** | 67,584 **over** | **58,368 OK** | **yes** | **yes** |
| 128 | 81,920 over | 69,632 over | yes | yes |

So `k = 96` is **illegal today and legal under the Stationary-B budget**
`2mk·T_in + kn·T_in + 2mn·T_out` that `CLAUDE.md` trap 3 records from
ICPP '25 — the paper that says Stationary-B
beats the Stationary-C algorithm our stock IRON matmul uses.

Using [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s fit, `k` 64 → 96
took `ffn_up` from **4,198 → 3,267 µs, 1.28×** — but that assumed the
per-iteration cost is fixed.

**T16 answered the condition, and it fails**
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): 78% of the
iteration is compute that scales with `m·k·n`, so `k = 96` amortises only the
~1,753 non-vector cycles — **≈1.08×, not 1.28×**. Still positive, and it
stacks with nothing else on the ledger, but it now has to beat its own costs:
single-buffering B risks losing fetch/compute overlap (which would eat the
8% directly), and `k` is part of `gemm_b_layout`, so the `.npue` repacks.

**A cheaper mechanism for the same 1.08×, 2026-08-19**: [ATB
(2511.16041)](https://arxiv.org/abs/2511.16041) reaches the L1 relief by shrinking **A's
buffered M** (its lifetime is one C-row accumulation) instead of
single-buffering B — everything stays double-buffered, so the overlap risk
above disappears. At our (64,64,48), ρ=2 on A frees 8,192 B. If this thread is
ever built, build it ATB-shaped, not Stationary-B-shaped.

### T20 — int8 · **OPEN** · promoted by T16: a datapath lever, and those are now the only big ones
[note 0007](notes/0007-unused-iron-surface.md) §3.5. `_MM_MAC_DIMS["aie2p"]`
gives int8 **(8, 8, 8)** against bf16's **(4, 8, 8)** — twice the `r` — and
whisper-xdna measured **1.33×**, not the 2× the geometry suggests.
[T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19)
sharpened why this matters: the production GEMM runs the fp32 vector datapath
at its 32-MACs/cycle limit while the MMAC unit sits idle, so **changing
datapath is the only identified multi-× lever left on array time** (the other
route is bfp16 emulation, [T23](#t23--the-gemm-datapath-itself-bfp16-emulation-is-29-of-array-gemm-time--open--an-accuracy-decision)).
int8 would cut *cycles per iteration*, not iteration count. Accuracy is an
MTEB question, and [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) is the
precedent for how to settle it.

### T21 — One tile geometry for four shapes · **OPEN** · a tension, not yet a lever
whisper-xdna measured **2.5× and 4.6×** swings from per-shape tile sizes
([note 0007](notes/0007-unused-iron-surface.md) §3.5). Our M2/M5 finding that
per-core cost is flat to 0.4% across the four MiniLM shapes is not contradicted —
ours are all fat, theirs were not — but the one-xclbin architecture *requires* a
single geometry, and [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19) has
just made geometry the thing that matters. Unmeasured here.

### T22 — mlir-aie 1.4.1 · **ANSWERED 2026-08-20** · Python design surface migrated and independently re-verified; the one tool-side regression is now fixed and verified
[note 0007](notes/0007-unused-iron-surface.md) §3.6. We are on 1.3.4;
phoenix-sdr-dsp pins 1.4.1 and documents that the rolling wheel channel silently
resolves back to 1.3.4. Not urgent — every feature note 0007 lists is already in
our 1.3.4 — but the gap will widen.

**The upgrade happened** ([`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md)):
`C:\dev\mlir-aie` moved to `main` (mlir-aie 1.4.2.dev16+g7e00b57), which
substantially rewrote `aie.iron`'s `Runtime`/`ObjectFifo` surface (see
[note 0008](notes/0008-iron-1.4-migration.md) for the concrete before/after
patterns) and broke all 15 of our `experiments/`/`tools/` files that
constructed a `Runtime`. A first session migrated all 15 and hardware-verified
one (`exp2_probe.py`) before being interrupted; **a second session
independently re-ran the other 14 from scratch and confirmed 14 of 15 are
bit/value-exact against the historical numbers already on record** in earlier
task logs — not just spot-checked, every printed number in 0058's table was
re-derived this session, not read off the first session's claims: 335/541,662-cycle
SAXPY (1,617×, exact), 25.0→137.2 MACs/cyc bf16→bfp16 single-core and 141.7
MACs/cyc 4-column/16-core whole-array GEMM (exact against CLAUDE.md's M2
headline), all three eltwise kernels' floor/RNE error pairs (GELU
4.312e-03→2.494e-03, LayerNorm 3.326e-03→2.059e-03 with implementation error
3.659e-03→3.967e-05, softmax 4.278e-03→3.325e-03 with the exact row-sum
mechanism — min 0.994581, max 1.000000 under floor), `gemm_pretiled.py`'s
140.9 MACs/cyc rowmajor ffn_down at 0.0% spread over 3 runs (exact against
task 0007's "140.9 (0.1%)"), the four M10 pipeline/join probes (3.550e-08 /
3.474e-08 family, `pipeline_gemm_gelu_probe.py`'s real-GELU 9.052e-04 exact),
and `build_passthrough.py`'s fwd/rev order-independence — **67 of 23,304
xclbin bytes differ, all outside configuration**, the same 67-byte count task
0029 recorded (the total xclbin size differs only because the two
measurements are from different toolchain states; the invariant that matters,
67 identity-metadata bytes and zero configuration bytes, is exact). The 15th
(`experiments/m7-unified/unified_design.py`) migrated at the code level
(including solving a genuine execution-order incompatibility — the old API's
`rt._fifos` idle-endpoint-pinning hack ran too late under the new lazy-body
model and had to move into `Runtime`'s `fn_args` instead) but **cannot be run
to completion under either toolchain version** — it compiles cleanly through
30 of 37 `aiecc` stages (every Python/IRON construction step succeeds) and
then hits the pre-existing, already-documented
([`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md)) 16 KB
program-memory overflow the file's own header says it was abandoned for,
reproduced with the *identical* `XAie_LoadElf failed with XAIE_INVALID_ELF`
error. Not a migration regression; the design has never run to completion.

**A new, DIFFERENT instance of the marker-specificity fail-open class turned
up during re-verification**, inside `@iron.jit`'s own cache lookup rather
than any script's `purge()`: running `cross_column_join_probe.py`'s N=8
column case right after its N=2 case (separate processes, no in-memory
state) returned a stale N=2-compiled kernel — `RuntimeError: Tensor argument
'A' has 32768 elements but the kernel was compiled for 8192 elements` — even
though the file's own `markers_for()`/`purge()` correctly found 0 stale
candidates for the (first-time) N=8 marker. A full `rm -rf ~/.npu/cache/*`
before the retest produced the correct, previously-documented compiler error
(`"tile (0, 1) requires 8 input/1 output DMA channels, but only 4 input/4
output available"`). Fourth documented instance of this fail-open class in
this project (0030, 0053, 0054 are the other three) — this one is upstream
of any of this project's own `purge()` implementations, so no script-level
fix is possible; the working mitigation is to purge fully rather than trust
a script's scoped marker match when switching configurations within one
session.

**FIXED and verified** ([`0060`](../tasks/0060-m11-export-gemm-rtp-marker-fix/TASK.md)):
`tools/export_gemm_rtp.py`'s `markers_for()` cache-marker string match was
broken by the unrelated MLIR pretty-printer format change described above —
the sequence-body `aie.dma_bd` op's textual form changed from bracket-tuples
(`<size = 64, stride = 48>`, still used for `ObjectFifo` `dimensionsToStream`
attributes) to a flat `sizes = [...] strides = [...]` array — so the
substring `f"<size = {k}, stride = {n}>"` the marker relied on no longer
appeared anywhere in freshly-built `aie.mlir`. Confirmed by building all four
production shapes and grepping their `aie.mlir`: B's (`%arg1`) `aie.dma_bd`
always ends its access pattern with the tile dims as the last two entries of
`sizes`, immediately before `strides` (`sizes = [.., .., 64, 48] strides =
[..]`), exactly twice per build (the ping/pong pair) and nowhere else in the
file. `markers_for()`'s second marker is now `f"{k}, {n}] strides = ["` — the
direct translation of the old marker into the new textual form, same two
numbers, same adjacency requirement, so it stays as specific as the six prior
marker-fail-open fixes demanded rather than merely permissive.
**Verified on a fully purged cache**: the small case (`--batch 4 --cols 2`)
now finds exactly 1 candidate per shape and completes with all identity
checks `OK`; the real production shape (`--batch 128 --cols 8`, the `0032`
recipe) rebuilds into a separate directory
(`runtime/artifacts_verify_t22fix/`, gitignored) and, combined with the
shipped weights/manifest/eltwise designs, **reproduces `0038`'s and `0059`'s
exact historical numbers on hardware**: golden-vs-`.npue` `1-cos` **1.086e-05**
(identical) and `verify_embed_e2e.py` worst `1-cos` **2.644e-05** (matches to
4 s.f.), both `PASS`. `runtime/artifacts_b128il/` itself was never modified.
`gemm_pretiled.py` (which `export_gemm_rtp.py` imports `pretiled_array`
from) was already confirmed correctly migrated in `0058` — the break, and
now the fix, is entirely in `export_gemm_rtp.py`'s own string-matching, not
in the Runtime migration.

### T24 — 0048's iteration fit missed bge-base by 27% · **ANSWERED 2026-08-20**
**The fit was fine; the miss was the host + overlap term the prediction
ignored.** [`0052`](../tasks/0052-m10-research-night/TASK.md) §1 ran the one
command: `--probe-streams` on the h=768 design. The discriminating pair
reproduces (identical-MAC shapes 0.3% apart across 1.27× bytes), marginal
per-iteration is **4.49 µs against the h=384 fit's 4.72 (−5%)**, and the
array account closes exactly: predicted 1,053 ms of array time for two
pipelined lanes against a measured 1,047 ms NPU share. The wrong move in
0051 was predicting *end-to-end* from an *array-only* model. The array fit
is now validated at both widths; what has no model is the host side.

### T26 — Why is bfp16 + bf16-C 6.6× MORE accurate than bfp16 + fp32-C? · **OPEN** · mechanism CLASS confirmed (compounds across chained GEMMs), numerical root cause still open
[`0052`](../tasks/0052-m10-research-night/TASK.md) §6. Adding a rounding
(bf16 C transport) to the emulated datapath should cost accuracy; it measured
1-cos 2.395e-03 → **3.615e-04**, confirmed independently by e2e text
(9.7e-04) and MTEB (worst task −0.33 → −0.06). The two workers differ only in
where the fp32 accumulator lives: the C fifo object (fp32-C path) versus a
core-local `Buffer` + `narrow_f32_bf16` (bf16-C path).

**PROBED 2026-08-20** ([`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md)):
the named hypothesis (the fp32-C fifo path re-quantises C partials at every
k-block boundary — six times at K=384) is **REFUTED on two independent
grounds**. (1) The matmul kernel object is **byte-for-byte identical**
between the fp32-C and bf16-C builds (`llvm-objdump` diff, one shared
`matmul_bf16_f32_<hash>.o` in both cache dirs) — there is only one matmul
kernel, used identically regardless of accumulator location, so no per-block
conversion exists to be the mechanism. The per-core wrapper differs only in
the accumulator's address and (bf16-C only) one `narrow_f32_bf16` call
*after* the k-loop, using `conv_even` rounding — nothing changes *inside*
the loop. (2) Numerically, a single-GEMM probe (M=256,N=192, 4 cols) run
"full" (one dispatch, K/64 k-blocks in one loop) vs "split" (K/64 SEPARATE
one-k-block dispatches, host-summed) shows the fp32-C full/split ratio is
**flat at 1.000× at BOTH K=384 (6 blocks) and K=1536 (24 blocks)** — if
boundary-crossing degraded it, the ratio should grow with block count; it
does not move at all.

**The anomaly itself also does not reproduce on isolated synthetic data**:
at full K, bf16-C is statistically tied with fp32-C (a hair *worse*) at both
K=384 (1.480e-02 vs 1.472e-02) and K=1536 (1.539e-02 vs 1.530e-02) — nothing
like production's 6.6×. A smaller, real, reproducible effect exists only in
"split" mode (single k-block, no boundary by construction): bf16-C beats
fp32-C by 27–36%, growing with block count (6→24) but present even at N=1
block, so it cannot be a boundary-count effect either.

**PROBED FURTHER 2026-08-20** ([`0056`](../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md)):
the rounding-mode-asymmetry hypothesis above is **REFUTED BY THE CODE**, and
a REFINED structural hypothesis is **CONFIRMED as a genuine compounding
mechanism** on a controlled chained probe.

1. **Refuted**: `runtime/src/main.cpp`'s `to_bf16()` (line 403) — the
   function used at the ONE point where an activation is narrowed to bf16 to
   feed the *next* GEMM's input, for BOTH the fp32-C and bf16-C paths alike,
   at every layer boundary — is explicitly round-to-nearest-even
   (`(u + 0x7FFF + ((u >> 16) & 1)) >> 16`), mirrored bit-for-bit by
   `tools/npue.py`'s `to_bf16_bits` used when packing weights offline. AIE's
   default `floor` (trap 2b) does not appear anywhere in this narrowing
   chain. There is no floor-vs-`conv_even` asymmetry in production.
2. **What the reading DID surface**: a real structural difference — the
   bf16-C path narrows its raw fp32 accumulator ONCE, EARLY (on-core,
   `conv_even`, before bias-add), *in addition to* the same late host RNE
   narrow both paths share right before the next GEMM; the fp32-C path keeps
   full fp32 precision through bias-add (and any host eltwise op) and only
   narrows once, late.
3. **Tested this refined hypothesis on a chained multi-GEMM probe**
   (`t26_chain_probe.py`, one constant shape reused for every stage so only
   2 device builds are needed for any chain length): tied at 1 stage
   (`1-cos` ratio 0.99, reproducing 0053's isolated-GEMM near-tie exactly),
   then **diverging monotonically** — 1.35×, 1.46×, **1.67× by stage 4**
   (`rel_fro`: 1.00×, 1.16×, 1.21×, 1.29×). The gap genuinely compounds with
   chain length, and extrapolating the growth rate to a 24-GEMM chain lands
   in the right order of magnitude for production's 6.6×.
4. **Still open**: the numerical *why* — does an extra early RNE-family
   narrowing reduce the growth of bfp16-emulation quantisation error because
   independent per-block noise partially cancels when narrowed-and-summed
   rather than accumulated-then-narrowed-once? 0053's own split-mode data is
   suggestive (bf16-C's split error *decreases* with more blocks: 1.073e-2 at
   6 blocks → 9.878e-3 at 24), but this was not isolated from the
   chain-compounding effect measured here — they may be the same phenomenon
   or two different ones. Also still open, carried from 0053: whether real
   weight/activation distributions (vs synthetic Gaussian, used in both the
   0053 and 0056 probes) matter — 0008 already showed real-vs-uniform data
   changes plain-bfp16 error by 0.85×, not the 6× a simulated prediction
   assumed, so distribution-dependence is an established mechanism class
   here even though it isn't confirmed as *this* mechanism. **Not
   examined**: whether today's shipping plain-bf16 (non-emulated) path
   carries any analogous, smaller effect — an explicit non-goal of 0056,
   left for a future decision.

### T27 — The emulated datapath is traffic-bound: which byte-lever pays first? · **OPEN** · conditional on T23
[`0052`](../tasks/0052-m10-research-night/TASK.md) §3. On bfp16 the array is
DMA-bound (GB/s 32–46 flat-ish, GMAC/ms spread; ffn_up slower than ffn_down
by its byte ratio) — the inverse of the plain-bf16 economy 0048 measured. If
T23 reopens, the retired levers re-price on the new path: **B-reuse** (still
channel-blocked per 0046/0047 — cascade or CascadeFlow was the route),
**ATB/bigger tiles** (now amortising DMA, not overhead), **A in bf16→bfp16
offline** (halves A bytes?), and the host readback (bf16-C already in). Also
note the host wall: at pipeline 4 the bfp16 encode is 47% NPU busy, so array
levers beyond 2.2× buy little until T3-class host work lands.

### T25 — bge-base has no MTEB gate and no interleaved CPU ratio · **ANSWERED 2026-08-20**
[`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md) validated it to
`1-cos` 1.353e-05 and end-to-end 2.613e-05 with top-10 overlap 1.0000, which is
the correctness bar. It lacked the two things
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md) and
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) established as the bars
for *quality* and for *speed claims*: five MTEB tasks against the CPU on the
same checkpoint, and a round-robin interleaved throughput ratio.

**Both landed in [`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md).**
MTEB, five tasks, seq 64, both NPU datapaths against the same CPU column:

| task | CPU | NPU plain-bf16 | Δ plain | NPU bfp16+bf16C | Δ bfp16 |
|---|---:|---:|---:|---:|---:|
| STSBenchmark | 86.418 | 86.422 | +0.004 | 86.407 | −0.011 |
| SICK-R | 80.301 | 80.299 | −0.002 | 80.286 | −0.016 |
| STS12 | 78.028 | 78.027 | −0.002 | 77.984 | −0.045 |
| Banking77Classification | 83.984 | 83.981 | −0.003 | 83.925 | −0.058 |
| TwentyNewsgroupsClustering | 50.576 | 50.695 | +0.119 | 50.383 | −0.193 |
| **mean / worst** | | | **+0.023 / −0.003** | | **−0.065 / −0.193** |

Both **PASS** (gate: \|mean\| ≤ 0.5 AND worst ≥ −0.5) — bge-base is not more
fragile than MiniLM under either datapath; bfp16+bf16C's bge-base worst
(−0.193) sits well inside MiniLM's own worst (−0.06 per 0052 §7). Artifacts:
`experiments/m8-npu-vs-cpu/artifacts/mteb_bge_base.json` (plain, self-computed
gate), `mteb_bge_base_bfp_cbf16.json` + `mteb_bge_base_bfp_cbf16_gate.json`
(bfp16 NPU-only run, gate computed by merging against the plain run's already-
recorded CPU column rather than re-running CPU a second time — `run_mteb.py
--sides npu` does not self-compute a delta, since its gate logic reads
`results["cpu"]`, which is absent from a NPU-only run; noted as a real, minor
gap in the harness, not fixed this session).

**Interleaved CPU ratio (0040 protocol, mains power, `compare_three.py
--rounds 8`, artifacts_base / plain bf16 only)**: torch 111.2 seq/s (steady,
strongest CPU), ORT 55.4 seq/s, NPU 181.5 seq/s → **NPU / strongest CPU =
1.633×**. Machine state recorded: `Online / NoSystemBattery / Balanced`.
Artifact: `experiments/m8-npu-vs-cpu/artifacts/compare_bge_base.json`. (The
script crashed *after* writing the artifact, in its final cosmetic
`print(f"wrote {out.relative_to(REPO)}")` — `out` is a relative `Path`,
`REPO` absolute, so `relative_to` raises `ValueError` on Windows; a real small
bug in `compare_three.py`, not investigated further since the data was
already on disk.) The bfp16+bf16C interleaved ratio was **not** measured this
session (only its MTEB gate) — left for a follow-up night alongside T27.

### T23 — The GEMM datapath itself: bfp16 emulation is 2.9× of array GEMM time · **OPEN** · an accuracy decision, not ours to make
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md). On the same design,
same shape, the emulated path costs 2,684 cycles per k-block against plain
bf16's 7,897 — **2.9× of array GEMM time**, worth roughly **1.35× end to end
on MiniLM (39.4% wait) and 1.66× on bge-large (60.8% wait)**. `--emulate-bfp16`
was retired when it was worth +2.2% end to end
([`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)) and failed the 1-cos gate at
3.470e-03; **the pricing half of that retirement is superseded, the accuracy
half stands.** T16 also showed the emulated design is DMA-bound (39% vector
busy, lock-stall gaps), so under emulation the retired byte-levers (B-reuse,
T2) would partially revive. Reopening this means running MTEB on the bfp16
path per [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) — flagged for the user;
per the 0045 precedent, datapath accuracy decisions are made by the user on
MTEB evidence, not unilaterally. [T20](#t20--int8--open--promoted-by-t16-a-datapath-lever-and-those-are-now-the-only-big-ones)
(int8) is the same decision shape with different numbers.

**Upside re-priced 2026-08-19 by [ATB](https://arxiv.org/abs/2511.16041)** (UCLA+AMD, our
SKU, our toolchain, web-indexed): with asymmetric tile buffering and a
hand-optimised microkernel, BFP16 GEMM reaches **24.3 TFLOPS** on this silicon
— ~8× our production array rate, of which 2.88× is microkernel work alone
(stock 0.32 → 0.92 TFLOPS/core). The 2.9× above is what the *stock* emulated
path buys; it is the floor of this decision, not the ceiling.

**MEASURED 2026-08-20 ([`0052`](../tasks/0052-m10-research-night/TASK.md)) —
every gate now passes, on the bfp16+bf16C combination:**

| | bfp16, fp32 C | **bfp16 + bf16 C** |
|---|---:|---:|
| `1-cos` vs HF (validation) | 2.395e-03 FAIL | **3.615e-04 PASS** |
| e2e real text, worst `1-cos` | — | **9.701e-04 PASS** |
| top-10 neighbour overlap | — | **0.9923 PASS** |
| MTEB, 5 tasks, mean Δ vs CPU | +0.01 (worst −0.33) | **+0.16 (worst −0.06)** |
| array GEMM vs plain bf16 | 1.74× | **2.20×** |
| e2e at pipeline 4 (MiniLM) | 982.1 seq/s | **1,046.2 seq/s (+8.7%)** |

Two big caveats travel with this. (1) **The e2e gain is only +9% because the
encode is host-walled at 47% NPU busy** — the datapath decision pays its
1.35–1.66× only together with host-side work (T3, readback). (2) **The
emulated array is TRAFFIC-bound** (GB/s 32–46, ffn_up slower than ffn_down in
proportion to bytes), so the byte-levers T1 retired come back to life on this
path: B-reuse, ATB, and bf16-C (already in). Decision remains the user's; the
mechanism of the accuracy jump is T26 (below), which as of 2026-08-20 has
moved from "unknown mechanism" through "original hypothesis refuted,
isolated-GEMM effect only 27–36% against production's 6.6×" to "the
rounding-mode-asymmetry hypothesis (`conv_even` vs `floor`) is ALSO refuted
by reading `runtime/src/main.cpp` (both paths' downstream narrowing is
round-to-nearest-even), but a refined structural hypothesis — bf16-C narrows
early, before bias, in addition to a late narrow both paths share — is
CONFIRMED to compound on a chained probe: tied at 1 stage, 1.67× by 4 stages
([`0056`](../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md))." The
mechanism *class* (compounding across chained narrowing) is now evidenced;
the numerical root cause within that class is not yet nailed down.

**bge-base MTEB landed 2026-08-20 ([`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md),
T25 above): mean −0.065, worst −0.193, still comfortably PASS** — so "MiniLM is measured,
bge-base/large are not" is now half-retired: bge-base's bfp16+bf16C accuracy
is measured and passes, on a **second, independent model geometry** (h=768 vs
384, N sets that force `tile_n=48` the same way, but a different layer count
and a completely different real-weight distribution). **Still not measured**:
bge-base's bfp16+bf16C *throughput* (only the plain-bf16 interleaved ratio —
1.633× — was run this session; T27's traffic-bound question and bge-large
remain open). This strengthens rather than settles the decision: two models
now clear the accuracy bar, but the speed case for reopening bfp16 is still
argued from MiniLM's host-walled +9% alone.

---

## Closed

| thread | status | where |
|---|---|---|
| Is `aie::vector<float>` really IEEE fp32? | **ANSWERED** — yes, ~24 mantissa bits | [`0016`](../tasks/0016-m5-fp32-probe/TASK.md), refuting [`0015`](../tasks/0015-m5-gelu-polynomial/TASK.md) |
| What carries GELU's 3.886e-03 implementation error, if not fp32 precision? | **ANSWERED** — the default `floor` rounding mode | [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3, chasing [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)'s own hypothesis after 28 tasks |
| Can B reuse be expressed with `consumer_obj_type`? | **ANSWERED** — no; no spare DMA channel exists | [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) |
| Does cascade free the channels B-reuse needs? | **ANSWERED** — frees inputs 6/6→3/6, costs outputs 3/6→6/6 | [`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) |
| LayerNorm still opens three fifos per core | **ANSWERED** — params broadcast from the mem tile, 8 columns | [`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) |
| M6 speed not measured | **ANSWERED** — deliberately deferred to M7, then measured | [`0023`](../tasks/0023-m7-full-cpp-encode/TASK.md) onward |
| Is bge-small a byte-identical drop-in? | **ANSWERED** — no; 12 layers and CLS pooling are data, not constants | [`0039`](../tasks/0039-m9-bge-small/TASK.md) |
| `pack_npue.py` had not run in months | **ANSWERED** — broken import found and fixed | [`0036`](../tasks/0036-m8-tokenizer/TASK.md) |
| Is the centred polynomial basis worth 2.5×? | **RETIRED** — measured, worth nothing at fp32 | [note 0007](notes/0007-unused-iron-surface.md) §3.2 |
| `AIE_LOOP_UNROLL_FULL` | **RETIRED** — 14% slower on straight vector loops | [note 0007](notes/0007-unused-iron-surface.md) §1.9 |
| `burst_length` tuning | **RETIRED** — already maximal by default | [note 0007](notes/0007-unused-iron-surface.md) §2 |
| MTEB on the bf16-C datapath | **RETIRED** — datapath decided as bf16 in / fp32 out, 2026-08-19 | [`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) |
| `--emulate-bfp16` | **RETIRED** — fails accuracy; closed by the MTEB gate | [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) |
| Pre-tiling as a performance lever | **RETIRED** — a wash under isolation | [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md), [`0008`](../tasks/0008-m5-bfp16-real-data/TASK.md) |
