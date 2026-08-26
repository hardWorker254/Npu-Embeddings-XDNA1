# 0100 — docs catch-up: CLAUDE.md and CURRENT_STATUS.md against 0088–0098

- **Date** 2026-08-23
- **Milestone** documentation (post-M13)
- **Status** done

## Goal

Today's session closed eleven threads and revised four
([`0088`](../0088-references-external/TASK.md)–[`0098`](../0098-t26-kernel-source/TASK.md)).
`CLAUDE.md` and `docs/CURRENT_STATUS.md` had not caught up: `0089` explicitly
found two contradictions between `CLAUDE.md` and the task logs it cites and
left them standing for someone to fix properly, and several other tasks
produced durable facts (traps, a hardware constraint, an unconfirmed
hypothesis) that belonged in one of the two files or nowhere. This task reads
the source tasks and commits directly (never `0089`'s summary alone, never
this session's own prose) and corrects both files in place, per `CLAUDE.md`'s
own closing instruction: no dated block, fix what is wrong, keep the voice.

**Constraint carried over from the brief:** documentation only, no NPU access
(the array was in use by another agent this session). `research/OPEN-THREADS.md`,
`research/CLOSED-THREADS.md` and `tasks/README.md` are owned by the
coordinator and were not touched.

**Numbering note:** `tasks/0099` was not found on disk when this task was
created (checked with `Glob tasks/0099*` — no match) and this task was
assigned `0100` directly by the coordinator. Recorded rather than
investigated further, in the spirit of the `0072` gap precedent in
`tasks/README.md` — a numbering gap costs nothing and is not this task's to
resolve.

## Context — what was read, and against what claim

Every correction below is checked against the task log(s) named, read in
full, not against `0089`'s paraphrase of them:

- [`0079`](../0079-m13-int8-why-only-1.1x/TASK.md) — the original "1.10×,
  quantisation costs more than conversion" finding, and its own §2/§3
  measurement refuting the second half of that claim (1.04×, and the real
  gap is the C drain).
- [`0080`](../0080-m13-int8-traffic-bound/TASK.md),
  [`0081`](../0081-m13-int8-everywhere/TASK.md),
  [`0082`](../0082-m13-fused-ffn-epilogue/TASK.md) — the host-side fusion
  work and the `tile_n=64` finding for bge-large that closed most of the
  gap 0079 measured.
- [`0085`](../0085-m13-release-sweep/TASK.md) — the twelve-row 0.4.0 sweep,
  the newest and authoritative throughput numbers (1.71×–2.39×).
- [`0089`](../0089-t15-task-index-backfill/TASK.md) — Problem #3, which names
  the two CLAUDE.md contradictions this task fixes, and states explicitly
  that fixing them was out of scope for 0089 itself.
- [`0090`](../0090-t5-dma-compression/TASK.md) — the shim-DMA-has-no-compression
  finding (T5) and the `iron.jit` module-globals cache bug found along the
  way.
- [`0092`](../0092-t28-relay-bf16-output/TASK.md) — Parts 1 and 4: the
  `arg_types`-shapes-only-the-call-site finding, and the hardcoded `Y_ty`
  host-argument-dtype bug, both read for their exact mechanism (not just the
  headline) before writing a CLAUDE.md trap.
- [`0094`](../0094-t32-golden-gate-rows/TASK.md) — the golden-gate
  truncation bug (4 of N rows) and T32's aliasing bug (indistinguishable
  tile copies), including the "neither fix alone suffices" demonstration
  that had to survive into the docs.
- [`0095`](../0095-t6-t10-probes/TASK.md) — confirmed the `docs/CURRENT_STATUS.md`
  §3 "unresolved bug" correction (0030 diagnosed it) was already applied and
  reads correctly; not re-touched.
- [`0097`](../0097-t18-t21-t4-measurements/TASK.md) — T18/T21/T4 closures,
  used to write the current §9 priority list and the T38 (padding) entry.
- [`0098`](../0098-t26-kernel-source/TASK.md) — the rounding-mode
  dead-code-elimination hypothesis for T26, explicitly unconfirmed on
  hardware; written into CLAUDE.md as a hypothesis with that status stated,
  not as fact.
- `research/OPEN-THREADS.md` — read (not edited) to get the current list of
  9 live threads (`T3, T9, T13, T17, T23, T26, T28, T34, T38`) for §9.
- `git log --oneline` from `4f7740e` back through the session's start, to
  confirm the shape of today's work before trusting any task's own framing.

## What was done

### `CLAUDE.md`

1. **Rewrote the int8 headline** (`## Current state`). The old text claimed
   "1.10× on MiniLM, not 7×" as the current number and "the int8
   quantisation pass costs more than the bf16 conversion it replaces" as the
   explanation. Both are stale: 1.10× was 0079's *first* measurement and
   already the catalogue's worst case at the time (bge-large was 1.44× in
   the same task); 0085's twelve-row sweep is the current, authoritative
   number (1.71×–2.39×). And 0079 **itself** measured and refuted the
   quantisation-cost claim (1.04×, not "more") — the real explanation is the
   C drain (int32 C is still 4 bytes, so int8 saves no output-transport
   bytes despite 7× faster MACs). Rewrote to state the refuted claim, what
   refuted it, the real mechanism, and the current number, in that order —
   rule 3b.
2. **Rewrote the geometry table's `tile_n` claim** (`## What geometry the
   array actually wants`). The old text said `tile_n` 32 for bge-large,
   unconditionally. `0081` moved bge-large's **int8** design to `tile_n=64`
   — legal only because `in` (bytes/element) halves in the L1 budget
   inequality for int8, and only because bge-large's N values divide 64·8.
   Rewrote to make the dtype-dependence explicit: 32 for bf16, 64 for int8,
   with the exact budget numbers (65,536 B vs 49,152 B against the 63 KB
   ceiling) so the claim is checkable, not just asserted. Also fixed the
   model table row for `bge-large-en-v1.5`, which repeated the stale
   unconditional "N=1024 forces `tile_n` 32."
3. **Added trap 7d** — `iron.jit`'s cache key never inspects a generator's
   own module globals (0090 §4, confirmed independently in 0092 Part 4's
   addendum). Sixth instance of the project's "stale binary fails open"
   class, first one found inside IRON itself.
4. **Added trap 8** — IRON's declared argument types are cosmetic at two
   separate boundaries that nothing checks: `arg_types` shapes only the MLIR
   call-site declaration, never the linked object (0092 Part 1, an
   objdump-confirmed 2×buffer overflow); and a design's host-facing argument
   dtype must independently track its internal pipeline's dtype, or the
   shim DMA is sized against the wrong byte count and hangs with **no
   diagnostic** (0092 Part 4, the `Y_ty` bug that caused T28's whole
   production-width hang).
5. **Extended trap 3b** with two short, hard facts: the shim DMA has no
   compression or padding hardware at all (only mem-tile and compute-tile
   do — confirmed directly in `xaie2pgbl_reginit.c`'s register tables, 0090
   §3), and `aie::accum<accfloat,N>` on aie2p is a plain 32-bit-per-lane
   accumulator, not extended precision (0098 item 9).
6. **Added an addendum to trap 2b**, clearly marked as an unconfirmed
   hypothesis rather than fact: 0098's finding that the emulated bf16
   matmul's own rounding-mode fix-up appears to be dead code in the
   compiled object, which would make T26's still-open 6.6× accuracy anomaly
   a rounding-state leak between kernels sharing a core. Stated with its
   evidence and its status ("NOT YET CONFIRMED ON HARDWARE") in the same
   sentence, per the brief's explicit instruction — if it could not be
   marked clearly as a hypothesis it was to be left out entirely and left to
   the register.

**Not added to CLAUDE.md, and why:**
- **0088's external-repo survey** (references-external.md) — a research
  artifact, not a ground rule or a trap; belongs where it already lives
  (`research/references-external.md`), not duplicated here.
- **0091, 0093, 0096** (GELU-poly reading, T11/T12/T13 research, the T26
  numerical model that failed its own control) — skimmed for anything
  CLAUDE.md-shaped; none produced a new trap or a ground-rule change. 0096's
  negative result (the host model does not reproduce the hardware control)
  is exactly the kind of finding that belongs in the register/task log, not
  in CLAUDE.md, since CLAUDE.md states current, not-yet-superseded facts and
  0096's own conclusion is "this doesn't explain it, keep looking" — 0098
  picked up from there and is the version that made it into CLAUDE.md.
- **0097's T18/T21 findings** (probe-vs-bench gap did not reproduce;
  per-shape tile geometry worth ~4% on narrow models) — measurement-doctrine
  and optimisation-backlog material respectively, not a ground rule; T21's
  finding is exactly the shape of thing the register should carry, and it
  already does (T21, now closed). Used only to write `docs/CURRENT_STATUS.md`
  §9's current priority list, not added to CLAUDE.md itself.

### `docs/CURRENT_STATUS.md`

1. **Updated the header pointer** from "after tasks/0082" to point at this
   catch-up and name the three tasks its corrections are checked against
   (0079, 0085, 0094).
2. **§4 "Four bugs that failed open" → "Five bugs that failed open."** Added
   bug #5 from `0094`'s own proposed text (lightly trimmed to match the
   section's existing density), covering both the row-truncation bug and
   the tile-aliasing bug it found alongside it, and the "neither fix alone
   is sufficient" demonstration. Changed the closing line from "All four are
   now fail-closed..." to "All five are now fail-closed...".
3. **Golden-gate description in §1** — the printed `worst 1-cos` line and a
   new paragraph directly under it explain that the gate now compares every
   row (not 4), and that comparing every row alone would **not** have caught
   the aliasing bug — the rotation is what makes it catchable — matching the
   brief's instruction that this distinction (all-rows-alone insufficient)
   survive into the docs, not just "we now check all rows."
4. **Checked §3's "unresolved bug" entry** — already corrected in place by
   [`0095`](../0095-t6-t10-probes/TASK.md) (the paragraph now says "0030
   diagnosed and fixed it," with the worker-stack mechanism). Read it in
   full; it is accurate and was not re-touched, per the brief's instruction
   not to redo already-correct work.
5. **Rewrote §9 "Next steps, in priority order."** The old list was M7/M8-era
   (a tokenizer, MTEB, energy — all long done) and referenced tasks
   0029–0030. Replaced with a pointer to `research/OPEN-THREADS.md` as the
   authority (rule 3) plus this file's own priority ordering of the current
   9 live threads (T28/T3, T38, T26, T23, T17, T13, T9, T34), each with a
   one-line status pulled from the thread text and, where a task closed
   part of it this session, that task. Explicitly notes the 8 threads this
   session closed (T4, T5, T6, T10, T15, T18, T21, T32) live in
   `CLOSED-THREADS.md` now, not here.

**Not touched, per the brief's explicit instruction:** the `THE 0.4.0
NUMBERS` table and the other measured-performance tables in
`docs/CURRENT_STATUS.md`. Nothing read this session re-measured them, and
rule 1/6 mean a documentation task does not get to edit a number it did not
re-derive on hardware.

## Commands

None — this is a read-and-edit documentation task, consistent with the
"do not run anything on the NPU" constraint. Every file read is named in
Context above; every edit is in `CLAUDE.md` and `docs/CURRENT_STATUS.md`
(see `git diff` for the exact changes).

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
git log --oneline 4f7740e~30..4f7740e   # confirmed the session's shape before trusting any task's framing
git diff CLAUDE.md docs/CURRENT_STATUS.md   # the actual changes made by this task
```

## Result

`CLAUDE.md`: two stale claims corrected (int8 headline, bge-large `tile_n`),
four new trap entries added (7d, 8, plus additions to 3b and 2b), one
unconfirmed hypothesis added with its status stated explicitly.
`docs/CURRENT_STATUS.md`: header pointer updated, §4 gained a fifth
fail-open bug, the golden-gate description now states the two-part fix and
why one part alone is insufficient, §9 replaced with a current, register-derived
priority list. No measured-performance table was touched.

## Problems hit

- **None that blocked the work.** The main risk was writing from `0089`'s
  summary of the contradictions rather than the underlying tasks (0079,
  0085) themselves — avoided by reading 0079 and 0085 in full before writing
  either CLAUDE.md paragraph, which surfaced detail (the C-drain mechanism,
  the exact 1.04× figure) that `0089`'s one-paragraph description did not
  carry.
- **Numbering**: `tasks/0099` does not exist on disk; not investigated
  further per the note under Goal above.

## Artifacts

- `CLAUDE.md` — edited in place (see Result above for the six changes).
- `docs/CURRENT_STATUS.md` — edited in place (see Result above for the five
  changes).
- This file.

## Next

The two contradictions `0089` flagged are now fixed; nothing else in this
session's task range (0088–0098) was found to need a CLAUDE.md or
`docs/CURRENT_STATUS.md` change beyond what is listed above. The 9 live
threads named in §9 are the register's own list and this task does not
change it — `research/OPEN-THREADS.md` remains the authority per rule 3.

## Proposed register update

*(`tasks/README.md` is owned by the coordinator this session; index row
prepared for that agent to paste.)*

```markdown
| [0100](0100-docs-catch-up/TASK.md) | **CLAUDE.md and docs/CURRENT_STATUS.md caught up to 0088–0098.** Fixed the two contradictions `0089` found and left standing: the int8 headline (1.10×/"quantisation costs more than conversion" → 1.71×–2.39× per the 0085 sweep, and 0079's own §2/§3 refuting the cost claim — the real gap was the C drain), and the geometry table's `tile_n` claim for bge-large (32 for bf16, 64 for int8, per 0081, made dtype-explicit rather than picking one number). Added four CLAUDE.md traps from today's session: `iron.jit`'s cache blind to a generator's module globals (0090, sixth instance of the stale-binary class), IRON's `arg_types`/host-argument-dtype boundaries being unchecked in two separate places (0092 Parts 1 and 4, one of them a hang with no diagnostic), the shim DMA having no compression or padding hardware at all (0090/0097), and `accfloat` being a plain 32-bit accumulator (0098) — plus one clearly-marked, NOT-hardware-confirmed hypothesis for T26's rounding-mode mystery (0098). In `docs/CURRENT_STATUS.md`: a fifth "failed open" bug (0094's golden-gate row-truncation and tile-aliasing fixes, including the "neither fix alone is sufficient" finding), and §9's stale M7-era next-steps list replaced with the register's current 9 live threads, prioritised. Did not touch any measured-performance table. | documentation | done |
```
