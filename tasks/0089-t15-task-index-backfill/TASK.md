# 0089 — T15: `tasks/README.md` index backfill for 0038–0043, and the 15 more that had also fallen out

- **Date** 2026-08-23
- **Milestone** none (documentation debt, not a build)
- **Status** done

## Goal

Close [`T15`](../../research/OPEN-THREADS.md#t15) (now moved to
[`CLOSED-THREADS.md`](../../research/CLOSED-THREADS.md#t15)): `tasks/README.md`'s
index table had no rows for tasks 0038–0043, and per rule 3 those rows must be
written **from the task logs themselves**, not backfilled from `CLAUDE.md`'s
*Current state* summary — that is the entire reason T15 was filed instead of
just being fixed on the spot in [`0044`](../0044-m9-optimisation-sweep/TASK.md).

## Context

T15's own text scoped the debt at six tasks (0038–0043). Before reading
anything, `ls tasks/` was compared against the index table in
`tasks/README.md`, and the debt turned out to be **21 rows, not six**:
0038–0043 (T15's own six) plus **0073–0087**, fifteen further tasks that had
never been indexed either, which T15 never mentioned because it was filed
against 0044's narrower observation and nobody re-checked the gap as the
project grew past task 0071. `tasks/0072` is correctly not one of them —
`tasks/README.md` already carries a note explaining that deliberate gap
(a merged M13 sub-task and a live benchmark run at the time the gap was
noticed), and this task leaves that note untouched.

## What was done

1. Read `tasks/README.md` in full to learn the table's exact format (four
   columns: `#` / `Task` / `Milestone` / `Status`) and the project's house
   style for a row: a bolded headline phrase, real numbers cited from the
   task's own `Result` section, named bugs/traps, and explicit
   "corrects/is corrected by" cross-references where a task revises another.
2. `ls tasks/` against the table's existing entries (0001–0071, minus the
   already-documented 0072 gap) found the 21 missing numbers above.
3. Read **every one of the 21 `TASK.md` files in full** (not summaries) —
   `Goal`, `Context`, `What was done`, `Result`, `Problems hit`, `Next` —
   split across three parallel reading passes (0038–0043, 0073–0079,
   0080–0087) so each task got a close, single-purpose read rather than a
   skim across all 21 at once. Each pass cross-checked its own tasks against
   both `CLAUDE.md`'s *Current state* and `docs/CURRENT_STATUS.md`, flagging
   — not silently reconciling — anything the task log said that the summary
   documents said differently.
4. Wrote one index row per task, in numeric order, matching the table's
   existing column format and density. Two formatting corrections applied to
   the drafted text before it went in: HTML-entity-escaped angle brackets
   (`&lt;repo&gt;`) were replaced with plain `<repo>` inside backticks, since
   CommonMark code spans already escape angle brackets safely and the manual
   entity would have rendered as literal `&lt;repo&gt;` text instead of
   `<repo>`; and two overly long `Status` cells (0043, 0079) were trimmed to
   single words, moving the explanatory detail (missing header fields, an
   unfilled `Results` section) into the `Task` cell, matching how the
   existing table handles similar cases (e.g. row 0003's compound status is
   the exception, not the rule).
5. Removed the stale "Index gap" note (the paragraph that used to sit between
   rows 0044 and 0045 explaining why 0038–0043 were missing) — it is no
   longer accurate once the rows exist — and inserted the six rows in its
   place, and the fifteen 0073–0087 rows after the existing last row (0071).
6. Moved T15 out of `OPEN-THREADS.md`'s live section into
   `CLOSED-THREADS.md`, verbatim, following the convention already used by
   T30/T31 (the heading's own status marker changes from `**OPEN**` to
   `**ANSWERED 2026-08-23**`; the original body text is kept unedited; a new
   paragraph is appended below it recording the resolution and pointing at
   this task). Added a bullet for T15 to the "Closed threads" pointer list at
   the bottom of `OPEN-THREADS.md`, matching the existing bullet format
   (`* [Tn](CLOSED-THREADS.md#tn) — <title> · **STATUS** ...`).
7. Grepped the whole repo for `#t15` and `\bT15\b` before and after the
   move — the only occurrence was the heading itself (no external file linked
   to `#t15`), so no other link needed fixing. The `<a id="t15"></a>` anchor
   now lives in `CLOSED-THREADS.md` and the new `OPEN-THREADS.md` bullet
   points at it correctly (`CLOSED-THREADS.md#t15`).
8. Ran `tools/check_register.py` before and after every register edit.

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
ls tasks | sort                         # confirmed 0001-0087 minus 0072, 86 dirs
python tools\check_register.py          # baseline: register OK (before any edit)
# ... edits to tasks/README.md, research/OPEN-THREADS.md, research/CLOSED-THREADS.md ...
python tools\check_register.py          # final: register OK
```

## Result

`tasks/README.md`'s index now has a row for every task directory that exists
(0001–0087, minus the documented 0072 gap) — 86 rows total, up from 65.

**The debt was 21 rows, not the six T15 named**: 0038–0043 (T15's own count)
plus 0073–0087. T15 itself is now evidence for the rule it enforces — a
register entry that is not re-checked as the project grows silently
undercounts its own problem.

`tools/check_register.py` output, before any edit in this task (baseline,
already passing):

```
note: 4 of 86 task logs are referenced from no thread, doc or note:
  0014 0037 0041 0086
note: OPEN-THREADS.md       785 lines, 20 threads
note: CLOSED-THREADS.md    1276 lines, 19 threads
register OK
```

Final output, after moving T15 and backfilling the index (see Problems hit
for the two register-format traps this uncovered):

```
note: 4 of 86 task logs are referenced from no thread, doc or note:
  0014 0037 0041 0086
note: OPEN-THREADS.md       763 lines, 19 threads
note: CLOSED-THREADS.md    1311 lines, 20 threads
register OK
```

(0041 and 0086 are both now indexed in `tasks/README.md` — the "referenced
from no thread, doc or note" check is advisory and scans threads/docs/notes,
not the task index itself, so being in the index does not remove a task from
this list; it is unaffected by this task and expected to stay unaffected.)

## Problems hit

1. **Symptom**: three subagents drafting rows independently produced one row
   (0076) with manually HTML-entity-escaped angle brackets,
   `` `npuembeddings add &lt;repo&gt; [&lt;sha256&gt;]` ``, inside a backtick
   code span. **Cause**: written defensively against `<repo>` being
   misinterpreted as an HTML tag, but CommonMark code spans already
   HTML-escape their contents when rendering — writing the entities as
   literal source text makes the renderer escape the leading `&` a second
   time, so the reader would see the literal characters `&lt;repo&gt;`
   instead of `<repo>`. **Fix**: replaced with plain `<repo>` inside the
   backticks, matching how `CLAUDE.md` itself already writes
   `` `npuembeddings add <org/model> [<sha256>]` `` unescaped.
2. **Symptom**: two drafted `Status` cells (0043, 0079) carried multi-clause
   parentheticals (`in-progress (inferred — no explicit Status field; Results
   section unfilled)`) that did not match the table's one-or-two-word `Status`
   convention. **Cause**: both tasks genuinely lack a `Status` header field
   (0043 has no `Date`/`Milestone`/`Status` block at all; 0079 has no
   `Status`/`Milestone` field), and the drafting agents surfaced that
   correctly but put the explanation in the wrong cell. **Fix**: trimmed
   `Status` to a single word (`in-progress` for 0043, since its own `Results`
   section is an unfilled placeholder and [T4](../../research/OPEN-THREADS.md#t4)
   already treats it as open; `done` for 0079, since its own text states every
   claimed result and the missing field is a documentation gap, not an
   in-progress claim) and moved the explanation into the `Task` cell's prose,
   which is where the rest of the table keeps this kind of detail.
3. **Not a bug, a finding**: two contradictions between task logs and the
   summary documents that cite them, both left standing rather than
   reconciled, per the assignment's explicit instruction not to fix either
   document in this task:
   - **CLAUDE.md cites [`0043`](../0043-m9-attention-geometry/TASK.md) for a
     "not worth the fight" cost/benefit verdict that 0043 never reaches.**
     CLAUDE.md's "What geometry the array actually wants" section says
     *"[`0043`] has not found it worth the fight"* about running attention on
     the array. 0043's own `## Results` section is the literal placeholder
     text `(filled in below)` — the task built four candidate configurations
     specifically to measure the tile-size-vs-column-count tradeoff, and
     never recorded a single throughput number. What 0043 *does* establish,
     fully and correctly, is the **structural** argument (`cols ≤ 4` follows
     from the AIE microkernel's `n ≥ 16` constraint and attention's `64×64`
     shape) — that part of CLAUDE.md's sentence is accurate. The "not worth
     the fight" framing is not; it describes an economic judgement 0043
     never made. [`T4`](../../research/OPEN-THREADS.md#t4) already flags the
     missing-measurements half of this independently.
   - **CLAUDE.md's int8 headline (1.10× on MiniLM, "the int8 quantisation
     pass costs more than the bf16 conversion it replaces") is stale on two
     counts.** It is [`0079`](../0079-m13-int8-why-only-1.1x/TASK.md)'s
     finding, and 0079 itself measured that hypothesis and found it **false**:
     quantisation costs 1.04× bf16's conversion time, not more (0079 §2), and
     the real bottleneck is the C drain (int32 C is still 4 bytes, so int8
     moves no fewer output bytes despite 7× faster arithmetic). Separately,
     the 1.10×/60.8%-array-share numbers CLAUDE.md quotes are superseded by
     later work in the same 0080–0087 range this task also indexed:
     0080–0082's traffic-bound narrowing and fused epilogues, and 0081's
     `tile_n = 64` for bge-large (CLAUDE.md's own geometry table still says
     32 is bge-large's forced ceiling). [`0085`](../0085-m13-release-sweep/TASK.md)'s
     twelve-row sweep — the newest number in the tree — measures MiniLM
     int8/bf16 at **1.71×** and bge-large at **2.39×**, not 1.10×. This task
     does not edit `CLAUDE.md` or `docs/CURRENT_STATUS.md`; it only records
     the discrepancy, per rule 3's own logic that the register (and now the
     task index) is the place a stale claim gets caught, not silently fixed
     in passing.
4. **Not a bug**: 0078's own log contradicts itself internally (its §6 "Not
   built" says the runtime quant/dequant path is unbuilt, its §4c
   "BUILT AND RUNNING ON HARDWARE" reports real hardware numbers). Flagged in
   the 0078 index row rather than resolved — it is a property of the source
   task, not of this indexing task, and rewriting a `TASK.md` after the fact
   would violate "failures are the valuable part."

## Artifacts

- `tasks/README.md` — 21 new rows (0038–0043, 0073–0087), stale "Index gap"
  note removed.
- `research/OPEN-THREADS.md` — T15 removed from the live section; one bullet
  added to the "Closed threads" pointer list.
- `research/CLOSED-THREADS.md` — T15 added verbatim, plus an appended
  resolution paragraph pointing at this task.
- This file.

## Next

The two stale-summary contradictions found in Problems #3 are not this task's
to fix — `CLAUDE.md` and `docs/CURRENT_STATUS.md` are living documents whose
maintainers should decide how to word the correction, and this task's whole
point was to report rather than reconcile. A natural follow-up: update
CLAUDE.md's int8 headline and bge-large `tile_n` claim against 0085's sweep,
and either fill in 0043's `Results` section (closing [T4](../../research/OPEN-THREADS.md#t4))
or retitle its CLAUDE.md citation to name the structural finding only.
