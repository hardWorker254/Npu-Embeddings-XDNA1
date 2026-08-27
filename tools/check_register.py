#!/usr/bin/env python3
"""Check the thread register against the task logs.

WHY THIS EXISTS
---------------
CLAUDE.md rule 3 makes `research/OPEN-THREADS.md` "the authority on what is
still open", and twice before this script was written that authority was wrong
in the same way: a task answered a question and the thread never said so, so a
later session re-derived work that was already in the tree.

  * The rule itself cites two instances -- tasks/0044 Part 3 and note 0005 6b,
    each unread for 20+ tasks after the blocker went away.
  * 2026-08-23 was the third, and the first where the register misled its own
    reader: T28 ran to 308 lines without mentioning tasks/0062, which had BUILT
    and PASSED the thing T28 still described as open. An hour went into
    re-deriving it on paper.

A register is only an authority if something checks it. This is that something.

WHAT IT CHECKS, and why each one is worth its false-positive rate
-----------------------------------------------------------------
1. A task whose TITLE or opening names a thread, which that thread does not
   link back.  <- the one that matters. Filtering on the task's first lines
   rather than its whole body takes this from 41 hits to 1: a task that merely
   CITES a thread should not be linked from it, but a task whose title says
   "T28 -- the hierarchical merge" is the thread's own history.
2. Threads that cite NEITHER a task nor a research note -- a claim with nothing
   behind it. Notes count: T8 and T9 come from note 0007's survey of unused
   IRON surface and have no task, which is legitimate, and a check that
   demanded a task would have pushed someone to invent one.
2b. A closed thread annotated `**REOPENED` that never moved back (T49 check 1).
   T2 carried "**REOPENED for the int8 datapath, 2026-08-22**" for five days
   while CLAUDE.md kept saying "three live threads" -- the annotation was a
   tombstone, not an action. A reopening counts as acted on only when the
   thread has a section back in OPEN-THREADS.md, or a later `SUPERSEDED by
   [Tnn]` block hands the question to a successor. The bold marker is the
   trigger on purpose: prose that merely mentions the word (as T49's own
   closure does) must not fire it.
3. Task logs no thread, doc or note references -- work that happened and left
   no way to find it.
4. The open/closed split, reported rather than enforced: rule 3b says the
   closed threads are kept deliberately, so the number is a prompt to consider
   splitting the file, not a failure.

Exit code is 1 if check 1 or 2 finds anything, so this can gate a release.
Checks 3 and 4 are advisory and never fail the run.

    python tools/check_register.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# BOTH files. The register was split on 2026-08-23 (64% of it was closed
# threads); a checker that read only the open half would report every answered
# thread as missing its task, which is the opposite of useful.
REGISTERS = [REPO / "research" / "OPEN-THREADS.md",
             REPO / "research" / "CLOSED-THREADS.md"]
# Where else a task log may legitimately be referenced from.
REFERRERS = ["docs/*.md", "docs/*/*.md", "research/notes/*.md",
             "research/*.md", "CLAUDE.md"]
# How many lines of a TASK.md count as "what this task is about".
HEAD_LINES = 12


def thread_bodies(text: str) -> dict[str, str]:
    """Map T-number -> its section text. A thread may appear more than once
    (a superseded entry kept beside its answer); the sections concatenate."""
    lines = text.split("\n")
    heads = [(i, m.group(1))
             for i, l in enumerate(lines)
             if (m := re.match(r"^### (T\d+)", l))]
    out: dict[str, str] = {}
    for k, (i, tid) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out[tid] = out.get(tid, "") + "\n".join(lines[i:end])
    return out


def main() -> int:
    reg = chr(10).join(r.read_text(encoding="utf-8") for r in REGISTERS)
    bodies = thread_bodies(reg)
    fail = 0

    # --- 1. a task ABOUT a thread that the thread does not link -------------
    gaps = []
    for d in sorted(glob.glob(str(REPO / "tasks" / "0*/"))):
        num = os.path.basename(d.rstrip("/\\"))[:4]
        f = Path(d) / "TASK.md"
        if not f.exists():
            continue
        head = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                         .split("\n")[:HEAD_LINES])
        for t in sorted(set(re.findall(r"\bT(\d+)\b", head)), key=int):
            tid = "T" + t
            if tid in bodies and f"tasks/{num}" not in bodies[tid]:
                gaps.append((tid, num))
    if gaps:
        print("FAIL: a task log is ABOUT a thread the thread never links back:")
        for tid, num in gaps:
            print(f"  {tid:<5} <- tasks/{num}")
        print("  (this is the failure mode rule 3 exists for -- the register")
        print("   said open, the work was done, and a later session redid it)")
        fail = 1

    # --- 2. threads with no task behind them --------------------------------
    bare = [t for t, b in bodies.items()
            if not re.search(r"tasks/\d{4}|notes/\d{4}|note \d{4}", b)]
    if bare:
        print(f"FAIL: {len(bare)} thread(s) cite neither a task nor a note: "
              f"{' '.join(sorted(bare, key=lambda x: int(x[1:])))}")
        fail = 1

    # --- 2b. a REOPENED closed thread that never moved back (T49) -----------
    open_bodies = thread_bodies(REGISTERS[0].read_text(encoding="utf-8"))
    closed_bodies = thread_bodies(REGISTERS[1].read_text(encoding="utf-8"))
    stuck = []
    for tid, b in closed_bodies.items():
        m = re.search(r"\*\*REOPENED", b)
        if not m or tid in open_bodies:
            continue
        if not re.search(r"SUPERSEDED by \[T\d+\]", b[m.start():]):
            stuck.append(tid)
    if stuck:
        print("FAIL: REOPENED annotation in CLOSED-THREADS that was never "
              "acted on:")
        for tid in sorted(stuck, key=lambda x: int(x[1:])):
            print(f"  {tid} says **REOPENED but has no section back in "
                  f"OPEN-THREADS.md")
            print("       and no later 'SUPERSEDED by [Tnn]' block")
        print("  (T2 sat like this for five days while CLAUDE.md said "
              "'three live threads')")
        fail = 1

    # --- 3. task logs nothing points at (advisory) --------------------------
    seen = set(re.findall(r"tasks/(\d{4})", reg))
    for pat in REFERRERS:
        for f in glob.glob(str(REPO / pat)):
            seen |= set(re.findall(r"tasks/(\d{4})",
                                   Path(f).read_text(encoding="utf-8",
                                                     errors="ignore")))
    allt = sorted(os.path.basename(d.rstrip("/\\"))[:4]
                  for d in glob.glob(str(REPO / "tasks" / "0*/")))
    orphan = [t for t in allt if t not in seen]
    if orphan:
        print(f"note: {len(orphan)} of {len(allt)} task logs are referenced "
              f"from no thread, doc or note:")
        print(f"  {' '.join(orphan)}")

    # --- 4. the split itself ------------------------------------------------
    # A thread filed OPEN in the closed file, or closed in the open one, is how
    # this structure rots: the split is only worth having if the open file
    # really is only what is open.
    for r in REGISTERS:
        t = r.read_text(encoding="utf-8")
        n = len(re.findall(r"^### T\d+", t, flags=re.M))
        print(f"note: {r.name:<19} {len(t.splitlines()):>5} lines, {n:>2} threads")

    o_txt = REGISTERS[0].read_text(encoding="utf-8")
    c_txt = REGISTERS[1].read_text(encoding="utf-8")
    closed_in_open = [m.group(1) for m in
                      re.finditer(r"^### (T\d+)([^\n]*)$", o_txt, flags=re.M)
                      if not re.search(r"\*\*(OPEN|PARTLY|BLOCKED)", m.group(2))]
    open_in_closed = [m.group(1) for m in
                      re.finditer(r"^### (T\d+)([^\n]*)$", c_txt, flags=re.M)
                      if re.search(r"\*\*(OPEN|PARTLY|BLOCKED)", m.group(2))]
    if closed_in_open or open_in_closed:
        print("FAIL: threads on the wrong side of the split")
        if closed_in_open:
            print(f"  answered/retired, still in OPEN-THREADS: "
                  f"{' '.join(closed_in_open)}")
        if open_in_closed:
            print(f"  still open, filed in CLOSED-THREADS:     "
                  f"{' '.join(open_in_closed)}")
        fail = 1

    # --- 5. dangling cross-references ---------------------------------------
    # Every thread heading carries an explicit `<a id="tNN">` so its anchor
    # survives the heading being reworded. Before that, 33 of the register's
    # own links were dead -- most of them broken not by the 2026-08-23 split
    # but by ordinary edits months earlier, because a GitHub anchor is the
    # slugified heading and every rewording silently invalidates it.
    ids: dict[str, str] = {}
    for r in REGISTERS:
        for a in re.findall(r'<a id="([a-z0-9-]+)"></a>',
                            r.read_text(encoding="utf-8")):
            ids[a] = r.name
    dead = []
    for r in REGISTERS:
        t = r.read_text(encoding="utf-8")
        for m in re.finditer(
                r"\]\((CLOSED-THREADS\.md|OPEN-THREADS\.md)?#([a-z0-9-]+)\)", t):
            target_file = m.group(1) or r.name
            if ids.get(m.group(2)) != target_file:
                dead.append((r.name, m.group(1) or "", m.group(2)))
    if dead:
        print(f"FAIL: {len(dead)} dangling thread link(s):")
        for f, tgt, anc in dead[:12]:
            print(f"  {f}: ]({tgt}#{anc})")
        fail = 1

    if not fail:
        print("register OK")
    return fail


if __name__ == "__main__":
    sys.exit(main())
