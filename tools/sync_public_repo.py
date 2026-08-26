# NpuEmbeddings -- assemble the public repository into repo/.
# SPDX-License-Identifier: Apache-2.0
#
# THIS repository is the working one and stays private: it holds the indexed
# summaries of other people's papers, the PDFs themselves, and everything else
# that is useful to us but not ours to publish. `repo/` is the subset that is.
#
# WHAT DECIDES WHAT SHIPS
# -----------------------
# The file list comes from `git ls-files`, not from walking the filesystem.
# That is deliberate: .gitignore already encodes "does this belong in a
# repository" for the model checkpoint, the build outputs, the exported
# designs, the virtualenv and the release staging area. Reusing it means none
# of those can leak here through an oversight in a second, parallel list.
#
# On top of that, EXCLUDE below removes what is ours-to-keep-private rather
# than ours-to-publish.
#
# THE LINK PROBLEM, AND WHY IT IS THE INTERESTING PART
# ----------------------------------------------------
# docs/ and tasks/ reference the paper summaries ~38 times. Dropping the
# summaries without touching the links would ship a documentation set full of
# 404s. So links into excluded material are REWRITTEN:
#
#   research/papers/<arxiv-id>.md  ->  https://arxiv.org/abs/<arxiv-id>
#         a link to our summary becomes a citation of the actual paper, which
#         is what a public document should have had in the first place
#   any other excluded target     ->  the link text, unlinked
#
# Then everything is scanned again, and a surviving reference to excluded
# material FAILS the sync. A broken public repo is worse than an unpublished
# one, and this is the check that makes "I think I got them all" unnecessary.
#
# Env: any Python 3. Run from the repository root.
# Usage:
#   python tools\sync_public_repo.py
#   python tools\sync_public_repo.py --no-notes      # also drop research/notes
#   python tools\sync_public_repo.py --dry-run

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "repo"

# Ours to keep private, not ours to publish.
#
# research/papers/ and OthersResarch/ are the reason this script exists: the
# summaries are written to be genuine substitutes for the papers, which makes
# them exactly the thing not to republish.
#
# research/notes/ is a different animal and ships by default -- see NOTES_NOTE.
EXCLUDE_PREFIXES = [
    "OthersResarch/",
    "research/papers/",
    "research/README.md",
    "research/prior-art.md",
    # A draft comment for someone else's issue tracker, including its own
    # retraction. Useful history for us; not something to publish about
    # another project.
    "FASTFLOW_ISSUE.md",
]

# CLAUDE.md SHIPS, after being excluded and put back.
#
# It is operating instructions for this repository, and a fifth of it is about
# material the public repo does not contain -- which argued for dropping it.
# What settled the question was measuring the other direction: 23 files make
# 33 references to it, because it is where every hardware trap is written
# down. Excluding it would have left 33 pointers into nothing, aimed at the
# single most reusable document here. Its links into excluded material are
# rewritten like any other file's, and its rule about reading summaries
# instead of PDFs simply describes how the working repository is organised.

NOTES_PREFIX = "research/notes/"
NOTES_NOTE = """research/notes/ holds five write-ups of OUR OWN hardware findings -- a
  silent architecture fallback, a device-sync bug we misdiagnosed and
  retracted, the context-switch cost model. They quote nobody, they are the
  most reusable thing here, and docs/ and tasks/ link them ~15 times.
  Excluded with --no-notes if you disagree."""

# Files above this are measurement artifacts that are evidence for a number in
# a task log. Kept, but reported, because a public repo that is mostly trace
# JSON is a worse advert than one that is mostly source.
BIG_FILE_MB = 0.5

ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")

# [text](path) and [text](path#anchor), including ../ prefixes.
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def is_excluded(path: str, drop_notes: bool) -> bool:
    if any(path.startswith(p) for p in EXCLUDE_PREFIXES):
        return True
    if drop_notes and path.startswith(NOTES_PREFIX):
        return True
    return False


def normalise(src_file: str, target: str) -> str:
    """Resolve a relative markdown link to a repo-root-relative path."""
    target = target.split("#", 1)[0].strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return ""
    base = Path(src_file).parent
    try:
        return (base / target).resolve().relative_to(REPO).as_posix()
    except (ValueError, OSError):
        return target.lstrip("./")


def rewrite_links(text: str, src_file: str, drop_notes: bool, stats: dict) -> str:
    def sub(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        resolved = normalise(src_file, target)
        if not resolved or not is_excluded(resolved, drop_notes):
            return m.group(0)
        # A summary of an arXiv paper becomes a citation of the paper.
        stem = Path(resolved).stem
        if resolved.startswith("research/papers/") and ARXIV_RE.match(stem):
            stats["arxiv"] += 1
            return f"[{label}](https://arxiv.org/abs/{stem})"
        stats["unlinked"] += 1
        return label
    return LINK_RE.sub(sub, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-notes", action="store_true",
                    help="also exclude research/notes/ (our own findings)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)

    files = tracked_files()
    keep, dropped = [], []
    for f in files:
        (dropped if is_excluded(f, args.no_notes) else keep).append(f)

    print(f"tracked {len(files)} files -> keep {len(keep)}, drop {len(dropped)}")
    by_prefix: dict[str, int] = {}
    for f in dropped:
        p = next((p for p in EXCLUDE_PREFIXES + [NOTES_PREFIX]
                  if f.startswith(p)), "?")
        by_prefix[p] = by_prefix.get(p, 0) + 1
    for p, n in sorted(by_prefix.items()):
        print(f"  excluded  {p:<24} {n:>3} files")
    if not args.no_notes:
        print(f"\n  NOTE: {NOTES_NOTE}\n")

    if args.dry_run:
        print("dry run -- nothing written")
        return 0

    # Rebuild from scratch: a stale file left behind from a previous run with
    # different settings is exactly the kind of leak this script exists to
    # prevent. .git is preserved so the public repo keeps its own history.
    if out_root.exists():
        for child in out_root.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    stats = {"arxiv": 0, "unlinked": 0}
    n_rewritten = 0
    big: list[tuple[float, str]] = []
    total = 0

    for f in keep:
        src, dst = REPO / f, out_root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        total += size
        if size > BIG_FILE_MB * 1024 * 1024:
            big.append((size / 1048576, f))
        if f.endswith(".md"):
            text = src.read_text(encoding="utf-8")
            new = rewrite_links(text, f, args.no_notes, stats)
            if new != text:
                n_rewritten += 1
            dst.write_text(new, encoding="utf-8", newline="\n")
        else:
            shutil.copy2(src, dst)

    print(f"copied {len(keep)} files, {total / 1048576:.1f} MB")
    print(f"  rewrote links in {n_rewritten} markdown files: "
          f"{stats['arxiv']} -> arxiv.org, {stats['unlinked']} unlinked")

    # ---- the check that makes this trustworthy ---------------------------
    # Two different things look alike here and only one is a defect:
    #
    #   a surviving LINK to excluded material -- the reader clicks and gets a
    #     404. This fails the sync.
    #   a PROSE mention ("the PDFs were indexed under OthersResarch/") -- an
    #     honest description of how the work was done in the working
    #     repository. Not a defect; reported so it stays a conscious choice,
    #     and explained once in the README rather than edited out of 30 task
    #     logs, which would falsify the record.
    dead_links: list[tuple[str, int, str]] = []
    mentions: list[tuple[str, int, str]] = []
    probes = [p.rstrip("/") for p in EXCLUDE_PREFIXES]
    if args.no_notes:
        probes.append(NOTES_PREFIX.rstrip("/"))
    for f in keep:
        p = out_root / f
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            hit = next((pr for pr in probes if pr in line), None)
            if hit is None:
                continue
            # Inside a markdown link target -> the reader would follow it.
            linked = any(hit in m.group(2) for m in LINK_RE.finditer(line))
            (dead_links if linked else mentions).append(
                (f, i, line.strip()[:110]))

    if big:
        print(f"\n  {len(big)} files over {BIG_FILE_MB} MB "
              f"(measurement evidence; fine, but be aware):")
        for mb, f in sorted(big, reverse=True)[:6]:
            print(f"    {mb:5.1f} MB  {f}")

    if mentions:
        print(f"\n  {len(mentions)} prose mention(s) of excluded material "
              f"(history, not defects):")
        seen_files = sorted({f for f, _, _ in mentions})
        for f in seen_files[:10]:
            n = sum(1 for g, _, _ in mentions if g == f)
            print(f"    {n:>2}x  {f}")
        if len(seen_files) > 10:
            print(f"    ... and {len(seen_files) - 10} more files")
        print("    README explains that the working repository keeps an "
              "indexed literature review that is not republished.")

    # ---- and every OTHER relative link, while we are here ----------------
    # Excluded material is not the only way to ship a 404: an ordinary link
    # can rot too, and a public repository is where someone else notices.
    # Checking all of them costs nothing once the tree exists.
    broken: list[tuple[str, int, str]] = []
    for f in keep:
        if not f.endswith(".md"):
            continue
        text = (out_root / f).read_text(encoding="utf-8")
        # CODE IS NOT A LINK. A task log that quotes the tasks/README.md row it
        # added -- inside a ```markdown fence, which is how nearly all of them
        # do it -- is showing SOURCE, and GitHub renders it as source. Scanning
        # it for links reports 17 false 404s and, worse, would refuse the very
        # file explaining a link repair. Same for an inline code span: 0089
        # spells out the FORMAT of a register bullet, `* [Tn](CLOSED-THREADS.md#tn)`,
        # which is a description rather than a destination.
        fenced = False
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            for m in LINK_RE.finditer(line):
                before = line[:m.start()]
                if before.count("`") % 2 == 1:          # inside a code span
                    continue
                target = m.group(2).split("#", 1)[0].strip()
                if not target or target.startswith(
                        ("http://", "https://", "mailto:", "#")):
                    continue
                dest = (out_root / f).parent.joinpath(target)
                try:
                    inside = dest.resolve().is_relative_to(out_root.resolve())
                except OSError:
                    inside = False
                # Escaping the tree means a GitHub-relative link such as
                # ../../releases or ../../issues, which is correct on GitHub
                # and unverifiable here.
                if not inside:
                    continue
                if not dest.exists():
                    broken.append((f, i, m.group(0)[:90]))

    if broken:
        print()
        print(f"  {len(broken)} relative link(s) with no target in repo/:")
        for f, i, link in broken[:15]:
            print(f"    {f}:{i}: {link}")
        if len(broken) > 15:
            print(f"    ... and {len(broken) - 15} more")

    if dead_links:
        print()
        print(f"FAIL -- {len(dead_links)} link(s) to excluded material "
              f"survive; a reader would get a 404:")
        for f, i, line in dead_links[:25]:
            print(f"  {f}:{i}: {line}")
        if len(dead_links) > 25:
            print(f"  ... and {len(dead_links) - 25} more")
        print("Fix the source in this repository, then re-run.")
        return 1

    if broken:
        print()
        print("FAIL -- the links above resolve to nothing. Fix the source "
              "in this repository, then re-run.")
        return 1

    where = (out_root.relative_to(REPO) if out_root.is_relative_to(REPO)
             else out_root)
    print()
    print(f"PASS -- {len(keep)} files in {where}, no dead links")
    print()
    print("Next: cd repo && git init && git add -A && git commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
