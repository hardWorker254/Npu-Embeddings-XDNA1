#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Parity gate for the tools/export/exporters/ split.
#
# The exporters were split out of two monoliths. A refactor that changes what
# the tool PRINTS is a behaviour change, and the only cheap way to know it did
# not is to run both and compare. So this reconstructs the pre-split monoliths
# from git (HEAD, i.e. the last committed state, which is the monolith because
# the split is not committed) and diffs stdout, stderr and the exit code of
# every documented invocation.
#
#   python tools/verify/parity_exporters.py            # the 17 cases below
#   python tools/verify/parity_exporters.py --rev HEAD~1
#
# The reference tree is rebuilt in a temp dir on every run: the two monoliths
# come out of git, everything else they import is symlinked from the working
# tree, so a change in a shared helper counts as a behaviour change on both
# sides instead of hiding behind a stale copy.
#===----------------------------------------------------------------------===//

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
# Where the two entry points live now, and where the shared modules they import
# by flat name live. The reference tree built below is deliberately left FLAT --
# that is the layout the monoliths were written against, so it is the layout they
# have to be run in to be comparable.
NEW = TOOLS / "export"
LIB = TOOLS / "lib"
MONOLITHS = ("export_gemm_rtp.py", "export_eltwise.py")
# Everything the monoliths import by module name. Symlinked, not copied, so a
# shared edit is a behaviour change on both sides.
SHARED = ("gemm_pretiled.py", "npue.py", "toolchain_provenance.py",
          "onnx_weights.py", "whisper_bpe.py")

CASES = [
    ("gemm", ["--list-targets"]),
    ("gemm", ["--target", "all-MiniLM-L6-v2", "--arch", "1", "--dry-run"]),
    ("gemm", ["--target", "bge-large-en-v1.5", "--arch", "2", "--dry-run"]),
    ("gemm", ["--target", "nomic-embed-text-v1.5", "--arch", "all", "--dry-run"]),
    ("gemm", ["--target", "embeddinggemma-300m", "--arch", "1", "--dry-run", "--batches", "4,8"]),
    ("gemm", ["--target", "gte-multilingual-base", "--arch", "1", "--seq", "128", "--dry-run"]),
    ("gemm", ["--arch", "1", "--batch", "8", "--cols", "4", "--hidden", "384", "--dry-run"]),
    ("gemm", ["--target", "no-such-model", "--arch", "1", "--dry-run"]),
    ("gemm", ["--target", "all-MiniLM-L6-v2", "--seq", "7", "--dry-run"]),
    ("gemm", ["--target", "all-MiniLM-L6-v2", "--seq", "64", "--batches", "4,6", "--dry-run"]),
    ("gemm", ["--target", "all-MiniLM-L6-v2", "--int8", "--emulate-bfp16", "--dry-run"]),
    # ACCEPTED DIFFERENCE: the M-not-tileable error's HINT now names -m and
    # --rows as well, because they are what can fix it. The numbers and the
    # refusal are unchanged.
    ("gemm", ["--target", "all-MiniLM-L6-v2", "--seq", "64", "-m", "48", "--dry-run"],
     [("Pick a tier or a seq whose product divides it.",
       "Pick a tier, a seq, -m or --rows whose product divides it.")]),
    ("gemm", ["--target", "all-MiniLM-L6-v2", "-m", "0", "--dry-run"]),
    ("gemm", ["--targets-file", "BAD", "--list-targets"]),
    # The two elt cases whose output STARTS with argparse's usage block. The
    # block is normalised before the comparison (see `usage_differs` below) and
    # the flags it lists are asserted by REQUIRED_FLAGS_ELT instead -- this fork
    # added --extra-ops and renamed --npu-eltwise, so the usage text cannot
    # match a monolith that predates both. Everything AFTER the block -- the
    # refusal itself, its wording, the exit code -- is still compared byte for
    # byte, which is the part that is behaviour.
    ("elt", ["--help"], "usage"),
    ("elt", ["--arch", "1", "--elt-cols", "9"]),
    ("elt", ["--arch", "1", "--gelu-tile", "2048"], "usage"),
]

# The gemm --help is not a parity case at all: this fork added --rows,
# --stream-set and --dec-seq, so its usage text cannot match the pre-split
# monolith's. What is asserted instead is that the flags are all still there --
# a refactor that quietly dropped one would pass a byte-comparison of nothing.
# The elt cases keep their comparison, with argparse's leading usage block
# normalised (usage_differs above), because the refusal after it is behaviour
# and the block is not.
#
# What a RENAME costs, stated plainly: the monolith still has --npu-eltwise, so
# no case here can notice a script that was never updated to --npu-extra-ops.
# That is the price of an intentional CLI change, and the runtime's own refusal
# of the old flag name is what covers it at the other end.
REQUIRED_FLAGS = ("--cols", "--seq", "-m", "-k", "-n", "--batches", "--rows",
                  "--stream-set", "--dec-seq", "--npu-extra-ops", "--int8",
                  "--emulate-bfp16", "--per-arch-cache", "--identity-threshold")

REQUIRED_FLAGS_ELT = ("--arch", "--out", "--batch", "--seq", "--hidden",
                      "--elt-cols", "--gelu-tile", "--ln-variant",
                      "--sm-variant", "--cache-root", "--per-arch-cache",
                      "--extra-ops")


def blob_at(rev: str, name: str):
    """The bytes of `name` at `rev`, or None. Follows the re-layout.

    Every file these tools touch moved once: the entry points into
    tools/export/, the shared modules into tools/lib/, the data into
    tools/data/. A `git show REV:tools/export/<name>` therefore fails for every
    revision from before that move, which is exactly the set of revisions the
    monoliths live in -- so the old flat path is tried second rather than
    concluding the file is absent.
    """
    for path in (f"tools/export/{name}", f"tools/lib/{name}",
                 f"tools/data/{name}", f"tools/{name}"):
        p = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                           capture_output=True)
        if p.returncode == 0:
            return p.stdout
    return None


def history(entry: str) -> list[str]:
    """Revs touching `entry`, newest first, following renames.

    Asked about tools/export/ FIRST and the old flat path second, because those
    are the two layouts this file has had: `git log --follow` walks back through
    a rename, but only from the path that exists. While the re-layout is an
    uncommitted working-tree move the new path matches no commit at all, so
    asking only about it yields nothing and the caller silently falls back to
    HEAD -- which is a shim onto the split package, i.e. the reference side
    compared against itself. Once the move is committed the first path answers
    and the second is never reached.
    """
    for path in (f"tools/export/{entry}", f"tools/{entry}"):
        revs = subprocess.run(["git", "log", "--follow", "--format=%H", "--", path],
                              capture_output=True, text=True).stdout.split()
        if revs:
            return revs
    return []


def reference_rev(entry: str) -> tuple[str, str]:
    """The newest revision at which `entry` was still a MONOLITH, and a note.

    "Still a monolith" is decided by reading the file, not by asking when the
    package appeared, because the two do not answer the same question: at one
    commit in this history tools/export_eltwise.py was already a shim importing
    `exporters.eltwise.main` while the package it imported had never been
    committed. Diffing against that shim compares a shim with itself, or -- since
    the reference tree does not have the package -- crashes it on a missing
    import and reports the traceback as a behavioural difference.

    So: walk the file's history newest-first and take the first revision whose
    blob does not mention the package. `--rev` still wins when given.
    """
    revs = history(entry)
    for rev in revs:
        blob = blob_at(rev, entry)
        if blob is None:
            continue
        if b"exporters" in blob:
            continue    # a shim onto the split package
        short = subprocess.run(["git", "rev-parse", "--short", rev],
                               capture_output=True, text=True).stdout.strip()
        subject = subprocess.run(["git", "log", "-1", "--format=%s", rev],
                                 capture_output=True,
                                 text=True).stdout.strip()
        skipped = len(revs) - revs.index(rev) - 1
        return rev, (f"{entry}: newest monolith is {short} ({subject})"
                     + (f"; {skipped} later revision(s) are shims onto the "
                        f"split" if skipped else ""))
    return "HEAD", f"{entry}: no monolith in history, diffing against HEAD"


def build_reference(revs: dict, root):
    """Materialise the pre-split tools/ into root, per tool at its own rev.

    Returns (tools dir, targets file). The targets file is the rev's own, and
    BOTH sides are pointed at it: the split's npu_targets.json has gained keys
    (per-model `batches`, STT entries) that the monolith's strict validator
    rejects outright, so feeding each side its own default would compare a
    refusal against a listing and call it parity. Same input, same output is
    the only comparison that means anything.
    """
    ref = root / "ref"
    (ref / "tools").mkdir(parents=True)
    for name in MONOLITHS + ("npu_targets.json",):
        rev = revs[name.split(".")[0].replace("export_", "")] \
            if name.endswith(".py") else revs["targets"]
        blob = blob_at(rev, name)
        if blob is None:
            raise SystemExit(
                f"git show {rev} -- {name} failed. That revision does not "
                f"hold a monolith for it, so there is nothing to diff against; "
                f"pass --rev with a commit that does.")
        (ref / "tools" / name).write_bytes(blob)
    for name in SHARED:
        src = LIB / name
        if src.exists():
            (ref / "tools" / name).symlink_to(src)
    return ref / "tools", ref / "tools" / "npu_targets.json"


def run(cwd, script, args):
    p = subprocess.run([sys.executable, f"{cwd}/{script}"] + args,
                       capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff the split exporters against the monoliths in git.")
    ap.add_argument("--rev", default=None,
                    help=(
                        "git rev to diff BOTH tools against. Omit it and each "
                        "tool is diffed against its own pre-split revision, "
                        "resolved from history -- the two halves of this "
                        "exporter were split in different commits, so a single "
                        "rev cannot serve both."))
    args = ap.parse_args()

    if args.rev:
        revs = {k: args.rev for k in ("gemm_rtp", "eltwise", "targets")}
        notes = [f"both tools diffed against {args.rev}"]
    else:
        revs, notes = {}, []
        for pkg, entry in (("gemm_rtp", "export_gemm_rtp.py"),
                           ("eltwise", "export_eltwise.py")):
            rev, note = reference_rev(entry)
            revs[pkg] = rev
            if note:
                notes.append(note)
        # The targets file must come from the SAME era as the monoliths, and
        # from the OLDEST of them when they differ: the monolith's validator is
        # strict, and it rejects override keys the split has since learned
        # (per-model `batches`, and the STT entries). Pointing one side at HEAD's
        # targets file turns a behaviour comparison into a comparison of a
        # refusal against a listing -- which is what this file's own header says
        # must not happen.
        def _when(rev):
            out = subprocess.run(["git", "show", "-s", "--format=%ct", rev],
                                 capture_output=True, text=True).stdout.strip()
            return int(out) if out else 0
        revs["targets"] = min(revs["gemm_rtp"], revs["eltwise"], key=_when)
        _t_short = subprocess.run(
            ["git", "rev-parse", "--short", revs["targets"]],
            capture_output=True, text=True).stdout.strip()
        notes.append(f"both tools' targets file: {_t_short} (the older of the "
                     f"two, for the strict validator's sake)")

    bad = 0
    for n in notes:
        print(f"reference: {n}")
    if notes:
        print()
    with tempfile.TemporaryDirectory(prefix="exporter-parity-") as tmp:
        ref_tools, targets = build_reference(revs, Path(tmp))
        new_tools = NEW

        for case in CASES:
            kind, argv = case[0], list(case[1])
            label = " ".join(argv)
            # A case's third element is either a list of accepted text
            # differences or the marker "usage" (argparse's leading block is
            # normalised -- see the note by CASES).
            usage_differs = len(case) > 2 and case[2] == "usage"
            accept = case[2] if len(case) > 2 and isinstance(case[2], list) else []
            script = "export_gemm_rtp.py" if kind == "gemm" else "export_eltwise.py"
            if kind == "gemm" and "--targets-file" not in argv:
                argv = ["--targets-file", str(targets)] + argv
            argv = [a.replace("BAD", "/nonexistent/npu_targets.json") for a in argv]

            # Each side runs from its own script's directory: the monoliths were
            # written against a flat tools/, while the split resolves
            # `exporters.` and tools/lib/ relative to its own file.
            ro, so, eo = run(str(ref_tools), script, argv)
            rn, sn, en = run(str(new_tools), script, argv)

            # The two trees are different directories, so both the module paths
            # and the cwd-relative --out values differ textually. Normalise
            # both roots; anything left is a real behavioural difference.
            # The script's own path is normalised separately: --dry-run echoes
            # the command line it would run, and that line names the script, so
            # the re-layout into tools/export/ would otherwise read as a
            # difference in every dry-run case.
            def norm(t, tree):
                # The script's own path first, while it is still absolute: the
                # root substitution below rewrites the prefix and would leave
                # nothing to match against.
                t = t.replace(f"{tree}/{script}", "ENTRY POINT")
                t = t.replace(str(ref_tools.parent), "TREE").replace(str(REPO), "TREE")
                if usage_differs:
                    # argparse's leading block: everything from "usage:" up to the
                    # first blank line, which is where the options and the
                    # per-flag help start. Replaced whole, because the two texts
                    # wrap at different columns and a line-wise substitution
                    # would compare wrapping rather than behaviour.
                    i = t.find("usage:")
                    if i >= 0:
                        j = t.find("\n\n", i)
                        t = t[:i] + "USAGE BLOCK (normalised: this fork added "
                        "flags to it)\n" + (t[j:] if j >= 0 else "")
                return t
            so_n, eo_n = norm(so, ref_tools), norm(eo, ref_tools)
            sn_n, en_n = norm(sn, new_tools), norm(en, new_tools)

            # THE PER-MODEL OUTPUT LAYOUT, and the only difference here that is
            # not attached to a case. Two parts, one change:
            #
            #   split:   <root>/artifacts/<model>[-i8]/artifacts_npu<N>
            #   monolith <root>/<model>/artifacts_npu<N>
            #
            # The `artifacts/` level groups every per-model set under one
            # parent, so `ls runtime/` tells build artifacts from source
            # directories. The `-i8` suffix gives a model's two datapaths two
            # directories, because they are indistinguishable by geometry and
            # differ only in `b_layout_hash` -- so the runtime proposes both
            # names and the writer has to produce both.
            #
            # Both are normalised HERE, once and globally, rather than as `accept`
            # substitutions on the cases that happen to print an `--out`.
            # Per-case would be seven chances to forget one, and a missed case
            # would read as a regression rather than as the change it is.
            #
            # What is NOT done: the substitution touches nothing but this path
            # shape, so a real difference elsewhere in the command line still
            # fails, and so does a difference in which model or generation is
            # named -- including `-i8` appearing where the split did not put it.
            _OUT = re.compile(r"TREE/runtime(?:/artifacts)?/"
                              r"([A-Za-z0-9_.-]+?)(?:-i8)?"
                              r"(/artifacts_npu\d)")
            so_n = _OUT.sub(r"TREE/runtime/artifacts/\1\2", so_n)
            sn_n = _OUT.sub(r"TREE/runtime/artifacts/\1\2", sn_n)
            eo_n = _OUT.sub(r"TREE/runtime/artifacts/\1\2", eo_n)
            en_n = _OUT.sub(r"TREE/runtime/artifacts/\1\2", en_n)
            for old, new in accept:
                so_n, sn_n = so_n.replace(old, new), sn_n.replace(old, new)
                eo_n, en_n = eo_n.replace(old, new), en_n.replace(old, new)

            if (ro, so_n, eo_n) == (rn, sn_n, en_n):
                print(f"OK   {kind:5s} {label[:70]}")
            else:
                bad += 1
                print(f"DIFF {kind:5s} {label[:70]}  rc {ro} vs {rn}")
                for line in difflib.unified_diff(
                        (so_n + eo_n).splitlines(), (sn_n + en_n).splitlines(),
                        "monolith", "split", lineterm="", n=1):
                    print("   " + line)

        h = subprocess.run([sys.executable, f"{new_tools}/export_gemm_rtp.py", "--help"],
                           capture_output=True, text=True, cwd=new_tools).stdout
        for flag in REQUIRED_FLAGS:
            if flag not in h:
                print(f"MISSING FLAG in gemm_rtp --help: {flag}")
                bad += 1
        he = subprocess.run([sys.executable, f"{new_tools}/export_eltwise.py", "--help"],
                            capture_output=True, text=True, cwd=new_tools).stdout
        for flag in REQUIRED_FLAGS_ELT:
            if flag not in he:
                print(f"MISSING FLAG in export_eltwise --help: {flag}")
                bad += 1

    # The presence checks count too: they are cases in everything but name, and
    # a run that fails one has NOT reproduced the monolith's behaviour.
    total = len(CASES) + len(REQUIRED_FLAGS) + len(REQUIRED_FLAGS_ELT)
    print(f"\n{total - bad}/{total} identical "
          f"({len(CASES)} behaviour cases, "
          f"{len(REQUIRED_FLAGS) + len(REQUIRED_FLAGS_ELT)} flag-presence "
          f"checks)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
