#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Regenerate NPU_OPS.md from tools/lib/npu_ops.py.
#
# The 40 cells (5 architectures x 8 codes) live in the registry as Python because
# the exporter reads them to decide what to compile. A hand-written Markdown copy
# of the same table would be a third copy of the truth and would drift the first
# time somebody added a code, so NPU_OPS.md is GENERATED and
# tools/verify/verify_npu_op_matrix.py asserts the checked-in file is exactly what
# this script prints.
#
#   python tools/gen_npu_ops_doc.py            # rewrite NPU_OPS.md
#   python tools/gen_npu_ops_doc.py --check    # exit 1 if it is stale, print the diff
#
# What this does NOT do is generate prose. The reasons come from the registry
# verbatim, so the document cannot flatter a model; the framing sentences around
# the tables are the only hand-written part and they live in this file, where a
# reviewer sees them change.
#===----------------------------------------------------------------------===//

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "lib"))
import npu_ops  # noqa: E402  -- needs the path above

DOC = REPO / "NPU_OPS.md"

# The order the architectures are printed in. Deliberately NOT the registry's
# dict order: this is the order a reader meets them in -- the default first, then
# the two that have audio, then the two that are not transformers at all.
ARCHES = [
    ("gemm_rtp", "Text embedders", "BERT and friends: bge, MiniLM, nomic, gte. "
     "This is also the default for a text embedder with no `kind` of its own."),
    ("stt", "Speech to text", "Whisper (all sizes) and anything else that "
     "carries an audio front end."),
    ("cls", "Image classification", "ViT. Its four GEMM streams are "
     "`gemm_rtp`'s, which is why the classification head lives here."),
    ("pose", "Pose", "YOLO pose. No transformer, no normalisation, no "
     "attention -- most of this row is `absent`, and that is the answer."),
    ("embeddinggemma-300m", "Gemma", "The one model whose ENCODER differs from "
     "its kind, so it gets a row of its own rather than a fifth kind."),
]


# Status -> the one word that goes in the matrix. Short because a table cell has
# to stay a cell; the per-architecture sections carry the reasons.
MARK = {
    "honours": "**yes**",
    "on_array": "already",
    "unimplemented": "no code",
    "blocked": "blocked",
    "absent": "-",
}


def reg_of(label):
    """The row for one architecture, whether it is a kind or a lone model.

    MODEL_REGISTRY wins over KIND_REGISTRY because a model entry REPLACES its
    kind's row rather than patching it -- registry_for() does the same thing for
    the exporter, and two places answering that question differently is the bug
    this function exists to prevent.
    """
    if label in npu_ops.MODEL_REGISTRY:
        return npu_ops.registry_for(None, label)
    return npu_ops.KIND_REGISTRY[label]


def cell_rows(label):
    """(code, long name, status, reason) in CODE order, for one architecture."""
    reg = reg_of(label)
    return [(code, npu_ops.OPS[code][1], reg[code][0], reg[code][1])
            for code in npu_ops.OPS]


def matrix_table():
    """The whole 40-cell matrix, one row per architecture, one column per code.

    The cell is the status and nothing else. An earlier version put the op's long
    name under it and the table was unreadable -- worse, the names are WRONG for
    some cells: `conv` is Whisper's audio front end in the gemm_rtp and stt rows
    and a ViT's patch embedding in the cls row, and one string per column cannot
    say both. The names are in the per-architecture section below, where the
    reason for each cell has room to explain what that cell's op actually is.
    """
    head = "| architecture | " + " | ".join(
        f"`{c}`" for c in npu_ops.OPS) + " |"
    rule = "| --- | " + " | ".join("---" for _ in npu_ops.OPS) + " |"
    lines = [head, rule]
    for label, _title, _blurb in ARCHES:
        reg = reg_of(label)
        cells = [MARK[reg[c][0]] for c in npu_ops.OPS]
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return lines


def _are(n):
    """'is' for 1, 'are' otherwise.

    The tally is generated and the sentence around it is not, so the two drift
    apart the moment a cell changes status -- "1 are already there" is exactly
    what a table with one `on_array` cell produces, and it is the sort of thing
    that makes a reader stop believing the numbers.
    """
    return "is" if n == 1 else "are"


def _do(n):
    return "does" if n == 1 else "do"


def doc():
    # One tally over every cell in the document, so the number at the top is the
    # number of rows below it and the gate can assert both.
    counts = {s: 0 for s in npu_ops.STATUSES}
    for label, _t, _b in ARCHES:
        reg = reg_of(label)
        for code in npu_ops.OPS:
            counts[reg[code][0]] += 1

    L = []
    a = L.append
    a("# Which architecture honours which `--npu-ops` code")
    a("")
    a("<!-- GENERATED by tools/gen_npu_ops_doc.py from tools/lib/npu_ops.py.")
    a("     Do not edit: run the script, or tools/verify/verify_npu_op_matrix.py")
    a("     fails. The registry is the source of truth; this file is its prose.")
    a("-->")
    a("")
    a("The runtime has ONE flag for this, `--npu-ops CODES`, and its default is")
    a("the empty set: **every op runs on the CPU, which is the measured-faster")
    a("path in every case measured so far.** Passing codes is how you ask for the")
    a("array instead, one op at a time.")
    a("")
    a("The exporter has **no such flag**. One command builds every design the")
    a("target can use:")
    a("")
    a("```")
    a("python tools/export/export_gemm_rtp.py --target <model> --arch 1")
    a("```")
    a("")
    a("It prints the list it chose and the reason for every code it skipped, and")
    a("it takes no op flag at all -- `--npu-ops`, `--npu-extra-ops` and")
    a("`--npu-eltwise` are refused by name there, each with its own message. The")
    a("table below is what it reads.")
    a("")
    a(f"There are {len(npu_ops.OPS)} codes and "
      f"{len(ARCHES)} architectures: {len(npu_ops.OPS) * len(ARCHES)} cells. "
      f"Of those, "
      f"{counts['honours']} run on the array today, "
      f"{counts['on_array']} {_are(counts['on_array'])} already there without a "
      f"code, {counts['unimplemented']} {_are(counts['unimplemented'])} "
      f"operations the model has and no array branch reaches, "
      f"{counts['blocked']} cannot be moved on this board for a stated "
      f"reason, and {counts['absent']} "
      f"{_do(counts['absent'])} not exist in that model at all.")
    a("")
    a("## The eight codes")
    a("")
    a("Three are elementwise designs of their own, compiled into a sibling")
    a("directory next to the GEMM set:")
    a("")
    for code in ("gelu", "layn", "softm"):
        a(f"- `{code}` -- {npu_ops.OPS[code][1]}, in `{npu_ops.OPS[code][0]}/`.")
    a("")
    a("Five are **not** designs of their own. They are streams added to a design")
    a("set the GEMM export already produces, so asking for them costs no extra")
    a("xclbin -- which is why `build.py` compiles nothing for them and only")
    a("`resolve.py` has to know they exist:")
    a("")
    for code in sorted(npu_ops.STREAM_ONLY):
        a(f"- `{code}` -- {npu_ops.OPS[code][1]}: "
          f"{npu_ops.NO_DESIGN[code]}.")
    a("")
    a("## What the five statuses mean")
    a("")
    for s in npu_ops.STATUSES:
        a(f"- **{s}** -- {npu_ops.STATUS_MEANING[s]}.")
    a("")
    a("`unimplemented` is the one worth distinguishing carefully from `blocked`:")
    a("it is **a branch of code that was never written**, not a property of the")
    a("model and not something the exporter can fix on its own. `blocked` means")
    a("the operation is real and cannot be moved here, and the reason says which")
    a("of the three it is: fused into another op, needs a different kernel and a")
    a("new code, or does not tile.")
    a("")
    a("## The matrix")
    a("")
    L.extend(matrix_table())
    a("")
    a("`already` means the cell is empty because the work is done without a code,")
    a("which is better than a tick. `-` means the model has no such operation,")
    a("which is permanent and correct. `no code` is the honest middle: the")
    a("operation is there, and nothing reaches it.")
    a("")
    for label, title, blurb in ARCHES:
        a(f"## `{label}` -- {title}")
        a("")
        a(blurb)
        a("")
        for code, long_name, status, reason in cell_rows(label):
            head = f"### `{code}` -- {long_name}: **{status}**"
            a(head)
            a("")
            a(reason)
            a("")
            design = npu_ops.OPS[code][0]
            if not design and status == "honours":
                a(f"Design directory: none -- "
                  f"{npu_ops.NO_DESIGN.get(code, 'nothing to compile')}. See "
                  f"`STREAM_ONLY` in `tools/lib/npu_ops.py`.")
                a("")
            elif status == "honours":
                a(f"Design directory: `{design}/`, built automatically by the "
                  f"exporter for any target of this architecture.")
                a("")
    a("## Where this is checked")
    a("")
    a("- `tools/lib/npu_ops.py` -- the registry. The exporter builds from it.")
    a("- `runtime/include/common/npu_ops_flag.hpp` -- the runtime's second copy")
    a("  of the eight codes and their long names. Neither side can include the")
    a("  other, so each points at the other, and the failure mode of the")
    a("  duplication is a refusal by name on both sides -- never a wrong number.")
    a("- `tools/verify/verify_npu_op_matrix.py` -- asserts this file is exactly")
    a("  what `tools/gen_npu_ops_doc.py` prints, asserts the 40-cell count and")
    a("  the per-status tally, and asserts the runtime accepts or refuses each")
    a("  cell's codes the way this document says.")
    a("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 and print the diff if stale")
    ns = ap.parse_args()
    text = doc()
    if ns.check:
        if not DOC.exists():
            print(f"{DOC.name} does not exist; run tools/gen_npu_ops_doc.py")
            return 1
        cur = DOC.read_text()
        if cur != text:
            print(f"{DOC.name} is stale; tools/gen_npu_ops_doc.py prints:")
            for line in difflib.unified_diff(
                    cur.splitlines(), text.splitlines(),
                    DOC.name, "generated", lineterm="", n=1):
                print("  " + line)
            return 1
        print(f"{DOC.name} matches tools/lib/npu_ops.py")
        return 0
    DOC.write_text(text)
    print(f"wrote {DOC} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())