#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Gate for the op registry: 8 codes x 5 architectures = 40 cells.
#
# tools/lib/npu_ops.py is the registry. It is not prose and it cannot be checked
# by reading it, so this file checks the four things that can be checked:
#
#   1. the registry's own shape -- 40 cells, five statuses, one tally, and the
#      set of codes with no design directory equal to STREAM_ONLY;
#   2. NPU_OPS.md, which is GENERATED from the registry, is not stale;
#   3. the runtime's C++ table (runtime/include/common/npu_ops_flag.hpp) carries
#      the same eight codes with the same design directory and the same long
#      name, in the same order. This is the duplication the two files' headers
#      both apologise for; the apology is only worth anything if something checks
#      it, and nothing else does -- verify_cli_flags.py checks that the FLAG
#      exists, not what the table says.
#   4. the runtime accepts or refuses each cell the way the registry says, by
#      RUNNING it, one code at a time, for every architecture that has a
#      container in this checkout.
#
# What check 4 does and does not claim. It asserts a binary fact per cell: a cell
# whose status is `honours` or `on_array` runs to completion with that code on
# the command line, and a cell whose status is anything else exits non-zero and
# names the code it refused. It does NOT assert the refusal's wording, which is
# prose and changes; and it does NOT assert that an accepted run computed the
# right numbers, which is what verify_vit / verify_whisper / verify_tail are for.
# The two together are the pair: this gate says the switch means what the table
# says, those say the arithmetic behind it is right.
#
# The pose row is SKIPPED, loudly, because there is no pose container in this
# checkout. Its eight cells are counted in check 1 and absent from check 4, and
# the summary line says so rather than reporting 32/40 as if it were the whole
# thing. tools/verify/verify_pose.py needs a checkpoint this machine does not
# have either.
#===----------------------------------------------------------------------===//

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))
import npu_ops  # noqa: E402  -- needs the path above

sys.path.insert(0, str(REPO / "tools"))
import gen_npu_ops_doc  # noqa: E402  -- the generator NPU_OPS.md must match

BIN = REPO / "runtime" / "build" / "npuembeddings"
HPP = REPO / "runtime" / "include" / "common" / "npu_ops_flag.hpp"
EXPORTER = REPO / "tools" / "export" / "export_gemm_rtp.py"
PY = REPO / ".venv" / "bin" / "python"
DOC = REPO / "NPU_OPS.md"

# The tally, pinned. These are not decorative: they are the counts a reader of
# NPU_OPS.md is looking at, and they were measured one cell at a time (see each
# cell's reason in the registry). Changing one means a cell's status changed,
# which is a behaviour change -- so this gate should have to be edited on purpose
# rather than a number quietly moving under a document that still says 14.
EXPECTED_COUNTS = {"honours": 14, "on_array": 1, "unimplemented": 5,
                   "blocked": 3, "absent": 17}

# One container per architecture row, and the command line that reaches it.
# `kind` is npu_targets.json's word; the row in the registry is looked up by it,
# except for gemma which is a MODEL override rather than a kind -- so each entry
# names which of the two it is, because a gate that guessed would silently check
# the wrong row.
#   argv(...) is the part before the container, with one {} for the input.
FIXTURES = {
    "gemm_rtp": dict(kind="gemm_rtp", target=None,
                     container="models/bge-base-en-v1.5.npue",
                     artifacts="runtime/artifacts/bge-base-en-v1.5",
                     argv=["embed"], input="txt"),
    "stt": dict(kind="stt", target=None,
                container="models/whisper-base.npue",
                artifacts="runtime/artifacts/whisper-base",
                argv=["transcribe"], input="wav"),
    "cls": dict(kind="cls", target=None,
                container="models/vit-base-patch16-224.npue",
                artifacts="runtime/artifacts/vit-base-patch16-224",
                argv=["classify"], input="image"),
    "embeddinggemma-300m": dict(kind="gemm_rtp",
                                target="embeddinggemma-300m",
                                container="models/embeddinggemma-300m.npue",
                                artifacts="runtime/artifacts/"
                                          "embeddinggemma-300m",
                                argv=["embed"], input="txt"),
}
# Reported as skipped, not silently absent.
UNRUNNABLE = {
    "pose": "no pose container in this checkout (models/ has none); "
            "tools/verify/verify_pose.py needs a checkpoint this machine "
            "does not have either",
}

# The cells the runtime must let RUN. Only `honours`: `on_array` is the case
# this file's author got wrong first and the reason is worth keeping --
# `on_array` means the work is ALREADY dispatched, so the code has nothing to
# select, and accepting it would be the failure this project treats as worst (a
# requested op that quietly did nothing). The runtime refuses cls/conv and says
# the patch embedding rides attn_out's slot already; the gate asserts the
# refusal, and `on_array` is therefore on the REFUSED side with everything else.
ACCEPTS = (npu_ops.HONOURS,)

# The C++ table's rows. `{...}` literals, three strings each, and the source
# comments contain braces too, so the pattern is deliberately narrow: three
# double-quoted runs with no brace or quote inside.
CPP_ROW = re.compile(r'\{"([a-z0-9]+)",\s*"([a-z0-9]*)",\s*"([^"]+)"\}')


def cpp_table():
    """(code, design, long_name) in source order, from npu_op_table() only.

    Scoped to the function body rather than the whole header: the header also
    holds the refusal tables, which are pairs of strings and would otherwise be
    read as a row with an empty design and a bogus long name.
    """
    text = HPP.read_text()
    start = text.index("inline const std::vector<NpuOp> &npu_op_table()")
    end = text.index("return table;", start)
    return CPP_ROW.findall(text[start:end])


def fixtures():
    """Where the input files are, or None when a fixture is not on this machine.

    Deliberately explicit paths under /tmp rather than repo files: the gate needs
    one line of text, one wav and one jpeg, none of which belong in a repository
    of source, and inventing a fixture by hand would be a second thing that can
    be wrong. A missing fixture makes the row print why and skip.
    """
    return {
        "txt": Path("/tmp/opencode/t.txt"),
        "wav": Path("/home/prof/jfk.wav"),
        "image": Path("/tmp/opencode/bus.jpg"),
    }


def design_set_dir(spec):
    """The generation directory under an artifacts root, or None.

    `runtime/artifacts/<model>/artifacts_npu1/` is the layout, and the trailing
    `npu1` is the GENERATION -- a machine with an NPU2 set on disk has
    artifacts_npu2 instead, and one built for both has both. Globbing beats
    hard-coding: a hard-coded path fails on every other machine with an error
    that reads as a missing export rather than as the wrong directory. Two
    directories is ambiguous for a machine that has both, so that skips with a
    reason instead of picking one and pretending.
    """
    root = REPO / spec["artifacts"]
    if not root.is_dir():
        return None
    gens = sorted(p for p in root.glob("artifacts_npu*") if p.is_dir())
    if len(gens) == 1:
        return gens[0]
    if not gens:
        return root if (root / "gemm_rtp" / "design.json").exists() else None
    return "AMBIGUOUS"


def main() -> int:
    bad = 0
    kinds = list(npu_ops.KIND_REGISTRY)
    arches = kinds + list(npu_ops.MODEL_REGISTRY)

    print(f"NpuEmbeddings -- op matrix gate")
    print(f"  registry     {len(npu_ops.OPS)} codes x {len(arches)} "
          f"architectures = {len(npu_ops.OPS) * len(arches)} cells")

    # --- 1. the registry's own shape ---------------------------------------
    if len(arches) != 5 or len(npu_ops.OPS) != 8:
        print(f"  FAIL  {len(npu_ops.OPS)} codes x {len(arches)} architectures "
              f"is not the 8 x 5 the document is built around")
        bad += 1
    total = {}
    for label in arches:
        row = npu_ops.registry_for(None, label) if label in \
            npu_ops.MODEL_REGISTRY else npu_ops.KIND_REGISTRY[label]
        missing = [c for c in npu_ops.OPS if c not in row]
        if missing:
            print(f"  FAIL  {label}: no entry for {missing}")
            bad += 1
        for status, _ in row.values():
            total[status] = total.get(status, 0) + 1
        for code, (status, reason) in row.items():
            if status not in npu_ops.STATUSES:
                print(f"  FAIL  {label}/{code}: status '{status}' is not one of "
                      f"{list(npu_ops.STATUSES)}")
                bad += 1
            if not reason.strip():
                print(f"  FAIL  {label}/{code}: empty reason -- a cell with no "
                      f"reason is a claim, not a table")
                bad += 1
    if total != EXPECTED_COUNTS:
        print(f"  FAIL  tally {total} != the pinned {EXPECTED_COUNTS}")
        bad += 1
    else:
        print(f"  ok    tally {total}")

    stream_only = {c for c, (d, _n) in npu_ops.OPS.items() if not d}
    if stream_only != npu_ops.STREAM_ONLY:
        print(f"  FAIL  codes with no design directory {sorted(stream_only)} "
              f"!= STREAM_ONLY {sorted(npu_ops.STREAM_ONLY)}")
        bad += 1
    else:
        print(f"  ok    STREAM_ONLY is exactly the {len(stream_only)} codes "
              f"with no directory of their own")
    if npu_ops.RUNTIME_FLAG != npu_ops.EXPORTER_FLAG:
        print("  FAIL  RUNTIME_FLAG and EXPORTER_FLAG differ")
        bad += 1

    # --- 2. NPU_OPS.md is generated from the registry -----------------------
    rc = subprocess.run([sys.executable, str(REPO / "tools" /
                                             "gen_npu_ops_doc.py"), "--check"],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"  FAIL  {DOC.name} is stale -- run tools/gen_npu_ops_doc.py")
        for line in rc.stdout.splitlines()[:40]:
            print("   " + line)
        bad += 1
    else:
        print(f"  ok    {DOC.name} matches the registry")

    # --- 3. the runtime's C++ table says the same thing ---------------------
    cpp = cpp_table()
    mine = [(c, d, n) for c, (d, n) in npu_ops.OPS.items()]
    if cpp != mine:
        print(f"  FAIL  npu_op_table() in {HPP.name} differs from OPS")
        print(f"        cpp   {cpp}")
        print(f"        python {mine}")
        bad += 1
    else:
        print(f"  ok    npu_op_table() matches OPS: {len(cpp)} codes, same "
              f"order, same design directory, same long name")

    # --- 4. the runtime accepts or refuses each cell -----------------------
    fx = fixtures()
    ran = skipped = 0
    for label in arches:
        spec = FIXTURES.get(label)
        if label in UNRUNNABLE or spec is None:
            why = UNRUNNABLE.get(
                label, "no container wired up for this row yet")
            print(f"  skip  {label}: {why}")
            skipped += 1
            continue
        row = npu_ops.registry_for(None, label) if label in \
            npu_ops.MODEL_REGISTRY else npu_ops.KIND_REGISTRY[label]
        gens = design_set_dir(spec)
        missing = [] if (REPO / spec["container"]).exists() else \
            [f"container {spec['container']}"]
        if gens is None:
            missing.append(f"no design set under {spec['artifacts']}")
        elif gens == "AMBIGUOUS":
            missing.append(f"{spec['artifacts']} holds more than one generation; "
                           f"say which with --artifacts")
        if spec["input"] not in fx or not fx[spec["input"]].exists():
            missing.append(f"input fixture ({fx[spec['input']]})")
        if missing:
            print(f"  skip  {label}: not on this machine -- {', '.join(missing)}")
            skipped += 1
            continue
        cmd = [str(BIN)] + spec["argv"] + [str(REPO / spec["container"]),
                                           str(fx[spec["input"]]),
                                           "--artifacts", str(gens)]
        for code in npu_ops.OPS:
            status, _reason = row[code]
            proc = subprocess.run(cmd + ["--npu-ops", code],
                                  capture_output=True, text=True, timeout=900)
            err = proc.stderr
            ran += 1
            if status in ACCEPTS:
                # Only the exit code is asserted. What an accepted run PRINTS is
                # not this gate's business: a run that honours a code says so in
                # a status line whose wording changes, and freezing it here
                # would make a prose edit look like a behaviour change. That a
                # passing run also produced the RIGHT numbers is the job of
                # verify_vit / verify_whisper / verify_tail.
                if proc.returncode != 0:
                    print(f"  FAIL  {label}/{code} is '{status}' but the runtime "
                          f"refused it (rc {proc.returncode})")
                    print("   " + (err.strip().splitlines() or
                                   ["<no stderr>"])[0][:160])
                    bad += 1
            else:
                if proc.returncode == 0:
                    print(f"  FAIL  {label}/{code} is '{status}' but the "
                          f"runtime ACCEPTED it and ran -- the registry and the "
                          f"binary disagree about what exists")
                    bad += 1
                elif code not in err:
                    print(f"  FAIL  {label}/{code}: refused, but the message "
                          f"never names the code, so the user cannot tell which "
                          f"of the eight was rejected")
                    print("   " + err.strip().splitlines()[0][:160])
                    bad += 1
    print(f"  ran   {ran} cells against {BIN.name}" if ran else
          "  ran   no cells (nothing to run against)")

    if skipped:
        print(f"  skip  {skipped} architecture row(s) not run here -- the cells "
              f"are counted above and were NOT checked against the binary")

    if bad:
        print(f"FAIL -- {bad} disagreement(s) between the registry, the "
              f"generated document, the C++ table and what the runtime does")
        return 1
    print("PASS -- the registry, NPU_OPS.md, the C++ table and the runtime all "
          "say the same thing about every cell that could be run")
    return 0


if __name__ == "__main__":
    sys.exit(main())