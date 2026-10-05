#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Gate for the op registry: 8 codes x 7 architectures = 56 cells.
#
# tools/lib/npu_ops.py is the registry. It is not prose and it cannot be checked
# by reading it, so this file checks the four things that can be checked:
#
#   1. the registry's own shape -- 40 cells, five statuses, one tally, and the
#      set of codes with no design directory equal to STREAM_ONLY;
#   2. NPU_OPS.md and NPU_OPS.ru.md (architectures x codes) and NPU_MODELS.md
#      and NPU_MODELS.ru.md (models x codes), all GENERATED from the registry
#      and npu_targets.json, are not stale -- and all 40 reasons are translated,
#      so a Russian reader is never left with one English cell among forty
#      Russian ones;
#   2b. the model tables' cells agree with the architecture tables', except for
#      the gated FFNs, which is the ONE difference the model table exists to
#      record -- so a difference appearing there is a bug, not news;
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
import gen_npu_ops_doc  # noqa: E402  -- the generators NPU_OPS*.md must match
import gen_npu_models_doc  # noqa: E402  -- ... and NPU_MODELS*.md

BIN = REPO / "runtime" / "build" / "npuembeddings"
HPP = REPO / "runtime" / "include" / "common" / "npu_ops_flag.hpp"
EXPORTER = REPO / "tools" / "export" / "export_gemm_rtp.py"
PY = REPO / ".venv" / "bin" / "python"
# Both generated documents. Kept as a list rather than one name because there are
# two languages now and the gate has to check both -- a single DOC constant would
# have made the Russian one silently unchecked.
DOCS = (REPO / "NPU_OPS.md", REPO / "NPU_OPS.ru.md")

# The tally, pinned. These are not decorative: they are the counts a reader of
# NPU_OPS.md is looking at, and they were measured one cell at a time (see each
# cell's reason in the registry). Changing one means a cell's status changed,
# which is a behaviour change -- so this gate should have to be edited on purpose
# rather than a number quietly moving under a document that still says 14.
EXPECTED_COUNTS = {"honours": 19, "on_array": 1, "unimplemented": 0,
                   "blocked": 5, "absent": 31}
# blocked went 4 -> 5 and absent 24 -> 31 when arch=8 was added, and that is the
# whole of the change: the new row contributes exactly eight cells, seven of them
# `absent` and one `blocked`, for the same reason arch=7's seven-and-one is --
# there is no design set carrying its convolutions. `honours` and `on_array` did
# not move, which is the check worth making: a new architecture that quietly
# started claiming dispatched work would move them, and this gate is what says so.
# Why these numbers are 19 and 0 rather than 14 and 5: five cells were flipped
# from `unimplemented` to `honours` when the branches were written --
# gemm_rtp/attn, cls/attn, embeddinggemma-300m/attn, cls/softm and
# embeddinggemma-300m/softm. Each was verified against the host before it was
# counted (see NPU_OPS.md's per-cell reason), and section 4 below then runs
# every cell of every row this harness CAN reach, so a cell that says `honours`
# and does not dispatch fails this same gate. It cannot reach all 48: the
# `hands` row is in UNRUNNABLE with the reason, and rows without a container or
# a design set on this machine are skipped and counted separately rather than
# quietly passed. The pin is edited ON PURPOSE, with the cells named, because
# the alternative -- a number moving under a document that still says 14 -- is
# the failure the comment above describes.
#
# The move from 40 cells to 48 is arch=7 joining as a KIND, which is eight new
# cells and nothing else: honours is UNCHANGED at 19, so no row that claimed to
# work stopped working. The eight are one `blocked` (hands/conv -- the
# convolutions are GEMM-shaped and there is no design set built for them, which
# is a missing artefact rather than missing code) and seven `absent` (no audio
# front end, no vocabulary, no normalisation, no attention, and the activation
# is ReLU/ReLU6/PReLU fused into the convolution's epilogue rather than a GELU
# pass). Each of the seven names why, and the reason is the claim.

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
                                # `--prefix query` in `tail`, because this
                                # container carries task prefixes and refuses to
                                # pick one on its own. Without it EVERY cell of
                                # this row exits 2 at the prefix check, so the
                                # six refused cells passed for a reason that was
                                # not theirs and the two honoured cells failed
                                # for one that was not theirs either -- a gate
                                # that measures the wrong refusal is worse than
                                # no gate. The value is one of the container's
                                # own listed prefixes; it goes in `tail` and not
                                # `argv` because the reader takes the model as
                                # the first non-flag argument.
                                argv=["embed"], tail=["--prefix", "query"],
                                input="txt"),
    # Pose reads the SAME two command-line words as every other row -- `pose` then
    # the container then the image -- so there is nothing pose-shaped about the
    # invocation. What was missing was this entry, and with it the whole row was
    # skipped while its container, its design set and its image fixture all sat on
    # this machine. The skip message it printed ("models/ has none") was false.
    "pose": dict(kind="pose", target=None,
                 container="models/yolov8n-pose.npue",
                 artifacts="runtime/artifacts/yolov8n-pose",
                 argv=["pose"], input="image"),
    # arch=7. Present even though section 4 skips it, because a fixture is a
    # declaration of what the row IS and a skip is a statement about this
    # machine -- keeping them apart is what lets the skip print a true reason
    # instead of "no container wired up for this row yet". Its `artifacts` is a
    # path that does not exist and is not meant to: hands_mode refuses
    # --artifacts by name, so there is no directory for it to name.
    "hands": dict(kind="hands", target=None,
                  container="models/mediapipe-hands/hands.npue",
                  artifacts="runtime/artifacts/mediapipe-hands",
                  argv=["hands"], input="image"),
    # arch=8, for the same reason and with the same shape: a declaration of what
    # the row IS, plus the UNRUNNABLE reason below that says why no cell of it
    # can be reached here. Its `artifacts` is a path that does not exist and is
    # not meant to -- mppose_mode refuses --artifacts by name, so there is no
    # directory for it to name. The verb is `mppose` and NOT a second spelling of
    # `pose`: two architectures can both be called pose and share no flag, no
    # threshold and no head.
    "mppose": dict(kind="mppose", target=None,
                   container="models/mediapipe-pose/mppose.npue",
                   artifacts="runtime/artifacts/mediapipe-pose",
                   argv=["mppose"], input="image"),
}
# Reported as skipped, not silently absent. `hands` is the first entry and it is
# NOT one of the two kinds of skip this file used to have: every other row has a
# container, a design set and a fixture on disk and skips only when the machine
# lacks one of them, whereas this row has no array path to be lacking. The
# distinction matters because the reader is left with one number either way.
#
# The comment that used to sit here claimed this dict was EMPTY and that was the
# point -- it had held `pose` with the reason "no pose container in this checkout
# (models/ has none)", a claim that stopped being true the moment the container
# was packed and which nothing checked, because the gate only ever prints these
# strings. It is not empty now, and the reason is written where it can be read
# rather than asserted here.
UNRUNNABLE: dict[str, str] = {
    "hands": "this architecture has no array path at all, so this harness "
             "cannot reach it anywhere rather than not-on-this-machine. "
             "runtime/include/runtime/hands_mode.hpp REFUSES --npu-ops conv "
             "and --artifacts by name, and both of its 61 dense convolutions "
             "pad down to 12 shapes that no design under runtime/artifacts/ "
             "carries. Its one `blocked` cell is that same fact; had it been "
             "written `honours` there would be nothing here to run either, "
             "which is the failure the status exists to prevent.",
    "mppose": "the same fact as hands, and for the same reason: this "
              "architecture has no array path, so this harness cannot reach any "
              "cell of it rather than not-on-this-machine. "
              "runtime/include/runtime/mppose_mode.hpp REFUSES --npu-ops conv "
              "and --artifacts by name, and its 99 dense convolutions pad down "
              "to 22 shapes, which no design under runtime/artifacts/ carries "
              "and which -- unlike hands' twelve and pose's fourteen -- has "
              "never been built at all. Its one `blocked` cell is that same "
              "fact. The row is here rather than absent for hands' reason too: "
              "a declaration of what the row IS is not a statement about this "
              "machine, and keeping the two apart is what lets the skip print a "
              "true reason.",
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
    if len(arches) != 7 or len(npu_ops.OPS) != 8:
        print(f"  FAIL  {len(npu_ops.OPS)} codes x {len(arches)} architectures "
              f"is not the 8 x 7 the document is built around. Adding a row here "
              f"means adding it to npu_ops.KINDS, to the model's kind in "
              f"tools/data/npu_targets.json and to ARCHES below, and every one "
              f"of those is a place the row can be half-added.")
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
    # Zero entries dropped on both sides: a status with no cells at all is not
    # a KEY in the tally, so comparing a dict that lacks `unimplemented` against
    # one that carries it as 0 would fail forever while saying nothing. The
    # counts that a reader of NPU_OPS.md sees are the non-zero ones anyway.
    total_nz = {k: v for k, v in total.items() if v}
    pinned_nz = {k: v for k, v in EXPECTED_COUNTS.items() if v}
    if total_nz != pinned_nz:
        print(f"  FAIL  tally {total_nz} != the pinned {pinned_nz}")
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

    # --- 2. both documents are generated from the registry ------------------
    # Two files, not one: the Russian one carries a translation of each reason,
    # kept in the generator rather than in the registry (the registry is imported
    # by the exporter and has no business carrying two languages). The
    # completeness of that translation is asserted here, because a cell left in
    # English reads to a Russian reader as an oversight rather than as a hole.
    missing = [f"{lbl}/{c}" for lbl in arches for c in npu_ops.OPS
               if (lbl, c) not in gen_npu_ops_doc.REASONS_RU]
    if missing:
        print(f"  FAIL  NPU_OPS.ru.md: {len(missing)} cell(s) with no Russian "
              f"reason -- {missing[:4]}")
        bad += 1
    else:
        print(f"  ok    all {len(npu_ops.OPS) * len(arches)} reasons "
              f"translated into NPU_OPS.ru.md")
    rc = subprocess.run([sys.executable, str(REPO / "tools" /
                                             "gen_npu_ops_doc.py"), "--check"],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print("  FAIL  a generated document is stale -- run "
              "tools/gen_npu_ops_doc.py")
        for line in rc.stdout.splitlines()[:40]:
            print("   " + line)
        bad += 1
    else:
        for line in rc.stdout.splitlines():
            print("  " + line)

    # --- 2b. the model tables agree with the architecture tables -----------
    # Every cell of a model's row must be its kind's cell, except for the gated
    # FFNs. That is not a tautology today -- the model generator applies the
    # gated rule itself -- it is a statement about what the model table is FOR:
    # the gated FFN is the only reason it exists, so if a second kind of
    # difference turns up there, it has to be a deliberate decision recorded
    # somewhere, not a silent divergence. `gated` is spelled out rather than
    # listed, because the check is "differs from the kind" and gated cells are
    # exactly the ones that do.
    gen_npu_models_doc.load()
    unexpected, gated_cells = [], 0
    catalogue = gen_npu_models_doc.TARGETS_MODEL
    for model, spec in catalogue.items():
        kind = gen_npu_models_doc.kind_of(spec)
        row = npu_ops.registry_for(kind, model)
        for code in npu_ops.OPS:
            mc = gen_npu_models_doc.cell(model, code)
            if mc == gen_npu_models_doc.GATED:
                gated_cells += 1
            elif mc != row[code][0]:
                unexpected.append(f"{model}/{code}: model says {mc!r}, kind "
                                  f"{kind!r} says {row[code][0]!r}")
    if unexpected:
        print(f"  FAIL  {len(unexpected)} model cell(s) differ from their kind "
              f"without the gated-FFN reason -- {unexpected[:3]}")
        bad += 1
    else:
        print(f"  ok    all {len(catalogue) * len(npu_ops.OPS)} model cells "
              f"equal their kind's, except {gated_cells} gated-FFN cell(s)")

    rc = subprocess.run([sys.executable, str(REPO / "tools" /
                                             "gen_npu_models_doc.py"),
                         "--check"], capture_output=True, text=True)
    if rc.returncode != 0:
        print("  FAIL  a generated model document is stale -- run "
              "tools/gen_npu_models_doc.py")
        for line in rc.stdout.splitlines()[:40]:
            print("   " + line)
        bad += 1
    else:
        for line in rc.stdout.splitlines():
            print("  " + line)

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
        # Flags that have to come AFTER the positionals, because the argument
        # reader takes the model as the first non-flag argument and stops
        # collecting positionals at the first flag. `argv` therefore holds only
        # the verb; `tail` holds what this particular container needs and cannot
        # be given earlier. It is appended here rather than folded into argv so
        # the distinction stays visible: a fixture that put a flag in `argv` for
        # a container that reads positionals first would fail as "needs a model
        # name", which names the wrong thing entirely.
        cmd += spec.get("tail", [])
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
    print("PASS -- the registry, all four generated documents, the C++ table and "
          "the runtime all say the same thing about every cell that could be "
          "run")
    return 0


if __name__ == "__main__":
    sys.exit(main())