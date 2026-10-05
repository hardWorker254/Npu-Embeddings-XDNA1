#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#===----------------------------------------------------------------------===//
# The way in. One command that answers "what do I run, and what do I need
# first", because tools/ is thirty-odd entry points and the answer used to live
# in BUILD.md's prose.
#
#   python tools/pipeline.py                 # the map: task -> command
#   python tools/pipeline.py list            # every tool, what it needs
#   python tools/pipeline.py list gates      # only the gates
#   python tools/pipeline.py check           # what this machine is missing
#   python tools/pipeline.py run <tool> ...  # check, then run it
#   python tools/pipeline.py gates           # the gates, cheapest first
#
# It runs nothing by itself. Every command either prints or execs the real tool
# with your arguments, so `run` is a preflight in front of
# `python tools/<bucket>/<tool>.py` and never a reimplementation of it -- the
# tools own their own flags, and this file is the map to them, not a second
# CLI over them.
#
# Stdlib only, deliberately: `check` has to work in an interpreter that cannot
# import numpy, or it cannot report that numpy is the thing missing.
#===----------------------------------------------------------------------===//

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS / "lib"))

from design_sets import design_sets                                # noqa: E402

# ---------------------------------------------------------------------------
# The inventory. One row per entry point, and the row is the only place in the
# repository that has to know all of them: the buckets say WHAT a tool is, this
# says what it needs before it will run, which is the question nobody could
# answer by looking at a directory listing.
#
# needs: names checked by `check` and by `run` before exec. A missing one is a
#        refusal with the command that fixes it, not a traceback.
# ---------------------------------------------------------------------------
TOOLSET = [
    # name, path, one line, needs
    ("pack_npue", "pack/pack_npue.py",
     "HuggingFace checkpoint -> pre-tiled, pre-fused .npue",
     ("numpy", "checkpoint")),
    ("export_gemm_rtp", "export/export_gemm_rtp.py",
     "the gemm_rtp design set (final.xclbin + instruction streams)",
     ("iron",)),
    ("export_eltwise", "export/export_eltwise.py",
     "the gelu / layernorm / softmax design sets",
     ("iron",)),
    ("export_validation", "export/export_validation.py",
     "the runtime's golden check vectors",
     ("numpy", "container")),
    ("gen_tokenizer_tables", "gen/gen_tokenizer_tables.py",
     "Unicode tables for the WordPiece tokenizer -> runtime/include/",
     ()),
    ("gen_xlmr_unicode_tables", "gen/gen_xlmr_unicode_tables.py",
     "the same for XLM-R",
     ()),
    ("gen_whisper_unicode_tables", "gen/gen_whisper_unicode_tables.py",
     "the same for Whisper (L/N/S as range tables)",
     ()),
    ("make_tail_reference", "gen/make_tail_reference.py",
     "the fp32 reference the tail gate measures against",
     ("torch", "transformers")),

    ("verify_onnx_reader", "verify/verify_onnx_reader.py",
     "the ONNX weight reader vs onnx, vs the goldens, and vs a broken copy",
     ("numpy", "onnx", "checkpoint")),
    ("verify_npue", "verify/verify_npue.py",
     "the .npue: spec, bit-exact round trip, stale layout, goldens",
     ("numpy", "container", "reference")),
    ("verify_pack_parity", "verify/verify_pack_parity.py",
     "the Python and C++ packers agree byte for byte",
     ("numpy", "checkpoint", "g++")),
    ("verify_i8_scheme", "verify/verify_i8_scheme.py",
     "the int8 scheme against itself, and seven injected faults",
     ("numpy",)),
    ("verify_i8_kernels", "verify/verify_i8_kernels.py",
     "the runtime's int8 host kernels, AVX2 vs scalar, byte for byte",
     ("g++",)),
    ("verify_i4_scheme", "verify/verify_i4_scheme.py",
     "the int4 scheme, ten injected faults, and the C++ decode byte for byte",
     ("numpy", "g++")),
    ("verify_design_numerics", "verify/verify_design_numerics.py",
     "does a design set COMPUTE what it claims to (needs the NPU)",
     ("numpy", "npu", "designs")),
    # The registry gate. It pins what all 40 (code x architecture) cells say --
    # the registry's own shape, all four GENERATED documents against it, the
    # C++ table against the Python one -- and then runs every cell against the
    # binary, so a cell that says `honours` and does not dispatch fails it.
    # The needs are the union of those two halves: the documents need no
    # device, the 40 runs need the built binary, a container and a design set.
    ("verify_npu_op_matrix", "verify/verify_npu_op_matrix.py",
     "every code against every architecture: registry, docs, C++ table, binary",
     ("npu", "runtime", "container", "designs")),
    ("verify_whisper_tokenizer", "verify/verify_whisper_tokenizer.py",
     "the Whisper tokenizer three ways: C++, reference, HuggingFace",
     ("numpy", "transformers")),
    ("verify_whisper_features", "verify/verify_whisper_features.py",
     "the C++ audio front end against WhisperFeatureExtractor",
     ("numpy", "torch", "transformers")),
    ("verify_whisper_model", "verify/verify_whisper_model.py",
     "the Whisper NPU stacks against transformers, step by step",
     ("numpy", "torch", "transformers", "npu", "container", "designs")),
    ("verify_whisper_cli", "verify/verify_whisper_cli.py",
     "the transcribe CLI and the endpoint, against transformers",
     ("numpy", "transformers", "runtime")),
    ("verify_whisper", "verify/verify_whisper.py",
     "word error rate against a HUMAN transcript (no audio is committed)",
     ("numpy", "torch", "transformers", "runtime")),
    ("verify_semantics", "verify/verify_semantics.py",
     "near/far ordering, no reference and no tolerance",
     ("runtime",)),
    ("verify_tail", "verify/verify_tail.py",
     "the p99 tail ratchet, per model and datapath",
     ("runtime",)),
    ("verify_endpoint", "verify/verify_endpoint.py",
     "the endpoint driven by the official OpenAI client",
     ("numpy", "openai", "runtime")),
    # No third-party requirements: stdlib urllib only, and it starts and stops the
    # server itself. In the same tier as verify_endpoint because it answers a
    # question ABOUT that endpoint -- which container answers on which path -- and
    # that gate only ever looks at one of the four.
    ("verify_serve_dispatch", "verify/verify_serve_dispatch.py",
     "serve's four endpoints: each container answers for its own path and refuses "
     "the rest, and the image endpoints refuse rather than fall back",
     ("runtime",)),
    ("verify_npue_nomic", "verify/verify_npue_nomic.py",
     "the arch=2 gate for the nomic container",
     ("numpy", "container", "reference")),
    # The four-spelling check for the conv-only stream sets, and the gate that
    # caught the encoder-set bug: it compares geometry.py, npu_targets.json, the
    # design set and the packed container, in both directions, for pose, hands
    # and mppose. It is in the `container` tier rather than `cheap` because one of
    # the four is a container and the whole claim is that all four agree.
    #
    # It takes a design directory and a container as arguments, so unlike every
    # other gate here it cannot be run without naming both. That is stated rather
    # than defaulted: a design set this machine cannot build is not a thing to
    # synthesise, and the toolchain that builds one is not installed here.
    ("verify_conv_streamset", "verify/verify_pose_streamset.py",
     "the conv-only stream sets against the targets file, a design set and the "
     "packed container, for pose, hands and mppose",
     ("numpy", "container", "designs")),
    ("verify_vit", "verify/verify_vit.py",
     "the arch=5 gate: pool + attention + pre-LN + head, both schemes",
     ("numpy", "torch", "transformers", "container", "checkpoint")),
    ("verify_vit_image", "verify/verify_vit_image.py",
     "the C++ image front end against PIL, per resample code",
     ("numpy", "pillow", "g++", "container")),
    ("verify_vit_model", "verify/verify_vit_model.py",
     "the host-only arch=5 half: geometry, the head's stride, the int8 kernels",
     ("numpy", "g++", "container")),
    # No requirements at all: it reads the runtime's own sources and the CLI's
    # own table, which is the point of putting it in this tier -- it is the one
    # check that can say the refusal in cli.cpp is wrong before anything is built.
    # The arch=7 host gate. No NPU, no design set, no binary: it reads the
    # container and the two checkpoints and runs the whole two-network pipeline
    # in numpy, against onnxruntime and against the geometry numbers recorded
    # from the OpenCV zoo. `--inject` breaks the code fourteen ways and checks
    # that every break fails it.
    ("verify_hands", "verify/verify_hands.py",
     "the arch=7 gate: two networks against ORT, the front end against the zoo",
     ("numpy", "onnx", "onnxruntime", "pillow", "hands_models")),
    # The arch=8 gate, and the same three questions in the same order: the packed
    # op list against ORT, the anchor table against an independent derivation of
    # the same pyramid, and the C++ runtime against the numbers recorded from the
    # zoo. It needs the BUILT BINARY -- where arch=7's section 4 can re-implement
    # the front end in numpy and compare, arch=8's third section runs the real
    # thing, because the residual that section measures (a resampler's) is only
    # measurable against the real resampler. So it sits in the `container` tier
    # and its environment list says what it needs.
    ("verify_mppose", "verify/verify_mppose.py",
     "the arch=8 gate: two networks against ORT, the anchors, and the C++ "
     "runtime against the zoo",
     ("numpy", "onnx", "onnxruntime", "pillow", "mppose_models", "runtime")),
    ("verify_cli_flags", "verify/verify_cli_flags.py",
     "the CLI's flag table against the flags the runtime reads off argv",
     ()),
    ("parity_exporters", "verify/parity_exporters.py",
     "the split exporters against the monoliths in git",
     ("git",)),

    ("gemm_pretiled_research", "research/gemm_pretiled_research.py",
     "presets, traces and wall-clock benchmarks for the GEMM library",
     ("iron", "npu")),
]

GATES = [t for t in TOOLSET if t[1].startswith("verify/")
         or t[0] == "parity_exporters"]

# What has to be on the machine, and the command that puts it there. Checked in
# order, and the first failure is the one reported: a machine with no numpy has
# a dozen problems and only one of them is worth printing.
ENVIRONMENTS = [
    ("numpy", "the packer, every container gate and both exporters",
     "pip install numpy"),
    ("onnx", "verify_onnx_reader.py -- the reference the ONNX reader is checked "
     "against; the reader itself never imports it",
     "pip install onnx"),
    ("torch", "the reference encoders and the Whisper gates",
     "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
    ("transformers", "the Whisper gates and the tokenizer references",
     "pip install transformers"),
    ("onnxruntime", "verify_hands.py -- the reference both networks are "
     "checked against; the gate's own arithmetic is numpy",
     "pip install onnxruntime"),
    ("pillow", "verify_hands.py and verify_vit_image.py -- the image decoders",
     "pip install pillow"),
    ("openai", "verify_endpoint.py only -- the official client, imported",
     "pip install openai"),
    ("aie.iron", "the exporters: MLIR-AIE, dot-source its env script first",
     ". C:\\dev\\mlir-aie\\iron_env.ps1   (or: source iron_env.sh)"),
    ("g++", "the two gates that compile C++ to compare it byte for byte",
     "apt install g++   (or: any C++17 compiler)"),
    ("git", "parity_exporters.py, which resolves its reference from history",
     "apt install git"),
]

# Artefacts, not packages: things a previous step in the build produced. Each
# is (name, how to test, the command that makes it).
ARTEFACTS = [
    ("checkpoint", "a HuggingFace checkpoint under models/",
     "the runtime fetches it on first use:  npuembeddings add <org/model>"),
    ("container", "a .npue under models/",
     "python tools/pipeline.py run pack_npue"),
    ("designs", "a design.json under runtime/artifacts/*/artifacts_npu*/",
     "python tools/pipeline.py run export_gemm_rtp --target <model> --arch 1"),
    ("runtime", "the npuembeddings executable",
     "cmake -S runtime -B runtime/build && cmake --build runtime/build"),
    ("hands_models", "the arch=7 checkpoint pair, the test photograph and "
     "hands.npue under models/mediapipe-hands/",
     "python tools/pipeline.py run pack_npue -- --hands-onnx "
     "models/mediapipe-hands --out models/mediapipe-hands/hands.npue"),
    # arch=8's pair, and the photograph the golden was recorded on. The
    # photograph is NOT in this list and is NOT meant to be fetched with the
    # checkpoints: it lives in docs/, which is tracked, because it is the golden's
    # own input and its sha256 is in the golden. Two ONNX files are here, the
    # container is here, and the golden is tracked -- the same split arch=7 has,
    # with the image moved because opencv_zoo does not ship raw input images at
    # all and this one came from elsewhere.
    ("mppose_models", "the arch=8 checkpoint pair and mppose.npue under "
     "models/mediapipe-pose/ (the golden's image is in docs/, not here)",
     "python tools/pipeline.py run pack_npue -- --mppose-onnx "
     "models/mediapipe-pose --out models/mediapipe-pose/mppose.npue"),
    ("npu", "an XRT device (xrt-smi sees it)",
     "the Ryzen AI driver; without it only the host-side gates can run"),
]

# The map. Deliberately short and in the order a build actually goes: this is
# what someone who has never opened this directory reads first.
MAP = [
    ("I want a model to RUN",
     "run pack_npue, then export_gemm_rtp, then build the runtime",
     "BUILD.md §2.2-2.4"),
    ("I changed a weight or a fusion",
     "run pack_npue, then verify_npue, then verify_pack_parity",
     "BUILD.md §2.5"),
    ("I changed a kernel or the dataflow",
     "run export_gemm_rtp, then verify_design_numerics (needs the NPU)",
     "BUILD.md §2.3, §2.5"),
    ("I want int8 (W8A8) instead of bf16",
     "pack_npue --int8, export_gemm_rtp --int8, then verify_i8_scheme, "
     "verify_i8_kernels, verify_npue",
     "tools/README.md § int8"),
    ("I want int4 (W4A8) -- half the weight bytes, same int8 array",
     "pack_npue --dtype i4 --int4-group 32, then verify_i4_scheme and "
     "verify_npue; no re-export, run it against an int8 design set",
     "tools/README.md § int4"),
    ("I changed the Whisper path",
     "verify_whisper_tokenizer, verify_whisper_features, verify_whisper_model",
     "BUILD.md §2.5"),
    ("I changed the image-classification path",
     "verify_vit_image (front end), verify_vit_model (host half), then "
     "verify_vit (the whole stack)",
     "BUILD.md §2.5"),
    ("I am about to ship",
     "run gates --only release",
     "BUILD.md §2.5, §2.6"),
    ("I touched tools/export/",
     "run parity_exporters",
     "tools/README.md"),
    ("I want to know what a tool does",
     "run list",
     ""),
    ("I am not sure what is missing here",
     "run check",
     ""),
]

# Which gates need what, so `gates --only` can promise an honest subset: the
# ones below run with nothing but this interpreter and a C++ compiler.
GATE_TIERS = {
    # "cheap" is the promise this tier makes: nothing but this interpreter and a
    # C++ compiler, no checkpoint and no NPU. verify_vit_image and
    # verify_vit_model earn their place here because they are the only gates on
    # the image path that need neither -- they read a container, but not a
    # design set and not a device. verify_i4_scheme compiles its own probe from
    # runtime/src/model.cpp for the same reason verify_i8_kernels does: the
    # arithmetic it has to hold apart lives in a header, and only a compile can
    # show that header and npue.fold_i4() still agree.
    "cheap": ("verify_i8_scheme", "verify_i8_kernels", "verify_i4_scheme",
              "parity_exporters", "verify_vit_image", "verify_vit_model",
              # Reads no container and needs no device: it compares the CLI's flag
              # table against the flags the sources actually read, which is the
              # one check that can run before anything is built. The
              # unrecognised-option refusal in cli.cpp is written against that
              # table, so a flag the runtime reads but the table has not heard of
              # is a command that fails on a flag it spelled correctly -- which
              # is how it broke 21 of them, including the one verify_pack_parity
              # passes and the one tools/release_benchmark.ps1 passes.
              "verify_cli_flags"),
    # "container" reads checkpoints and containers. verify_onnx_reader belongs
    # here rather than in "cheap" because both of its claims need one: the
    # per-model read needs an ONNX file, and the golden that proves the reader
    # returns the RIGHT tensor needs the model directory beside it.
    # verify_hands is here rather than in "npu" because it needs none of that:
    # the claim it makes is about the CONTAINER and the front end, both of which
    # are host-side, and a gate that needs a device to say a convolution is
    # packed correctly cannot be run by the person who packed it.
    "container": ("verify_onnx_reader", "verify_npue", "verify_pack_parity",
                  "verify_npue_nomic", "verify_vit", "verify_hands",
                  "verify_mppose", "verify_conv_streamset"),
    # verify_npu_op_matrix is here rather than in "cheap" for the same reason
    # verify_design_numerics is: it runs every registry cell against the BINARY,
    # so it needs the build and the device, and its other claims (the registry's
    # shape, all four generated documents, the C++ table) are the ones that keep
    # NPU_OPS*.md and NPU_MODELS*.md from drifting from the code. A gate that
    # pins what every cell says about every architecture and is not in the list
    # that gets run is a gate that is only run by hand, which is not a gate.
    "npu": ("verify_design_numerics", "verify_whisper_model",
            "verify_npu_op_matrix"),
    "whisper": ("verify_whisper_tokenizer", "verify_whisper_features",
                "verify_whisper_cli", "verify_whisper"),
    # verify_serve_dispatch is here rather than under "release" because it needs no
    # openai client and no human transcript: it starts and stops its own servers.
    # What it does need is the built binary, so it is not in "cheap".
    "release": ("verify_semantics", "verify_tail", "verify_endpoint",
                "verify_serve_dispatch"),
}
# The order a gate list is printed and run in. Cheapest and most local first:
# a container fault is cheaper to find here than as a wrong embedding later.
GATE_ORDER = [n for names in GATE_TIERS.values() for n in names]


def tool(name):
    for t in TOOLSET:
        if t[0] == name:
            return t
    return None


def command_for(name):
    """The command as the user types it, with `python` and the repo's own
    interpreter spelled the way BUILD.md spells them."""
    return f"python tools/{tool(name)[1]}"


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _module_present(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _have(kind):
    """(ok, remedy) for one requirement. Probes are cheap and side-effect
    free -- an import of a module that is present but broken is not this
    file's problem to diagnose."""
    if kind == "numpy":
        return _module_present("numpy"), "pip install numpy"
    if kind == "torch":
        return _module_present("torch"), "pip install torch"
    if kind == "transformers":
        return _module_present("transformers"), "pip install transformers"
    # These three were listed as needs by gates that use them before this file
    # knew how to ask. An unknown kind falls through to "present", so the check
    # said yes to an interpreter that then died on the import -- which is the
    # one thing `check` exists to prevent.
    if kind == "onnx":
        return _module_present("onnx"), "pip install onnx"
    if kind == "onnxruntime":
        return _module_present("onnxruntime"), "pip install onnxruntime"
    if kind == "pillow":
        return _module_present("PIL"), "pip install pillow"
    if kind == "openai":
        return _module_present("openai"), "pip install openai"
    if kind == "iron":
        return _module_present("aie.iron"), "dot-source MLIR-AIE's env script"
    if kind == "git":
        return shutil.which("git") is not None, "install git"
    if kind == "g++":
        return any(shutil.which(c) for c in ("g++", "clang++", "cl")), \
            "install a C++17 compiler"
    if kind == "hands_models":
        # The arch=7 pair, both float files, and the photograph the gate's
        # geometry numbers were recorded on. Named separately from "container"
        # because a container can exist with the checkpoints gone, and the gate
        # then cannot say whether it is holding this model or an empty shell.
        d = REPO / "models" / "mediapipe-hands"
        want = ["palm_detection_mediapipe_2023feb.onnx",
                "handpose_estimation_mediapipe_2023feb.onnx",
                "hand_plain.png", "hands.npue"]
        missing_ = [w for w in want if not (d / w).exists()]
        return (not missing_), (
            f"missing in models/mediapipe-hands/: {', '.join(missing_)}\n"
            f"  fetch the two ONNX files from opencv/palm_detection_mediapipe and "
            f"opencv/handpose_estimation_mediapipe (Apache-2.0) and hand_plain.png "
            f"from opencv_zoo. The golden is NOT in that list: it is tracked, at "
            f"reference/goldens/hands_mediapipe.json. Then "
            f"`python tools/pipeline.py run pack_npue -- "
            f"--hands-onnx models/mediapipe-hands --out models/mediapipe-hands/hands.npue`")
    if kind == "mppose_models":
        # The arch=8 pair, both FLOAT files, and the container. The image is NOT
        # in this list: it is docs/bus.jpg, which is TRACKED, because it is the
        # golden's own input rather than a fetched checkpoint, and opencv_zoo
        # ships no raw input image at all -- 385 paths, not one of them a
        # photograph.
        d = REPO / "models" / "mediapipe-pose"
        want = ["person_detection_mediapipe_2023mar.onnx",
                "pose_estimation_mediapipe_2023mar.onnx", "mppose.npue"]
        missing_ = [w for w in want if not (d / w).exists()]
        img = REPO / "docs" / "bus.jpg"
        if not img.exists():
            missing_ = missing_ + ["../../docs/bus.jpg"]
        return (not missing_), (
            f"missing: {', '.join(missing_)}\n"
            f"  fetch the two ONNX files from opencv/person_detection_mediapipe "
            f"and opencv/pose_estimation_mediapipe (Apache-2.0); both sha256s and "
            f"the image's are in models/mediapipe-pose/CHECKPOINT.json. The "
            f"_int8bq variants are deliberately NOT accepted -- they are "
            f"slower AND wrong here, measured, and CHECKPOINT.json says by how "
            f"much. The goldens are tracked, at reference/goldens/mppose_det.json "
            f"and mppose_pose.json, and their recorder is NOT in this repository "
            f"so that no gate can rewrite them. Then "
            f"`python tools/pipeline.py run pack_npue -- --mppose-onnx "
            f"models/mediapipe-pose --out models/mediapipe-pose/mppose.npue`")
    if kind == "checkpoint":
        found = [p for p in (REPO / "models").glob("*")
                 if (p / "config.json").exists()] if (REPO / "models").is_dir() else []
        return bool(found), "fetch one: npuembeddings add <org/model>"
    if kind == "container":
        found = list((REPO / "models").glob("*.npue")) \
            if (REPO / "models").is_dir() else []
        return bool(found), "python tools/pipeline.py run pack_npue"
    if kind == "designs":
        # Via design_sets, not a glob written here: the layout is stated once, in
        # tools/lib/design_sets.py, beside the exporter that writes it and the
        # note about what it replaced. A copy of the pattern in this file is a
        # copy that can stop matching.
        found = design_sets()
        return bool(found), "python tools/pipeline.py run export_gemm_rtp"
    if kind == "runtime":
        for name in ("npuembeddings", "npuembeddings.exe"):
            if (REPO / "runtime" / "build" / name).exists():
                return True, ""
        return False, "cmake -S runtime -B runtime/build && cmake --build ..."
    if kind == "npu":
        # xrt-smi is the only honest answer, and asking it is cheap. Absent
        # xrt-smi means no driver, which is the same answer.
        exe = shutil.which("xrt-smi")
        if not exe:
            return False, "the Ryzen AI driver (xrt-smi is not on PATH)"
        try:
            p = subprocess.run([exe, "examine", "-r", "runtime"],
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return False, "xrt-smi did not answer"
        return p.returncode == 0, "no device: check the driver and the array"
    return True, ""


def missing(needs):
    """[(requirement, remedy)] for everything this tool needs and lacks."""
    out = []
    for kind in needs:
        ok, remedy = _have(kind)
        if not ok:
            out.append((kind, remedy))
    return out


def cmd_check(args):
    print("python   ", sys.version.split()[0], sys.executable)
    print("repo     ", REPO)
    print()
    print("PACKAGES")
    for kind, why, install in ENVIRONMENTS:
        ok, remedy = _have(kind)
        mark = "ok  " if ok else "MISS"
        print(f"  [{mark}] {kind:<13} {why}")
        if not ok:
            print(f"         -> {remedy or install}")
    print()
    print("ARTEFACTS (produced by earlier steps, not installed)")
    for kind, what, make in ARTEFACTS:
        ok, remedy = _have(kind)
        mark = "ok  " if ok else "no  "
        print(f"  [{mark}] {kind:<11} {what}")
        if not ok and kind != "npu":
            print(f"         -> {remedy}")
    print()
    runnable = [t for t in TOOLSET if not missing(t[3])]
    blocked = [t for t in TOOLSET if missing(t[3])]
    print(f"{len(runnable)} of {len(TOOLSET)} tools can run here. "
          f"Blocked: {', '.join(t[0] for t in blocked) or 'none'}.")
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args):
    rows = TOOLSET
    if args.role:
        rows = [t for t in rows
                if t[1].split("/")[0] == args.role
                or (args.role == "gates" and t in GATES)]
        if not rows:
            print(f"no tools with role {args.role!r}. "
                  f"Roles: {', '.join(sorted({t[1].split('/')[0] for t in TOOLSET}))}"
                  f", gates.")
            return 1
    width = max(len(t[0]) for t in rows)
    for name, path, what, needs in rows:
        lack = missing(needs)
        mark = "ok " if not lack else "-- "
        print(f"  {mark}{name:<{width}}  {what}")
        print(f"     {' ' * width}  tools/{path}"
              + (f"   needs: {', '.join(k for k, _ in lack)}" if lack else ""))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args):
    name = args.tool
    t = tool(name)
    if t is None:
        import difflib
        near = difflib.get_close_matches(name, [x[0] for x in TOOLSET], n=3)
        print(f"no tool named {name!r}."
              + (f" Did you mean: {', '.join(near)}?" if near else
                 " Run `python tools/pipeline.py list` for the index."))
        return 2
    lack = missing(t[3])
    if lack and not args.force:
        print(f"{name} needs {len(lack)} thing(s) this machine does not have:")
        for kind, remedy in lack:
            print(f"  - {kind}: {remedy}")
        print(f"\nIt would fail on the first of those. Override with --force "
              f"if the message it prints is the one you wanted.")
        return 1
    script = TOOLS / t[1]
    print(f"-> {command_for(name)} {' '.join(args.rest)}", flush=True)
    # Not exec(): the tools' own tracebacks and exit codes are the contract,
    # and a wrapper that swallowed them would be a second thing to debug.
    return subprocess.run([sys.executable, str(script)] + args.rest).returncode


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def cmd_gates(args):
    names = list(GATE_ORDER)
    if args.only:
        if args.only not in GATE_TIERS:
            print(f"no tier {args.only!r}. Tiers: "
                  f"{', '.join(GATE_TIERS)}")
            return 2
        names = list(GATE_TIERS[args.only])
    if not args.run:
        for name in names:
            t = tool(name)
            lack = missing(t[3])
            mark = "ok " if not lack else "-- "
            print(f"  {mark}{name:<24} {t[2]}")
            if lack:
                print(f"     {' ' * 24} needs {', '.join(k for k, _ in lack)}")
        print("\n--run to execute them in this order, or --only TIER for one "
              "tier:\n  " + ", ".join(GATE_TIERS))
        return 0
    failed = []
    for name in names:
        rc = cmd_run(argparse.Namespace(tool=name, rest=[], force=False))
        if rc:
            failed.append(name)
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(names)} gate(s) passed")
    return 0


# ---------------------------------------------------------------------------
# the map (no arguments)
# ---------------------------------------------------------------------------

def cmd_map(args):
    print("tools/ -- build-time tooling. What do you want to do?\n")
    for what, how, where in MAP:
        print(f"  {what}")
        print(f"    {how}")
        if where:
            print(f"    ({where})")
        print()
    print("  Every tool, with what it needs before it runs:")
    print("    python tools/pipeline.py list")
    print()
    print("  What this machine is missing:")
    print("    python tools/pipeline.py check")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python tools/pipeline.py",
        description="The way into tools/: what to run, and what you need "
                    "first. Runs no tool by itself except under `run`/`gates "
                    "--run`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with no arguments: it prints the same map.")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("list", help="every tool, and what it needs")
    p.add_argument("role", nargs="?", default=None,
                   help="only one bucket (pack, export, gen, verify, "
                        "research) or `gates`")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("check", help="what this machine is missing")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="check the requirements, then run a tool")
    p.add_argument("tool")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="arguments passed through untouched")
    p.add_argument("--force", action="store_true",
                   help="run even if a requirement is missing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("gates", help="the verify_* gates, cheapest first")
    p.add_argument("--only", choices=sorted(GATE_TIERS), default=None,
                   help="one tier only")
    p.add_argument("--run", action="store_true", help="execute them in order")
    p.set_defaults(func=cmd_gates)

    args = ap.parse_args()
    if not getattr(args, "func", None):
        return cmd_map(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
