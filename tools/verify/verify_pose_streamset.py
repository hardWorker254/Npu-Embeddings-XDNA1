#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The exporter's pose stream set against a packed container's.

    python tools/verify/verify_pose_streamset.py artifacts/pose/artifacts_npu1/gemm_rpu \
        --container /tmp/opencode/npu_i8.npue

WHY THIS EXISTS AND WHY IT IS A SEPARATE FILE
---------------------------------------------
The twenty-one `conv{K}x{N}` stream names are written down TWICE: once in
tools/export/exporters/gemm_rtp/geometry.py, which builds the design set, and
once in tools/pack/packers/pose.py, which writes them into the container's
npu_streams so the runtime can match a convolution to a slot. They are the same
list, and nothing but this tool holds them together.

That is not a theoretical risk. An earlier version of geometry.py listed the
strides by hand, and the hand-written table disagreed with the checkpoint on
twenty-one of twenty-five entries -- a stride-2 convolution listed as stride 4,
which moves that design's M by 4x. Both files were internally consistent, both
built, and the array would have run the wrong rows and returned confident wrong
numbers. Nothing in either exporter would have noticed.

WHAT IS CHECKED, AND WHAT IS NOT
--------------------------------
Checked: the set of (K, N) stream names, exactly, in both directions -- a name
in one and not the other is an error, not a warning, because the failure it
causes is a slot that resolves to nothing.

Checked: every design's M against POSE_DISPATCH_M, and against the m*rows the
design was built with. A design whose M is not the one the runtime reads back
means the runtime's chunk loop either dispatches short chunks (correct, slow) or
overruns the xclbin (wrong, and not always loudly).

NOT checked: that the K and N are what this checkpoint's convolutions actually
need. That is tools/verify/verify_conv_quant.py's job for the weights, and
geometry.py's list is checked against the container HERE -- the container is
built from the checkpoint, so agreement with it is agreement with the
checkpoint. What is deliberately NOT re-derived is the padding rule: both sides
pad K up to a multiple of tile_k and N up to a multiple of tile_n, and a
disagreement about THAT would show up as a name mismatch, which is checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "export"))
sys.path.insert(0, str(ROOT / "tools" / "lib"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", nargs="?", default=None,
                    help=".../gemm_rtp -- the design set directory. Optional: "
                         "without it this file checks the two spellings that "
                         "live IN the repository (consts, geometry.py and "
                         "npu_targets.json) and says plainly which two it could "
                         "not. A design set needs MLIR-AIE to build and that is "
                         "not installed everywhere, so making it mandatory would "
                         "make this gate unrunnable on exactly the machines "
                         "where its in-repo half is worth checking.")
    ap.add_argument("--container", default=None,
                    help="a container packed with --npu. Optional for the same "
                         "reason, and for a second one: arch=8 has no design set, "
                         "so a container packed with --npu has nothing to be "
                         "measured against and the interesting half of the check "
                         "is the two in-repo lists.")
    ap.add_argument("--expect-n", type=int, default=32,
                    help="tile_n the design was built with (default 32, the "
                         "value tools/pack/packers/pose.py tiles to)")
    ap.add_argument("--kind", choices=("pose", "hands", "mppose"), default="pose",
                    help="which conv-only stream set to check against the design "
                         "and the container (default: pose). The design "
                         "directory is shared by name across all three kinds -- "
                         "every one writes to artifacts_npu<N>/gemm_rtp -- so "
                         "which set is being checked is the caller's claim, and "
                         "it is checked against both the design and the "
                         "container rather than taken on trust.")
    args = ap.parse_args(argv)

    from exporters.gemm_rtp.geometry import (CONV_ONLY_DISPATCH_M,
                                             conv_shapes_for, conv_stream_order)
    from npue import Reader

    design = Path(args.design) if args.design else None
    d = None
    if design is not None:
        dj = design / "design.json"
        if not dj.is_file():
            print(f"FAIL {dj} is not there. Build the set first:\n"
                  f"  python tools/export/export_gemm_rtp.py --target <model> "
                  f"--arch 1 -n {args.expect_n} --batch 1\n"
                  f"  Building one needs MLIR-AIE (`import aie.iron`) and that is "
                  f"not installed on this machine, so there is no honest way to "
                  f"produce one here.")
            return 2
        d = json.loads(dj.read_text())

    # The set under test is the caller's --kind, resolved through geometry.py
    # rather than written out here, so this file has no list of its own to go
    # stale.
    kind = args.kind
    want = [(n, s["K"], s["N"])
            for n, s in conv_shapes_for(kind).items()]

    fails: list[str] = []

    # -- 1-3. the design's streams, its M and its tile_n ----------------------
    #
    # All three need a design, and all three are SKIPPED WITH A LINE SAYING SO
    # when none was named -- not defaulted to a design this file invents. A gate
    # that reported "22 streams agree" because it compared a list against itself
    # would be worse than one that says it checked nothing.
    if d is None:
        print(f"  streams    NOT CHECKED -- no design set named, and the {kind} "
              f"set has {len(want)} of them")
        print(f"  M          NOT CHECKED for the same reason "
              f"(CONV_ONLY_DISPATCH_M is {CONV_ONLY_DISPATCH_M})")
        print(f"  tile_n     NOT CHECKED for the same reason "
              f"(expected {args.expect_n})")
    else:
        got = [(s["op"], s["K"], s["N"]) for s in d.get("streams", [])]
        gset, wset = set(got), set(want)
        for name, k, n in sorted(wset - gset):
            fails.append(f"the design has no stream {name} (K={k}, N={n}); "
                         f"geometry.py lists it")
        for name, k, n in sorted(gset - wset):
            fails.append(f"the design carries {name} (K={k}, N={n}) and "
                         f"geometry.py does not list it")
        if gset == wset:
            print(f"  streams    {len(wset)} names agree, both directions")

        top = d.get("M", 0)
        if top != CONV_ONLY_DISPATCH_M:
            fails.append(f"the design's top-level M is {top}, but geometry.py's "
                         f"CONV_ONLY_DISPATCH_M -- the value the runtime reads "
                         f"back as its chunk size -- is "
                         f"{CONV_ONLY_DISPATCH_M}. These are the same number in "
                         f"two places and they must agree.")
        others = sorted({s.get("M") for s in d.get("streams", [])})
        if others != [top]:
            fails.append(f"the design's per-stream M values are {others} but "
                         f"its top-level M is {top}. NpuConvBackend reads the "
                         f"top-level one (runtime/src/pose/session.cpp) and "
                         f"ignores the rest, so a stream M that differs is a "
                         f"number nothing uses.")
        if top % (d.get("tile", {}).get("m", 64)
                  * d.get("tile", {}).get("rows", 4)):
            fails.append(f"M = {top} is not a multiple of m*rows = "
                         f"{d.get('tile', {}).get('m')}"
                         f"*{d.get('tile', {}).get('rows')}")
        else:
            print(f"  M          {top} = CONV_ONLY_DISPATCH_M, a multiple of "
                  f"m*rows, and the same in all "
                  f"{len(d.get('streams', []))} streams")

        tn = d.get("b_layout", {}).get("tile_n")
        if tn != args.expect_n:
            fails.append(f"the design's B layout is tiled at n={tn}, but the "
                         f"container's panels are tiled at {args.expect_n}. The "
                         f"runtime compares layout_hash and would refuse at "
                         f"stage time; the panel tiling is decided by the "
                         f"packer's TILE_N, not by this file.")
        else:
            print(f"  tile_n     {tn}, the value the packer tiles its panels to")

    # -- 4. the targets file's third spelling of the same list ---------------
    #
    # The names now exist in FOUR places: geometry.py (the build order and the
    # K/N), npu_targets.json's kinds.pose.streams (which the schema requires to
    # be a list), the design set's design.json (what was built), and the packed
    # container's npu_streams (what the runtime will look up). Any two agreeing
    # proves nothing about the other two, so all four are compared here rather
    # than pairwise in three tools.
    from exporters.common.targets import load_targets

    t = load_targets(ROOT / "tools" / "data" / "npu_targets.json")
    # The conv-only kinds are one list here against three there, and the two
    # sides cannot be allowed to drift: the exporter builds a set from
    # CONV_ONLY_SHAPES and the runtime looks a slot up by the name the packer
    # wrote, so a name in one file and not the other is a slot that resolves to
    # nothing AT DISPATCH TIME -- after a design set has been built, installed
    # and found plausible.
    from exporters.common.consts import CONV_ONLY_KINDS as _COK
    from exporters.gemm_rtp.geometry import CONV_ONLY_SHAPES as _COS
    if set(_COK) != set(_COS):
        print(f"FAIL consts.CONV_ONLY_KINDS is {sorted(_COK)} and "
              f"geometry.CONV_ONLY_SHAPES is {sorted(_COS)}. One kind with a "
              f"stream set the other does not know is an export that builds the "
              f"wrong four streams, or a validation that demands a hidden width "
              f"that does not exist.")
        fails.append("consts.CONV_ONLY_KINDS != geometry.CONV_ONLY_SHAPES")
    for k in sorted(set(_COS) & set(_COK)):
        kn = len(conv_stream_order(k))
        tn = len(t["kinds"].get(k, {}).get("streams", []))
        if kn != tn:
            print(f"FAIL {k}: geometry.py lists {kn} streams and npu_targets.json "
                  f"lists {tn}")
            fails.append(f"{k}: {kn} streams in geometry.py, {tn} in "
                         f"npu_targets.json")
    if not fails or all("conv-only kind" not in f and "streams in geometry" not in f
                        for f in fails):
        print(f"  kinds      {sorted(_COK)} agree across consts, geometry and "
              f"the targets file: "
              + ", ".join(f"{k} {len(conv_stream_order(k))}" for k in sorted(_COK)))
    # `streams` in the targets file is a list of NAMES, not a list of objects --
    # the schema requires a list of strings (common/targets.py), which is why
    # this reads the strings directly rather than the objects it reads out of a
    # design.json.
    tnames = list(t["kinds"][kind]["streams"])
    gnames = conv_stream_order(kind)
    if tnames != gnames:
        for nm in sorted(set(gnames) - set(tnames)):
            fails.append(f"npu_targets.json's kinds.{kind}.streams is missing "
                         f"{nm}")
        for nm in sorted(set(tnames) - set(gnames)):
            fails.append(f"npu_targets.json's kinds.{kind}.streams carries {nm}, "
                         f"which geometry.py does not list")
        if set(tnames) == set(gnames) and tnames != gnames:
            fails.append(f"the same {len(set(tnames))} {kind} names are present "
                         f"in both files but in a different ORDER, so the slot a "
                         f"convolution resolves to depends on which file the "
                         f"build read")
    else:
        print(f"  targets    {len(tnames)} {kind} names agree with geometry.py, "
              f"in the same order")

    # -- 5. the container against geometry.py ---------------------------------
    if args.container is None:
        print(f"  container  NOT CHECKED -- no container named. The fourth "
              f"spelling is the packed container's npu_streams, and it is the "
              f"one that caught the arch=8 26-vs-22 bug, so it is the one worth "
              f"naming:")
        print(f"              --container <file packed with --npu> --kind {kind}")
        return _verdict(fails)
    r = Reader(args.container)
    graphs = [g for g in ("det_graph", "pose_graph") if g in r.config]
    if not graphs:
        graphs = ["graph"]
    if "npu_streams" not in r.config:
        fails.append(f"{args.container} carries no npu_streams, so it was packed "
                     f"WITHOUT --npu and has no array panels at all. Repack it "
                     f"with the packer's --npu flag.")
    else:
        cs = json.loads(r.config["npu_streams"])
        cn = {(s["op"], s["k"], s["n"]) for s in cs}
        wset = {(n, k, nn) for n, k, nn in want}
        for name, k, n in sorted(wset - cn):
            fails.append(f"the container has no stream {name} (K={k}, N={n})")
        for name, k, n in sorted(cn - wset):
            fails.append(f"the container carries stream {name} (K={k}, N={n}) "
                         f"and geometry.py does not list it. Every stream in this "
                         f"container should be one a convolution names; one that "
                         f"none names is a slot with nothing behind it, which is "
                         f"how the depthwise leak showed up here.")
        if cn == wset:
            print(f"  container  {len(cn)} streams agree with geometry.py")

        # Does this container carry array panels at all? Every conv-only kind can
        # be packed WITHOUT --npu, which is the correct default on this machine
        # because there is no design set to dispatch into. So a container with no
        # panels is not an error -- it is the host-only build, and saying so is
        # the difference between a real disagreement and a missing input.
        conv_n = sum(1 for gk in graphs
                     for s in json.loads(r.config[gk])
                     if s["op"] == "conv")
        paneled = sum(1 for gk in graphs
                      for s in json.loads(r.config[gk])
                      if s["op"] == "conv" and s.get("stream"))
        print(f"  panels     {paneled} of {conv_n} dense convolutions in "
              f"{len(graphs)} graph(s) name a stream"
              + ("" if paneled else " -- the HOST-ONLY build, packed without "
                                    "--npu, and nothing to dispatch to"))

        # And every convolution in EVERY graph must name a stream that exists --
        # the specific way this pair goes wrong at run time rather than at load.
        # This is the check that catches the bug found today: hands and mppose
        # were being exported against the ENCODER's four streams, and nothing in
        # the exporter could see it, because nothing here ever compared the two
        # network lists against the container's.
        gnames = {n for n, _, _ in want}
        named: set[str] = set()
        n_dense = n_dw = 0
        for gk in graphs:
            for s in json.loads(r.config[gk]):
                if s["op"] == "conv":
                    n_dense += 1
                    named.add(s.get("stream",
                                    f"<dense conv in {gk} names no stream>"))
                elif s["op"] == "dwconv":
                    n_dw += 1
                    if "stream" in s:
                        fails.append(
                            f"a depthwise convolution in the container's {gk} "
                            f"names the stream {s['stream']}. A depthwise filter "
                            f"reduces within one channel -- K is kh*kw and N is 1 "
                            f"-- so there is no [M, N] GEMM in it to dispatch "
                            f"and a slot for it is a name with nothing behind it. "
                            f"This is not hypothetical: an unfiltered packer "
                            f"produces exactly these four stream names for "
                            f"arch=8 (conv64x512, conv64x640, conv64x768, "
                            f"conv64x1152) and a 26-stream container where the "
                            f"design has 22.")
        for s in sorted(named - gnames):
            fails.append(f"a convolution in the container's graphs names the "
                         f"stream {s}, which neither the design nor geometry.py "
                         f"has. It would resolve to no slot at dispatch time.")
        if named and named <= gnames:
            print(f"  graph      all {len(named)} distinct streams the "
                  f"{n_dense} dense convolutions name across {len(graphs)} "
                  f"graph(s) exist in the design set; {n_dw} depthwise name "
                  f"none, as they must")

    return _verdict(fails)


def _verdict(fails: list[str]) -> int:
    # The wording says which of the four spellings were actually compared, because
    # the set that matters changes with the arguments and a PASS that read as
    # "four spellings agree" when two were skipped would be the exact failure
    # this tool exists to catch, one level up.
    if fails:
        print(f"\nFAIL {len(fails)} disagreement(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS -- every stream spelling that was NAMED agrees with the next: "
          "consts' conv-only kinds, geometry.py's build order, npu_targets.json's "
          "list, a design set's streams and M, and a packed container's "
          "npu_streams and per-convolution names. So a convolution's stream "
          "resolves to a real slot rather than to nothing, and no slot exists "
          "that no convolution names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())