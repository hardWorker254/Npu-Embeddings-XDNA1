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
    ap.add_argument("design", help=".../gemm_rtp -- the design set directory")
    ap.add_argument("--container", required=True,
                    help="a container packed with --npu")
    ap.add_argument("--expect-n", type=int, default=32,
                    help="tile_n the design was built with (default 32, the "
                         "value tools/pack/packers/pose.py tiles to)")
    args = ap.parse_args(argv)

    from exporters.gemm_rtp.geometry import (POSE_CONV_SHAPES, POSE_DISPATCH_M,
                                             pose_shapes_for)
    from npue import Reader

    design = Path(args.design)
    dj = design / "design.json"
    if not dj.is_file():
        print(f"FAIL {dj} is not there. Build the set first:\n"
              f"  source /tmp/opencode/npu-env.sh\n"
              f"  python tools/export/export_gemm_rtp.py --target pose --arch 1 "
              f"--n {args.expect_n} --batch 1")
        return 2
    d = json.loads(dj.read_text())

    fails: list[str] = []

    # -- 1. the design's streams against geometry.py's -----------------------
    got = [(s["op"], s["K"], s["N"]) for s in d.get("streams", [])]
    want = [(f"conv{k}x{n}", k, n) for k, n in POSE_CONV_SHAPES]
    gset, wset = set(got), set(want)
    for name, k, n in sorted(wset - gset):
        fails.append(f"the design has no stream {name} (K={k}, N={n}); "
                     f"geometry.py lists it")
    for name, k, n in sorted(gset - wset):
        fails.append(f"the design carries {name} (K={k}, N={n}) and "
                     f"geometry.py does not list it")
    if gset == wset:
        print(f"  streams    {len(wset)} names agree, both directions")

    # -- 2. the design's M against the dispatch chunk -------------------------
    top = d.get("M", 0)
    if top != POSE_DISPATCH_M:
        fails.append(f"the design's top-level M is {top}, but geometry.py's "
                     f"POSE_DISPATCH_M -- the value the runtime reads back as "
                     f"its chunk size -- is {POSE_DISPATCH_M}. These are the "
                     f"same number in two places and they must agree.")
    others = sorted({s.get("M") for s in d.get("streams", [])})
    if others != [top]:
        fails.append(f"the design's per-stream M values are {others} but its "
                     f"top-level M is {top}. NpuConvBackend reads the top-level "
                     f"one (runtime/src/pose/session.cpp) and ignores the rest, "
                     f"so a stream M that differs is a number nothing uses.")
    if top % (d.get("tile", {}).get("m", 64) * d.get("tile", {}).get("rows", 4)):
        fails.append(f"M = {top} is not a multiple of m*rows = "
                     f"{d.get('tile', {}).get('m')}*{d.get('tile', {}).get('rows')}")
    else:
        print(f"  M          {top} = POSE_DISPATCH_M, a multiple of m*rows, and "
              f"the same in all {len(d.get('streams', []))} streams")

    # -- 3. the tile_n the panels were tiled to -------------------------------
    tn = d.get("b_layout", {}).get("tile_n")
    if tn != args.expect_n:
        fails.append(f"the design's B layout is tiled at n={tn}, but the "
                     f"container's panels are tiled at {args.expect_n}. The "
                     f"runtime compares layout_hash and would refuse at stage "
                     f"time; the panel tiling is decided by "
                     f"tools/pack/packers/pose.py's TILE_N, not by this file.")
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
    # `streams` in the targets file is a list of NAMES, not a list of objects --
    # the schema requires a list of strings (common/targets.py), which is why
    # this reads the strings directly rather than the objects it reads out of a
    # design.json.
    tnames = list(t["kinds"]["pose"]["streams"])
    gnames = [f"conv{k}x{n}" for k, n in POSE_CONV_SHAPES]
    if tnames != gnames:
        for nm in sorted(set(gnames) - set(tnames)):
            fails.append(f"npu_targets.json's kinds.pose.streams is missing {nm}")
        for nm in sorted(set(tnames) - set(gnames)):
            fails.append(f"npu_targets.json's kinds.pose.streams carries {nm}, "
                         f"which geometry.py does not list")
        if set(tnames) == set(gnames) and tnames != gnames:
            fails.append("the same 21 names are present in both files but in a "
                         "different ORDER, so the slot a convolution resolves to "
                         "depends on which file the build read")
    else:
        print(f"  targets    {len(tnames)} names agree with geometry.py, in the "
              f"same order")

    # -- 5. the container against geometry.py ---------------------------------
    r = Reader(args.container)
    if "npu_streams" not in r.config:
        fails.append(f"{args.container} carries no npu_streams, so it was packed "
                     f"WITHOUT --npu and has no array panels at all. Repack it "
                     f"with `--pose-onnx FILE --npu`.")
    else:
        cs = json.loads(r.config["npu_streams"])
        cn = {(s["op"], s["k"], s["n"]) for s in cs}
        for name, k, n in sorted(wset - cn):
            fails.append(f"the container has no stream {name} (K={k}, N={n})")
        for name, k, n in sorted(cn - wset):
            fails.append(f"the container carries stream {name} (K={k}, N={n}) "
                         f"and the design set does not")
        if cn == wset:
            print(f"  container  {len(cn)} streams agree with the design set")

        # And every convolution in the graph must name a stream that exists --
        # the specific way this pair goes wrong at run time rather than at load.
        gnames = {n for n, _, _ in wset}
        g = json.loads(r.config["graph"])
        named = {s["stream"] for s in g if s["op"] == "conv"}
        for s in sorted(named - gnames):
            fails.append(f"a convolution in the container's graph names the "
                         f"stream {s}, which neither the design nor geometry.py "
                         f"has. It would resolve to no slot at dispatch time.")
        if named and named <= gnames:
            print(f"  graph      all {len(named)} distinct streams the "
                  f"convolutions name exist in the design set")

    # -- verdict ---------------------------------------------------------------
    if fails:
        print(f"\nFAIL {len(fails)} disagreement(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS -- the design set, the exporter's list and the packed "
          "container name the same streams at the same M, so a "
          "convolution's stream resolves to a real slot rather than to nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())