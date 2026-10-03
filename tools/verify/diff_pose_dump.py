#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the pose runtime's graph against the ONNX checkpoint, NODE BY NODE.

    python tools/verify/diff_pose_dump.py MODEL.onnx CONTAINER.npue IMAGE \
        --dump runtime.bin

`npuembeddings pose ... --pose-dump FILE` writes every graph node's output as
raw fp32 NCHW behind a JSON header. This file reads that back, runs the same
graph independently in NumPy (verify_pose.py's interpreter, transcribed from the
ONNX file rather than from the container), and prints the FIRST node where the
two disagree, plus the whole per-node error table.

WHY NODE BY NODE, AND NOT END TO END
------------------------------------
End to end, a wrong first convolution, a wrong strided stem and a wrong DFL fold
are all the same sentence -- "the detections are wrong" -- and this bug's symptom
was scores of 0.003 against a reference's 0.88, which is equally consistent with
a transposed weight, a half-pixel upsample and a sign error in the box decoder.
The node table says which, on the first run.

WHAT IS AND IS NOT INDEPENDENT
------------------------------
The two graphs are not the same object read twice. The container's op list is
written by tools/pack/packers/pose.py from the ONNX file; this file reads the
container only to learn (a) the index -> checkpoint-node correspondence, which
`node_names` records, and (b) nothing else. Every number compared here comes from
executing the ONNX graph itself in NumPy. That is what makes a disagreement
localise: the runtime and the reference disagree about a VALUE, and each value
has exactly one node on each side.

The image front end is the exception, and it is deliberate: both sides call PIL's
bilinear resize (see verify_pose.py's letterbox) because resize_to's contract is
PIL-equivalence, already measured in tools/verify/verify_vit_image.py. Node 0's
INPUT is compared anyway, so a front-end difference would show up here rather
than hide.

TOLERANCE, AND WHY IT IS NOT A CONSTANT
---------------------------------------
fp32 accumulate order differs between a blocked host GEMM and NumPy's BLAS, so
the two are bit-identical nowhere and everywhere near-identical. The check is
therefore RELATIVE, per node, against the node's own scale:

    err = max|a - b| / max(1, max|b|)

with the default threshold 2e-4. A node that fails it is reported with its own
scale, so a divergence that grows through the network is visible as a growing
column rather than one pass/fail at the end. What this build has NOT measured is
the exact threshold at which a downstream detection degrades -- it is an
engineering bound for "the arithmetic is the same computation", and it is set
loose enough to absorb fp32 reassociation and tight enough to catch a wrong
weight, which is off by O(1) and not by O(1e-5).
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import verify_pose as V  # noqa: E402  (after sys.path, deliberately)
from npue import Reader  # noqa: E402


def read_dump(path):
    """The runtime's --pose-dump file: (header dict, body bytes, header size).

    The header is the first 64 KiB of the file, space-padded to that size (see
    pose_mode.hpp). Reading a fixed prefix rather than "up to the first brace" is
    deliberate: the writer reserved the region, so the length is known, and a
    reader that stopped at a brace would silently accept a truncated header -- the
    one failure mode this tool exists to detect.
    """
    raw = open(path, "rb").read()
    k = 1 << 16
    if len(raw) < k:
        raise SystemExit(f"{path}: {len(raw)} bytes, shorter than the 65536-byte "
                         f"header region. The writer did not finish.")
    head = raw[:k].decode("ascii").strip()
    try:
        return json.loads(head), raw[k:], k
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path}: the header region is not JSON ({e}). Either "
                         f"the run was killed before the header was written back, "
                         f"or this is not a --pose-dump file.")


def reference_nodes(onnx_path, image, node_names, graph):
    """{graph index: NCHW float64 tensor} from an independent ONNX execution.

    The SILU IS APPLIED HERE, and it has to be. The container's op list fuses the
    activation into the convolution -- one `conv` node carries `"silu": 1` where
    the checkpoint has a Conv and then a separate activation node -- so the
    runtime's node 0 output is silu(conv) while the ONNX's node 0 output is
    conv. Comparing them directly is a mismatch of about 1.0 relative that looks
    like a catastrophically broken first convolution and is not: the pre- and
    post-activation tensors differ by tens of units wherever the pre-activation
    value is very negative, because silu(-33.8) is -1e-14.

    That is why the fused flag is read from the CONTAINER rather than guessed from
    the ONNX: the whole point of the flag is that the checkpoint does not say.
    """
    import onnx
    m = onnx.load(onnx_path)
    x, scale, px, py = V.letterbox(V.load_image(image), 640)
    want = set()
    for nm in node_names:
        base, _, half = nm.rpartition(":")
        want.add(base if half.isdigit() else nm)
    want.add("images")
    vals, got = V.run_onnx(m, {"images": x[None, ...].astype(np.float64)}, want=want)
    silu = lambda a: a / (1.0 + np.exp(-a))
    out = {}
    for i, nm in enumerate(node_names):
        base, _, half = nm.rpartition(":")
        if half.isdigit():
            # A Split became one `slice` op per half, in channel order. The half
            # index is the ORDER of the halves, which is what the packer wrote,
            # so it indexes the Split's output list rather than a channel count.
            outs = [o for o in m.graph.node if o.name == base][0].output
            out[i] = vals[outs[int(half)]]
        elif nm == "detect":
            # The one op with no checkpoint node of its own: it is the packer's
            # fold of the head's own 78 nodes into ONE runtime op, so the
            # reference has nothing to compare it to node-wise. The head tensor
            # itself is checked by verify_pose.py's --compare-head, which reads
            # the runtime's canonical output rather than a graph node.
            continue
        else:
            if nm not in got:
                raise SystemExit(f"the reference did not produce {nm}; the "
                                 f"container's node_names names a node this "
                                 f"checkpoint does not have")
            t = np.asarray(got[nm], dtype=np.float64)
            out[i] = silu(t) if graph[i].get("silu") else t
    out[-1] = x[None, ...].astype(np.float64)   # the graph's input tensor
    return out, (scale, px, py)


def check_head(args, rows, graph, body, k, node_names):
    """Compare the runtime's CANONICAL head tensor with a reference dump.

    The node loop above cannot check the `detect` node: everything it compares
    is a layer of the packed graph, and the head is not a layer -- it is the
    20-odd head-plumbing nodes the packer absorbed, re-assembled by
    Network::head into one [4 + nc + nk*3, cells] tensor. That function is where
    the DFL, the grid arithmetic, the keypoint affine and the xyxy conversion
    live, so it is the part with the most arithmetic and no node-by-node diff.

    --head is verify_pose.py's --dump-head: the ONNX graph's OWN output0, which
    is a DIFFERENT tensor -- xywh boxes and the exporter's own keypoint
    arithmetic -- rather than the runtime's canonical xyxy. So --head-box says
    how to read the reference's first four channels. It is an argument and not a
    guess because both layouts are correct ONNX outputs for different heads, and
    inferring one from the other would compare the two boxes in different units
    and call a correct head wrong.
    """
    if not args.head:
        print("\nthe head was NOT checked: --head was not given. Everything above "
              "is the packed graph; the DFL, the grid, the keypoint affine and "
              "the box conversion are below it and are exactly where a head "
              "that reads the right values in the wrong order still looks "
              "right.")
        return 0
    det = [i for i, o in enumerate(graph) if o["op"] == "detect"]
    if len(det) != 1:
        print(f"\nthe graph has {len(det)} detect nodes, so there is no one head "
              f"tensor to compare. The container is malformed in a way "
              f"read_geometry should have refused.")
        return 1
    row = next(r for r in rows if r["i"] == det[0])
    got = np.frombuffer(body, dtype="<f4", count=row["n"],
                        offset=row["off"] - k).reshape(
                            row["c"], row["h"], row["w"]).astype(np.float64)
    ref = np.fromfile(args.head, dtype="<f4").astype(np.float64)
    if ref.size != got.size:
        print(f"\nthe head reference is {ref.size} floats and the runtime's is "
              f"{got.size}. verify_pose.py wrote it from a different image or a "
              f"different checkpoint.")
        return 1
    ref = ref.reshape(got.shape)
    if args.head_box == "xywh":
        cx, cy, w, h = ref[0], ref[1], ref[2], ref[3]
        ref = ref.copy()
        ref[0], ref[1] = cx - w / 2, cy - h / 2
        ref[2], ref[3] = cx + w / 2, cy + h / 2
    elif args.head_box != "ltrb":
        print(f"\n--head-box {args.head_box} is not ltrb or xywh. This tool will "
              f"not guess how the reference's four box channels are laid out.")
        return 1
    scale = max(1.0, float(np.abs(ref).max()))
    diff = float(np.abs(got - ref).max())
    err = diff / scale
    bad = diff > max(args.atol, args.tol * scale)
    print(f"\nhead         detect node {det[0]} {str(got.shape)} "
          f"[{got.shape[0]} channels x {got.shape[1]} cells], reference read as "
          f"{args.head_box}")
    print(f"{'':>4} {'':9} {'':22} {scale:>10.4g} {err:>10.3g}  "
          f"{'MISMATCH -- see the channel breakdown below' if bad else 'agrees'}")
    if bad:
        for ch in range(got.shape[0]):
            e = float(np.abs(got[ch] - ref[ch]).max()) / max(
                1.0, float(np.abs(ref[ch]).max()))
            if e > args.tol:
                print(f"     channel {ch:>3}: {e:.3g} relative")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("container")
    ap.add_argument("image")
    ap.add_argument("--dump", required=True, help="the runtime's --pose-dump file")
    ap.add_argument("--image-index", type=int, default=1,
                    help="which image in the dump (1-based); pose_mode.hpp writes "
                         "one set of nodes per input image, in order")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="relative per-node tolerance (see the module docstring)")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="absolute floor, so a node whose reference is all zeros "
                         "is not failed by its own fp32 noise")
    ap.add_argument("--all", action="store_true",
                    help="print every node, not just up to the first failure")
    ap.add_argument("--head",
                    help="verify_pose.py --dump-head: the ONNX graph's own "
                         "output0, to compare the runtime's CANONICAL head "
                         "tensor against. Without it the head is not checked at "
                         "all and this tool says so")
    ap.add_argument("--head-box", choices=("ltrb", "xywh"), default="xywh",
                    help="how to read the reference head's first four channels. "
                         "This exporter emits (cx, w, cy, h) in letterbox "
                         "pixels; a head that emitted (l, t, r, b) would be "
                         "xyxy and the argument is 'ltrb'. It is stated rather "
                         "than guessed because both are valid outputs for "
                         "different heads, and reading one as the other "
                         "compares boxes in different units")
    args = ap.parse_args()

    r = Reader(args.container)
    node_names = json.loads(r.config["node_names"])
    graph = json.loads(r.config["graph"])
    head, body, k = read_dump(args.dump)

    rows = sorted((n for n in head["nodes"] if n["image"] == args.image_index),
                  key=lambda n: n["i"])
    if [r["i"] for r in rows] != list(range(-1, len(node_names))):
        raise SystemExit(f"the dump's node indices for image {args.image_index} are "
                         f"not 0..{len(node_names) - 1} in order. It is missing the "
                         f"graph input record (-1) or a node, and a diff over a "
                         f"truncated list reads as agreement on what is left.")
    inp = [r for r in rows if r["i"] == -1][0]
    rows = [r for r in rows if r["i"] >= 0]
    if len(rows) != len(node_names):
        raise SystemExit(f"the dump has {len(rows)} nodes for image "
                         f"{args.image_index}, the container lists "
                         f"{len(node_names)}. One of them is wrong and a diff over "
                         f"a truncated list reads as agreement on the nodes that "
                         f"are left.")

    ref, _ = reference_nodes(args.onnx, args.image, node_names, graph)

    print(f"image       {args.image}  ({args.image_index} of the dump)")
    print(f"nodes       {len(rows)}   tolerance {args.tol:g} relative, "
          f"{args.atol:g} absolute\n")
    print(f"{'idx':>4} {'op':<9} {'shape':<22} {'ref scale':>10} {'rel err':>10}  "
          f"checkpoint node")

    first_bad = None
    worst = 0.0
    worst_at = None

    # The graph's INPUT first, before any node. It is the front end's own
    # arithmetic -- letterbox, pad colour, channel order -- and if it is wrong
    # then every node below is wrong in a way that looks like the network's, so
    # this row is what turns "the convolutions are broken" into "the front end is
    # broken, or the convolutions are".
    print(f"{'in':>4} {'input':<9} {str((inp['c'], inp['h'], inp['w'])):<22} "
          f"{'':>10} {'':>10}  the letterbox both sides fed the network")
    gi = np.frombuffer(body, dtype="<f4", count=inp["n"],
                       offset=inp["off"] - k).reshape(
                           1, inp["c"], inp["h"], inp["w"]).astype(np.float64)
    gw = np.asarray(ref[-1], dtype=np.float64)
    gerr = float(np.abs(gi - gw).max()) / max(1.0, float(np.abs(gw).max()))
    gscale = float(np.abs(gw).max())
    print(f"{'':>4} {'':9} {'':22} {gscale:>10.4g} {gerr:>10.3g}  "
          f"{'MISMATCH -- the front end disagrees, so every node below is '
             'downstream of it' if gerr > args.tol else 'agrees'}\n")
    if gerr > args.tol:
        first_bad = -1

    for row in rows:
        i = row["i"]
        if graph[i]["op"] == "detect":
            continue
        n = row["n"]
        # The header's `off` is an offset into the FILE (the writer asked
        # tellp() on the whole stream, which is the only number that means
        # something before the header exists), so it has the reserved region's
        # size subtracted here. Reading it as an offset into `body` instead
        # would land 64 KiB early -- inside the previous node's last row, for
        # every node, which reads as a large-but-not-crazy diff everywhere.
        flat = np.frombuffer(body, dtype="<f4", count=n, offset=row["off"] - k)
        got = flat.reshape(1, row["c"], row["h"], row["w"]).astype(np.float64)
        want = np.asarray(ref[i], dtype=np.float64)
        if got.shape != want.shape:
            print(f"{i:>4} {graph[i]['op']:<9} {str(got.shape[1:]):<22} "
                  f"{'--':>10} {'SHAPE':>10}  {node_names[i]}\n"
                  f"     the runtime produced {got.shape}, the ONNX graph "
                  f"{want.shape}. A shape difference here is a graph-structure "
                  f"bug, not a numeric one, and no later node can be compared.")
            first_bad = first_bad if first_bad is not None else i
            continue
        scale = max(1.0, float(np.abs(want).max()))
        diff = float(np.abs(got - want).max())
        # ONE threshold, not two tests. `atol` is a FLOOR: it keeps a node whose
        # reference is all zeros (or all tiny) from being failed by its own fp32
        # noise, and it does that by raising the bar where the relative test is
        # meaningless -- not by also acting as a ceiling on a node with a large
        # reference, where 3e-5 of 54 is 6e-7 relative and is exactly the
        # reassociation a different GEMM blocking produces.
        bad = diff > max(args.atol, args.tol * scale)
        err = diff / scale
        if bad and first_bad is None:
            first_bad = i
        if err > worst:
            worst, worst_at = err, i
        if bad or args.all:
            mark = "  <-- FIRST MISMATCH" if i == first_bad else ""
            print(f"{i:>4} {graph[i]['op']:<9} {str(got.shape[1:]):<22} "
                  f"{scale:>10.4g} {err:>10.3g}  {node_names[i]}{mark}")
        if first_bad is not None and not args.all and not bad:
            break

    if first_bad is None:
        print(f"\nno node exceeded the tolerance; worst was node {worst_at} at "
              f"{worst:.3g} relative")
        return check_head(args, rows, graph, body, k, node_names)
    if first_bad < 0:
        print("\nFIRST MISMATCH is the graph INPUT: the two front ends disagree, so "
              "no node's comparison means anything yet. Node -1 is the letterbox "
              "-- resize, pad colour, channel order -- not the network.")
        return 1
    print(f"\nFIRST MISMATCH at node {first_bad} ({node_names[first_bad]}). The "
          f"runtime and the ONNX graph agree on every node before it.")
    print("A node before this one that also failed but was skipped is not "
          "agreement -- re-run with --all.")
    return 1


if __name__ == "__main__":
    sys.exit(main())