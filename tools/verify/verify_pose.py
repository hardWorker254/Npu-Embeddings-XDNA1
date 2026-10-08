#!/usr/bin/env python3
"""An independent NumPy reference for the YOLOv8-pose checkpoint.

WHAT THIS IS, AND WHY IT IS AN INTERPRETER RATHER THAN A TRANSCRIPTION
--------------------------------------------------------------------
It executes the ONNX graph itself -- every Conv, Mul, Sigmoid, Add, Concat,
Split, MaxPool, Resize and all of the head's Reshape/Slice/Transpose/Softmax
plumbing -- from the graph's own attributes and initialisers. Nothing is written
down twice and then compared against itself: the weights, the padding, the
stride, the split sizes, the softmax axis and the affine constants all come from
the file.

That matters because the alternative -- transcribing YOLOv8-pose by hand from
knowledge of the architecture -- would compare the runtime against MY IDEA of
the network, which is the thing most likely to be wrong. Reading the graph means
a disagreement between this file and the runtime is a disagreement about the
file, and the node that causes it can be named.

It is slow. 4.59 GMAC of im2col in NumPy is minutes, not milliseconds, and that
is fine: this runs to decide whether the runtime is right, not to be fast.

WHAT IT IS COMPARED AGAINST
---------------------------
By default it prints the reference's own decode -- boxes, scores, keypoints in
SOURCE pixels -- so the answer can be compared with `npuimage pose <model>
<image> --text` by eye. With --dump-head it writes the reference's [56, 8400]
head tensor to a binary file for a numeric comparison against the runtime's.

It does NOT compare against a container, and --npue is REFUSED rather than
accepted and ignored: the node-by-node comparison is
tools/verify/diff_pose_dump.py, which drives this same interpreter over a
--pose-dump file. An accepted flag that does nothing is a gate that passes
without having compared anything.

WHAT IS NOT MEASURED
--------------------
Nothing here is a claim about the runtime. It is a claim about the ONNX file, and
only the comparison in --npue says anything about the runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

# COCO 17, in the order this architecture's head emits them.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
NUM_CLASSES = 1
NUM_KEYPOINTS = 17


# -- a minimal interpreter over the op types this checkpoint actually uses -----


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == 2:
                return a.i
            if a.type == 1:
                return a.f
            if a.type == 3:
                return a.s.decode("utf-8")
            if a.type == 7:
                return list(a.ints)
            if a.type == 6:
                return list(a.floats)
            if a.type == 8:
                return [x.decode("utf-8") for x in a.strings]
            raise SystemExit(f"{node.name}: attribute {name} has a type this "
                             f"reference does not read (type {a.type})")
    return default


def sigmoid(x):
    # The branch matters only for extreme logits: np.exp(-1000) overflows to inf
    # and 1/(1+inf) is 0, which is the right answer for a negative logit anyway.
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def conv(x, w, b, pads, strides):
    """NCHW convolution. `pads` is ONNX's (top, left, bottom, right)."""
    n, c, h, wd = x.shape
    co, ci, kh, kw = w.shape
    pt, pl, pb, pr = pads
    st, sw = strides
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (h + pt + pb - kh) // st + 1
    ow = (wd + pl + pr - kw) // sw + 1
    # im2col once per output pixel. ci*kh*kw is at most 256*9 = 2304 and oh*ow
    # is at most 640*640, so the temporary is the single biggest allocation in
    # this file and it is why a run takes minutes.
    k = ci * kh * kw
    cols = np.empty((n, oh * ow, k), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i:i + oh * st:st, j:j + ow * sw:sw]
            # ONNX's kernel order is [C, kh, kw], so position (i, j) of the
            # window is the strided slice starting at i*kw + j with step kh*kw.
            # Anything else would still multiply the right numbers and still add
            # up, and the network would be a different one.
            part = patch.transpose(0, 2, 3, 1).reshape(n, oh * ow, ci)
            cols[:, :, i * kw + j:: kh * kw] = part
    # One tensordot rather than `co` passes: the same arithmetic, and BLAS
    # instead of a Python loop over output channels. 4.59 GMAC of the latter is
    # hours; of the former, minutes.
    out = np.tensordot(cols, w.reshape(co, k).T, axes=([2], [0]))
    out = out.transpose(0, 2, 1).reshape(n, co, oh, ow)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def maxpool(x, k, strides, pads):
    n, c, h, w = x.shape
    st = strides[0]
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    oh = (h + pt + pb - k) // st + 1
    ow = (w + pl + pr - k) // st + 1
    out = np.full((n, c, oh, ow), -np.inf)
    for i in range(k):
        for j in range(k):
            sl = xp[:, :, i:i + oh * st:st, j:j + ow * st:st]
            np.maximum(out, sl, out=out)
    return out


def run_onnx(model, feeds, want=None):
    """Execute the graph. `want` limits the recorded tensors, by node name."""
    g = model.graph
    vals = {i.name: numpy_helper.to_array(i).astype(np.float64)
            for i in g.initializer}
    for k, v in feeds.items():
        vals[k] = np.asarray(v, dtype=np.float64)
    for n in g.node:
        if n.op_type == "Constant":
            a = next((x for x in n.attribute if x.name == "value"), None)
            if a is not None:
                try:
                    vals[n.output[0]] = numpy_helper.to_array(a.t).astype(np.float64)
                except Exception:
                    pass
    got = {}
    for n in g.node:
        if n.op_type == "Constant":
            continue
        ins = [vals.get(i) for i in n.input]
        op = n.op_type
        if op == "Conv":
            w = vals[n.input[1]]
            b = vals[n.input[2]] if len(n.input) > 2 else None
            pads = _attr(n, "pads", [0, 0, 0, 0])
            strides = _attr(n, "strides", [1, 1])
            auto = _attr(n, "auto_pad", b"NOTSET".decode())
            if auto in ("SAME_UPPER", "SAME_LOWER"):
                kh, kw = w.shape[2], w.shape[3]
                pads = [kh // 2, kw // 2, kh - 1 - kh // 2, kw - 1 - kw // 2]
            out = conv(ins[0], w, b, pads, strides)
        elif op == "Mul":
            out = ins[0] * ins[1]
        elif op == "Add":
            out = ins[0] + ins[1]
        elif op == "Sub":
            out = ins[0] - ins[1]
        elif op == "Div":
            out = ins[0] / ins[1]
        elif op == "Sigmoid":
            out = sigmoid(ins[0])
        elif op == "Softmax":
            out = ins[0] - ins[0].max(axis=_attr(n, "axis", -1), keepdims=True)
            out = np.exp(out)
            out /= out.sum(axis=_attr(n, "axis", -1), keepdims=True)
        elif op == "Concat":
            out = np.concatenate(ins, axis=_attr(n, "axis", 0))
        elif op == "Split":
            sizes = _attr(n, "split")
            if sizes is None and len(n.input) > 1:
                sizes = [int(v) for v in np.atleast_1d(ins[1]).ravel()]
            axis = _attr(n, "axis", 0)
            at = 0
            for t, sz in zip(n.output, sizes):
                sl = [slice(None)] * out_rank(ins[0])
                sl[axis] = slice(at, at + int(sz))
                vals[t] = ins[0][tuple(sl)]
                at += int(sz)
            continue
        elif op == "MaxPool":
            out = maxpool(ins[0], _attr(n, "kernel_shape")[0],
                          _attr(n, "strides", [1]), _attr(n, "pads", [0, 0, 0, 0]))
        elif op == "Resize":
            mode = _attr(n, "mode", "nearest")
            assert "nearest" in (mode or "nearest"), f"{n.name}: mode {mode}"
            scales = _attr(n, "scales")
            if scales is None and len(n.input) > 2:
                scales = [float(v) for v in np.atleast_1d(ins[2]).ravel()]
            assert scales is not None, f"{n.name}: no scales"
            sh = [int(round(ins[0].shape[i] * scales[i])) for i in range(4)]
            yi = (np.arange(sh[2]) / scales[2]).astype(int).clip(0, ins[0].shape[2] - 1)
            xi = (np.arange(sh[3]) / scales[3]).astype(int).clip(0, ins[0].shape[3] - 1)
            out = ins[0][:, :, yi][:, :, :, xi]
        elif op == "Transpose":
            out = np.transpose(ins[0], _attr(n, "perm"))
        elif op == "Reshape":
            target = [int(v) for v in np.atleast_1d(ins[1]).ravel()]
            out = ins[0].reshape(target)
        elif op == "Shape":
            out = np.array(ins[0].shape, dtype=np.float64)
        elif op == "Gather":
            out = np.take(ins[0], np.atleast_1d(ins[1]).astype(int), axis=_attr(n, "axis", 0))
        elif op == "Slice":
            data = ins[0]
            axis = int(np.atleast_1d(ins[3]).ravel()[0]) if len(ins) > 3 else 0
            starts = _to_ints(ins[1])
            ends = _to_ints(ins[2])
            steps = _to_ints(ins[4]) if len(ins) > 4 else [1] * len(starts)
            n_axis = data.ndim
            for k, (a, b, st) in enumerate(zip(starts, ends, steps)):
                ax = axis + k
                assert -n_axis <= ax < n_axis
                ax %= n_axis
                dim = data.shape[ax]
                # ONNX's rule, in full: a negative bound is relative to the END
                # of the axis, and then BOTH kinds are clamped into [0, dim]. An
                # earlier version of this read "ends >= 0" as "the whole
                # remainder of the axis", which made the head's ltrb split
                # [0:2] into [0:4] -- the wrong half of every box, silently,
                # because a slice that keeps everything is still a valid tensor
                # of a plausible width.
                lo = a + dim if a < 0 else a
                lo = max(0, min(lo, dim))
                hi = b + dim if b < 0 else b
                hi = max(0, min(hi, dim))
                count = max(0, -(-(hi - lo) // st))
                sl = [slice(None)] * n_axis
                sl[ax] = slice(lo, lo + count * st, st)
                data = data[tuple(sl)]
            out = data
        else:
            raise SystemExit(f"{n.name}: op {op} is not implemented by this "
                             f"reference. It implements what this checkpoint "
                             f"uses; an op it does not know is a refusal, not "
                             f"a zero.")
        vals[n.output[0]] = out
        if want and n.name in want:
            got[n.name] = out
    return vals, got


def out_rank(x):
    return x.ndim


def _to_ints(x):
    """ONNX's Slice takes int64 tensors, but the exporter wrote floats.

    A float end of 2.5 on a dimension of 4 means 2, not 3: the value is a
    ceiling of a real division and the slice truncates. Truncating toward zero
    is what an int64 conversion does, so this matches what the runtime sees.
    """
    return [int(v) for v in np.atleast_1d(x).ravel()]


# -- the front end, so the reference and the runtime see the same pixels -------


def letterbox(rgb, size, pad=114.0):
    """rgb: uint8 HxWx3 -> (float32 3xsize x size, scale, pad_x, pad_y).

    The RESIZE IS PIL's BILINEAR, on purpose and with a reason. The runtime's
    front end is npue::vit::resize_to, whose contract is PIL-equivalence rather
    than "a smooth interpolation" -- measured in tools/verify/verify_vit_image.py
    to within one 8-bit LSB. So transcribing a bilinear filter here would be
    writing down a third implementation and calling it a reference; asking PIL
    for the resize makes the two sides agree on the one function that was already
    measured against a third party, and leaves the part this file is here to
    check -- the 116-node graph and the decode -- genuinely independent.

    The node-by-node diff in diff_pose_dump.py compares node 0's INPUT as well as
    its output, so a front-end disagreement is still caught rather than hidden by
    sharing the code.
    """
    from PIL import Image
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    small = np.asarray(
        Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), dtype=np.float64)
    canvas = np.full((size, size, 3), float(pad), dtype=np.float64)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = small
    x = canvas.transpose(2, 0, 1) / 255.0
    return x.astype(np.float32), scale, px, py


def load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# -- the decode, so both sides can be compared in the same units --------------


def nms(boxes, scores, iou_thr, max_det):
    order = np.argsort(-scores)
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(head, scale, pad_x, pad_y, src_w, src_h, conf, iou_thr, kpt_thr,
           max_det):
    """head: [4 + nc + nk*3, cells] -- the runtime's CANONICAL layout.

    Read here rather than reused, on purpose. The canonical layout is the one
    thing the two implementations share by design, so reading it independently
    here is a check that the layout is the documented one; a shared reader would
    make the comparison agree by construction.
    """
    nc, nk = NUM_CLASSES, NUM_KEYPOINTS
    xs = np.arange(head.shape[1])
    people = []
    for c in range(head.shape[1]):
        row = head[:, c]
        score = row[4]
        if score < conf:
            continue
        box = row[:4].copy()
        kp = row[4 + nc:4 + nc + nk * 3].reshape(nk, 3)
        # letterbox -> source. x' = (x_lb - pad)/scale, clipped to the image.
        box[0] = np.clip((box[0] - pad_x) / scale, 0, src_w)
        box[2] = np.clip((box[2] - pad_x) / scale, 0, src_w)
        box[1] = np.clip((box[1] - pad_y) / scale, 0, src_h)
        box[3] = np.clip((box[3] - pad_y) / scale, 0, src_h)
        kp[:, 0] = np.clip((kp[:, 0] - pad_x) / scale, 0, src_w)
        kp[:, 1] = np.clip((kp[:, 1] - pad_y) / scale, 0, src_h)
        people.append({"cell": c, "score": float(score), "box": box, "kpt": kp})
    people.sort(key=lambda p: -p["score"])
    if not people:
        return []
    boxes = np.array([p["box"] for p in people])
    keep = nms(boxes, np.array([p["score"] for p in people]), iou_thr, max_det)
    return [people[i] for i in keep]


def print_people(people, kpt_thr, src):
    print(f"{len(people)} person/people   (source {src[1]}x{src[0]})")
    for p in people:
        vis = int((p["kpt"][:, 2] >= kpt_thr).sum())
        b = p["box"]
        print(f"  cell {p['cell']:5d}  score {p['score']:.3f}  box "
              f"{b[0]:.0f},{b[1]:.0f} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}  "
              f"{vis}/{NUM_KEYPOINTS} keypoints visible")
        for i, j in SKELETON:
            a, c = p["kpt"][i], p["kpt"][j]
            if a[2] < kpt_thr or c[2] < kpt_thr:
                continue
            print(f"    {KEYPOINT_NAMES[i]:>12s}-{KEYPOINT_NAMES[j]:<12s} "
                  f"({a[0]:.0f},{a[1]:.0f}) -> ({c[0]:.0f},{c[1]:.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--kpt", type=float, default=0.50)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--dump-head", metavar="FILE",
                    help="write the head tensor (float32, [56, 8400]) to FILE")
    ap.add_argument("--compare-head", metavar="FILE",
                    help="compare the head tensor in FILE against this one")
    ap.add_argument("--npue", metavar="FILE",
                    help="REFUSED, and it always was: the node-by-node "
                         "comparison lives in tools/verify/diff_pose_dump.py, "
                         "which runs the runtime with --pose-dump and this "
                         "file's NumPy interpreter against it. Run that "
                         "instead.")
    ap.add_argument("--cache", metavar="FILE",
                    help="npz of every graph node's output. WRITTEN if it does "
                         "not exist and READ if it does. One ONNX run here is "
                         "about 20 minutes, and every experiment after the "
                         "first would otherwise pay it again; the cache is the "
                         "whole graph, so it is invalidated by any change to "
                         "the model or the image, which are the only two "
                         "inputs.")
    ap.add_argument("--head-inputs", metavar="FILE",
                    help="npz of the head's six TERMINAL tensors, as this "
                         "reference computed them. Used with --from-head to "
                         "re-run the head alone, which takes a second and is "
                         "where every head convention lives")
    args = ap.parse_args()
    if args.npue:
        # An ACCEPTED FLAG THAT DOES NOTHING is the failure this project treats
        # as the worst one, and this flag has been doing it: --npue was declared
        # to "compare against a packed container's conv outputs, node by node"
        # and the argument was never read. A caller who passed it, saw the
        # reference's own decode printed, and read that as a comparison, had a
        # passing gate that compared nothing. The comparison is real and it is
        # one file over: diff_pose_dump.py runs the runtime with --pose-dump and
        # compares every node against the interpreter in this file.
        raise SystemExit(
            "--npue does not do anything here. The node-by-node comparison "
            "against a packed container is tools/verify/diff_pose_dump.py:\n"
            "    python tools/verify/diff_pose_dump.py MODEL.onnx CONTAINER.npue "
            "IMAGE --dump runtime.bin\n"
            "which runs `npuimage pose CONTAINER.npue IMAGE --pose-dump "
            "runtime.bin` and compares every node against this file's NumPy "
            "interpreter. This file is the ONNX-side reference on its own; what "
            "it prints by default is the reference's own decode, not a "
            "comparison with the runtime.")

    model = onnx.load(args.onnx)
    rgb = load_image(args.image)
    src_h, src_w = rgb.shape[:2]
    x, scale, pad_x, pad_y = letterbox(rgb, 640)
    print(f"reference front end: {src_w}x{src_h} -> letterbox 640 "
          f"(scale {scale:.6f}, pad {pad_x},{pad_y})", file=sys.stderr)

    terminals = ["/model.22/Concat_1", "/model.22/cv4.0/cv4.0.2/Conv",
                 "/model.22/Concat_2", "/model.22/cv4.1/cv4.1.2/Conv",
                 "/model.22/Concat_3", "/model.22/cv4.2/cv4.2.2/Conv"]
    out_name = "/model.22/Concat_7"

    got = None
    if args.cache and os.path.exists(args.cache):
        with np.load(args.cache) as z:
            got = {k: z[k] for k in z.files}
        print(f"cache: read {len(got)} tensors from {args.cache}", file=sys.stderr)
    if got is None:
        want = {out_name} | set(terminals)
        _, got = run_onnx(model, {"images": x[None, ...]}, want=want)
        if args.cache:
            np.savez_compressed(args.cache, **got)
            print(f"cache: wrote {len(got)} tensors to {args.cache}",
                  file=sys.stderr)

    head_raw = got[out_name]                     # [1, 56, 8400] -- xywh
    if args.dump_head:
        np.asarray(head_raw[0], dtype=np.float32).tofile(args.dump_head)
        print(f"wrote {args.dump_head}", file=sys.stderr)

    # The runtime's CANONICAL tensor, derived from the ONNX output by the
    # documented conversion: the ONNX's first four channels are xywh in letterbox
    # pixels and the runtime's are xyxy in the same pixels.
    canonical = np.zeros_like(head_raw[0])
    x1 = head_raw[0, 0] - head_raw[0, 2] * 0.5
    y1 = head_raw[0, 1] - head_raw[0, 3] * 0.5
    canonical[0] = x1
    canonical[1] = y1
    canonical[2] = head_raw[0, 0] + head_raw[0, 2] * 0.5
    canonical[3] = head_raw[0, 1] + head_raw[0, 3] * 0.5
    canonical[4:5] = head_raw[0, 4:5]
    canonical[5:] = head_raw[0, 5:]

    if args.compare_head:
        other = np.fromfile(args.compare_head, dtype=np.float32)
        if other.size != canonical.size:
            raise SystemExit(f"reference head has {canonical.size} values and "
                             f"{args.compare_head} has {other.size}")
        other = other.reshape(canonical.shape)
        diff = np.abs(other - canonical)
        scale_ = np.maximum(np.abs(canonical), 1e-6)
        print(f"head tensor: max abs {diff.max():.6g}, "
              f"max rel {(diff / scale_).max():.6g}, "
              f"mean abs {diff.mean():.6g}")
        worst = int(np.argmax(diff.max(axis=0)))
        print(f"  worst cell {worst} (row {worst % canonical.shape[1]}), "
              f"ref {canonical[:, worst][:8]} vs other {other[:, worst][:8]}")

    people = decode(canonical, scale, pad_x, pad_y, src_w, src_h,
                    args.conf, args.iou, args.kpt, args.max_det)
    print_people(people, args.kpt, (src_h, src_w))


if __name__ == "__main__":
    main()