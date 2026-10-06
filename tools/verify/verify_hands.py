# Verify an arch=7 MediaPipe Hands container against onnxruntime.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# The packer is tools/pack/packers/hands.py. It does two separable things: it
# extracts weights from the checkpoint, and it rewrites the graph into an op
# list. Both are checked here, by RUNNING the container and comparing against
# onnxruntime on the same input tensor.
#
# Comparing end to end is not enough on its own. The palm detector's 2016 output
# rows index the anchor table positionally, so an anchor table in the wrong
# ORDER still produces 2016 well-shaped rows and a detector that confidently
# reports the wrong boxes. So the graph check is on the raw graph output, and
# the ORDER is checked separately, end to end: a wrong order moves the detection,
# and the detection is compared landmark by landmark against the oracle.
#
# WHAT IS NOT CHECKED HERE, AND WHY
# =================================
# The arithmetic of a convolution. This file implements convolutions in numpy
# because it is checking the PACKER's weights and geometry; it is not the
# runtime, and it is not evidence about anyone's kernels. Its job is narrower
# and it says so: "the container holds this checkpoint's numbers, in this
# arrangement, and running it reproduces onnxruntime."
#
# The reference is ORT rather than the OpenCV zoo demo on purpose: ORT executes
# the same ONNX file the container was packed from, so a disagreement is the
# packer's, not a difference of front ends. The zoo comparison lives in the
# packaging task's own notes, where the crop geometry was checked end to end.

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))

from npue import Reader  # noqa: E402

ARCH_MEDIAPIPE_HANDS = 7
ARCH_STRING = "mediapipe_hands_palm_ssd_lm_heatmap"


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def close(a, b, rtol_scale, what, floor=1.0):
    """Agreement measured RELATIVE TO THE TENSOR'S OWN SCALE, with a floor.

    Not relative element by element, because these are 53- and 47-convolution
    float32 chains and the honest question is "does the container hold this
    checkpoint", not "do two float32 summation orders agree to the last bit".
    An element whose expected value is 1e-7 while the tensor's scale is 1e3 tells
    you nothing; the max deviation over the whole tensor does.

    The floor is 1.0, and it is there for the two scalar heads. The landmark
    graph's presence and handedness outputs are sigmoids, and on a random input
    the presence probability lands at 1.9e-03 -- so "the tensor's scale" is
    1.9e-03 and the whole tensor has no dynamic range to speak of. Its error,
    4.7e-08 absolute, is 2.5e-05 of that 1.9e-03 and therefore fails a purely
    proportional test, while being 4.7e-08 of the logit it came from (about
    -6.3), which is where float32 through 47 convolutions actually earns it.
    With the floor at 1.0 the same number is 4.7e-08 of scale and the allowance
    is a flat 2e-05 -- which, on a probability in [0,1], is still an extremely
    tight statement: any structural error in that Gemm moves it by O(1).

    Measured against onnxruntime, whose kernels are a different implementation
    of the same arithmetic, on one random input per network, the worst
    deviations as a fraction of scale are 1.37e-06 (palm boxes, own scale
    1519), 1.06e-06 (palm scores, 216), 4.77e-07 (landmarks, 176),
    2.18e-07 (world landmarks, own scale 0.043 -> floored to 1),
    8.23e-07 (handedness, own scale 0.193 -> floored to 1) and 4.69e-08
    (presence, own scale 0.0019 -> floored to 1). The gate allows 2e-05, so
    there is headroom for a BLAS that sums in a different order -- while the
    transposed resize this file once had (off by 3e-01, five orders of magnitude
    worse) cannot pass.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise Fail(f"{what}: shape {a.shape} against {b.shape}")
    scale = max(float(np.abs(b).max()), floor)
    d = float(np.abs(a - b).max())
    if d > rtol_scale * scale:
        i = int(np.argmax(np.abs(a - b)))
        raise Fail(f"{what}: max deviation {d:.3g} is {d / scale:.3g} of the "
                   f"tensor's scale {scale:.4g}, over the {rtol_scale:g} allowed; "
                   f"worst element [{i}] {a.ravel()[i]:.9g} against "
                   f"{b.ravel()[i]:.9g}")
    return d / scale


# -- the ops ----------------------------------------------------------------
# Every one of these is a reference implementation, deliberately plain. The
# point of this file is to be obviously correct about the CONTAINER's metadata,
# and a clever convolution is not obviously correct about anything.

def _im2col(x, kh, kw, pad, stride=(1, 1)):
    """[1,Cin,H,W] -> [1, Cin*kh*kw, OH, OW], zero padded, strided. NCHW.

    Written with as_strided rather than a loop over taps because a 5x5 stem is
    25 taps and the landmark network is mostly 1x1, and both shapes then go
    through exactly one code path.

    The stride is applied to the OUTPUT SPATIAL axes only. Subsampling the
    flattened OH*OW axis instead looks equivalent and is not: it would skip a
    whole tap block per output pixel, which for a 3x3 is a different network.
    """
    _, c, h, w = x.shape
    pt, pl, pb, pr = pad
    sh, sw = stride
    oh, ow = h + pt + pb - kh + 1, w + pl + pr - kw + 1
    xp = np.zeros((1, c, h + pt + pb, w + pl + pr), np.float32)
    xp[:, :, pt:pt + h, pl:pl + w] = x
    s = xp.strides
    # Shape is [1, C, OH, kh, OW, kw], so the strides have to be
    # (batch, channel, ROW, ROW, COL, COL): the kh axis steps rows and the kw
    # axis steps columns. Swapping the middle pair reads past the end of the
    # buffer, which numpy answers with a segmentation fault rather than an
    # error -- the one place this file can die without saying why.
    view = np.lib.stride_tricks.as_strided(
        xp, shape=(1, c, oh, kh, ow, kw),
        strides=(s[0], s[1], s[2], s[2], s[3], s[3]), writeable=False)
    col = view.transpose(0, 1, 3, 5, 2, 4).reshape(1, c * kh * kw, oh, ow)
    return np.ascontiguousarray(col[:, :, ::sh, ::sw])


def op_conv(reader, op, xs, pfx):
    x = xs[op["inputs"][0]]
    sh, sw = op["stride"]
    i = op["conv"]
    w = reader.tensor(f"{pfx}.conv.{i}.w")
    b = reader.tensor(f"{pfx}.conv.{i}.b") if f"{pfx}.conv.{i}.b" in reader.entries else None
    cout, cin, kh, kw = w.shape
    col = _im2col(x, kh, kw, op["pad"], (sh, sw))
    oh, ow = col.shape[2], col.shape[3]
    y = (w.reshape(cout, -1) @ col[0].reshape(cin * kh * kw, -1))
    y = y.reshape(1, cout, oh, ow)
    if b is not None:
        y = y + b.reshape(1, -1, 1, 1)
    return y


def op_dwconv(reader, op, xs, pfx):
    """Depthwise: ONE filter per channel, so there is no [K,N] matrix and no
    single GEMM. Loop over channels -- slow, and deliberately so: this is the
    op the array backend cannot express, and a gate that quietly turned it into
    a dense convolution would stop being able to tell the difference."""
    x = xs[op["inputs"][0]]
    sh, sw = op["stride"]
    i = op["conv"]
    w = reader.tensor(f"{pfx}.conv.{i}.w")
    check(w.shape[1] == 1, f"dwconv {i}: weight is {w.shape}, not one input "
                           f"channel per filter")
    b = reader.tensor(f"{pfx}.conv.{i}.b") if f"{pfx}.conv.{i}.b" in reader.entries else None
    cout, _, kh, kw = w.shape
    col = _im2col(x, kh, kw, op["pad"], (sh, sw))
    oh, ow = col.shape[2], col.shape[3]
    y = np.empty((1, cout, oh, ow), np.float32)
    k = kh * kw
    flat = col[0].reshape(cout, k, oh * ow)
    for c in range(cout):
        # (k, OH*OW), not (k*OH*OW,): the filter is k long and the output is
        # OH*OW wide, and flattening the patch to one vector would make the
        # product a shape error at best.
        y[0, c] = (w[c].reshape(-1) @ flat[c]).reshape(oh, ow)
    if b is not None:
        y = y + b.reshape(1, -1, 1, 1)
    return y


def op_add(reader, op, xs, pfx):
    a, b = xs[op["inputs"][0]], xs[op["inputs"][1]]
    check(a.shape == b.shape, f"add: {a.shape} + {b.shape}")
    return a + b


def op_maxpool(reader, op, xs, pfx):
    x = xs[op["inputs"][0]]
    kh, kw = op["kernel"]
    sh, sw = op["stride"]
    pt, pl, pb, pr = op["pad"]
    _, c, h, w = x.shape
    xp = np.zeros((1, c, h + pt + pb, w + pl + pr), np.float32)
    xp[:, :, pt:pt + h, pl:pl + w] = x
    oh, ow = (h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp[0], shape=(c, oh, kh, ow, kw),
        strides=(s[1], sh * s[2], s[2], sw * s[3], s[3]), writeable=False)
    return view.max(axis=(2, 4))[None]


def op_pad_c(reader, op, xs, pfx):
    x = xs[op["inputs"][0]]
    k = op["count"]
    return np.concatenate([x, np.zeros((1, k) + x.shape[2:], np.float32)], axis=1)


def op_resize(reader, op, xs, pfx):
    """ONNX Resize, mode=linear, coordinate_transformation_mode=half_pixel,
    exclude_outside=0 -- the FPN neck's upsampling, spelled out because
    half-pixel is off by half a source pixel from the more common variants, and a
    resize a quarter-pixel out still produces a plausible-looking detection."""
    x = xs[op["inputs"][0]]
    sy, sx = op["scale"]
    _, c, h, w = x.shape
    nh, nw = op["out"][2], op["out"][3]
    yy = (np.arange(nh) + 0.5) / sy - 0.5
    xx = (np.arange(nw) + 0.5) / sx - 0.5
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    wy = (yy - y0).astype(np.float32)      # weight along the ROW axis
    wx = (xx - x0).astype(np.float32)      # weight along the COLUMN axis
    y0c, x0c = np.clip(y0, 0, h - 1), np.clip(x0, 0, w - 1)
    y1c, x1c = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
    # Two passes, and each weight belongs to the axis its pass moves along:
    # `bot` differs from `top` in the ROW index, so it is blended by wy; `right`
    # differs from `left` in the COLUMN index, so it is blended by wx. Putting
    # them the other way round is a TRANSPOSITION -- still bilinear, still the
    # right shape, still within a factor of two of right, and still a detector
    # that finds a hand. It is caught by comparing against ORT, not by looking.
    top = x[0][:, y0c][:, :, x0c]
    bot = x[0][:, y1c][:, :, x0c]
    left = top * (1 - wy)[None, :, None] + bot * wy[None, :, None]
    top = x[0][:, y0c][:, :, x1c]
    bot = x[0][:, y1c][:, :, x1c]
    right = top * (1 - wy)[None, :, None] + bot * wy[None, :, None]
    return (left * (1 - wx)[None, None, :] + right * wx[None, None, :])[None]


_BODY = {"conv": op_conv, "dwconv": op_dwconv, "add": op_add,
         "maxpool": op_maxpool, "pad_c": op_pad_c, "resize": op_resize}


def apply_act(reader, y, op, pfx):
    a = op.get("act", "none")
    if a == "none":
        return y
    if a == "prelu":
        k = op["prelu"]
        slope = reader.tensor(f"{pfx}.prelu.{k}.slope").reshape(1, -1, 1, 1)
        if slope.shape[1] != y.shape[1]:
            raise Fail(f"prelu {pfx}.{k} has {slope.shape[1]} slopes for a "
                       f"{y.shape[1]}-channel tensor")
        return np.where(y >= 0, y, y * slope).astype(np.float32)
    if a == "relu6":
        return np.clip(y, 0.0, 6.0).astype(np.float32)
    raise Fail(f"activation {a!r} is not one of none/prelu/relu6")


def run_body(reader, ops, image, pfx):
    """The op list -> every node's output. `image` is the input in NCHW, which is
    what operand -1 means, and `pfx` namespaces this network's tensors -- both
    networks number their convolutions from zero, so an unprefixed lookup reads
    the OTHER network's weights and returns a tensor of the wrong width."""
    xs = {-1: image}
    for n, op in enumerate(ops):
        for k in op["inputs"]:
            if k >= n:
                raise Fail(f"body op {n} takes operand {k}, which is not an "
                           f"earlier op or the image")
        fn = _BODY.get(op["op"])
        if fn is None:
            raise Fail(f"body op {n} is {op['op']!r}, which this reference "
                       f"does not implement; the vocabulary is "
                       f"{sorted(_BODY)}")
        y = fn(reader, op, xs, pfx)
        want = op.get("out")
        if want is not None and list(y.shape) != list(want):
            raise Fail(f"body op {n} ({op['op']}) produced {list(y.shape)}, "
                       f"but the container says {want}")
        xs[n] = apply_act(reader, y, op, pfx)
    return xs


# -- the two heads ----------------------------------------------------------

def palm_head(reader, head, xs):
    """Four NCHW feature maps -> the graph's [1,2016,18] and [1,2016,1].

    The reshape is the delicate part and is done in the graph's own axis order:
    the checkpoint transposes NCHW to NHWC and reshapes [H,W,C] to [H*W*A, 18],
    so row (h*W + w)*A + a, column term, is channel a*18 + term of cell (h, w).
    Doing it as a plain C-major flatten instead would permute every row, and the
    detector would still return 2016 rows of 18 numbers.
    """
    boxes, scores = [], []
    for lv in head:
        b = xs[lv["box"]]
        s = xs[lv["score"]]
        h, w, a = lv["h"], lv["w"], lv["anchors_per_cell"]
        check(b.shape == (1, a * 18, h, w), f"palm head: box level {lv} has "
                                           f"feature {b.shape}")
        check(s.shape == (1, a, h, w), f"palm head: score level {lv} has "
                                       f"feature {s.shape}")
        bb = b.reshape(1, a, 18, h, w).transpose(0, 3, 4, 1, 2)
        boxes.append(bb.reshape(1, h * w * a, 18))
        scores.append(s.reshape(1, a, h, w).transpose(0, 2, 3, 1)
                      .reshape(1, h * w * a, 1))
    bx, sc = np.concatenate(boxes, 1), np.concatenate(scores, 1)
    check(bx.shape[1] == reader.config["palm_num_anchors"],
          f"palm head: {bx.shape[1]} rows against "
          f"{reader.config['palm_num_anchors']} anchors in the config")
    return bx, sc


def lm_head(reader, head, xs):
    """The pooled feature map -> the graph's four outputs.

    Shapes are kept as the graph declares them -- [1,63], [1,1], [1,1], [1,63] --
    batch axis included, so a squeeze here would compare equal in value and
    unequal in shape against the reference.
    """
    x = xs[head["pool"]]
    pooled = x.reshape(x.shape[1], -1).mean(axis=1)
    check(int(pooled.shape[0]) == head["features"],
          f"lm head: pooled {pooled.shape[0]} channels against "
          f"{head['features']} in the config")
    outs = []
    for i, p in enumerate(head["projs"]):
        W = reader.tensor(p["w"])
        # Stored [in, out], ONNX Gemm with transB=0 is A @ W -- so the
        # projection is W.T @ pooled. Reading it as W @ pooled is a shape error
        # here, but a silent transpose is what a runtime would do, and 63 and 1
        # are far enough apart to be noticed, which is the only reason this is
        # a crash and not a wrong hand.
        y = W.T @ pooled + reader.tensor(p["b"])
        check(int(y.shape[0]) == p["n"], f"lm head: projection {i} produced "
                                         f"{y.shape[0]}, config says {p['n']}")
        outs.append(y.astype(np.float64)[None])
    result = []
    for o in head["outputs"]:
        v = outs[o["proj"]]
        result.append(1.0 / (1.0 + np.exp(-v)) if o["sigmoid"] else v)
    return result


# -- the host front end ------------------------------------------------------
# The geometry below is the OpenCV zoo demo's, and it is part of the model in
# every sense that matters: WHICH two of the palm detector's seven landmarks
# choose the rotation, and by how much the box is shifted and enlarged, decide
# where the landmark network looks. A landmark network pointed at a plausible
# but wrong crop returns a plausible hand, so none of this can be checked by
# looking at it.
#
# Two resamplers, and they are not interchangeable. The letterbox is
# cv2.INTER_LINEAR with no antialiasing; the final 224 crop is cv2.INTER_AREA,
# an exact box average. Using bilinear where INTER_AREA belongs re-reads the
# same one or two source rows over and over; measured against cv2 on random
# crops that mistake is a mean error of 0.21 of the full range, five times the
# model's own 8-bit quantisation.

def _blend(img, ys, xs):
    """bilinear sample at float coordinates, clamping at the border. warp()
    masks the outside itself, because clamping and zeroing are different
    answers and the zoo's warpAffine is border=0."""
    ys = np.asarray(ys, np.float64)
    xs = np.asarray(xs, np.float64)
    h, w = img.shape[:2]
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - y0, xs - x0
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    a = img[y0c, x0c].astype(np.float64)
    b = img[y0c, x1c].astype(np.float64)
    c = img[y1c, x0c].astype(np.float64)
    d = img[y1c, x1c].astype(np.float64)
    fx = fx[..., None]
    fy = fy[..., None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _resize_bilinear(img, nh, nw):
    """cv2.INTER_LINEAR, no antialiasing -- the letterbox.

    The half-pixel grid: dst(x) reads src at (x + 0.5) * scale - 0.5. Off by
    half a source pixel it is not, and the result looks like a bad model rather
    than a bad resample.
    """
    h, w = img.shape[:2]
    ys = (np.arange(nh) + 0.5) * h / nh - 0.5
    xs = (np.arange(nw) + 0.5) * w / nw - 0.5
    return _blend(img, ys[:, None], xs[None, :])


def _area_axis(n_src, n_dst):
    """Exact INTER_AREA weights for one axis, as (dst, src, weight) triples that
    cover every destination index once and sum to 1.

    No special case for upscaling. When the axis grows, the box [i*s, (i+1)*s)
    is narrower than one source cell and the same loop degrades on its own into
    a linear blend of the two cells it straddles -- which is what cv2 does, and
    NOT the nearest-neighbour its INTER_AREA documentation claims.
    """
    s = n_src / n_dst
    dst_idx, src_idx, wts = [], [], []
    for i in range(n_dst):
        y0, y1 = i * s, (i + 1) * s
        acc = 0.0
        for j in range(int(np.floor(y0)), min(int(np.ceil(y1)), n_src)):
            lo, hi = max(j, y0), min(j + 1, y1)
            if hi > lo:
                dst_idx.append(i)
                src_idx.append(j)
                wts.append((hi - lo) / s)
                acc += (hi - lo) / s
        if abs(acc - 1.0) > 1e-12 and len(wts):
            # floating-point residue on the coverage sum, onto the first tap
            k = len(wts) - len([x for x in wts if x > 0])
            wts[k] += 1.0 - acc
    return dst_idx, src_idx, wts


def _resize_area(img, nh, nw):
    """cv2.INTER_AREA -- an exact box average, separable, each axis exact."""
    h, w = img.shape[:2]
    iy, iy_j, iy_w = _area_axis(h, nh)
    ix, ix_j, ix_w = _area_axis(w, nw)
    rows = np.zeros((nh, w, img.shape[2]), np.float64)
    for a, j, wt in zip(iy, iy_j, iy_w):
        rows[a] += img[j] * wt
    out = np.zeros((nh, nw, img.shape[2]), np.float64)
    for a, j, wt in zip(ix, ix_j, ix_w):
        out[:, a] += rows[:, j] * wt
    return out


def _rotation_matrix(centre, angle):
    a = np.deg2rad(angle)
    ca, sa = np.cos(a), np.sin(a)
    m = np.array([[ca, sa, 0.0], [-sa, ca, 0.0]])
    m[0, 2] = centre[0] - m[0, 0] * centre[0] - m[0, 1] * centre[1]
    m[1, 2] = centre[1] - m[1, 0] * centre[0] - m[1, 1] * centre[1]
    return m


def _warp(img, m):
    """cv2.warpAffine(img, M, INTER_LINEAR, border=0).

    M maps SOURCE to DEST, so the resampling for a destination pixel uses M^-1.
    Using M directly rotates the hand the wrong way and the landmark network
    absorbs it well enough to still return a plausible hand: the worst kind of
    bug, because it does not look wrong.
    """
    h, w = img.shape[:2]
    a, b, tx = m[0, 0], m[0, 1], m[0, 2]
    c, d, ty = m[1, 0], m[1, 1], m[1, 2]
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    # the inverse translation pairs row 0 against the row-0 inverse entries
    itx = -(ia * tx + ib * ty)
    ity = -(ic * tx + id_ * ty)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = ia * xx + ib * yy + itx
    y = ic * xx + id_ * yy + ity
    out = _blend(img, y, x)
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return out * inside[..., None]


def _border(img, top, bottom, left, right):
    if top or bottom or left or right:
        img = np.pad(img, ((top, bottom), (left, right), (0, 0)))
    return img


def _nms(boxes, scores, score_thr, nms_thr, top_k):
    order = np.argsort(-scores)
    keep = []
    while len(order) and len(keep) < top_k:
        i = order[0]
        if scores[i] < score_thr:
            break
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (ai + ar - inter) <= nms_thr]
    return np.array(keep, np.int64)


# -- two backends, one geometry ----------------------------------------------
# The geometry is shared deliberately. Comparing the container's numbers against
# ORT's through the SAME front end is what makes section 2 sharp: any difference
# is a difference in the networks, and there is no second implementation of the
# geometry for it to be blamed on. Section 3 is where the geometry itself is
# pinned, against recorded numbers -- because two sides agreeing proves they
# agree, not that either is right.

class ContainerBackend:
    """The two networks as the .npue holds them."""

    def __init__(self, reader):
        self.r = reader
        self.palm_ops = json.loads(reader.config["palm_graph"])
        self.palm_head = json.loads(reader.config["palm_head"])
        self.lm_ops = json.loads(reader.config["lm_graph"])
        self.lm_head = json.loads(reader.config["lm_head"])
        self.anchors = reader.tensor("palm_anchors").astype(np.float64)

    def palm(self, blob):
        x = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))
        return palm_head(self.r, self.palm_head,
                         run_body(self.r, self.palm_ops, x, "palm"))

    def lm(self, blob):
        x = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))
        return lm_head(self.r, self.lm_head,
                       run_body(self.r, self.lm_ops, x, "lm"))


class OrtBackend:
    """The same two networks as ORT sees them: the ONNX files the container was
    packed from. The reference is ORT rather than the OpenCV zoo demo on
    purpose -- ORT executes the same file, so a disagreement is the packer's and
    not a difference of front ends."""

    def __init__(self, palm_path, lm_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # the sessions are NOT called palm/lm: those names are the methods, and
        # an attribute would shadow the method the pipeline drives
        self.palm_sess = ort.InferenceSession(str(palm_path), so,
                                              providers=["CPUExecutionProvider"])
        self.lm_sess = ort.InferenceSession(str(lm_path), so,
                                            providers=["CPUExecutionProvider"])
        self.palm_in = self.palm_sess.get_inputs()[0].name
        self.lm_in = self.lm_sess.get_inputs()[0].name
        self.anchors = None       # set by the caller: the anchor table is not
                                  # in the ONNX file, so the ORT side is handed
                                  # the container's and section 2 compares
                                  # networks, not tables

    def palm(self, blob):
        return self.palm_sess.run(None, {self.palm_in: blob})

    def lm(self, blob):
        return self.lm_sess.run(None, {self.lm_in: blob})


class Hands:
    """The zoo demo's two stages, driven through either backend."""

    def __init__(self, backend, cfg, verbose=False):
        self.b = backend
        self.c = cfg
        self.verbose = verbose
        self.palm_size = int(cfg["palm_input_size"])
        self.lm_size = int(cfg["lm_input_size"])
        self.anchors = backend.anchors

    # -- palm detection -----------------------------------------------------
    def detect_palms(self, img):
        """HxWx3 uint8 RGB -> boxes [N,4] x1y1x2y2, keypoints [N,7,2], scores [N]."""
        h, w = img.shape[:2]
        # The anchors are FRACTIONS of the long side, so the decode multiplies
        # by max(h, w) and not by the resized one.
        scale = float(max(h, w))
        blob, pad_bias = self._letterbox(img)
        x = (blob.astype(np.float32) / 255.0)[None]
        b, s = self.b.palm(x)
        b, s = b[0], s[0][:, 0]
        # The score is a LOGIT: the graph's last node is the sigmoid's input and
        # the zoo applies 1/(1+exp(-x)) here, not in the graph.
        s = 1.0 / (1.0 + np.exp(-s.astype(np.float64)))
        A = self.anchors
        cxy = b[:, :2] / self.palm_size
        wh = b[:, 2:4] / self.palm_size
        xy1 = (cxy - wh / 2 + A) * scale
        xy2 = (cxy + wh / 2 + A) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]
        keep = _nms(boxes, s, self.c["score_threshold"],
                    self.c["nms_threshold"], self.c["top_k"])
        if self.verbose:
            print(f"    palm: {int((s >= self.c['score_threshold']).sum())}"
                  f" over threshold, {len(keep)} after NMS")
        if len(keep) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0, 7, 2), np.float32),
                    np.zeros((0,), np.float64))
        s, boxes = s[keep], boxes[keep]
        lm = b[keep, 4:].reshape(-1, 7, 2) / self.palm_size + A[keep][:, None, :]
        lm = lm * scale - pad_bias
        return boxes.astype(np.float32), lm.astype(np.float32), s

    def _letterbox(self, img):
        """Short side to palm_size, symmetric pad. Returns the image and the
        (left, top) pad in ORIGINAL pixels."""
        n = self.palm_size
        h, w = img.shape[:2]
        r = min(n / h, n / w)
        rs = (np.array([h, w]) * r).astype(np.int32)
        img = _resize_bilinear(img, int(rs[0]), int(rs[1]))
        ph, pw = n - rs[0], n - rs[1]
        left, top = pw // 2, ph // 2
        img = _border(img, top, ph - top, left, pw - left)
        return img, (np.array([left, top]) / r).astype(np.int32)

    # -- landmark network ---------------------------------------------------
    def landmarks(self, img, box, kps):
        blob, rot_bbox, angle, rot_m, pad_bias = self._preprocess(img, box, kps)
        out = self.b.lm(blob)
        return self._postprocess(out, rot_bbox, angle, rot_m, pad_bias)

    def _crop_pad(self, img, bbox, for_rotation):
        shift = np.array(self.c["palm_pre_shift"] if for_rotation
                         else self.c["palm_shift"], np.float64)
        enlarge = (self.c["palm_pre_enlarge"] if for_rotation
                   else self.c["palm_enlarge"])
        wh = bbox[1] - bbox[0]
        bbox = bbox + shift * wh
        ctr = bbox.sum(0) / 2
        wh = bbox[1] - bbox[0]
        half = wh * enlarge / 2
        bbox = np.array([ctr - half, ctr + half])
        bbox = np.clip(bbox.astype(np.int32), [0, 0],
                       [img.shape[1], img.shape[0]])
        img = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        # The rotation stage squares the crop by its DIAGONAL, not its long
        # side: the palm is rotated about its centre and a long-side square
        # clips the fingertips on a wide palm.
        side = (int(np.linalg.norm(img.shape[:2])) if for_rotation
                else int(max(img.shape[:2])))
        ph, pw = side - img.shape[0], side - img.shape[1]
        left, top = pw // 2, ph // 2
        img = _border(img, top, ph - top, left, pw - left)
        bias = bbox[0] - np.array([left, top])
        return img, bbox, bias

    def _preprocess(self, img, box, kps):
        pad_bias = np.zeros(2, np.int32)
        bbox = np.asarray(box, np.float64).reshape(2, 2)
        img, bbox, bias = self._crop_pad(img, bbox, True)
        pad_bias = pad_bias + bias
        bbox = bbox - pad_bias
        pl = np.asarray(kps, np.float64) - pad_bias
        p1, p2 = pl[self.c["palm_lm_wrist"]], pl[self.c["palm_lm_middle_base"]]
        rad = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        rad -= 2 * np.pi * np.floor((rad + np.pi) / (2 * np.pi))
        angle = np.rad2deg(rad)
        ctr = bbox.sum(0) / 2
        rot_m = _rotation_matrix(ctr, angle)
        img = _warp(img, rot_m)
        hom = np.c_[pl, np.ones(len(pl))]
        rl = np.array([hom @ rot_m[0], hom @ rot_m[1]])
        rot_bbox = np.array([rl.min(axis=1), rl.max(axis=1)])
        crop, rot_bbox, _ = self._crop_pad(img, rot_bbox, False)
        n = self.lm_size
        blob = _resize_area(crop, n, n).astype(np.float32) / 255.0
        return blob[None], rot_bbox, angle, rot_m, pad_bias

    def _postprocess(self, out, rot_bbox, angle, rot_m, pad_bias):
        lm, pres, hand, world = out[0][0], out[1][0], out[2][0], out[3][0]
        lm = lm.reshape(-1, 3)
        world = world.reshape(-1, 3)
        wh = rot_bbox[1] - rot_bbox[0]
        sc = wh / float(self.lm_size)
        lm[:, :2] = (lm[:, :2] - self.lm_size / 2.0) * sc.max()
        lm[:, 2] *= sc.max()
        rot2 = _rotation_matrix((0, 0), angle)
        lm_rot = np.c_[lm[:, :2] @ rot2[:, :2], lm[:, 2]]
        w_rot = np.c_[world[:, :2] @ rot2[:, :2], world[:, 2]]
        rc = np.array([[rot_m[0][0], rot_m[1][0]], [rot_m[0][1], rot_m[1][1]]])
        tc = np.array([rot_m[0][2], rot_m[1][2]])
        inv_t = np.array([-rc[0] @ tc, -rc[1] @ tc])
        inv = np.c_[rc, inv_t]
        ctr = np.append(rot_bbox.sum(0) / 2, 1.0)
        oc = np.array([ctr @ inv[0], ctr @ inv[1]])
        lm[:, :2] = lm_rot[:, :2] + oc + pad_bias
        bb = np.array([lm[:, :2].min(0), lm[:, :2].max(0)])
        whb = bb[1] - bb[0]
        bb = bb + np.array(self.c["hand_shift"]) * whb
        half = (bb[1] - bb[0]) * self.c["hand_enlarge"] / 2
        c = bb.sum(0) / 2
        bb = np.array([c - half, c + half])
        return {"bbox": bb, "landmarks": lm, "world": w_rot,
                "handedness": hand, "presence": float(pres[0])}

    # -- the whole thing ----------------------------------------------------
    def run(self, path, all_hands=False):
        img = np.asarray(Image.open(path).convert("RGB"))
        boxes, kps, scores = self.detect_palms(img)
        if len(scores) == 0:
            return []
        if all_hands:
            return [self.landmarks(img, boxes[i], kps[i]) for i in range(len(scores))]
        i = int(np.argmax(scores))
        return [self.landmarks(img, boxes[i], kps[i])]


# -- the gate ----------------------------------------------------------------

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MODEL_DIR = os.path.join(REPO, "models", "mediapipe-hands")
PALM_ONNX = os.path.join(MODEL_DIR, "palm_detection_mediapipe_2023feb.onnx")
LM_ONNX = os.path.join(MODEL_DIR, "handpose_estimation_mediapipe_2023feb.onnx")
IMAGE = os.path.join(MODEL_DIR, "hand_plain.png")
# The SAME container with the pre-tiled bf16 B panels staged, which is what the
# array path dispatches against. A separate file on purpose: staging takes it from
# 8.9 MB to 13.1 MB, and one path for both would mean the container section 4
# checked against the golden is not the container section 5 ran the array on.
NPU_CONTAINER = os.path.join(MODEL_DIR, "hands-npu.npue")
# The two paths' landmark distance, measured on this frame at mean 1.63 px, median
# 1.72 and worst 4.71 -- bf16 panel precision, the same order as arch=6's array
# path at 1.1 px. The budget is 1.7x the measured worst, and it is HERE rather
# than in the registry because this is the gate's own claim: the array path is not
# interchangeable with the host, and a reader who wants to know by how much should
# not have to open another file to find it.
RT_LANDMARK_XY_ARRAY = 8.0
# The golden lives in reference/goldens/, NOT beside the container in models/.
# models/** is gitignored except for CHECKPOINT.json -- see .gitignore's block on
# that -- so a golden written there would be present on this machine and absent
# from every fresh checkout, and section 3 would then refuse on a machine that has
# done nothing wrong. reference/goldens/ is where this tree keeps the tracked
# goldens (the *_taps.npz and encode_npu_*.json files), and the IMAGE stays
# untracked in models/mediapipe-hands/ next to the two ONNX files it was fetched
# with: it is an input fixture, not a claim, and the golden's own image_sha256 is
# what proves the one on disk is the one the numbers were recorded from.
GOLDEN = os.path.join(REPO, "reference", "goldens", "hands_mediapipe.json")
# The gate's own tolerance, not a fudge factor: 2e-05 of a tensor's scale is
# fourteen times the 1.4e-06 measured against ORT in close()'s docstring, and
# 2e-05 of a 520-pixel frame is a hundredth of a pixel.
TOL_SCALE = 2e-05
# The front end, against the zoo, on this frame. Every one of these is a
# MEASURED deviation, not a round number chosen to make the gate pass:
#
#   bbox                1.086 px     landmark x/y  1.271 px (mean 0.452)
#   landmark depth      0.338 px     world        7.2e-04
#   presence            4.42e-05     handedness    8.11e-05
#
# and the allowance is about 20% over the measurement. The residual is a
# systematic one part in a hundred in the crop's scale, and it has one cause:
# this file recomputes the letterbox in float64 where cv2 rounds back to 8 bits,
# so the detector looks at a very slightly different image and the detection
# moves by about a hundredth of its own size. That is a property of being a
# second implementation of a resampler, not slack in the test -- and it is
# nowhere near the size of the failures the section exists to catch. The wrong
# pair of landmarks for the rotation axis puts the fingertips tens of pixels
# out; bilinear where INTER_AREA belongs moves them further; an anchor table in
# the wrong order moves the whole detection. None of those is sub-pixel.
TOL_BBOX_PX = 1.5
TOL_LANDMARK_PX = 1.5
# The depth axis is in the same PIXELS as x and y, relative to the wrist, so it
# rides the same 1% scale and its own allowance is a quarter of a pixel.
TOL_DEPTH_PX = 0.5
TOL_WORLD = 1e-2
TOL_PROB = 2e-4
# The palm detector's own score. Its own number, not a probability: it is the
# sigmoid of the head's logit and on this frame it lands at 0.894.
TOL_SCORE = 2e-3

# -- section 4, the C++ runtime ------------------------------------------------
# The tolerances for the runtime against THIS file's own front end. They are
# measured, like every other number in this file, and the measurement is
# recorded in tools/verify/README or in the section's docstring when it drifts.
#
#   detection box  0.618 px   landmarks  0.650 px worst, 0.470 mean
#   depth          0.240 px   hand box   0.510 px
#   score          2.78e-03   presence   1.0e-05   handedness  1.1e-05
#   world          1.85e-04
#
# The cause of the sub-pixel part is the same one section 3 documents, and it
# runs the OTHER way round: the runtime rounds the letterboxed palm input back
# to 8 bits because cv2 does, so it is the CLOSER of the two to the zoo and
# this file is the one that differs. The two disagree by about a hundredth of
# the detection's own size, and everything downstream of that inherits it.
#
# The score's allowance is 4e-3 rather than a share of a pixel, because the
# score is a LOGIT passed through a sigmoid and the logit is where the
# quantisation lands: 0.0028 on a score of 0.894 is a shift of about 0.02 in
# the logit, which is the same 8-bit rounding seen through a steep part of the
# sigmoid rather than through a coordinate. It is a big number next to 2e-4
# next to it, and it is the right size: the same fault that moves the box by
# half a pixel moves the score by three thousandths, and a tolerance that
# tracked the other quantities would either hide this or fire on nothing.
RT_BOX_PX = 1.0
RT_LANDMARK_PX = 1.0
RT_DEPTH_PX = 0.5
RT_HANDBOX_PX = 1.0
RT_SCORE = 4e-3
RT_PROB = 1e-4
RT_WORLD = 1e-3
# Where the runtime binary is expected. It is BUILT, not shipped, so a gate that
# cannot find it says so and stops -- a runtime section that quietly skipped
# itself would leave the C++ path unmeasured while reporting OK.
BINARY = os.path.join(REPO, "runtime", "build", "npuembeddings")


def _need(path, what):
    if not os.path.exists(path):
        raise Fail(f"{what} is not there: {path}\n"
                   f"Run tools/pack/pack_npue.py for this model first "
                   f"(the packer is tools/pack/packers/hands.py), or fetch the "
                   f"checkpoint into models/mediapipe-hands/ -- see that "
                   f"directory's CHECKPOINT.json.")


def anchors_from_levels(levels):
    """The 2016 SSD anchor centres, derived here from the head's level list.

    Written out in this file rather than imported from the packer, because a
    check that calls the thing it is checking is not a check. Two independent
    spellings of one derivation still catch each other's typos, which is what
    this is for; what it cannot catch is a formula that is wrong in both, and
    the honest limit of it is written in section 3.

    Cell (h, w) of a level is at x = (w + 0.5)/W, y = (h + 0.5)/H, and
    carries `a` identical rows. The row order is the graph's own: the head
    reshapes its [H, W, a*18] feature map to [H*W*a, 18] with h outer and w
    inner, so the flat row index advances w -- the X coordinate -- fastest,
    which is why the first component of a row is x and not y. Swapping the two
    components transposes the whole table into the mirror image, which is still
    2016 well-shaped rows and still a confident detector.
    """
    out = []
    for H, W, a in levels:
        for h in range(H):
            for w in range(W):
                for _ in range(a):
                    out.append(((w + 0.5) / W, (h + 0.5) / H))
    return np.array(out, np.float64)


def check_anchor_order(reader, verbose):
    """The container's anchor table against the one derived from its own levels.

    This exists because of a MEASURED blind spot in the end-to-end check, not
    out of caution. On hand_plain.png the highest-scoring palm cell is row
    1766, which belongs to the SECOND level (rows 1152..2016), so permuting the
    first level's 1152 rows leaves the winning anchor untouched and the
    landmarks do not move at all: measured, transposing the 24x24 grid
    reproduces the correct answer to within 0.001 px while the same fault in
    the second level moves a landmark by 224 px. The end-to-end check cannot
    see a wrong order in the first level on this image, so this is what sees
    it.
    """
    levels = [tuple(l) for l in json.loads(reader.config["palm_anchor_levels"])]
    got = reader.tensor("palm_anchors").astype(np.float64)
    want = anchors_from_levels(levels)
    check(got.shape == want.shape,
          f"the container's anchor table is {got.shape}, the levels "
          f"{levels} derive {want.shape}")
    # 1e-6 relative: the table is stored F32, so it cannot agree to more than
    # about 6e-8 of 1.0, and the derived values are exact in float64.
    d = float(np.abs(got - want).max())
    check(d <= 1e-6,
          f"the anchor table is {d:.3g} from the one its own levels derive; "
          f"first disagreement at row "
          f"{int(np.argmax(np.abs(got - want).max(axis=1)))}: "
          f"{got[np.argmax(np.abs(got - want).max(axis=1))].tolist()} against "
          f"{want[np.argmax(np.abs(got - want).max(axis=1))].tolist()}")
    check(int(reader.config["palm_num_anchors"]) == want.shape[0],
          f"the config says {reader.config['palm_num_anchors']} anchors, the "
          f"levels derive {want.shape[0]}")
    # the pyramid's own strides, as fractions of the 192 input
    for H, _, _ in levels:
        stride = 192 / H
        check(stride == int(stride) and stride in (8, 16, 32),
              f"a level of {H} cells is a stride of {stride}, which is not one "
              f"of the pyramid's 8, 16, 32 over a 192 input")
    if verbose:
        print(f"     {want.shape[0]} anchors from {levels}, worst "
              f"disagreement {d:.2e}")


def section_1(reader, verbose):
    """The graphs, node for node, against ORT on the same input tensor."""
    print("  1. graph output against onnxruntime")
    cb = ContainerBackend(reader)
    ob = OrtBackend(PALM_ONNX, LM_ONNX)
    rng = np.random.default_rng(20230517)
    worst = 0.0
    for tag, cbn, obn, size in (("palm", cb.palm, ob.palm,
                                 int(reader.config["palm_input_size"])),
                                ("landmark", cb.lm, ob.lm,
                                 int(reader.config["lm_input_size"]))):
        x = rng.standard_normal((1, 3, size, size), dtype=np.float32)
        blob = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        got, exp = cbn(blob), obn(blob)
        check(len(got) == len(exp), f"{tag}: container returns {len(got)} "
                                     f"outputs, onnxruntime {len(exp)}")
        for i, (g, e) in enumerate(zip(got, exp)):
            worst = max(worst, close(g, e, TOL_SCALE, f"{tag} output {i}"))
    print(f"     worst deviation {worst:.2e} of scale, allowed "
          f"{TOL_SCALE:.0e}")


def section_2(reader, verbose):
    """End to end on a real photograph, container against ORT.

    This is what says the container is WIRED, not merely that it holds the
    numbers: the same letterbox, the same 2016 rows decoded against the same
    anchor table, the same NMS, the same rotation crop, and both networks.
    """
    print("  2. the whole pipeline, container against onnxruntime")
    cb = ContainerBackend(reader)
    ob = OrtBackend(PALM_ONNX, LM_ONNX)
    ob.anchors = cb.anchors     # the anchor table is the CONTAINER's; section 1
                                # already pinned the rows it indexes
    cfg = reader.config
    a = Hands(cb, cfg, verbose=verbose)
    b = Hands(ob, cfg, verbose=verbose)
    ra_list, rb_list = a.run(IMAGE), b.run(IMAGE)
    check(len(ra_list) == 1 and len(rb_list) == 1,
          f"the container found {len(ra_list)} hands and onnxruntime "
          f"{len(rb_list)}; this frame has one. If BOTH are zero the fault is "
          f"in the front end, not in the container -- the two sides share it, "
          f"so a geometry mistake takes them out together and there is nothing "
          f"to disagree about.")
    ra, rb = ra_list[0], rb_list[0]
    close(ra["bbox"], rb["bbox"], TOL_SCALE, "hand bbox", floor=520.0)
    close(ra["landmarks"], rb["landmarks"], TOL_SCALE, "landmarks",
          floor=520.0)
    close(ra["world"], rb["world"], TOL_SCALE, "world landmarks", floor=1.0)
    close(ra["handedness"], rb["handedness"], TOL_SCALE, "handedness")
    close(np.array([ra["presence"]]), np.array([rb["presence"]]), TOL_SCALE,
          "presence")
    print(f"     one hand, bbox {np.round(ra['bbox'], 1).ravel().tolist()}, "
          f"presence {ra['presence']:.4f}")


def _pixels(a):
    return np.asarray(a, np.float64).reshape(-1)


def section_3(reader, verbose):
    """The front end, against the OpenCV zoo's own numbers on this frame.

    Sections 1 and 2 share their front end -- same letterbox, same SSD decode,
    same rotation crop, same resamplers -- so between them they can only show
    that the container and onnxruntime agree. They cannot show that either is
    RIGHT. This section is the only thing that does, and it is why the golden
    exists: it is recorded from the zoo's MPPalmDet and MPHandPose, run
    verbatim, rather than from this file.

    It is also what a wrong anchor ORDER cannot survive. A shape-valid anchor
    table in the wrong order still gives 2016 rows of 18 numbers and a
    confident detector pointing somewhere else; here that detector's landmarks
    land 1.27 px from the zoo's when the order is right, and nowhere near them
    when it is not.
    """
    print("  3. the front end against the recorded OpenCV zoo numbers")
    if not os.path.exists(GOLDEN):
        raise Fail(f"{GOLDEN} is not there, and this file is the only thing "
                   f"that checks the front end -- it cannot check itself. See "
                   f"the section's docstring for where the numbers come from.")
    want = json.load(open(GOLDEN))
    _need_image_is(want)
    hands = Hands(ContainerBackend(reader), reader.config, verbose=verbose)
    # The DETECTOR's own output, before the landmark net gets a say. Without
    # this the section looks only at the single best hand, and the whole NMS is
    # untested: with the IoU threshold at 1.0 the frame yields seventeen boxes
    # instead of one, and `run` -- which by design reports only the best --
    # cannot tell the difference. Measured, because it was measured.
    img = np.asarray(Image.open(IMAGE).convert("RGB"))
    boxes, kps, scores = hands.detect_palms(img)
    check(len(scores) == want["n_detections_after_nms"],
          f"the detector returned {len(scores)} hands after NMS at IoU "
          f"{hands.c['nms_threshold']}, the zoo returned "
          f"{want['n_detections_after_nms']}. A threshold that stops "
          f"suppressing gives seventeen boxes and one good answer, and a check "
          f"that only reads the best box sees nothing wrong with either.")
    d_s = abs(float(scores[0]) - want["palm_score"])
    check(d_s <= TOL_SCORE,
          f"the winning palm score is {float(scores[0]):.6f} against the zoo's "
          f"{want['palm_score']:.6f}, off by {d_s:.3g}")

    h = hands.run(IMAGE)
    check(len(h) == 1, f"the pipeline reported {len(h)} hands")
    ra = h[0]

    d_bb = float(np.abs(_pixels(ra["bbox"]) - np.array(want["bbox"])).max())
    check(d_bb <= TOL_BBOX_PX,
          f"hand bbox is {d_bb:.3f} px from the zoo's, over {TOL_BBOX_PX} px: "
          f"{np.round(ra['bbox'], 1).ravel().tolist()} against "
          f"{np.round(want['bbox'], 1).tolist()}")

    lm_g = np.asarray(ra["landmarks"], np.float64)
    lm_w = np.array(want["landmarks"], np.float64)
    check(lm_g.shape == lm_w.shape,
          f"landmarks are {lm_g.shape} against the recorded {lm_w.shape}")
    d = np.abs(lm_g - lm_w)
    d_xy = float(d[:, :2].max())
    if d_xy > TOL_LANDMARK_PX:
        i = int(np.unravel_index(d[:, :2].argmax(), d[:, :2].shape)[0])
        raise Fail(f"landmark {i} is {d_xy:.3f} px from the zoo's, over "
                   f"{TOL_LANDMARK_PX} px: {np.round(lm_g[i], 2).tolist()} "
                   f"against {np.round(lm_w[i], 2).tolist()}. The rotation "
                   f"axis is the first thing to suspect -- it is two landmarks "
                   f"out of seven, and the wrong pair still returns a hand.")
    d_z = float(d[:, 2].max())
    check(d_z <= TOL_DEPTH_PX,
          f"a landmark depth is {d_z:.3f} px from the zoo's, over "
          f"{TOL_DEPTH_PX} px")

    w_g = np.asarray(ra["world"], np.float64)
    w_w = np.array(want["world"], np.float64)
    check(w_g.shape == w_w.shape,
          f"world landmarks are {w_g.shape} against {w_w.shape}")
    d_w = float(np.abs(w_g - w_w).max())
    check(d_w <= TOL_WORLD,
          f"a world landmark is {d_w:.3g} from the zoo's, over {TOL_WORLD:g} "
          f"(metres, on a hand about 0.18 m across)")

    d_p = abs(ra["presence"] - want["presence"])
    check(d_p <= TOL_PROB,
          f"presence is {ra['presence']:.6f} against the zoo's "
          f"{want['presence']:.6f}, off by {d_p:.3g}")
    d_h = float(np.abs(np.asarray(ra["handedness"], np.float64).ravel()
                       - np.array(want["handedness"])).max())
    check(d_h <= TOL_PROB,
          f"handedness is {d_h:.3g} from the zoo's, over {TOL_PROB:g}")

    print(f"     bbox within {d_bb:.3f} px, landmarks within {d_xy:.3f} px "
          f"(depth {d_z:.3f}), world within {d_w:.1e}, presence within "
          f"{d_p:.1e}")


def section_5_array(reader, verbose):
    """The array path against the host path, on the same frame.

    WHY A WHOLE SECTION FOR A FLAG WHOSE DEFAULT IS THE HOST: because the two
    answers are NOT the same and nothing else here can see it. Sections 1 to 4 all
    run the host, so a gate built only from them would pass with the array path
    returning anything at all.

    WHAT IS ASSERTED AND WHAT IS ONLY MEASURED. Asserted: the array path runs, it
    dispatches a plausible number of times, the count of hands agrees with the
    host's, and the LANDMARKS land within 8 px -- the bf16 panel budget, measured
    at a mean 1.63 px and a worst 4.71. Measured and reported: the exact
    disagreement and the frame times, because any tolerance on the timing would be
    a claim that the array wins, and it does not.

    THE DESIGN SET AND THE PANELS ARE BOTH NAMED IN THE REFUSAL, because the two
    are separate requirements and a caller who has one has not got the other. The
    container for the array path is a DIFFERENT FILE from the golden's: staging the
    panels takes it from 8.9 MB to 13.1 MB, and reusing one path for both would
    mean the container section 4 checked is not the container this ran on.
    """
    print("  5. the array path against the host path, same frame")
    design = os.path.join(REPO, "runtime", "artifacts", "mediapipe-hands",
                          "artifacts_npu1", "gemm_rtp", "design.json")
    if not os.path.exists(design):
        raise Fail(
            f"{design} is not there, so there is no array path to compare the host "
            f"against. Build it with\n"
            f"  source /opt/xilinx/xrt/setup.sh\n"
            f"  .venv/bin/python tools/export/export_gemm_rtp.py --target "
            f"mediapipe-hands --arch 1 -n 32\n"
            f"which needs MLIR-AIE, and that is in .venv/ rather than on the "
            f"system PATH.")
    # A NAME and not the absolute directory: artifacts_candidates(root, named)
    # treats its argument as a name and builds the candidates under it, so handing
    # it an absolute path makes it concatenate the two. The runtime's own refusal
    # message is the good message here -- it prints every path it looked at, and
    # the doubled ones make the mistake obvious.
    art = os.path.join("runtime", "artifacts", "mediapipe-hands", "artifacts_npu1")
    container = NPU_CONTAINER
    if not os.path.exists(container):
        raise Fail(
            f"{container} is not there, so the array path has no panels to "
            f"dispatch into. Pack it with\n"
            f"  python tools/pack/pack_npue.py --hands-onnx "
            f"models/mediapipe-hands --out {container} --npu --device npu1\n"
            f"and note --npu needs --device: N pads to tile_n*cols and cols is a "
            f"property of the array.")

    def run(extra):
        cmd = [BINARY, "hands", container, IMAGE] + extra
        p = subprocess.run(cmd, capture_output=True, text=True)
        check(p.returncode == 0,
              f"`{' '.join(cmd)}` exited {p.returncode}\n{p.stderr[-600:]}")
        line = [l for l in p.stderr.splitlines() if "array:" in l]
        return json.loads(p.stdout), (line[0].strip() if line else "")

    host, _ = run([])
    arr, line = run(["--npu-ops", "conv", "--artifacts", art])
    check(line, "the status block printed no array line, so nothing was "
                "dispatched and this section would be comparing two host runs")

    check(len(arr["hands"]) == len(host["hands"]) == 1,
          f"the host found {len(host['hands'])} hands and the array "
          f"{len(arr['hands'])}; the two must agree on the COUNT for the rest of "
          f"this section to mean anything")
    ph, pa = host["hands"][0], arr["hands"][0]
    lh = np.asarray(ph["landmarks"], np.float64)
    la = np.asarray(pa["landmarks"], np.float64)
    check(lh.shape == la.shape,
          f"the host returned {lh.shape} landmark rows and the array "
          f"{la.shape}")
    xy = np.abs(lh[:, :2] - la[:, :2]).max(axis=1)
    check(xy.max() <= RT_LANDMARK_XY_ARRAY,
          f"the array's landmarks are {xy.max():.3f} px from the host's, "
          f"outside the {RT_LANDMARK_XY_ARRAY} px the bf16 panels allow. "
          f"Measured on this frame: mean {xy.mean():.3f}, median "
          f"{np.median(xy):.3f}, worst {xy.max():.3f}.")
    th = host["timings_ms"]["total"]
    ta = arr["timings_ms"]["total"]
    print(f"     landmarks mean {xy.mean():.3f} / median {np.median(xy):.3f} / "
          f"max {xy.max():.3f} px between the two paths (bf16 panels)")
    print(f"     {line}")
    print(f"     frame     {ta:.0f} ms on the array against {th:.0f} ms on the "
          f"host -- {ta / th:.2f}x")
    check(ta > 0 and th > 0, "a frame time of zero would make the ratio a "
                              "division by nothing")
    check(ta > th,
          f"the array path is now {th / ta:.2f}x FASTER than the host "
          f"({ta:.0f} ms against {th:.0f} ms). That is a real result and a good "
          f"one, and the registry cell, NPU_OPS.md and NPU_MODELS.md all say the "
          f"array is slower here -- so this failing is the signal to re-measure "
          f"and rewrite them, not a regression.")


def section_4(reader, verbose):
    """The C++ runtime, against this file's own front end and networks.

    SECTIONS 1 TO 3 NEVER TOUCH THE RUNTIME
    ---------------------------------------
    They run the container through numpy. Every one of them would pass, in full,
    with runtime/src/hands/ deleted: the packer, the graph walk written twice,
    the anchor order, the zoo's numbers. That is a real gap, because the thing
    a caller runs is the binary, and the two implementations differ in more than
    their language:

      * the runtime reads the container's graphs with common/json_min.hpp and
        walks them in C++, where this file reads the same JSON with json.loads;
      * it ROUNDS the letterboxed palm input back to 8 bits, because cv2 does
        and this file deliberately does not, so on this frame the two
        implementations' detection boxes differ by 0.618 px before anything else
        is considered;
      * it accumulates the convolutions in its own GEMM order, so its tensors
        differ from numpy's by float32 rounding rather than by agreement.

    So this section is not "does the C++ agree with itself". It is the one place
    where the shipped binary is measured at all, and its tolerances come from
    the run recorded above rather than from a round number.

    AND THE TOLERANCES ARE PIXELS, NOT TENSORS
    -------------------------------------------
    Sections 1 and 2 hold 2e-05 of a tensor's scale, which is the right unit for
    two float32 evaluations of the same arithmetic. This section cannot use it,
    because the two sides here are NOT two evaluations of the same arithmetic:
    the runtime looks at a slightly different photograph. Every allowance below
    is therefore in the caller's own units -- pixels, metres, probabilities --
    and the failures this section exists to catch are the ones that are NOT
    sub-pixel: a wrong rotation axis moves fingertips by tens of pixels, an
    anchor table in the wrong order moves the whole detection, and INTER_LINEAR
    where INTER_AREA belongs moves the fingertips by a pixel and a half.

    A MISSING BINARY STOPS THE GATE
    --------------------------------
    Not skips it. The runtime is built, not committed, so on a fresh checkout
    this section has nothing to run and saying "OK" would report the C++ path as
    measured when it was not measured at all. Build it first:
        cmake -S runtime -B runtime/build && cmake --build runtime/build
    """
    print("  4. the C++ runtime against this file's own front end")
    if not os.path.exists(BINARY):
        raise Fail(
            f"the runtime binary is not at {BINARY}, and this is the only "
            f"place any gate measures it. Sections 1 to 3 run the container "
            f"through numpy and would pass in full without it, so skipping is "
            f"not an option -- report OK would mean the C++ path was measured. "
            f"Build it: cmake -S runtime -B runtime/build && "
            f"cmake --build runtime/build")
    hands = Hands(ContainerBackend(reader), reader.config, verbose=verbose)
    img = np.asarray(Image.open(IMAGE).convert("RGB"))
    boxes, kps, scores = hands.detect_palms(img)
    ref = hands.run(IMAGE)
    check(len(ref) == 1, f"this file's own pipeline found {len(ref)} hands, so "
                         f"there is nothing to compare the runtime against")

    # The runtime writes its JSON to stdout and its diagnostics to stderr, so
    # the capture keeps the two apart rather than interleaving a status block
    # into a JSON document.
    import subprocess
    # No -q: the runtime's status block goes to stderr, which this captures and
    # only prints if the run fails, so the run stays quiet in the normal case.
    cmd = [BINARY, "hands", CONTAINER[0], IMAGE]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise Fail(f"`{' '.join(cmd)}` exited {p.returncode}:\n"
                   f"{p.stderr.strip()[-2000:]}")
    try:
        got = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise Fail(f"the runtime's stdout is not one JSON object ({e}). Its "
                   f"diagnostics are supposed to go to stderr; the first 200 "
                   f"characters of stdout were: {p.stdout[:200]!r}")
    check(len(got.get("hands", [])) == 1 and len(got.get("detections", [])) == 1,
          f"the runtime returned {len(got.get('detections', []))} detections "
          f"and {len(got.get('hands', []))} hands; this file's own front end "
          f"returns 1 and 1 on this frame")
    rh = got["hands"][0]
    rd = got["detections"][0]
    ra = ref[0]

    d_box = float(np.abs(np.asarray(rd["box"], np.float64) - boxes[0]).max())
    if d_box > RT_BOX_PX:
        raise Fail(f"the runtime's detection box is {d_box:.3f} px from this "
                   f"file's, over {RT_BOX_PX} px: "
                   f"{[round(v, 2) for v in rd['box']]} against "
                   f"{[round(float(v), 2) for v in boxes[0]]}. Sub-pixel on "
                   f"this frame is the letterbox's 8-bit rounding; anything "
                   f"larger is the SSD decode -- the anchor order, the pad, or "
                   f"the sigmoid on the score.")

    d_lm = np.abs(np.asarray(rh["landmarks"], np.float64)
                  - np.asarray(ra["landmarks"], np.float64))
    d_xy = float(d_lm[:, :2].max())
    if d_xy > RT_LANDMARK_PX:
        i = int(np.unravel_index(d_lm[:, :2].argmax(), d_lm[:, :2].shape)[0])
        raise Fail(f"the runtime's landmark {i} is {d_xy:.3f} px from this "
                   f"file's, over {RT_LANDMARK_PX} px: "
                   f"{np.round(np.asarray(rh['landmarks'])[i], 2).tolist()} "
                   f"against {np.round(np.asarray(ra['landmarks'])[i], 2).tolist()}"
                   f". The rotation's centre and the crop's scale are the first "
                   f"two to suspect, and both of them move every joint together "
                   f"rather than one.")
    d_z = float(d_lm[:, 2].max())
    check(d_z <= RT_DEPTH_PX,
          f"a runtime landmark depth is {d_z:.3f} px from this file's, over "
          f"{RT_DEPTH_PX} px. The depth rides the crop's single scale, so a "
          f"depth-only failure means x and y are being scaled by different "
          f"factors.")

    d_hb = float(np.abs(np.asarray(rh["bbox"], np.float64)
                        - np.asarray(ra["bbox"], np.float64).ravel()).max())
    check(d_hb <= RT_HANDBOX_PX,
          f"the runtime's hand box is {d_hb:.3f} px from this file's, over "
          f"{RT_HANDBOX_PX} px. It is fitted to the LANDMARKS, so it cannot "
          f"differ by more than they do by much -- unless the enlarge factor or "
          f"the shift is being applied to one corner instead of both.")

    d_s = abs(rd["score"] - float(scores[0]))
    check(d_s <= RT_SCORE,
          f"the runtime's palm score is {rd['score']:.6f} against this file's "
          f"{float(scores[0]):.6f}, off by {d_s:.3g}, over {RT_SCORE:g}. The "
          f"score is a logit through a sigmoid, so the same 8-bit rounding that "
          f"moves the box by half a pixel arrives here as several thousandths.")

    for key, want_val, what in (("presence", ra["presence"], "presence"),
                                ("handedness",
                                 float(np.asarray(ra["handedness"],
                                                  np.float64).ravel()[0]),
                                 "handedness")):
        d = abs(rh[key] - want_val)
        check(d <= RT_PROB,
              f"the runtime's {what} is {rh[key]:.6f} against this file's "
              f"{want_val:.6f}, off by {d:.3g}, over {RT_PROB:g}")

    d_w = float(np.abs(np.asarray(rh["world_landmarks"], np.float64)
                       - np.asarray(ra["world"], np.float64)).max())
    check(d_w <= RT_WORLD,
          f"a runtime world landmark is {d_w:.3g} from this file's, over "
          f"{RT_WORLD:g} (metres, on a hand about 0.18 m across). The world "
          f"points take the rotation but not the translation, so a difference "
          f"proportional to the hand's POSITION in the frame is a postprocess "
          f"that applied one.")

    print(f"     detection box within {d_box:.3f} px, landmarks within "
          f"{d_xy:.3f} px (depth {d_z:.3f}), hand box within {d_hb:.3f} px, "
          f"score within {d_s:.1e}, world within {d_w:.1e}")


def _need_image_is(want):
    """The golden names the image it was recorded on, and says so by hash.

    A landmark list is only a landmark list for a particular photograph: on a
    different one these numbers are not slightly wrong, they are unrelated.
    """
    got = hashlib.sha256(open(IMAGE, "rb").read()).hexdigest()
    check(got == want["image_sha256"],
          f"the golden was recorded on a {want['image']} with sha256 "
          f"{want['image_sha256'][:16]}..., and this is one with "
          f"{got[:16]}...; the landmarks it holds are not wrong, they are "
          f"about a different photograph")


# -- injected faults ---------------------------------------------------------
# Fourteen of them, each a mistake this file's own comments name as one that a
# shape check cannot catch, and each one confirmed to fail this gate. A gate
# nobody has watched fail is a gate nobody knows what it covers.
#
# Run with --inject. It does not check the container: it breaks the code on
# purpose and asserts that every break is caught, and that the un-broken code
# is not. Anything reported MISSED is a hole in the sections above, and the
# holes are printed rather than counted, because a count of 13/14 tells nobody
# which one.

_PRISTINE = {}


def _snapshot():
    if _PRISTINE:
        return
    _PRISTINE.update({
        "Reader.tensor": Reader.tensor, "op_resize": op_resize,
        "_BODY.resize": _BODY["resize"], "_resize_area": _resize_area,
        "_nms": _nms, "pre": Hands._preprocess, "post": Hands._postprocess,
        "crop": Hands._crop_pad, "main": main,
    })


def _restore():
    _snapshot()
    Reader.tensor = _PRISTINE["Reader.tensor"]
    _BODY["resize"] = _PRISTINE["_BODY.resize"]
    _resize_area = _PRISTINE["_resize_area"]
    _nms = _PRISTINE["_nms"]
    Hands._preprocess = _PRISTINE["pre"]
    Hands._postprocess = _PRISTINE["post"]
    Hands._crop_pad = _PRISTINE["crop"]


def _cfg_override(key, value, where="crop"):
    """Run one front-end method with one config key changed under it.

    The geometry constants live in the container, so the realistic fault is a
    wrong VALUE in there, and it has to be injected where it is read.
    """
    _snapshot()
    slot = {"crop": "_crop_pad", "pre": "_preprocess",
            "post": "_postprocess"}[where]
    real = getattr(Hands, slot)

    def bad(self, *a, **k):
        keep = self.c
        self.c = dict(keep, **{key: value(keep[key])})
        try:
            return real(self, *a, **k)
        finally:
            self.c = keep
    setattr(Hands, slot, bad)
    return lambda: setattr(Hands, slot, real)


def _swap(key, value):
    """Replace one module-level function or one entry of the op table."""
    _snapshot()
    if key in _BODY:
        real = _BODY[key]
        _BODY[key] = value
        return lambda: _BODY.__setitem__(key, real)
    real = globals()[key]
    globals()[key] = value
    return lambda: globals().__setitem__(key, real)


def _anchor_table(z):
    _snapshot()
    real = Reader.tensor

    def patched(self, name):
        return z if name == "palm_anchors" else real(self, name)
    Reader.tensor = patched
    return lambda: setattr(Reader, "tensor", real)


def _transpose_level(z, n0, rows):
    blk = z[n0:n0 + rows].reshape(-1, 2, 2)
    return np.concatenate([z[:n0],
                           np.ascontiguousarray(blk.transpose(1, 0, 2).reshape(rows, 2)),
                           z[n0 + rows:]])


def inject(quiet=True):
    """Every fault, one at a time. Returns (caught, missed) as lists of
    (tag, what-said)."""
    _snapshot()
    z = Reader(CONTAINER[0]).tensor("palm_anchors").astype(np.float64)

    def resize_transposed(reader, op, xs, pfx):
        # the bug this file really had: each weight applied to the other axis
        x = xs[op["inputs"][0]]
        sy, sx = op["scale"]
        _, c, h, w = x.shape
        nh, nw = op["out"][2], op["out"][3]
        yy = (np.arange(nh) + 0.5) / sy - 0.5
        xx = (np.arange(nw) + 0.5) / sx - 0.5
        y0 = np.floor(yy).astype(int)
        x0 = np.floor(xx).astype(int)
        wy = (yy - y0).astype(np.float32)
        wx = (xx - x0).astype(np.float32)
        y0c, x0c = np.clip(y0, 0, h - 1), np.clip(x0, 0, w - 1)
        y1c, x1c = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
        t = x[0][:, y0c][:, :, x0c]
        b = x[0][:, y1c][:, :, x1c]
        left = t * (1 - wx)[None, None, :] + b * wx[None, None, :]
        t = x[0][:, y0c][:, :, x1c]
        b = x[0][:, y1c][:, :, x1c]
        right = t * (1 - wx)[None, None, :] + b * wx[None, None, :]
        return (left * (1 - wy)[None, :, None] + right * wy[None, :, None])[None]

    def area_as_bilinear(img, nh, nw):
        return np.asarray(_resize_bilinear(img, nh, nw))

    cases = [
        ("anchor: level 1 transposed", lambda: _anchor_table(_transpose_level(z, 0, 1152))),
        ("anchor: level 2 transposed", lambda: _anchor_table(_transpose_level(z, 1152, 864))),
        ("anchor: x and y swapped", lambda: _anchor_table(np.ascontiguousarray(z[:, ::-1]))),
        ("anchor: not cell centres", lambda: _anchor_table(z / 2)),
        ("resize: axes transposed", lambda: _swap("resize", resize_transposed)),
        ("resize: half-pixel as asymmetric",
         lambda: _swap("resize", lambda r, o, xs, p: _resize_asym(o, xs))),
        ("front end: rotation axis on the wrong landmark",
         lambda: _cfg_override("palm_lm_middle_base", lambda v: 5, "pre")),
        ("front end: palm pre-enlarge halved",
         lambda: _cfg_override("palm_pre_enlarge", lambda v: v * 0.5)),
        ("front end: palm shift sign flipped",
         lambda: _cfg_override("palm_shift", lambda v: [-x for x in v])),
        ("front end: landmark crop enlarged instead",
         lambda: _cfg_override("palm_enlarge", lambda v: v * 4)),
        ("front end: hand box not re-enlarged",
         lambda: _cfg_override("hand_enlarge", lambda v: 1.0, "post")),
        ("front end: INTER_AREA replaced by bilinear",
         lambda: _swap("_resize_area", area_as_bilinear)),
        ("front end: NMS IoU threshold at 1.0",
         lambda: _swap("_nms", lambda bx, s, a, n, k: _PRISTINE["_nms"](bx, s, a, 1.0, k))),
        ("front end: score threshold at 0.99",
         lambda: _swap("_nms", lambda bx, s, a, n, k: _PRISTINE["_nms"](bx, s, 0.99, n, k))),
    ]
    _restore()
    # Each fault re-enters through main(), so the flag that got us here has to
    # leave argv first: main() would otherwise read --inject again and recurse
    # until the stack runs out, which is a confusing way to learn that a fault
    # injection harness calls its own entry point.
    argv = list(sys.argv)
    sys.argv = [a for a in argv if a != "--inject"]
    out = []
    out.append(("baseline, no fault", _run_and_report(quiet)))
    for tag, setup in cases:
        undo = setup()
        out.append((tag, _run_and_report(quiet)))
        undo()
        _restore()
    sys.argv = argv
    return out


def _resize_asym(op, xs):
    """half_pixel, with the half-pixel dropped -- the neighbouring convention,
    a quarter of a source pixel out."""
    x = xs[op["inputs"][0]]
    sy, sx = op["scale"]
    _, c, h, w = x.shape
    nh, nw = op["out"][2], op["out"][3]
    yy = np.arange(nh) / sy
    xx = np.arange(nw) / sx
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    wy = (yy - y0).astype(np.float32)
    wx = (xx - x0).astype(np.float32)
    y0c, x0c = np.clip(y0, 0, h - 1), np.clip(x0, 0, w - 1)
    y1c, x1c = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
    left = (x[0][:, y0c][:, :, x0c] * (1 - wy)[None, :, None]
            + x[0][:, y1c][:, :, x0c] * wy[None, :, None])
    right = (x[0][:, y0c][:, :, x1c] * (1 - wy)[None, :, None]
             + x[0][:, y1c][:, :, x1c] * wy[None, :, None])
    return (left * (1 - wx)[None, None, :] + right * wx[None, None, :])[None]


def _run_and_report(quiet):
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
            main()
        return "MISSED"
    except Fail as e:
        return "caught: " + str(e).split("\n")[0]
    except SystemExit as e:
        return f"caught (exit {e.code}): " + (str(e)[:70] if e else "")
    except Exception as e:  # a traceback is still a failure, but say so plainly
        return f"caught ({type(e).__name__}): {str(e)[:70]}"


# Filled in by main(): the injection cases re-enter through main(), and the
# path they read is the one the operator asked for.
CONTAINER = [None]


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="verify an arch=7 MediaPipe Hands container against "
                    "onnxruntime and the recorded zoo geometry")
    ap.add_argument("--container", default=None,
                    help="the .npue to check (default: the packed one)")
    # There is deliberately no --dump-golden. A golden this file could
    # regenerate is a golden that records whatever the file currently believes,
    # which is the one thing it must not do. hands_golden.json is written by
    # hand from the OpenCV zoo and its provenance is in its own _doc field.
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--inject", action="store_true",
                    help="break the code on purpose, fourteen ways, and check "
                         "that every break fails the gate -- then check that "
                         "the unbroken code passes. Does not inspect the "
                         "container except to read its anchor table.")
    args = ap.parse_args()

    container = args.container or os.path.join(MODEL_DIR, "hands.npue")
    CONTAINER[0] = container
    _need(container, "the packed container")
    reader = Reader(container)
    check(reader.arch == ARCH_MEDIAPIPE_HANDS,
          f"arch is {reader.arch}, this gate is for {ARCH_MEDIAPIPE_HANDS}")
    check(reader.config.get("kind") == "hands",
          f"kind is {reader.config.get('kind')!r}, not 'hands'")
    _need(IMAGE, "the test photograph")
    if args.inject:
        _need(PALM_ONNX, "the palm checkpoint")
        _need(LM_ONNX, "the landmark checkpoint")
        print("injected faults, each one a mistake this file's comments name")
        results = inject(quiet=not args.verbose)
        missed = []
        for tag, said in results:
            print(f"  {tag:52s} {said}")
            if said == "MISSED":
                missed.append(tag)
        _restore()
        bad = [t for t in missed if not t.startswith("baseline")]
        if results[0][1] != "MISSED":
            raise Fail(f"the unbroken code did not pass: {results[0][1]}")
        if bad:
            raise Fail(f"{len(bad)} of {len(results) - 1} injected faults were "
                       f"NOT caught: {bad}")
        print(f"OK: {len(results) - 1} injected faults, all caught; the "
              f"unbroken code passes.")
        return
    _need(PALM_ONNX, "the palm checkpoint")
    _need(LM_ONNX, "the landmark checkpoint")
    print(f"arch {reader.arch} ({ARCH_STRING}), kind hands, "
          f"{reader.config['palm_num_anchors']} anchors, "
          f"{reader.config['palm_num_convs']} + {reader.config['lm_num_convs']} "
          f"convolutions")
    check_anchor_order(reader, args.verbose)
    for fn in (section_1, section_2, section_3, section_4,
                section_5_array):
        fn(reader, args.verbose)
    print("OK: the container holds this checkpoint, in this arrangement, the "
          "front end is the one the numbers were recorded from, the C++ "
          "runtime reproduces all four stages of it, and the array path "
          "dispatches against a design set checked in both directions -- "
          "slower than the host and a different answer, both measured and both "
          "in the registry cell.")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
