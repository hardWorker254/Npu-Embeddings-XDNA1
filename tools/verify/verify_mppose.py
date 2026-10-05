#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify an arch=8 MediaPipe Pose container and the C++ runtime built from it.

WHAT THIS FILE IS, AND WHAT IT IS NOT
-------------------------------------
The arithmetic of the convolutions is written in numpy here, because this file is
checking the PACKER's weights and graph and then the RUNTIME's front end, and
neither is evidence about anyone's kernels. Its job is narrower than that of a
kernel benchmark and it says so: "this container holds these two checkpoints in
this arrangement, and the C++ front end answers the recorded OpenCV zoo's
question."

FOUR SECTIONS, AND THE SPLIT IS THE USEFUL PART
-----------------------------------------------
  1. the packed op list, against onnxruntime. ORT executes the same ONNX file the
     container was packed from, so a disagreement here is the PACKER's and not a
     difference of front ends. All seven graph outputs of both networks.
  2. the anchor table, against an independent derivation from the container's own
     pyramid levels.
  3. the C++ runtime, against the numbers recorded from the OpenCV zoo.
  4. the array path, against the host path, on the same frame.

Section 1 passing and section 3 failing is a FRONT END fault and nothing else.
That separation is the reason the gate is split and not one pass: end to
end, a wrong letterbox and a wrong last convolution are the same sentence.

Section 4 exists because `--npu-ops conv` dispatches, is 2.4x SLOWER than the
host, and its landmarks sit up to 44 px from the host's -- three facts that are
invisible to sections 1 to 3, all of which run the host. It needs a design set
and a container packed with --npu; the design set needs MLIR-AIE, which is in
.venv/ and not on the system PATH, and section 4's refusal says so.

WHY THE REFERENCE FOR SECTION 3 IS THE ZOO AND NOT ORT
-------------------------------------------------------
ORT says the graphs are right. It says nothing about whether the region of
interest is the right square, whether the rotation is about the right point, or
whether the mask lands at the right offset -- none of which is in either graph.
Those are questions only a front end answers, and the answer of record is the
OpenCV zoo's, recorded in reference/goldens/mppose_{det,pose}.json.

WHAT THE GOLDEN IS, AND WHY THIS FILE CANNOT WRITE IT
-----------------------------------------------------
There is deliberately no --dump-golden, and there is no path in this file that
writes reference/goldens/. A golden this file could regenerate is a golden that
records whatever the file currently believes, which is the one thing it must not
do. The two JSONs were written by hand from a recorder that ran the zoo's
_preprocess and _postprocess on docs/bus.jpg; their _doc fields say exactly how,
including the one adapter cv2 5.0.0 forces on the detector's output order.

THE NUMBERS BELOW ARE MEASUREMENTS, AND EVERY ONE OF THEM HAS ITS CAUSE
----------------------------------------------------------------------
Against the recorded zoo answer on docs/bus.jpg (810x1080), this runtime:

  detector  score       5.3e-07            box + 4 keypoints   0.00049 px
  pose      conf        2.05e-03
           landmarks x/y  mean 0.575  median 0.272  p90 0.861  max 4.380 px
           landmarks z    mean 0.851  median 0.629  p90 1.724  max 3.243 px
           visibility     6.2e-03           presence  6.7e-05
           world          mean 0.00216  max 0.01514 m
           mask non-zero  53705 against 53767, 0.115 %

The detector is exact because its two halves are exact: the letterbox is a plain
bilinear at cv2's own grid and the network is the one ORT agrees with. The pose
residuals all come from ONE cause, and it is not slack in the test:

    cv2.warpAffine evaluates bilinear in FIXED POINT -- five bits of weight and
    ten of source coordinate -- so an exact float64 bilinear is not the same
    function. Measured on this stage the two disagree on 0.27 % of pixels with a
    mean absolute difference of 0.16, and that reaches 0.81 % of the resampled
    crop.

Reproducing cv2's quantisation is not done, and the reason is in tools/pack/
packers/mppose.py's sibling note: these resamplers are shared with arch=7, whose
numbers were recorded with the exact ones, so changing them is its own decision
and not one to smuggle in here. What matters for this gate is that the residual
is an order of magnitude smaller than any fault the sections exist to catch --
the wrong DepthToSpace order moves a landmark by hundreds of pixels, the wrong
pair of points for the rotation by tens, and an anchor table transposed by the
width of a whole person.

THE FOUR WORST LANDMARKS ARE NOT RANDOM
---------------------------------------
They are the left foot, ankle, heel and knee -- the far-side limb, which the
crop carries least information about, and which is exactly where a different
resampler and a different network would disagree most. The six auxiliary rows
are 0.10 to 0.25 px. A gate that checked only the worst row would be measuring
the model and a gate that checked only the mean would be measuring the torso.

SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))

from npue import Reader  # noqa: E402

ARCH_MEDIAPIPE_POSE = 8
ARCH_STRING = "mediapipe_pose_det_ssd_lm_regress"

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
MODEL_DIR = os.path.join(REPO, "models", "mediapipe-pose")
DET_ONNX = os.path.join(MODEL_DIR, "person_detection_mediapipe_2023mar.onnx")
POSE_ONNX = os.path.join(MODEL_DIR, "pose_estimation_mediapipe_2023mar.onnx")
CONTAINER = os.path.join(MODEL_DIR, "mppose.npue")
# The SAME container with the pre-tiled bf16 B panels staged, which is what the
# array path dispatches against. It is a separate file on purpose: staging the
# panels takes it from 18.3 MB to 29.9 MB, and reusing one path for both would
# mean the container section 3 checked against the golden is not the container
# section 4 ran the array on.
NPU_CONTAINER = os.path.join(MODEL_DIR, "mppose-npu.npue")
IMAGE = os.path.join(REPO, "docs", "bus.jpg")
GOLDEN_DET = os.path.join(REPO, "reference", "goldens", "mppose_det.json")
GOLDEN_POSE = os.path.join(REPO, "reference", "goldens", "mppose_pose.json")
BINARY = os.path.join(REPO, "runtime", "build", "npuembeddings")

# Seeded, and the same every run: section 1 must fail the SAME way twice for the
# message to be worth anything.
SEED = 20261005

# Section 1's tolerance, peak-relative to each output's own largest value. The
# measured worst is 9.9e-05, on the 256x256 segmentation mask, and this is 1e-3.
# The number that decides the threshold is not that measurement: it is that the
# wrong DepthToSpace order measures 9.9e-01 against the same reference, so the
# threshold sits four orders of magnitude above anything a real fault does and
# one order above float32 accumulation over ~120 layers.
TOL_PEAK_REL = 1e-3

# Section 2 is exact. The container stores a table the packer generated and
# checked bit for bit against the OpenCV zoo's literal 2254 x 2 table; a
# "tolerance" here would be an invitation.
TOL_ANCHOR_BITS = 0

# Section 3, against the recorded zoo numbers, with about 2x to 20x of headroom
# on each. The detector's numbers are effectively exact and are held tightly
# because there is no reason for them to move; the pose numbers are the
# resampler's residual and are held at twice what it measures.
DET_TOL_PX = 0.01          # measured 0.00049
DET_TOL_SCORE = 1e-5       # measured 5.3e-07
POSE_TOL_CONF = 5e-3       # measured 2.05e-03
LM_TOL_XY_MEAN = 1.5       # measured 0.575
LM_TOL_XY_MAX = 6.0        # measured 4.380
LM_TOL_Z_MEAN = 2.5        # measured 0.851
LM_TOL_Z_MAX = 5.0         # measured 3.243
LM_TOL_VIS = 2e-2          # measured 6.2e-03
LM_TOL_PRES = 5e-4         # measured 6.7e-05
WORLD_TOL_MEAN = 5e-3      # measured 0.00216 m
WORLD_TOL_MAX = 3e-2       # measured 0.01514 m
MASK_TOL_FRAC = 5e-3       # measured 1.15e-03 of the non-zero count


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)
    return True


def _need(path, what):
    if not os.path.exists(path):
        raise Fail(f"{what} is not there: {path}\n"
                   f"  Pack it with `python tools/pack/pack_npue.py "
                   f"--mppose-onnx models/mediapipe-pose --out "
                   f"{CONTAINER}`, or fetch the two checkpoints into "
                   f"models/mediapipe-pose/ -- that directory's CHECKPOINT.json "
                   f"names both repositories and both sha256s.")


# ==============================================================================
# -- a numpy reader for the packed graph --------------------------------------
#
# The arithmetic of a convolution, written out so this file checks the packer's
# weights and geometry rather than trusting them. NOT the runtime, and not
# evidence about anyone's kernels. The vocabulary is the container's and is
# closed: an op this file does not implement is a refusal, not a skip.
# ==============================================================================


def _im2col(x, kh, kw, pad, stride):
    """[C,H,W] -> [C*kh*kw, oh*ow], with the column order (c, ki, kj) and c
    OUTERMOST -- the same order the filter's own [Cout, Cin, kh, kw] flattens
    in, so a dense convolution is one matrix product with no permutation."""
    pt, pl, pb, pr = pad
    sh, sw = stride
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))
    Hp, Wp = xp.shape[1], xp.shape[2]
    oh = (Hp - kh) // sh + 1
    ow = (Wp - kw) // sw + 1
    col = np.empty((x.shape[0] * kh * kw, oh * ow), dtype=np.float32)
    i = 0
    for c in range(x.shape[0]):
        for ki in range(kh):
            for kj in range(kw):
                blk = xp[c, ki:ki + (oh - 1) * sh + 1:sh,
                         kj:kj + (ow - 1) * sw + 1:sw]
                check(blk.shape == (oh, ow),
                      f"im2col: the {kh}x{kw} gather at stride "
                      f"{sh},{sw} ran off a {Hp}x{Wp} padded input")
                col[i] = blk.reshape(-1)
                i += 1
    return col, oh, ow


def _act(v, name):
    if name is None or name == "none":
        return v
    if name == "relu":
        return np.maximum(v, 0.0)
    if name == "relu6":
        return np.clip(v, 0.0, 6.0)
    raise Fail(f"activation {name!r} is not one this file implements")


def _resize(x, sy, sx):
    """ONNX Resize, linear, half_pixel, exclude_outside=0. Every `Resize` in
    these two graphs was checked to carry exactly that, by the packer."""
    C, H, W = x.shape
    oh, ow = int(round(H * sy)), int(round(W * sx))
    ys = (np.arange(oh) + 0.5) * (H / oh) - 0.5
    xs = (np.arange(ow) + 0.5) * (W / ow) - 0.5
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0).astype(np.float32)
    fx = (xs - x0).astype(np.float32)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
    # fy AND fx ARE EXPLICITLY BROADCAST ONTO THEIR OWN AXES. Writing
    # `a * (1 - fy)` broadcasts a length-oh vector over the LAST axis, which is
    # ow -- so the row weights are applied to the columns, and for a square output
    # with a uniform scale the two vectors are EQUAL and the mistake is invisible.
    #
    # That is not hypothetical: this file was written that way, and every Resize
    # in BOTH of these graphs happens to be a square uniform-scale upsample of a
    # square map, so all seven outputs still matched onnxruntime to 1e-06 and the
    # defect would have shipped in the reference the whole gate is measured
    # against. A reference that is wrong in a way the model does not exercise is
    # worse than no reference: it looks checked.
    a = x[:, y0c][:, :, x0c] * (1 - fx)[None, None, :] + \
        x[:, y0c][:, :, x1c] * fx[None, None, :]
    b = x[:, y1c][:, :, x0c] * (1 - fx)[None, None, :] + \
        x[:, y1c][:, :, x1c] * fx[None, None, :]
    return (a * (1 - fy)[None, :, None] + b * fy[None, :, None]).astype(np.float32)


def _d2s(x, blk, mode="dcr"):
    """DepthToSpace: out[co, h*blk + d1, w*blk + d2] = in[ci, h, w].

    The two ONNX orders differ ONLY in how the input channel axis splits across
    the block: CRD as (co, d1, d2) and DCR as (d1, d2, co). ONNX's default when
    the attribute is absent is DCR, and this model's nodes carry `blocksize` and
    nothing else -- so a reader that guesses here is guessing at a rearrangement
    that produces the right shape and a plausible range either way.
    """
    C, H, W = x.shape
    cout = C // (blk * blk)
    if mode == "crd":
        xp = x.reshape(cout, blk, blk, H, W)        # co, d1, d2, h, w
        y = np.transpose(xp, (0, 3, 1, 4, 2))       # co, h, d1, w, d2
    elif mode == "dcr":
        xp = x.reshape(blk, blk, cout, H, W)        # d1, d2, co, h, w
        y = np.transpose(xp, (2, 3, 0, 4, 1))       # co, h, d1, w, d2
    else:
        raise Fail(f"DepthToSpace mode {mode!r} is neither DCR nor CRD")
    return y.reshape(cout, H * blk, W * blk)


_BODY = {}


# The container's tensor names, built once per section_1 call.
#
# It was keyed by id(reader) at first, which is a trap for this file in
# particular: --inject re-enters main(), each pass builds a NEW Reader, and a
# cache keyed by identity misses on the second one. Every injected fault then
# "failed" on that KeyError -- a memory address as the diagnostic -- and six
# cases reported themselves caught while section 1 had not run at all. A fault
# harness that passes for the wrong reason is worse than no harness, because it
# is a claim of coverage.
_NAMES = set()


def _has_bias(reader, name):
    if not _NAMES:
        raise Fail("the container's tensor names were never indexed; "
                   "section_1 builds them and no one else may")
    return name in _NAMES


def _op(op, xs, reader, pfx):
    k = op["op"]
    if k == "add":
        return _act(xs[op["inputs"][0]] + xs[op["inputs"][1]], op.get("act"))
    inp = xs[op["inputs"][0]]
    if inp.ndim == 4:
        inp = inp[0]                                # the frame arrives [1,C,H,W]
    if k in ("conv", "dwconv"):
        w = reader.tensor(f"{pfx}.conv.{op['conv']}.w")
        cout, cin, kh, kw = w.shape
        col, oh, ow = _im2col(inp, kh, kw, op["pad"], op["stride"])
        if k == "conv":
            y = w.reshape(cout, -1) @ col
        else:
            check(col.shape[0] == cout * kh * kw,
                  f"a dwconv reads {inp.shape[0]} channels and writes {cout}: "
                  f"one filter per channel is the same number")
            y = np.empty((cout, col.shape[1]), np.float32)
            for c in range(cout):
                y[c] = w[c].reshape(-1) @ col[c * kh * kw:(c + 1) * kh * kw]
        if _has_bias(reader, f"{pfx}.conv.{op['conv']}.b"):
            y = y + reader.tensor(f"{pfx}.conv.{op['conv']}.b").reshape(-1, 1)
        return _act(y.reshape(cout, oh, ow), op["act"])
    if k == "maxpool":
        kh, kw = op["kernel"]
        c, h, w = inp.shape
        pt, pl, pb, pr = op["pad"]
        sh, sw = op["stride"]
        xp = np.full((c, h + pt + pb, w + pl + pr), -np.inf, np.float32)
        xp[:, pt:pt + h, pl:pl + w] = inp
        oh = (h + pt + pb - kh) // sh + 1
        ow = (w + pl + pr - kw) // sw + 1
        # A sliding view, not im2col. The im2col rows run (c, ki, kj) with the
        # channel OUTERMOST, so regrouping them as (kh*kw, c, ...) reduces over
        # channels and still returns exactly the right shape.
        s = xp.strides
        view = np.lib.stride_tricks.as_strided(
            xp, shape=(c, oh, kh, ow, kw),
            strides=(s[0], sh * s[1], s[1], sw * s[2], s[2]), writeable=False)
        return view.max(axis=(2, 4))
    if k == "resize":
        return _resize(inp, op["scale"][0], op["scale"][1])
    if k == "d2s":
        return _d2s(inp, op["block"], op.get("mode", "dcr"))
    raise Fail(f"op {k!r} is not one this file implements")


def run_body(reader, ops, image, pfx):
    """The op list -> every node's output. Operand -1 is the frame, and a dict
    rather than a list because -1 in a list would quietly mean "the last node
    computed" -- which is right for the first convolution and wrong for every
    later one."""
    xs = {-1: image}
    for n, op in enumerate(ops):
        for i in op["inputs"]:
            check(i < n or i == -1,
                  f"body op {n} takes operand {i}, which is not an earlier op "
                  f"and not the frame")
        y = _op(op, xs, reader, pfx)
        want = op.get("out")
        if want is not None:
            check(list(np.shape(y)) == list(want[1:]),
                  f"body op {n} ({op['op']}) produced {list(np.shape(y))}, "
                  f"the container says {list(want[1:])}")
        xs[n] = np.asarray(y, np.float32)
    return xs


# -- the two heads ------------------------------------------------------------


def det_head(xs, head):
    """Three levels of two 1x1 convolution outputs -> the graph's [1,2254,12] and
    [1,2254,1].

    The reshape is the delicate part and it is the graph's own: the checkpoint
    transposes NCHW to NHWC and reshapes [H,W,C] to [H*W*A, terms], so row
    (h*W + w)*A + a, term t, is channel a*terms + t of cell (h, w). The flat row
    index therefore advances w -- the X coordinate -- fastest, which is the
    order section 2's anchor table is in. A C-major flatten instead permutes
    every row and still leaves 2254 well-shaped rows and a confident detector.
    """
    boxes, scores = [], []
    for L in head:
        b, s = xs[L["box_op"]], xs[L["score_op"]]
        per, terms = L["per"], L["terms"]
        check(b.shape == (per * terms, L["h"], L["w"]),
              f"the box feature is {b.shape}, the head says "
              f"{(per * terms, L['h'], L['w'])}")
        check(s.shape == (per, L["h"], L["w"]),
              f"the score feature is {s.shape}, the head says "
              f"{(per, L['h'], L['w'])}")
        for y in range(L["h"]):
            for x in range(L["w"]):
                for a in range(per):
                    boxes.extend(b[a * terms + t, y, x] for t in range(terms))
                    scores.append(s[a, y, x])
    return np.array(boxes, np.float64), np.array(scores, np.float64)


def pose_head(xs, head):
    """Five single convolutions. `transposed` and `sigmoid` come out of the
    CONTAINER rather than out of the output's name: the heatmap is genuinely
    transposed and the confidence is genuinely squashed, and a reader that took
    either from the name would be hard-coding what the graph states."""
    out = {}
    for name, e in head.items():
        y = xs[e["op"]]
        if e["transposed"]:
            cc, hh, ww = e["decl"][3], e["decl"][1], e["decl"][2]
            check(y.shape == (cc, hh, ww),
                  f"{name} declares [{cc},{hh},{ww}] and the node holds "
                  f"{y.shape}; it is the one output whose ORDER is the content")
            v = np.transpose(y, (1, 2, 0))
        else:
            v = y
        if e.get("sigmoid"):
            v = 1.0 / (1.0 + np.exp(-v))
        out[name] = v.reshape(e["decl"])
    return out


# ==============================================================================
# section 1 -- the packed op list, against onnxruntime
# ==============================================================================


def _peak_rel(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        raise Fail(f"shape {got.shape} against onnxruntime's {want.shape}")
    return float(np.abs(got - want).max()) / (1.0 + float(np.abs(want).max()))


def section_1(reader, verbose):
    import onnxruntime as ort
    global _NAMES
    _NAMES = set(reader.entries)

    print("  1. the packed op list, against onnxruntime")
    for pfx, onnx_path, head_key, graphs_key in (
            ("det", DET_ONNX, "det_head", "det_graph"),
            ("pose", POSE_ONNX, "pose_head", "pose_graph")):
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        shape = [d if isinstance(d, int) and d > 0 else 1
                 for d in sess.get_inputs()[0].shape]
        # A standard normal, seeded. A real photograph would also be a fair test
        # but it is a WEAK one: it exercises the network on a distribution the
        # checkpoint has never seen, where a transposed filter still produces a
        # plausible range. This is the input that finds a wrong graph.
        x = np.random.default_rng(SEED).standard_normal(shape).astype(np.float32)
        names = [o.name for o in sess.get_outputs()]
        want = sess.run(None, {sess.get_inputs()[0].name: x})

        # The landmark net's checkpoint takes [1,256,256,3]; the packer ABSORBS
        # its leading NHWC->NCHW transpose, so operand -1 means "the frame in
        # NCHW" and that transpose is done here. The detector's input is already
        # NCHW and has no leading transpose to absorb. The test is on the
        # DECLARED shape, not on which network this is.
        body = x
        if len(shape) == 4 and shape[3] == 3 and shape[1] != 3:
            body = np.transpose(x, (0, 3, 1, 2))
        xs = run_body(reader, json.loads(reader.config[graphs_key]), body, pfx)

        if pfx == "det":
            boxes, scores = det_head(xs, json.loads(reader.config[head_key]))
            got = [boxes.reshape(1, -1, 12), scores.reshape(1, -1, 1)]
        else:
            h = pose_head(xs, json.loads(reader.config[head_key]))
            got = [h["landmarks"], h["conf"], h["mask"], h["heatmap"], h["world"]]

        # Paired by SHAPE, not by position: cv2 5.0.0 returns the detector's two
        # Concat outputs in the reverse of the order the graph declares them, and
        # the zoo's own harness reorders them by declared shape for the same
        # reason. ORT happens to agree with the declaration, but nothing here
        # should depend on that.
        used, worst, rows = set(), 0.0, []
        for name, w in zip(names, want):
            hit = next((i for i, g in enumerate(got)
                        if i not in used and np.shape(g) == w.shape), None)
            check(hit is not None,
                  f"{pfx}: no packed output of shape {w.shape} for {name}")
            used.add(hit)
            rel = _peak_rel(got[hit], w)
            rows.append((name, rel))
            worst = max(worst, rel)
        check(len(used) == len(want),
              f"{pfx}: {len(want) - len(used)} of onnxruntime's outputs were "
              f"never matched")
        for name, rel in rows:
            if verbose or rel > 1e-5:
                print(f"     ok   {name:<12} peak-rel {rel:.3e}")
        check(worst <= TOL_PEAK_REL,
              f"{pfx}: the packed graph reaches {worst:.3e} of onnxruntime's "
              f"peak, above the {TOL_PEAK_REL:.0e} this section allows. The "
              f"per-output numbers are above. A number of the ORDER OF ONE here "
              f"is a rearranged graph that still has the right shape and a "
              f"plausible range: ONNX's DepthToSpace order is the one that does "
              f"it, and `--inject` makes this exact check catch that case.")
    print(f"     both networks reproduce onnxruntime on all seven outputs "
          f"(worst {TOL_PEAK_REL:.0e} allowed)")


# ==============================================================================
# section 2 -- the anchor table
# ==============================================================================


def anchors_from_levels(levels):
    """The 2254 SSD anchor centres, derived here from the head's level list.

    Written out rather than imported from the packer, because a check that calls
    the thing it is checking is not a check. Cell (h, w) is at x = (w+0.5)/W,
    y = (h+0.5)/H and carries `a` identical rows; the row order is the graph's,
    with h outer and w inner, so w -- the X coordinate -- advances fastest.
    Transposing the two components mirrors the whole table and still leaves 2254
    well-shaped rows and a confident detector.
    """
    out = []
    for H, W, a in levels:
        for h in range(H):
            for w in range(W):
                for _ in range(a):
                    out.append(((w + 0.5) / W, (h + 0.5) / H))
    return np.array(out, np.float32)


def section_2(reader, verbose):
    print("  2. the anchor table, against an independent derivation")
    levels = [tuple(l) for l in json.loads(reader.config["det_anchor_levels"])]
    head = json.loads(reader.config["det_head"])
    check(len(levels) == len(head),
          f"det_anchor_levels lists {len(levels)} levels and det_head has "
          f"{len(head)}; two spellings of one pyramid that disagree are a "
          f"misaligned table rather than a tolerance")
    for i, (L, lev) in enumerate(zip(head, levels)):
        check((L["h"], L["w"], L["per"]) == lev,
              f"level {i}: det_head says {L['h']}x{L['w']} at {L['per']}, "
              f"det_anchor_levels says {lev}")

    got = reader.tensor("det_anchors").astype(np.float32)
    want = anchors_from_levels(levels)
    check(got.shape == want.shape,
          f"the container's table is {got.shape}, the levels derive "
          f"{want.shape}")
    d = np.abs(got - want)
    n_bad = int((d > TOL_ANCHOR_BITS).sum())
    check(n_bad == 0,
          f"{n_bad} of {d.size} table entries differ from the derivation, "
          f"worst {d.max():.3e}. This section is EXACT and has no tolerance: the "
          f"packer generated this table and reproduced the OpenCV zoo's literal "
          f"2254 x 2 one bit for bit, so a difference here is a wrong order or "
          f"a wrong level, not rounding.")
    check(int(reader.config["det_num_anchors"]) == want.shape[0],
          f"the container declares {reader.config['det_num_anchors']} anchors "
          f"and the levels derive {want.shape[0]}")
    if verbose:
        print(f"     first three: {got[:3].tolist()}  last: {got[-1].tolist()}")
    print(f"     2254 x 2, exact: 0 entries differ, levels {levels}")


# ==============================================================================
# section 3 -- the C++ runtime, against the recorded zoo answer
# ==============================================================================


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_runtime(container):
    cmd = [BINARY, "mppose", container, IMAGE]
    p = subprocess.run(cmd, capture_output=True, text=True)
    check(p.returncode == 0,
          f"`{' '.join(cmd)}` exited {p.returncode}\n{p.stderr[-800:]}")
    try:
        return json.loads(p.stdout), p.stderr
    except json.JSONDecodeError as e:
        raise Fail(f"the runtime printed something that is not JSON ({e}); "
                   f"its stdout is the result and stderr is the status block:\n"
                   f"{p.stderr[-400:]}")


def _run_array(container):
    """The same run with the dense convolutions on the array.

    Read from the STATUS BLOCK and not from stdout, because the block is the
    runtime's own account of where the work went -- a number printed next to a
    flag the caller supplied would be printed whether or not anything moved.
    """
    design = os.path.join(REPO, "runtime", "artifacts", "mediapipe-pose",
                          "artifacts_npu1")
    cmd = [BINARY, "mppose", container, IMAGE, "--npu-ops", "conv",
           "--artifacts", design]
    p = subprocess.run(cmd, capture_output=True, text=True)
    check(p.returncode == 0,
          f"`{' '.join(cmd)}` exited {p.returncode}\n{p.stderr[-800:]}")
    line = [l for l in p.stderr.splitlines() if "array:" in l]
    check(line, "the status block printed no array line, so nothing was "
                 "dispatched and this section would be comparing two host runs")
    return json.loads(p.stdout), line[0].strip()


def section_4(reader, verbose):
    """The array path against the host path, on the same frame.

    WHY A WHOLE SECTION FOR A FLAG WHOSE DEFAULT IS THE HOST. Because the two
    answers are NOT the same and the difference is not visible anywhere else:
    `--npu-ops conv` dispatches, it is 2.4x slower, and its landmarks sit up to
    44 px from the host's. A gate that only checked the host path would pass with
    the array path returning anything at all, and a reader of the registry row
    would have no way to tell how far apart the two are.

    WHAT IS ASSERTED, AND WHAT IS ONLY MEASURED. Asserted: the array path runs,
    it dispatches a plausible number of times, and the DETECTOR agrees with the
    host to bf16 precision -- which is the part that is supposed to be true, and
    the reason the pose difference below is not a bug in the array network.
    Measured and reported: the pose difference, which is a property of this
    architecture's geometry rather than of the array, and which has no tolerance
    here because any tolerance would be a claim that the two paths are
    interchangeable. They are not, and this file says so with numbers instead.
    """
    print("  4. the array path against the host path, same frame")
    art = os.path.join(REPO, "runtime", "artifacts", "mediapipe-pose",
                       "artifacts_npu1", "gemm_rtp", "design.json")
    if not os.path.exists(art):
        raise Fail(
            f"{art} is not there, so there is no array path to compare the host "
            f"against. Build it with\n"
            f"  source /opt/xilinx/xrt/setup.sh\n"
            f"  .venv/bin/python tools/export/export_gemm_rtp.py --target "
            f"mediapipe-pose --arch 1 --cols 4 -n 32\n"
            f"which needs MLIR-AIE -- and that is in .venv/, not on the system "
            f"PATH, which is the whole reason this gate says so rather than "
            f"reporting a missing directory and leaving it at that.")
    # A SEPARATE CONTAINER, and not a flag on the golden's one. The array needs
    # the pre-tiled bf16 panels staged, which makes it 29.9 MB against the
    # host-only 18.3 MB -- so reusing one path for both would mean the container
    # the golden was recorded against is not the container section 3 checked.
    if not os.path.exists(NPU_CONTAINER):
        raise Fail(
            f"{NPU_CONTAINER} is not there, so the array path has no panels to "
            f"dispatch into. Pack it with\n"
            f"  python tools/pack/pack_npue.py --mppose-onnx models/mediapipe-pose "
            f"--out {NPU_CONTAINER} --npu --device npu1\n"
            f"and note that --npu needs --device: N pads to tile_n*cols and cols "
            f"is a property of the array, so a panel built without one is not a "
            f"panel the design can read.")
    host, _ = _run_runtime(CONTAINER)
    arr, line = _run_array(NPU_CONTAINER)

    hd, ad = host["detections"][0], arr["detections"][0]
    check(len(host["detections"]) == len(arr["detections"]) == 1,
          f"the host found {len(host['detections'])} people and the array "
          f"{len(arr['detections'])}; the two paths must agree on the COUNT for "
          f"the rest of this section to mean anything")
    ds = abs(hd["score"] - ad["score"])
    check(ds <= 3e-3,
          f"detector score differs by {ds:.3e}. The panels are bf16, so this is "
          f"the expected order of magnitude; a larger one means the array is "
          f"running something other than this network.")
    dk = max(abs(hd["keypoints"][i][j] - ad["keypoints"][i][j])
             for i in range(4) for j in range(2))
    check(dk <= 3.0,
          f"the detector's keypoints differ by {dk:.3f} px, outside the 3 px "
          f"bf16 budget. These four points decide the landmark network's crop "
          f"AND its rotation angle, so this number is what the pose difference "
          f"below is made of.")
    if verbose:
        print(f"     detector  score within {ds:.1e}, keypoints within {dk:.2f} "
              f"px -- bf16 panel precision")

    ph, pa = host["poses"][0], arr["poses"][0]
    lh = np.asarray(ph["landmarks"], np.float64)
    la = np.asarray(pa["landmarks"], np.float64)
    xy = np.abs(lh[:, :2] - la[:, :2]).max(axis=1)
    dc = abs(ph["pose_confidence"] - pa["pose_confidence"])
    print(f"     pose      landmarks mean {xy.mean():.1f} / median "
          f"{np.median(xy):.1f} / max {xy.max():.1f} px, pose confidence "
          f"{ph['pose_confidence']:.4f} against {pa['pose_confidence']:.4f}")
    print(f"     {line}")
    th = host["timings_ms"]["total"]
    ta = arr["timings_ms"]["total"]
    print(f"     frame     {ta:.0f} ms on the array against {th:.0f} ms on the "
          f"host -- {ta / th:.2f}x")
    check(ta > 0 and th > 0, "a frame time of zero would make the ratio a "
                              "division by nothing")
    # The registry cell says `honours` and says 2.4x slower. If a future design
    # set makes the array WINNER, this fails and the cell has to be rewritten --
    # which is the point of asserting a direction rather than a range.
    check(ta > th,
          f"the array path is now {th / ta:.2f}x FASTER than the host "
          f"({ta:.0f} ms against {th:.0f} ms). That is a real result and a good "
          f"one, and the registry cell, NPU_OPS.md and NPU_MODELS.md all say the "
          f"array is slower here -- so this gate failing is the signal to "
          f"re-measure and rewrite them, not a regression.")


def section_3(reader, verbose):
    print("  3. the C++ runtime, against the numbers recorded from the OpenCV zoo")
    with open(GOLDEN_DET) as f:
        gd = json.load(f)
    with open(GOLDEN_POSE) as f:
        gp = json.load(f)

    # The image is checked BY HASH and not by name: a substitute photograph would
    # make every number below meaningless while every one of them still passed.
    for name, g in (("mppose_det.json", gd), ("mppose_pose.json", gp)):
        check(g["image_sha256"] == _sha256(IMAGE),
              f"{name} was recorded on a {g['image']} with sha256 "
              f"{g['image_sha256'][:16]}..., and {IMAGE} hashes to "
              f"{_sha256(IMAGE)[:16]}...")
        check(g["model_sha256"] == _sha256(
                  DET_ONNX if "person_detection" in g["model"]
                  else POSE_ONNX),
              f"{name}'s recorded checkpoint is not the one on disk")

    got, _ = _run_runtime(CONTAINER)

    check(len(got["detections"]) == gd["n_kept"],
          f"the runtime found {len(got['detections'])} people and the recorded "
          f"answer has {gd['n_kept']}. THIS IS THE ONE COUNTS CHECK. The "
          f"detector's score and NMS thresholds come from the container, so a "
          f"different count is either a different graph or a different "
          f"threshold, and section 1 has already ruled out the graph.")
    d = got["detections"][0]
    row = gd["rows"][0]
    check(abs(d["score"] - row[12]) <= DET_TOL_SCORE,
          f"detector score {d['score']:.9f} against the recorded "
          f"{row[12]:.9f}, outside {DET_TOL_SCORE:.0e}")
    deltas = [abs(d["box"][i] - row[i]) for i in range(4)]
    deltas += [abs(d["keypoints"][k][j] - row[4 + 2 * k + j])
               for k in range(4) for j in range(2)]
    check(max(deltas) <= DET_TOL_PX,
          f"the detector's box or keypoints are {max(deltas):.5f} px from the "
          f"recorded row, outside {DET_TOL_PX} px")
    if verbose:
        print(f"     score {d['score']:.9f} vs {row[12]:.9f}; box and keypoints "
              f"within {max(deltas):.5f} px")

    check(len(got["poses"]) == gp["n_persons"],
          f"the runtime reported {len(got['poses'])} poses and the recorded "
          f"answer has {gp['n_persons']}. The confidence gate is "
          f"pose_conf_threshold, read from the container.")

    p, q = got["poses"][0], gp["poses"][0]
    check(abs(p["pose_confidence"] - q["conf"]) <= POSE_TOL_CONF,
          f"pose confidence {p['pose_confidence']:.9f} against the recorded "
          f"{q['conf']:.9f}, outside {POSE_TOL_CONF:.0e}")

    want_bbox = [q["bbox"][0][0], q["bbox"][0][1], q["bbox"][1][0], q["bbox"][1][1]]
    bb = [abs(p["bbox"][i] - want_bbox[i]) for i in range(4)]
    check(max(bb) <= LM_TOL_XY_MAX,
          f"the reported pose box is {max(bb):.4f} px away, outside "
          f"{LM_TOL_XY_MAX} px")

    cl = np.asarray(p["landmarks"], np.float64)
    gl = np.asarray(q["landmarks"], np.float64)
    check(cl.shape == gl.shape,
          f"the runtime returned {cl.shape} landmark rows, the golden has "
          f"{gl.shape}")
    xy = np.abs(cl[:, :2] - gl[:, :2]).max(axis=1)
    z = np.abs(cl[:, 2] - gl[:, 2])
    vis = np.abs(cl[:, 3] - gl[:, 3]).max()
    pres = np.abs(cl[:, 4] - gl[:, 4]).max()
    check(xy.mean() <= LM_TOL_XY_MEAN,
          f"landmarks differ by {xy.mean():.4f} px on average, outside "
          f"{LM_TOL_XY_MEAN}")
    check(xy.max() <= LM_TOL_XY_MAX,
          f"the worst landmark is {xy.max():.4f} px out, outside "
          f"{LM_TOL_XY_MAX}. On this frame the four worst are the far-side "
          f"foot, ankle, heel and knee, which is where a different resampler "
          f"and a different network disagree most.")
    check(z.mean() <= LM_TOL_Z_MEAN,
          f"depth differs by {z.mean():.4f} px on average, outside "
          f"{LM_TOL_Z_MEAN}")
    check(z.max() <= LM_TOL_Z_MAX,
          f"the worst depth is {z.max():.4f} px out, outside {LM_TOL_Z_MAX}")
    check(vis <= LM_TOL_VIS,
          f"visibility differs by {vis:.3e}, outside {LM_TOL_VIS:.0e}")
    check(pres <= LM_TOL_PRES,
          f"presence differs by {pres:.3e}, outside {LM_TOL_PRES:.0e}")

    cw = np.abs(np.asarray(p["world_landmarks"], np.float64)
                - np.asarray(q["world"], np.float64)).max(axis=1)
    check(cw.mean() <= WORLD_TOL_MEAN,
          f"world points differ by {cw.mean():.5f} m on average, outside "
          f"{WORLD_TOL_MEAN}")
    check(cw.max() <= WORLD_TOL_MAX,
          f"the worst world point is {cw.max():.5f} m out, outside "
          f"{WORLD_TOL_MAX}")

    nz_cpp = p["segmentation_mask"]["nonzero"]
    nz_want = q["mask_nnz"]
    frac = abs(nz_cpp - nz_want) / max(nz_want, 1)
    check(frac <= MASK_TOL_FRAC,
          f"the mask has {nz_cpp} non-zero pixels against the recorded "
          f"{nz_want}, {frac:.4%} away, outside {MASK_TOL_FRAC:.2%}. The mask is "
          f"a binary silhouette whose boundary is where a resampler and a "
          f"threshold disagree first, so this counts boundary pixels and says "
          f"nothing about the interior.")

    print(f"     detector  score within {abs(d['score'] - row[12]):.1e}, "
          f"box+keypoints within {max(deltas):.5f} px")
    print(f"     pose      conf within {abs(p['pose_confidence'] - q['conf']):.1e}"
          f", landmarks mean {xy.mean():.3f} / median {np.median(xy):.3f} / "
          f"max {xy.max():.3f} px, world max {cw.max():.5f} m")
    print(f"               mask {nz_cpp} non-zero against {nz_want} ({frac:.3%})")


# ==============================================================================
# -- fault injection ------------------------------------------------------------
#
# Every case below is a mistake one of this file's or the runtime's comments
# names, and each one must make the gate FAIL. A fault that slips through is
# worse than no case at all: it is a claim that the section covers something it
# does not.
#
# They are all faults in THIS file's reader or in the container it reads. The C++
# binary is not faulted from here -- a binary would have to be rebuilt per case,
# and arch=8's own refusals (--serve, --npu-ops, --artifacts, arch=6's and
# arch=7's flags) are covered by verify_cli_flags and verify_serve_dispatch
# rather than by re-injecting them here.
# ==============================================================================


# The module's own globals, for the injection harness. `sys.modules[__name__]`
# and NOT `import verify_mppose`: run as a script this file IS __main__, so an
# import of the same path would make a SECOND copy of every function, and the
# harness would then be patching the copy while the gate called the original --
# which reports six faults caught by patching nothing at all.
_SELF = sys.modules[__name__]
_PRISTINE = {}


def _snapshot():
    _PRISTINE["_d2s"] = _SELF._d2s
    _PRISTINE["_act"] = _SELF._act
    _PRISTINE["_resize"] = _SELF._resize
    _PRISTINE["anchors_from_levels"] = _SELF.anchors_from_levels
    _PRISTINE["_run_runtime"] = _SELF._run_runtime


def _restore():
    _SELF._d2s = _PRISTINE["_d2s"]
    _SELF._act = _PRISTINE["_act"]
    _SELF._resize = _PRISTINE["_resize"]
    _SELF.anchors_from_levels = _PRISTINE["anchors_from_levels"]
    _SELF._run_runtime = _PRISTINE["_run_runtime"]


def inject(quiet=True):
    def d2s_crd(x, blk, mode="dcr"):
        return _PRISTINE["_d2s"](x, blk, "crd")

    def act_off(v, name):
        return v

    def resize_identity(x, sy, sx):
        return x

    def anchors_transposed(levels):
        out = []
        for H, W, a in levels:
            for h in range(H):
                for w in range(W):
                    for _ in range(a):
                        out.append(((h + 0.5) / H, (w + 0.5) / W))
        return np.array(out, np.float32)

    def runtime_drops_a_person(container):
        got, err = _PRISTINE["_run_runtime"](container)
        got = dict(got)
        got["poses"] = []
        return got, err

    def runtime_half_the_mask(container):
        got, err = _PRISTINE["_run_runtime"](container)
        got = json.loads(json.dumps(got))
        if got["poses"]:
            got["poses"][0]["segmentation_mask"]["nonzero"] //= 2
        return got, err

    cases = [
        ("section 1: DepthToSpace in CRD order",
         lambda: setattr(_SELF, "_d2s", d2s_crd)),
        ("section 1: the activation dropped from the op",
         lambda: setattr(_SELF, "_act", act_off)),
        ("section 1: the detector's Resize treated as an identity",
         lambda: setattr(_SELF, "_resize", resize_identity)),
        ("section 2: the anchor table's x and y swapped",
         lambda: setattr(_SELF, "anchors_from_levels", anchors_transposed)),
        ("section 3: the runtime's answer read as having no poses",
         lambda: setattr(_SELF, "_run_runtime", runtime_drops_a_person)),
        ("section 3: half the mask's non-zero pixels",
         lambda: setattr(_SELF, "_run_runtime", runtime_half_the_mask)),
    ]

    # main() re-enters through _run_and_report, and the injection flag must be
    # off argv by then or it recurses until the stack runs out -- which is a
    # confusing way to learn that a fault-injection harness calls its own entry
    # point.
    argv = list(sys.argv)
    sys.argv = [a for a in argv if a != "--inject"]
    _snapshot()
    out = [("baseline, no fault", _run_and_report(quiet))]
    for tag, setup in cases:
        setup()
        out.append((tag, _run_and_report(quiet)))
        _restore()
    sys.argv = argv
    return out


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
    except Exception as e:
        return f"caught ({type(e).__name__}): {str(e)[:70]}"


# ==============================================================================


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="verify an arch=8 MediaPipe Pose container and the C++ "
                    "runtime built from it")
    ap.add_argument("--container", default=None,
                    help="the .npue to check (default: models/mediapipe-pose/"
                         "mppose.npue)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--inject", action="store_true",
                    help="break this file's reader six ways and check that "
                         "every break fails the gate, then check that the "
                         "unbroken code passes")
    args = ap.parse_args()

    container = args.container or CONTAINER
    _need(container, "the packed container")
    _need(IMAGE, "the golden's photograph")
    _need(GOLDEN_DET, "the recorded detector answer")
    _need(GOLDEN_POSE, "the recorded pose answer")
    _need(BINARY, "the built runtime")

    reader = Reader(container)
    check(reader.arch == ARCH_MEDIAPIPE_POSE,
          f"arch is {reader.arch}, this gate is for {ARCH_MEDIAPIPE_POSE}")
    check(reader.config.get("kind") == "mppose",
          f"kind is {reader.config.get('kind')!r}, not 'mppose'. It is not "
          f"'pose' either: that is arch=6's, and the two share no flag, no "
          f"threshold, no head and no input size.")
    # The two normalisations, because one pair would be a detector fed the wrong
    # range and a container that does not say which is which.
    check(reader.config.get("det_image_mean") == [0.5] * 3,
          f"det_image_mean is {reader.config.get('det_image_mean')}, and this "
          f"detector is (x/255 - 0.5) * 2 -- one pair for both networks is "
          f"arch=7's situation and not this one's")
    check(reader.config.get("det_image_std") == [0.5] * 3,
          f"det_image_std is {reader.config.get('det_image_std')}")
    check(reader.config.get("pose_image_mean") == [0.0] * 3,
          f"pose_image_mean is {reader.config.get('pose_image_mean')}; the "
          f"landmark net divides by 255 and stops")
    check(reader.config.get("pose_image_std") == [1.0] * 3,
          f"pose_image_std is {reader.config.get('pose_image_std')}")

    if args.inject:
        _need(DET_ONNX, "the person detector checkpoint")
        _need(POSE_ONNX, "the landmark checkpoint")
        print("injected faults, each one a mistake this file's comments name")
        results = inject(quiet=not args.verbose)
        missed = []
        for tag, said in results:
            print(f"  {tag:56s} {said}")
            if said == "MISSED":
                missed.append(tag)
        _restore()
        if results[0][1] != "MISSED":
            raise Fail(f"the unbroken code did not pass: {results[0][1]}")
        bad = [t for t in missed if not t.startswith("baseline")]
        if bad:
            raise Fail(f"{len(bad)} of {len(results) - 1} injected faults were "
                       f"NOT caught: {bad}")
        print(f"OK: {len(results) - 1} injected faults, all caught; the "
              f"unbroken code passes.")
        return

    _need(DET_ONNX, "the person detector checkpoint")
    _need(POSE_ONNX, "the landmark checkpoint")
    print(f"arch {reader.arch} ({ARCH_STRING}), kind pose, "
          f"{reader.config['det_num_anchors']} anchors, "
          f"{reader.config['det_num_convs']} + {reader.config['pose_num_convs']} "
          f"convolutions, "
          f"{sum(1 for o in json.loads(reader.config['det_graph']) if o['op'] == 'd2s')}"
          f" + "
          f"{sum(1 for o in json.loads(reader.config['pose_graph']) if o['op'] == 'd2s')}"
          f" d2s")
    for fn in (section_1, section_2, section_3, section_4):
        fn(reader, args.verbose)
    print("OK: the container holds these two checkpoints in this arrangement, "
          "the anchor table is the one the packer verified against the zoo's "
          "literal one, the C++ runtime reproduces the recorded answer, and the "
          "array path dispatches against a design set checked in both "
          "directions -- slower than the host and a different answer, both "
          "measured and both in the registry cell.")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
