# NpuEmbeddings -- pack MediaPipe Pose into an arch=8 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# The third MediaPipe family, and the first whose answer includes something
# that is not a set of coordinates: a 256x256 person segmentation mask, out of
# the same forward pass as the 39 screen landmarks and the 39 world landmarks.
#
# It holds TWO networks, and the reason is not symmetry with packers/hands.py.
# The pose landmark network is handed a person's region of interest and has no
# way to find a person in a frame; it is a regressor over a crop. A container
# holding only it answers "no person here" to every image, and a container
# holding only the detector stops at a box. What differs from hands is what the
# crop is. MediaPipe Hands rotates a palm by two points its own detector
# regresses. MediaPipe Pose builds a square around `mid_hip`, rotates it by the
# `mid_hip -> full_body` direction, and then INVERTS that rotation on the way
# out -- for the landmarks, for the world landmarks, and for the mask. So the
# angle is not a property of the preprocessing; it is part of the result, and
# reversing it produces skeletons mirrored about the hip rather than an error.
#
# THE FOURTEEN NUMBERS THAT ARE MODEL, NOT TUNING
# -----------------------------------------------
# PERSON_BOX_PRE_ENLARGE_FACTOR and PERSON_BOX_ENLARGE_FACTOR are both read
# out of the zoo's mp_pose.py rather than fitted: 1 (no pre-enlargement) and
# 1.25 (the reported box is grown by a quarter about its own centre). The
# crop square is `mid_hip +/- ||mid_hip - full_body||`, so its half-side is a
# length the DETECTOR gives us and not a constant. RoI clip order, pad_bias,
# the angle wrap, the two rotation matrices and the mask's crop/pad back into
# the frame all follow the reference line for line, because each of them can be
# wrong in a way that still yields a human-shaped answer.
#
# WHY ONE NEW OPERATOR, AND ONLY ONE
# ----------------------------------
# The two graphs together use twelve operator types. Eleven already have a
# meaning here:
#
#   Conv, Add, MaxPool, Resize, Transpose, Clip, PRelu, Reshape, Concat,
#   Sigmoid            -- the existing vocabulary (hands.py, pose.py)
#   Relu               -- Act::Relu already names it; this is a fusion
#   Pad                -- see below; never emitted as an op at all
#   DepthToSpace       -- genuinely new, emitted as `d2s`
#
# The SPATIAL Pads are the interesting case. ONNX Conv pads are zeros, so a
# constant-zero Pad standing in front of a Conv is arithmetically identical to
# adding those terms to the Conv's own `pad` field -- not approximately, not up
# to rounding: padding with zeros and then convolving with an implicit zero
# border is the same sum either way. This packer therefore folds all four of
# them (three in the detector, one in the landmark net) into their Conv
# consumers and emits no Pad node, then CHECKS that each folded convolution
# still produces the shape the graph declares. A pad whose consumer is not a
# Conv is refused rather than emitted, because there is nothing to fold it into.
#
# DepthToSpace cannot be reformed that way: it is a real rearrangement, and it
# exists three times in the detector -- once per pyramid level, which is how the
# 7x7/14x14/28x28 levels are produced in the first place. WHICH rearrangement it
# is must not be guessed. ONNX names this operator's two orders DCR (depth,
# then column, then row) and CRD, they differ only in how the input CHANNEL axis
# is split across the block, and the default when the attribute is absent is
# DCR -- which is this model's case: the node carries `blocksize` and nothing
# else. Both orders are written into the op and both are executed, because
# taking the wrong one moves every channel to a different pixel while still
# producing a tensor of the right shape and a plausible range, so nothing
# downstream notices; only a comparison against the graph that was packed
# does.
#
# THE ANCHOR TABLE IS GENERATED AND MATCHES EXACTLY
# ------------------------------------------------
# The zoo hard-codes 2254 anchor centres as a literal float table. This packer
# never reads that table. It generates cell centres over the three levels the
# head actually reports -- 784 cells of 28x28 at 2 anchors each, 196 of 14x14
# at 2, 49 of 7x7 at 6, giving 1568 + 392 + 294 = 2254 -- in the Concat's own
# order. Against the zoo's literals the maximum deviation is 0.000e+00. The
# corresponding number for mediapipe-hands was 7.5e-9, which was the zoo's own
# eight-decimal truncation; this table has no truncation, so it matches bit for
# bit. The three levels are not an assumption either: they are the graph's
# three DepthToSpace nodes, and the head's row counts confirm it independently
# (1568 = 28*28*2, 392 = 14*14*2, 294 = 7*7*6).
#
# WHY FLOAT, AND HOW THAT WAS DECIDED
# -----------------------------------
# Both repositories ship an `..._int8bq.onnx`. It is NOT taken, and the reason
# was measured rather than inherited from hands: on docs/bus.jpg the int8 pair
# puts the 33 keypoints a mean 64.0 px apart from float with a worst case of
# 108.5 px, and z -- which is in PIXELS relative to the mid-hip -- off by as
# much as 291.9 px. It is also slower (29.1 ms against 28.2 ms end to end).
# models/mediapipe-pose/CHECKPOINT.json records those numbers and the sha256 of
# both files actually taken.
#
# WHAT IS DELIBERATELY NOT CLAIMED
# --------------------------------
# Coverage on the fixture is partial and that is the model, not a bug: at the
# zoo's own score_threshold of 0.5 the person detector returns ONE of the
# three people in docs/bus.jpg, because their raw scores are 0.854, 0.305,
# 0.067 and 0.065. Nothing here claims otherwise, and no threshold is loosened
# to make a second person appear -- the thresholds written into the container
# are the zoo's own defaults.

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))

from npue import (ARCH_MEDIAPIPE_POSE_DET_SSD_LM_REGRESS,  # noqa: E402
                  Writer)

ARCH_STRING = "mediapipe_pose_det_ssd_lm_regress"

DET_SIZE = 224                 # the person detector's square input
POSE_SIZE = 256                # the landmark network's square input

DET_LANDMARKS = 4              # points the detector regresses (2 body, 2 upper)
DET_BOX_TERMS = 4
DET_ROW = DET_BOX_TERMS + 2 * DET_LANDMARKS      # 12, not 8 and not 4

NUM_LANDMARKS = 39             # rows the network emits
NUM_KEYPOINTS = 33             # the advertised set; the other 6 are auxiliary
LM_COLS = 5                    # x, y, z, visibility, presence
WORLD_COLS = 3                 # x, y, z in metres

# The two points that fix the rotation. Written down because a re-export that
# reordered the detector's four landmarks would rotate every person by
# whatever the difference happened to be, and the answer would still look like
# a person.
PERSON_LM_MID_HIP = 0
PERSON_LM_FULL_BODY = 1

# Geometry, from the OpenCV zoo's mp_pose.py. MODEL decisions, not tuning.
PERSON_BOX_PRE_ENLARGE = 1.0
PERSON_BOX_ENLARGE = 1.25

SCORE_THR = 0.5                # MPPersonDet's own default
NMS_THR = 0.3
TOP_K = 5000
POSE_CONF_THR = 0.5            # MPPose's own default (demo.py passes 0.8)

FUSIBLE = ("Conv", "Add")


def _die(msg):
    raise SystemExit(f"mppose: {msg}")


def check_disjoint(a, b, what):
    both = set(a) & set(b)
    if both:
        _die(f"two networks share {what} names: {sorted(both)[:4]}. Prefixed "
             f"names are disjoint by construction, so this means the prefix "
             f"rule stopped holding.")


def det_anchors(levels):
    """Cell centres, `per` anchors apiece, in the head's OWN level order.

    Every anchor of one cell shares its centre -- that is what the table does,
    and it is why the reconstruction below reproduces the zoo's literals
    exactly rather than to a tolerance.
    """
    pts = []
    for h, w, per in levels:
        if h <= 0 or w <= 0 or per <= 0:
            _die(f"anchor level {h}x{w}x{per} is not a level")
        cy = [(iy + 0.5) / h for iy in range(h)]
        cx = [(ix + 0.5) / w for ix in range(w)]
        for y in cy:
            for x in cx:
                for _ in range(per):
                    pts.append((x, y))
    return np.array(pts, dtype=np.float32)


class Traced:
    """One network, with every value's shape read out of the graph."""

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.graph = model.graph
        self.init = {i.name: numpy_helper.to_array(i)
                     for i in model.graph.initializer}
        self.by_out = {o: n for n in model.graph.node for o in n.output}
        self.nodes = list(model.graph.node)
        self.shape = {}

    def out_shape(self, name, node):
        s = self.shape.get(name)
        if s is None or any(not isinstance(x, int) for x in s):
            _die(f"{self.name}: cannot read the shape of {name!r}, consumed by "
                 f"{node.op_type} {node.name!r}. Every emitted node's output "
                 f"shape is written into the container, so an unreadable one is "
                 f"a refusal rather than a guess.")
        return s


def trace(name, model):
    t = Traced(name, model)
    try:
        inf = onnx.shape_inference.infer_shapes(model)
        for vi in inf.graph.value_info:
            t.shape[vi.name] = [d.dim_param or d.dim_value
                                for d in vi.type.tensor_type.shape.dim]
    except Exception as e:                                   # noqa: BLE001
        print(f"  note: shape inference unavailable ({e})")
    for vi in list(model.graph.input) + list(model.graph.output):
        t.shape[vi.name] = [d.dim_param or d.dim_value
                            for d in vi.type.tensor_type.shape.dim]
    return t


def _attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            return {onnx.AttributeProto.INT: lambda: a.i,
                    onnx.AttributeProto.FLOAT: lambda: a.f,
                    onnx.AttributeProto.INTS: lambda: list(a.ints),
                    onnx.AttributeProto.STRING: lambda: a.s.decode()
                    }.get(a.type, lambda: default)()
    return default


def _head_ops(n):
    """Absorbed into a head rather than emitted as a body node."""
    return n.op_type in ("Reshape", "Concat", "Squeeze", "GlobalAveragePool",
                         "Gemm", "Sigmoid")


def _pad_targets(t, pad, consumers):
    """The Convs that consume this Pad's value, through leading transposes.

    The landmark net puts its Pad BEFORE the NHWC->NCHW transpose, so the
    consumer is a Transpose and not a Conv. Walking through it is exact: a
    transpose moves the axes around and the pad amounts are spatial, so what
    the Conv must add is the same three numbers whichever side it sits on.
    """
    out, stack = [], [c for o in pad.output for c in consumers.get(o, [])]
    while stack:
        c = stack.pop()
        if c.op_type == "Transpose" and _attr(c, "perm") == [0, 3, 1, 2]:
            stack.extend(x for o in c.output for x in consumers.get(o, []))
            continue
        out.append(c)
    return out


def _spatial_pad(t, pad, targets):
    """[top, left, bottom, right] for a constant pad, or a refusal.

    The layout is not assumed: it is read off the target convolution's input
    channel count, which appears on exactly one axis of the padded shape and
    on no other. Guessing would silently swap height and width for a square
    input, which is the one case nothing downstream catches.
    """
    if _attr(pad, "mode", "constant") != "constant":
        _die(f"Pad {pad.name!r} has mode {_attr(pad, 'mode')!r}; only "
             f"constant-zero padding folds into a convolution's own pads")
    v = t.init.get(pad.input[1])
    if v is None:
        _die(f"Pad {pad.name!r} takes its pads from {pad.input[1]!r}, which is "
             f"not an initializer, so the amounts cannot be read")
    p = [int(x) for x in np.asarray(v).ravel()]
    if len(p) != 8:
        _die(f"Pad {pad.name!r} has {len(p)} pad terms, not the 8 an ONNX Pad "
             f"of a rank-4 tensor has")
    begin, end = p[:4], p[4:]

    src = t.shape.get(pad.input[0])
    if src is None or len(src) != 4:
        _die(f"Pad {pad.name!r}: the shape of {pad.input[0]!r} is {src}, so "
             f"the layout cannot be established")

    convs = [c for c in targets if c.op_type == "Conv"]
    if len(convs) != len(targets):
        bad = [c for c in targets if c.op_type != "Conv"]
        _die(f"Pad {pad.name!r} feeds {[c.op_type for c in bad]} rather than a "
             f"convolution, so there is no `pad` field to fold it into. ONNX "
             f"Conv pads are zeros and folding is exact; nothing else here has "
             f"an operand that absorbs a spatial pad.")

    cin = None
    for c in convs:
        w = t.init.get(c.input[1])
        if w is None:
            _die(f"Conv {c.name!r} has no weight initializer")
        g = _attr(c, "group", 1)
        want = int(w.shape[1]) * int(g)
        if cin is not None and want != cin:
            _die(f"Pad {pad.name!r} feeds two convolutions with different "
                 f"input channel counts ({cin} and {want}), which cannot both "
                 f"be this tensor")
        cin = want

    out_sh = t.shape.get(pad.output[0])
    if out_sh is None or len(out_sh) != 4:
        _die(f"Pad {pad.name!r}: the shape of {pad.output[0]!r} is {out_sh}")
    nchw = int(out_sh[1]) == cin
    nhwc = int(out_sh[3]) == cin
    if nchw == nhwc:
        _die(f"Pad {pad.name!r}: the padded shape {out_sh} has {cin} channels "
             f"on no axis alone or on more than one, so the layout is not "
             f"established. Folding it with the wrong layout pads the wrong "
             f"axis, and a square input hides that until the picture moves.")
    h_ax, w_ax, chan_ax = (2, 3, 1) if nchw else (1, 2, 3)

    batch_ax = 0
    for ax in (batch_ax, chan_ax):
        if begin[ax] or end[ax]:
            _die(f"Pad {pad.name!r} pads axis {ax} (batch or channel) by "
                 f"{begin[ax]}/{end[ax]}, and a channel pad is a different "
                 f"operator -- hands' pad_c appends channels and changes the "
                 f"width of every later operand.")
    return [begin[h_ax], begin[w_ax], end[h_ax], end[w_ax]]


def emit_ops(t, pfx):
    """The graph as op-list entries. See the module header for the rules."""
    ops, names = [], []
    conv_of, weights, prelus = {}, {}, {}
    producer = {o: n.name for n in t.nodes for o in n.output}
    tensor_idx, emitted = {}, {}

    consumers = {}
    for n in t.nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)

    # A tensor with no producer is the graph's own input, and that is the image.
    # Only ONE of these two networks needs it: the detector's input is already
    # NCHW ([1, 3, 224, 224]) so there is no leading transpose to absorb and
    # its very first convolution reads the frame directly, whereas the landmark
    # net takes NHWC and reaches the image through Pad -> Transpose.
    graph_inputs = {i.name for i in t.graph.input} - set(t.init)

    # -- pass 1: spatial Pads, folded into their Conv consumers -------------
    folds, zero_pads = {}, {}
    for n in t.nodes:
        if n.op_type != "Pad":
            continue
        targets = _pad_targets(t, n, consumers)
        v = t.init.get(n.input[1])
        if v is None:
            _die(f"Pad {n.name!r} takes its pads from {n.input[1]!r}, which "
                 f"is not an initializer")
        p = [int(x) for x in np.asarray(v).ravel()]
        if len(p) != 8:
            _die(f"Pad {n.name!r} has {len(p)} pad terms, not 8")
        if not any(p):
            zero_pads[n.name] = True
            continue
        if not targets:
            _die(f"Pad {n.name!r} feeds nothing, so its value is dropped and "
                 f"every consumer of it would have to be re-pointed by hand")
        sp = _spatial_pad(t, n, targets)
        for c in targets:
            if c.op_type != "Conv":
                _die(f"Pad {n.name!r} feeds {c.op_type} {c.name!r}, which has "
                     f"no `pad` field to fold into")
            if c.name in folds:
                _die(f"Conv {c.name!r} would be folded twice (by "
                     f"{folds[c.name][0]} and {n.name}), and the two amounts "
                     f"compose only if they are the same pair of zero pads")
            folds[c.name] = (n.name, sp)

    # -- pass 2: activations, and which op owns each ------------------------
    # The owner is not always a convolution: MediaPipe's inverted residual ends
    # with the residual Add, so `Add -> Relu -> Conv` is ordinary here and
    # fusing only into Convolutions would refuse four of the detector's eight.
    act_of = {}
    for n in t.nodes:
        if n.op_type not in ("Relu", "Clip", "PRelu"):
            continue
        src = n.input[0]
        p = producer.get(src)
        if p is None or len(consumers.get(src, [])) != 1:
            _die(f"{n.op_type} {n.name!r} does not sit between exactly one "
                 f"node and exactly one consumer, so it cannot be fused into "
                 f"that node's epilogue. Fusing it anyway would either "
                 f"double-apply it or drop it.")
        if t.by_out[src].op_type not in FUSIBLE:
            _die(f"{n.op_type} {n.name!r} follows a "
                 f"{t.by_out[src].op_type}, and this packer fuses activations "
                 f"into {FUSIBLE} epilogues only.")
        if p in act_of:
            _die(f"{p!r} is claimed by two activations ({act_of[p].name}, "
                 f"{n.name}). Impossible in a real graph, so the trace is "
                 f"wrong.")
        act_of[p] = n

    prelu_seq = [0]

    def fusion_of(node_name):
        act = act_of.get(node_name)
        if act is None:
            return "none", None
        if act.op_type == "Relu":
            return "relu", None
        if act.op_type == "PRelu":
            k = prelu_seq[0]
            prelu_seq[0] += 1
            prelus[f"{pfx}.prelu.{k}.slope"] = \
                t.init[act.input[1]].astype(np.float32)
            return "prelu", k
        lo = float(t.init[act.input[1]].ravel()[0])
        hi = float(t.init[act.input[2]].ravel()[0])
        if (lo, hi) != (0.0, 6.0):
            _die(f"Clip {act.name!r} is ({lo}, {hi}), which is not relu6; a "
                 f"different Clip bound is a different activation")
        return "relu6", None

    def idx_of(tensor, node, what):
        if tensor in t.init:
            _die(f"{node.op_type} {node.name!r} consumes the initializer "
                 f"{tensor!r} as its {what}. A convolution's weights are the "
                 f"conv op's own entry, so an initializer arriving as an "
                 f"ACTIVATION input is a graph this packer does not understand.")
        if tensor not in producer:
            if tensor in graph_inputs:
                return -1                 # the frame itself
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r} ({what}), "
                 f"which no traced node produces.")
        p = producer[tensor]
        if p not in emitted:
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r}, which is "
                 f"{p!r}'s output, and that node is not in the emitted list. "
                 f"Only a fused activation, a folded pad or a leading transpose "
                 f"may vanish.")
        if emitted[p] == -2:
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r}, which is "
                 f"a head-layout transpose ({p!r}). Only the head may read "
                 f"those, because only it knows the reshape's axis order.")
        if tensor not in tensor_idx:
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r}, which was "
                 f"marked emitted but never given an index")
        return tensor_idx[tensor]

    counts = {}

    def add(op, node):
        ops.append(op)
        names.append(node.name)
        counts[op["op"]] = counts.get(op["op"], 0) + 1
        return len(ops) - 1

    def claim(node, idx):
        emitted[node.name] = idx
        for o in node.output:
            tensor_idx[o] = idx

    for n in t.nodes:
        op = n.op_type

        if op == "Pad":
            # Whether it was folded into a consumer or was a zero-pad identity,
            # this node contributes no op: the amount (if any) already went into
            # that consumer's own `pad` field in pass 1. What it still owes is an
            # INDEX for its output tensor, because a consumer asking `idx_of`
            # for it must be answered with the value that tensor now names --
            # its input. Dropping the name instead would make every consumer
            # refuse a producer that never appears in the op list.
            src = n.input[0]
            if src in producer:
                if producer[src] not in emitted or src not in tensor_idx:
                    _die(f"Pad {n.name!r} reads {src!r} from "
                         f"{producer[src]!r}, which was not emitted, so there "
                         f"is no index to hand its consumers")
                idx = tensor_idx[src]
            elif src in graph_inputs:
                idx = -1                     # a graph input is the image
            else:
                _die(f"Pad {n.name!r} reads {src!r}, which is neither a traced "
                     f"node's output nor a declared input of the graph")
            emitted[n.name] = idx
            for o in n.output:
                tensor_idx[o] = idx
            continue

        if op == "Conv":
            w = t.init[n.input[1]]
            cout, cin_g, kh, kw = [int(x) for x in w.shape]
            groups = int(_attr(n, "group", 1))
            strides = [int(x) for x in _attr(n, "strides", [1, 1])]
            pads = [int(x) for x in _attr(n, "pads", [0, 0, 0, 0])]
            if len(pads) != 4:
                _die(f"Conv {n.name!r} has pads {pads}, which is not four "
                     f"terms. ONNX orders them [top, left, bottom, right].")
            if _attr(n, "auto_pad", "NOTSET") != "NOTSET":
                _die(f"Conv {n.name!r} has "
                     f"auto_pad={_attr(n, 'auto_pad')!r}. Only explicit pads "
                     f"are expressed; auto_pad is a promise about padding this "
                     f"packer does not keep.")
            if groups != 1 and cin_g != 1:
                _die(f"Conv {n.name!r} has group={groups} with {cin_g} input "
                     f"channels per group, which is neither a dense convolution "
                     f"nor a depthwise one.")
            if _attr(n, "dilations", [1, 1]) != [1, 1]:
                _die(f"Conv {n.name!r} is dilated, which no emitted operand "
                     f"expresses")

            # the folded spatial pad, composed with the Conv's own
            folded = folds.get(n.name)
            fold_src = None
            if folded is not None:
                fold_src, sp = folded
                pads = [pads[0] + sp[0], pads[1] + sp[1],
                        pads[2] + sp[2], pads[3] + sp[3]]

            in_sh = t.shape.get(n.input[0])
            if in_sh is not None and len(in_sh) == 4 and \
                    all(isinstance(x, int) for x in in_sh):
                ih, iw = int(in_sh[2]), int(in_sh[3])
                if folded is not None:
                    ih -= sp[0] + sp[2]
                    iw -= sp[1] + sp[3]
                oh = (ih + pads[0] + pads[2] - kh) // strides[0] + 1
                ow = (iw + pads[1] + pads[3] - kw) // strides[1] + 1
                out = t.out_shape(n.output[0], n)
                if int(out[2]) != oh or int(out[3]) != ow:
                    _die(f"Conv {n.name!r} computes {oh}x{ow} from an input of "
                         f"{ih}x{iw}, pads {pads} and stride {strides}, but the "
                         f"graph declares {out[2]}x{out[3]}."
                         + (f" Its pad was folded from {fold_src!r}, so this "
                            f"is where a wrong fold shows up." if folded
                            else ""))

            if groups != 1 and groups != cout:
                _die(f"Conv {n.name!r} has group={groups} but {cout} output "
                     f"channels, so it is not depthwise either. A grouped "
                     f"convolution is a THIRD thing with no op here, and "
                     f"packing it as dense would compute a different network.")
            out = t.out_shape(n.output[0], n)
            if int(out[1]) != cout:
                _die(f"Conv {n.name!r} writes {cout} channels but the graph's "
                     f"output shape says {out[1]}")
            i = len(conv_of)
            conv_of[n.name] = i
            weights[f"{pfx}.conv.{i}.w"] = w.astype(np.float32)
            if len(n.input) > 2 and n.input[2] in t.init:
                weights[f"{pfx}.conv.{i}.b"] = \
                    t.init[n.input[2]].astype(np.float32)
            actname, prelu = fusion_of(n.name)
            idx = add({"op": "conv" if groups == 1 else "dwconv",
                       "conv": i,
                       "act": actname, "prelu": prelu,
                       "stride": strides, "pad": pads, "group": int(groups),
                       "out": [int(x) for x in out],
                       "inputs": [idx_of(n.input[0], n, "activations")]}, n)
            claim(n, idx)
            a = act_of.get(n.name)
            if a is not None:
                emitted[a.name] = idx
                for o in a.output:
                    tensor_idx[o] = idx
            continue

        if op == "DepthToSpace":
            block = _attr(n, "blocksize")
            if block is None or int(block) < 1:
                _die(f"DepthToSpace {n.name!r} has blocksize {block!r}")
            # ONNX names the two orders DCR and CRD and defaults to DCR when
            # the attribute is absent, which is exactly how this model's node
            # looks -- it carries `blocksize` and nothing else. The default has
            # to be the schema's own: substituting anything else makes this
            # check pass vacuously while the rearrangement the executor then
            # performs is the other one, and the wrong one still yields a tensor
            # of the right shape.
            mode = _attr(n, "mode", "DCR")
            if mode not in ("DCR", "CRD"):
                _die(f"DepthToSpace {n.name!r} has mode {mode!r}; the ONNX "
                     f"schema for this operator names exactly two orders -- DCR "
                     f"and CRD -- and anything else is a value no executor here "
                     f"knows how to arrange channels for")
            out = t.out_shape(n.output[0], n)
            src = t.out_shape(n.input[0], n)
            b = int(block)
            if int(out[1]) * b * b != int(src[1]) or \
                    int(out[2]) != int(src[2]) * b or \
                    int(out[3]) != int(src[3]) * b:
                _die(f"DepthToSpace {n.name!r} goes {src} -> {out} with "
                     f"blocksize {b}, which is not the "
                     f"[N, C/b^2, H*b, W*b] this operator means")
            idx = add({"op": "d2s", "block": b, "mode": mode.lower(),
                       "out": [int(x) for x in out],
                       "inputs": [idx_of(n.input[0], n, "input")]}, n)
            claim(n, idx)
            continue

        if op == "Add":
            out = t.out_shape(n.output[0], n)
            a, b = (idx_of(x, n, "operands") for x in n.input[:2])
            if t.out_shape(n.input[0], n) != t.out_shape(n.input[1], n):
                _die(f"Add {n.name!r} has operands of different shapes, so it "
                     f"is a broadcast add and the runtime's elementwise add "
                     f"would not be the same operation")
            actname, prelu = fusion_of(n.name)
            idx = add({"op": "add", "act": actname, "prelu": prelu,
                       "out": [int(x) for x in out], "inputs": [a, b]}, n)
            claim(n, idx)
            a_node = act_of.get(n.name)
            if a_node is not None:
                emitted[a_node.name] = idx
                for o in a_node.output:
                    tensor_idx[o] = idx
            continue

        if op == "MaxPool":
            k = [int(x) for x in _attr(n, "kernel_shape", [1, 1])]
            strides = [int(x) for x in _attr(n, "strides", [1, 1])]
            pads = [int(x) for x in _attr(n, "pads", [0, 0, 0, 0])]
            if len(pads) != 4:
                _die(f"MaxPool {n.name!r} has pads {pads}, not four terms")
            out = t.out_shape(n.output[0], n)
            idx = add({"op": "maxpool", "kernel": k, "stride": strides,
                       "pad": pads, "out": [int(x) for x in out],
                       "inputs": [idx_of(n.input[0], n, "input")]}, n)
            claim(n, idx)
            continue

        if op == "Resize":
            mode = _attr(n, "mode", "nearest")
            if mode != "linear":
                _die(f"Resize {n.name!r} has mode {mode!r}; both necks in this "
                     f"family upsample bilinearly and a nearest one would be a "
                     f"different graph")
            ctm = _attr(n, "coordinate_transformation_mode", "half_pixel")
            if ctm != "half_pixel":
                _die(f"Resize {n.name!r} uses coordinate_transformation_mode "
                     f"{ctm!r}; half-pixel is the convention the weights were "
                     f"exported against")
            if _attr(n, "exclude_outside", 0) != 0:
                _die(f"Resize {n.name!r} sets exclude_outside, which changes "
                     f"which samples fall outside the edge")
            src = t.out_shape(n.input[0], n)
            out = t.out_shape(n.output[0], n)
            scales = t.init.get(n.input[2]) if len(n.input) > 2 else None
            sizes = t.init.get(n.input[3]) if len(n.input) > 3 else None
            if scales is not None and scales.size:
                scale = [float(scales[2]), float(scales[3])]
                want = (int(round(src[2] * scale[0])),
                        int(round(src[3] * scale[1])))
                if sizes is not None and sizes.size and (
                        int(sizes[2]) != want[0] or int(sizes[3]) != want[1]):
                    _die(f"Resize {n.name!r} carries BOTH scales {scale} and "
                         f"sizes {sizes.ravel().tolist()}, and they disagree")
            elif sizes is not None and sizes.size:
                want = (int(sizes[2]), int(sizes[3]))
                scale = [want[0] / int(src[2]), want[1] / int(src[3])]
                if want[0] % int(src[2]) or want[1] % int(src[3]):
                    _die(f"Resize {n.name!r} goes {src[2]}x{src[3]} to "
                         f"{want[0]}x{want[1]}, a non-integral ratio. A "
                         f"fractional upsample is a different sampling pattern, "
                         f"not a scale this op expresses.")
            else:
                _die(f"Resize {n.name!r} supplies neither a non-empty `scales` "
                     f"nor a non-empty `sizes`")
            if (int(out[2]), int(out[3])) != want:
                _die(f"Resize {n.name!r} goes {src[2]}x{src[3]} to "
                     f"{out[2]}x{out[3]}, which is not the {want[0]}x{want[1]} "
                     f"its own operand asks for")
            idx = add({"op": "resize",
                       "scale": [float(scale[0]), float(scale[1])],
                       "out": [int(x) for x in out],
                       "inputs": [idx_of(n.input[0], n, "input")]}, n)
            claim(n, idx)
            continue

        if op == "Transpose":
            perm = _attr(n, "perm")
            src = t.out_shape(n.input[0], n)
            if perm == [0, 3, 1, 2] and len(src) == 4 and src[3] == 3:
                # The landmark net's LEADING NHWC->NCHW, absorbed because the
                # runtime writes NCHW natively. The Pad in front of it was
                # already folded into the convolution downstream, so the tensor
                # this now names is the un-padded image.
                emitted[n.name] = -1
                tensor_idx[n.output[0]] = -1
                continue
            if perm == [0, 2, 3, 1]:
                # NCHW->NHWC, one per level per regressor. NOT absorbed: the
                # head traces Concat -> Reshape -> Transpose -> Conv and reads
                # the convolution's tensor itself, building the NHWC axis order
                # the reshape wants. Marked -2 rather than -1 so a body node
                # consuming it is refused rather than silently rewired.
                emitted[n.name] = -2
                tensor_idx[n.output[0]] = -2
                continue
            _die(f"Transpose {n.name!r} has perm {perm} on {src}. Only the "
                 f"leading NHWC->NCHW and the head's NCHW->NHWC are "
                 f"understood; a transpose anywhere else reorders elements in "
                 f"a way nothing else in this file describes.")
            continue

        if op in ("PRelu", "Clip", "Relu"):
            continue                      # fused in pass 2
        if _head_ops(n):
            continue                      # absorbed into a head
        _die(f"{t.name}: operator {op} ({n.name!r}) is not in this packer's "
             f"vocabulary")

    heads = build_heads(t, ops, names, emitted, tensor_idx, producer,
                        weights, pfx)
    return ops, names, conv_of, weights, prelus, heads


def build_heads(t, ops, names, emitted, tensor_idx, producer, weights, pfx):
    if t.name == "det":
        return _det_head(t, emitted, producer)
    return _pose_head(t, emitted, producer)


def _det_head(t, emitted, producer):
    """The SSD head, in the graph's OWN level order.

    Which output is boxes and which is scores comes from the declared trailing
    dimension -- 12 against 1 -- and never from the order the outputs happen to
    be listed in. The level order inside each comes from the Concat's input
    order, and `det_anchors()` is handed THAT order, so the anchor table is a
    consequence of the graph instead of a table somebody keeps in step with it.
    """
    by_terms = {}
    for vi in t.graph.output:
        cn = t.by_out.get(vi.name)
        if cn is None or cn.op_type != "Concat":
            _die(f"det head: the graph output {vi.name!r} is produced by "
                 f"{cn.op_type if cn else 'nothing'}, not by a Concat, so this "
                 f"is not a per-level SSD head")
        by_terms[int(t.out_shape(vi.name, cn)[-1])] = cn
    if DET_ROW not in by_terms or 1 not in by_terms:
        _die(f"det head: the graph outputs have trailing dimensions "
             f"{sorted(by_terms)}, which is neither a {DET_ROW}-term row nor a "
             f"1-term score column. The {DET_BOX_TERMS} box and "
             f"{2 * DET_LANDMARKS} landmark terms this packer splits a row "
             f"into do not describe this head.")

    def levels_of(cn, what):
        got = []
        for inp in cn.input:
            rs = t.by_out.get(inp)
            if rs is None or rs.op_type != "Reshape":
                _die(f"det head: the {what} Concat {cn.name!r} takes {inp!r}, "
                     f"produced by "
                     f"{rs.op_type if rs else 'nothing'}, but this packer reads "
                     f"the head through Reshape->Transpose->Conv")
            tr = t.by_out.get(rs.input[0])
            if tr is None or tr.op_type != "Transpose" or \
                    _attr(tr, "perm") != [0, 2, 3, 1]:
                _die(f"det head: {what} level {inp!r} is not reached through a "
                     f"NCHW->NHWC transpose, so the reshape's axis order is "
                     f"not the one this head assumes")
            conv = t.by_out.get(tr.input[0])
            if conv is None or conv.op_type != "Conv":
                _die(f"det head: the transpose above {what} level {inp!r} does "
                     f"not take a convolution, so the head is not a per-level "
                     f"regressor")
            out = t.out_shape(conv.output[0], conv)
            want = t.out_shape(inp, rs)
            if len(want) != 3:
                _die(f"det head: the reshape producing {inp!r} makes {want}, "
                     f"which is not a rows x terms table")
            terms = int(want[2])
            if terms <= 0 or int(out[1]) % terms:
                _die(f"det head: the level's convolution writes {out[1]} "
                     f"channels, which is not a whole number of {terms}-term "
                     f"rows -- the reshape would be inventing channels")
            per = int(out[1]) // terms
            rows = int(want[1])
            if rows != int(out[2]) * int(out[3]) * per:
                _die(f"det head: {what} level {rows} rows against a "
                     f"{out[2]}x{out[3]} feature map at {per} anchors per "
                     f"cell ({int(out[2]) * int(out[3]) * per}). The reshape's "
                     f"row count is what makes the anchor table line up with "
                     f"the predictions, so a disagreement here is the failure "
                     f"this whole file is written against.")
            got.append({"op": int(emitted[conv.name]), "c": int(out[1]),
                        "h": int(out[2]), "w": int(out[3]), "rows": rows,
                        "terms": terms, "per": per})
        return got

    boxes = levels_of(by_terms[DET_ROW], "box")
    scores = levels_of(by_terms[1], "score")
    if len(boxes) != len(scores):
        _die(f"det head: {len(boxes)} box levels against {len(scores)} score "
             f"levels; the two heads assemble different pyramids")
    merged = []
    for b, s in zip(boxes, scores):
        for k in ("h", "w", "per", "rows"):
            if b[k] != s[k]:
                _die(f"det head: the box level {b['h']}x{b['w']}x{b['per']} "
                     f"and the score level {s['h']}x{s['w']}x{s['per']} differ "
                     f"in {k}, so the two tables index different pyramids")
        if b["terms"] != DET_ROW:
            _die(f"det head: a box level has {b['terms']} terms, not the "
                 f"{DET_ROW} = {DET_BOX_TERMS} box + 2*{DET_LANDMARKS} "
                 f"landmark terms this decoder splits a row into")
        merged.append({"h": b["h"], "w": b["w"], "per": b["per"],
                       "rows": b["rows"], "terms": b["terms"],
                       "box_op": b["op"], "score_op": s["op"]})
    return {"det_head": merged}


def _pose_head(t, emitted, producer):
    """The five outputs, each resolved to the body node that computes it.

    Every one of them is a single convolution followed by a Reshape or a
    Transpose, and the head records WHICH convolution plus what must be done to
    its tensor -- because a Reshape is not a transpose and the mask's own
    reshape (`[1,1,256,256]` read as `[1,256,256,1]`) leaves the flat order
    alone while the heatmap's `[0,2,3,1]` genuinely reorders it.
    """
    # How many VALUES each output declares. Both a Reshape and a Transpose
    # preserve the element count, so equality with the convolution's own tensor
    # is the one invariant covering all five without assuming any layout --
    # and it is what catches a re-export that changed 39 rows to something else.
    want = {
        "landmarks": NUM_LANDMARKS * LM_COLS,        # 39 * 5 = 195
        "conf": 1,
        "mask": POSE_SIZE * POSE_SIZE,               # 256 * 256
        "world": NUM_LANDMARKS * WORLD_COLS,         # 39 * 3 = 117
    }
    found = {}
    for vi in t.graph.output:
        n = t.by_out.get(vi.name)
        if n is None:
            _die(f"pose head: the graph output {vi.name!r} is produced by "
                 f"nothing")
        # walk back through the absorbed plumbing to the body node
        cur, hops, sigmoid = n, 0, False
        while cur is not None and cur.op_type in ("Reshape", "Transpose",
                                                  "Sigmoid"):
            hops += 1
            # Recorded rather than assumed from the output's NAME. `conf` is the
            # one output this front end squashes and the other four are not, and
            # a reader that took it from the name would be hard-coding what the
            # graph states -- so a re-export that dropped the Sigmoid would
            # report a confidence of -9.7 and be believed.
            if cur.op_type == "Sigmoid":
                sigmoid = True
            cur = t.by_out.get(cur.input[0])
        if cur is None or cur.op_type != "Conv":
            _die(f"pose head: the output {vi.name!r} is produced by "
                 f"{n.op_type} over {cur.op_type if cur else 'nothing'}, not by "
                 f"a single convolution -- this head reads one tensor per "
                 f"output")
        if int(emitted[cur.name]) < 0:
            _die(f"pose head: the output {vi.name!r} resolves to "
                 f"{cur.name!r}, which was absorbed rather than emitted, so "
                 f"there is no tensor to read")
        decl = t.out_shape(vi.name, n)
        conv_out = t.out_shape(cur.output[0], cur)
        key = _pose_key(decl, conv_out)
        if key in found:
            _die(f"pose head: two graph outputs both resolve to {key!r} "
                 f"({found[key]['node']} and {vi.name})")
        transposed = n.op_type == "Transpose"
        # A Reshape and a Transpose both preserve the element count, so the
        # output and the convolution that computes it must hold the same number
        # of values -- the invariant that covers all five without assuming any
        # layout. The four whose count is fixed by the model are then checked
        # against that count as well; the heatmap is checked only for this,
        # because the reference never reads it.
        if int(np.prod(decl)) != int(np.prod(conv_out)):
            _die(f"pose head: {vi.name!r} declares {decl} = "
                 f"{int(np.prod(decl))} values but the convolution "
                 f"{cur.name!r} writes {conv_out} = "
                 f"{int(np.prod(conv_out))}. Neither a reshape nor a transpose "
                 f"changes how many values there are.")
        if key in want and int(np.prod(conv_out)) != want[key]:
            _die(f"pose head: {key} should be {want[key]} values but the "
                 f"convolution {cur.name!r} writes {conv_out} = "
                 f"{int(np.prod(conv_out))}. This decoder reads "
                 f"{NUM_LANDMARKS} rows of {LM_COLS} screen columns and "
                 f"{WORLD_COLS} world columns, so a different count is a "
                 f"different model.")
        found[key] = {"node": vi.name, "op": int(emitted[cur.name]),
                      "conv": cur.name, "decl": [int(x) for x in decl],
                      "conv_out": [int(x) for x in conv_out],
                      "transposed": transposed, "sigmoid": sigmoid,
                      "hops": hops}
    # The one output whose CONSUMER needs to know its range: the threshold and
    # the reported number both assume a probability, and a raw logit compared
    # against 0.5 accepts a person the model rejects.
    if not found.get("conf", {}).get("sigmoid", False):
        _die(f"pose head: {found.get('conf', {}).get('node', 'the confidence')} "
             f"has no Sigmoid between it and its convolution. The zoo's "
             f"_postprocess compares this number against "
             f"confThreshold={POSE_CONF_THR:g} and returns it, and both assume "
             f"a probability rather than a logit.")
    missing = set(want) - set(found)
    if missing:
        _die(f"pose head: the graph does not produce {sorted(missing)}; it "
             f"produces {sorted(found)}. This decoder reads exactly the five "
             f"tensors MediaPipe Pose declares.")
    # heatmap: produced, recorded, and known to be unused by the reference.
    hm = found.get("heatmap")
    return {"pose_head": {k: found[k] for k in sorted(found)},
            "pose_unused": ["heatmap"] if hm else []}


def _pose_key(decl, conv_out):
    """Which of the five this output is, from its DECLARED shape."""
    d = [int(x) for x in decl]
    c = [int(x) for x in conv_out]
    flat = int(np.prod(d))
    if flat == NUM_LANDMARKS * LM_COLS:
        return "landmarks"
    if flat == 1:
        return "conf"
    if flat == NUM_LANDMARKS * WORLD_COLS:
        return "world"
    if len(d) == 4 and d[1] == 1 and d[2] == d[3] == POSE_SIZE:
        return "mask"
    if len(d) == 4 and d[3] == NUM_LANDMARKS:
        return "heatmap"
    if len(d) == 4 and c[1] == 1 and c[2] == c[3] == POSE_SIZE:
        return "mask"
    _die(f"pose head: an output declared {d} (convolution writes {c}) is none "
         f"of landmarks/{NUM_LANDMARKS * LM_COLS}, conf/1, world/"
         f"{NUM_LANDMARKS * WORLD_COLS}, mask/{POSE_SIZE}x{POSE_SIZE} or "
         f"heatmap/{NUM_LANDMARKS} channels")


def npu_panels(t, pfx, ops, conv_of, weights, device):
    """Pre-tiled bf16 B panels for the array backend, one per DENSE convolution.

    Returns (panels, streams, (tile_k, tile_n)) where panels maps a convolution
    INDEX (the `conv` field an op carries) to (ndarray, layout, pk, pn) and
    streams is the sorted list of distinct padded (K, N) shapes, which is the set
    tools/export/exporters/gemm_rtp/geometry.py builds a design from.

    WHY ONLY THE DENSE ONES, AND IT IS NOT A SUBCASE
    -------------------------------------------------
    62 of this container's 161 convolutions are depthwise, and a depthwise filter
    reduces within ONE channel: its K is kh*kw and its N is 1, so there is no
    [M, N] GEMM in it to dispatch and a stream named for it would be a name with
    nothing behind it. They are counted, not converted, and the runtime keeps
    them on the host whatever a design set says. That share -- 77.2M MAC, 12.3 %
    of the container -- is one of the two numbers in the arch=8 registry row.

    EVERYTHING ELSE HERE IS arch=6's npu_panels(), AND IT IS COPIED RATHER THAN
    REUSED ON PURPOSE. pose.py's takes a traced graph and a conv-name map; this
    container stores two graphs under two PREFIXES in one file, so the signature
    cannot match without the prefix threaded through a shared helper that
    arch=6's caller does not have. The arithmetic that decides a panel -- pad K
    up to tile_k, pad N up to tile_n*COLS, and TRANSPOSE the weight -- is the
    part that must be identical, and it is the part transcribed. The transpose
    in particular is not a detail: a reshape here has the right SIZE, the right
    tileability and a matching layout_hash, and produces a transposed weight, so
    the array returns a network that finds hundreds of people in a photograph
    with three. pose.py's comment on it is the reason this paragraph exists.
    """
    from npue import (cols_for_device, gemm_b_layout, mac_for_device,  # noqa: E402
                      tile_b)

    TILE_K, TILE_N = 64, 32
    mac_s, mac_t = mac_for_device(device, "BF16")
    # N pads to tile_n * COLS. On npu1 that is 32*4 = 128, and this network's
    # output channel counts are not all multiples of it -- 384, 640 and 1152
    # appear and pad up, while 256 is already a multiple. Padding to tile_n alone
    # would store a 32-wide panel for a design that reads 128, and the columns
    # past the end are whatever follows in the container's data region.
    N_MULT = TILE_N * cols_for_device(device)
    layout = gemm_b_layout(TILE_K, TILE_N, mac_s, mac_t)
    panels, streams = {}, set()
    # DENSE ONLY, AND THE FILTER IS `op == "conv"` RATHER THAN `group == 1` BECAUSE
    # THE FIRST VERSION OF THIS FUNCTION DID NOT FILTER AT ALL, and the container
    # it produced carried TWENTY-SIX streams where the design has TWENTY-TWO. The
    # four extras -- conv64x512, conv64x640, conv64x768, conv64x1152 -- are every
    # depthwise convolution in the landmark network: their Cin is 1 (per group),
    # so K pads from 3x3=9 up to 64, and N is their channel count. A depthwise
    # filter has no [M, N] GEMM in it, so those four slots were names with
    # nothing behind them, built by indexing `conv_of` when the caller had
    # already said which ops were dense.
    #
    # It was caught by tools/verify/verify_pose_streamset.py comparing the
    # container's npu_streams against geometry.py's list, in both directions --
    # which is that tool's whole reason for existing, applied to the container
    # rather than to a design set.
    dense = {o["conv"] for o in ops if o["op"] == "conv"}
    for idx in sorted(dense):
        w = weights[f"{pfx}.conv.{idx}.w"]
        cout, cin, kh, kw = w.shape
        K, N = cin * kh * kw, cout
        pk = ((K + TILE_K - 1) // TILE_K) * TILE_K
        pn = ((N + N_MULT - 1) // N_MULT) * N_MULT
        b = np.zeros((pk, pn), dtype=np.float32)
        b[:K, :N] = w.reshape(N, K).T
        panels[idx] = (tile_b(b, TILE_K, TILE_N, s=mac_s, t=mac_t), layout,
                       pk, pn)
        streams.add((pk, pn))
    return panels, sorted(streams), (TILE_K, TILE_N)


def mac_split(t, conv_of):
    """MACs the array could take (dense) and MACs it cannot (depthwise)."""
    dense = dw = 0
    shapes = {}
    for n in t.nodes:
        if n.op_type != "Conv":
            continue
        w = t.init[n.input[1]]
        cout, cin_g, kh, kw = [int(x) for x in w.shape]
        groups = int(_attr(n, "group", 1))
        out = t.shape.get(n.output[0])
        if out is None or any(not isinstance(x, int) for x in out):
            continue
        # `cin_g` is already PER GROUP, so it is used as-is: dividing by groups
        # again would make every depthwise convolution 1 // 32 = 0 MAC and
        # report the host share as zero.
        m = int(out[2]) * int(out[3]) * cout * cin_g * kh * kw
        if groups == 1:
            dense += m
        else:
            dw += m
        key = f"{int(out[1])}x{int(out[2])}x{int(out[3])}"
        shapes[key] = shapes.get(key, 0) + 1
    return dense, dw, shapes


def pack_mppose(det_onnx, pose_onnx, out_path, dry_run=False, device=None,
                npu=False):
    td = trace("det", onnx.load(str(det_onnx)))
    tp = trace("pose", onnx.load(str(pose_onnx)))

    ops_d, names_d, conv_d, w_d, prelu_d, head_d = emit_ops(td, "det")
    ops_p, names_p, conv_p, w_p, prelu_p, head_p = emit_ops(tp, "pose")

    a_d, dw_d, shapes_d = mac_split(td, conv_d)
    a_p, dw_p, shapes_p = mac_split(tp, conv_p)

    # The anchor table comes FROM the head's level list, in the head's own
    # order, so it cannot drift out of step with the predictions it indexes.
    levels = [(l["h"], l["w"], l["per"]) for l in head_d["det_head"]]
    anchors = det_anchors(levels)
    rows = sum(l["rows"] for l in head_d["det_head"])
    if int(anchors.shape[0]) != rows:
        _die(f"det head: the levels report {rows} rows but the generated "
             f"anchor table has {anchors.shape[0]}. The table is indexed by "
             f"row, so they must be the same number.")

    if dry_run:
        for tag, ops, conv, w, a, dw, shapes in (
                ("det", ops_d, conv_d, w_d, a_d, dw_d, shapes_d),
                ("pose", ops_p, conv_p, w_p, a_p, dw_p, shapes_p)):
            counts = {}
            for o in ops:
                counts[o["op"]] = counts.get(o["op"], 0) + 1
            print(f"{tag}: {len(ops)} body ops: " + ", ".join(
                f"{counts[k]} {k}" for k in sorted(counts)))
            dense = sum(1 for o in ops if o["op"] == "conv")
            dwc = sum(1 for o in ops if o["op"] == "dwconv")
            print(f"  convs {len(conv)}: {dense} array-eligible, {dwc} "
                  f"depthwise on the host; weights "
                  f"{sum(v.size for v in w.values()) / 1e6:.2f} M params")
            print(f"  MAC   array {a / 1e6:.1f} M, host(depthwise) "
                  f"{dw / 1e6:.1f} M  ({100.0 * dw / (a + dw):.1f}% on the "
                  f"host)")
        print(f"anchors  {anchors.shape[0]} generated from levels {levels}")
        print(f"det head {[(l['h'], l['w'], l['per'], l['rows']) for l in head_d['det_head']]}")
        print(f"pose head: " + ", ".join(
            f"{k} <- node {v['op']}"
            + (" (transposed)" if v["transposed"] else "")
            for k, v in sorted(head_p["pose_head"].items())))
        if head_p["pose_unused"]:
            print(f"  unused by the reference: {head_p['pose_unused']}")
        return

    config = {
        "arch": ARCH_STRING,
        # Its OWN kind and not "pose", which is arch=6's. Two architectures can
        # both be called pose in the ordinary sense -- they are both skeletons on
        # a frame -- and they share no flag, no threshold, no head and no input
        # size. The kind string is what the registry's rows and the exporter's
        # validation key on, so a shared one would merge two unrelated rows.
        "kind": "mppose",
        "model": "mediapipe-pose",
        "det_input_size": DET_SIZE,
        "pose_input_size": POSE_SIZE,
        "num_landmarks": NUM_LANDMARKS,
        "num_keypoints": NUM_KEYPOINTS,
        "lm_cols": LM_COLS,
        "world_cols": WORLD_COLS,
        "det_num_anchors": int(anchors.shape[0]),
        "det_anchor_levels": json.dumps([list(l) for l in levels],
                                        separators=(",", ":")),
        "det_row_terms": DET_ROW,
        "det_landmarks": DET_LANDMARKS,
        "person_lm_mid_hip": PERSON_LM_MID_HIP,
        "person_lm_full_body": PERSON_LM_FULL_BODY,
        "person_box_pre_enlarge": PERSON_BOX_PRE_ENLARGE,
        "person_box_enlarge": PERSON_BOX_ENLARGE,
        "score_threshold": SCORE_THR,
        "nms_threshold": NMS_THR,
        "top_k": TOP_K,
        "pose_conf_threshold": POSE_CONF_THR,
        # TWO NORMALISATIONS, AND THEY ARE NOT THE SAME. This is the one place
        # where mppose and mediapipe-hands differ in more than geometry, and
        # writing a single image_mean/image_std pair -- which is what the hands
        # packer can do -- would be a silently wrong detector.
        #
        #   detector  (mp_persondet._preprocess)   (u8/255 - 0.5) * 2  -> [-1, 1]
        #   landmark  (mp_pose._preprocess)        u8/255              -> [ 0, 1]
        #
        # It follows from the zoo code and was verified against it, not inferred:
        # hands' palm detector really is u8/255, which is why one pair served
        # both of its networks and one does not here.
        #
        # IT ALSO SAYS WHERE THE ZERO PAD IS APPLIED, because that is a tensor
        # value and not a raster one. The detector divides by 255 and rescales
        # FIRST and pads with 0 in the [-1,1] tensor afterwards, so its border
        # is 0.0 -- the middle of the range, not the bottom of it. A reader that
        # letterboxes a uint8 raster with 0 and normalises afterwards puts -1.0
        # in that border, which is the one pixel value that means "saturated
        # black" to this stem. The landmark net pads in the uint8 raster BEFORE
        # the rotation and the resize and divides by 255 at the end, so its
        # border is 0.0 in the same way -- but for the opposite reason, and the
        # two are not interchangeable.
        "det_image_mean": [0.5, 0.5, 0.5],
        "det_image_std": [0.5, 0.5, 0.5],
        "pose_image_mean": [0.0, 0.0, 0.0],
        "pose_image_std": [1.0, 1.0, 1.0],
        "det_num_convs": len(conv_d),
        "pose_num_convs": len(conv_p),
        "array_mac": int(a_d + a_p),
        "host_mac": int(dw_d + dw_p),
        "det_graph": json.dumps(ops_d, separators=(",", ":")),
        "pose_graph": json.dumps(ops_p, separators=(",", ":")),
        "det_graph_nodes": json.dumps(names_d, separators=(",", ":")),
        "pose_graph_nodes": json.dumps(names_p, separators=(",", ":")),
        "det_head": json.dumps(head_d["det_head"], separators=(",", ":")),
        "pose_head": json.dumps(head_p["pose_head"], separators=(",", ":")),
        "pose_unused": json.dumps(head_p["pose_unused"], separators=(",", ":")),
    }

    w = Writer(config, arch=ARCH_MEDIAPIPE_POSE_DET_SSD_LM_REGRESS)
    both = {**w_d, **w_p, **prelu_d, **prelu_p}
    check_disjoint(w_d, w_p, "convolution weights")
    check_disjoint(prelu_d, prelu_p, "prelu slopes")
    for name, arr in sorted(both.items()):
        w.add(name, arr, "F32",
              "conv" if ".conv." in name else "prelu", list(arr.shape))
    w.add("det_anchors", anchors.astype(np.float32), "F32", "anchors",
          list(anchors.shape))

    if npu:
        # A device is REQUIRED and not defaulted. `cols_for_device(None)` returns
        # a column count for a device nobody named, and N pads to
        # tile_n * cols -- so a container built that way stores panels narrower
        # than the design that will read them, the layout_hash still matches
        # because both sides derive it from the same constants, and the array
        # multiplies whatever follows in the data region. Refused here rather
        # than guessed, for the same reason arch=6's --npu path is.
        if device is None:
            _die("--npu needs a device: pass --device npu1 (or npu2). N pads to "
                 "tile_n*cols and cols is a property of the array, so a panel "
                 "built without one is not a panel the design can read.")
        from npue import to_bf16_bits  # noqa: E402
        # BOTH graphs' panels, because both graphs' dense convolutions dispatch.
        # A stream set for one network and not the other is a design whose slots
        # half match, which resolves for the detector and not for the landmarks.
        pd, sd, tile = npu_panels(td, "det", ops_d, conv_d, w_d, device)
        pp, sp, tile2 = npu_panels(tp, "pose", ops_p, conv_p, w_p, device)
        if tile != tile2:
            _die(f"the two networks were tiled differently ({tile} and {tile2}), "
                 f"so one container would carry two panel layouts under one "
                 f"design's layout_hash")
        allstreams = sorted(set(sd) | set(sp))
        name_of = {k: f"conv{k[0]}x{k[1]}" for k in allstreams}
        for pfx, ops, panels in (("det", ops_d, pd), ("pose", ops_p, pp)):
            for o in ops:
                if o["op"] != "conv":
                    continue
                i = o["conv"]
                panel, layout, pk, pn = panels[i]
                w.add(f"{pfx}.conv.{i}.btile", to_bf16_bits(panel), "BF16",
                      "gemm_b", [pk, pn], layout=layout,
                      padded_shape=[pk, pn])
                o["stream"] = name_of[(pk, pn)]
        # Into w.config and NOT into the local `config` dict: Writer.__init__ took
        # a copy of `config` when it was constructed above, so a key set on the
        # local dict now is a dict the file never sees. That mistake is written
        # down at length in tools/pack/packers/pose.py, where every --npu
        # container was once shipped with no npu_streams at all.
        w.config["npu_streams"] = json.dumps(
            [{"op": name_of[k], "k": k[0], "n": k[1], "tile_k": tile[0],
              "tile_n": tile[1]} for k in allstreams],
            separators=(",", ":"))
        # Re-serialised because the stream names above went into `ops` AFTER the
        # config dict was built, so the strings the writer already holds do not
        # have them.
        w.config["det_graph"] = json.dumps(ops_d, separators=(",", ":"))
        w.config["pose_graph"] = json.dumps(ops_p, separators=(",", ":"))

    w.write(out_path)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({size / 1e6:.1f} MB, arch={ARCH_STRING})")
    print(f"  {len(both)} weight tensors, {anchors.shape[0]} anchors")
    if npu:
        print(f"  npu        {len(pd) + len(pp)} pre-tiled bf16 panels over "
              f"{len(allstreams)} padded (K, N) shapes, tile {tile[0]}x{tile[1]}, "
              f"device {device}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("det_onnx")
    ap.add_argument("pose_onnx")
    ap.add_argument("out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--npu", action="store_true",
                    help="add the pre-tiled bf16 B panels and the stream table, "
                         "so `--npu-ops conv` has slots to dispatch into. OFF by "
                         "default because there is no design set for arch=8 on "
                         "this machine, and a container with panels and nothing "
                         "to dispatch into is 5x the size for no path.")
    ap.add_argument("--device", default=None,
                    help="which array the panels are tiled for (npu1, npu2). "
                         "Required with --npu, because N pads to tile_n*cols and "
                         "cols is a property of the array.")
    a = ap.parse_args()
    pack_mppose(a.det_onnx, a.pose_onnx, a.out, dry_run=a.dry_run,
                device=a.device, npu=a.npu)


if __name__ == "__main__":
    main()
