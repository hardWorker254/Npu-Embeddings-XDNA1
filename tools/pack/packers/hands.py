# NpuEmbeddings -- pack MediaPipe Hands into an arch=7 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# The fourth family, and the first whose container holds TWO networks. That is
# not a packaging preference -- it is the model's shape. The palm detector finds
# a hand; the landmark network consumes a cropped, ROTATED palm ROI and has no
# way to find a hand in a frame by itself. A container with only the second one
# would return a healthy-looking "no hand here" for every image, and a container
# with only the first would stop at a box. So the container holds both, and the
# geometry that connects them -- which two of the detector's seven landmarks fix
# the rotation, and by how much the box is shifted and enlarged -- is written
# into the config as NUMBERS rather than left to a runtime's idea of the
# pipeline. Those numbers decide where the landmark network looks, and a
# landmark network pointed at the wrong crop returns a plausible hand rather
# than an error, which is exactly why they belong to the container.
#
# WHAT RUNS WHERE, AND WHY DEPTHWISE IS NOT A REFUSAL
# ===================================================
# 39 of the 100 convolutions are DEPTHWISE (23 in the palm detector, 16 in the
# landmark network) and they stay on the host, in the container, by name. It is
# tempting to call that a refusal, the way this packer's pose sibling refuses a
# graph it cannot trace; it is not, and the difference matters.
#
# A depthwise convolution is not a [K, N] matrix: out[o] depends on in[o] alone,
# so its matrix is block-diagonal, N blocks of kh*kw. The array's operand is
# dense, so there is no dense panel to hand it. The three ways out are all worse
# than the host:
#   * one GEMM per output channel -- 672 dispatches for a single 672-channel
#     convolution, each paying a full dispatch for a kh*kw inner product;
#   * im2col with K = cin*kh*kw -- valid only for a DENSE convolution, and it
#     would quietly compute the dense convolution instead of the sparse one;
#   * folding the scale into the following pointwise convolution -- impossible,
#     because a per-channel scale does not commute with the nonlinearity
#     MobileNet puts between the depthwise and the pointwise.
# Measured share of the work this leaves on the host, from the graph's own
# declared output shapes: 51.1M of 283.2M MAC for the palm detector (18.1%),
# 21.6M of 145.5M for the landmark network (14.9%).
#
# (An earlier pass at these numbers said 359.9M and 183.1M. Both were wrong, and
# worth recording why: they came from multiplying by the INPUT spatial size of
# the 5x5 stems instead of their OUTPUT size. The palm stem is stride 2, so it
# emits 96x96 and costs 22.1M MAC, not the 88.5M that a 192x192 output implies.
# The split that matters -- dense versus depthwise -- was not affected, because
# both figures divided the same two totals the same wrong way; the totals
# themselves were off by 27% and 26%.)
#
# So they are graph NODES -- "dwconv", executed by the host -- while the 61
# dense convolutions are "conv", array-eligible. The distinction is recorded per
# node, and the packer's own report separates array_mac from host_mac so that a
# claim about "how much goes to the array" has a number to be checked against.
#
# THE OP VOCABULARY IS MOBILET'S, NOT YOLOV8'S
# =============================================
# PReLU (per-channel slope) in the palm detector and relu6 (a Clip, 0..6) in the
# landmark network, where YOLOv8 has SiLU; a bilinear Resize with half-pixel
# coordinates in the FPN neck; and a channel-axis Pad that MediaPipe inserts so
# widths reach the next block's divisibility. Each activation is fused into the
# convolution that owns it, so no node reads 283.2M MAC of activation output
# from memory for a nonlinearity the checkpoint had already folded in.
#
# THE ANCHORS ARE GENERATED, AND THAT WAS NOT OBVIOUS
# ===================================================
# The palm detector's 2016 SSD anchor centres look like a magic table -- the zoo
# demo ships one as 2016 literal pairs. It is not one. It is two grids of cell
# CENTRES, (j + 0.5) / n, at two pyramid levels with different anchors per
# cell: 24x24 cells at 2 each (576 * 2 = 1152) then 12x12 cells at 6 each
# (144 * 6 = 864), 1152 + 864 = 2016, in exactly the order the graph's Concat
# assembles them.
#
# The first guess -- the obvious one, j / n over a 24/16/32 pyramid -- produced
# 2116 rows and was discarded; so did "24 per level, four times". What settled it
# was the graph: Concat takes the 1152-row tensor first, and 1152 = 576 * 2 pins
# the 24x24 level at 2 anchors per cell with no reference to the table at all.
# `palm_anchors()` below is that derivation, and tools/verify/verify_hands.py
# checks it against the zoo's literal table: they agree to 7.5e-09, which is the
# zoo's own 8-decimal truncation of these same fractions -- 4e-07 px on a
# 520-pixel frame -- and not a different formula.

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))

from npue import ARCH_MEDIAPIPE_HANDS_PALM_SSD_LM_HEATMAP, Writer  # noqa: E402

ARCH_STRING = "mediapipe_hands_palm_ssd_lm_heatmap"

PALM_SIZE = 192
LM_SIZE = 224
NUM_LANDMARKS = 21
LM_FLAT = NUM_LANDMARKS * 3          # 63: screen landmarks x, y, z
BOX_TERMS = 4                        # x1, y1, x2, y2
PALM_LM_COUNT = 7                    # landmarks the detector regresses
# Each landmark costs TWO terms, not one: a point in the image is (x, y). So the
# row is 4 + 7*2 = 18 and not 4 + 7 = 11 -- and the graph's [1, 2016, 18] is
# what catches that, which is why the head reads the row's width from the
# output's declared shape rather than from this constant.
PALM_LM_TERMS = 2 * PALM_LM_COUNT
PALM_ROW = BOX_TERMS + PALM_LM_TERMS  # 18

# The detector's seven landmarks arrive already selected, and it is the LAST TWO
# -- the wrist (0) and the middle-finger base (2) -- that fix the rotation.
# Written down because a re-export that reordered them would rotate every hand
# by whatever the difference happened to be, and the landmark network would
# still return a plausible hand.
PALM_LM_WRIST = 0
PALM_LM_MIDDLE_BASE = 2

# Crop geometry, from the reference implementation. MODEL decisions, not tuning.
PALM_PRE_SHIFT = (0.0, 0.0)      # before the rotation crop
PALM_PRE_ENLARGE = 4.0
PALM_SHIFT = (0.0, -0.4)         # after it
PALM_ENLARGE = 3.0
HAND_SHIFT = (0.0, -0.1)         # the reported hand box
HAND_ENLARGE = 1.65

SCORE_THR = 0.5
NMS_THR = 0.3
TOP_K = 5000


def _die(msg):
    raise SystemExit(f"{ARCH_STRING}: {msg}")


def check_disjoint(a, b, what):
    """Two networks' tensor namespaces must not overlap.

    Both number their convolutions from zero, so an unprefixed write silently
    overwrites. This is the check that would have caught it, and it is here
    because the alternative is a container that looks fine.
    """
    clash = sorted(set(a) & set(b))
    if clash:
        _die(f"the two networks both contribute {what} under the same names "
             f"({len(clash)} of them, e.g. {clash[:3]}). Prefixes exist for "
             f"this; without them the later one silently replaces the earlier.")


def palm_anchors(levels):
    """The SSD anchor centres, in the order the graph's output rows use.

    Derived from `levels` -- the graph's OWN per-level (height, width, anchors
    per cell), in the order the head's Concat assembles them -- and from nothing
    else. No pyramid is hardcoded here on purpose: see this file's header for the
    two wrong guesses that a hardcoded pyramid invites, and for why the only
    thing worth pinning is the structure (cell centres) rather than the two
    levels' sizes.
    """
    rows = []
    for h, w, per_cell in levels:
        for i in range(h):
            for j in range(w):
                # Cell CENTRES, (j + 0.5) / w -- not the left edges a detector
                # "usually" uses. The distinction is visible in the table: cell
                # centres put every coordinate on the odd multiples of 1/(2*w),
                # and the zoo's literal values are exactly those, truncated to
                # eight decimals.
                rows.extend([((j + 0.5) / w, (i + 0.5) / h)] * per_cell)
    return np.array(rows, np.float64)


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
    # Shape inference is an optimisation, not a gate: the graph's own value_info
    # plus the graph inputs already cover what the emitted nodes need, and the
    # refusal above fires per node rather than once, at import.
    try:
        inf = onnx.shape_inference.infer_shapes(model)
        for vi in inf.graph.value_info:
            t.shape[vi.name] = [d.dim_param or d.dim_value
                                for d in vi.type.tensor_type.shape.dim]
    except Exception as e:
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
                    onnx.AttributeProto.STRING: lambda: a.s.decode(),
                    }.get(a.type, lambda: default)()
    return default


def emit_ops(t, pfx):
    """The traced graph -> the runtime's op list, plus its two heads.

    `pfx` namespaces every tensor this network contributes. It is not cosmetic:
    both networks number their convolutions from zero, so writing `conv.0.w`
    from both would let the landmark network's 24-channel stem silently REPLACE
    the palm detector's 32-channel one. Nothing would say so -- the container
    would be a well-formed arch 7 with half the weights missing and the other
    half wrong, and the first symptom would be a shape mismatch a long way from
    the cause.

    Returns (ops, names, conv_of, weights, prelus, heads).

    Three things happen here that are not serialisation:

    1. ACTIVATIONS VANISH. A PRelu or a relu6 Clip that follows exactly one
       node and is consumed exactly once is fused into that convolution as
       "act". The activation does not change a tensor's shape, so the
       convolution and its activation share a shape and every downstream
       resolution stays valid. Anything else -- a second consumer, a producer
       that is not a convolution, two activations on one convolution -- is a
       refusal, because "who does this value belong to" has no answer that does
       not involve guessing.

    2. TRANSPOSES VANISH, and each one is CHECKED. There are exactly five in
       these two graphs: one leading NHWC->NCHW per network (the image arrives
       NHWC, because both checkpoints take 1x192x192x3 and 1x224x224x3) and
       four NCHW->NHWC inside the palm head, which the head op reads directly.
       A transpose anywhere else reorders elements in a way nothing else here
       describes.

    3. THE OPERAND IS AN INDEX INTO THE EMITTED LIST, not a tensor name. Node
       0's operand is -1, which is the image. A consumer of a dropped tensor is
       a refusal, and the runtime's geometry reader refuses an operand that is
       neither -1 nor an earlier node.
    """
    ops, names = [], []
    conv_of, weights, prelus = {}, {}, {}
    # Built up front, not during the walk: which activations can be fused is a
    # question about the graph's SHAPE, and it has to be answered before the
    # first node is emitted.
    producer = {o: n.name for n in t.nodes for o in n.output}
    tensor_idx = {}
    emitted = {}

    consumers = {}
    for n in t.nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)

    # -- pass 1: activations, and which op owns each
    #
    # The owner is NOT always a convolution. MediaPipe's inverted-residual block
    # ends with the residual Add and the NEXT block's depthwise convolution
    # starts with a PReLU, so `Add -> PRelu -> Conv` is the ordinary shape here
    # and fusing only into Convolutions refuses eleven of the twenty-six PReLUs.
    # The activation is fused into whatever produced its input, and Add grows an
    # epilogue the same way a convolution does.
    FUSIBLE = ("Conv", "Add")
    act_of = {}
    for n in t.nodes:
        if n.op_type not in ("PRelu", "Clip"):
            continue
        src = n.input[0]
        p = producer.get(src)
        if p is None or len(consumers.get(src, [])) != 1:
            _die(f"{n.op_type} {n.name!r} does not sit between exactly one node "
                 f"and exactly one consumer, so it cannot be fused into that "
                 f"node's epilogue. Fusing it anyway would either double-apply "
                 f"it or drop it.")
        kind = t.by_out[src].op_type
        if kind not in FUSIBLE:
            _die(f"{n.op_type} {n.name!r} follows a {kind}, and this packer "
                 f"fuses activations into {FUSIBLE} epilogues only. A "
                 f"nonlinearity after a {kind} is a separate pass over that "
                 f"tensor's output, which is exactly what fusing exists to "
                 f"avoid.")
        if p in act_of:
            _die(f"{p!r} is claimed by two activations ({act_of[p]}, {n.name}). "
                 f"Impossible in a real graph, so the trace is wrong.")
        act_of[p] = n

    prelu_seq = [0]

    def fusion_of(node_name, cout):
        """(act_name, prelu_index) for a node, or ('none', None)."""
        act = act_of.get(node_name)
        if act is None:
            return "none", None
        if act.op_type == "PRelu":
            k = prelu_seq[0]
            prelu_seq[0] += 1
            prelus[f"{pfx}.prelu.{k}.slope"] = t.init[act.input[1]].astype(np.float32)
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
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r} ({what}), "
                 f"which no traced node produces.")
        p = producer[tensor]
        if p not in emitted:
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r}, which is "
                 f"{p!r}'s output, and that node is not in the emitted list. "
                 f"Only a fused activation may vanish.")
        if emitted[p] == -2:
            _die(f"{node.op_type} {node.name!r} consumes {tensor!r}, which is a "
                 f"head-layout transpose ({p!r}). Only `palm_head` may read "
                 f"those, because only it knows the reshape's axis order.")
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

        if op == "Conv":
            w = t.init[n.input[1]]
            cout, cin, kh, kw = w.shape
            groups = _attr(n, "group", 1)
            strides = _attr(n, "strides", [1, 1])
            pads = _attr(n, "pads", [0, 0, 0, 0])
            if len(pads) != 4:
                _die(f"Conv {n.name!r} has pads {pads}, which is not four "
                     f"terms. ONNX orders them [top, left, bottom, right] and "
                     f"they are NOT symmetric -- this detector's 5x5 stem pads "
                     f"1 above and 2 below -- so keeping only the first pair "
                     f"loses a row of the output.")
            if _attr(n, "auto_pad", "NOTSET") != "NOTSET":
                _die(f"Conv {n.name!r} has auto_pad="
                     f"{_attr(n, 'auto_pad')!r}. Only explicit pads are "
                     f"expressed; auto_pad is a promise about padding this "
                     f"packer does not keep.")
            # A depthwise convolution stores ONE input channel per group, so its
            # weight is [cout, 1, kh, kw] with groups == cout -- `groups == cin`
            # is the wrong test and rejects every one of the 39.
            if groups != 1 and w.shape[1] != 1:
                _die(f"Conv {n.name!r} has group={groups} with "
                     f"{w.shape[1]} input channels per group, which is neither 1 "
                     f"(dense) nor 1-per-group (depthwise). A grouped "
                     f"convolution is a THIRD thing with no op here, and packing "
                     f"it as dense would compute a different network.")
            if groups != 1 and groups != cout:
                _die(f"Conv {n.name!r} has group={groups} but {cout} output "
                     f"channels, so it is not depthwise either")
            dense = groups == 1
            out = t.out_shape(n.output[0], n)
            if out[1] != cout:
                _die(f"Conv {n.name!r} emits {out[1]} channels from a weight "
                     f"with cout={cout}")
            i = len(conv_of)
            conv_of[n.name] = i
            weights[f"{pfx}.conv.{i}.w"] = w.astype(np.float32)
            if len(n.input) > 2:
                weights[f"{pfx}.conv.{i}.b"] = t.init[n.input[2]].astype(np.float32)
            act = act_of.get(n.name)
            actname, prelu = fusion_of(n.name, cout)
            claim(n, add({
                "op": "conv" if dense else "dwconv",
                "conv": i,
                "act": actname,
                "prelu": prelu,
                "stride": [int(strides[0]), int(strides[1])],
                "pad": [int(pads[0]), int(pads[1]), int(pads[2]),
                        int(pads[3])],
                "group": int(groups),
                "out": [int(x) for x in out],
                "inputs": [idx_of(n.input[0], n, "activations")],
            }, n))
            # The fused activation's OUTPUT stands for this convolution's output
            if act is not None:
                _rewire(t, n, act, emitted[n.name], emitted, tensor_idx)
            continue

        if op == "Add":
            out = t.out_shape(n.output[0], n)
            a, b = (idx_of(x, n, "operands") for x in n.input[:2])
            if t.out_shape(n.input[0], n) != t.out_shape(n.input[1], n):
                _die(f"Add {n.name!r} has operands of different shapes, so it "
                     f"is a broadcast add and the runtime's elementwise add "
                     f"would not be the same operation")
            actname, prelu = fusion_of(n.name, out[1])
            idx = add({"op": "add", "act": actname, "prelu": prelu,
                       "out": [int(x) for x in out], "inputs": [a, b]}, n)
            claim(n, idx)
            _rewire(t, n, act_of.get(n.name), idx, emitted, tensor_idx)
            continue

        if op == "MaxPool":
            k = _attr(n, "kernel_shape", [1, 1])
            strides = _attr(n, "strides", [1, 1])
            pads = _attr(n, "pads", [0, 0, 0, 0])
            if len(pads) != 4:
                _die(f"MaxPool {n.name!r} has pads {pads}, not four terms")
            out = t.out_shape(n.output[0], n)
            claim(n, add({
                "op": "maxpool",
                "kernel": [int(k[0]), int(k[1])],
                "stride": [int(strides[0]), int(strides[1])],
                "pad": [int(pads[0]), int(pads[1]), int(pads[2]),
                        int(pads[3])],
                "out": [int(x) for x in out],
                "inputs": [idx_of(n.input[0], n, "input")],
            }, n))
            continue

        if op == "Pad":
            pads = t.init[n.input[1]].astype(int).ravel().tolist()
            src = t.out_shape(n.input[0], n)
            out = t.out_shape(n.output[0], n)
            # ONNX's Pad order for a 4-D graph is
            # [x1_begin, x2_begin, x1_end, x2_end, c1_begin, c2_begin, c1_end,
            #  c2_end] -- the CHANNEL terms are 4..8, not 2..8. Reading them as
            # spatial refuses the padding this model is full of.
            if (len(pads) != 8 or any(v != 0 for v in pads[:4])
                    or pads[4] or pads[7]):
                _die(f"Pad {n.name!r} has pads {pads}, which is not a plain "
                     f"zero-pad of the channel axis at its end. This packer "
                     f"only expresses MediaPipe's channel padding; anything else "
                     f"is a different operator than the one it would emit.")
            count = pads[5]
            if out[1] - src[1] != count:
                _die(f"Pad {n.name!r} asks for {count} channels on the end of a "
                     f"{src[1]}-channel tensor, which is {src[1] + count}, but "
                     f"the graph's output shape says {out[1]}")
            claim(n, add({"op": "pad_c", "count": int(count),
                          "out": [int(x) for x in out],
                          "inputs": [idx_of(n.input[0], n, "input")]}, n))
            continue

        if op == "Resize":
            mode = _attr(n, "mode", "nearest")
            if mode != "linear":
                _die(f"Resize {n.name!r} has mode {mode!r}; the FPN neck's "
                     f"upsampling is bilinear and a nearest one would be a "
                     f"different graph")
            ctm = _attr(n, "coordinate_transformation_mode", "half_pixel")
            if ctm != "half_pixel":
                _die(f"Resize {n.name!r} uses coordinate_transformation_mode "
                     f"{ctm!r}; half-pixel is the convention the weights were "
                     f"exported against")
            # ONNX gives a Resize EITHER `scales` (input 2) or `sizes` (input
            # 3), and which one is present is a property of the exporter, not of
            # the operator. These two graphs pass an EMPTY scales and give
            # absolute sizes, so reading input 2 unconditionally finds a
            # zero-length array. Both are accepted; the one actually present is
            # the one the op's scale is taken from, and the other is CHECKED
            # rather than assumed to agree.
            src = t.out_shape(n.input[0], n)
            out = t.out_shape(n.output[0], n)
            scales = t.init.get(n.input[2])
            sizes = t.init.get(n.input[3]) if len(n.input) > 3 else None
            if scales is not None and scales.size:
                scale = [float(scales[2]), float(scales[3])]
                if sizes is not None and sizes.size and (
                        int(sizes[2]) != int(round(src[2] * scale[0]))
                        or int(sizes[3]) != int(round(src[3] * scale[1]))):
                    _die(f"Resize {n.name!r} carries BOTH scales {scale} and "
                         f"sizes {sizes.ravel().tolist()}, and they disagree "
                         f"about the output size. Picking one would be picking "
                         f"which of two different graphs to run.")
                want = (int(round(src[2] * scale[0])),
                        int(round(src[3] * scale[1])))
            elif sizes is not None and sizes.size:
                # sizes is absolute; the scale is derived so the op can be a
                # pure geometric upsample, and a fractional ratio is a refusal
                # because the half-pixel bilinear sampling is only exact for one
                want = (int(sizes[2]), int(sizes[3]))
                scale = [want[0] / src[2], want[1] / src[3]]
                if want[0] % src[2] or want[1] % src[3]:
                    _die(f"Resize {n.name!r} goes {src[2]}x{src[3]} to "
                         f"{want[0]}x{want[1]}, a ratio of "
                         f"{want[0] / src[2]}:1. A non-integral upsample is a "
                         f"different sampling pattern, not a scale this op "
                         f"expresses.")
            else:
                _die(f"Resize {n.name!r} supplies neither a non-empty `scales` "
                     f"nor a non-empty `sizes`, so the output size is a "
                     f"constant this op does not have")
            if (out[2], out[3]) != want:
                _die(f"Resize {n.name!r} goes {src[2]}x{src[3]} to "
                     f"{out[2]}x{out[3]}, which is not the {want[0]}x{want[1]} "
                     f"its own operand asks for")
            claim(n, add({"op": "resize",
                          "scale": [float(scale[0]), float(scale[1])],
                          "out": [int(x) for x in out],
                          "inputs": [idx_of(n.input[0], n, "input")]}, n))
            continue

        if op == "Transpose":
            perm = _attr(n, "perm")
            src = t.out_shape(n.input[0], n)
            if perm == [0, 3, 1, 2] and src[3] == 3:
                # The graph's LEADING NHWC->NCHW, one per network: absorbed,
                # because the image arrives NHWC (both checkpoints take
                # 1x192x192x3 and 1x224x224x3) and the runtime writes NCHW
                # natively. Its operand is therefore the image, index -1.
                emitted[n.name] = -1
                tensor_idx[n.output[0]] = -1
                continue
            if perm == [0, 2, 3, 1]:
                # The palm head's NCHW->NHWC, one per level per regressor. NOT
                # absorbed into the body: `palm_head` traces
                # Concat -> Reshape -> Transpose -> Conv and reads the
                # convolution's tensor directly, building the NHWC axis order
                # the reshape wants itself. Marked -2 rather than -1 so that a
                # body node consuming it (which would mean the head is not the
                # only thing downstream) is refused by idx_of instead of being
                # silently rewired to the image.
                emitted[n.name] = -2
                tensor_idx[n.output[0]] = -2
                continue
            _die(f"Transpose {n.name!r} has perm {perm} on {src}. Only the "
                 f"leading NHWC->NCHW and the palm head's NCHW->NHWC are "
                 f"understood; a transpose anywhere else reorders elements in a "
                 f"way nothing else in this file describes.")
            continue

        if op in ("PRelu", "Clip"):
            continue                      # fused in pass 1
        if _head_ops(t, n):
            continue                      # absorbed into a head op
        _die(f"{t.name}: operator {op} ({n.name!r}) is not in this packer's "
             f"vocabulary")

    heads = build_heads(t, ops, names, emitted, tensor_idx, producer, conv_of,
                        weights, pfx)
    return ops, names, conv_of, weights, prelus, heads


def _rewire(t, node, act, idx, emitted, tensor_idx):
    """Point a fused activation's output at the op that owns it.

    The activation does not change a tensor's shape, so the convolution (or the
    add) and its epilogue have the same shape and every downstream resolution is
    the same number. This is the ONE place a dropped tensor's consumer is allowed
    to exist, which is what lets idx_of refuse every other one.
    """
    if act is None:
        return
    emitted[act.name] = idx
    for o in act.output:
        tensor_idx[o] = idx


def _head_ops(t, n):
    """Is this node absorbed into a head rather than emitted as a body node?"""
    return n.op_type in ("Reshape", "Concat", "Squeeze", "GlobalAveragePool",
                         "Gemm", "Sigmoid")


def build_heads(t, ops, names, emitted, tensor_idx, producer, conv_of,
                weights, pfx):
    """The two heads, as op-list entries.

    Each head is a separate node because each is a different mathematical object:
    the palm head is a reshape of four feature maps into SSD rows plus the anchor
    decode, and the landmark head is a global average plus four dense
    projections. Neither is "a convolution with extra steps".
    """
    if t.name == "palm":
        return _palm_head(t, ops, names, emitted, tensor_idx, conv_of)
    return _lm_head(t, ops, names, emitted, tensor_idx, conv_of, weights, pfx)


def _palm_head(t, ops, names, emitted, tensor_idx, conv_of):
    """The palm head, in the graph's OWN level order.

    Traced from the Concats rather than assumed, because getting the order wrong
    permutes the anchor table against the predictions -- boxes that are
    plausible and wrong, which is the failure mode this whole file is written
    against. Two things are read rather than guessed:

    * WHICH output is boxes and which is scores, from the graph's declared
      output shapes ([1, 2016, 18] against [1, 2016, 1]) and not from the order
      the outputs happen to be listed in;
    * the order of the pyramid levels within each, from the order the Concat
      takes its inputs. `palm_anchors()` is then handed THAT order, so the
      anchor table is a consequence of the graph rather than a table somebody
      has to keep in step with it by hand.
    """
    def level_from_concat(cn, what):
        levels = []
        for inp in cn.input:
            rs = t.by_out.get(inp)
            if rs is None or rs.op_type != "Reshape":
                _die(f"palm head: the {what} Concat {cn.name!r} takes "
                     f"{inp!r}, produced by "
                     f"{rs.op_type if rs else 'nothing'}, but this packer reads "
                     f"the head through Reshape->Transpose->Conv")
            tr = t.by_out.get(rs.input[0])
            if tr is None or tr.op_type != "Transpose":
                _die(f"palm head: Reshape {rs.name!r} does not take a Transpose")
            if _attr(tr, "perm") != [0, 2, 3, 1]:
                _die(f"palm head: Transpose {tr.name!r} has perm "
                     f"{_attr(tr, 'perm')}, not the NCHW->NHWC the reshape's "
                     f"axis order assumes")
            conv = t.by_out.get(tr.input[0])
            if conv is None or conv.op_type != "Conv":
                _die(f"palm head: Transpose {tr.name!r} does not take a "
                     f"Convolution, so the head is not a per-level regressor")
            out = t.out_shape(conv.output[0], conv)
            # The reshape's own TARGET is the ground truth for the level's shape
            # and row count, and it is checked against the feature map rather
            # than assumed to follow from it -- the reshape is where the
            # anchor-per-cell count actually lives.
            want = t.out_shape(inp, rs)
            if len(want) != 3:
                _die(f"palm head: the reshape producing {inp!r} makes {want}, "
                     f"which is not a rows x terms table")
            levels.append({"op": emitted[conv.name], "c": int(out[1]),
                           "h": int(out[2]), "w": int(out[3]),
                           "rows": int(want[1]), "terms": int(want[2])})
        return levels

    by_out_node = {}
    for vi in t.graph.output:
        cn = t.by_out.get(vi.name)
        if cn is None or cn.op_type != "Concat":
            _die(f"palm head: the graph output {vi.name!r} is produced by "
                 f"{cn.op_type if cn else 'nothing'}, not by a Concat, so this "
                 f"is not a per-level SSD head")
        by_out_node[int(t.out_shape(vi.name, cn)[-1])] = cn
    if PALM_ROW not in by_out_node or 1 not in by_out_node:
        _die(f"palm head: the graph outputs have trailing dimensions "
             f"{sorted(by_out_node)}, which is neither a {PALM_ROW}-term box "
             f"table nor a 1-term score column. The {BOX_TERMS} box and "
             f"{PALM_LM_TERMS} landmark terms this packer splits a row into do "
             f"not describe this head.")

    boxes = level_from_concat(by_out_node[PALM_ROW], "box")
    scores = level_from_concat(by_out_node[1], "score")
    if len(boxes) != len(scores):
        _die(f"palm head: {len(boxes)} box levels against {len(scores)} score "
             f"levels; the two heads assemble different pyramids")
    for vi in t.graph.output:
        shape = t.shape[vi.name]
        table = boxes if int(shape[-1]) == PALM_ROW else scores
        rows = sum(x["rows"] for x in table)
        if int(shape[1]) != rows:
            _die(f"palm head: the graph output {vi.name!r} has {shape[1]} rows "
                 f"but its levels assemble {rows}. The anchor table and the "
                 f"predictions would be indexed by different things.")

    lv = []
    for b, s in zip(boxes, scores):
        if (b["h"], b["w"]) != (s["h"], s["w"]):
            _die(f"palm head: a box level {b['h']}x{b['w']} has a score level "
                 f"{s['h']}x{s['w']}")
        if b["c"] != s["c"] * PALM_ROW or b["terms"] != PALM_ROW:
            _die(f"palm head: {b['c']} box channels against {s['c']} score "
                 f"channels, and a {b['terms']}-term row, is not the "
                 f"{PALM_ROW}:1 ratio of {BOX_TERMS} box terms to "
                 f"{PALM_LM_TERMS} landmark terms")
        if b["rows"] != s["rows"] or b["rows"] != s["c"] * s["h"] * s["w"]:
            _die(f"palm head: a {s['h']}x{s['w']} level with {s['c']} anchors "
                 f"per cell should produce {s['c'] * s['h'] * s['w']} rows; its "
                 f"reshape asks for {b['rows']}")
        lv.append({"box": b["op"], "score": s["op"], "h": b["h"], "w": b["w"],
                   "anchors_per_cell": s["c"], "rows": b["rows"]})

    # The anchor table is a CONSEQUENCE of the levels above, not a constant
    # beside them. Nothing downstream has to keep the two in step by hand.
    total = sum(x["rows"] for x in lv)
    anchors = palm_anchors([(x["h"], x["w"], x["anchors_per_cell"]) for x in lv])
    if anchors.shape != (total, 2):
        _die(f"palm head: levels give {total} rows but {anchors.shape[0]} anchors "
             f"were generated from them")
    return {"palm_head": lv}


def _through_fusion(t, tensor_name):
    """The Node that really produced `tensor_name`, stepping over fused
    activations.

    Needed because a head can sit behind an activation: this checkpoint's global
    average pool consumes the relu6 of the last convolution, not the convolution.
    Resolving it by stepping back is not a guess -- the fusion is a rewrite this
    packer itself performed a few lines earlier, so the owner is recorded rather
    than inferred. Returns None if nothing produces the value.
    """
    name = tensor_name
    seen = set()
    while True:
        node = t.by_out.get(name)
        if node is None or node.op_type not in ("PRelu", "Clip"):
            return node
        if name in seen:
            _die(f"fusion chain from {tensor_name!r} loops at {name!r}")
        seen.add(name)
        name = node.input[0]


def _lm_head(t, ops, names, emitted, tensor_idx, conv_of, weights, pfx):
    """The landmark head: global average, four dense projections, two sigmoids.

    Worth spelling out because it is NOT a heatmap network once exported. The
    checkpoint's own graph has already folded the per-landmark argmax into it:
    the outputs are [1, 63] coordinates (21 x, y, z) and the same again for the
    metric world set, plus two scalars. So there is no heatmap to decode and no
    sub-pixel offset to add, and a runtime written for the usual two-stage
    landmark head would go looking for tensors this graph does not contain.

    The two sigmoids are NOT free-floating activations: each one is the graph's
    output, so which projection is a probability and which is a coordinate is a
    property of the OUTPUT ORDER, and it is recorded per projection here rather
    than left to the runtime to infer from a width.
    """
    gap = [n for n in t.nodes if n.op_type == "GlobalAveragePool"]
    if len(gap) != 1:
        _die(f"landmark head: {len(gap)} GlobalAveragePool nodes, want 1")
    # The pool consumes this checkpoint's LAST RELU6, not the convolution under
    # it, so the owner has to be resolved through the fusion rather than read
    # off the node itself.
    conv = _through_fusion(t, gap[0].input[0])
    if conv is None or conv.op_type != "Conv":
        _die(f"landmark head: the pool consumes "
             f"{conv.op_type if conv else 'nothing'}, not a convolution")
    feats = t.out_shape(conv.output[0], conv)
    pooled, fh, fw = int(feats[1]), int(feats[2]), int(feats[3])

    gemms = [n for n in t.nodes if n.op_type == "Gemm"]
    projs = []
    for g in gemms:
        W = t.init.get(g.input[1])
        B = t.init.get(g.input[2])
        if W is None or B is None:
            _die(f"landmark head: Gemm {g.name!r} does not have both a weight "
                 f"and a bias initializer; an unaugmented projection would need "
                 f"a different op")
        if W.shape[0] != pooled or int(W.shape[1]) not in (LM_FLAT, 1):
            _die(f"landmark head: Gemm {g.name!r} has weight {W.shape}, which is "
                 f"neither ({pooled}, {LM_FLAT}) nor ({pooled}, 1)")
        k = len(projs)
        projs.append({"w": f"{pfx}.gemm.{k}.w", "b": f"{pfx}.gemm.{k}.b",
                      "n": int(W.shape[1])})
        weights[f"{pfx}.gemm.{k}.w"] = W.astype(np.float32)
        weights[f"{pfx}.gemm.{k}.b"] = B.astype(np.float32)
    if len(projs) != 4:
        _die(f"landmark head: {len(projs)} projections, want 4 -- two of "
             f"{LM_FLAT} landmark terms and two scalars")

    # Which projection is which OUTPUT, in the graph's own output order. Width
    # alone does not separate the two landmark sets, so the order has to be
    # recorded or the runtime would be free to swap them.
    outs = []
    for vi in t.graph.output:
        src = vi.name
        node = t.by_out.get(src)
        sig = False
        if node is not None and node.op_type == "Sigmoid":
            sig = True
            src = node.input[0]
            node = t.by_out.get(src)
        if node is None or node.op_type != "Gemm":
            _die(f"landmark head: the graph output {vi.name!r} comes from "
                 f"{node.op_type if node else 'nothing'}, expected a projection "
                 f"possibly through a sigmoid")
        width = int(t.out_shape(vi.name, node)[-1])
        idx0 = next((i for i, g in enumerate(gemms)
                     if g.output[0] == src), None)
        if idx0 is None:
            _die(f"landmark head: cannot place the output {vi.name!r} among the "
                 f"projections")
        if width != projs[idx0]["n"]:
            _die(f"landmark head: the graph declares {vi.name!r} as "
                 f"{width} wide, but its projection produces "
                 f"{projs[idx0]['n']}")
        outs.append({"n": width, "sigmoid": sig, "proj": idx0})

    return {"lm_head": {"pool": emitted[conv.name], "features": pooled,
                        "height": fh, "width": fw, "projs": projs,
                        "outputs": outs}}



def mac_split(t, conv_of):
    """MACs the array could take (dense) and MACs it cannot (depthwise)."""
    dense = dw = 0
    shapes = {}
    for n in t.nodes:
        if n.op_type != "Conv":
            continue
        w = t.init[n.input[1]]
        cout, cin_g, kh, kw = w.shape
        groups = _attr(n, "group", 1)
        out = t.shape.get(n.output[0])
        if out is None or any(not isinstance(x, int) for x in out):
            continue
        # `cin_g` is already the PER-GROUP input channel count, so it is used
        # as-is: dividing it by groups again would make every depthwise
        # convolution 1 // 32 = 0 MAC and report the host share as zero.
        m = out[2] * out[3] * cout * cin_g * kh * kw
        if groups == 1:
            dense += m
        else:
            dw += m
        key = f"{int(out[1])}x{int(out[2])}x{int(out[3])}"
        shapes[key] = shapes.get(key, 0) + 1
    return dense, dw, shapes


def pack_hands(palm_onnx, lm_onnx, out_path, dry_run=False):
    models = {}
    tp = trace("palm", onnx.load(str(palm_onnx)))
    tl = trace("landmark", onnx.load(str(lm_onnx)))
    ops_p, names_p, conv_p, w_p, prelu_p, head_p = emit_ops(tp, "palm")
    ops_l, names_l, conv_l, w_l, prelu_l, head_l = emit_ops(tl, "lm")

    a_palm, dw_palm, shapes_palm = mac_split(tp, conv_p)
    a_lm, dw_lm, shapes_lm = mac_split(tl, conv_l)

    # The anchor table is generated FROM the head's level list, in the head's
    # own order, so it cannot drift out of step with the predictions it indexes.
    levels = [(l["h"], l["w"], l["anchors_per_cell"])
              for l in head_p["palm_head"]]
    anchors = palm_anchors(levels)

    if dry_run:
        for tag, ops, conv, w, a, dw, shapes in (
                ("palm", ops_p, conv_p, w_p, a_palm, dw_palm, shapes_palm),
                ("landmark", ops_l, conv_l, w_l, a_lm, dw_lm, shapes_lm)):
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
            print(f"  MAC   array {a / 1e6:.1f} M, host(depthwise) {dw / 1e6:.1f} M"
                  f"  ({100.0 * dw / (a + dw):.1f}% on the host)")
        print(f"anchors  {anchors.shape[0]} generated from levels {levels}")
        print(f"palm head levels {[(l['h'], l['w'], l['anchors_per_cell'], l['rows']) for l in head_p['palm_head']]}")
        print(f"lm head   pool {head_l['lm_head']['features']} -> "
              f"{[p['n'] for p in head_l['lm_head']['projs']]}")
        return

    config = {
        "arch": ARCH_STRING,
        "kind": "hands",
        "palm_input_size": PALM_SIZE,
        "lm_input_size": LM_SIZE,
        "num_landmarks": NUM_LANDMARKS,
        "palm_num_anchors": int(anchors.shape[0]),
        "palm_anchor_levels": json.dumps([list(l) for l in levels],
                                         separators=(",", ":")),
        "palm_row_terms": PALM_ROW,
        "palm_lm_terms": PALM_LM_TERMS,
        "palm_lm_wrist": PALM_LM_WRIST,
        "palm_lm_middle_base": PALM_LM_MIDDLE_BASE,
        "palm_pre_shift": list(PALM_PRE_SHIFT),
        "palm_pre_enlarge": PALM_PRE_ENLARGE,
        "palm_shift": list(PALM_SHIFT),
        "palm_enlarge": PALM_ENLARGE,
        "hand_shift": list(HAND_SHIFT),
        "hand_enlarge": HAND_ENLARGE,
        "score_threshold": SCORE_THR,
        "nms_threshold": NMS_THR,
        "top_k": TOP_K,
        # Identity normalisation, WRITTEN rather than assumed: both front ends
        # divide the resized crop by 255 and stop, so the container says so.
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "palm_num_convs": len(conv_p),
        "lm_num_convs": len(conv_l),
        "array_mac": int(a_palm + a_lm),
        "host_mac": int(dw_palm + dw_lm),
        "palm_graph": json.dumps(ops_p, separators=(",", ":")),
        "lm_graph": json.dumps(ops_l, separators=(",", ":")),
        "palm_graph_nodes": json.dumps(names_p, separators=(",", ":")),
        "lm_graph_nodes": json.dumps(names_l, separators=(",", ":")),
        "palm_head": json.dumps(head_p["palm_head"], separators=(",", ":")),
        "lm_head": json.dumps(head_l["lm_head"], separators=(",", ":")),
    }

    w = Writer(config, arch=ARCH_MEDIAPIPE_HANDS_PALM_SSD_LM_HEATMAP)
    # Prefixed names are disjoint by construction, so these two merges are a
    # union rather than a collision -- but checked, because "by construction"
    # is exactly the kind of claim that stops being true when someone adds a
    # third network.
    both = {**w_p, **w_l, **prelu_p, **prelu_l}
    check_disjoint(w_p, w_l, "convolution weights")
    check_disjoint(prelu_p, prelu_l, "prelu slopes")
    for name, arr in sorted(both.items()):
        w.add(name, arr, "F32",
              "conv" if ".conv." in name else "prelu", list(arr.shape))
    w.add("palm_anchors", anchors.astype(np.float32), "F32", "anchors",
          list(anchors.shape))
    w.write(out_path)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({size / 1e6:.1f} MB, arch={ARCH_STRING})")
    print(f"  {len({**w_p, **w_l})} weight tensors, "
          f"{len({**prelu_p, **prelu_l})} slopes, "
          f"{anchors.shape[0]} anchors")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("palm_onnx")
    ap.add_argument("lm_onnx")
    ap.add_argument("out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pack_hands(a.palm_onnx, a.lm_onnx, a.out, dry_run=a.dry_run)


if __name__ == "__main__":
    main()