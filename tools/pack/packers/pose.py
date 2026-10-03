# NpuEmbeddings -- pack a YOLOv8-pose checkpoint into an arch=6 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS FILE IS
# =================
# The third family, and the first whose structure is not recoverable from three
# numbers. A BERT container is a shape and a weight list, because the runtime
# knows what a transformer layer is. A YOLOv8-pose container is A SHAPE AND A
# GRAPH: which convolutions there are, in which order, what feeds what, where the
# SiLUs are, how the neck joins the pyramid. That is the model, and a packer that
# did not emit it would be a packer that invented a network.
#
# So this file TRACES the ONNX and emits an explicit op list. `graph` in the
# container is what the runtime walks, and runtime/src/pose/geometry.cpp
# typechecks it: operand counts, channel counts, resolutions, and the head's
# arity. A graph that would not execute is refused at LOAD time rather than
# producing a plausible pose.
#
# WHAT RUNS WHERE -- stated here because the split is the design
# ==============================================================
#   ON THE HOST, by default, and that is MEASURED rather than preferred:
#     the 72 graph convolutions. 150 ms on 16 CPU threads at 640x640, against
#     290 ms as dispatched GEMMs on this array. Three structural reasons:
#       * a design's N must be a multiple of tile_n*cols = 128 on npu1 and most
#         of this network's N are <= 64, so 4.59 GMAC of useful work becomes
#         11.95 GMAC of dispatched work -- 2.6x waste before anything runs;
#       * a dispatch is M <= 1024 rows and the stem alone has M = 102400, which
#         is 100 of the network's 436 dispatches;
#       * those 436 dispatches cost 660 us each inside conv(), of which 140 us
#         is the device's own GEMM -- the other 520 us is this host's per-chunk A
#         repack and the C transpose, which no tile size removes.
#     Per layer, 19 of the 72 are faster on the array than on the host, and an
#     oracle putting each layer on its faster side measured 146 ms against the
#     host's 150 -- 2.6%.
#   ON THE HOST, unavoidably: the front end (decode, letterbox, normalise), the
#     SiLUs (fused into the convolution's epilogue -- elementwise on that GEMM's
#     own output), the concat/add/maxpool/upsample, and the head.
#   ON THE ARRAY, only under --npu: the same 72 graph convolutions as im2col plus
#     dispatched GEMMs, with the B panels pre-tiled and a stream named per
#     distinct padded (K, N).
#
# THE BOUNDARY, AND WHY IT IS WHERE IT IS
# =======================================
# The exported graph's last ~30 nodes are not convolutions: a Softmax, a weighted
# sum over 16 bins, a Sub/Add/Div turning left-top-right-bottom distances into a
# centre and a half-extent, a meshgrid of anchors, a multiply by the strides, two
# more Sigmoids, and a pile of Reshape/Split/Slice/Transpose moving numbers
# around. None of it is a GEMM, so the conv engine cannot run it, and turning
# each piece into an op kind would mean a softmax and a weighted sum per level
# with nothing to check them against but a careful reading.
#
# So the boundary is drawn AT THAT ARITHMETIC and made EXPLICIT. The packer
# traces which nodes are convolutions and which are not, keeps the former in the
# graph, and hands the latter's inputs to a single `detect` op. The runtime
# implements that op once (Network::head) and it emits ONE CANONICAL tensor: xyxy
# boxes in letterbox pixels, class score and keypoint visibility as
# probabilities, keypoints in letterbox pixels. decode.cpp then reads a canonical
# tensor and knows nothing about what the exporter chose -- which is what stops
# the packer and the reader from ever disagreeing about a convention.
#
# tools/verify/verify_pose.py is what makes that claim checkable: it implements
# the network AND the whole tail independently, in NumPy, by reading the ONNX
# graph, and compares keypoints and boxes against this runtime's output on a
# real photograph.
#
# Env: numpy + onnx. Both build-time only, as this whole directory is.

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import conv_quant                                        # noqa: E402
from npue import ARCH_YOLOV8_POSE_C2F_SILU_DFL, Writer  # noqa: E402

# Kept as literals and compared against the C++ header by
# tools/verify/verify_pose_geometry.py, because the two must agree and a test
# that reads both is the only thing that can say they do.
ARCH_STRING = "yolov8_pose_c2f_silu_dfl"
NUM_KEYPOINTS = 17
DFL_BINS = 16
NUM_CLASSES = 1

# The node types this packer maps to a graph op. Every other type is
# either head plumbing (PLUMBING) or a refusal -- see _executable.
GRAPH_OPS = {"Conv", "Mul", "Sigmoid", "Add", "Concat", "Split",
              "MaxPool", "Resize"}

# Node types allowed ONLY inside the head. Listing them is what turns "a node I
# did not understand" from a silent drop into a refusal by name -- a dropped node
# is a network that computes something else and returns a plausible pose.
PLUMBING = {
    "Split", "Reshape", "Slice", "Transpose", "Gather", "Shape", "Softmax",
    "Sigmoid", "Div", "Sub", "Constant", "Identity", "Cast", "Flatten",
    "Gelu", "Tanh", "Exp", "Sqrt", "ReduceMax", "Range", "Expand",
    "Unsqueeze", "Squeeze", "Max", "Sum", "Where", "Equal", "Not", "And",
    "Or", "Pow", "Mod", "Floor", "Round", "Clip", "Neg", "Abs", "Log",
}

# -- the header the runtime compares against -----------------------------------
REQUIRED_CONFIG_KEYS = (
    "arch", "input_size", "num_keypoints", "num_classes", "dfl_bins",
    "detect_cout", "detect_h", "detect_w", "num_levels", "strides", "grid",
    "image_mean", "image_std", "head_box_format", "head_score",
    "head_keypoint_visibility", "num_convs", "head_dfl_conv", "bias_free",
    "head_box_grid_offset", "head_kpt_grid_offset",
    "conv_weight_dtype", "conv_int4_group",
    "graph",
)


def _die(msg):
    raise SystemExit(f"pose packer: {msg}")


def _chw(vinfo, name, what):
    """[C, H, W] out of an ONNX value_info, or a refusal.

    Rank 4 is required for everything in the network's body, and the reason is
    worth stating: the runtime's tensors are [C, H, W] NCHW, so a rank-4 ONNX
    value is [1, C, H, W] with a batch of one. The head's plumbing reshapes to
    rank 2 and 3, and those values belong to the detect op -- which is precisely
    why the boundary is drawn where it is.
    """
    t = vinfo[name].type.tensor_type
    dims = [d.dim_value for d in t.shape.dim]
    if t.elem_type != onnx.TensorProto.FLOAT:
        _die(f"{what} ({name}) has element type {t.elem_type}, not float. This "
             f"packer stores F32 and the runtime's host kernels read F32; a "
             f"half-precision graph would need its own storage and is not "
             f"approximated here.")
    if len(dims) != 4 or dims[0] != 1:
        _die(f"{what} ({name}) is {dims}, not [1, C, H, W]. Every tensor in the "
             f"network's body is rank 4 with a batch of one; a value of another "
             f"rank is head plumbing and belongs to the detect op, not to a "
             f"graph node.")
    if any(d <= 0 for d in dims[1:]):
        _die(f"{what} ({name}) is {dims} with a dynamic extent. Every spatial "
             f"dimension has to be a number: the runtime allocates each tensor up "
             f"front, and these resolutions are what the head's cell-to-pixel "
             f"map is checked against.")
    return dims[1], dims[2], dims[3]


def _ancestors(by_out, roots):
    """The NAMES of the nodes that produce `roots`, transitively.

    Node names and not tensor names: every caller below asks "is this node part
    of the network", and a set of tensors would make that question look like it
    had an answer. Returning the producer's name also keeps ONNX's unnamed nodes
    -- which export as empty strings, and which a name-keyed set would collapse
    into one -- from silently dropping out of the set.
    """
    seen, stack = set(), list(roots)
    while stack:
        v = stack.pop()
        n = by_out.get(v)
        if n is None or n.name in seen:
            continue
        seen.add(n.name)
        stack.extend(n.input)
    return seen


def _topo(t, wanted):
    """`wanted` in ONNX's OWN node order, after verifying that order is usable.

    ONNX requires a graph's nodes to be in topological order, and every exporter
    honours that, so the exporter's order is both correct and deterministic --
    the emitted container is byte-identical across runs for the same checkpoint,
    and every cache keyed on the container's hash hits. A name-sorted order would
    also be deterministic and would be WRONG: node names carry no dependency
    information, and `/model.10/Resize` sorts before `/model.9/cv2/conv/Conv`
    while depending on it. Sorting names and hoping is how a graph that reads
    values it has not computed gets emitted; it fails at load, but only because
    the loader happened to be strict.

    Which is why the spec's guarantee is CHECKED rather than trusted: a
    producer that comes later means the exporter broke its own rule, and the
    only correct response is to say which node, not to re-sort and hope.
    """
    order, done = [], set()
    for name in t.node_order:
        if name not in wanted:
            continue
        n = t.by_name[name]
        for i in n.input:
            if not i:
                continue                    # an omitted optional input
            p = t.by_out.get(i)
            if p is None or p.name not in wanted:
                continue                    # an initializer, or folded away
            if p.name not in done:
                _die(f"{name} ({n.op_type}) reads {i}, which {p.name} "
                     f"produces, but {p.name} comes AFTER it in the ONNX node "
                     f"list. The ONNX spec requires a topologically ordered node "
                     f"list; re-sorting here would produce a different graph "
                     f"from a different exporter, and the graph below is only "
                     f"meaningful if it is the checkpoint's own order.")
        order.append(name)
        done.add(name)
    if len(order) != len(wanted):
        missing = sorted(wanted - set(order))
        _die(f"{len(missing)} placed node(s) are not in the ONNX node list at "
             f"all (e.g. {missing[:4]}), so this packer's graph and the "
             f"checkpoint's node list have diverged.")
    return order


# onnx.AttributeProto field types
_AT = {"float": 1, "int": 2, "string": 3, "tensor": 4, "graph": 5,
       "floats": 6, "ints": 7, "strings": 8}
_AT_KIND = {v: k for k, v in _AT.items()}


def _raw(a):
    """An attribute's value, read by the field it is actually stored in.

    ONNX changed the representation of several attributes between opsets -- and
    even WITHIN one opset, `Resize.mode` is an int enum at opset 11 and a string
    at opset 18 -- so reading a field that is not there returns a zero rather than
    an error. That is the failure worth guarding: a nearest-neighbour resample is
    mode 0, and so is an attribute that was read from the wrong field.
    """
    kind = _AT_KIND.get(a.type)
    if kind == "int":
        return a.i
    if kind == "float":
        return a.f
    if kind == "string":
        return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
    if kind == "ints":
        return list(a.ints)
    if kind == "floats":
        return list(a.floats)
    if kind == "strings":
        return [x.decode("utf-8") if isinstance(x, bytes) else x
                for x in a.strings]
    return None                    # tensor/graph: not a value this reads


def _attribute(n, kind, name):
    """The attribute `name`, required to be stored as `kind` ("i" or "ints").

    A type mismatch is an error rather than a coercion. Coercing a string "1"
    into an int works on the attributes this packer reads today and silently
    reads a float 0.0 as int 0 on the ones an older exporter wrote, and an
    unnoticed wrong 0 is a wrong axis or a wrong split.
    """
    for a in n.attribute:
        if a.name != name:
            continue
        if _AT_KIND.get(a.type) != ("ints" if kind == "ints" else "int"):
            raise SystemExit(
                f"pose packer: {n.name} carries \"{name}\" as "
                f"{_AT_KIND.get(a.type)}, not {kind}. This reader does not "
                f"coerce between representations, because the two "
                f"representations of this attribute have different value "
                f"spaces and a coercion that happens to work on one export will "
                f"read a zero on the next.")
        return list(a.ints) if kind == "ints" else a.i
    return None


# Resize's mode, as an int enum (opset 11) and as a string (opset 18). The two
# agree on 0 = nearest, which is also the DEFAULT when the attribute is absent.
_RESIZE_MODE = {0: "nearest", 1: "linear", 2: "cubic", 3: "area",
                4: "nearest_cubic", 5: "linear_cubic"}


def _resize_mode(n):
    for a in n.attribute:
        if a.name != "mode":
            continue
        kind = _AT_KIND.get(a.type)
        if kind == "int":
            return _RESIZE_MODE.get(a.i, f"enum {a.i}")
        if kind == "string":
            return a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
        raise SystemExit(
            f"pose packer: {n.name} carries \"mode\" as {kind}, which this "
            f"reader does not interpret. Guessing which resample a graph meant "
            f"is how a keypoint moves by half a pixel on one pyramid level.")
    return "nearest"          # the ONNX default


class Traced:
    """Everything the emitter needs, plus every refusal the tracer already made.

    A class rather than a dict so that a missing field is a compile error rather
    than a KeyError at 200 lines into the emitter.
    """

    def __init__(self):
        self.model = None
        self.by_name = {}        # node name -> NodeProto
        self.by_out = {}         # tensor name -> NodeProto
        self.vals = {}           # tensor name -> ndarray, for initializers only
        self.vinfo = {}          # tensor name -> ValueInfoProto
        self.src = ""
        self.dst = ""
        self.input = (0, 0, 0)
        # [channels, cells, 1] -- the runtime's head tensor, NCHW with a 1x1
        # spatial extent. Kept because the config writes detect_cout/h/w and
        # read_geometry cross-checks them against the graph it walks, so a
        # mismatch between "what the ONNX emits" and "what the head emits" is
        # caught rather than assumed away.
        self.output = (0, 0, 1)
        self.cells = 0
        self.impl = set()
        self.plumbing = set()
        self.consumers = {}       # node name -> [consumer node names]
        self.node_order = []       # every node name, in the ONNX list's order
        self.anc_nodes = []        # the ancestry, in ONNX's own order
        self.stop_reason = {}     # plumbing node -> (bad input, type)
        self.terminals = {}      # node name -> NodeProto
        self.graph_nodes = []
        self.dfl = ""
        self.strides = []
        self.grid = []
        self.box_grid = None
        self.kpt_grid = None
        self.box_grid_offset = 0.0
        self.kpt_grid_offset = 0.0


def trace(model):
    t = Traced()
    t.model = model
    g = model.graph
    for n in g.node:
        t.by_name[n.name] = n
        t.node_order.append(n.name)
        for o in n.output:
            t.by_out[o] = n
        for i in n.input:
            t.consumers.setdefault(i, []).append(n.name)
    # Initializers AND Constant nodes. The second is here for one reason: the
    # head's per-anchor stride vector is a Constant, not an initializer, and it
    # is the only place the pyramid's strides appear in the graph as DATA.
    # Reading them from there instead of assuming 8/16/32 is what turns
    # _trace_strides_and_grid into a check.
    t.vals = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type != "Constant" or not n.output:
            continue
        attr = next((a for a in n.attribute if a.name == "value"), None)
        if attr is None:
            continue            # a Constant carrying a scalar/expression, rare
        try:
            t.vals[n.output[0]] = numpy_helper.to_array(attr.t)
        except Exception:       # noqa: BLE001 -- a non-tensor constant is fine
            continue

    if len(g.input) != 1:
        _die(f"the graph has {len(g.input)} inputs, not 1. This architecture "
             f"takes one letterboxed image tensor and nothing else.")
    if len(g.output) != 1:
        _die(f"the graph has {len(g.output)} outputs, not 1. A pose export "
             f"emits one [1, 4 + {NUM_CLASSES} + {NUM_KEYPOINTS}*3, cells] "
             f"tensor. A two-output one is the RAW head, with the DFL and the box "
             f"decode NOT applied, and this runtime does not run a second DFL on "
             f"a distribution that has already been summed out.")
    t.src, t.dst = g.input[0].name, g.output[0].name

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as e:  # noqa: BLE001 -- the message is the payload
        _die(f"ONNX shape inference failed: {e}")
    t.vinfo = {v.name: v for v in
               list(inferred.graph.value_info) + list(g.input) + list(g.output)}

    cin, hin, win = _chw(t.vinfo, t.src, "the graph input")
    # The OUTPUT is rank 3, unlike every tensor in the body: the exporter has
    # already concatenated the pyramid's cells into one axis, so there is no H and
    # W left. It is checked here rather than through _chw precisely because that
    # difference is real and load-bearing -- the head re-derives the cells from
    # `grid`, and the total is compared against this number as an independent
    # check rather than assumed.
    tout = t.vinfo[t.dst].type.tensor_type
    odims = [d.dim_value for d in tout.shape.dim]
    if tout.elem_type != onnx.TensorProto.FLOAT or len(odims) != 3 or odims[0] != 1:
        _die(f"the graph output is {odims}, not [1, C, cells]. A pose export's "
             f"output has the pyramid's cells already concatenated into one "
             f"axis; a rank-4 output would mean the exporter kept a [H, W] layout "
             f"that this head does not read.")
    cout, cells = odims[1], odims[2]
    if cells <= 0:
        _die(f"the graph output has {cells} cells, which is not a number. Every "
             f"spatial extent has to be static: the head reads the tensor cell by "
             f"cell and the pyramid's resolutions are what its cell-to-pixel map "
             f"is checked against.")
    if cin != 3:
        _die(f"the input has {cin} channels, not 3. This front end produces RGB.")
    if hin != win:
        _die(f"the input is {hin}x{win}, not square. The letterbox pads to a "
             f"SQUARE so one uniform scale and one offset are invertible; a "
             f"non-square canvas needs two scales, and the network was trained on "
             f"people of one shape.")
    want = 4 + NUM_CLASSES + NUM_KEYPOINTS * 3
    if cout != want:
        _die(f"the output has {cout} channels; this architecture's head emits "
             f"4 box + {NUM_CLASSES} class + {NUM_KEYPOINTS} x 3 keypoints = "
             f"{want}. A different width is a different head, and the decode "
             f"would read past the end of a row into the next anchor's.")
    t.input = (cin, hin, win)
    t.output = (cout, cells, 1)
    t.cells = cells

    # -- SPLIT THE ANCESTRY INTO THE GRAPH AND THE HEAD'S PLUMBING
    #
    # STRUCTURALLY, by walking FORWARD from the input rather than by op type.
    # The type test does not work, and the reason is worth recording because it
    # is the trap in this packer: the head is not a different SET of types, it
    # is the same types in the wrong PLACE. The head multiplies by the strides
    # and adds a meshgrid -- both Mul and both Add, the very two types that make
    # up the convolutional body. Sorting by type would put a stride multiply
    # into the graph as if it were a residual, and the graph would refuse to
    # run it (its two operands have different shapes), so the failure would be
    # visible; but a type sort that happened to typecheck would compute a
    # different network.
    #
    # So the rule is positional and it is one sentence: a node belongs to the
    # GRAPH exactly when every value it reads was produced by the graph. Start at
    # the input, follow consumers, and stop at the first node that reads
    # something the graph did not produce. Everything from there to output0 is
    # the head's plumbing, and the last graph node before each stop is a
    # TERMINAL -- a tensor the head reads.
    #
    # The stop is what makes the SiLU work: `Mul(x, Sigmoid(x))` is a graph op,
    # and so is `x * stride_vector` when both operands come from the graph. In
    # the body the first form holds; in the head the second one's constant comes
    # from plumbing, so it stops there. That is checked rather than assumed, and
    # emit_ops refuses any Mul in the graph that is not the activation pattern.
    anc = _ancestors(t.by_out, [t.dst])
    nodes = [t.by_name[x] for x in anc]
    t.anc_nodes = nodes
    graph_produced = {t.src}
    undecided = [n for n in nodes]
    rounds = 0

    # A FIXPOINT, not a single sweep, and the reason is a specific bug this
    # comment records. The C2f block joins two branches: a Split feeding one
    # branch, and a convolution in that branch feeding a residual Add whose
    # OTHER operand is the Split's second output. A single sweep that enqueues
    # a node the moment any of its producers is placed reaches that Add as soon
    # as the Split is placed -- before the branch that produces its other
    # operand has been walked -- and declares it head plumbing, because at that
    # instant one of its two inputs is not yet accounted for.
    #
    # That failure is invisible in the output and catastrophic in the graph: the
    # C2f block's residual join vanishes, the channel count after it is wrong,
    # and the network that runs is not the checkpoint.
    #
    # So a node is decided only when EVERY value it reads has an accounting:
    # either the graph produced it, or the graph will never. Each round decides
    # everything that became decidable, and the loop ends when a round decides
    # nothing new. What is left over is plumbing by elimination, and the loop's
    # `rounds` is printed in the dry run so a future change to this rule that
    # silently needs more rounds than it should is visible.
    while undecided:
        rounds += 1
        if rounds > len(nodes) + 2:
            _die(f"the graph/body split did not converge after {rounds} rounds "
                 f"over {len(nodes)} nodes. The walk decides a node only when "
                 f"every value it reads is accounted for, so a round that "
                 f"decides nothing means the remainder is unreachable from the "
                 f"input -- which no export produces.")
        progressed, still = False, []
        for n in undecided:
            silu = silu_parts(t.by_out, n) if n.op_type == "Mul" else None
            pending = False
            for i in n.input:
                # ONNX writes an omitted OPTIONAL input as the empty name. The
                # neck's Resize has one (the `roi` operand), and reading it as a
                # tensor whose producer does not exist would make the node
                # permanently undecidable and drop the whole rest of the network
                # into the head by elimination.
                if not i or i in t.vals or i == t.src:
                    continue     # absent, a weight, a Constant, or the input
                if silu is not None and i == t.by_name[silu[1]].output[0]:
                    continue     # the activation's own Sigmoid, decided with it
                if i not in graph_produced:
                    pending = True
                    break
            verdict = _executable(t, n)
            if verdict is None:
                _die(f"node {n.name} ({n.op_type}) is in the network's ancestry "
                     f"and is neither an operation the graph engine implements "
                     f"({sorted(GRAPH_OPS)}) nor one of this architecture's "
                     f"head operations ({sorted(PLUMBING)}). This packer refuses "
                     f"it by name rather than skipping it, because a skipped node "
                     f"is a network that computes something else and returns a "
                     f"pose.")
            if pending:
                still.append(n)
                continue
            if verdict:
                t.impl.add(n.name)
                graph_produced.update(n.output)
            else:
                t.plumbing.add(n.name)
            progressed = True
        undecided = still
        if not progressed:
            break
    for n in undecided:
        # By elimination: every value it reads is produced by plumbing, so it is
        # head arithmetic. Recorded so the dry run can report how big the head
        # is, which is the number that says whether this architecture's head
        # really is the small part the design assumed.
        t.plumbing.add(n.name)
        t.stop_reason[n.name] = (None, n.op_type)

    if not t.impl:
        _die("nothing in the network's ancestry is an operation this packer can "
             "place. Is this actually a YOLOv8-pose export?")
    if not t.plumbing:
        _die("the network's ancestry contains no head plumbing at all. This "
             "packer's detect op exists to replace a DFL plus the exporter's box "
             "decode; a graph whose last op already produces the output tensor is "
             "a different architecture and is not guessed at.")

    # -- the terminals: graph nodes whose OUTPUT is read by plumbing.
    #
    # Derived structurally rather than by name, so a re-export that renames every
    # node still packs. These are exactly the tensors the head reads, and
    # finding them structurally is the whole reason the boundary can be
    # automatic.
    for pn in t.plumbing:
        for i in t.by_name[pn].input:
            n = t.by_out.get(i)
            if n is not None and n.name in t.impl:
                t.terminals.setdefault(n.name, n)
    if not t.terminals:
        _die("nothing the head reads is a graph node. The plumbing and the "
             "network are disconnected, which no export produces.")

    t.graph_nodes = _topo(t, t.impl)

    # -- the DFL's projection: a CONVOLUTION that the walk classified as
    # plumbing, because its input is the distribution's Softmax.
    #
    # This is the one conv the graph engine cannot run -- not because of its
    # shape but because of what precedes it -- and it is handed to the head by
    # INDEX instead. read_geometry checks its shape and its bias, and
    # Network::head folds it in exactly:
    #
    #     sum_bin w[0][bin] * softmax(logits)[bin] + b
    #       = w[0] . E[bin] + b          (because the softmax sums to 1)
    #
    # so it is a per-bin SCALE on the expectation rather than a second pass over
    # the distribution. Algebra, not approximation, and the reason the head needs
    # a weight index rather than a graph node.
    #
    # Any OTHER convolution inside the plumbing is a refusal, because there is no
    # second op kind in Network::head for it and silently dropping it would drop
    # the arithmetic it performs.
    found = []
    for pn in t.plumbing:
        n = t.by_name[pn]
        if n.op_type != "Conv":
            continue
        w = t.vals.get(n.input[1])
        if w is not None and w.ndim == 4 and list(w.shape) == [1, DFL_BINS, 1, 1]:
            found.append(n.name)
        else:
            _die(f"{pn} is a convolution inside the head's plumbing, of shape "
                 f"{None if w is None else list(w.shape)}, and it is not the "
                 f"DFL's [1, {DFL_BINS}, 1, 1] projection. This build runs "
                 f"exactly one convolution in its head op; a second one is "
                 f"arithmetic this runtime does not implement, and dropping it "
                 f"would return a pose from a network that is not the one "
                 f"packed.")
    if not found:
        _die(f"the head's plumbing contains no convolution of shape "
             f"[1, {DFL_BINS}, 1, 1]. That is the DFL's projection from the "
             f"{DFL_BINS} bins to one distance, and without it the head has no "
             f"scale for the expectation it takes -- so the boxes would be in "
             f"the wrong units by whatever factor it should have carried. A "
             f"checkpoint with a different bin count is a different head.")
    if len(found) > 1:
        _die(f"the head's plumbing contains {len(found)} convolutions of shape "
             f"[1, {DFL_BINS}, 1, 1] ({', '.join(found)}). The head folds "
             f"exactly one into the distribution's expectation and would use the "
             f"wrong one's weights for the other.")
    t.dfl = found[0]

    _trace_strides_and_grid(t)
    return t


def _executable(t, n):
    """TRIPLE-VERDICT on whether the graph engine could run this node.

      True   it is a graph op
      False  it is head plumbing this architecture is known to use
      None   it is neither, and the caller must refuse it BY NAME

    The answer is deliberately about the TYPE and nothing else: positional
    reasoning already decided whether the node is in the graph, and this says
    whether a node of this type could be. Keeping the two apart is what makes the
    `None` verdict possible, and the `None` is the whole point -- a type this
    packer has never seen is refused rather than treated as head arithmetic and
    absorbed, because "absorbed" and "dropped" look identical from outside and
    only one of them is honest.

    Four types are executable in only one shape, and the refusals are in the
    emitter rather than here so that each carries a message naming the node:

    * `Mul` only as `x * Sigmoid(x)` (silu_of). A general elementwise product
      is a different op.
    * `Concat` only on axis 1, with any number of operands -- SPPF joins four
      and the neck joins two, so the count is not fixed and is checked per
      node in the emitter instead.
    * `Split` only on the channel axis. The C2f block halves its channels, and
      it is emitted as ONE SLICE NODE PER HALF rather than a multi-output split
      node, so that every operand reference in the whole graph format stays a
      single node index. The cost is one extra node per C2f; the alternative is
      a second dimension of addressing on every operand in the format.
    * `MaxPool` only 5x5 / stride 1 / pad 2 -- the spatial pyramid's pooling,
      which PRESERVES the extent and whose output is concatenated with its
      input. Only that padding is consistent with the concatenation.
    * `Resize` only nearest-neighbour. A linear or cubic resample is a different
      network, and point-sampling one would move every keypoint on that level by
      up to half a pixel.
    """
    op = n.op_type
    if op == "Mul":
        return silu_parts(t.by_out, n) is not None
    if op == "Sigmoid":
        # Graph only as the sigmoid half of an activation. The head's OTHER
        # sigmoids -- the class score's and the keypoint visibility's -- are
        # plumbing, and the two are told apart by their operand rather than by
        # their type, so both can exist in one graph.
        return silu_sigmoid_of(t.by_out, n) is not None
    if op == "Conv":
        return True
    if op == "Add":
        return True
    if op in ("Concat", "Split", "MaxPool", "Resize"):
        return True     # the shape checks live in emit_ops, which names the node
    if op in PLUMBING:
        return False
    return None


def _trace_strides_and_grid(t):
    """Derive the pyramid's strides and grids from the exporter's own constants.

    The exporter multiplies the decoded boxes and the keypoints by a per-anchor
    stride vector, so the strides ARE in the graph as a constant -- and deriving
    them from that constant rather than from input_size is what makes this a
    check instead of an assumption. The grid is then NOT derived: it is
    computed as input_size / stride and the total cell count is compared against
    the graph's output, which is an independent number. A checkpoint whose
    padding differed from its stride (the `auto` letterbox) would fail here
    rather than produce a pyramid whose last row of cells sits off the image.
    """
    cands = []
    for pn in t.plumbing:
        n = t.by_name[pn]
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        for a, b in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
            w = t.vals.get(a) if t.by_out.get(a) is None or t.by_out[a].op_type == "Constant" else None
            if w is None:
                continue
            w = np.asarray(w)
            if w.size != t.cells or w.ndim > 2:
                continue
            uniq = [int(x) for x in np.unique(w)]
            if 2 <= len(uniq) <= 4:
                cands.append((n.name, uniq))
    if not cands:
        _die("no per-anchor stride constant of the right length is in the "
             f"head's plumbing. The head multiplies decoded boxes and keypoints "
             f"by a [{1}, {t.cells}] stride vector, and this packer reads the "
             f"strides from THAT rather than assuming 8/16/32, so a checkpoint "
             f"with a different pyramid still packs correctly.")
    # All candidates must agree. They are the same vector read at several points
    # in the head; a disagreement means one of them is a different vector that
    # happens to have the same length, and picking the first would be a coin flip.
    base = cands[0][1]
    for nm, u in cands[1:]:
        if u != base:
            _die(f"the head's stride constants disagree: {cands[0][0]} says "
                 f"{base} and {nm} says {u}. They are the same vector read at "
                 f"several points, so a disagreement means one is a different "
                 f"vector of the same length and the first is not a safe choice.")
    t.strides = sorted(base)
    t.grid = [t.input[1] // s for s in t.strides]
    for s, gg in zip(t.strides, t.grid):
        if s <= 0 or t.input[1] % s:
            _die(f"input_size {t.input[1]} is not a whole number of stride-{s} "
                 f"cells. This build letterboxes to a square and does NOT pad "
                 f"the pyramid to a stride multiple, because a padded level "
                 f"produces cells whose centres are outside the image.")
        if t.input[1] // s != gg:  # unreachable, and deliberately so
            _die("internal: grid derivation disagrees with itself")
    total = sum(g * g for g in t.grid)
    if total != t.cells:
        _die(f"the pyramid at strides {t.strides} over {t.input[1]}px holds "
             f"{total} cells, but the graph's output has {t.cells}. The runtime "
             f"reads the output cell by cell in stride order, so a mismatch "
             f"would attach every keypoint to the wrong cell -- and since a "
             f"cell index only changes which of a few dozen coordinates is "
             f"reported, that looks like a slightly worse pose rather than an "
             f"error.")
    _trace_grid_offsets(t)


def _trace_grid_offsets(t):
    """The half-cell offset each of the head's two grids adds to (column, row).

    The exporter builds the grid twice, from two different constants, and in
    this checkpoint the two do NOT agree:

        box grid       [1, 2, cells]   = j + 0.5, i + 0.5     cell CENTRES
        keypoint grid  [2, cells]      = j + 0,   i + 0       cell CORNERS

    So the offset is read out of the file and recorded, once per branch, rather
    than taken from a table of what exporters do. Assuming they agree puts every
    joint of every person half a cell out -- 4, 8 and 16 pixels at strides 8, 16
    and 32 -- while the boxes stay exactly right, so the pose reads as broadly
    plausible and every limb is misplaced by a distance that grows with the
    pyramid level and with nothing else.

    WHICH grid is which comes from the RANK, and that is worth spelling out
    rather than hiding, because a coin flip between two same-length constants is
    exactly the sort of thing that produces one plausible checkpoint's pose on
    another's data. The rank differs for a structural reason: the box grid is
    added to a [1, 2, cells] pair of distances, so it carries its own leading
    batch axis, while the keypoint grid is broadcast against the 17 axis that
    precedes it, so it is [2, cells] and gets that axis from the other operand.

    Both are then CHECKED against the pyramid this same function derived: within
    each level both rows must be arange(base, base + grid) for that level's
    width. That verifies the offset, the level order and the level widths
    together, from three constants that were independently derived.
    """
    found = {}
    for pn in t.plumbing:
        n = t.by_name[pn]
        if n.op_type not in ("Add", "Sub"):
            continue
        for x in n.input:
            w = t.vals.get(x)
            if w is None:
                continue
            w = np.asarray(w, dtype=np.float64)
            if w.ndim not in (2, 3) or w.shape[-1] != t.cells or w.shape[-2] != 2:
                continue
            prev = found.get(w.ndim)
            if prev is not None and not np.array_equal(prev[1], w):
                _die(f"two different rank-{w.ndim} per-anchor grids are in the "
                     f"head's plumbing ({prev[0]} and {pn}), and this runtime "
                     f"adds one recorded offset per branch. Picking the first "
                     f"would be a coin flip between two same-length constants.")
            found.setdefault(w.ndim, (pn, w))
    if sorted(found) != [2, 3]:
        _die(f"the head's plumbing has per-anchor grids of ranks {sorted(found)}, "
             f"not one of rank 2 and one of rank 3. Those are the two grids this "
             f"architecture builds -- one added to the box distances and one "
             f"broadcast over the keypoints -- and their offsets are recorded "
             f"separately because in this checkpoint they are 0.5 and 0. Any "
             f"other pair is a head whose grid this runtime does not reproduce.")
    t.box_grid = found[3][1][0]      # [1, 2, cells]
    t.kpt_grid = found[2][1]         # [2, cells]

    for name, arr, per_row in (("the box grid", t.box_grid, True),
                               ("the keypoint grid", t.kpt_grid, True)):
        base = float(arr.ravel()[0])
        if not np.isfinite(base) or abs(base) > 1.0:
            _die(f"{name} starts at {base}, which is not a half-cell offset. "
                 f"It is added to the cell's column and row before the stride "
                 f"multiply, so it has to be finite and inside one cell.")
        at = 0
        for gg in t.grid:
            want = np.arange(base, base + gg, dtype=np.float64)
            rows = [arr.reshape(-1, t.cells)[0, at:at + gg * gg].reshape(gg, gg),
                    arr.reshape(-1, t.cells)[1, at:at + gg * gg].reshape(gg, gg)]
            for axis, row in enumerate(rows):
                # row[i][j] is the coordinate of cell (i, j): the exporter's
                # meshgrid indexes x by column and y by row, which is the one
                # remaining way to transpose every pose that the shape checks
                # cannot see -- both rows would still be a correct arange if the
                # two axes were swapped, and a person would still have a head
                # above its hips.
                want_row = want[None, :] if axis == 0 else want[:, None]
                if not np.allclose(row, want_row, atol=1e-6):
                    _die(f"{name}, level of {gg} cells per side, is not "
                         f"arange({base}, {base + gg}) broadcast over its "
                         f"{'columns (x)' if axis == 0 else 'rows (y)'}. It is "
                         f"{row.ravel()[:4]} ... {row.ravel()[-2:]}. This "
                         f"runtime derives that grid from the stride constant "
                         f"and the level widths and adds one recorded offset to "
                         f"both axes, so a grid it cannot reproduce is refused "
                         f"rather than approximated -- every keypoint would be "
                         f"placed against the wrong cell.")
            at += gg * gg
    t.box_grid_offset = float(t.box_grid.ravel()[0])
    t.kpt_grid_offset = float(t.kpt_grid.ravel()[0])


def silu_parts(by_out, mul_node):
    """The (conv, sigmoid) pair of an activation `Mul(x, Sigmoid(x))`, or None.

    Matched on IDENTITY of the sigmoid's input, never on a node's NAME, so a
    re-export that renames everything still packs. `x` must be produced by a
    Convolution: the runtime fuses the activation into that convolution's
    epilogue, so a Mul whose first operand is not a conv's output has no epilogue
    to fuse into and is a different op.

    Two halves, two answers. The Sigmoid is asked about separately by
    `silu_sigmoid_of`, because the forward walk reaches the Mul before the
    Sigmoid (the Mul is enqueued as a consumer of the conv) and the Mul's second
    operand is a tensor the Sigmoid has not produced yet. Treating that operand
    as external would classify the activation as head plumbing and the walk
    would stop there -- which is exactly the bug this comment exists to prevent.
    """
    if len(mul_node.input) != 2:
        return None
    a, b = mul_node.input
    for x, y in ((a, b), (b, a)):
        n = by_out.get(y)
        if (n is not None and n.op_type == "Sigmoid" and len(n.input) == 1
                and n.input[0] == x):
            src = by_out.get(x)
            if src is not None and src.op_type == "Conv":
                return (src.name, n.name)
    return None


def silu_sigmoid_of(by_out, sigmoid_node):
    """The convolution whose activation this Sigmoid is, or None.

    The mirror of silu_parts: found by looking for the Mul that has this
    Sigmoid's output as one operand and the Sigmoid's own input as the other.
    Anything else is a head Sigmoid -- the class score's and the keypoint
    visibility's -- and belongs to the plumbing, which is why the distinction has
    to be made per node rather than per type.
    """
    if len(sigmoid_node.input) != 1:
        return None
    x = sigmoid_node.input[0]
    for n in by_out.values():
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        for a, b in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
            if a == sigmoid_node.output[0] and b == x:
                src = by_out.get(x)
                if src is not None and src.op_type == "Conv":
                    return src.name
    return None


def _must(d, key, what):
    if key not in d:
        _die(f"{what} ({key}) is not a node this packer placed, so the emitted "
             f"graph would read a value no earlier node computes. The runtime "
             f"refuses this too (runtime/src/pose/geometry.cpp), and naming the "
             f"ONNX node here is more useful than naming it at load time.")
    return key


def _conv_geometry(n, w, name):
    """(stride, pad_h, pad_w) for one Convolution, read off the node.

    These three numbers are the difference between this network and a slightly
    different one that looks the same. A 3x3 with pad 1 and stride 1 keeps a
    pyramid level the size it was; the SAME 3x3 with stride 2 is the stem and
    halves it. An im2col that assumed "pad k/2, stride 1" -- which is what this
    runtime's first version did -- would be right for 71 of 73 convolutions and
    produce a 642x642 first layer for the other two, and no shape check would
    complain, because the graph would be self-consistent about the wrong number.

    So they are read, and both spellings ONNX allows are handled: the explicit
    `pads`, and the `auto_pad` string, which is NOTSET for the exporters that
    write pads out and SAME_UPPER for the ones that do not. An unknown auto_pad
    value is refused by name rather than treated as VALID, because VALID means
    "no padding" and treating an unknown as VALID is the wrong default twice
    out of the three possible readings.
    """
    strides = _attribute(n, "ints", "strides")
    if strides is None:
        strides = [1, 1]                    # the ONNX default
    if strides != [strides[0], strides[0]]:
        _die(f"{name} has strides {strides}. This runtime's im2col steps both "
             f"axes by the same amount, because a convolution that walks its "
             f"window at a different rate horizontally and vertically is a "
             f"stride-1 convolution with a skew, and its output extent is not "
             f"the one below.")
    stride = int(strides[0])

    auto_pad = None
    for a in n.attribute:
        if a.name == "auto_pad":
            v = _raw(a)
            auto_pad = v.decode("utf-8") if isinstance(v, bytes) else v
    pads = _attribute(n, "ints", "pads")
    kh, kw = int(w.shape[2]), int(w.shape[3])
    if pads is None:
        if auto_pad in (None, "NOTSET"):
            pads = [0, 0, 0, 0]
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            pads = [kh // 2, kw // 2, kh - 1 - kh // 2, kw - 1 - kw // 2]
        elif auto_pad == "VALID":
            pads = [0, 0, 0, 0]
        else:
            _die(f"{name} sets auto_pad='{auto_pad}'. This packer reads NOTSET, "
                 f"SAME_UPPER, SAME_LOWER and VALID, and refuses the rest by "
                 f"name. Guessing is not an option here: VALID means no padding "
                 f"at all, so an unknown read as VALID would crop every layer "
                 f"this network pads, and read as SAME it would grow them.")
    if len(pads) != 4:
        _die(f"{name} has pads {pads}, not four values (top, left, bottom, "
             f"right). ONNX's order, and a shorter list would mean the "
             f"exporter's convention rather than this one's.")
    if pads[0] != pads[2] or pads[1] != pads[3]:
        _die(f"{name} pads [{pads[0]}, {pads[1]}, {pads[2]}, {pads[3]}] -- "
             f"asymmetrically, which is what auto_pad=SAME_LOWER produces (one "
             f"extra row at the bottom rather than at the top). The runtime's "
             f"im2col pads symmetrically and its conv node carries one pad per "
             f"axis, so this convolution would be packed with a padding it does "
             f"not have. It moves the whole feature map by half a pixel, which "
             f"no shape check would catch.")
    return stride, int(pads[0]), int(pads[1])


class Ops(list):
    """`ops`, plus the checkpoint node each entry came from.

    Not decoration. tools/verify/verify_pose.py compares the runtime's graph
    against the ONNX graph NODE BY NODE, and a positional comparison has to know
    which checkpoint node each index is: end to end, a mismatch in the first
    convolution and a mismatch in the twentieth are both "the numbers differ",
    and the difference between fixing them is one re-run of the packer and one
    20-minute reference against twenty.

    The Split name carries the half it became -- "node:0", "node:1" -- because
    one checkpoint node emits two runtime nodes and writing the same name twice
    would make the two indistinguishable in a diff.
    """

    def __init__(self):
        super().__init__()
        self.names = []

    def append(self, op, node_name):
        super().append(op)
        self.names.append(node_name)


def emit_ops(t):
    """The traced graph -> the runtime's op list, with the head wired up.

    Returns (ops, conv_of, dfl_index, bias_free) where `conv_of` maps an ONNX
    node NAME to a convolution index and `bias_free` is the set of node names
    whose Convolution listed no bias initializer.

    THREE THINGS HAPPEN HERE THAT ARE NOT JUST SERIALISATION
    -------------------------------------------------------
    1. ACTIVATIONS DISAPPEAR. The walk classified `Mul(x, Sigmoid(x))` and its
       Sigmoid as graph ops, because they ARE part of the body. They do not
       become graph NODES: the runtime fuses the activation into the
       convolution's epilogue, and a separate SiLU node over 4.59 GMAC of
       activation output would be a memory-bound pass the checkpoint never asked
       for. So both are dropped here and the convolution that owns them is
       emitted with "silu": 1.

       The consequence is handled explicitly rather than hoped for: a consumer of
       a dropped node's output must be rewired to the CONVOLUTION that produced
       the operand, and one such consumer exists (the next convolution in the
       chain). Any other consumer is a refusal, because "who does this value
       belong to" has no answer that does not involve guessing.

    2. CHANNEL SPLITS BECOME SLICES. One slice node per half; see the note in
       emit_ops' Split branch and in the C++ header.

    3. THE OPERAND IS AN INDEX INTO THE EMITTED LIST, not an ONNX tensor name.
       Node 0's operand is -1, which is the image. Both ends of that convention
       are checked: a consumer of a dropped value is refused here, and the
       runtime's geometry reader refuses an operand that is neither -1 nor an
       earlier node.
    """
    # -- 1. which nodes vanish, and what stands in for them
    fused = {}     # dropped node name -> the node whose output replaces it
    silu_conv = {}
    for name in t.graph_nodes:
        n = t.by_name[name]
        if n.op_type != "Mul":
            continue
        parts = silu_parts(t.by_out, n)
        if parts is None:
            _die(f"{name} is a Mul in the network's body that is not the "
                 f"activation pattern x * Sigmoid(x). This packer has no other "
                 f"Mul in the graph: a network with a different elementwise "
                 f"expression in its body is a different architecture, and "
                 f"guessing which Mul is a scale and which is an activation is "
                 f"how a container computes a plausible wrong model.")
        if parts[0] in silu_conv:
            _die(f"convolution {parts[0]} is claimed by two activation Muls "
                 f"({silu_conv[parts[0]]} and {name}). Structurally impossible "
                 f"in a real graph, so the trace is wrong and the container "
                 f"would fuse the activation twice.")
        silu_conv[parts[0]] = name
        # The Mul's own output stands for the CONVOLUTION's output: the
        # activation does not change a tensor's shape, so the convolution and
        # its activation have the same shape and every downstream resolution is
        # identical. What differs is the VALUE, and that is carried by the
        # "silu" flag on the convolution rather than by a separate node.
        fused[name] = parts[0]
        fused[parts[1]] = parts[0]

    ops = Ops()
    emitted = {}      # node name -> index in `ops`
    producer = {}     # ONNX tensor name -> node name, for EMITTED nodes only
    # ONNX tensor name -> the index of the op that produced THAT tensor. Needed
    # because a Split is several runtime nodes and `emitted` can only name one of
    # them: YOLOv8's C2f block joins [Split:0, Split:1, m.0], and with `emitted`
    # alone both halves resolved to the LAST one, so the join concatenated the
    # same 16 channels twice and dropped the first 16 -- a 48-channel tensor of
    # exactly the right width holding half the block. Nothing downstream can see
    # it: the channel counts still add up and the graph is still well formed.
    tensor_idx = {}
    conv_of = {}
    bias_free = set()

    def idx_of(tensor, who, role):
        if tensor == t.src:
            if ops:
                _die(f"{who} reads the image as its {role}, but node 0 has "
                     f"already been emitted. The image is operand -1 and is "
                     f"available to any node, but a second reader of it in this "
                     f"graph would mean the input is being consumed twice -- "
                     f"which is this architecture's business, not the packer's "
                     f"guess.")
            return -1
        # Follow the fold. A dropped activation's output tensor has no producer
        # in the emitted list -- but the CONVOLUTION that owns it has the same
        # shape and the same slot, because an activation does not change a
        # tensor's extent. So `fused` is a rename of the operand, and every
        # consumer of an activation's output is rewired to the convolution,
        # which is exactly what the runtime's fused epilogue computes.
        seen = 0
        node = producer.get(tensor)
        while node is None:
            src = t.by_out.get(tensor)
            if src is None or src.name not in fused or seen > 4:
                _die(f"{who} reads {tensor} as its {role} and no earlier node "
                     f"produces it."
                     + ("" if src is None else
                        f" It is produced by {src.name}, which is neither a "
                        f"graph node this packer emitted nor an activation with "
                        f"a convolution behind it -- so nothing computed it."))
            node = fused[src.name]
            tensor = t.by_name[node].output[0]
            seen += 1
        # The INDEX, not the name. The graph's operands are positions in the
        # emitted list, and writing the name here produced a container whose
        # every operand after the first was a string the runtime cannot resolve.
        #
        # `tensor_idx` is asked first. For a folded activation it and `emitted`
        # agree -- the tensor that reaches here is the CONVOLUTION's output. For
        # a SPLIT output they do not, and `emitted` answers the last half.
        at = tensor_idx.get(tensor)
        if at is not None:
            return at
        return emitted[_must(emitted, node, who)]

    for name in t.graph_nodes:
        n = t.by_name[name]
        if name in fused:
            continue
        src = n.input[0] if n.input else ""

        if n.op_type == "Conv":
            w = t.vals.get(n.input[1])
            if w is None:
                _die(f"{name} has no initializer for its weight. A convolution "
                     f"whose weights are not in the graph is not an export this "
                     f"packer can read.")
            if len(n.input) == 3:
                if n.input[2] not in t.vals:
                    _die(f"{name} lists a bias initializer it does not carry.")
            elif len(n.input) == 2:
                bias_free.add(name)
            else:
                _die(f"{name} has {len(n.input)} inputs. A convolution has an "
                     f"activation and a weight, optionally a bias.")
            conv_of[name] = len(conv_of)
            stride, pad_h, pad_w = _conv_geometry(n, w, name)
            ops.append({"op": "conv", "conv": conv_of[name],
                        "silu": 1 if name in silu_conv else 0,
                        "stride": stride, "pad": [pad_h, pad_w],
                        "inputs": [idx_of(src, name, "input")]}, name)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name
                tensor_idx[o] = emitted[name]
            continue

        if n.op_type == "Add":
            ops.append({"op": "add",
                        "inputs": [idx_of(src, name, "left"),
                                   idx_of(n.input[1], name, "right")]}, name)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name
                tensor_idx[o] = emitted[name]
            continue

        if n.op_type == "Concat":
            axis = _attribute(n, "i", "axis")
            if axis != 1:
                _die(f"{name} concatenates on axis {axis}, not 1. The runtime's "
                     f"join is channel-wise by construction. A spatial or batch "
                     f"join is a different op, and transposing it here would "
                     f"produce a tensor of the right shape holding the wrong "
                     f"pixels.")
            # Any arity from two up. SPPF joins four (its own input and three
            # pooled copies) and the neck joins two, so pinning this at two would
            # refuse the spatial pyramid's only join.
            if len(n.input) < 2:
                _die(f"{name} joins {len(n.input)} tensor(s). A join needs at "
                     f"least two; a single operand is a copy, and a copy reads "
                     f"as a plausible layer.")
            ops.append({"op": "concat",
                        "inputs": [idx_of(i, name, f"operand {k}")
                                   for k, i in enumerate(n.input)]}, name)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name
                tensor_idx[o] = emitted[name]
            continue

        if n.op_type == "Split":
            # ONE SLICE NODE PER OUTPUT. See the note in _executable: this keeps
            # every operand in the graph format a single node index, at the cost
            # of one extra node per channel split.
            #
            # The sizes come from the node's own `split` attribute when it has one
            # and from the second OPERAND otherwise -- ONNX permits both, and
            # which one the exporter used is not something to guess at. A node
            # with neither is refused rather than assumed to be an even split,
            # because an even split is exactly the assumption a missed odd split
            # fails on.
            axis = _attribute(n, "i", "axis")
            if axis is not None and axis != 1:
                _die(f"{name} splits on axis {axis}, not 1. The runtime's slice "
                     f"is a contiguous run of CHANNELS; a spatial split is a "
                     f"different op and is not approximated by a channel one, "
                     f"which would produce a tensor of the right size holding "
                     f"the wrong pixels.")
            sizes = _attribute(n, "ints", "split")
            if sizes is None:
                if len(n.input) < 2 or n.input[1] not in t.vals:
                    _die(f"{name} carries no split sizes, as an attribute or as "
                         f"an operand this packer can read. An even split is "
                         f"the obvious assumption and it is exactly the "
                         f"assumption a missed odd split fails on, silently: "
                         f"the channel counts would still add up.")
                sizes = [int(x) for x in np.asarray(t.vals[n.input[1]]).ravel()]
            if not sizes or sum(sizes) <= 0:
                _die(f"{name} has split sizes {sizes}")
            si = idx_of(src, name, "input")
            if len(n.output) != len(sizes):
                _die(f"{name} splits into {len(sizes)} sizes but has "
                     f"{len(n.output)} outputs. One runtime `slice` per output "
                     f"means the two have to line up, and pairing them by "
                     f"position without checking is a graph whose halves are in "
                     f"the wrong order.")
            at = 0
            for half, wd in enumerate(sizes):
                ops.append({"op": "slice", "chan": at, "nch": int(wd),
                            "inputs": [si]}, f"{name}:{half}")
                tensor_idx[n.output[half]] = len(ops) - 1
                at += int(wd)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name   # the halves DIFFER; tensor_idx says which
            continue

        if n.op_type == "MaxPool":
            k = _attribute(n, "ints", "kernel_shape")
            st = _attribute(n, "ints", "strides")
            pads = _attribute(n, "ints", "pads")
            # The extent-preserving condition, stated as such: stride 1 and
            # padding k/2 on every side gives
            #     out = floor((H + 2*(k/2) - k)/1) + 1 = H
            # which is what makes SPPF's join possible at all -- it concatenates
            # the pooled copies with the UNPOOLED input, so the two extents have
            # to agree. A pooling that shrank the map would not be a different
            # number in an error message; it would change every downstream
            # resolution and still produce a box.
            if st != [1, 1]:
                _die(f"{name} pools {k} with stride {st}. The spatial pyramid "
                     f"pools with stride 1, which PRESERVES the extent, and its "
                     f"output is concatenated with its own input -- only "
                     f"possible if the extents agree. A pooling that shrank the "
                     f"map is a different op.")
            if pads != [k[0] // 2, k[1] // 2, k[0] // 2, k[1] // 2]:
                _die(f"{name} pads {pads} for a {k} window. Padding is not a "
                     f"field on the runtime's maxpool node -- it is implied by "
                     f"k/2 -- because that is the only padding under which this "
                     f"pooling preserves the extent. Padding it any other way "
                     f"would change every downstream resolution.")
            ops.append({"op": "maxpool", "k": k[0], "stride": st[0],
                        "inputs": [idx_of(src, name, "input")]}, name)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name
                tensor_idx[o] = emitted[name]
            continue

        if n.op_type == "Resize":
            mode = _resize_mode(n)
            if "nearest" not in mode:
                _die(f"{name} resizes with mode {mode!r}, not nearest. The "
                     f"neck's up path is nearest-neighbour; a linear or cubic "
                     f"resample is a different network and is not approximated "
                     f"by point-sampling.")
            # The scales are read per AXIS, not as a list to all-match. The
            # exporter writes them for NCHW, so a nearest x2 upsample is
            # [1, 1, 2, 2] -- and checking "every factor is 2" would refuse
            # every real upsample in the file, which is a check that looks
            # strict and is really just wrong.
            sc = next((_raw(a) for a in n.attribute if a.name == "scales"), None)
            if sc is None and len(n.input) > 2 and n.input[2] in t.vals:
                sc = np.asarray(t.vals[n.input[2]]).ravel().tolist()
            if sc is not None:
                if not isinstance(sc, list) or len(sc) != 4:
                    _die(f"{name} carries scales {sc!r}. A four-axis NCHW "
                         f"scale vector is the only shape this reader takes, "
                         f"and a three-element or one-element one would place "
                         f"the doubling on the wrong axis.")
                got = [round(float(x), 6) for x in sc]
                if got != [1.0, 1.0, 2.0, 2.0]:
                    _die(f"{name} resamples by {got}. The neck doubles the "
                         f"spatial resolution and leaves the batch and the "
                         f"channels alone. The runtime's upsample node doubles "
                         f"by construction and carries no factor, because one "
                         f"that was not 2 -- or that doubled the channel axis -- "
                         f"would silently produce a pyramid of the wrong size "
                         f"whose channel counts still add up.")
            ops.append({"op": "upsample", "scale": 2,
                        "inputs": [idx_of(src, name, "input")]}, name)
            emitted[name] = len(ops) - 1
            for o in n.output:
                producer[o] = name
                tensor_idx[o] = emitted[name]
            continue

        _die(f"internal: {name} ({n.op_type}) reached the emitter, which only "
             f"handles the traced graph op set")

    # -- the head's inputs, PAIRED BY RESOLUTION
    #
    # Paired structurally rather than by the order the plumbing happens to list
    # them in. The box tensor for level i and the keypoint tensor for level i are
    # both at grid_i; the grids are distinct (input_size / 8, /16, /32); and that
    # is the only consistent pairing. Two keypoint tensors at one resolution
    # would make it ambiguous and are refused rather than resolved by order --
    # order would be a coin flip, and both orders produce poses that look
    # plausible and belong to the wrong people.
    box_t, kpt_t = {}, {}
    for name, n in t.terminals.items():
        if name == t.dfl:
            continue
        c, h, w = _chw(t.vinfo, n.output[0], f"terminal {name}")
        if h != w:
            _die(f"terminal {name} is {h}x{w}. Every head input sits on a "
                 f"square pyramid level; a non-square one means the export "
                 f"padded to a stride multiple, which this build does not do -- "
                 f"a padded level produces cells whose centres are outside the "
                 f"image.")
        if c == 4 * DFL_BINS + NUM_CLASSES:
            box_t[h] = name
        elif c == NUM_KEYPOINTS * 3:
            kpt_t[h] = name
        else:
            _die(f"terminal {name} has {c} channels. The head reads a "
                 f"box+class tensor ({4 * DFL_BINS} + {NUM_CLASSES}) or a "
                 f"keypoint tensor ({NUM_KEYPOINTS} * 3). A third width means "
                 f"the head's inputs are not the ones this runtime reads, and "
                 f"reading them anyway would return numbers with no meaning.")
    want = sorted(t.grid)
    if sorted(box_t) != want:
        _die(f"the head's box tensors are at resolutions {sorted(box_t)} and "
             f"the pyramid's grids are {want}.")
    if sorted(kpt_t) != want:
        _die(f"the head's keypoint tensors are at resolutions {sorted(kpt_t)} "
             f"and the pyramid's grids are {want}. A keypoint head at another "
             f"resolution puts every joint on the neighbouring cell's pixel.")

    detect_inputs = []
    for gg in t.grid:
        for role, table in (("box", box_t), ("keypoint", kpt_t)):
            node = table[gg]
            if node not in emitted:
                _die(f"the head's {role} tensor at grid {gg} is {node}, which "
                     f"this packer did not emit. A head input the graph does not "
                     f"compute is a graph with a hole in it.")
            detect_inputs.append(emitted[node])

    # The DFL's projection gets the LAST convolution index. It is not a graph
    # node -- its input is a Softmax, which the graph engine cannot run -- and it
    # is not dropped: read_geometry checks its shape and Network::head folds it
    # into the distribution's expectation exactly.
    dfl_index = len(conv_of)
    conv_of[t.dfl] = dfl_index
    # The DFL projection is bias-free in this architecture and the fold in
    # Network::head is only EXACT without one:
    #     sum_bin w[0][bin]*p[bin] + b  ==  w[0] . E[bin] + b
    # uses sum(p) == 1, which holds for the softmax, and needs nothing about b --
    # but b is 0 here, so recording it as bias-free is what tells read_geometry
    # the head's fold is a fold rather than an affine it would have to carry.
    if len(t.by_name[t.dfl].input) == 2:
        bias_free.add(t.dfl)
    ops.append({"op": "detect", "inputs": detect_inputs}, "detect")
    return ops, conv_of, dfl_index, bias_free


def collect_weights(t, conv_of):
    """node name -> (weight [Cout, Cin, kh, kw] F32, bias or None).

    Every index in the runtime's im2col depends on that channel order -- [Cout,
    Cin, kh, kw], not [Cin, Cout, ...] and not a transposed kernel -- so the
    shape is checked to be rank 4 and the kernel square, and nothing is
    reinterpreted. A weight stored any other way is refused here rather than
    folded at load, because folding it would produce a network whose
    convolutions are transposes of the checkpoint's and whose output is still a
    plausible pose.
    """
    out = {}
    for name in conv_of:
        n = t.by_name[name]
        w = np.asarray(t.vals[n.input[1]], dtype=np.float32)
        if w.ndim != 4:
            _die(f"{name}'s weight is {list(w.shape)}, not rank 4. A "
                 f"convolution's weight is [Cout, Cin, kh, kw] and every index "
                 f"in the runtime's im2col depends on that order.")
        if w.shape[2] != w.shape[3]:
            _die(f"{name}'s kernel is {w.shape[2]}x{w.shape[3]}, not square. "
                 f"The runtime's im2col assumes a square kernel and refuses an "
                 f"anisotropic one rather than transposing it.")
        b = None
        if len(n.input) == 3:
            b = np.asarray(t.vals[n.input[2]], dtype=np.float32).reshape(-1)
            if b.size != w.shape[0]:
                _die(f"{name}'s bias has {b.size} values and its weight has "
                     f"{w.shape[0]} output channels.")
        out[name] = (np.ascontiguousarray(w), b)
    return out


def npu_panels(t, conv_of, weights, device):
    """Pre-tiled bf16 B panels for the array backend, one per graph convolution.

    Returns (panels, streams, (tile_k, tile_n)) where panels maps a node NAME to
    (ndarray, layout, padded_k, padded_n) and streams is the sorted list of
    distinct padded (K, N) shapes, which is the design set the exporter builds.

    WHY THE PADDING IS 3.3x WASTE AND IT IS STILL OFFERED
    ------------------------------------------------------
    A design's N must be a multiple of tile_n * AIE columns. On npu1 that is
    32 * 4 = 128, and this network's output channel counts are 16, 32, 48, 51,
    64, 96, 128, 192 and 256 -- so most pad to 128 or more, and the useful
    4.59 GMAC becomes 15.2 GMAC of dispatched work. K pads to a multiple of
    tile_k and is nearly always already one, since a conv's K is Cin*kh*kw and
    this network's channel counts are multiples of 8.

    The reason this is done anyway is that --npu-extra-ops conv exists so the
    ARRAY path can be MEASURED on this model rather than estimated in a comment,
    and a measured 105 ms against a measured 20.5 ms is worth having written
    down. tile_n is 32 rather than ViT's 48 because 48 divides none of these
    channel counts usefully; 32 is the sub-tile width Whisper uses and the one
    npue.py's measurements cover at this array.
    """
    from npue import (cols_for_device, gemm_b_layout, mac_for_device,  # noqa: E402
                      tile_b, to_bf16_bits)

    TILE_K, TILE_N = 64, 32
    mac_s, mac_t = mac_for_device(device, "BF16")
    # N PADS TO tile_n * COLS, NOT TO tile_n. This is the whole reason this
    # function needed a device at all, and getting it wrong is a read past the
    # end of a tensor rather than a wrong number.
    #
    # A design's N must be a multiple of tile_n * AIE columns -- 128 on npu1 --
    # and the compiled core runs that many columns every dispatch. The B panel
    # it multiplies against therefore has to BE that wide. Padding to tile_n
    # alone stores a 32-column panel for a convolution whose design reads 128,
    # and the 96 columns past the end are whatever follows this tensor in the
    # container's data region: the products are plausible, the layout_hash
    # matches because both sides derived it from the same constants, and nothing
    # in the loader can see it.
    #
    # Every model in this repository until now had N already a multiple of 128 --
    # embedder widths are 384, 768, 1536, 3072 -- so this never had a reader.
    # YOLOv8-pose is the first with channels of 16, 32, 48, 51, 64, 96, 128,
    # 192 and 256.
    N_MULT = TILE_N * cols_for_device(device)
    layout = gemm_b_layout(TILE_K, TILE_N, mac_s, mac_t)
    panels, streams = {}, set()
    for name in t.graph_nodes:
        if t.by_name[name].op_type != "Conv":
            continue
        w = weights[name][0]
        cout, cin, kh, kw = w.shape
        K, N = cin * kh * kw, cout
        pk = ((K + TILE_K - 1) // TILE_K) * TILE_K
        pn = ((N + N_MULT - 1) // N_MULT) * N_MULT
        # tile_b refuses a shape that does not tile, so the panel is padded HERE
        # and the padding is ZERO -- which is exact, not an approximation: a
        # padded K column is multiplied by nothing, and a padded N column is a
        # channel the runtime never reads back. The A side is not padded the
        # same way, which is why the runtime zero-fills the A panel's tail to the
        # dispatch's row count; see whisper/npu_ops.hpp.
        b = np.zeros((pk, pn), dtype=np.float32)
        # A TRANSPOSE, NOT A RESHAPE. `w` is the checkpoint's [cout, cin, kh, kw]
        # and the panel is [K, N] -- `tile_b` tiles the FIRST axis in tile_k and
        # the second in tile_n, and `add_gemm_b`/`gemm_b_panel` declare
        # logical_shape [K, N] to match. `w.reshape(K, N)` happens to have the
        # right SIZE and is not the same matrix: it is w's flat order laid into a
        # K-by-N grid, which for every convolution in this network is a
        # permutation that no arithmetic rejects.
        #
        # The failure is silent in every direction that exists. tile_b accepts it
        # (the padded shape tiles), the layout_hash matches (both sides derive it
        # from the same constants, and it does not describe the DATA), the C++
        # reader hands the bytes to the device unchanged, and the array multiplies
        # a transposed weight -- so `--npu-extra-ops conv` returns a network that
        # finds 300 people in a photograph with three, with the stem convolution
        # already wrong and no error anywhere. `b[:K, :N] = w.reshape(N, K).T` is
        # the transpose; the flat order is what the reshape above preserved.
        b[:K, :N] = w.reshape(N, K).T
        panels[name] = (tile_b(b, TILE_K, TILE_N, s=mac_s, t=mac_t), layout,
                        pk, pn)
        streams.add((pk, pn))
    return panels, sorted(streams), (TILE_K, TILE_N)


def pack_pose(onnx_path, out_path, device=None, npu=False, dry_run=False,
              dtype="f32", int4_group=None):
    """ONNX -> arch=6 .npue. `npu=True` adds the pre-tiled B panels and the
    per-convolution stream names.

    The container's mean/std are the identity, and they are WRITTEN rather than
    assumed: YOLOv8's own preprocessor divides by 255 and stops, and a checkpoint
    packed from a pipeline that also subtracted a mean would need different ones.
    The runtime reads the two triples and normalises by them, so writing them is
    what makes the front end a property of the CONTAINER rather than of this
    file -- a later packer with a different front end changes two numbers here
    instead of a function in C++.
    """
    model = onnx.load(str(onnx_path))
    t = trace(model)
    ops, conv_of, dfl_index, bias_free = emit_ops(t)
    wdtype = conv_quant.conv_weight_dtype(dtype)
    if wdtype == "i4":
        if int4_group is not None and int(int4_group) < 0:
            _die(f"--int4-group {int4_group} is negative. 0 means one group over "
                 f"the whole input axis, i.e. per-channel int4, and is the only "
                 f"value below 1 that has a meaning.")
    elif int4_group is not None:
        _die(f"--int4-group {int4_group} was given with --dtype {wdtype}. It "
             f"only means anything for i4, and ignoring it would leave a "
             f"container whose `conv_int4_group` says 0 while the operator "
             f"asked for {int4_group} -- the kind of disagreement that is only "
             f"discovered when the accuracy turns out to be somebody else's.")
    weights = collect_weights(t, conv_of)

    n_conv = sum(1 for o in ops if o["op"] == "conv")
    if dry_run:
        counts = {k: sum(1 for o in ops if o["op"] == k)
                  for k in ("conv", "concat", "add", "maxpool", "upsample")}
        print(f"graph        {len(ops)} ops: " + ", ".join(
            f"{counts[k]} {k}" for k in
            ("conv", "concat", "add", "maxpool", "upsample")) + ", 1 detect")
        print(f"conv         {len(conv_of)} convolutions "
              f"({n_conv} in the graph + 1 DFL projection), "
              f"{sum(w[0].size for w in weights.values()) / 1e6:.2f} M params")
        print(f"strides      {t.strides}  grids {t.grid}  cells {t.cells}")
        print(f"grids        box offset {t.box_grid_offset} (cell centres), "
              f"keypoint offset {t.kpt_grid_offset} (cell corners)")
        print(f"head         detect({len(ops[-1]['inputs'])} inputs = "
              f"{len(t.grid)} levels x (boxes, keypoints)), dfl conv "
              f"#{dfl_index}, bias-free {sorted(conv_of[n] for n in bias_free)}")
        print(f"channels     detect_cout={t.output[0]}, "
              f"box+class={4 * DFL_BINS + NUM_CLASSES}, "
              f"keypoints={NUM_KEYPOINTS * 3}")
        print(f"plumbing     {len(t.plumbing)} head nodes absorbed by detect, "
              f"{len(t.terminals)} graph terminals")
        return

    panels = streams = tile = None
    if npu:
        # The panels are built from the DEQUANTISED weights, not from the fp32
        # source. One flag, one network: on a --dtype i8 container the host
        # multiplies `q * s` and the array multiplies bf16(q * s), and those
        # agree to the bf16 rounding. Tiling the checkpoint's own weights instead
        # would make `--npu-extra-ops conv` switch the MODEL as well as the
        # backend -- an i8 container whose array path is quietly an fp32 one,
        # with nothing in the file that says so, and a detection that moves when
        # a performance flag is added.
        if wdtype != "f32":
            # Reshaped BACK to the source's own 4-D shape, because `npu_panels`
            # unpacks `cout, cin, kh, kw = w.shape` and a [N, K] view would
            # fail there with a ValueError about the number of values. The shape
            # is the checkpoint's and is not something the quantiser changes.
            weights = {
                n: (conv_quant.dequantise(v[0], wdtype, int4_group).reshape(
                        v[0].shape), v[1])
                for n, v in weights.items()}
        panels, streams, tile = npu_panels(t, conv_of, weights, device)

    config = {
        "arch": ARCH_STRING,
        "kind": "pose",
        "input_size": int(t.input[1]),
        "num_keypoints": NUM_KEYPOINTS,
        "num_classes": NUM_CLASSES,
        "dfl_bins": DFL_BINS,
        "num_levels": len(t.strides),
        "strides": [int(x) for x in t.strides],
        "grid": [int(x) for x in t.grid],
        "detect_cout": int(t.output[0]),
        "detect_h": int(t.cells),
        "detect_w": 1,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "letterbox_pad_value": 114.0,
        # What the head's INPUTS mean, traced out of the graph above -- not what
        # the graph's own output means, which is a different tensor entirely and
        # which this runtime deliberately never builds. See the header note on
        # BoxFormat in runtime/include/pose/geometry.hpp.
        "head_box_format": "ltrb_dfl",
        "head_score": "logit",
        "head_keypoint_visibility": "logit",
        "head_dfl_conv": int(dfl_index),
        # The two grids' half-cell offsets, in CELLS, traced out of the graph's
        # own constants above. They differ in this checkpoint -- the boxes are
        # anchored at cell centres and the keypoints at cell corners -- so the
        # runtime is given both rather than one default applied twice.
        "head_box_grid_offset": float(t.box_grid_offset),
        "head_kpt_grid_offset": float(t.kpt_grid_offset),
        "num_convs": len(conv_of),
        # Which checkpoint INITIALISER each conv.N.w came from, by N. Recorded
        # because the shape cannot stand in for it: YOLOv8's C2f branches run
        # cv1 and cv2 side by side with identical shapes, and the head runs nine
        # of those, so six of the 73 convolutions are shape-indistinguishable
        # from a sibling. A checker that resolved weights by shape would gate
        # conv.3 against cv1's weights, find conv.4 disagreeing, and report a
        # packing bug that is not there -- while genuinely missing the case where
        # the two got swapped, which is indistinguishable by shape and is the
        # one that matters. The runtime does not read this; tools/verify/
        # verify_conv_quant.py does, and it is the only way it can name the
        # weight it is checking.
        "conv_sources": json.dumps(
            [t.by_name[n].input[1]
             for n, _ in sorted(conv_of.items(), key=lambda kv: kv[1])],
            separators=(",", ":")),
        # The precision the WEIGHTS are stored at, and the int4 group along the
        # input axis. Both are REQUIRED rather than defaulted, for the same
        # reason head_box_format is: a reader that had to guess would read fp32
        # bytes as int8 ones and produce a network with the right shapes.
        # `conv_int4_group` ships on an f32 or i8 container too, as 0 -- the same
        # "one code path" rule gemm_i8 follows by shipping an all-ones
        # `.asmooth`.
        "conv_weight_dtype": wdtype,
        "conv_int4_group": int(int4_group) if wdtype == "i4" else 0,
        # The convolutions whose ONNX node carried NO bias initializer, by conv
        # index. The runtime requires a bias for every other one and refuses a
        # container that does not say which these are -- see read_geometry.
        "bias_free": sorted(conv_of[n] for n in bias_free),
        "graph": json.dumps(ops, separators=(",", ":")),
        # Which checkpoint node each graph op came from. Read only by
        # tools/verify/verify_pose.py, which compares the two graphs node by
        # node and needs the correspondence to do it. Kept in the container
        # rather than recomputed, because the packer is not available to
        # anything that only has the file.
        "node_names": json.dumps(ops.names, separators=(",", ":")),
    }

    w = Writer(config, arch=ARCH_YOLOV8_POSE_C2F_SILU_DFL)
    worst = (0.0, "")
    werr = 0
    for name, idx in sorted(conv_of.items(), key=lambda kv: kv[1]):
        weight, bias = weights[name]
        err, nbytes = conv_quant.add_conv_w(w, f"conv.{idx}", weight, wdtype,
                                            int4_group)
        werr += nbytes
        if err > worst[0]:
            worst = (err, name)
        if bias is not None:
            w.add(f"conv.{idx}.b", bias, "F32", "conv_b", list(bias.shape))

    if npu:
        from npue import to_bf16_bits  # noqa: E402
        tile_k, tile_n = tile
        stream_name = {k: f"conv{k[0]}x{k[1]}" for k in streams}
        pos_of_conv = {o["conv"]: i for i, o in enumerate(ops) if o["op"] == "conv"}
        for name in [n for n in t.graph_nodes
                     if t.by_name[n].op_type == "Conv"]:
            idx = conv_of[name]
            panel, layout, pk, pn = panels[name]
            w.add(f"conv.{idx}.btile", to_bf16_bits(panel), "BF16", "gemm_b",
                  [pk, pn], layout=layout, padded_shape=[pk, pn])
            ops[pos_of_conv[idx]]["stream"] = stream_name[(pk, pn)]
        # Into w.config, NOT into the local `config` dict.
        #
        # `Writer.__init__` does `self.config = dict(config)` -- a COPY, taken
        # when the writer was constructed twenty lines above. Setting a key on
        # the local dict after that point changes a dictionary the file will
        # never see, and it did: every `--npu` container was written with NO
        # `npu_streams` at all and with the PRE-stream `graph`, so each of the
        # 73 convolutions in the recorded graph had no `stream` field and the
        # runtime had nothing to validate a design against. The comment below
        # used to claim the opposite could be caught at load, which is only
        # true of a container that records the streams in the first place.
        #
        # It is not a `Writer` bug to fix there: five other packers build the
        # whole dict before constructing it, and making `write()` re-read a
        # caller's dict would hide this class of mistake rather than catch it.
        w.config["npu_streams"] = json.dumps(
            [{"op": stream_name[k], "k": k[0], "n": k[1],
              "tile_k": tile_k, "tile_n": tile_n} for k in streams],
            separators=(",", ":"))
        # Re-serialised because the stream names above went into `ops` AFTER the
        # config dict was built, so the string the container already holds does
        # not have them. Both writes go to the writer for the same reason.
        w.config["graph"] = json.dumps(ops, separators=(",", ":"))

    w.write(str(out_path))
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"wrote {out_path}  ({size_mb:.1f} MB, arch={ARCH_STRING})")
    print(f"  graph      {len(ops)} ops, {len(conv_of)} convolutions, "
          f"strides {t.strides}, {t.cells} cells, "
          f"{sum(w[0].size for w in weights.values()) / 1e6:.2f} M params")
    print(f"  grids      box offset {t.box_grid_offset}, keypoint offset "
          f"{t.kpt_grid_offset}  (traced from the graph's own grid constants; "
          f"they differ, and one default applied twice would displace every "
          f"joint by half a cell)")
    print(f"  weights    {wdtype.upper()}"
          + (f", int4 group {int4_group or 'per-channel'}"
             if wdtype == "i4" else "")
          + f", {werr / 1e6:.2f} MB of payload"
          + (f", worst relative Frobenius {worst[0]:.3g} ({worst[1]})"
             if wdtype != "f32" else ""))
    if wdtype != "f32":
        print(f"            REPORTED, not gated: this buys container bytes and "
              f"changes the host's arithmetic by the amount above. Nothing "
              f"here measures a detection against fp32 -- "
              f"tools/verify/verify_pose_quant.py does, and there is no "
              f"threshold to fail because there is no accuracy claim to make "
              f"one for.")
        if wdtype == "i4":
            # Said at pack time because the payload is the SMALLER number and
            # the file is the larger one: on this checkpoint i4 group 8 is a
            # 1.64 MB payload in a 4.0 MB container, behind i8's 3.28 MB payload
            # in a 3.8 MB file. An operator comparing --dtype i8 against --dtype
            # i4 --int4-group 8 would otherwise conclude the smaller number is
            # the file.
            print(f"            the {werr / 1e6:.2f} MB above is the WEIGHTS "
                  f"ONLY. The group scales are fp32 and there are G of them per "
                  f"output channel, so a small group can make the file LARGER "
                  f"than i8's -- compare the file size on the first line, not "
                  f"this one.")
    if npu:
        print(f"  array      {len(streams)} distinct padded (K, N) designs, "
              f"tile ({tile_k}, {tile_n}), {len(panels)} pre-tiled panels")
        for (pk, pn) in streams:
            used = [n for n, v in panels.items() if v[2] == pk and v[3] == pn]
            real = sorted({int(weights[n][0][0].size) for n in used})
            print(f"    {stream_name[(pk, pn)]:<14s} {len(used):3d} convolutions, "
                  f"real K in {real}")
