# NpuEmbeddings -- a minimal, streaming ONNX weight reader.
#
# WHY THIS EXISTS: the project packs `.npue` from an ONNX checkpoint rather
# than a flat one, and every reason that made the reader this replaces
# hand-rolled applies here -- more so.
#
#   1. The iron env is numpy and nothing else (CLAUDE.md). torch, transformers
#      and onnxruntime are build-time only, and so is `onnx`: importing it pulls
#      in a protobuf runtime, and `onnx.load()` parses the WHOLE file into RAM.
#      Measured on this machine: 212 MB resident for a 90 MB model -- 2.35x the
#      file. whisper-large-v3 is 3.1 GB of checkpoint, so a reader that
#      materialises it is the exact OOM coin-flip a streaming reader was
#      written to avoid. This reader walks the protobuf wire format to FIND
#      each tensor's byte range, then reads it through a memory map: constant
#      memory, nothing copied until array() is called, and the 3 GB of weights
#      are never resident unless somebody asks for them.
#
#   2. The interface is deliberately the established one -- keys(),
#      __contains__(), info(), raw(), array(), close(), context manager -- so a
#      call site swaps the old reader for `OnnxWeights(p)` and nothing else
#      moves. Every tensor is presented under the name AND the orientation the
#      checkpoint used, which is what makes the swap safe: the
#      packers, the verifiers and the goldens keep working unchanged, and the
#      equivalence is testable byte-for-byte against the old checkpoint.
#
# WHAT ONNX COSTS THAT A FLAT DICT DOES NOT
# -----------------------------------------
# A flat container is a dict {name: bytes}. ONNX is a graph, and exporters do
# not keep the checkpoint's names or its layout:
#
#   (a) NAMES DIE. HF's exports rewrite every nn.Linear into MatMul, which
#       orphans the weight: `encoder.layer.0.attention.self.query.weight`
#       becomes `onnx::MatMul_977`. Measured: 36/36 on MiniLM, 72/72 on
#       bge-small, 72/72 on ViT, 60 on whisper's decoder.
#
#   (b) THE WEIGHT IS TRANSPOSED. MatMul computes x @ W and so stores W as
#       [in, out]; torch.nn.Linear stores [out, in]. Byte-compared against the
#       checkpoint: the mangled tensor is exactly `weight.T`. transpose is
#       lossless -- no rounding happens -- but the reader MUST undo it, or every
#       GEMM operand arrives with K and N swapped.
#
#   (c) THE ROOT CAN BE MISSING. whisper's encoder was exported as a submodule,
#       so it carries `layers.0...` where the checkpoint says
#       `model.encoder.layers.0...`. Pass `prefix="model.encoder."`.
#
# Both (a) and (b) are recovered from the GRAPH, never guessed from a name or a
# shape: the generated names carry no information and the shapes are not
# distinctive (24 of the 36 orphaned MiniLM weights are 384x384). _recover()
# runs three tiers, in order; every one of them was verified to 100% on the
# models that were checked:
#
#   tier 1  the initializer already has a real name. This covers whisper's
#           decoder and ViT, and the dynamo exports, which spell Linear weights
#           `...down_proj.MatMul.weight` -- drop the `.MatMul`.
#   tier 2  orphan -> follow the MatMul to the node consuming its output and
#           read the BIAS name, which exporters keep: `.bias` -> `.weight`.
#           36/36 MiniLM, 72/72 bge-small, 72/72 ViT, 48/60 whisper-dec.
#   tier 3  orphan -> take the module path out of the node's own output name,
#           which torch writes as `/model/decoder/layers.0/self_attn/k_proj/
#           MatMul_output_0`: drop the last component, join on `.`, append
#           `.weight`. This rescues whisper's k_proj, whose MatMul feeds a
#           Reshape (GQA) instead of an Add, leaving tier 2 nothing to hold on
#           to -- 6 in the encoder, 12 in the decoder.
#
# Nothing is guessed on failure either: the tensor stays under its generated
# name and a request for the real one raises a KeyError naming the file, which
# is what every gate here expects.
#
# Env: numpy only, plus the standard library.
# Usage:
#   from onnx_weights import OnnxWeights, MODEL_ONNX
#   with OnnxWeights(model_dir / MODEL_ONNX) as w:
#       q = w.array("encoder.layer.0.attention.self.query.weight")

import mmap as _mmap
import os

import numpy as np

# ---------------------------------------------------------------------------
# Where a model's weights live. models/** is gitignored, so this is a
# CONVENTION rather than a cache path: the ONNX is placed into models/<name>/
# by hand (the project deliberately does not fetch it -- see BUILD.md) and
# CHECKPOINT.json pins its digest instead. Three places have to agree on it:
# the Python packers, CHECKPOINT.json's `file` field, and the runtime's
# manifest in runtime/src/common/hub.cpp.
#
# whisper is the exception that makes this a named constant instead of a
# literal: it ships one ONNX per sub-model against a single checkpoint, and
# the encoder's export drops the `model.encoder.` root that the checkpoint's
# tensor names carry (see OnnxWeights.prefix).
# ---------------------------------------------------------------------------
MODEL_ONNX = "onnx/model.onnx"
WHISPER_ENCODER_ONNX = "onnx/encoder_model.onnx"
WHISPER_DECODER_ONNX = "onnx/decoder_model.onnx"

# ---------------------------------------------------------------------------
# How the onnx-community EmbeddingGemma export spells the checkpoint's names.
#
# `strip` alone is not enough for this export: it drops a root the checkpoint
# does not have (see OnnxWeights.strip), but it also RENAMES three families of
# tensors mid-name, and no amount of prefix surgery reaches those. The names
# the packers and encoder_gemma.py ask for are the checkpoint's own inventory
# recorded in tasks/0055 -- that inventory is the ground truth, so the mapping
# belongs here rather than in four separate call sites.
#
#   (".attn.", ".self_attn.")
#       the export shortens the attention module: model.layers.0.attn.q_proj
#       .MatMul.weight where the checkpoint says model.layers.0.self_attn
#       .q_proj.weight. `_recover()`'s tier 1 has already dropped the
#       `.MatMul` by the time this runs.
#   (".layernorm.weight", ".weight")
#       the export wraps each norm in a `.layernorm` submodule: q_norm and
#       k_norm become `...q_norm.layernorm.weight`. This fires on NOTHING
#       else -- the four per-layer norms are spelled `input_layernorm.weight`
#       and friends, where `layernorm` is preceded by `_`, not `.`.
#   ("layers.24.final_norm_layernorm.weight", "norm.weight")
#       the post-stack norm is filed as one more layer: the checkpoint's
#       `model.norm.weight` arrives as `model.layers.24.` + the module path.
#       24 IS num_hidden_layers (one past the last block, layers 0..23) and
#       reference/fetch_model_gemma.py::EXPECT_CONFIG asserts it, so a
#       different Gemma checkpoint fails the lookup loudly instead of packing
#       some other tensor into the final norm's slot.
#
# Both facts have to travel together, so call sites pass both.
EMBEDDINGGEMMA_STRIP = "model."
EMBEDDINGGEMMA_RENAME = (
    (".attn.", ".self_attn."),
    (".layernorm.weight", ".weight"),
    ("layers.24.final_norm_layernorm.weight", "norm.weight"),
)

# ONNX TensorProto.DataType -> (numpy dtype, the .npue tag it maps to).
#
# info() hands back the .npue tag on purpose: callers that switch on
# `dt == "BF16"` or `dt == "F32"` were written for the container reader and
# must not have to know which container they are standing in.
#
# ONNX also has UINT16/UINT32/UINT64. The .npue tag set has no entry for those, so they
# are refused rather than retyped -- inventing a tag would let a mismatched
# tensor through a shape check that exists to catch exactly that.
# ---------------------------------------------------------------------------
ONNX_TO_ST = {
    1: ("<f4", "F32"),     # FLOAT
    2: ("<u1", "U8"),      # UINT8
    3: ("<i1", "I8"),      # INT8
    5: ("<i2", "I16"),     # INT16
    6: ("<i4", "I32"),     # INT32
    7: ("<i8", "I64"),     # INT64
    9: ("?", "BOOL"),      # BOOL
    10: ("<f2", "F16"),    # FLOAT16
    11: ("<f8", "F64"),    # DOUBLE
    16: ("<u2", "BF16"),   # BFLOAT16: no numpy dtype, kept as raw bits
}

# Inlined-data field -> (numpy dtype, .npue tag). Reached only for
# tensors small enough that the exporter wrote typed fields instead of
# raw_data, which is how torch emits scalar constants.
_INLINED = {
    4: ("<f4", "F32"),     # float_data
    5: ("<i4", "I32"),     # int32_data
    7: ("<i8", "I64"),     # int64_data
    10: ("<f8", "F64"),    # double_data
    11: ("<u8", "I64"),    # uint64_data, surfaced as I64 (never a weight)
}


def from_bf16_bits(bits):
    """bf16 bit pattern -> fp32. Exact: bf16 is a strict subset of fp32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Protobuf wire format.
#
# Only what parsing an ONNX file needs, and it is here rather than in the `onnx`
# package for the memory reason in the header: everything below runs against an
# mmap and jumps OVER tensor payloads in O(1), so a 3 GB file costs a few
# thousand byte reads instead of 3 GB of RAM.
#
# Field numbers are not guessed -- they were read off the descriptors of the
# installed onnx package, and tools/verify/verify_onnx_reader.py checks this
# table against `onnx` so a wrong number cannot reach a container. That matters
# because protobuf will not complain about a wrong field number: it would
# quietly return a different tensor.
#
#   ModelProto.graph                    = 7   (length-delimited)
#   GraphProto.node / .initializer      = 1 / 5
#   NodeProto.input/output/op_type/attr = 1 / 2 / 4 / 5
#   TensorProto.dims/data_type/name     = 1 / 2 / 8
#   TensorProto.raw_data/external_data  = 9 / 13
#   AttributeProto.name/i/type          = 1 / 3 / 20
#   StringStringEntryProto.key/value    = 1 / 2
# ---------------------------------------------------------------------------

def _varint(buf, i):
    """LEB128 -> (value, index just past it)."""
    v = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v, i
        shift += 7
        if shift > 63:
            raise ValueError(f"onnx: unterminated varint at byte {i}")


def _fields(buf, start, end):
    """Yield (field_number, wire_type, payload_start, payload_end) in [start, end).

    wire 0 yields the varint's own byte range, so the caller -- who knows the
    field's type -- decodes it with _varint. wire 2 yields the payload with the
    length already applied, which is what makes skipping a 500 MB raw_data
    block free: we read a length and jump.
    """
    i = start
    while i < end:
        key, j = _varint(buf, i)
        no, wt = key >> 3, key & 7
        if wt == 0:
            ps = j
            pe, j = _varint(buf, ps)
        elif wt == 1:
            ps, pe, j = j, j + 8, j + 8
        elif wt == 2:
            n, j = _varint(buf, j)
            ps, pe, j = j, j + n, j + n
        elif wt == 5:
            ps, pe, j = j, j + 4, j + 4
        else:
            raise ValueError(f"onnx: wire type {wt} at byte {i} is not protobuf")
        yield no, wt, ps, pe
        i = j


def _text(buf, ps, pe):
    return buf[ps:pe].decode("utf-8")


def _parse_kv(buf, start, end):
    k = v = None
    for no, wt, ps, pe in _fields(buf, start, end):
        if wt != 2:
            continue
        if no == 1:
            k = _text(buf, ps, pe)
        elif no == 2:
            v = _text(buf, ps, pe)
    return k, v


def _parse_attr(buf, start, end):
    """AttributeProto -> (name, value-if-scalar-INT).

    transA/transB on Gemm are the entire reason attributes are decoded: they say
    whether the weight sitting in the node is already [out, in]. Graphs,
    tensors, floats and strings are skipped -- none of them can change an answer
    this reader gives.
    """
    name, ival, is_int = None, 0, False
    for no, wt, ps, pe in _fields(buf, start, end):
        if no == 1 and wt == 2:
            name = _text(buf, ps, pe)
        elif no == 3 and wt == 0:
            ival = _varint(buf, ps)[0]
            if ival >= 1 << 63:            # int64 arrives as 64-bit two's complement
                ival -= 1 << 64
            is_int = True
        elif no == 20 and wt == 0:
            is_int = _varint(buf, ps)[0] == 2   # AttributeType.INT == 2
    return name, (ival if is_int else None)


def _parse_node(buf, start, end):
    node = {"inputs": [], "outputs": [], "op": "", "attrs": {}}
    for no, wt, ps, pe in _fields(buf, start, end):
        if no == 1 and wt == 2:
            node["inputs"].append(_text(buf, ps, pe))
        elif no == 2 and wt == 2:
            node["outputs"].append(_text(buf, ps, pe))
        elif no == 4 and wt == 2:
            node["op"] = _text(buf, ps, pe)
        elif no == 5 and wt == 2:
            k, v = _parse_attr(buf, ps, pe)
            if k is not None and v is not None:
                node["attrs"][k] = v
    return node


def _parse_tensor(buf, start, end):
    """TensorProto -> dict with name, dims, data_type and a `loc` descriptor.

    `loc` is where the bytes are, in one of three shapes:
      ("raw",   offset, length)              -- inside this file
      ("ext",   path, offset, length)        -- ONNX external data, another file
      ("inline", field_no, wire, ps, pe)     -- typed fields, not raw_data
    """
    t = {"name": "", "dims": [], "dtype": 0, "loc": None, "ext": {}}
    for no, wt, ps, pe in _fields(buf, start, end):
        if no == 1:                                   # dims: packed or not
            if wt == 0:
                t["dims"].append(_varint(buf, ps)[0])
            else:
                j = ps
                while j < pe:
                    v, j = _varint(buf, j)
                    t["dims"].append(v)
        elif no == 2 and wt == 0:
            t["dtype"] = _varint(buf, ps)[0]
        elif no == 8 and wt == 2:
            t["name"] = _text(buf, ps, pe)
        elif no == 9 and wt == 2:
            t["loc"] = ("raw", ps, pe - ps)
        elif no == 13 and wt == 2:
            k, v = _parse_kv(buf, ps, pe)
            if k is not None:
                t["ext"][k] = v
        elif no in _INLINED and t["loc"] is None:
            t["loc"] = ("inline", no, wt, ps, pe)
    # An external-data path in the graph is RELATIVE to the graph file, and
    # this function does not know where that is -- `_build()` resolves it
    # against the real base directory. Leaving `loc` None here is what makes
    # that happen; resolving it here as well would produce a path relative to
    # the current working directory, which silently reads nothing (or the
    # wrong file) for every model that keeps its weights in a side file.
    return t


# ---------------------------------------------------------------------------
# Name recovery and orientation.
# ---------------------------------------------------------------------------

def _recover(name, uses):
    """(recovered_checkpoint_name, how) for an initializer, or (name, None).

    `uses` are the nodes taking `name` as input. See the module header for the
    three tiers and the evidence behind them.
    """
    if not name.startswith("onnx::"):
        # tier 1: dynamo-style exports spell Linear weights `...proj.MatMul.weight`
        if ".MatMul." in name:
            return name.replace(".MatMul.", ".", 1), "tier1-drop-.MatMul"
        return name, None

    for node in uses:
        if node["op"] not in ("MatMul", "Gemm"):
            continue
        # tier 2: the bias paired with this weight keeps its real name
        for other in node.get("out_consumers", ()):
            if other["op"] != "Add":
                continue
            for cand in other["inputs"]:
                if cand.endswith(".bias") and cand != name:
                    return cand[:-5] + ".weight", "tier2-bias-anchor"
        # tier 3: `/a/b/c/MatMul_output_0` -> `a.b.c.weight`
        if node["outputs"]:
            out = node["outputs"][0]
            parts = [p for p in out.split("/") if p]
            if out.startswith("/") and len(parts) >= 2 and \
                    not parts[-1].startswith("onnx::"):
                return ".".join(parts[:-1]) + ".weight", "tier3-node-path"
    return name, None


def _transpose_for(uses, orig, dims, bias0):
    """True when the stored tensor is [in, out] and must become [out, in].

    Decided by how the graph USES the tensor, because that is the only thing
    that says what the layout is: MatMul's second operand is [in, out] by
    definition of the op, Gemm states it in transA/transB, and a Gather operand
    is an embedding table already laid out [vocab, dim] == [out, in].

    When the uses disagree -- what a tied lm_head produces, one initializer
    serving both a lookup and the logits MatMul -- the bias settles it, because
    bias.shape[0] IS out. With no bias and no agreement we refuse to guess: a
    wrong answer here builds a container that passes its own layout hash and
    emits embeddings that look like noise, which is the worst failure mode this
    project has.
    """
    if len(dims) != 2:
        return False

    flags = set()
    for node in uses:
        op = node["op"]
        if op == "MatMul":
            if orig in node["inputs"]:
                flags.add(node["inputs"].index(orig) == 1)
        elif op == "Gemm":
            flags.add(not bool(node["attrs"].get("transB", 0)))
        elif op == "Gather":
            flags.add(False)

    if len(flags) == 1:
        return flags.pop()
    if len(flags) > 1:
        if bias0 is None:
            raise ValueError(
                f"onnx: layout of {orig!r} is ambiguous (MatMul, Gemm and "
                f"Gather all consume it) and there is no bias to settle it; "
                f"refusing to guess")
        return dims[0] != bias0
    if bias0 is not None:
        if dims[0] == bias0:
            return False
        if dims[1] == bias0:
            return True
        raise ValueError(
            f"onnx: {orig!r} has shape {dims} but its bias has {bias0}; "
            f"neither orientation matches")
    return False


class OnnxWeights:
    """Read-only, mmap-backed view of an ONNX model, under checkpoint names.

    Same contract as the reader this replaces: nothing is copied until
    array() is called, array() copies exactly one tensor, and close() releases
    the mapping -- which matters because a dict of views into a closed mapping
    faults on first read rather than raising.

    `prefix` is prepended to every recovered name, `strip` removed from the
    front of it. Both exist for one reason: the packers ask for the names THE
    CHECKPOINT used, and exporters disagree with it in both directions.

      prefix="model.encoder."   whisper's encoder was exported as a submodule,
                                so it ships `layers.0...` where the checkpoint
                                says `model.encoder.layers.0...`. The export
                                dropped a root.
      strip="model."            embeddinggemma's export keeps `model.` where
                                the checkpoint -- and therefore
                                encoder_gemma.py, which is the verified
                                ground truth -- has none (documented at
                                encoder_gemma.py:208). The export added a root.

    `rename` is the third disagreement, one prefix surgery cannot express: a
    sequence of (old, new) pairs, each applied with str.replace to the name
    AFTER prefix/strip have had their say. embeddinggemma needs it for three
    families of mid-name rewrites; see EMBEDDINGGEMMA_RENAME, which is what
    every caller passes alongside EMBEDDINGGEMMA_STRIP. Nothing is applied
    implicitly -- a caller that forgets it gets the export's spelling back and
    a KeyError naming the file, which is the same failure the reader has
    always produced for a name it does not know.

    A strip or a rename that makes two tensors collide is refused by _build()
    rather than resolved, because silently keeping one would pack one weight
    into the other's slot.
    """

    def __init__(self, path, prefix="", strip="", rename=()):
        self.path = str(path)
        self.prefix = prefix
        self.strip = strip
        self.rename = tuple(rename)
        self._ext = {}
        self._tensors = {}
        self._order = []
        with open(self.path, "rb") as f:
            self._map = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        try:
            self._build()
        except Exception:
            self.close()
            raise

    # -- construction -------------------------------------------------------

    def _build(self):
        buf = self._map
        g_start = g_end = None
        for no, wt, ps, pe in _fields(buf, 0, len(buf)):
            if no == 7 and wt == 2:                    # ModelProto.graph
                g_start, g_end = ps, pe
                break
        if g_start is None:
            raise ValueError(f"{self.path}: no graph in this ONNX file")

        nodes, inits = [], []
        for no, wt, ps, pe in _fields(buf, g_start, g_end):
            if wt != 2:
                continue
            if no == 1:
                nodes.append(_parse_node(buf, ps, pe))
            elif no == 5:
                inits.append(_parse_tensor(buf, ps, pe))

        # Two walks over the graph:
        #   by_input      initializer name -> nodes taking it. An initializer
        #                 uses this to find the ops that consume it, which is
        #                 how both the name (tier 2/3) and the layout are read.
        #   out_consumers node -> nodes taking ITS OUTPUT. Tier 2 walks one hop
        #                 forward from a MatMul to the Add carrying the bias
        #                 that pairs with the weight; without this direction it
        #                 reaches backwards to whatever produced the input and
        #                 recovers an unrelated name (it once returned
        #                 embeddings.LayerNorm.weight for three orphaned
        #                 attention weights, which the collision check caught).
        by_input = {}
        for n in nodes:
            for i in n["inputs"]:
                by_input.setdefault(i, []).append(n)
        for n in nodes:
            seen, seen_ids = [], set()
            for o in n["outputs"]:
                for c in by_input.get(o, ()):
                    if c is not n and id(c) not in seen_ids:
                        seen_ids.add(id(c))
                        seen.append(c)
            n["out_consumers"] = seen

        base = os.path.dirname(os.path.abspath(self.path))
        named = {}
        for t in inits:
            if not t["name"]:
                continue
            t["orig"] = t["name"]
            if t["loc"] is None and t["ext"]:
                t["loc"] = ("ext",
                            os.path.join(base, t["ext"].get("location", "")),
                            int(t["ext"].get("offset", 0)),
                            int(t["ext"].get("length", 0)))
            uses = by_input.get(t["name"], ())
            fixed, how = _recover(t["name"], uses)
            if self.strip and fixed.startswith(self.strip):
                fixed = fixed[len(self.strip):]
            fixed = self.prefix + fixed
            for old, new in self.rename:
                fixed = fixed.replace(old, new)
            t["how"] = how
            named.setdefault(fixed, []).append(t)
            t["_uses"] = uses

        # pass 2: orientation, now that every sibling bias is addressable
        for fixed, group in named.items():
            if len(group) != 1:
                raise ValueError(
                    f"{self.path}: {len(group)} tensors recover to {fixed!r}: "
                    f"{[t['orig'] for t in group]}")
            t = group[0]
            bias0 = None
            if fixed.endswith(".weight") and len(t["dims"]) == 2:
                sib = named.get(fixed[:-7] + ".bias")
                if sib and len(sib[0]["dims"]) == 1:
                    bias0 = sib[0]["dims"][0]
            t["transpose"] = _transpose_for(t["_uses"], t["orig"],
                                            t["dims"], bias0)
            shown = t["dims"][::-1] if t["transpose"] else t["dims"]
            if bias0 is not None and len(shown) == 2 and shown[0] != bias0:
                raise ValueError(
                    f"{self.path}: {t['orig']!r} reads as {shown} but its bias "
                    f"has {bias0} elements; orientation and the checkpoint "
                    f"disagree, refusing to pack")
            self._tensors[fixed] = t
            self._order.append(fixed)

    # -- reading ------------------------------------------------------------

    def _mmap_for(self, src):
        if src not in self._ext:
            with open(src, "rb") as f:
                self._ext[src] = _mmap.mmap(f.fileno(), 0,
                                            access=_mmap.ACCESS_READ)
        return self._ext[src]

    def _get(self, name):
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None

    def keys(self):
        return list(self._order)

    def __contains__(self, name):
        return name in self._tensors

    def info(self, name):
        """(dtype tag, shape, [begin, end) byte range in the owning file).

        The shape is what array() returns, not necessarily the order the bytes
        are stored in -- see (b) in the module header.
        """
        t = self._get(name)
        dims = t["dims"][::-1] if t["transpose"] else list(t["dims"])
        return _tag_of(t), dims, _range_of(t)

    def raw(self, name):
        """The tensor's bytes as a read-only view. No copy.

        May be a non-contiguous view when the weight had to be transposed back
        to [out, in]; use array() where a contiguous buffer is required.
        """
        t = self._get(name)
        tag = _tag_of(t)
        dt = np.dtype("<u2") if tag == "BF16" else np.dtype(ONNX_TO_ST[t["dtype"]][0])
        view = self._read(t, dt)
        return view.T if t["transpose"] else view

    def array(self, name, dtype=np.float32):
        """One tensor, widened to `dtype`. The only allocation this module makes.

        fp16 is the interesting case: the Whisper checkpoints are fp16, and a
        weight already rounded to fp16 must be widened to fp32 BEFORE it is
        rounded again to bf16, or the second rounding compounds the first.

        THE RETURN VALUE IS OWNED, never a view into the mapping.

        The flat-file reader this replaces returned a view for a same-dtype read
        and documents
        that the caller must copy before close(), because close() UNMAPS the
        memory out from under a live array and the next read faults -- which is
        a real hazard: packers hold `st.array(...)` results across close(). This
        reader refuses to hand out a view from array(), so close() cannot be
        reached with the mmap's buffer still exported (it raises BufferError
        instead of silently unmapping) and no caller can segfault. One copy per
        tensor is the price, array() already allocates for every dtype change,
        and raw() remains available for anyone who genuinely wants zero copy.

        fp16 is the interesting case: the Whisper checkpoints are fp16, and a
        weight already rounded to fp16 must be widened to fp32 BEFORE it is
        rounded again to bf16, or the second rounding compounds the first.
        """
        t = self._get(name)
        if _tag_of(t) == "BF16":
            return from_bf16_bits(self.raw(name)).astype(dtype)
        return np.array(self.raw(name), dtype=dtype)

    def _read(self, t, dt):
        loc = t["loc"]
        if loc is None:
            raise ValueError(f"{self.path}: {t['orig']!r} carries no data")
        # The element count comes from the SHAPE, never from the byte length:
        # a truncated or padded payload then fails loudly in frombuffer instead
        # of silently reshaping into the wrong tensor.
        count = 1
        for d in t["dims"]:
            count *= int(d)
        kind = loc[0]
        if kind == "raw":
            mm, off = self._map, loc[1]
        elif kind == "ext":
            mm, off = self._mmap_for(loc[1]), loc[2]
        else:                                          # inline typed fields
            return self._read_inlined(t, loc, dt)
        return np.frombuffer(mm, dtype=dt, count=count, offset=off).reshape(t["dims"])

    def _read_inlined(self, t, loc, dt):
        _, no, wt, ps, pe = loc
        d = np.dtype(_INLINED[no][0])
        buf = self._map
        if wt == 2:                                    # packed
            flat = np.frombuffer(buf, dtype=d, count=(pe - ps) // d.itemsize,
                                 offset=ps)
        elif wt == 0:                                  # single varint
            v, _ = _varint(buf, ps)
            flat = np.array([v], dtype=d)
        else:                                          # 32-/64-bit fixed
            flat = np.frombuffer(buf, dtype=d, count=1, offset=ps)
        return flat.reshape(t["dims"])

    def source_files(self):
        """Every file whose BYTES this reader can hand back, graph first.

        An ONNX model may keep its weights in a side file (`external_data`),
        and the ones that do are the large ones -- whisper-large-v3's graph is
        0.7 MB and the data it points at is 2.5 GB. Hashing only the graph,
        which is what a plain `sha256(model.onnx)` would do, then pins the
        tensor names and none of the WEIGHTS: swap the checkpoint out under an
        unchanged graph and the pin verifies clean. This is the list
        `model_digest()` walks instead.
        """
        out = [self.path]
        seen = {os.path.abspath(self.path)}
        for t in self._tensors.values():
            loc = t.get("loc")
            if not loc or loc[0] != "ext":
                continue
            ap = os.path.abspath(loc[1])
            if ap not in seen:
                seen.add(ap)
                out.append(loc[1])
        return out

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        for m in list(self._ext.values()):
            m.close()
        self._ext.clear()
        if self._map is not None:
            self._map.close()
            self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Combined:
    """Two or more OnnxWeights behind one shared read-only surface.

    whisper ships its encoder and its decoder as separate ONNX files against a
    single checkpoint, and pack_whisper() reads both inside one function: it
    asks for `model.encoder.conv1.weight` a handful of lines before
    `model.decoder.embed_tokens.weight`. Rather than make the packer track two
    handles and choose correctly at sixty call sites, each file is opened with
    the prefix that restores its root -- the encoder export dropped
    `model.encoder.`, the decoder export kept `model.decoder.` -- and this
    presents them as one namespace.

    A name found in two readers is REFUSED, not resolved. It can only mean the
    prefixes did not do their job, and silently picking one would pack one
    sub-model's weights into the other's slots -- a container that builds,
    hashes and produces wrong embeddings.
    """

    def __init__(self, readers):
        self._readers = list(readers)
        if not self._readers:
            raise ValueError("Combined needs at least one reader")
        self._where = {}
        for r in self._readers:
            for k in r.keys():
                if k in self._where:
                    raise ValueError(
                        f"{k!r} resolves in both ONNX files "
                        f"({self._where[k].path} and {r.path}); the prefixes "
                        f"were meant to keep the two sub-models apart")
                self._where[k] = r

    def keys(self):
        return list(self._where)

    def __contains__(self, name):
        return name in self._where

    def _r(self, name):
        try:
            return self._where[name]
        except KeyError:
            raise KeyError(
                f"no tensor named {name!r} in "
                + ", ".join(r.path for r in self._readers)) from None

    def info(self, name):
        return self._r(name).info(name)

    def raw(self, name):
        return self._r(name).raw(name)

    def array(self, name, dtype=np.float32):
        return self._r(name).array(name, dtype)

    def source_files(self):
        """Every file behind all of the readers, in reader order."""
        out, seen = [], set()
        for r in self._readers:
            for p in r.source_files():
                ap = os.path.abspath(p)
                if ap not in seen:
                    seen.add(ap)
                    out.append(p)
        return out

    def close(self):
        for r in self._readers:
            r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _tag_of(t):
    if t["dtype"] not in ONNX_TO_ST:
        raise ValueError(f"onnx: tensor {t['orig']!r} has data_type "
                         f"{t['dtype']}, which this container has no tag for")
    return ONNX_TO_ST[t["dtype"]][1]


def _range_of(t):
    loc = t["loc"]
    if loc is None:
        return 0, 0
    if loc[0] == "raw":
        return loc[1], loc[1] + loc[2]
    if loc[0] == "ext":
        return loc[2], loc[2] + loc[3]
    return loc[3], loc[4]


def sha256_file(path, chunk=1 << 20):
    """Digest of one file, read once and never held in memory."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def model_digest(*paths):
    """One digest identifying an ONNX model: its graph and everything it holds.

    Every ONNX path that names "the checkpoint" -- `source_sha256` in the
    .npue, `sha256` in CHECKPOINT.json, the pin the gate compares -- passes
    through here, so there is one answer to "is this the same weights?".

    The digest covers each of the model's source files as that file's own
    digest together with its BASENAME, in a fixed order:

      * each file's own digest, so a 2.5 GB side file costs 2.5 GB of reading
        and then 32 bytes of combining rather than being read a second time;
      * the basename with it, because two checkpoints can be byte-identical
        under different names and that is not the same checkpoint -- the
        external-data location is part of the model's layout, and a reader
        that resolves `encoder_model.onnx_data` when the graph asks for a
        different name has not read this model;
      * a fixed order, because a set that hashes differently depending on how
        it was enumerated is not a fingerprint of anything.

    It is deliberately NOT the digest of any single file, so it cannot be
    mistaken for one: `sha256_file()` is still there for the plain question
    "is this file this file?".
    """
    import hashlib
    files = []
    for p in paths:
        with OnnxWeights(p) as w:
            for f in w.source_files():
                if f not in files:
                    files.append(f)
    outer = hashlib.sha256()
    for p in files:
        outer.update(os.path.basename(p).encode("utf-8"))
        outer.update(b"\0")
        outer.update(sha256_file(p).encode("ascii"))
        outer.update(b"\n")
    return outer.hexdigest()
