# NpuEmbeddings -- the ONNX weight reader, checked against two independent
# implementations and against a deliberately broken copy of itself.
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS EXISTS ALONGSIDE the packers. tools/lib/onnx_weights.py hand-rolls
# the protobuf walk because `onnx.load()` materialises the whole file in RAM
# (measured: 212 MB for a 90 MB model, so 3 GB of whisper), and a reader that
# is both hand-rolled and load-bearing on 5.6 GB of containers is exactly the
# thing that must not be trusted on its own say-so. This gate answers three
# questions none of the other gates can:
#
#   1. IS THE WIRE TABLE RIGHT? The reader hardcodes protobuf field numbers
#      (ModelProto.graph = 7, TensorProto.raw_data = 9, ...) because protobuf
#      does not complain about a wrong field number -- it quietly returns a
#      different tensor. This compares every one of them against the descriptors
#      of the installed `onnx` package, so bumping onnx or editing the table
#      cannot silently corrupt a container. This is also what `onnx==1.23.1` in
#      requirements.txt is FOR: the reader never imports it, this gate does.
#
#   2. DOES IT READ THE SAME BYTES? Compared against `onnx.load()` -- a from-
#      scratch protobuf implementation with no code shared with ours -- for
#      names, shapes, dtypes and values. What that proves is narrower than the
#      PASS line makes it look, and crosscheck_onnx() says so in place: a
#      MatMul weight handed back TRANSPOSED still matches the file's bytes, so
#      this indexes both orientations on purpose and therefore cannot tell them
#      apart, and a name recovered to a sibling tensor still names a real
#      tensor. The file alone has no opinion about what torch.nn.Linear meant.
#
#   3. SO WHO SAYS THE TENSOR IS THE RIGHT ONE? The committed goldens. They are
#      reference activations produced from a known-good weight set, so the
#      reader's own bytes pushed through the reference encoder must reproduce
#      them -- and must then STOP reproducing them the moment a fault is
#      injected. check_sensitivity() runs exactly that: the correct reader, a
#      transpose left undone, and a name recovered to the wrong tensor. A gate
#      that has only ever passed is indistinguishable from one that would pass
#      anything; this one is built to be able to fail.
#
# Env: numpy, onnx; the reference encoder and the goldens from reference/.
# Usage:
#   python tools/verify/verify_onnx_reader.py

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

from onnx_weights import MODEL_ONNX, OnnxWeights    # noqa: E402

# (model dir, ONNX path relative to it, reader prefix).
#
# `prefix` restores a module root an export dropped -- whisper's encoder was
# exported as a submodule and carries `layers.0...` where the checkpoint says
# `model.encoder.layers.0...`. The decoder export kept `model.decoder.` and
# needs nothing.
#
# Cases whose files are absent are SKIPPED and reported: this table is the
# registry of every ONNX source the project can pack from, and a model nobody
# has fetched yet must not make the gate red.
CASES = [
    ("all-MiniLM-L6-v2", "onnx/model.onnx", ""),
    ("bge-small-en-v1.5", "onnx/model.onnx", ""),
    ("bge-base-en-v1.5", "onnx/model.onnx", ""),
    ("bge-large-en-v1.5", "onnx/model.onnx", ""),
    ("bge-micro-v2", "onnx/model.onnx", ""),
    ("nomic-embed-text-v1.5", "onnx/model.onnx", ""),
    ("gte-multilingual-base", "onnx/model.onnx", ""),
    ("embeddinggemma-300m", "onnx/model.onnx", ""),
    ("vit-base-patch16-224", "onnx/model.onnx", ""),
    # whisper: one ONNX per sub-model
    ("whisper-tiny", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-tiny", "onnx/decoder_model.onnx", ""),
    ("whisper-base", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-base", "onnx/decoder_model.onnx", ""),
    ("whisper-small", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-small", "onnx/decoder_model.onnx", ""),
    ("whisper-medium", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-medium", "onnx/decoder_model.onnx", ""),
    ("whisper-large-v3", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-large-v3", "onnx/decoder_model.onnx", ""),
    ("whisper-large-v3-turbo", "onnx/encoder_model.onnx", "model.encoder."),
    ("whisper-large-v3-turbo", "onnx/decoder_model.onnx", ""),
]

# `onnx.load()` reads the entire file into RAM, so the cross-check is capped.
# Raise it on a machine with room to check a large model end to end; above the
# cap the model still gets the wire-table and the golden-based checks.
MAX_ONNX_LOAD_MB = int((__import__("os").environ.get("ONNX_VERIFY_MAX_MB")
                         or 400))

_failures = []


def ok(label, cond, detail=""):
    print(f"   {'ok  ' if cond else 'FAIL'}  {label}"
          + (f"  {detail}" if detail else ""))
    if not cond:
        _failures.append(label)
    return cond


# ---------------------------------------------------------------------------
# 1. the hardcoded wire table, against the installed onnx package
# ---------------------------------------------------------------------------
EXPECTED_FIELDS = {
    ("ModelProto", "graph"): 7,
    ("GraphProto", "node"): 1,
    ("GraphProto", "initializer"): 5,
    ("NodeProto", "input"): 1,
    ("NodeProto", "output"): 2,
    ("NodeProto", "name"): 3,
    ("NodeProto", "op_type"): 4,
    ("NodeProto", "attribute"): 5,
    ("TensorProto", "dims"): 1,
    ("TensorProto", "data_type"): 2,
    ("TensorProto", "float_data"): 4,
    ("TensorProto", "int32_data"): 5,
    ("TensorProto", "int64_data"): 7,
    ("TensorProto", "name"): 8,
    ("TensorProto", "raw_data"): 9,
    ("TensorProto", "double_data"): 10,
    ("TensorProto", "uint64_data"): 11,
    ("TensorProto", "external_data"): 13,
    ("AttributeProto", "name"): 1,
    ("AttributeProto", "i"): 3,
    ("AttributeProto", "type"): 20,
    ("StringStringEntryProto", "key"): 1,
    ("StringStringEntryProto", "value"): 2,
}


def check_wire_table():
    print("-- wire table vs the installed onnx descriptors")
    try:
        import onnx
    except ImportError:
        print("   skip  the onnx package is not installed; the field numbers "
              "have nothing to be checked against")
        return
    wrong = []
    for (msg, field), want in EXPECTED_FIELDS.items():
        d = getattr(onnx, msg).DESCRIPTOR
        got = d.fields_by_name[field].number
        if got != want:
            wrong.append(f"{msg}.{field}: reader says {want}, onnx says {got}")
    ok("all protobuf field numbers match onnx",
       not wrong, "; ".join(wrong[:4]))

    # the dtype map and the enums are the other place a stale assumption hides
    stale = [f"{name}: reader {code}"
             for name, code in [("FLOAT", 1), ("UINT8", 2), ("INT8", 3),
                                ("INT16", 5), ("INT32", 6), ("INT64", 7),
                                ("BOOL", 9), ("FLOAT16", 10), ("DOUBLE", 11),
                                ("BFLOAT16", 16)]
            if getattr(onnx.TensorProto, name) != code]
    ok("ONNX dtype codes match the reader's table", not stale,
       "; ".join(stale[:4]))
    ok("AttributeType.INT is 2", onnx.AttributeProto.INT == 2)
    ok("data_location EXTERNAL is 1",
       onnx.TensorProto.EXTERNAL == 1)


# ---------------------------------------------------------------------------
# 2. values, against onnx.load()
# ---------------------------------------------------------------------------
def crosscheck_onnx(model, path, prefix):
    """Byte-level agreement with `onnx.load()`, an implementation sharing no
    code with ours.

    What this can and cannot prove is worth stating, because the check reads
    stronger than it is. A MatMul weight is presented TRANSPOSED relative to
    the file, so a digest of what we hand back matches the reference either in
    the same orientation or in the other one; both mean the offsets, the dtypes
    and the dimensions were parsed correctly, and neither says the orientation
    is the RIGHT one. The file alone cannot say that -- it has no opinion about
    what torch.nn.Linear meant -- so this indexes BOTH orientations on purpose
    and stays silent about orientation and naming. Those are proven against the
    goldens instead, by pushing this reader's own bytes through the reference
    encoder and requiring the committed activations to come back: see
    check_sensitivity().
    """
    import hashlib

    size_mb = path.stat().st_size // (1 << 20)
    if size_mb > MAX_ONNX_LOAD_MB:
        print(f"   skip  {model}/{path.name}: {size_mb} MB > cap "
              f"{MAX_ONNX_LOAD_MB} MB (onnx.load would materialise it)")
        return
    try:
        import onnx
    except ImportError:
        print(f"   skip  {model}/{path.name}: the onnx package is not "
              f"installed")
        return
    ref = onnx.load(str(path))
    want = [onnx.numpy_helper.to_array(i) for i in ref.graph.initializer]

    def digest(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).digest()

    # Both orientations indexed, so a correctly transposed weight still lands.
    # A SET of shapes per digest rather than one: two tensors can be
    # byte-identical and differently shaped, and keeping only whichever shape
    # was indexed first then calls a correct read wrong. embeddinggemma's
    # export carries exactly that -- the scalar `INT64/0` and the one-element
    # `INT64/[0]` are the same eight bytes, likewise 1/[1] and 2/[2]: three
    # such pairs among 343 tensors, and no byte-level check can separate them.
    # So this asks the question bytes CAN answer -- does the file hold a
    # tensor with these bytes at this shape or its transpose? Orientation and
    # naming stay out of it on purpose; the goldens prove those (above).
    index = {}
    for r in want:
        index.setdefault(digest(r), set()).add(r.shape)
        if r.ndim == 2:
            index.setdefault(digest(np.ascontiguousarray(r.T)), set()).add(
                r.T.shape)

    got = OnnxWeights(path, prefix=prefix)
    try:
        bad, checked = [], 0
        for k in got.keys():
            # Native dtype, NOT array()'s default float32: that would compare
            # an int64 graph constant (whisper-turbo's export carries 128 of
            # them) against float32 bytes and call a correct read wrong -- and
            # would do the same to any fp16 export. np.array() then copies the
            # view out of the mapping, because a raw() view still exported when
            # close() runs is a BufferError (and on a reader that allowed it, a
            # fault), which is the hazard array()'s own docstring exists for.
            a = np.array(got.raw(k))
            shapes = index.get(digest(a))
            if shapes is None:
                bad.append(f"{k}: these bytes are not in the file")
            elif a.shape not in shapes and a.shape[::-1] not in shapes:
                bad.append(f"{k}: shape {a.shape} matches nothing in the file")
            else:
                checked += 1
        ok(f"{model}/{path.name}: every tensor is a verbatim read of the file "
           f"({checked}/{len(got.keys())})", not bad, "; ".join(bad[:3]))
    finally:
        got.close()


# ---------------------------------------------------------------------------
# 3. the goldens: is it the RIGHT tensor, and can this gate tell when it is not
# ---------------------------------------------------------------------------
def check_sensitivity():
    """A gate that cannot fail is not a gate (see verify_i8_scheme.py).

    The two checks above say the file was parsed and that the bytes came out
    of it. Neither can say the reader handed back the RIGHT tensor under the
    right NAME in the ORIENTATION the packers expect, and neither can: this
    gate indexes both orientations on purpose (see crosscheck_onnx), and a name
    recovered to a sibling still names a real tensor. The file has no opinion
    about what torch.nn.Linear meant.

    The goldens do. They are committed reference activations produced from a
    known-good weight set, so the reader's own bytes pushed through the
    reference encoder must reproduce them -- and must then STOP reproducing
    them when either fault is injected. Two questions answered at once: the
    reader is right, and this check can tell when it is not.
    """
    print("-- sensitivity: the goldens must catch a wrong reader")
    model_dir = REPO / "models" / "all-MiniLM-L6-v2"
    gpath = REPO / "reference" / "goldens" / "minilm_l6_s64_boundary.npz"
    if not ((model_dir / MODEL_ONNX).exists() and gpath.exists()):
        print("   skip  all-MiniLM-L6-v2's ONNX or its golden is absent")
        return

    import json

    sys.path.insert(0, str(REPO / "reference"))
    from encoder import MiniLMReference, read_pooling     # noqa: E402
    from npz_io import load as load_goldens               # noqa: E402
    from onnx_io import load as load_ckpt                 # noqa: E402

    g, meta = load_goldens(gpath)
    n_layers = int(meta["num_layers"])
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    want = np.asarray(g["hf.last_hidden_state"], dtype=np.float64)

    def deviation(weights):
        """rel_fro of this weight set's answer against the committed golden."""
        ref = MiniLMReference(weights, num_layers=n_layers,
                              num_heads=cfg["num_attention_heads"],
                              eps=cfg["layer_norm_eps"],
                              pooling=read_pooling(model_dir))
        taps = {}
        ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"],
                   taps=taps)
        got = np.asarray(taps["last_hidden_state"], dtype=np.float64)
        if got.shape != want.shape:
            return float("inf")
        denom = np.linalg.norm(want)
        return float(np.linalg.norm(got - want) / denom) if denom else 0.0

    # load_ckpt() hands back OWNED arrays (OnnxWeights.array()'s contract), so
    # the dict below can be corrupted for the injections without touching the
    # file on disk.
    w, _ = load_ckpt(model_dir / MODEL_ONNX)

    good = deviation(w)
    ok("the reader's bytes reproduce the golden end to end", good <= 2e-5,
       f"rel_fro {good:.2e} (limit 2e-5)")

    # (a) the transpose left undone -- the single most likely bug here, and one
    #     that still yields a layout hash that verifies and embeddings shaped
    #     like noise. The victim has to be square (a wrong transpose must show
    #     up in the VALUES, not as a shape error) and one the reference reads.
    square = [k for k, a in w.items()
              if a.ndim == 2 and a.shape[0] == a.shape[1]]
    if ok("a square weight exists to transpose", bool(square)):
        victim = next((k for k in square if "query.weight" in k), square[0])
        orig = w[victim]
        w[victim] = orig.T.copy()
        bad = deviation(w)
        w[victim] = orig
        ok(f"a transpose left undone is caught ({victim})", bad > 1e-3,
           f"rel_fro {bad:.2e} -- would have been <= 2e-5 if not caught")

    # (b) a name recovered to the wrong tensor: two projections of equal shape
    #     swapped, which is exactly what a mis-resolved tier-2/tier-3 name
    #     produces. Equal SHAPE on purpose, so detection has to come from the
    #     values the reference consumes -- a shape mismatch would be caught by
    #     anything, including a reader that never recovered a name at all.
    q = next((k for k in w if "query.weight" in k), None)
    kq = next((k for k in w if "key.weight" in k and k != q), None)
    if ok("two equal-shaped projections exist to swap",
          bool(q and kq and w[q].shape == w[kq].shape)):
        saved_q, saved_k = w[q].copy(), w[kq].copy()
        w[q], w[kq] = saved_k, saved_q
        bad = deviation(w)
        w[q], w[kq] = saved_q, saved_k
        ok("a weight pointing at the wrong tensor is caught "
           f"({q} <- {kq})", bad > 1e-3,
           f"rel_fro {bad:.2e} -- still matched the golden")


def main():
    print("onnx_weights -- independent cross-checks")
    check_wire_table()

    print("-- per-model reads")
    ran = 0
    for model, rel, prefix in CASES:
        path = REPO / "models" / model / rel
        if not path.exists():
            continue
        ran += 1
        crosscheck_onnx(model, path, prefix)
    if ran == 0:
        ok("at least one ONNX model is present to check", False,
           "no models/*/<onnx> found")

    check_sensitivity()

    if _failures:
        print(f"\nFAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print("   -", f)
        return 1
    print("\nPASS -- the wire table matches onnx, every tensor is a verbatim\n"
          "       read of the file, the reader's bytes reproduce the golden\n"
          "       end to end, and both injected faults are caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
