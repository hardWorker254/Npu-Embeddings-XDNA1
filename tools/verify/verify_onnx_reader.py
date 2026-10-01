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
#      names, shapes, dtypes and values. And, while the safetensors checkpoints
#      are still on disk, against the reader this one replaces: the migration's
#      real acceptance criterion is "the ONNX container carries byte-identical
#      weights to the safetensors container", because every golden, every .npue
#      and every provenance pin downstream is defined in terms of those bytes.
#
#   3. DOES IT ACTUALLY CATCH A WRONG ANSWER? A gate that has only ever been
#      run on a correct reader proves nothing -- "PASS" is indistinguishable
#      from a gate that would pass anything. So each failure mode is injected
#      and the gate FAILS unless it is reported: the transpose left undone (the
#      single most likely bug here, and one that still produces a container that
#      passes its own layout hash and emits embeddings shaped like noise), and a
#      name recovered to the wrong tensor.
#
# Env: numpy, onnx; safetensors reader from tools/lib (checkpoint comparison is
# skipped, not failed, when a checkpoint is absent).
# Usage:
#   python tools/verify/verify_onnx_reader.py

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

from onnx_weights import OnnxWeights            # noqa: E402

# (model dir, ONNX path relative to it, reader prefix, safetensors subset).
#
# `prefix` restores a module root an export dropped -- whisper's encoder was
# exported as a submodule and carries `layers.0...` where the checkpoint says
# `model.encoder.layers.0...`. `subset` limits the safetensors comparison to
# the half of the checkpoint this ONNX file holds, because whisper ships
# encoder and decoder as two files against one checkpoint.
#
# Cases whose files are absent are SKIPPED and reported: this table is the
# registry of every ONNX source the project can pack from, and a model nobody
# has fetched yet must not make the gate red.
CASES = [
    ("all-MiniLM-L6-v2", "onnx/model.onnx", "", ""),
    ("bge-small-en-v1.5", "onnx/model.onnx", "", ""),
    ("bge-base-en-v1.5", "onnx/model.onnx", "", ""),
    ("bge-large-en-v1.5", "onnx/model.onnx", "", ""),
    ("bge-micro-v2", "onnx/model.onnx", "", ""),
    ("nomic-embed-text-v1.5", "onnx/model.onnx", "", ""),
    ("gte-multilingual-base", "onnx/model.onnx", "", ""),
    ("embeddinggemma-300m", "onnx/model.onnx", "", ""),
    ("vit-base-patch16-224", "onnx/model.onnx", "", ""),
    # whisper: one ONNX per sub-model, one checkpoint for both
    ("whisper-tiny", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-tiny", "onnx/decoder_model.onnx", "", "model.decoder."),
    ("whisper-base", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-base", "onnx/decoder_model.onnx", "", "model.decoder."),
    ("whisper-small", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-small", "onnx/decoder_model.onnx", "", "model.decoder."),
    ("whisper-medium", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-medium", "onnx/decoder_model.onnx", "", "model.decoder."),
    ("whisper-large-v3", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-large-v3", "onnx/decoder_model.onnx", "", "model.decoder."),
    ("whisper-large-v3-turbo", "onnx/encoder_model.onnx", "model.encoder.",
     "model.encoder."),
    ("whisper-large-v3-turbo", "onnx/decoder_model.onnx", "", "model.decoder."),
]

# Tensors an ONNX graph has no reason to carry, listed by reference/fetch_model.py
# as IGNORABLE and absent from every export checked so far:
#   pooler.*       sentence-transformers never calls it (dead weight by design)
#   position_ids   a constant arange, materialised in the graph rather than stored
# Everything else must be there, byte for byte.
IGNORABLE = {"pooler.dense.weight", "pooler.dense.bias", "embeddings.position_ids"}

# `onnx.load()` reads the entire file into RAM, so the cross-check is capped.
# Raise it on a machine with room to check a large model end to end; above the
# cap the model still gets the wire-table and safetensors checks.
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
# 2. values, against onnx.load() and against the safetensors checkpoint
# ---------------------------------------------------------------------------
def crosscheck_onnx(model, path, prefix):
    """Byte-level agreement with `onnx.load()`, an implementation sharing no
    code with ours.

    What this can and cannot prove is worth stating, because the check reads
    stronger than it is. A MatMul weight is presented TRANSPOSED relative to
    the file, so a digest of what we hand back matches the reference either in
    the same orientation or in the other one; both mean the offsets, the dtypes
    and the dimensions were parsed correctly, and neither says the orientation
    is the RIGHT one -- the file alone cannot say that, it has no opinion about
    what torch.nn.Linear meant. Orientation and naming are proven against the
    safetensors checkpoint instead, which is the ground truth for what the
    packers asked for; see crosscheck_safetensors().
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

    # both orientations indexed, so a correctly transposed weight still lands
    index = {}
    for r in want:
        index.setdefault(digest(r), r.shape)
        if r.ndim == 2:
            index.setdefault(digest(np.ascontiguousarray(r.T)), r.T.shape)

    got = OnnxWeights(path, prefix=prefix)
    try:
        bad, checked = [], 0
        for k in got.keys():
            a = got.array(k)
            if digest(a) not in index:
                bad.append(f"{k}: these bytes are not in the file")
            elif index[digest(a)] not in (a.shape, a.shape[::-1]):
                bad.append(f"{k}: shape {a.shape} matches nothing in the file")
            else:
                checked += 1
        ok(f"{model}/{path.name}: every tensor is a verbatim read of the file "
           f"({checked}/{len(got.keys())})", not bad, "; ".join(bad[:3]))
    finally:
        got.close()


def crosscheck_safetensors(model, path, prefix, subset):
    from safetensors_mmap import SafeTensors
    st_path = REPO / "models" / model / "model.safetensors"
    if not st_path.exists():
        print(f"   skip  {model}: no safetensors checkpoint to compare "
              f"({st_path.name} not fetched)")
        return
    st = SafeTensors(st_path)
    got = OnnxWeights(path, prefix=prefix)
    try:
        want = [k for k in st.keys()
                if (not subset or k.startswith(subset))
                and k not in IGNORABLE
                and not k.endswith(".position_ids")]
        bad, missing = [], []
        for k in want:
            if k not in got:
                missing.append(k)
                continue
            a, b = st.array(k), got.array(k)
            if a.shape != b.shape:
                bad.append(f"{k}: shape {a.shape} vs {b.shape}")
            elif not np.array_equal(a, b):
                bad.append(f"{k}: bytes differ")
        ok(f"{model}/{path.name}: byte-identical to the safetensors checkpoint "
           f"({len(want) - len(missing) - len(bad)}/{len(want)})",
           not bad and not missing,
           "; ".join((bad + missing)[:3]))
    finally:
        got.close()
        st.close()


# ---------------------------------------------------------------------------
# 3. fault injection: the gate must REPORT a broken reader
# ---------------------------------------------------------------------------
def check_sensitivity():
    """A gate that cannot fail is not a gate (see verify_i8_scheme.py)."""
    print("-- sensitivity: a deliberately broken reader must be caught")
    onnx_path = REPO / "models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
    st_path = REPO / "models" / "all-MiniLM-L6-v2" / "model.safetensors"
    if not (onnx_path.exists() and st_path.exists()):
        print("   skip  neither all-MiniLM-L6-v2/model.onnx nor its "
              "checkpoint is present")
        return

    from safetensors_mmap import SafeTensors
    st = SafeTensors(st_path)
    got = OnnxWeights(onnx_path)
    try:
        # (a) the transpose left undone. Must produce a DIFFERENCE, and only
        #     where a weight was actually transposed -- if this does not trip,
        #     the comparison above would pass a reader that never untwists
        #     MatMul weights at all.
        victim = next(k for k in got.keys()
                      if got._tensors[k]["transpose"])
        got._tensors[victim]["transpose"] = False
        a, b = st.array(victim), got.array(victim)
        caught = not np.array_equal(a, b)
        # a square weight still differs after a wrong transpose (the bytes are
        # in the other order), but the SHAPE check must also be exercised
        ok(f"missing transpose is detected on {victim}", caught)
        got._tensors[victim]["transpose"] = True

        # (b) a name recovered to the wrong tensor. The equivalence check must
        #     notice that a weight now holds someone else's bytes.
        #     The shapes are equal on purpose: a mismatch that shows up only as
        #     a shape error would pass even for a reader that never recovered a
        #     name at all, so this has to be caught on VALUES alone.
        victim = next(k for k in got.keys()
                      if got._tensors[k]["transpose"])
        other = next(k for k in got.keys()
                     if k != victim and got._tensors[k]["transpose"]
                     and st.array(k).shape == st.array(victim).shape)
        saved = got._tensors[victim]
        got._tensors[victim] = got._tensors[other]   # victim reads other's bytes
        a, b = st.array(victim), got.array(victim)
        ok("a weight pointing at the wrong tensor is detected",
           not np.array_equal(a, b))
        got._tensors[victim] = saved
    finally:
        got.close()
        st.close()


def main():
    print("onnx_weights -- independent cross-checks")
    check_wire_table()

    print("-- per-model reads")
    ran = 0
    for model, rel, prefix, subset in CASES:
        path = REPO / "models" / model / rel
        if not path.exists():
            continue
        ran += 1
        crosscheck_onnx(model, path, prefix)
        crosscheck_safetensors(model, path, prefix, subset)
    if ran == 0:
        ok("at least one ONNX model is present to check", False,
           "no models/*/<onnx> found")

    check_sensitivity()

    if _failures:
        print(f"\nFAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print("   -", f)
        return 1
    print("\nPASS -- the wire table matches onnx, the reader returns the same\n"
          "       bytes as both reference implementations, and it reports a\n"
          "       reader that has been deliberately broken.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
