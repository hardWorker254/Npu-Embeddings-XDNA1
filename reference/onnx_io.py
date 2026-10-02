# NpuEmbeddings -- read an ONNX checkpoint the way this file's predecessor
# read a flat weight file.
#
# reference/ is the oracle: every golden, and every comparison of a kernel
# against the thing it is supposed to compute, traces back to weights loaded
# here. It therefore has to run with numpy and nothing else -- the iron env
# where the NPU work happens carries no torch and no transformers -- and
# tools/lib/onnx_weights.py is numpy plus the standard library, so this stays
# inside that boundary. (It walks the protobuf wire format itself rather than
# importing onnx, both because onnx is not in the iron env and because
# onnx.load() materialises the whole file: 212 MB resident for a 90 MB model,
# measured, which is the OOM coin-flip the previous reader was written to
# avoid. The field numbers it hardcodes are checked against the installed
# onnx package by tools/verify/verify_onnx_reader.py.)
#
# The signature is its predecessor load()'s on purpose: ({name: ndarray},
# metadata). ONNX carries no metadata block, so the second element is always
# {} -- and every call site in this tree already writes `w, _ = load(...)`,
# discarding it, so nothing downstream had to change shape.
#
# Env: numpy only.
# Usage:
#   from onnx_io import MODEL_ONNX, load
#   w, _ = load(model_dir / MODEL_ONNX)

import sys
from pathlib import Path

# Self-contained: make_goldens.py used to insert tools/ (not tools/lib) and
# then failed on `from npue import golden_slug` with ModuleNotFoundError, so
# the file had not run since npue.py moved. A module that fixes its own path
# cannot regress that way again.
_LIB = Path(__file__).resolve().parent.parent / "tools" / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from onnx_weights import (EMBEDDINGGEMMA_RENAME,             # noqa: E402
                          EMBEDDINGGEMMA_STRIP, MODEL_ONNX,
                          WHISPER_DECODER_ONNX, WHISPER_ENCODER_ONNX,
                          OnnxWeights, model_digest, sha256_file)

__all__ = ["EMBEDDINGGEMMA_RENAME", "EMBEDDINGGEMMA_STRIP", "MODEL_ONNX",
           "WHISPER_ENCODER_ONNX", "WHISPER_DECODER_ONNX",
           "checkpoint_digest", "checkpoint_files", "load", "model_digest",
           "reader", "sha256_file"]


def reader(model_dir, prefix="", strip="", rename=()):
    """An OPEN OnnxWeights for a checkpoint directory, for callers that want
    to stream rather than materialise. The caller owns the handle and must
    close it (or use it as a context manager)."""
    return OnnxWeights(Path(model_dir) / MODEL_ONNX, prefix=prefix,
                       strip=strip, rename=rename)


def load(path, strip="", rename=()):
    """Read one ONNX file into ({name: ndarray}, {}).

    Materialises every tensor, which is what the previous reader did and
    what these callers expect: they hold the dict across a whole reference run
    and index it by name. OnnxWeights.array() returns an OWNED array (see its
    docstring), so the dict survives close() -- no copy is needed on the way
    out, which matters at gemma's 1.2 GB.

    `strip` is a root the export ADDED and the checkpoint does not have; only
    embeddinggemma needs it, and it needs `rename` with it for the three
    mid-name rewrites prefix surgery cannot reach. Pass both together as
    EMBEDDINGGEMMA_STRIP / EMBEDDINGGEMMA_RENAME. See OnnxWeights for the
    other direction (a root the export dropped).
    """
    w = OnnxWeights(path, strip=strip, rename=rename)
    try:
        return {k: w.array(k) for k in w.keys()}, {}
    finally:
        w.close()


def checkpoint_files(model_dir):
    """Every file a checkpoint's BYTES live in, relative to its own directory.

    The ONNX graph file(s) -- one for most models, encoder plus decoder for
    whisper -- plus any external-data side file those graphs point at. This is
    exactly the list `model_digest()` hashes, so CHECKPOINT.json's `file` and
    its `sha256` always describe the same set; a pin that named one file and
    hashed another would not be a pin of anything.

    An incomplete checkpoint reports NOTHING: a graph whose side file has not
    arrived carries no weights anyone can read, and listing it would let
    `file` promise bytes `sha256` refuses to cover. The pair is either both
    set or both empty.
    """
    d = Path(model_dir)
    try:
        root = d.resolve()
    except OSError:
        root = d.absolute()
    out, seen = [], set()
    for g in (d / MODEL_ONNX, d / WHISPER_ENCODER_ONNX, d / WHISPER_DECODER_ONNX):
        if not g.exists():
            continue
        with OnnxWeights(g) as w:
            found = []
            for f in w.source_files():
                p = Path(f)
                # `source_files()` hands back the graph exactly as it was
                # opened (possibly relative to wherever the caller was
                # standing) and each side file already absolutised, so both
                # have to be resolved before they can be made relative to the
                # model directory -- otherwise the list mixes two roots and
                # reads as if the weights lived in two places.
                if not p.is_absolute():
                    p = p.resolve()
                if not p.exists():
                    found = None
                    break
                rel = str(p.relative_to(root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)
            if found is None:
                return []
            out.extend(found)
    return out


def checkpoint_digest(model_dir):
    """The digest of the checkpoint in `model_dir`, or None if it has no
    complete ONNX placed yet. `checkpoint_files()` lists what it covers."""
    files = checkpoint_files(model_dir)
    if not files:
        return None
    d = Path(model_dir)
    graphs = [d / f for f in files if f.endswith(".onnx")]
    return model_digest(*graphs)
