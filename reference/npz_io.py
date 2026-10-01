# NpuEmbeddings -- read and write the committed goldens, as .npz.
#
# A golden is data that crosses an environment boundary: it is produced in an
# env with torch and transformers, and consumed in the iron env with numpy and
# nothing else. It has to be a format both ends open without a package, and it
# has to carry its own provenance, because a golden compared against a
# different checkpoint is worse than no golden.
#
# It used to be a .safetensors file. .npz is numpy's own container: a zip of
# .npy arrays, so dtypes and shapes round-trip exactly, `allow_pickle=False`
# on read keeps it data rather than code, and numpy is in both environments
# already. The string metadata map that lived in safetensors' `__metadata__`
# becomes one uint8 array holding the JSON, under META_KEY.
#
# Env: numpy only.
# Usage:
#   from npz_io import load, save
#   save(path, tensors, {"source_sha256": digest, "seq_len": "64"})
#   g, meta = load(path)

import json
from pathlib import Path

import numpy as np

__all__ = ["META_KEY", "load", "save"]

# The one key in an .npz that is not a tensor. Golden tensors are named after
# HF's activations (`input_ids`) or the reference's taps (`hf.L0.ln2`), so this
# cannot collide by accident -- and save() REFUSES a tensor that is called
# this rather than letting the provenance be overwritten by an array.
META_KEY = "__meta__"


def save(path, tensors, metadata=None):
    """Write {name: ndarray} plus a {str: str} metadata map to `path`."""
    meta = dict(metadata or {})
    bad = [k for k, v in meta.items() if not isinstance(v, str)]
    if bad:
        raise ValueError(f"metadata values must be str; offenders: {bad}")
    if META_KEY in tensors:
        raise ValueError(f"a tensor may not be called {META_KEY!r}: that key "
                         f"is where the metadata lives")
    payload = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    # JSON rather than a pickle object array: the point of this format is that
    # it opens as data in an env that will not unpickle anything.
    payload[META_KEY] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{path}: goldens are .npz (numpy would append the "
                         f"suffix silently and the caller would look for the "
                         f"wrong file)")
    np.savez(path, **payload)


def load(path):
    """Read a .npz written by save(). Returns ({name: ndarray}, {str: str})."""
    with np.load(path, allow_pickle=False) as z:
        if META_KEY not in z.files:
            raise ValueError(
                f"{path}: no {META_KEY!r} entry -- not a golden written by "
                f"npz_io.save(), so its provenance cannot be checked")
        meta = json.loads(z[META_KEY].tobytes().decode("utf-8"))
        tensors = {k: z[k] for k in z.files if k != META_KEY}
    return tensors, meta
