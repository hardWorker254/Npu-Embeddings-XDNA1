# NpuEmbeddings -- minimal safetensors reader/writer, numpy only.
#
# Why not just `pip install safetensors`
# -------------------------------------
# Golden vectors cross an environment boundary (CLAUDE.md): they are produced in
# .venv-ref (transformers, torch) and consumed in the iron env, which must stay
# clean -- a pip accident there breaks the toolchain that took the most work to
# get running. A dependency-free reader means the consuming side needs nothing
# but numpy.
#
# It is also ~80 lines, and M4/M7 have to parse this format from C++ anyway, so
# writing it once in Python is the cheap way to be sure we understand it.
#
# Format (as documented in docs/04-model/README.md):
#   [0:8]    uint64 LE  header length N
#   [8:8+N]  UTF-8 JSON header
#   [8+N:]   raw data buffer, little-endian, row-major, C-contiguous
# Header maps tensor name -> {"dtype", "shape", "data_offsets": [start, end]},
# where offsets are RELATIVE to the start of the data buffer. The reserved key
# "__metadata__" holds a flat str->str dict.

import json

import numpy as np

# safetensors dtype tag -> numpy. Only what this project can produce.
DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": None,          # no numpy dtype; handled as raw uint16, see below
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}
TAGS = {v: k for k, v in DTYPES.items() if v is not None}


def load(path, *, bf16_as="float32"):
    """Read a .safetensors file. Returns (tensors: dict, metadata: dict).

    BF16 has no numpy equivalent. `bf16_as="float32"` widens it losslessly (the
    natural thing for a comparison oracle); `bf16_as="uint16"` hands back the
    raw bit pattern.
    """
    with open(path, "rb") as f:
        buf = f.read()

    n = int.from_bytes(buf[:8], "little")
    header = json.loads(buf[8 : 8 + n].decode("utf-8"))
    data = memoryview(buf)[8 + n :]

    metadata = header.pop("__metadata__", {})
    out = {}
    for name, spec in header.items():
        start, end = spec["data_offsets"]
        raw = data[start:end]
        shape = tuple(spec["shape"])
        if spec["dtype"] == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").reshape(shape)
            if bf16_as == "uint16":
                out[name] = bits.copy()
            else:
                # bf16 is the top 16 bits of fp32 -- shift left, reinterpret.
                out[name] = (bits.astype(np.uint32) << 16).view(np.float32)
            continue
        dt = DTYPES.get(spec["dtype"])
        if dt is None:
            raise ValueError(f"{name}: unsupported dtype {spec['dtype']}")
        arr = np.frombuffer(raw, dtype=dt).reshape(shape)
        out[name] = arr.copy()      # detach from the file buffer
    return out, metadata


def save(path, tensors, metadata=None):
    """Write a .safetensors file. `metadata` values must all be strings."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        tag = TAGS.get(arr.dtype.newbyteorder("<"))
        if tag is None:
            raise ValueError(f"{name}: no safetensors tag for dtype {arr.dtype}")
        blob = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {
            "dtype": tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    if metadata:
        bad = [k for k, v in metadata.items() if not isinstance(v, str)]
        if bad:
            raise ValueError(f"metadata values must be str; offenders: {bad}")
        header["__metadata__"] = dict(metadata)

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The spec requires the data buffer to start 8-byte aligned; pad the header.
    pad = (-len(blob)) % 8
    blob += b" " * pad

    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for b in blobs:
            f.write(b)
