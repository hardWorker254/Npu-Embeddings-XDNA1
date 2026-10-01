# NpuEmbeddings -- a minimal, streaming safetensors reader.
#
# WHY THIS EXISTS: tools/pack/pack_npue.py imports `safetensors_io` from the
# reference/ tree, which this fork does not carry, so the historical packer
# cannot run here at all. This module is the replacement, and it differs from a
# "load the whole file" reader in one way that matters for Whisper:
#
#   openai/whisper-large-v3 is 3.1 GB of fp16 and large-v3-turbo 1.6 GB. This
#   machine has ~7 GB of free RAM, so a reader that materialises the whole
#   checkpoint as one dict is a coin flip against the OOM killer on the models
#   we are explicitly asked to support. Here the file is mmap'd once and each
#   tensor is decoded on demand, so packing peaks at the size of the largest
#   single tensor (large-v3's token embedding is 51866 x 1280 fp32 = 265 MB).
#
# The format is deliberately trivial: an 8-byte little-endian header length, a
# JSON header, then the data blob with every tensor at the offset the header
# gives it. No alignment requirements beyond what the header states.
#
# Env: numpy only.

import json
import struct

import numpy as np

# safetensors dtype tag -> numpy dtype. BF16 has no numpy dtype of its own and
# is widened by the caller, exactly as npue.py does for its own BF16 tensors.
ST_DTYPE = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i8").newbyteorder("<"),   # placeholder, replaced below
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}
ST_DTYPE["I8"] = np.dtype("<i1")


def from_bf16_bits(bits):
    """bf16 bit pattern -> fp32. Exact: bf16 is a strict subset of fp32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class SafeTensors:
    """Read-only, mmap-backed view of a .safetensors file.

    Nothing is copied until array() is called, and array() copies exactly one
    tensor. The mapping stays open for the object's lifetime; close() releases
    it, which matters on Windows and is cheap everywhere.
    """

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        header.pop("__metadata__", None)
        self.header = header
        # data blob starts right after the 8-byte length and the JSON header,
        # with no padding: safetensors is not the .npue format.
        self._data_offset = 8 + n
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def close(self):
        m = getattr(self._map, "_mmap", None)
        if m is not None:
            m.close()
        self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def keys(self):
        return list(self.header.keys())

    def __contains__(self, name):
        return name in self.header

    def info(self, name):
        """(dtype tag, shape, [begin, end) byte range in the file)."""
        try:
            e = self.header[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None
        if e is None or "data_offsets" not in e:
            raise KeyError(f"{name!r} is metadata, not a tensor")
        b, en = e["data_offsets"]
        return e["dtype"], list(e["shape"]), (self._data_offset + b,
                                               self._data_offset + en)

    def raw(self, name):
        """The bytes exactly as stored, as a read-only view. No copy."""
        dt, shape, (b, e) = self.info(name)
        if dt == "BF16":
            dt = np.dtype("<u2")
        else:
            dt = ST_DTYPE[dt]
        buf = self._map[b:e]
        return np.frombuffer(buf, dtype=dt).reshape(shape)

    def array(self, name, dtype=np.float32):
        """One tensor, widened to `dtype`. The only allocation this module makes.

        fp16 is the interesting case: the Whisper checkpoints are fp16, and a
        weight that has already been rounded to fp16 must be widened to fp32
        BEFORE it is rounded again to bf16, or the second rounding compounds the
        first. Widening here keeps that ordering explicit.

        THE RETURN VALUE MAY BE A VIEW INTO THE MAPPING, and no longer valid
        after close(). A widening conversion copies, but a same-dtype read does
        not, so `array(k).copy()` is required whenever the mapping is closed
        before the last use -- returning a dict of views into a closed mmap
        segfaults on first read rather than raising.
        """
        dt, shape, (b, e) = self.info(name)
        if dt == "BF16":
            return from_bf16_bits(self.raw(name)).astype(dtype, copy=False)
        return np.ascontiguousarray(self.raw(name), dtype=dtype)
