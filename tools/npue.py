# NpuEmbeddings -- the .npue runtime weight container.
#
# Spec lives in docs/04-model/README.md; this is the implementation, and it is
# the reference for the C++ loader in M7. numpy only, so it runs in the iron env.
#
# Layout of the file:
#   [0:64]                   FileHeader, exactly 64 bytes, little-endian
#   [json_offset:+json_len]  UTF-8 JSON: config + tensor directory
#   [data_offset:+data_len]  raw tensor data, every tensor 4096-byte aligned
#
# Design rules, all from docs/04-model and each with a reason:
#
#  * 4096-byte alignment per tensor -- page size, and a multiple of any DMA
#    burst. Costs a few KB on a 21 MB file.
#  * mmap-able: the runtime hands raw pointers to DMA descriptors and never
#    memcpys weights at load. That is why nothing here is compressed and why
#    tile data is stored in exactly the order the DMA will read it.
#  * The layout descriptor is DATA, not code. Retuning to a different tile size
#    regenerates the file rather than editing the loader. `layout_hash` makes a
#    stale file fail loudly instead of producing plausible garbage embeddings.
#  * `source_sha256` of the upstream model.safetensors travels with the file, so
#    a golden comparison can assert it ran against the same checkpoint.
#
# SPEC CORRECTION (M4): docs/04-model declared `uint8_t reserved[24]` and
# "exactly 64 bytes", but 4+4+4+4 + 8*4 = 48, so 24 reserved bytes gives 72.
# The header is 64 bytes and reserved is 16. See tasks/0006.

import hashlib
import json
import struct

import numpy as np

MAGIC = b"NPUE"
VERSION = 1

ARCH_BERT_ABS_GELU_POSTLN = 0

FLAG_PRETILED = 1 << 0

HEADER_FORMAT = "<4sIII QQQQ 16s"      # see SPEC CORRECTION above
HEADER_SIZE = 64
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE

ALIGN = 4096

# dtype tags. BF16 has no numpy dtype, so it travels as raw uint16 and is
# widened on read -- the same convention as reference/safetensors_io.py.
NP_DTYPE = {"F32": np.dtype("<f4"), "I32": np.dtype("<i4"), "I64": np.dtype("<i8"),
            "U16": np.dtype("<u2"), "BF16": np.dtype("<u2"),
            # U8 carries opaque bytes -- the tokenizer vocabulary, so a
            # deployed model is ONE file rather than a file plus a
            # vocab.txt that must not get separated from it.
            "U8": np.dtype("<u1")}


def to_bf16_bits(x):
    """fp32 -> bf16 bit pattern (uint16), round-to-nearest-even.

    bf16 is the top 16 bits of fp32, so this is a rounding of the bit pattern.
    RNE and not truncation: truncation biases every weight toward zero, which
    over 10.6M parameters is a systematic error rather than noise.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def from_bf16_bits(bits):
    """bf16 bit pattern -> fp32. Exact: bf16 is a strict subset of fp32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# -- tiling ----------------------------------------------------------------

def tile_b(b, tile_k, tile_n, s=None, t=None, order="k,n"):
    """Pre-tile a [K, N] GEMM operand into the order the DMA will stream it.

    Absorbs BOTH re-layouts the M2 design did at runtime:

      1. L3->L2: gathering a [tile_k, tile_n] tile out of a row-major [K, N]
         DDR buffer, via TensorTiler2D.step_tiler. This is the strided access
         pattern whose dimension hits the 10-bit (max 1023) DMA BD size field --
         the reason ffn_down (K=1536) could not be expressed at all.
      2. L2->L1: the intrinsic sub-tile order, which the design expressed as
         dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)].

    Output is a flat array of tiles in (kb, nb) order, each tile internally in
    (s, t) order when s and t are given. The runtime then reads whole tiles by
    index, so no access-pattern dimension exceeds K/tile_k or N/tile_n -- 24 and
    32 at MiniLM's largest, comfortably under 1023.
    """
    K, N = b.shape
    if K % tile_k or N % tile_n:
        raise ValueError(f"[{K},{N}] does not tile into ({tile_k},{tile_n})")

    # [K,N] -> [kb, tile_k, nb, tile_n] -> [kb, nb, ...] or [nb, kb, ...]
    #
    # `order` decides which of kb/nb is major, and it is a PERFORMANCE decision,
    # not a cosmetic one. A core's inner loop walks all K/k k-blocks for one
    # n-block, so with "k,n" those consecutive tiles sit (N/n)*k*n elements
    # apart -- 48 KB at ffn_down -- and the DMA scatters across ~1.1 MB. With
    # "n,k" the same walk is one contiguous run. Measured in M5.
    out = b.reshape(K // tile_k, tile_k, N // tile_n, tile_n)
    if order == "k,n":
        out = out.transpose(0, 2, 1, 3)
    elif order == "n,k":
        out = out.transpose(2, 0, 1, 3)
    else:
        raise ValueError(f"order must be 'k,n' or 'n,k', got {order!r}")

    if s is not None and t is not None:
        if tile_k % s or tile_n % t:
            raise ValueError(f"tile ({tile_k},{tile_n}) not divisible by mac ({s},{t})")
        # within a tile: [tile_k, tile_n] -> [tile_k/s, s, tile_n/t, t]
        #                                 -> [tile_k/s, tile_n/t, s, t]
        out = out.reshape(out.shape[0], out.shape[1],
                          tile_k // s, s, tile_n // t, t).transpose(0, 1, 2, 4, 3, 5)
    return np.ascontiguousarray(out).reshape(-1)


def untile_b(flat, K, N, tile_k, tile_n, s=None, t=None, order="k,n"):
    """Exact inverse of tile_b. The round-trip check is bit-exact, not close."""
    kb, nb = K // tile_k, N // tile_n
    outer = (kb, nb) if order == "k,n" else (nb, kb)
    if s is not None and t is not None:
        x = flat.reshape(*outer, tile_k // s, tile_n // t, s, t)
        x = x.transpose(0, 1, 2, 4, 3, 5).reshape(*outer, tile_k, tile_n)
    else:
        x = flat.reshape(*outer, tile_k, tile_n)
    if order == "n,k":
        x = x.transpose(1, 0, 2, 3)                 # -> [kb, nb, tile_k, tile_n]
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3).reshape(K, N))


def gemm_b_layout(tile_k, tile_n, mac_s=8, mac_t=8, dtype="BF16"):
    """The canonical B layout descriptor. Build it HERE, never inline.

    Every copy of this dict is a chance for two sides to drift, and the drift
    is invisible: `layout_hash` changes, the bytes do not, and the check that
    exists to catch wrong layouts starts reporting a mismatch that is not one.
    That happened -- tools/export_xclbin.py wrote the dict by hand and omitted
    `dtype`, so a correct file failed the check. The packer had it twice, too.
    """
    return {"kind": "block_panel", "tile_k": tile_k, "tile_n": tile_n,
            "order": "k,n,kt,nt", "inner": "s,t",
            "mac_s": mac_s, "mac_t": mac_t, "dtype": dtype}


def layout_hash(layout):
    """Stable hash of everything that changes how bytes are laid out.

    A file packed with different tile dimensions is not merely suboptimal for a
    kernel expecting others -- it is wrong, silently. The loader compares this
    and refuses rather than producing plausible garbage.
    """
    canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- writer ----------------------------------------------------------------

class Writer:
    """Accumulates tensors, then writes the container.

    Tensors are staged rather than streamed because the JSON directory carries
    every offset and must be complete before the first data byte is written.
    """

    def __init__(self, config, arch=ARCH_BERT_ABS_GELU_POSTLN, flags=FLAG_PRETILED):
        self.config = dict(config)
        self.arch = arch
        self.flags = flags
        self.entries = []
        self.blobs = []
        self._offset = 0

    def add(self, name, array, dtype_tag, role, logical_shape,
            layout=None, padded_shape=None):
        arr = np.ascontiguousarray(array, dtype=NP_DTYPE[dtype_tag])
        blob = arr.tobytes()
        entry = {
            "name": name,
            "role": role,
            "dtype": dtype_tag,
            "logical_shape": list(logical_shape),
            "padded_shape": list(padded_shape or logical_shape),
            "offset": self._offset,
            "nbytes": len(blob),
        }
        if layout is not None:
            entry["layout"] = layout
            entry["layout_hash"] = layout_hash(layout)
        self.entries.append(entry)
        self.blobs.append(blob)
        # Pad AFTER each tensor so the next one starts aligned.
        self._offset += len(blob)
        pad = (-self._offset) % ALIGN
        self._offset += pad
        self.blobs.append(b"\0" * pad)
        return entry

    def write(self, path):
        directory = {"config": self.config, "tensors": self.entries}
        js = json.dumps(directory, separators=(",", ":")).encode("utf-8")

        json_offset = HEADER_SIZE
        data_offset = json_offset + len(js)
        data_offset += (-data_offset) % ALIGN          # 4096-align the data blob
        data_length = self._offset

        header = struct.pack(
            HEADER_FORMAT, MAGIC, VERSION, self.arch, self.flags,
            json_offset, len(js), data_offset, data_length, b"\0" * 16,
        )
        with open(path, "wb") as f:
            f.write(header)
            f.write(js)
            f.write(b"\0" * (data_offset - json_offset - len(js)))
            for b in self.blobs:
                f.write(b)
        return {"json_offset": json_offset, "json_length": len(js),
                "data_offset": data_offset, "data_length": data_length,
                "total": data_offset + data_length}


# -- reader ----------------------------------------------------------------

class Reader:
    """Reads a .npue container. Uses np.memmap -- the point of the format is
    that the runtime never copies weights at load."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            head = f.read(HEADER_SIZE)
        (magic, self.version, self.arch, self.flags,
         json_offset, json_length, self.data_offset, self.data_length,
         _reserved) = struct.unpack(HEADER_FORMAT, head)

        if magic != MAGIC:
            raise ValueError(f"{path}: not a .npue file (magic {magic!r})")
        if self.version != VERSION:
            raise ValueError(f"{path}: version {self.version}, expected {VERSION}")

        with open(self.path, "rb") as f:
            f.seek(json_offset)
            directory = json.loads(f.read(json_length).decode("utf-8"))
        self.config = directory["config"]
        self.entries = {e["name"]: e for e in directory["tensors"]}
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def close(self):
        """Release the mapping. Required on Windows before the file can be
        deleted or replaced -- an open memmap holds a lock, and `del` is not
        enough because derived views keep it alive."""
        m = getattr(self._map, "_mmap", None)
        if m is not None:
            m.close()
        self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def raw(self, name):
        """The tensor exactly as stored -- tiled, bf16 as uint16. What DMA sees."""
        e = self.entries[name]
        start = self.data_offset + e["offset"]
        buf = self._map[start:start + e["nbytes"]]
        return np.frombuffer(buf, dtype=NP_DTYPE[e["dtype"]])

    def tensor(self, name):
        """The logical tensor: de-tiled and widened to fp32. For verification
        and for the Python encoder -- the C++ runtime uses raw() instead."""
        e = self.entries[name]
        x = self.raw(name)
        if e["dtype"] == "BF16":
            x = from_bf16_bits(x)
        lay = e.get("layout")
        if lay and lay.get("kind") == "block_panel":
            K, N = e["padded_shape"]
            x = untile_b(x, K, N, lay["tile_k"], lay["tile_n"],
                         lay.get("mac_s"), lay.get("mac_t"))
            kl, nl = e["logical_shape"]
            x = x[:kl, :nl]
        else:
            x = x.reshape(e["logical_shape"])
        # copy(), not ascontiguousarray(): the latter can hand back a view into
        # the mapping, which would keep the file locked and give the caller an
        # array that dies when close() is called.
        return np.ascontiguousarray(x).copy()

    def check_layout(self, name, layout):
        """Refuse a file whose layout is not the one the caller expects."""
        e = self.entries[name]
        want = layout_hash(layout)
        got = e.get("layout_hash")
        if got != want:
            raise ValueError(
                f"{name}: layout_hash mismatch -- file has {got}, caller wants "
                f"{want}. Repack with tools/pack_npue.py.")


# --- one naming rule, three callers ----------------------------------------
# reference/make_goldens.py writes them, tools/export_validation.py and
# tools/verify_npue.py read them. Three copies of a naming convention is how
# the three pooling implementations started, so it lives here.

# all-MiniLM-L6-v2's goldens predate the derived scheme and are cited by name
# in tasks/0005 and by six scripts under experiments/. Renaming them would
# falsify a task log, so its historical slug is kept.
LEGACY_GOLDEN_SLUGS = {"all-MiniLM-L6-v2": "minilm_l6"}


def golden_slug(model_name, n_layers):
    """Stem of this model's golden files, without the _s<seq>_* suffix."""
    import os
    name = os.path.basename(str(model_name).rstrip("/\\"))
    if name.endswith(".npue"):
        name = name[:-5]
    return LEGACY_GOLDEN_SLUGS.get(name, f"{name.lower()}_l{n_layers}")


def find_goldens(goldens_dir, source_sha256, seq, load):
    """The goldens for a checkpoint, found by CONTENT rather than by name.

    Goldens belong to a checkpoint; a `.npue` is one packing of it. Two
    containers of the same weights at different tile sizes share goldens, and
    deriving the filename from the container's name made that unexpressible --
    `bge-large-n16.npue` went looking for `bge-large-n16_l24_s64_*`.

    `load` is passed in because reference/safetensors_io is not importable from
    here without dragging reference/ onto the path of every caller.

    Returns (boundary_path, taps_path). Raises if the match is not exactly one:
    zero means the goldens were never generated, and more than one means two
    checkpoints share a sha256, which is not a thing to guess about.
    """
    from pathlib import Path as _P
    gdir = _P(goldens_dir)
    hits = []
    for cand in sorted(gdir.glob(f"*_s{seq}_boundary.safetensors")):
        try:
            _, meta = load(cand)
        except Exception:
            continue
        if meta.get("source_sha256") == source_sha256:
            hits.append(cand)
    if len(hits) != 1:
        names = ", ".join(h.name for h in hits) if hits else "none"
        raise FileNotFoundError(
            f"{len(hits)} goldens in {gdir} match checkpoint "
            f"{source_sha256[:16]}... at seq {seq} ({names}). Generate them "
            f"with reference/make_goldens.py --model-dir <dir> --taps")
    return hits[0], hits[0].with_name(hits[0].name.replace("_boundary.", "_taps."))
