"""Extract real weight bytes from shipped .npue containers for the T5 DMA
compression probe. Pulls the RAW on-disk bytes (Reader.raw(), the exact byte
order the DMA reads -- pre-tiled block_panel for bf16, plain for int8), slices
the first N=4096 uint32-equivalent elements (16384 bytes) from each of four
GEMM-B shapes, and writes them as flat .bin files plus a manifest with
sha256 so the exact bytes used are traceable.

Not a general tool -- one-off for tasks/0090. Run with any numpy-having
Python (does not touch the NPU): C:\\Users\\vegar\\.conda\\envs\\iron\\python.exe
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import npue
import numpy as np

N = 4096  # matches dma_compression.py's fixed N (int32 elements per dispatch)
NBYTES = N * 4  # 16384 bytes

OUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "tiles")
os.makedirs(OUT_DIR, exist_ok=True)

SHAPES = ["qkv", "attn_out", "ffn_up", "ffn_down"]
CONTAINERS = {
    "bf16": os.path.join("models", "bge-base-en-v1.5.npue"),
    "int8": os.path.join("models", "bge-base-en-v1.5.int8.npue"),
}

manifest = []
for dtype, path in CONTAINERS.items():
    r = npue.Reader(path)
    for shape in SHAPES:
        name = f"layer.0.{shape}"
        raw = r.raw(name)  # exact on-disk dtype (uint16 for BF16, int8 for I8)
        raw_bytes = raw.tobytes()
        if len(raw_bytes) < NBYTES:
            raise ValueError(f"{path}:{name} only {len(raw_bytes)} bytes, need {NBYTES}")
        chunk = raw_bytes[:NBYTES]
        as_u32 = np.frombuffer(chunk, dtype=np.uint32)
        assert as_u32.nbytes == NBYTES
        out_name = f"{dtype}_{shape}.bin"
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(chunk)
        sha = hashlib.sha256(chunk).hexdigest()
        # crude entropy/zero-byte stats, informational only
        byte_arr = np.frombuffer(chunk, dtype=np.uint8)
        zero_frac = float(np.mean(byte_arr == 0))
        manifest.append({
            "dtype": dtype, "shape": shape, "container": path, "tensor": name,
            "file": out_name, "nbytes": NBYTES, "sha256": sha,
            "zero_byte_fraction": zero_frac,
        })
        print(f"{dtype:5s} {shape:9s} sha256={sha[:16]}... zero_byte_frac={zero_frac:.4f}")
    r.close()

with open(os.path.join(os.path.dirname(__file__), "artifacts", "tiles_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("wrote", len(manifest), "tiles")
