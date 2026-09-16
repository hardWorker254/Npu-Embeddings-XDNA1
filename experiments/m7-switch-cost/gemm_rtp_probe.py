# NpuEmbeddings -- step 1 of the one-xclbin architecture: do RTP-ified GEMM
# shapes share a static configuration?
#
# tasks/0029 proved the mechanism on a passthrough: two runtime sequences over
# one static design differ only in insts.bin, and alternating them in one
# context costs alone-price. The GEMMs could not use it because core_fn
# compiled n_tiles_per_core and K//k into the ELF -- the ONLY shape-dependent
# values in the static design.
#
# gemm_pretiled now has rtp=True: those two bounds live in a
# Buffer(use_write_rtp=True) written from the runtime sequence, behind a
# WorkerRuntimeBarrier (the scale_shift pattern). If that worked, building two
# DIFFERENT shapes must yield final.xclbin files that differ only in UUID
# metadata -- the same test 0029 ran, now on the real thing.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python experiments\m7-switch-cost\gemm_rtp_probe.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))
CACHE = Path.home() / ".npu" / "cache"

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402

M = 1024
SHAPES = {"qkv": (384, 1152), "attn_out": (384, 384)}


def build(K, N):
    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    pretiled_array(A, B, C, M=M, K=K, N=N, m=64, k=64, n=48, n_aie_cols=2,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=False,
                   pretiled=True, trace_config=None, rtp=True)


def newest_cache_with(marker):
    cands = []
    for d in CACHE.iterdir():
        m = d / "aie.mlir"
        if d.is_dir() and m.exists() and (d / "final.xclbin").exists():
            if marker in m.read_text(encoding="utf-8", errors="ignore"):
                cands.append(d)
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    dirs = {}
    for name, (K, N) in SHAPES.items():
        build(K, N)
        # fresh build in this session; identified by its unique C memref + rtp
        d = newest_cache_with(f"memref<{M * N}xf32>")
        # the RTP buffers appear as aie.buffer ... sym_name = "rtp_r_c"
        assert 'sym_name = "rtp_0_0"' in (d / "aie.mlir").read_text(
            encoding="utf-8", errors="ignore"), f"{name}: rtp not in design!"
        dirs[name] = d
        print(f"  {name:<9} K={K:<5} N={N:<5} -> {d.name}")

    a = (dirs["qkv"] / "final.xclbin").read_bytes()
    b = (dirs["attn_out"] / "final.xclbin").read_bytes()
    print(f"\n  xclbin sizes: {len(a)} vs {len(b)}")
    if len(a) != len(b):
        print("  VERDICT: sizes differ -- static configs are NOT shared")
        return 1
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    runs = []
    for i in diffs:
        if runs and i - runs[-1][1] <= 8:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    print(f"  differing bytes: {len(diffs)} in {len(runs)} cluster(s)")
    for lo, hi in runs[:12]:
        print(f"    0x{lo:06x}-0x{hi:06x} ({hi - lo + 1} B): "
              f"A={a[lo:hi + 1].hex()[:48]}")
    # 0029's UUID footprint was 67 bytes in <=7 clusters. Anything materially
    # beyond that is real configuration divergence.
    verdict = "SHARED (UUID-only)" if len(diffs) <= 80 else "NOT shared"
    print(f"\n  VERDICT: {verdict}")
    if len(diffs) > 80:
        return 1

    # Export both for the C++ cross-stream probe: one xclbin (qkv's), each
    # shape's insts.bin, and a minimal design.json. Buffer sizes are qkv's --
    # the largest -- so attn_out's stream runs inside over-allocated BOs,
    # which is exactly how a unified runtime would work.
    import json
    import shutil
    out_root = REPO / "runtime" / "artifacts_rtp"
    for name, (K, N) in SHAPES.items():
        dst = out_root / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(dirs[name] / "final.xclbin", dst / "final.xclbin")
        shutil.copy(dirs[name] / "insts.bin", dst / "insts.bin")
        (dst / "design.json").write_text(json.dumps({
            "name": f"rtp_{name}", "kernel": "MLIR_AIE", "kind": "gemm",
            "M": M, "K": K, "N": N,
            "bytes_a": M * K * 2, "bytes_b": K * N * 2, "bytes_c": M * N * 4,
            "b_layout_hash": "probe-only",
            "insts_bytes": (dst / "insts.bin").stat().st_size,
            "source_cache_dir": dirs[name].name,
        }, indent=2), encoding="utf-8")
    print(f"  exported to {out_root.relative_to(REPO)}/{{qkv,attn_out}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
