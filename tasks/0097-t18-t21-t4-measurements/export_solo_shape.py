# T21 probe: build ONE GEMM shape as a standalone unified-style design,
# at a custom tile n, to measure what a per-shape geometry costs without
# needing all four shapes to share one xclbin. Mirrors tools/export_gemm_rtp.py
# but for a single (M,K,N) at a chosen n. int8 + c-bf16 only (T21's regime).
#
# Usage:
#   python export_solo_shape.py --out <dir> --op ffn_up --M 8192 --K 384 --N 1536 -n 64 --cols 8

from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))
CACHE = Path.home() / ".npu" / "cache"

import aie.iron as iron
from aie.iron.device import from_name
from gemm_pretiled import pretiled_array
from npue import gemm_b_layout, layout_hash

sys.path.insert(0, str(REPO / "tools"))
from export_gemm_rtp import markers_for, purge, find_cache  # reuse identity infra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu1", n_cols=None))

    a_str, acc_str, a_np = "i8", "i32", np.int8
    c_np, c_marker, c_bytes_out = bfloat16, "bf16", 2

    sh = dict(M=args.M, K=args.K, N=args.N)
    mk = markers_for(sh, args.m, args.k, args.n, c_marker, a_str)
    purge(mk, args.cols, args.op)
    A = iron.zeros((args.M, args.K), dtype=a_np, device="npu")
    B = iron.zeros((args.K, args.N), dtype=a_np, device="npu")
    C = iron.zeros(args.M * args.N, dtype=c_np, device="npu")
    pretiled_array(A, B, C, M=args.M, K=args.K, N=args.N, m=args.m, k=args.k,
                    n=args.n, n_aie_cols=args.cols,
                    dtype_in_str=a_str, dtype_out_str=acc_str,
                    emulate_bf16_mmul_with_bfp16=False,
                    pretiled=True, trace_config=None, rtp=True,
                    c_bf16=True)
    d = find_cache(mk, args.cols, args.op)
    print(f"  {args.op:<10} {[args.M,args.K,args.N]} n={args.n} cols={args.cols} -> {d.name}")

    out = Path(args.out) / "gemm_rtp"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("insts_*.bin"):
        f.unlink()
    shutil.copy(d / "final.xclbin", out / "final.xclbin")
    shutil.copy(d / "insts.bin", out / "insts.bin")
    fn = f"insts_{args.op}_b{args.M // 64}.bin"
    shutil.copy(d / "insts.bin", out / fn)

    b_layout = gemm_b_layout(args.k, args.n, dtype="I8")
    meta = {
        "name": "gemm_rtp", "kind": "gemm_rtp", "kernel": "MLIR_AIE",
        "M": args.M,
        "buffers": [args.M * args.K * 1, args.K * args.N * 1,
                    args.M * args.N * c_bytes_out],
        "c_dtype": c_marker, "a_dtype": a_str,
        "b_layout_hash": layout_hash(b_layout), "b_layout": b_layout,
        "cols": args.cols, "batch": args.M // 64, "tiers": [args.M // 64],
        "seq": 64, "hidden": args.K, "intermediate": args.K,
        "gated_ffn": False, "qkv_n": args.N,
        "tile": {"m": args.m, "k": args.k, "n": args.n},
        "streams": [{"op": args.op, "batch": args.M // 64, "slot": 1,
                      "file": fn, "M": args.M, "K": args.K, "N": args.N,
                      "src": d.name}],
    }
    (out / "design.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {out}")

if __name__ == "__main__":
    sys.exit(main())
