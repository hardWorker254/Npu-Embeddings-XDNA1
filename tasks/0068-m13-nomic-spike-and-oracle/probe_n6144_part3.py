# NpuEmbeddings -- M13 nomic spike, part 3: two clean below-threshold
# controls. Part 2's "G" control (N=4096, tile_n=48) was a bad test -- 4096
# is not divisible by 48 at all (tile_b rejects it outright, unrelated to
# the guard), which is exactly WHY production bge-large uses tile_n=32 for
# its real N=4096 ffn_up (docs/CURRENT_STATUS.md: "N in {1024,3072,4096}
# makes 48 illegal; 32 is the largest legal value"). This script redoes the
# sanity control properly:
#   I. N=4096, tile_n=32, cols=8 -- bge-large's REAL production shape.
#      m*n_aie_rows*N = 64*4*4096 = 1048576 = 2**20 exactly, NOT > 2**20,
#      so the guard does NOT fire. Must PASS (it already runs in production).
#   J. N=3840, tile_n=48, cols=8 -- same tile_n as nomic's ffn_up (shape A),
#      strictly BELOW the guard threshold (983040 < 1048576). Isolates
#      "does tile_n=48 itself work below the boundary" from "is N=6144 the
#      problem" -- if this PASSES and shape A FAILS, tile_n=48 is exonerated
#      and the guard boundary (N>4096 at m=64) is the whole story.
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np                            # noqa: E402
import aie.iron as iron                        # noqa: E402
from aie.iron.device import from_name          # noqa: E402
from ml_dtypes import bfloat16                 # noqa: E402

iron.set_current_device(from_name("npu2", n_cols=None))

from aie.iron import kernels, str_to_dtype     # noqa: E402
from gemm_pretiled import pretiled_array       # noqa: E402
from npue import tile_b                        # noqa: E402


def probe(label, M, K, N, m, k, n, cols):
    print(f"\n{'=' * 78}")
    print(f"{label}: M={M} K={K} N={N} tile=({m},{k},{n}) cols={cols}")
    n_aie_rows = 4
    guard = m * n_aie_rows * N > 2**20
    print(f"  guard 'm*n_aie_rows*N > 2**20' -> {m*n_aie_rows*N} > 1048576 = {guard}")
    print(f"  divisibility: M%(m*4)={M % (m * 4)}  K%k={K % k}  "
          f"N%(n*cols)={N % (n * cols)}  N%n={N % n}")
    t0 = time.time()
    try:
        dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
        A = iron.rand((M, K), dtype=dt_in, device="npu")
        B = iron.rand((K, N), dtype=dt_in, device="npu")
        C = iron.zeros(M * N, dtype=dt_out, device="npu")
        A_np = A.numpy().copy()
        B_logical = B.numpy().copy()
        r, s, t = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
            b_col_maj=False, c_col_maj=False, use_chess=False,
            emulate_bf16_mmul_with_bfp16=False, vectorized=True).mac_dims
        tiled = tile_b(B_logical.view(np.uint16), k, n, s, t, order="k,n")
        B[:] = tiled.view(bfloat16).reshape(K, N)
        assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
        pretiled_array(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
                       dtype_in_str="bf16", dtype_out_str="f32",
                       emulate_bf16_mmul_with_bfp16=False, pretiled=True,
                       tile_order="k,n", inner_st=True, b_reuse=False,
                       trace_config=None)
        got = C.numpy().reshape(M, N).astype(np.float64)
        ref = A_np.astype(np.float64) @ B_logical.astype(np.float64)
        rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        ok = rel_fro <= 5e-3
        dt = time.time() - t0
        print(f"  COMPILED AND RAN in {dt:.1f}s")
        print(f"  rel_frobenius = {rel_fro:.3e}  {'PASS' if ok else 'FAIL'}")
        return dict(label=label, compiled=True, rel_fro=rel_fro, ok=ok)
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAILED TO COMPILE/RUN after {dt:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout)
        return dict(label=label, compiled=False, error=str(e))


def main() -> int:
    results = [
        probe("I: N=4096 tile_n=32 (bge-large's REAL production shape, sanity control)",
              8192, 768, 4096, 64, 64, 32, 8),
        probe("J: N=3840 tile_n=48 (below guard threshold, same tile_n as nomic ffn_up)",
              8192, 768, 3840, 64, 64, 48, 8),
    ]
    print(f"\n{'#' * 78}\n# PART 3 SUMMARY\n{'#' * 78}")
    for r in results:
        if r["compiled"]:
            print(f"  {r['label']:<70} {'PASS' if r['ok'] else 'FAIL'} rel_fro={r['rel_fro']:.3e}")
        else:
            print(f"  {r['label']:<70} COMPILE-FAIL: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
