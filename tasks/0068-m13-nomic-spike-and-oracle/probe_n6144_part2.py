# NpuEmbeddings -- M13 nomic spike, part 2: boundary-narrowing the NUMERIC
# failure found in part 1 (probe_n6144.py / probe_n6144.txt).
#
# PART 1 RESULT (already on disk, exit code 0, run completed):
#   A: M=8192 K=768 N=6144 cols=8 tile(64,64,48)  COMPILED in 22.7s,
#      rel_frobenius = 7.074e-01  FAIL
#   B: M=8192 K=3072 N=768 (positive control)     COMPILED in 9.8s,
#      rel_frobenius = 8.173e-07 PASS  <- harness itself is correct
#   C: M=1024 K=768 N=6144 (small M)               COMPILED in 8.3s,
#      rel_frobenius = 7.067e-01 FAIL
#
# So N=6144 COMPILES (no aie.dma_bd stride/size error) but produces WRONG
# numbers, both at M=8192 and M=1024. rel_fro ~0.707 = sqrt(1/2) on BOTH is
# the signature of "half the output is right, half is wrong/stale" -- not
# random noise.
#
# HYPOTHESIS (read directly in experiments/m5-pretiled-gemm/gemm_pretiled.py
# `_build_design`, around line 498-608): the `m * n_aie_rows * N > 2**20`
# guard (added to fix the C-drain stride wall, tasks/0030 5b) sets
#     tb_n_rows = 1
# which sizes the C_tiles drain TAP's `tile_group_repeats`. But the runtime
# `sequence()` function's per-iteration row count,
#     current_tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
# is computed independently from the HARD-CODED `tb_max_n_rows = 4` (i.e.
# always up to 2), and never reads the guarded `tb_n_rows`. So when the guard
# fires, the C drain tap is sized for 1 row-block per call while the A/B fill
# loop still streams up to 2 row-blocks before the next drain -- an
# accounting mismatch between what is drained and what is filled, whenever
# N is large enough to trip the guard. This would explain wrong-but-compiling
# results, and would NOT depend on cols or tile_n (the guard formula
# `m * n_aie_rows * N > 2**20` has neither in it) -- it depends only on N
# (threshold N > 2**20/(m*4) = 4096 at m=64).
#
# THIS SCRIPT tests that hypothesis:
#   D. Same shape A but cols=4            -- guard still fires (N-only test)
#   E. Same shape A but tile_n=32, cols=8 -- guard still fires
#   F. Same shape A but tile_n=16, cols=8 -- guard still fires
#   G. N=4096 (bge-large's real ffn_up N, exactly AT the guard threshold,
#      2**20/(64*4) = 4096 exactly -- guard does NOT fire since the compare
#      is strict '>'). Sanity control: known-good in production, must PASS.
#   H. N=4224, just ABOVE 4096 -- smallest N over the boundary. If this also
#      fails, the wall is a hard N>4096 boundary at m=64, independent of
#      tile_n/cols, and every one of nomic's 6144 is on the wrong side of it.
#
# All shapes use dtype bf16 in / fp32 out (this project's standing contract).
# No trace (trap 7: 8 columns is not traceable anyway; not needed for a
# correctness probe).
#
# Env: iron env WITH C:\dev\mlir-aie\iron_env.ps1 dot-sourced first.
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
    print(f"  guard 'm*n_aie_rows*N > 2**20' -> {m*n_aie_rows*N} > 1048576 = {guard}"
          f"  (tb_n_rows {'FORCED TO 1' if guard else 'stays default'})")
    print(f"  divisibility: M%(m*4)={M % (m * 4)}  K%k={K % k}  "
          f"N%(n*cols)={N % (n * cols)}")
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
        diff = got - ref
        rel_fro = float(np.linalg.norm(diff) / np.linalg.norm(ref))
        max_abs = float(np.max(np.abs(diff)))
        max_abs_ref = float(np.max(np.abs(ref)))
        ok = rel_fro <= 5e-3
        dt = time.time() - t0
        print(f"  COMPILED AND RAN in {dt:.1f}s")
        print(f"  rel_frobenius = {rel_fro:.3e}   max_abs_diff = {max_abs:.3e}"
              f"  (max|ref|={max_abs_ref:.3e})   {'PASS' if ok else 'FAIL'}")
        # Cheap structural check for the "half wrong" hypothesis: fraction of
        # (m*n_aie_rows)-row bands whose max abs error is "small" vs "huge".
        band = m * n_aie_rows
        n_bands = M // band
        band_errs = [float(np.max(np.abs(diff[i * band:(i + 1) * band, :])))
                     for i in range(n_bands)]
        bad_bands = sum(1 for e in band_errs if e > 1.0)
        print(f"  row-band structural check: {bad_bands}/{n_bands} bands of "
              f"{band} rows have max|err| > 1.0 "
              f"(bf16 numeric error alone would be ~1e-2, not >1.0)")
        return dict(label=label, compiled=True, rel_fro=rel_fro,
                    max_abs=max_abs, ok=ok, bad_bands=bad_bands,
                    n_bands=n_bands, seconds=dt)
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAILED TO COMPILE/RUN after {dt:.1f}s: {type(e).__name__}")
        traceback.print_exc(file=sys.stdout)
        return dict(label=label, compiled=False, error=str(e), seconds=dt)


def main() -> int:
    results = []
    results.append(probe("D: shape A at cols=4", 8192, 768, 6144, 64, 64, 48, 4))
    results.append(probe("E: shape A, tile_n=32, cols=8", 8192, 768, 6144, 64, 64, 32, 8))
    results.append(probe("F: shape A, tile_n=16, cols=8", 8192, 768, 6144, 64, 64, 16, 8))
    results.append(probe("G: N=4096 (guard boundary, AT threshold, sanity control)",
                         8192, 768, 4096, 64, 64, 48, 8))
    results.append(probe("H: N=4224 (guard boundary, just ABOVE threshold)",
                         8192, 768, 4224, 64, 64, 48, 8))

    print(f"\n{'#' * 78}")
    print("# PART 2 SUMMARY")
    print(f"{'#' * 78}")
    for r in results:
        if not r["compiled"]:
            print(f"  {r['label']:<55} COMPILE-FAIL")
        else:
            print(f"  {r['label']:<55} "
                  f"{'PASS' if r['ok'] else 'FAIL'}  rel_fro={r['rel_fro']:.3e}  "
                  f"bad_bands={r['bad_bands']}/{r['n_bands']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
