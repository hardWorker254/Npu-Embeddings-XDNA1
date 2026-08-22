# NpuEmbeddings -- verification for the tb_n_rows fill/drain fix (tasks/0068)
# SPDX-License-Identifier: Apache-2.0
#
# Verifies experiments/m5-pretiled-gemm/gemm_pretiled.py's _build_design fix:
# the C-drain fill/drain walk now steps by the GUARDED tb_n_rows instead of
# the hardcoded tb_max_n_rows // 2, so the guard ('m*n_aie_rows*N > 2**20',
# which forces tb_n_rows to 1) and the fill loop agree on how many row
# blocks are covered per drain call.
#
# Four groups, matching tasks/0068-m13-nomic-spike-and-oracle/TASK.md's
# verification requirements 1-3 (requirement 4, the byte-identical
# production-xclbin check, is a separate script using export_gemm_rtp.py):
#
#   GROUP 1 -- the fix works (guard-fired shapes, previously ~7.07e-01 FAIL):
#     A: M=8192 K=768 N=6144 tile(64,64,48) cols=8  (nomic's real ffn_up)
#     B: M=1024 K=768 N=6144 tile(64,64,48) cols=8  (small M, same N)
#     C: M=8192 K=768 N=4224 tile(64,64,48) cols=8  (just above the 4096
#        threshold -- probe_n6144.txt's shape H)
#
#   GROUP 2 -- below-threshold shapes MUST reproduce probe_n6144.txt's
#   numbers exactly (guard never fires for these -- the fix must be a
#   strict no-op here):
#     D: M=8192 K=3072 N=768  tile_n=48 cols=8  -> was 8.173e-07 (probe B)
#     E: M=8192 K=768  N=3840 tile_n=48 cols=8  -> was 3.291e-07 (probe J)
#     F: M=8192 K=768  N=4096 tile_n=32 cols=8  -> was 3.283e-07 (probe I)
#
#   GROUP 3 -- small-M tail case (the NPUE-M5 comment at gemm_pretiled.py:499,
#   M // m // n_aie_rows == 1 row block), at a below-threshold N so this
#   exercises the pre-existing tail logic under the new formulation:
#     G: M=256  K=768 N=768 tile_n=48 cols=8  (exactly 1 row block)
#     H: M=1024 K=768 N=768 tile_n=48 cols=8  (4 row blocks)
#
# Uses gemm_pretiled.run_one(), which already implements the correct
# reference pattern per CLAUDE.md traps 6b/6c: A_np/B_logical are captured
# from the host-intended values before repacking, C is compared against
# A_np @ B_logical, never against a device read-back of a value the device
# itself produced.
#
# Env: iron env WITH C:\dev\mlir-aie\iron_env.ps1 dot-sourced first.
# Usage (from repo root):
#   python tasks\0068-m13-nomic-spike-and-oracle\verify_fix_tb_n_rows.py
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))

import aie.iron as iron                      # noqa: E402
from aie.iron.device import from_name        # noqa: E402

iron.set_current_device(from_name("npu2", n_cols=None))

from gemm_pretiled import run_one            # noqa: E402


def try_one(label, M, K, N, m, k, n, cols, expect=None):
    print(f"\n{'=' * 78}")
    print(f"{label}: M={M} K={K} N={N} tile=({m},{k},{n}) cols={cols}")
    n_aie_rows = 4
    guard = m * n_aie_rows * N > 2**20
    print(f"  guard 'm*n_aie_rows*N > 2**20' -> {m*n_aie_rows*N} > 1048576 = "
          f"{guard}  (tb_n_rows {'FORCED TO 1' if guard else 'stays default'})")
    print(f"  row blocks M//m//n_aie_rows = {M // m // n_aie_rows}")
    if expect is not None:
        print(f"  expected (from probe_n6144.txt, must reproduce): {expect:.3e}")
    t0 = time.time()
    try:
        res = run_one(M, K, N, m, k, n, cols,
                      emulate=False, trace_size=0, pretiled=True,
                      tile_order="k,n", inner_st=True, b_reuse=False,
                      dtype_in="bf16", dtype_out="f32",
                      verbose=True, trace=False)
        dt = time.time() - t0
        if res is None:
            print(f"  run_one returned None (SKIP), {dt:.1f}s")
            return {"label": label, "compiled": False, "skipped": True,
                    "seconds": dt}
        print(f"  COMPILED AND RAN in {dt:.1f}s")
        print(f"  rel_frobenius = {res['rel_frobenius']:.6e}  "
              f"{'PASS' if res['correctness_pass'] else 'FAIL'} "
              f"(tol 5e-3 for plain bf16)")
        res["label"] = label
        res["compiled"] = True
        res["seconds"] = dt
        return res
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAILED TO COMPILE/RUN after {dt:.1f}s: {type(e).__name__}")
        traceback.print_exc(file=sys.stdout)
        return {"label": label, "compiled": False, "error": str(e),
                "seconds": dt}


def main() -> int:
    results = []

    print(f"\n{'#' * 78}")
    print("# GROUP 1 -- the fix: guard-fired shapes must now PASS")
    print(f"{'#' * 78}")
    results.append(try_one(
        "A: nomic ffn_up (real shape)", M=8192, K=768, N=6144,
        m=64, k=64, n=48, cols=8))
    results.append(try_one(
        "B: nomic ffn_up, small M", M=1024, K=768, N=6144,
        m=64, k=64, n=48, cols=8))
    results.append(try_one(
        "C: N=4224, just above guard threshold", M=8192, K=768, N=4224,
        m=64, k=64, n=48, cols=8))

    print(f"\n{'#' * 78}")
    print("# GROUP 2 -- below-threshold shapes must be UNCHANGED (no-op check)")
    print(f"{'#' * 78}")
    results.append(try_one(
        "D: K=3072,N=768 tile_n=48 (was 8.173e-07)", M=8192, K=3072, N=768,
        m=64, k=64, n=48, cols=8, expect=8.173e-07))
    results.append(try_one(
        "E: K=768,N=3840 tile_n=48 (was 3.291e-07)", M=8192, K=768, N=3840,
        m=64, k=64, n=48, cols=8, expect=3.291e-07))
    results.append(try_one(
        "F: K=768,N=4096 tile_n=32 (was 3.283e-07, bge-large real shape)",
        M=8192, K=768, N=4096, m=64, k=64, n=32, cols=8, expect=3.283e-07))

    print(f"\n{'#' * 78}")
    print("# GROUP 3 -- small-M tail case (NPUE-M5 comment), below threshold")
    print(f"{'#' * 78}")
    results.append(try_one(
        "G: M=256 (exactly 1 row block), N=768 tile_n=48", M=256, K=768,
        N=768, m=64, k=64, n=48, cols=8))
    results.append(try_one(
        "H: M=1024 (4 row blocks), N=768 tile_n=48", M=1024, K=768,
        N=768, m=64, k=64, n=48, cols=8))

    print(f"\n{'#' * 78}")
    print("# SUMMARY")
    print(f"{'#' * 78}")
    for r in results:
        status = ("COMPILE-FAIL" if not r.get("compiled") else
                  "SKIP" if r.get("skipped") else
                  ("PASS" if r.get("correctness_pass") else "NUMERIC-FAIL"))
        extra = ""
        if r.get("compiled"):
            extra = f"  rel_fro={r.get('rel_frobenius'):.6e}"
        print(f"  {r['label']:<58} {status:<14}{extra}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
