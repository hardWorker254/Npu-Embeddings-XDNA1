# NpuEmbeddings -- M13 nomic-embed-text-v1.5 spike: does N=6144 compile?
# SPDX-License-Identifier: Apache-2.0
#
# QUESTION (the ONE job of this script): nomic-embed-text-v1.5 has a SwiGLU
# gated FFN, so its ffn_up GEMM is [M, 768] x [768, 6144] -- N=6144, the
# widest single GEMM stream this project has ever tried to build. Does it
# compile at all, and if it compiles does it produce a correct result?
#
# This is a GO/NO-GO probe. It does NOT build a production artifact set and
# does NOT modify anything outside tasks/0068-m13-nomic-spike-and-oracle/.
#
# Background (read before touching this file):
#   - docs/CURRENT_STATUS.md's "known walls" table: `hidden >= 1536` in the
#     old width sweep hit `'aie.dma_bd' op Stride 3 exceeds the
#     [1:1048576] range` -- the C-drain row-block stride, NOT the K<=1023
#     BD-size wall (trap 4). tasks/0027 found it; tasks/0030 (expert review
#     5b, commit 4d62053) diagnosed the exact mechanism and
#     experiments/m5-pretiled-gemm/gemm_pretiled.py's `_build_design` now
#     guards it explicitly:
#         if m * n_aie_rows * N > 2**20:
#             tb_n_rows = 1
#     This probe exists to find out whether that guard is actually
#     sufficient at N=6144 (docs say the wall is still "Open" -- possibly
#     just undocumented-as-fixed, possibly a second wall the guard misses).
#   - CLAUDE.md trap 3: L1 budget is 2*(m*k*in + k*n*in + m*n*out); N does
#     NOT enter it. At (m,k,n)=(64,64,48) bf16-in/fp32-out that is 53,248 B
#     of the 63 KB limit, same as every shipping model. If this probe fails
#     for an L1 reason, that contradicts trap 3 and must be reported loudly.
#   - CLAUDE.md trap 4: DMA BD size field is 10 bits (max 1023). K/k and
#     other loop-count dimensions must stay under that ceiling.
#   - CLAUDE.md traps 6b/6c: never write device tensors through .numpy();
#     never validate against a device read-back. This probe uses
#     experiments/m5-pretiled-gemm/gemm_pretiled.py's existing `run_one()`,
#     which already implements the correct pattern (B[:] = ..., reference
#     computed from the host-side A_np/B_logical captured before any
#     device write) -- verified by reading that function before use.
#
# Env: iron env WITH C:\dev\mlir-aie\iron_env.ps1 dot-sourced first.
# Usage (from repo root):
#   python tasks\0068-m13-nomic-spike-and-oracle\probe_n6144.py
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

# CLAUDE.md trap 1: set the device BEFORE any kernels.* call, or IRON
# silently falls back to NPU1 (aie2, not aie2p) and bf16 mac_dims/burst
# sizes change with no error. n_cols=None or from_name("npu2") defaults to
# ONE column.
iron.set_current_device(from_name("npu2", n_cols=None))

from gemm_pretiled import run_one            # noqa: E402


def l1_bytes(m, k, n, in_bytes=2, out_bytes=4):
    return 2 * (m * k * in_bytes + k * n * in_bytes + m * n * out_bytes)


def try_one(label, M, K, N, m, k, n, cols):
    print(f"\n{'=' * 78}")
    print(f"{label}: M={M} K={K} N={N}  tile=({m},{k},{n})  cols={cols}")
    print(f"  L1 budget (Stationary-C, trap 3): {l1_bytes(m, k, n)} B "
          f"of 65536 (63 KB nominal limit)  -- N does not enter this formula")
    print(f"  divisibility checks: M%(m*4)={M % (m * 4)}  K%k={K % k}  "
          f"N%(n*cols)={N % (n * cols)}")
    t0 = time.time()
    try:
        res = run_one(M, K, N, m, k, n, cols,
                      emulate=False, trace_size=0, pretiled=True,
                      tile_order="k,n", inner_st=True, b_reuse=False,
                      dtype_in="bf16", dtype_out="f32",
                      verbose=True, trace=False)
        dt = time.time() - t0
        if res is None:
            print(f"  run_one returned None (SKIP -- see message above), "
                  f"{dt:.1f}s")
            return {"label": label, "M": M, "K": K, "N": N, "cols": cols,
                    "compiled": False, "skipped": True, "seconds": dt}
        print(f"  COMPILED AND RAN in {dt:.1f}s")
        print(f"  rel_frobenius = {res['rel_frobenius']:.3e}  "
              f"{'PASS' if res['correctness_pass'] else 'FAIL'} "
              f"(tol 5e-3 for plain bf16)")
        res["label"] = label
        res["compiled"] = True
        res["seconds"] = dt
        return res
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAILED TO COMPILE/RUN after {dt:.1f}s")
        print(f"  exception type: {type(e).__name__}")
        print("  ---- full exception text ----")
        traceback.print_exc(file=sys.stdout)
        print("  ---- end exception text ----")
        msg = str(e)
        mechanism = "UNKNOWN"
        if "Stride" in msg and "exceeds" in msg:
            mechanism = "DMA BD STRIDE range [1:1048576] exceeded"
        elif "Size" in msg and "[0:1023]" in msg:
            mechanism = "DMA BD SIZE field (10 bits, max 1023) exceeded"
        elif "number of input DMA channel" in msg or "DMA channel" in msg:
            mechanism = "DMA channel count exceeded (2 in / 2 out per core, 6/6 per mem tile)"
        elif "Basic sequential allocation" in msg:
            mechanism = "L1 allocation failure (63 KB budget)"
        elif "legal routing" in msg or "packet ID" in msg:
            mechanism = "routing / packet-ID exhaustion (trace-related, unexpected here)"
        print(f"  BELIEVED MECHANISM: {mechanism}")
        return {"label": label, "M": M, "K": K, "N": N, "cols": cols,
                "m": m, "k": k, "n": n,
                "compiled": False, "error": msg, "mechanism": mechanism,
                "seconds": dt}


def main() -> int:
    results = []

    # A. The real nomic ffn_up at batch 128, SEQ=64 -> M=8192.
    results.append(try_one(
        "A: nomic ffn_up (real shape)", M=8192, K=768, N=6144,
        m=64, k=64, n=48, cols=8))

    # B. Positive control: nomic ffn_down. Same K/N as bge-base's ffn_up
    #    transposed shape-class (K=3072,N=768) -- should behave identically
    #    to a shape this project already ships.
    results.append(try_one(
        "B: nomic ffn_down (positive control)", M=8192, K=3072, N=768,
        m=64, k=64, n=48, cols=8))

    # C. Smaller M, same N=6144 -- isolates whether M*N buffer/stride sizing
    #    (not N alone) is the wall.
    results.append(try_one(
        "C: nomic ffn_up, small M", M=1024, K=768, N=6144,
        m=64, k=64, n=48, cols=8))

    # If A failed, establish the boundary: does N=6144 work at 4 columns?
    # At tile_n 32? At tile_n 16? Each answer narrows which limit it is.
    a_result = results[0]
    if not a_result.get("compiled"):
        print(f"\n{'#' * 78}")
        print("# A FAILED -- establishing the boundary")
        print(f"{'#' * 78}")

        # A1: same shape, 4 columns instead of 8 (halves several strides
        # that scale with n_aie_cols; also halves n_aie_rows' interaction
        # with N in the tb_n_rows guard formula, which does NOT depend on
        # cols, only on m*n_aie_rows*N -- so this isolates whether the wall
        # is COLUMN-COUNT dependent or N-alone dependent).
        # N % (n * cols) must hold: 6144 % (48*4=192) = 0. OK.
        results.append(try_one(
            "A1: N=6144 at 4 columns", M=8192, K=768, N=6144,
            m=64, k=64, n=48, cols=4))

        # A2: tile_n=32 at 8 columns. 6144 % (32*8=256) = 0. OK.
        results.append(try_one(
            "A2: N=6144, tile_n=32, 8 columns", M=8192, K=768, N=6144,
            m=64, k=64, n=32, cols=8))

        # A3: tile_n=16 at 8 columns. 6144 % (16*8=128) = 0. OK.
        results.append(try_one(
            "A3: N=6144, tile_n=16, 8 columns", M=8192, K=768, N=6144,
            m=64, k=64, n=16, cols=8))

        # A4: tile_n=48, 4 columns, but also smaller M (combine A1 x C) to
        # separate an M-dependent wall from a pure-N-at-8-cols wall.
        results.append(try_one(
            "A4: N=6144, tile_n=48, 4 columns, small M", M=1024, K=768,
            N=6144, m=64, k=64, n=48, cols=4))

    print(f"\n{'#' * 78}")
    print("# SUMMARY")
    print(f"{'#' * 78}")
    for r in results:
        status = ("COMPILE-FAIL" if not r.get("compiled") else
                  "SKIP" if r.get("skipped") else
                  ("PASS" if r.get("correctness_pass") else "NUMERIC-FAIL"))
        extra = ""
        if not r.get("compiled") and not r.get("skipped"):
            extra = f"  mechanism={r.get('mechanism')}"
        elif r.get("compiled"):
            extra = f"  rel_fro={r.get('rel_frobenius'):.3e}"
        print(f"  {r['label']:<48} {status:<14}{extra}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
