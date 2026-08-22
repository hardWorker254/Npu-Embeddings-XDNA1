# NpuEmbeddings -- T28 Del B / B2 (tasks/0054): dispatch-latency context for
# the pipeline probe. LABELLED wall-clock-derived (aie.utils.benchmark.
# run_iters, host timer around kernel.wait()) -- NOT a hardware trace (the
# pipeline design's own trace attempts both failed to route or produced no
# events; see pipeline_gemm_gelu_probe.py). This is context, not a
# performance claim: the two designs differ in scale from any production
# shape (2 rows, one tiny tile), so this is "does one pipelined dispatch
# cost roughly what two dispatches of comparable size would", not a
# production number.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage: python pipeline_bench.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

import aie.iron as iron                                # noqa: E402
from aie.iron.device import from_name                    # noqa: E402
from aie.utils.benchmark import run_iters                 # noqa: E402
from npue import tile_b, to_bf16_bits                      # noqa: E402
from pipeline_gemm_gelu_probe import (                      # noqa: E402
    pipeline_gemm_gelu, M, K, N, TK, TN)
from pipeline_diag_gemm_only import gemm_only               # noqa: E402


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)

    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b), TK, TN, 8, 8)
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    Y = iron.zeros(M * N, dtype=np.float32, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = a.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K, N)

    res_pipe = run_iters(pipeline_gemm_gelu, A, B, Y, trace_config=None,
                         identity=False, warmup=5, iters=20)
    res_gemm = run_iters(gemm_only, A, B, C, warmup=5, iters=20)

    def fmt(res, label):
        npu = res.npu
        e2e = res.e2e
        print(f"  {label:<28} npu(avg/min)="
              f"{(npu.avg_us if npu else float('nan')):7.1f}/"
              f"{(npu.min_us if npu else float('nan')):7.1f} us   "
              f"e2e(avg/min)={e2e.avg_us:7.1f}/{e2e.min_us:7.1f} us")

    print("Dispatch latency (wall-clock-derived, NOT a hardware trace):")
    fmt(res_pipe, "pipeline (GEMM->GELU, 1 dispatch)")
    fmt(res_gemm, "GEMM-only (same M/K/N, 1 dispatch)")
    if res_pipe.npu and res_gemm.npu:
        print(f"  ratio (pipeline / gemm-only) npu avg: "
              f"{res_pipe.npu.avg_us / res_gemm.npu.avg_us:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
