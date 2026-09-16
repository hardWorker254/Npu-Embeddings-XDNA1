# NpuEmbeddings -- NPU vs CPU on the MiniLM GEMMs.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS IS AND IS NOT
# -----------------------
# Wall clock, on both sides. Under docs/05-measurement that is legitimate for
# end-to-end throughput and forbidden as a kernel-cycle claim, so nothing here
# is presented as a statement about kernel quality -- only about how long the
# whole dispatch takes versus how long the CPU takes to do the same arithmetic.
#
# The NPU is a shared resource. Quiesce it first: `flm` and any Lemonade /
# Ryzen AI server hold it and will silently inflate these numbers. This script
# prints what it found still running so the reader can judge.
#
# The CPU side is numpy, which on this machine dispatches to a multithreaded
# BLAS -- i.e. the CPU gets all its cores. That is the fair comparison: it is
# what someone would actually run.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m8-npu-vs-cpu\bench_gemm.py

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))

import aie.iron as iron                       # noqa: E402
from aie.iron.device import from_name         # noqa: E402
import gemm_pretiled as g                     # noqa: E402

SHAPES = [
    ("qkv", 384, 1152),
    ("attn_out", 384, 384),
    ("ffn_up", 384, 1536),
    ("ffn_down", 1536, 384),
]
M_VALUES = [256, 1024, 4096]


def cpu_gemm_us(M, K, N, iters=20, warmup=5):
    """numpy fp32 matmul, best-of. Best-of rather than mean: we want the
    machine's capability, not its current interruptions -- same courtesy the
    NPU side gets from run_iters' min."""
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    for _ in range(warmup):
        a @ b
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def npu_variants():
    who = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process | Where-Object { $_.ProcessName -match 'flm|Lemonade|ryzen' } "
         "| ForEach-Object { $_.ProcessName }"],
        capture_output=True, text=True).stdout.split()
    return sorted(set(who))


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    others = npu_variants()
    print(f"host: {platform.processor()}")
    print(f"numpy {np.__version__}, BLAS threads = all cores")
    print(f"other NPU users still running: {others or 'none'}")
    if any(o.lower().startswith('flm') for o in others):
        print("  WARNING: flm is running and holds the NPU -- numbers are void")
    print()

    rows = []
    print(f"  {'shape':<10} {'M':>5} {'[K,N]':>12} {'CPU us':>9} {'NPU us':>9} "
          f"{'NPU/CPU':>8} {'CPU TF/s':>9} {'NPU TF/s':>9}")
    for M in M_VALUES:
        for name, K, N in SHAPES:
            cpu_us = cpu_gemm_us(M, K, N)
            b = g.bench_one(M, K, N, 64, 64, 48, 8, True, pretiled=False,
                            iters=50, warmup=10)
            npu_us = b.get("npu_min_us") or float("nan")
            flop = 2 * M * K * N
            row = dict(shape=name, M=M, K=K, N=N,
                       cpu_us=cpu_us, npu_us=npu_us,
                       cpu_tflops=flop / (cpu_us * 1e-6) / 1e12,
                       npu_tflops=flop / (npu_us * 1e-6) / 1e12,
                       npu_over_cpu=npu_us / cpu_us)
            rows.append(row)
            print(f"  {name:<10} {M:>5} {str([K, N]):>12} {cpu_us:>9.1f} "
                  f"{npu_us:>9.1f} {row['npu_over_cpu']:>7.2f}x "
                  f"{row['cpu_tflops']:>9.2f} {row['npu_tflops']:>9.2f}")

    print("\n  ('NPU/CPU' below 1.00 means the NPU is faster.)")
    for M in M_VALUES:
        sub = [r for r in rows if r["M"] == M]
        tot_cpu = sum(r["cpu_us"] for r in sub)
        tot_npu = sum(r["npu_us"] for r in sub)
        print(f"  M={M:<5} all four GEMMs: CPU {tot_cpu:8.1f} us, "
              f"NPU {tot_npu:8.1f} us  ->  {tot_cpu / tot_npu:.2f}x "
              f"{'NPU faster' if tot_npu < tot_cpu else 'CPU faster'}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "bench_gemm.json").write_text(json.dumps({
        "kind": "wall clock, end-to-end dispatch (docs/05-measurement)",
        "note": "NOT a kernel-cycle claim on either side",
        "other_npu_users": others,
        "numpy": np.__version__,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {(ARTIFACTS / 'bench_gemm.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
