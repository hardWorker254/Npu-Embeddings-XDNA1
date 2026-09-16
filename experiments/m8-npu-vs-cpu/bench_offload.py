# NpuEmbeddings -- how much CPU does the NPU path actually free up?
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS IS THE MEASUREMENT THAT MATTERS
# ----------------------------------------
# tasks/0018 measured throughput and found the GEMM engine 4.3-4.8x faster than
# a 12-thread CPU. But throughput was never the whole case for an NPU: parity
# alone is worth it if the work moves off the CPU and costs less energy.
# docs/04-model states that directly -- "<5 W package, CPU stays free" sits
# alongside the throughput tiers as a requirement.
#
# Package power is not readable on this machine: the Windows "Power Meter" and
# "Energy Meter" counter sets exist but expose no instances, so there is no
# energy figure to be had without external hardware.
#
# CPU TIME CONSUMED is the robust stand-in, and it is the direct measure of
# offload rather than a proxy for it. `time.process_time()` sums user+kernel CPU
# across every thread of the process, so a 12-thread BLAS burning 12 cores for
# one second reads as 12 CPU-seconds. It also tracks energy reasonably: on a
# fixed part, CPU-seconds at full clock is roughly proportional to CPU energy.
#
# Reported per unit of work:
#   cpu_seconds / wall_seconds  =  cores occupied
#   cpu_seconds per GEMM        =  what the caller has to pay in CPU
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU), quiesced.
# Usage:
#   python experiments\m8-npu-vs-cpu\bench_offload.py

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))

import aie.iron as iron                       # noqa: E402
from aie.iron import str_to_dtype             # noqa: E402
from aie.iron.device import from_name         # noqa: E402
from ml_dtypes import bfloat16                # noqa: E402
from gemm_pretiled import pretiled_array      # noqa: E402

SHAPES = [("qkv", 384, 1152), ("attn_out", 384, 384),
          ("ffn_up", 384, 1536), ("ffn_down", 1536, 384)]


def measure(fn, iters):
    """Wall and CPU seconds for `iters` calls. process_time() counts every
    thread, so a multithreaded BLAS shows up as more CPU than wall."""
    fn()                                       # warm
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(iters):
        fn()
    return time.perf_counter() - w0, time.process_time() - c0


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    # time.process_time() on Windows ticks at ~15.6 ms. At 20 iterations the
    # NPU's CPU cost lands ON that floor (readings of 0.0 / 15.6 / 31.2), so the
    # ratio would be a bound rather than a measurement. Enough iterations to
    # clear the floor by an order of magnitude.
    M, cols = 4096, 8
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")

    print(f"offload: {iters} iterations of each GEMM at M={M}, {cols} columns")
    print("  cores = CPU-seconds / wall-seconds; 1.0 means one core fully busy\n")
    print(f"  {'shape':<10} {'path':<5} {'wall ms':>9} {'cpu ms':>9} "
          f"{'cores':>7} {'cpu ms/GEMM':>12}")

    rows = []
    for name, K, N in SHAPES:
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)

        w_cpu, c_cpu = measure(lambda: a @ b, iters)

        A = iron.zeros((M, K), dtype=dt_in, device="npu")
        B = iron.zeros((K, N), dtype=dt_in, device="npu")
        C = iron.zeros(M * N, dtype=dt_out, device="npu")
        A[:] = a.astype(bfloat16)
        B[:] = b.astype(bfloat16)

        def npu_call():
            pretiled_array(A, B, C, M=M, K=K, N=N, m=64, k=64, n=48,
                           n_aie_cols=cols, dtype_in_str="bf16",
                           dtype_out_str="f32",
                           emulate_bf16_mmul_with_bfp16=True,
                           pretiled=False, trace_config=None)

        w_npu, c_npu = measure(npu_call, iters)

        for path, w, c in (("cpu", w_cpu, c_cpu), ("npu", w_npu, c_npu)):
            rows.append(dict(shape=name, path=path, M=M, K=K, N=N,
                             wall_ms=w * 1e3, cpu_ms=c * 1e3,
                             cores=c / w, cpu_ms_per_gemm=c * 1e3 / iters))
            print(f"  {name:<10} {path:<5} {w * 1e3:>9.1f} {c * 1e3:>9.1f} "
                  f"{c / w:>7.2f} {c * 1e3 / iters:>12.2f}")

    print()
    tot = {p: sum(r["cpu_ms_per_gemm"] for r in rows if r["path"] == p)
           for p in ("cpu", "npu")}
    wall = {p: sum(r["wall_ms"] for r in rows if r["path"] == p) / iters
            for p in ("cpu", "npu")}
    print(f"  all four GEMMs, per encode-layer-equivalent:")
    print(f"    CPU path: {wall['cpu']:7.2f} ms wall, {tot['cpu']:7.2f} ms CPU")
    print(f"    NPU path: {wall['npu']:7.2f} ms wall, {tot['npu']:7.2f} ms CPU")
    TICK_MS = 15.6                    # Windows process_time resolution
    floor = TICK_MS * len(SHAPES) / iters
    print(f"    -> the NPU path costs {tot['cpu'] / tot['npu']:.1f}x LESS CPU "
          f"for {wall['cpu'] / wall['npu']:.1f}x less wall time")
    print(f"    (timer floor is {floor:.3f} ms/GEMM-set; NPU measured "
          f"{tot['npu']:.2f} ms, i.e. {tot['npu'] / floor:.0f}x above it)"
          if tot["npu"] > 0 else
          "    (NPU CPU cost is BELOW the timer floor -- raise iterations)")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "bench_offload.json").write_text(json.dumps({
        "kind": "wall clock + CPU time (docs/05-measurement: end-to-end only)",
        "note": "package power not readable on this machine; Windows Power "
                "Meter / Energy Meter counter sets expose no instances",
        "M": M, "cols": cols, "iters": iters, "rows": rows,
        "cpu_ms_total": tot, "wall_ms_total": wall,
        "cpu_reduction": tot["cpu"] / tot["npu"],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {(ARTIFACTS / 'bench_offload.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
