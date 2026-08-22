# NpuEmbeddings -- M1 "Hello NPU"
# SPDX-License-Identifier: Apache-2.0
#
# Derived from mlir-aie programming_examples/getting_started/01_SAXPY/saxpy.py
#   Copyright (C) 2025 Advanced Micro Devices, Inc.
#   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# WHY THIS EXISTS
# ---------------
# M1's gate: prove the whole native-Windows chain works end to end --
# IRON design -> Peano kernel -> aiecc -> xclbin -> NPU -> hardware trace ->
# cycle counts. Everything after M1 is built on that chain, so it gets proven
# first, on the simplest kernel that still exercises all of it.
#
# Differences from the upstream example, and why:
#   1. All artifacts are written into THIS repo (artifacts/), never into
#      C:\dev\mlir-aie. That tree is a read-only reference.
#   2. trace.json is generated automatically. Upstream leaves the
#      trace_to_json() call commented out, so you get a raw trace.txt and no
#      cycle counts unless you know to run parse.py by hand.
#   3. The cycle summary is printed at the end -- the actual deliverable.
#   4. vec_size is passed to the kernel as -DSAXPY_VEC_SIZE, so the design and
#      the kernel cannot silently disagree about the tensor length.
#   5. --scalar runs the non-vectorised kernel, as a cheap ablation.
#
# Usage (from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced):
#     python saxpy.py
#     python saxpy.py --scalar
#     python saxpy.py --size 8192 --trace-size 16384

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.utils.config import cxx_header_path
from aie.utils.trace import TraceConfig

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"


@iron.jit
def saxpy(
    input0: In,
    input1: In,
    output: Out,
    *,
    N: CompileTime[int],
    element_type: CompileTime[type],
    kernel_name: CompileTime[str] = "saxpy",
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """z = 3*x + y on a single AIE compute tile.

    One worker, three ObjectFifos (x, y in; z out). Deliberately the smallest
    design that still has a real kernel, real DMA, and a real trace.
    """
    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N,), np.dtype[element_type]]

    # --- data movement -----------------------------------------------------
    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # --- the compute kernel ------------------------------------------------
    # use_chess is left at its default (False): Peano is the only kernel
    # compiler available on native Windows. See docs/02-toolchain/.
    saxpy_kernel = ExternalFunction(
        kernel_name,
        source_file=str(HERE / "saxpy.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DSAXPY_VEC_SIZE={N}"],
    )

    def core_body(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    # trace=1 marks this worker for event tracing.
    worker = Worker(
        core_body,
        fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), saxpy_kernel],
        trace=1,
    )

    # --- host <-> NPU dispatch --------------------------------------------
    def sequence(a_x, a_y, c_z, x_prod, y_prod, z_cons):
        x_prod.fill(a_x)
        y_prod.fill(a_y)
        z_cons.drain(c_z, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, in_ty, out_ty, of_x.prod(), of_y.prod(), of_z.cons()],
    )

    program = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size, workers=[worker])
    return program.resolve_program()


def main() -> int:
    ap = argparse.ArgumentParser(description="M1 Hello NPU: saxpy + trace")
    ap.add_argument("--size", type=int, default=4096, help="vector length")
    ap.add_argument("--trace-size", type=int, default=8192, help="trace buffer bytes")
    ap.add_argument(
        "--scalar",
        action="store_true",
        help="use the non-vectorised kernel (ablation)",
    )
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "scalar" if args.scalar else "vector"
    kernel_name = "saxpy_scalar" if args.scalar else "saxpy"

    trace_txt = ARTIFACTS / f"trace_{tag}_{args.size}.txt"
    trace_json = ARTIFACTS / f"trace_{tag}_{args.size}.json"
    mlir_copy = ARTIFACTS / f"input_with_addresses_{tag}_{args.size}.mlir"

    print(f"device : {iron.get_current_device()}")
    print(f"kernel : {kernel_name}  N={args.size}  dtype=bfloat16")

    x = iron.arange(args.size, dtype=bfloat16, device="npu")
    y = iron.arange(args.size, dtype=bfloat16, device="npu")
    z = iron.zeros_like(x)

    trace_cfg = TraceConfig(trace_size=args.trace_size, trace_file=str(trace_txt))

    saxpy(
        x,
        y,
        z,
        N=args.size,
        element_type=bfloat16,
        kernel_name=kernel_name,
        trace_config=trace_cfg,
    )

    # --- correctness -------------------------------------------------------
    # Checked in float32 to avoid comparing bf16 against bf16 rounding noise.
    got = z.numpy().astype(np.float32)
    ref = (3.0 * x.numpy().astype(np.float32)) + y.numpy().astype(np.float32)
    abs_err = np.abs(got - ref)
    denom = np.linalg.norm(ref)
    rel_fro = float(np.linalg.norm(got - ref) / denom) if denom else 0.0
    ok = rel_fro <= 5e-3

    print(f"\ncorrectness: max_abs_err={abs_err.max():.6g}  rel_frobenius={rel_fro:.3e}")
    print(f"             {'PASS' if ok else 'FAIL'} (threshold 5e-3, see docs/05-measurement)")

    # --- trace -------------------------------------------------------------
    # physical_mlir_path points at input_with_addresses.mlir in the JIT cache.
    # That is the file parse.py needs -- NOT the source MLIR. Copy it next to
    # the trace so this run stays reproducible even if the cache is cleared.
    phys = getattr(trace_cfg, "physical_mlir_path", None)
    print(f"\ntrace.txt  : {trace_txt}  ({trace_txt.stat().st_size if trace_txt.exists() else 0} bytes)")
    print(f"cache mlir : {phys}")

    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print("\nWARNING: trace.txt is missing or empty.")
        print("  Usual causes: trace buffer too small, ddr_id mismatch, or too few")
        print("  events to form a packet. See docs/05-measurement/.")
        return 1 if ok else 2

    if phys:
        shutil.copy(phys, mlir_copy)
        trace_cfg.trace_to_json(phys, str(trace_json))
        print(f"trace.json : {trace_json}")

        from aie.utils.trace.utils import print_cycles_summary

        print("\n--- cycle summary ---")
        print_cycles_summary(str(trace_json))

        try:
            from aie.utils.trace.utils import get_vector_time

            print(f"vector time fraction: {get_vector_time(str(trace_json))}")
        except Exception as exc:  # not fatal -- the cycle count is the deliverable
            print(f"(vector time unavailable: {exc})")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
