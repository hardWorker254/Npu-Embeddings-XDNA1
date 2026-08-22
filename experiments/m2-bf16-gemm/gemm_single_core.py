# NpuEmbeddings -- M2: bf16 GEMM on a single AIE core, traced
# SPDX-License-Identifier: Apache-2.0
#
# Derived from mlir-aie
#   programming_examples/getting_started/03_matrix_multiplication_single_core/
#   Copyright (C) 2025 Advanced Micro Devices, Inc.
#   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# WHY THIS EXISTS
# ---------------
# M2 needs a bf16 GEMM cycle baseline we can trust before scaling to the whole
# array. Single core first, because:
#
#   * It is the same design the pre-existing int16 traces came from, so bf16 vs
#     int16 is a clean single-variable comparison.
#   * `whole_array.py` (the multi-core design) has NO trace support at all
#     (`grep -c trace` == 0), so it cannot answer a cycles question yet.
#   * A single core has a closed-form expected cycle count, so the measurement
#     can be checked against theory rather than against hope.
#
# THE THEORY WE CHECK AGAINST
# ---------------------------
# The kernel issues `aie::mmul` intrinsics of shape (r, s, t) taken from
# `kernels.mm(...).mac_dims`. On aie2p:
#
#     bf16 -> bf16            (r,s,t) = (4, 8, 8)   -> 256 MACs per intrinsic
#     bf16 -> bf16 + bfp16    (r,s,t) = (8, 8, 8)   -> 512 MACs per intrinsic
#
# One m x k x n tile therefore needs (m/r) * (k/s) * (n/t) intrinsics. If the
# core issued one intrinsic per cycle with no overhead, that count IS the cycle
# count -- AIE cores never stall, so there is nothing else to wait for.
# Measured / ideal is then a real efficiency number, not a guess.
#
# Usage (from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced):
#     python gemm_single_core.py                          # 256^3 bf16 baseline
#     python gemm_single_core.py --emulate-bfp16          # the 8x8x8 A/B
#     python gemm_single_core.py --preset qkv             # MiniLM fused QKV
#     python gemm_single_core.py --dtype i16              # reproduce prior data

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron import kernels
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.trace import TraceConfig

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"

# MiniLM-L6-v2 / bge-small GEMM shapes. M is (batch * seq_len).
# See docs/04-model/ -- these are the four per-layer shapes; head_dim=32 only
# appears inside attention, which is 5.3% of the work at seq 128.
PRESETS = {
    "square":   dict(M=256, K=256,  N=256),   # sanity baseline
    "qkv":      dict(M=256, K=384,  N=1152),  # fused Q,K,V projection
    "proj":     dict(M=256, K=384,  N=384),   # attention output projection
    "ffn_up":   dict(M=256, K=384,  N=1536),  # FFN up
    "ffn_down": dict(M=256, K=1536, N=384),   # FFN down
}

DTYPES = {"bf16": bfloat16, "i16": np.int16, "i8": np.int8}
OUT_DTYPES = {"bf16": bfloat16, "f32": np.float32, "i16": np.int16, "i32": np.int32}
# Sensible accumulator per input dtype -- bf16 must accumulate in fp32.
DEFAULT_OUT = {"bf16": "f32", "i16": "i32", "i8": "i32"}


@iron.jit
def gemm_single_core(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    element_type: CompileTime[type],
    out_type: CompileTime[type],
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """C = A @ B on one compute tile, tiled m x k x n, accumulating over K.

    out_type is deliberately separate from element_type. With bf16 in AND out,
    the L1 accumulator tile is bf16, so every one of the K/k accumulation steps
    re-rounds to 8 mantissa bits -- the error you then measure is the storage
    format's, not the kernel's. Every paper we read (2504.03083, 2607.11211)
    specifies bf16 in / fp32 accumulate. Use out_type=float32 for real work.
    """
    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=element_type,
        output_dtype=out_type,
        vectorized=True,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
    )
    # Read the geometry from the kernel we actually compiled -- hardcoding
    # (r,s,t) breaks the moment the dtype or the bfp16 flag changes.
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[element_type]]
    B_ty = np.ndarray[(K, N), np.dtype[element_type]]
    C_ty = np.ndarray[(M, N), np.dtype[out_type]]
    a_ty = np.ndarray[(m * k,), np.dtype[element_type]]
    b_ty = np.ndarray[(k * n,), np.dtype[element_type]]
    c_ty = np.ndarray[(m * n,), np.dtype[out_type]]

    # DMA-level layout transforms repack m*k / k*n / m*n tiles into the
    # r*s / s*t / r*t sub-tiles the MMUL intrinsic expects.
    fifo_A_L3L2 = ObjectFifo(a_ty, name="A_L3L2")
    tap_A = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = ObjectFifo(b_ty, name="B_L3L2")
    tap_B = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = ObjectFifo(c_ty, name="C_L1L2")
    tap_C = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=tap_C.transformation_dims, name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in range_(m * n):
                elem_out[i] = 0
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
        trace=1,
    )

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for tile_row in range(M // m):
            tg = TaskGroup()
            a_prod.fill(A, tap=a_taps[tile_row], group=tg)
            b_prod.fill(B, tap=b_tap, group=tg)
            c_cons.drain(C, tap=c_taps[tile_row], wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            fifo_A_L3L2.prod(), fifo_B_L3L2.prod(), fifo_C_L2L3.cons()])

    program = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size, workers=[worker])
    return program.resolve_program()


def make_inputs(M, K, N, dt):
    """Allocate NPU-resident inputs and return them with their numpy views.

    Built with iron.rand / iron.randint rather than iron.tensor(np_array):
    iron.tensor() cannot ingest an ml_dtypes bfloat16 array (it tries to cast
    to uint32 and raises). Constructing on the device and reading .numpy()
    back gives us both the device tensor and the exact values the reference
    must match.

    Magnitudes are kept small on purpose. With K=1536, products of large
    values accumulate past what bf16 can represent, and the 'error' you then
    measure is the reference's fault rather than the kernel's.
    """
    if np.issubdtype(dt, np.integer):
        A = iron.randint(-16, 16, (M, K), dtype=dt, device="npu")
        B = iron.randint(-16, 16, (K, N), dtype=dt, device="npu")
    else:
        A = iron.rand((M, K), dtype=dt, device="npu")
        B = iron.rand((K, N), dtype=dt, device="npu")
    return A, B, A.numpy().copy(), B.numpy().copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="M2: bf16 GEMM on one AIE core, traced")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="square")
    ap.add_argument("-M", type=int)
    ap.add_argument("-K", type=int)
    ap.add_argument("-N", type=int)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--out-dtype", choices=sorted(OUT_DTYPES), default=None,
                    help="accumulator/output dtype; defaults to f32 for bf16 "
                         "inputs (correct) and to --dtype for integer inputs")
    ap.add_argument("--emulate-bfp16", action="store_true",
                    help="AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16: 4x8x8 -> 8x8x8")
    ap.add_argument("--trace-size", type=int, default=262144)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    # MUST come before any kernels.mm() call and before the JIT design runs.
    #
    # kernels._common._detect_arch() calls get_current_device(probe_runtime=False),
    # which raises unless a device was set EXPLICITLY -- and the handler then
    # silently falls back to 'aie2' (NPU1). The consequences on this NPU2 box:
    #   * bf16 mac_dims resolve to (4,8,4) instead of (4,8,8)  -> HALF the MACs
    #   * emulate_bf16_mmul_with_bfp16 becomes a NO-OP, because the emulated
    #     table is only consulted when arch == 'aie2p'
    #   * the aie_kernels/aie2/ source is compiled instead of aie_kernels/aie2p/
    # None of this errors or warns at the user level. Verified: before this call
    # mac_dims == (4,8,4) for both plain and bfp16; after it, (4,8,8) and (8,8,8).
    iron.set_current_device(from_name("npu2", n_cols=None))

    shape = dict(PRESETS[args.preset])
    for key in ("M", "K", "N"):
        if getattr(args, key) is not None:
            shape[key] = getattr(args, key)
    M, K, N = shape["M"], shape["K"], shape["N"]
    m, k, n = args.m, args.k, args.n
    dt = DTYPES[args.dtype]
    out_name = args.out_dtype or DEFAULT_OUT[args.dtype]
    odt = OUT_DTYPES[out_name]

    # Fail loudly and early rather than deep inside MLIR.
    problems = []
    if M % m: problems.append(f"M({M}) % m({m}) != 0")
    if K % k: problems.append(f"K({K}) % k({k}) != 0")
    if N % n: problems.append(f"N({N}) % n({n}) != 0")
    if problems:
        print("shape constraints violated:\n  " + "\n  ".join(problems))
        return 2

    probe = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=dt, output_dtype=odt,
                       vectorized=True,
                       emulate_bf16_mmul_with_bfp16=args.emulate_bfp16)
    r, s, t = probe.mac_dims

    # mm.cc uses a 2x2 accumulator expansion, so m and n need 2r / 2t.
    for label, dim, need in (("m", m, 2 * r), ("k", k, s), ("n", n, 2 * t)):
        if dim % need:
            print(f"tile constraint violated: {label}({dim}) % {need} != 0 "
                  f"(mac_dims r,s,t = {r},{s},{t}; mm.cc needs m%2r, k%s, n%2t)")
            return 2

    # --- L1 budget pre-flight ---------------------------------------------
    # aiecc's failure mode here is an opaque "'aie.tile' op Basic sequential
    # allocation also failed", which says nothing about what overflowed.
    # ObjectFifos default to depth 2 (double buffering), so each of A, B and C
    # is resident twice. 64 KB L1 per compute tile (docs/01-hardware/).
    #
    # Worked example that cost us a compile: m=k=n=64 with f32 output is
    #   2*(64*64*2 + 64*64*2 + 64*64*4) = 65536 B = exactly 64 KB -> fails,
    # while the same tile with bf16 output is 48 KB and fits. fp32 accumulation
    # doubles the C tile, so it buys correctness at the cost of tile size.
    L1_BYTES = 64 * 1024
    in_sz = np.dtype(dt).itemsize
    out_sz = np.dtype(odt).itemsize
    l1_bytes = 2 * (m * k * in_sz + k * n * in_sz + m * n * out_sz)
    print(f"L1 est.    : {l1_bytes} B of {L1_BYTES} B "
          f"(A {m*k*in_sz}B + B {k*n*in_sz}B + C {m*n*out_sz}B, double-buffered)")
    if l1_bytes >= L1_BYTES:
        print(f"\nERROR: tile does not fit L1 ({l1_bytes} >= {L1_BYTES}).")
        print("  Reduce -m/-k/-n, or use a narrower --out-dtype.")
        print(f"  e.g. -m {m} -k {k} -n {n // 2} would need "
              f"{2 * (m*k*in_sz + k*(n//2)*in_sz + m*(n//2)*out_sz)} B")
        return 2

    tag = args.tag or (
        f"{args.dtype}_{out_name}{'_bfp16' if args.emulate_bfp16 else ''}"
        f"_{M}x{K}x{N}_t{m}x{k}x{n}"
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trace_txt = ARTIFACTS / f"trace_{tag}.txt"
    trace_json = ARTIFACTS / f"trace_{tag}.json"
    mlir_copy = ARTIFACTS / f"mlir_{tag}.mlir"
    result_json = ARTIFACTS / f"result_{tag}.json"

    # --- theory -----------------------------------------------------------
    intrinsics_per_tile = (m // r) * (k // s) * (n // t)
    macs_per_tile = m * k * n
    tile_invocations = (M // m) * (N // n) * (K // k)
    total_macs = M * K * N

    print(f"shape      : M={M} K={K} N={N}   tile m={m} k={k} n={n}   "
          f"dtype={args.dtype}->{out_name}")
    print(f"bfp16 emul : {args.emulate_bfp16}")
    print(f"mac_dims   : (r,s,t) = ({r},{s},{t})  -> {r*s*t} MACs per mmul intrinsic")
    print(f"per tile   : {macs_per_tile} MACs = {intrinsics_per_tile} intrinsics")
    print(f"total      : {tile_invocations} tile invocations, {total_macs/1e6:.1f} MMAC")

    # --- run --------------------------------------------------------------
    A, B, A_np, B_np = make_inputs(M, K, N, dt)
    C = iron.zeros(M * N, dtype=odt, device="npu")

    trace_cfg = TraceConfig(trace_size=args.trace_size, trace_file=str(trace_txt))
    gemm_single_core(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, element_type=dt,
                     out_type=odt,
                     emulate_bf16_mmul_with_bfp16=args.emulate_bfp16,
                     trace_config=trace_cfg)

    # --- correctness ------------------------------------------------------
    got = C.numpy().reshape(M, N).astype(np.float64)
    ref = A_np.astype(np.float64) @ B_np.astype(np.float64)
    denom = np.linalg.norm(ref)
    rel_fro = float(np.linalg.norm(got - ref) / denom) if denom else 0.0
    # Integer paths must be exact. bf16 tolerance is the documented 5e-3
    # (docs/05-measurement); bfp16 emulation shares an exponent across each
    # 8-element block, so it is legitimately coarser -- tracked separately
    # rather than silently accepted under the bf16 threshold.
    if np.issubdtype(dt, np.integer):
        tol = 0.0
    elif args.emulate_bfp16:
        tol = 5e-2
    else:
        tol = 5e-3
    ok = rel_fro <= tol
    print(f"\ncorrectness: rel_frobenius={rel_fro:.3e}  tol={tol:.0e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # --- trace ------------------------------------------------------------
    # A silently-empty trace is the failure mode that produced a 0-byte
    # trace_512x512x512.txt in the reference tree. Never treat it as a result.
    size = trace_txt.stat().st_size if trace_txt.exists() else 0
    print(f"trace.txt  : {size} bytes")
    if size == 0:
        print("ERROR: trace is empty -- raise --trace-size or shrink the shape.")
        return 3

    phys = getattr(trace_cfg, "physical_mlir_path", None)
    shutil.copy(phys, mlir_copy)
    trace_cfg.trace_to_json(phys, str(trace_json))

    from aie.utils.trace.utils import get_cycles_summary, get_vector_time

    summary = get_cycles_summary(str(trace_json))
    record = dict(shape=dict(M=M, K=K, N=N), tile=dict(m=m, k=k, n=n),
                  dtype=args.dtype, out_dtype=out_name,
                  emulate_bfp16=args.emulate_bfp16,
                  mac_dims=[r, s, t], intrinsics_per_tile=intrinsics_per_tile,
                  macs_per_tile=macs_per_tile,
                  rel_frobenius=rel_fro, tol=tol, correctness_pass=ok, cores=[])

    # Efficiency is measured as MACs/cycle against the datapath peak, NOT as
    # "intrinsics per cycle". aie::mmul is not one hardware instruction -- the
    # disassembly shows it decomposed into many vmac.f plus vextbcst/vshuffle
    # operand preparation, so an intrinsic-per-cycle model is meaningless.
    # Peak figures: docs/01-hardware/ (aie2p, MACs/cycle/core).
    peak = {"bf16": 256, "i16": 128, "i8": 512}[args.dtype]
    print("\n--- cycle summary ---")
    print(f"peak assumed: {peak} MACs/cycle/core ({args.dtype}, aie2p)")
    for entry in summary:
        core, deltas = entry[0], [d for d in entry[1:] if d is not None]
        if not deltas:
            continue
        mn, mx = min(deltas), max(deltas)
        avg = sum(deltas) / len(deltas)
        macs_per_cycle = macs_per_tile / avg
        eff = macs_per_cycle / peak * 100.0
        print(f"{core}: n={len(deltas)}  min={mn}  avg={avg:.1f}  max={mx}")
        print(f"    {macs_per_cycle:.1f} MACs/cycle  ->  {eff:.1f}% of peak")
        record["cores"].append(dict(core=core, n=len(deltas), min=mn, avg=avg,
                                    max=mx, macs_per_cycle=macs_per_cycle,
                                    peak_macs_per_cycle=peak, efficiency_pct=eff))

    try:
        vt = get_vector_time(str(trace_json))
        print(f"vector time fraction: {vt}")
        record["vector_time"] = vt
    except Exception as exc:
        print(f"(vector time unavailable: {exc})")

    result_json.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {result_json.name}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
