# NpuEmbeddings -- M2 (multi-core): whole-array bf16 GEMM, TRACED
# SPDX-License-Identifier: Apache-2.0
#
# Derived from mlir-aie
#   programming_examples/basic/matrix_multiplication/whole_array/whole_array.py
#   Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
#   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# WHY A COPY EXISTS HERE
# ----------------------
# The upstream whole_array design has NO trace support at all:
#     grep -c trace whole_array.py  ->  0
# no trace_config parameter, no worker marked trace=1, no rt.enable_trace(). It
# therefore cannot answer a cycles question, and our measurement doctrine
# (docs/05-measurement/) forbids using wall clock for an NPU claim. Rather than
# depend on and patch the read-only reference tree, we own a copy -- which is
# also what constraint C2 requires ("build everything here").
#
# OUR CHANGES vs UPSTREAM, all marked `# NPUE:` below
#   1. trace_config / trace_row / trace_col parameters.
#   2. Exactly ONE worker is marked trace=1. Tracing all 32 cores would flood
#      the buffer and tell us nothing extra -- every core runs the same program
#      on a different tile, so one core's cycles ARE the per-core cost.
#   3. rt.enable_trace(...) inside the runtime sequence.
#   4. Explicit set_current_device -- without it IRON silently compiles for
#      NPU1 and the bfp16 flag becomes a no-op (research/notes/0002).
#   5. AOT/CLI/taps-visualisation machinery dropped; measurement driver added.
#
# Usage (from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced):
#     python gemm_whole_array.py --cols 4 --emulate-bfp16
#     python gemm_whole_array.py --cols 8 --emulate-bfp16 -M 512 -K 512 -N 512
#     python gemm_whole_array.py --scaling            # 1,2,4,8 column sweep

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.utils.trace import TraceConfig

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"

PRESETS = {
    "square":   dict(M=512, K=512,  N=512),
    "qkv":      dict(M=512, K=384,  N=1152),
    "proj":     dict(M=512, K=384,  N=384),
    "ffn_up":   dict(M=512, K=384,  N=1536),
}


def _build_design(dev, M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str,
                  emulate_bf16_mmul_with_bfp16, trace_config, trace_row, trace_col,
                  trace_egress_col=0):
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=dtype_in, output_dtype=dtype_out,
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    assert M % (m * n_aie_rows) == 0, "A must tile into (m*n_aie_rows, k) blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0, "B must tile into (k, n*n_aie_cols) blocks"
    assert m % r == 0 and k % s == 0 and n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # A: one L3->L2 fifo per shim, split down the rows of a column.
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * (stop_row - start_row)
        tmp = A_l3l2_fifos[i].cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=dims,
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = tmp[j]

    # B: one fifo per column, forwarded through the mem tile. C: joined back.
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}",
            dims_to_stream=[(k // s, s * n), (n // t, t), (s, n), (t, 1)],
        )
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        tmp = C_l2l3_fifos[col].prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = tmp[j]

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    # NPUE: mark exactly one core for tracing. Every core runs the same program
    # on a different tile, so one core's event0..event1 window IS the per-core
    # cost; tracing 32 cores would only overflow the buffer.
    def _mk(row, col):
        return Worker(
            core_fn,
            [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
             C_l1l2_fifos[row][col].prod(), zero_kernel, matmul_kernel],
            stack_size=0xD00,
            trace=1 if (row == trace_row and col == trace_col) else None,
        )

    workers = Worker.grid(n_aie_rows, n_aie_cols, _mk)

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False)
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n), tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), tile_group_col_major=True, prune_step=False)
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), prune_step=False)
    c_index = 0

    def sequence(A, B, C, A_prods, B_prods, C_conss):
        nonlocal c_index
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([tb_max_n_rows // 2,
                                         M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    C_conss[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        off = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            A_prods[col].fill(A, tap=A_tiles[off], group=tg)
                        B_prods[col].fill(B, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            [f.prod() for f in A_l3l2_fifos],
                            [f.prod() for f in B_l3l2_fifos],
                            [f.cons() for f in C_l2l3_fifos]])

    program = Program(dev, rt, workers=[w for row in workers for w in row])
    # NPUE: enable tracing on the single marked worker.
    # egress_shim_col matters: the trace stream needs a route to a shim, and
    # on a fully-packed array the obvious column is already occupied. Moving
    # the egress is the difference between "Unable to find a legal routing"
    # and a working 8-column trace.
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size,
                             workers=[workers[trace_row][trace_col]],
                             egress_shim_col=trace_egress_col)
    return program.resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    m: CompileTime[int], k: CompileTime[int], n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str], dtype_out_str: CompileTime[str],
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    trace_config: CompileTime[TraceConfig | None] = None,
    trace_row: CompileTime[int] = 0,
    trace_col: CompileTime[int] = 0,
    trace_egress_col: CompileTime[int] = 0,
):
    return _build_design(iron.get_current_device(), M, K, N, m, k, n, n_aie_cols,
                         dtype_in_str, dtype_out_str,
                         emulate_bf16_mmul_with_bfp16,
                         trace_config, trace_row, trace_col, trace_egress_col)


# Adding ONE trace flow exhausts routing on most column counts -- the design
# itself compiles at 1/2/4/8, but with trace only 2 and 4 are routable, and only
# with these specific (trace_col, egress_shim_col) pairs. Found by exhaustive
# search over trace_col 0..3 x egress 0..cols-1; see tasks/0004.
#   cols=1 : no working combination (packet_rules DMA0 collision)
#   cols=2 : (1, 1)
#   cols=4 : (0, 0)
#   cols=8 : no working combination -- "Unable to find a legal routing", and
#            --packet-sw-objFifos instead hits "max number of packet IDs reached"
TRACE_ROUTING = {2: (1, 1), 4: (0, 0)}


def run_one(M, K, N, m, k, n, cols, emulate, trace_size, dtype_in="bf16",
            dtype_out="f32", verbose=True):
    """Compile+run one configuration. Returns a result dict (or None on failure)."""
    dt_in = str_to_dtype(dtype_in)
    dt_out = str_to_dtype(dtype_out)

    in_sz, out_sz = np.dtype(dt_in).itemsize, np.dtype(dt_out).itemsize
    l1 = 2 * (m * k * in_sz + k * n * in_sz + m * n * out_sz)
    if l1 >= 64 * 1024:
        print(f"  SKIP cols={cols}: tile needs {l1} B of L1 (max 65536)")
        return None

    tag = f"wa{cols}c_{dtype_in}_{dtype_out}{'_bfp16' if emulate else ''}_{M}x{K}x{N}_t{m}x{k}x{n}"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trace_txt = ARTIFACTS / f"trace_{tag}.txt"
    trace_json = ARTIFACTS / f"trace_{tag}.json"

    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")
    A_np, B_np = A.numpy().copy(), B.numpy().copy()

    if cols not in TRACE_ROUTING:
        print(f"  cols={cols}: NOT TRACEABLE (routing exhausted); "
              f"traceable widths are {sorted(TRACE_ROUTING)}")
        return None
    tcol, egress = TRACE_ROUTING[cols]

    cfg = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt))
    whole_array(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
                dtype_in_str=dtype_in, dtype_out_str=dtype_out,
                emulate_bf16_mmul_with_bfp16=emulate,
                trace_config=cfg, trace_row=0, trace_col=tcol,
                trace_egress_col=egress)

    got = C.numpy().reshape(M, N).astype(np.float64)
    ref = A_np.astype(np.float64) @ B_np.astype(np.float64)
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    tol = 5e-2 if emulate else 5e-3
    ok = rel_fro <= tol

    size = trace_txt.stat().st_size if trace_txt.exists() else 0
    if size == 0:
        print(f"  cols={cols}: EMPTY TRACE -- raise --trace-size")
        return None

    phys = getattr(cfg, "physical_mlir_path", None)
    cfg.trace_to_json(phys, str(trace_json))
    from aie.utils.trace.utils import get_cycles_summary

    deltas = []
    for entry in get_cycles_summary(str(trace_json)):
        deltas += [d for d in entry[1:] if d is not None]
    if not deltas:
        print(f"  cols={cols}: no event0/event1 pairs in trace")
        return None

    avg = sum(deltas) / len(deltas)
    macs_tile = m * k * n
    per_core = macs_tile / avg
    total = per_core * 4 * cols                 # 4 rows x cols cores
    peak_core = {"bf16": 256, "i16": 128, "i8": 512}[dtype_in]

    if verbose:
        print(f"  cols={cols:>2}  cores={4*cols:>2}  n={len(deltas):>5}  "
              f"avg={avg:8.1f} cyc  per-core={per_core:6.1f} MACs/cyc "
              f"({per_core/peak_core*100:5.1f}%)  array={total:8.1f} MACs/cyc  "
              f"relfro={rel_fro:.2e} {'PASS' if ok else 'FAIL'}")

    return dict(cols=cols, cores=4 * cols, M=M, K=K, N=N, m=m, k=k, n=n,
                dtype_in=dtype_in, dtype_out=dtype_out, emulate_bfp16=emulate,
                invocations=len(deltas), avg_cycles=avg,
                min_cycles=min(deltas), max_cycles=max(deltas),
                macs_per_cycle_per_core=per_core,
                macs_per_cycle_array=total,
                peak_per_core=peak_core,
                efficiency_pct=per_core / peak_core * 100.0,
                rel_frobenius=rel_fro, correctness_pass=ok)


def bench_one(M, K, N, m, k, n, cols, emulate, iters=50, warmup=10,
              dtype_in="bf16", dtype_out="f32"):
    """Wall-clock end-to-end throughput, NO trace.

    Legitimate under docs/05-measurement/: wall clock is valid for end-to-end
    throughput and host/dispatch cost -- it is only forbidden as a *kernel
    cycle* claim. This exists because a fully-packed 8-column array CANNOT be
    core-traced (routing exhausted), so it is the only way to check whether the
    trace-based extrapolation from 2/4 columns actually holds at 32 cores.

    Interpret with care: this includes dispatch and DMA, and the NPU is a
    shared resource -- quiesce other users first.
    """
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)
    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, trace_config=None)

    from aie.utils.benchmark import run_iters
    res = run_iters(whole_array, A, B, C, warmup=warmup, iters=iters, **kw)

    # BenchmarkResult.{e2e,npu} are Stats with avg_us / min_us / max_us
    # (microseconds). npu is None when the callable exposes no npu_time.
    npu_us = getattr(getattr(res, "npu", None), "avg_us", None)
    npu_min = getattr(getattr(res, "npu", None), "min_us", None)
    e2e_us = getattr(getattr(res, "e2e", None), "avg_us", None)
    total_macs = M * K * N
    out = dict(cols=cols, cores=4 * cols, iters=iters, total_macs=total_macs,
               npu_avg_us=npu_us, npu_min_us=npu_min, e2e_avg_us=e2e_us)
    # 2 FLOP per MAC
    if npu_us:
        out["tflops_npu"] = 2 * total_macs / (npu_us * 1e-6) / 1e12
    if npu_min:
        out["tflops_npu_best"] = 2 * total_macs / (npu_min * 1e-6) / 1e12
    if e2e_us:
        out["tflops_e2e"] = 2 * total_macs / (e2e_us * 1e-6) / 1e12
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 multi-core: whole-array bf16 GEMM, traced")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="square")
    ap.add_argument("-M", type=int); ap.add_argument("-K", type=int); ap.add_argument("-N", type=int)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--cols", type=int, choices=[1, 2, 4, 8], default=4)
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--trace-size", type=int, default=262144)
    ap.add_argument("--scaling", action="store_true",
                    help="sweep 1,2,4,8 columns and report scaling (traced)")
    ap.add_argument("--bench", action="store_true",
                    help="wall-clock end-to-end sweep, no trace (works at 8 cols)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    # See research/notes/0002 -- without this IRON silently targets NPU1.
    iron.set_current_device(from_name("npu2", n_cols=None))

    shape = dict(PRESETS[args.preset])
    for key in ("M", "K", "N"):
        if getattr(args, key) is not None:
            shape[key] = getattr(args, key)
    M, K, N = shape["M"], shape["K"], shape["N"]

    print(f"shape M={M} K={K} N={N}  tile {args.m}x{args.k}x{args.n}  "
          f"bf16->f32  bfp16={args.emulate_bfp16}")
    print(f"peak/core = 256 MACs/cyc (bf16 aie2p); array peak = 256 * 4 * cols\n")

    if args.bench:
        print("WALL-CLOCK end-to-end (no trace). NOT a kernel-cycle claim.")
        print("Quiesce other NPU users before trusting these numbers.\n")
        rows = []
        for cols in ([1, 2, 4, 8] if args.scaling else [args.cols]):
            if N % (args.n * cols):
                print(f"  SKIP cols={cols}: N({N}) % (n*cols) != 0")
                continue
            try:
                r = bench_one(M, K, N, args.m, args.k, args.n, cols,
                              args.emulate_bfp16, iters=args.iters)
            except Exception as exc:
                print(f"  cols={cols}: FAILED -- {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]}")
                continue
            rows.append(r)
            npu = f"{r['npu_avg_us']:8.1f}us" if r.get("npu_avg_us") else "    n/a"
            e2e = f"{r['e2e_avg_us']:8.1f}us" if r.get("e2e_avg_us") else "    n/a"
            tf = f"{r.get('tflops_npu', 0):5.2f}" if r.get("tflops_npu") else "  n/a"
            tfb = f"{r.get('tflops_npu_best', 0):5.2f}" if r.get("tflops_npu_best") else "  n/a"
            print(f"  cols={cols:>2} cores={r['cores']:>2}  npu={npu} e2e={e2e}  "
                  f"{tf} TFLOP/s (best {tfb})")
        if rows:
            base = rows[0]
            print(f"\n--- wall-clock scaling vs {base['cols']} cols ---")
            for r in rows:
                if r.get("tflops_npu") and base.get("tflops_npu"):
                    ideal = r["cores"] / base["cores"]
                    actual = r["tflops_npu"] / base["tflops_npu"]
                    print(f"  {r['cols']:>2} cols ({r['cores']:>2} cores): {actual:5.2f}x  "
                          f"(ideal {ideal:4.1f}x, {actual/ideal*100:5.1f}%)")
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            out = ARTIFACTS / f"bench_{'bfp16' if args.emulate_bfp16 else 'native'}_{M}x{K}x{N}.json"
            out.write_text(json.dumps(rows, indent=2))
            print(f"\nwrote {out.name}")
        return 0 if rows else 1

    cols_list = [1, 2, 4, 8] if args.scaling else [args.cols]
    results = []
    for cols in cols_list:
        if N % (args.n * cols):
            print(f"  SKIP cols={cols}: N({N}) % (n*cols={args.n*cols}) != 0")
            continue
        try:
            res = run_one(M, K, N, args.m, args.k, args.n, cols,
                          args.emulate_bfp16, args.trace_size)
        except Exception as exc:
            print(f"  cols={cols}: FAILED -- {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:140]}")
            continue
        if res:
            results.append(res)

    if len(results) > 1:
        base = results[0]
        print("\n--- scaling vs {} cols ---".format(base["cols"]))
        for r in results:
            ideal = r["cores"] / base["cores"]
            actual = r["macs_per_cycle_array"] / base["macs_per_cycle_array"]
            print(f"  {r['cols']:>2} cols ({r['cores']:>2} cores): "
                  f"{actual:5.2f}x  (ideal {ideal:4.1f}x, "
                  f"{actual/ideal*100:5.1f}% scaling efficiency)")

    if results:
        out = ARTIFACTS / f"scaling_{'bfp16' if args.emulate_bfp16 else 'native'}_{M}x{K}x{N}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out.name}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
