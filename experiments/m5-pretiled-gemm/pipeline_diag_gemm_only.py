# NpuEmbeddings -- T28 Del B / B2 diagnostic: GEMM-only, 2 rows, straight to
# host. Isolates whether the reduced 2-row geometry (A dims_to_stream, B
# broadcast, join, drain) is correct BEFORE blaming the core-to-core GELU
# hop for pipeline_gemm_gelu_probe.py's runtime timeout. If this hangs too,
# the bug is in the base geometry, not the pipeline mechanism.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage: python pipeline_diag_gemm_only.py

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aie.iron as iron                                # noqa: E402
from aie.iron import (                                  # noqa: E402
    CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels,
    str_to_dtype,
)
from aie.iron.device import Tile, from_name             # noqa: E402
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D  # noqa: E402
from npue import tile_b, to_bf16_bits                    # noqa: E402

CACHE = Path.home() / ".npu" / "cache"
TM, TK, TN = 64, 64, 48
N_GEMM_ROWS = 2
M, K, N = TM * N_GEMM_ROWS, TK, TN


def purge():
    marker = f"aie.runtime_sequence(%arg0: memref<{M*K}xbf16>, %arg1: memref<{K*N}xbf16>, %arg2: memref<{M*N}xf32>)"
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if d.is_dir() and mlir.exists():
            text = mlir.read_text(encoding="utf-8", errors="ignore")
            if marker in text and "gelu_epilogue_3072_f32_io" not in text:
                shutil.rmtree(d)
                n += 1
    print(f"  purged {n} cache candidate(s)")


def _build(dev):
    matmul_kernel = kernels.mm(
        dim_m=TM, dim_k=TK, dim_n=TN,
        input_dtype=str_to_dtype("bf16"), output_dtype=str_to_dtype("f32"),
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(TM, TK), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(TK, TN), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(TM, TN), np.dtype[np.float32]]

    A_l2_ty = np.ndarray[(TM * TK * N_GEMM_ROWS,), np.dtype[bfloat16]]
    A_shim = ObjectFifo(A_l2_ty, name="A_L3L2", depth=2)
    a_offsets = [TM * TK * j for j in range(N_GEMM_ROWS)]
    a_dims = [[(TM // r, r * TK), (TK // s, s), (r, TK), (s, 1)]] * N_GEMM_ROWS
    A_rows = A_shim.cons().split(a_offsets, obj_types=[A_l1_ty] * N_GEMM_ROWS,
                                 names=[f"A_row{j}" for j in range(N_GEMM_ROWS)],
                                 dims_to_stream=a_dims)

    B_l2_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    B_shim = ObjectFifo(B_l2_ty, name="B_L3L2", depth=2)
    B_fwd = B_shim.cons().forward(obj_type=B_l1_ty, name="B_fwd", depth=2)

    # gemm_pretiled.py sets dims_to_stream ON THE BASE OBJECT (not on the
    # join's subfifos) for exactly this C join -- omitted on the first two
    # attempts here. Copied verbatim from _build_design's C_l2l3_fifos.
    C_mem = ObjectFifo(
        np.ndarray[(TM * TN * N_GEMM_ROWS,), np.dtype[np.float32]],
        name="C_mem", depth=2,
        dims_to_stream=[(TM // r, r * TN), (r, t), (TN // t, r * t), (t, 1)])
    c_offsets = [TM * TN * j for j in range(N_GEMM_ROWS)]
    C_rows = C_mem.prod().join(c_offsets, obj_types=[C_l1_ty] * N_GEMM_ROWS,
                               names=[f"C_row{j}" for j in range(N_GEMM_ROWS)])

    def gemm_core_fn(in_a, in_b, out_c, zero, matmul):
        ea = out_c.acquire(1)
        zero(ea)
        a = in_a.acquire(1)
        b = in_b.acquire(1)
        matmul(a, b, ea)
        in_a.release(1)
        in_b.release(1)
        out_c.release(1)

    workers = [Worker(gemm_core_fn,
                      [A_rows[j].cons(), B_fwd.cons(), C_rows[j].prod(),
                       zero_kernel, matmul_kernel],
                      tile=Tile(0, 2 + j), stack_size=0xD00)
              for j in range(N_GEMM_ROWS)]

    # Swapped hand-built TensorAccessPattern for the PROVEN helper
    # (TensorTiler2D.simple_tiler with tile_dims == tensor_dims -- "one big
    # tile covering everything", the same idiom gelu_kernel.py uses) as a
    # diagnostic step: does the tap CONSTRUCTION matter, or is the hang
    # elsewhere? tile_dims=None defaults to tensor_dims per its docstring.
    a_tap = TensorTiler2D.simple_tiler((M, K))[0]
    b_tap = TensorTiler2D.simple_tiler((K, N))[0]
    c_tap = TensorTiler2D.simple_tiler((M, N))[0]

    def sequence(A, B, C, A_shim_prod, B_shim_prod, C_mem_cons):
        tg = TaskGroup()
        A_shim_prod.fill(A, tap=a_tap, group=tg)
        B_shim_prod.fill(B, tap=b_tap, group=tg)
        C_mem_cons.drain(C, tap=c_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, A_shim.prod(), B_shim.prod(), C_mem.cons()])
    return Program(dev, rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemm_only(A: In, B: In, C: Out):
    return _build(iron.get_current_device())


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    purge()
    rng = np.random.default_rng(11)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    ref = (a.astype(bfloat16).astype(np.float32)
           @ b.astype(bfloat16).astype(np.float32)).astype(np.float64)

    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b), TK, TN, 8, 8)
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = a.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K, N)

    gemm_only(A, B, C)
    got = C.numpy().reshape(M, N).astype(np.float64)
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    print(f"GEMM-only, 2 rows, straight to host: rel_fro={rel_fro:.3e} "
          f"{'PASS' if rel_fro < 5e-3 else 'FAIL'}")
    return 0 if rel_fro < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
