# NpuEmbeddings -- T28 Del C (tasks/0057): does a mem-tile JOIN's OWN .cons()
# work as an ordinary input handle for a THIRD, on-chip Worker -- i.e. can a
# gathered (joined) tile feed further COMPUTE without ever leaving the array,
# as opposed to being drained to host (the only thing production code and
# 0054's B2 pipeline ever did with a join's consumer side)?
#
# WHY THIS MATTERS: 0054 (tasks/0054) proved a GEMM core's tile can reach a
# DIFFERENT core's GELU computation through a mem tile via POINT-TO-POINT
# .forward() links (2 GEMM rows -> 2 GELU rows, paired 1:1), and separately
# found that a SINGLE ObjectFifo cannot be BOTH a join-destination and a
# split/forward-SOURCE (confirmed in dataflow/objectfifo.py's
# ObjectFifoLink.__init__ AND in the MLIR verifier). What was never tested:
# whether that restriction is about the LINK MECHANISM specifically (i.e.
# calling .split()/.forward() a second time on an already-linked object), or
# whether ANY second consumption of a joined buffer is forbidden -- including
# the ordinary way every OTHER fifo in this codebase is consumed, by simply
# handing `.cons()` to a Worker (no second Link call at all; see
# `B_fwd.cons()` in pipeline_gemm_gelu_probe.py, `A_l2l1_fifos[row].cons()`
# in gemm_pretiled.py -- neither of those calls a SECOND split/join/forward,
# they just read a fifo that ALREADY has exactly one link).
#
# If C_mem.cons() -- the mem-tile buffer that TWO GEMM cores .join() into --
# can be handed straight to a THIRD Worker as an ordinary consumer (same
# pattern as everything else in this codebase), that is a genuinely different
# mechanism from split()/forward()-chaining and might sidestep finding (1)
# entirely for the WITHIN-COLUMN case (gathering multiple ROWS' tiles for
# further on-chip compute). It does NOT by itself solve the CROSS-COLUMN
# regather ffn_down needs (see the module-end note and TASK.md) -- that is a
# separate, adjacency/routing question this probe does not test.
#
# ARCHITECTURE: 2 GEMM cores (rows 2,3) join into C_mem (mem tile, one link).
# NO rt.drain on C_mem. Instead C_mem.cons() is handed directly to a THIRD
# Worker (row 4) running a plain 6144-element (2-tile) copy kernel, whose
# output feeds a FRESH point-to-point pipe (Y_out -> Y_in, the same
# .forward() pattern 0054 already proved) into a final drain to host. rt.
# sequence() is (A, B, Y) -- three buffers, no C -- same proof-by-construction
# 0054 used: if this compiles, runs, and is numerically correct, the gathered
# (joined) tile never left the array before being consumed by real compute.
#
# RESULT (tasks/0057): PASS, rel_fro 3.550e-08. C_mem.cons() -- the buffer
# that IS the join's destination -- worked as an ordinary Worker input with
# NO second split()/forward()/join() call anywhere in this file. The first
# compile attempt (depth=2 on both C_mem and Y_out) hit a DIFFERENT, mundane
# wall first -- 'aie.tile' op allocated buffers exceeded available memory,
# 98 KB requested of 63 (trap 3) -- fixed by dropping both to depth=1 (this
# is a single-shot probe, not a pipelined design, so single buffering costs
# nothing here). Once THAT was fixed it compiled, ran, and was numerically
# correct on the first attempt: the restriction 0054 found (an ObjectFifo
# cannot be BOTH a join-destination and a split/forward-SOURCE) is about the
# LINK mechanism specifically, not about "no second read of a joined buffer,
# period". See cross_column_join_probe.py for the follow-on question this
# does NOT answer by itself (can the join's SOURCES live in different
# columns than the mem tile they join into?).
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage: python join_then_consume_probe.py

from __future__ import annotations

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
from aie.helpers.taplib import TensorTiler2D             # noqa: E402
from npue import tile_b, to_bf16_bits                    # noqa: E402

CACHE = Path.home() / ".npu" / "cache"
TM, TK, TN = 64, 64, 48
N_GEMM_ROWS = 2
M, K, N = TM * N_GEMM_ROWS, TK, TN
GATHER_TILE = TM * TN * N_GEMM_ROWS   # 6144 -- the full joined buffer


def markers_for():
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}xf32>)",
            "identity_copy_6144_f32"]


def purge():
    n = 0
    markers = markers_for()
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers):
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

    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config as _cfg
    _inc = _include_dirs()
    _inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels"))
    _inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    copy_kernel = ExternalFunction(
        "identity_copy_6144_f32",
        source_file=str(HERE.parent / "m5-eltwise" / "kernels" / "gelu_poly.cc"),
        arg_types=[np.ndarray[(GATHER_TILE,), np.dtype[np.float32]]] * 2,
        include_dirs=_inc,
    )

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(GATHER_TILE,), np.dtype[np.float32]]

    A_l1_ty = np.ndarray[(TM, TK), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(TK, TN), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(TM, TN), np.dtype[np.float32]]
    gather_ty = np.ndarray[(GATHER_TILE,), np.dtype[np.float32]]

    # --- A: shim -> mem tile -> split (one tile per GEMM row), same as 0054.
    A_l2_ty = np.ndarray[(TM * TK * N_GEMM_ROWS,), np.dtype[bfloat16]]
    A_shim = ObjectFifo(A_l2_ty, name="A_L3L2", depth=2)
    a_offsets = [TM * TK * j for j in range(N_GEMM_ROWS)]
    a_dims = [[(TM // r, r * TK), (TK // s, s), (r, TK), (s, 1)]] * N_GEMM_ROWS
    A_rows = A_shim.cons().split(
        a_offsets, obj_types=[A_l1_ty] * N_GEMM_ROWS,
        names=[f"A_row{j}" for j in range(N_GEMM_ROWS)],
        dims_to_stream=a_dims)

    # --- B: shim -> mem tile -> forward, broadcast to both GEMM rows.
    B_l2_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    B_shim = ObjectFifo(B_l2_ty, name="B_L3L2", depth=2)
    B_fwd = B_shim.cons().forward(obj_type=B_l1_ty, name="B_fwd", depth=2)

    # --- C: THE PROBE. Two GEMM cores JOIN into C_mem (ONE link -- this is
    # legal, exactly what gemm_pretiled.py's own production C join does).
    # dims_to_stream copied verbatim from gemm_pretiled.py's C_l2l3_fifos /
    # 0054's Y_mem -- a mem-tile join of multiple producer tiles needs this
    # unscrambling formula or the result is wrong-but-finite (0054 problem 5).
    # depth=1 on the gather-consumer's own L1 buffers (C_mem, Y_out below):
    # GATHER_TILE=6144 f32 elements = 24576 B/buffer, and this worker's tile
    # (row 4) must hold BOTH C_mem's consumer buffer and Y_out's producer
    # buffer -- depth 2 on both overflows the 63 KB L1 budget (trap 3) at
    # ~98 KB. This is a single-shot probe (one acquire/release each), so
    # single buffering costs nothing here; it would need real double
    # buffering for anything pipelined.
    C_mem = ObjectFifo(
        np.ndarray[(GATHER_TILE,), np.dtype[np.float32]],
        name="C_mem", depth=1,
        dims_to_stream=[(TM // r, r * TN), (r, t), (TN // t, r * t), (t, 1)])
    c_offsets = [TM * TN * j for j in range(N_GEMM_ROWS)]
    C_rows = C_mem.prod().join(
        c_offsets, obj_types=[C_l1_ty] * N_GEMM_ROWS,
        names=[f"C_row{j}" for j in range(N_GEMM_ROWS)])

    # --- Y: a FRESH point-to-point pipe, core -> mem tile -> shim, the same
    # .forward() primitive 0054's C_outs[j]->C_pipes[j] used. This is the
    # gather-consumer core's OWN output; it is NOT a second link on C_mem.
    Y_out = ObjectFifo(gather_ty, name="Y_out", depth=1)
    Y_pipe = Y_out.cons().forward(obj_type=gather_ty, name="Y_pipe", depth=2)

    def gemm_core_fn(in_a, in_b, out_c, zero, matmul):
        ea = out_c.acquire(1)
        zero(ea)
        a = in_a.acquire(1)
        b = in_b.acquire(1)
        matmul(a, b, ea)
        in_a.release(1)
        in_b.release(1)
        out_c.release(1)

    def gather_core_fn(in_gathered, out_y, copy):
        g = in_gathered.acquire(1)
        y = out_y.acquire(1)
        copy(g, y)
        in_gathered.release(1)
        out_y.release(1)

    gemm_workers = [
        Worker(gemm_core_fn,
               [A_rows[j].cons(), B_fwd.cons(), C_rows[j].prod(),
                zero_kernel, matmul_kernel],
               tile=Tile(0, 2 + j), stack_size=0xD00)
        for j in range(N_GEMM_ROWS)
    ]
    # THE PROBE ITSELF: C_mem.cons() -- the mem-tile buffer that is ALREADY
    # the destination of the join above -- handed straight to a THIRD Worker
    # as an ordinary consumer. No second split()/forward()/join() call on
    # C_mem anywhere in this file.
    gather_worker = Worker(
        gather_core_fn,
        [C_mem.cons(), Y_out.prod(), copy_kernel],
        tile=Tile(0, 4), stack_size=0x800,
    )

    a_full_tap = TensorTiler2D.simple_tiler((M, K))[0]
    b_full_tap = TensorTiler2D.simple_tiler((K, N))[0]
    y_full_tap = TensorTiler2D.simple_tiler((M, N))[0]

    def sequence(A, B, Y, A_shim_prod, B_shim_prod, Y_pipe_cons):
        tg = TaskGroup()
        A_shim_prod.fill(A, tap=a_full_tap, group=tg)
        B_shim_prod.fill(B, tap=b_full_tap, group=tg)
        Y_pipe_cons.drain(Y, tap=y_full_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, Y_ty,
                            A_shim.prod(), B_shim.prod(), Y_pipe.cons()])

    return Program(dev, rt, workers=[*gemm_workers, gather_worker]).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def join_then_consume(A: In, B: In, Y: Out):
    return _build(iron.get_current_device())


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    purge()

    rng = np.random.default_rng(11)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    a16 = a.astype(bfloat16).astype(np.float32)
    b16 = b.astype(bfloat16).astype(np.float32)
    want = a16 @ b16   # [M, N] = [128, 48]; the copy kernel is identity, so
                        # the answer is still plain GEMM, just routed through
                        # a THIRD core's L1 on the way out.

    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b), TK, TN, 8, 8)
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    Y = iron.zeros(GATHER_TILE, dtype=np.float32, device="npu")
    A[:] = a.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K, N)
    assert np.array_equal(A.numpy(), a.astype(bfloat16)), "A did not reach the device"
    assert np.array_equal(B.numpy(), Bt.view(bfloat16).reshape(K, N)), "B did not reach the device"

    join_then_consume(A, B, Y)

    got = Y.numpy().reshape(N_GEMM_ROWS, TM, TN).astype(np.float64)
    # got[j] should equal the GEMM result for GEMM-row j's slice of A (rows
    # j*TM:(j+1)*TM), since the join lays row j at offset j*TM*TN and the
    # gather-consumer copies it through unchanged.
    want_split = want.reshape(N_GEMM_ROWS, TM, TN)
    finite = np.isfinite(got).all()
    if not finite:
        print(f"FAIL -- {int((~np.isfinite(got)).sum())} non-finite outputs "
              f"(join.cons() handed to a third Worker did NOT hang, but the "
              f"gathered data is garbage)")
        return 1
    rel_fro = float(np.linalg.norm(got - want_split) / np.linalg.norm(want_split))
    ok = rel_fro < 5e-3
    print("join.cons() handed DIRECTLY to a third on-chip Worker "
          "(no second split/forward/join on C_mem):")
    print(f"  rel_fro {rel_fro:.3e}")
    print("PASS" if ok else "FAIL", "-- tolerance 5e-3 rel_fro")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
