# NpuEmbeddings -- T28 Del C (tasks/0057): can a JOIN gather from GEMM cores
# in TWO DIFFERENT COLUMNS into ONE mem tile? join_then_consume_probe.py just
# proved the "read a join's .cons() from a third on-chip Worker" mechanism is
# legal WITHIN one column (2 rows -> 1 mem tile -> a 3rd row consumes it).
# That is necessary but not sufficient for the real ffn_down problem: ffn_up
# splits its N=4h output ACROSS COLUMNS (each column owns a different N-slice
# for ALL M rows), not across rows within one column -- so ffn_down's
# K-reduction needs a gather from OTHER COLUMNS, not other rows of the same
# column. This probe tests that directly: two GEMM cores in DIFFERENT
# columns (Tile(0,2) and Tile(1,2)), each fed from ITS OWN column's shim (A
# and B, exactly like production), joined into ONE mem tile explicitly
# pinned at column 0.
#
# If this compiles and routes, the cross-column regather is at least
# EXPRESSIBLE (subject to the mem-tile port budget 0046/0047 already
# measured as saturated at production scale -- see TASK.md). If it fails to
# compile/route, or compiles but hangs, that is direct evidence the
# restriction is topological (mem tiles only take inputs from their OWN
# column's cores), not just the Link-arity rule already found.
#
# RESULT (tasks/0057): cross-column routing is NOT the wall -- it is NOT
# limited to physically adjacent columns either. Set NPUE_SRC_COLS /
# NPUE_DEST_COL to reproduce each finding:
#
#   NPUE_SRC_COLS=0,1   NPUE_DEST_COL=0  -> PASS, rel_fro 3.474e-08
#     (adjacent-column join: column 1's GEMM core -> column 0's mem tile)
#   NPUE_SRC_COLS=0,7   NPUE_DEST_COL=0  -> PASS, rel_fro 3.474e-08 (identical)
#     (MAXIMUM-DISTANCE join, full array span: column 7 -> column 0. Distance
#     changed NOTHING -- the stream-switch fabric routes it at zero
#     numerical cost either way.)
#
# The wall is the mem-tile port budget (CLAUDE.md trap 3b / 0046 / 0047: 6
# in / 6 out per mem tile), and it is now compiler-QUANTIFIED, not just
# counted from placed MLIR after the fact:
#
#   NPUE_SRC_COLS=0,1,2,3,4,5,6,7 (8, incl. DEST_COL -- local A/B present)
#     FAILS TO COMPILE: "tile (0, 1) requires 8 input/1 output DMA channels,
#     but only 4 input/4 output available" -- exactly 6 - {A,B} = 4.
#   NPUE_SRC_COLS=1,2,3,4,5,6,7 (7, DEST_COL excluded -- gather-only tile)
#     FAILS TO COMPILE: "...requires 7 input/1 output... but only 6 input/6
#     output available" -- the bare 6-port ceiling, no local A/B to subtract.
#   NPUE_SRC_COLS=1,2,3,4,5,6 (6, gather-only) FAILS TOO, but at a DIFFERENT
#     point: the join itself fits (uses all 6 IN), but the gather-consumer's
#     OWN output (Y_out, relayed back OUT through the SAME mem tile to reach
#     the shim) needs a 7th IN port at that tile ("requires 1 input/1
#     output... only 0 input/5 output available") -- routing a result back
#     OUT through a mem tile that is already a full 6-way join destination
#     costs a port too. A gather-only tile that also must relay its result
#     onward through itself tops out at 5 sources, not 6.
#   NPUE_SRC_COLS=0,1,2,3 (4, incl. DEST_COL, local A/B + outbound Y forward
#     all through the SAME tile) FAILS on the SAME "no ports left for the Y
#     forward" pattern: A(1)+B(1)+join(4)=6, 0 left for Y_pipe.
#
# A SECOND, SEPARATE wall showed up at N=3 and N=5 gather-only: L1 capacity
# on the GATHER-CONSUMER core itself (row 4), because this probe's gather
# kernel acquires the WHOLE joined tile in ONE flat shot (identity_copy_*)
# rather than streaming it k-block-by-k-block the way gemm_pretiled.py's
# real K-loop does. At (64,48) tiles this one-shot design tops out at N=2
# (2*TM*TN*4B * 2 buffers = 49152 B, just under the 62464 B budget after the
# 2048 B stack) -- NOT a fundamental blocker, just evidence that any real
# fused ffn_down consumer must acquire/release per k-block like every other
# GEMM core in this codebase, not gather-then-copy in one piece. Not fixed
# here -- see TASK.md's Next section.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   $env:NPUE_SRC_COLS="0,1"; $env:NPUE_DEST_COL="0"
#   python cross_column_join_probe.py

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
# NPUE_SRC_COLS / NPUE_DEST_COL: which physical columns the GEMM sources sit
# in, and which column's mem tile the join's destination is pinned to. Read
# from env so the SAME script can probe adjacent (0,1 -> 0) vs
# maximum-distance (0,7 -> 0) cross-column routing without duplicating the
# whole design.
import os                                                # noqa: E402
SRC_COLS = [int(x) for x in os.environ.get("NPUE_SRC_COLS", "0,1").split(",")]
DEST_COL = int(os.environ.get("NPUE_DEST_COL", "0"))
N_GEMM_COLS = len(SRC_COLS)
M, K, N = TM, TK, TN     # each GEMM core computes its OWN [TM,K]x[K,TN] tile
GATHER_TILE = TM * TN * N_GEMM_COLS


def markers_for():
    return [f"aie.runtime_sequence(%arg0: memref<{TM * K * N_GEMM_COLS}xbf16>, "
            f"%arg1: memref<{K * N * N_GEMM_COLS}xbf16>, "
            f"%arg2: memref<{GATHER_TILE}xf32>)",
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

    # Each column gets its OWN A/B, exactly like the production whole-array
    # GEMM (A per column-worth-of-rows, B per column). Two independent
    # [TM,K]x[K,TN] problems, computed by cores in DIFFERENT columns.
    A_ty = np.ndarray[(TM * K * N_GEMM_COLS,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N * N_GEMM_COLS,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(GATHER_TILE,), np.dtype[np.float32]]

    A_l1_ty = np.ndarray[(TM, TK), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(TK, TN), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(TM, TN), np.dtype[np.float32]]
    gather_ty = np.ndarray[(GATHER_TILE,), np.dtype[np.float32]]

    A_shims, A_l2l1 = [], []
    B_shims, B_l2l1 = [], []
    for col in range(N_GEMM_COLS):
        pcol = SRC_COLS[col]      # physical column this GEMM source lives in
        A_l2_ty = np.ndarray[(TM * TK,), np.dtype[bfloat16]]
        a_shim = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=2)
        a_dims = [(TM // r, r * TK), (TK // s, s), (r, TK), (s, 1)]
        a_row = a_shim.cons().split(
            [0], obj_types=[A_l1_ty], names=[f"A_row_{col}"],
            dims_to_stream=[a_dims], tile=Tile(pcol, 1))[0]
        A_shims.append(a_shim)
        A_l2l1.append(a_row)

        B_l2_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
        b_shim = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        b_fwd = b_shim.cons().forward(obj_type=B_l1_ty, name=f"B_fwd_{col}",
                                      depth=2, tile=Tile(pcol, 1))
        B_shims.append(b_shim)
        B_l2l1.append(b_fwd)

    # THE PROBE: join from TWO DIFFERENT COLUMNS' compute tiles into ONE mem
    # tile, explicitly pinned at column 0 (tile=Tile(0, 1)). Column 1's GEMM
    # core (Tile(1, 2)) is NOT adjacent to column 0's mem tile in the
    # "N/S/E/W direct neighbour" sense note 0007 describes for cross-tile
    # Buffer reads -- this exercises the ObjectFifo/DMA stream-switch routing
    # path instead, which (unlike direct buffer reads) is not restricted to
    # immediate physical neighbours in the rest of this codebase (B's shim
    # already broadcasts across a whole row of columns in gemm_pretiled.py).
    C_mem = ObjectFifo(
        np.ndarray[(GATHER_TILE,), np.dtype[np.float32]],
        name="C_mem", depth=1,
        dims_to_stream=[(TM // r, r * TN), (r, t), (TN // t, r * t), (t, 1)])
    c_offsets = [TM * TN * j for j in range(N_GEMM_COLS)]
    # tile=Tile(0, 1): explicitly pin the join's DESTINATION (the mem tile
    # itself) at column 0. Column 1's GEMM core is a SOURCE of this join --
    # its own tile is fixed by its Worker's tile= below (Tile(1, 2)), not by
    # this call.
    C_cols = C_mem.prod().join(
        c_offsets, obj_types=[C_l1_ty] * N_GEMM_COLS,
        names=[f"C_col{j}" for j in range(N_GEMM_COLS)],
        tile=Tile(DEST_COL, 1))

    Y_out = ObjectFifo(gather_ty, name="Y_out", depth=1)
    Y_pipe = Y_out.cons().forward(obj_type=gather_ty, name="Y_pipe", depth=2,
                                  tile=Tile(DEST_COL, 1))

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
               [A_l2l1[col].cons(), B_l2l1[col].cons(), C_cols[col].prod(),
                zero_kernel, matmul_kernel],
               tile=Tile(SRC_COLS[col], 2), stack_size=0xD00)
        for col in range(N_GEMM_COLS)
    ]
    gather_worker = Worker(
        gather_core_fn,
        [C_mem.cons(), Y_out.prod(), copy_kernel],
        tile=Tile(DEST_COL, 4), stack_size=0x800,
    )

    # BUG FOUND AND FIXED (tasks/0057): the first version of this probe built
    # each column's A/B fill tap as `simple_tiler((TM, K))[0]` independently
    # -- IDENTICAL for every column, always covering OFFSET 0 of the host
    # buffer. Both GEMM cores therefore read the SAME A and SAME B (column
    # 0's), and column 1's "gathered" output was byte-identical to column
    # 0's -- not a routing bug at all, a host-side tiling bug that looked
    # exactly like one (rel_fro col1=1.375, col1 values == col0 values to 8
    # digits). Fix: tile the FULL (N_GEMM_COLS*TM, K) tensor and let
    # simple_tiler emit one correctly-offset TAP per tile.
    a_full_taps = TensorTiler2D.simple_tiler((N_GEMM_COLS * TM, K), (TM, K))
    b_full_taps = TensorTiler2D.simple_tiler((N_GEMM_COLS * K, N), (K, N))
    y_full_tap = TensorTiler2D.simple_tiler((TM, N * N_GEMM_COLS))[0]

    # `tile=` moves from the old rt.fill(..., tile=...) call to the handle
    # itself: prod(tile=)/cons(tile=) pin the runtime-driven shim endpoint now
    # (see Runtime.__init__'s docstring -- shim endpoints bind at fn_args
    # registration time, before the sequence body runs).
    A_prods = [A_shims[col].prod(tile=Tile(SRC_COLS[col], 0))
              for col in range(N_GEMM_COLS)]
    B_prods = [B_shims[col].prod(tile=Tile(SRC_COLS[col], 0))
              for col in range(N_GEMM_COLS)]
    Y_pipe_cons = Y_pipe.cons(tile=Tile(DEST_COL, 0))

    def sequence(A, B, Y, A_prod_hs, B_prod_hs, Y_pipe_cons_h):
        tg = TaskGroup()
        for col in range(N_GEMM_COLS):
            A_prod_hs[col].fill(A, tap=a_full_taps[col], group=tg)
            B_prod_hs[col].fill(B, tap=b_full_taps[col], group=tg)
        Y_pipe_cons_h.drain(Y, tap=y_full_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, Y_ty, A_prods, B_prods, Y_pipe_cons])

    return Program(dev, rt, workers=[*gemm_workers, gather_worker]).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def cross_column_join(A: In, B: In, Y: Out):
    return _build(iron.get_current_device())


def main() -> int:
    iron.set_current_device(from_name("npu2", n_cols=None))
    print(f"  SRC_COLS={SRC_COLS} DEST_COL={DEST_COL} "
          f"(set via NPUE_SRC_COLS / NPUE_DEST_COL env vars)")
    purge()

    rng = np.random.default_rng(11)
    a = (rng.standard_normal((N_GEMM_COLS, TM, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((N_GEMM_COLS, K, N)) * 0.05).astype(np.float32)
    a16 = a.astype(bfloat16).astype(np.float32)
    b16 = b.astype(bfloat16).astype(np.float32)
    want = np.stack([a16[j] @ b16[j] for j in range(N_GEMM_COLS)])  # [2,TM,TN]

    A = iron.zeros((N_GEMM_COLS * TM, K), dtype=bfloat16, device="npu")
    B = iron.zeros((N_GEMM_COLS * K, N), dtype=bfloat16, device="npu")
    Y = iron.zeros(GATHER_TILE, dtype=np.float32, device="npu")
    a_flat = a.reshape(N_GEMM_COLS * TM, K).astype(bfloat16)
    A[:] = a_flat
    b_tiled = np.concatenate(
        [tile_b(to_bf16_bits(b[j]), TK, TN, 8, 8).view(bfloat16).reshape(K, N)
         for j in range(N_GEMM_COLS)], axis=0)
    B[:] = b_tiled
    assert np.array_equal(A.numpy(), a_flat), "A did not reach the device"
    assert np.array_equal(B.numpy(), b_tiled), "B did not reach the device"

    cross_column_join(A, B, Y)

    got = Y.numpy().reshape(N_GEMM_COLS, TM, TN).astype(np.float64)
    finite = np.isfinite(got).all()
    if not finite:
        print(f"FAIL -- {int((~np.isfinite(got)).sum())} non-finite outputs")
        return 1
    rel_fro = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    ok = rel_fro < 5e-3
    print("JOIN from TWO DIFFERENT COLUMNS' GEMM cores into ONE mem tile "
          "(Tile(0,1)), then consumed by a third on-chip Worker:")
    print(f"  rel_fro {rel_fro:.3e}")
    for j in range(N_GEMM_COLS):
        rf = float(np.linalg.norm(got[j] - want[j]) / np.linalg.norm(want[j]))
        print(f"  col {j}: rel_fro={rf:.3e} got[0,0:4]={got[j,0,:4]} "
              f"want[0,0:4]={want[j,0,:4]} got_norm={np.linalg.norm(got[j]):.3e} "
              f"want_norm={np.linalg.norm(want[j]):.3e}")
    # Also check the swapped hypothesis (columns landed transposed).
    if N_GEMM_COLS == 2:
        swapped = got[::-1]
        rf_swapped = float(np.linalg.norm(swapped - want) / np.linalg.norm(want))
        print(f"  swapped-column-order rel_fro={rf_swapped:.3e}")
    print("PASS" if ok else "FAIL", "-- tolerance 5e-3 rel_fro")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
