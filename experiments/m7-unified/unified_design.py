# NpuEmbeddings -- M7: the SPATIAL-SPLIT unified design. ABANDONED at the
# 16 KB program-memory wall (tasks/0032): GELU + LayerNorm-il4 + softmax-il4
# in one core ELF overflow PM (the il4 softmax ELF alone is 12.2 KB). Kept
# because the techniques in it are load-bearing for any future multi-stream
# heterogeneous design: idle-half fifo endpoint pinning (RuntimeEndpoint +
# rt._fifos registration), pinned shim tiles for static-image identity, and
# the include-order trap (our kernels dir must precede aie_kernels, which
# ships its own softmax.cc). Production went a different way: pure-GEMM NPU
# (tools/export_gemm_rtp.py) + host eltwise. One static configuration, one
# hw_context, every encoder op an instruction stream.
# SPDX-License-Identifier: Apache-2.0
#
# WHY
# ---
# tasks/0031 measured the design-switch bill at ~115 ms of a 420 ms encode --
# 2.2-2.6 ms per switch, 49 switches, 28% of everything. tasks/0029 proved
# switches follow the hw_context, not the instruction stream; tasks/0030
# proved RTP loop bounds let one xclbin serve every GEMM shape. What kept the
# ops in seven designs was that a core runs exactly one program: GEMM cores
# cannot do GELU. This design resolves that by SPLITTING THE ARRAY:
#
#   columns 0..3   the whole-array GEMM, verbatim from gemm_pretiled
#                  (rtp=True), all four shapes as streams
#   columns 4..7   an OPCODE-SWITCHED eltwise worker: one core program
#                  containing GELU + LayerNorm + softmax, selected per
#                  dispatch by a runtime parameter
#
# The two halves never run simultaneously (dispatches serialize on the array,
# note 0004), so each op simply uses its own half. The pricing (notes/0005
# section 4) said a 4-column split was a wash at ~60 ms of switches; the
# measured bill is ~115 ms and il4 shrank the eltwise penalty, so the split
# now nets clearly positive. The pipelined form (GEMM streaming INTO eltwise)
# remains the endgame; this is the step that removes the switch bill and
# builds the multi-stream runtime it needs.
#
# GEOMETRY
# --------
# GEMM keeps m,k,n = 64,64,48 -- same .npue B layout, same layout hash, no
# weight repack. At batch 128 the GEMM half is movement-bound and 4 columns
# measured the same as 8 (8,104 vs 8,132 us per call, notes/0005).
#
# The eltwise tile is 6,144 bf16 elements -- ONE size that fits all three ops:
#   GELU        6,144 flat        (elementwise; the kernel loops in 64s)
#   LayerNorm   16 rows x 384     (exactly the block validated in isolation)
#   softmax     96 rows x 64      (96 % 4 = 0 for the il4 kernel)
# L1 per elt core: in 2x12 KB + out 2x12 KB + params 3 KB = 51 KB < 63 KB.
#
# BUFFER CONTRACT (the same three slots for every stream)
# -------------------------------------------------------
#   arg0  in       GEMM: A activations        eltwise: X input
#   arg1  aux      GEMM: B weights (pretiled) eltwise: params (LN) or dummy
#   arg2  out      GEMM: C fp32               eltwise: Y bf16
# Sizes differ per stream; 0030 established that buffer sizes live in the
# instruction stream and the host, never in the static image.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage (smoke-test all seven streams at batch 4):
#   python experiments\m7-unified\unified_design.py --batch 4

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
    WorkerRuntimeBarrier, kernels, str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron.runtime.endpoint import RuntimeEndpoint

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

from npue import tile_b, untile_b                      # noqa: E402

SEQ = 64
HEADS = 12
GEMM_COLS = 4          # columns 0..3
ELT_COLS = 4           # columns 4..7
N_ROWS = 4
M_TILE, K_TILE, N_TILE = 64, 64, 48

ELT_OBJ = 6144         # bf16 elements per eltwise fifo object; see header
LN_COLS = 384
LN_ROWS_PER_OBJ = ELT_OBJ // LN_COLS      # 16
SM_COLS = 64
SM_ROWS_PER_OBJ = ELT_OBJ // SM_COLS      # 96

ELT_OPS = {"gelu": 0, "layernorm": 1, "softmax": 2}
GEMM_STREAMS = ("qkv", "attn_out", "ffn_up", "ffn_down")


def gemm_shapes(batch, hidden=384):
    M, h = batch * SEQ, hidden
    return {
        "qkv":      dict(M=M, K=h,     N=3 * h),
        "attn_out": dict(M=M, K=h,     N=h),
        "ffn_up":   dict(M=M, K=h,     N=4 * h),
        "ffn_down": dict(M=M, K=4 * h, N=h),
    }


def elt_n_elem(stream, batch, hidden=384):
    """Total elements the eltwise stream processes, encode-shaped."""
    if stream == "gelu":
        return batch * SEQ * 4 * hidden
    if stream == "layernorm":
        return batch * SEQ * LN_COLS
    return batch * HEADS * SEQ * SM_COLS   # softmax


def elt_kernel():
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"
                       / _detect_arch()))
    # OUR kernels dir FIRST: the shipped aie_kernels tree has its own
    # softmax.cc, and quote-include falls back to -I order -- without this the
    # universal TU includes AMD's softmax instead of ours and dies with
    # "use of undeclared identifier softmax_il4_impl".
    include.insert(0, str(REPO / "experiments" / "m5-eltwise" / "kernels"))
    tile_ty = np.ndarray[(ELT_OBJ,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(2 * LN_COLS,), np.dtype[np.float32]]
    return ExternalFunction(
        "eltwise_universal_6144",
        source_file=str(REPO / "experiments" / "m5-eltwise" / "kernels"
                        / "eltwise_universal.cc"),
        arg_types=[tile_ty, vec_ty, tile_ty, np.int32],
        include_dirs=include,
    )


def _build(dev, stream, batch, hidden):
    M_by_shape = gemm_shapes(batch, hidden)
    m, k, n = M_TILE, K_TILE, N_TILE

    # ------------------------------------------------------------------ GEMM
    # Verbatim from gemm_pretiled._build_design at n_aie_cols=4, rtp=True,
    # workers pinned to columns 0..3. Loop bounds are the ONLY shape-dependent
    # values and they are runtime parameters, so the static half is
    # shape-independent (proven in 0030 section 1).
    dtype_in = str_to_dtype("bf16")
    dtype_out = str_to_dtype("f32")
    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out,
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True)
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    fifo_depth = 2
    A_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * N_ROWS,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = [None] * GEMM_COLS
    A_l2l1_fifos = [None] * N_ROWS
    B_l3l2_fifos = [None] * GEMM_COLS
    B_l2l1_fifos = [None] * GEMM_COLS
    C_l1l2_fifos = [[None] * GEMM_COLS for _ in range(N_ROWS)]
    C_l2l3_fifos = [None] * GEMM_COLS

    for i in range(GEMM_COLS):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}",
                                     depth=fifo_depth)
        dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]]
        tmp = A_l3l2_fifos[i].cons().split(
            [0], obj_types=[A_l1_ty], names=[f"A_L2L1_{i}"],
            dims_to_stream=dims)
        A_l2l1_fifos[i] = tmp[0]

    for col in range(GEMM_COLS):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}",
                                       depth=fifo_depth)
        B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}")
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)])
        tmp = C_l2l3_fifos[col].prod().join(
            [m * n * i for i in range(N_ROWS)],
            obj_types=[C_l1_ty] * N_ROWS,
            names=[f"C_L1L2_{col}_{row}" for row in range(N_ROWS)],
            depths=[fifo_depth] * N_ROWS)
        for j in range(N_ROWS):
            C_l1l2_fifos[j][col] = tmp[j]

    gemm_rtps = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]],
                         name=f"grtp_{rr}_{cc}",
                         initial_value=np.zeros(2, dtype=np.int32),
                         use_write_rtp=True)
                  for cc in range(GEMM_COLS)] for rr in range(N_ROWS)]
    gemm_barriers = [[WorkerRuntimeBarrier()
                      for _ in range(GEMM_COLS)] for _ in range(N_ROWS)]

    def gemm_core_fn(in_a, in_b, out_c, zero, matmul, my_rtp, barrier):
        barrier.wait_for_value(1)
        n_out_tiles = my_rtp[0]
        n_k_blocks = my_rtp[1]
        for _ in range_(n_out_tiles):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(n_k_blocks):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)
        barrier.release_with_value(1)

    gemm_workers = []
    for row in range(N_ROWS):
        for col in range(GEMM_COLS):
            gemm_workers.append(Worker(
                gemm_core_fn,
                [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
                 C_l1l2_fifos[row][col].prod(), zero_kernel, matmul_kernel,
                 gemm_rtps[row][col], gemm_barriers[row][col]],
                tile=Tile(col, row + 2),
                stack_size=0xD00))

    # --------------------------------------------------------------- ELTWISE
    # The split/join topology that took every eltwise design to width
    # (tasks/0027/0030/0031), on columns 4..7, with the params fifo broadcast
    # from the mem tile and ONE opcode-switched worker program.
    elt_tile_ty = np.ndarray[(ELT_OBJ,), np.dtype[bfloat16]]
    elt_l2_ty = np.ndarray[(N_ROWS * ELT_OBJ,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(2 * LN_COLS,), np.dtype[np.float32]]
    elt_k = elt_kernel()
    elt_offsets = [ELT_OBJ * rr for rr in range(N_ROWS)]

    elt_rtps = [Buffer(np.ndarray[(2,), np.dtype[np.int32]],
                       name=f"ertp_{i}",
                       initial_value=np.zeros(2, dtype=np.int32),
                       use_write_rtp=True)
                for i in range(ELT_COLS * N_ROWS)]
    elt_barriers = [WorkerRuntimeBarrier() for _ in range(ELT_COLS * N_ROWS)]

    def elt_core_fn(a, pm, c, kern, my_rtp, barrier):
        barrier.wait_for_value(1)
        op = my_rtp[0]
        n_objs = my_rtp[1]
        ep = pm.acquire(1)
        for _ in range_(n_objs):
            ea = a.acquire(1)
            ec = c.acquire(1)
            kern(ea, ep, ec, op)
            a.release(1)
            c.release(1)
        pm.release(1)
        barrier.release_with_value(1)

    elt_in, elt_out, elt_p, elt_workers = [], [], [], []
    for ci in range(ELT_COLS):
        col = GEMM_COLS + ci                     # physical column 4..7
        fin = ObjectFifo(elt_l2_ty, name=f"ein_{ci}", depth=2)
        fout = ObjectFifo(elt_l2_ty, name=f"eout_{ci}", depth=2)
        fp = ObjectFifo(vec_ty, name=f"ep_{ci}", depth=1)
        fp_l1 = fp.cons().forward(name=f"ep_l1_{ci}", depth=1)
        elt_in.append(fin); elt_out.append(fout); elt_p.append(fp)
        ins = fin.cons().split(
            elt_offsets, obj_types=[elt_tile_ty] * N_ROWS,
            names=[f"ein_{ci}_{rr}" for rr in range(N_ROWS)])
        # OUT single-buffered: the 0x4000 stack the il4 softmax needs plus
        # double-buffered 12 KB objects both ways is 68.6 KB against 64. The
        # drain of one 12 KB object is ~1.5k cycles against >=8k of compute
        # per object, so the stall is the cheapest thing to give up.
        outs = fout.prod().join(
            elt_offsets, obj_types=[elt_tile_ty] * N_ROWS,
            names=[f"eout_{ci}_{rr}" for rr in range(N_ROWS)],
            depths=[1] * N_ROWS)
        for rr in range(N_ROWS):
            wi = ci * N_ROWS + rr
            elt_workers.append(Worker(
                elt_core_fn,
                [ins[rr].cons(), fp_l1.cons(), outs[rr].prod(), elt_k,
                 elt_rtps[wi], elt_barriers[wi]],
                tile=Tile(col, rr + 2),
                # 0x4000: the il4 softmax frame; 0x2000 TIMES OUT (tasks/0031)
                stack_size=0x4000))

    all_workers = gemm_workers + elt_workers

    # ------------------------------------------------------- runtime sequence
    # Every stream must leave the OTHER half's L3-side fifos with placed
    # endpoints, or Program.resolve() fails with "prod was not created".
    # They get endpoints but no DMA tasks -- statically identical routing,
    # nothing moves. Shim tiles are pinned so the placement (and therefore
    # the static image) cannot differ between streams.
    #
    # NPUE-IRON1.4: the old API pinned these from INSIDE the sequence body via
    # a private rt._fifos.add() (the body ran eagerly, so rt._fifos was fully
    # populated by the time Program(dev, rt).resolve_program() read it). The
    # new API's sequence body runs LAST, inside Program.resolve_program() --
    # AFTER Program has already collected `self._rt.fifos` for tile
    # resolution -- so a handle registered only from inside the body would
    # never be seen. Passing the idle handles through Runtime's `fn_args`
    # instead registers them at Runtime.__init__ time (see
    # Runtime._register_fn_args / its docstring: "fifo shim endpoints bind
    # now ... so the fifo has both ends known when the Program resolves it"),
    # which is early enough. The sequence body receives them as parameters
    # and simply never touches them -- same "placed but idle" effect as the
    # old rt._fifos hack, reached through the sanctioned pathway instead of a
    # private attribute.
    def _pin_idle_gemm():
        handles = []
        for col in range(GEMM_COLS):
            shim = Tile(col, 0)
            handles += [A_l3l2_fifos[col].prod(tile=shim),
                       B_l3l2_fifos[col].prod(tile=shim),
                       C_l2l3_fifos[col].cons(tile=shim)]
        return handles

    def _pin_idle_elt():
        handles = []
        for ci in range(ELT_COLS):
            shim = Tile(GEMM_COLS + ci, 0)
            handles += [elt_in[ci].prod(tile=shim), elt_p[ci].prod(tile=shim),
                       elt_out[ci].cons(tile=shim)]
        return handles

    if stream in GEMM_STREAMS:
        shape = M_by_shape[stream]
        M, K, N = shape["M"], shape["K"], shape["N"]
        assert M % (m * N_ROWS) == 0 and K % k == 0
        assert N % (n * GEMM_COLS) == 0
        n_tiles_per_core = (M // m) * (N // n) // (N_ROWS * GEMM_COLS)

        A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
        B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
        C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]

        tb_max_n_rows = 4
        tb_n_rows = min(tb_max_n_rows // 2, M // m // N_ROWS)
        if m * N_ROWS * N > 2**20:
            tb_n_rows = 1

        A_tiles = TensorTiler2D.group_tiler(
            (M, K), (m, k), (1, K // k),
            pattern_repeat=N // n // GEMM_COLS, prune_step=False)

        TE = k * n
        KB, NB = K // k, N // n
        NBC = NB // GEMM_COLS
        kb_stride, nb_stride = NB * TE, TE
        B_taps = [
            TensorAccessPattern(
                (K * N,), col * nb_stride,
                [NBC, KB, k, n],
                [GEMM_COLS * nb_stride, kb_stride, n, 1],
            )
            for col in range(GEMM_COLS)
        ]
        C_tiles = TensorTiler2D.step_tiler(
            (M, N), (m * N_ROWS, n),
            tile_group_repeats=(tb_n_rows, N // n // GEMM_COLS),
            tile_group_steps=(1, GEMM_COLS), prune_step=False)
        c_index = 0

        A_prods = [A_l3l2_fifos[col].prod(tile=Tile(col, 0))
                  for col in range(GEMM_COLS)]
        B_prods = [B_l3l2_fifos[col].prod(tile=Tile(col, 0))
                  for col in range(GEMM_COLS)]
        C_conss = [C_l2l3_fifos[col].cons(tile=Tile(col, 0))
                  for col in range(GEMM_COLS)]
        idle_elt = _pin_idle_elt()

        def sequence(A, B, C, A_prod_hs, B_prod_hs, C_cons_hs, _idle):
            nonlocal c_index
            for rr in range(N_ROWS):
                for cc in range(GEMM_COLS):
                    gemm_rtps[rr][cc][0] = n_tiles_per_core
                    gemm_rtps[rr][cc][1] = K // k
            for rr in range(N_ROWS):
                for cc in range(GEMM_COLS):
                    gemm_barriers[rr][cc].set(1)
            tg = TaskGroup()
            for tb in range(iron.ceildiv(M // m // N_ROWS, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    if c_index >= len(C_tiles):
                        break
                    row_base = (tb * tb_max_n_rows
                                + pingpong * tb_max_n_rows // 2)
                    current_tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // N_ROWS - row_base])
                    for col in range(GEMM_COLS):
                        C_cons_hs[col].drain(C, tap=C_tiles[c_index],
                                             wait=True, group=tg)
                        c_index += 1
                        for tile_row in range(current_tb_n_rows):
                            off = ((row_base + tile_row) * GEMM_COLS + col) \
                                % len(A_tiles)
                            A_prod_hs[col].fill(A, tap=A_tiles[off], group=tg)
                            B_prod_hs[col].fill(B, tap=B_taps[col], group=tg)
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        tg.finish()
                        tg = TaskGroup()
            tg.finish()

        rt = Runtime(sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conss, idle_elt])
    else:
        op = ELT_OPS[stream]
        n_elem = elt_n_elem(stream, batch, hidden)
        n_objs_total = n_elem // ELT_OBJ
        n_cores = ELT_COLS * N_ROWS
        assert n_elem % (ELT_OBJ * n_cores) == 0, \
            f"{stream}: {n_elem} elements do not split into {ELT_OBJ}-blocks " \
            f"over {n_cores} cores"
        n_objs_per_core = n_objs_total // n_cores

        X_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]
        P_ty = np.ndarray[(2 * LN_COLS,), np.dtype[np.float32]]
        Y_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]

        data_taps = TensorTiler2D.simple_tiler(
            (1, n_elem), (1, n_elem // ELT_COLS))
        param_tap = TensorTiler2D.simple_tiler(
            (1, 2 * LN_COLS), (1, 2 * LN_COLS))[0]

        p_prods = [elt_p[ci].prod(tile=Tile(GEMM_COLS + ci, 0))
                  for ci in range(ELT_COLS)]
        in_prods = [elt_in[ci].prod(tile=Tile(GEMM_COLS + ci, 0))
                   for ci in range(ELT_COLS)]
        out_conss = [elt_out[ci].cons(tile=Tile(GEMM_COLS + ci, 0))
                    for ci in range(ELT_COLS)]
        idle_gemm = _pin_idle_gemm()

        def sequence(X, P, Y, p_prod_hs, in_prod_hs, out_cons_hs, _idle):
            for b in elt_rtps:
                b[0] = op
                b[1] = n_objs_per_core
            for i in range(n_cores):
                elt_barriers[i].set(1)
            tg = TaskGroup()
            for ci in range(ELT_COLS):
                p_prod_hs[ci].fill(P, tap=param_tap, group=tg)
                in_prod_hs[ci].fill(X, tap=data_taps[ci], group=tg)
                out_cons_hs[ci].drain(Y, tap=data_taps[ci], wait=True, group=tg)
            tg.finish()

        rt = Runtime(sequence, [X_ty, P_ty, Y_ty,
                                p_prods, in_prods, out_conss, idle_gemm])

    return Program(dev, rt, workers=all_workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def unified_array(A: In, B: In, C: Out, *, stream: CompileTime[str],
                  batch: CompileTime[int], hidden: CompileTime[int] = 384):
    return _build(iron.get_current_device(), stream, batch, hidden)


# ------------------------------------------------------------------ smoke test
def _check_gemm(stream, batch, hidden, rng):
    shape = gemm_shapes(batch, hidden)[stream]
    M, K, N = shape["M"], shape["K"], shape["N"]
    m, k, n = M_TILE, K_TILE, N_TILE

    A_np = (rng.standard_normal((M, K)) * 0.1).astype(bfloat16)
    B_np = (rng.standard_normal((K, N)) * 0.1).astype(bfloat16)

    A = iron.zeros(M * K, dtype=bfloat16, device="npu")
    B = iron.zeros(K * N, dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = A_np.reshape(-1)
    mk = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=str_to_dtype("bf16"),
                    output_dtype=str_to_dtype("f32"), b_col_maj=False,
                    c_col_maj=False, use_chess=False,
                    emulate_bf16_mmul_with_bfp16=False, vectorized=True)
    r, s, t = mk.mac_dims
    tiled = tile_b(B_np.view(np.uint16), k, n, s, t, order="k,n")
    B[:] = tiled.view(bfloat16).reshape(-1)
    assert np.array_equal(A.numpy(), A_np.reshape(-1)), "A did not reach device"

    unified_array(A, B, C, stream=stream, batch=batch, hidden=hidden)

    got = C.numpy().reshape(M, N).astype(np.float64)
    ref = A_np.astype(np.float64) @ B_np.astype(np.float64)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    ok = rel < 5e-3
    print(f"  {stream:<10} rel_fro {rel:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_elt(stream, batch, hidden, rng):
    n_elem = elt_n_elem(stream, batch, hidden)
    if stream == "layernorm":
        x = (rng.standard_normal(n_elem) * 2.0).astype(np.float32)
    elif stream == "softmax":
        x = (rng.standard_normal(n_elem) * 3.0).astype(np.float32)
    else:
        x = (rng.standard_normal(n_elem) * 2.0).astype(np.float32)
    x16 = x.astype(bfloat16)
    gamma = rng.standard_normal(LN_COLS).astype(np.float32)
    beta = rng.standard_normal(LN_COLS).astype(np.float32)

    X = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    P = iron.zeros(2 * LN_COLS, dtype=np.float32, device="npu")
    Y = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    X[:] = x16
    P[:] = np.concatenate([gamma, beta])
    assert np.array_equal(X.numpy(), x16), "X did not reach device"

    unified_array(X, P, Y, stream=stream, batch=batch, hidden=hidden)
    got = Y.numpy().astype(np.float64)

    xf = x16.astype(np.float32)
    if stream == "gelu":
        from math import erf
        v = np.vectorize(lambda u: 0.5 * u * (1.0 + erf(u / np.sqrt(2.0))))
        ref = v(xf.astype(np.float64))
        tol = 5e-3
    elif stream == "layernorm":
        xr = xf.reshape(-1, LN_COLS).astype(np.float64)
        mu = xr.mean(axis=1, keepdims=True)
        var = ((xr - mu) ** 2).mean(axis=1, keepdims=True)
        ref = ((xr - mu) / np.sqrt(var + 1e-12) * gamma + beta).reshape(-1)
        tol = 5e-3
    else:
        xr = xf.reshape(-1, SM_COLS).astype(np.float64)
        e = np.exp(xr - xr.max(axis=1, keepdims=True))
        ref = (e / e.sum(axis=1, keepdims=True)).reshape(-1)
        tol = 2e-2
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    ok = rel < tol
    print(f"  {stream:<10} rel_fro {rel:.3e}  (tol {tol:.0e})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--streams", default=None,
                    help="comma-separated subset; default all seven")
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    rng = np.random.default_rng(7)

    streams = (args.streams.split(",") if args.streams
               else list(GEMM_STREAMS) + list(ELT_OPS))
    ok = True
    print(f"unified design smoke test, batch {args.batch}")
    for stre in streams:
        if stre in GEMM_STREAMS:
            ok &= _check_gemm(stre, args.batch, args.hidden, rng)
        else:
            ok &= _check_elt(stre, args.batch, args.hidden, rng)
    print("ALL PASS" if ok else "FAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
