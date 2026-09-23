# NpuEmbeddings -- M5: whole-array bf16 GEMM consuming PRE-TILED B, traced
# SPDX-License-Identifier: Apache-2.0
#
# Derived from our own experiments/m2-bf16-gemm/gemm_whole_array.py, which is in
# turn derived from mlir-aie programming_examples/.../whole_array/whole_array.py
#   Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
#   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# WHAT THIS CHANGES vs M2, and why
# --------------------------------
# M2 could not compile ffn_down at all. Reproduced before writing a line of this
# file, so the fix is measured against a real failure rather than a remembered
# one:
#
#   aie.mlir:80:9: error: 'aie.dma_bd' op Size 1 exceeds the [0:1023] range.
#     aie.dma_bd(%arg1 : memref<1536x384xbf16>, 0, 589824,
#       [<size=1, stride=0>, <size=12, stride=32>,
#        <size=1536, stride=384>, <size=32, stride=1>])
#
# Read the failing dimension: `<size=1536, stride=384>` is K itself, walking all
# 1536 rows of a row-major [K, N] with stride N. The DMA BD size field is 10
# bits, so K <= 1023 -- bisected in M2 at K=960 works, K=1024 fails.
#
# With B pre-tiled offline (M4), the transfer stops being a strided gather over
# the tensor and becomes a walk over TILE INDICES:
#
#   sizes   [N/n/cols, K/k, k, n]      <- all small
#   strides [cols*k*n, (N/n)*k*n, n, 1]
#
# For ffn_down at 4 columns that is [2, 24, 64, 48] instead of a 1536. The two
# inner dims are just a contiguous k*n run expressed in two dimensions, which is
# also why k*n = 3072 never appears as a single size.
#
# The second change is the one that is easy to miss: the L2->L1 forward drops
# its `dims_to_stream`. In M2 that argument reordered each tile into the MAC
# intrinsic's (s, t) sub-tile order on the way into L1. M4 bakes that order into
# the file, so the forward is now a plain linear copy. Both re-layouts M2 did at
# runtime are gone.
#
# OUR CHANGES vs the M2 file, all marked `# NPUE-M5:`
#   1. B is a pre-tiled buffer; the L3->L2 tap is an explicit
#      TensorAccessPattern over tile indices instead of TensorTiler2D.step_tiler.
#   2. B's L2->L1 forward has no dims_to_stream.
#   3. The host harness (gemm_pretiled_research.py) builds B via
#      tools/npue.tile_b -- the SAME function that packs the .npue file, so it
#      validates the shipped layout rather than a lookalike.
#
# This file is the design LIBRARY. The research driver (presets, tracing,
# benchmarking) lives in gemm_pretiled_research.py; the production exporter
# imports pretiled_array() from here.

from __future__ import annotations

from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker,
    WorkerRuntimeBarrier, kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.trace import TraceConfig

# HERE is tools/; REPO is the repository root (this file lives in tools/).
HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _build_design(dev, M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str,
                  emulate_bf16_mmul_with_bfp16, trace_config, trace_row, trace_col,
                  trace_egress_col=0, pretiled=True, tile_order="k,n", inner_st=True,
                  rtp=False, c_bf16=False,
                  b_l1_depth=2, fifo_depth=2):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    # NPUE-M9 (tasks/0045): c_bf16 narrows C to bf16 ON THE CORE, after the
    # full K reduction, so the C DMA moves half the bytes. `dtype_out` stays
    # the ACCUMULATOR type and must stay fp32 -- CLAUDE.md trap 2 forbids
    # output_dtype=bf16 on the matmul kernel because that re-rounds at every K
    # step (7.4e-3 against 1.21e-07). Only the TRANSPORT type changes here.
    dtype_c = str_to_dtype("bf16") if c_bf16 else dtype_out
    if c_bf16:
        # NPUE-M13 (tasks/0080): the int8 datapath narrows from int32. Same
        # transport saving, different reason -- under bf16 the GEMM is
        # iteration-bound (0048) and this bought +4.9%; under int8 it is
        # traffic-bound again (0010's model, refitted at R2 0.987) and C is 61%
        # of the traffic on three of four production shapes. int32 -> bf16
        # needs no extra core operand, so unlike a per-column rescale on the
        # core it does not run into trap 3b's 2-in / 2-out wall.
        assert dtype_out in (np.float32, np.int32),             "c_bf16 narrows FROM the accumulator; fp32 or int32 only"
        assert rtp, "c_bf16 is only wired into the rtp worker (tasks/0045)"

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

    # DEPTH IS THE PREFETCH DISTANCE. depth=2 is double buffering: the DMA
    # fills one object while the core computes on the other, which is how an
    # AIE design overlaps movement with compute at all. It is also the leading
    # `2 *` in trap 3's L1 budget -- the overlap is not free, it is paid for in
    # L1 bytes, and a deeper prefetch buys distance at the cost of tile size.
    # Parameterised in tasks/0083 to measure which of those two the datapath
    # actually wants.
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_c]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_c]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_c]]
    # The core-local accumulator the matmul writes into when c_bf16 -- fp32 on
    # the bf16 datapath, int32 on the int8 one, both 4 bytes.
    # SINGLE buffered on purpose: it is filled and drained inside one
    # iteration, so it costs m*n*4 while the C fifo it feeds saves
    # 2*m*n*2 -- exactly cancelling. 53,248 B at (64,64,48) either way, and
    # 38,912 B for the int8 operands -- unchanged by narrowing in both cases.
    C_acc_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # A is untouched: activations are runtime data, not weights, so they are not
    # pre-tiled. Its BD sizes are (K//k, k) and (m, K) -- 24 and 64 at ffn_down,
    # nowhere near 1023. Only B ever hit the limit.
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

    # B is single-buffered at the L3->L2 stage and forwarded to L1. The B-reuse
    # experiments (L2 slice staging, "mega", "asym") are removed here; they
    # never built and are recorded in docs/CURRENT_STATUS.md §"Known walls".
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}",
                                       depth=fifo_depth)
        # NPUE-M5 change 2: with the (s, t) sub-tile order baked into the file
        # the forward becomes a plain linear copy.
        #
        # `inner_st=False` keeps the tile interior row-major and leaves this
        # dims_to_stream in place, which ISOLATES change 1 (the L3->L2 access
        # pattern) from change 2 (moving the sub-tile reorder offline). Both
        # variants are numerically identical; only where the reorder happens
        # differs.
        # NPUE-M9 (T19, tasks/0052): b_l1_depth=1 single-buffers B in L1 --
        # the Stationary-B budget `2mk + kn + 2mn` from ICPP'25 (trap 3),
        # which is what makes k=96 legal where the all-double-buffered
        # default overflows. The cost it risks is the fetch/compute overlap
        # on B; tasks/0049 says the DMA idles in the compute's shadow on the
        # plain-bf16 path, and the traced per-iteration cycles decide.
        if pretiled and inner_st:
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}",
                depth=b_l1_depth)
        else:
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}",
                depth=b_l1_depth,
                dims_to_stream=[(k // s, s * n), (n // t, t), (s, n), (t, 1)])
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)])
        tmp = C_l2l3_fifos[col].prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = tmp[j]

    # NPUE-M9 (tasks/0045): the bf16 narrowing epilogue -- converts fp32 -> bf16
    # into the fifo object the DMA drains, which is what halves the transport.
    narrow_kernel = None
    if c_bf16:
        from aie.iron.kernel import ExternalFunction as _EF
        from aie.iron.kernels._common import _detect_arch as _da, _include_dirs as _id
        from aie.utils import config as _cfg2
        from pathlib import Path as _P2
        _inc2 = _id()
        _inc2.append(str(_P2(_cfg2.cxx_header_path()) / "aie_kernels"))
        _inc2.append(str(_P2(_cfg2.cxx_header_path()) / "aie_kernels" / _da()))
        # The accumulator dtype picks the source file and the entry point. Both
        # kernels expose the same three tile sizes and the same signature, so
        # nothing downstream of here knows which one it got.
        _acc_tag = "i32" if dtype_out is np.int32 else "f32"
        # 4096 (tile_n=64) exists only on the int8 side -- at bf16's 2-byte
        # operands that tile needs 65,536 B of a 63 KB L1 (tasks/0081).
        _ok = (1024, 2048, 3072, 4096) if _acc_tag == "i32" else (1024, 2048, 3072)
        _narrow_src = str(REPO / "kernels" / f"narrow_{_acc_tag}_bf16.cc")
        # narrow_f32_bf16.cc ALWAYS writes bf16 -- use a dedicated bf16 type
        # for the second arg rather than the outer C_l1_ty, which is only
        # bf16 when c_bf16 is set.
        _narrow_out_ty = np.ndarray[(m, n), np.dtype[str_to_dtype("bf16")]]
        assert m * n in _ok, (
            f"no narrow entry point for tile m*n={m*n}; "
            f"narrow_{_acc_tag}_bf16.cc has {'/'.join(str(v) for v in _ok)}")
        narrow_kernel = _EF(
            f"narrow_{m * n}_{_acc_tag}_bf16",
            source_file=_narrow_src,
            arg_types=[C_acc_ty, _narrow_out_ty],
            include_dirs=_inc2,
        )

    # NPUE-M7, the one-xclbin architecture (tasks/0029, notes/0005 section 1):
    # the ONLY shape-dependent values in the static design are these two loop
    # bounds. Compiled in (rtp=False) they make each GEMM shape its own ELF and
    # therefore its own xclbin and hw_context -- and every design change costs
    # a context switch. Hoisted into runtime parameters (rtp=True), one ELF
    # serves every shape and each GEMM becomes an instruction stream plus two
    # RTP writes over ONE context. The barrier orders the RTP write before the
    # core reads it, exactly as in programming_examples/ml/scale_shift.
    if rtp:
        rtp_bufs = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]],
                            name=f"rtp_{r}_{c}",
                            # ZEROS, not the real bounds: the initial value
                            # is baked into the static image, and a
                            # shape-dependent initializer was exactly the 8
                            # bytes that kept two shapes' xclbins from being
                            # identical. The runtime sequence writes the real
                            # bounds before the barrier releases the core.
                            initial_value=np.zeros(2, dtype=np.int32),
                            use_write_rtp=True)
                     for c in range(n_aie_cols)] for r in range(n_aie_rows)]
        rtp_barriers = [[WorkerRuntimeBarrier()
                         for _ in range(n_aie_cols)] for _ in range(n_aie_rows)]

        if c_bf16:
            # One fp32 accumulator per core. The acquire stays BEFORE the K
            # loop, exactly where it was, so the fifo's double buffering and
            # the DMA overlap behave identically to the fp32 design -- only
            # what gets written into the acquired object changes.
            acc_bufs = [[Buffer(C_acc_ty, name=f"cacc_{r}_{c}")
                         for c in range(n_aie_cols)] for r in range(n_aie_rows)]

            def core_fn(in_a, in_b, out_c, acc, zero, matmul, narrow,
                        my_rtp, barrier):
                barrier.wait_for_value(1)
                n_out_tiles = my_rtp[0]
                n_k_blocks = my_rtp[1]
                for _ in range_(n_out_tiles):
                    elem_out = out_c.acquire(1)
                    zero(acc)
                    for _ in range_(n_k_blocks):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, acc)
                        in_a.release(1)
                        in_b.release(1)
                    narrow(acc, elem_out)
                    out_c.release(1)
                barrier.release_with_value(1)

            def _mk(row, col):
                return Worker(
                    core_fn,
                    [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
                     C_l1l2_fifos[row][col].prod(), acc_bufs[row][col],
                     zero_kernel, matmul_kernel, narrow_kernel,
                     rtp_bufs[row][col], rtp_barriers[row][col]],
                    # Same 0xD00 as the fp32 path: narrow_f32_bf16 keeps two
                    # accumulators live and spills nothing (17 instructions,
                    # 2-line ZOL -- see tasks/0045). If this ever hangs or
                    # corrupts, the stack is the first suspect (traps 5b/0031).
                    stack_size=0xD00,
                    trace=1 if (row == trace_row and col == trace_col) else None,
                )
        else:
            def core_fn(in_a, in_b, out_c, zero, matmul, my_rtp, barrier):
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

            def _mk(row, col):
                return Worker(
                    core_fn,
                    [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
                     C_l1l2_fifos[row][col].prod(), zero_kernel, matmul_kernel,
                     rtp_bufs[row][col], rtp_barriers[row][col]],
                    stack_size=0xD00,
                    trace=1 if (row == trace_row and col == trace_col) else None,
                )
    else:
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
    # NPUE-M5: M2 hard-coded tb_max_n_rows//2 and only ever ran M=512, which has
    # exactly 2 row blocks. MiniLM's real single-sequence shape is M=256 -- ONE
    # row block -- and the C tiler then rejects the design outright
    # ("tensor does not divide evenly into tile groups in dimension 0").
    tb_n_rows = min(tb_max_n_rows // 2, M // m // n_aie_rows)
    # NPUE-M7 (research/notes/0005 section 5b): the C-drain tap repeats over row
    # blocks with stride m*n_aie_rows*N elements, and the DMA stride field is
    # 20 bits ([1:1048576], INCLUSIVE -- measured: N=4096 at exactly 2^20
    # builds). Above that the whole design fails with `'aie.dma_bd' op Stride 3
    # exceeds the range`, which is what walled off hidden >= 1536 in
    # tasks/0027. With one row block per drain the repeat dimension carries no
    # stride, at the cost of twice as many drain tasks.
    if m * n_aie_rows * N > 2**20:
        tb_n_rows = 1

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False)

    # NPUE-M5 change 1: B's access pattern.
    #
    # Tile (kb, nb) lives at (kb*NB + nb) * k*n in the pre-tiled buffer -- the
    # "k,n,kt,nt" order of the .npue layout. Column `col` consumes the n-blocks
    # congruent to col mod n_aie_cols, and for each one walks all K/k k-blocks,
    # because the core's inner loop is `for _ in range_(K//k)`. So kb must vary
    # fastest, which is what the KB dimension being inner expresses.
    #
    # The two innermost dims are one contiguous k*n run written as (k, n); that
    # factoring is what keeps k*n = 3072 from ever appearing as a single size.
    TE = k * n
    KB, NB = K // k, N // n
    NBC = NB // n_aie_cols
    if pretiled:
        # Only the KB stride differs between the two orders, and that is exactly
        # the point: the DMA does the same number of transfers of the same size,
        # so any difference is locality alone.
        #   "k,n": tile (kb,nb) at (kb*NB + nb)*TE -> KB stride NB*TE
        #   "n,k": tile (nb,kb) at (nb*KB + kb)*TE -> KB stride TE (contiguous)
        if tile_order == "k,n":
            kb_stride, nb_stride = NB * TE, TE
        else:
            kb_stride, nb_stride = TE, KB * TE
        B_taps = [
            TensorAccessPattern(
                (K * N,), col * nb_stride,
                [NBC, KB, k, n],
                [n_aie_cols * nb_stride, kb_stride, n, 1],
            )
            for col in range(n_aie_cols)
        ]
    else:
        B_taps = TensorTiler2D.step_tiler(
            (K, N), (k, n), tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols), tile_group_col_major=True,
            prune_step=False)

    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), prune_step=False)
    c_index = 0

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conss = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_prod_hs, B_prod_hs, C_cons_hs):
        nonlocal c_index
        if rtp:
            # A Buffer with use_write_rtp=True emits its write inline when
            # assigned inside the active sequence body -- no separate
            # inline_ops() call needed (matches
            # programming_examples/ml/scale_shift.py's `rtp[0] = value`).
            for r in range(n_aie_rows):
                for c in range(n_aie_cols):
                    rtp_bufs[r][c][0] = n_tiles_per_core
                    rtp_bufs[r][c][1] = K // k
            for r in range(n_aie_rows):
                for c in range(n_aie_cols):
                    rtp_barriers[r][c].set(1)

        # NPUE-M13 (tasks/0068): the C-drain tap's row-group width is
        # `tb_n_rows` (computed above, line ~503, and forced to 1 by the
        # 20-bit DMA-stride guard at line ~511). This fill/drain walk used to
        # step by the hardcoded `tb_max_n_rows // 2` regardless of what
        # `tb_n_rows` actually was, so whenever the guard forced tb_n_rows
        # down to 1 the loop kept filling+computing 2 row blocks per drain
        # call while each drain tap only covered 1 -- half of every C tile
        # was silently never DMA'd out (stale host memory, not a numeric
        # error): rel_fro ~7.07e-01, 28/32 row-bands with max|err| > 1.0.
        # tb_step below is the outer (2-way ping-pong) stride expressed in
        # units of tb_n_rows; at the historical unguarded tb_n_rows=2 it
        # equals tb_max_n_rows=4 exactly, so every shape below the guard
        # threshold is bit-for-bit unaffected by this change.
        #
        # The guard has never fired in any shipped design: bge-large's real
        # production N=4096 sits at EXACTLY 2**20 and the guard is a strict
        # '>', so this bug was latent until nomic's N=6144 ffn_up crossed it.
        tb_step = 2 * tb_n_rows
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_step)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_step + pingpong * tb_n_rows
                current_tb_n_rows = min([tb_n_rows,
                                         M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    C_cons_hs[col].drain(C, tap=C_tiles[c_index], wait=True)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        off = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            A_prod_hs[col].fill(A, tap=A_tiles[off])
                        B_prod_hs[col].fill(B, tap=B_taps[col])

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conss])

    program = Program(dev, rt, workers=[w for row in workers for w in row])
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size,
                             workers=[workers[trace_row][trace_col]],
                             egress_shim_col=trace_egress_col)
    return program.resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def pretiled_array(
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
    pretiled: CompileTime[bool] = True,
    tile_order: CompileTime[str] = "k,n",
    inner_st: CompileTime[bool] = True,
    rtp: CompileTime[bool] = False,
    c_bf16: CompileTime[bool] = False,
    b_l1_depth: CompileTime[int] = 2,
    fifo_depth: CompileTime[int] = 2,
):
    return _build_design(iron.get_current_device(), M, K, N, m, k, n, n_aie_cols,
                         dtype_in_str, dtype_out_str,
                         emulate_bf16_mmul_with_bfp16,
                         trace_config, trace_row, trace_col, trace_egress_col,
                         pretiled, tile_order, inner_st, rtp=rtp,
                         c_bf16=c_bf16,
                         b_l1_depth=b_l1_depth, fifo_depth=fifo_depth)
