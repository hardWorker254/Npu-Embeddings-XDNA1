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
#   3. The host builds B via tools/npue.tile_b -- the SAME function that packs
#      the .npue file, so this validates the shipped layout rather than a
#      lookalike.
#
# Usage (from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced):
#     python gemm_pretiled.py --preset ffn_down --cols 4 -n 48
#     python gemm_pretiled.py --all-shapes --cols 4 -n 48
#     python gemm_pretiled.py --preset ffn_down --cols 4 -n 48 --baseline

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
    Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker,
    WorkerRuntimeBarrier, kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.trace import TraceConfig

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))

from npue import tile_b, untile_b            # noqa: E402

# The four MiniLM GEMMs. M=256 is one sequence at the model's real max length.
PRESETS = {
    "qkv":      dict(M=256, K=384,  N=1152),   # fused Q,K,V projection
    "proj":     dict(M=256, K=384,  N=384),    # attention output projection
    "ffn_up":   dict(M=256, K=384,  N=1536),   # FFN up
    "ffn_down": dict(M=256, K=1536, N=384),    # FFN down -- inexpressible in M2
}

# From tasks/0004: adding one trace flow exhausts routing at most widths.
TRACE_ROUTING = {2: (1, 1), 4: (0, 0)}


def _build_design(dev, M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str,
                  emulate_bf16_mmul_with_bfp16, trace_config, trace_row, trace_col,
                  trace_egress_col=0, pretiled=True, tile_order="k,n", inner_st=True,
                  b_reuse=False, rtp=False, epilogue=None, c_bf16=False):
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
        assert dtype_out is np.float32,             "c_bf16 narrows FROM fp32; the accumulator must be fp32"
        assert rtp, "c_bf16 is only wired into the rtp worker (tasks/0045)"
        assert epilogue is None,             "c_bf16 and epilogue='gelu' both own the post-K step; pick one"

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
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_c]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_c]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_c]]
    # The core-local fp32 accumulator the matmul writes into when c_bf16.
    # SINGLE buffered on purpose: it is filled and drained inside one
    # iteration, so it costs m*n*4 while the C fifo it feeds saves
    # 2*m*n*2 -- exactly cancelling. 53,248 B at (64,64,48) either way.
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

    # NPUE-M5 change 4: B REUSE ACROSS ROW BLOCKS.
    #
    # The design re-fills B from DDR once per row block. M2 estimated B being
    # re-streamed 16x at M=4096, and after pre-tiling turned out to be a wash
    # (tasks/0007) this is the remaining hypothesis for why the array starves.
    #
    # ObjectFifo's `repeat_count` is exactly the mechanism: "causes the MemTile
    # DMA to replay the buffer descriptor this many times WITHOUT a new DMA
    # transfer from L3". So we stage the whole per-column B slice in L2 once and
    # let the mem tile replay it to L1 for every row block.
    #
    # L2 budget per column, ffn_down at 4 cols:
    #   B  48 tiles x 64x48 x 2 B = 288 KB
    #   C  64x48 x 4 rows x 4 B x depth 2 =  98 KB
    #   A  64x64 x 2 B x depth 2        =  16 KB   -> 402 KB of 512 KB
    # At 8 columns B halves to 147 KB, so the wide configuration is comfortable
    # and the 4-column one is the tight case.
    n_row_blocks = M // m // n_aie_rows
    b_slice_tiles = (N // n // n_aie_cols) * (K // k)
    # b_reuse: False | int (stage that many tiles) | "mega".
    #
    # The obvious form -- an L2 fifo deep enough to hold the whole column slice
    # -- does not compile: the mem tile BD pool caps depth at 6 tiles at 4
    # columns and 4 at 8, while a slice needs 48 and 24. Measured, see
    # tasks/0010.
    #
    # "mega" is the way around it: one L2 object holding the ENTIRE slice, so
    # the same bytes cost one buffer descriptor instead of 24.
    # NPUE-M9 (tasks/0046): "asym" is the third attempt at B reuse, and the
    # first with a primitive that addresses BOTH ends of the transfer.
    #
    #   b_reuse=True   -- an L2 fifo deep enough for the slice. Mem-tile BD
    #                     pool caps depth at 6 (4 cols) / 4 (8 cols); a slice
    #                     wants 24-48. Fails at the MEM TILE.
    #   b_reuse="mega" -- one L2 object holding the whole slice, 1 descriptor.
    #                     The core then has to receive the whole klump:
    #                     "number of input DMA channel exceeded". Fails at the
    #                     CORE.
    #   b_reuse="asym" -- ONE fifo, big on the producer side (few descriptors)
    #                     and TILE-sized on the consumer side, via
    #                     consumer_obj_type=. Upstream's mobilenet
    #                     bottleneck/post_l1.py uses exactly this shape to hold
    #                     a weight buffer on a mem tile and feed compute tiles
    #                     chunks of it, with repeat_count replaying it.
    #
    # Capacity was never the blocker (tasks/0044 note 0007 s1.4): the column
    # slice is 108-144 KB at h=384 against ~400 KB free in a 512 KB mem tile.
    # VERDICT (tasks/0046): "asym" DOES NOT BUILD, and neither can any other B
    # reuse form at 8 columns. `repeat_count` is "unavailable for shim tiles",
    # so a single L3-fed fifo cannot replay; and the two-fifo route cannot carry
    # consumer_obj_type because forward()/split() do not expose it. Underneath
    # both: `tools/count_dma_channels.py` on the SHIPPING design shows every one
    # of the 32 core tiles at 2/2 input channels and five of eight mem tiles at
    # 6/6. There is no channel to give B. The mem-tile arithmetic is exact --
    # A(1) + B(1) + C(4 core rows) = 6 -- so the C JOIN is what spends the
    # budget, and freeing it needs CascadeFlow (C returned through one core
    # instead of four), not a fifo option.
    #
    # Kept in the tree, guarded, because the primitive looks right and the next
    # person will reach for it.
    asym = b_reuse == "asym"
    mega = b_reuse == "mega"
    B_l2_mega_ty = np.ndarray[(b_slice_tiles * k * n,), np.dtype[dtype_in]]
    b_depth = (2 if mega else
               0 if asym else            # asym sizes its own fifo below
               b_slice_tiles if b_reuse is True else int(b_reuse or 0))
    for col in range(n_aie_cols):
        if asym:
            # One fifo, no .forward(): the mem tile holds the whole slice and
            # each core acquires (k, n) tiles out of it. repeat_count replays
            # the staged bytes for every row block WITHOUT a new L3 fetch,
            # which is the entire point -- B is streamed from DDR once instead
            # of n_row_blocks times.
            B_l2l1_fifos[col] = ObjectFifo(
                B_l2_mega_ty, name=f"B_L3L2_{col}", depth=1,
                consumer_obj_type=B_l1_ty,
                repeat_count=n_row_blocks)
            B_l3l2_fifos[col] = B_l2l1_fifos[col]
        elif mega:
            B_l3l2_fifos[col] = ObjectFifo(B_l2_mega_ty, name=f"B_L3L2_{col}",
                                           depth=2)
        elif b_reuse:
            B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}",
                                           depth=b_depth)
        else:
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
        rc = n_row_blocks if b_reuse else None
        if asym:
            pass                      # asym has no second fifo to forward into
        elif pretiled and inner_st:
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}", repeat_count=rc)
        else:
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}", repeat_count=rc,
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

    # NPUE-M7 (expert review section 2, notes/0005): optional GELU epilogue,
    # applied by the core to the fp32 output tile before it is released. The
    # bias must already be inside the product -- ride it in as an augmented
    # K-block (A ones-column, B bias-row) so elem_out holds A@B + bias when
    # this runs. No extra fifos, no extra DMA, no extra dispatch.
    # NPUE-M9 (tasks/0045): the bf16 narrowing epilogue. Separate from the
    # `epilogue="gelu"` path above -- that one rewrites the fp32 tile in place
    # and C still leaves as fp32; this one converts fp32 -> bf16 into the fifo
    # object the DMA drains, which is what halves the transport.
    narrow_kernel = None
    if c_bf16:
        from aie.iron.kernel import ExternalFunction as _EF
        from aie.iron.kernels._common import _detect_arch as _da, _include_dirs as _id
        from aie.utils import config as _cfg2
        from pathlib import Path as _P2
        _inc2 = _id()
        _inc2.append(str(_P2(_cfg2.cxx_header_path()) / "aie_kernels"))
        _inc2.append(str(_P2(_cfg2.cxx_header_path()) / "aie_kernels" / _da()))
        assert m * n in (1024, 2048, 3072), (
            f"no narrow entry point for tile m*n={m*n}; "
            "narrow_f32_bf16.cc has 1024/2048/3072")
        narrow_kernel = _EF(
            f"narrow_{m * n}_f32_bf16",
            source_file=str(_P2(__file__).resolve().parent.parent
                            / "m5-eltwise" / "kernels" / "narrow_f32_bf16.cc"),
            arg_types=[C_acc_ty, C_l1_ty],
            include_dirs=_inc2,
        )

    epilogue_kernel = None
    if epilogue == "gelu":
        from aie.iron.kernel import ExternalFunction
        from aie.iron.kernels._common import _detect_arch, _include_dirs
        from aie.utils import config as _cfg
        from pathlib import Path as _P
        _inc = _include_dirs()
        _inc.append(str(_P(_cfg.cxx_header_path()) / "aie_kernels"))
        _inc.append(str(_P(_cfg.cxx_header_path()) / "aie_kernels"
                        / _detect_arch()))
        assert m * n == 3072, "gelu epilogue entry point is fixed at 3072"
        epilogue_kernel = ExternalFunction(
            "gelu_epilogue_3072_f32",
            source_file=str(_P(__file__).resolve().parent.parent
                            / "m5-eltwise" / "kernels" / "gelu_poly.cc"),
            arg_types=[np.ndarray[(m * n,), np.dtype[np.float32]]],
            include_dirs=_inc,
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
    elif epilogue == "gelu":
        def core_fn(in_a, in_b, out_c, zero, matmul, gelu_epi):
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
                gelu_epi(elem_out)
                out_c.release(1)

        def _mk(row, col):
            return Worker(
                core_fn,
                [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(),
                 C_l1l2_fifos[row][col].prod(), zero_kernel, matmul_kernel,
                 epilogue_kernel],
                # 0x2000: the 4-chain epilogue keeps 12 vectors live, and
                # 0xD00 corrupts silently (tasks/0026)
                stack_size=0x2000,
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

    def _set_rtps(*bufs):
        for b in bufs:
            b[0] = n_tiles_per_core
            b[1] = K // k

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        if rtp:
            rt.inline_ops(_set_rtps,
                          [rtp_bufs[r][c] for r in range(n_aie_rows)
                           for c in range(n_aie_cols)])
            for r in range(n_aie_rows):
                for c in range(n_aie_cols):
                    rt.set_barrier(rtp_barriers[r][c], 1)
        if trace_config is not None:
            rt.enable_trace(trace_config.trace_size,
                            [workers[trace_row][trace_col]],
                            egress_shim_col=trace_egress_col)
        rt.start(*[w for row in workers for w in row])

        tg = rt.task_group()
        if b_reuse:
            # Stream each column's B slice from DDR exactly ONCE, before any row
            # block runs. The mem tile replays it n_row_blocks times, so DDR
            # traffic for B drops by that factor -- 2x at M=512, 16x at M=4096.
            for col in range(n_aie_cols):
                rt.fill(B_l3l2_fifos[col].prod(), B, tap=B_taps[col],
                        task_group=tg)
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([tb_max_n_rows // 2,
                                         M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    rt.drain(C_l2l3_fifos[col].cons(), C, tap=C_tiles[c_index],
                             wait=True, task_group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        off = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(A_l3l2_fifos[col].prod(), A, tap=A_tiles[off],
                                    task_group=tg)
                        if not b_reuse:
                            rt.fill(B_l3l2_fifos[col].prod(), B, tap=B_taps[col],
                                    task_group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


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
    b_reuse: CompileTime[bool] = False,
    rtp: CompileTime[bool] = False,
    epilogue: CompileTime[str | None] = None,
    c_bf16: CompileTime[bool] = False,
):
    return _build_design(iron.get_current_device(), M, K, N, m, k, n, n_aie_cols,
                         dtype_in_str, dtype_out_str,
                         emulate_bf16_mmul_with_bfp16,
                         trace_config, trace_row, trace_col, trace_egress_col,
                         pretiled, tile_order, inner_st, b_reuse, rtp=rtp,
                         epilogue=epilogue, c_bf16=c_bf16)


def run_one(M, K, N, m, k, n, cols, emulate, trace_size, pretiled=True,
            tile_order="k,n", inner_st=True, b_reuse=False,
            dtype_in="bf16", dtype_out="f32", verbose=True, trace=True):
    """Compile + run one configuration. Returns a result dict, or None."""
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)

    in_sz, out_sz = np.dtype(dt_in).itemsize, np.dtype(dt_out).itemsize
    l1 = 2 * (m * k * in_sz + k * n * in_sz + m * n * out_sz)
    if l1 >= 64 * 1024:
        print(f"  SKIP cols={cols}: tile needs {l1} B of L1 (max 65536)")
        return None

    # Display label vs filesystem tag are deliberately separate: '|' and '[' are
    # invalid in Windows filenames, and the failure mode is an OSError from deep
    # inside the trace writer AFTER the kernel has already run.
    if pretiled:
        kind = f"pretiled[{tile_order}|{'st' if inner_st else 'rowmaj'}]"
        slug = f"pretiled_{tile_order.replace(',', '')}_{'st' if inner_st else 'rowmaj'}"
    else:
        kind = slug = "rowmajor"
    if b_reuse:
        kind += "+reuse"
        slug += "_reuse"
    tag = (f"{slug}_{cols}c_{dtype_in}_{dtype_out}{'_bfp16' if emulate else ''}"
           f"_{M}x{K}x{N}_t{m}x{k}x{n}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trace_txt = ARTIFACTS / f"trace_{tag}.txt"
    trace_json = ARTIFACTS / f"trace_{tag}.json"
    mlir_copy = ARTIFACTS / f"mlir_{tag}.mlir"

    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")
    A_np = A.numpy().copy()
    B_logical = B.numpy().copy()          # the mathematical [K,N] operand

    if pretiled:
        # Build the pre-tiled buffer with the SAME tile_b() that packs .npue,
        # so this exercises the shipped layout rather than a lookalike. The
        # device tensor's .numpy() is a live writable view, which is the only
        # way in: iron.tensor() cannot ingest an ml_dtypes bfloat16 array.
        r, s, t = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
            b_col_maj=False, c_col_maj=False, use_chess=False,
            emulate_bf16_mmul_with_bfp16=emulate, vectorized=True).mac_dims
        st = (s, t) if inner_st else (None, None)
        tiled = tile_b(B_logical.view(np.uint16), k, n, *st, order=tile_order)
        # Write through Tensor.__setitem__, not through .numpy().
        # `B.numpy()` syncs FROM the device and returns the host buffer; writing
        # into that array never syncs back, and only the first dispatch in a
        # process happens to come out right. `B[:] = x` syncs both ways.
        # See tasks/0009 -- this cost a full misdiagnosis.
        B[:] = tiled.view(bfloat16).reshape(K, N)
        # Prove the permutation is invertible on exactly these bytes before
        # trusting a hardware result that depends on it.
        back = untile_b(B.numpy().reshape(-1).view(np.uint16), K, N, k, n, *st,
                        order=tile_order)
        assert np.array_equal(back, B_logical.view(np.uint16)), "tile_b round-trip failed"

    tcol, egress = TRACE_ROUTING.get(cols, (None, None))
    cfg = None
    if trace:
        if tcol is None:
            print(f"  cols={cols}: NOT TRACEABLE; traceable widths are "
                  f"{sorted(TRACE_ROUTING)}")
            return None
        cfg = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt))

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, pretiled=pretiled,
              tile_order=tile_order, inner_st=inner_st, b_reuse=b_reuse,
              trace_config=cfg)
    if cfg is not None:
        kw.update(trace_row=0, trace_col=tcol, trace_egress_col=egress)
    pretiled_array(A, B, C, **kw)

    got = C.numpy().reshape(M, N).astype(np.float64)
    ref = A_np.astype(np.float64) @ B_logical.astype(np.float64)
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    tol = 5e-2 if emulate else 5e-3
    ok = rel_fro <= tol

    out = dict(kind=kind, tile_order=tile_order if pretiled else None,
               cols=cols, cores=4 * cols, M=M, K=K, N=N, m=m, k=k, n=n,
               dtype_in=dtype_in, dtype_out=dtype_out, emulate_bfp16=emulate,
               rel_frobenius=rel_fro, correctness_pass=ok)

    if cfg is None:
        if verbose:
            print(f"  cols={cols:>2} {kind:<8} relfro={rel_fro:.2e} "
                  f"{'PASS' if ok else 'FAIL'} (no trace)")
        return out

    size = trace_txt.stat().st_size if trace_txt.exists() else 0
    if size == 0:
        print(f"  cols={cols}: EMPTY TRACE -- raise --trace-size")
        out["trace"] = "empty"
        return out

    # physical_mlir_path is set by the JIT only when it actually compiles. On a
    # repeat run of an identical config the cache hits, nothing is compiled, and
    # the attribute stays None -- trace_to_json then dies with
    # "expected str, bytes or os.PathLike object, not NoneType". The copy we
    # keep for offline trace regeneration doubles as the fallback.
    phys = getattr(cfg, "physical_mlir_path", None)
    if phys:
        shutil.copy(phys, mlir_copy)
    elif mlir_copy.exists():
        phys = str(mlir_copy)
    if phys is None:
        print(f"  cols={cols}: no physical MLIR (cache hit, no stored copy) -- "
              f"clear {mlir_copy.name} or the JIT cache")
        return out
    cfg.trace_to_json(phys, str(trace_json))
    from aie.utils.trace.utils import get_cycles_summary

    deltas = []
    for entry in get_cycles_summary(str(trace_json)):
        deltas += [d for d in entry[1:] if d is not None]
    if not deltas:
        print(f"  cols={cols}: no event0/event1 pairs in trace")
        return out

    avg = sum(deltas) / len(deltas)
    per_core = (m * k * n) / avg
    peak_core = {"bf16": 256, "i16": 128, "i8": 512}[dtype_in]
    out.update(invocations=len(deltas), avg_cycles=avg,
               min_cycles=min(deltas), max_cycles=max(deltas),
               macs_per_cycle_per_core=per_core,
               macs_per_cycle_array=per_core * 4 * cols,
               peak_per_core=peak_core,
               efficiency_pct=per_core / peak_core * 100.0)
    if verbose:
        print(f"  cols={cols:>2} {kind:<8} cores={4*cols:>2} n={len(deltas):>5}  "
              f"avg={avg:8.1f} cyc  per-core={per_core:6.1f} MACs/cyc "
              f"({per_core/peak_core*100:5.1f}%)  relfro={rel_fro:.2e} "
              f"{'PASS' if ok else 'FAIL'}")
    return out


def bench_one(M, K, N, m, k, n, cols, emulate, pretiled, tile_order="k,n",
              inner_st=True, iters=50, warmup=10,
              dtype_in="bf16", dtype_out="f32"):
    """Wall-clock end-to-end throughput, NO trace.

    This is the metric M4 actually claimed to improve. Per-core cycles measure
    the compute window including DMA stalls, but "the array is starved" is a
    statement about the whole dispatch, and docs/05-measurement permits wall
    clock for exactly that -- labelled, never as a kernel-cycle claim, with the
    NPU quiesced.
    """
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)
    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")

    if pretiled:
        r, s, t = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
            b_col_maj=False, c_col_maj=False, use_chess=False,
            emulate_bf16_mmul_with_bfp16=emulate, vectorized=True).mac_dims
        st = (s, t) if inner_st else (None, None)
        B.numpy().reshape(-1).view(np.uint16)[:] = tile_b(
            B.numpy().copy().view(np.uint16), k, n, *st, order=tile_order)

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, pretiled=pretiled,
              tile_order=tile_order, inner_st=inner_st, trace_config=None)

    from aie.utils.benchmark import run_iters
    res = run_iters(pretiled_array, A, B, C, warmup=warmup, iters=iters, **kw)
    npu_us = getattr(getattr(res, "npu", None), "avg_us", None)
    npu_min = getattr(getattr(res, "npu", None), "min_us", None)
    e2e_us = getattr(getattr(res, "e2e", None), "avg_us", None)
    total_macs = M * K * N
    out = dict(kind="pretiled" if pretiled else "rowmajor",
               tile_order=tile_order if pretiled else None,
               inner_st=inner_st if pretiled else None,
               cols=cols, M=M, K=K, N=N, m=m, k=k, n=n, iters=iters,
               npu_avg_us=npu_us, npu_min_us=npu_min, e2e_avg_us=e2e_us)
    if npu_us:
        out["tflops_npu"] = 2 * total_macs / (npu_us * 1e-6) / 1e12
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M5: whole-array bf16 GEMM on pre-tiled B, traced")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ffn_down")
    ap.add_argument("--all-shapes", action="store_true",
                    help="run all four MiniLM GEMMs")
    ap.add_argument("-M", type=int); ap.add_argument("-K", type=int)
    ap.add_argument("-N", type=int)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--trace-size", type=int, default=262144)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="also run the M2 row-major B path as a control")
    ap.add_argument("--orders", default="k,n",
                    help="pre-tiled orders to try, ';'-separated: 'k,n;n,k'")
    ap.add_argument("--inner", default="st", choices=["st", "rowmaj", "both"],
                    help="'st' bakes the sub-tile order into the file; "
                         "'rowmaj' leaves it to dims_to_stream, isolating the "
                         "L3->L2 access-pattern change on its own")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each config; two runs of the SAME config "
                         "differed by 4.7%%, so a single number cannot support "
                         "a pretiled-vs-rowmajor claim")
    ap.add_argument("--bench", action="store_true",
                    help="wall-clock end-to-end instead of tracing")
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Without this IRON silently compiles for NPU1 and the bfp16 flag becomes a
    # no-op. research/notes/0002. n_cols=None or it defaults to a single column.
    iron.set_current_device(from_name("npu2", n_cols=None))

    shapes = (list(PRESETS.items()) if args.all_shapes
              else [(args.preset, dict(PRESETS[args.preset]))])
    results = []
    for name, shape in shapes:
        for key in ("M", "K", "N"):
            if getattr(args, key) is not None:
                shape[key] = getattr(args, key)
        M, K, N = shape["M"], shape["K"], shape["N"]
        print(f"\n{name}: {M}x{K}x{N}  tile ({args.m},{args.k},{args.n})  "
              f"cols={args.cols}  bfp16={args.emulate_bfp16}")

        inners = [True, False] if args.inner == "both" else [args.inner == "st"]

        if args.bench:
            print(f"  {'variant':<22} {'npu avg us':>11} {'npu best us':>12} "
                  f"{'TFLOP/s':>9}")
            bvars = ([(False, "k,n", True)] if args.baseline else [])
            bvars += [(True, o, i) for o in args.orders.split(";") for i in inners]
            for pt, order, inner in bvars:
                b = bench_one(M, K, N, args.m, args.k, args.n, args.cols,
                              args.emulate_bfp16, pretiled=pt, tile_order=order,
                              inner_st=inner, iters=args.bench_iters)
                b["shape_name"] = name
                results.append(b)
                lbl = (f"pretiled[{order}|{'st' if inner else 'rowmaj'}]"
                       if pt else "rowmajor")
                print(f"  {lbl:<22} {b.get('npu_avg_us') or float('nan'):>11.1f} "
                      f"{b.get('npu_min_us') or float('nan'):>12.1f} "
                      f"{b.get('tflops_npu') or float('nan'):>9.2f}")
            continue
        variants = [(False, None, True)] if args.baseline else []
        variants += [(True, o, i) for o in args.orders.split(";") for i in inners]
        for kind_pretiled, order, inner in variants:
            per = []
            for rep in range(args.repeat):
                try:
                    res = run_one(M, K, N, args.m, args.k, args.n, args.cols,
                                  args.emulate_bfp16, args.trace_size,
                                  pretiled=kind_pretiled,
                                  tile_order=order or "k,n", inner_st=inner,
                                  trace=not args.no_trace)
                except Exception as e:
                    msg = str(e)
                    hit = "exceeds the [0:1023] range" in msg
                    print(f"  cols={args.cols} "
                          f"{'pretiled' if kind_pretiled else 'rowmajor'} "
                          f"FAILED TO COMPILE"
                          f"{' -- BD size limit' if hit else ''}")
                    for line in msg.splitlines():
                        if "aie.dma_bd" in line or "exceeds" in line:
                            print(f"    {line.strip()[:150]}")
                    results.append({"kind": "pretiled" if kind_pretiled else "rowmajor",
                                    "tile_order": order,
                                    "shape_name": name, "cols": args.cols,
                                    "M": M, "K": K, "N": N,
                                    "compile_failed": True, "bd_limit": hit})
                    break
                if res:
                    res["shape_name"] = name
                    res["repeat"] = rep
                    results.append(res)
                    if "macs_per_cycle_per_core" in res:
                        per.append(res["macs_per_cycle_per_core"])
            if len(per) > 1:
                lo, hi, mean = min(per), max(per), sum(per) / len(per)
                label = (f"pretiled[{order}|{'st' if inner else 'rowmaj'}]"
                         if kind_pretiled else "rowmajor")
                print(f"     {label:<16} "
                      f"over {len(per)} runs: mean {mean:6.1f}  "
                      f"range {lo:.1f}-{hi:.1f}  spread {(hi-lo)/mean*100:.1f}%")

    if args.out:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
