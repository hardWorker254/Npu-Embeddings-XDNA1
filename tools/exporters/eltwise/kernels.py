"""The IRON programs for gelu / layernorm / softmax, wrapped around kernels/*.cc.

The device code itself lives in kernels/ and is byte-for-byte what the M5
experiments produced. This module only builds the host program around it: the
extern-function declaration, the buffers, and the sequence. A missing Peano
install surfaces here as aiecc failing on kernels/*.cc.
"""

# NOTE: no `from __future__ import annotations` in this package on purpose.
# aie.iron.jit reads the annotations of the kernels it decorates at DECORATION
# time to tell a CompileTime[T] parameter from a runtime scalar. PEP 563 would
# turn every annotation into a string, iron would stop recognising
# CompileTime[T], and it rejects a parameter that has both a default and no
# recognised annotation. The failure is a TypeError raised from
# @iron.jit, several frames away from the cause.

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from .consts import (GELU_TILE, KERNELS, LN_ROWS_PER_CALL,
                     SM_ROWS_PER_CALL)

def _extern_kernel(iron, symbol: str, filename: str, arg_types, tile: int | None,
                  compile_flags: list[str] | None = None):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    # Our kernels dir FIRST: aie_kernels ships same-named sources, and the
    # `*_rne` variants #include the implementation they wrap by name.
    include.insert(0, str(KERNELS))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"
                           / _detect_arch()))
    return ExternalFunction(
        symbol,
        source_file=str(KERNELS / filename),
        arg_types=arg_types,
        include_dirs=include,
        compile_flags=compile_flags,
    )

def _arrays():
    """Import IRON and define the three array programs.

    Deferred so that `IRON_CACHE_DIR` can be set from `--per-arch-cache` before
    the first import of the AIE stack (only honoured if the toolchain reads it).
    """
    if getattr(_arrays, "_cache", None) is not None:
        return _arrays._cache

    import aie.iron as iron
    from aie.iron import (
        CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile, from_name
    from aie.helpers.taplib import TensorTiler2D

    # -- GELU: elementwise, 1024 or 4096 elements per tile -------------------
    #
    # Two FUNCTIONS, not one function with a tolerance: `poly` is a degree-8 fit
    # of the even part (2.49e-3 relative against exact erf) and `erf` is the
    # activation a Whisper container declares. A packer refuses a checkpoint
    # whose activation is not `gelu`, so the tanh polynomial is not a faster
    # GELU -- it is a different model, and its error is larger than the bf16
    # datapath's own.
    GELU_SYMBOLS = {1024: "gelu_poly_bf16", 4096: "gelu_poly_bf16_4k"}
    GELU_ERF_SYMBOLS = {1024: "gelu_erf_bf16", 4096: "gelu_erf_bf16_4k"}
    GELU_SRC = {"poly": "gelu_poly.cc", "erf": "gelu_erf.cc"}

    def _gelu_build(dev, n_elem, n_cols, tile, variant="poly"):
        n_rows = 4
        n_cores = n_rows * n_cols
        n_tiles = n_elem // tile
        assert n_elem % tile == 0, f"{n_elem} is not a multiple of {tile}"
        assert n_tiles % n_cores == 0, \
            f"{n_tiles} tiles do not spread over {n_cores} cores"
        per_core = n_tiles // n_cores
        syms = GELU_ERF_SYMBOLS if variant == "erf" else GELU_SYMBOLS
        k = _extern_kernel(
            iron, syms[tile], GELU_SRC[variant],
            [np.ndarray[(tile,), np.dtype[bfloat16]]] * 2, tile)

        def core_fn(a, c, gelu):
            for _ in range_(per_core):
                ea = a.acquire(1)
                ec = c.acquire(1)
                gelu(ea, ec)
                a.release(1)
                c.release(1)

        tile_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
        buf_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]
        l2_ty = np.ndarray[(n_rows * tile,), np.dtype[bfloat16]]
        offsets = [tile * r for r in range(n_rows)]
        workers, in_l3l2, out_l2l3 = [], [], []
        for c in range(n_cols):
            fin = ObjectFifo(l2_ty, name=f"gin_{c}", depth=2)
            fout = ObjectFifo(l2_ty, name=f"gout_{c}", depth=2)
            in_l3l2.append(fin)
            out_l2l3.append(fout)
            ins = fin.cons().split(offsets, obj_types=[tile_ty] * n_rows,
                                   names=[f"gin_{c}_{r}" for r in range(n_rows)])
            outs = fout.prod().join(offsets, obj_types=[tile_ty] * n_rows,
                                    names=[f"gout_{c}_{r}" for r in range(n_rows)],
                                    depths=[2] * n_rows)
            for r in range(n_rows):
                workers.append(Worker(core_fn,
                                      [ins[r].cons(), outs[r].prod(), k],
                                      tile=Tile(c, r + 2), stack_size=0x2000))
        taps = TensorTiler2D.simple_tiler((1, n_elem), (1, n_elem // n_cols))

        def sequence(X, Y, in_prods, out_conss):
            tg = TaskGroup()
            for c in range(n_cols):
                in_prods[c].fill(X, tap=taps[c], group=tg)
                out_conss[c].drain(Y, tap=taps[c], wait=True, group=tg)
            tg.finish()

        rt = Runtime(sequence, [buf_ty, buf_ty,
                                [f.prod() for f in in_l3l2],
                                [f.cons() for f in out_l2l3]])
        return Program(dev, rt, workers=workers).resolve_program()

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def gelu_array(X: In, Y: Out, *, n_elem: CompileTime[int],
                   variant: CompileTime[str] = "poly",
                   n_cols: CompileTime[int] = 1,
                   tile: CompileTime[int] = GELU_TILE):
        # n_cols/tile are CompileTime so a per-call change recompiles rather
        # than being silently ignored (the jit decorator refuses bare defaults).

        return _gelu_build(iron.get_current_device(), n_elem, n_cols, tile,
                           variant)

    # -- LayerNorm: [rows, hidden] -> [rows, hidden], gamma|beta as one fp32
    #    vector because a core tile has only two input DMA channels ----------
    #
    # The row WIDTH and the EPSILON are the caller's, not the kernel's: they
    # arrive as CompileTime and reach the device as -DLN_COLS/-DLN_EPS/-DLN_ROWS,
    # so a Whisper design (d_model wide, 1e-5) and a MiniLM one (384, 1e-12) are
    # the same source compiled twice rather than two sources. `rows_per_call` is
    # picked by the caller from the L1 budget, because the ping-pong buffers are
    # 2 * rows * cols * 2 B and L1 is 64 KB: sixteen rows of 1280 columns do not
    # fit and the design has to be exported with fewer.
    LN_SYMBOLS = {
        "base": ("layernorm_bf16", 0xD00),
        "il4": ("layernorm_il4_bf16", 0x2000),
        "rne": ("layernorm_rne_bf16", 0xD00),
        "il4_rne": ("layernorm_il4_rne_bf16", 0x2000),
        # The row count is part of the symbol, not a runtime argument: the JIT
        # cache identifies a design by the kernel it links.
        "il4_8": ("layernorm_il4_8_bf16", 0x2000),
        "il4_4": ("layernorm_il4_4_bf16", 0x2000),
    }

    def _ln_build(dev, rows, n_cols, variant, cols, eps, rows_per_call):
        symbol, stack = LN_SYMBOLS[variant]
        src = "layernorm_rne.cc" if "rne" in variant else "layernorm.cc"
        n_cores = 4 * n_cols
        assert rows % (rows_per_call * n_cores) == 0, \
            f"{rows} rows do not split into {rows_per_call}-row blocks over {n_cores} cores"
        per_core = rows // (rows_per_call * n_cores)
        blk = rows_per_call * cols
        tile_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
        vec_ty = np.ndarray[(2 * cols,), np.dtype[np.float32]]
        buf_ty = np.ndarray[(rows * cols,), np.dtype[bfloat16]]
        k = _extern_kernel(iron, symbol, src, [tile_ty, vec_ty, tile_ty], None,
                           compile_flags=[f"-DLN_COLS={cols}",
                                          f"-DLN_EPS={eps!r}f",
                                          f"-DLN_ROWS={rows_per_call}"])

        def core_fn(a, pm, c, ln):
            # The params object is acquired ONCE, outside the loop, because it
            # is the same gamma|beta for all `per_core` blocks -- so it MUST be
            # released once at the end to balance that acquire.
            #
            # It was not, and the imbalance is a correctness bug, not a leak.
            # `sequence()` issues exactly one `p_prods[c].fill(P, ...)` per
            # program run, so a run consumes 1 and returns 0: the L1-forwarded
            # buffer `lnp_l1_c` (depth 1) still holds an object when the program
            # ends. The next dispatch restarts the program, and the core's
            # `pm.acquire(1)` can then be satisfied by THAT leftover before the
            # new fill lands -- so a LayerNorm computes with the PREVIOUS
            # dispatch's gamma|beta. Measured as the same request answering
            # 0.38 and 0.68 on successive calls, and only ever on LayerNorm:
            # the GELU and softmax core_fn below have no params fifo at all and
            # are balanced. Releasing here leaves the buffer empty at exit, so
            # the next acquire blocks until its own fill arrives.
            ep = pm.acquire(1)
            for _ in range_(per_core):
                ea = a.acquire(1)
                ec = c.acquire(1)
                ln(ea, ep, ec)
                a.release(1)
                c.release(1)
            pm.release(1)

        n_rows_grid = 4
        l2_ty = np.ndarray[(n_rows_grid * blk,), np.dtype[bfloat16]]
        offsets = [blk * r for r in range(n_rows_grid)]
        in_l3l2, out_l2l3, p_l3l2, workers = [], [], [], []
        for c in range(n_cols):
            fin = ObjectFifo(l2_ty, name=f"lnin_{c}", depth=2)
            fout = ObjectFifo(l2_ty, name=f"lnout_{c}", depth=2)
            fp = ObjectFifo(vec_ty, name=f"lnp_{c}", depth=1)
            fp_l1 = fp.cons().forward(name=f"lnp_l1_{c}", depth=1)
            in_l3l2.append(fin)
            out_l2l3.append(fout)
            p_l3l2.append(fp)
            ins = fin.cons().split(offsets, obj_types=[tile_ty] * n_rows_grid,
                                   names=[f"lnin_{c}_{r}" for r in range(n_rows_grid)])
            outs = fout.prod().join(offsets, obj_types=[tile_ty] * n_rows_grid,
                                   names=[f"lnout_{c}_{r}" for r in range(n_rows_grid)],
                                   depths=[2] * n_rows_grid)
            for r in range(n_rows_grid):
                workers.append(Worker(core_fn,
                                      [ins[r].cons(), fp_l1.cons(),
                                       outs[r].prod(), k],
                                      tile=Tile(c, r + 2), stack_size=stack))
        data_taps = TensorTiler2D.simple_tiler((1, rows * cols),
                                               (1, (rows * cols) // n_cols))
        param_tap = TensorTiler2D.simple_tiler((1, 2 * cols),
                                               (1, 2 * cols))[0]

        def sequence(X, P, Y, in_prods, p_prods, out_conss):
            tg = TaskGroup()
            for c in range(n_cols):
                p_prods[c].fill(P, tap=param_tap, group=tg)
                in_prods[c].fill(X, tap=data_taps[c], group=tg)
                out_conss[c].drain(Y, tap=data_taps[c], wait=True, group=tg)
            tg.finish()

        rt = Runtime(sequence, [buf_ty, vec_ty, buf_ty,
                                [f.prod() for f in in_l3l2],
                                [f.prod() for f in p_l3l2],
                                [f.cons() for f in out_l2l3]])
        return Program(dev, rt, workers=workers).resolve_program()

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def ln_array(X: In, P: In, Y: Out, *, rows: CompileTime[int],
                 n_cols: CompileTime[int] = 1,
                 variant: CompileTime[str] = "base",
                 cols: CompileTime[int] = 384,
                 eps: CompileTime[float] = 1e-12,
                 rows_per_call: CompileTime[int] = LN_ROWS_PER_CALL):
        return _ln_build(iron.get_current_device(), rows, n_cols, variant,
                         cols, eps, rows_per_call)

    # -- softmax: [rows, cols] -> [rows, cols], row-wise --------------------
    #
    # The ROW WIDTH is the caller's, and for two very different reasons: BERT's
    # score row is the sequence length (64 here, the shipped kernel's own
    # compile-time constant), while a Whisper score row is n_kv -- 1500, padded
    # to 1536 -- which the 64-column kernel cannot hold at all, in registers or
    # on a worker's stack (kernels/softmax_w.cc says why). `cols = 0` in a row
    # below means "the caller decides", and those variants are the ones that
    # take -DSW_COLS.
    SM_COLS = 64
    SM_SYMBOLS = {
        "lib": ("softmax_bf16", "softmax.cc", 0xD00, SM_COLS, SM_ROWS_PER_CALL),
        "poly": ("softmax_poly_bf16", "softmax.cc", 0x2000, SM_COLS,
                 SM_ROWS_PER_CALL),
        "poly_il4": ("softmax_poly_il4_bf16", "softmax.cc", 0x4000, SM_COLS,
                     SM_ROWS_PER_CALL),
        "poly_rne": ("softmax_poly_rne_bf16", "softmax_rne.cc", 0x2000, SM_COLS,
                     SM_ROWS_PER_CALL),
        "poly_il4_rne": ("softmax_poly_il4_rne_bf16", "softmax_rne.cc", 0x4000,
                         SM_COLS, SM_ROWS_PER_CALL),
        "wide": ("softmax_w_bf16", "softmax_w.cc", 0x4000, 0, 1),
        "wide2": ("softmax_w2_bf16", "softmax_w.cc", 0x4000, 0, 2),
    }

    def _sm_build(dev, rows, n_cols, variant, cols=0, rows_per_call=0):
        symbol, src, stack, dcols, drpc = SM_SYMBOLS[variant]
        if dcols:
            cols = dcols
        if not rows_per_call:
            rows_per_call = drpc
        flags = None if dcols else [f"-DSW_COLS={cols}"]
        n_cores = 4 * n_cols
        assert rows % (rows_per_call * n_cores) == 0, (
            f"{rows} softmax rows do not split into {rows_per_call}-row blocks "
            f"over {n_cores} cores")
        per_core = rows // (rows_per_call * n_cores)
        blk = rows_per_call * cols
        tile_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
        buf_ty = np.ndarray[(rows * cols,), np.dtype[bfloat16]]
        k = _extern_kernel(iron, symbol, src, [tile_ty, tile_ty], None, flags)

        def core_fn(a, c, sm):
            for _ in range_(per_core):
                ea = a.acquire(1)
                ec = c.acquire(1)
                sm(ea, ec)
                a.release(1)
                c.release(1)

        n_rows_grid = 4
        l2_ty = np.ndarray[(n_rows_grid * blk,), np.dtype[bfloat16]]
        offsets = [blk * r for r in range(n_rows_grid)]
        in_l3l2, out_l2l3, workers = [], [], []
        for c in range(n_cols):
            fin = ObjectFifo(l2_ty, name=f"smin_{c}", depth=2)
            fout = ObjectFifo(l2_ty, name=f"smout_{c}", depth=2)
            in_l3l2.append(fin)
            out_l2l3.append(fout)
            ins = fin.cons().split(offsets, obj_types=[tile_ty] * n_rows_grid,
                                   names=[f"smin_{c}_{r}" for r in range(n_rows_grid)])
            outs = fout.prod().join(offsets, obj_types=[tile_ty] * n_rows_grid,
                                    names=[f"smout_{c}_{r}" for r in range(n_rows_grid)],
                                    depths=[2] * n_rows_grid)
            for r in range(n_rows_grid):
                workers.append(Worker(core_fn,
                                      [ins[r].cons(), outs[r].prod(), k],
                                      tile=Tile(c, r + 2), stack_size=stack))
        taps = TensorTiler2D.simple_tiler((1, rows * cols),
                                          (1, (rows * cols) // n_cols))

        def sequence(X, Y, in_prods, out_conss):
            tg = TaskGroup()
            for c in range(n_cols):
                in_prods[c].fill(X, tap=taps[c], group=tg)
                out_conss[c].drain(Y, tap=taps[c], wait=True, group=tg)
            tg.finish()

        rt = Runtime(sequence, [buf_ty, buf_ty,
                                [f.prod() for f in in_l3l2],
                                [f.cons() for f in out_l2l3]])
        return Program(dev, rt, workers=workers).resolve_program()

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def sm_array(X: In, Y: Out, *, rows: CompileTime[int],
                 n_cols: CompileTime[int] = 1,
                 variant: CompileTime[str] = "lib",
                 cols: CompileTime[int] = 0,
                 rows_per_call: CompileTime[int] = 0):
        return _sm_build(iron.get_current_device(), rows, n_cols, variant,
                         cols, rows_per_call)

    _arrays._cache = SimpleNamespace(
        iron=iron, from_name=from_name,
        gelu_array=gelu_array, ln_array=ln_array, sm_array=sm_array,
        gelu_symbols=GELU_SYMBOLS, gelu_erf_symbols=GELU_ERF_SYMBOLS,
        ln_symbols=LN_SYMBOLS, sm_symbols=SM_SYMBOLS)
    return _arrays._cache
