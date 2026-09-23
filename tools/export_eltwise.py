#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Build the three elementwise design directories the encoder needs to keep
# LayerNorm, softmax and GELU on the array instead of the host:
#
#   <out>/artifacts_npu<N>/gelu
#   <out>/artifacts_npu<N>/layernorm
#   <out>/artifacts_npu<N>/softmax
#
# The runtime only touches these when `--npu-eltwise` is given; without the
# flag the unified `gemm_rtp` set runs the three ops on the host, which is the
# measured-faster path. The exporter and the flag therefore agree on one layout:
# the eltwise directories sit next to `gemm_rtp` under the generation root, the
# same seven-design names the runtime already resolved before the unified path
# existed.
#
# The array programs are the ones the M5 experiments developed and are kept
# byte-for-byte in kernels/: `gelu_poly.cc`, `layernorm.cc`, `softmax.cc`. This
# tool only wraps them in an IRON program, compiles it once per (op, batch), and
# copies the two artifacts XRT needs out of the JIT cache. The C++ runtime
# compiles nothing (ground rule 3).
#
# Usage:
#   python tools/export_eltwise.py --arch all --batch 128 --hidden 384
#   python tools/export_eltwise.py --arch 1   --batch 128
#
# The eltwise designs are normally emitted alongside the GEMM set by
# `tools/export_gemm_rtp.py --with-eltwise`; this tool exists so they can be
# rebuilt on their own.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16


REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "kernels"
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))


DEFAULT_SEQ = 64
HEADS = 12
GELU_TILE = 1024
DEFAULT_CACHE_ROOT = Path.home() / ".npu" / "cache"

ARCH_DEVICES = {
    "1": os.environ.get("NPU_ARCH1_DEVICE", "npu1"),
    "2": os.environ.get("NPU_ARCH2_DEVICE", "npu2"),
}
ARCHES = ("1", "2")

# LayerNorm and softmax still open three and two fifos per core respectively;
# above two columns the shimNOC DMA capacity is exhausted and aiecc refuses with
# `no ShimNOCTile has sufficient DMA capacity` (docs/CURRENT_STATUS.md
# "Known walls"). GELU was rewritten around `.split()`/`.join()` and runs wider.
# The exporter refuses before compiling, and names the constraint, rather than
# failing deep inside the placer.
MAX_LN_SM_COLS = 2
MAX_GELU_COLS = 8


TILE_RE = re.compile(r"aie\.tile\((\d+)\s*,\s*(\d+)\)")


def core_columns(d: Path) -> int:
    """Count distinct core columns from the placed MLIR. Raise if unreadable.

    `aie.mlir` is pre-placement and carries no coordinates, so counting there
    yields 0 for every design; the placed form is `input_with_addresses.mlir`.
    A design whose width cannot be established is never accepted -- that is how
    a 2-column GELU once shipped as a 1-column one.
    """
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        raise RuntimeError(f"{d.name}: no input_with_addresses.mlir")
    text = m.read_text(encoding="utf-8", errors="ignore")
    cols = {int(c) for c, r in TILE_RE.findall(text) if int(r) >= 2}
    if not cols:
        raise RuntimeError(f"{d.name}: placed MLIR has no core tiles")
    return len(cols)


def _markers_match(text: str, symbols: list[str], n_elem: int) -> bool:
    if not all(s in text for s in symbols):
        return False
    return re.search(rf"memref<\s*{n_elem}\s*x\s*bf16\s*>", text) is not None


def purge(cache_dir: Path, symbols: list[str], n_elem: int, what: str) -> int:
    """Remove every cached design for this op before rebuilding.

    Symbol and buffer size cannot distinguish a design built at one column count
    from the same op at another, and a JIT cache hit does not restamp the
    directory, so mtime is not a tie-break. Removing first and requiring exactly
    one match afterwards is the same determinism the GEMM exporter uses.
    """
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for d in list(cache_dir.iterdir()):
        if not d.is_dir():
            continue
        mlir = d / "aie.mlir"
        if not mlir.exists():
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if _markers_match(text, symbols, n_elem):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        print(f"  {what}: purged {removed} cache candidate(s)")
    return removed


def find_cache(cache_dir: Path, symbols: list[str], n_elem: int,
               n_cols: int, what: str) -> Path:
    if not cache_dir.is_dir():
        raise SystemExit(f"{what}: cache directory does not exist: {cache_dir}")
    hits = []
    for d in cache_dir.iterdir():
        if not d.is_dir():
            continue
        if not ((d / "aie.mlir").exists() and (d / "final.xclbin").exists()
                and (d / "insts.bin").exists()):
            continue
        text = (d / "aie.mlir").read_text(encoding="utf-8", errors="ignore")
        if not _markers_match(text, symbols, n_elem):
            continue
        try:
            if core_columns(d) != n_cols:
                continue
        except RuntimeError:
            continue
        hits.append(d)
    if len(hits) != 1:
        raise SystemExit(
            f"{what}: {len(hits)} cache candidates after purge -- expected "
            f"exactly 1 in {cache_dir}")
    return hits[0]


def _extern_kernel(iron, symbol: str, filename: str, arg_types, tile: int | None):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    # Our kernels dir FIRST: aie_kernels ships same-named sources, and the
    # `*_rne.cc` variants #include the implementation they wrap by name.
    include.insert(0, str(KERNELS))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"
                           / _detect_arch()))
    return ExternalFunction(
        symbol,
        source_file=str(KERNELS / filename),
        arg_types=arg_types,
        include_dirs=include,
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
    GELU_SYMBOLS = {1024: "gelu_poly_bf16", 4096: "gelu_poly_bf16_4k"}

    def _gelu_build(dev, n_elem, n_cols, tile):
        n_rows = 4
        n_cores = n_rows * n_cols
        n_tiles = n_elem // tile
        assert n_elem % tile == 0, f"{n_elem} is not a multiple of {tile}"
        assert n_tiles % n_cores == 0, \
            f"{n_tiles} tiles do not spread over {n_cores} cores"
        per_core = n_tiles // n_cores
        k = _extern_kernel(
            iron, GELU_SYMBOLS[tile], "gelu_poly.cc",
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
                   n_cols: CompileTime[int] = 1,
                   tile: CompileTime[int] = GELU_TILE):
        # n_cols/tile are CompileTime so a per-call change recompiles rather
        # than being silently ignored (the jit decorator refuses bare defaults).

        return _gelu_build(iron.get_current_device(), n_elem, n_cols, tile)

    # -- LayerNorm: [rows, hidden] -> [rows, hidden], gamma|beta as one fp32
    #    vector because a core tile has only two input DMA channels ----------
    LN_COLS = 384
    LN_ROWS_PER_CALL = 16
    LN_SYMBOLS = {
        "base": ("layernorm_bf16", 0xD00),
        "il4": ("layernorm_il4_bf16", 0x2000),
        "rne": ("layernorm_rne_bf16", 0xD00),
        "il4_rne": ("layernorm_il4_rne_bf16", 0x2000),
    }

    def _ln_build(dev, rows, n_cols, variant):
        symbol, stack = LN_SYMBOLS[variant]
        src = "layernorm_rne.cc" if "rne" in variant else "layernorm.cc"
        n_cores = 4 * n_cols
        assert rows % (LN_ROWS_PER_CALL * n_cores) == 0, \
            f"{rows} rows do not split into {LN_ROWS_PER_CALL}-row blocks over {n_cores} cores"
        per_core = rows // (LN_ROWS_PER_CALL * n_cores)
        blk = LN_ROWS_PER_CALL * LN_COLS
        tile_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
        vec_ty = np.ndarray[(2 * LN_COLS,), np.dtype[np.float32]]
        buf_ty = np.ndarray[(rows * LN_COLS,), np.dtype[bfloat16]]
        k = _extern_kernel(iron, symbol, src, [tile_ty, vec_ty, tile_ty], None)

        def core_fn(a, pm, c, ln):
            ep = pm.acquire(1)
            for _ in range_(per_core):
                ea = a.acquire(1)
                ec = c.acquire(1)
                ln(ea, ep, ec)
                a.release(1)
                c.release(1)

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
        data_taps = TensorTiler2D.simple_tiler((1, rows * LN_COLS),
                                               (1, (rows * LN_COLS) // n_cols))
        param_tap = TensorTiler2D.simple_tiler((1, 2 * LN_COLS),
                                               (1, 2 * LN_COLS))[0]

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
                 variant: CompileTime[str] = "base"):
        return _ln_build(iron.get_current_device(), rows, n_cols, variant)

    # -- softmax: [rows, seq] -> [rows, seq], row-wise ----------------------
    SM_COLS = 64
    SM_ROWS_PER_CALL = 64
    SM_SYMBOLS = {
        "lib": ("softmax_bf16", "softmax.cc", 0xD00),
        "poly": ("softmax_poly_bf16", "softmax.cc", 0x2000),
        "poly_il4": ("softmax_poly_il4_bf16", "softmax.cc", 0x4000),
        "poly_rne": ("softmax_poly_rne_bf16", "softmax_rne.cc", 0x2000),
        "poly_il4_rne": ("softmax_poly_il4_rne_bf16", "softmax_rne.cc", 0x4000),
    }

    def _sm_build(dev, rows, n_cols, variant):
        symbol, src, stack = SM_SYMBOLS[variant]
        n_cores = 4 * n_cols
        assert rows % (SM_ROWS_PER_CALL * n_cores) == 0
        per_core = rows // (SM_ROWS_PER_CALL * n_cores)
        blk = SM_ROWS_PER_CALL * SM_COLS
        tile_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
        buf_ty = np.ndarray[(rows * SM_COLS,), np.dtype[bfloat16]]
        k = _extern_kernel(iron, symbol, src, [tile_ty, tile_ty], None)

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
        taps = TensorTiler2D.simple_tiler((1, rows * SM_COLS),
                                          (1, (rows * SM_COLS) // n_cols))

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
                 variant: CompileTime[str] = "lib"):
        return _sm_build(iron.get_current_device(), rows, n_cols, variant)

    _arrays._cache = SimpleNamespace(
        iron=iron, from_name=from_name,
        gelu_array=gelu_array, ln_array=ln_array, sm_array=sm_array,
        gelu_symbols=GELU_SYMBOLS, ln_symbols=LN_SYMBOLS,
        sm_symbols=SM_SYMBOLS)
    return _arrays._cache


def _write_dir(dst: Path, src: Path, meta: dict) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        shutil.copy2(src / f, dst / f)
    meta = dict(meta)
    meta["insts_bytes"] = (dst / "insts.bin").stat().st_size
    meta["source_cache_dir"] = src.name
    (dst / "design.json").write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    print(f"  {meta['name']:<10} arch {meta['arch']} {meta['device']:<5} "
          f"xclbin {(dst / 'final.xclbin').stat().st_size / 1024:7.1f} KB  "
          f"insts {meta['insts_bytes'] / 1024:6.1f} KB  ({src.name})")
    return meta


def export_arch(out_dir: Path, arch: str, batch: int, hidden: int = 384,
                seq: int = DEFAULT_SEQ, n_cols: int = 1, gelu_tile: int = GELU_TILE,
                ln_variant: str = "il4", sm_variant: str = "poly_il4",
                cache_root: Path | None = DEFAULT_CACHE_ROOT,
                per_arch_cache: bool = False) -> list[dict]:
    """Build the three eltwise designs into `out_dir` for one generation."""
    device = ARCH_DEVICES.get(arch)
    if not device:
        raise SystemExit(f"unknown architecture: {arch}")

    # The documented width walls, refused before compiling rather than deep in
    # the placer. Name the constraint so the fix is obvious.
    if n_cols > MAX_LN_SM_COLS:
        raise SystemExit(
            f"eltwise: {n_cols} columns is above the LayerNorm/softmax wall "
            f"({MAX_LN_SM_COLS}). Both still open three and two fifos per core "
            f"and exhaust the shimNOC DMA capacity "
            f"(`no ShimNOCTile has sufficient DMA capacity`) -- they must be "
            f"joined hierarchically through the mem tiles first, the way GELU "
            f"was. Use a smaller --elt-cols, or rewrite the kernels' dataflow.")
    if n_cols > MAX_GELU_COLS:
        raise SystemExit(
            f"eltwise: {n_cols} columns is above the GELU wall "
            f"({MAX_GELU_COLS}).")
    if batch % 4:
        raise SystemExit(f"--batch {batch} must be a multiple of 4")

    cache_dir = (Path(cache_root).expanduser() / f"arch{arch}"
                 if per_arch_cache else Path(cache_root).expanduser())
    if per_arch_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["IRON_CACHE_DIR"] = str(cache_dir)

    arr = _arrays()
    arr.iron.set_current_device(arr.from_name(device, n_cols=None))
    print(f"[eltwise arch {arch}] device={device} cache_dir={cache_dir}")

    n_gelu = batch * seq * 4 * hidden
    ln_rows = batch * seq
    sm_rows = batch * HEADS * seq

    metas: list[dict] = []

    # -- GELU ----------------------------------------------------------------
    gelu_sym = arr.gelu_symbols[gelu_tile]
    purge(cache_dir, [gelu_sym], n_gelu, f"gelu@b{batch}")
    X = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
    Y = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
    arr.gelu_array(X, Y, n_elem=n_gelu, n_cols=n_cols, tile=gelu_tile)
    src = find_cache(cache_dir, [gelu_sym], n_gelu, n_cols, f"gelu@b{batch}")
    metas.append(_write_dir(out_dir / "gelu", src, {
        "name": "gelu", "kind": "eltwise", "kernel": "MLIR_AIE",
        "arch": int(arch), "device": device,
        "a_dtype": "bf16", "c_dtype": "bf16",
        "buffers": [n_gelu * 2, n_gelu * 2],
        "rows": n_gelu, "hidden": hidden, "seq": seq, "batch": batch,
        "cols": n_cols, "tile": gelu_tile}))

    # -- LayerNorm -----------------------------------------------------------
    ln_sym, _ = arr.ln_symbols[ln_variant]
    purge(cache_dir, [ln_sym], ln_rows * 384, f"layernorm@b{batch}")
    X = arr.iron.zeros(ln_rows * 384, dtype=bfloat16, device="npu")
    P = arr.iron.zeros(2 * 384, dtype=np.float32, device="npu")
    Y = arr.iron.zeros(ln_rows * 384, dtype=bfloat16, device="npu")
    arr.ln_array(X, P, Y, rows=ln_rows, n_cols=n_cols, variant=ln_variant)
    src = find_cache(cache_dir, [ln_sym], ln_rows * 384, n_cols,
                     f"layernorm@b{batch}")
    metas.append(_write_dir(out_dir / "layernorm", src, {
        "name": "layernorm", "kind": "layernorm", "kernel": "MLIR_AIE",
        "arch": int(arch), "device": device,
        "a_dtype": "bf16", "c_dtype": "bf16",
        "buffers": [ln_rows * 384 * 2, 2 * 384 * 4, ln_rows * 384 * 2],
        "rows": ln_rows, "cols": 384, "hidden": hidden, "seq": seq,
        "batch": batch, "variant": ln_variant, "aie_cols": n_cols}))

    # -- softmax -------------------------------------------------------------
    sm_sym = arr.sm_symbols[sm_variant][0]
    purge(cache_dir, [sm_sym], sm_rows * 64, f"softmax@b{batch}")
    X = arr.iron.zeros(sm_rows * 64, dtype=bfloat16, device="npu")
    Y = arr.iron.zeros(sm_rows * 64, dtype=bfloat16, device="npu")
    arr.sm_array(X, Y, rows=sm_rows, n_cols=n_cols, variant=sm_variant)
    src = find_cache(cache_dir, [sm_sym], sm_rows * 64, n_cols,
                     f"softmax@b{batch}")
    metas.append(_write_dir(out_dir / "softmax", src, {
        "name": "softmax", "kind": "softmax", "kernel": "MLIR_AIE",
        "arch": int(arch), "device": device,
        "a_dtype": "bf16", "c_dtype": "bf16",
        "buffers": [sm_rows * 64 * 2, sm_rows * 64 * 2],
        "rows": sm_rows, "cols": 64, "hidden": hidden, "seq": seq,
        "batch": batch, "variant": sm_variant, "aie_cols": n_cols}))

    return metas


def arch_out_dir(base: str | Path, arch: str) -> Path:
    return Path(base) / f"artifacts_npu{arch}"


def build_child_argv(args, arch: str, outdir: Path) -> list[str]:
    cmd = [
        sys.executable or "python", str(Path(__file__).resolve()),
        "--arch", arch, "--out", str(outdir),
        "--batch", str(args.batch), "--seq", str(args.seq),
        "--hidden", str(args.hidden), "--elt-cols", str(args.elt_cols),
        "--gelu-tile", str(args.gelu_tile),
        "--ln-variant", args.ln_variant, "--sm-variant", args.sm_variant,
        "--cache-root", str(args.cache_root),
    ]
    if args.per_arch_cache:
        cmd.append("--per-arch-cache")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the gelu/layernorm/softmax design directories for "
                    "one or more NPU generations.")
    ap.add_argument("--arch", choices=["1", "2", "all"], default="all")
    ap.add_argument("--out", default=str(REPO / "runtime"),
                    help="artifacts root; each generation is written to "
                         "<out>/artifacts_npu<N>/{gelu,layernorm,softmax}. "
                         "Default: runtime/")
    ap.add_argument("--batch", type=int, default=128,
                    help="largest batch tier; also the eltwise buffer sizing")
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--elt-cols", type=int, default=1,
                    help="AIE columns for the eltwise designs. LayerNorm and "
                         f"softmax refuse above {MAX_LN_SM_COLS}.")
    ap.add_argument("--gelu-tile", type=int, default=GELU_TILE,
                    choices=[1024, 4096])
    ap.add_argument("--ln-variant", default="il4",
                    choices=["base", "il4", "rne", "il4_rne"])
    ap.add_argument("--sm-variant", default="poly_il4",
                    choices=["lib", "poly", "poly_il4", "poly_rne",
                             "poly_il4_rne"])
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--per-arch-cache", action="store_true")
    args = ap.parse_args()

    if args.arch == "all":
        if not args.per_arch_cache:
            args.per_arch_cache = True
        for arch in ARCHES:
            outdir = arch_out_dir(args.out, arch)
            cmd = build_child_argv(args, arch, outdir)
            print("[export] " + " ".join(repr(x) for x in cmd))
            subprocess.run(cmd, check=True)
        return 0

    args.out = str(arch_out_dir(args.out, args.arch))
    metas = export_arch(
        Path(args.out), args.arch, args.batch, args.hidden, args.seq,
        args.elt_cols, args.gelu_tile, args.ln_variant, args.sm_variant,
        args.cache_root, args.per_arch_cache)
    print(f"\nwrote {len(metas)} eltwise design(s) under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
