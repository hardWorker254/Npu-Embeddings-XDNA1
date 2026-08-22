# NpuEmbeddings -- build the minimal 1-column design, as a control.
#
# tasks/0024 established two things that sit in tension:
#
#   A->A' with the SAME xclbin in two contexts costs the same as two different
#   designs, so the switch does not depend on how DIFFERENT the configurations
#   are.  ...and yet...
#   The cost is ~286 us per column, so it clearly depends on how much of the
#   array the context covers.
#
# Both can hold, and then the diagnosis is that the driver tears down and
# rebuilds per-column state unconditionally, without taking the shortcut that
# it is identical. That is a lost fast path, not inevitable work -- a different
# claim with different consequences.
#
# 286 us/column is far too much to be bulk DMA of state (a column's ELFs are
# tens of KB) and entirely plausible as a few thousand individual MMIO writes to
# BD registers, stream switches and locks. That predicts the cost tracks the
# NUMBER of configured objects, not the column count as such.
#
# This design is the low end of that scale: 1 column, 1 worker, 2 ObjectFifos,
# a copy. Compared against a 1-column GEMM -- 4 workers, layout-transforming
# access patterns, many more BDs -- at identical width.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m7-switch-cost\build_passthrough.py

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
CACHE = Path.home() / ".npu" / "cache"

TILE = 1024
# The same element count GELU moves, so the comparison is not confounded by how
# much data crosses the shim. --batch/--cols make it match any GELU build:
# a passthrough at GELU's exact shape separates "DMA bound" from "compute
# bound", which tile size alone could not (tasks/0026).
N_ELEM = 256 * 1536


def passthrough_kernel():
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"
                       / _detect_arch()))
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    return ExternalFunction(
        "passthrough_bf16",
        source_file=str(HERE / "kernels" / "passthrough.cc"),
        arg_types=[tile_ty, tile_ty],
        include_dirs=include,
    )


def _build(dev, n_elem, n_cols, order="fwd"):
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    buf_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]
    k = passthrough_kernel()

    def core_fn_n(n):
        def core_fn(a, c, f):
            for _ in range_(n):
                ea = a.acquire(1)
                ec = c.acquire(1)
                f(ea, ec)
                a.release(1)
                c.release(1)
        return core_fn

    n_cores = 4 * n_cols
    per_core = (n_elem // TILE) // n_cores
    fins = [ObjectFifo(tile_ty, name=f"ptin{i}", depth=2) for i in range(n_cores)]
    fouts = [ObjectFifo(tile_ty, name=f"ptout{i}", depth=2) for i in range(n_cores)]
    ws = [Worker(core_fn_n(per_core), [fins[i].cons(), fouts[i].prod(), k],
                 stack_size=0xD00) for i in range(n_cores)]
    taps = TensorTiler2D.simple_tiler((1, n_elem), (1, per_core * TILE))
    # `order` changes ONLY the runtime sequence -- the same fills and drains
    # issued in a different order. The static design (workers, fifos, routing)
    # is untouched. This is the step-0 probe for the one-xclbin hypothesis:
    # if the runtime sequence lives entirely in insts.bin, the two builds must
    # produce a BYTE-IDENTICAL final.xclbin, and two insts streams can then be
    # dispatched through one hw_context with no design switch at all.
    idx = list(range(n_cores)) if order == "fwd" else list(reversed(range(n_cores)))

    def sequence(X, Y, fin_prods, fout_conss):
        tg = TaskGroup()
        for i in idx:
            fin_prods[i].fill(X, tap=taps[i], group=tg)
            fout_conss[i].drain(Y, tap=taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [buf_ty, buf_ty,
                            [f.prod() for f in fins], [f.cons() for f in fouts]])
    return Program(dev, rt, workers=ws).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def passthrough_array(X: In, Y: Out, *, n_elem: CompileTime[int],
                      n_cols: CompileTime[int] = 1,
                      order: CompileTime[str] = "fwd"):
    return _build(iron.get_current_device(), n_elem, n_cols, order)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--order", default="fwd", choices=["fwd", "rev"])
    a = ap.parse_args()
    n_elem = a.batch * 64 * 1536

    # Trap 1 in CLAUDE.md: without this the arch falls back to aie2 silently.
    iron.set_current_device(from_name("npu2", n_cols=None))

    X = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    Y = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    passthrough_array(X, Y, n_elem=n_elem, n_cols=a.cols, order=a.order)

    from export_xclbin import find_cache_by_markers, cache_core_columns

    src, n_hits = find_cache_by_markers(
        ["passthrough_bf16", f"memref<{n_elem}xbf16>"], "passthrough",
        n_cols=a.cols)
    out = (REPO / "runtime" / (a.out or "artifacts_pass")) / "passthrough"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        shutil.copy(src / f, out / f)

    placed = (src / "input_with_addresses.mlir").read_text(
        encoding="utf-8", errors="ignore")
    n_bd = placed.count("aie.dma_bd")
    n_lock = placed.count("aie.lock")

    (out / "design.json").write_text(json.dumps({
        "name": "passthrough", "kernel": "MLIR_AIE",
        "kind": "eltwise", "elems": n_elem, "cols": a.cols,
        "buffers": [n_elem * 2, n_elem * 2],
        "insts_bytes": (out / "insts.bin").stat().st_size,
        "source_cache_dir": src.name,
        "dma_bds": n_bd, "locks": n_lock,
    }, indent=2), encoding="utf-8")

    print(f"passthrough -> {out.relative_to(REPO)}  "
          f"({n_hits} cache dir{'s' if n_hits != 1 else ''} by symbol)")
    print(f"  columns {cache_core_columns(src)}  dma_bd {n_bd}  "
          f"aie.lock {n_lock}  xclbin "
          f"{(out / 'final.xclbin').stat().st_size / 1024:.1f} KB")

    # The same counts for the 1-column GEMM, so the comparison is quantified
    # rather than asserted.
    for name in ("qkv",):
        d = REPO / "runtime" / "artifacts1" / name / "design.json"
        if not d.exists():
            continue
        cd = CACHE / json.loads(d.read_text())["source_cache_dir"]
        m = cd / "input_with_addresses.mlir"
        if m.exists():
            t = m.read_text(encoding="utf-8", errors="ignore")
            print(f"  {name} (1 col)  dma_bd {t.count('aie.dma_bd')}  "
                  f"aie.lock {t.count('aie.lock')}  xclbin "
                  f"{(cd / 'final.xclbin').stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
