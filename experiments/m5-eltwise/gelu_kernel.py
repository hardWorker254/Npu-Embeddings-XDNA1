# NpuEmbeddings -- M5: the first ELEMENTWISE op on the array, validated against
# the M3 goldens.
# SPDX-License-Identifier: Apache-2.0
#
# Every kernel so far has been a GEMM. A fused encoder layer also needs GELU,
# LayerNorm, softmax and bias adds, and none of those has been on a core yet.
# This is the smallest one that matters, and it answers a question we cannot
# answer on paper.
#
# WHAT IS ACTUALLY IN DOUBT
# -------------------------
# IRON ships `kernels.gelu()`, and two things about it disagree with
# docs/04-model:
#
#   1. It is the **tanh approximation**, not exact erf. The doc calls tanh a
#      landmine ("~1e-3 systematically-biased error").
#   2. It is **LUT-backed in bf16**, and IRON's own docstring suggests verifying
#      it with `rtol=0.128` -- a 12.8% relative tolerance.
#
# On (1) we already have a CPU answer: on real L0.ffn_up data, tanh costs
# 5.62e-04 relative in fp32, but once the output is bf16 the rounding floor is
# 1.69e-03 and tanh only moves it to 1.78e-03 -- about 5% worse, not a doubling.
# The doc's warning is right for an fp32 reference and too strict for a bf16
# datapath.
#
# On (2) there is no CPU answer. A LUT's real accuracy is whatever the table
# says, and the only way to find out is to run it. That is what this file is
# for: feed the genuine L0.ffn_up activations from the goldens through the
# hardware kernel and compare against the exact-erf golden L0.gelu.
#
# The kernel is fixed at 1024-element tiles and bf16 in/out, so [256, 1536] =
# 393,216 elements divides into exactly 384 tiles.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m5-eltwise\gelu_kernel.py
#   python experiments\m5-eltwise\gelu_kernel.py --cols 4

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "reference"))

from safetensors_io import load                    # noqa: E402

TILE = 1024          # kernels.gelu accepts nothing else; ours matches it

# Our own polynomial kernel also has a 4096-element entry point. The tile size
# turned out to BE the cost at scale: at batch 64, GELU moved 25 MB per call in
# 16.5 ms (1.5 GB/s against tasks/0010's 33 GB/s) and spent 21.5 us on a
# 1024-element tile that holds ~2 us of arithmetic. Per-transaction overhead,
# so the fix is fewer, larger transactions. L1 stays inside trap 3's budget:
# 2 fifos x depth 2 x 4096 x 2 B = 32 KB of 64 KB.  -> tasks/0026
POLY_TILES = {1024: "gelu_poly_bf16", 4096: "gelu_poly_bf16_4k"}
# Same kernel, one line different: aie::set_rounding(conv_even). The AIE
# default is `floor` -- a systematic downward bias on every bf16 store, never
# set by us until tasks/0044 went looking. Separate FILE, not a flag, because
# an edit-in-place A/B is the exact shape that has served stale builds here.
POLY_RNE_TILES = {1024: "gelu_poly_rne_bf16", 4096: "gelu_poly_rne_bf16_4k"}

# Speed probe, numerically WRONG on purpose: identical structure with the Horner
# chain cut from 8 steps to 2. Never shipped; use_ours="probe2" selects it.
PROBE_SYMBOL = "gelu_probe_deg2_bf16"


def ext_kernel(symbol, filename, tile=TILE):
    """Build an ExternalFunction the same way IRON builds its own."""
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    # OUR kernels dir FIRST: aie_kernels ships its own softmax.cc, and a
    # sibling #include (kernels/*_rne.cc pull in the impl they wrap) would
    # otherwise silently resolve to the upstream file -- which compiles and
    # then fails with "undeclared identifier". This is the include-order trap
    # CLAUDE.md records from the m7-unified design, hit again in tasks/0044.
    include.insert(0, str(HERE / "kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    tile_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    return ExternalFunction(
        symbol,
        source_file=str(HERE / "kernels" / filename),
        arg_types=[tile_ty, tile_ty],
        include_dirs=include,
    )


def our_gelu():
    """Our own kernel: same tanh polynomial, fp32 intermediates.

    Built the same way IRON builds its own -- ExternalFunction over a .cc
    compiled by Peano. Include dirs are lifted from IRON's helper so the
    aie_api and aie_kernels headers resolve exactly as they do for the
    shipped kernels.
    """
    return ext_kernel("gelu_fp32_bf16", "gelu_fp32.cc")


def _build_design(dev, n_elem, n_cols, use_ours, tile=TILE):
    n_rows = 4
    n_cores = n_rows * n_cols
    n_tiles = n_elem // tile
    assert n_elem % tile == 0, f"{n_elem} is not a multiple of {tile}"
    assert n_tiles % n_cores == 0, f"{n_tiles} tiles do not spread over {n_cores} cores"
    per_core = n_tiles // n_cores

    tile_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    buf_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]
    if use_ours == "ours":
        gelu_k = our_gelu()
    elif use_ours == "poly":
        if tile not in POLY_TILES:
            raise ValueError(f"no gelu_poly entry point for tile {tile}; "
                             f"have {sorted(POLY_TILES)}")
        gelu_k = ext_kernel(POLY_TILES[tile], "gelu_poly.cc", tile)
    elif use_ours == "polyrne":
        if tile not in POLY_RNE_TILES:
            raise ValueError(f"no gelu_poly_rne entry point for tile {tile}; "
                             f"have {sorted(POLY_RNE_TILES)}")
        gelu_k = ext_kernel(POLY_RNE_TILES[tile], "gelu_poly_rne.cc", tile)
    elif use_ours == "probe2":
        gelu_k = ext_kernel(PROBE_SYMBOL, "gelu_poly.cc", tile)
    elif use_ours == "control":
        # IRON's own source, compiled through OUR ExternalFunction path.
        # Isolates "is our C++ wrong" from "is our build setup wrong".
        gelu_k = ext_kernel("gelu_ctrl_bf16", "gelu_control.cc")
    else:
        gelu_k = kernels.gelu(tile)

    # ONE shim stream per COLUMN, split across the four cores by the mem tile.
    #
    # The first version gave every core its own rt.fill/rt.drain -- 4 per column
    # -- and that is why the design stopped at 2 columns: at 4 it needs 16 shim
    # channels each way and aiecc refuses with `no ShimNOCTile has sufficient
    # DMA capacity`. That was read as a hardware wall. It is not: the GEMM in
    # this same repo runs at 8 columns by routing L3->L2->L1, and ObjectFifo
    # has exactly the operators for it. .split() fans one L2 buffer out to the
    # four cores, .join() collects them back.
    #
    # Shim cost drops from 4 per column to 1, so 8 columns needs 8 channels each
    # way instead of 32.
    #
    # This matters because the eltwise kernels are compute bound PER CORE (the
    # degree probe in tasks/0026), and compute-bound work scales with cores:
    # 1 -> 2 columns measured 1.98x on GELU, 1.99x on LayerNorm, 1.97x on
    # softmax. Nothing about that stops at 2.
    def core_fn(a, c, gelu):
        for _ in range_(per_core):
            ea = a.acquire(1)
            ec = c.acquire(1)
            gelu(ea, ec)
            a.release(1)
            c.release(1)

    l2_ty = np.ndarray[(n_rows * tile,), np.dtype[bfloat16]]
    offsets = [tile * r for r in range(n_rows)]

    workers = []
    in_l3l2, out_l2l3 = [], []
    for c in range(n_cols):
        fin = ObjectFifo(l2_ty, name=f"gin_{c}", depth=2)
        fout = ObjectFifo(l2_ty, name=f"gout_{c}", depth=2)
        in_l3l2.append(fin)
        out_l2l3.append(fout)
        ins = fin.cons().split(
            offsets, obj_types=[tile_ty] * n_rows,
            names=[f"gin_{c}_{r}" for r in range(n_rows)])
        outs = fout.prod().join(
            offsets, obj_types=[tile_ty] * n_rows,
            names=[f"gout_{c}_{r}" for r in range(n_rows)],
            depths=[2] * n_rows)
        for r in range(n_rows):
            # split()/join() hand back ObjectFifos; a Worker takes handles.
            #
            # Pinned, not placed. With AnyComputeTile the placer failed at 4 and
            # 8 columns with "no ShimNOCTile has sufficient DMA capacity ... near
            # centroid column 0" -- and its own hint says to rebalance so the
            # centroid lands elsewhere. The layout here is perfectly regular
            # (column c, rows 2..5), so there is nothing for a heuristic to
            # discover: state it. Row 0 is the shim and row 1 the mem tile on
            # npu2, so compute starts at row 2.
            workers.append(Worker(core_fn,
                                  [ins[r].cons(), outs[r].prod(), gelu_k],
                                  tile=Tile(c, r + 2),
                                  stack_size=0x2000))

    # Column c owns one contiguous slice; within it the mem tile hands core r
    # every fourth tile. Elementwise, so any bijection will do, and the join
    # puts the output back at the offsets the split took it from.
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
               use_ours: CompileTime[str] = "iron",
               tile: CompileTime[int] = TILE):
    return _build_design(iron.get_current_device(), n_elem, n_cols, use_ours,
                         tile)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"))
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--kernel",
                    choices=["iron", "ours", "control", "poly", "polyrne"],
                    default="poly")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    taps, _ = load(Path(args.goldens) / "minilm_l6_s64_taps.safetensors")

    x_f32 = taps["L0.ffn_up"].reshape(-1)          # pre-activation, real data
    want = taps["L0.gelu"].reshape(-1).astype(np.float64)   # exact erf, fp32
    n_elem = x_f32.size

    print(f"GELU on the array [{args.kernel}]: {n_elem:,} elements = "
          f"{n_elem // TILE} tiles of {TILE}, {args.cols} cols x 4 rows")
    print("  input  = golden L0.ffn_up (real pre-activations)")
    print("  target = golden L0.gelu   (exact erf, fp32)\n")

    x16 = x_f32.astype(bfloat16)
    X = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    Y = iron.zeros(n_elem, dtype=bfloat16, device="npu")
    X[:] = x16                                     # __setitem__ syncs to device
    assert np.array_equal(X.numpy(), x16), "X did not reach the device"

    gelu_array(X, Y, n_elem=n_elem, n_cols=args.cols,
               use_ours=args.kernel)
    got = Y.numpy().astype(np.float64)

    # Three references, so a failure says which layer of approximation caused it.
    import math
    xf = x16.astype(np.float32)
    tanh_ref = (0.5 * xf * (1.0 + np.tanh(math.sqrt(2 / math.pi)
                                          * (xf + 0.044715 * xf ** 3)))).astype(np.float64)
    rel = lambda g, w: float(np.linalg.norm(g - w) / np.linalg.norm(w))

    r_golden = rel(got, want)
    r_tanh = rel(got, tanh_ref)
    print(f"  {'comparison':<44} {'rel_fro':>11}")
    print(f"  {'NPU vs exact-erf golden (what we need)':<44} {r_golden:>11.3e}")
    print(f"  {'NPU vs fp32 tanh-GELU (is the LUT faithful?)':<44} {r_tanh:>11.3e}")
    print(f"  {'fp32 tanh-GELU vs golden (formula cost alone)':<44} "
          f"{rel(tanh_ref, want):>11.3e}")

    # When we run our own polynomial, compare the hardware against the CPU
    # model of the SAME polynomial. That separates "the design is imperfect"
    # from "the hardware arithmetic differs from numpy fp32".
    if args.kernel in ("poly", "polyrne"):
        import json as _json
        d = _json.loads((ARTIFACTS / "gelu_poly.json").read_text(encoding="utf-8"))
        coef, R = d["coefficients_highest_first"], np.float32(d["clamp_R"])
        u = np.minimum(np.abs(xf), R)
        acc = np.full_like(u, np.float32(coef[0]))
        for c in coef[1:]:
            acc = acc * u + np.float32(c)
        cpu_poly = (np.maximum(xf, np.float32(0.0)) + acc).astype(np.float64)
        print(f"  {'NPU vs CPU model of the same polynomial':<44} "
              f"{rel(got, cpu_poly):>11.3e}")
        print(f"  {'CPU model vs golden (design limit)':<44} "
              f"{rel(cpu_poly, want):>11.3e}")

    nz = want != 0
    max_rel = float(np.abs((got[nz] - want[nz]) / want[nz]).max())
    print(f"\n  max abs error            {np.abs(got - want).max():.3e}")
    print(f"  max pointwise rel error  {max_rel:.3e}  "
          f"(IRON suggests rtol 0.128 for its LUT kernels)")

    # The bar: bf16 output rounding alone costs 1.69e-03 on this data, so a
    # faithful kernel lands near there. 5e-3 catches a bad LUT without failing
    # on the format.
    TOL = 5e-3
    ok = r_golden <= TOL
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or ARTIFACTS / f"gelu_kernel_{args.kernel}.json")
    out_path.write_text(json.dumps({
        "kind": "hardware measurement", "op": "gelu", "kernel": args.kernel, "n_elem": int(n_elem),
        "tile": TILE, "cols": args.cols,
        "rel_fro_vs_exact_erf_golden": r_golden,
        "rel_fro_vs_fp32_tanh": r_tanh,
        "max_pointwise_rel": max_rel,
        "tolerance": TOL, "pass": bool(ok),
    }, indent=2), encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'} -- tolerance {TOL:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
