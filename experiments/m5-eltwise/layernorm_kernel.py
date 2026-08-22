# NpuEmbeddings -- LayerNorm on the array, validated against the M3 goldens.
# SPDX-License-Identifier: Apache-2.0
#
# The first op with a REDUCTION. Everything before this was elementwise (GELU)
# or a GEMM; LayerNorm has to sum across a row before it can produce any output,
# which is a different shape of problem for a dataflow array.
#
# The shipped aie_kernels/aie2p/layer_norm.cc is unusable here for reasons
# visible in its source -- hardcoded gamma=1/beta=0, eps=1e-5 where MiniLM needs
# 1e-12, and the unstable one-pass variance formula. See kernels/layernorm.cc.
#
# WHAT IS ACTUALLY IN DOUBT
# -------------------------
# `aie::invsqrt`. tasks/0015 found `aie::tanh` accurate to only ~1%, and closed
# by warning that invsqrt and exp2 should be assumed equally coarse until
# measured. LayerNorm divides by it, so if it is 1% off the output is 1% off.
# This is the measurement.
#
# Three-way split, the same as the GELU work: comparing only against the golden
# would blame the wrong component.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m5-eltwise\layernorm_kernel.py
#   python experiments\m5-eltwise\layernorm_kernel.py --tap L0.ln2

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

from encoder import layernorm as layernorm_ref          # noqa: E402
from npue import Reader                                 # noqa: E402
from safetensors_io import load                         # noqa: E402

COLS = 384
ROWS_PER_CALL = 16          # must match the extern "C" wrapper; set by the L1 budget

# The inputs each LayerNorm site consumes, and the golden it must reproduce.
# `attn_proj + x` and `ffn_down + x` are the post-LN residual adds, done on the
# host here: they are elementwise and .npue keeps biases fp32 by design.
SITES = {
    "L0.ln1": dict(pre=("L0.attn_proj", "emb.ln"), g="layer.0.ln1.weight",
                   b="layer.0.ln1.bias"),
    "L0.ln2": dict(pre=("L0.ffn_down", "L0.ln1"), g="layer.0.ln2.weight",
                   b="layer.0.ln2.bias"),
}


def ln_kernel(symbol="layernorm_bf16", src="layernorm.cc"):
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    # OUR kernels dir FIRST -- kernels/layernorm_rne.cc #includes the impl it
    # wraps, and aie_kernels ships same-named sources. See tasks/0044.
    include.insert(0, str(HERE / "kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    in_ty = np.ndarray[(ROWS_PER_CALL * COLS,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(2 * COLS,), np.dtype[np.float32]]   # gamma then beta
    return ExternalFunction(
        symbol,
        source_file=str(HERE / "kernels" / src),
        arg_types=[in_ty, vec_ty, in_ty],
        include_dirs=include,
    )


def _build(dev, rows, n_cols, variant="base", stack=0xD00):
    n_cores = 4 * n_cols
    assert rows % (ROWS_PER_CALL * n_cores) == 0, \
        f"{rows} rows do not split into {ROWS_PER_CALL}-row blocks over {n_cores} cores"
    per_core = rows // (ROWS_PER_CALL * n_cores)

    blk = ROWS_PER_CALL * COLS
    tile_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(2 * COLS,), np.dtype[np.float32]]
    buf_ty = np.ndarray[(rows * COLS,), np.dtype[bfloat16]]
    # *_rne variants differ by exactly one line: set_rounding(conv_even).
    _SYM = {"base": ("layernorm_bf16", "layernorm.cc"),
            "il4": ("layernorm_il4_bf16", "layernorm.cc"),
            "rne": ("layernorm_rne_bf16", "layernorm_rne.cc"),
            "il4_rne": ("layernorm_il4_rne_bf16", "layernorm_rne.cc")}
    k = ln_kernel(*_SYM[variant])

    # ONE shim stream per column for data in, one for params, one for data out
    # -- split/join/broadcast through the mem tile, the pattern that took GELU
    # to 8 columns (tasks/0027). The first version opened THREE fifos per CORE:
    # 96 shim streams at 8 columns, and aiecc's refusal was misread in 0027 as
    # a GELU failure. The parameter fifo is the interesting one: 768 floats,
    # identical for every core, so it goes L3->L2 once per column and is
    # BROADCAST from the mem tile -- multiple workers consuming the same
    # ObjectFifo handle is exactly how the GEMM broadcasts B down a column.
    def core_fn(a, pm, c, ln):
        # The parameters are the same 768 floats for every row, so they are
        # acquired once and held -- filled once, never released, which makes
        # them behave like resident parameters rather than a stream.
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
        # Params hop L3 -> mem tile once; the forward gives the four cores a
        # single L2-resident copy to broadcast from.
        fp_l1 = fp.cons().forward(name=f"lnp_l1_{c}", depth=1)
        in_l3l2.append(fin); out_l2l3.append(fout); p_l3l2.append(fp)
        ins = fin.cons().split(
            offsets, obj_types=[tile_ty] * n_rows_grid,
            names=[f"lnin_{c}_{r}" for r in range(n_rows_grid)])
        outs = fout.prod().join(
            offsets, obj_types=[tile_ty] * n_rows_grid,
            names=[f"lnout_{c}_{r}" for r in range(n_rows_grid)],
            depths=[2] * n_rows_grid)
        for r in range(n_rows_grid):
            workers.append(Worker(core_fn,
                                  [ins[r].cons(), fp_l1.cons(),
                                   outs[r].prod(), k],
                                  tile=Tile(c, r + 2),
                                  stack_size=stack))

    data_taps = TensorTiler2D.simple_tiler((1, rows * COLS),
                                           (1, (rows * COLS) // n_cols))
    param_tap = TensorTiler2D.simple_tiler((1, 2 * COLS), (1, 2 * COLS))[0]

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
             stack: CompileTime[int] = 0xD00):
    return _build(iron.get_current_device(), rows, n_cols, variant, stack)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=sorted(SITES), default="L0.ln1")
    ap.add_argument("--npue", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"))
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--variant", default="base",
                    choices=["base", "il4", "rne", "il4_rne"])
    ap.add_argument("--stack", type=lambda v: int(v, 0), default=0xD00)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    taps, _ = load(Path(args.goldens) / "minilm_l6_s64_taps.safetensors")
    site = SITES[args.tap]

    # The residual add feeding this LayerNorm, on the host.
    a, b = site["pre"]
    x = (taps[a].reshape(-1, COLS).astype(np.float32)
         + taps[b].reshape(-1, COLS).astype(np.float32))
    want = taps[args.tap].reshape(-1, COLS).astype(np.float64)
    rows = x.shape[0]

    with Reader(args.npue) as r:
        gamma = r.tensor(site["g"]).astype(np.float32)
        beta = r.tensor(site["b"]).astype(np.float32)
        eps = r.config["layer_norm_eps"]

    print(f"LayerNorm on the array: {rows} rows x {COLS}, "
          f"{args.cols} cols x 4 rows, {ROWS_PER_CALL} rows/call")
    print(f"  site {args.tap}, gamma/beta from .npue, eps {eps:g}\n")

    x16 = x.astype(bfloat16)
    X = iron.zeros(rows * COLS, dtype=bfloat16, device="npu")
    P = iron.zeros(2 * COLS, dtype=np.float32, device="npu")
    Y = iron.zeros(rows * COLS, dtype=bfloat16, device="npu")
    X[:] = x16.reshape(-1)
    P[:] = np.concatenate([gamma, beta]).astype(np.float32)
    assert np.array_equal(X.numpy(), x16.reshape(-1)), "X did not reach the device"

    print(f"  variant={args.variant}  stack=0x{args.stack:X}")
    ln_array(X, P, Y, rows=rows, n_cols=args.cols, variant=args.variant,
             stack=args.stack)
    got = Y.numpy().astype(np.float64).reshape(rows, COLS)

    # Three references, so a failure says WHICH component caused it.
    cpu_f32 = layernorm_ref(x16.astype(np.float32), gamma, beta, eps).astype(np.float64)
    u = np.ascontiguousarray(cpu_f32, dtype=np.float32).view(np.uint32)
    cpu_bf16 = (((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
                ).astype(np.float64)

    rel = lambda g, w: float(np.linalg.norm(g - w) / np.linalg.norm(w))
    r_golden, r_cpu, r_floor = rel(got, want), rel(got, cpu_bf16), rel(cpu_bf16, want)
    print(f"  {'comparison':<46} {'rel_fro':>11}")
    print(f"  {'NPU vs golden ' + args.tap:<46} {r_golden:>11.3e}")
    print(f"  {'NPU vs CPU model of the same formula':<46} {r_cpu:>11.3e}")
    print(f"  {'CPU model in bf16 vs golden (the floor)':<46} {r_floor:>11.3e}")

    # Two kernels now diverge from their own numpy models by ~3.7e-03 (GELU
    # 3.886e-03 in tasks/0015, this one 3.659e-03). A shared cause is more
    # likely than two coincidences, and the shared step is the fp32 -> bf16
    # store. Round-to-nearest-even and truncation are distinguishable, so ask.
    def bf16_rne(v):
        u = np.ascontiguousarray(v, np.float32).view(np.uint32)
        return (((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32))

    def bf16_trunc(v):
        u = np.ascontiguousarray(v, np.float32).view(np.uint32)
        return ((u & 0xFFFF0000).view(np.float32))

    g32 = got.astype(np.float32)
    rne, trunc = bf16_rne(cpu_f32.astype(np.float32)), bf16_trunc(cpu_f32.astype(np.float32))
    both = rne == trunc                       # exactly representable: no information
    inf = ~both
    if inf.sum():
        m_rne = float((g32[inf] == rne[inf]).mean())
        m_trunc = float((g32[inf] == trunc[inf]).mean())
        print(f"\n  rounding of the fp32 -> bf16 store, on the "
              f"{inf.sum() / inf.size:.1%} of values where it is decidable:")
        print(f"    matches round-to-nearest-even : {m_rne:6.1%}")
        print(f"    matches truncation            : {m_trunc:6.1%}")
        bias = float(np.mean(np.abs(g32[inf]) - np.abs(cpu_f32.astype(np.float32)[inf])))
        print(f"    mean |NPU| - |exact|          : {bias:+.3e}  "
              f"(truncation biases toward zero)")

    TOL = 5e-3
    ok = r_golden <= TOL
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Keyed by VARIANT too -- see the note in softmax_kernel.py (tasks/0044).
    (ARTIFACTS / f"layernorm_{args.tap.replace('.', '_')}_{args.variant}.json").write_text(
        json.dumps({
            "kind": "hardware measurement", "op": "layernorm", "site": args.tap,
            "rows": int(rows), "cols": COLS, "eps": eps,
            "rel_fro_vs_golden": r_golden,
            "rel_fro_vs_cpu_model": r_cpu,
            "bf16_floor": r_floor,
            "tolerance": TOL, "pass": bool(ok),
        }, indent=2), encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'} -- tolerance {TOL:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
