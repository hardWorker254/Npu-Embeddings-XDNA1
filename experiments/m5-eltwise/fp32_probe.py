# NpuEmbeddings -- measure the mantissa width of aie::vector<float> arithmetic.
#
# See kernels/fp32_probe.cc for the method. Computes (1.0f + eps) - 1.0f on the
# array for eps = 2^-1 .. 2^-24 and reports the smallest eps that survives.
#
#   IEEE fp32 -> survives to 2^-23  (24-bit mantissa)
#   bf16      -> dies below 2^-8    ( 8-bit mantissa)
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m5-eltwise\fp32_probe.py

from __future__ import annotations

import json
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
ARTIFACTS = HERE / "artifacts"
TILE = 1024
N_ELEM = 1024


def probe_kernel(symbol="fp32_probe_bf16"):
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config

    include = _include_dirs()
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    include.append(str(Path(config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    return ExternalFunction(
        symbol,
        source_file=str(HERE / "kernels" / "fp32_probe.cc"),
        arg_types=[tile_ty, tile_ty],
        include_dirs=include,
    )


def _build(dev, n_elem, symbol):
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    buf_ty = np.ndarray[(n_elem,), np.dtype[bfloat16]]
    k = probe_kernel(symbol)
    fin = ObjectFifo(tile_ty, name="pin", depth=2)
    fout = ObjectFifo(tile_ty, name="pout", depth=2)

    def core_fn(a, c, f):
        for _ in range_(n_elem // TILE):
            ea = a.acquire(1)
            ec = c.acquire(1)
            f(ea, ec)
            a.release(1)
            c.release(1)

    w = Worker(core_fn, [fin.cons(), fout.prod(), k], stack_size=0xD00)
    taps = TensorTiler2D.simple_tiler((1, n_elem), (1, n_elem))

    def sequence(X, Y, fin_prod, fout_cons):
        tg = TaskGroup()
        fin_prod.fill(X, tap=taps[0], group=tg)
        fout_cons.drain(Y, tap=taps[0], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [buf_ty, buf_ty, fin.prod(), fout.cons()])
    return Program(dev, rt, workers=[w]).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe_array(X: In, Y: Out, *, n_elem: CompileTime[int],
                symbol: CompileTime[str] = "fp32_probe_bf16"):
    return _build(iron.get_current_device(), n_elem, symbol)


OPS = {"add": ("fp32_probe_bf16", "(1.0f + eps) - 1.0f"),
       "mul": ("fp32_probe_mul_bf16", "((1.0f + eps) * 1.0f) - 1.0f")}


def main() -> int:
    op = sys.argv[1] if len(sys.argv) > 1 else "add"
    symbol, formula = OPS[op]
    iron.set_current_device(from_name("npu2", n_cols=None))

    exps = np.arange(1, 25)                       # 2^-1 .. 2^-24
    eps = (2.0 ** -exps).astype(np.float32)
    x = np.zeros(N_ELEM, dtype=np.float32)
    x[: eps.size] = eps
    x16 = x.astype(bfloat16)
    # Powers of two are exact in bf16; if that ever stops being true the whole
    # experiment is void, so check rather than assume.
    assert np.array_equal(x16.astype(np.float32)[: eps.size], eps), \
        "eps values did not survive bf16 conversion"

    X = iron.zeros(N_ELEM, dtype=bfloat16, device="npu")
    Y = iron.zeros(N_ELEM, dtype=bfloat16, device="npu")
    X[:] = x16
    assert np.array_equal(X.numpy(), x16), "X did not reach the device"

    probe_array(X, Y, n_elem=N_ELEM, symbol=symbol)
    got = Y.numpy().astype(np.float32)[: eps.size]

    print("aie::vector<float> arithmetic:  out = (1.0f + eps) - 1.0f\n")
    print(f"  {'eps':>12} {'2^-n':>6} {'returned':>14} {'exact?':>8}")
    survived = []
    for e, ex, g in zip(eps, exps, got):
        ok = (g == e)
        if ok:
            survived.append(int(ex))
        print(f"  {e:>12.3e} {ex:>6} {g:>14.3e} {'yes' if ok else 'no':>8}")

    deepest = max(survived) if survived else 0
    mantissa = deepest + 1          # eps = 2^-n resolvable => n+1 significant bits
    print(f"\n  deepest eps that survived : 2^-{deepest}")
    print(f"  implied mantissa bits     : ~{mantissa}")
    print(f"  IEEE fp32 would give      : 24  (eps to 2^-23)")
    print(f"  bf16 would give           : 8   (eps to 2^-7)")

    verdict = ("fp32" if mantissa >= 20 else
               "bf16-class" if mantissa <= 12 else "intermediate")
    print(f"\n  VERDICT: aie::vector<float> carries ~{mantissa} mantissa bits "
          f"-> {verdict}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"fp32_probe_{op}.json").write_text(json.dumps({
        "kind": "hardware measurement",
        "op": op, "test": formula,
        "eps_exponents": [int(e) for e in exps],
        "returned": [float(g) for g in got],
        "deepest_surviving_exponent": deepest,
        "implied_mantissa_bits": mantissa,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
