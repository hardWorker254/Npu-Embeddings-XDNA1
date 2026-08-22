# NpuEmbeddings -- what does exp2_poly actually return?
#
# The softmax kernel produced rows summing to zero, which requires
# exp2_poly(0) != 1. Measure it directly rather than reasoning about which of
# to_fixed / upshift / vector_cast differs from the assumption.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
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
N = 1024


def kern():
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config
    inc = _include_dirs()
    inc.append(str(Path(config.cxx_header_path()) / "aie_kernels"))
    inc.append(str(Path(config.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    inc.append(str(HERE / "kernels"))
    ty = np.ndarray[(N,), np.dtype[bfloat16]]
    return ExternalFunction("exp2_probe_bf16",
                            source_file=str(HERE / "kernels" / "exp2_probe.cc"),
                            arg_types=[ty, ty], include_dirs=inc)


def _build(dev):
    ty = np.ndarray[(N,), np.dtype[bfloat16]]
    k = kern()
    fin, fout = ObjectFifo(ty, name="ein", depth=2), ObjectFifo(ty, name="eout", depth=2)

    def core_fn(a, c, f):
        ea = a.acquire(1); ec = c.acquire(1)
        f(ea, ec)
        a.release(1); c.release(1)

    w = Worker(core_fn, [fin.cons(), fout.prod(), k], stack_size=0xD00)
    tap = TensorTiler2D.simple_tiler((1, N), (1, N))[0]

    def sequence(X, Y, fin_prod, fout_cons):
        tg = TaskGroup()
        fin_prod.fill(X, tap=tap, group=tg)
        fout_cons.drain(Y, tap=tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [ty, ty, fin.prod(), fout.cons()])
    return Program(dev, rt, workers=[w]).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(X: In, Y: Out):
    return _build(iron.get_current_device())


iron.set_current_device(from_name("npu2", n_cols=None))
xs = np.linspace(-120.0, 0.0, N).astype(np.float32)
x16 = xs.astype(bfloat16)
X = iron.zeros(N, dtype=bfloat16, device="npu")
Y = iron.zeros(N, dtype=bfloat16, device="npu")
X[:] = x16
probe(X, Y)
got = Y.numpy().astype(np.float64)
want = np.exp2(x16.astype(np.float64))

print(f"{'x':>10} {'exp2_poly':>14} {'numpy 2^x':>14} {'ratio':>10}")
for xv in (0.0, -0.5, -1.0, -2.0, -10.0, -40.0, -80.0, -119.0):
    i = int(np.abs(x16.astype(np.float64) - xv).argmin())
    r = got[i] / want[i] if want[i] else float("nan")
    print(f"{x16.astype(np.float64)[i]:>10.3f} {got[i]:>14.6e} {want[i]:>14.6e} {r:>10.4f}")

ok = want > 0
rel = np.abs(got[ok] - want[ok]) / want[ok]
print(f"\nmax relative error over x in [-120, 0]: {rel.max():.3e}")
print(f"median relative error                 : {np.median(rel):.3e}")
