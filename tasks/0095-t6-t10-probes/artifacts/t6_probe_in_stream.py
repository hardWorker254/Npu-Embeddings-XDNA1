# T6 probe -- does aie_stream=(1, 0) on the CONSUMER (core) side of an input
# fifo remove the core's input DMA channel? Compile-only, no NPU execution.
# Based on experiments/m5-eltwise/exp2_probe.py, minimally modified.
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

HERE = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-eltwise")
N = 1024

MODE = sys.argv[1] if len(sys.argv) > 1 else "off"  # "off" or "instream"


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
    kwargs = {}
    if MODE == "instream":
        kwargs["aie_stream"] = (1, 0)  # end=1 -> consumer (the core) is wire-only
    fin = ObjectFifo(ty, name="ein", depth=2, **kwargs)
    fout = ObjectFifo(ty, name="eout", depth=2)

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
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)
spec = probe.specialize()
spec.compile(xclbin_path=str(out_dir / "probe.xclbin"), inst_path=str(out_dir / "probe.insts.bin"))
print("BUILD OK:", MODE, "->", out_dir)
