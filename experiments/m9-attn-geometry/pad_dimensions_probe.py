# NpuEmbeddings -- T38 (tasks/0114): does mem-tile `pad_dimensions` actually
# route and work at ATTENTION's geometry, on all 8 columns?
#
# WHY. tasks/0043 established a structural wall: attention's per-head GEMM is
# [64,64] x [64,64], the whole-array design requires N % (n * cols) == 0, and
# the bf16 microkernel requires n % 16 == 0. So n*cols must divide 64 with
# n >= 16, hence cols <= 4 -- a design that can express attention can use at
# most HALF the array. tasks/0097 then measured what that costs: 2.229x on the
# projections, of which 1.729x is the column halving alone.
#
# research/notes/0007 §1.1 proposes removing it: N=64 is not given. AIE-ML DMAs
# apply constant padding at the buffer-descriptor level, so the per-column
# 8-wide N slice can be padded 8 -> 16 IN THE MEM TILE, with no host-side
# buffer and no change to the weights. Then n=16, cols=8 is legal.
#
# Two hardware facts support it and neither is a design:
#   - xaie2pgbl_reginit.c:1667 gives Aie2PMemTileDmaMod .Padding = AVAILABLE,
#     while the compute tile (1905) and shim (2158) do not have it.
#   - AIEDialect.cpp's DMABDOp::verify() restricts pad_dimensions to memtiles,
#     requires an n-d access pattern, and requires the inner-most pad to land
#     on 32-bit word boundaries.
#
# WHAT IS NOT KNOWN, and what this probe answers: whether it ROUTES AND WORKS
# in an IRON design at this geometry and this width. T38 says it is "worth an
# actual build rather than an argument". This is that build.
#
# TWO STAGES, because they fail differently:
#
#   --stage transport (default)
#       8 columns, each: shim -> mem tile (PAD innermost 8 -> 16) -> shim.
#       The drain takes 8 of every 16 with a strided tap -- note 0007 predicts
#       "that is where a first attempt will fail", because padding is
#       one-directional and the output side has to undo it.
#       PASS = every column's 8 real values round-trip exactly, AND a
#       full-width drain shows the pad value where the padding should be.
#
#   --stage compute
#       1 column: shim -> mem tile (PAD 8 -> 16) -> CORE -> mem tile -> shim.
#       Proves a compute tile can consume a padded stream at all -- the pad is
#       created upstream of it, and the verifier forbids padding ON a compute
#       tile, so "can a core read a padded buffer" is a separate question from
#       "can a mem tile make one".
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage: python pad_dimensions_probe.py [--stage transport|compute] [--cols 8]

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aie.iron as iron                                  # noqa: E402
from aie.iron import (                                    # noqa: E402
    CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.device import Tile, from_name               # noqa: E402
from aie.helpers.taplib import TensorAccessPattern        # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

# Attention's real geometry, from tasks/0043: per-head [64,64] x [64,64], so
# N=64 spread over 8 columns is 8 per column. 16 is the microkernel's minimum.
ROWS = 64            # the K dimension of the slice being moved
N_REAL = 8           # what a column actually owns
N_PAD = 16           # what the microkernel needs
PAD_VALUE = 0        # zero columns of B give exactly-zero columns of C


def purge(markers):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers):
            shutil.rmtree(d)
            n += 1
    print(f"  purged {n} cache candidate(s)")


# --------------------------------------------------------------------------
# Stage 1: transport. 8 columns, pad on the mem tile, un-pad on the drain.
# --------------------------------------------------------------------------
def build_transport(dev, cols, mode):
    """mode="pad": shim -> memtile(PAD 8->16) -> shim, drained 16 wide.
       mode="unpad": shim(16 wide) -> memtile(strided read, 8 of every 16)
                     -> shim, drained 8 wide.

    They are SEPARATE designs on purpose. The first attempt at this probe put
    both on one hop -- pad 8->16 on the mem tile, then try to take 8 of every
    16 back with a strided tap on the host drain -- and it MISMATCHED at
    2.121e+00. That failure is the one note 0007 §1.1 predicted ("the output
    side needs a strided dims_to_stream that takes 8 of every 16 ... that is
    where a first attempt will fail"), and the reason is worth keeping: a
    shim DMA consumes the stream in order and cannot skip, so the drain tap
    describes where bytes LAND in host memory, not which stream elements to
    keep. Selecting 8 of 16 is a strided READ on the mem tile, a different BD
    on a different hop -- which is exactly the topology the real design has
    (pad on the way IN to the cores, un-pad on the way OUT of them).
    """
    real_ty = np.ndarray[(ROWS, N_REAL), np.dtype[bfloat16]]
    pad_ty = np.ndarray[(ROWS, N_PAD), np.dtype[bfloat16]]

    if mode == "pad":
        in_n, out_n = N_REAL, N_PAD
    else:
        in_n, out_n = N_PAD, N_REAL

    in_ty = np.ndarray[(cols * ROWS * in_n,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(cols * ROWS * out_n,), np.dtype[bfloat16]]

    ins, outs = [], []
    for c in range(cols):
        if mode == "pad":
            f_in = ObjectFifo(real_ty, name=f"in{c}", depth=2)
            # THE PROBE. The pad happens here, on the mem tile's MM2S channel:
            # innermost dimension 8 -> 16, pad AFTER the real data. The n-d
            # access pattern is required by the verifier -- padding without one
            # is rejected with "Padding requires n-d data layouts".
            f_out = f_in.cons().forward(
                obj_type=pad_ty,
                dims_to_stream=[(ROWS, N_REAL), (N_REAL, 1)],
                pad_dimensions=[(0, 0), (0, N_PAD - N_REAL)],
                pad_value=PAD_VALUE, name=f"out{c}", depth=2)
        else:
            f_in = ObjectFifo(pad_ty, name=f"in{c}", depth=2)
            # THE UN-PAD, and it is an ORDINARY access pattern: read 8 of every
            # 16 out of the mem tile's 16-wide buffer. No padding feature
            # involved -- note 0007 says so, and this is the check.
            f_out = f_in.cons().forward(
                obj_type=real_ty,
                dims_to_stream=[(ROWS, N_PAD), (N_REAL, 1)],
                name=f"out{c}", depth=2)
        ins.append(f_in)
        outs.append(f_out)

    def sequence(x, y, *handles):
        tg = TaskGroup()
        prods, conss = handles[:cols], handles[cols:]
        for c in range(cols):
            src = TensorAccessPattern(
                (1, cols * ROWS * in_n), offset=c * ROWS * in_n,
                sizes=[1, 1, ROWS, in_n], strides=[0, 0, in_n, 1])
            dst = TensorAccessPattern(
                (1, cols * ROWS * out_n), offset=c * ROWS * out_n,
                sizes=[1, 1, ROWS, out_n], strides=[0, 0, out_n, 1])
            prods[c].fill(x, tap=src, group=tg)
            conss[c].drain(y, tap=dst, wait=True, group=tg)
        tg.finish()

    args = [in_ty, out_ty]
    args += [f.prod(tile=Tile(c, 0)) for c, f in enumerate(ins)]
    args += [f.cons(tile=Tile(c, 0)) for c, f in enumerate(outs)]
    return Program(dev, Runtime(sequence, args)).resolve_program()


# --------------------------------------------------------------------------
# Stage 2: can a COMPUTE tile consume a padded stream?
# --------------------------------------------------------------------------
def build_compute(dev):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config as _cfg

    inc = _include_dirs()
    inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels"))
    inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels" / _detect_arch()))

    n_elems = ROWS * N_PAD
    scale = ExternalFunction(
        f"scale_bf16_{n_elems}",
        source_file=str(HERE / "kernels" / "scale_bf16.cc"),
        arg_types=[np.ndarray[(n_elems,), np.dtype[bfloat16]]] * 2,
        include_dirs=inc,
    )

    small = np.ndarray[(ROWS, N_REAL), np.dtype[bfloat16]]
    big = np.ndarray[(n_elems,), np.dtype[bfloat16]]

    in_ty = np.ndarray[(ROWS * N_REAL,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(n_elems,), np.dtype[bfloat16]]

    f_in = ObjectFifo(small, name="in", depth=2)
    f_pad = f_in.cons().forward(
        obj_type=big,
        dims_to_stream=[(ROWS, N_REAL), (N_REAL, 1)],
        pad_dimensions=[(0, 0), (0, N_PAD - N_REAL)],
        pad_value=PAD_VALUE, name="padded", depth=2)
    f_y = ObjectFifo(big, name="y", depth=2)
    f_pipe = f_y.cons().forward(obj_type=big, name="y_pipe", depth=2)

    def core_fn(a_in, y_out, k):
        a = a_in.acquire(1)
        y = y_out.acquire(1)
        k(a, y)
        a_in.release(1)
        y_out.release(1)

    w = Worker(core_fn, [f_pad.cons(), f_y.prod(), scale],
               tile=Tile(0, 2), stack_size=0x800)

    def sequence(x, y, p, c):
        tg = TaskGroup()
        p.fill(x, group=tg)
        c.drain(y, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, f_in.prod(), f_pipe.cons()])
    return Program(dev, rt, workers=[w]).resolve_program()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pad", "unpad", "compute"],
                    default="pad")
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    # Without this IRON silently compiles for NPU1. research/notes/0002.
    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(38)

    if args.stage in ("pad", "unpad"):
        cols = args.cols
        pad_mode = args.stage == "pad"
        in_n = N_REAL if pad_mode else N_PAD
        out_n = N_PAD if pad_mode else N_REAL
        purge([f"aie.runtime_sequence(%arg0: memref<{cols * ROWS * in_n}xbf16>, "
               f"%arg1: memref<{cols * ROWS * out_n}xbf16>)"])

        # n_cols and mode MUST be CompileTime kwargs, not closure variables:
        # CLAUDE.md trap 7d -- iron.jit's cache key derives from the call's
        # arguments plus (device, full_elf) and never inspects the generator's
        # globals or closure. Hit live on the first run of this probe: a
        # second variant was served the FIRST variant's binary and died with
        # "Tensor argument 'Y' has 512 elements but the kernel was compiled
        # for 1024". As kwargs they are in the key and each variant is its own
        # build.
        @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
        def pad_transport(X: In, Y: Out, *,
                          n_cols: CompileTime[int] = 8,
                          do_pad: CompileTime[int] = 1):
            return build_transport(iron.get_current_device(), n_cols,
                                   "pad" if do_pad else "unpad")

        x_np = (rng.standard_normal((cols, ROWS, in_n)) * 0.5).astype(bfloat16)
        X = iron.zeros((cols * ROWS * in_n,), dtype=bfloat16, device="npu")
        Y = iron.zeros((cols * ROWS * out_n,), dtype=bfloat16, device="npu")
        X[:] = x_np.reshape(-1)
        assert np.array_equal(X.numpy(), x_np.reshape(-1)),             "X did not reach the device"

        pad_transport(X, Y, n_cols=cols, do_pad=1 if pad_mode else 0)
        got = Y.numpy().reshape(cols, ROWS, out_n).astype(np.float32)
        src = x_np.astype(np.float32)

        if pad_mode:
            real, pad = got[:, :, :N_REAL], got[:, :, N_REAL:]
            ok_real = np.array_equal(real, src)
            ok_pad = bool(np.all(pad == PAD_VALUE))
            print(f"  PAD    cols={cols}: real {'MATCH' if ok_real else 'MISMATCH'}, "
                  f"padded half all-{PAD_VALUE} {'YES' if ok_pad else 'NO'} "
                  f"(max |pad| = {np.abs(pad).max():.3e})")
            return 0 if (ok_real and ok_pad) else 1

        want = src[:, :, :N_REAL]           # 8 of every 16, kept in order
        ok = np.array_equal(got, want)
        print(f"  UNPAD  cols={cols}: {'EXACT' if ok else 'MISMATCH'}  "
              f"max|diff| = {float(np.abs(got - want).max()):.3e}")
        return 0 if ok else 1

    # ---- compute stage
    purge([f"aie.runtime_sequence(%arg0: memref<{ROWS * N_REAL}xbf16>, "
           f"%arg1: memref<{ROWS * N_PAD}xbf16>)"])

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def pad_compute(X: In, Y: Out):
        return build_compute(iron.get_current_device())

    x_np = (rng.standard_normal((ROWS, N_REAL)) * 0.5).astype(bfloat16)
    X = iron.zeros((ROWS * N_REAL,), dtype=bfloat16, device="npu")
    Y = iron.zeros((ROWS * N_PAD,), dtype=bfloat16, device="npu")
    X[:] = x_np.reshape(-1)
    assert np.array_equal(X.numpy(), x_np.reshape(-1)), \
        "X did not reach the device"

    pad_compute(X, Y)
    got = Y.numpy().reshape(ROWS, N_PAD).astype(np.float32)
    want = np.zeros((ROWS, N_PAD), np.float32)
    want[:, :N_REAL] = x_np.astype(np.float32) * 2.0   # the kernel doubles

    ok = np.array_equal(got, want)
    print(f"  compute over a padded stream: "
          f"{'EXACT' if ok else 'MISMATCH'}  max|diff| = "
          f"{float(np.abs(got - want).max()):.3e}")
    print(f"    padded half after compute: max|.| = "
          f"{float(np.abs(got[:, N_REAL:]).max()):.3e} "
          f"(2 x {PAD_VALUE} must still be {2 * PAD_VALUE})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
