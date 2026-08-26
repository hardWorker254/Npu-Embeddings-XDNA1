# NpuEmbeddings -- tasks/0083, T28: how many sources can ONE mem tile join,
# when L1 is deliberately not the binding constraint?
#
# WHY THIS EXISTS
# ---------------
# 0057 measured the wall with the compiler's own error text and concluded "a
# single JOIN cannot express the full 8-column ffn_down regather, full stop".
# 0062 then built the hierarchical 2-hop form and PASSED on hardware, and
# recorded that GROUP=4 was ruled out by L1 arithmetic -- 49,152 B for the
# one-shot gather alone -- while noting "a version that streams GROUP=4 in
# SMALLER sub-chunks might still fit -- not investigated".
#
# Re-running 0057's own cross_column_join_probe at SRC_COLS=0,1,2,3 today
# reproduces that exactly, and the failure is:
#
#   Y_out_buff_0      : 49152 bytes
#   C_mem_cons_buff_0 : 49152 bytes
#   error: 'aie.tile' op Basic sequential allocation failed.
#
# That is L1, not ports. The port budget was never reached, so "4 sources do
# not fit" has never actually been tested -- only "4 sources AT PRODUCTION
# TILE SIZE do not fit", which is a different claim and the one 0062 already
# knew.
#
# THIS PROBE separates them. Tiles are 512 floats (2 KB), so the gathered
# buffer at 8 sources is 16 KB and the consumer's pair is 32 KB -- half the
# L1 budget, with no production geometry anywhere near it. The ONLY thing
# that can fail is the topology.
#
# Sweep NPUE_N_SRC=2..8. What we are reading is not pass/fail but WHICH error:
#   * "requires N input/M output DMA channels, but only ..."  -> port budget
#   * "Basic sequential allocation failed"                    -> L1
#   * "Unable to find a legal routing"                        -> stream switch
# Three different walls that 0057's single data point could not distinguish.

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.device import Tile, from_name
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CACHE = Path.home() / ".npu" / "cache"

iron.set_current_device(from_name("npu2", n_cols=None))

# Deliberately tiny. TM*TN = 512 floats = 2 KB per source tile.
TM, TN = 32, 16
TILE = TM * TN
N_SRC = int(os.environ.get("NPUE_N_SRC", "4"))
DEST_COL = int(os.environ.get("NPUE_DEST_COL", "0"))
# Which physical columns the producers live in. Default spreads them across
# the array so the test is cross-column, as the real regather is.
SRC_COLS = [int(x) for x in os.environ.get(
    "NPUE_SRC_COLS", ",".join(str(i) for i in range(N_SRC))).split(",")]
assert len(SRC_COLS) == N_SRC, "NPUE_SRC_COLS must have NPUE_N_SRC entries"
GATHER = TILE * N_SRC


def purge():
    """Purge EVERY build of this probe, not just this N.

    THE MARKER MUST IDENTIFY THE FAMILY, NOT THE INSTANCE. The first version
    keyed on `memref<{GATHER}xf32>`, which is DIFFERENT for every N_SRC -- so
    sweeping N=2,3,4... never matched the previous build, `@iron.jit` decided
    it was a cache hit (it keys on the decorated function, and N_SRC is read
    inside build() where the decorator cannot see it), and every N silently ran
    the N=2 binary. It announced itself only because the host-side element
    count disagreed:

        Tensor argument 'X' has 1536 elements but the kernel was compiled
        for 1024 elements

    which is luck -- a sweep where the shapes happened to match would have
    reported N=2's result under N=8's label. This is the FOURTH instance of
    the marker-specificity fail-open class (tasks/0030, 0053, 0054).

    `probe_copy_512_f32` is the per-source kernel and appears in every N, so it
    identifies the family. Match on contents, never on mtime (trap 7c).
    """
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        if f"probe_copy_{TILE}_f32" in mlir.read_text(encoding="utf-8",
                                                      errors="ignore"):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def build():
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config as _cfg
    inc = _include_dirs()
    inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels"))
    inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels" / _detect_arch()))

    tile_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
    gather_ty = np.ndarray[(GATHER,), np.dtype[np.float32]]

    src_copy = ExternalFunction(
        f"probe_copy_{TILE}_f32",
        source_file=str(HERE.parent / "m5-eltwise" / "kernels" / "probe_copy.cc"),
        arg_types=[tile_ty, tile_ty], include_dirs=inc)
    gather_copy = ExternalFunction(
        f"probe_copy_{GATHER}_f32",
        source_file=str(HERE.parent / "m5-eltwise" / "kernels" / "probe_copy.cc"),
        arg_types=[gather_ty, gather_ty], include_dirs=inc)

    # One shim feed per producer column, exactly like production's per-column
    # A feed -- so each producer's mem tile carries real traffic of its own
    # and the join is not the only thing on it.
    X_ty = np.ndarray[(TILE * N_SRC,), np.dtype[np.float32]]
    Y_ty = np.ndarray[(GATHER,), np.dtype[np.float32]]

    x_shims, x_l2l1 = [], []
    for j in range(N_SRC):
        sh = ObjectFifo(tile_ty, name=f"X_L3L2_{j}", depth=2)
        fwd = sh.cons().forward(obj_type=tile_ty, name=f"X_fwd_{j}", depth=2,
                                tile=Tile(SRC_COLS[j], 1))
        x_shims.append(sh)
        x_l2l1.append(fwd)

    # THE THING UNDER TEST: an N_SRC-way join into ONE mem tile.
    C_mem = ObjectFifo(gather_ty, name="C_mem", depth=1)
    C_cols = C_mem.prod().join(
        [TILE * j for j in range(N_SRC)],
        obj_types=[tile_ty] * N_SRC,
        names=[f"C_src{j}" for j in range(N_SRC)],
        tile=Tile(DEST_COL, 1))

    Y_out = ObjectFifo(gather_ty, name="Y_out", depth=1)
    Y_pipe = Y_out.cons().forward(obj_type=gather_ty, name="Y_pipe", depth=2,
                                  tile=Tile(DEST_COL, 1))

    def src_fn(xin, cout, cp):
        a = xin.acquire(1)
        c = cout.acquire(1)
        cp(a, c)
        xin.release(1)
        cout.release(1)

    def gather_fn(gin, yout, cp):
        g = gin.acquire(1)
        y = yout.acquire(1)
        cp(g, y)
        gin.release(1)
        yout.release(1)

    workers = [
        Worker(src_fn, [x_l2l1[j].cons(), C_cols[j].prod(), src_copy],
               tile=Tile(SRC_COLS[j], 2), stack_size=0x800)
        for j in range(N_SRC)
    ]
    workers.append(Worker(gather_fn, [C_mem.cons(), Y_out.prod(), gather_copy],
                          tile=Tile(DEST_COL, 4), stack_size=0x800))

    # 0057's own bug, avoided the way 0057 fixed it: tile the FULL tensor and
    # let simple_tiler emit one correctly-offset tap per source, rather than
    # building N identical taps that all read offset 0.
    x_taps = TensorTiler2D.simple_tiler((N_SRC * TM, TN), (TM, TN))
    y_tap = TensorTiler2D.simple_tiler((N_SRC * TM, TN))[0]

    x_prods = [x_shims[j].prod(tile=Tile(SRC_COLS[j], 0)) for j in range(N_SRC)]
    y_cons = Y_pipe.cons(tile=Tile(DEST_COL, 0))

    def sequence(X, Y, x_hs, y_h):
        tg = TaskGroup()
        for j in range(N_SRC):
            x_hs[j].fill(X, tap=x_taps[j], group=tg)
        y_h.drain(Y, tap=y_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [X_ty, Y_ty, x_prods, y_cons])
    return Program(iron.get_current_device(), rt, workers).resolve_program()


# `--alloc-scheme=basic-sequential` is what 0054/0057/0062's probes all use;
# without it the allocator's default scheme reports a different (and less
# legible) failure for the same L1 exhaustion.
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def join_port_budget(X: In, Y: Out):
    return build()


def main():
    print(f"  sources {N_SRC} in columns {SRC_COLS} -> mem tile at column "
          f"{DEST_COL}")
    print(f"  tile {TILE} floats ({TILE*4} B), gathered {GATHER*4} B, "
          f"consumer pair {GATHER*8} B of 64512 -- L1 is NOT the constraint")
    print(f"  purged {purge()} cache candidate(s)")

    rng = np.random.default_rng(3)
    X_np = rng.standard_normal((N_SRC * TM, TN)).astype(np.float32)
    # Trap 6b: write through __setitem__, never through .numpy(), which is a
    # host-only view the device never sees.
    X = iron.zeros(TILE * N_SRC, dtype=np.float32, device="npu")
    X[:] = X_np.reshape(-1)
    Y = iron.zeros(GATHER, dtype=np.float32, device="npu")
    try:
        join_port_budget(X, Y)
    except Exception as e:
        msg = str(e)
        kind = ("PORT BUDGET" if "DMA channel" in msg
                else "L1" if "allocation failed" in msg
                else "ROUTING" if "legal routing" in msg
                else "other")
        print(f"  FAILED [{kind}]: {msg[:500]}")
        return 1
    got = Y.numpy().reshape(N_SRC * TM, TN)
    # Reference is what we INTENDED, never a device read-back (trap 6c).
    ok = np.array_equal(got, X_np)
    print(f"  rel_fro {np.linalg.norm(got - X_np) / np.linalg.norm(X_np):.3e}  "
          f"{'PASS (bit-exact)' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
