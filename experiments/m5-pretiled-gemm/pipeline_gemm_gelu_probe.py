# NpuEmbeddings -- T28 Del B / B2 (tasks/0054): a genuine CROSS-CORE pipeline.
# SPDX-License-Identifier: Apache-2.0
#
# B1 (epilogue_gelu_survey.py) proved GEMM+GELU fusion WITHIN one core (the
# same core that computes A@B applies GELU in place before releasing the
# tile). That is real, but it is not the mechanism note 0005 SS4 / T28
# actually names as the big win: AMD's 15->3 dispatch reduction, STEEL's
# 22.8x and ARIES' adjacent-tile handoff are all about DIFFERENT cores
# (different silicon, different program-memory budgets) exchanging a tile
# THROUGH THE MEM TILE, never through host DRAM, inside ONE dispatch.
#
# This is that mechanism, built at the smallest scale that proves it rather
# than performs at production scale (a milestone-scale full ffn_up->GELU->
# ffn_down chain does not fit one session -- see the module-end note on why
# the natural next step, adding ffn_down as a THIRD stage, is a much harder
# problem and was NOT attempted here).
#
# ARCHITECTURE (one column, all 4 compute rows, ONE static design):
#
#   host A ----shim--> mem tile --split--> GEMM core (row 2) --\
#                                       \-> GEMM core (row 3) --+-- C_mem
#   host B ----shim--> mem tile --fwd (broadcast to both GEMM cores)
#                                                                |
#                                              mem tile (C_mem) -+
#                                               |          |
#                                       --split-+          |
#                                       |                  |
#                              GELU core (row 4)   GELU core (row 5)
#                                       |                  |
#                                       \--- join ---> Y_mem --shim--> host Y
#
# THE PROOF: the design's I/O CONTRACT has no C tensor at all. rt.sequence()
# takes exactly (A, B, Y) -- three buffers. The GEMM output tile is an
# ObjectFifo named C_mem that is filled by a .join() FROM the two GEMM cores
# and drained by a .split() TO the two GELU cores, both operations pinned to
# the SAME mem tile, with no rt.fill/rt.drain touching it at all. If this
# compiles and the numbers come out right, the intermediate genuinely never
# leaves the array -- provably, by the absence of a host-facing C in the
# design's own signature, not by inference from timing.
#
# Deliberately NOT reused from gemm_pretiled.py's _build_design: that
# function hardcodes n_aie_rows=4 (one shape per column) and has no notion of
# "some rows do X, other rows do Y" -- this needed a fresh, small design
# rather than a parameter added to a function already carrying six option
# flags. K is a single un-augmented k-block (TK=64) and N is a single tile
# (TN=48) so this probe measures ROUTING correctness, not GELU-fusion
# accuracy again -- B1 already covered that at production scale.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python pipeline_gemm_gelu_probe.py
#   python pipeline_gemm_gelu_probe.py --trace --out artifacts\pipeline_probe.json

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aie.iron as iron                                # noqa: E402
from aie.iron import (                                  # noqa: E402
    Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
    kernels, str_to_dtype,
)
from aie.iron.device import Tile, from_name             # noqa: E402
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D  # noqa: E402
from aie.utils.trace import TraceConfig                  # noqa: E402
from npue import tile_b, to_bf16_bits                    # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

TM, TK, TN = 64, 64, 48       # one MiniLM-shaped tile
N_GEMM_ROWS = 2                # physical rows 2, 3
N_GELU_ROWS = 2                # physical rows 4, 5
M = TM * N_GEMM_ROWS           # 128
K = TK                         # single k-block, no augmentation -- this
N = TN                         # probe is about ROUTING, not fusion accuracy
GELU_TILE = TM * TN            # 3072, matches gelu_epilogue_3072_f32_io


def markers_for():
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}xf32>)",
            "gelu_epilogue_3072_f32_io"]


def purge():
    n = 0
    markers = markers_for()
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers):
            shutil.rmtree(d)
            n += 1
    print(f"  purged {n} cache candidate(s)")


def _build_design(dev, trace_config=None, identity=False):
    matmul_kernel = kernels.mm(
        dim_m=TM, dim_k=TK, dim_n=TN,
        input_dtype=str_to_dtype("bf16"), output_dtype=str_to_dtype("f32"),
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config as _cfg
    _inc = _include_dirs()
    _inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels"))
    _inc.append(str(Path(_cfg.cxx_header_path()) / "aie_kernels" / _detect_arch()))
    gelu_io_kernel = ExternalFunction(
        "identity_copy_3072_f32" if identity else "gelu_epilogue_3072_f32_io",
        source_file=str(HERE.parent / "m5-eltwise" / "kernels" / "gelu_poly.cc"),
        arg_types=[np.ndarray[(GELU_TILE,), np.dtype[np.float32]]] * 2,
        include_dirs=_inc,
    )

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    A_l1_ty = np.ndarray[(TM, TK), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(TK, TN), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(TM, TN), np.dtype[np.float32]]
    gelu_flat_ty = np.ndarray[(GELU_TILE,), np.dtype[np.float32]]

    # --- A: shim -> mem tile -> split (one tile per GEMM row) ---
    # Same dims_to_stream reorder gemm_pretiled.py uses for A -- activations
    # are not pre-tiled offline, so the mem tile reorders into the MAC's
    # (r, s) sub-tile order on the way into L1.
    A_l2_ty = np.ndarray[(TM * TK * N_GEMM_ROWS,), np.dtype[bfloat16]]
    A_shim = ObjectFifo(A_l2_ty, name="A_L3L2", depth=2)
    a_offsets = [TM * TK * j for j in range(N_GEMM_ROWS)]
    a_dims = [[(TM // r, r * TK), (TK // s, s), (r, TK), (s, 1)]] * N_GEMM_ROWS
    A_rows = A_shim.cons().split(
        a_offsets, obj_types=[A_l1_ty] * N_GEMM_ROWS,
        names=[f"A_row{j}" for j in range(N_GEMM_ROWS)],
        dims_to_stream=a_dims)

    # --- B: shim -> mem tile -> forward, BROADCAST to both GEMM rows.
    # Pre-tiled offline (tile_b, the SAME function that packs .npue -- M4/M5
    # precedent) so the forward is a plain linear copy, no dims_to_stream.
    # Both GEMM-row workers are handed the SAME ObjectFifo object below --
    # multiple workers consuming one handle is the documented broadcast
    # pattern (note 0005 SS3, LayerNorm's gamma/beta).
    B_l2_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    B_shim = ObjectFifo(B_l2_ty, name="B_L3L2", depth=2)
    B_fwd = B_shim.cons().forward(obj_type=B_l1_ty, name="B_fwd", depth=2)

    # --- C: THE PROBE, take 2. First attempt tried GEMM-cores-join -> C_mem
    # -> split-to-GELU-cores on ONE ObjectFifo and the compiler refused:
    # "objectfifo cannot be in more than one ObjectFifoLinkOp" -- an
    # ObjectFifoLink (which is what join()/split()/forward() all create) may
    # use a given object as a link ENDPOINT exactly once, so chaining
    # join-then-split through a shared intermediate buffer is not
    # expressible (ObjectFifoLink itself also refuses N:M -- "may only have
    # > 1 of either sources or destinations, but not both", confirmed by
    # reading dataflow/objectfifo.py directly). RECORDED, not worked around
    # silently -- see the module-end note.
    #
    # Take 2 compiled (no join/split at all, a bare point-to-point ObjectFifo
    # per pairing) but HUNG at runtime (ERT_CMD_STATE_TIMEOUT) -- the
    # compiler accepted it and placed/routed *something*, but nothing in
    # this design's own rt.sequence() ever references the pipe, and neither
    # A's nor B's cores.cons() do either (those work fine with no
    # rt.sequence entry) -- the difference is A/B's mem-tile buffer is
    # PRIMED by an explicit rt.fill from a shim, which is what an
    # ObjectFifoLink (join/split/forward) apparently also anchors even
    # without a shim on either side. A bare ObjectFifo with both endpoints
    # on compute tiles and NO Link at all is not provably wired up.
    #
    # Take 3, which is what actually runs: give each pipe an EXPLICIT
    # mem-tile Link via forward() -- the SAME primitive B already uses
    # (shim-fed L2 buffer forwarded onward), just with the producer side
    # fed by a CORE instead of a shim. Each `C_out_j` fifo has exactly ONE
    # link (the forward, anchoring `C_out_j` as SRC), so this does not hit
    # take 1's "already linked" conflict -- there is no join in this
    # design, and each `C_out_j`/`C_in_j` pair is a fully independent
    # two-object chain.
    C_outs = [ObjectFifo(C_l1_ty, name=f"C_out_{j}", depth=2)
              for j in range(N_GEMM_ROWS)]
    C_pipes = [C_outs[j].cons().forward(obj_type=C_l1_ty, name=f"C_in_{j}",
                                        depth=2)
              for j in range(N_GEMM_ROWS)]

    # --- Y: GELU cores JOIN into Y_mem; Y_mem drains to L3 -- the FINAL
    # result, for verification. (A real ffn_down stage would read straight
    # from here instead, per the module docstring's "not attempted" note.)
    # dims_to_stream on the BASE object (not the join's subfifos) --
    # pipeline_diag_gemm_only.py needed this EXACT formula on its C join to
    # go from rel_fro 1.3-ish (wrong, finite -- the identity-copy diagnostic
    # reproduced this same magnitude here) to 3.888e-08. gemm_pretiled.py's
    # own C_l2l3_fifos carries it for the identical reason: multiple
    # producer tiles landing in one L2 buffer are NOT simply concatenated by
    # the join -- the mem tile's join DMA interleaves them, and this
    # transform is what the shim-facing drain needs to undo it back into
    # plain row-major.
    y_offsets = [GELU_TILE * j for j in range(N_GELU_ROWS)]
    Y_mem = ObjectFifo(
        np.ndarray[(GELU_TILE * N_GELU_ROWS,), np.dtype[np.float32]],
        name="Y_mem", depth=2,
        dims_to_stream=[(TM // r, r * TN), (r, t), (TN // t, r * t), (t, 1)])
    Y_rows = Y_mem.prod().join(
        y_offsets, obj_types=[gelu_flat_ty] * N_GELU_ROWS,
        names=[f"Y_row{j}" for j in range(N_GELU_ROWS)])

    def gemm_core_fn(in_a, in_b, out_c, zero, matmul):
        ea = out_c.acquire(1)
        zero(ea)
        a = in_a.acquire(1)
        b = in_b.acquire(1)
        matmul(a, b, ea)
        in_a.release(1)
        in_b.release(1)
        out_c.release(1)

    def gelu_core_fn(in_c, out_y, gelu_io):
        ec = in_c.acquire(1)
        ey = out_y.acquire(1)
        gelu_io(ec, ey)
        in_c.release(1)
        out_y.release(1)

    gemm_workers = [
        Worker(gemm_core_fn,
               [A_rows[j].cons(), B_fwd.cons(), C_outs[j].prod(),
                zero_kernel, matmul_kernel],
               tile=Tile(0, 2 + j), stack_size=0xD00,
               trace=1 if (trace_config is not None and j == 0) else None)
        for j in range(N_GEMM_ROWS)
    ]
    gelu_workers = [
        Worker(gelu_core_fn,
               [C_pipes[j].cons(), Y_rows[j].prod(), gelu_io_kernel],
               tile=Tile(0, 4 + j), stack_size=0x2000,
               trace=1 if (trace_config is not None and j == 0) else None)
        for j in range(N_GELU_ROWS)
    ]

    # A flat single-dim [size],[1] tap lowers to a huge repeat count (BD
    # field is [0:255] -- hit at 8191 on the first attempt), and a HAND-BUILT
    # 2D TensorAccessPattern([M,K],[K,1]) compiles cleanly but HANGS THE
    # HARDWARE at runtime (ERT_CMD_STATE_TIMEOUT) -- root-caused in
    # pipeline_diag_gemm_only.py (see that file's own history): the identical
    # full-tensor copy via TensorTiler2D.simple_tiler(tensor_dims)[0] (tile
    # size defaults to the whole tensor) compiles AND runs correctly
    # (rel_fro 3.888e-08). Something about the raw TensorAccessPattern
    # constructor's output differs from what the tiler helper emits in a way
    # that passed every static/verifier check yet produced an unrunnable
    # design -- not chased further this session (see the module-end note).
    # MORAL: use the tiler helpers, never hand-build a TensorAccessPattern
    # for a plain full-region copy, even though the API allows it.
    a_full_tap = TensorTiler2D.simple_tiler((M, K))[0]
    b_full_tap = TensorTiler2D.simple_tiler((K, N))[0]
    y_full_tap = TensorTiler2D.simple_tiler((M, N))[0]

    def sequence(A, B, Y, A_shim_prod, B_shim_prod, Y_mem_cons):
        tg = TaskGroup()
        A_shim_prod.fill(A, tap=a_full_tap, group=tg)
        B_shim_prod.fill(B, tap=b_full_tap, group=tg)
        Y_mem_cons.drain(Y, tap=y_full_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, Y_ty,
                            A_shim.prod(), B_shim.prod(), Y_mem.cons()])

    program = Program(dev, rt, workers=[*gemm_workers, *gelu_workers])
    if trace_config is not None:
        # Tried tracing BOTH a GEMM core and a GELU core first -- failed
        # to route ("packet switched source DMA0 cannot match another
        # connect or masterset operation"). This design already has more
        # mem-tile hops than gemm_pretiled.py's own traceable designs
        # (A split, B forward, C forward PER PIPE, Y join), so it has
        # less routing headroom left for a second trace flow -- CLAUDE.md
        # trap 7 ("adding one trace flow exhausts routing") turns out to
        # bite at ONE flow here, not just at two. One worker only.
        # Tracing gemm_workers[0] ALSO fails to route with the identical
        # "packet switched source DMA0" error -- this design's routing
        # headroom (A split + B forward + a forward PER pipe + Y join,
        # all sharing column 0's shim) is tighter than a plain GEMM
        # design's, and it is gone before a trace flow is added at all,
        # regardless of which worker. gelu_workers[0] is the one that at
        # least COMPILES (produces an empty trace -- no event0/event1
        # pairs, so no cycle numbers came out of it either). Recorded as
        # a genuine open item, not chased further -- see the module-end
        # note.
        program.enable_trace(trace_config.trace_size, workers=[gelu_workers[0]],
                             egress_shim_col=0)
    return program.resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def pipeline_gemm_gelu(A: In, B: In, Y: Out, *,
                       trace_config: CompileTime[TraceConfig | None] = None,
                       identity: CompileTime[bool] = False):
    return _build_design(iron.get_current_device(), trace_config, identity)


def gelu_exact(x):
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / np.sqrt(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-size", type=int, default=262144)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--identity", action="store_true",
                    help="diagnostic: swap the GELU kernel for a pure copy, "
                         "and compare against raw A@B (no GELU) -- isolates "
                         "whether the cross-core hop delivers the right "
                         "bytes from whether the GELU math is right")
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    a16 = a.astype(bfloat16).astype(np.float32)
    b16 = b.astype(bfloat16).astype(np.float32)
    want = (a16 @ b16) if args.identity else gelu_exact(a16 @ b16)

    purge()

    r, s, t = 8, 8, 8  # bf16 mac_dims on npu2 (confirmed by earlier tasks)
    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b), TK, TN, s, t)
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    Y = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = a.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K, N)
    assert np.array_equal(A.numpy(), a.astype(bfloat16)), "A did not reach the device"
    assert np.array_equal(B.numpy(), Bt.view(bfloat16).reshape(K, N)), "B did not reach the device"

    cfg = None
    trace_txt = trace_json = mlir_copy = None
    if args.trace:
        trace_txt = ARTIFACTS / "trace_pipeline_probe.txt"
        trace_json = ARTIFACTS / "trace_pipeline_probe.json"
        mlir_copy = ARTIFACTS / "mlir_pipeline_probe.mlir"
        cfg = TraceConfig(trace_size=args.trace_size, trace_file=str(trace_txt))

    pipeline_gemm_gelu(A, B, Y, trace_config=cfg, identity=args.identity)

    got = Y.numpy().reshape(M, N).astype(np.float64)
    finite = np.isfinite(got).all()
    result = dict(M=M, K=K, N=N, finite=bool(finite))
    if not finite:
        print(f"FAIL -- {int((~np.isfinite(got)).sum())} non-finite outputs "
              f"(the intermediate DID move, but the pipeline computed garbage)")
        result["correctness_pass"] = False
    else:
        rel_fro = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        worst = float(np.abs(got - want).max())
        ok = rel_fro < 1.5e-2
        print(f"pipelined GEMM->mem-tile->GELU, NO C in the I/O signature:")
        print(f"  rel_fro   {rel_fro:.3e}")
        print(f"  worst abs {worst:.3e}")
        print("PASS" if ok else "FAIL",
              "-- tolerance 1.5e-2 rel_fro (same bar as B1's epilogue probe)")
        result.update(rel_frobenius=rel_fro, worst_abs=worst, correctness_pass=ok)

    if cfg is not None:
        size = trace_txt.stat().st_size if trace_txt.exists() else 0
        if size == 0:
            print("  EMPTY TRACE")
            result["trace"] = "empty"
        else:
            phys = getattr(cfg, "physical_mlir_path", None)
            if phys:
                shutil.copy(phys, mlir_copy)
            elif mlir_copy.exists():
                phys = str(mlir_copy)
            if phys is None:
                print("  no physical MLIR (cache hit, no stored copy)")
                result["trace"] = "no-mlir"
            else:
                cfg.trace_to_json(phys, str(trace_json))
                from aie.utils.trace.utils import get_cycles_summary
                deltas = []
                for entry in get_cycles_summary(str(trace_json)):
                    deltas += [d for d in entry[1:] if d is not None]
                if deltas:
                    avg = sum(deltas) / len(deltas)
                    print(f"  traced: n={len(deltas)} avg={avg:.1f} cyc "
                          f"min={min(deltas)} max={max(deltas)}")
                    result["trace"] = dict(invocations=len(deltas), avg_cycles=avg,
                                           min_cycles=min(deltas), max_cycles=max(deltas))
                else:
                    print("  no event0/event1 pairs")
                    result["trace"] = "no-events"

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if result.get("correctness_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
