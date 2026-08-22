# NpuEmbeddings -- tasks/0061, T28: the hierarchical 2-hop merge, built.
#
# 0054 proved a 1:1 GEMM->mem-tile->GELU pipeline. 0057 proved the two
# sub-mechanisms a full ffn_up->GELU->ffn_down chain needs -- (Q1) a JOIN's
# own .cons() feeding a THIRD on-chip Worker with no second link, and (Q2)
# cross-column JOIN routing at any distance -- and quantified, with the
# compiler's own error text, exactly why an 8-way single JOIN cannot express
# ffn_down's full regather (no mem tile has more than 6 ports) while a
# HIERARCHICAL 2-hop merge (two smaller joins, each landing in a different
# mem tile, read by a relay core over its 2 input DMA channels) is
# arithmetically sound. Neither prior session built that hierarchical form,
# and neither used anything but an identity-copy diagnostic kernel -- no
# real GELU, no real second matmul.
#
# This does both, at a scale chosen to fit L1 with REAL (non-identity)
# compute end to end -- see the module-end note for exactly why this scale
# and not a bigger one, and TASK.md for the byte arithmetic and the wall
# that forced it.
#
# ARCHITECTURE
# ------------
#   4 GEMM+GELU producer columns (0,1,2,3), each: real bf16xbf16->fp32 GEMM
#   (TM,TK,TN)=(64,64,48), then a REAL GELU epilogue that narrows straight to
#   bf16 on the SAME core (gelu_epilogue_3072_f32_to_bf16) -- this is what
#   0054/0057's identity_copy_* kernels never did.
#
#   Two merge mem tiles, each a JOIN of GROUP=2 producer columns' GELU'd bf16
#   tiles, EACH mem tile ALSO carrying the local A/B feed for its own
#   column's GEMM core (columns 0 and 2 double as merge destinations) --
#   the same "mem tile does double duty" topology 0057's boundary-case ports
#   test exercised, just at group=2 instead of the tightest group=4 case
#   (see TASK.md for why: a REAL second matmul, unlike an identity copy,
#   also needs L1 for weights + accumulator + output on the RELAY side, and
#   group=4's one-shot gather alone (49,152 B) already leaves no room for
#   any of that).
#
#   ONE relay/"ffn_down"-consuming core, reading BOTH merge results directly
#   off their mem tiles' .cons() (Q1's mechanism -- no .forward(), which is
#   what let group=2 avoid the extra output-port cost 0057's 4-source case
#   hit when it ALSO needed an outbound relay hop). It streams: acquire hop0,
#   partial-matmul-accumulate against a resident weight slice, release hop0;
#   acquire hop1, partial-matmul-accumulate, release hop1 -- NEVER holding
#   both hops' buffers plus weights plus accumulator plus output all at their
#   full production size simultaneously, exactly the acquire/release
#   discipline 0057 flagged as the thing its own one-shot gather probes
#   never needed to solve.
#
#   The relay's own matmul is hand-written (ffn_down_relay_g2.cc), not
#   kernels.mm() -- see that file's docstring for why chaining a SECOND
#   kernels.mm() matmul onto a JOIN's gathered output is a genuinely
#   different, unexplored question (composing the join's own dims_to_stream
#   with the MMAC intrinsic's (r,s) sub-tile order) that this session
#   deliberately did not chase, in favour of building the mechanism this
#   task was actually asked to prove: hierarchical merge + real GELU + real
#   (if not MMAC-accelerated) second-stage matmul, correctness-checked
#   against an independent fp64 reference, never a device read-back.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage: python hierarchical_merge_ffn_probe.py

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
from aie.helpers.taplib import TensorTiler2D             # noqa: E402
from npue import tile_b, to_bf16_bits                    # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

TM, TK, TN = 64, 64, 48       # one MiniLM-shaped ffn_up tile per producer
N_DOWN = 16                    # reduced-scale ffn_down output width (see docstring)
GROUP = 2                      # producer columns joined per merge mem tile
N_HOPS = 2                     # merge groups == relay input channels (core: 2 in / 2 out)
PRODUCER_COLS = [0, 1, 2, 3]
DEST_COLS = [0, 2]             # merge mem tiles double as these columns' local A/B feed
WEIGHT_SEED = 42                # fixed: _build() bakes weights at compile time; main()
                                 # must reconstruct the identical arrays for the reference
                                 # (trap 6c -- never verify against a device read-back)

GELU_TILE = TM * TN            # 3072
Y_SIZE = TM * N_DOWN           # 1024


def markers_for():
    return [f"aie.runtime_sequence(%arg0: memref<{TM * TK}xbf16>, "
            f"%arg1: memref<{len(PRODUCER_COLS) * TK * TN}xbf16>, "
            f"%arg2: memref<{Y_SIZE}xf32>)",
            "ffn_down_hop_matmul_g2_64x48x16"]


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


def _weight_arrays():
    """The relay's resident weight, (N_HOPS, GROUP, TN, N_DOWN) fp32 -> bf16.

    Generated from a FIXED seed so _build() (compile time, baking these into
    the design as constant Buffers) and main() (the independent host
    reference) derive byte-identical arrays without the device ever being
    asked to reveal what it was compiled with.
    """
    rng = np.random.default_rng(WEIGHT_SEED)
    w = (rng.standard_normal((N_HOPS, GROUP, TN, N_DOWN)) * 0.1).astype(np.float32)
    return w, w.astype(bfloat16)


def _build(dev):
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

    gelu_narrow_kernel = ExternalFunction(
        "gelu_epilogue_3072_f32_to_bf16",
        source_file=str(HERE.parent / "m5-eltwise" / "kernels" / "gelu_poly.cc"),
        arg_types=[np.ndarray[(GELU_TILE,), np.dtype[np.float32]],
                   np.ndarray[(GELU_TILE,), np.dtype[bfloat16]]],
        include_dirs=_inc,
    )
    # NPUE-M11 (trap found this session): pulling more than one
    # ExternalFunction entry point from the SAME multi-symbol .cc file in
    # one design duplicate-symbol-fails the link -- every requested symbol
    # compiles the WHOLE file, and no other build in this codebase has ever
    # asked for >1 symbol from one file. Each of these three now lives in
    # its own file (ffn_down_zero.cc / ffn_down_hop_matmul_g2.cc /
    # ffn_down_copy_out.cc) -- see their headers.
    kdir = HERE.parent / "m5-eltwise" / "kernels"
    zero_relay_kernel = ExternalFunction(
        "zero_f32_1024", source_file=str(kdir / "ffn_down_zero.cc"),
        arg_types=[np.ndarray[(Y_SIZE,), np.dtype[np.float32]]],
        include_dirs=_inc,
    )
    hop_matmul_kernel = ExternalFunction(
        "ffn_down_hop_matmul_g2_64x48x16",
        source_file=str(kdir / "ffn_down_hop_matmul_g2.cc"),
        arg_types=[np.ndarray[(GROUP * TM * TN,), np.dtype[bfloat16]],
                   np.ndarray[(GROUP * TN * N_DOWN,), np.dtype[bfloat16]],
                   np.ndarray[(Y_SIZE,), np.dtype[np.float32]]],
        include_dirs=_inc,
    )
    copy_relay_kernel = ExternalFunction(
        "copy_f32_1024", source_file=str(kdir / "ffn_down_copy_out.cc"),
        arg_types=[np.ndarray[(Y_SIZE,), np.dtype[np.float32]]] * 2,
        include_dirs=_inc,
    )

    A_l1_ty = np.ndarray[(TM, TK), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(TK, TN), np.dtype[bfloat16]]
    C_acc_ty = np.ndarray[(TM * TN,), np.dtype[np.float32]]
    C_out_bf16_ty = np.ndarray[(TM * TN,), np.dtype[bfloat16]]

    A_ty = np.ndarray[(TM * TK,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(len(PRODUCER_COLS) * TK * TN,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(Y_SIZE,), np.dtype[np.float32]]

    # --- per-column A/B feed, one shim + mem tile per producer column,
    # exactly like cross_column_join_probe.py. Columns 0 and 2 (DEST_COLS)
    # get their mem tile pinned to the SAME tile the merge join below also
    # uses -- this IS the "mem tile does double duty" topology.
    A_shims, A_rows, B_shims, B_fwds, A_prods, B_prods = {}, {}, {}, {}, {}, {}
    for col in PRODUCER_COLS:
        a_dims = [(TM // r, r * TK), (TK // s, s), (r, TK), (s, 1)]
        a_shim = ObjectFifo(np.ndarray[(TM * TK,), np.dtype[bfloat16]],
                            name=f"A_L3L2_{col}", depth=2)
        a_row = a_shim.cons().split(
            [0], obj_types=[A_l1_ty], names=[f"A_row_{col}"],
            dims_to_stream=[a_dims], tile=Tile(col, 1))[0]
        A_shims[col], A_rows[col] = a_shim, a_row
        A_prods[col] = a_shim.prod(tile=Tile(col, 0))

        b_shim = ObjectFifo(np.ndarray[(TK * TN,), np.dtype[bfloat16]],
                            name=f"B_L3L2_{col}", depth=2)
        b_fwd = b_shim.cons().forward(obj_type=B_l1_ty, name=f"B_fwd_{col}",
                                      depth=2, tile=Tile(col, 1))
        B_shims[col], B_fwds[col] = b_shim, b_fwd
        B_prods[col] = b_shim.prod(tile=Tile(col, 0))

    # --- two merge mem tiles. dims_to_stream is the SAME join-undo formula
    # gemm_pretiled.py's C_l2l3_fifos / pipeline_gemm_gelu_probe.py's Y_mem
    # already use for (TM,TN)=(64,48) tiles at these r,t -- a join of
    # multiple producer tiles is not a byte-for-byte concatenation (0054
    # Problem #5); this undoes whatever the join DMA does, so what a
    # downstream reader sees IS the plain block-concatenated layout
    # ffn_down_relay_g2.cc's own docstring assumes.
    group_mems, C_cols = {}, {}
    for g, dest_col in enumerate(DEST_COLS):
        cols = PRODUCER_COLS[g * GROUP:(g + 1) * GROUP]
        gm = ObjectFifo(
            np.ndarray[(GROUP * TM * TN,), np.dtype[bfloat16]],
            name=f"C_mem_g{g}", depth=1,
            dims_to_stream=[(TM // r, r * TN), (r, t), (TN // t, r * t), (t, 1)])
        cc = gm.prod().join(
            [TM * TN * j for j in range(GROUP)],
            obj_types=[C_out_bf16_ty] * GROUP,
            names=[f"C_g{g}_{j}" for j in range(GROUP)],
            depths=[1] * GROUP,
            tile=Tile(dest_col, 1))
        group_mems[g] = gm
        for j, col in enumerate(cols):
            C_cols[col] = cc[j]

    # --- resident relay weights, baked at compile time (see _weight_arrays).
    _, w_bf16 = _weight_arrays()
    weight_bufs = [
        Buffer(np.ndarray[(GROUP * TN * N_DOWN,), np.dtype[bfloat16]],
              initial_value=w_bf16[g].reshape(-1), name=f"w_hop{g}")
        for g in range(N_HOPS)
    ]

    acc_buf = Buffer(np.ndarray[(Y_SIZE,), np.dtype[np.float32]], name="ffn_down_acc")

    Y_out = ObjectFifo(np.ndarray[(Y_SIZE,), np.dtype[np.float32]], name="Y_out", depth=1)
    Y_pipe = Y_out.cons().forward(obj_type=np.ndarray[(Y_SIZE,), np.dtype[np.float32]],
                                  name="Y_pipe", depth=2)

    def gemm_gelu_core_fn(in_a, in_b, out_c, acc, zero, matmul, gelu_narrow):
        a = in_a.acquire(1)
        b = in_b.acquire(1)
        zero(acc)
        matmul(a, b, acc)
        in_a.release(1)
        in_b.release(1)
        ec = out_c.acquire(1)
        gelu_narrow(acc, ec)
        out_c.release(1)

    def relay_core_fn(hop0, hop1, w0, w1, acc, out_c, zero_r, hop_mm, copy_r):
        zero_r(acc)
        h0 = hop0.acquire(1)
        hop_mm(h0, w0, acc)
        hop0.release(1)
        h1 = hop1.acquire(1)
        hop_mm(h1, w1, acc)
        hop1.release(1)
        ec = out_c.acquire(1)
        copy_r(acc, ec)
        out_c.release(1)

    producer_workers = []
    for col in PRODUCER_COLS:
        acc = Buffer(C_acc_ty, name=f"cacc_{col}")
        producer_workers.append(Worker(
            gemm_gelu_core_fn,
            [A_rows[col].cons(), B_fwds[col].cons(), C_cols[col].prod(),
             acc, zero_kernel, matmul_kernel, gelu_narrow_kernel],
            tile=Tile(col, 2), stack_size=0x2000))

    relay_worker = Worker(
        relay_core_fn,
        [group_mems[0].cons(), group_mems[1].cons(),
         weight_bufs[0], weight_bufs[1], acc_buf, Y_out.prod(),
         zero_relay_kernel, hop_matmul_kernel, copy_relay_kernel],
        tile=Tile(0, 4), stack_size=0x800)

    a_full_tap = TensorTiler2D.simple_tiler((TM, TK))[0]
    b_full_taps = TensorTiler2D.simple_tiler(
        (len(PRODUCER_COLS) * TK, TN), (TK, TN))
    y_full_tap = TensorTiler2D.simple_tiler((TM, N_DOWN))[0]

    A_prod_list = [A_prods[col] for col in PRODUCER_COLS]
    B_prod_list = [B_prods[col] for col in PRODUCER_COLS]
    Y_pipe_cons = Y_pipe.cons(tile=Tile(0, 0))

    def sequence(A, B, Y, A_prod_hs, B_prod_hs, Y_pipe_cons_h):
        tg = TaskGroup()
        for i, col in enumerate(PRODUCER_COLS):
            A_prod_hs[i].fill(A, tap=a_full_tap, group=tg)
            B_prod_hs[i].fill(B, tap=b_full_taps[i], group=tg)
        Y_pipe_cons_h.drain(Y, tap=y_full_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, Y_ty, A_prod_list, B_prod_list, Y_pipe_cons])

    return Program(dev, rt,
                   workers=[*producer_workers, relay_worker]).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def hierarchical_merge_ffn(A: In, B: In, Y: Out):
    return _build(iron.get_current_device())


def gelu_exact(x):
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / np.sqrt(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    purge()

    rng = np.random.default_rng(args.seed)
    a = (rng.standard_normal((TM, TK)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((len(PRODUCER_COLS), TK, TN)) * 0.05).astype(np.float32)
    a16 = a.astype(bfloat16).astype(np.float32)
    b16 = [b[j].astype(bfloat16).astype(np.float32) for j in range(len(PRODUCER_COLS))]

    # --- independent fp64 reference (trap 6c: never against a device read-back).
    up = [a16.astype(np.float64) @ b16[j].astype(np.float64) for j in range(len(PRODUCER_COLS))]
    gelu_fp32 = [gelu_exact(u).astype(np.float32) for u in up]
    # Emulate the device's own RNE bf16 narrow (aie::set_rounding(conv_even)
    # in gelu_epilogue_3072_f32_to_bf16) so the reference carries the SAME
    # quantisation the hardware actually applies, not zero quantisation.
    gelu_bf16 = [g.astype(bfloat16).astype(np.float64) for g in gelu_fp32]

    w_fp32, _ = _weight_arrays()
    acc_ref = np.zeros((TM, N_DOWN), dtype=np.float64)
    for g in range(N_HOPS):
        for j in range(GROUP):
            col = g * GROUP + j
            acc_ref += gelu_bf16[col] @ w_fp32[g, j].astype(bfloat16).astype(np.float64)

    A = iron.zeros((TM, TK), dtype=bfloat16, device="npu")
    B = iron.zeros((len(PRODUCER_COLS) * TK, TN), dtype=bfloat16, device="npu")
    Y = iron.zeros(Y_SIZE, dtype=np.float32, device="npu")

    A[:] = a.astype(bfloat16)
    b_tiled = np.concatenate(
        [tile_b(to_bf16_bits(b[j]), TK, TN, 8, 8).view(bfloat16).reshape(TK, TN)
         for j in range(len(PRODUCER_COLS))], axis=0)
    B[:] = b_tiled
    assert np.array_equal(A.numpy(), a.astype(bfloat16)), "A did not reach the device"
    assert np.array_equal(B.numpy(), b_tiled), "B did not reach the device"

    hierarchical_merge_ffn(A, B, Y)

    got = Y.numpy().reshape(TM, N_DOWN).astype(np.float64)
    result = dict(TM=TM, TK=TK, TN=TN, N_DOWN=N_DOWN, group=GROUP, n_hops=N_HOPS)
    finite = np.isfinite(got).all()
    if not finite:
        print(f"FAIL -- {int((~np.isfinite(got)).sum())} non-finite outputs")
        result["correctness_pass"] = False
    else:
        rel_fro = float(np.linalg.norm(got - acc_ref) / np.linalg.norm(acc_ref))
        worst = float(np.abs(got - acc_ref).max())
        ok = rel_fro < 3e-2
        print("hierarchical 2-hop merge: 4 GEMM+GELU producer cols -> 2 merge "
              "mem tiles (each ALSO local A/B feed) -> 1 relay ffn_down-shaped "
              "matmul, k-block (per-hop) acquire/release, NO C tensor for "
              "either GEMM stage in the design's own I/O signature:")
        print(f"  rel_fro   {rel_fro:.3e}")
        print(f"  worst abs {worst:.3e}")
        print("PASS" if ok else "FAIL", "-- tolerance 3e-2 rel_fro "
              "(two chained bf16 narrows plus a non-MMAC second matmul)")
        result.update(rel_frobenius=rel_fro, worst_abs=worst, correctness_pass=ok)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if result.get("correctness_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
