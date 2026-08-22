# NpuEmbeddings -- T26 chain probe (tasks/0056, research/OPEN-THREADS.md T26)
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS ANSWERS
# ------------------
# 0053's t26_probe.py refuted the "k-block-boundary re-quantisation" hypothesis
# and the "the anomaly reproduces on a single isolated GEMM" hypothesis, and
# proposed a THIRD one, untested: rounding-mode asymmetry between the fp32-C
# path's downstream (production, host-side) bf16 narrowing and the bf16-C
# path's on-core (conv_even) narrowing -- if the host used AIE's default
# `floor` instead of round-to-nearest-even, the bias would compound across
# MiniLM's 24 chained GEMMs in a way one isolated GEMM cannot show.
#
# tasks/0056 read runtime/src/main.cpp and found this hypothesis, AS STATED,
# is REFUTED BY THE CODE: `to_bf16()` (line ~403) is explicitly
# round-to-nearest-even (`(u + 0x7FFF + ((u >> 16) & 1)) >> 16`), the same
# formula as tools/npue.py's `to_bf16_bits` used when packing weights. There
# is no floor-vs-conv_even asymmetry in production's host narrowing path --
# both are unbiased round-to-nearest-even.
#
# So THIS probe tests a refined, structural hypothesis instead: production's
# `gemm()` (runtime/src/main.cpp ~line 1004) does, per layer boundary:
#   fp32-C path:  raw fp32 accumulator -> [+bias, fp32]        -> RNE narrow -> next GEMM's bf16 A
#   bf16-C path:  raw fp32 accumulator -> [conv_even narrow]   -> +bias (fp32, from bf16) -> RNE narrow -> next GEMM's bf16 A
# i.e. bf16-C does an EXTRA narrowing step, EARLY (before bias), that fp32-C
# does not do; both paths do the SAME final RNE narrow before the next GEMM's
# A. Does this extra-early-narrow structure make bf16-C's error grow SLOWER
# (or fp32-C's grow FASTER) as more GEMM stages are chained, under bfp16
# emulation? If the ratio between the two paths' final-stage error widens
# with chain length, that supports "compounds across chained GEMMs" (even
# though the *mechanism* is structural narrowing position, not rounding mode).
# If the ratio stays flat, the anomaly is NOT a chaining effect either, and
# whatever else it is remains open.
#
# Design: ONE constant GEMM shape (K == N, so output width equals next
# stage's input width) reused for every stage -- only 2 device BUILDS total
# (fp32-C rtp=False, bf16-C rtp=True), dispatched L times each for a length-L
# chain. Same random weights/bias schedule feeds BOTH paths at each stage, so
# only the narrowing structure differs. Reference is a pure fp64 chain with NO
# intermediate narrowing at all (the "ideal" trajectory) -- same spirit as
# t26_probe.py's `ref`, extended over stages.
#
# CLAUDE.md trap 6c: C is read back as the ACTUAL device output (bf16 bits or
# fp32 bits IRON already synced), never re-derived from "intended" values.
#
# Env: iron env with C:\dev\mlir-aie\iron_env.ps1 dot-sourced.
# Usage:
#   python t26_chain_probe.py --out artifacts/t26_chain_probe.json

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import kernels, str_to_dtype
from aie.iron.device import from_name

HERE = Path(__file__).parent
REPO = HERE.parent.parent
ARTIFACTS = HERE / "artifacts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))

from gemm_pretiled import pretiled_array          # noqa: E402
from npue import tile_b, to_bf16_bits             # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

# Constant shape across every stage: K == N so a stage's output width feeds
# the next stage's input width directly. Legal at tile (64,64,48), cols=4:
# M%64=0, K%64=0 (3 k-blocks), N%(48*4)=0.
M, K, N = 256, 192, 192
TM, TK, TN = 64, 64, 48
COLS = 4
L_MAX = 4


def to_bf16_np(x_f32: np.ndarray) -> np.ndarray:
    """Host RNE narrow, bit-identical formula to runtime/src/main.cpp's
    to_bf16() and tools/npue.py's to_bf16_bits -- the ACTUAL production
    downstream-narrowing step between two chained GEMMs."""
    bits = to_bf16_bits(x_f32.astype(np.float32))
    return bits.astype(np.uint16).view(bfloat16)


# ---------------------------------------------------------------------------
# Cache identity (same ordered aie.runtime_sequence signature as t26_probe.py)
# ---------------------------------------------------------------------------

def core_columns(d):
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                        m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def markers_for(c_dtype):
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}x{c_dtype}>)",
            f"<size = {TK}, stride = {TN}>"]


def purge(markers, cols, what):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers) and core_columns(d) in (cols, None):
            shutil.rmtree(d)
            n += 1
    if n:
        print(f"  [{what}] purged {n} cache candidate(s)")


# ---------------------------------------------------------------------------
# One dispatch (constant M,K,N,tile,cols; emulate always True in this probe --
# we are studying the bfp16-emulation regime T26 lives in)
# ---------------------------------------------------------------------------

def dispatch(A_bf16, B_bf16, c_bf16, rtp):
    """A_bf16: (M,K) bfloat16, B_bf16: (K,N) bfloat16 -- both already the
    values written to the device; returns the ACTUAL device C, reshaped
    (M,N), as float64 (never a re-derived/read-back-implied value, trap 6c)."""
    dt_in = str_to_dtype("bf16")
    dt_out = str_to_dtype("f32")
    dt_c = str_to_dtype("bf16") if c_bf16 else dt_out

    A = iron.zeros((M, K), dtype=dt_in, device="npu")
    B = iron.zeros((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_c, device="npu")

    A[:] = A_bf16

    r, s, t = kernels.mm(dim_m=TM, dim_k=TK, dim_n=TN, input_dtype=dt_in,
                          output_dtype=dt_out, b_col_maj=False, c_col_maj=False,
                          use_chess=False, emulate_bf16_mmul_with_bfp16=True,
                          vectorized=True).mac_dims
    tiled = tile_b(B_bf16.view(np.uint16), TK, TN, s, t, order="k,n")
    B[:] = tiled.view(bfloat16).reshape(K, N)

    pretiled_array(A, B, C, M=M, K=K, N=N, m=TM, k=TK, n=TN, n_aie_cols=COLS,
                    dtype_in_str="bf16", dtype_out_str="f32",
                    emulate_bf16_mmul_with_bfp16=True,
                    pretiled=True, trace_config=None, rtp=rtp, c_bf16=c_bf16)

    got = C.numpy()
    if c_bf16:
        got = got.view(bfloat16)
    return got.astype(np.float64).reshape(M, N)


def build_once(c_bf16, rtp):
    c_dtype = "bf16" if c_bf16 else "f32"
    mk = markers_for(c_dtype)
    tag = f"build/c_bf16={c_bf16}"
    purge(mk, COLS, tag)


def rel_fro(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def one_minus_cos(got, ref):
    g, r = got.reshape(-1), ref.reshape(-1)
    return float(1.0 - (g @ r) / (np.linalg.norm(g) * np.linalg.norm(r)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ARTIFACTS / "t26_chain_probe.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l-max", type=int, default=L_MAX)
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # trap: without this, IRON silently compiles for NPU1 and bfp16 becomes a
    # no-op (research/notes/0002).
    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(args.seed)

    def randn_bf16(shape):
        return (rng.standard_normal(shape).astype(np.float32)
                .astype(bfloat16))

    L = args.l_max

    # Stage-0 input, and per-stage weights/biases -- fixed schedule shared by
    # BOTH the fp32-C and bf16-C chains, and by the fp64 reference.
    X0 = randn_bf16((M, K))
    Ws = [randn_bf16((K, N)) for _ in range(L)]
    biases = [rng.standard_normal(N).astype(np.float32) * 0.1 for _ in range(L)]

    # Purge + (first-dispatch) build each design ONCE; every subsequent
    # dispatch at this constant shape hits the cache.
    build_once(False, False)   # fp32-C, rtp=False
    build_once(True, True)     # bf16-C, rtp=True

    # --- fp64 "ideal" reference chain: no MAC emulation error, no
    # intermediate narrowing at all, full fp64 precision end to end. ---
    X_ref = X0.astype(np.float64)
    ref_states = []
    for i in range(L):
        W64 = Ws[i].astype(np.float64)
        b64 = biases[i].astype(np.float64)
        X_ref = X_ref @ W64 + b64
        ref_states.append(X_ref.copy())

    # --- fp32-C device chain ---
    X_fp32c = X0
    fp32c_states_bf16 = []
    for i in range(L):
        C = dispatch(X_fp32c, Ws[i], c_bf16=False, rtp=False)   # fp64 fp32-C
        Y = C + biases[i].astype(np.float64)
        X_fp32c_bf16 = to_bf16_np(Y.astype(np.float32))
        fp32c_states_bf16.append(X_fp32c_bf16.astype(np.float64).copy())
        X_fp32c = X_fp32c_bf16

    # --- bf16-C device chain ---
    X_bf16c = X0
    bf16c_states_bf16 = []
    for i in range(L):
        C = dispatch(X_bf16c, Ws[i], c_bf16=True, rtp=True)     # fp64 bf16-C
        Y = C + biases[i].astype(np.float64)   # C already conv_even-narrowed
        X_bf16c_bf16 = to_bf16_np(Y.astype(np.float32))
        bf16c_states_bf16.append(X_bf16c_bf16.astype(np.float64).copy())
        X_bf16c = X_bf16c_bf16

    rows = []
    print(f"\n{'stage':>5} {'fp32C rel_fro':>15} {'bf16C rel_fro':>15} "
          f"{'ratio (fp32C/bf16C)':>20}")
    for i in range(L):
        rf_fp32c = rel_fro(fp32c_states_bf16[i], ref_states[i])
        rf_bf16c = rel_fro(bf16c_states_bf16[i], ref_states[i])
        omc_fp32c = one_minus_cos(fp32c_states_bf16[i], ref_states[i])
        omc_bf16c = one_minus_cos(bf16c_states_bf16[i], ref_states[i])
        ratio = rf_fp32c / rf_bf16c if rf_bf16c else float("nan")
        print(f"{i + 1:>5} {rf_fp32c:>15.6e} {rf_bf16c:>15.6e} {ratio:>20.3f}")
        rows.append({"stage": i + 1, "fp32C_rel_fro": rf_fp32c,
                      "bf16C_rel_fro": rf_bf16c,
                      "fp32C_1mcos": omc_fp32c, "bf16C_1mcos": omc_bf16c,
                      "ratio_fp32C_over_bf16C": ratio})

    results = {"shape": {"M": M, "K": K, "N": N, "m": TM, "k": TK, "n": TN,
                          "cols": COLS, "L_max": L},
               "seed": args.seed, "stages": rows}
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
