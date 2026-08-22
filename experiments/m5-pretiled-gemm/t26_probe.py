# NpuEmbeddings -- T26 probe (tasks/0053, research/OPEN-THREADS.md T26)
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS ANSWERS
# ------------------
# tasks/0052 found bfp16-emulated GEMM + bf16-C transport (core-local `Buffer`
# + narrow_f32_bf16) measures 1-cos 3.615e-04 against bfp16-emulated + fp32-C
# transport (the C ObjectFifo directly)'s 2.395e-03 -- BF16-C IS 6.6x MORE
# ACCURATE, which is backwards (adding a rounding should never help). Named
# hypothesis, untested: the fp32-C fifo path re-quantises the C partial sum
# through a bfp16-shaped conversion at every k-block boundary (6x at K=384,
# k=64), while the Buffer path's accumulator stays clean until the single
# narrow() at the end.
#
# THE PROBE. If the hypothesis is right, a SINGLE k-block dispatch (K=64,
# k=64 -- the loop body executes ONCE, there is no boundary to cross) should
# show fp32-C =~ bf16-C, and the gap should open specifically as more
# k-blocks accumulate in one dispatch (K=384, 6 blocks). We test this by
# running the SAME logical GEMM (M=256, K=384, N=192) two ways:
#
#   "full"  -- ONE dispatch, K=384, tile k=64  -> 6 k-blocks inside one loop
#   "split" -- SIX dispatches, K=64 each, tile k=64 -> 1 k-block per dispatch,
#              summed on the HOST in fp64 (no host rounding added beyond what
#              each dispatch's own C narrowing already did)
#
# ...crossed with {fp32-C, bf16-C} under bfp16 emulation, plus a bonus pair on
# PLAIN bf16 (not emulated) to test whether the same k-block-boundary defect
# exists in miniature in TODAY'S SHIPPING datapath (bf16 fp32-C, 1-cos
# 1.086e-05 in production).
#
# Six cells, same random A/B data throughout (sliced for the split case), same
# fp64 reference (A_bf16 @ B_bf16 computed at fp64 -- the "what was actually
# written to the device" reference per CLAUDE.md trap 6c, not the true
# pre-quantisation random values):
#
#   1. emulate=True,  c_bf16=False, full   (matches 0052's 2.395e-03 regime)
#   2. emulate=True,  c_bf16=False, split
#   3. emulate=True,  c_bf16=True,  full   (matches 0052's 3.615e-04 regime)
#   4. emulate=True,  c_bf16=True,  split
#   5. emulate=False, c_bf16=False, full   (today's shipping bf16 datapath)
#   6. emulate=False, c_bf16=False, split
#
# Plus a 7th, disassembly-only build: emulate=True, c_bf16=False, RTP=True,
# full -- structurally identical to cell 3 except for c_bf16, so the object
# diff isolates exactly one variable instead of two (cell 1 is rtp=False,
# which changes the worker's core_fn shape too).
#
# CRITICAL (CLAUDE.md, 0030's fifth fail-open, 0045's sixth): bfp16 and
# plain-bf16 builds are AMBIGUOUS in the JIT cache when M/K/N/c_dtype match --
# cells (1,5) and (2,6) collide. purge() runs before every DISTINCT shape
# signature, not just once, using the ordered aie.runtime_sequence marker
# (export_gemm_rtp.py's approach, not the three-loose-strings form 0045 found
# broken).
#
# Env: iron env with C:\dev\mlir-aie\iron_env.ps1 dot-sourced.
# Usage:
#   python t26_probe.py --out artifacts/t26_probe.json
#   python t26_probe.py --disasm-only     # skip the accuracy matrix, just
#                                          # rebuild cells 3 and 7 and dump .o

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
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
from npue import tile_b                           # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

M, N = 256, 192          # smallest shapes legal at m=64,k=64,n=48,cols=4
TM, TK, TN = 64, 64, 48
COLS = 4
K_FULL = 384
K_SPLIT = 64
N_SPLITS = K_FULL // K_SPLIT   # 6


# ---------------------------------------------------------------------------
# Cache identity (adapted from tools/export_gemm_rtp.py -- same ordered
# aie.runtime_sequence signature match, not loose memref substrings).
# ---------------------------------------------------------------------------

def core_columns(d):
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                        m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def markers_for(M, K, N, c_dtype):
    """BUG FOUND AND FIXED (this task): the first version of this function was
    called with the TILE dims (64,64,48), not the actual per-call GEMM shape
    (M, K, N differ between the full-K=384 and split-K=64 cells) -- so the
    memref sizes it built (4096/3072/3072 bytes-worth of elements) never
    appeared in any real aie.mlir, purge() matched ZERO candidates every
    time, and the 'purge before every build' safety net silently did
    nothing for the whole first run. Matches export_gemm_rtp.py's
    ordered-signature convention, which exists for exactly this reason
    (tasks/0045's sixth fail-open)."""
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


def find_cache(markers, cols, what):
    hits = []
    for d in CACHE.iterdir():
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists() and (d / "final.xclbin").exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not all(x in text for x in markers):
            continue
        if core_columns(d) != cols:
            continue
        hits.append(d)
    if len(hits) != 1:
        raise SystemExit(f"[{what}] {len(hits)} cache candidates after purge "
                          f"-- expected exactly 1: {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# One dispatch
# ---------------------------------------------------------------------------

def dispatch(A_slice, B_slice, K, cols, emulate, c_bf16, rtp):
    """A_slice: (M,K) bfloat16, B_slice: (K,N) bfloat16 -- BOTH already the
    values written to A/B; returns C reshaped (M,N) float64."""
    dt_in = str_to_dtype("bf16")
    dt_out = str_to_dtype("f32")
    dt_c = str_to_dtype("bf16") if c_bf16 else dt_out

    A = iron.zeros((M, K), dtype=dt_in, device="npu")
    B = iron.zeros((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_c, device="npu")

    A[:] = A_slice

    r, s, t = kernels.mm(dim_m=TM, dim_k=TK, dim_n=TN, input_dtype=dt_in,
                          output_dtype=dt_out, b_col_maj=False, c_col_maj=False,
                          use_chess=False, emulate_bf16_mmul_with_bfp16=emulate,
                          vectorized=True).mac_dims
    tiled = tile_b(B_slice.view(np.uint16), TK, TN, s, t, order="k,n")
    B[:] = tiled.view(bfloat16).reshape(K, N)

    pretiled_array(A, B, C, M=M, K=K, N=N, m=TM, k=TK, n=TN, n_aie_cols=cols,
                    dtype_in_str="bf16", dtype_out_str="f32",
                    emulate_bf16_mmul_with_bfp16=emulate,
                    pretiled=True, trace_config=None, rtp=rtp, c_bf16=c_bf16)

    return C.numpy().reshape(M, N).astype(np.float64)


def run_cell(tag, A_full, B_full, mode, emulate, c_bf16, rtp, want_cachedir=False):
    """mode: 'full' (one K=384 dispatch) or 'split' (six K=64 dispatches
    summed on the host in fp64)."""
    K = K_FULL if mode == "full" else K_SPLIT
    c_dtype = "bf16" if c_bf16 else "f32"
    mk = markers_for(M, K, N, c_dtype)
    purge(mk, COLS, tag)

    if mode == "full":
        got = dispatch(A_full, B_full, K_FULL, COLS, emulate, c_bf16, rtp)
    else:
        got = np.zeros((M, N), dtype=np.float64)
        for i in range(N_SPLITS):
            A_i = A_full[:, i * K_SPLIT:(i + 1) * K_SPLIT]
            B_i = B_full[i * K_SPLIT:(i + 1) * K_SPLIT, :]
            got += dispatch(A_i, B_i, K_SPLIT, COLS, emulate, c_bf16, rtp)

    cache_dir = find_cache(mk, COLS, tag) if want_cachedir else None
    return got, cache_dir


def rel_fro(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def one_minus_cos(got, ref):
    g, r = got.reshape(-1), ref.reshape(-1)
    return float(1.0 - (g @ r) / (np.linalg.norm(g) * np.linalg.norm(r)))


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

def find_kernel_objs(cache_dir, pattern):
    return sorted(cache_dir.glob(pattern))


def objdump(peano_bin, obj_path, out_path):
    exe = peano_bin / "llvm-objdump.exe"
    if not exe.exists():
        return f"llvm-objdump.exe not found at {exe}"
    res = subprocess.run([str(exe), "-d", "-r", str(obj_path)],
                          capture_output=True, text=True)
    out_path.write_text(res.stdout, encoding="utf-8")
    if res.returncode != 0:
        return f"llvm-objdump exit {res.returncode}: {res.stderr[:2000]}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ARTIFACTS / "t26_probe.json"))
    ap.add_argument("--disasm-only", action="store_true",
                     help="skip the accuracy matrix, just rebuild the two "
                          "rtp=True full-K=384 cells (c_bf16 True/False) and "
                          "objdump the matmul kernels")
    ap.add_argument("--no-disasm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-full", type=int, default=None,
                     help="total K for the 'full' single-dispatch cell; "
                          "'split' always uses k=64 chunks of this. Try 1536 "
                          "(ffn_down's real K, 24 k-blocks) to see whether "
                          "the full/split gap grows with more k-blocks.")
    args = ap.parse_args()

    global K_FULL, N_SPLITS
    if args.k_full is not None:
        K_FULL = args.k_full
    N_SPLITS = K_FULL // K_SPLIT
    assert K_FULL % K_SPLIT == 0

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # trap: without this, IRON silently compiles for NPU1 and bfp16 becomes a
    # no-op (research/notes/0002). n_cols=None or it defaults to one column.
    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(args.seed)
    A_full = (rng.standard_normal((M, K_FULL)).astype(np.float32)
              .astype(bfloat16))
    B_full = (rng.standard_normal((K_FULL, N)).astype(np.float32)
              .astype(bfloat16))
    ref = A_full.astype(np.float64) @ B_full.astype(np.float64)
    print(f"reference ||ref||_F = {np.linalg.norm(ref):.4e}")

    results = {"shape": {"M": M, "K": K_FULL, "N": N, "m": TM, "k": TK,
                          "n": TN, "cols": COLS},
               "cells": []}

    if not args.disasm_only:
        cells = [
            ("emulate+fp32C",  True,  False),
            ("emulate+bf16C",  True,  True),
            ("plain+fp32C",    False, False),
        ]
        for label, emulate, c_bf16 in cells:
            rtp = c_bf16  # c_bf16 requires rtp=True (gemm_pretiled.py guard)
            row = {"label": label, "emulate": emulate, "c_bf16": c_bf16}
            for mode in ("full", "split"):
                tag = f"{label}/{mode}"
                print(f"\n== {tag} ==")
                got, _ = run_cell(tag, A_full, B_full, mode, emulate, c_bf16, rtp)
                rf = rel_fro(got, ref)
                omc = one_minus_cos(got, ref)
                row[f"{mode}_rel_fro"] = rf
                row[f"{mode}_1mcos"] = omc
                print(f"  {tag}: rel_fro={rf:.6e}  1-cos={omc:.6e}")
            ratio = (row["full_rel_fro"] / row["split_rel_fro"]
                     if row["split_rel_fro"] else float("nan"))
            row["full_over_split_ratio"] = ratio
            print(f"  {label}: full/split rel_fro ratio = {ratio:.3f}x "
                  f"({'grows with k-blocks' if ratio > 1.2 else 'flat'})")
            results["cells"].append(row)

    disasm = {}
    if not args.no_disasm:
        print("\n== disassembly: rtp=True, full K=384, c_bf16 False vs True ==")
        peano = Path(__import__("os").environ.get("PEANO_INSTALL_DIR", ""))
        if not peano:
            print("  PEANO_INSTALL_DIR not set -- skipping disassembly "
                  "(dot-source iron_env.ps1)")
        else:
            peano_bin = peano / "bin"
            for label, c_bf16 in (("fp32C_rtp", False), ("bf16C_rtp", True)):
                tag = f"disasm/{label}"
                got, cache_dir = run_cell(tag, A_full, B_full, "full",
                                           True, c_bf16, True,
                                           want_cachedir=True)
                print(f"  {label}: cache dir = {cache_dir}")
                objs = find_kernel_objs(cache_dir, "*.o")
                print(f"  {label}: objects = {[o.name for o in objs]}")
                dumped = []
                for o in objs:
                    out_txt = ARTIFACTS / f"objdump_{label}_{o.stem}.txt"
                    err = objdump(peano_bin, o, out_txt)
                    if err:
                        print(f"    {o.name}: {err}")
                    else:
                        n_lines = len(out_txt.read_text(encoding="utf-8")
                                       .splitlines())
                        print(f"    {o.name} -> {out_txt.name} "
                              f"({n_lines} lines)")
                        dumped.append(str(out_txt))
                disasm[label] = {"cache_dir": str(cache_dir),
                                  "objects": [o.name for o in objs],
                                  "dumps": dumped}
    results["disassembly"] = disasm

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
