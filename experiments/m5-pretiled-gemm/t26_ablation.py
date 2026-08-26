# NpuEmbeddings -- T26 rounding-mode ablation (tasks/0099, research/OPEN-THREADS.md T26)
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS ANSWERS
# ------------------
# tasks/0098 read the kernel source and found: `mmul_bf16_bf16.hpp`'s emulated
# bf16xbf16 mac() calls the AMBIENT-mode `to_v64bfp16ebs8(acc)` intrinsic, not
# the `_conf` form that actually sets the rounding-mode register -- so
# `mm.cc`'s own `aie::swap_rounding(conv_even)`/`set_rounding(saved)` around
# the k-loop is dead code, and every A/B-tile bfp16 quantisation inside the
# emulated matmul runs under whatever rounding mode is ALREADY sitting in that
# physical core's control register. A core that has never run narrow_f32_bf16
# (every fp32-C core, always) is stuck at AIE's default `floor` -- biased low,
# compounding over K. A bf16-C core inherits `conv_even` from its OWN prior
# narrow() call (which never restores) from its second output tile onward.
#
# PREDICTION: priming an fp32-C core's rounding-mode register to `conv_even`
# via a THROWAWAY narrow()-calling dispatch, with NO change to C's transport
# dtype, should close most of the 6.6x gap (1-cos 2.395e-03 -> 3.615e-04,
# tasks/0052 S6) by itself.
#
# THE ABLATION, and why it is built this way instead of literally what 0098's
# "Next" section sketched (warm up with a c_bf16=True dispatch, then switch to
# the c_bf16=False design): 0053's own objdump diff shows fp32-C and bf16-C
# are DIFFERENT compiled objects (bf16-C's wrapper has one extra basic block
# and a different accumulator address) -- i.e. different xclbins. Warming up
# on one xclbin and measuring on another crosses a context switch (CLAUDE.md
# trap 7b), and it is genuinely unknown whether `crrnd` survives that switch
# (trap 7b's own finding is that a switch is NOT eviction -- contexts stay
# Active, Suspensions=0 -- which is suggestive but not proof). A negative
# result from THAT design would be ambiguous: "mechanism wrong" and "crrnd
# resets on context switch, mechanism untested" look identical.
#
# So this ablation adds a THIRD gemm_pretiled.py worker variant, `poison`
# (rtp=True, c_bf16=False, poison=True): structurally the PLAIN fp32-C worker
# -- the real output goes straight into the C ObjectFifo, C's transport dtype
# NEVER changes -- plus one extra, discarded call after each dispatch's real
# output is released: zero a scratch accumulator and narrow() it into a
# scratch tile connected to no ObjectFifo (never DMA'd, never read). This
# means "cold" and "warmed" fp32-C are the SAME xclbin, the SAME hw_context,
# dispatched twice in the SAME process -- byte-for-byte identical machine
# code both times, zero ambiguity about which physical cores are involved
# (there is only ever one context). The only variable between the two
# measured dispatches is operational history: has this core's own prior
# dispatch already executed narrow() once, or not.
#
#   "cold"  -- FRESH process, ONE dispatch (the very first thing this
#              process's hw_context ever runs), measured directly. No core
#              in this context has ever executed narrow().
#   "warm"  -- FRESH process, ONE throwaway dispatch (random discard data,
#              result never read) to execute narrow() once per core and leave
#              `crrnd`=conv_even, THEN a SECOND dispatch (same context, no
#              rebuild) with the real data, measured.
#   "bf16c" -- FRESH process, the SAME warm-up-then-measure protocol on the
#              standard (unmodified) c_bf16=True design, so it is measured at
#              the same "already-settled register state" operating point as
#              "warm" -- production's 6.6x anomaly was measured across many
#              CHAINED GEMMs (tasks/0052/0056), and 0053 already showed a
#              bf16-C core's OWN first dispatch is "tied" with cold fp32-C
#              (its own narrow() hasn't run yet) -- so bf16-C's first-ever
#              dispatch is not the right operating point to compare against.
#
# "Fresh process" is the mechanism used to force a cold register: each trial
# is a SEPARATE python invocation (a brand-new hw_context), on the
# expectation that opening a new hw_context resets per-core control state.
# This is exactly the assumption under test alongside the rounding mechanism
# itself, so `plain_check` (below) cross-checks it independently.
#
# `plain_check` -- FRESH process, ONE dispatch of the ORIGINAL, unmodified
# fp32-C design (rtp=False, no poison plumbing at all -- narrow() does not
# exist anywhere in its compiled code, structurally CANNOT be warmed by
# anything). If "cold" (poison design, first dispatch) and "plain_check"
# (original design, only dispatch) do not agree, the "fresh process = fresh
# register" assumption, or the poison plumbing itself, is suspect.
#
# CACHE IDENTITY (CLAUDE.md's cache-marker trap, six prior incidents: 0030,
# 0053, 0054, 0083, 0087, 0090). `poison=True` changes NO tensor memref shape
# in the generated aie.mlir (C's transport dtype is fp32 in both the plain
# and the poison design) -- so the ordinary M/K/N/dtype marker used by
# t26_probe.py/t26_chain_probe.py CANNOT tell a poison build apart from a
# plain rtp=True fp32-C build at the same shape. `markers_for()` below adds
# an explicit, unique marker string (the `poison_acc_0_0` buffer name) that
# only exists in the poison design's MLIR, and `purge()`/`find_cache()` treat
# it as a REQUIRE-marker for poison lookups and an EXCLUDE-marker for
# non-poison lookups, so the two can never be confused for each other.
#
# Env: iron env with C:\dev\mlir-aie\iron_env.ps1 dot-sourced.
# Usage (one trial per invocation, by design -- see "fresh process" above):
#   python t26_ablation.py --prebuild             # one-time: build + verify
#   python t26_ablation.py --mode cold  --seed 0 --out artifacts\t26_ablation\cold_0.json
#   python t26_ablation.py --mode warm  --seed 0 --out artifacts\t26_ablation\warm_0.json
#   python t26_ablation.py --mode bf16c --seed 0 --out artifacts\t26_ablation\bf16c_0.json
#   python t26_ablation.py --mode plain_check --seed 0 --out artifacts\t26_ablation\plain_0.json

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
ARTIFACTS = HERE / "artifacts" / "t26_ablation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))

from gemm_pretiled import pretiled_array          # noqa: E402
from npue import tile_b                           # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

# Same regime t26_probe.py used for its isolated single-GEMM cell: smallest
# shape legal at m=64,k=64,n=48,cols=4, K=384 (6 k-blocks, MiniLM's proj/qkv
# order of magnitude).
M, K, N = 256, 384, 192
TM, TK, TN = 64, 64, 48
COLS = 4


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

def core_columns(d):
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                        m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def markers_for(c_dtype, poison):
    # Just the ordered aie.runtime_sequence signature (shape identity) plus,
    # for poison builds, the buffer name that only exists in that variant's
    # MLIR. No `<size=.., stride=..>` sub-tile marker (t26_probe.py's second
    # marker) here -- checked directly against this shape's actual aie.mlir
    # and it does not appear verbatim (its exact (s,t) sub-tile factoring
    # differs), so it would silently fail to match rather than fail loudly;
    # the two markers below were independently confirmed present with
    # `grep -c` before being trusted (see task log).
    base = [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}x{c_dtype}>)"]
    if poison:
        base.append("poison_acc_0_0")
    return base


def _matches(text, markers, excludes):
    return (all(x in text for x in markers)
            and not any(x in text for x in excludes))


def purge(markers, excludes, cols, what):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if _matches(text, markers, excludes) and core_columns(d) in (cols, None):
            shutil.rmtree(d)
            n += 1
    if n:
        print(f"  [{what}] purged {n} cache candidate(s)")


def find_cache(markers, excludes, cols, what):
    hits = []
    for d in CACHE.iterdir():
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists() and (d / "final.xclbin").exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not _matches(text, markers, excludes):
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

def dispatch(A_np, B_np, poison, c_bf16, rtp=True):
    """A_np: (M,K) bfloat16, B_np: (K,N) bfloat16 -- both already the values
    written to A/B; returns the ACTUAL device C (trap 6c), reshaped (M,N),
    as float64."""
    dt_in = str_to_dtype("bf16")
    dt_out = str_to_dtype("f32")
    dt_c = str_to_dtype("bf16") if c_bf16 else dt_out

    A = iron.zeros((M, K), dtype=dt_in, device="npu")
    B = iron.zeros((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_c, device="npu")

    A[:] = A_np
    assert np.array_equal(A.numpy(), A_np), "A did not reach the device"

    r, s, t = kernels.mm(dim_m=TM, dim_k=TK, dim_n=TN, input_dtype=dt_in,
                          output_dtype=dt_out, b_col_maj=False, c_col_maj=False,
                          use_chess=False, emulate_bf16_mmul_with_bfp16=True,
                          vectorized=True).mac_dims
    tiled = tile_b(B_np.view(np.uint16), TK, TN, s, t, order="k,n")
    B[:] = tiled.view(bfloat16).reshape(K, N)

    pretiled_array(A, B, C, M=M, K=K, N=N, m=TM, k=TK, n=TN, n_aie_cols=COLS,
                    dtype_in_str="bf16", dtype_out_str="f32",
                    emulate_bf16_mmul_with_bfp16=True,
                    pretiled=True, trace_config=None, rtp=rtp, c_bf16=c_bf16,
                    poison=poison)

    got = C.numpy()
    if c_bf16:
        got = got.view(bfloat16)
    return got.astype(np.float64).reshape(M, N)


def rel_fro(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def one_minus_cos(got, ref):
    g, r = got.reshape(-1), ref.reshape(-1)
    return float(1.0 - (g @ r) / (np.linalg.norm(g) * np.linalg.norm(r)))


def randn_bf16(rng, shape):
    return rng.standard_normal(shape).astype(np.float32).astype(bfloat16)


# ---------------------------------------------------------------------------
# Disassembly (--prebuild only)
# ---------------------------------------------------------------------------

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


def prebuild_and_verify():
    """Force-build both designs ONCE, confirm cache identity by CONTENTS
    (not mtime, trap 7c), and objdump their kernel objects."""
    import os
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(12345)
    A = randn_bf16(rng, (M, K))
    B = randn_bf16(rng, (K, N))

    results = {}
    for label, poison, c_bf16 in (("poison", True, False), ("bf16c", False, True)):
        c_dtype = "bf16" if c_bf16 else "f32"
        mk = markers_for(c_dtype, poison)
        ex = [] if poison else ["poison_acc_0_0"]
        purge(mk, ex, COLS, f"prebuild/{label}")
        _ = dispatch(A, B, poison, c_bf16)
        d = find_cache(mk, ex, COLS, f"prebuild/{label}")
        print(f"  [{label}] cache dir = {d}")
        mlir_text = (d / "aie.mlir").read_text(encoding="utf-8", errors="ignore")
        has_poison_marker = "poison_acc_0_0" in mlir_text
        print(f"  [{label}] 'poison_acc_0_0' in aie.mlir: {has_poison_marker} "
              f"(expected {poison})")
        assert has_poison_marker == poison, "cache identity check FAILED"

        peano = Path(os.environ.get("PEANO_INSTALL_DIR", ""))
        dumps = {}
        if peano:
            peano_bin = peano / "bin"
            for obj in sorted(d.glob("*.o")):
                out_txt = ARTIFACTS / f"objdump_{label}_{obj.stem}.txt"
                err = objdump(peano_bin, obj, out_txt)
                if err:
                    print(f"    {obj.name}: {err}")
                    continue
                text = out_txt.read_text(encoding="utf-8")
                n_lines = len(text.splitlines())
                crrnd_hits = len(re.findall(r"crrnd|\brnd\b|round", text, re.I))
                print(f"    {obj.name} -> {out_txt.name} ({n_lines} lines, "
                      f"{crrnd_hits} rnd/round/crrnd matches)")
                dumps[obj.name] = {"lines": n_lines, "rnd_matches": crrnd_hits}
        else:
            print("  PEANO_INSTALL_DIR not set -- skipping disassembly")

        results[label] = {"cache_dir": str(d), "has_poison_marker": has_poison_marker,
                           "objdump": dumps}

    Path(ARTIFACTS / "prebuild_verify.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {ARTIFACTS / 'prebuild_verify.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prebuild", action="store_true",
                     help="one-time: force-build both designs, verify cache "
                          "identity by contents, objdump kernel objects")
    ap.add_argument("--mode", choices=["cold", "warm", "bf16c", "plain_check"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.prebuild:
        prebuild_and_verify()
        return 0

    if args.mode is None or args.out is None:
        raise SystemExit("--mode and --out are required unless --prebuild")

    iron.set_current_device(from_name("npu2", n_cols=None))

    rng = np.random.default_rng(args.seed)
    A = randn_bf16(rng, (M, K))
    B = randn_bf16(rng, (K, N))
    ref = A.astype(np.float64) @ B.astype(np.float64)

    if args.mode == "cold":
        poison, c_bf16, rtp = True, False, True
        got = dispatch(A, B, poison, c_bf16, rtp)
        n_dispatches = 1
    elif args.mode == "warm":
        poison, c_bf16, rtp = True, False, True
        wrng = np.random.default_rng(args.seed + 999_000)
        A_w, B_w = randn_bf16(wrng, (M, K)), randn_bf16(wrng, (K, N))
        _ = dispatch(A_w, B_w, poison, c_bf16, rtp)   # throwaway, discarded
        got = dispatch(A, B, poison, c_bf16, rtp)     # measured
        n_dispatches = 2
    elif args.mode == "bf16c":
        poison, c_bf16, rtp = False, True, True
        wrng = np.random.default_rng(args.seed + 999_000)
        A_w, B_w = randn_bf16(wrng, (M, K)), randn_bf16(wrng, (K, N))
        _ = dispatch(A_w, B_w, poison, c_bf16, rtp)   # throwaway, discarded
        got = dispatch(A, B, poison, c_bf16, rtp)     # measured
        n_dispatches = 2
    else:  # plain_check
        poison, c_bf16, rtp = False, False, False
        got = dispatch(A, B, poison, c_bf16, rtp)
        n_dispatches = 1

    rf = rel_fro(got, ref)
    omc = one_minus_cos(got, ref)
    result = {"mode": args.mode, "seed": args.seed, "n_dispatches": n_dispatches,
              "rel_fro": rf, "1mcos": omc,
              "shape": {"M": M, "K": K, "N": N, "cols": COLS}}
    print(json.dumps(result))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
