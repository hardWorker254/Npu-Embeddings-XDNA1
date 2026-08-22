# NpuEmbeddings -- T28 Del B / B1 (tasks/0054): epilogue="gelu" at REAL shapes.
# SPDX-License-Identifier: Apache-2.0
#
# 0030 (notes/0005 section 2) proved the fused ffn_up+GELU epilogue mechanism
# on ONE synthetic shape: M=1024, K=384->448 (K-augmented), N=1536, tile
# (64,64,48), 2 columns -- rel_fro 3.167e-04, ~1.35x faster than the separate
# GEMM+GELU-dispatch path. That is MiniLM/bge-small's ffn_up shape (they share
# every GEMM shape -- CLAUDE.md's model table). This script extends the SAME
# probe (experiments/m7-switch-cost/gelu_fusion_probe.py's pattern,
# generalised over shape/tile_n) to bge-base (h=768) and bge-large (h=1024),
# which 0030 never touched.
#
# THREE THINGS MEASURED PER SHAPE, in order (accuracy before performance, per
# tasks/README's convention):
#   1. ACCURACY -- rel_fro of the fused NPU output against an exact-erf
#      gelu(A@B + bias) reference computed at the datapath's own precision
#      (bf16-quantised inputs, fp32 accumulate -- the SAME quantisation the
#      device actually stores, not a device read-back per CLAUDE.md trap 6c).
#   2. HARDWARE TRACE (per-core cycles) for the MiniLM/bge-small shape only,
#      fused vs plain -- CLAUDE.md rule 1: an array-compute claim needs a
#      trace, not a wall-clock number. Traced at cols=4 (trap 7: 8 columns is
#      fully packed and untraceable; 4 is the documented traceable width).
#   3. DISPATCH LATENCY (aie.utils.benchmark.run_iters) for fused vs the
#      plain (unaugmented) GEMM alone, at all three shapes -- LABELLED as
#      wall-clock-derived host-side timing around kernel.wait(), never
#      presented as a hardware trace (docs/05-measurement; bench_one()'s own
#      docstring in gemm_pretiled.py makes the same distinction). This isolates
#      the array-side cost of K-augmentation + the epilogue kernel; it does
#      NOT attempt to re-measure the full separate-path host round trip
#      (readback + host GELU + reconvert) -- CLAUDE.md's own "Current state"
#      already carries that number (0044: ~70.7 ms/encode at batch 128,
#      "read out + bias" 18.8% of wall clock) and this session does not
#      re-derive it.
#
# A fresh JIT cache candidate is purged before EVERY build matching that
# build's exact runtime_sequence signature (M*K, K*N, M*N, dtypes) -- 0053's
# bug (matching on TILE dims instead of per-call SHAPE, so purge() silently
# purged nothing) and 0045's marker collision (two shapes, same three sizes,
# purge() for one deleted the other) are both instances of the class this
# guards against. Purging is safe to over-trigger (costs a recompile, never a
# wrong result); the failure mode that bit prior sessions was UNDER-triggering.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python epilogue_gelu_survey.py --out artifacts\epilogue_gelu_survey.json
#   python epilogue_gelu_survey.py --shapes minilm --no-trace   # fast smoke test

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))

import aie.iron as iron                               # noqa: E402
from aie.iron.device import from_name                 # noqa: E402
from aie.utils.trace import TraceConfig                # noqa: E402
from aie.utils.benchmark import run_iters               # noqa: E402
from gemm_pretiled import pretiled_array                # noqa: E402
from npue import tile_b, to_bf16_bits                   # noqa: E402

CACHE = Path.home() / ".npu" / "cache"

# The four shipped models' ffn_up shape, at batch 1024 tokens (16 sequences x
# seq 64 -- 0030's own M, kept for continuity with its numbers). K_REAL = h,
# N = 4h.  tile_n is NOT a constant: 48 for the h in {384, 768} where every N
# is a multiple of 384 (0042/0051), 32 for bge-large where 48 is illegal
# (N=4096, 4096/(48*8) is not integral even at 8 cols; 32 is the largest
# legal choice that still fits L1).
SHAPES = {
    "minilm_bge_small": dict(h=384, tile_n=48,
                              note="MiniLM-L6 and bge-small-en-v1.5 share "
                                   "every GEMM shape (CLAUDE.md model table) "
                                   "-- this IS 0030's own probe shape, "
                                   "reproduced here rather than re-derived"),
    "bge_base":  dict(h=768,  tile_n=48,
                       note="every N a multiple of 384 (0051) so tile_n "
                            "stays 48, same as MiniLM"),
    "bge_large": dict(h=1024, tile_n=32,
                       note="N=4096 illegal at tile_n=48 (0042); 32 is the "
                            "largest legal value that fits L1"),
}

M = 1024
TM, TK = 64, 64
COLS = 4  # traceable (trap 7), and matches the trace-comparison's needs


def gelu_exact(x):
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / np.sqrt(2.0)))


def markers_for(M, K, N, c_dtype="f32"):
    """Match the ORDERED runtime_sequence signature, not three loose memref
    strings -- 0045's exact collision class (see gemm_pretiled.py's own
    comment on this, and note 0005's fifth fail-open)."""
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}x{c_dtype}>)"]


def purge(markers, what):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers):
            shutil.rmtree(d)
            n += 1
    print(f"  [{what}] purged {n} cache candidate(s)")


def check_accuracy(name, h, tile_n, seed=7):
    """Fused epilogue vs exact-erf gelu(A@B+bias) at datapath precision."""
    K_real, K_aug, N = h, h + TK, 4 * h
    rng = np.random.default_rng(seed)
    a_real = (rng.standard_normal((M, K_real)) * 0.5).astype(np.float32)
    b_real = (rng.standard_normal((K_real, N)) * 0.05).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)

    a_aug = np.zeros((M, K_aug), np.float32)
    a_aug[:, :K_real] = a_real
    a_aug[:, K_real] = 1.0
    b_aug = np.zeros((K_aug, N), np.float32)
    b_aug[:K_real] = b_real
    b_aug[K_real] = bias

    a16 = a_aug.astype(bfloat16).astype(np.float32)
    b16 = b_aug.astype(bfloat16).astype(np.float32)
    want = gelu_exact(a16 @ b16)

    markers = markers_for(M, K_aug, N)
    purge(markers, f"{name} accuracy (fused, K_aug={K_aug})")

    A = iron.zeros((M, K_aug), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b_aug), TK, tile_n, 8, 8)
    B = iron.zeros((K_aug, N), dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = a_aug.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K_aug, N)
    assert np.array_equal(A.numpy(), a_aug.astype(bfloat16)), "A sync"
    assert np.array_equal(B.numpy(), Bt.view(bfloat16).reshape(K_aug, N)), "B sync"

    pretiled_array(A, B, C, M=M, K=K_aug, N=N, m=TM, k=TK, n=tile_n,
                    n_aie_cols=COLS, dtype_in_str="bf16", dtype_out_str="f32",
                    emulate_bf16_mmul_with_bfp16=False, pretiled=True,
                    trace_config=None, epilogue="gelu")

    got = C.numpy().reshape(M, N).astype(np.float64)
    finite = np.isfinite(got).all()
    rel_fro = worst = float("nan")
    if finite:
        rel_fro = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        worst = float(np.abs(got - want).max())
    ok = finite and rel_fro < 1.5e-2
    print(f"  [{name}] fused gelu(A@B+bias): rel_fro={rel_fro:.3e} "
          f"worst_abs={worst:.3e} {'PASS' if ok else 'FAIL'}  "
          f"(K_real={K_real} K_aug={K_aug} N={N} tile_n={tile_n})")
    return dict(shape=name, h=h, tile_n=tile_n, K_real=K_real, K_aug=K_aug,
                N=N, M=M, cols=COLS, finite=bool(finite),
                rel_frobenius=rel_fro, worst_abs=worst, correctness_pass=ok)


def dispatch_latency(name, h, tile_n, iters=30, warmup=8):
    """run_iters-based comparison: fused (K-augmented+epilogue) vs plain GEMM
    alone (K real, no epilogue), same M/N/cols. LABELLED wall-clock-derived
    dispatch latency (host timer around kernel.wait()) -- NOT a hardware
    trace. See module docstring point 3."""
    K_real, K_aug, N = h, h + TK, 4 * h
    out = {}
    for tag, K, epi in (("fused", K_aug, "gelu"), ("plain", K_real, None)):
        markers = markers_for(M, K, N)
        purge(markers, f"{name} dispatch-latency ({tag}, K={K})")

        A = iron.rand((M, K), dtype=bfloat16, device="npu")
        B = iron.rand((K, N), dtype=bfloat16, device="npu")
        C = iron.zeros(M * N, dtype=np.float32, device="npu")
        r, s, t = 8, 8, 8  # bf16 mac_dims on npu2, matched by tile_b below
        B[:] = tile_b(B.numpy().copy().view(np.uint16), TK, tile_n, s, t
                      ).view(bfloat16).reshape(K, N)

        kw = dict(M=M, K=K, N=N, m=TM, k=TK, n=tile_n, n_aie_cols=COLS,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=False, pretiled=True,
                   trace_config=None, epilogue=epi)
        res = run_iters(pretiled_array, A, B, C, warmup=warmup, iters=iters, **kw)
        npu = res.npu
        e2e = res.e2e
        print(f"  [{name}] {tag:<6} K={K:<5} npu(avg/min)="
              f"{(npu.avg_us if npu else float('nan')):7.1f}/"
              f"{(npu.min_us if npu else float('nan')):7.1f} us   "
              f"e2e(avg/min)={e2e.avg_us:7.1f}/{e2e.min_us:7.1f} us")
        out[tag] = dict(K=K, npu_avg_us=npu.avg_us if npu else None,
                         npu_min_us=npu.min_us if npu else None,
                         e2e_avg_us=e2e.avg_us, e2e_min_us=e2e.min_us,
                         iters=iters, warmup=warmup)
    if out["fused"]["npu_avg_us"] and out["plain"]["npu_avg_us"]:
        out["fused_over_plain_npu_avg"] = (out["fused"]["npu_avg_us"]
                                            / out["plain"]["npu_avg_us"])
    return dict(shape=name, h=h, tile_n=tile_n, M=M, N=N, cols=COLS, **out)


def traced_cycles(name, h, tile_n, trace_size=262144):
    """Hardware trace (per-core cycles), fused vs plain -- the rule-1-
    compliant number. cols=4 -> trace routing (trace_row=0, trace_col=0,
    egress_shim_col=0) per TRACE_ROUTING in gemm_pretiled.py."""
    from gemm_pretiled import TRACE_ROUTING
    from aie.utils.trace.utils import get_cycles_summary
    K_real, K_aug, N = h, h + TK, 4 * h
    tcol, egress = TRACE_ROUTING[COLS]
    out = {}
    for tag, K, epi in (("fused", K_aug, "gelu"), ("plain", K_real, None)):
        markers = markers_for(M, K, N)
        purge(markers, f"{name} trace ({tag}, K={K})")

        A = iron.rand((M, K), dtype=bfloat16, device="npu")
        B = iron.rand((K, N), dtype=bfloat16, device="npu")
        C = iron.zeros(M * N, dtype=np.float32, device="npu")
        B[:] = tile_b(B.numpy().copy().view(np.uint16), TK, tile_n, 8, 8
                      ).view(bfloat16).reshape(K, N)

        trace_txt = ARTIFACTS / f"trace_epi_{name}_{tag}.txt"
        trace_json = ARTIFACTS / f"trace_epi_{name}_{tag}.json"
        mlir_copy = ARTIFACTS / f"mlir_epi_{name}_{tag}.mlir"
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cfg = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt))

        pretiled_array(A, B, C, M=M, K=K, N=N, m=TM, k=TK, n=tile_n,
                        n_aie_cols=COLS, dtype_in_str="bf16",
                        dtype_out_str="f32",
                        emulate_bf16_mmul_with_bfp16=False, pretiled=True,
                        trace_config=cfg, trace_row=0, trace_col=tcol,
                        trace_egress_col=egress, epilogue=epi)

        size = trace_txt.stat().st_size if trace_txt.exists() else 0
        if size == 0:
            print(f"  [{name}] {tag}: EMPTY TRACE")
            out[tag] = {"trace": "empty", "K": K}
            continue
        phys = getattr(cfg, "physical_mlir_path", None)
        if phys:
            shutil.copy(phys, mlir_copy)
        elif mlir_copy.exists():
            phys = str(mlir_copy)
        if phys is None:
            print(f"  [{name}] {tag}: no physical MLIR (cache hit, no stored copy)")
            out[tag] = {"trace": "no-mlir", "K": K}
            continue
        cfg.trace_to_json(phys, str(trace_json))
        deltas = []
        for entry in get_cycles_summary(str(trace_json)):
            deltas += [d for d in entry[1:] if d is not None]
        if not deltas:
            print(f"  [{name}] {tag}: no event0/event1 pairs")
            out[tag] = {"trace": "no-events", "K": K}
            continue
        avg = sum(deltas) / len(deltas)
        print(f"  [{name}] {tag:<6} K={K:<5} traced: n={len(deltas):>4} "
              f"avg={avg:8.1f} cyc  min={min(deltas)}  max={max(deltas)}")
        out[tag] = dict(K=K, invocations=len(deltas), avg_cycles=avg,
                         min_cycles=min(deltas), max_cycles=max(deltas))
    if (isinstance(out.get("fused"), dict) and "avg_cycles" in out["fused"]
            and isinstance(out.get("plain"), dict) and "avg_cycles" in out["plain"]):
        out["fused_over_plain_avg_cycles"] = (out["fused"]["avg_cycles"]
                                               / out["plain"]["avg_cycles"])
    return dict(shape=name, h=h, tile_n=tile_n, M=M, N=N, cols=COLS, **out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="minilm_bge_small,bge_base,bge_large",
                     help="comma-separated subset of " + ",".join(SHAPES))
    ap.add_argument("--no-trace", action="store_true",
                     help="skip the hardware-trace comparison (accuracy + "
                          "dispatch-latency only, all shapes)")
    ap.add_argument("--trace-shapes", default="minilm_bge_small",
                     help="which shapes to trace (traces are slow to build)")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))

    wanted = [s.strip() for s in args.shapes.split(",") if s.strip()]
    trace_wanted = set(s.strip() for s in args.trace_shapes.split(",") if s.strip())
    results = {"accuracy": [], "dispatch_latency": [], "traced_cycles": []}

    for name in wanted:
        h, tile_n = SHAPES[name]["h"], SHAPES[name]["tile_n"]
        print(f"\n=== {name}  h={h}  tile_n={tile_n} ===")
        results["accuracy"].append(check_accuracy(name, h, tile_n))
        if not args.no_latency:
            results["dispatch_latency"].append(
                dispatch_latency(name, h, tile_n, iters=args.iters))
        if not args.no_trace and name in trace_wanted:
            results["traced_cycles"].append(traced_cycles(name, h, tile_n))

    if args.out:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
