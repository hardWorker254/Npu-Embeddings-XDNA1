# NpuEmbeddings -- research driver for the pre-tiled bf16 GEMM design.
# SPDX-License-Identifier: Apache-2.0
#
# The design itself is the library in gemm_pretiled.py. This file keeps the
# research harness that was split out of it: the four MiniLM presets, the
# traced correctness run, the wall-clock benchmark, and the CLI. It targets
# either NPU generation from one switch; the device is set before any design
# is compiled, exactly as tools/export_gemm_rtp.py does.
#
# Usage (from a shell where the iron environment is active):
#     python tools/gemm_pretiled_research.py --preset ffn_down --cols 4 -n 48
#     python tools/gemm_pretiled_research.py --all-shapes --cols 4 -n 48
#     python tools/gemm_pretiled_research.py --preset ffn_down --arch 2 --cols 8
#     python tools/gemm_pretiled_research.py --preset ffn_down --bench

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import kernels, str_to_dtype
from aie.iron.device import from_name
from aie.utils.trace import TraceConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"
sys.path.insert(0, str(HERE))

from gemm_pretiled import pretiled_array       # noqa: E402
from npue import tile_b, untile_b               # noqa: E402

# The four MiniLM GEMMs. M=256 is one sequence at the model's real max length.
PRESETS = {
    "qkv":      dict(M=256, K=384,  N=1152),   # fused Q,K,V projection
    "proj":     dict(M=256, K=384,  N=384),    # attention output projection
    "ffn_up":   dict(M=256, K=384,  N=1536),   # FFN up
    "ffn_down": dict(M=256, K=1536, N=384),    # FFN down -- inexpressible in M2
}

# From tasks/0004: adding one trace flow exhausts routing at most widths.
TRACE_ROUTING = {2: (1, 1), 4: (0, 0)}

# The same override tools/export_gemm_rtp.py uses, so one environment variable
# retargets both the research driver and the exporter.
ARCH_DEVICES = {
    "1": os.environ.get("NPU_ARCH1_DEVICE", "npu1"),
    "2": os.environ.get("NPU_ARCH2_DEVICE", "npu2"),
}


def resolve_device(arch: str | None, dev: str | None) -> str:
    if dev:
        return dev
    return ARCH_DEVICES[arch or "1"]


def run_one(M, K, N, m, k, n, cols, emulate, trace_size, pretiled=True,
            tile_order="k,n", inner_st=True,
            dtype_in="bf16", dtype_out="f32", verbose=True, trace=True,
            b_l1_depth=2):
    """Compile + run one configuration. Returns a result dict, or None."""
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)

    in_sz, out_sz = np.dtype(dt_in).itemsize, np.dtype(dt_out).itemsize
    # Stationary-B budget when B is single-buffered (b_l1_depth=1):
    # 2mk + kn + 2mn instead of 2(mk + kn + mn). Trap 3 / ICPP'25.
    l1 = (2 * m * k * in_sz + b_l1_depth * k * n * in_sz
          + 2 * m * n * out_sz)
    if l1 >= 64 * 1024:
        print(f"  SKIP cols={cols}: tile needs {l1} B of L1 (max 65536)")
        return None

    # Display label vs filesystem tag are deliberately separate: '|' and '[' are
    # invalid in Windows filenames, and the failure mode is an OSError from deep
    # inside the trace writer AFTER the kernel has already run.
    if pretiled:
        kind = f"pretiled[{tile_order}|{'st' if inner_st else 'rowmaj'}]"
        slug = f"pretiled_{tile_order.replace(',', '')}_{'st' if inner_st else 'rowmaj'}"
    else:
        kind = slug = "rowmajor"
    if b_l1_depth != 2:
        kind += f"+bd{b_l1_depth}"
        slug += f"_bd{b_l1_depth}"
    tag = (f"{slug}_{cols}c_{dtype_in}_{dtype_out}{'_bfp16' if emulate else ''}"
           f"_{M}x{K}x{N}_t{m}x{k}x{n}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trace_txt = ARTIFACTS / f"trace_{tag}.txt"
    trace_json = ARTIFACTS / f"trace_{tag}.json"
    mlir_copy = ARTIFACTS / f"mlir_{tag}.mlir"

    # INTEGER dtypes need their own generator (tasks/0077). `iron.rand` draws
    # floats in [0,1) and casts, which for i8 gives an ALL-ZERO operand -- the
    # reference norm is then 0 and rel_fro comes out `nan`, i.e. the harness
    # reports a failure that is entirely its own. Caught while probing T20.
    if np.issubdtype(np.dtype(dt_in), np.integer):
        _rng = np.random.default_rng(7)
        _lim = np.iinfo(np.dtype(dt_in)).max
        A_np = _rng.integers(-_lim, _lim + 1, size=(M, K)).astype(dt_in)
        B_logical = _rng.integers(-_lim, _lim + 1, size=(K, N)).astype(dt_in)
        A = iron.zeros((M, K), dtype=dt_in, device="npu")
        B = iron.zeros((K, N), dtype=dt_in, device="npu")
        C = iron.zeros(M * N, dtype=dt_out, device="npu")
        # Tensor.__setitem__, never .numpy()[:] -- CLAUDE.md trap 6b.
        A[:] = A_np
        B[:] = B_logical
        assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
    else:
        A = iron.rand((M, K), dtype=dt_in, device="npu")
        B = iron.rand((K, N), dtype=dt_in, device="npu")
        C = iron.zeros(M * N, dtype=dt_out, device="npu")
        A_np = A.numpy().copy()
        B_logical = B.numpy().copy()      # the mathematical [K,N] operand

    if pretiled:
        # Build the pre-tiled buffer with the SAME tile_b() that packs .npue,
        # so this exercises the shipped layout rather than a lookalike. The
        # device tensor's .numpy() is a live writable view, which is the only
        # way in: iron.tensor() cannot ingest an ml_dtypes bfloat16 array.
        r, s, t = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
            b_col_maj=False, c_col_maj=False, use_chess=False,
            emulate_bf16_mmul_with_bfp16=emulate, vectorized=True).mac_dims
        st = (s, t) if inner_st else (None, None)
        # ITEMSIZE-GENERAL (tasks/0077). `tile_b`/`untile_b` in tools/npue.py
        # are already dtype-agnostic -- they only reshape and transpose -- but
        # this call site viewed everything as uint16, which is a fact about
        # bf16 rather than about the layout. The view exists at all because
        # ml_dtypes' bfloat16 does not survive some numpy ops; an unsigned
        # integer of the same width does, and reinterpreting is free.
        # With i8 the old form failed as "cannot reshape array of size 73728
        # into shape (384,384)" -- exactly half of 384x384, i.e. the array read
        # as 2-byte elements.
        _uview = {1: np.uint8, 2: np.uint16, 4: np.uint32}[np.dtype(dt_in).itemsize]
        tiled = tile_b(B_logical.view(_uview), k, n, *st, order=tile_order)
        # Write through Tensor.__setitem__, not through .numpy().
        # `B.numpy()` syncs FROM the device and returns the host buffer; writing
        # into that array never syncs back, and only the first dispatch in a
        # process happens to come out right. `B[:] = x` syncs both ways.
        # See tasks/0009 -- this cost a full misdiagnosis.
        B[:] = tiled.view(dt_in).reshape(K, N)
        # Prove the permutation is invertible on exactly these bytes before
        # trusting a hardware result that depends on it.
        back = untile_b(B.numpy().reshape(-1).view(_uview), K, N, k, n, *st,
                        order=tile_order)
        assert np.array_equal(back, B_logical.view(_uview)), "tile_b round-trip failed"

    tcol, egress = TRACE_ROUTING.get(cols, (None, None))
    cfg = None
    if trace:
        if tcol is None:
            print(f"  cols={cols}: NOT TRACEABLE; traceable widths are "
                  f"{sorted(TRACE_ROUTING)}")
            return None
        cfg = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt))

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, pretiled=pretiled,
              tile_order=tile_order, inner_st=inner_st,
              b_l1_depth=b_l1_depth,
              trace_config=cfg)
    if cfg is not None:
        kw.update(trace_row=0, trace_col=tcol, trace_egress_col=egress)
    pretiled_array(A, B, C, **kw)

    got = C.numpy().reshape(M, N).astype(np.float64)
    ref = A_np.astype(np.float64) @ B_logical.astype(np.float64)
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    # An int8 x int8 -> int32 GEMM has NO rounding anywhere in the reduction,
    # so the honest gate is EXACT equality, not a tolerance -- 2608.13756's
    # "integer alibi" used as a test. A tolerance here would pass a kernel that
    # is subtly wrong. Overflow is what would break the argument, so it is
    # asserted rather than assumed.
    if np.issubdtype(np.dtype(dt_in), np.integer):
        assert K * int(np.iinfo(np.dtype(dt_in)).max) ** 2 < 2 ** 31, \
            "this K could overflow the int32 accumulator; the exactness gate " \
            "below would then be testing the wrong thing"
        ok = bool(np.array_equal(got, ref))
        tol = 0.0
    else:
        tol = 5e-2 if emulate else 5e-3
        ok = rel_fro <= tol

    out = dict(kind=kind, tile_order=tile_order if pretiled else None,
               cols=cols, cores=4 * cols, M=M, K=K, N=N, m=m, k=k, n=n,
               b_l1_depth=b_l1_depth,
               dtype_in=dtype_in, dtype_out=dtype_out, emulate_bfp16=emulate,
               rel_frobenius=rel_fro, correctness_pass=ok)

    if cfg is None:
        if verbose:
            print(f"  cols={cols:>2} {kind:<8} relfro={rel_fro:.2e} "
                  f"{'PASS' if ok else 'FAIL'} (no trace)")
        return out

    size = trace_txt.stat().st_size if trace_txt.exists() else 0
    if size == 0:
        print(f"  cols={cols}: EMPTY TRACE -- raise --trace-size")
        out["trace"] = "empty"
        return out

    # physical_mlir_path is set by the JIT only when it actually compiles. On a
    # repeat run of an identical config the cache hits, nothing is compiled, and
    # the attribute stays None -- trace_to_json then dies with
    # "expected str, bytes or os.PathLike object, not NoneType". The copy we
    # keep for offline trace regeneration doubles as the fallback.
    phys = getattr(cfg, "physical_mlir_path", None)
    if phys:
        shutil.copy(phys, mlir_copy)
    elif mlir_copy.exists():
        phys = str(mlir_copy)
    if phys is None:
        print(f"  cols={cols}: no physical MLIR (cache hit, no stored copy) -- "
              f"clear {mlir_copy.name} or the JIT cache")
        return out
    cfg.trace_to_json(phys, str(trace_json))
    from aie.utils.trace.utils import get_cycles_summary

    deltas = []
    for entry in get_cycles_summary(str(trace_json)):
        deltas += [d for d in entry[1:] if d is not None]
    if not deltas:
        print(f"  cols={cols}: no event0/event1 pairs in trace")
        return out

    avg = sum(deltas) / len(deltas)
    per_core = (m * k * n) / avg
    peak_core = {"bf16": 256, "i16": 128, "i8": 512}[dtype_in]
    out.update(invocations=len(deltas), avg_cycles=avg,
               min_cycles=min(deltas), max_cycles=max(deltas),
               macs_per_cycle_per_core=per_core,
               macs_per_cycle_array=per_core * 4 * cols,
               peak_per_core=peak_core,
               efficiency_pct=per_core / peak_core * 100.0)
    if verbose:
        print(f"  cols={cols:>2} {kind:<8} cores={4*cols:>2} n={len(deltas):>5}  "
              f"avg={avg:8.1f} cyc  per-core={per_core:6.1f} MACs/cyc "
              f"({per_core/peak_core*100:5.1f}%)  relfro={rel_fro:.2e} "
              f"{'PASS' if ok else 'FAIL'}")
    return out


def bench_one(M, K, N, m, k, n, cols, emulate, pretiled, tile_order="k,n",
              inner_st=True, iters=50, warmup=10,
              dtype_in="bf16", dtype_out="f32"):
    """Wall-clock end-to-end throughput, NO trace.

    This is the metric M4 actually claimed to improve. Per-core cycles measure
    the compute window including DMA stalls, but "the array is starved" is a
    statement about the whole dispatch, and docs/05-measurement permits wall
    clock for exactly that -- labelled, never as a kernel-cycle claim, with the
    NPU quiesced.
    """
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)
    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")

    if pretiled:
        r, s, t = kernels.mm(
            dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
            b_col_maj=False, c_col_maj=False, use_chess=False,
            emulate_bf16_mmul_with_bfp16=emulate, vectorized=True).mac_dims
        st = (s, t) if inner_st else (None, None)
        B.numpy().reshape(-1).view(np.uint16)[:] = tile_b(
            B.numpy().copy().view(np.uint16), k, n, *st, order=tile_order)

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, pretiled=pretiled,
              tile_order=tile_order, inner_st=inner_st, trace_config=None)

    from aie.utils.benchmark import run_iters
    res = run_iters(pretiled_array, A, B, C, warmup=warmup, iters=iters, **kw)
    npu_us = getattr(getattr(res, "npu", None), "avg_us", None)
    npu_min = getattr(getattr(res, "npu", None), "min_us", None)
    e2e_us = getattr(getattr(res, "e2e", None), "avg_us", None)
    total_macs = M * K * N
    out = dict(kind="pretiled" if pretiled else "rowmajor",
               tile_order=tile_order if pretiled else None,
               inner_st=inner_st if pretiled else None,
               cols=cols, M=M, K=K, N=N, m=m, k=k, n=n, iters=iters,
               npu_avg_us=npu_us, npu_min_us=npu_min, e2e_avg_us=e2e_us)
    if npu_us:
        out["tflops_npu"] = 2 * total_macs / (npu_us * 1e-6) / 1e12
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Research driver for the pre-tiled GEMM design library")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="ffn_down")
    ap.add_argument("--all-shapes", action="store_true",
                    help="run all four MiniLM GEMMs")
    ap.add_argument("-M", type=int); ap.add_argument("-K", type=int)
    ap.add_argument("-N", type=int)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--b-depth", type=int, default=2,
                    help="L1 depth of the B fifo. 1 = Stationary-B single "
                         "buffering (frees k*n*2 bytes of L1, enabling k=96; "
                         "risks losing B fetch/compute overlap -- T19)")
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--trace-size", type=int, default=262144)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="also run the M2 row-major B path as a control")
    ap.add_argument("--orders", default="k,n",
                    help="pre-tiled orders to try, ';'-separated: 'k,n;n,k'")
    ap.add_argument("--inner", default="st", choices=["st", "rowmaj", "both"],
                    help="'st' bakes the sub-tile order into the file; "
                         "'rowmaj' leaves it to dims_to_stream, isolating the "
                         "L3->L2 access-pattern change on its own")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each config; two runs of the SAME config "
                         "differed by 4.7%%, so a single number cannot support "
                         "a pretiled-vs-rowmajor claim")
    ap.add_argument("--bench", action="store_true",
                    help="wall-clock end-to-end instead of tracing")
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", choices=sorted(ARCH_DEVICES), default=None,
                    help="NPU generation to target. Default: 1 "
                         "(NPU_ARCH1_DEVICE/NPU_ARCH2_DEVICE override the "
                         "device name).")
    ap.add_argument("--dev", choices=["npu1", "npu2"], default=None,
                    help="Explicit device name; overrides --arch.")
    args = ap.parse_args()

    # Without this IRON silently compiles for NPU1 and the bfp16 flag becomes a
    # no-op. research/notes/0002. n_cols=None or it defaults to a single column.
    iron.set_current_device(
        from_name(resolve_device(args.arch, args.dev), n_cols=None))

    shapes = (list(PRESETS.items()) if args.all_shapes
              else [(args.preset, dict(PRESETS[args.preset]))])
    results = []
    for name, shape in shapes:
        for key in ("M", "K", "N"):
            if getattr(args, key) is not None:
                shape[key] = getattr(args, key)
        M, K, N = shape["M"], shape["K"], shape["N"]
        print(f"\n{name}: {M}x{K}x{N}  tile ({args.m},{args.k},{args.n})  "
              f"cols={args.cols}  bfp16={args.emulate_bfp16}")

        inners = [True, False] if args.inner == "both" else [args.inner == "st"]

        if args.bench:
            print(f"  {'variant':<22} {'npu avg us':>11} {'npu best us':>12} "
                  f"{'TFLOP/s':>9}")
            bvars = ([(False, "k,n", True)] if args.baseline else [])
            bvars += [(True, o, i) for o in args.orders.split(";") for i in inners]
            for pt, order, inner in bvars:
                b = bench_one(M, K, N, args.m, args.k, args.n, args.cols,
                              args.emulate_bfp16, pretiled=pt, tile_order=order,
                              inner_st=inner, iters=args.bench_iters)
                b["shape_name"] = name
                results.append(b)
                lbl = (f"pretiled[{order}|{'st' if inner else 'rowmaj'}]"
                       if pt else "rowmajor")
                print(f"  {lbl:<22} {b.get('npu_avg_us') or float('nan'):>11.1f} "
                      f"{b.get('npu_min_us') or float('nan'):>12.1f} "
                      f"{b.get('tflops_npu') or float('nan'):>9.2f}")
            continue
        variants = [(False, None, True)] if args.baseline else []
        variants += [(True, o, i) for o in args.orders.split(";") for i in inners]
        for kind_pretiled, order, inner in variants:
            per = []
            for rep in range(args.repeat):
                try:
                    res = run_one(M, K, N, args.m, args.k, args.n, args.cols,
                                  args.emulate_bfp16, args.trace_size,
                                  pretiled=kind_pretiled,
                                  tile_order=order or "k,n", inner_st=inner,
                                  trace=not args.no_trace,
                                  b_l1_depth=args.b_depth)
                except Exception as e:
                    msg = str(e)
                    hit = "exceeds the [0:1023] range" in msg
                    print(f"  cols={args.cols} "
                          f"{'pretiled' if kind_pretiled else 'rowmajor'} "
                          f"FAILED TO COMPILE"
                          f"{' -- BD size limit' if hit else ''}")
                    for line in msg.splitlines():
                        if "aie.dma_bd" in line or "exceeds" in line:
                            print(f"    {line.strip()[:150]}")
                    results.append({"kind": "pretiled" if kind_pretiled else "rowmajor",
                                    "tile_order": order,
                                    "shape_name": name, "cols": args.cols,
                                    "M": M, "K": K, "N": N,
                                    "compile_failed": True, "bd_limit": hit})
                    break
                if res:
                    res["shape_name"] = name
                    res["repeat"] = rep
                    results.append(res)
                    if "macs_per_cycle_per_core" in res:
                        per.append(res["macs_per_cycle_per_core"])
            if len(per) > 1:
                lo, hi, mean = min(per), max(per), sum(per) / len(per)
                label = (f"pretiled[{order}|{'st' if inner else 'rowmaj'}]"
                         if kind_pretiled else "rowmajor")
                print(f"     {label:<16} "
                      f"over {len(per)} runs: mean {mean:6.1f}  "
                      f"range {lo:.1f}-{hi:.1f}  spread {(hi-lo)/mean*100:.1f}%")

    if args.out:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
