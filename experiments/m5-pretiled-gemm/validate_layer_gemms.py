# NpuEmbeddings -- M5 GATE: every per-layer GEMM on the NPU, validated against
# the M3 goldens.
# SPDX-License-Identifier: Apache-2.0
#
# Generalises the layer-0 QKV check from tasks/0011 to all four GEMMs an encoder
# layer performs. The whole chain runs on real data with nothing re-derived:
#
#   HuggingFace  ->  M3 goldens  ->  M4 .npue  ->  NPU  ->  compare to golden
#
# INPUTS COME FROM THE TAP FILE, NOT THE BOUNDARY FILE
# ----------------------------------------------------
# `L0.ctx` and `L0.gelu` have no HuggingFace counterpart -- HF exposes no hook
# for the attention interior or the FFN interior. The taps are our reference's
# own values, and they are trustworthy because check_reference.py proves that
# reference agrees with HF at every point HF *does* expose (<= 9.9e-07 at every
# layer boundary). Using taps uniformly keeps the four checks comparable.
#
# THE SCALE FOLD
# --------------
# M3 deliberately left 1/sqrt(head_dim) unfolded so M4's fold would be provable;
# the packed weights fold it into Q. Rather than unfold the weights -- which
# would test a different program than the one we ship -- the GOLDEN is scaled
# into the space the packed pipeline produces. Only `qkv` is affected.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m5-pretiled-gemm\validate_layer_gemms.py
#   python experiments\m5-pretiled-gemm\validate_layer_gemms.py --emulate-bfp16
#   python experiments\m5-pretiled-gemm\validate_layer_gemms.py --gemm ffn_down

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

import aie.iron as iron                                    # noqa: E402
from aie.iron import str_to_dtype                          # noqa: E402
from aie.iron.device import from_name                      # noqa: E402
from aie.utils.trace import TraceConfig                    # noqa: E402

from gemm_pretiled import TRACE_ROUTING, pretiled_array    # noqa: E402
from npue import Reader                                    # noqa: E402
from safetensors_io import load                            # noqa: E402

# Tolerance depends on the number format, and conflating them is how a correct
# bfp16 run gets thrown away as a failure.
#   bf16 : M3 measured the whole 6-layer path at 2.4e-3 for L0, so one GEMM must
#          sit under that.
#   bfp16: tasks/0008 and tasks/0011 both measured ~9e-3 on real data.
# Neither leaves room to hide an actual error, which shows up as O(1).
TOL_REL_FRO = {"bf16": 5e-3, "bfp16": 2e-2}

# The four GEMMs of one encoder layer, in execution order.
GEMMS = {
    "qkv":      dict(a="emb.ln",  w="layer.0.qkv",      out="L0.qkv",
                     bias="layer.0.qkv.bias",      fold_q=True),
    "attn_out": dict(a="L0.ctx",  w="layer.0.attn_out", out="L0.attn_proj",
                     bias="layer.0.attn_out.bias", fold_q=False),
    "ffn_up":   dict(a="L0.ln1",  w="layer.0.ffn_up",   out="L0.ffn_up",
                     bias="layer.0.ffn_up.bias",   fold_q=False),
    "ffn_down": dict(a="L0.gelu", w="layer.0.ffn_down", out="L0.ffn_down",
                     bias="layer.0.ffn_down.bias", fold_q=False),
}


def run_gemm(name, spec, taps, r, cfg, args):
    hidden, head_dim = cfg["hidden"], cfg["head_dim"]
    a_f32 = taps[spec["a"]].reshape(-1, taps[spec["a"]].shape[-1])
    w = r.tensor(spec["w"])
    b = r.tensor(spec["bias"])
    M, K = a_f32.shape
    N = w.shape[1]
    if w.shape[0] != K:
        raise SystemExit(f"{name}: activation K={K} != weight K={w.shape[0]}")

    want = taps[spec["out"]].reshape(-1, N).astype(np.float64).copy()
    if spec["fold_q"] and cfg["fusions"]["qk_scale_folded_into_q"]:
        want[:, :hidden] *= 1.0 / math.sqrt(head_dim)

    dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")
    a16, b16 = a_f32.astype(bfloat16), w.astype(bfloat16)
    A[:] = a16                                   # __setitem__ syncs to device
    B[:] = b16
    assert np.array_equal(A.numpy(), a16), f"{name}: A did not reach the device"
    assert np.array_equal(B.numpy(), b16), f"{name}: B did not reach the device"

    tag = f"layer0_{name}_{args.cols}c{'_bfp16' if args.emulate_bfp16 else ''}"
    cfg_trace = None
    if not args.no_trace:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cfg_trace = TraceConfig(trace_size=262144,
                                trace_file=str(ARTIFACTS / f"trace_{tag}.txt"))
    tcol, egress = TRACE_ROUTING.get(args.cols, (0, 0))
    kw = dict(M=M, K=K, N=N, m=args.m, k=args.k, n=args.n, n_aie_cols=args.cols,
              dtype_in_str="bf16", dtype_out_str="f32",
              emulate_bf16_mmul_with_bfp16=args.emulate_bfp16,
              pretiled=False, trace_config=cfg_trace)
    if cfg_trace is not None:
        kw.update(trace_row=0, trace_col=tcol, trace_egress_col=egress)
    pretiled_array(A, B, C, **kw)

    # Bias add is elementwise and stays on the host: it is fp32 in .npue by
    # design, and M5 has not put an eltwise kernel on the array yet.
    got = C.numpy().reshape(M, N).astype(np.float64) + b.astype(np.float64)

    rel_fro = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    gn = got / np.linalg.norm(got, axis=1, keepdims=True)
    wn = want / np.linalg.norm(want, axis=1, keepdims=True)
    return {
        "gemm": name, "M": M, "K": K, "N": N,
        "rel_frobenius": rel_fro,
        "max_abs": float(np.abs(got - want).max()),
        "worst_row_1_minus_cos": float(1 - (gn * wn).sum(1).min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npue", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"))
    ap.add_argument("--gemm", choices=sorted(GEMMS), help="default: all four")
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.no_trace and args.cols not in TRACE_ROUTING:
        print(f"cols={args.cols} is not traceable; use --no-trace or "
              f"{sorted(TRACE_ROUTING)}")
        return 1

    iron.set_current_device(from_name("npu1", n_cols=None))
    _, meta = load(Path(args.goldens) / "minilm_l6_s64_boundary.safetensors")
    taps, _ = load(Path(args.goldens) / "minilm_l6_s64_taps.safetensors")

    fmt = "bfp16" if args.emulate_bfp16 else "bf16"
    tol = TOL_REL_FRO[fmt]
    out_path = Path(args.out or ARTIFACTS / f"validate_layer_gemms_{fmt}.json")

    with Reader(args.npue) as r:
        cfg = r.config
        if cfg["source_sha256"] != meta["source_sha256"]:
            print(f"FAIL -- .npue {cfg['source_sha256'][:16]} and goldens "
                  f"{meta['source_sha256'][:16]} are different checkpoints")
            return 1
        print(f"layer 0, all GEMMs on the NPU -- {fmt}, {args.cols} cols, "
              f"tile ({args.m},{args.k},{args.n})")
        print(f"  checkpoint {meta['source_sha256'][:16]}...  (both sources agree)")
        print(f"  weights straight from {Path(args.npue).name}\n")
        print(f"  {'gemm':<10} {'[M,K,N]':>18} {'rel_fro':>11} "
              f"{'max_abs':>10} {'1-cos':>10}")

        rows = []
        for name in ([args.gemm] if args.gemm else list(GEMMS)):
            res = run_gemm(name, GEMMS[name], taps, r, cfg, args)
            res["pass"] = res["rel_frobenius"] <= tol
            rows.append(res)
            print(f"  {name:<10} {str([res['M'], res['K'], res['N']]):>18} "
                  f"{res['rel_frobenius']:>11.3e} {res['max_abs']:>10.3e} "
                  f"{res['worst_row_1_minus_cos']:>10.3e} "
                  f"{'ok' if res['pass'] else 'FAIL'}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "kind": "hardware measurement", "layer": 0, "format": fmt,
        "tolerance": tol, "cols": args.cols,
        "tile": {"m": args.m, "k": args.k, "n": args.n},
        "source_sha256": meta["source_sha256"],
        "weights_from": Path(args.npue).name,
        "gemms": rows,
    }, indent=2), encoding="utf-8")

    bad = [r for r in rows if not r["pass"]]
    print(f"\n{'PASS' if not bad else 'FAIL'} -- {len(rows) - len(bad)}/{len(rows)} "
          f"GEMMs within {fmt} tolerance {tol:.0e}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
