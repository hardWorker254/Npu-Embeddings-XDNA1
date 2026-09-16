# NpuEmbeddings -- M5: what does bfp16 cost on REAL activations? On hardware.
# SPDX-License-Identifier: Apache-2.0
#
# The question, and why it has taken this long
# -------------------------------------------
# M2 measured --emulate-bf16-mmul-with-bfp16 at 5.5x throughput for a relative
# Frobenius error of 1.040e-02. That was on iron.rand uniform [0,1) inputs, and
# CLAUDE.md has carried the caveat ever since: uniform data may not represent
# what a real encoder feeds a GEMM.
#
# M3 answered it in SIMULATION and got an unwelcome answer -- real activations
# were 6.0x WORSE than uniform (5.05e-02 vs 8.42e-03), not better, because
# post-LayerNorm BERT has outlier dimensions that set the shared block exponent
# and crush the other seven elements. But that rested on a bfp16 model fitted to
# a single hardware number, and the fit itself only landed after the first guess
# missed by 19x.
#
# This settles it on the device. Same GEMM, same kernel, same flag, four input
# sets, on real hardware:
#
#   uniform [0,1)          x  uniform [0,1)        <- M2's conditions
#   real activations       x  real trained weights <- what the model does
#   real activations       x  uniform
#   uniform                x  real trained weights <- which operand carries it
#
# The last two are the part the simulation never separated: block floating point
# quantises BOTH operands along K, and post-LN outliers live in the activations,
# not the weights. If the damage is one-sided, that is worth knowing -- it would
# mean bfp16 is usable for weights and not for activations.
#
# Every run is also done with the flag OFF, which is the control: bf16 in with
# fp32 accumulate should give ~1e-7 regardless of distribution, and if it does
# not, the harness rather than the format is at fault.
#
# PROCESS ISOLATION -- not optional, see tasks/0008
# ------------------------------------------------
# Every measurement runs in its own subprocess. The first version of this file
# did not, and its control column read 6.66e+01 where bf16 + fp32 accumulate must
# read ~1e-7. That was the harness, not the format: once TWO different compiled
# designs have been dispatched in one process WITHOUT tracing, every dispatch
# after them returns garbage -- including repeats of a design that had just
# worked. Correctness is silently wrong; nothing raises.
#
#   run0 emulate=False relfro=1.723e-07     <- correct
#   run1 emulate=True  relfro=9.015e-03     <- correct
#   run2 emulate=True  relfro=6.833e+01     <- same design as run1, garbage
#
# So each configuration gets a fresh interpreter. Slow and worth it: this script
# exists to produce a number that decides whether bfp16 is usable at all.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python experiments\m5-pretiled-gemm\bfp16_real_activations.py

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference"))

from safetensors_io import load                     # noqa: E402

# M3's numbers, to compare the hardware against. From
# reference/goldens/precision_study.json (a SIMULATION, fitted to M2).
SIM_UNIFORM = 8.416e-03
SIM_REAL = 5.052e-02
SIM_RATIO = 6.00


def bf16_round(x):
    """fp32 -> bf16 values, via the device's own dtype."""
    return np.asarray(x, dtype=np.float32).astype(bfloat16)


def block_dynamic_range(x, block=8, axis=-1):
    """Median within-block max/min along the reduction axis.

    This is the quantity block floating point actually cares about: one large
    value sets the shared exponent for its whole block. Reported alongside the
    error so the mechanism is visible in the same table, not just asserted.
    """
    x = np.abs(np.asarray(x, np.float64))
    x = np.moveaxis(x, axis, -1)
    n = (x.shape[-1] // block) * block
    b = x[..., :n].reshape(*x.shape[:-1], -1, block)
    return float(np.median(b.max(axis=-1) / np.maximum(b.min(axis=-1), 1e-30)))


def real_inputs(M, K, N):
    """Real post-LayerNorm activations and real trained weights.

    A is the M3 golden `hf.emb.ln` -- the actual input to layer 0's QKV GEMM,
    produced by HuggingFace and verified against our oracle to 8.5e-08.
    B is the fused Q,K,V weight, exactly as M4 packs it.
    """
    g, meta = load(REPO / "reference" / "goldens"
                   / "minilm_l6_s64_boundary.safetensors")
    w, _ = load(REPO / "models" / "all-MiniLM-L6-v2" / "model.safetensors")

    a = g["hf.emb.ln"].reshape(-1, g["hf.emb.ln"].shape[-1])      # [256, 384]
    b = np.ascontiguousarray(np.concatenate(
        [w[f"encoder.layer.0.attention.self.{n}.weight"]
         for n in ("query", "key", "value")], axis=0).T)          # [384, 1152]
    if a.shape != (M, K) or b.shape != (K, N):
        raise SystemExit(f"golden shapes {a.shape} x {b.shape} != {(M,K)} x {(K,N)}; "
                         f"this experiment is pinned to layer 0 QKV")
    return a, b, meta["source_sha256"]


SETS = ["uniform_uniform", "real_real", "real_uniform", "uniform_real"]


def build_set(name, seed, M, K, N):
    """The four input pairs. Built identically in parent and child."""
    a_real, b_real, sha = real_inputs(M, K, N)
    rng = np.random.default_rng(seed)
    a_unif = rng.random((M, K), dtype=np.float32)
    b_unif = rng.random((K, N), dtype=np.float32)
    pair = {"uniform_uniform": (a_unif, b_unif),
            "real_real": (a_real, b_real),
            "real_uniform": (a_real, b_unif),
            "uniform_real": (a_unif, b_real)}[name]
    return bf16_round(pair[0]), bf16_round(pair[1]), sha


def run(A_np, B_np, M, K, N, m, k, n, cols, emulate):
    """One hardware GEMM on given values. Returns relative Frobenius error.

    Only ever called in a child process, exactly once per process.
    """
    import aie.iron as iron
    from aie.iron import str_to_dtype
    from aie.iron.device import from_name
    from gemm_pretiled import pretiled_array

    iron.set_current_device(from_name("npu1", n_cols=None))
    dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
    A = iron.rand((M, K), dtype=dt_in, device="npu")
    B = iron.rand((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")

    # Write through Tensor.__setitem__, NOT through .numpy().
    #
    # `A.numpy()` syncs FROM the device and hands back the host buffer; writing
    # into that array never syncs back, so the device keeps running on whatever
    # was there before. `A[:] = x` syncs both ways. The first dispatch after
    # construction happens to be correct either way, which is exactly what makes
    # the wrong form so easy to ship. See tasks/0009.
    A[:] = A_np
    B[:] = B_np

    # rowmajor B: M5 showed pre-tiling is a wash, so use the faster, stable path.
    pretiled_array(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=emulate, pretiled=False,
                   trace_config=None)

    got = C.numpy().reshape(M, N).astype(np.float64)
    # Reference against the values we INTENDED to send, not against a read-back.
    # A read-back re-syncs from the device, so if the write never landed the
    # reference would silently agree with the wrong inputs and the check would
    # pass while measuring nothing. Inputs are already bf16, so this counts only
    # what the MAC path does.
    ref = A_np.astype(np.float64) @ B_np.astype(np.float64)
    assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
    assert np.array_equal(B.numpy(), B_np), "B did not reach the device"
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def child(args) -> int:
    """One configuration, one process, one dispatch. Prints a JSON line."""
    M, K, N = 256, 384, 1152
    a16, b16, sha = build_set(args.set, args.seed, M, K, N)
    rel = run(a16, b16, M, K, N, args.m, args.k, args.n, args.cols, args.emulate)
    print("RESULT " + json.dumps({
        "set": args.set, "emulate": args.emulate, "rel_fro": rel,
        "A_median_block_dynamic_range":
            block_dynamic_range(a16.astype(np.float32), axis=1),
        "B_median_block_dynamic_range":
            block_dynamic_range(b16.astype(np.float32), axis=0),
        "source_sha256": sha,
    }))
    return 0


def spawn(setname, emulate, args):
    """Run one configuration in a fresh interpreter and parse its result."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--child", "--set", setname,
           "-m", str(args.m), "-k", str(args.k), "-n", str(args.n),
           "--cols", str(args.cols), "--seed", str(args.seed)]
    if emulate:
        cmd.append("--emulate")
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise SystemExit(f"child failed for {setname} emulate={emulate}:\n"
                     f"{res.stdout[-800:]}\n{res.stderr[-800:]}")


LABELS = {
    "uniform_uniform": "uniform x uniform      (M2's conditions)",
    "real_real":       "real act x real weight (the model)",
    "real_uniform":    "real act x uniform",
    "uniform_real":    "uniform  x real weight",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--set", choices=SETS, help=argparse.SUPPRESS)
    ap.add_argument("--emulate", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(ARTIFACTS / "bfp16_real_activations.json"))
    args = ap.parse_args()

    if args.child:
        return child(args)

    M, K, N = 256, 384, 1152
    print(f"hardware bfp16 on real data -- layer 0 QKV, {M}x{K}x{N}, "
          f"{args.cols} cols, tile ({args.m},{args.k},{args.n})")
    print("one fresh process per measurement (see the header of this file)\n")
    print(f"  {'inputs':<42} {'bf16+f32':>10} {'bfp16':>10} {'A blk':>7} {'B blk':>7}")

    rows, sha = [], None
    for name in SETS:
        off = spawn(name, False, args)
        on = spawn(name, True, args)
        sha = off["source_sha256"]
        rows.append({"inputs": LABELS[name], "set": name,
                     "bf16_f32_rel_fro": off["rel_fro"],
                     "bfp16_rel_fro": on["rel_fro"],
                     "A_median_block_dynamic_range":
                         off["A_median_block_dynamic_range"],
                     "B_median_block_dynamic_range":
                         off["B_median_block_dynamic_range"]})
        r = rows[-1]
        flag = "" if r["bf16_f32_rel_fro"] < 1e-5 else "   <-- CONTROL FAILED"
        print(f"  {LABELS[name]:<42} {r['bf16_f32_rel_fro']:>10.2e} "
              f"{r['bfp16_rel_fro']:>10.2e} "
              f"{r['A_median_block_dynamic_range']:>7.1f} "
              f"{r['B_median_block_dynamic_range']:>7.1f}{flag}")

    bad = [r for r in rows if r["bf16_f32_rel_fro"] >= 1e-5]
    if bad:
        print(f"\nABORT -- the bf16 control must be ~1e-7 and is not for "
              f"{len(bad)} set(s). The harness is wrong; the numbers mean nothing.")
        return 1

    by = {r["set"]: r["bfp16_rel_fro"] for r in rows}
    unif, real = by["uniform_uniform"], by["real_real"]
    ratio = real / unif
    verdict = ("WORSE" if ratio > 1.15 else
               "BETTER" if ratio < 0.87 else "COMPARABLE")

    print(f"\n  real / uniform            : {ratio:>6.2f}x   ({verdict})")
    print(f"  real activations only     : {by['real_uniform']/unif:>6.2f}x")
    print(f"  real weights only         : {by['uniform_real']/unif:>6.2f}x")
    print(f"\n  M2 hardware, uniform      : 1.040e-02   (this run: {unif:.3e})")
    print(f"  M3 simulation predicted   : {SIM_RATIO:.2f}x worse on real data")
    agree = 0.5 <= ratio / SIM_RATIO <= 2.0
    print(f"  simulation vs hardware    : predicted {SIM_RATIO:.2f}x, "
          f"measured {ratio:.2f}x  -> {'CONFIRMED' if agree else 'NOT CONFIRMED'}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "kind": "hardware measurement",
        "shape": {"M": M, "K": K, "N": N, "m": args.m, "k": args.k,
                  "n": args.n, "cols": args.cols},
        "source_sha256": sha,
        "process_isolated": True,
        "rows": rows,
        "ratios": {"real_over_uniform": ratio,
                   "real_activations_only": by["real_uniform"] / unif,
                   "real_weights_only": by["uniform_real"] / unif},
        "verdict": verdict,
        "m2_hardware_uniform": 1.040e-02,
        "m3_simulation": {"uniform": SIM_UNIFORM, "real": SIM_REAL,
                          "predicted_ratio": SIM_RATIO,
                          "confirmed_by_hardware": agree},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
