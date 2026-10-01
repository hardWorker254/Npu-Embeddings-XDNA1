# NpuEmbeddings -- M3 extra: what does the NPU's numeric format cost the
# EMBEDDING, on real activations?
#
# Why this exists
# ---------------
# M2 measured --emulate-bf16-mmul-with-bfp16 at 5.5x throughput for ~1e5x the
# error (1.21e-07 -> 1.040e-02 relative Frobenius). But that error was measured
# on uniform [0,1) inputs, which is plausibly adversarial for a block float
# format: within a block of 8, uniform samples span an unbounded dynamic range
# as values approach zero, so the shared exponent is set by the largest and the
# smallest are crushed. CLAUDE.md carries this forward as an open question --
# "re-measure bfp16 error against realistic activation distributions".
#
# M3 is the first point where that is answerable, because we now have a
# validated fp32 oracle and therefore real post-LayerNorm activations.
#
# Two parts, and the first one is what makes the second trustworthy:
#
#   1. CALIBRATE. Reproduce M2's hardware number in simulation. We do not know
#      the exact (block, mantissa) geometry the Peano flag emits, so we sweep
#      and keep the configuration that lands on the measured 1.040e-02 under
#      M2's own conditions. A simulation that cannot reproduce the hardware
#      measurement has no standing to predict anything.
#
#   2. APPLY. Run the full encoder with that GEMM and report what actually
#      matters: cosine between the bfp16 embedding and the fp32 embedding, and
#      the distortion of the sentence-similarity matrix. Relative Frobenius on
#      an intermediate tensor is not the deliverable; retrieval behaviour is.
#
# NOTE ON STATUS: this is a SIMULATION, not a hardware measurement, and it is
# labelled as such everywhere. It predicts; M5 will measure. Per
# docs/05-measurement it may not be quoted as an NPU result.
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" reference\precision_study.py

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder import MiniLMReference          # noqa: E402
from safetensors_io import load              # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# The hardware numbers this simulation must reproduce, from M2. Source:
# experiments/m2-bf16-gemm/artifacts/result_bf16_f32{,_bfp16}_256x256x256_*.json
M2_SHAPE = (256, 256, 256)
M2_BF16_F32_RELFRO = 1.213e-07      # bf16 in, fp32 accumulate, no emulation
M2_BFP16_RELFRO = 1.040e-02         # + --emulate-bf16-mmul-with-bfp16

# M5 measured the SAME flag on the SAME hardware with REAL layer-0 QKV data:
# real activations x real weights gives 9.02e-03, against 1.06e-02 for uniform
# in the same session -- 0.85x, i.e. indistinguishable. Source:
# experiments/m5-pretiled-gemm/artifacts/bfp16_real_activations.json
#
# That refutes what this file originally concluded. Calibrating the block-float
# model on uniform data and then applying it to real activations predicted 6.0x
# WORSE; the hardware says the distribution barely matters. The model's
# sensitivity to dynamic range was its own artifact, not the datapath's
# behaviour -- so the mantissa width is now fitted against the REAL-data
# measurement, which is the distribution the encoder actually runs on.
M5_BFP16_REAL_RELFRO = 9.02e-03
M5_BFP16_UNIFORM_RELFRO = 1.057e-02


# -- number formats --------------------------------------------------------

def to_bf16(x):
    """Round fp32 to bfloat16 (round-to-nearest-even), return as fp32.

    bf16 is fp32 with the low 16 mantissa bits dropped, so this is a rounding
    of the bit pattern rather than a dtype change -- which is why the M2 GEMM
    could feed already-bf16 values and see only accumulation error.
    """
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even on bit 16
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def to_bfp(x, axis, block=8, mant_bits=8):
    """Block floating point: `block` consecutive values along `axis` share one
    exponent, each keeping 1 sign + (mant_bits-1) magnitude bits.

    The shared exponent is set by the largest magnitude in the block, so the
    error of any element is relative to the block maximum, not to itself. That
    is the whole mechanism -- and the reason a block spanning a wide dynamic
    range loses its small elements.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    n = shape[-1]
    pad = (-n) % block
    if pad:
        x = np.concatenate([x, np.zeros(shape[:-1] + (pad,), np.float32)], axis=-1)
    blocks = x.reshape(*x.shape[:-1], -1, block)

    amax = np.abs(blocks).max(axis=-1, keepdims=True)
    m = mant_bits - 1                                   # magnitude bits
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
    scale = np.exp2(e + 1 - m)                          # so (2^m-1)*scale ~ 2^(e+1)
    scale = np.where(amax > 0, scale, 1.0)

    q = np.rint(blocks / scale)
    q = np.clip(q, -(2 ** m - 1), 2 ** m - 1)
    out = (q * scale).astype(np.float32)

    out = out.reshape(*x.shape)[..., :n]
    return np.moveaxis(out, -1, axis)


def make_gemm(mode, block=8, mant_bits=8):
    """Build a GEMM primitive matching an NPU numeric path.

    Accumulation is fp32 in every mode -- M2 established that bf16 accumulation
    re-rounds at every K step and is not an option (7.4e-3 vs 1.21e-07).
    """
    if mode == "fp32":
        def g(a, b):
            return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)
    elif mode == "bf16":
        def g(a, b):
            return (to_bf16(a).astype(np.float32) @ to_bf16(b).astype(np.float32)
                    ).astype(np.float32)
    elif mode == "bfp16":
        def g(a, b):
            # K is axis 1 of A and axis 0 of B -- the reduction axis, which is
            # where the hardware forms its blocks.
            qa = to_bfp(to_bf16(a), axis=1, block=block, mant_bits=mant_bits)
            qb = to_bfp(to_bf16(b), axis=0, block=block, mant_bits=mant_bits)
            return (qa @ qb).astype(np.float32)
    else:
        raise ValueError(mode)
    return g


def rel_fro(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# -- part 1: calibrate against the M2 hardware measurement ------------------

def calibrate(rng, verbose=True):
    M, K, N = M2_SHAPE
    # M2 built inputs with iron.rand -> uniform [0,1) in bf16 on device.
    a = to_bf16(rng.random((M, K), dtype=np.float32))
    b = to_bf16(rng.random((K, N), dtype=np.float32))
    ref = a.astype(np.float64) @ b.astype(np.float64)

    got_bf16 = rel_fro(make_gemm("bf16")(a, b), ref)
    if verbose:
        print("part 1 -- calibrate the bfp16 model against M2 hardware")
        print(f"  M2 shape {M}x{K}x{N}, uniform [0,1) bf16 inputs, fp32 accumulate\n")
        print(f"  bf16 + fp32 accum      sim {got_bf16:.3e}   "
              f"hw {M2_BF16_F32_RELFRO:.3e}   "
              f"ratio {got_bf16/M2_BF16_F32_RELFRO:5.2f}x")
        print(f"\n  {'block':>5} {'mant':>5} {'sim rel_fro':>13} {'hw':>11} {'ratio':>8}")

    best, results = None, []
    # Swept wide on purpose. The first pass assumed bfp16 keeps 1 sign + 7
    # magnitude bits and landed at 0.05x the measured hardware error -- i.e. the
    # hardware is ~19x WORSE than that model. Rather than guess a mechanism, we
    # sweep mantissa width and let the measurement say how many bits the
    # hardware behaves as if it keeps.
    for block in (8, 16, 32, 64):
        for mant in (3, 4, 5, 6, 7, 8, 9):
            g = make_gemm("bfp16", block=block, mant_bits=mant)
            r = rel_fro(g(a, b), ref)
            ratio = r / M2_BFP16_RELFRO
            results.append({"block": block, "mant_bits": mant, "rel_fro": r,
                            "ratio_vs_hw": ratio})
            if verbose:
                print(f"  {block:>5} {mant:>5} {r:>13.3e} {M2_BFP16_RELFRO:>11.3e} "
                      f"{ratio:>7.2f}x")
    # Block size is NOT identifiable from this experiment: 8 through 64 differ
    # by <10% at every mantissa width, because uniform [0,1) has the same
    # dynamic range in a block of 8 as in a block of 64. So we fix block=8 --
    # the physically plausible value for an AIE bfp16 datapath -- and fit only
    # the mantissa width, which the data DOES resolve (a factor ~3 per bit).
    FIT_BLOCK = 8
    for r in results:
        if r["block"] != FIT_BLOCK:
            continue
        if best is None or abs(np.log(r["ratio_vs_hw"])) < abs(np.log(best["ratio_vs_hw"])):
            best = r
    if verbose:
        print(f"\n  block size is not identifiable here (<10% spread across 8..64);")
        print(f"  fixing block={FIT_BLOCK} and fitting mantissa width only:")
        print(f"    best fit: {best['mant_bits']} bits/element "
              f"(1 sign + {best['mant_bits']-1} magnitude) -> "
              f"{best['rel_fro']:.3e}, {best['ratio_vs_hw']:.2f}x the measured "
              f"{M2_BFP16_RELFRO:.3e}")
    return best, results, got_bf16


# -- part 1b: is uniform [0,1) actually adversarial for block FP? -----------

def fit_on_real(model_dir, goldens, block, target=M5_BFP16_REAL_RELFRO):
    """Fit the mantissa width against the M5 HARDWARE measurement on real data.

    The original fit used uniform inputs because that was the only hardware
    number available. M5 measured the real thing, so the model is now anchored
    where the encoder actually operates. Same layer-0 QKV GEMM the hardware ran.
    """
    g, _ = load(goldens)
    w, _ = load(Path(model_dir) / "model.safetensors")
    a = to_bf16(g["hf.emb.ln"].reshape(-1, 384))
    b = to_bf16(np.ascontiguousarray(np.concatenate(
        [w[f"encoder.layer.0.attention.self.{n}.weight"]
         for n in ("query", "key", "value")], axis=0).T))
    ref = a.astype(np.float64) @ b.astype(np.float64)

    print("\npart 1c -- refit the mantissa width against M5 HARDWARE on real data")
    print(f"  target: {target:.3e} (layer 0 QKV, real activations x real weights)\n")
    print(f"  {'bits/elem':>10} {'sim rel_fro':>13} {'ratio vs hw':>13}")
    best = None
    for mant in range(3, 11):
        r = rel_fro(make_gemm("bfp16", block=block, mant_bits=mant)(a, b), ref)
        ratio = r / target
        print(f"  {mant:>10} {r:>13.3e} {ratio:>12.2f}x")
        if best is None or abs(np.log(ratio)) < abs(np.log(best[1] / target)):
            best = (mant, r)
    print(f"\n  best fit on REAL data: {best[0]} bits/element -> {best[1]:.3e} "
          f"({best[1]/target:.2f}x the measured {target:.3e})")
    return best[0]


def single_gemm_comparison(rng, model_dir, goldens, block, mant_bits):
    """One GEMM, same shape, two input distributions.

    CLAUDE.md's open question assumed uniform [0,1) was probably ADVERSARIAL for
    a block float format and that real activations would look better. This
    isolates that: identical shape and identical bfp16 model, only the data
    differs. Accumulated end-to-end error cannot answer it -- depth confounds it.
    """
    g, _ = load(goldens)
    w, _ = load(Path(model_dir) / "model.safetensors")

    # A real GEMM: layer 0 QKV. A is post-LayerNorm activations, B is the
    # trained weight -- exactly what the NPU will be handed.
    a_real = g["hf.emb.ln"].reshape(-1, 384).astype(np.float32)
    b_real = np.ascontiguousarray(np.concatenate(
        [w[f"encoder.layer.0.attention.self.{n}.weight"]
         for n in ("query", "key", "value")], axis=0).T)          # [384, 1152]

    M, K, N = a_real.shape[0], a_real.shape[1], b_real.shape[1]
    a_unif = to_bf16(rng.random((M, K), dtype=np.float32))
    b_unif = to_bf16(rng.random((K, N), dtype=np.float32))

    gemm = make_gemm("bfp16", block=block, mant_bits=mant_bits)

    def block_range(x, axis):
        """Median within-block dynamic range: max|x| / min|x| over `block`
        consecutive values along the reduction axis. This is the quantity block
        floating point actually cares about -- one large value sets the shared
        exponent for the whole block."""
        x = np.abs(np.moveaxis(np.asarray(x, np.float64), axis, -1))
        n = (x.shape[-1] // block) * block
        b = x[..., :n].reshape(*x.shape[:-1], -1, block)
        lo = np.maximum(b.min(axis=-1), 1e-30)
        return float(np.median(b.max(axis=-1) / lo))

    print("\npart 1b -- is uniform [0,1) really the adversarial case?")
    print(f"  one GEMM, {M}x{K}x{N} (layer 0 QKV), bfp16 block={block} "
          f"mant={mant_bits}\n")
    print(f"  {'inputs':<34} {'rel_fro':>11} {'median block max/min':>22}")
    rows = {}
    for label, (a, b) in {
        "uniform [0,1)  (M2's test data)": (a_unif, b_unif),
        "real activations x real weights": (a_real, b_real),
    }.items():
        ref = a.astype(np.float64) @ b.astype(np.float64)
        r = rel_fro(gemm(a, b), ref)
        dr = block_range(a, axis=1)
        rows[label] = {"rel_fro": r, "median_block_dynamic_range_A": dr}
        print(f"  {label:<34} {r:>11.3e} {dr:>22.1f}")
    ratio = (rows["real activations x real weights"]["rel_fro"]
             / rows["uniform [0,1)  (M2's test data)"]["rel_fro"])
    verdict = ("WORSE" if ratio > 1.15 else "BETTER" if ratio < 0.87 else "COMPARABLE")
    print(f"\n  real activations are {ratio:.2f}x the uniform error -> {verdict}")
    return rows, ratio, verdict


# -- part 2: run the real encoder under each format ------------------------

def study(model_dir, goldens, block, mant_bits):
    g, meta = load(goldens)
    n_layers = int(meta["num_layers"])
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    w, _ = load(Path(model_dir) / "model.safetensors")

    ids, mask, tti = g["input_ids"], g["attention_mask"], g["token_type_ids"]
    runs = {}
    for mode in ("fp32", "bf16", "bfp16"):
        ref = MiniLMReference(w, num_layers=n_layers,
                              num_heads=cfg["num_attention_heads"],
                              eps=cfg["layer_norm_eps"],
                              gemm=make_gemm(mode, block=block, mant_bits=mant_bits))
        taps = {}
        emb = ref.encode(ids, mask, tti, taps=taps)
        runs[mode] = (emb, taps)

    base_emb, base_taps = runs["fp32"]
    hf = g["hf.out.embedding"]

    print("\npart 2 -- the same formats on REAL activations (the full encoder)")
    print(f"  bfp16 model: block={block} along K, {mant_bits} bits/element "
          f"(1 sign + {mant_bits-1} magnitude), shared exponent per block")
    print(f"  corpus: batch {meta['batch']}, seq {meta['seq_len']}, "
          f"{n_layers} layers\n")

    names = ["emb.ln"] + [f"L{i}.ln2" for i in range(n_layers)] + ["last_hidden_state"]
    print(f"  {'tensor':<20} {'bf16 rel_fro':>14} {'bfp16 rel_fro':>15}")
    rows = []
    for nm in names:
        r16 = rel_fro(runs["bf16"][1][nm], base_taps[nm])
        rbfp = rel_fro(runs["bfp16"][1][nm], base_taps[nm])
        rows.append({"tensor": nm, "bf16": r16, "bfp16": rbfp})
        print(f"  {nm:<20} {r16:>14.3e} {rbfp:>15.3e}")

    print(f"\n  {'metric':<34} {'bf16':>13} {'bfp16':>13}")
    out = {}
    for mode in ("bf16", "bfp16"):
        emb = runs[mode][0]
        cos_fp32 = (emb.astype(np.float64) * base_emb.astype(np.float64)).sum(1)
        cos_hf = (emb.astype(np.float64) * hf.astype(np.float64)).sum(1)
        # The similarity matrix is what a retrieval system actually consumes.
        sim_ref = base_emb @ base_emb.T
        sim_got = emb @ emb.T
        out[mode] = {
            "min_cos_vs_fp32": float(cos_fp32.min()),
            "max_1mcos_vs_fp32": float(1 - cos_fp32.min()),
            "min_cos_vs_hf": float(cos_hf.min()),
            "max_abs_sim_shift": float(np.abs(sim_got - sim_ref).max()),
            "embedding_rel_fro": rel_fro(emb, base_emb),
        }
    for label, key in [("1 - cos vs fp32 (worst)", "max_1mcos_vs_fp32"),
                       ("1 - cos vs HuggingFace (worst)", "min_cos_vs_hf"),
                       ("max shift in sentence similarity", "max_abs_sim_shift"),
                       ("embedding rel_fro", "embedding_rel_fro")]:
        a = out["bf16"][key]
        b = out["bfp16"][key]
        if key == "min_cos_vs_hf":
            a, b = 1 - a, 1 - b
        print(f"  {label:<34} {a:>13.3e} {b:>13.3e}")
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"
                                             / "minilm_l6_s64_boundary.safetensors"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "reference" / "goldens"
                                         / "precision_study.json"))
    args = ap.parse_args()

    print("SIMULATION -- not a hardware measurement. See docs/05-measurement.\n")
    rng = np.random.default_rng(args.seed)
    best, sweep, bf16_cal = calibrate(rng)
    dist_rows, dist_ratio, verdict = single_gemm_comparison(
        rng, args.model_dir, args.goldens, best["block"], best["mant_bits"])

    # Anchor on the real-data hardware measurement, not the uniform one. The
    # end-to-end numbers below are what M8 will budget against, so they must be
    # calibrated on the distribution the encoder actually runs.
    mant_real = fit_on_real(args.model_dir, args.goldens, best["block"])
    rows, summary = study(args.model_dir, args.goldens,
                          best["block"], mant_real)

    Path(args.out).write_text(json.dumps({
        "kind": "simulation",
        "m2_hardware_reference": {
            "shape": list(M2_SHAPE),
            "bf16_f32_rel_fro": M2_BF16_F32_RELFRO,
            "bfp16_rel_fro": M2_BFP16_RELFRO,
            "inputs": "uniform [0,1) bf16, iron.rand",
        },
        "calibration": {"sweep": sweep, "chosen": best,
                        "bf16_sim_rel_fro": bf16_cal,
                        "mant_bits_fitted_on_real_hardware": mant_real,
                        "m5_hardware_real_rel_fro": M5_BFP16_REAL_RELFRO,
                        "m5_hardware_uniform_rel_fro": M5_BFP16_UNIFORM_RELFRO,
                        "note": "The encoder numbers use the REAL-data fit. The "
                                "uniform fit is kept only to show why the "
                                "distribution-sensitivity claim was an artifact."},
        "distribution_comparison": {"single_gemm": dist_rows,
                                    "real_over_uniform": dist_ratio,
                                    "verdict": verdict},
        "encoder": {"per_tensor": rows, "summary": summary},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
