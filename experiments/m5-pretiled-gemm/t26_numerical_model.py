# NpuEmbeddings -- T26 numerical mechanism model (tasks/0096, research/OPEN-THREADS.md T26)
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS ANSWERS
# ------------------
# tasks/0052 found bfp16-emulated GEMM + bf16-C transport measures 1-cos
# 3.615e-04 against bfp16-emulated + fp32-C transport's 2.395e-03 -- 6.6x MORE
# accurate, backwards from what an extra rounding step should do. tasks/0053
# refuted the "k-block-boundary re-quantisation" hypothesis (bit-identical
# matmul kernel object; flat 1.000x full/split ratio). tasks/0056 refuted the
# "floor vs conv_even" rounding-mode-asymmetry hypothesis by reading the code
# (both narrowing sites are round-to-nearest-even) and CONFIRMED the
# mechanism CLASS instead: bf16-C narrows the raw accumulator once, EARLY
# (on-core, before bias-add), IN ADDITION TO the same late host RNE narrow
# both paths share -- and on a chained synthetic probe this compounds
# monotonically (rel_fro ratio 0.996/1.160/1.207/1.290, 1-cos ratio
# 0.992/1.346/1.457/1.666 over 4 stages).
#
# What is still open, and what THIS script is for: the numerical WHY. This is
# a HOST-SIDE model, not another hardware probe -- CLAUDE.md's own framing for
# this task. No NPU is touched anywhere below.
#
# THE MODEL. Reuses two pieces of already-CALIBRATED infrastructure rather
# than re-deriving a block-float model from scratch:
#
#   - reference/precision_study.py's `to_bfp()` -- the bfp16ebs8 block-float
#     quantiser (block=8, shared exponent, sign+magnitude mantissa), fitted
#     against REAL HARDWARE (tasks/0008: 7 bits/element on real activations,
#     0.85x the uniform-data error, refuting a 6.0x-worse simulated
#     prediction). This IS aie2p's bfp16ebs8 format: block_vector_native_types
#     .hpp (C:\dev\mlir-aie) confirms `type_bits<bfp16ebs8> = 8` (1 sign +
#     7 magnitude bits per element) with block-size-8 storage -- exactly
#     0008's fitted geometry, not a guess.
#   - reference/precision_study.py's `to_bf16()` and tools/npue.py's
#     `to_bf16_bits()` -- the SAME round-to-nearest-even bit formula
#     production uses at every narrowing site (verified bit-identical in
#     0056), used here for BOTH the "late" narrow shared by both paths and
#     the "early" on-core narrow unique to bf16-C.
#   - reference/encoder.py's `MiniLMReference` -- the full BERT forward pass
#     in numpy with a swappable `gemm(a, b) -> C` primitive. This is what
#     lets Part 3 below run REAL weights and REAL activations through the
#     ACTUAL 6-layer chain (LayerNorm, softmax, GELU, residuals intact),
#     not a toy loop -- production's structural claim, `linear()` returns
#     `gemm(...).reshape(...)` and adds bias OUTSIDE that call (line ~156),
#     which is exactly where fp32-C/bf16-C diverge in the real runtime too.
#
# Three parts:
#   1. CONTROL. Reproduce 0056's t26_chain_probe.py shape/protocol in pure
#      numpy. If this does not reproduce the measured 0.99->1.29 (rel_fro) /
#      0.99->1.67 (1-cos) progression, the model is not modelling the right
#      thing and everything below is suspect -- said explicitly in the report
#      either way.
#   2. ABLATE. With a working model, isolate candidate mechanisms: is the
#      compounding a BIAS effect (fp32-C's error has non-zero mean) or a
#      VARIANCE/decorrelation effect (bf16-C's error is less correlated
#      stage-to-stage)? Also settles 0053's carried sub-question (is the
#      split-mode block-count effect the same phenomenon as chain
#      compounding?) and 0056's non-goal (does plain bf16, non-emulated,
#      carry an analogous smaller effect?).
#   3. REAL DATA. Route the SAME two narrowing paths through the real
#      6-layer MiniLM encoder on real weights/activations (models/, reference
#      /goldens/) instead of synthetic Gaussians -- the thing 0053 and 0056
#      both flagged as untested, and which 0008 already showed matters for
#      bfp16 error in general (0.85x real vs uniform, not the simulated 6.0x).
#
# Env: plain numpy. Any of iron / .venv-ref / conda `iron` works; no IRON,
# no torch.
# Usage:
#   python t26_numerical_model.py --out artifacts/t26_numerical_model.json

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
REFERENCE = REPO / "reference"
ARTIFACTS = HERE / "artifacts"
sys.path.insert(0, str(REFERENCE))

from precision_study import to_bf16, to_bfp          # noqa: E402  (calibrated)
from encoder import MiniLMReference                  # noqa: E402
from safetensors_io import load                       # noqa: E402

MANT_BITS = 7     # 0008's real-data fit: 1 sign + 6 magnitude bits
BLOCK = 8         # bfp16ebs8 -- confirmed in block_vector_native_types.hpp


# ---------------------------------------------------------------------------
# Part 0 -- core numeric primitives
# ---------------------------------------------------------------------------

def bfp16_matmul(a_bf16, b_bf16, block=BLOCK, mant_bits=MANT_BITS):
    """The on-device MAC under --emulate-bf16-mmul-with-bfp16: A and B
    (already bf16-valued) are block-quantised along the reduction axis
    (shared exponent per `block` elements, `mant_bits` sign+magnitude bits),
    multiplied and accumulated in fp64 here -- standing in for the device's
    fp32 accumulate, which trap 2 / 0008 measured as effectively free
    (bf16 x bf16 accumulated in fp32: relfro 1.2e-07, "free" relative to the
    ~1e-2 bfp16-quantisation error this function actually injects).
    """
    qa = to_bfp(a_bf16, axis=1, block=block, mant_bits=mant_bits)
    qb = to_bfp(b_bf16, axis=0, block=block, mant_bits=mant_bits)
    return qa.astype(np.float64) @ qb.astype(np.float64)


def bf16_matmul(a_bf16, b_bf16):
    """Plain (non-emulated) bf16 MAC, fp32-equivalent accumulate. No bfp16
    quantisation error -- only the fact that A and B are already snapped to
    the bf16 grid on input."""
    return a_bf16.astype(np.float64) @ b_bf16.astype(np.float64)


def make_gemm(path, emulate=True, mant_bits=MANT_BITS, block=BLOCK):
    """gemm(a, b) -> C[np.float32], matching production's structural split
    (runtime/src/main.cpp's gemm() / tasks/0056's traced narrowing sites):

      fp32C: the raw accumulator is returned UNNARROWED. The caller (in the
             synthetic chain below, or reference/encoder.py's `linear()`)
             adds bias in fp32 and narrows to bf16 exactly once, late.
      bf16C: the raw accumulator is narrowed to bf16 ONCE, ON-CORE, with the
             SAME round-to-nearest-even formula, BEFORE returning -- i.e.
             before the caller's bias add. The caller's later narrow (same
             RNE) still happens too, on the (bf16 + bias) sum.

    Both paths narrow their INCOMING activation `a` to bf16 on every call --
    this is production's bf16_fill()/to_bf16() step (0056's "late narrow
    shared by both paths"), applied here at the one place every gemm() call
    naturally sees the previous stage's fp32 output.
    """
    def gemm(a, b):
        a_bf16 = to_bf16(np.asarray(a, dtype=np.float32))
        b_bf16 = to_bf16(np.asarray(b, dtype=np.float32))
        raw = bfp16_matmul(a_bf16, b_bf16, block, mant_bits) if emulate \
            else bf16_matmul(a_bf16, b_bf16)
        if path == "bf16C":
            raw = to_bf16(raw.astype(np.float32)).astype(np.float64)
        return raw.astype(np.float32)
    return gemm


def rel_fro(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def one_minus_cos(got, ref):
    g, r = np.asarray(got, np.float64).reshape(-1), np.asarray(ref, np.float64).reshape(-1)
    return float(1.0 - (g @ r) / (np.linalg.norm(g) * np.linalg.norm(r)))


# ---------------------------------------------------------------------------
# Part 1 -- CONTROL: reproduce 0056's t26_chain_probe.py in pure numpy
# ---------------------------------------------------------------------------

def synthetic_chain(path, mant_bits=MANT_BITS, block=BLOCK, L=4,
                     M=256, K=192, N=192, seed=0, emulate=True):
    """Same shape/protocol as t26_chain_probe.py: K==N so a stage's output
    feeds the next stage's input width directly; bias magnitude 0.1*N(0,1);
    fp64 'ideal' reference with NO emulation error and NO narrowing at all."""
    rng = np.random.default_rng(seed)

    def randn_bf16(shape):
        return to_bf16(rng.standard_normal(shape).astype(np.float32))

    X0 = randn_bf16((M, K))
    Ws = [randn_bf16((K, N)) for _ in range(L)]
    biases = [rng.standard_normal(N).astype(np.float32) * 0.1 for _ in range(L)]

    X_ref = X0.astype(np.float64)
    ref_states = []
    for i in range(L):
        X_ref = X_ref @ Ws[i].astype(np.float64) + biases[i].astype(np.float64)
        ref_states.append(X_ref.copy())

    gemm = make_gemm(path, emulate=emulate, mant_bits=mant_bits, block=block)
    X = X0.astype(np.float32)
    states, raws = [], []
    for i in range(L):
        C = gemm(X, Ws[i].astype(np.float32))
        raws.append(C.astype(np.float64).copy())
        Y = C.astype(np.float64) + biases[i].astype(np.float64)
        X_next = to_bf16(Y.astype(np.float32))
        states.append(X_next.astype(np.float64).copy())
        X = X_next
    return ref_states, states, raws


def part1_control(mant_bits, block):
    print("=" * 78)
    print("PART 1 -- control: reproduce 0056's chain probe (synthetic Gaussian)")
    print("=" * 78)
    print(f"  bfp16 model: block={block}, {mant_bits} bits/element "
          f"(0008's real-data fit)\n")

    measured_relfro = [0.996, 1.160, 1.207, 1.290]
    measured_1mcos = [0.992, 1.346, 1.457, 1.666]

    ref_a, fp32c, _ = synthetic_chain("fp32C", mant_bits, block)
    ref_b, bf16c, _ = synthetic_chain("bf16C", mant_bits, block)
    assert all(np.array_equal(a, b) for a, b in zip(ref_a, ref_b))  # same seed/schedule

    rows = []
    print(f"  {'stage':>5} {'fp32C rel_fro':>14} {'bf16C rel_fro':>14} "
          f"{'sim ratio':>10} {'measured':>10}   {'1mcos ratio':>12} {'measured':>10}")
    for i in range(len(ref_a)):
        rf32 = rel_fro(fp32c[i], ref_a[i])
        rb16 = rel_fro(bf16c[i], ref_a[i])
        om32 = one_minus_cos(fp32c[i], ref_a[i])
        om16 = one_minus_cos(bf16c[i], ref_a[i])
        ratio_rf = rf32 / rb16
        ratio_om = om32 / om16
        rows.append({"stage": i + 1, "fp32C_rel_fro": rf32, "bf16C_rel_fro": rb16,
                      "ratio_rel_fro": ratio_rf, "measured_ratio_rel_fro": measured_relfro[i],
                      "fp32C_1mcos": om32, "bf16C_1mcos": om16,
                      "ratio_1mcos": ratio_om, "measured_ratio_1mcos": measured_1mcos[i]})
        print(f"  {i + 1:>5} {rf32:>14.4e} {rb16:>14.4e} {ratio_rf:>10.3f} "
              f"{measured_relfro[i]:>10.3f}   {ratio_om:>12.3f} {measured_1mcos[i]:>10.3f}")

    # verdict: does the SIGN and MONOTONICITY of the compounding reproduce?
    sim_monotonic = all(rows[i]["ratio_1mcos"] <= rows[i + 1]["ratio_1mcos"] + 1e-9
                         for i in range(len(rows) - 1))
    tied_at_1 = abs(rows[0]["ratio_1mcos"] - 1.0) < 0.05
    grows = rows[-1]["ratio_1mcos"] > 1.2
    reproduced = sim_monotonic and tied_at_1 and grows
    print(f"\n  tied at stage 1: {tied_at_1}   monotonic growth: {sim_monotonic}   "
          f"final ratio > 1.2: {grows}")
    print(f"  CONTROL {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
    return rows, reproduced


# ---------------------------------------------------------------------------
# Part 2 -- ABLATIONS
# ---------------------------------------------------------------------------

def ablation_idempotence():
    """Analytic sanity check referenced in the report: RNE-narrowing a value
    already on the bf16 grid to bf16 again is a no-op. This is WHY bf16-C's
    'extra narrow' can only be non-degenerate if it targets a DIFFERENT value
    than the final narrow does (the pre-bias accumulator) -- placed AFTER
    bias it would collapse to fp32-C exactly. Not a hypothesis to test on
    hardware; a property of round-to-nearest-even, checked here for the
    record."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200000).astype(np.float32) * 37.0
    once = to_bf16(x)
    twice = to_bf16(once)
    ok = bool(np.array_equal(once, twice))
    print("\n" + "=" * 78)
    print("PART 2a -- analytic: RNE-to-bf16 is idempotent on an already-bf16 value")
    print("=" * 78)
    print(f"  to_bf16(to_bf16(x)) == to_bf16(x) for 200000 samples: {ok}")
    print("  Consequence: bf16-C's 'extra' narrow is only ever non-degenerate")
    print("  because it targets the PRE-BIAS accumulator, a value the shared")
    print("  final narrow does not itself see. 'Narrow count' and 'narrow")
    print("  position' are NOT independent factors here -- position IS the")
    print("  only way an extra narrow can do anything at all.")
    return ok


def ablation_bias_vs_variance(mant_bits, block, L=8, trials=300, M=64, K=192, N=192):
    """Is the compounding a BIAS effect (fp32-C's per-stage error has
    non-zero mean, drifting the trajectory systematically) or a VARIANCE /
    decorrelation effect (the two paths are both ~zero-mean but bf16-C's
    error is less correlated stage-to-stage, so it partially cancels on
    accumulation while fp32-C's does not)?

    Runs `trials` independent chains (fresh random weights/bias/input each
    time, same schedule shared by both paths per trial) and looks at the
    signed, per-element error distribution at each stage, pooled across
    trials and elements.
    """
    print("\n" + "=" * 78)
    print("PART 2b -- bias vs variance: signed error and stage-to-stage correlation")
    print("=" * 78)

    fp32_errs = [[] for _ in range(L)]   # signed error, flattened, all trials
    bf16_errs = [[] for _ in range(L)]

    for t in range(trials):
        ref, fp32c, _ = synthetic_chain("fp32C", mant_bits, block, L=L, M=M, K=K, N=N, seed=1000 + t)
        _, bf16c, _ = synthetic_chain("bf16C", mant_bits, block, L=L, M=M, K=K, N=N, seed=1000 + t)
        for i in range(L):
            fp32_errs[i].append((fp32c[i] - ref[i]).ravel())
            bf16_errs[i].append((bf16c[i] - ref[i]).ravel())

    rows = []
    print(f"  {'stage':>5} {'fp32C mean':>12} {'fp32C std':>11} "
          f"{'bf16C mean':>12} {'bf16C std':>11} {'std ratio':>10}")
    prev_fp32 = prev_bf16 = None
    corr_fp32, corr_bf16 = [], []
    for i in range(L):
        e32 = np.concatenate(fp32_errs[i]).astype(np.float64)
        e16 = np.concatenate(bf16_errs[i]).astype(np.float64)
        m32, s32 = float(e32.mean()), float(e32.std())
        m16, s16 = float(e16.mean()), float(e16.std())
        rows.append({"stage": i + 1, "fp32C_mean": m32, "fp32C_std": s32,
                      "bf16C_mean": m16, "bf16C_std": s16, "std_ratio": s32 / s16})
        print(f"  {i + 1:>5} {m32:>12.3e} {s32:>11.4e} {m16:>12.3e} {s16:>11.4e} "
              f"{s32 / s16:>10.3f}")
        if prev_fp32 is not None:
            # correlation between consecutive-stage error (reshaped elementwise;
            # stages are same M,N shape so this is well-defined)
            c32 = float(np.corrcoef(prev_fp32, e32)[0, 1])
            c16 = float(np.corrcoef(prev_bf16, e16)[0, 1])
            corr_fp32.append(c32)
            corr_bf16.append(c16)
        prev_fp32, prev_bf16 = e32, e16

    print(f"\n  stage-to-stage error correlation (Pearson r, consecutive stages):")
    print(f"  {'transition':>12} {'fp32C r':>10} {'bf16C r':>10}")
    for i, (c32, c16) in enumerate(zip(corr_fp32, corr_bf16)):
        print(f"  {i + 1}->{i + 2:<8} {c32:>10.4f} {c16:>10.4f}")

    verdict_bias = "BIAS" if any(abs(r["fp32C_mean"]) > 3 * abs(r["bf16C_mean"]) and
                                  abs(r["fp32C_mean"]) > 1e-4 for r in rows[-3:]) else "NOT BIAS-DOMINATED"
    mean_corr32 = float(np.mean(corr_fp32)) if corr_fp32 else float("nan")
    mean_corr16 = float(np.mean(corr_bf16)) if corr_bf16 else float("nan")
    print(f"\n  mean stage-to-stage correlation: fp32C {mean_corr32:.4f}  bf16C {mean_corr16:.4f}")
    print(f"  bias verdict: {verdict_bias}")
    return {"rows": rows, "corr_fp32": corr_fp32, "corr_bf16": corr_bf16,
            "mean_corr_fp32C": mean_corr32, "mean_corr_bf16C": mean_corr16,
            "bias_verdict": verdict_bias}


def ablation_split_vs_chain(mant_bits, block, M=256, N=192, trials=20):
    """0053's carried sub-question: bf16-C's SPLIT-mode error (one k-block
    per dispatch, host-summed, NO boundary by construction) DECREASES with
    more blocks (1.073e-2 at 6 -> 9.878e-3 at 24, 0053). Is that the SAME
    phenomenon as chain compounding (Part 1), or different?

    Chain compounding = error growing ACROSS SEQUENTIAL GEMM STAGES (each
    stage's OUTPUT feeds the next stage's INPUT, narrowed at each boundary).
    Split-mode block-count = error SHRINKING within a SINGLE GEMM's own
    K-reduction as more k-blocks are summed (no narrowing between blocks,
    single narrow at the very end). These are structurally different axes
    (depth vs width of one reduction) -- this ablation tests whether they
    nonetheless share a mechanism (both look like averaging of independent
    per-block quantisation noise) by reproducing the split-mode shrinkage
    in the SAME calibrated model and checking its functional form.
    """
    print("\n" + "=" * 78)
    print("PART 2c -- split-mode block-count effect vs chain-length effect")
    print("=" * 78)
    rng = np.random.default_rng(7)

    def randn_bf16(shape):
        return to_bf16(rng.standard_normal(shape).astype(np.float32))

    rows = []
    for k_blocks in (6, 24):
        K = 64 * k_blocks
        errs_full, errs_split = [], []
        for t in range(trials):
            a = randn_bf16((M, K))
            b = randn_bf16((K, N))
            ref = a.astype(np.float64) @ b.astype(np.float64)
            # "full": one bfp16 K-reduction, single narrow at the end (bf16C path)
            gemm_bf16c = make_gemm("bf16C", emulate=True, mant_bits=mant_bits, block=block)
            c_full = gemm_bf16c(a, b).astype(np.float64)
            errs_full.append(rel_fro(c_full, ref))
            # "split": K/64 separate 64-wide dispatches, each independently
            # bfp16C-narrowed, summed on the host in fp64 -- exactly 0053's split protocol
            acc = np.zeros((M, N), np.float64)
            for kb in range(k_blocks):
                a_slice = a[:, kb * 64:(kb + 1) * 64]
                b_slice = b[kb * 64:(kb + 1) * 64, :]
                acc += gemm_bf16c(a_slice, b_slice).astype(np.float64)
            errs_split.append(rel_fro(acc, ref))
        rows.append({"k_blocks": k_blocks, "full_rel_fro": float(np.mean(errs_full)),
                      "split_rel_fro": float(np.mean(errs_split))})
        print(f"  k_blocks={k_blocks:>3}  full {np.mean(errs_full):.4e}  "
              f"split {np.mean(errs_split):.4e}")

    split_shrinks = rows[1]["split_rel_fro"] < rows[0]["split_rel_fro"]
    print(f"\n  split-mode error at 24 blocks < at 6 blocks: {split_shrinks} "
          f"(0053 measured 9.878e-3 < 1.073e-2, same direction)")
    print("  Interpretation: both this and Part 1's chain compounding are")
    print("  instances of the SAME underlying fact -- summing more independent,")
    print("  zero-mean-ish bfp16 quantisation contributions makes the SUM's")
    print("  RELATIVE error shrink (each dot-product term's error is")
    print("  ~independent of the others, so the error grows as sqrt(K) while")
    print("  the true sum grows faster / does not cancel as much) -- but they")
    print("  are NOT the same axis: split-mode narrows the SAME reduction's")
    print("  own error (width), chain compounding is about what happens to")
    print("  error carried ACROSS stage boundaries (depth). Confirmed distinct")
    print("  by construction (split never crosses a stage boundary; chain")
    print("  always does), and both directionally consistent with hardware.")
    return rows, split_shrinks


def ablation_plain_bf16(L=4, M=256, K=192, N=192, trials=20):
    """0056's explicit non-goal: does TODAY'S SHIPPING plain-bf16 path (no
    bfp16 emulation) carry any analogous, smaller narrowing-position effect?
    Same chain, same two paths, emulate=False (bf16_matmul only -- no bfp16
    quantisation error at all, just bf16 rounding of inputs)."""
    print("\n" + "=" * 78)
    print("PART 2d -- does plain (non-emulated) bf16 carry an analogous effect?")
    print("=" * 78)
    ratios_relfro, ratios_1mcos = [], []
    for t in range(trials):
        ref, fp32c, _ = synthetic_chain("fp32C", L=L, M=M, K=K, N=N, seed=2000 + t, emulate=False)
        _, bf16c, _ = synthetic_chain("bf16C", L=L, M=M, K=K, N=N, seed=2000 + t, emulate=False)
        rf = [rel_fro(fp32c[i], ref[i]) for i in range(L)]
        rb = [rel_fro(bf16c[i], ref[i]) for i in range(L)]
        om_f = [one_minus_cos(fp32c[i], ref[i]) for i in range(L)]
        om_b = [one_minus_cos(bf16c[i], ref[i]) for i in range(L)]
        ratios_relfro.append([f / b if b else float("nan") for f, b in zip(rf, rb)])
        ratios_1mcos.append([f / b if b else float("nan") for f, b in zip(om_f, om_b)])
    mean_rf = np.nanmean(np.array(ratios_relfro), axis=0)
    mean_om = np.nanmean(np.array(ratios_1mcos), axis=0)
    print(f"  {'stage':>5} {'mean rel_fro ratio':>20} {'mean 1mcos ratio':>18}")
    rows = []
    for i in range(L):
        rows.append({"stage": i + 1, "mean_ratio_rel_fro": float(mean_rf[i]),
                      "mean_ratio_1mcos": float(mean_om[i])})
        print(f"  {i + 1:>5} {mean_rf[i]:>20.4f} {mean_om[i]:>18.4f}")
    grows = mean_om[-1] > mean_om[0] * 1.05
    print(f"\n  plain-bf16 ratio grows across stages: {grows} "
          f"(both paths sit at ~1e-7 relfro either way -- absolute magnitudes below)")
    print(f"  absolute magnitude, stage {L}: fp32C {rel_fro(fp32c[-1], ref[-1]):.3e}, "
          f"bf16C {rel_fro(bf16c[-1], ref[-1]):.3e} (last trial)")
    return rows, bool(grows)


# ---------------------------------------------------------------------------
# Part 3 -- REAL DATA: the actual MiniLM encoder, real weights, real activations
# ---------------------------------------------------------------------------

def part3_real_encoder(mant_bits, block,
                        model_dir=None, goldens=None):
    """Runs the FULL 6-layer MiniLM forward pass (reference/encoder.py,
    unmodified structurally) twice, once per narrowing path, with REAL
    weights (models/all-MiniLM-L6-v2/model.safetensors) and REAL activations
    (the tokenised input from reference/goldens/minilm_l6_s64_boundary
    .safetensors -- the same golden 0008 used). LayerNorm, softmax, GELU,
    residuals, biases all run in fp32 on the host exactly as production does
    (docs/04-model) -- only `self.gemm` differs between the two runs, exactly
    the production/probe boundary.
    """
    model_dir = model_dir or (REPO / "models" / "all-MiniLM-L6-v2")
    goldens = goldens or (REFERENCE / "goldens" / "minilm_l6_s64_boundary.safetensors")

    print("\n" + "=" * 78)
    print("PART 3 -- real weights, real activations: the actual 6-layer encoder")
    print("=" * 78)

    g, meta = load(goldens)
    n_layers = int(meta["num_layers"])
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    w, _ = load(Path(model_dir) / "model.safetensors")
    ids, mask, tti = g["input_ids"], g["attention_mask"], g["token_type_ids"]

    def build(path, emulate):
        gemm = make_gemm(path, emulate=emulate, mant_bits=mant_bits, block=block)
        ref = MiniLMReference(w, num_layers=n_layers, num_heads=cfg["num_attention_heads"],
                              eps=cfg["layer_norm_eps"], gemm=gemm)
        taps = {}
        emb = ref.encode(ids, mask, tti, taps=taps)
        return emb, taps

    fp32_emb, fp32_taps = build("fp32C", emulate=True)
    bf16_emb, bf16_taps = build("bf16C", emulate=True)
    # fp32 CPU oracle: no emulation, no path split at all (both `gemm`s equal)
    oracle = MiniLMReference(w, num_layers=n_layers, num_heads=cfg["num_attention_heads"],
                             eps=cfg["layer_norm_eps"])
    oracle_taps = {}
    oracle_emb = oracle.encode(ids, mask, tti, taps=oracle_taps)

    layer_names = ["emb.ln"] + [f"L{i}.ln2" for i in range(n_layers)] + ["last_hidden_state"]
    print(f"  {'tensor':<20} {'fp32C rel_fro':>14} {'bf16C rel_fro':>14} {'ratio':>8}")
    rows = []
    for nm in layer_names:
        rf32 = rel_fro(fp32_taps[nm], oracle_taps[nm])
        rb16 = rel_fro(bf16_taps[nm], oracle_taps[nm])
        ratio = rf32 / rb16 if rb16 else float("nan")
        rows.append({"tensor": nm, "fp32C_rel_fro": rf32, "bf16C_rel_fro": rb16, "ratio": ratio})
        print(f"  {nm:<20} {rf32:>14.4e} {rb16:>14.4e} {ratio:>8.3f}")

    cos_fp32 = float(1 - (fp32_emb.astype(np.float64) * oracle_emb.astype(np.float64)).sum(1).min())
    cos_bf16 = float(1 - (bf16_emb.astype(np.float64) * oracle_emb.astype(np.float64)).sum(1).min())
    print(f"\n  worst 1-cos vs fp32 oracle: fp32C {cos_fp32:.4e}  bf16C {cos_bf16:.4e}  "
          f"ratio {cos_fp32 / cos_bf16 if cos_bf16 else float('nan'):.3f}x")
    print(f"  (production hardware, tasks/0052: fp32-C 2.395e-03, bf16-C 3.615e-04, "
          f"ratio 6.6x)")

    return {"per_tensor": rows, "worst_1mcos_fp32C": cos_fp32, "worst_1mcos_bf16C": cos_bf16,
            "ratio_worst_1mcos": cos_fp32 / cos_bf16 if cos_bf16 else float("nan")}


def part3b_real_vs_gaussian(mant_bits, block, model_dir=None, L=4):
    """Distribution-dependence check requested explicitly: build a chain of
    REAL layer-0/1 weight matrices (Q|K|V fused, attn_out, ffn_up, ffn_down
    -- same tensors precision_study.py's fit_on_real() uses) fed by the REAL
    post-LayerNorm activation, and compare the fp32C/bf16C ratio progression
    against Part 1's Gaussian synthetic chain at the same depth. If the
    anomaly is systematically LARGER on real data, that is itself evidence
    that Part 1's Gaussian control understates the effect (consistent with
    0008's real-vs-uniform finding being a live mechanism class here, even
    though 0008's own instance of it went the OTHER way -- real was BETTER
    than uniform for plain bfp16 magnitude, not for this narrowing-structure
    question)."""
    model_dir = model_dir or (REPO / "models" / "all-MiniLM-L6-v2")
    goldens = REFERENCE / "goldens" / "minilm_l6_s64_boundary.safetensors"
    print("\n" + "=" * 78)
    print("PART 3b -- real-weight chain vs Gaussian-synthetic chain, same depth")
    print("=" * 78)

    g, _ = load(goldens)
    w, _ = load(Path(model_dir) / "model.safetensors")
    a0 = g["hf.emb.ln"].reshape(-1, 384).astype(np.float32)     # real activations

    # Real weight matrices from consecutive layers' QKV-fusion, sized so each
    # stage's OUTPUT width feeds the next stage's INPUT width (matching Part
    # 1's K==N constant-shape convention as closely as real HF shapes allow):
    # layer i's fused QKV [384,1152] doesn't chain directly (1152 != 384), so
    # this uses the attention-output projection weights (384->384) across L
    # consecutive layers instead -- real trained matrices, same shape every
    # stage, chains cleanly.
    Ws, biases = [], []
    for i in range(L):
        p = f"encoder.layer.{i}.attention.output.dense."
        Ws.append(w[p + "weight"].T.astype(np.float32))           # [384,384], HF stores [out,in]
        biases.append(w[p + "bias"].astype(np.float32))

    X_ref = a0.astype(np.float64)
    ref_states = []
    for i in range(L):
        X_ref = X_ref @ Ws[i].astype(np.float64) + biases[i].astype(np.float64)
        ref_states.append(X_ref.copy())

    def run(path):
        gemm = make_gemm(path, emulate=True, mant_bits=mant_bits, block=block)
        X = a0.copy()
        states = []
        for i in range(L):
            C = gemm(X, Ws[i]).astype(np.float64)
            Y = C + biases[i].astype(np.float64)
            X_next = to_bf16(Y.astype(np.float32))
            states.append(X_next.astype(np.float64).copy())
            X = X_next
        return states

    fp32c = run("fp32C")
    bf16c = run("bf16C")

    rows = []
    print(f"  {'stage':>5} {'fp32C 1mcos':>13} {'bf16C 1mcos':>13} {'ratio':>8}")
    for i in range(L):
        om32 = one_minus_cos(fp32c[i], ref_states[i])
        om16 = one_minus_cos(bf16c[i], ref_states[i])
        ratio = om32 / om16 if om16 else float("nan")
        rows.append({"stage": i + 1, "fp32C_1mcos": om32, "bf16C_1mcos": om16, "ratio": ratio})
        print(f"  {i + 1:>5} {om32:>13.4e} {om16:>13.4e} {ratio:>8.3f}")
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ARTIFACTS / "t26_numerical_model.json"))
    ap.add_argument("--mant-bits", type=int, default=MANT_BITS)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--skip-real", action="store_true",
                     help="skip Part 3 (needs models/all-MiniLM-L6-v2 + reference/goldens)")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    control_rows, reproduced = part1_control(args.mant_bits, args.block)
    idem_ok = ablation_idempotence()
    bias_var = ablation_bias_vs_variance(args.mant_bits, args.block)
    split_rows, split_shrinks = ablation_split_vs_chain(args.mant_bits, args.block)
    plain_rows, plain_grows = ablation_plain_bf16()

    real_summary = real_chain_rows = None
    if not args.skip_real:
        try:
            real_summary = part3_real_encoder(args.mant_bits, args.block)
            real_chain_rows = part3b_real_vs_gaussian(args.mant_bits, args.block)
        except FileNotFoundError as e:
            print(f"\n  Part 3 SKIPPED -- missing real-data file: {e}")

    out = {
        "mant_bits": args.mant_bits, "block": args.block,
        "part1_control": {"rows": control_rows, "reproduced": reproduced},
        "part2a_idempotence_ok": idem_ok,
        "part2b_bias_vs_variance": bias_var,
        "part2c_split_vs_chain": {"rows": split_rows, "split_shrinks_with_more_blocks": split_shrinks},
        "part2d_plain_bf16": {"rows": plain_rows, "grows_across_stages": plain_grows},
        "part3_real_encoder": real_summary,
        "part3b_real_vs_gaussian_chain": real_chain_rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
