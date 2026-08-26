# 0096 — T26: a host-side numerical model of the bfp16+bf16-C anomaly — control NOT reproduced

- **Date** 2026-08-23
- **Milestone** post-M13 (research thread, `research/OPEN-THREADS.md` T26)
- **Status** done — **negative result**: the model does not reproduce the
  hardware control, so no numerical mechanism is confirmed. What is
  established is what does NOT explain the anomaly, at the precision this
  model can express.

## Goal

T26 (`research/OPEN-THREADS.md`) asks *why* bfp16-emulated GEMM + bf16-C
transport measures 6.6× **more** accurate than bfp16 + fp32-C
(`1-cos` 2.395e-03 → 3.615e-04, [`0052`](../0052-m10-research-night/TASK.md)
§6) — backwards from what an extra rounding step should do. Two hypotheses
are refuted and must stay refuted (k-block-boundary re-quantisation,
[`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md); floor-vs-conv_even
rounding asymmetry, [`0056`](../0056-m10-t26-rounding-and-chain-probe/TASK.md)).
0056 confirmed a mechanism **class**: bf16-C narrows its raw accumulator
once, early, on-core, before bias-add, in addition to the same late host RNE
narrow both paths share; on a 4-stage chained synthetic probe this compounds
monotonically (rel_fro ratio 0.996→1.290, `1-cos` ratio 0.992→1.666). What
was still open is the **numerical why** — does the extra early narrow reduce
error growth through noise cancellation, a bias effect, or a range/scale
interaction — and this task's brief was to build a calibrated host-side
model in numpy/fp64 to answer it, since 0056 is a hardware fact but not an
explanation.

## Context

- [`0008`](../0008-m5-bfp16-real-data/TASK.md) measured bfp16's real-data
  hardware behaviour and fitted a block-float model to it: **7 bits/element**
  (1 sign + 6 magnitude), block size 8, refuting a 5-bit/6.0×-worse
  simulated prediction. `reference/precision_study.py`'s `to_bfp()` is that
  calibrated model, already checked in.
- `C:\dev\mlir-aie\ironenv\...\aie_api\detail\aie2p\block_vector_native_types.hpp`
  confirms this is not a guess: `type_bits<bfp16ebs8> = 8` and
  `block_vector_fill_frequency<bfp16ebs8> = 8` — 8 bits/element, block size
  8, exactly 0008's fitted geometry, not assumed.
- `reference/encoder.py`'s `MiniLMReference` already runs the full 6-layer
  BERT forward pass in numpy with a swappable `gemm(a,b)->C` primitive,
  bias added by the *caller* (`linear()`, line ~156) — structurally the same
  split point production's `runtime/src/main.cpp gemm()` uses between the
  raw accumulator and the bias add, which is exactly where fp32-C and
  bf16-C diverge (0056).
- `runtime/src/main.cpp`'s `to_bf16()` (line 403) and `tools/npue.py`'s
  `to_bf16_bits()` are bit-identical RNE formulas (0056), reused here so the
  model's narrowing steps match production exactly, not an approximation.

## What was done

Wrote `experiments/m5-pretiled-gemm/t26_numerical_model.py` (numpy/fp64
only, no NPU, no IRON). Three parts, per the brief:

**Part 0 — primitives.** `bfp16_matmul(a,b)` block-quantises A along its
K axis and B along its K axis (`to_bfp`, block=8, mant_bits=7, imported
unmodified from `precision_study.py`) then computes the exact fp64 dot
product — standing in for the device's fp32 accumulate, which trap 2/0008
established as effectively free relative to the ~1e-2 bfp16-quantisation
error itself. `make_gemm(path, emulate)` returns a `gemm(a,b)` closure
matching 0056's traced structure exactly: both paths narrow the *incoming*
activation to bf16 via `to_bf16()` (production's late narrow, applied at
the one place every call naturally sees the previous stage's output);
`bf16C` additionally narrows the *raw accumulator* to bf16 (same RNE
formula) before returning, i.e. before the caller's bias add; `fp32C`
returns the raw accumulator unnarrowed.

**Part 1 — control.** Reproduces `t26_chain_probe.py`'s exact shape/protocol
(M=256, K=192, N=192, K==N so output feeds next input, bias×0.1, L=4 stages)
in pure numpy, and compares against 0056's measured ratios. **This is the
gate**: per the brief, if the model can't reproduce the measured chain, it
is not modelling the right thing and everything downstream is suspect.

**Part 2 — ablations**, run regardless (see Result — Problems for why they
are reported as diagnostic rather than as confirmed mechanism evidence):
(a) an analytic idempotence check (RNE-narrowing an already-bf16 value is a
no-op — a fact about round-to-nearest-even, not a hypothesis); (b) a
300-trial bias-vs-variance sweep out to 8 chained stages, tracking signed
per-element error mean/std and stage-to-stage error correlation for both
paths; (c) 0053's carried sub-question — does bf16-C's *split*-mode error
shrink with k-block count (1.073e-2 at 6 → 9.878e-3 at 24, measured), and is
that the same phenomenon as chain compounding; (d) 0056's non-goal — does
*plain* (non-emulated) bf16 carry an analogous, smaller effect.

**Part 3 — real data.** Routes both paths through the actual 6-layer
`MiniLMReference` forward pass (LayerNorm, softmax, GELU, residuals intact)
on real weights (`models/all-MiniLM-L6-v2/model.safetensors`) and real
activations (`reference/goldens/minilm_l6_s64_boundary.safetensors`, the
same golden 0008 used), comparing per-layer `1-cos`/`rel_fro` against a
true fp32 oracle (no emulation). Part 3b builds a 4-stage chain from real,
consecutive attention-output-projection weight matrices (384→384, same
shape every stage) fed by the real post-LayerNorm activation, as a
real-data analogue of Part 1's synthetic control.

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm
& "C:\Users\vegar\.conda\envs\iron\python.exe" t26_numerical_model.py --out artifacts\t26_numerical_model.json

# mantissa-bit sweep (diagnostic, not saved to the JSON artifact — run inline
# to check whether the compounding effect appears at any calibration):
& "C:\Users\vegar\.conda\envs\iron\python.exe" -c "
import sys; sys.path.insert(0, '.')
from t26_numerical_model import synthetic_chain, one_minus_cos
for mant in (3,4,5,6,7,8,9):
    ref, fp32c, _ = synthetic_chain('fp32C', mant_bits=mant, L=4, seed=0)
    _, bf16c, _   = synthetic_chain('bf16C', mant_bits=mant, L=4, seed=0)
    print(mant, ['%.3f' % (one_minus_cos(fp32c[i], ref[i]) / one_minus_cos(bf16c[i], ref[i])) for i in range(4)])
"
```

Env: `C:\Users\vegar\.conda\envs\iron\python.exe` — plain numpy 2.4.3, no
torch, no IRON. Per the task brief, this thread is answerable entirely on
the host; no NPU was touched.

## Result

### Part 1 — CONTROL NOT REPRODUCED

```
stage  fp32C rel_fro  bf16C rel_fro  sim ratio  measured(0056)   1mcos sim ratio  measured(0056)
    1     1.8084e-02     1.8146e-02      0.997      0.996              0.993          0.992
    2     2.5285e-02     2.5319e-02      0.999      1.160              0.997          1.346
    3     3.0951e-02     3.0994e-02      0.999      1.207              0.997          1.457
    4     3.6107e-02     3.6220e-02      0.997      1.290              0.994          1.666
```

Absolute magnitudes land within ~17% of 0056's hardware numbers at stage 1
(1.81e-2 sim vs 1.55e-2 hw), confirming the bfp16 quantisation *magnitude*
calibration (0008's 7-bit fit, block=8) is in the right regime. But the
**ratio between the two paths stays pinned at ~0.99–1.00 across all 4
stages** — tied at stage 1 (matches hardware), then **flat**, where
hardware climbs monotonically to 1.29–1.67×. Re-run with 300-trial
averaging out to 8 stages (Part 2b, below) removes any doubt this is
single-seed noise: the std ratio between paths sits at 0.996–1.000 for
every one of 8 stages, with **no growth trend at all**.

**A full mantissa-bit sweep (3–9 bits) at the same shape/L=4 finds the
compounding effect at NO calibration.** At mant=3–7 the ratio stays pinned
near 1.00 throughout; at mant=8–9 (less block-float error, closer to plain
bf16) the ratio actually falls *below* 1 (bf16-C become worse, e.g. 0.881 at
mant=9, stage 1) and **rises back toward 1 with depth** — the opposite
direction from hardware's climb away from 1. This is not a calibration
miss; the effect this model can express saturates at "no compounding" over
the entire physically plausible mantissa range.

**Per the task brief's explicit instruction, this fails the gate: the
model is not reproducing the mechanism, and the ablations below are
reported as diagnostic/exploratory, not as confirmed evidence for a
mechanism.**

### Part 2 — ablations (diagnostic only, given the failed control)

**2a — idempotence (analytic, stands independent of the control failure).**
`to_bf16(to_bf16(x)) == to_bf16(x)` for 200,000 samples: **True**. RNE
narrowing to bf16 is idempotent on an already-bf16 value. Consequence:
bf16-C's "extra" narrow can only be non-degenerate because it targets the
*pre-bias* accumulator, a value the shared final narrow never itself sees —
narrow-count and narrow-position are not independent factors in this
system; position is the only way an extra narrow can do anything. This is
a fact about the arithmetic 0056's structural finding rests on, and it
holds regardless of what follows.

**2b — bias vs variance (300 trials, L=8, M=64,K=192,N=192).** Both paths'
per-element signed error is close to zero-mean at every stage (largest
|mean| relative to std is ~1e-4 relative — noise, not a systematic drift),
and stage-to-stage error correlation is ~0 for both paths (|r| < 0.002
throughout, see the printed table). The std ratio (fp32C/bf16C) sits at
0.996–1.000 across all 8 stages — **no detectable divergence, bias, or
decorrelation difference between the two paths under this model.** If the
real mechanism were "independent, zero-mean per-block bfp16 noise,
narrowed early vs late," it should show approximately this — i.e. this
ablation is consistent with that candidate mechanism being **insufficient
by itself**, not with it being confirmed.

**2c — split-mode block count (0053's sub-question).** Reproducing 0053's
split protocol (K/64 independent 64-wide dispatches, each independently
bf16-C-narrowed, host-summed in fp64) at k_blocks=6 and 24: full ≈ split ≈
1.80e-2 at both block counts — **no shrinkage reproduced** (0053 measured
9.878e-3 < 1.073e-2, a real ~8% shrink; this model shows none). **Analytic
distinction stands regardless**: split-mode narrows the *same reduction's
own* error (a width question — independent bf16-C narrows are host-summed
before the reduction is "done"), chain compounding is about what happens to
error carried *across* stage boundaries (a depth question) — the two are
different axes by construction, whether or not either one shows a
detectable effect in this model. **This sub-question is not answered by
this model**: since neither effect reproduces, I cannot say whether they
are "the same phenomenon" on hardware — only that they are structurally
distinct questions, and that a model with no compounding in either axis
gives no evidence either way.

**2d — plain (non-emulated) bf16.** Same chain machinery, `emulate=False`
(bf16_matmul only, no bfp16 quantisation error). The `1-cos` ratio *does*
show a mild, monotonic pattern here — 0.504 → 0.655 → 0.739 → 0.792 at
stages 1–4 (ratio < 1 means bf16-C is *worse*, shrinking toward parity with
depth) — but at absolute magnitudes near fp32 accumulate's floor (~1e-7 to
~1e-3 range for this shape, not comparable to bfp16's ~1e-2 regime) and in
the *opposite* direction (bf16-C worse, converging toward tie) from what
T26 needs explained. **Answer to the carried sub-question: today's shipping
plain-bf16 path shows a small, structurally-present but sign-and-behaviour-
different effect from the bfp16+bf16-C anomaly — not "the same effect,
smaller," but a different (and much smaller-magnitude, and opposite-
trending) artifact of the same narrow-position asymmetry.** Not itself
concerning for production (bf16-C is not the shipping default), but worth
carrying forward if `--c-bf16` is ever considered for the plain path.

### Part 3 — real weights, real activations (full 6-layer encoder)

```
tensor              fp32C rel_fro  bf16C rel_fro   ratio
emb.ln                  0.0000e+00    0.0000e+00     nan   (identical input, no gemm yet)
L0.ln2                  2.2212e-02    2.2294e-02   0.996
L1.ln2                  3.1051e-02    3.1036e-02   1.000
L2.ln2                  3.6626e-02    3.7964e-02   0.965
L3.ln2                  4.3404e-02    4.4698e-02   0.971
L4.ln2                  4.6940e-02    4.9237e-02   0.953
L5.ln2                  5.9875e-02    6.1793e-02   0.969
last_hidden_state       5.9875e-02    6.1793e-02   0.969

worst 1-cos vs fp32 oracle: fp32C 1.7729e-03  bf16C 1.4831e-03  ratio 1.195x
(production hardware, 0052: fp32-C 2.395e-03, bf16-C 3.615e-04, ratio 6.6x)
```

Real data, real nonlinearities, 6 real layers: **bf16-C is ~1.2× better
than fp32-C**, the *right direction* but **an order of magnitude short of
hardware's 6.6×**, and most per-layer ratios sit *below* 1.0 (fp32-C
slightly better per-layer; the aggregate embedding-level number is the one
that ends up favouring bf16-C). Part 3b (a 4-stage chain of real,
consecutive attention-output-projection weight matrices, real activations)
shows the same story: ratios pinned at 0.994–1.006, i.e. **essentially tied,
same as the synthetic control** — real weight/activation distributions do
**not** produce the anomaly at any scale this model can express, either.
This directly answers the distribution-dependence question the brief
raised (0008 precedent): distribution does not appear to be the missing
ingredient here, at least not as a first-order effect expressible in this
model — the missing factor is elsewhere.

### Interpretation

This model was built to be structurally faithful (same narrowing sites,
same RNE formula 0056 verified bit-identical to production, same
calibrated block-float geometry 0008 fitted against real hardware) and it
robustly — across a full mantissa sweep, 300-trial statistical averaging,
and real production weights/activations — **fails to reproduce 0056's own
measured compounding.** Given the matmul kernel object is proven
bit-identical between the two builds (0053), and my model's accumulation
is exact fp64 (matching CLAUDE.md trap 2/0008's "fp32 accumulate is
effectively free" finding), the gap between this model and hardware must
live in something the model cannot express: the actual numerical behaviour
of the real on-core `narrow_f32_bf16` kernel / accumulator representation
(five 2048-bit accumulator registers, CLAUDE.md trap 3b) at a level below
"apply an RNE bit-formula to a host-side fp32 value" — i.e. something
about how the real device's vectorised narrow instruction, or its
interaction with the emulated bfp16 MAC's actual internal accumulation
order/register packing, differs from a scalar bitwise round. This is a
hardware/kernel-source-reading question (in the spirit of trap 2b's own
discovery method — reading `aie_api/aie.hpp` directly, not simulating),
not a numerical-modelling one, and it is out of this task's scope (host
only, per the brief).

## Problems hit

1. **The control check (Part 1) failed on the first run**, and stayed
   failed after a full mantissa sweep (3–9 bits) and 300-trial statistical
   averaging out to 8 stages — ruling out both "wrong calibration" and
   "single-seed noise" as the cause. Per the brief's explicit instruction
   ("if the model does not reproduce the measured chain... you must say so
   rather than proceeding"), the ablations in Part 2 are reported as
   diagnostic evidence about what the failing model does and does not show,
   not as a confirmed mechanism for T26 itself.
2. Real-data Part 3 initially risked confusing "bf16-C wins on the
   aggregate embedding metric" with "the anomaly reproduces" — the
   per-layer table makes clear most individual layers are closer to tied
   or slightly favour fp32-C (ratios 0.953–1.000), and only the pooled/
   normalized embedding-level `1-cos` shows a consistent (if small, 1.195×)
   advantage for bf16-C. Reported both levels rather than only the
   flattering one.
3. No hardware was touched (per the task brief) — nothing to report about
   NPU contention, process isolation, or device state.

## Artifacts

- `experiments/m5-pretiled-gemm/t26_numerical_model.py` — new, checked in.
- `experiments/m5-pretiled-gemm/artifacts/t26_numerical_model.json` — the
  run above, checked in (all four parts, full per-stage/per-layer detail).

## Next

The numerical "why" is **still open**. What this task adds to the register:
a calibrated, structurally-faithful host model that **rules out** "narrow-
position applied to independent, magnitude-calibrated per-block bfp16
quantisation noise" as a *sufficient* explanation on its own — at any
mantissa calibration, under heavy statistical averaging, and under real
production weights/activations, the effect this model can express tops out
around 1.0–1.2×, not 6.6×. The next step is not another numpy model; it is
reading the actual `narrow_f32_bf16.cc` kernel and the emulated-bfp16
matmul's accumulator handling at the AIE-intrinsic level (mirroring how
trap 2b's floor-rounding bias was found — by reading `aie_api/aie.hpp`
directly, not by simulating), or a hardware ablation that varies *only* the
on-core narrow's timing/position while holding everything else fixed, to
check whether 0056's structural hypothesis holds up with a device-level
control rather than a host-side stand-in for the device's arithmetic.

## Proposed register update

**T26 verdict: still OPEN.** Do NOT mark ANSWERED. The mechanism *class*
from 0056 (structural: early-narrow-before-bias vs late-only-narrow)
remains the best-evidenced hardware-observed correlate of the anomaly, but
this task's host-side model — built specifically to test whether that
structural difference, combined with 0008's calibrated bfp16 noise model,
is *numerically sufficient* to explain the compounding — **does not
reproduce 0056's own chain-probe measurement**, at any mantissa
calibration tested (3–9 bits) or under real production weights/activations
routed through the full 6-layer encoder. This localizes the open question
more precisely than before: it is not simply "narrow position changes how
independent bfp16 quantisation noise propagates" (ruled out here); whatever
it is must live in a hardware/kernel-level numerical detail this host model
cannot express.

Suggested replacement text for T26's entry (append after the existing
0056 paragraph, before the "Closed" table):

> **PROBED FURTHER 2026-08-23** ([`0096`](TASK.md)):
> a host-side numpy/fp64 model, calibrated against 0008's real-hardware
> bfp16 fit (7 bits/element, block=8) and structurally faithful to 0056's
> traced narrowing sites (same bit-identical RNE formula at both the early
> on-core and late host narrow), was built to explain the numerical *why*.
> **It does not reproduce 0056's own chain-probe measurement** — the
> fp32C/bf16C ratio stays pinned at ~0.99–1.00 across 4 (and, with 300-trial
> averaging, 8) chained stages in the model, against hardware's measured
> climb to 1.29–1.67×, at every mantissa calibration tested (3–9 bits) and
> under real production weights/activations through the full 6-layer
> encoder (real data tops out at 1.195× on the embedding-level metric,
> mostly *below* 1.0 per-layer). **This rules out "narrow position applied
> to independent, magnitude-calibrated per-block bfp16 noise" as a
> sufficient explanation on its own** — the matmul kernel object is proven
> bit-identical (0053) and fp32 accumulation is proven exact (0008/trap 2),
> so the missing ingredient must be a hardware/kernel-level numerical
> detail (the actual `narrow_f32_bf16` / accumulator-register behaviour)
> below what a host-side bit-formula model can express, not a
> distribution- or chain-length-dependence effect. The two carried
> sub-questions are also unresolved by this model: 0053's split-mode
> block-count shrinkage does not reproduce either (so whether it is "the
> same phenomenon" as chain compounding is still open, though the two
> remain structurally distinct — width of one reduction vs depth across
> stages — regardless); today's shipping plain-bf16 (non-emulated) path
> shows a small effect in this model, but of the *opposite* sign and trend
> (bf16-C mildly worse, converging toward parity with depth, not diverging)
> — a different artifact of the same narrow-position asymmetry, not "the
> same anomaly, smaller." **Next step is hardware/kernel-source-level, not
> a numpy model**: read `narrow_f32_bf16.cc` and the emulated bfp16 matmul's
> accumulator handling directly (trap 2b's own discovery method), or run a
> device-level ablation that isolates narrow timing/position with real
> hardware arithmetic rather than a host stand-in.

**T23 consequent note.** T26's mechanism is still unexplained, which
matters for T23's framing: the 6.6× accuracy gain (bfp16+bf16-C vs
bfp16+fp32-C) is now **evidenced as hardware-real and reproduced on two
model geometries** (0052 MiniLM, 0053 bge-base) but its **numerical origin
is not understood well enough to predict how it will generalise** — this
task specifically shows a plausible first-principles account (independent
per-block noise, narrow-position) does *not* explain the observed
magnitude, so there is currently **no theoretical basis to expect the gain
to hold at other chain lengths or on other model depths beyond
"empirically, it did on the two geometries measured."** T23 remains an
accuracy decision for the user; this task adds that the 6.6× number should
be treated as an **empirical, unexplained result specific to what has been
measured**, not (yet) as a generalisable property of the bfp16+bf16-C
datapath. No change to T23's PASS/FAIL gate table (0052/0053's numbers
stand); this only tempers the confidence with which the gain can be
extrapolated to bge-large or to a different layer count.

**`tasks/README.md` index row:**

```
| [0096](0096-t26-numerical-mechanism/TASK.md) | T26 numerical mechanism model — control NOT reproduced; rules out independent-noise narrow-position as sufficient, real cause still open | research (T26) | done |
```
