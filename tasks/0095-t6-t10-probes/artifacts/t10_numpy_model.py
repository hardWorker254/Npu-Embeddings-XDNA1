# T10 -- numpy accuracy model for exp2_poly, over softmax's actual input
# range, against an fp64 reference. aie::exp2 has no public source (a
# hardware/library primitive) so it cannot be independently modeled here;
# its accuracy against the SAME fp64 reference (np.exp2 exact) is the
# already-measured hardware number from tasks/0021 (1.711e-02), cited
# alongside for the decision.
#
# Coefficients are copied verbatim from experiments/m5-eltwise/kernels/exp2_poly.h
import numpy as np
from ml_dtypes import bfloat16

C = [1.5483275463e-05, 1.5669833174e-04, 1.3331825236e-03, 9.6164605538e-03,
     5.5504156855e-02, 2.4022684109e-01, 6.9314717694e-01, 9.9999998955e-01]


def exp2_poly_f64(x):
    """Bit-for-bit the same Horner recurrence and exponent-field trick as
    exp2_poly.h, done in float64 to isolate the ALGORITHM's error from any
    bf16 storage-rounding effects."""
    x = np.asarray(x, dtype=np.float64)
    k = np.trunc(x)  # aie::to_fixed<int32_t> truncates toward zero, same as C trunc
    f = x - k
    p = np.full_like(f, C[0])
    for c in C[1:]:
        p = p * f + c
    # 2^k by writing the IEEE-754 double exponent field directly (52-bit
    # mantissa base, 11-bit exponent, bias 1023) -- the float64 analogue of
    # exp2_poly.h's "add 127, shift into the fp32 exponent field".
    two_k = np.exp2(k)  # exact for integer k (no rounding at all)
    return p * two_k


def exp2_poly_bf16_pipeline(x_bf16_in, rounding="floor"):
    """Reproduce the kernel's actual bf16 datapath: input already bf16
    (as softmax hands it), widened to fp32 for the Horner chain (this is
    what the AIE core does -- fp32 vector arithmetic throughout, see
    exp2_poly.h), then the *single* SRS back to bf16 that the kernel does
    when it stores ev[i]. `rounding` selects floor (AIE default, never
    overridden by this kernel -- CLAUDE.md trap 2b) or conv_even (round to
    nearest, ties to even)."""
    x = x_bf16_in.astype(np.float32).astype(np.float64)
    k = np.trunc(x)
    f = (x - k).astype(np.float32).astype(np.float64)  # fp32 subtract, as the core does
    p = np.full_like(f, np.float32(C[0]))
    for c in C[1:]:
        p = (p * f).astype(np.float32).astype(np.float64) + np.float32(c)
    two_k = np.exp2(k)  # exact for integer k up to bf16/fp32 range used here
    result_fp32 = (p * two_k).astype(np.float32)

    if rounding == "floor":
        # bf16 = fp32 truncated to 7 mantissa bits (AIE default rounding
        # mode is floor -- "always round towards negative infinity", not
        # round-to-nearest). Implemented as: drop the low 16 bits of the
        # fp32 bit pattern (equivalent to floor toward -inf when the value
        # is positive, which every exp2 output is).
        u = result_fp32.view(np.uint32)
        bf16_bits = (u & 0xFFFF0000).astype(np.uint32)
        return bf16_bits.view(np.float32).astype(np.float64)
    elif rounding == "conv_even":
        return result_fp32.astype(bfloat16).astype(np.float64)
    else:
        raise ValueError(rounding)


def main():
    # softmax's actual input range for exp2_poly: arg = max(d*log2e, -120)
    # with d = max(x-m, -100), so the argument to exp2_poly is always <= 0
    # and floored at -120 (SM_ARG_FLOOR, softmax.cc). Sample it the way
    # 0021's exp2_probe did: a dense ramp over the full domain, PLUS a
    # realistic pre-softmax-shifted-scores sample so the report isn't
    # biased toward the (rare) deep-clamp tail.
    N = 100_000
    rng = np.random.default_rng(0)

    # (a) dense ramp over the full legal domain [-120, 0], same as 0021.
    ramp = np.linspace(-120.0, 0.0, N)

    # (b) realistic distribution: x - max over a row of BERT attention
    # scores concentrates most mass near 0 with a long negative tail before
    # the -100 clamp ever engages (docs/04-model: post-softmax scores are
    # well-behaved once masked entries are clamped upstream). Model with a
    # half-Cauchy-ish heavy tail then clamp exactly like the kernel.
    d = -np.abs(rng.standard_cauchy(N)) * 8.0
    d = np.maximum(d, -100.0)
    realistic = np.maximum(d * np.float64(1.4426950408889634), -120.0)

    for label, xs in (("dense ramp [-120,0]", ramp),
                      ("realistic pre-softmax d*log2e", realistic)):
        xs_bf16 = xs.astype(bfloat16)  # what actually reaches the kernel: bf16 arg
        ref = np.exp2(xs.astype(np.float64))          # fp64 exact
        ref_from_bf16in = np.exp2(xs_bf16.astype(np.float64))  # fp64 exact, bf16-quantized input

        # --- algorithm-only error (float64 Horner, no bf16 anywhere) ---
        got_f64 = exp2_poly_f64(xs)
        rel_algo = np.abs(got_f64 - ref) / np.maximum(ref, 1e-300)

        # --- full bf16 datapath, AIE DEFAULT rounding (floor) ---
        got_floor = exp2_poly_bf16_pipeline(xs_bf16, "floor")
        rel_floor = np.abs(got_floor - ref_from_bf16in) / np.maximum(ref_from_bf16in, 1e-300)

        # --- full bf16 datapath, conv_even (round-to-nearest-even) ---
        got_rne = exp2_poly_bf16_pipeline(xs_bf16, "conv_even")
        rel_rne = np.abs(got_rne - ref_from_bf16in) / np.maximum(ref_from_bf16in, 1e-300)

        print(f"\n=== {label} (N={N}) ===")
        # Only score where the reference isn't flushed to zero in fp64 either
        # (exp2(-120) ~ 6e-37, still representable) -- all N points qualify.
        print(f"  algorithm only (fp64 Horner)      : max {rel_algo.max():.3e}  rms {np.sqrt((rel_algo**2).mean()):.3e}")
        print(f"  full bf16 datapath, floor (AIE default) : max {rel_floor.max():.3e}  rms {np.sqrt((rel_floor**2).mean()):.3e}")
        print(f"  full bf16 datapath, conv_even           : max {rel_rne.max():.3e}  rms {np.sqrt((rel_rne**2).mean()):.3e}")
        ratio = rel_floor.max() / max(rel_rne.max(), 1e-300)
        print(f"  floor/conv_even max-error ratio         : {ratio:.3f}x")


if __name__ == "__main__":
    main()
