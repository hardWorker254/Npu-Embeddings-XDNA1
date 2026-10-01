//===- gelu_erf.cc -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the exact-erf GELU, which is the activation a Whisper
// container declares. SPDX-License-Identifier: Apache-2.0
//
// WHY THIS FILE EXISTS
// --------------------
// kernels/gelu_poly.cc is a degree-8 fit of the even part, and it measures
// 2.49e-3 relative against exact erf -- which is the bf16 output floor times
// 1.01 on an embedder and a DIFFERENT ACTIVATION on a Whisper. The packer
// refuses a checkpoint whose `activation` is not `gelu`, and tools/pack/packers/
// whisper.py:345 is what makes that binding: this model runs the erf GELU, so a
// tanh-polynomial GELU is not a faster version of the same function, it is a
// different model, and its 2.5e-3 is larger than the error the bf16 datapath
// itself introduces. The host reference is 0.5*x*(1+erf(x/sqrt(2))) in double
// (app::gelu_erf_exact), and this kernel computes the same function.
//
// HOW, WITHOUT A TRANSCENDENTAL
// ----------------------------
// The AIE API has no vector erf and no vector exp (aie_api/detail/elementary.hpp
// says so in as many words), which is why kernels/softmax.cc carries its own
// exp2. This one borrows that: erf's A&S 7.1.26 rational form needs one exp, one
// reciprocal and five Horner steps, and the exp is exp2_poly over a clamped
// base-2 argument -- the same construction, the same clamp, the same reason
// (a negative exponent field is a NaN bit pattern, not a small number).
//
//   |erf(u)| = 1 - t P(t) exp(-u^2),  t = 1/(1 + 0.3275911 |u|)
//
// with P(t) = a1 + a2 t + ... + a5 t^4. The t in front is part of the formula,
// not a normalisation: A&S 7.1.26 is 1 - (a1 t + a2 t^2 + ... + a5 t^5) e^-x^2,
// and dropping that leading t makes erf(0.7) read 0.609 instead of 0.683 -- a
// GELU that is 0.8 instead of 0.84 at x = 1, which is a different activation and
// not a rounding of this one.
//
// The ABSOLUTE value is what is computed, and the sign is free because GELU's
// even part is even -- the same identity gelu_poly.cc is built on:
//
//   GELU(x) = 0.5 x + 0.5 |x| |erf(x / sqrt 2)|
//
// which is 0.5x(1 + erf(u)) for x >= 0 and 0.5x(1 - erf(|u|)) for x < 0. No
// select, no mask, no branch: the alternative is a signed erf and a vector
// comparison, and the identity is the reason this file has neither.
//
// The approximation error of A&S 7.1.26 is 1.5e-7 absolute, which is four
// orders below the bf16 grid the result is stored on (2^-8 relative) and five
// below the 1e-2 the conv path's own activations are held to. Rounding on the
// way out is RNE, like every other kernel in this tree that narrows (the AIE
// default is floor, and a systematic downward bias on an activation is a
// systematic bias on every layer after it).

#include "aie_kernel_utils.h"
#include "exp2_poly.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// A&S 7.1.26: |erf(x)| ~ 1 - t (a1 + t (a2 + t (a3 + t (a4 + t a5)))) exp(-x^2),
// t = 1/(1 + p x). p = 0.3275911.
#define GE_P 0.3275911f
#define GE_A1 0.254829592f
#define GE_A2 (-0.284496736f)
#define GE_A3 1.421413741f
#define GE_A4 (-1.453152027f)
#define GE_A5 1.061405429f
#define GE_INV_SQRT2 0.70710678118654752440f
// The base-2 argument floor. 2^-125 is 2.4e-38, so the exp it returns is zero
// for every purpose here, and -125 keeps k + 127 = 2 inside a legal exponent
// field -- the exact failure softmax.cc documents for the unclamped version.
#define GE_ARG_FLOOR -125.0f

// |x| through max(x, -x), never aie::abs: on aie2 that resolves to the INTEGER
// vabs.gtz32 and is not |x| at all (kernels/gelu_poly.cc, with the measurement:
// GELU(-1) came out -5.2e-05).
static inline aie::vector<float, 16> ge_fabs(const aie::vector<float, 16> &x) {
  return aie::max(x, aie::sub(aie::broadcast<float, 16>(0.0f), x));
}

static inline void gelu_erf_impl(bfloat16 *restrict in, bfloat16 *restrict out,
                                 int n) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<float, 16> vone = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vhalf = aie::broadcast<float, 16>(0.5f);
  const aie::vector<float, 16> vs2 = aie::broadcast<float, 16>(GE_INV_SQRT2);
  const aie::vector<float, 16> vp = aie::broadcast<float, 16>(GE_P);
  const aie::vector<float, 16> va1 = aie::broadcast<float, 16>(GE_A1);
  const aie::vector<float, 16> va2 = aie::broadcast<float, 16>(GE_A2);
  const aie::vector<float, 16> va3 = aie::broadcast<float, 16>(GE_A3);
  const aie::vector<float, 16> va4 = aie::broadcast<float, 16>(GE_A4);
  const aie::vector<float, 16> va5 = aie::broadcast<float, 16>(GE_A5);
  const aie::vector<float, 16> vlog2e =
      aie::broadcast<float, 16>(1.4426950408889634f);
  const aie::vector<float, 16> vfloor =
      aie::broadcast<float, 16>(GE_ARG_FLOOR);
  const aie::vector<float, 16> vzero = aie::broadcast<float, 16>(0.0f);

  for (int i = 0; i < n; i += 16) {
    // The widen rides the accumulator's native cast, not a multiply by 1.0
    // (tasks/0045: the latter is an emulated fp32 multiply, four instructions
    // where a cast is one).
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::load_v<16>(in + i));
    const aie::vector<float, 16> x = acc.to_vector<float>();

    const aie::vector<float, 16> u = aie::mul(x, vs2).to_vector<float>();
    const aie::vector<float, 16> au = ge_fabs(u);
    // Only `mul` and `div` return an accumulator; add/sub/max already answer a
    // vector, and asking a vector for to_vector is a compile error rather than a
    // silent no-op.
    const aie::vector<float, 16> t = aie::div(
        vone, aie::add(vone, aie::mul(vp, au).to_vector<float>()))
                                          .to_vector<float>();
    aie::vector<float, 16> poly = va5;
    poly = aie::add(aie::mul(poly, t).to_vector<float>(), va4);
    poly = aie::add(aie::mul(poly, t).to_vector<float>(), va3);
    poly = aie::add(aie::mul(poly, t).to_vector<float>(), va2);
    poly = aie::add(aie::mul(poly, t).to_vector<float>(), va1);
    // -u^2 in log2, clamped. u^2 overflows to +inf for |x| above ~1.9e19, and
    // -inf clamped to the floor is the right answer there, not a NaN: the exp
    // becomes 2^-125 and the even part becomes 1, which is erf to every digit a
    // bf16 can hold.
    const aie::vector<float, 16> z =
        aie::mul(aie::sub(vzero, aie::mul(u, u).to_vector<float>()), vlog2e)
            .to_vector<float>();
    const aie::vector<float, 16> ex = exp2_poly(aie::max(z, vfloor));
    // t * P(t): the leading t of a1 t + ... + a5 t^5, which the Horner above
    // leaves outside the polynomial.
    const aie::vector<float, 16> tp =
        aie::mul(t, poly).to_vector<float>();
    const aie::vector<float, 16> eabs =
        aie::sub(vone, aie::mul(tp, ex).to_vector<float>());

    // GELU(x) = 0.5x + 0.5|x| |erf(x/sqrt 2)| -- see the header for why the
    // absolute value is enough.
    const aie::vector<float, 16> ax = ge_fabs(x);
    const aie::vector<float, 16> y = aie::add(
        aie::mul(x, vhalf).to_vector<float>(),
        aie::mul(aie::mul(ax, vhalf).to_vector<float>(), eabs)
            .to_vector<float>());
    aie::accum<accfloat, 16> out_acc;
    out_acc.from_vector(y);
    aie::store_v(out + i, out_acc.to_vector<bfloat16>());
  }
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

// 1024 elements per call, the tile the eltwise GELU program splits at.
void gelu_erf_bf16(bfloat16 *restrict in, bfloat16 *restrict out) {
  gelu_erf_impl(in, out, 1024);
}

// 4096, for the design that asks for the wider tile.
void gelu_erf_bf16_4k(bfloat16 *restrict in, bfloat16 *restrict out) {
  gelu_erf_impl(in, out, 4096);
}

}  // extern "C"
#endif  // NPUE_ELTWISE_IMPL_ONLY
