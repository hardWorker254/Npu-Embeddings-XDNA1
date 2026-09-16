//===- layernorm.cc -----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- LayerNorm for post-LN BERT: fp32, eps inside the sqrt,
// biased variance, learned per-channel gamma/beta.
// SPDX-License-Identifier: Apache-2.0
//
// WHY NOT THE SHIPPED aie_kernels/aie2p/layer_norm.cc
// ---------------------------------------------------
// Three things in its source make it unusable here, and none needed measuring:
//
//   1. `const float gamma = 1.0f; const float beta = 0.0f;` -- hardcoded. It
//      takes no learned parameters at all, and BERT's LayerNorm has 384 of each
//      per site.
//   2. `constexpr float epsilon = 1e-5f`. MiniLM uses **1e-12**
//      (docs/04-model). Four orders of magnitude, inside a square root.
//   3. `variance = (sum_sq / cols) - mean*mean` -- the naive one-pass formula.
//      It subtracts two nearly equal large numbers, and docs/04-model records
//      that post-LN BERT carries hidden dims at +/-50-100 among values near
//      +/-1. That is precisely the input that makes this formula cancel.
//
// WHAT THIS DOES INSTEAD
// ----------------------
//   * TWO PASSES. sum -> mean, then sum of (x-mean)^2 -> variance. Costs one
//     extra read of a row that is already resident in L1, and is stable no
//     matter how large the outliers are.
//   * BIASED variance (divide by N, not N-1). PyTorch's convention; using the
//     unbiased estimator is one of the documented landmines.
//   * eps INSIDE the sqrt: x / sqrt(var + eps), not sqrt(var) + eps.
//   * fp32 throughout. tasks/0016 measured aie::vector<float> at a full 24-bit
//     mantissa on both add and multiply, so this is real fp32, not nominal.
//   * gamma and beta as fp32 inputs -- .npue stores them fp32 by design because
//     they are 384 floats and numerically sensitive.
//
// research/notes/0001 forbids scalar float in a kernel body. The per-row
// scalars here are three: two multiplies by a precomputed reciprocal (never a
// divide) and one invsqrt. That is 3 per 384 elements, not 3 per element.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#define LN_COLS 384
#define LN_VECS (LN_COLS / 16)
#define LN_EPS 1e-12f
#define LN_INV_COLS (1.0f / (float)LN_COLS)

// Fast approximate inverse square root without sqrtf (bare-metal compatible)
static inline float fast_invsqrt(float x) {
    // Initial approximation using bit manipulation (Quake III algorithm)
    float xhalf = 0.5f * x;
    int i = *reinterpret_cast<int*>(&x);
    i = 0x5f3759df - (i >> 1);  // magic number for initial guess
    float y = *reinterpret_cast<float*>(&i);
    // Two Newton-Raphson iterations for accuracy
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}

// gamma and beta arrive as ONE buffer of 2*LN_COLS floats, gamma first.
//
// Not a packaging preference -- a core tile has only 2 input and 2 output DMA
// channels, and passing them separately makes 3 inputs:
//   "tile (0,3) requires 3 input/1 output DMA channels, but only 2 input/2
//    output available"
// One buffer, one channel. See tasks/0020.
#ifndef NPUE_ELTWISE_IMPL_ONLY
void layernorm_impl(bfloat16 *restrict input, float *restrict params,
                    bfloat16 *restrict output, const int32_t rows) {
  float *gamma = params;
  float *beta = params + LN_COLS;
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  // Narrowing fp32 -> bf16 also goes through an accumulator: to_vector<T>() is
  // an accum method, never a vector one. Multiplying by 1.0 is the exact
  // identity that gets us one. Same trap as tasks/0014, second time.
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);

  for (int r = 0; r < rows; r++) {
    bfloat16 *in = input + r * LN_COLS;
    bfloat16 *out = output + r * LN_COLS;

    // -- pass 1: mean ------------------------------------------------------
    aie::vector<float, 16> acc = aie::zeros<float, 16>();
    for (int i = 0; i < LN_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      // Widen through a multiply by 1.0: exact, and to_vector<T>() is an
      // accumulator method, not a vector one.
      acc = aie::add(acc, aie::mul(xb, vone_bf).to_vector<float>());
    }
    float mean = aie::reduce_add(acc) * LN_INV_COLS;
    aie::vector<float, 16> mean_v = aie::broadcast<float, 16>(mean);

    // -- pass 2: variance about the measured mean --------------------------
    // Two passes rather than E[x^2] - mean^2. The extra read is of a row
    // already in L1; the stability is not optional on this data.
    aie::vector<float, 16> acc2 = aie::zeros<float, 16>();
    for (int i = 0; i < LN_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      aie::vector<float, 16> d = aie::sub(x, mean_v);
      acc2 = aie::add(acc2, aie::mul(d, d).to_vector<float>());
    }
    float variance = aie::reduce_add(acc2) * LN_INV_COLS;   // biased: / N
    float inv_std = fast_invsqrt(variance + LN_EPS);        // eps INSIDE
    aie::vector<float, 16> inv_std_v = aie::broadcast<float, 16>(inv_std);

    // -- normalize, scale, shift -------------------------------------------
    for (int i = 0; i < LN_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      aie::vector<float, 16> d = aie::sub(x, mean_v);
      aie::vector<float, 16> nrm = aie::mul(d, inv_std_v).to_vector<float>();
      aie::vector<float, 16> g = aie::load_v<16>(gamma + i * 16);
      aie::vector<float, 16> b = aie::load_v<16>(beta + i * 16);
      aie::vector<float, 16> y = aie::add(aie::mul(nrm, g).to_vector<float>(), b);
      aie::store_v(out + i * 16, aie::mul(y, vone_f).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}


extern "C" {

// 16 rows of 384 per call.
//
// Set by the L1 budget, not by taste. With double buffering on input and
// output: 2*(rows*384*2) * 2 + 2*384*4 < 65536 gives rows <= 20. 64 rows was
// the first attempt and produced CLAUDE.md trap 3 --
// "'aie.tile' op Basic sequential allocation failed" -- at ~200 KB against a
// 64 KB L1. 16 rows costs 52,224 B and leaves headroom.
void layernorm_bf16(bfloat16 *restrict input, float *restrict params,
                    bfloat16 *restrict output) {
  layernorm_impl(input, params, output, 16);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY

// ---------------------------------------------------------------------------
// FOUR ROWS INTERLEAVED (tasks/0031). The one-row impl above is latency
// bound, not throughput bound: at 8 columns it measures ~7,800 cycles per row
// against ~1,000 of issued work. Every pass is a serial dependency chain --
// pass 1 and 2 are 24 dependent accumulator adds, pass 3 a 6-op chain per
// vector -- and with one row in flight the pipeline drains between every
// step. Same diagnosis and same cure as the 4-chain GELU (tasks/0026): give
// the compiler four INDEPENDENT rows to interleave.
//
// Numerics are bit-identical to the one-row impl: each row performs the same
// operations in the same order on its own data. No summation order changes.
//
// Pass 3 also loads gamma/beta once per vector index instead of once per row
// per vector index -- shared across the four rows, a pure win.
void layernorm_il4_impl(bfloat16 *restrict input, float *restrict params,
                        bfloat16 *restrict output, const int32_t rows) {
  float *gamma = params;
  float *beta = params + LN_COLS;
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);

  for (int r = 0; r < rows; r += 4) {
    bfloat16 *in0 = input + (r + 0) * LN_COLS;
    bfloat16 *in1 = input + (r + 1) * LN_COLS;
    bfloat16 *in2 = input + (r + 2) * LN_COLS;
    bfloat16 *in3 = input + (r + 3) * LN_COLS;
    bfloat16 *out0 = output + (r + 0) * LN_COLS;
    bfloat16 *out1 = output + (r + 1) * LN_COLS;
    bfloat16 *out2 = output + (r + 2) * LN_COLS;
    bfloat16 *out3 = output + (r + 3) * LN_COLS;

    // -- pass 1: means, four independent accumulator chains ----------------
    aie::vector<float, 16> a0 = aie::zeros<float, 16>();
    aie::vector<float, 16> a1 = aie::zeros<float, 16>();
    aie::vector<float, 16> a2 = aie::zeros<float, 16>();
    aie::vector<float, 16> a3 = aie::zeros<float, 16>();
    for (int i = 0; i < LN_VECS; i++) {
      a0 = aie::add(a0, aie::mul(aie::load_v<16>(in0 + i * 16), vone_bf)
                            .to_vector<float>());
      a1 = aie::add(a1, aie::mul(aie::load_v<16>(in1 + i * 16), vone_bf)
                            .to_vector<float>());
      a2 = aie::add(a2, aie::mul(aie::load_v<16>(in2 + i * 16), vone_bf)
                            .to_vector<float>());
      a3 = aie::add(a3, aie::mul(aie::load_v<16>(in3 + i * 16), vone_bf)
                            .to_vector<float>());
    }
    const aie::vector<float, 16> mean0 =
        aie::broadcast<float, 16>(aie::reduce_add(a0) * LN_INV_COLS);
    const aie::vector<float, 16> mean1 =
        aie::broadcast<float, 16>(aie::reduce_add(a1) * LN_INV_COLS);
    const aie::vector<float, 16> mean2 =
        aie::broadcast<float, 16>(aie::reduce_add(a2) * LN_INV_COLS);
    const aie::vector<float, 16> mean3 =
        aie::broadcast<float, 16>(aie::reduce_add(a3) * LN_INV_COLS);

    // -- pass 2: variances about the measured means -------------------------
    a0 = aie::zeros<float, 16>();
    a1 = aie::zeros<float, 16>();
    a2 = aie::zeros<float, 16>();
    a3 = aie::zeros<float, 16>();
    for (int i = 0; i < LN_VECS; i++) {
      aie::vector<float, 16> d0 = aie::sub(
          aie::mul(aie::load_v<16>(in0 + i * 16), vone_bf).to_vector<float>(),
          mean0);
      aie::vector<float, 16> d1 = aie::sub(
          aie::mul(aie::load_v<16>(in1 + i * 16), vone_bf).to_vector<float>(),
          mean1);
      aie::vector<float, 16> d2 = aie::sub(
          aie::mul(aie::load_v<16>(in2 + i * 16), vone_bf).to_vector<float>(),
          mean2);
      aie::vector<float, 16> d3 = aie::sub(
          aie::mul(aie::load_v<16>(in3 + i * 16), vone_bf).to_vector<float>(),
          mean3);
      a0 = aie::add(a0, aie::mul(d0, d0).to_vector<float>());
      a1 = aie::add(a1, aie::mul(d1, d1).to_vector<float>());
      a2 = aie::add(a2, aie::mul(d2, d2).to_vector<float>());
      a3 = aie::add(a3, aie::mul(d3, d3).to_vector<float>());
    }
    const aie::vector<float, 16> is0 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a0) * LN_INV_COLS + LN_EPS));
    const aie::vector<float, 16> is1 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a1) * LN_INV_COLS + LN_EPS));
    const aie::vector<float, 16> is2 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a2) * LN_INV_COLS + LN_EPS));
    const aie::vector<float, 16> is3 = aie::broadcast<float, 16>(
        fast_invsqrt(aie::reduce_add(a3) * LN_INV_COLS + LN_EPS));

    // -- pass 3: normalize, scale, shift; gamma/beta shared across rows -----
    for (int i = 0; i < LN_VECS; i++) {
      const aie::vector<float, 16> g = aie::load_v<16>(gamma + i * 16);
      const aie::vector<float, 16> b = aie::load_v<16>(beta + i * 16);
      aie::vector<float, 16> n0 = aie::mul(
          aie::sub(aie::mul(aie::load_v<16>(in0 + i * 16), vone_bf)
                       .to_vector<float>(), mean0), is0).to_vector<float>();
      aie::vector<float, 16> n1 = aie::mul(
          aie::sub(aie::mul(aie::load_v<16>(in1 + i * 16), vone_bf)
                       .to_vector<float>(), mean1), is1).to_vector<float>();
      aie::vector<float, 16> n2 = aie::mul(
          aie::sub(aie::mul(aie::load_v<16>(in2 + i * 16), vone_bf)
                       .to_vector<float>(), mean2), is2).to_vector<float>();
      aie::vector<float, 16> n3 = aie::mul(
          aie::sub(aie::mul(aie::load_v<16>(in3 + i * 16), vone_bf)
                       .to_vector<float>(), mean3), is3).to_vector<float>();
      aie::vector<float, 16> y0 =
          aie::add(aie::mul(n0, g).to_vector<float>(), b);
      aie::vector<float, 16> y1 =
          aie::add(aie::mul(n1, g).to_vector<float>(), b);
      aie::vector<float, 16> y2 =
          aie::add(aie::mul(n2, g).to_vector<float>(), b);
      aie::vector<float, 16> y3 =
          aie::add(aie::mul(n3, g).to_vector<float>(), b);
      aie::store_v(out0 + i * 16, aie::mul(y0, vone_f).to_vector<bfloat16>());
      aie::store_v(out1 + i * 16, aie::mul(y1, vone_f).to_vector<bfloat16>());
      aie::store_v(out2 + i * 16, aie::mul(y2, vone_f).to_vector<bfloat16>());
      aie::store_v(out3 + i * 16, aie::mul(y3, vone_f).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

// Same 16 rows per call as layernorm_bf16 -- the fifo geometry is unchanged,
// only the core's schedule differs. Needs the 0x2000 worker stack: the
// interleaved body keeps ~16 vectors live, and 0xD00 corrupts silently
// (tasks/0026, tasks/0030 section 5a -- third time this trap has bitten).
void layernorm_il4_bf16(bfloat16 *restrict input, float *restrict params,
                        bfloat16 *restrict output) {
  layernorm_il4_impl(input, params, output, 16);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY
