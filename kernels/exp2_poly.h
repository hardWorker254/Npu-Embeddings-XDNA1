//===- exp2_poly.h ------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- exp2 without a library transcendental.
// SPDX-License-Identifier: Apache-2.0
//
// Shared by softmax.cc and the probe that validates it, so the thing measured
// is byte-for-byte the thing shipped.
//
//     2^x = 2^k * 2^f,   k = trunc(x),  f = x - k
//
// 2^k is exact: an integer add of 127 and a 23-bit shift into a float's
// exponent field. Only 2^f is approximated, and it is smooth on a bounded
// interval -- degree 7 gives 1.75e-07, four orders below the bf16 grid.
//
// Fitted on [-1, 1] rather than [-1, 0] so the result does not depend on
// whether aie::to_fixed truncates or rounds.
//
// CALLER CONTRACT: x >= -126, or the exponent field goes negative and the bit
// pattern is a NaN rather than a small number.

#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

// 2^f on [-1, 1], Chebyshev-fitted, Horner order. Max relative error 1.75e-07.
#define EXP2_C0 1.5483275463e-05f
#define EXP2_C1 1.5669833174e-04f
#define EXP2_C2 1.3331825236e-03f
#define EXP2_C3 9.6164605538e-03f
#define EXP2_C4 5.5504156855e-02f
#define EXP2_C5 2.4022684109e-01f
#define EXP2_C6 6.9314717694e-01f
#define EXP2_C7 9.9999998955e-01f

// exp2 for x in [-100, 0]. Vector ops only: no library transcendental.
static inline aie::vector<float, 16> exp2_poly(const aie::vector<float, 16> &x) {
  const aie::vector<int32_t, 16> v127 = aie::broadcast<int32_t, 16>(127);
  aie::vector<int32_t, 16> k = aie::to_fixed<int32_t>(x);
  aie::vector<float, 16> f = aie::sub(x, aie::to_float<float>(k));

  aie::vector<float, 16> p = aie::broadcast<float, 16>(EXP2_C0);
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C1));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C2));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C3));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C4));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C5));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C6));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(EXP2_C7));

  // 2^k by writing the IEEE exponent field. Valid only while k + 127 stays a
  // legal exponent, i.e. k >= -126; below that the field goes negative and the
  // bit pattern is a NaN, not a small number. The caller clamps the BASE-2
  // argument for exactly this reason -- clamping the natural-log difference
  // instead is what produced 256 NaNs on the first attempt, because
  // -100 * log2(e) = -144.27 and 127 - 144 is negative.
  aie::vector<int32_t, 16> bits = aie::upshift(aie::add(k, v127), 23);
  return aie::mul(p, aie::vector_cast<float>(bits)).to_vector<float>();
}


