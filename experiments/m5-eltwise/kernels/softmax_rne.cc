//===- softmax_rne.cc ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- softmax with ONE line changed: the rounding mode.
// SPDX-License-Identifier: Apache-2.0
//
// The companion to gelu_poly_rne.cc, and the one with a REASON TO FAIL.
//
// tasks/0044: the AIE default rounding mode is `floor` and we have never set
// it. whisper-xdna measured `conv_even` as a free accuracy win on GELU and
// reported that the same change BREAKS their softmax (cosine 0.879), because
// the stock `getExpBf16` lookup table is calibrated for the default mode.
//
// That objection should not apply to us: tasks/0030 replaced the library exp
// with our own `exp2_poly`, so there is no mode-calibrated table left to
// break. "Should not apply" is a prediction, and this file is how it gets
// tested rather than assumed -- if `conv_even` makes softmax worse here, the
// mechanism is not the LUT and the finding needs re-thinking.
//
// A SECOND, INDEPENDENT PREDICTION lives in this kernel that does not live in
// GELU. Softmax's output rows must sum to 1. Under `floor` every element is
// rounded DOWN, so the row sums can only come in LOW -- and the baseline run
// reports exactly that: min 0.994581, max 1.000000, worst |1-sum| 5.419e-03,
// with no row above 1. Under `conv_even` the error becomes symmetric, so row
// sums should STRADDLE 1.0 and max should exceed it. That is a signature, not
// a magnitude, and it can confirm the mechanism independently of any rel_fro.
//
// Separate FILE rather than a flag, for the reason gelu_poly_rne.cc gives.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#define NPUE_ELTWISE_IMPL_ONLY
#include "softmax.cc"
#undef NPUE_ELTWISE_IMPL_ONLY

extern "C" {

// Identical to softmax_poly_bf16 / softmax_poly_il4_bf16 except for the
// set_rounding call -- same 64 rows per call, so the fifo geometry the harness
// builds is unchanged and this is a true A/B.
void softmax_poly_rne_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  softmax_impl<true>(input, output, 64);
}

// Needs the 0x2000 worker stack for the same reason softmax_poly_il4_bf16 does.
void softmax_poly_il4_rne_bf16(bfloat16 *restrict input,
                               bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  softmax_il4_impl(input, output, 64);
}

}  // extern "C"
