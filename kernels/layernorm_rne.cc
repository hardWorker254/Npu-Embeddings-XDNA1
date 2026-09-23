//===- layernorm_rne.cc -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- LayerNorm with ONE line changed: the rounding mode.
// SPDX-License-Identifier: Apache-2.0
//
// Third of the set (gelu_poly_rne.cc, softmax_rne.cc), completing the sweep
// of every kernel we have that narrows fp32 to bf16 on its way out.
//
// tasks/0044: the AIE default is `aie::rounding_mode::floor` and we had never
// called set_rounding, so every bf16 store this project ever executed rounded
// toward negative infinity. GELU improved 4.312e-03 -> 2.494e-03 and softmax
// 4.278e-03 -> 3.325e-03 from this line alone.
//
// LayerNorm is the interesting third case because it is the one whose output
// is CENTRED: it subtracts the row mean, so roughly half the values are
// negative. A downward bias on a symmetric distribution shifts the whole row
// rather than shrinking it, which is a different error shape from softmax's
// one-sided sum deficit. Whether that makes the mode matter more or less here
// is not predictable from the other two, which is why it gets measured.
//
// Separate FILE rather than a flag, for the reason gelu_poly_rne.cc gives.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// NOTE: unlike gelu_poly.cc and softmax.cc, layernorm.cc puts `layernorm_impl`
// ITSELF inside the `#ifndef NPUE_ELTWISE_IMPL_ONLY` block, so defining that
// guard here strips the very function this file wraps -- Peano says "use of
// undeclared identifier 'layernorm_impl'; did you mean 'layernorm_il4_impl'?",
// the il4 one surviving because it sits outside the guard. So include it
// plainly. The baseline entry points come along for the ride, which is
// harmless: they have different names from the two below and no design links
// both objects.
#include "layernorm.cc"

extern "C" {

// Identical to layernorm_bf16 / layernorm_il4_bf16 except for set_rounding --
// same 16 rows per call, so the fifo geometry is unchanged and this is a true
// A/B against the baseline variants.
void layernorm_rne_bf16(bfloat16 *restrict input, float *restrict params,
                        bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  layernorm_impl(input, params, output, 16);
}

void layernorm_il4_rne_bf16(bfloat16 *restrict input, float *restrict params,
                            bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  layernorm_il4_impl(input, params, output, 16);
}

}  // extern "C"
