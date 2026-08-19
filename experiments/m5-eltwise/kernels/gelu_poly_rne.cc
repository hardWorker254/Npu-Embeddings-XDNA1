//===- gelu_poly_rne.cc -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- gelu_poly with ONE line changed: the rounding mode.
// SPDX-License-Identifier: Apache-2.0
//
// WHY
// ---
// tasks/0044 found that the AIE default rounding mode is
// `aie::rounding_mode::floor` -- "always round towards negative infinity", as
// aie_api/aie.hpp says twice and aie_types.hpp defines. We have never called
// `aie::set_rounding` in any kernel, so every bf16 SRS this project has ever
// executed carries a SYSTEMATIC DOWNWARD BIAS rather than symmetric noise.
//
// gelu_poly.cc is the sharpest possible test of that, because its own comment
// says "Horner, fp32 throughout, rounded exactly once on the store below."
// One rounding, and it is the biased one. If the mode matters anywhere, it
// matters here, and nothing else in the kernel changes.
//
// THE PREDICTION THIS IS BUILT TO FALSIFY
// ---------------------------------------
// The baseline harness reports three numbers that separate the error sources:
//
//   NPU vs exact-erf golden                4.312e-03   <- what we care about
//   NPU vs CPU model of the same polynomial 3.886e-03  <- implementation error
//   CPU model vs golden (design limit)      1.923e-03  <- the polynomial itself
//
// The CPU model evaluates the identical coefficients in numpy and rounds to
// bf16 round-to-nearest. So if the 3.886e-03 implementation gap is carried by
// the rounding mode, `conv_even` should collapse it, and the NPU-vs-golden
// number should fall toward the 1.923e-03 design limit.
//
// If it does NOT move, the bias is real but not the carrier here, and the
// finding is worth exactly one line in the note instead of a production change.
// Either answer is a result; this file exists to get one.
//
// It is a SEPARATE FILE rather than a flag on gelu_poly.cc on purpose:
// CLAUDE.md trap 7c ("never identify a build artifact by mtime", and the JIT
// does not hash everything you might change) means an edit-in-place A/B is
// exactly the shape that has silently served a stale build here five times.
// Two filenames cannot collide.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// Pull in gelu_poly_impl without its extern "C" entry points, so this
// translation unit exports only the symbols below and the two objects can
// coexist in one design if anyone ever wants them side by side.
#define NPUE_ELTWISE_IMPL_ONLY
#include "gelu_poly.cc"
#undef NPUE_ELTWISE_IMPL_ONLY

extern "C" {

// Identical to gelu_poly_bf16 except for the set_rounding call.
//
// set_rounding writes a core control register, so it is once per kernel call,
// not once per element -- the cost should be unmeasurable against 1024
// elements of Horner. That is part of what this measures.
void gelu_poly_rne_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gelu_poly_impl(input, output, 1024);
}

void gelu_poly_rne_bf16_4k(bfloat16 *restrict input, bfloat16 *restrict output) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gelu_poly_impl(input, output, 4096);
}

}  // extern "C"
