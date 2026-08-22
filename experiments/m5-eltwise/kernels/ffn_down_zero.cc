//===- ffn_down_zero.cc -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- tasks/0061, T28: split out of ffn_down_relay_g2.cc.
// SPDX-License-Identifier: Apache-2.0
//
// TRAP FOUND BUILDING THE HIERARCHICAL MERGE PROBE: pulling more than one
// ExternalFunction entry point from the SAME multi-symbol .cc file within
// ONE design produces `ld.lld: error: duplicate symbol` -- every requested
// entry point compiles the WHOLE source file into its own object (there is
// no per-symbol extraction), so linking N objects built from the same
// multi-function file links N copies of every function in it. No prior
// build in this codebase does this (gelu_poly.cc and narrow_f32_bf16.cc
// each hold several entry points, but every existing design picks exactly
// ONE per build); this probe needs THREE from what was one file
// (ffn_down_relay_g2.cc), so each now lives in its own file. See
// hierarchical_merge_ffn_probe.py and TASK.md.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

extern "C" {

// Zero a (64, 16) fp32 accumulator (1024 elements) before the first of the
// two hop-matmul calls in ffn_down_hop_matmul_g2.cc.
void zero_f32_1024(float *restrict acc) {
  auto ot = aie::begin_restrict_vector<16>(acc);
  const aie::vector<float, 16> z = aie::broadcast<float, 16>(0.0f);
  for (int i = 0; i < 1024; i += 16) {
    *ot++ = z;
  }
}

} // extern "C"
