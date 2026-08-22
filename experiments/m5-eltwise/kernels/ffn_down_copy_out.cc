//===- ffn_down_copy_out.cc ----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- tasks/0061, T28: split out of ffn_down_relay_g2.cc.
// SPDX-License-Identifier: Apache-2.0. See ffn_down_zero.cc's header for why
// this is its own file rather than a third symbol in one.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

extern "C" {

// Drain the (64,16) fp32 accumulator into the output ObjectFifo's freshly
// acquired object -- a plain copy, the same role narrow_f32_bf16.cc's
// separate-buffer narrow plays, minus the narrow (this probe's ffn_down
// output stays fp32, matching CLAUDE.md's decided "bf16 in / fp32 out"
// datapath contract).
void copy_f32_1024(float *restrict in, float *restrict out) {
  auto it = aie::begin_restrict_vector<16>(in);
  auto ot = aie::begin_restrict_vector<16>(out);
  for (int i = 0; i < 1024; i += 16) {
    *ot++ = *it++;
  }
}

} // extern "C"
