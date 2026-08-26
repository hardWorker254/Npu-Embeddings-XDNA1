//===- probe_copy.cc ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- identity copies at PROBE sizes, for topology experiments.
// SPDX-License-Identifier: Apache-2.0
//
// WHY A SEPARATE FILE
// -------------------
// `gelu_poly.cc` carries two identity copies (3072 and 6144 elements) added
// by tasks/0054 and 0057 for their own probes. Those sizes are production
// tile sizes, which is exactly wrong for a PORT-BUDGET experiment: a 4-way
// join of 6144-element tiles fails on L1 (two 49,152 B buffers) before the
// compiler ever gets to counting DMA channels, so the experiment answers a
// question nobody asked. tasks/0083 needed tiles small enough that L1 is
// never the binding constraint, so the only thing that can fail is the thing
// under test.
//
// These are diagnostic-only and no shipped design calls them.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <unsigned N>
static inline void copy_impl(const float *restrict in, float *restrict out) {
  auto it = aie::begin_restrict_vector<16>((float *)in);
  auto ot = aie::begin_restrict_vector<16>(out);
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned i = 0; i < N; i += 16) *ot++ = *it++;
}

extern "C" {

void probe_copy_512_f32(const float *restrict in, float *restrict out) {
  copy_impl<512>(in, out);
}
void probe_copy_1024_f32(const float *restrict in, float *restrict out) {
  copy_impl<1024>(in, out);
}
void probe_copy_1536_f32(const float *restrict in, float *restrict out) {
  copy_impl<1536>(in, out);
}
void probe_copy_2048_f32(const float *restrict in, float *restrict out) {
  copy_impl<2048>(in, out);
}
void probe_copy_2560_f32(const float *restrict in, float *restrict out) {
  copy_impl<2560>(in, out);
}
void probe_copy_3072_f32(const float *restrict in, float *restrict out) {
  copy_impl<3072>(in, out);
}
void probe_copy_3584_f32(const float *restrict in, float *restrict out) {
  copy_impl<3584>(in, out);
}
void probe_copy_4096_f32(const float *restrict in, float *restrict out) {
  copy_impl<4096>(in, out);
}

// bf16 IN, bf16 OUT -- the dtype-vs-kernel discriminator for tasks/0087's
// hang. The relay's narrowed output hangs the hardware; this splits "a bf16
// output ObjectFifo hangs" from "narrow_f32_bf16 hangs", which the single
// failing configuration could not.
void probe_copy_1024_bf16(const bfloat16 *restrict in, bfloat16 *restrict out) {
  auto it = aie::begin_restrict_vector<16>((bfloat16 *)in);
  auto ot = aie::begin_restrict_vector<16>(out);
  for (unsigned i = 0; i < 1024; i += 16) *ot++ = *it++;
}

// tasks/0092: WRITES A KNOWN CONSTANT PATTERN, IGNORES `in` ENTIRELY --
// element i = (bfloat16)i, i in 0..1023. This is the cheapest rung on the
// bisection ladder tasks/0092's brief asked for: does a bf16 core->mem-tile
// ->shim ObjectFifo `forward()` drain arrive correctly on the host with NO
// upstream compute at all (no accumulator read, no weight buffer touched)?
// If this hangs too, the fifo/forward path itself is the bug, independent of
// everything the relay's own matmul does upstream of it. If it PASSES, the
// hang tracks back into the compute chain instead.
void ffn_down_const_pattern_1024_bf16(const bfloat16 *restrict in,
                                      bfloat16 *restrict out) {
  (void)in;
  for (unsigned i = 0; i < 1024; i++) out[i] = (bfloat16)(float)i;
}

}  // extern "C"
