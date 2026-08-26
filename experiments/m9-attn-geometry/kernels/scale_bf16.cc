//===- scale_bf16.cc ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- T38 (tasks/0114): the smallest possible compute over a
// mem-tile-PADDED stream.
// SPDX-License-Identifier: Apache-2.0
//
// This kernel exists only to answer one question: can a compute tile consume
// a buffer that a mem tile padded on the way in? The arithmetic is
// deliberately trivial -- y = 2x -- so that any mismatch is about the DATA
// MOVEMENT and cannot be about the kernel.
//
// Doubling is chosen over copying on purpose: a copy cannot distinguish "the
// kernel ran" from "the buffer happened to already hold the right bytes", and
// 2 x 0 is still 0, so the padded half stays checkable as zero.
//
// SIZE IS COMPILE-TIME AND MUST MATCH THE PYTHON SIDE. CLAUDE.md trap 8: a
// Kernel's declared arg_types shape only the MLIR call-site, never the linked
// object, which is compiled straight from this signature. 64 rows x 16 padded
// columns = 1024 bf16 elements.
//
// research/notes/0001 forbids scalar float in a kernel body (measured 1,617x
// slower via __mulsf3). Everything below is a native vector op.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#define NPUE_PAD_PROBE_ELEMS 1024

extern "C" {

void scale_bf16_1024(bfloat16 *restrict in_v, bfloat16 *restrict out_v) {
  event0();

  auto it = aie::begin_restrict_vector<16>((bfloat16 *)in_v);
  auto ot = aie::begin_restrict_vector<16>((bfloat16 *)out_v);
  const aie::vector<float, 16> two = aie::broadcast<float, 16>(2.0f);

  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < NPUE_PAD_PROBE_ELEMS; i += 16) {
    // Widen bf16 -> fp32 with accum::from_vector, the idiom tasks/0091 (T7)
    // established: it compiles to `vlda.conv.fp32.bf16`, a load-with-
    // conversion. An implicit vector<bfloat16> -> vector<float> assignment
    // does not compile at all (no such conversion), and the multiply-by-1.0f
    // form is emulated on aie2p.
    aie::accum<accfloat, 16> ax;
    ax.from_vector(*it++);
    aie::accum<accfloat, 16> a = aie::mul(ax.to_vector<float>(), two);
    *ot++ = a.to_vector<bfloat16>();
  }

  event1();
}

}  // extern "C"
