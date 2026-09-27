//===- softmax_w.cc -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- row-wise softmax for a row as wide as an attention's score
// vector, one row per call. SPDX-License-Identifier: Apache-2.0
//
// WHY NOT softmax.cc
// ------------------
// softmax.cc holds a row's exponentials in REGISTERS (SM_VECS of them, 4 at
// 64 columns) and the four-row variant spills 4 * SM_COLS bf16 to the worker's
// stack. A Whisper encoder row is n_kv scores -- 1500, padded to 1536 -- and both
// shapes stop working: 96 vectors is more than the register file holds, and
// 4 * 1536 * 2 B is 12 KB of stack against a worker's 8 KB. The arithmetic is
// not the problem and the three clamps in softmax.cc are not the problem; only
// the storage of the exponentials is.
//
// SO: one row per call, and the exponentials go to a LOCAL buffer of one row
// (3 KB at 1536 columns) instead of to registers or to the output fifo. Passing
// through `out` is what softmax.cc tried first and had to abandon (tasks/0021:
// 386 non-finite values in a stride-128 pattern, a read-after-write through the
// output buffer that did not land before the read). The worker's own stack is an
// ordinary dependent L1 access inside one invocation, which is a different thing
// and is how the four-row variant has always done it.
//
// The clamps, the exp2_poly and the bf16 narrowing are softmax.cc's, unchanged:
// a score row that reaches this kernel is the same number either way, and a
// different exp2 would be a different model.

#include "aie_kernel_utils.h"
#include "exp2_poly.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SW_COLS
#define SW_COLS 1536
#endif
#define SW_VECS (SW_COLS / 16)
#define SW_LOG2E 1.4426950408889634f
#define SW_FLOOR -100.0f
#define SW_ARG_FLOOR -120.0f

void softmax_w_impl(bfloat16 *restrict input, bfloat16 *restrict output,
                    const int32_t rows) {
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vlog2e = aie::broadcast<float, 16>(SW_LOG2E);
  const aie::vector<float, 16> vfloor = aie::broadcast<float, 16>(SW_FLOOR);
  const aie::vector<float, 16> vargfloor =
      aie::broadcast<float, 16>(SW_ARG_FLOOR);
  // Clamp 1, in bf16 and before anything widens: -3.4e38 is the value HF's own
  // mask uses and bf16 cannot hold it, so an unmasked -inf arrives as a NaN.
  const aie::vector<bfloat16, 16> vinlo_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)-1.0e30f);

  alignas(32) bfloat16 evbuf[SW_COLS];

  for (int r = 0; r < rows; r++) {
    bfloat16 *in = input + r * SW_COLS;
    bfloat16 *out = output + r * SW_COLS;

    aie::vector<float, 16> mx = aie::broadcast<float, 16>(-1.0e30f);
    for (int i = 0; i < SW_VECS; i++) {
      const aie::vector<bfloat16, 16> xb =
          aie::max(aie::load_v<16>(in + i * 16), vinlo_bf);
      mx = aie::max(mx, aie::mul(xb, vone_bf).to_vector<float>());
    }
    const aie::vector<float, 16> m_v =
        aie::broadcast<float, 16>(aie::reduce_max(mx));

    aie::vector<float, 16> acc = aie::zeros<float, 16>();
    for (int i = 0; i < SW_VECS; i++) {
      const aie::vector<bfloat16, 16> xb =
          aie::max(aie::load_v<16>(in + i * 16), vinlo_bf);
      const aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      const aie::vector<float, 16> d = aie::max(aie::sub(x, m_v), vfloor);
      const aie::vector<float, 16> arg =
          aie::max(aie::mul(d, vlog2e).to_vector<float>(), vargfloor);
      const aie::vector<bfloat16, 16> e =
          aie::mul(exp2_poly(arg), vone_f).to_vector<bfloat16>();
      aie::store_v(evbuf + i * 16, e);
      acc = aie::add(acc, aie::mul(e, vone_bf).to_vector<float>());
    }
    const aie::vector<float, 16> inv_v =
        aie::broadcast<float, 16>(aie::inv(aie::reduce_add(acc)));

    for (int i = 0; i < SW_VECS; i++) {
      const aie::vector<float, 16> e =
          aie::mul(aie::load_v<16>(evbuf + i * 16), vone_bf).to_vector<float>();
      aie::store_v(out + i * 16, aie::mul(e, inv_v).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}

extern "C" {

// One row of SW_COLS per call. The row width is compiled in (-DSW_COLS) because
// the spill buffer is a fixed-size local array: a runtime width would need a
// heap, and a heap in a worker is how a design ends up non-deterministic.
void softmax_w_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  softmax_w_impl(input, output, 1);
}

// Two rows per call, for the designs whose object fifo is better off moving two
// rows at a time. Identical arithmetic per row.
void softmax_w2_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  softmax_w_impl(input, output, 2);
}

}  // extern "C"
