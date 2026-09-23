//===- softmax.cc -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- row-wise softmax over the sequence, fp32, max-subtracted.
// SPDX-License-Identifier: Apache-2.0
//
// WHY NOT THE SHIPPED aie_kernels/aie2p/softmax.cc
// ------------------------------------------------
//   1. It computes softmax over the whole tile as one row. Attention needs it
//      per row of the sequence -- [B, H, S, S] reduces along the last axis
//      only. tasks/0013 recorded this; it makes the shipped kernel
//      structurally unusable for us unless S happens to equal the tile.
//   2. Every intermediate is bf16 -- the max accumulator, the exponentials,
//      and the reciprocal of the sum. docs/04-model requires softmax in fp32,
//      and tasks/0016 established fp32 is genuinely available here.
//
// SAFETY AROUND THE MASK -- three clamps, each for a different reason
// -------------------------------------------------------------------
// HF's additive mask is (1 - mask) * finfo(f32).min = -3.4028e38. docs/04-model
// warns that using -inf here produces NaN. The value IS finite in fp32 -- but
// bf16's largest finite magnitude is 3.3895e38, so **the mask fill becomes -inf
// the moment the datapath is bf16**. Measured: 135,168 of 196,608 entries in
// the golden L0.scores_masked arrive as -inf. Same landmine, different door.
//
//   1. ON LOAD, IN bf16, BEFORE THE WIDENING:  xb = max(xb, -1e30).
//      Clamping after the widening is too late -- widening goes through
//      aie::mul(xb, 1.0) and an accumulator, and -inf does not survive that
//      trip intact. Clamping in fp32 afterwards still left 386 NaNs; clamping
//      the bf16 vector first leaves none. -1e30 is exactly representable in
//      bf16, which shares fp32's exponent range.
//   2. ON THE DIFFERENCE:  d = max(x - m, -100). exp(-100) = 3.7e-44, zero for
//      any purpose here, and it bounds everything downstream.
//   3. ON THE BASE-2 ARGUMENT:  arg = max(d * log2e, -120). exp2_poly builds
//      2^k by writing the IEEE exponent field, which needs k >= -126. Clamping
//      the natural-log difference is NOT enough: -100 * log2(e) = -144.27, and
//      127 - 144 is a negative exponent field, i.e. a NaN bit pattern. That
//      mistake produced exactly 256 NaNs on the first attempt.
//
// WHY exp2 IS OURS TOO
// --------------------
// The first version called aie::exp2<bfloat16> and measured 1.744e-02 against
// a bf16 floor of 3.225e-03 -- and 1.711e-02 against a CPU model of the same
// formula, so the divergence was the library call, not the algorithm. That is
// the same verdict aie::tanh got in tasks/0014, and tasks/0015's warning about
// exp2 turns out to have been right.
//
// So exp2 is built here, the same way GELU was: split the exponent from the
// fraction, do the fraction with a polynomial, and construct the power of two
// by writing the exponent field directly.
//
//     2^x  =  2^k * 2^f,   k = trunc(x),  f = x - k
//
// 2^k is exact -- an integer add of 127 and a 23-bit shift into a float's
// exponent field. Only 2^f needs approximating, and it is smooth on a bounded
// interval: degree 7 gives 1.75e-07, which is four orders below the bf16 grid.
//
// The polynomial is fitted on [-1, 1] rather than [-1, 0] deliberately, so the
// kernel does not depend on whether aie::to_fixed truncates or rounds. That
// costs one degree and removes an assumption we would otherwise have to verify.

#include "aie_kernel_utils.h"
#include "exp2_poly.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#define SM_COLS 64
#define SM_VECS (SM_COLS / 16)
#define SM_LOG2E 1.4426950408889634f
#define SM_FLOOR -100.0f
// Floor on the base-2 argument, not on the natural-log difference. 2^-120 is
// 6e-37 -- zero for any purpose here -- and it keeps k + 127 >= 7, well inside
// the normal-float exponent range that exp2_poly's bit construction needs.
#define SM_ARG_FLOOR -120.0f

template <bool kUsePoly>
void softmax_impl(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t rows) {
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vlog2e = aie::broadcast<float, 16>(SM_LOG2E);
  const aie::vector<float, 16> vfloor = aie::broadcast<float, 16>(SM_FLOOR);
  const aie::vector<float, 16> vargfloor =
      aie::broadcast<float, 16>(SM_ARG_FLOOR);
  // Clamp 1: applied in bf16, before anything converts.
  const aie::vector<bfloat16, 16> vinlo_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)-1.0e30f);

  for (int r = 0; r < rows; r++) {
    bfloat16 *in = input + r * SM_COLS;
    bfloat16 *out = output + r * SM_COLS;

    // -- pass 1: row max ---------------------------------------------------
    // Lane-wise max first, one reduce at the end: SM_VECS vector ops and a
    // single horizontal reduction, rather than a reduction per vector.
    aie::vector<float, 16> mx = aie::broadcast<float, 16>(-1.0e30f);
    for (int i = 0; i < SM_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      xb = aie::max(xb, vinlo_bf);
      mx = aie::max(mx, aie::mul(xb, vone_bf).to_vector<float>());
    }
    aie::vector<float, 16> m_v = aie::broadcast<float, 16>(aie::reduce_max(mx));

    // -- pass 2: exponentials and their sum --------------------------------
    // The exponentials are HELD IN REGISTERS, not staged through `out`.
    //
    // The first version stored them to `out` and read them back to normalize.
    // That produced 386 non-finite values in a structured pattern -- exactly
    // one 16-lane vector, in rows 1, 129, 257, ... at stride 128 -- i.e. a
    // read-after-write through the output buffer that did not always land
    // before the read. Not an arithmetic bug at all. At S=64 this is 4
    // registers; correctness beats the register-pressure argument that led to
    // the round trip.
    aie::vector<bfloat16, 16> ev[SM_VECS];
    aie::vector<float, 16> acc = aie::zeros<float, 16>();
    for (int i = 0; i < SM_VECS; i++) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(in + i * 16);
      xb = aie::max(xb, vinlo_bf);
      aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
      aie::vector<float, 16> d = aie::max(aie::sub(x, m_v), vfloor);
      aie::vector<float, 16> arg =
          aie::max(aie::mul(d, vlog2e).to_vector<float>(), vargfloor);
      // Two implementations, selected at compile time so both symbols coexist
      // in the same source and the cache marker (the symbol) stays unambiguous.
      //
      // The exp2_poly composition failed in tasks/0021 with output reading as
      // fp32 reinterpreted as bf16 pairs, and was never diagnosed. External
      // review proposed two candidate causes, and this template tests them
      // orthogonally (research/notes/0005):
      //   (1) the worker stack: 0xD00, the same size the 4-chain GELU
      //       overran -- and stack overrun corrupts rather than faults;
      //   (2) the narrowing: exp2_poly returns fp32, ev[i] is bf16, and the
      //       original composition's narrowing path was lost in the revert.
      //       Here it goes through the accumulator (mul by 1.0 then
      //       to_vector<bfloat16>), the only correct route -- never a cast.
      if constexpr (kUsePoly) {
        ev[i] = aie::mul(exp2_poly(arg), vone_f).to_vector<bfloat16>();
      } else {
          aie::vector<float, 16> exp_f = exp2_poly(arg);
          const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);
          ev[i] = aie::mul(exp_f, vone_f).to_vector<bfloat16>();
      }
      acc = aie::add(acc, aie::mul(ev[i], vone_bf).to_vector<float>());
    }
    float inv_sum = aie::inv(aie::reduce_add(acc));
    aie::vector<float, 16> inv_v = aie::broadcast<float, 16>(inv_sum);

    // -- pass 3: normalize and store, once ---------------------------------
    for (int i = 0; i < SM_VECS; i++) {
      aie::vector<float, 16> e = aie::mul(ev[i], vone_bf).to_vector<float>();
      aie::store_v(out + i * 16, aie::mul(e, inv_v).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

// 64 rows of 64 per call = 4,096 elements. L1 with double buffering on in and
// out: 2*(64*64*2)*2 = 32,768 B, comfortably inside 64 KB.
void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  softmax_impl<false>(input, output, 64);
}

// The exp2_poly composition, resurrected for the 0021 diagnosis.
void softmax_poly_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  softmax_impl<true>(input, output, 64);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY

// ---------------------------------------------------------------------------
// FOUR ROWS INTERLEAVED (tasks/0031). The one-row impl above is latency
// bound: at 8 columns it measures ~3,100 cycles per row against ~500 of
// issued work. A row is only SM_VECS = 4 vectors, and everything in it is one
// dependency chain -- max -> subtract -> exp2_poly (a 7-step serial Horner)
// -> sum -> reciprocal -> scale -- so the pipeline drains at every step.
// Same cure as the 4-chain GELU: four independent rows in flight.
//
// The exponentials go to a small LOCAL spill buffer instead of 16 live
// registers (4 rows x 4 vectors would exhaust the 12 x 512-bit register
// file). This is NOT the tasks/0021 output-buffer round trip: evbuf is the
// worker stack, not a fifo buffer, and the stores and loads are ordinary
// dependent L1 accesses within one kernel invocation.
//
// Numerics are bit-identical to softmax_impl<true>: each row performs the
// same operations in the same order, and the exponentials pass through the
// same bf16 narrowing they always did (ev[] was already bf16 above).
void softmax_il4_impl(bfloat16 *restrict input, bfloat16 *restrict output,
                      const int32_t rows) {
  event0();

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vlog2e = aie::broadcast<float, 16>(SM_LOG2E);
  const aie::vector<float, 16> vfloor = aie::broadcast<float, 16>(SM_FLOOR);
  const aie::vector<float, 16> vargfloor =
      aie::broadcast<float, 16>(SM_ARG_FLOOR);
  const aie::vector<bfloat16, 16> vinlo_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)-1.0e30f);

  alignas(32) bfloat16 evbuf[4 * SM_COLS];

  for (int r = 0; r < rows; r += 4) {
    bfloat16 *in0 = input + (r + 0) * SM_COLS;
    bfloat16 *in1 = input + (r + 1) * SM_COLS;
    bfloat16 *in2 = input + (r + 2) * SM_COLS;
    bfloat16 *in3 = input + (r + 3) * SM_COLS;
    bfloat16 *out0 = output + (r + 0) * SM_COLS;
    bfloat16 *out1 = output + (r + 1) * SM_COLS;
    bfloat16 *out2 = output + (r + 2) * SM_COLS;
    bfloat16 *out3 = output + (r + 3) * SM_COLS;

    // -- pass 1: row maxima, four independent chains ------------------------
    aie::vector<float, 16> mx0 = aie::broadcast<float, 16>(-1.0e30f);
    aie::vector<float, 16> mx1 = mx0, mx2 = mx0, mx3 = mx0;
    for (int i = 0; i < SM_VECS; i++) {
      mx0 = aie::max(mx0, aie::mul(aie::max(aie::load_v<16>(in0 + i * 16),
                                            vinlo_bf), vone_bf)
                              .to_vector<float>());
      mx1 = aie::max(mx1, aie::mul(aie::max(aie::load_v<16>(in1 + i * 16),
                                            vinlo_bf), vone_bf)
                              .to_vector<float>());
      mx2 = aie::max(mx2, aie::mul(aie::max(aie::load_v<16>(in2 + i * 16),
                                            vinlo_bf), vone_bf)
                              .to_vector<float>());
      mx3 = aie::max(mx3, aie::mul(aie::max(aie::load_v<16>(in3 + i * 16),
                                            vinlo_bf), vone_bf)
                              .to_vector<float>());
    }
    const aie::vector<float, 16> m0 =
        aie::broadcast<float, 16>(aie::reduce_max(mx0));
    const aie::vector<float, 16> m1 =
        aie::broadcast<float, 16>(aie::reduce_max(mx1));
    const aie::vector<float, 16> m2 =
        aie::broadcast<float, 16>(aie::reduce_max(mx2));
    const aie::vector<float, 16> m3 =
        aie::broadcast<float, 16>(aie::reduce_max(mx3));

    // -- pass 2: exponentials (to the spill buffer) and their sums ----------
    aie::vector<float, 16> ac0 = aie::zeros<float, 16>();
    aie::vector<float, 16> ac1 = aie::zeros<float, 16>();
    aie::vector<float, 16> ac2 = aie::zeros<float, 16>();
    aie::vector<float, 16> ac3 = aie::zeros<float, 16>();
    for (int i = 0; i < SM_VECS; i++) {
      aie::vector<float, 16> x0 =
          aie::mul(aie::max(aie::load_v<16>(in0 + i * 16), vinlo_bf), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> x1 =
          aie::mul(aie::max(aie::load_v<16>(in1 + i * 16), vinlo_bf), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> x2 =
          aie::mul(aie::max(aie::load_v<16>(in2 + i * 16), vinlo_bf), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> x3 =
          aie::mul(aie::max(aie::load_v<16>(in3 + i * 16), vinlo_bf), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> g0 = aie::max(
          aie::mul(aie::max(aie::sub(x0, m0), vfloor), vlog2e)
              .to_vector<float>(), vargfloor);
      aie::vector<float, 16> g1 = aie::max(
          aie::mul(aie::max(aie::sub(x1, m1), vfloor), vlog2e)
              .to_vector<float>(), vargfloor);
      aie::vector<float, 16> g2 = aie::max(
          aie::mul(aie::max(aie::sub(x2, m2), vfloor), vlog2e)
              .to_vector<float>(), vargfloor);
      aie::vector<float, 16> g3 = aie::max(
          aie::mul(aie::max(aie::sub(x3, m3), vfloor), vlog2e)
              .to_vector<float>(), vargfloor);
      // exp2, STEP-INTERLEAVED across the four rows. Calling exp2_poly()
      // four times in sequence gave ZERO speedup over the one-row kernel
      // (5,548 vs 5,356 us, measured): the inliner emits each call as a
      // complete 7-step serial Horner chain, and the scheduler does not
      // interleave across them. The GELU kernel dodged this by interleaving
      // each HORNER STEP across its four chains with a macro -- so exp2 gets
      // the same treatment here. Per-lane operations and order are identical
      // to exp2_poly(); only the instruction schedule differs.
      aie::vector<int32_t, 16> k0 = aie::to_fixed<int32_t>(g0);
      aie::vector<int32_t, 16> k1 = aie::to_fixed<int32_t>(g1);
      aie::vector<int32_t, 16> k2 = aie::to_fixed<int32_t>(g2);
      aie::vector<int32_t, 16> k3 = aie::to_fixed<int32_t>(g3);
      aie::vector<float, 16> f0 = aie::sub(g0, aie::to_float<float>(k0));
      aie::vector<float, 16> f1 = aie::sub(g1, aie::to_float<float>(k1));
      aie::vector<float, 16> f2 = aie::sub(g2, aie::to_float<float>(k2));
      aie::vector<float, 16> f3 = aie::sub(g3, aie::to_float<float>(k3));
      aie::vector<float, 16> p0 = aie::broadcast<float, 16>(EXP2_C0);
      aie::vector<float, 16> p1 = p0, p2 = p0, p3 = p0;
#define SM_EXP2_STEP(C)                                                              {                                                                                const aie::vector<float, 16> vc = aie::broadcast<float, 16>(C);                p0 = aie::add(aie::mul(p0, f0).to_vector<float>(), vc);                        p1 = aie::add(aie::mul(p1, f1).to_vector<float>(), vc);                        p2 = aie::add(aie::mul(p2, f2).to_vector<float>(), vc);                        p3 = aie::add(aie::mul(p3, f3).to_vector<float>(), vc);                      }
      SM_EXP2_STEP(EXP2_C1)
      SM_EXP2_STEP(EXP2_C2)
      SM_EXP2_STEP(EXP2_C3)
      SM_EXP2_STEP(EXP2_C4)
      SM_EXP2_STEP(EXP2_C5)
      SM_EXP2_STEP(EXP2_C6)
      SM_EXP2_STEP(EXP2_C7)
#undef SM_EXP2_STEP
      const aie::vector<int32_t, 16> v127 = aie::broadcast<int32_t, 16>(127);
      aie::vector<int32_t, 16> t0 = aie::upshift(aie::add(k0, v127), 23);
      aie::vector<int32_t, 16> t1 = aie::upshift(aie::add(k1, v127), 23);
      aie::vector<int32_t, 16> t2 = aie::upshift(aie::add(k2, v127), 23);
      aie::vector<int32_t, 16> t3 = aie::upshift(aie::add(k3, v127), 23);
      aie::vector<float, 16> q0 =
          aie::mul(p0, aie::vector_cast<float>(t0)).to_vector<float>();
      aie::vector<float, 16> q1 =
          aie::mul(p1, aie::vector_cast<float>(t1)).to_vector<float>();
      aie::vector<float, 16> q2 =
          aie::mul(p2, aie::vector_cast<float>(t2)).to_vector<float>();
      aie::vector<float, 16> q3 =
          aie::mul(p3, aie::vector_cast<float>(t3)).to_vector<float>();
      aie::vector<bfloat16, 16> e0 =
          aie::mul(q0, vone_f).to_vector<bfloat16>();
      aie::vector<bfloat16, 16> e1 =
          aie::mul(q1, vone_f).to_vector<bfloat16>();
      aie::vector<bfloat16, 16> e2 =
          aie::mul(q2, vone_f).to_vector<bfloat16>();
      aie::vector<bfloat16, 16> e3 =
          aie::mul(q3, vone_f).to_vector<bfloat16>();
      aie::store_v(evbuf + 0 * SM_COLS + i * 16, e0);
      aie::store_v(evbuf + 1 * SM_COLS + i * 16, e1);
      aie::store_v(evbuf + 2 * SM_COLS + i * 16, e2);
      aie::store_v(evbuf + 3 * SM_COLS + i * 16, e3);
      ac0 = aie::add(ac0, aie::mul(e0, vone_bf).to_vector<float>());
      ac1 = aie::add(ac1, aie::mul(e1, vone_bf).to_vector<float>());
      ac2 = aie::add(ac2, aie::mul(e2, vone_bf).to_vector<float>());
      ac3 = aie::add(ac3, aie::mul(e3, vone_bf).to_vector<float>());
    }
    const aie::vector<float, 16> iv0 =
        aie::broadcast<float, 16>(aie::inv(aie::reduce_add(ac0)));
    const aie::vector<float, 16> iv1 =
        aie::broadcast<float, 16>(aie::inv(aie::reduce_add(ac1)));
    const aie::vector<float, 16> iv2 =
        aie::broadcast<float, 16>(aie::inv(aie::reduce_add(ac2)));
    const aie::vector<float, 16> iv3 =
        aie::broadcast<float, 16>(aie::inv(aie::reduce_add(ac3)));

    // -- pass 3: normalize and store ----------------------------------------
    for (int i = 0; i < SM_VECS; i++) {
      aie::vector<float, 16> e0 =
          aie::mul(aie::load_v<16>(evbuf + 0 * SM_COLS + i * 16), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> e1 =
          aie::mul(aie::load_v<16>(evbuf + 1 * SM_COLS + i * 16), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> e2 =
          aie::mul(aie::load_v<16>(evbuf + 2 * SM_COLS + i * 16), vone_bf)
              .to_vector<float>();
      aie::vector<float, 16> e3 =
          aie::mul(aie::load_v<16>(evbuf + 3 * SM_COLS + i * 16), vone_bf)
              .to_vector<float>();
      aie::store_v(out0 + i * 16, aie::mul(e0, iv0).to_vector<bfloat16>());
      aie::store_v(out1 + i * 16, aie::mul(e1, iv1).to_vector<bfloat16>());
      aie::store_v(out2 + i * 16, aie::mul(e2, iv2).to_vector<bfloat16>());
      aie::store_v(out3 + i * 16, aie::mul(e3, iv3).to_vector<bfloat16>());
    }
  }

  event1();
  return;
}

#ifndef NPUE_ELTWISE_IMPL_ONLY
extern "C" {

// Same 64 rows per call as softmax_poly_bf16 -- fifo geometry unchanged.
// Needs the 0x2000 worker stack: four exp2_poly chains plus the spill buffer
// overrun 0xD00, which corrupts silently rather than faulting.
void softmax_poly_il4_bf16(bfloat16 *restrict input,
                           bfloat16 *restrict output) {
  softmax_il4_impl(input, output, 64);
}

} // extern "C"
#endif // NPUE_ELTWISE_IMPL_ONLY
