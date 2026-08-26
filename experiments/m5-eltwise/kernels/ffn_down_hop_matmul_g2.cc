//===- ffn_down_hop_matmul_g2.cc ------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- tasks/0061, T28: the relay/ffn_down-consuming core's own
// matmul kernel, for the hierarchical 2-hop merge probe. Split out of
// ffn_down_relay_g2.cc -- see ffn_down_zero.cc's header for why.
// SPDX-License-Identifier: Apache-2.0
//
// WHY A HAND-WRITTEN KERNEL INSTEAD OF kernels.mm()
// --------------------------------------------------
// gemm_pretiled.py's `matmul_kernel` (kernels.mm()) expects its A operand
// pre-arranged in the MMAC intrinsic's (r,s) sub-tile order -- the exact
// transform `A_l2l1_fifos`' dims_to_stream applies on every OTHER GEMM in
// this codebase. Getting a JOIN's gathered, GELU'd, bf16-narrowed output
// into that same sub-tile order would need EITHER a second dims_to_stream
// layered on top of the join-undo transform every prior join in this
// codebase already carries (untested composition, two transforms on one
// hop), or a per-producer-subfifo reorder threaded through a chain of THREE
// kernels (matmul -> GELU -> narrow) that no probe in this codebase has
// combined before. Both are real, unexplored IRON questions in their own
// right and were judged out of scope for tonight's session (see TASK.md).
//
// This kernel sidesteps that question entirely by NOT using the MMAC
// intrinsic for the second-stage matmul. It operates on the join's OWN
// output layout directly -- block-concatenated per producer tile, each
// block internally PLAIN ROW-MAJOR (established by
// pipeline_gemm_gelu_probe.py's Y_mem: an elementwise GELU kernel, reading
// a GEMM's own accumulator tile with a straight linear loop and no
// permutation at all, reproduces the exact-erf reference -- so a GEMM
// core's own accumulator tile, as seen by C++ code on the SAME core, is
// already plain row-major; nothing needs undoing at that boundary). The one
// join-mechanism artifact that DOES need undoing (0054 Problem #5: "a
// mem-tile JOIN of multiple producer tiles does not simply concatenate them
// byte-for-byte") is undone the usual way, on the join's own base
// ObjectFifo dims_to_stream, exactly as gemm_pretiled.py's C_l2l3_fifos and
// pipeline_gemm_gelu_probe.py's Y_mem already do it -- see
// hierarchical_merge_ffn_probe.py.
//
// This is slower than kernels.mm() (CLAUDE.md note 0001's 1,617x scalar-math
// trap is about SCALAR FLOAT ARITHMETIC becoming library calls; the
// broadcast-then-vector-FMA structure here keeps every multiply/add a real
// vector op, just not MMAC-accelerated). No performance claim is made for
// this kernel or this probe -- CLAUDE.md rule 1, and it is far too small a
// design to trace meaningfully anyway (trap 7).

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

extern "C" {

// hop:  (G=2, TM=64, TN=48) bf16, block-major -- producer column b's whole
//       (64,48) tile contiguous at offset b*64*48, the join's own output
//       layout after its dims_to_stream join-undo transform (see module
//       docstring). w: (G=2, TN=48, N=16) bf16, same block order -- a
//       compile-time-constant weight slice, one (48,16) block per producer
//       column this hop gathers. acc: (TM=64, N=16) fp32, ACCUMULATED (not
//       overwritten) -- the caller zeroes it once, then calls this kernel
//       once per hop (twice total in this probe), so the K-reduction spans
//       BOTH hops the same way gemm_pretiled.py's K-loop spans k-blocks.
//
// N=16 is chosen so one output row is exactly one 16-lane fp32 vector --
// this probe's ffn_down output width is 16, not production's 48, purely to
// keep this kernel's structure (one vector register per output row) simple;
// see TASK.md for the L1-budget arithmetic behind every scale choice here.
void ffn_down_hop_matmul_g2_64x48x16(bfloat16 *restrict hop,
                                     bfloat16 *restrict w,
                                     float *restrict acc) {
  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  for (int b = 0; b < 2; b++) {
    bfloat16 *hb = hop + b * 64 * 48;
    bfloat16 *wb = w + b * 48 * 16;
    for (int m = 0; m < 64; m++) {
      float *acc_row_ptr = acc + m * 16;
      aie::vector<float, 16> acc_row =
          *aie::begin_restrict_vector<16>(acc_row_ptr);
      for (int k = 0; k < 48; k++) {
        float a_val = (float)hb[m * 48 + k];
        aie::vector<float, 16> a_bcast = aie::broadcast<float, 16>(a_val);
        aie::vector<bfloat16, 16> w_row_bf =
            *aie::begin_restrict_vector<16>(wb + k * 16);
        aie::vector<float, 16> w_row =
            aie::mul(w_row_bf, vone_bf).to_vector<float>();
        acc_row =
            aie::add(acc_row, aie::mul(a_bcast, w_row).to_vector<float>());
      }
      *aie::begin_restrict_vector<16>(acc_row_ptr) = acc_row;
    }
  }
}

// tasks/0092: the GENUINE bf16-accumulator twin of the kernel above, added to
// FIX the NPUE_BF16_COPY discriminator tasks/0087 built. That discriminator
// declared `acc_buf` bf16 (1024 elements, 2048 B) on the Python/MLIR side but
// kept calling the SAME `ffn_down_hop_matmul_g2_64x48x16` symbol above, whose
// C++ signature is `float *restrict acc` -- IRON's `arg_types` only shapes the
// MLIR func.call declaration (`external_func(..., inputs=self._arg_types)` in
// kernel.py); it does not touch the linked object, which is compiled straight
// from this source regardless of what the caller declares. `llvm-objdump` on
// the cached `ffn_down_hop_matmul_g2_64x48x16.o` confirms the callee body: a
// `vst ..., [p5], #0x40` / `vst ..., [p5, #0x0]` pair per row, 64 rows, i.e.
// 64 x 64 B = 4096 B written through that pointer -- double the 2048 B the
// bf16 Buffer actually allocates (verified against `input_with_addresses.mlir`
// for the BF16_COPY/N_DOWN=16 build: `ffn_down_acc` sits at address 38144 as
// `memref<1024xbf16>`, i.e. exactly 2048 B, with nothing else on tile_0_4
// until... there IS nothing else -- it is the last buffer allocated, so the
// extra 2048 B lands in unclaimed DM, not inside a named neighbour). So the
// discriminator's OWN new bug is real and objdump-confirmed, but whether it,
// rather than the fifo, is what actually hangs the hardware is exactly what
// this genuinely-bf16 twin re-tests: same acquire/release shape, same
// resident-Buffer accumulator dtype as the MLIR declares, no overflow.
// Accumulating in bf16 (not fp32) is numerically wrong per CLAUDE.md trap 2
// and this is NEVER used by a design that computes anything real -- same
// caveat as `zero_bf16_1024` above -- it exists only to isolate the hang.
void ffn_down_hop_matmul_g2_64x48x16_bf16acc(bfloat16 *restrict hop,
                                             bfloat16 *restrict w,
                                             bfloat16 *restrict acc) {
  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  for (int b = 0; b < 2; b++) {
    bfloat16 *hb = hop + b * 64 * 48;
    bfloat16 *wb = w + b * 48 * 16;
    for (int m = 0; m < 64; m++) {
      bfloat16 *acc_row_ptr = acc + m * 16;
      aie::vector<bfloat16, 16> acc_row =
          *aie::begin_restrict_vector<16>(acc_row_ptr);
      for (int k = 0; k < 48; k++) {
        aie::vector<bfloat16, 16> a_bcast =
            aie::broadcast<bfloat16, 16>(hb[m * 48 + k]);
        aie::vector<bfloat16, 16> w_row =
            *aie::begin_restrict_vector<16>(wb + k * 16);
        acc_row = aie::add(acc_row, aie::mul(a_bcast, w_row).to_vector<bfloat16>());
      }
      *aie::begin_restrict_vector<16>(acc_row_ptr) = acc_row;
    }
  }
}

// N = 48, WHICH IS PRODUCTION'S OWN ffn_down TILE WIDTH (tasks/0087).
//
// The N=16 entry point above exists because one output row is then exactly one
// 16-lane fp32 vector, which keeps the register structure trivial. At N=48 a
// row is THREE vectors, so they are held as three explicit accumulators across
// the k loop rather than one -- the same shape gemm_pretiled's own core uses,
// and the reason the L1 budget in tasks/0083 counted `TM * N * 4` for the
// accumulator rather than a single register.
//
// L1 for the relay at this width, from tasks/0083's streaming-relay
// arithmetic: two gathered hops 24,576 B + B(down) 18,432 + acc 12,288 +
// out 6,144 = 61,440 of 64,512. It fits, with 3,072 B spare -- which is why
// N=64 does not (73,728) and why this is the widest step available.
void ffn_down_hop_matmul_g2_64x48x48(bfloat16 *restrict hop,
                                     bfloat16 *restrict w,
                                     float *restrict acc) {
  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  for (int b = 0; b < 2; b++) {
    bfloat16 *hb = hop + b * 64 * 48;
    bfloat16 *wb = w + b * 48 * 48;
    for (int m = 0; m < 64; m++) {
      float *acc_row_ptr = acc + m * 48;
      aie::vector<float, 16> a0 = *aie::begin_restrict_vector<16>(acc_row_ptr);
      aie::vector<float, 16> a1 =
          *aie::begin_restrict_vector<16>(acc_row_ptr + 16);
      aie::vector<float, 16> a2 =
          *aie::begin_restrict_vector<16>(acc_row_ptr + 32);
      for (int k = 0; k < 48; k++) {
        float a_val = (float)hb[m * 48 + k];
        aie::vector<float, 16> a_bcast = aie::broadcast<float, 16>(a_val);
        bfloat16 *wk = wb + k * 48;
        aie::vector<bfloat16, 16> w0 = *aie::begin_restrict_vector<16>(wk);
        aie::vector<bfloat16, 16> w1 = *aie::begin_restrict_vector<16>(wk + 16);
        aie::vector<bfloat16, 16> w2 = *aie::begin_restrict_vector<16>(wk + 32);
        a0 = aie::add(a0, aie::mul(a_bcast,
              aie::mul(w0, vone_bf).to_vector<float>()).to_vector<float>());
        a1 = aie::add(a1, aie::mul(a_bcast,
              aie::mul(w1, vone_bf).to_vector<float>()).to_vector<float>());
        a2 = aie::add(a2, aie::mul(a_bcast,
              aie::mul(w2, vone_bf).to_vector<float>()).to_vector<float>());
      }
      *aie::begin_restrict_vector<16>(acc_row_ptr) = a0;
      *aie::begin_restrict_vector<16>(acc_row_ptr + 16) = a1;
      *aie::begin_restrict_vector<16>(acc_row_ptr + 32) = a2;
    }
  }
}

} // extern "C"
