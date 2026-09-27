//===- attention_npu.hpp -----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's attention as two GEMMs on the design set's own
// attn_qk and attn_av streams. SPDX-License-Identifier: Apache-2.0
//
// THE SHAPE OF IT
// ---------------
// Attention is three operations, and only two of them are matrices:
//
//   QK^T      [rows, head_dim] @ [head_dim, n_kv]   a GEMM
//   softmax   [rows, n_kv]                          a reduction, and the
//                                                   elementwise design's job
//   softmax.V [rows, n_kv] @ [n_kv, head_dim]       a GEMM
//
// So the two GEMMs are the array's, one dispatch per head per query chunk, and
// the softmax is whichever path --npu-extra-ops softm named. The dispatch count
// is the price: heads x chunks x 2 x layers, which for large-v3 is 3840 per
// window against 150 us of fixed cost each. That is why this exists anyway: the
// arithmetic is 184 GFLOP per window on the host and the point of the exercise
// is that the operation CAN run here, not that it runs faster.
//
// WHAT IS DIFFERENT FROM THE HOST PASS, AND WHY IT IS NOT A MASK
// -----------------------------------------------------------
// The B panel of a score row is built from the K|V block of the fused qkv
// operand with ZEROS past n_kv, because the panel has to tile evenly and n_kv is
// padded up to a multiple of the tile. A zero key column gives a score of 0 --
// a perfectly ordinary score, not a forbidden one -- so the padded columns are
// written to -1e30 before the softmax, which is the same sentinel the
// suppression policy uses and which the kernel's own load-time clamp turns into
// exp(-100) = 0. Without that line the padded keys take a real share of the
// attention mass and the encoder output is wrong in a way that looks like a
// model problem.
//
// The two B panels are rebuilt for EVERY HEAD, every layer, every window: they
// are activations, not weights, so there is nothing to stage once. They go into
// ONE slot per stream and are overwritten, which is why the fill and the
// sync-to-device happen back to back here and not once per session.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/pool.hpp"
#include "whisper/eltwise.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

class NpuAttention {
public:
  // `n_kv` is the design set's padded key count and `ctx_cols` the attn_av
  // stream's N, which is head_dim rounded up to the design's N granularity (a
  // design's N must be a multiple of tile_n * AIE columns, and 64 is not a
  // multiple of 128). Only the first head_dim columns of every C row are the
  // head's context; the rest is the zero half of the panel.
  NpuAttention(npu::Design &design, app::Pool &pool, int64_t n_kv,
               int64_t head_dim, int64_t ctx_cols)
      : d_(design), pool_(pool), n_kv_(n_kv), head_dim_(head_dim),
        ctx_cols_(ctx_cols), qk_(design, pool), av_(design, pool) {}

  // The two streams' instruction-stream slots and the row count one dispatch
  // computes, read off the loaded design set by the caller. A set without those
  // streams is not a set this class can run on, and the caller refuses by name.
  void set_streams(size_t qk_slot, size_t av_slot, int64_t rows);

  // The A/C slots of both GEMMs and the single B slot each of them rewrites per
  // head. Must be called once, before the first run().
  void alloc_buffers();

  // Where the softmax runs. Null is the host pass, which is a real path and not
  // a fallback: --npu-extra-ops softm has to name it.
  void set_softmax(NpuEltwise *s) { softm_ = s; }
  bool softmax_on_array() const { return softm_ != nullptr; }

  // One attention, for every head:
  //   `q` is [n_q, q_stride] with a head's slice at offset h*head_dim,
  //   `kv` is [n_kv, kv_stride] as a K|V block (K first, then V),
  //   `out` is [n_q, d_model] and receives each head's context in place.
  void run(const float *q, int64_t q_stride, const float *kv, int64_t kv_stride,
           int64_t n_q, int64_t n_kv, int64_t d_model, float scale, float *out);

  int64_t n_dispatch = 0;
  // What the status line says this costs, per window: the dispatches the
  // geometry implies, not a counter that reads zero before the first request.
  std::string note(int64_t n_q, int64_t heads, int64_t layers) const;

private:
  // The [head_dim, n_kv] K panel, or the [n_kv, head_dim] V panel, tiled for the
  // MMAC. `which` picks which: the two are the same numbers transposed, and one
  // buffer holds whichever is being built.
  void build_panel(const float *kv, int64_t kv_stride, int64_t n_kv,
                   int64_t d_model, int64_t head, bool v_side);
  // The two panels' byte counts, checked against the design's B buffer in
  // alloc_buffers() -- a design exported without these streams has a B buffer
  // sized for the set's own weights, and writing past it is heap corruption.
  void host_softmax(float *scores, int64_t n_rows, int64_t n_kv, int64_t n_kv_pad);

  npu::Design &d_;
  app::Pool &pool_;
  int64_t n_kv_;      // padded: the design's N (and attn_av's K)
  int64_t head_dim_ = 0;
  int64_t ctx_cols_ = 0;   // attn_av's N: head_dim rounded up to the design's
  size_t instr_qk_ = 0, instr_av_ = 0;
  int64_t rows_ = 0;
  NpuGemm qk_, av_;
  size_t bslot_qk_ = 0, bslot_av_ = 0;
  NpuEltwise *softm_ = nullptr;
  // Per-run scratch, sized once: the gathered Q block, one chunk of scores, one
  // chunk of context, and the two panels' staging matrices.
  std::vector<float> aq_, scores_, ctx_, zero_k_, zero_v_;
  std::vector<float> mat_;
  std::vector<uint16_t> panel_;
};

}  // namespace npue::whisper
