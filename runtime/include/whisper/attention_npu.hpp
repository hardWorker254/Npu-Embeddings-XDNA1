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
// the softmax is whichever path --npu-ops softm named. The dispatch count
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
        ctx_cols_(ctx_cols), qk_(design, pool, "attn"), av_(design, pool, "attn") {}

  // The two streams' instruction-stream slots, the row count one dispatch
  // computes, and attn_qk's K -- read off the loaded design set by the caller. A
  // set without those streams is not a set this class can run on, and the caller
  // refuses by name.
  //
  // `k_qk` is the design's own attn_qk K, which is head_dim PADDED UP to the
  // tile_k. It has to be passed rather than derived: one xclbin carries every
  // stream of the set and `info()` reports the largest, so the number that
  // matters is only in the stream table. When it equals head_dim -- Whisper,
  // whose head_dim is 64 against a tile_k of 64 -- nothing is padded and this
  // class behaves exactly as it did before the parameter existed. Zero means
  // "not padded", so an existing caller cannot be wrong by omission.
  void set_streams(size_t qk_slot, size_t av_slot, int64_t rows,
                   int64_t k_qk = 0);

  // The K|V block's shape, for the architectures where it is not the output
  // row's shape.
  //
  // `kv_width` is the distance from a key row to its value, which for BERT and
  // Whisper IS d_model because K and V are each one d_model wide. Gemma is
  // MULTI-QUERY -- 3 query heads of 256 over ONE key/value head of 256 -- so its
  // kv_width is 256 while its output row is 768. Zero means d_model.
  //
  // `kv_heads` is how many of those key/value heads exist. Query head h reads
  // key/value head `h / (heads / kv_heads)`, which is the identity when the two
  // counts are equal and 0 for every head when there is only one. Zero means
  // "one per query head". This is the parameter that makes an MQA model
  // expressible at all: without it the loop would walk off the end of a
  // one-head K|V block on the second head, and the scores it produced would look
  // like a model problem rather than a geometry one.
  void set_kv_geometry(int64_t kv_width, int64_t kv_heads);

  // The runtime's dispatch mutex, the SAME one BertEncoder and GemmaNpuEncoder
  // take around their own bind/dispatch window. NpuAttention dispatches through
  // two NpuGemm objects, and a Design's `active` binding is shared mutable
  // state -- npu_ops.cpp's comment on the subject is blunt about it: "two
  // threads binding concurrently read a torn binding and each other's buffers".
  // Leaving these null makes every attention dispatch unlocked, which is what a
  // single-lane caller (Whisper, ViT) already does, so an unwired caller
  // behaves exactly as it did before this method existed.
  //
  // This is the second half of a fix, not the whole one. The failure it was
  // chased after was `Invalid BO offset and size for sync'ing: 0, 147456` on
  // bge-base with --npu-ops attn at --pipeline 4, and THAT came from
  // build_panel() binding argument 1 and syncing through the live binding,
  // where a second lane could rebind in between. build_panel now addresses its
  // slot by name and needs no lock; this handles the GEMMs' own window, which
  // does bind and cannot avoid it.
  void set_npu_mutex(std::mutex *mu) {
    qk_.npu_mu = mu;
    av_.npu_mu = mu;
  }

  // The A/C slots of both GEMMs and the single B slot each of them rewrites per
  // head. Must be called once, before the first run().
  void alloc_buffers();

  // Where the softmax runs. Null is the host pass, which is a real path and not
  // a fallback: --npu-ops softm has to name it.
  void set_softmax(NpuEltwise *s) { softm_ = s; }
  bool softmax_on_array() const { return softm_ != nullptr; }

  // An additive mask of n_kv floats, added to every score before the softmax --
  // the extended-attention-mask convention, MASK_FILL in the positions the
  // caller wants ignored. Null for the models that have none: Whisper's encoder
  // attends over every position it was given, and a ViT's encoder has no mask
  // at all. A BERT-family embedder does, and it is not optional there -- the
  // padded positions are real rows of the tensor, so skipping the mask is not a
  // faster attention, it is a different one.
  //
  // `mask_rows` is how many QUERY ROWS one mask covers, and `mask_stride` the
  // distance between consecutive masks. Zero for both means one mask for the
  // whole call, which is the per-sequence caller (one sequence, one mask) and
  // every caller without a mask. They are separate parameters because a batched
  // call needs one mask per SEQUENCE while the mask is indexed by score column:
  // row i of the score matrix is query row i, and its mask belongs to sequence
  // i / mask_rows. Setting mask_rows to the sequence length turns [batch, seq]
  // into the right mask for every row without the caller looping per sequence
  // -- which matters, because the loop would also throw away the design's row
  // count: a dispatch computes attn_qk's M rows whatever it is handed, so a
  // 16-sequence call at 64 rows each would compute 1024 rows sixteen times over.
  //
  // The pointer is the CALLER's and must outlive the calls: it is read per
  // dispatch, not copied.
  void set_additive_mask(const float *mask, int64_t mask_rows = 0,
                         int64_t mask_stride = 0) {
    mask_ = mask;
    mask_rows_ = mask_rows;
    mask_stride_ = mask_stride > 0 ? mask_stride : mask_rows;
  }

  // One attention, for every head:
  //   `q` is [n_q, q_stride] with a head's slice at offset h*head_dim,
  //   `kv` is [n_kv, kv_stride] as a K|V block (K first, then V),
  //   `out` is [n_q, d_model] and receives each head's context in place.
  void run(const float *q, int64_t q_stride, const float *kv, int64_t kv_stride,
           int64_t n_q, int64_t n_kv, int64_t d_model, float scale, float *out);

  int64_t n_dispatch = 0;
  // QK^T (+ scale, mask, softmax) and (softmax).V timed apart, because the
  // status block prints them as sub-items of the attention total. Leaving them
  // at zero while t_attn grows would print "QK^T 0.0 ms" under a run that spent
  // all of its attention time in them -- the intention-versus-value failure
  // this project's status block exists to avoid.
  //
  // These ACCUMULATE, and take_timers() is how a caller reads them: the same
  // instance runs once per layer, so a caller that adds the current value each
  // time sums 1+2+...+N calls' worth and reports N times the real time. That is
  // not a rounding error -- it prints 293 ms for 87 ms of measured attention,
  // a larger total than the block it is a sub-item of.
  struct Timers {
    double qk = 0, av = 0;
    int64_t dispatch = 0;
  };
  Timers take_timers() {
    Timers t{t_qk, t_av, n_dispatch};
    t_qk = t_av = 0.0;
    n_dispatch = 0;
    return t;
  }
  double t_qk = 0, t_av = 0;
  // What the status line says this costs, per window: the dispatches the
  // geometry implies, not a counter that reads zero before the first request.
  std::string note(int64_t n_q, int64_t heads, int64_t layers) const;

private:
  // The [head_dim, n_kv] K panel, or the [n_kv, head_dim] V panel, tiled for the
  // MMAC. `which` picks which: the two are the same numbers transposed, and one
  // buffer holds whichever is being built.
  void build_panel(const float *kv, int64_t kv_stride, int64_t n_kv,
                   int64_t d_model, int64_t head, int64_t kv_head,
                   bool v_side);
  // The two panels' byte counts, checked against the design's B buffer in
  // alloc_buffers() -- a design exported without these streams has a B buffer
  // sized for the set's own weights, and writing past it is heap corruption.
  void host_softmax(float *scores, int64_t n_rows, int64_t n_kv, int64_t n_kv_pad);

  npu::Design &d_;
  app::Pool &pool_;
  int64_t n_kv_;      // padded: the design's N (and attn_av's K)
  int64_t head_dim_ = 0;
  int64_t ctx_cols_ = 0;   // attn_av's N: head_dim rounded up to the design's
  // attn_qk's K, padded up to tile_k; 0 means "not padded", i.e. head_dim_.
  int64_t k_qk_ = 0;
  int64_t kv_width_ = 0;   // key row -> value row; 0 means d_model
  int64_t kv_heads_ = 0;   // 0 means one per query head
  size_t instr_qk_ = 0, instr_av_ = 0;
  int64_t rows_ = 0;
  NpuGemm qk_, av_;
  size_t bslot_qk_ = 0, bslot_av_ = 0;
  NpuEltwise *softm_ = nullptr;
  const float *mask_ = nullptr;   // n_kv floats per mask
  int64_t mask_rows_ = 0;         // query rows one mask covers; 0 = all of them
  int64_t mask_stride_ = 0;       // distance between masks
  // Per-run scratch, sized once: the gathered Q block, one chunk of scores, one
  // chunk of context, and the two panels' staging matrices.
  std::vector<float> aq_, scores_, ctx_, zero_k_, zero_v_;
  std::vector<float> mat_;
  std::vector<uint16_t> panel_;
};

}  // namespace npue::whisper
