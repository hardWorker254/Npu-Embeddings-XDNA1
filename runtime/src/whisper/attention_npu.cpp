//===- attention_npu.cpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's attention as two GEMMs. See attention_npu.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/attention_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/host_kernels.hpp"   // now_s

namespace npue::whisper {

void NpuAttention::set_streams(size_t qk_slot, size_t av_slot, int64_t rows,
                               int64_t k_qk) {
  instr_qk_ = qk_slot;
  instr_av_ = av_slot;
  rows_ = rows;
  if (rows <= 0)
    throw std::runtime_error("whisper attention: the design set's attn rows "
                             "are zero, so no dispatch has a row count");
  // Zero means "not padded". Resolving it here rather than at every use means
  // the rest of the file reads one number, and an omitted argument is the
  // Whisper case rather than a silent zero in a GEMM's K.
  k_qk_ = k_qk > 0 ? k_qk : head_dim_;
  if (k_qk_ < head_dim_)
    throw std::runtime_error(
        "whisper attention: attn_qk's K is " + std::to_string(k_qk_) +
        " and a head is " + std::to_string(head_dim_) +
        " wide. K is padded UP to the tile, never down -- a K below one head is "
        "a design exported against a different model.");
  if (k_qk_ % d_.info().b_tile_k)
    throw std::runtime_error(
        "whisper attention: attn_qk's K is " + std::to_string(k_qk_) +
        ", which is not a multiple of the design's b_tile_k " +
        std::to_string(d_.info().b_tile_k) +
        ". The builder refuses such a design, so this set was not produced by "
        "the current exporter.");
}

void NpuAttention::set_kv_geometry(int64_t kv_width, int64_t kv_heads) {
  kv_width_ = kv_width;
  kv_heads_ = kv_heads;
}

void NpuAttention::alloc_buffers() {
  const auto &in = d_.info();
  // Both GEMMs live on ONE xclbin -- they are two instruction streams of the set
  // -- so they share the design's buffers, and each needs its own A/C slots
  // because a design's A buffer is bound per dispatch, not per stream.
  qk_.alloc_buffers();
  av_.alloc_buffers();
  // ONE B slot each, rewritten per head. The panels are activations, not
  // weights, so staging one per head per layer per window would allocate
  // heads*layers slots and never reuse one.
  if (ctx_cols_ < head_dim_)
    throw std::runtime_error(
        in.name + ": the attn_av stream's N is " + std::to_string(ctx_cols_) +
        " and a head is " + std::to_string(head_dim_) +
        " wide. A design's N is padded up to the design's own granularity, so "
        "this cannot be what the export meant.");
  const size_t bk = static_cast<size_t>(k_qk_) * n_kv_ * 2;
  const size_t bv = static_cast<size_t>(n_kv_) * ctx_cols_ * 2;
  if (static_cast<size_t>(in.buffer_bytes[1]) < bk ||
      static_cast<size_t>(in.buffer_bytes[1]) < bv)
    throw std::runtime_error(
        in.name + ": the B operands of attn_qk/attn_av are " +
        std::to_string(bk) + " and " + std::to_string(bv) +
        " bytes and the design's B buffer is " +
        std::to_string(in.buffer_bytes[1]) + ". This set was exported without "
        "the attention streams (tools/export/export_gemm_rtp.py --npu-ops attn).");
  bslot_qk_ = d_.stage_alloc(1, bk);
  bslot_av_ = d_.stage_alloc(1, bv);
  // One row per query row, k_qk_ wide rather than head_dim_ wide. The columns
  // past a head are ZERO and stay zero: they are written once here and the
  // per-dispatch gather below only ever rewrites the first head_dim_ of each
  // row, which is what makes the padded K contribute 0*0 to every score instead
  // of whatever the previous dispatch left behind.
  aq_.assign(static_cast<size_t>(rows_) * k_qk_, 0.f);
  scores_.assign(static_cast<size_t>(rows_) * n_kv_, 0.f);
  ctx_.assign(static_cast<size_t>(rows_) * ctx_cols_, 0.f);
  zero_k_.assign(static_cast<size_t>(n_kv_), 0.f);
  zero_v_.assign(static_cast<size_t>(ctx_cols_), 0.f);
  // The larger of the two staging matrices: the K panel is head_dim x n_kv and
  // the V panel is n_kv x ctx_cols.
  mat_.assign(static_cast<size_t>(n_kv_) * std::max(k_qk_, ctx_cols_), 0.f);
}

void NpuAttention::build_panel(const float *kv, int64_t kv_stride, int64_t n_kv,
                               int64_t d_model, int64_t head, int64_t kv_head,
                               bool v_side) {
  const int64_t hd = head_dim_;
  // Zeroed first, INCLUDING the columns past n_kv and the ROWS past a head:
  // the panel has to tile evenly and a stale value there would be a real key or
  // value from the previous head taking part in this one's attention. The
  // runtime masks the matching score columns, but a zeroed V is what makes the
  // score masking sufficient. The rows [head_dim, k_qk) are the K padding and
  // are zeroed here for the same reason -- paired with the zero tail of the
  // gathered Q they contribute 0*0, which is what makes a 32-wide head
  // computable on a tile_k of 64.
  // k and n first, because the staging matrix is k*n and NOT n*n: for the K
  // panel n is the padded key count and k is the padded head, and zeroing n*n
  // of a head x n_kv buffer is a heap overflow that the allocator finds later,
  // in a different function, as an invalid free.
  const int64_t k = v_side ? n_kv_ : k_qk_;
  const int64_t n = v_side ? ctx_cols_ : n_kv_;
  std::fill(mat_.begin(), mat_.begin() + static_cast<long>(k * n), 0.f);
  float *m = mat_.data();
  if (!v_side) {
    // B[k_qk, n_kv]: row t of the panel is key t's components. `kv_head` is the
    // key/value head this query head reads -- the identity for BERT and
    // Whisper, and 0 for every head of a multi-query model.
    for (int64_t t = 0; t < hd; ++t)
      for (int64_t j = 0; j < n_kv; ++j)
        m[t * n_kv_ + j] = kv[j * kv_stride + kv_head * hd + t];
  } else {
    // B[n_kv, ctx_cols]: row j of the panel is value j's components, and the
    // values are the SECOND half of the K|V block. The columns past head_dim
    // stay zero -- the design's N granularity is wider than a head -- and the
    // C rows they produce are never read.
    const int64_t v_off = kv_width_ > 0 ? kv_width_ : d_model;
    for (int64_t j = 0; j < n_kv; ++j)
      for (int64_t t = 0; t < hd; ++t)
        m[j * ctx_cols_ + t] = kv[j * kv_stride + v_off + kv_head * hd + t];
  }
  const std::vector<uint16_t> tiled =
      tile_b_panel(m, k, n, d_.info().b_tile_k, d_.info().tile_n,
                   d_.info().b_mac_s, d_.info().b_mac_t);
  // Written through the NAMED slot, not through "whichever B is bound": the two
  // panels are two different slots and the last dispatch left one of them live,
  // so a sync into the live binding writes the V panel into the K panel's
  // smaller buffer and the driver says `Invalid BO offset and size for
  // sync'ing` -- or, when the sizes happen to match, silently runs the previous
  // head.
  //
  // This used to be bind(1, slot) + sync_to_device(1, ...), and under
  // --pipeline it was a race rather than a lookup: `active` is per-Design and
  // shared, so a second lane's build_panel() could rebind argument 1 between
  // our bind and our sync, and we would push 147456 bytes (n_kv * ctx_cols * 2,
  // the V panel) into the other lane's 49152-byte K panel BO. That is exactly
  // what XRT reported on bge-base with --npu-ops attn -- err=22, offset 0, size
  // 147456 -- and no amount of locking around it fixes the reason the bind was
  // here. sync_slot_to_device() names the BO instead, needs no binding, and so
  // needs no mutex either; the dispatch below binds for itself inside the one
  // the rest of the runtime already takes.
  const size_t slot = v_side ? bslot_av_ : bslot_qk_;
  auto *dst = static_cast<uint16_t *>(d_.slot_ptr(1, slot));
  std::memcpy(dst, tiled.data(), tiled.size() * sizeof(uint16_t));
  // Written through map(): without this the device reads whatever the host cache
  // last flushed, which looks like a design that computed the previous head.
  d_.sync_slot_to_device(1, slot, tiled.size() * sizeof(uint16_t));
}

void NpuAttention::host_softmax(float *scores, int64_t n_rows, int64_t n_kv,
                                int64_t n_kv_pad) {
  pool_.run([&](int w, int nw) {
    for (int64_t r = w; r < n_rows; r += nw) {
      float *sc = scores + r * n_kv_pad;
      float mx = -3.4e38f;
      for (int64_t j = 0; j < n_kv_pad; ++j) mx = std::max(mx, sc[j]);
      float sum = 0.f;
      for (int64_t j = 0; j < n_kv_pad; ++j) {
        sc[j] = std::exp(sc[j] - mx);
        sum += sc[j];
      }
      const float inv = sum > 0.f ? 1.0f / sum : 0.f;
      for (int64_t j = 0; j < n_kv_pad; ++j) sc[j] *= inv;
    }
  });
}

void NpuAttention::run(const float *q, int64_t q_stride, const float *kv,
                       int64_t kv_stride, int64_t n_q, int64_t n_kv,
                       int64_t d_model, float scale, float *out) {
  const int64_t hd = head_dim_;
  if (n_kv > n_kv_)
    throw std::runtime_error(
        "whisper attention: " + std::to_string(n_kv) +
        " keys, but the design's attn streams are built for " +
        std::to_string(n_kv_) +
        ". Re-export the design set with this model's own window (the N of "
        "attn_qk is the padded n_kv and the K of attn_av is the same number).");
  if (d_model % hd)
    throw std::runtime_error("whisper attention: head_dim " +
                             std::to_string(hd) + " does not divide d_model " +
                             std::to_string(d_model));
  const int64_t heads = d_model / hd;
  // A multi-query model has fewer key/value heads than query heads, and query
  // head h reads key/value head `h / (heads / kv_heads)`. The divisor is
  // checked rather than assumed: a kv_heads that does not divide the head count
  // has no such mapping, and without the check the division would silently pick
  // the nearest head and produce a plausible-looking wrong answer.
  const int64_t kv_heads = kv_heads_ > 0 ? kv_heads_ : heads;
  if (heads % kv_heads)
    throw std::runtime_error(
        "whisper attention: " + std::to_string(kv_heads) +
        " key/value heads do not divide " + std::to_string(heads) +
        " query heads, so no head mapping exists");
  if (kv_width_ > 0 && kv_width_ % hd)
    throw std::runtime_error("whisper attention: the K|V half-width " +
                             std::to_string(kv_width_) +
                             " is not a multiple of head_dim " +
                             std::to_string(hd));

  for (int64_t h = 0; h < heads; ++h) {
    const int64_t kvh = h / (heads / kv_heads);
    // The panels depend on the head and on nothing else -- not on the query
    // chunk -- so they are built once and the chunks run inside them. With one
    // key/value head and several query heads they are rebuilt to the SAME
    // numbers, which is redundant work rather than a wrong answer.
    build_panel(kv, kv_stride, n_kv, d_model, h, kvh, false);
    build_panel(kv, kv_stride, n_kv, d_model, h, kvh, true);
    for (int64_t r0 = 0; r0 < n_q; r0 += rows_) {
      const int64_t n = std::min<int64_t>(rows_, n_q - r0);
      // A = this chunk's Q rows, one head's slice, contiguous, at the PADDED
      // row stride. The design's K is k_qk_ and may be wider than a head; the
      // columns past head_dim are left at the zeros alloc_buffers() put there,
      // which is the whole reason that fill happens once.
      for (int64_t i = 0; i < n; ++i)
        std::memcpy(aq_.data() + i * k_qk_,
                    q + (r0 + i) * q_stride + h * hd,
                    static_cast<size_t>(hd) * sizeof(float));
      const double t0 = app::now_s();
      qk_.run(instr_qk_, aq_.data(), n, rows_, k_qk_, bslot_qk_, zero_k_.data(),
              n_kv_, scores_.data());
      // scale, then the caller's additive mask, then the padded keys. All
      // three are per-score-column and the score matrix is the only thing any
      // of them touches, so they are one pass -- but they are three STEPS in
      // that pass, in this order, matching the host path's qk -> mask ->
      // softmax. Folding the mask in before the scale would scale MASK_FILL.
      for (int64_t i = 0; i < n; ++i) {
        float *sc = scores_.data() + i * n_kv_;
        // The mask for score row i is the one belonging to the SEQUENCE query
        // row i sits in. mask_rows_ is that sequence's length in query rows;
        // zero means the whole call shares one mask, which is every caller
        // without a mask and the per-sequence one.
        const float *mk = mask_ ? mask_ +
            (mask_rows_ ? (i / mask_rows_) * mask_stride_ : 0)
                              : nullptr;
        for (int64_t j = 0; j < n_kv; ++j)
          sc[j] = sc[j] * scale + (mk ? mk[j] : 0.f);
        for (int64_t j = n_kv; j < n_kv_; ++j) sc[j] = -1.0e30f;
      }
      if (softm_)
        softm_->softmax(scores_.data(), n);
      else
        host_softmax(scores_.data(), n, n_kv, n_kv_);
      t_qk += app::now_s() - t0;
      const double t1 = app::now_s();
      av_.run(instr_av_, scores_.data(), n, rows_, n_kv_, bslot_av_,
              zero_v_.data(), ctx_cols_, ctx_.data());
      // Row i of the C buffer starts at i * ctx_cols_, NOT at i * head_dim:
      // the design's N is padded up to its own granularity, so a context row is
      // 128 floats wide here and holds 64 of answer. Reading it with a head_dim
      // stride takes the first 64 floats of row 0, then the SECOND 64 floats of
      // row 0 -- which are the padding zeros -- and every other row's output
      // lands one row early. It is a plausible-looking context of the wrong
      // values, which is the worst shape a bug in this file can take.
      for (int64_t i = 0; i < n; ++i)
        std::memcpy(out + (r0 + i) * d_model + h * hd,
                    ctx_.data() + i * ctx_cols_,
                    static_cast<size_t>(hd) * sizeof(float));
      t_av += app::now_s() - t1;
      n_dispatch += 2;
    }
  }
}

std::string NpuAttention::note(int64_t n_q, int64_t heads, int64_t layers) const {
  const int64_t chunks = (n_q + rows_ - 1) / rows_;
  const int64_t per_window = chunks * heads * 2 * layers;
  return std::to_string(rows_) + " rows per dispatch, " + std::to_string(heads) +
         " heads x " + std::to_string(chunks) + " chunks x 2 GEMMs x " +
         std::to_string(layers) + " layers = " + std::to_string(per_window) +
         " dispatches per window; " +
         (softm_ ? "softmax on the array" : "softmax on the host") +
         "; n_kv padded to " + std::to_string(n_kv_);
}

}  // namespace npue::whisper
