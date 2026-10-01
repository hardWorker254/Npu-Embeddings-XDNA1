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

void NpuAttention::set_streams(size_t qk_slot, size_t av_slot, int64_t rows) {
  instr_qk_ = qk_slot;
  instr_av_ = av_slot;
  rows_ = rows;
  if (rows <= 0)
    throw std::runtime_error("whisper attention: the design set's attn rows "
                             "are zero, so no dispatch has a row count");
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
  const size_t bk = static_cast<size_t>(head_dim_) * n_kv_ * 2;
  const size_t bv = static_cast<size_t>(n_kv_) * ctx_cols_ * 2;
  if (static_cast<size_t>(in.buffer_bytes[1]) < bk ||
      static_cast<size_t>(in.buffer_bytes[1]) < bv)
    throw std::runtime_error(
        in.name + ": the B operands of attn_qk/attn_av are " +
        std::to_string(bk) + " and " + std::to_string(bv) +
        " bytes and the design's B buffer is " +
        std::to_string(in.buffer_bytes[1]) + ". This set was exported without "
        "the attention streams (tools/export/export_gemm_rtp.py --npu-extra-ops attn).");
  bslot_qk_ = d_.stage_alloc(1, bk);
  bslot_av_ = d_.stage_alloc(1, bv);
  aq_.assign(static_cast<size_t>(rows_) * head_dim_, 0.f);
  scores_.assign(static_cast<size_t>(rows_) * n_kv_, 0.f);
  ctx_.assign(static_cast<size_t>(rows_) * ctx_cols_, 0.f);
  zero_k_.assign(static_cast<size_t>(n_kv_), 0.f);
  zero_v_.assign(static_cast<size_t>(ctx_cols_), 0.f);
  // The larger of the two staging matrices: the K panel is head_dim x n_kv and
  // the V panel is n_kv x ctx_cols.
  mat_.assign(static_cast<size_t>(n_kv_) * std::max(head_dim_, ctx_cols_), 0.f);
}

void NpuAttention::build_panel(const float *kv, int64_t kv_stride, int64_t n_kv,
                               int64_t d_model, int64_t head, bool v_side) {
  const int64_t hd = head_dim_;
  // Zeroed first, INCLUDING the columns past n_kv: the panel has to tile evenly
  // and a stale value there would be a real key or value from the previous head
  // taking part in this one's attention. The runtime masks the matching score
  // columns, but a zeroed V is what makes the score masking sufficient.
  // k and n first, because the staging matrix is k*n and NOT n*n: for the K
  // panel n is the padded key count and k is a head, and zeroing n*n of a
  // head x n_kv buffer is a heap overflow that the allocator finds later, in a
  // different function, as an invalid free.
  const int64_t k = v_side ? n_kv_ : hd;
  const int64_t n = v_side ? ctx_cols_ : n_kv_;
  std::fill(mat_.begin(), mat_.begin() + static_cast<long>(k * n), 0.f);
  float *m = mat_.data();
  if (!v_side) {
    // B[head_dim, n_kv]: row t of the panel is key t's head_dim components.
    for (int64_t t = 0; t < hd; ++t)
      for (int64_t j = 0; j < n_kv; ++j)
        m[t * n_kv_ + j] = kv[j * kv_stride + head * hd + t];
  } else {
    // B[n_kv, ctx_cols]: row j of the panel is value j's components, and the
    // values are the SECOND half of the K|V block. The columns past head_dim
    // stay zero -- the design's N granularity is wider than a head -- and the
    // C rows they produce are never read.
    for (int64_t j = 0; j < n_kv; ++j)
      for (int64_t t = 0; t < hd; ++t)
        m[j * ctx_cols_ + t] = kv[j * kv_stride + d_model + head * hd + t];
  }
  const std::vector<uint16_t> tiled =
      tile_b_panel(m, k, n, d_.info().b_tile_k, d_.info().tile_n,
                   d_.info().b_mac_s, d_.info().b_mac_t);
  // Bound BEFORE it is written, not by the dispatch that will read it. The two
  // panels are two different slots and the last dispatch left one of them live,
  // so a sync into "whatever B is bound right now" writes the V panel into the
  // K panel's smaller buffer and the driver says `Invalid BO offset and size
  // for sync'ing` -- or, when the sizes happen to match, silently runs the
  // previous head.
  const size_t slot = v_side ? bslot_av_ : bslot_qk_;
  d_.bind(1, slot);
  auto *dst = static_cast<uint16_t *>(d_.slot_ptr(1, slot));
  std::memcpy(dst, tiled.data(), tiled.size() * sizeof(uint16_t));
  // Written through map(): without this the device reads whatever the host cache
  // last flushed, which looks like a design that computed the previous head.
  d_.sync_to_device(1, tiled.size() * sizeof(uint16_t));
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

  for (int64_t h = 0; h < heads; ++h) {
    // The panels depend on the head and on nothing else -- not on the query
    // chunk -- so they are built once and the chunks run inside them.
    build_panel(kv, kv_stride, n_kv, d_model, h, false);
    build_panel(kv, kv_stride, n_kv, d_model, h, true);
    for (int64_t r0 = 0; r0 < n_q; r0 += rows_) {
      const int64_t n = std::min<int64_t>(rows_, n_q - r0);
      // A = this chunk's Q rows, one head's slice, contiguous. The design's K is
      // head_dim, so the gather is a copy of `n` short rows.
      for (int64_t i = 0; i < n; ++i)
        std::memcpy(aq_.data() + i * hd, q + (r0 + i) * q_stride + h * hd,
                    static_cast<size_t>(hd) * sizeof(float));
      qk_.run(instr_qk_, aq_.data(), n, rows_, hd, bslot_qk_, zero_k_.data(),
              n_kv_, scores_.data());
      // scale, then the padded keys. Both in one pass because they are both
      // per-score-column and the score matrix is the only thing either touches.
      for (int64_t i = 0; i < n; ++i) {
        float *sc = scores_.data() + i * n_kv_;
        for (int64_t j = 0; j < n_kv; ++j) sc[j] *= scale;
        for (int64_t j = n_kv; j < n_kv_; ++j) sc[j] = -1.0e30f;
      }
      if (softm_)
        softm_->softmax(scores_.data(), n);
      else
        host_softmax(scores_.data(), n, n_kv, n_kv_);
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
