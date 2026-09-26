//===- encoder.cpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper encoder stack. See whisper/encoder.hpp for the
// schedule and why it differs from BertEncoder's.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/encoder.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "common/host_kernels.hpp"

namespace npue::whisper {

WhisperEncoder::WhisperEncoder(npue::File &model, npu::Design &design,
                               app::Pool &pool, const Geometry &geom)
    : model_(model), g_(design, pool), pool_(pool), geom_(geom) {}

size_t WhisperEncoder::stage_all() {
  if (streams_.rows <= 0)
    throw std::runtime_error("whisper encoder: set_streams() before stage_all()");
  const int64_t d = geom_.d_model;
  size_t bytes = 0;
  auto operand = [&](const std::string &name, std::vector<size_t> &slots,
                     std::vector<const float *> &bias) {
    slots.push_back(g_.stage_operand(model_, name));
    bias.push_back(model_.raw(name + ".bias").as<float>());
    bytes += model_.raw(name).bytes;
  };
  auto norm = [&](const std::string &name, std::vector<const float *> &gamma,
                  std::vector<const float *> &beta) {
    gamma.push_back(model_.raw(name + ".weight").as<float>());
    beta.push_back(model_.raw(name + ".bias").as<float>());
    bytes += 2 * static_cast<size_t>(d) * sizeof(float);
  };
  for (int64_t L = 0; L < geom_.enc_layers; ++L) {
    const std::string p = "encoder.layers." + std::to_string(L) + ".";
    operand(p + "qkv", s_qkv, b_qkv);
    operand(p + "attn_out", s_ao, b_ao);
    operand(p + "ffn_up", s_fu, b_fu);
    operand(p + "ffn_down", s_fd, b_fd);
    norm(p + "ln1", ln1_gamma, ln1_beta);
    norm(p + "ln2", ln2_gamma, ln2_beta);
  }
  enc_pos_ = model_.raw("encoder.embed_positions").as<float>();
  const std::string f = "encoder.layer_norm.";
  final_gamma_ = model_.raw(f + "weight").as<float>();
  final_beta_ = model_.raw(f + "bias").as<float>();
  bytes += 2 * static_cast<size_t>(d) * sizeof(float);
  g_.alloc_buffers();
  return bytes;
}

void WhisperEncoder::reset_timers() {
  g_.n_dispatch = 0;
  g_.t_dispatch = 0.0;
  g_.t_convert = 0.0;
}

std::vector<float> WhisperEncoder::run(const std::vector<float> &conv_out,
                                       int64_t n_src) {
  const int64_t d = geom_.d_model, inter = geom_.enc_intermediate;
  const int64_t rows = streams_.rows;
  if (rows <= 0)
    throw std::runtime_error("whisper encoder: set_streams() before run()");
  if (n_src <= 0 || n_src > geom_.max_seq)
    throw std::runtime_error(
        "whisper encoder: " + std::to_string(n_src) + " positions, the "
        "checkpoint has weights for " + std::to_string(geom_.max_seq));
  if (static_cast<int64_t>(conv_out.size()) < n_src * d)
    throw std::runtime_error("whisper encoder: conv output is " +
                             std::to_string(conv_out.size() / d) +
                             " rows, " + std::to_string(n_src) + " wanted");

  // The attention scale is already inside the Q weight and the Q bias, unless
  // the container was packed with --no-fold-scale. Applying it here when it is
  // folded would be applying it twice, which is a different model.
  const float scale = geom_.qkv_scale_folded
                          ? 1.0f
                          : static_cast<float>(geom_.attn_scale);

  std::vector<float> x(conv_out.begin(), conv_out.begin() + n_src * d);
  // + embed_positions, before the first LayerNorm: HF adds the position to the
  // convolutional output, not after the first norm.
  for (int64_t t = 0; t < n_src; ++t) {
    const float *pe = enc_pos_ + t * d;
    float *row = x.data() + t * d;
    for (int64_t j = 0; j < d; ++j) row[j] += pe[j];
  }

  // Per-chunk scratch. Sized by the DESIGN's row count, not by n_src, because
  // the design computes all `rows` rows whatever the chunk holds.
  std::vector<float> norm(static_cast<size_t>(rows) * d);
  std::vector<float> qkv_chunk(static_cast<size_t>(rows) * 3 * d);
  std::vector<float> proj(static_cast<size_t>(rows) * d);
  std::vector<float> up(static_cast<size_t>(rows) * inter);
  std::vector<float> down(static_cast<size_t>(rows) * d);
  std::vector<float> qkv_all(static_cast<size_t>(n_src) * 3 * d);
  std::vector<float> ctx(static_cast<size_t>(n_src) * d);
  // The encoder attends over every position, so the score matrix is
  // n_src * heads * n_src -- 54 MB at whisper-tiny's 6 heads and 180 MB at
  // large-v3's 20. That is the price of a full 1500-position attention on the
  // host, and it is the next wall if a bigger Whisper has to be fast.
  std::vector<float> scores(static_cast<size_t>(n_src) * geom_.heads * n_src);

  for (int64_t L = 0; L < geom_.enc_layers; ++L) {
    // ---- self-attention, pre-LN -----------------------------------------
    chunks(n_src, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      std::copy(x.begin() + r0 * d, x.begin() + r1 * d, norm.begin());
      layernorm_rows(norm.data(), n, d, ln1_gamma[L], ln1_beta[L], geom_.ln_eps,
                     pool_);
      g_.run(streams_.qkv, norm.data(), n, rows, d, s_qkv[L], b_qkv[L], 3 * d,
             qkv_chunk.data());
      std::copy(qkv_chunk.begin(), qkv_chunk.begin() + n * 3 * d,
                qkv_all.begin() + r0 * 3 * d);
    });
    // The fused [Q|K|V] row is read as Q at offset 0 and a K|V block at offset
    // d, both with row stride 3*d_model.
    attention(qkv_all.data(), 3 * d, qkv_all.data() + d, 3 * d, n_src, n_src, d,
              geom_.heads, geom_.head_dim, scale, ctx.data(), scores.data(),
              pool_);
    chunks(n_src, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      g_.run(streams_.attn_out, ctx.data() + r0 * d, n, rows, d, s_ao[L],
             b_ao[L], d, proj.data());
      for (int64_t i = 0; i < n * d; ++i)
        x[r0 * d + i] += proj[static_cast<size_t>(i)];
    });

    // ---- feed-forward, pre-LN --------------------------------------------
    chunks(n_src, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      std::copy(x.begin() + r0 * d, x.begin() + r1 * d, norm.begin());
      layernorm_rows(norm.data(), n, d, ln2_gamma[L], ln2_beta[L], geom_.ln_eps,
                     pool_);
      g_.run(streams_.ffn_up, norm.data(), n, rows, d, s_fu[L], b_fu[L], inter,
             up.data());
      gelu_erf_inplace(up.data(), static_cast<size_t>(n) * inter, pool_);
      g_.run(streams_.ffn_down, up.data(), n, rows, inter, s_fd[L], b_fd[L], d,
             down.data());
      for (int64_t i = 0; i < n * d; ++i)
        x[r0 * d + i] += down[static_cast<size_t>(i)];
    });
  }

  // Whisper's encoder output is AFTER the final LayerNorm -- that is what
  // transformers hands the decoder and therefore what the gate compares.
  layernorm_rows(x.data(), n_src, d, final_gamma_, final_beta_, geom_.ln_eps,
                 pool_);
  return x;
}

}  // namespace npue::whisper
