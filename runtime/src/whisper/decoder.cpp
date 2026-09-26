//===- decoder.cpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper decoder stack. See whisper/decoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/host_kernels.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace npue::whisper {

WhisperDecoder::WhisperDecoder(npue::File &model, npu::Design &design,
                               app::Pool &pool, const Geometry &geom)
    : model_(model), g_(design, pool), pool_(pool), geom_(geom) {}

const DecoderTier &WhisperDecoder::tier(int64_t batch) const {
  for (const auto &t : tiers_)
    if (t.batch == batch) return t;
  throw std::runtime_error("whisper decoder: no tier for batch " +
                           std::to_string(batch) + " (the design set has " +
                           std::to_string(tiers_.size()) + " tiers)");
}

void WhisperDecoder::set_step_tier(int64_t batch) {
  step_batch_ = batch;
  (void)tier(batch);
}

void WhisperDecoder::set_prefill_tier(int64_t batch) {
  prefill_batch_ = batch;
  (void)tier(batch);
}

size_t WhisperDecoder::stage_all() {
  if (step_batch_ <= 0 || prefill_batch_ <= 0)
    throw std::runtime_error(
        "whisper decoder: set_step_tier() and set_prefill_tier() before "
        "stage_all()");
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
    bytes += 2 * static_cast<size_t>(geom_.d_model) * sizeof(float);
  };
  for (int64_t L = 0; L < geom_.dec_layers; ++L) {
    const std::string p = "decoder.layers." + std::to_string(L) + ".";
    operand(p + "self_qkv", s_sqkv, b_sqkv);
    operand(p + "self_attn_out", s_sao, b_sao);
    operand(p + "cross_q", s_cq, b_cq);
    operand(p + "cross_kv", s_ckv, b_ckv);
    operand(p + "cross_attn_out", s_cao, b_cao);
    operand(p + "ffn_up", s_fu, b_fu);
    operand(p + "ffn_down", s_fd, b_fd);
    norm(p + "ln1", ln1_gamma, ln1_beta);
    norm(p + "ln2", ln2_gamma, ln2_beta);
    norm(p + "ln3", ln3_gamma, ln3_beta);
  }
  dec_pos_ = model_.raw("decoder.embed_positions").as<float>();
  const std::string f = "decoder.layer_norm.";
  final_gamma_ = model_.raw(f + "weight").as<float>();
  final_beta_ = model_.raw(f + "bias").as<float>();
  embed_tokens_ = model_.raw("decoder.embed_tokens").as<float>();
  bytes += 2 * static_cast<size_t>(geom_.d_model) * sizeof(float) +
           static_cast<size_t>(geom_.vocab) * geom_.d_model * sizeof(float);
  self_kv_.resize(static_cast<size_t>(geom_.dec_layers));
  cross_kv_.resize(static_cast<size_t>(geom_.dec_layers));
  for (int64_t L = 0; L < geom_.dec_layers; ++L)
    self_kv_[L].reserve(static_cast<size_t>(geom_.max_target) * 2 * geom_.d_model);
  g_.alloc_buffers();
  return bytes;
}

void WhisperDecoder::set_source(const std::vector<float> &enc, int64_t n_src) {
  const int64_t d = geom_.d_model;
  if (n_src <= 0 || static_cast<int64_t>(enc.size()) < n_src * d)
    throw std::runtime_error("whisper decoder: encoder output is " +
                             std::to_string(enc.size() / d) + " rows, " +
                             std::to_string(n_src) + " wanted");
  n_src_ = n_src;
  const DecoderTier &t = tier(prefill_batch_);
  std::vector<float> out(static_cast<size_t>(t.rows) * 2 * d);
  for (int64_t L = 0; L < geom_.dec_layers; ++L) {
    std::vector<float> &cache = cross_kv_[L];
    cache.assign(static_cast<size_t>(n_src) * 2 * d, 0.f);
    for (int64_t r0 = 0; r0 < n_src; r0 += t.rows) {
      // n_src - r0, not n_src: clamping to `n_src` alone yields r0 + rows on
      // every chunk but the last, which is longer than the dispatch.
      const int64_t n = std::min<int64_t>(n_src - r0, t.rows);
      g_.run(t.streams.cross_kv, enc.data() + r0 * d, n, t.rows, d, s_ckv[L],
             b_ckv[L], 2 * d, out.data());
      std::copy(out.begin(), out.begin() + n * 2 * d,
                cache.begin() + static_cast<long>(r0 * 2 * d));
    }
  }
  reset();
}

void WhisperDecoder::set_suppression(std::vector<int32_t> suppress,
                                     std::vector<int32_t> begin) {
  const int64_t vocab = geom_.vocab;
  for (const auto *list : {&suppress, &begin})
    for (int32_t id : *list)
      if (id < 0 || id >= vocab)
        throw std::runtime_error(
            "whisper decoder: the decoding policy names id " +
            std::to_string(id) + ", outside this container's " +
            std::to_string(vocab) + "-entry vocabulary -- the lists were "
            "packed against a different checkpoint");
  if (suppress.empty() && begin.empty())
    throw std::runtime_error(
        "whisper decoder: no decoding policy. The container carries neither "
        "decoder.suppress_tokens nor decoder.begin_suppress_tokens, so a "
        "greedy step here would pick a different token than the reference "
        "implementation does. Re-pack it OVER THIS FILE, or the old container "
        "keeps being opened: python tools/pack_npue.py --model-dir <ckpt> "
        "--out <this file> --device <npu1|npu2> --max-seq 1500");
  suppress_ = std::move(suppress);
  begin_suppress_ = std::move(begin);
}

void WhisperDecoder::reset() {
  n_cached_ = 0;
  n_generated_ = 0;
  for (auto &kv : self_kv_) kv.clear();
}

void WhisperDecoder::logits_from(const std::vector<float> &h,
                                 std::vector<float> *out, int32_t *argmax) {
  // The checkpoint has no proj_out: the logit matrix is the tied token
  // embedding, so logits = h @ embed_tokens^T. Host, because a 51866 x 384
  // matvec is 20 MFLOP -- nothing next to the 28 dispatches that produced h.
  const int64_t d = geom_.d_model, vocab = geom_.vocab;
  const bool policy = has_suppression();
  // The logits are materialised whenever anyone will look at them OR the
  // policy has to be applied to them -- which is the point: the vector a caller
  // sees and the id the step chose have to come from the same numbers.
  std::vector<float> local;
  float *dst = nullptr;
  if (out) {
    out->assign(static_cast<size_t>(vocab), 0.f);
    dst = out->data();
  } else if (policy) {
    local.assign(static_cast<size_t>(vocab), 0.f);
    dst = local.data();
  }
  const float *hv = h.data();
  float best = 0.f;
  int32_t best_id = 0;
  bool have = false;
  pool_.run([&](int w, int nw) {
    float local_best = 0.f;
    int32_t local_id = 0;
    bool local_have = false;
    for (int64_t v = w; v < vocab; v += nw) {
      const float *e = embed_tokens_ + v * d;
      float acc = 0.f;
      int64_t j = 0;
#if defined(__AVX2__)
      __m256 a = _mm256_setzero_ps();
      for (; j + 8 <= d; j += 8)
        a = _mm256_fmadd_ps(_mm256_loadu_ps(hv + j), _mm256_loadu_ps(e + j), a);
      acc = app::hsum256(a);
#endif
      for (; j < d; ++j) acc += hv[j] * e[j];
      if (dst) dst[v] = acc;
      if (!local_have || acc > local_best) {
        local_best = acc;
        local_id = static_cast<int32_t>(v);
        local_have = true;
      }
    }
    // Ties broken by the lower id, so the result does not depend on which
    // worker got there first -- argmax is a function of the logits, not of the
    // pool's schedule.
    std::lock_guard<std::mutex> lk(argmax_mu_);
    if (local_have && (!have || local_best > best ||
                       (local_best == best && local_id < best_id))) {
      best = local_best;
      best_id = local_id;
      have = true;
    }
  });
  if (dst) {
    if (policy) {
      // transformers installs TWO processors and they OVERLAP on the first
      // step: SuppressTokensLogitsProcessor(suppress_tokens) at every step,
      // and SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens) at the
      // first generated step only. Applying only the begin list there -- which
      // is the obvious reading of "begin" -- lets a token the full list forbids
      // win the first step: on a 3 s tone that is 50362 instead of 50259, and
      // the whole transcript after it is a different language. Applied to the
      // vector, not to the argmax, so that a caller reading `logits` sees
      // -3.4e+38 exactly where the choice was forbidden.
      for (int32_t id : suppress_)
        if (id >= 0 && id < vocab) dst[id] = -3.4e38f;
      if (n_generated_ == 0)
        for (int32_t id : begin_suppress_)
          if (id >= 0 && id < vocab) dst[id] = -3.4e38f;
    }
    best = 0.f;
    best_id = 0;
    have = false;
    for (int64_t v = 0; v < vocab; ++v) {
      const float x = dst[v];
      if (!have || x > best) {
        best = x;
        best_id = static_cast<int32_t>(v);
        have = true;
      }
    }
  }
  if (argmax) *argmax = have ? best_id : 0;
  ++n_generated_;
}

int32_t WhisperDecoder::step(int32_t token, int64_t position,
                             std::vector<float> *hidden,
                             std::vector<float> *logits) {
  const int64_t d = geom_.d_model, inter = geom_.dec_intermediate;
  if (token < 0 || token >= geom_.vocab)
    throw std::runtime_error("whisper decoder: token " + std::to_string(token) +
                             " is outside the vocabulary (" +
                             std::to_string(geom_.vocab) + " ids)");
  // Positions are the cache's own index. Accepting a token at any other
  // position would produce a fluent transcript with the wrong position
  // embedding, which is the worst kind of wrong.
  if (position != n_cached_)
    throw std::runtime_error(
        "whisper decoder: token at position " + std::to_string(position) +
        " but the cache holds " + std::to_string(n_cached_) +
        " positions. Tokens must be fed in order, or reset() first.");
  if (n_cached_ >= geom_.max_target)
    throw std::runtime_error("whisper decoder: position " +
                             std::to_string(position) +
                             " is past max_target_positions " +
                             std::to_string(geom_.max_target));

  const DecoderTier &t = tier(step_batch_);
  const float scale = geom_.qkv_scale_folded
                          ? 1.0f
                          : static_cast<float>(geom_.attn_scale);

  // embed_tokens + embed_positions, no norm in between: HF's decoder adds the
  // position to the token embedding and enters the first pre-LN block.
  std::vector<float> h(static_cast<size_t>(d));
  const float *et = embed_tokens_ + static_cast<size_t>(token) * d;
  const float *pe = dec_pos_ + position * d;
  for (int64_t j = 0; j < d; ++j) h[j] = et[j] + pe[j];

  std::vector<float> norm(static_cast<size_t>(d)), qkv(static_cast<size_t>(3 * d));
  std::vector<float> proj(static_cast<size_t>(d)), up(static_cast<size_t>(inter));
  std::vector<float> down(static_cast<size_t>(d)), ctx(static_cast<size_t>(d));

  for (int64_t L = 0; L < geom_.dec_layers; ++L) {
    // ---- self-attention over the cache, one new row ----------------------
    norm = h;
    layernorm_rows(norm.data(), 1, d, ln1_gamma[L], ln1_beta[L], geom_.ln_eps,
                   pool_);
    g_.run(t.streams.self_qkv, norm.data(), 1, t.rows, d, s_sqkv[L], b_sqkv[L],
           3 * d, qkv.data());
    std::vector<float> &cache = self_kv_[L];
    cache.resize(static_cast<size_t>(n_cached_ + 1) * 2 * d);
    std::copy(qkv.begin() + d, qkv.begin() + 3 * d,
              cache.begin() + static_cast<long>(n_cached_ * 2 * d));
    attention(qkv.data(), 3 * d, cache.data(), 2 * d, 1, n_cached_ + 1, d,
              geom_.heads, geom_.head_dim, scale, ctx.data(), nullptr, pool_);
    g_.run(t.streams.self_attn_out, ctx.data(), 1, t.rows, d, s_sao[L], b_sao[L],
           d, proj.data());
    for (int64_t j = 0; j < d; ++j) h[j] += proj[static_cast<size_t>(j)];

    // ---- cross-attention over the encoder output -------------------------
    norm = h;
    layernorm_rows(norm.data(), 1, d, ln2_gamma[L], ln2_beta[L], geom_.ln_eps,
                   pool_);
    g_.run(t.streams.cross_q, norm.data(), 1, t.rows, d, s_cq[L], b_cq[L], d,
           qkv.data());
    // The Q row is all this step needs from cross_q: its K|V half was folded
    // into a separate operand precisely so the two could be fed from different
    // A operands.
    attention(qkv.data(), d, cross_kv_[L].data(), 2 * d, 1, n_src_, d,
              geom_.heads, geom_.head_dim, scale, ctx.data(), nullptr, pool_);
    g_.run(t.streams.cross_attn_out, ctx.data(), 1, t.rows, d, s_cao[L],
           b_cao[L], d, proj.data());
    for (int64_t j = 0; j < d; ++j) h[j] += proj[static_cast<size_t>(j)];

    // ---- feed-forward -----------------------------------------------------
    norm = h;
    layernorm_rows(norm.data(), 1, d, ln3_gamma[L], ln3_beta[L], geom_.ln_eps,
                   pool_);
    g_.run(t.streams.ffn_up, norm.data(), 1, t.rows, d, s_fu[L], b_fu[L], inter,
           up.data());
    gelu_erf_inplace(up.data(), static_cast<size_t>(inter), pool_);
    g_.run(t.streams.ffn_down, up.data(), 1, t.rows, inter, s_fd[L], b_fd[L], d,
           down.data());
    for (int64_t j = 0; j < d; ++j) h[j] += down[static_cast<size_t>(j)];
  }

  ++n_cached_;
  layernorm_rows(h.data(), 1, d, final_gamma_, final_beta_, geom_.ln_eps, pool_);
  int32_t next = 0;
  logits_from(h, logits, &next);
  if (hidden) *hidden = h;
  return next;
}

std::vector<int32_t> WhisperDecoder::greedy(const std::vector<int32_t> &prompt,
                                            int32_t stop_id, int64_t max_new) {
  if (prompt.empty())
    throw std::runtime_error("whisper decoder: greedy() needs at least one "
                             "prompt token (the control tokens are the model's "
                             "own; see tools/packers/whisper.py)");
  if (static_cast<int64_t>(prompt.size()) > geom_.max_target)
    throw std::runtime_error("whisper decoder: prompt of " +
                             std::to_string(prompt.size()) +
                             " tokens exceeds max_target_positions " +
                             std::to_string(geom_.max_target));
  reset();
  int32_t next = 0;
  for (size_t i = 0; i < prompt.size(); ++i) {
    next = step(prompt[i], static_cast<int64_t>(i));
    if (next == stop_id) return {};
  }
  std::vector<int32_t> out{next};
  while (out.back() != stop_id && n_cached_ < geom_.max_target &&
         (max_new <= 0 || static_cast<int64_t>(out.size()) < max_new))
    out.push_back(step(out.back(), n_cached_));
  if (out.back() == stop_id) out.pop_back();
  return out;
}

void WhisperDecoder::reset_timers() {
  g_.n_dispatch = 0;
  g_.t_dispatch = 0.0;
  g_.t_convert = 0.0;
}

}  // namespace npue::whisper
