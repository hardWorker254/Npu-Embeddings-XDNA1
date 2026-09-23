//===- gemma_host_encoder.cpp --------------------------------*- C++ -*-===//
//
// Gemma host-only encoder implementation. Moved from
// runtime/src/gemma_encode.cpp and runtime/include/gemma_encode.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "encoders/gemma_host_encoder.hpp"

#include "encoders/gemma_kernels.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace npue {

namespace {

void gemm_f32(const float *a, int64_t M, int64_t K, const float *b, int64_t N,
                float *c) {
  for (int64_t i = 0; i < M; ++i) {
    const float *arow = a + i * K;
    float *crow = c + i * N;
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(arow[k]) * static_cast<double>(b[k * N + j]);
      crow[j] = static_cast<float>(acc);
    }
  }
}

}  // namespace

GemmaHostEncoder::GemmaHostEncoder(npue::File &model) : model_(model) {
  if (model.config_string("arch") != "gemma3_mqa_rope_geglu")
    throw std::runtime_error(
        "GemmaHostEncoder given a .npue whose config arch is '" +
        model.config_string("arch") + "', not 'gemma3_mqa_rope_geglu'");

  hidden_ = model.config_int("hidden");
  heads_ = model.config_int("num_heads");
  kv_heads_ = model.config_int("num_key_value_heads");
  head_dim_ = model.config_int("head_dim");
  inter_ = model.config_int("intermediate");
  layers_ = model.config_int("num_layers");
  dense_hidden_ = model.config_int("dense_hidden");
  eps_ = model.config_double("rms_norm_eps");
  rope_theta_ = model.config_double("rope_theta");
  rope_theta_local_ = model.config_double("rope_local_base_freq");
  sliding_window_pattern_ = model.config_int("sliding_window_pattern");
  query_pre_attn_scalar_ = model.config_double("query_pre_attn_scalar");

  if (hidden_ <= 0 || heads_ <= 0 || kv_heads_ <= 0 || head_dim_ <= 0 ||
      layers_ <= 0 || dense_hidden_ <= 0)
    throw std::runtime_error("the .npue reports a non-positive Gemma shape");
  if (head_dim_ * heads_ != hidden_)
    throw std::runtime_error("head_dim * num_heads != hidden in the .npue");
  if (heads_ % kv_heads_)
    throw std::runtime_error("num_heads is not a multiple of "
                                 "num_key_value_heads (MQA/GQA repeat factor)");

  w_embed_ = model.raw("embed_tokens.weight").as<float>();
  w_norm_ = model.raw("norm.weight").as<float>();
  w_dense2_ = model.raw("dense2.weight").as<float>();
  w_dense3_ = model.raw("dense3.weight").as<float>();

  lp_.resize(static_cast<size_t>(layers_));
  for (int64_t i = 0; i < layers_; ++i) {
    const std::string p = "layer." + std::to_string(i) + ".";
    LayerPtrs &l = lp_[static_cast<size_t>(i)];
    l.q_proj = model.raw(p + "q_proj").as<float>();
    l.k_proj = model.raw(p + "k_proj").as<float>();
    l.v_proj = model.raw(p + "v_proj").as<float>();
    l.o_proj = model.raw(p + "o_proj").as<float>();
    l.q_norm = model.raw(p + "q_norm.weight").as<float>();
    l.k_norm = model.raw(p + "k_norm.weight").as<float>();
    l.ln_in = model.raw(p + "input_layernorm.weight").as<float>();
    l.ln_post_attn = model.raw(p + "post_attention_layernorm.weight").as<float>();
    l.ln_pre_ffn = model.raw(p + "pre_feedforward_layernorm.weight").as<float>();
    l.ln_post_ffn = model.raw(p + "post_feedforward_layernorm.weight").as<float>();
    l.gate_proj = model.raw(p + "gate_proj").as<float>();
    l.up_proj = model.raw(p + "up_proj").as<float>();
    l.down_proj = model.raw(p + "down_proj").as<float>();
  }

  auto tv = model.raw("tokenizer.gemma_table");
  tok = GemmaTokenizer::from_table_bytes(
      reinterpret_cast<const char *>(tv.data), tv.bytes);
}

std::vector<float> GemmaHostEncoder::encode(
    const std::vector<std::string> &texts,
    const std::string &prefix,
    int64_t *tokens) {
  std::vector<float> out;
  int64_t tok_count = 0;
  for (size_t i = 0; i < texts.size(); ++i) {
    const std::string &prefix_name = prefix.empty()
        ? GemmaTokenizer::default_prefix_name()
        : prefix;
    const std::vector<float> v = encode_one(
        texts[i], static_cast<int>(seq()), prefix_name,
        static_cast<size_t>(i), false);
    if (tokens) tok_count += static_cast<int64_t>(v.size() / hidden_);
    out.insert(out.end(), v.begin(), v.end());
  }
  if (tokens) *tokens = tok_count;
  return out;
}

std::vector<float> GemmaHostEncoder::encode_one(
    const std::string &text, int max_len,
    const std::string &prefix_name, size_t index,
    bool allow_truncation) const {
  const GemmaEncoded en = tok.encode(text, max_len, prefix_name);
  if (en.truncated && !allow_truncation)
    throw InputTooLong(index, en.n_tokens_full, max_len);
  const int64_t S = max_len;
  const int64_t H = heads_, KVH = kv_heads_, hd = head_dim_;
  const bool kv_heads_is_one = (KVH == 1);
  if (!kv_heads_is_one)
    throw std::runtime_error(
        "GemmaHostEncoder::encode_one's attention loop assumes num_key_value_heads "
        "== 1 (true for EmbeddingGemma-300M); a checkpoint with KVH > 1 needs "
        "the repeat_kv loop generalised before this can run it");

  const double embed_scale = std::sqrt(static_cast<double>(hidden_));
  std::vector<float> x(static_cast<size_t>(S * hidden_));
  for (int64_t s = 0; s < S; ++s) {
    const int32_t id = en.input_ids[static_cast<size_t>(s)];
    const float *wv = w_embed_ + static_cast<size_t>(id) * hidden_;
    float *dst = x.data() + s * hidden_;
    for (int64_t c = 0; c < hidden_; ++c)
      dst[c] = static_cast<float>(static_cast<double>(wv[c]) * embed_scale);
  }

  const float MASK_FILL = -3.4028235e38f;
  std::vector<float> add_mask(static_cast<size_t>(S));
  for (int64_t s = 0; s < S; ++s)
    add_mask[static_cast<size_t>(s)] =
        en.attention_mask[static_cast<size_t>(s)] ? 0.0f : MASK_FILL;

  std::vector<float> cos_g(static_cast<size_t>(S * hd)), sin_g(static_cast<size_t>(S * hd));
  std::vector<float> cos_l(static_cast<size_t>(S * hd)), sin_l(static_cast<size_t>(S * hd));
  gemma_rope_tables(S, hd, rope_theta_, cos_g.data(), sin_g.data());
  gemma_rope_tables(S, hd, rope_theta_local_, cos_l.data(), sin_l.data());

  const double attn_scale = std::pow(query_pre_attn_scalar_, -0.5);

  std::vector<float> h(static_cast<size_t>(S * hidden_));
  std::vector<float> q(static_cast<size_t>(S * H * hd));
  std::vector<float> k(static_cast<size_t>(S * hd));
  std::vector<float> v(static_cast<size_t>(S * hd));
  std::vector<float> rope_scratch(static_cast<size_t>(S * hd));
  std::vector<float> scores(static_cast<size_t>(S * S));
  std::vector<float> ctx(static_cast<size_t>(S * hidden_));
  std::vector<float> proj(static_cast<size_t>(S * hidden_));
  std::vector<float> gate(static_cast<size_t>(S * inter_));
  std::vector<float> up(static_cast<size_t>(S * inter_));
  std::vector<float> geglu(static_cast<size_t>(S * inter_));
  std::vector<float> down(static_cast<size_t>(S * hidden_));

  for (int64_t L = 0; L < layers_; ++L) {
    const LayerPtrs &lp = lp_[static_cast<size_t>(L)];
    const bool full_attn = gemma_is_full_attention_layer(L, sliding_window_pattern_);
    const float *cos_t = full_attn ? cos_g.data() : cos_l.data();
    const float *sin_t = full_attn ? sin_g.data() : sin_l.data();

    rms_norm_cpu(x.data(), lp.ln_in, h.data(), S, hidden_, static_cast<float>(eps_));

    gemm_f32(h.data(), S, hidden_, lp.q_proj, H * hd, q.data());
    gemm_f32(h.data(), S, hidden_, lp.k_proj, hd, k.data());
    gemm_f32(h.data(), S, hidden_, lp.v_proj, hd, v.data());

    rms_norm_cpu(q.data(), lp.q_norm, q.data(), S * H, hd, static_cast<float>(eps_));
    rms_norm_cpu(k.data(), lp.k_norm, k.data(), S, hd, static_cast<float>(eps_));

    for (int64_t hh = 0; hh < H; ++hh) {
      for (int64_t s = 0; s < S; ++s)
        std::memcpy(rope_scratch.data() + s * hd,
                    q.data() + (s * H + hh) * hd, sizeof(float) * static_cast<size_t>(hd));
      apply_rope_cpu(rope_scratch.data(), cos_t, sin_t, rope_scratch.data(), S, S, hd);
      for (int64_t s = 0; s < S; ++s)
        std::memcpy(q.data() + (s * H + hh) * hd,
                    rope_scratch.data() + s * hd, sizeof(float) * static_cast<size_t>(hd));
    }
    apply_rope_cpu(k.data(), cos_t, sin_t, k.data(), S, S, hd);

    for (int64_t hh = 0; hh < H; ++hh) {
      for (int64_t i = 0; i < S; ++i) {
        const float *qi = q.data() + (i * H + hh) * hd;
        float *srow = scores.data() + i * S;
        for (int64_t j = 0; j < S; ++j) {
          const float *kj = k.data() + j * hd;
          double acc = 0.0;
          for (int64_t d = 0; d < hd; ++d)
            acc += static_cast<double>(qi[d]) * static_cast<double>(kj[d]);
          const double scaled = acc * attn_scale;
          srow[j] = static_cast<float>(scaled) + add_mask[static_cast<size_t>(j)];
        }
      }
      for (int64_t i = 0; i < S; ++i) {
        float *srow = scores.data() + i * S;
        double mx = srow[0];
        for (int64_t j = 1; j < S; ++j) mx = std::max(mx, static_cast<double>(srow[j]));
        double sum = 0.0;
        std::vector<double> e(static_cast<size_t>(S));
        for (int64_t j = 0; j < S; ++j) {
          e[static_cast<size_t>(j)] = std::exp(static_cast<double>(srow[j]) - mx);
          sum += e[static_cast<size_t>(j)];
        }
        for (int64_t j = 0; j < S; ++j)
          srow[j] = static_cast<float>(e[static_cast<size_t>(j)] / sum);
      }
      for (int64_t i = 0; i < S; ++i) {
        const float *prow = scores.data() + i * S;
        float *co = ctx.data() + (i * H + hh) * hd;
        for (int64_t d = 0; d < hd; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < S; ++j)
            acc += static_cast<double>(prow[j]) * static_cast<double>(v[j * hd + d]);
          co[d] = static_cast<float>(acc);
        }
      }
    }

    gemm_f32(ctx.data(), S, hidden_, lp.o_proj, hidden_, proj.data());

    rms_norm_cpu(proj.data(), lp.ln_post_attn, proj.data(), S, hidden_, static_cast<float>(eps_));
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    rms_norm_cpu(x.data(), lp.ln_pre_ffn, h.data(), S, hidden_, static_cast<float>(eps_));
    gemm_f32(h.data(), S, hidden_, lp.gate_proj, inter_, gate.data());
    gemm_f32(h.data(), S, hidden_, lp.up_proj, inter_, up.data());
    geglu_cpu(gate.data(), up.data(), geglu.data(),
              static_cast<size_t>(S * inter_));
    gemm_f32(geglu.data(), S, inter_, lp.down_proj, hidden_, down.data());

    rms_norm_cpu(down.data(), lp.ln_post_ffn, down.data(), S, hidden_, static_cast<float>(eps_));
    for (size_t i = 0; i < x.size(); ++i) x[i] += down[i];
  }

  rms_norm_cpu(x.data(), w_norm_, x.data(), S, hidden_, static_cast<float>(eps_));

  std::vector<double> pooled(static_cast<size_t>(hidden_), 0.0);
  double denom = 0.0;
  for (int64_t s = 0; s < S; ++s) {
    const float m = static_cast<float>(en.attention_mask[static_cast<size_t>(s)]);
    if (m == 0.f) continue;
    denom += m;
    const float *row = x.data() + s * hidden_;
    for (int64_t c = 0; c < hidden_; ++c) pooled[static_cast<size_t>(c)] += row[c] * m;
  }
  denom = std::max(denom, 1e-9);
  std::vector<float> pooled_f(static_cast<size_t>(hidden_));
  for (int64_t c = 0; c < hidden_; ++c)
    pooled_f[static_cast<size_t>(c)] = static_cast<float>(pooled[static_cast<size_t>(c)] / denom);

  std::vector<float> d2(static_cast<size_t>(dense_hidden_));
  gemm_f32(pooled_f.data(), 1, hidden_, w_dense2_, dense_hidden_, d2.data());
  std::vector<float> d3(static_cast<size_t>(hidden_));
  gemm_f32(d2.data(), 1, dense_hidden_, w_dense3_, hidden_, d3.data());

  double nrm = 0.0;
  for (int64_t c = 0; c < hidden_; ++c)
    nrm += static_cast<double>(d3[static_cast<size_t>(c)]) * d3[static_cast<size_t>(c)];
  nrm = std::max(std::sqrt(nrm), 1e-12);
  std::vector<float> out(static_cast<size_t>(hidden_));
  for (int64_t c = 0; c < hidden_; ++c)
    out[static_cast<size_t>(c)] = static_cast<float>(d3[static_cast<size_t>(c)] / nrm);
  return out;
}

int64_t GemmaHostEncoder::hidden() const { return hidden_; }

int64_t GemmaHostEncoder::seq() const {
  return model_.config_int("max_seq_len");
}

}  // namespace npue