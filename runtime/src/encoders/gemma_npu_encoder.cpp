//===- gemma_npu_encoder.cpp --------------------------------*- C++ -*-===//
//
// Gemma NPU encoder implementation. Moved from
// runtime/include/gemma_npu_encoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "encoders/gemma_npu_encoder.hpp"

#include "common/app_state.hpp"

using namespace app;

namespace npue {

GemmaNpuEncoder::GemmaNpuEncoder(npue::File &model,
                                       npu::Design &design, Pool &pool)
    : d(design), model_(model), pool_(pool) {
  const std::string arch = model.config_string("arch");
  if (arch != "gemma3_mqa_rope_geglu")
    throw std::runtime_error("GemmaNpuEncoder given arch '" + arch + "'");
  const std::string layout = model.config_string("gemm_layout");
  if (layout != "pretiled_bf16")
    throw std::runtime_error(
        "this container's gemm_layout is '" + layout + "', not "
        "'pretiled_bf16' -- it holds host-side row-major F32 operands and "
        "has no tiled weights for the array. Repack it with "
        "tools/pack_npue.py");

  hidden_ = model.config_int("hidden");
  heads = model.config_int("num_heads");
  kv_heads = model.config_int("num_key_value_heads");
  head_dim = model.config_int("head_dim");
  inter = model.config_int("intermediate");
  layers = model.config_int("num_layers");
  dense_hidden = model.config_int("dense_hidden");
  swp = model.config_int("sliding_window_pattern");
  eps = model.config_double("rms_norm_eps");
  rope_theta = model.config_double("rope_theta");
  rope_theta_local = model.config_double("rope_local_base_freq");
  attn_scale = std::pow(model.config_double("query_pre_attn_scalar"), -0.5);
  qkv_n = model.config_int("qkv_n");
  kv_w = kv_heads * head_dim;

  if (kv_heads != 1)
    throw std::runtime_error(
        "this encoder's attention loop assumes num_key_value_heads == 1 "
        "(EmbeddingGemma-300M); a GQA checkpoint needs the K/V reuse "
        "generalised first");
  if (head_dim * heads != hidden_)
    throw std::runtime_error("head_dim * num_heads != hidden");
  if (head_dim % 2)
    throw std::runtime_error("odd head_dim -- RoPE cannot half-split it");
  if (model.config_string("geglu_halves") != "gate|up")
    throw std::runtime_error(
        "unrecognised geglu_halves '" + model.config_string("geglu_halves") +
        "' -- expected 'gate|up'; refusing rather than guessing which half "
        "of the fused ffn_up gets the GELU.");

  q_off = 0;
  k_off = hidden_;
  v_off = hidden_ + kv_w;
  if (qkv_n < v_off + kv_w)
    throw std::runtime_error("qkv_n is too small to hold Q|K|V");

  w_embed = model.raw("embed_tokens.weight").as<float>();
  w_norm = model.raw("norm.weight").as<float>();
  w_dense2 = model.raw("dense2.weight").as<float>();
  w_dense3 = model.raw("dense3.weight").as<float>();
  lh.resize(static_cast<size_t>(layers));
  for (int64_t L = 0; L < layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    LayerHost &l = lh[static_cast<size_t>(L)];
    l.q_norm = model.raw(p + "q_norm.weight").as<float>();
    l.k_norm = model.raw(p + "k_norm.weight").as<float>();
    l.ln_in = model.raw(p + "input_layernorm.weight").as<float>();
    l.ln_pa = model.raw(p + "post_attention_layernorm.weight").as<float>();
    l.ln_pf = model.raw(p + "pre_feedforward_layernorm.weight").as<float>();
    l.ln_pof = model.raw(p + "post_feedforward_layernorm.weight").as<float>();
  }
  auto tv = model.raw("tokenizer.gemma_table");
  tok = npue::GemmaTokenizer::from_table_bytes(
      reinterpret_cast<const char *>(tv.data), tv.bytes);
}

std::vector<float> GemmaNpuEncoder::encode(
    const std::vector<std::string> &texts,
    const std::string &prefix,
    int64_t *tokens) {
  return encode_batch(texts, prefix, 0, static_cast<size_t>(-1), tokens);
}

std::vector<float> GemmaNpuEncoder::encode_batch(
    const std::vector<std::string> &texts,
    const std::string &prefix,
    size_t index_base,
    size_t n_real,
    int64_t *tokens) {
  if (static_cast<int64_t>(texts.size()) != batch)
    throw std::runtime_error("encode_batch given " +
                                 std::to_string(texts.size()) +
                                 " texts, tier is " + std::to_string(batch));
  ensure_tables();
  double t0 = now_s();
  ids.assign(static_cast<size_t>(rows), 0);
  mask.assign(static_cast<size_t>(rows), 0);
  for (int64_t b = 0; b < batch; ++b) {
    const npue::GemmaEncoded en =
        tok.encode(texts[static_cast<size_t>(b)], static_cast<int>(seq_), prefix);
    if (static_cast<size_t>(b) < n_real) {
      check_truncation(en.truncated, en.n_tokens_full,
                       index_base + static_cast<size_t>(b), seq_);
      if (tokens) *tokens += en.n_tokens;
    }
    for (int64_t s = 0; s < seq_; ++s) {
      ids[static_cast<size_t>(b * seq_ + s)] = en.input_ids[static_cast<size_t>(s)];
      mask[static_cast<size_t>(b * seq_ + s)] =
          static_cast<uint8_t>(en.attention_mask[static_cast<size_t>(s)]);
    }
  }
  t_tok += now_s() - t0;

  x.assign(static_cast<size_t>(rows * hidden_), 0.f);
  hbuf.resize(x.size());
  qkvbuf.resize(static_cast<size_t>(rows * qkv_n));
  ctx.resize(x.size());
  proj.resize(x.size());
  upbuf.resize(static_cast<size_t>(rows * 2 * inter));
  gatedbuf.resize(static_cast<size_t>(rows * inter));
  down.resize(x.size());
  add_mask.resize(static_cast<size_t>(rows));

  const float MASK_FILL = -3.4028235e38f;
  for (int64_t r = 0; r < rows; ++r)
    add_mask[static_cast<size_t>(r)] = mask[static_cast<size_t>(r)] ? 0.f : MASK_FILL;

  const float escale = static_cast<float>(std::sqrt(static_cast<double>(hidden_)));
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *wv = w_embed + static_cast<size_t>(ids[static_cast<size_t>(r)]) * hidden_;
      float *dst = x.data() + r * hidden_;
      for (int64_t c = 0; c < hidden_; ++c) dst[c] = wv[c] * escale;
    }
  });

  for (int64_t L = 0; L < layers; ++L) {
    const LayerHost &l = lh[static_cast<size_t>(L)];
    const bool full = npue::gemma_is_full_attention_layer(L, swp);
    const float *cs_t = full ? cos_g.data() : cos_l.data();
    const float *sn_t = full ? sin_g.data() : sin_l.data();

    rms_norm(x.data(), hidden_, hbuf.data(), hidden_, rows, hidden_, l.ln_in);
    gemm(is_qkv, hbuf.data(), hbuf.size(), s_qkv[L], b_qkv[L], qkvbuf,
         qkv_n, at(ws_qkv, L), at(as_qkv, L));

    for (int64_t hh = 0; hh < heads; ++hh)
      rms_norm(qkvbuf.data() + q_off + hh * head_dim, qkv_n,
               qkvbuf.data() + q_off + hh * head_dim, qkv_n, rows, head_dim,
               l.q_norm);
    rms_norm(qkvbuf.data() + k_off, qkv_n, qkvbuf.data() + k_off, qkv_n,
             rows, head_dim, l.k_norm);

    apply_rope(qkvbuf, cs_t, sn_t);
    attention(qkvbuf, ctx);

    gemm(is_ao, ctx.data(), ctx.size(), s_ao[L], b_ao[L], proj,
         hidden_, at(ws_ao, L), at(as_ao, L));
    rms_norm(proj.data(), hidden_, proj.data(), hidden_, rows, hidden_, l.ln_pa);
    par(x.size(), [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) x[i] += proj[i];
    });

    rms_norm(x.data(), hidden_, hbuf.data(), hidden_, rows, hidden_, l.ln_pf);
    const bool fuse_ffn = fuse_ffn_epilogue &&
                          d.info().a_elem_bytes == 1 && at(as_fd, L);
    FusedNext fn{at(as_fd, L), static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                    nullptr};
    if (fuse_ffn) {
      a_scale_next.resize(static_cast<size_t>(rows));
      fn.scale = a_scale_next.data();
    }
    const bool fuse_ffn_bf16 = fuse_ffn_epilogue && d.info().a_elem_bytes == 2;
    FusedNextBf16 fn_bf16{static_cast<uint16_t *>(d.slot_ptr(0, slot_a))};
    gemm(is_fu, hbuf.data(), hbuf.size(), s_fu[L], b_fu[L], upbuf,
         2 * inter, at(ws_fu, L), at(as_fu, L), fuse_ffn ? &fn : nullptr,
         /*a_ready=*/false, fuse_ffn_bf16 ? &fn_bf16 : nullptr);
    if (fuse_ffn || fuse_ffn_bf16) {
      if (fuse_ffn) a_scale.swap(a_scale_next);
    } else {
      geglu(upbuf, gatedbuf);
    }
    gemm(is_fd, gatedbuf.data(), gatedbuf.size(), s_fd[L], b_fd[L], down,
         hidden_, at(ws_fd, L), at(as_fd, L), nullptr,
         /*a_ready=*/fuse_ffn || fuse_ffn_bf16);
    rms_norm(down.data(), hidden_, down.data(), hidden_, rows, hidden_, l.ln_pof);
    par(x.size(), [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) x[i] += down[i];
    });
  }

  rms_norm(x.data(), hidden_, x.data(), hidden_, rows, hidden_, w_norm);

  std::vector<float> pooled(static_cast<size_t>(batch * hidden_), 0.f);
  for (int64_t b = 0; b < batch; ++b) {
    double denom = 0.0;
    float *o = pooled.data() + b * hidden_;
    for (int64_t s = 0; s < seq_; ++s) {
      if (!mask[static_cast<size_t>(b * seq_ + s)]) continue;
      denom += 1.0;
      const float *row = x.data() + (b * seq_ + s) * hidden_;
      for (int64_t c = 0; c < hidden_; ++c) o[c] += row[c];
    }
    const float inv = static_cast<float>(1.0 / std::max(denom, 1e-9));
    for (int64_t c = 0; c < hidden_; ++c) o[c] *= inv;
  }

  std::vector<float> d2(static_cast<size_t>(batch * dense_hidden));
  gemm_host(pooled.data(), batch, hidden_, w_dense2, dense_hidden, d2.data());
  std::vector<float> out(static_cast<size_t>(batch * hidden_));
  gemm_host(d2.data(), batch, dense_hidden, w_dense3, hidden_, out.data());

  for (int64_t b = 0; b < batch; ++b) {
    float *o = out.data() + b * hidden_;
    double nrm = 0.0;
    for (int64_t c = 0; c < hidden_; ++c) nrm += static_cast<double>(o[c]) * o[c];
    const float inv = static_cast<float>(1.0 / std::max(std::sqrt(nrm), 1e-12));
    for (int64_t c = 0; c < hidden_; ++c) o[c] *= inv;
  }
  return out;
}

int64_t GemmaNpuEncoder::hidden() const { return hidden_; }

int64_t GemmaNpuEncoder::seq() const { return seq_; }

size_t GemmaNpuEncoder::stage_all() {
  size_t bytes = 0;
  const bool i8 = d.info().a_elem_bytes == 1;
  auto one = [&](const std::string &name, std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc,
                   std::vector<const float *> *asm_) {
    const std::string &want = d.info().b_layout_hash;
    const std::string &got = model_.info(name).layout_hash;
    if (want.empty() || got.empty() || want != got)
      throw std::runtime_error(
          name + ": B layout mismatch -- design wants " +
          (want.empty() ? std::string("(nothing stated)") : want.substr(0, 16)) +
          ", container has " +
          (got.empty() ? std::string("(nothing stated)") : got.substr(0, 16)) +
          ". The bytes would be the right size and the wrong order.");
    auto w = model_.raw(name);
    slots.push_back(d.stage(1, w.data, w.bytes));
    bias.push_back(model_.raw(name + ".bias").as<float>());
    if (i8) {
      wsc->push_back(model_.raw(name + ".wscale").as<float>());
      asm_->push_back(model_.raw(name + ".asmooth").as<float>());
    }
    bytes += w.bytes;
  };
  for (int64_t L = 0; L < layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    one(p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
    one(p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
    one(p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
    one(p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
  }
  return bytes;
}

int64_t GemmaNpuEncoder::use_tier(int64_t want) {
  if (tiers.empty()) return batch;
  size_t pick = tiers.size() - 1;
  for (size_t i = 0; i < tiers.size(); ++i)
    if (tiers[i] >= want) { pick = i; break; }
  batch = tiers[pick];
  rows = batch * seq_;
  is_qkv = tier_slots[pick][0];
  is_ao = tier_slots[pick][1];
  is_fu = tier_slots[pick][2];
  is_fd = tier_slots[pick][3];
  return batch;
}

void GemmaNpuEncoder::reset_timers() {
  t_conv = t_in = t_disp = t_out = t_bias = 0;
  t_norm = t_attn = t_rope = t_geglu = t_tok = 0;
  n_dispatch = 0;
}

template <typename F>
void GemmaNpuEncoder::par(size_t n, F &&f) const {
  if (pool_.size() == 1 || n < 65536) {
    f(size_t(0), n);
    return;
  }
  pool_.run([&](int w, int nw) {
    const size_t chunk = ((n / nw) + 63) & ~size_t(63);
    const size_t lo = std::min(n, chunk * size_t(w));
    const size_t hi = std::min(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

template <typename F>
void GemmaNpuEncoder::par_rows(int64_t n, F &&f) const {
  if (pool_.size() == 1) { f(int64_t(0), n); return; }
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n, chunk * w);
    const int64_t hi = std::min<int64_t>(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

double GemmaNpuEncoder::lap(double t0, double &bucket) {
  const double t = now_s();
  bucket += t - t0;
  return t;
}

const float *GemmaNpuEncoder::at(const std::vector<const float *> &v, int64_t L) {
  return L < static_cast<int64_t>(v.size()) ? v[static_cast<size_t>(L)]
                                            : nullptr;
}

void GemmaNpuEncoder::dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                                           int64_t out_n, const float *bias,
                                           uint16_t *dst) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      float *v = row.data();
      int64_t j = 0;
#if defined(__AVX2__)
      if (c_bytes == 2) {
        const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256i lo = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
          __m256i hi = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
          _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                _mm256_loadu_ps(bias + j + 8)));
        }
      } else {
        const float *cr = static_cast<const float *>(c) + r * N;
        for (; j + 8 <= N; j += 8) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                _mm256_loadu_ps(bias + j)));
        }
      }
#endif
      for (; j < N; ++j) {
        const float cf = c_bytes == 2
            ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
            : static_cast<const float *>(c)[r * N + j];
        v[j] = cf + bias[j];
      }
      const int64_t inter = N / 2;
      const float *u = v + inter;
      int64_t k = 0;
#if defined(__AVX2__)
      const __m256 c_half = _mm256_set1_ps(0.5f);
      const __m256 c_one = _mm256_set1_ps(1.0f);
      const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
      const __m256 c_k = _mm256_set1_ps(0.044715f);
      const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
      const __m256 c_lim = _mm256_set1_ps(15.0f);
      for (; k + 8 <= inter; k += 8) {
        __m256 xv = _mm256_loadu_ps(v + k);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
        __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
        y = _mm256_min_ps(
            _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
            c_lim);
        __m256 e = exp2_avx2(_mm256_mul_ps(y, c_2log2e));
        __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                  _mm256_add_ps(e, c_one));
        __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                   _mm256_add_ps(c_one, th));
        _mm256_storeu_ps(v + k, _mm256_mul_ps(act, _mm256_loadu_ps(u + k)));
      }
#endif
      for (; k < inter; ++k) {
        const double xv = v[k];
        const double y =
            0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
        const float act =
            static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
        v[k] = static_cast<float>(static_cast<double>(act) *
                                  static_cast<double>(u[k]));
      }
      bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
    }
  });
}

void GemmaNpuEncoder::gemm(size_t islot, const float *a, size_t a_len, size_t wslot,
                             const float *bias, std::vector<float> &out, int64_t N,
                             const float *wscale, const float *asmooth,
                             FusedNext *fuse, bool a_ready,
                             FusedNextBf16 *fuse_bf16) {
  const bool i8 = d.info().a_elem_bytes == 1;
  if (i8 && (wscale == nullptr || asmooth == nullptr))
    throw std::runtime_error(
        "int8 design but this encoder has no quantisation scales -- the "
        "container is bf16, or it was packed before tools/pack_npue.py "
        "--int8 supported arch=1");
  double t0 = now_s();
  if (i8 && a_ready) {
  } else if (i8) {
    const int64_t K = static_cast<int64_t>(a_len) / rows;
    a_scale.resize(static_cast<size_t>(rows));
    if (inv_smooth.size() != static_cast<size_t>(K) ||
        inv_smooth_src != asmooth) {
      inv_smooth.resize(static_cast<size_t>(K));
      for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
      inv_smooth_src = asmooth;
    }
    quantise_a_int8(a, rows, K, inv_smooth.data(),
                      static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                      a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
    t0 = lap(t0, t_conv);
  } else if (a_ready) {
  } else {
  auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
  par(a_len, [&](size_t lo, size_t hi) {
    bf16_fill(abuf + lo, a + lo, hi - lo);
  });
  t0 = lap(t0, t_conv);
  }
  const float *c;
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d.bind_instr(islot);
    d.bind(0, slot_a);
    d.bind(1, wslot);
    d.bind(2, slot_c);
    d.sync_to_device(0, a_len * d.info().a_elem_bytes);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    const size_t cb = d.info().c_elem_bytes;
    d.sync_from_device(2, static_cast<size_t>(rows) * N * cb);
    t0 = lap(t0, t_out);
    c = static_cast<const float *>(d.slot_ptr(2, slot_c));
  }
  if (i8 && fuse) {
    const int64_t out_n = N / 2;
    if (inv_smooth_next.size() != static_cast<size_t>(out_n) ||
        inv_smooth_next_src != fuse->asmooth) {
      inv_smooth_next.resize(static_cast<size_t>(out_n));
      for (int64_t j = 0; j < out_n; ++j)
        inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
      inv_smooth_next_src = fuse->asmooth;
    }
    dequant_act_quant(
        c, d.info().c_elem_bytes, rows, N, out_n,
        [](float *v, int64_t n) {
          const int64_t inter = n / 2;
          const float *u = v + inter;
          int64_t j = 0;
#if defined(__AVX2__)
          const __m256 c_half = _mm256_set1_ps(0.5f);
          const __m256 c_one = _mm256_set1_ps(1.0f);
          const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
          const __m256 c_k = _mm256_set1_ps(0.044715f);
          const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
          const __m256 c_lim = _mm256_set1_ps(15.0f);
          for (; j + 8 <= inter; j += 8) {
            __m256 xv = _mm256_loadu_ps(v + j);
            __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
            __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
            y = _mm256_min_ps(
                _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                c_lim);
            __m256 e = exp2_avx2(_mm256_mul_ps(y, c_2log2e));
            __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                      _mm256_add_ps(e, c_one));
            __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                       _mm256_add_ps(c_one, th));
            _mm256_storeu_ps(v + j,
                                _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
          }
#endif
          for (; j < inter; ++j) {
            const double xv = v[j];
            const double y =
                0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
            const float act =
                static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
            v[j] = static_cast<float>(static_cast<double>(act) *
                                      static_cast<double>(u[j]));
          }
        },
        a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
        fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
  } else if (i8) {
    dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                    bias, out.data(),
                    [&](int64_t n, auto f) { par_rows(n, f); });
  } else if (fuse_bf16) {
    dequant_act_bf16(c, d.info().c_elem_bytes, N, N / 2, bias,
                       fuse_bf16->dst);
  } else if (d.info().c_elem_bytes == 2) {
    const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb16 + r * N;
        float *o = out.data() + r * N;
        for (int64_t j = 0; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
      }
    });
  } else {
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *cr = c + r * N;
        float *o = out.data() + r * N;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= N; j += 8)
          _mm256_storeu_ps(o + j,
                             _mm256_add_ps(_mm256_castsi256_ps(
                                               _mm256_stream_load_si256(
                                                   reinterpret_cast<const __m256i *>(cr + j))),
                                           _mm256_loadu_ps(bias + j)));
#endif
        for (; j < N; ++j) o[j] = cr[j] + bias[j];
      }
    });
  }
  lap(t0, t_bias);
  ++n_dispatch;
}

void GemmaNpuEncoder::rms_norm(const float *xin, int64_t in_stride, float *xout,
                                 int64_t out_stride, int64_t n_rows, int64_t dim,
                                 const float *w) {
  const double t0 = now_s();
  const float e = static_cast<float>(eps);
  par_rows(n_rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *in = xin + r * in_stride;
      float *o = xout + r * out_stride;
      int64_t j = 0;
      float ss;
#if defined(__AVX2__)
      __m256 acc = _mm256_setzero_ps();
      for (; j + 8 <= dim; j += 8) {
        __m256 v = _mm256_loadu_ps(in + j);
        acc = _mm256_fmadd_ps(v, v, acc);
      }
      ss = hsum256(acc);
#else
      ss = 0.f;
#endif
      for (; j < dim; ++j) ss += in[j] * in[j];
      const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + e);
      j = 0;
#if defined(__AVX2__)
      const __m256 iv = _mm256_set1_ps(inv);
      const __m256 one = _mm256_set1_ps(1.0f);
      for (; j + 8 <= dim; j += 8)
        _mm256_storeu_ps(o + j,
                           _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + j), iv),
                                         _mm256_add_ps(one, _mm256_loadu_ps(w + j))));
#endif
      for (; j < dim; ++j) o[j] = in[j] * inv * (1.0f + w[j]);
    }
  });
  t_norm += now_s() - t0;
}

void GemmaNpuEncoder::apply_rope(std::vector<float> &qkv,
                                    const float *cs_t, const float *sn_t) {
  const double t0 = now_s();
  const int64_t half = head_dim / 2;
  float *__restrict p = qkv.data();
  auto rot = [half](float *v, const float *cs, const float *sn) {
    int64_t dd = 0;
#if defined(__AVX2__)
    for (; dd + 8 <= half; dd += 8) {
      __m256 x1 = _mm256_loadu_ps(v + dd);
      __m256 x2 = _mm256_loadu_ps(v + dd + half);
      __m256 c = _mm256_loadu_ps(cs + dd);
      __m256 s = _mm256_loadu_ps(sn + dd);
      _mm256_storeu_ps(v + dd,
                         _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s)));
      _mm256_storeu_ps(v + dd + half,
                         _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s)));
    }
#endif
    for (; dd < half; ++dd) {
      const float a = v[dd], b = v[dd + half];
      v[dd] = a * cs[dd] - b * sn[dd];
      v[dd + half] = b * cs[dd] + a * sn[dd];
    }
  };
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t s = r % seq_;
      const float *cs = cs_t + s * head_dim;
      const float *sn = sn_t + s * head_dim;
      float *base = p + r * qkv_n;
      for (int64_t hh = 0; hh < heads; ++hh)
        rot(base + q_off + hh * head_dim, cs, sn);
      rot(base + k_off, cs, sn);
    }
  });
  t_rope += now_s() - t0;
}

void GemmaNpuEncoder::geglu(const std::vector<float> &fused,
                               std::vector<float> &out) {
  const double t0 = now_s();
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *g = fused.data() + r * 2 * inter;
      const float *u = g + inter;
      float *o = out.data() + r * inter;
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 c_half = _mm256_set1_ps(0.5f);
      const __m256 c_one = _mm256_set1_ps(1.0f);
      const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
      const __m256 c_k = _mm256_set1_ps(0.044715f);
      const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
      const __m256 c_lim = _mm256_set1_ps(15.0f);
      for (; j + 8 <= inter; j += 8) {
        __m256 xv = _mm256_loadu_ps(g + j);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
        __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
        y = _mm256_min_ps(
            _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
            c_lim);
        __m256 e = exp2_avx2(_mm256_mul_ps(y, c_2log2e));
        __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                  _mm256_add_ps(e, c_one));
        __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                   _mm256_add_ps(c_one, th));
        _mm256_storeu_ps(o + j, _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
      }
#endif
      for (; j < inter; ++j) {
        const double xv = g[j];
        const double y = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
        const float act =
            static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
        o[j] = static_cast<float>(static_cast<double>(act) *
                                  static_cast<double>(u[j]));
      }
    }
  });
  t_geglu += now_s() - t0;
}

void GemmaNpuEncoder::attention(const std::vector<float> &qkv,
                                   std::vector<float> &out) {
  const double t0 = now_s();
  const int64_t pairs = batch * heads;
  par_rows(pairs, [&](int64_t p0, int64_t p1) {
    std::vector<float> row(static_cast<size_t>(seq_));
    for (int64_t pi = p0; pi < p1; ++pi) {
      const int64_t b = pi / heads, hh = pi % heads;
      const float *base = qkv.data() + b * seq_ * qkv_n;
      const float *mk = add_mask.data() + b * seq_;
      for (int64_t i = 0; i < seq_; ++i) {
        const float *qi = base + i * qkv_n + q_off + hh * head_dim;
        float mx = -3.4e38f;
        for (int64_t j = 0; j < seq_; ++j) {
          const float *kj = base + j * qkv_n + k_off;
          int64_t dd = 0;
          float acc;
#if defined(__AVX2__)
          __m256 a = _mm256_setzero_ps();
          for (; dd + 8 <= head_dim; dd += 8)
            a = _mm256_fmadd_ps(_mm256_loadu_ps(qi + dd),
                                _mm256_loadu_ps(kj + dd), a);
          acc = hsum256(a);
#else
          acc = 0.f;
#endif
          for (; dd < head_dim; ++dd) acc += qi[dd] * kj[dd];
          const float sv =
              acc * static_cast<float>(attn_scale) + mk[j];
          row[static_cast<size_t>(j)] = sv;
          mx = std::max(mx, sv);
        }
        float sum = 0.f;
        for (int64_t j = 0; j < seq_; ++j) {
          const float e = std::exp(row[static_cast<size_t>(j)] - mx);
          row[static_cast<size_t>(j)] = e;
          sum += e;
        }
        const float inv = 1.0f / sum;
        float *o = out.data() + (b * seq_ + i) * hidden_ + hh * head_dim;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(head_dim));
        for (int64_t j = 0; j < seq_; ++j) {
          const float w = row[static_cast<size_t>(j)] * inv;
          const float *vj = base + j * qkv_n + v_off;
          int64_t dd = 0;
#if defined(__AVX2__)
          const __m256 wv = _mm256_set1_ps(w);
          for (; dd + 8 <= head_dim; dd += 8)
            _mm256_storeu_ps(o + dd,
                               _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj + dd),
                                               _mm256_loadu_ps(o + dd)));
#endif
          for (; dd < head_dim; ++dd) o[dd] += w * vj[dd];
        }
      }
    }
  });
  t_attn += now_s() - t0;
}

void GemmaNpuEncoder::gemm_host(const float *a, int64_t M, int64_t K, const float *b,
                                  int64_t N, float *c) const {
  par_rows(M, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) {
      const float *ar = a + i * K;
      float *cr = c + i * N;
      std::memset(cr, 0, sizeof(float) * static_cast<size_t>(N));
      for (int64_t k = 0; k < K; ++k) {
        const float av = ar[k];
        if (av == 0.f) continue;
        const float *br = b + k * N;
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 avv = _mm256_set1_ps(av);
        for (; j + 8 <= N; j += 8)
          _mm256_storeu_ps(cr + j, _mm256_fmadd_ps(avv, _mm256_loadu_ps(br + j),
                                                    _mm256_loadu_ps(cr + j)));
#endif
        for (; j < N; ++j) cr[j] += av * br[j];
      }
    }
  });
}

void GemmaNpuEncoder::ensure_tables() {
  if (!cos_g.empty()) return;
  cos_g.resize(static_cast<size_t>(seq_ * head_dim));
  sin_g.resize(cos_g.size());
  cos_l.resize(cos_g.size());
  sin_l.resize(cos_g.size());
  npue::gemma_rope_tables(seq_, head_dim, rope_theta, cos_g.data(), sin_g.data());
  npue::gemma_rope_tables(seq_, head_dim, rope_theta_local, cos_l.data(),
                          sin_l.data());
}

}  // namespace npue