//===- bert_encoder.cpp ------------------------------------------------*- C++ -*-===//
//
// BERT-family NPU encoder implementation. Moved from
// runtime/include/bert_encoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "encoders/bert_encoder.hpp"

#include "common/app_state.hpp"
#include "encoders/gemma_kernels.hpp"
#include "runtime/model.hpp"
#include "tokenizers/tokenizer_facade.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace app;

namespace npue {

BertEncoder::BertEncoder(npue::File &model,
                           npu::Design &qkv_d, npu::Design &ao_d,
                           npu::Design &fu_d, npu::Design &fd_d,
                           npu::Design &gelu_d, npu::Design &ln_d,
                           npu::Design &sm_d, Pool &pool)
    : model_(model), qkv_(qkv_d), attn_out_(ao_d), ffn_up_(fu_d),
      ffn_down_(fd_d), gelu_(gelu_d), layernorm_(ln_d), softmax_(sm_d),
      pool_(pool) {}

int64_t BertEncoder::hidden() const { return g_hidden; }

int64_t BertEncoder::seq() const { return g_seq; }

size_t BertEncoder::stage_all() {
  size_t bytes = 0;
  const bool i8 = qkv_.info().a_elem_bytes == 1;
  auto one = [&](npu::Design &d, const std::string &name,
                 std::vector<size_t> &slots,
                 std::vector<const float *> &bias,
                 std::vector<const float *> *wsc = nullptr,
                 std::vector<const float *> *asm_ = nullptr) {
    const std::string &want = d.info().b_layout_hash;
    const std::string &got = model_.info(name).layout_hash;
    if (want.empty())
      throw std::runtime_error(d.info().name +
                               "/design.json has no "
                               "b_layout_hash -- re-export with "
                               "tools/export_xclbin.py");
    if (got.empty())
      throw std::runtime_error(name + ": .npue tensor carries no "
                               "layout_hash -- repack with "
                               "tools/pack_npue.py");
    if (want != got)
      throw std::runtime_error(
          name + ": layout mismatch -- design " + d.info().name +
          " wants " + want.substr(0, 16) + "..., file has " +
          got.substr(0, 16) + "... The bytes would be the right size and the "
          "wrong order.");
    auto w = model_.raw(name);
    slots.push_back(d.stage(1, w.data, w.bytes));
    bias.push_back(model_.raw(name + ".bias").as<float>());
    bytes += w.bytes;
    if (i8 && wsc) {
      wsc->push_back(model_.raw(name + ".wscale").as<float>());
      asm_->push_back(model_.raw(name + ".asmooth").as<float>());
    }
  };
  for (int64_t L = 0; L < g_layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    one(qkv_, p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
    one(attn_out_, p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
    one(ffn_up_, p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
    one(ffn_down_, p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
  }

  std::vector<float> gb(2 * g_hidden);
  auto ln_one = [&](const std::string &g, const std::string &b) {
    std::memcpy(gb.data(), model_.raw(g).data, g_hidden * sizeof(float));
    std::memcpy(gb.data() + g_hidden, model_.raw(b).data,
                g_hidden * sizeof(float));
    s_ln.push_back(layernorm_.stage(1, gb.data(), gb.size() * sizeof(float)));
    bytes += gb.size() * sizeof(float);
    h_gamma.push_back(model_.raw(g).as<float>());
    h_beta.push_back(model_.raw(b).as<float>());
  };
  if (host_ln) {
    auto ln_host = [&](const std::string &g, const std::string &b) {
      s_ln.push_back(s_ln.size() + 1);
      h_gamma.push_back(model_.raw(g).as<float>());
      h_beta.push_back(model_.raw(b).as<float>());
    };
    ln_host("embeddings.ln.weight", "embeddings.ln.bias");
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      ln_host(p + "ln1.weight", p + "ln1.bias");
      ln_host(p + "ln2.weight", p + "ln2.bias");
    }
    return bytes;
  }
  ln_one("embeddings.ln.weight", "embeddings.ln.bias");
  for (int64_t L = 0; L < g_layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    ln_one(p + "ln1.weight", p + "ln1.bias");
    ln_one(p + "ln2.weight", p + "ln2.bias");
  }
  return bytes;
}

int64_t BertEncoder::use_tier(int64_t want) {
  if (tiers.empty()) return batch;
  size_t pick = tiers.size() - 1;
  for (size_t i = 0; i < tiers.size(); ++i)
    if (tiers[i] >= want) { pick = i; break; }
  batch = tiers[pick];
  rows = batch * g_seq;
  is_qkv = tier_slots[pick][0];
  is_ao = tier_slots[pick][1];
  is_fu = tier_slots[pick][2];
  is_fd = tier_slots[pick][3];
  return batch;
}

void BertEncoder::reset_timers() {
  t_qk = t_av = 0.0;
  t_npu = t_attn = 0.0;
  t_hostln = t_hostsm = t_hostgelu = 0.0;
  t_conv = t_in = t_disp = t_out = t_bias = 0.0;
  n_dispatch = 0;
}

template <typename F>
void BertEncoder::par(size_t n, F &&f) const {
  if (pool_.size() == 1 || n < 65536) { f(size_t(0), n); return; }
  pool_.run([&](int w, int nw) {
    const size_t chunk = ((n / nw) + 63) & ~size_t(63);
    const size_t lo = std::min(n, chunk * size_t(w));
    const size_t hi = std::min(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

template <typename F>
void BertEncoder::par_rows(int64_t n, F &&f) const {
  if (pool_.size() == 1) { f(int64_t(0), n); return; }
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n, chunk * w);
    const int64_t hi = std::min<int64_t>(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

double BertEncoder::lap(double t0, double &bucket) {
  double t = now_s();
  bucket += t - t0;
  return t;
}

void BertEncoder::eltwise(npu::Design &d, const EltSlots &slots, float *x,
                          size_t n) {
  double t0 = now_s();
  // The A buffer is this lane's own, so the conversion runs unlocked and in
  // parallel. It is the dispatch window below that has to be exclusive.
  auto *in_bf16 = static_cast<uint16_t *>(d.slot_ptr(0, slots.a));
  par(n, [&](size_t lo, size_t hi) {
    bf16_fill(in_bf16 + lo, x + lo, hi - lo);
  });
  const size_t cap_elems = d.info().buffer_bytes[0] / sizeof(uint16_t);
  if (n < cap_elems)
      std::memset(in_bf16 + n, 0, (cap_elems - n) * sizeof(uint16_t));
  t0 = lap(t0, t_conv);
  // ONE mutex, exactly as gemm() takes it, because the thing being protected is
  // the same thing gemm() protects: Design's `active` slot table is shared
  // mutable state, so two lanes binding and dispatching one Design concurrently
  // read a torn binding and each other's buffers. Previously only gemm() took
  // it -- and gelu/softmax/layernorm, which have no per-lane weight slot to
  // hide behind, silently raced here.
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d.bind(0, slots.a);
    d.bind(1, slots.c);
    d.sync_to_device(0, cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(d.output_index(), cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_out);
  }
  // Read back through slot_ptr, not host_ptr: an explicit slot touches neither
  // the shared binding table nor the lock, and the C buffer is this lane's own.
  const auto *out_bf16 = static_cast<const uint16_t *>(
      d.slot_ptr(d.output_index(), slots.c));
  par(n, [&](size_t lo, size_t hi) {
    bf16_read(x + lo, out_bf16 + lo, hi - lo);
  });
  lap(t0, t_conv);
  ++n_dispatch;
}

void BertEncoder::layer_norm(std::vector<float> &x, size_t slot) {
  if (host_ln) { layer_norm_cpu(x, slot - 1); return; }
  double t0 = now_s();
  auto *in_bf16 = static_cast<uint16_t *>(
      layernorm_.slot_ptr(0, slots_ln.a));
  par(x.size(), [&](size_t lo, size_t hi) {
    bf16_fill(in_bf16 + lo, x.data() + lo, hi - lo);
  });
  const size_t cap_elems = layernorm_.info().buffer_bytes[0] / sizeof(uint16_t);
  if (x.size() < cap_elems)
      std::memset(in_bf16 + x.size(), 0, (cap_elems - x.size()) * sizeof(uint16_t));
  t0 = lap(t0, t_conv);
  // Same window, same reason, same mutex as eltwise() and gemm(). `slot` is the
  // gamma|beta site, and lanes sit at DIFFERENT sites at the same instant -- so
  // an unlocked bind(1, slot) handed this dispatch another layer's parameters,
  // which is the larger half of the corruption.
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    layernorm_.bind(0, slots_ln.a);
    layernorm_.bind(1, slot);
    layernorm_.bind(2, slots_ln.c);
    layernorm_.sync_to_device(0, cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_in);
    layernorm_.dispatch_only();
    t0 = lap(t0, t_disp);
    layernorm_.sync_from_device(layernorm_.output_index(),
                                cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_out);
  }
  const auto *out_bf16 = static_cast<const uint16_t *>(
      layernorm_.slot_ptr(layernorm_.output_index(), slots_ln.c));
  par(x.size(), [&](size_t lo, size_t hi) {
    bf16_read(x.data() + lo, out_bf16 + lo, hi - lo);
  });
  lap(t0, t_conv);
  ++n_dispatch;
}

void BertEncoder::layer_norm_cpu(std::vector<float> &x, size_t site) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t n_rows = static_cast<int64_t>(x.size()) / g_hidden;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
    for (int64_t r = lo; r < hi; ++r) {
      float *row = x.data() + r * g_hidden;
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_hidden; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / g_hidden;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_hidden; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / g_hidden;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (int64_t j = 0; j < g_hidden; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                    _mm256_loadu_ps(g + j),
                                    _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, y);
      }
#else
      double sm = 0.0;
      for (int64_t j = 0; j < g_hidden; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / g_hidden);
      double sv = 0.0;
      for (int64_t j = 0; j < g_hidden; ++j) {
        const float d = row[j] - mean;
        sv += static_cast<double>(d) * d;
      }
      const float var = static_cast<float>(sv / g_hidden);
      const float is = 1.0f / std::sqrt(var + 1e-12f);
      for (int64_t j = 0; j < g_hidden; ++j)
        row[j] = (row[j] - mean) * is * g[j] + b[j];
#endif
    }
  });
  t_hostln += now_s() - t0;
}

void BertEncoder::add_additive_mask(std::vector<float> &scores) {
  const int64_t rows_per_seq = g_heads * g_seq;
  const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
  pool_.run([&](int w, int nw) {
    for (int64_t r = w; r < n_rows; r += nw) {
      float *row = scores.data() + r * g_seq;
      const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
      for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
    }
  });
}

void BertEncoder::softmax_cpu(std::vector<float> &scores) {
  double t0 = now_s();
  const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
    const int64_t rows_per_seq = g_heads * g_seq;
    for (int64_t r = lo; r < hi; ++r) {
      float *row = scores.data() + r * g_seq;
      const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
      for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
#if defined(__AVX2__)
      __m256 mx = _mm256_loadu_ps(row);
      for (int64_t j = 8; j < g_seq; j += 8)
        mx = _mm256_max_ps(mx, _mm256_loadu_ps(row + j));
      __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx),
                             _mm256_extractf128_ps(mx, 1));
      m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
      m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
      const __m256 mv = _mm256_set1_ps(_mm_cvtss_f32(m4));
      const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
      const __m256 argfloor = _mm256_set1_ps(-120.0f);
      __m256 sum = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_seq; j += 8) {
        __m256 a = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv),
                                    log2e);
        __m256 e = exp2_avx2(_mm256_max_ps(a, argfloor));
        _mm256_storeu_ps(row + j, e);
        sum = _mm256_add_ps(sum, e);
      }
      const __m256 inv = _mm256_set1_ps(1.0f / hsum256(sum));
      for (int64_t j = 0; j < g_seq; j += 8)
        _mm256_storeu_ps(row + j,
                            _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
#else
      float m = row[0];
      for (int64_t j = 1; j < g_seq; ++j) m = std::max(m, row[j]);
      float sum = 0.f;
      for (int64_t j = 0; j < g_seq; ++j) {
        row[j] = std::exp(row[j] - m);
        sum += row[j];
      }
      const float inv = 1.0f / sum;
      for (int64_t j = 0; j < g_seq; ++j) row[j] *= inv;
#endif
    }
  });
  t_hostsm += now_s() - t0;
}

void BertEncoder::gelu_cpu(std::vector<float> &x) {
  double t0 = now_s();
  par(x.size(), [&](size_t lo, size_t hi) {
    size_t i = lo;
#if defined(__AVX2__)
    const __m256 vR = _mm256_set1_ps(4.0f);
    const __m256 vz = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 c0 = _mm256_set1_ps(-7.2340282171e-05f);
    const __m256 c1 = _mm256_set1_ps(1.8179518005e-03f);
    const __m256 c2 = _mm256_set1_ps(-1.7707383379e-02f);
    const __m256 c3 = _mm256_set1_ps(8.4577147641e-02f);
    const __m256 c4 = _mm256_set1_ps(-1.9228671834e-01f);
    const __m256 c5 = _mm256_set1_ps(9.8431124458e-02f);
    const __m256 c6 = _mm256_set1_ps(3.6137852062e-01f);
    const __m256 c7 = _mm256_set1_ps(-4.9454128936e-01f);
    const __m256 c8 = _mm256_set1_ps(-1.3007010117e-04f);
    for (; i + 8 <= hi; i += 8) {
      __m256 v = _mm256_loadu_ps(x.data() + i);
      __m256 u = _mm256_min_ps(_mm256_andnot_ps(sign, v), vR);
      __m256 pl = _mm256_fmadd_ps(c0, u, c1);
      pl = _mm256_fmadd_ps(pl, u, c2);
      pl = _mm256_fmadd_ps(pl, u, c3);
      pl = _mm256_fmadd_ps(pl, u, c4);
      pl = _mm256_fmadd_ps(pl, u, c5);
      pl = _mm256_fmadd_ps(pl, u, c6);
      pl = _mm256_fmadd_ps(pl, u, c7);
      pl = _mm256_fmadd_ps(pl, u, c8);
      _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_max_ps(v, vz), pl));
    }
#endif
    for (; i < hi; ++i) {
      const float v = x[i];
      const float u = std::min(std::fabs(v), 4.0f);
      float pl = -7.2340282171e-05f;
      pl = pl * u + 1.8179518005e-03f;
      pl = pl * u + -1.7707383379e-02f;
      pl = pl * u + 8.4577147641e-02f;
      pl = pl * u + -1.9228671834e-01f;
      pl = pl * u + 9.8431124458e-02f;
      pl = pl * u + 3.6137852062e-01f;
      pl = pl * u + -4.9454128936e-01f;
      pl = pl * u + -1.3007010117e-04f;
      x[i] = std::max(v, 0.0f) + pl;
    }
  });
  t_hostgelu += now_s() - t0;
}

void BertEncoder::swiglu_cpu(const std::vector<float> &x,
                              std::vector<float> &out) {
  double t0 = now_s();
  const int64_t inter = g_ffn;
  const int64_t n_rows = rows;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo_r = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi_r = std::min<int64_t>(n_rows, lo_r + chunk);
    for (int64_t r = lo_r; r < hi_r; ++r) {
      const float *lo = x.data() + r * 2 * inter;
      const float *hi = lo + inter;
      float *dst = out.data() + r * inter;
      if (g_gated_act == GatedAct::GeluErf) {
        for (int64_t j = 0; j < inter; ++j)
          dst[j] = lo[j] * gelu_erf_exact(hi[j]);
        continue;
      }
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
      const __m256 argfloor = _mm256_set1_ps(-120.0f);
      const __m256 one = _mm256_set1_ps(1.0f);
      for (; j + 8 <= inter; j += 8) {
        __m256 xv = _mm256_loadu_ps(hi + j);
        __m256 a = _mm256_max_ps(
            _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
            argfloor);
        __m256 e = exp2_avx2(a);
        __m256 s = _mm256_div_ps(xv, _mm256_add_ps(one, e));
        __m256 loV = _mm256_loadu_ps(lo + j);
        _mm256_storeu_ps(dst + j, _mm256_mul_ps(loV, s));
      }
#endif
      for (; j < inter; ++j) {
        const float xv = hi[j];
        float a = -xv * 1.4426950408889634f;
        if (a < -120.0f) a = -120.0f;
        const float e = std::exp2(a);
        const float s = xv / (1.0f + e);
        dst[j] = lo[j] * s;
      }
    }
  });
  t_hostgelu += now_s() - t0;
}

const float *BertEncoder::i8w(const std::vector<const float *> &v, int64_t L) {
  return v.empty() ? nullptr : v[static_cast<size_t>(L)];
}

void BertEncoder::dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                                    int64_t out_n, bool gated, const float *bias,
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
      if (!gated) {
        int64_t k = 0;
#if defined(__AVX2__)
        for (; k + 8 <= N; k += 8)
          _mm256_storeu_ps(v + k, gelu8(_mm256_loadu_ps(v + k)));
#endif
        for (; k < N; ++k) v[k] = gelu8(v[k]);
      } else if (g_gated_act == GatedAct::GeluErf) {
        const int64_t inter = N / 2;
        const float *hi = v + inter;
        for (int64_t k = 0; k < inter; ++k)
          v[k] = v[k] * gelu_erf_exact(hi[k]);
      } else {
        const int64_t inter = N / 2;
        const float *hi = v + inter;
        int64_t k = 0;
#if defined(__AVX2__)
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; k + 8 <= inter; k += 8) {
          __m256 xv = _mm256_loadu_ps(hi + k);
          __m256 a = _mm256_max_ps(
              _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
              argfloor);
          __m256 e = exp2_avx2(a);
          __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
          _mm256_storeu_ps(v + k, _mm256_mul_ps(_mm256_loadu_ps(v + k), sg));
        }
#endif
        for (; k < inter; ++k) {
          const float xv = hi[k];
          float a = -xv * 1.4426950408889634f;
          if (a < -120.0f) a = -120.0f;
          v[k] = v[k] * (xv / (1.0f + std::exp2f(a)));
        }
      }
      bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
    }
  });
}

void BertEncoder::gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
                         size_t wslot, const float *bias, std::vector<float> &out,
                         int64_t N, const float *wscale,
                         const float *asmooth,
                         FusedNext *fuse, bool a_ready,
                         FusedNextBf16 *fuse_bf16) {
  const bool i8 = d.info().a_elem_bytes == 1;
  if (i8 && (wscale == nullptr || asmooth == nullptr))
    throw std::runtime_error(
        "int8 design but this encoder has no quantisation scales -- the "
        "container is bf16, or a pipeline lane was constructed without "
        "copying ws_*/as_* from lane 0");
  double t0 = now_s();
  if (i8 && a_ready) {
  } else if (i8) {
    const int64_t K = static_cast<int64_t>(a.size()) / rows;
    a_scale.resize(static_cast<size_t>(rows));
    if (inv_smooth.size() != static_cast<size_t>(K) ||
        inv_smooth_src != asmooth) {
      inv_smooth.resize(static_cast<size_t>(K));
      for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
      inv_smooth_src = asmooth;
    }
    const float *ias = inv_smooth.data();
    auto *abuf = static_cast<int8_t *>(d.slot_ptr(0, slot_a));
    quantise_a_int8(a.data(), rows, K, ias, abuf, a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
    t0 = lap(t0, t_conv);
  } else if (a_ready) {
  } else {
  auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
  par(a.size(), [&](size_t lo, size_t hi) {
    bf16_fill(abuf + lo, a.data() + lo, hi - lo);
  });
  t0 = lap(t0, t_conv);
  }
  const float *c;
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    if (unified) d.bind_instr(islot);
    d.bind(0, slot_a);
    d.bind(1, wslot);
    d.bind(2, slot_c);
    d.sync_to_device(0, a.size() * d.info().a_elem_bytes);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    const size_t cb = d.info().c_elem_bytes;
    d.sync_from_device(2, static_cast<size_t>(rows) * N * cb);
    t0 = lap(t0, t_out);
    c = static_cast<const float *>(d.slot_ptr(2, slot_c));
  }
  if (i8 && fuse) {
    const int64_t Kn = fuse->gated ? N / 2 : N;
    if (inv_smooth_next.size() != static_cast<size_t>(Kn) ||
        inv_smooth_next_src != fuse->asmooth) {
      inv_smooth_next.resize(static_cast<size_t>(Kn));
      for (int64_t j = 0; j < Kn; ++j)
        inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
      inv_smooth_next_src = fuse->asmooth;
    }
    const int64_t out_n = fuse->gated ? N / 2 : N;
    dequant_act_quant(
        c, d.info().c_elem_bytes, rows, N, out_n,
        [gated = fuse->gated](float *v, int64_t n) {
          if (!gated) {
            int64_t j = 0;
#if defined(__AVX2__)
            for (; j + 8 <= n; j += 8)
              _mm256_storeu_ps(v + j, gelu8(_mm256_loadu_ps(v + j)));
#endif
            for (; j < n; ++j) v[j] = gelu8(v[j]);
            return;
          }
          if (g_gated_act == GatedAct::GeluErf) {
            const int64_t inter = n / 2;
            const float *hi = v + inter;
            for (int64_t j = 0; j < inter; ++j)
              v[j] = v[j] * gelu_erf_exact(hi[j]);
            return;
          }
          const int64_t inter = n / 2;
          const float *hi = v + inter;
          int64_t j = 0;
#if defined(__AVX2__)
          const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
          const __m256 argfloor = _mm256_set1_ps(-120.0f);
          const __m256 one = _mm256_set1_ps(1.0f);
          for (; j + 8 <= inter; j += 8) {
            __m256 xv = _mm256_loadu_ps(hi + j);
            __m256 a = _mm256_max_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                argfloor);
            __m256 e = exp2_avx2(a);
            __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
            _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(v + j), sg));
          }
#endif
          for (; j < inter; ++j) {
            const float xv = hi[j];
            float a = -xv * 1.4426950408889634f;
            if (a < -120.0f) a = -120.0f;
            v[j] = v[j] * (xv / (1.0f + std::exp2f(a)));
          }
        },
        a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
        fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
  } else if (i8) {
    dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); }, sim_c_bf16);
  } else if (fuse_bf16) {
    const int64_t out_n = fuse_bf16->gated ? N / 2 : N;
    dequant_act_bf16(c, d.info().c_elem_bytes, N, out_n, fuse_bf16->gated,
                       bias, fuse_bf16->dst);
  } else if (d.info().c_elem_bytes == 2) {
    const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
    par(size_t(rows), [&](size_t r0, size_t r1) {
      for (size_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb16 + r * N;
        float *o = out.data() + r * N;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256i lo = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
          __m256i hi = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
          _mm256_storeu_ps(o + j,
                           _mm256_add_ps(_mm256_castsi256_ps(lo),
                                         _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(o + j + 8,
                           _mm256_add_ps(_mm256_castsi256_ps(hi),
                                         _mm256_loadu_ps(bias + j + 8)));
        }
#endif
        for (; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
      }
    });
  } else {
    par(size_t(rows), [&](size_t r0, size_t r1) {
      for (size_t r = r0; r < r1; ++r) {
        const float *cr = c + r * N;
        float *o = out.data() + r * N;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= N; j += 8) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          _mm256_storeu_ps(o + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                _mm256_loadu_ps(bias + j)));
        }
#endif
        for (; j < N; ++j) o[j] = cr[j] + bias[j];
      }
    });
  }
  lap(t0, t_bias);
  ++n_dispatch;
}

void BertEncoder::apply_rope_qkv(std::vector<float> &qkv) {
  if (!rope_ready) {
    rope_cos.resize(static_cast<size_t>(g_seq * g_head_dim));
    rope_sin.resize(static_cast<size_t>(g_seq * g_head_dim));
    if (!g_rope_inv_freq.empty()) {
      const int64_t half = g_head_dim / 2;
      for (int64_t s = 0; s < g_seq; ++s) {
        float *cs = rope_cos.data() + s * g_head_dim;
        float *sn = rope_sin.data() + s * g_head_dim;
        for (int64_t j = 0; j < half; ++j) {
          const double ang =
              static_cast<double>(s) *
              static_cast<double>(g_rope_inv_freq[static_cast<size_t>(j)]);
          const float c = static_cast<float>(std::cos(ang));
          const float si = static_cast<float>(std::sin(ang));
          cs[j] = c;
          cs[half + j] = c;
          sn[j] = si;
          sn[half + j] = si;
        }
      }
    } else {
      npue::gemma_rope_tables(g_seq, g_head_dim, g_rope_theta,
                              rope_cos.data(), rope_sin.data());
    }
    rope_ready = true;
  }
  const int64_t half = g_head_dim / 2;
  const int64_t row_stride = 3 * g_hidden;
  const int64_t n_rows = batch * g_seq;
  float *__restrict p = qkv.data();
  const float *__restrict cos_p = rope_cos.data();
  const float *__restrict sin_p = rope_sin.data();

  auto rotate_pair = [half](float *v, const float *cs, const float *sn) {
    int64_t d = 0;
#if defined(__AVX2__)
    for (; d + 8 <= half; d += 8) {
      __m256 x1 = _mm256_loadu_ps(v + d);
      __m256 x2 = _mm256_loadu_ps(v + d + half);
      __m256 c = _mm256_loadu_ps(cs + d);
      __m256 s = _mm256_loadu_ps(sn + d);
      __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s));
      __m256 o2 = _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s));
      _mm256_storeu_ps(v + d, o1);
      _mm256_storeu_ps(v + d + half, o2);
    }
#endif
    for (; d < half; ++d) {
      const float x1 = v[d], x2 = v[d + half];
      v[d] = x1 * cs[d] - x2 * sn[d];
      v[d + half] = x2 * cs[d] + x1 * sn[d];
    }
  };

  pool_.run([&](int w, int nw) {
    for (int64_t row = w; row < n_rows; row += nw) {
      const int64_t s = row % g_seq;
      const float *cs = cos_p + s * g_head_dim;
      const float *sn = sin_p + s * g_head_dim;
      float *row_base = p + row * row_stride;
      for (int64_t h = 0; h < g_heads; ++h) {
        rotate_pair(row_base + h * g_head_dim, cs, sn);
        rotate_pair(row_base + g_hidden + h * g_head_dim, cs, sn);
      }
    }
  });
}

template <int NV>
void BertEncoder::qk_impl(const std::vector<float> &qkvbuf,
                          std::vector<float> &scores) {
  const int64_t pairs = batch * g_heads;
  const float *__restrict qkv_p = qkvbuf.data();
  float *__restrict sc_p = scores.data();
  pool_.run([&](int w, int nw) {
    for (int64_t p = w; p < pairs; p += nw) {
      const int64_t b = p / g_heads, h = p % g_heads;
      for (int64_t i = 0; i < g_seq; ++i) {
        const float *q = &qkv_p[(b * g_seq + i) * 3 * g_hidden + h * g_head_dim];
        float *dst = &sc_p[(p * g_seq + i) * g_seq];
#if defined(__AVX512F__)
        const int64_t nv = NV ? NV : g_head_dim / 8;
        const int64_t nz = nv / 2;
        const bool zt = (nv & 1) != 0;
        __m512 zq[(NV ? NV : kMaxHeadVecs) / 2 + 1];
        __m256 yq;
        for (int64_t v = 0; v < nz; ++v) zq[v] = _mm512_loadu_ps(q + v * 16);
        if (zt) yq = _mm256_loadu_ps(q + nz * 16);
#else
        __m256 qv[NV ? NV : kMaxHeadVecs];
        const int64_t nv = NV ? NV : g_head_dim / 8;
        for (int64_t v = 0; v < nv; ++v) qv[v] = _mm256_loadu_ps(q + v * 8);
#endif
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *k = &qkv_p[(b * g_seq + j) * 3 * g_hidden + g_hidden +
                                  h * g_head_dim];
#if defined(__AVX512F__)
          __m512 zs = _mm512_mul_ps(zq[0], _mm512_loadu_ps(k));
          for (int64_t v = 1; v < nz; ++v)
            zs = _mm512_fmadd_ps(zq[v], _mm512_loadu_ps(k + v * 16), zs);
          float acc = _mm512_reduce_add_ps(zs);
          if (zt) acc += hsum256(_mm256_mul_ps(yq,
                                               _mm256_loadu_ps(k + nz * 16)));
          dst[j] = acc;
#elif defined(__AVX2__)
          __m256 s = _mm256_mul_ps(qv[0], _mm256_loadu_ps(k));
          for (int64_t v = 1; v < nv; ++v)
            s = _mm256_fmadd_ps(qv[v], _mm256_loadu_ps(k + v * 8), s);
          dst[j] = hsum256(s);
#else
          float s = 0.f;
          for (int64_t d = 0; d < g_head_dim; ++d) s += q[d] * k[d];
          dst[j] = s;
#endif
        }
      }
    }
  });
}

void BertEncoder::qk(const std::vector<float> &qkv, std::vector<float> &scores) {
  switch (g_head_dim) {
    case 32: qk_impl<4>(qkv, scores); break;
    case 64: qk_impl<8>(qkv, scores); break;
    default: qk_impl<0>(qkv, scores); break;
  }
}

template <int NV>
void BertEncoder::av_impl(const std::vector<float> &scores,
                          const std::vector<float> &qkvbuf,
                          std::vector<float> &ctx) {
  const int64_t pairs = batch * g_heads;
  const float *__restrict sc_p = scores.data();
  const float *__restrict qkv_p = qkvbuf.data();
  float *__restrict ctx_p = ctx.data();
  pool_.run([&](int w, int nw) {
    for (int64_t p = w; p < pairs; p += nw) {
      const int64_t b = p / g_heads, h = p % g_heads;
      for (int64_t i = 0; i < g_seq; ++i) {
        const float *a = &sc_p[(p * g_seq + i) * g_seq];
        float *o = &ctx_p[(b * g_seq + i) * g_hidden + h * g_head_dim];
#if defined(__AVX2__)
#if defined(__AVX512F__)
        const int64_t nv = NV ? NV : g_head_dim / 8;
        const int64_t nz = nv / 2;
        __m512 zacc[(NV ? NV : kMaxHeadVecs) / 2 + 1];
        __m256 yacc{};
        for (int64_t v = 0; v < nz; ++v) zacc[v] = _mm512_setzero_ps();
        const bool tail = (nv & 1) != 0;
        if (tail) yacc = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          const __m512 zaj = _mm512_set1_ps(a[j]);
          for (int64_t k = 0; k < nz; ++k)
            zacc[k] = _mm512_fmadd_ps(zaj, _mm512_loadu_ps(v + k * 16),
                                      zacc[k]);
          if (tail)
            yacc = _mm256_fmadd_ps(_mm256_set1_ps(a[j]),
                                   _mm256_loadu_ps(v + nz * 16), yacc);
        }
        for (int64_t v = 0; v < nz; ++v)
          _mm512_storeu_ps(o + v * 16, zacc[v]);
        if (tail) _mm256_storeu_ps(o + nz * 16, yacc);
#else
        __m256 acc[NV ? NV : kMaxHeadVecs];
        const int64_t nv = NV ? NV : g_head_dim / 8;
        for (int64_t v = 0; v < nv; ++v) acc[v] = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          const __m256 aj = _mm256_set1_ps(a[j]);
          for (int64_t k = 0; k < nv; ++k)
            acc[k] = _mm256_fmadd_ps(aj, _mm256_loadu_ps(v + k * 8), acc[k]);
        }
        for (int64_t v = 0; v < nv; ++v)
          _mm256_storeu_ps(o + v * 8, acc[v]);
#endif
#else
        for (int64_t d = 0; d < g_head_dim; ++d) o[d] = 0.f;
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          for (int64_t d = 0; d < g_head_dim; ++d) o[d] += a[j] * v[d];
        }
#endif
      }
    }
  });
}

void BertEncoder::av(const std::vector<float> &scores,
                     const std::vector<float> &qkv, std::vector<float> &ctx) {
  switch (g_head_dim) {
    case 32: av_impl<4>(scores, qkv, ctx); break;
    case 64: av_impl<8>(scores, qkv, ctx); break;
    default: av_impl<0>(scores, qkv, ctx); break;
  }
}

void BertEncoder::add_into(std::vector<float> &x, const std::vector<float> &y) {
  par(x.size(), [&](size_t lo, size_t hi) {
    size_t i = lo;
#if defined(__AVX2__)
    for (; i + 8 <= hi; i += 8)
      _mm256_storeu_ps(x.data() + i,
                       _mm256_add_ps(_mm256_loadu_ps(y.data() + i),
                                     _mm256_loadu_ps(residual.data() + i)));
#endif
    for (; i < hi; ++i) x[i] = y[i] + residual[i];
  });
}

void BertEncoder::add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
                                   size_t site, const float *ias_next, int8_t *dst,
                                   float *scale_next) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t H = g_hidden;
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float *row = x.data() + r * H;
      const float *yr = y.data() + r * H;
      float *res = residual.data() + r * H;
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= H; j += 8)
        _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
      for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / H;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / H;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, yv);
        _mm256_storeu_ps(res + j, yv);
      }
#else
      double sm = 0.0;
      for (j = 0; j < H; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / H);
      double vs = 0.0;
      for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
      const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
      for (j = 0; j < H; ++j) {
        row[j] = (row[j] - mean) * is * g[j] + b[j];
        res[j] = row[j];
      }
#endif
      if (!dst) continue;
      float mx = 0.f;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= H; j += 8)
          acc = _mm256_max_ps(acc, _mm256_and_ps(
              _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        mx = _mm_cvtss_f32(h);
      }
#endif
      for (; j < H; ++j) {
        const float a = std::fabs(row[j] * ias_next[j]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      scale_next[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      int8_t *q = dst + r * H;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 invv = _mm256_set1_ps(inv);
        const __m256 vhi = _mm256_set1_ps(127.0f);
        const __m256 vlo = _mm256_set1_ps(-127.0f);
        for (; j + 8 <= H; j += 8) {
          __m256 t = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps(row + j),
                            _mm256_loadu_ps(ias_next + j)), invv);
          t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
          __m256i i32 = _mm256_cvtps_epi32(t);
          __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                        _mm256_extracti128_si256(i32, 1));
          _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                           _mm_packs_epi16(p16, p16));
        }
      }
#endif
      for (; j < H; ++j) {
        float t = std::nearbyintf(row[j] * ias_next[j] * inv);
        if (t > 127.f) t = 127.f;
        if (t < -127.f) t = -127.f;
        q[j] = static_cast<int8_t>(t);
      }
    }
  });
  t_hostln += now_s() - t0;
}

void BertEncoder::add_norm_bf16(std::vector<float> &x,
                                 const std::vector<float> &y,
                                 size_t site, uint16_t *dst) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t H = g_hidden;
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float *row = x.data() + r * H;
      const float *yr = y.data() + r * H;
      float *res = residual.data() + r * H;
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= H; j += 8)
        _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
      for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / H;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / H;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, yv);
        _mm256_storeu_ps(res + j, yv);
      }
#else
      double sm = 0.0;
      for (j = 0; j < H; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / H);
      double vs = 0.0;
      for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
      const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
      for (j = 0; j < H; ++j) {
        row[j] = (row[j] - mean) * is * g[j] + b[j];
        res[j] = row[j];
      }
#endif
      if (!dst) continue;
      for (j = 0; j < H; ++j)
        dst[r * H + j] = to_bf16(row[j]);
    }
  });
  t_hostln += now_s() - t0;
}
BertEncoder::~BertEncoder() = default;

// Text-in entry point: tokenize, gather embeddings, run the stack, pool and
// L2-normalise -- the same path EmbedService::chunk walks in the runtime, kept
// here so the abstract Encoder interface is self-contained.
std::vector<float> BertEncoder::encode(const std::vector<std::string> &texts,
                                       const std::string &prefix,
                                       int64_t *tokens) {
  if (!tok_)
    tok_ = std::make_unique<app::AnyTokenizer>(
        app::load_tokenizer(model_, std::string()));
  const float *w_word = model_.raw("embeddings.word").as<float>();
  const float *w_pos = model_.raw("embeddings.position").as<float>();
  const float *w_typ = model_.raw("embeddings.token_type").as<float>();
  const int64_t n = static_cast<int64_t>(texts.size());
  if (batch <= 0)
    throw std::runtime_error(
        "BertEncoder::encode: no batch tier configured -- call use_tier() "
        "during setup before the text-in path");
  std::vector<float> out(static_cast<size_t>(n) * g_hidden, 0.f);
  int64_t base = 0;
  while (base < n) {
    const int64_t take = std::min<int64_t>(batch, n - base);
    const int64_t bt = use_tier(take);
    std::vector<float> buf(static_cast<size_t>(bt) * g_seq * g_hidden, 0.f);
    std::vector<float> cmask(static_cast<size_t>(bt) * g_seq, -1.0e30f);
    std::vector<float> cam(static_cast<size_t>(bt) * g_seq, 0.f);
    int64_t ntok = 0;
    for (int64_t b = 0; b < take; ++b) {
      const std::string &raw = texts[static_cast<size_t>(base + b)];
      const auto en = prefix.empty()
          ? tok_->encode(raw, static_cast<int>(g_seq))
          : tok_->encode(prefix + raw, static_cast<int>(g_seq));
      check_truncation(en.truncated, en.n_tokens_full,
                       static_cast<size_t>(base + b), g_seq);
      ntok += en.n_tokens;
      for (int64_t s = 0; s < g_seq; ++s) {
        const int32_t id = en.input_ids[static_cast<size_t>(s)];
        const float m = static_cast<float>(en.attention_mask[static_cast<size_t>(s)]);
        cam[static_cast<size_t>(b * g_seq + s)] = m;
        cmask[static_cast<size_t>(b * g_seq + s)] = m > 0 ? 0.f : -1.0e30f;
        float *dst = buf.data() + static_cast<size_t>((b * g_seq + s) * g_hidden);
        const float *wv = w_word + static_cast<size_t>(id) * g_hidden;
        const float *pv = w_pos + static_cast<size_t>(s) * g_hidden;
        for (int64_t c = 0; c < g_hidden; ++c)
          dst[c] = wv[c] + pv[c] + w_typ[c];
      }
    }
    add_mask = cmask;
    auto h = run(buf);
    pool_rows(h.data(), cam.data(), take, out.data() + base * g_hidden);
    if (tokens) *tokens += ntok;
    base += take;
  }
  return out;
}

std::vector<float> BertEncoder::run(const std::vector<float> &emb_in) {
  std::vector<float> x = emb_in;
  layer_norm(x, s_ln[0]);

  qkvbuf.resize(rows * 3 * g_hidden);
  ctx.resize(rows * g_hidden);
  proj.resize(rows * g_hidden);
  up.resize(rows * (g_gated_ffn ? 2 : 1) * g_ffn);
  if (g_gated_ffn) gated.resize(rows * g_ffn);
  down.resize(rows * g_hidden);
  scores.resize(batch * g_heads * g_seq * g_seq);

  residual.resize(x.size());
  bool qkv_a_ready = false;
  for (int64_t L = 0; L < g_layers; ++L) {
    if (!qkv_a_ready)
      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

    gemm(qkv_, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden,
         i8w(ws_qkv, L), i8w(as_qkv, L), nullptr, /*a_ready=*/qkv_a_ready);
    qkv_a_ready = false;
    if (g_rope) apply_rope_qkv(qkvbuf);

    double ta = now_s();
    qk(qkvbuf, scores);
    t_attn += now_s() - ta; t_qk += now_s() - ta;

    if (host_sm) {
      softmax_cpu(scores);
    } else {
      add_additive_mask(scores);
      eltwise(softmax_, slots_sm, scores.data(), scores.size());
    }

    ta = now_s();
    av(scores, qkvbuf, ctx);
    t_attn += now_s() - ta; t_av += now_s() - ta;

    gemm(attn_out_, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden,
         i8w(ws_ao, L), i8w(as_ao, L));
    const bool fuse_ln = fuse_ffn_epilogue && host_ln &&
                         ffn_up_.info().a_elem_bytes == 1;
    const bool fuse_ln_bf16 = fuse_ffn_epilogue && host_ln &&
                              ffn_up_.info().a_elem_bytes == 2;
    if (fuse_ln) {
      const float *asf = i8w(as_fu, L);
      if (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
          inv_smooth_src != asf) {
        inv_smooth.resize(static_cast<size_t>(g_hidden));
        for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asf[j];
        inv_smooth_src = asf;
      }
      a_scale.resize(static_cast<size_t>(rows));
      add_norm_quant(x, proj, s_ln[1 + 2 * L] - 1, inv_smooth.data(),
                     static_cast<int8_t *>(ffn_up_.slot_ptr(0, slot_a)),
                     a_scale.data());
    } else if (fuse_ln_bf16) {
      add_norm_bf16(x, proj, s_ln[1 + 2 * L] - 1,
                    static_cast<uint16_t *>(ffn_up_.slot_ptr(0, slot_a)));
    } else {
      add_into(x, proj);
      layer_norm(x, s_ln[1 + 2 * L]);
      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
    }
    FusedNext fn{i8w(as_fd, L),
                 static_cast<int8_t *>(ffn_down_.slot_ptr(0, slot_a)),
                 nullptr, g_gated_ffn};
    const bool fuse_ffn = fuse_ffn_epilogue && host_gelu &&
                          ffn_up_.info().a_elem_bytes == 1 &&
                          ffn_down_.info().a_elem_bytes == 1;
    if (fuse_ffn) {
      a_scale_next.resize(static_cast<size_t>(rows));
      fn.scale = a_scale_next.data();
    }
    FusedNextBf16 fn_bf16{
        static_cast<uint16_t *>(ffn_down_.slot_ptr(0, slot_a)), g_gated_ffn};
    const bool fuse_ffn_bf16 = fuse_ffn_epilogue && host_gelu &&
                               ffn_up_.info().a_elem_bytes == 2 &&
                               ffn_down_.info().a_elem_bytes == 2;
    gemm(ffn_up_, is_fu, x, s_fu[L], b_fu[L], up,
         g_gated_ffn ? 2 * g_ffn : g_ffn, i8w(ws_fu, L), i8w(as_fu, L),
         fuse_ffn ? &fn : nullptr, /*a_ready=*/fuse_ln || fuse_ln_bf16,
         fuse_ffn_bf16 ? &fn_bf16 : nullptr);

    if (fuse_ffn) {
      a_scale.swap(a_scale_next);
      gemm(ffn_down_, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
           g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
    } else if (fuse_ffn_bf16) {
      gemm(ffn_down_, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
           g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
    } else if (g_gated_ffn) {
      swiglu_cpu(up, gated);
      gemm(ffn_down_, is_fd, gated, s_fd[L], b_fd[L], down, g_hidden,
           i8w(ws_fd, L), i8w(as_fd, L));
    } else {
      if (host_gelu)
        gelu_cpu(up);
      else
        eltwise(gelu_, slots_gelu, up.data(), up.size());
      gemm(ffn_down_, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
           i8w(ws_fd, L), i8w(as_fd, L));
    }
    const bool last = (L + 1 == g_layers);
    if (fuse_ffn_epilogue && host_ln && qkv_.info().a_elem_bytes == 1) {
      const float *asq = last ? nullptr : i8w(as_qkv, L + 1);
      if (asq && (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
                  inv_smooth_src != asq)) {
        inv_smooth.resize(static_cast<size_t>(g_hidden));
        for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asq[j];
        inv_smooth_src = asq;
      }
      if (asq) a_scale.resize(static_cast<size_t>(rows));
      add_norm_quant(x, down, s_ln[2 + 2 * L] - 1,
                     asq ? inv_smooth.data() : nullptr,
                     asq ? static_cast<int8_t *>(qkv_.slot_ptr(0, slot_a))
                         : nullptr,
                     asq ? a_scale.data() : nullptr);
      qkv_a_ready = !last;
    } else if (fuse_ffn_epilogue && host_ln && qkv_.info().a_elem_bytes == 2) {
      add_norm_bf16(x, down, s_ln[2 + 2 * L] - 1,
                    last ? nullptr
                         : static_cast<uint16_t *>(qkv_.slot_ptr(0, slot_a)));
      qkv_a_ready = !last;
    } else {
      add_into(x, down);
      layer_norm(x, s_ln[2 + 2 * L]);
    }
  }
  return x;
}

}  // namespace npue
