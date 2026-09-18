//===- gemma_npu_encoder.hpp ------------------*- C++ -*-===//
//
// EmbeddingGemma-300M on the array (arch=1): MQA, per-layer RoPE, four
// RMSNorms, GeGLU. A separate encoder from BERT on purpose -- it reuses the
// same machinery (Pool, bf16 conversions, npu::Design staging), not the same
// forward pass. Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "gemma_kernels.hpp"
#include "host_kernels.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "pool.hpp"
#include "tokenizer.hpp"
#include "tokenizer_gemma.hpp"

namespace app {

// ===========================================================================
// EmbeddingGemma-300M on the array (arch=1) -- tasks/0074.
// ===========================================================================
//
// WHY A SECOND ENCODER AND NOT A BRANCH IN Encoder::run().
//
// arch=2 (nomic) could be a branch because it IS a BERT block with two things
// swapped: RoPE instead of an absolute position table, and a gated FFN instead
// of a plain one. Gemma is not. Its block is a four-RMSNorm sandwich
// (pre-norm AND post-norm around both sub-layers), its attention carries
// q_norm/k_norm between the projection and RoPE, its RoPE base changes per
// layer, it scales the embedding by sqrt(hidden), and it ends in two post-pool
// Dense heads. Threading all of that through a function that serves five
// shipped models as `if (arch == ...)` would put the shipped models one typo
// away from a silent wrong answer, which is the failure this project keeps
// finding. So this is separate code that happens to reuse the same MACHINERY:
// Pool, the bf16 conversions, npu::Design staging, and the same
// bind/sync/dispatch/bias sequence Encoder::gemm() uses.
//
// The host-only npue::GemmaEncoder (tasks/0064, verified to 1-cos 5.496e-13
// against reference/encoder_gemma.py) is NOT replaced by this. It is the
// discriminating control: the same container geometry, the same tokenizer,
// every GEMM in double precision on the CPU. Any disagreement between the two
// beyond the bf16 floor is a bug in THIS file.
//
// WHAT RUNS WHERE. Four GEMMs per layer go to the array -- 97.7% of the
// model's MACs (tasks/0074 sec 4). Attention's QK^T and A.V stay on the host:
// at head_dim 256 and seq 64 their N is 64, which fails the design's
// `N % (n * n_aie_cols) == 0` outright, and F3 prices the whole of attention
// at 2.3% of MACs here. RMSNorm, RoPE and GeGLU stay on the host on
// tasks/0032's measured precedent that a host eltwise pass beats an NPU
// dispatch at these widths.
struct GemmaNpuEncoder {
  npue::File &model;
  npu::Design &d;
  Pool *pool = nullptr;
  npue::GemmaTokenizer tok;

  // Geometry, read from the container. Nothing here is a literal: this file
  // has no idea that hidden is 768 or that there are 24 layers.
  int64_t hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0,
          layers = 0, dense_hidden = 0, qkv_n = 0, swp = 6;
  int64_t q_off = 0, k_off = 0, v_off = 0, kv_w = 0;
  double eps = 1e-6, rope_theta = 0.0, rope_theta_local = 0.0,
         attn_scale = 1.0;

  int64_t seq = 0, batch = 0, rows = 0;
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  size_t slot_a = 0, slot_c = 0;
  // Lanes (tasks/0033's mechanism, applied to this arch): several encoders
  // share the ONE design, each owning its A and C slots, with every NPU
  // interaction under one mutex. The array serialises dispatches anyway, so
  // the lock only makes explicit what the hardware enforces -- what overlaps
  // is one lane's HOST work with another's array work. It matters more here
  // than on any BERT model: the first measurement of this path put the array
  // at 48% of wall clock and the host at 47%, which is as close to the ideal
  // case for overlap as this project has met.
  std::mutex *npu_mu = nullptr;

  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 (tasks/0081). Per-output-channel weight scales and per-input-channel
  // SmoothQuant divisors, one pointer per layer per op, filled by stage_all()
  // and empty on a bf16 container.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  // Per-row activation scales for the current GEMM, [rows].
  std::vector<float> a_scale;
  // 1/asmooth for the GEMM being run, rebuilt only when the pointer changes --
  // K floats of setup against 2*rows*K divisions (tasks/0080).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;
  std::vector<float> a_scale_next, inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  bool fuse_ffn_epilogue = true;

  struct LayerHost {
    const float *q_norm, *k_norm, *ln_in, *ln_pa, *ln_pf, *ln_pof;
  };
  std::vector<LayerHost> lh;
  const float *w_embed = nullptr, *w_norm = nullptr;
  const float *w_dense2 = nullptr, *w_dense3 = nullptr;

  // Scratch, sized once per batch and reused across layers and calls.
  std::vector<float> x, hbuf, qkvbuf, ctx, proj, upbuf, gatedbuf, down,
      scores, add_mask, cos_g, sin_g, cos_l, sin_l;
  std::vector<int32_t> ids;
  std::vector<uint8_t> mask;

  double t_conv = 0, t_in = 0, t_disp = 0, t_out = 0, t_bias = 0,
         t_norm = 0, t_attn = 0, t_rope = 0, t_geglu = 0, t_tok = 0;
  int n_dispatch = 0;
  void reset_timers() {
    t_conv = t_in = t_disp = t_out = t_bias = 0;
    t_norm = t_attn = t_rope = t_geglu = t_tok = 0;
    n_dispatch = 0;
  }

  GemmaNpuEncoder(npue::File &m, npu::Design &design, Pool &p)
      : model(m), d(design), pool(&p) {
    const std::string arch = m.config_string("arch");
    if (arch != "gemma3_mqa_rope_geglu")
      throw std::runtime_error("GemmaNpuEncoder given arch '" + arch + "'");
    // The container must say it holds PRE-TILED operands. A host-only
    // container carries the same tensor VALUES in the same file under
    // different names and a different layout; reading one as the other is
    // tasks/0022's rel_fro 1.186 all over again.
    const std::string layout = m.config_string("gemm_layout");
    if (layout != "pretiled_bf16")
      throw std::runtime_error(
          "this container's gemm_layout is '" + layout + "', not "
          "'pretiled_bf16' -- it holds host-side row-major F32 operands and "
          "has no tiled weights for the array. Repack it with "
          "tools/pack_npue.py (the NPU layout is now the default).");

    hidden = m.config_int("hidden");
    heads = m.config_int("num_heads");
    kv_heads = m.config_int("num_key_value_heads");
    head_dim = m.config_int("head_dim");
    inter = m.config_int("intermediate");
    layers = m.config_int("num_layers");
    dense_hidden = m.config_int("dense_hidden");
    swp = m.config_int("sliding_window_pattern");
    eps = m.config_double("rms_norm_eps");
    rope_theta = m.config_double("rope_theta");
    rope_theta_local = m.config_double("rope_local_base_freq");
    attn_scale = std::pow(m.config_double("query_pre_attn_scalar"), -0.5);
    qkv_n = m.config_int("qkv_n");
    kv_w = kv_heads * head_dim;

    if (kv_heads != 1)
      throw std::runtime_error(
          "this encoder's attention loop assumes num_key_value_heads == 1 "
          "(EmbeddingGemma-300M); a GQA checkpoint needs the K/V reuse "
          "generalised first");
    if (head_dim * heads != hidden)
      throw std::runtime_error("head_dim * num_heads != hidden");
    if (head_dim % 2)
      throw std::runtime_error("odd head_dim -- RoPE cannot half-split it");
    if (m.config_string("geglu_halves") != "gate|up")
      throw std::runtime_error(
          "unrecognised geglu_halves '" + m.config_string("geglu_halves") +
          "' -- expected 'gate|up'; refusing rather than guessing which half "
          "of the fused ffn_up gets the GELU. tasks/0068 Q2 measured the "
          "swapped variant on the sibling architecture at rel_fro 4.022e+00.");

    // Q/K/V offsets are READ, never derived. `3*hidden` is the BERT answer and
    // it is wrong here twice over: MQA makes K and V narrower than Q, and the
    // operand is zero-padded past them (tasks/0074).
    q_off = 0;
    k_off = hidden;
    v_off = hidden + kv_w;
    if (qkv_n < v_off + kv_w)
      throw std::runtime_error("qkv_n is too small to hold Q|K|V");

    w_embed = m.raw("embed_tokens.weight").as<float>();
    w_norm = m.raw("norm.weight").as<float>();
    w_dense2 = m.raw("dense2.weight").as<float>();
    w_dense3 = m.raw("dense3.weight").as<float>();
    lh.resize(static_cast<size_t>(layers));
    for (int64_t L = 0; L < layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      LayerHost &l = lh[static_cast<size_t>(L)];
      l.q_norm = m.raw(p + "q_norm.weight").as<float>();
      l.k_norm = m.raw(p + "k_norm.weight").as<float>();
      l.ln_in = m.raw(p + "input_layernorm.weight").as<float>();
      l.ln_pa = m.raw(p + "post_attention_layernorm.weight").as<float>();
      l.ln_pf = m.raw(p + "pre_feedforward_layernorm.weight").as<float>();
      l.ln_pof = m.raw(p + "post_feedforward_layernorm.weight").as<float>();
    }
    auto tv = m.raw("tokenizer.gemma_table");
    tok = npue::GemmaTokenizer::from_table_bytes(
        reinterpret_cast<const char *>(tv.data), tv.bytes);
  }

  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) {
      f(size_t(0), n);
      return;
    }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }
  // Row-parallel, for the strided passes where a flat byte range would split
  // a row.
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  double lap(double t0, double &bucket) {
    const double t = now_s();
    bucket += t - t0;
    return t;
  }

  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = d.info().a_elem_bytes == 1;
    auto one = [&](const std::string &name, std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc,
                   std::vector<const float *> *asm_) {
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty() || got.empty() || want != got)
        throw std::runtime_error(
            name + ": B layout mismatch -- design wants " +
            (want.empty() ? std::string("(nothing stated)") : want.substr(0, 16)) +
            ", container has " +
            (got.empty() ? std::string("(nothing stated)") : got.substr(0, 16)) +
            ". The bytes would be the right size and the wrong order.");
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      // Gemma has no biases anywhere; the packer zero-fills them so this
      // dispatch path stays byte-for-byte the BERT one (tasks/0074 sec 5).
      bias.push_back(model.raw(name + ".bias").as<float>());
      if (i8) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER passes the layout check only if the container is also
        // int8, but a container packed by an older packer would carry i8
        // bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
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

  int64_t use_tier(int64_t want) {
    if (tiers.empty()) return batch;
    size_t pick = tiers.size() - 1;
    for (size_t i = 0; i < tiers.size(); ++i)
      if (tiers[i] >= want) { pick = i; break; }
    batch = tiers[pick];
    rows = batch * seq;
    is_qkv = tier_slots[pick][0];
    is_ao = tier_slots[pick][1];
    is_fu = tier_slots[pick][2];
    is_fd = tier_slots[pick][3];
    return batch;
  }

  // Identical in shape to Encoder::gemm() -- convert A to bf16 into this
  // pipeline's own slot, bind, sync, dispatch, then read C back with a
  // streaming load and add the (zero) bias.
  // The scale vectors are empty on a bf16 container, so this yields nullptr
  // and gemm()'s own check decides whether that is legal for the design.
  static const float *at(const std::vector<const float *> &v, int64_t L) {
    return L < static_cast<int64_t>(v.size()) ? v[static_cast<size_t>(L)]
                                              : nullptr;
  }

  // T37 (tasks/0082): as Encoder::FusedNext, for arch=1's GeGLU. `gated` is
  // always true here -- Gemma has no un-gated FFN.
  struct FusedNext {
    const float *asmooth;
    int8_t *dst;
    float *scale;
  };

  // T37-BF16 (tasks/0108): as Encoder::FusedNextBf16, for arch=1's GeGLU.
  struct FusedNextBf16 {
    uint16_t *dst;
  };

  // T37-BF16 (tasks/0108): as Encoder::dequant_act_bf16, for arch=1's GeGLU
  // (gelu_pytorch_tanh(gate) * up, NOT the same function as BERT's GELU --
  // see geglu()'s own comment). Bit-identical to the unfused
  // gemm(...)+geglu() pair by construction: the C-readback+bias arm is
  // copied verbatim from this gemm()'s own unfused branches below, and the
  // activation arm is copied verbatim from the int8 fused epilogue's lambda
  // above (itself bit-identical to geglu(), including the scalar tail's
  // double-precision rounding).
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, const float *bias, uint16_t *dst) {
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
        // GeGLU, in place, narrowing to out_n -- COPIED VERBATIM from the
        // int8 fused epilogue's lambda / geglu() above.
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
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
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

  void gemm(size_t islot, const float *a, size_t a_len, size_t wslot,
            const float *bias, std::vector<float> &out, int64_t N,
            const float *wscale = nullptr, const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or it was packed before tools/pack_npue.py "
          "--int8 supported arch=1");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A is already in the device slot, written by the previous GEMM's fused
      // epilogue, and a_scale already holds its row scales.
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
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue.
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
      // FUSED GeGLU EPILOGUE (T37): dequantise, gate, and quantise ffn_down's
      // operand in one L1-resident pass. `out` is never written.
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
            // Identical intrinsics to geglu(), including its +-15 clamp before
            // exp2 -- a scalar rewrite of the same algebra is a different
            // number (tasks/0082 section 2).
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
              __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
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
      // Same rank-1 dequantisation the BERT encoder uses, from the same
      // helper -- the transport width comes from the design, so a narrowed-C
      // int8 set (tasks/0080) needs nothing extra here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
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
          // Streaming loads: C is an XRT write-combined host bo and ordinary
          // loads from it stall per line (tasks/0024). N is a multiple of 48
          // here, so the 8-wide tail is handled by the scalar loop.
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

  // Gemma3RMSNorm over `dim` contiguous elements per row, with independent
  // input and output row strides so it can work on a slice of the fused qkv
  // buffer in place. `out = x * rsqrt(mean(x^2) + eps) * (1 + w)` -- the
  // `1 +` is Gemma's and omitting it is the single easiest way to produce a
  // plausible wrong answer here (gemma_kernels.hpp's own warning).
  //
  // fp32 accumulation, not the double reduction npue::rms_norm_cpu uses. That
  // function exists to match reference/encoder_gemma.py bit-for-bit and is
  // still what the host-only control runs; this path already rounds every
  // GEMM through bf16, so a double-precision reduction here would buy nothing
  // measurable and costs a factor on 96 norm sites per encode. The end-to-end
  // check against the control is what decides whether that is true.
  void rms_norm(const float *xin, int64_t in_stride, float *xout,
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

  // RoPE on Q (every head) and K (the single KV head) in place, inside the
  // fused qkv buffer. Position is `row % seq`, which holds because rows are
  // laid out [batch][seq]. NeoX convention (concat(freqs,freqs), rotate-half),
  // matching gemma_rope_tables().
  void apply_rope(std::vector<float> &qkv, const float *cs_t, const float *sn_t) {
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
        const int64_t s = r % seq;
        const float *cs = cs_t + s * head_dim;
        const float *sn = sn_t + s * head_dim;
        float *base = p + r * qkv_n;
        for (int64_t hh = 0; hh < heads; ++hh)
          rot(base + q_off + hh * head_dim, cs, sn);
        rot(base + k_off, cs, sn);           // kv_heads == 1
      }
    });
    t_rope += now_s() - t0;
  }

  // out = gelu_pytorch_tanh(gate) * up, over the fused [gate | up] ffn_up
  // buffer. Gemma's activation is the tanh approximation, NOT the exact-erf
  // GELU the BERT path uses -- a different function, kept deliberately
  // separate (gemma_kernels.hpp).
  void geglu(const std::vector<float> &fused, std::vector<float> &out) {
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
        const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);  // sqrt(2/pi)
        const __m256 c_k = _mm256_set1_ps(0.044715f);
        // 2*log2(e): tanh(y) = (2^(2y*log2e) - 1) / (2^(2y*log2e) + 1)
        const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
        // Clamp before exp2 so a large activation saturates tanh instead of
        // overflowing to inf and producing (inf-1)/(inf+1) = NaN. |y| >= 15
        // is tanh = +-1 to well inside fp32 already.
        const __m256 c_lim = _mm256_set1_ps(15.0f);
        for (; j + 8 <= inter; j += 8) {
          __m256 xv = _mm256_loadu_ps(g + j);
          __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
          __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
          y = _mm256_min_ps(_mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                            c_lim);
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
          __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                    _mm256_add_ps(e, c_one));
          __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                     _mm256_add_ps(c_one, th));
          _mm256_storeu_ps(o + j, _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
        }
#endif
        // Scalar tail, in the reference's own two-stage rounding (round `act`
        // to fp32, THEN promote and multiply by `up`). It never runs at this
        // model's intermediate width -- 1152 is a multiple of 8 -- and is kept
        // matching the reference rather than matching the vector body above,
        // so a future width with a tail lands on the more accurate form.
        for (; j < inter; ++j) {
          const double xv = g[j];
          const double y = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
          const float act = static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
          o[j] = static_cast<float>(static_cast<double>(act) *
                                    static_cast<double>(u[j]));
        }
      }
    });
    t_geglu += now_s() - t0;
  }

  // MQA attention on the host. Every one of the `heads` query heads attends to
  // the SAME single K/V head -- mathematically identical to repeat_kv() but
  // without materialising the repeat.
  void attention(const std::vector<float> &qkv, std::vector<float> &out) {
    const double t0 = now_s();
    const int64_t pairs = batch * heads;
    par_rows(pairs, [&](int64_t p0, int64_t p1) {
      std::vector<float> row(static_cast<size_t>(seq));
      for (int64_t pi = p0; pi < p1; ++pi) {
        const int64_t b = pi / heads, hh = pi % heads;
        const float *base = qkv.data() + b * seq * qkv_n;
        const float *mk = add_mask.data() + b * seq;
        for (int64_t i = 0; i < seq; ++i) {
          const float *qi = base + i * qkv_n + q_off + hh * head_dim;
          float mx = -3.4e38f;
          for (int64_t j = 0; j < seq; ++j) {
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
          for (int64_t j = 0; j < seq; ++j) {
            const float e = std::exp(row[static_cast<size_t>(j)] - mx);
            row[static_cast<size_t>(j)] = e;
            sum += e;
          }
          const float inv = 1.0f / sum;
          float *o = out.data() + (b * seq + i) * hidden + hh * head_dim;
          std::memset(o, 0, sizeof(float) * static_cast<size_t>(head_dim));
          for (int64_t j = 0; j < seq; ++j) {
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

  // A plain threaded fp32 host GEMM, for the two post-pool Dense heads only.
  // They run once per SEQUENCE, not once per token (tasks/0074 sec 4).
  void gemm_host(const float *a, int64_t M, int64_t K, const float *b,
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

  void ensure_tables() {
    if (!cos_g.empty()) return;
    cos_g.resize(static_cast<size_t>(seq * head_dim));
    sin_g.resize(cos_g.size());
    cos_l.resize(cos_g.size());
    sin_l.resize(cos_g.size());
    npue::gemma_rope_tables(seq, head_dim, rope_theta, cos_g.data(), sin_g.data());
    npue::gemma_rope_tables(seq, head_dim, rope_theta_local, cos_l.data(),
                            sin_l.data());
  }

  // Encode `texts` (padded/tiled by the caller to exactly `batch` entries).
  // Returns [batch][hidden], L2-normalized.
  // `index_base` and `n_real` exist only so a truncation error can name the
  // CALLER's input. This function is handed a group that the caller has
  // padded up to the tier by repeating its last real text, so rows at
  // `b >= n_real` are duplicates whose index does not exist upstream --
  // checking them would report a row number the caller cannot look up. A
  // duplicate that truncates is a copy of a real row that also truncates, and
  // the real one is checked first, so nothing escapes by being skipped here.
  // `tokens`, when given, accumulates the REAL texts' token counts -- the same
  // thing the BERT path's chunk() reports and the same field `usage.
  // prompt_tokens` needs (tasks/0115). Padding repeats are excluded, which is
  // what `n_real` already distinguishes for the truncation check.
  std::vector<float> encode_batch(const std::vector<std::string> &texts,
                                  const std::string &prefix,
                                  size_t index_base = 0,
                                  size_t n_real = static_cast<size_t>(-1),
                                  int64_t *tokens = nullptr) {
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
          tok.encode(texts[static_cast<size_t>(b)], static_cast<int>(seq), prefix);
      if (static_cast<size_t>(b) < n_real) {
        check_truncation(en.truncated, en.n_tokens_full,
                         index_base + static_cast<size_t>(b), seq);
        if (tokens) *tokens += en.n_tokens;
      }
      for (int64_t s = 0; s < seq; ++s) {
        ids[static_cast<size_t>(b * seq + s)] = en.input_ids[static_cast<size_t>(s)];
        mask[static_cast<size_t>(b * seq + s)] =
            static_cast<uint8_t>(en.attention_mask[static_cast<size_t>(s)]);
      }
    }
    t_tok += now_s() - t0;

    x.assign(static_cast<size_t>(rows * hidden), 0.f);
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

    // embed: x = W[id] * sqrt(hidden)
    const float escale = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *wv = w_embed + static_cast<size_t>(ids[static_cast<size_t>(r)]) * hidden;
        float *dst = x.data() + r * hidden;
        for (int64_t c = 0; c < hidden; ++c) dst[c] = wv[c] * escale;
      }
    });

    for (int64_t L = 0; L < layers; ++L) {
      const LayerHost &l = lh[static_cast<size_t>(L)];
      const bool full = npue::gemma_is_full_attention_layer(L, swp);
      const float *cs_t = full ? cos_g.data() : cos_l.data();
      const float *sn_t = full ? sin_g.data() : sin_l.data();

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_in);
      gemm(is_qkv, hbuf.data(), hbuf.size(), s_qkv[L], b_qkv[L], qkvbuf,
           qkv_n, at(ws_qkv, L), at(as_qkv, L));

      // q_norm / k_norm: RMSNorm over head_dim, PER HEAD, strictly between the
      // projection and RoPE. Each (row, head) slice of head_dim floats is
      // contiguous inside the fused buffer even though consecutive rows are
      // qkv_n apart, so a strided call needs no repacking.
      for (int64_t hh = 0; hh < heads; ++hh)
        rms_norm(qkvbuf.data() + q_off + hh * head_dim, qkv_n,
                 qkvbuf.data() + q_off + hh * head_dim, qkv_n, rows, head_dim,
                 l.q_norm);
      rms_norm(qkvbuf.data() + k_off, qkv_n, qkvbuf.data() + k_off, qkv_n,
               rows, head_dim, l.k_norm);

      apply_rope(qkvbuf, cs_t, sn_t);
      attention(qkvbuf, ctx);

      gemm(is_ao, ctx.data(), ctx.size(), s_ao[L], b_ao[L], proj,
           hidden, at(ws_ao, L), at(as_ao, L));
      rms_norm(proj.data(), hidden, proj.data(), hidden, rows, hidden, l.ln_pa);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += proj[i];
      });

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_pf);
      const bool fuse_ffn = fuse_ffn_epilogue &&
                            d.info().a_elem_bytes == 1 && at(as_fd, L);
      FusedNext fn{at(as_fd, L), static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                   nullptr};
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- no quantisation scale on this path, so no `at(as_fd, L)`
      // gate is needed.
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
           hidden, at(ws_fd, L), at(as_fd, L), nullptr,
           /*a_ready=*/fuse_ffn || fuse_ffn_bf16);
      rms_norm(down.data(), hidden, down.data(), hidden, rows, hidden, l.ln_pof);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += down[i];
      });
    }

    rms_norm(x.data(), hidden, x.data(), hidden, rows, hidden, w_norm);

    // masked mean pool, include_prompt=true
    std::vector<float> pooled(static_cast<size_t>(batch * hidden), 0.f);
    for (int64_t b = 0; b < batch; ++b) {
      double denom = 0.0;
      float *o = pooled.data() + b * hidden;
      for (int64_t s = 0; s < seq; ++s) {
        if (!mask[static_cast<size_t>(b * seq + s)]) continue;
        denom += 1.0;
        const float *row = x.data() + (b * seq + s) * hidden;
        for (int64_t c = 0; c < hidden; ++c) o[c] += row[c];
      }
      const float inv = static_cast<float>(1.0 / std::max(denom, 1e-9));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }

    std::vector<float> d2(static_cast<size_t>(batch * dense_hidden));
    gemm_host(pooled.data(), batch, hidden, w_dense2, dense_hidden, d2.data());
    std::vector<float> out(static_cast<size_t>(batch * hidden));
    gemm_host(d2.data(), batch, dense_hidden, w_dense3, hidden, out.data());

    for (int64_t b = 0; b < batch; ++b) {
      float *o = out.data() + b * hidden;
      double nrm = 0.0;
      for (int64_t c = 0; c < hidden; ++c) nrm += static_cast<double>(o[c]) * o[c];
      const float inv = static_cast<float>(1.0 / std::max(std::sqrt(nrm), 1e-12));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }
    return out;
  }
};

}  // namespace app
