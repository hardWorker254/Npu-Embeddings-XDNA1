//===- bert_encoder.hpp --------------------------------------*- C++ -*-===//
//
// The BERT-family NPU encoder (absolute positions, post-LN; arch=0, and
// arch=2/3 via the data-driven RoPE/gated-FFN switches in the container
// config). Split out of main.cpp verbatim.
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
#include "host_kernels.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "pool.hpp"

namespace app {

// The whole encoder. Designs are constructed once by the caller and reused --
// that is the point of the exercise.
struct Encoder {
  npue::File &model;
  npu::Design &qkv, &attn_out, &ffn_up, &ffn_down, &gelu, &layernorm, &softmax;


  // Staged once: the mask, in the form softmax consumes.
  std::vector<float> add_mask;   // [batch, g_seq]

  // Unified gemm_rtp mode (tasks/0032): all four GEMM refs above point at ONE
  // design; each op is an instruction-stream slot bound before dispatch. The
  // slot order is the export contract of tools/export_gemm_rtp.py:
  // qkv=0 (insts.bin), attn_out=1, ffn_up=2, ffn_down=3 (load_instr order).
  bool unified = false;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;

  // Batch tiers (0037): one xclbin carries a stream per (op, batch), so the
  // encoder can size a request instead of padding it to the largest design.
  // `tiers` is ascending; `use_tier` picks the smallest that fits and points
  // is_* at its slots.
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;   // qkv, attn_out, ffn_up, ffn_down

  int64_t use_tier(int64_t want) {
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

  // Two-encode pipelining (tasks/0033): two Encoder instances share the ONE
  // unified design; each owns its A and C slots, and every NPU interaction
  // (bind + sync + dispatch) happens under this mutex. The array serializes
  // dispatches anyway (note 0004) -- the lock only makes explicit what the
  // hardware enforces -- while each pipeline's HOST work overlaps the other
  // pipeline's NPU work.
  std::mutex *npu_mu = nullptr;
  size_t slot_a = 0, slot_c = 0;

  int64_t batch = 0, rows = 0;   // rows = batch * g_seq, from the design's M
  Pool *pool = nullptr;

  // Chunk a flat range over the pool. Chunks are 64-element aligned so the AVX2
  // conversions never see a split vector and every worker takes the fast path.
  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) { f(size_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Row-parallel, for passes where a flat byte range would split a row --
  // int8 quantisation takes a per-row maximum, so a chunk boundary inside a
  // row would give two different scales to one row's halves (tasks/0078).
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Scratch reused across layers. `residual = x` used to allocate and copy
  // 12.6 MB per layer at batch 128 -- 75 MB per encode of pure copying.
  std::vector<float> residual;
  // Scratch, sized once. These used to be six fresh vectors per run() -- ~90
  // MB of allocate-and-touch per encode at batch 128, and ~280 MB at
  // bge-large's width. resize() after the first call is a no-op.
  std::vector<float> qkvbuf, ctx, proj, up, down, scores;
  // arch=2 only: swiglu_cpu()'s output, [rows][g_ffn] -- a separate buffer
  // from `up` because an in-place version of this compaction is NOT safe
  // under threading (see swiglu_cpu's own comment). Empty/unused for every
  // arch=0 container.
  std::vector<float> gated;

  // arch=2 RoPE tables, [g_seq, g_head_dim] each, built ONCE per Encoder (not
  // per layer, not per call) the first time apply_rope_qkv() runs. Every
  // layer shares the same table -- unlike Gemma, nomic has a single rope_theta
  // for the whole model, not a per-layer local/global split (gemma_kernels.hpp
  // trap 2b). Using npue::gemma_rope_tables() here even though nothing about
  // it is Gemma-specific: it is plain NeoX RoPE table construction, and
  // duplicating it for a second arch would be the actual mistake.
  std::vector<float> rope_cos, rope_sin;
  bool rope_ready = false;

  // Device-resident weights, one slot per layer per design, plus the bias
  // pointers straight into the mapped file. Filled by stage_all().
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 only (tasks/0078): per-output-channel weight scales and the
  // per-input-channel SmoothQuant divisor, straight out of the mapped .npue.
  // Empty for every bf16 container, and the int8 path is selected by the
  // DESIGN's a_dtype, so a mismatch is caught by stage_all() rather than by
  // dereferencing an empty vector.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  std::vector<size_t> s_ln;      // 0 = embeddings, then ln1/ln2 per layer
  // Host-side views of the same parameters, straight into the mapped .npue.
  // tasks/0031 measured a LayerNorm call at 725 us of kernel inside ~3 ms of
  // switch+conversion; a threaded fp32 AVX2 LayerNorm on the host costs
  // ~0.5 ms and removes 13 design switches outright. --host-ln selects it.
  std::vector<const float *> h_gamma, h_beta;
  bool host_ln = false;
  bool host_sm = false;
  bool host_gelu = false;
  // Measure-before-build (tasks/0080). Under int8 the GEMM is bandwidth-bound
  // again (0010's model, which 0048 superseded for bf16), and C is 61% of the
  // traffic on three of four shapes -- so narrowing C is the top lever. Doing
  // it needs NO extra core input, because int32 -> bf16 is a pure format
  // conversion: the host still applies sa[i]*wscale[j]. What it costs is
  // accuracy, and that is knowable without writing a kernel. This flag rounds
  // the accumulator exactly as such a design would (int32 -> fp32 -> bf16,
  // round-to-nearest-even, i.e. conv_even per trap 2b) and changes nothing
  // else, so the 1-cos gate prices the design before it exists.
  bool sim_c_bf16 = false;
  // T37 (tasks/0081): fuse the ffn_up epilogue -- dequantise, GELU and the
  // next GEMM's quantisation in ONE pass, so the widest tensor in the model is
  // never materialised in fp32. On by default where it applies; --no-fuse-ffn
  // turns it off, which is how the two paths are compared.
  // T37-BF16 (tasks/0108): the same flag also gates the bf16/bfp16 analogue
  // (narrow-to-bf16 instead of quantise, no scale) -- one flag, two datapaths,
  // so --no-fuse-ffn A/Bs whichever one the loaded container actually uses.
  bool fuse_ffn_epilogue = true;
  double t_hostln = 0.0, t_hostsm = 0.0, t_hostgelu = 0.0;


  // Where the time goes. A single number for the whole encode says "slow";
  // this says which half to fix.
  double t_npu = 0.0;      // memcpy + sync + dispatch, i.e. everything a
                           // fused design would subsume
  double t_attn = 0.0;     // the per-head QK^T and A.V loops on the host
  // Split, because they are different problems: QK^T ends in a horizontal
  // reduction and holds Q in registers (compute), A.V accumulates per output
  // element and streams scores + V (memory). tasks/0086 widened A.V to 512
  // bits for exactly zero, which only makes sense if it is the memory half --
  // and that is a claim this counter can check instead of infer.
  double t_qk = 0.0, t_av = 0.0;
  int n_dispatch = 0;

  // Splitting t_npu further, because removing 21 MB of memcpy and vectorising
  // 13.8 M conversions bought only 9%: the cost is not where it was assumed to
  // be, and one aggregate number cannot say where it is instead (tasks/0024).
  double t_conv = 0.0;     // fp32 <-> bf16 both directions
  double t_in = 0.0;       // sync_to_device
  double t_disp = 0.0;     // kernel(...) + wait
  double t_out = 0.0;      // sync_from_device
  double t_bias = 0.0;     // reading the result buffer, adding bias

  void reset_timers() {
    // t_qk/t_av belong here too. Left out, they accumulated the warm-up encode
    // as well as the benched ones and summed to 1.5x their own parent t_attn --
    // a new counter that is not reset is a counter measuring a different window
    // from everything printed beside it (tasks/0086).
    t_qk = t_av = 0.0;
    t_npu = t_attn = 0.0;
    t_hostln = t_hostsm = t_hostgelu = 0.0;
    t_conv = t_in = t_disp = t_out = t_bias = 0.0;
    n_dispatch = 0;
  }

  // Move every weight onto the device once. Returns the bytes staged, so the
  // banner can state the cost of the trade rather than hiding it.
  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = qkv.info().a_elem_bytes == 1;
    auto one = [&](npu::Design &d, const std::string &name,
                   std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc = nullptr,
                   std::vector<const float *> *asm_ = nullptr) {
      // The design says what layout it needs; the .npue says what it holds.
      // Refuse unless both spoke AND they agree.
      //
      // tasks/0022 shipped pre-tiled weights into a row-major design: right
      // sizes, wrong order, rel_fro 1.186 -- "a buffer-size check catches a
      // wrong size, never a wrong layout". The layout hash that catches it has
      // been in the file since M4 and was never read on this side.
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty())
        throw std::runtime_error(d.info().name + "/design.json has no "
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
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      bias.push_back(model.raw(name + ".bias").as<float>());
      bytes += w.bytes;
      if (i8 && wsc) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER would otherwise fail here with a bare "no such tensor",
        // and the layout-hash check above has already told the user which
        // half is wrong -- so this only fires if a container carries tiled
        // int8 bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
      }
    };
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(qkv, p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
      one(attn_out, p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
      one(ffn_up, p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
      one(ffn_down, p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
    }

    // gamma and beta share one buffer: a core tile has two input DMA channels
    // and the activations need one of them (tasks/0020).
    std::vector<float> gb(2 * g_hidden);
    auto ln_one = [&](const std::string &g, const std::string &b) {
      std::memcpy(gb.data(), model.raw(g).data, g_hidden * sizeof(float));
      std::memcpy(gb.data() + g_hidden, model.raw(b).data,
                  g_hidden * sizeof(float));
      s_ln.push_back(layernorm.stage(1, gb.data(), gb.size() * sizeof(float)));
      bytes += gb.size() * sizeof(float);
      h_gamma.push_back(model.raw(g).as<float>());
      h_beta.push_back(model.raw(b).as<float>());
    };
    if (host_ln) {
      // No device staging: only the host pointers and the site numbering.
      auto ln_host = [&](const std::string &g, const std::string &b) {
        s_ln.push_back(s_ln.size() + 1);
        h_gamma.push_back(model.raw(g).as<float>());
        h_beta.push_back(model.raw(b).as<float>());
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

  // `lap` charges the elapsed time to a bucket and returns the new mark, so
  // each stage is attributed without a timer call being able to drift.
  double lap(double t0, double &bucket) {
    double t = now_s();
    bucket += t - t0;
    return t;
  }

  // bf16 in, bf16 out, one input buffer -- GELU and softmax.
  void eltwise(npu::Design &d, float *x, size_t n) {
    double t0 = now_s();
    par(n, [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(d.host_ptr(0)) + lo, x + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    d.sync_to_device(0);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(1);
    t0 = lap(t0, t_out);
    par(n, [&](size_t lo, size_t hi) {
      bf16_read(x + lo, static_cast<const uint16_t *>(d.host_ptr(1)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // fp32 two-pass LayerNorm on the host, parallelized over rows: the same
  // two-pass mean/variance formula as the NPU kernel and the M3 oracle, in
  // fp32 throughout -- if anything MORE accurate than the bf16 round trip it
  // replaces. The golden check decides.
  void layer_norm_cpu(std::vector<float> &x, size_t site) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t n_rows = static_cast<int64_t>(x.size()) / g_hidden;
    pool->run([&](int w, int nw) {
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

#if defined(__AVX2__)
  static inline __m256 exp2_avx2(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(1.5483275463e-05f);
    const __m256 c1 = _mm256_set1_ps(1.5669833174e-04f);
    const __m256 c2 = _mm256_set1_ps(1.3331825236e-03f);
    const __m256 c3 = _mm256_set1_ps(9.6164605538e-03f);
    const __m256 c4 = _mm256_set1_ps(5.5504156855e-02f);
    const __m256 c5 = _mm256_set1_ps(2.4022684109e-01f);
    const __m256 c6 = _mm256_set1_ps(6.9314717694e-01f);
    const __m256 c7 = _mm256_set1_ps(9.9999998955e-01f);
    __m256i k = _mm256_cvttps_epi32(x);
    __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(k));
    __m256 pl = _mm256_fmadd_ps(c0, f, c1);
    pl = _mm256_fmadd_ps(pl, f, c2);
    pl = _mm256_fmadd_ps(pl, f, c3);
    pl = _mm256_fmadd_ps(pl, f, c4);
    pl = _mm256_fmadd_ps(pl, f, c5);
    pl = _mm256_fmadd_ps(pl, f, c6);
    pl = _mm256_fmadd_ps(pl, f, c7);
    __m256i bits = _mm256_slli_epi32(
        _mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(pl, _mm256_castsi256_ps(bits));
  }
#endif

  // fp32 softmax over rows of g_seq on the host. Same structure as the NPU
  // kernel (max-subtract, exp2 with the -120 argument floor, one reciprocal),
  // but fp32 end to end -- like host LayerNorm, it removes dispatches AND
  // beats the bf16 path on accuracy.
  // The padding mask, as an explicit pass. Only the NPU-softmax branch needs
  // this: softmax_cpu folds the same addition into its per-row prologue, where
  // the row is already in L1 and it costs nothing, while an aie softmax kernel
  // has no second operand to take it from.
  void add_additive_mask(std::vector<float> &scores) {
    const int64_t rows_per_seq = g_heads * g_seq;
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      for (int64_t r = w; r < n_rows; r += nw) {
        float *row = scores.data() + r * g_seq;
        const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
        for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
      }
    });
  }

  void softmax_cpu(std::vector<float> &scores) {
    double t0 = now_s();
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
      const int64_t rows_per_seq = g_heads * g_seq;
      for (int64_t r = lo; r < hi; ++r) {
        float *row = scores.data() + r * g_seq;
        // The additive padding mask, folded in here rather than in qk(): the
        // row is already resident, so this is free, and it leaves qk() as the
        // pure matmul an array kernel could run. Same single float addition
        // qk() used to do, so the result is unchanged to the bit.
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

  void gelu_cpu(std::vector<float> &x) {
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

  // arch=2 gated FFN activation: [rows][2*inter] -> [rows][inter].
  //   out[r][j] = lo[r][j] * silu(hi[r][j])
  // lo = cols [0, inter) (fc11, the untouched up-path), hi = cols
  // [inter, 2*inter) (fc12, the SiLU gate) -- pinned by the container's
  // swiglu_halves == "fc11_up|fc12_gate", asserted once in set_model_shape()
  // rather than trusted here.
  //
  // silu(x) = x / (1 + exp(-x)); exp(-x) = exp2(-x*log2e), reusing
  // exp2_avx2 instead of adding a second exponential. The argument floor
  // mirrors softmax_cpu's -120: when x (the gate, hi[j]) is large positive,
  // -x*log2e is large negative, and unfloored that corrupts exp2_avx2's
  // internal int32 conversion (cvttps2dq's "indefinite integer" case)
  // instead of cleanly underflowing toward 0 -- the same failure mode
  // softmax's own masked (very negative) rows hit without that floor.
  //
  // OUT OF PLACE, into a caller-supplied buffer -- NOT the in-place scheme
  // this function originally shipped with. The "safe in-place, compacting
  // forward" proof this comment used to carry was WRONG, and it was wrong in
  // exactly the way the task that wrote it demanded be checked for
  // ("VERIFY this claim numerically... rather than trusting the algebra") --
  // caught by that numerical check, on real hardware, tasks/0070:
  //   Row r WRITES [r*inter, (r+1)*inter) and READS [r*2*inter,(r+1)*2*inter).
  //   The original proof showed no row r' > r can have its READ range
  //   clobbered by row r's WRITE -- true, but it never checked r' < r. Row
  //   r's write range and row r' = floor(r/2)'s READ range overlap for
  //   EVERY r >= 1 (write=[r*inter,(r+1)*inter), read=[2r'*inter,(2r'+2)*inter),
  //   and r=2r' or r=2r'+1 both fall inside that read range by construction).
  //   Sequentially this is harmless (r' < r is always processed first in an
  //   ascending loop, so its read completes before r's write). Threaded, it
  //   is not: pool->run() hands CONTIGUOUS chunks to independent threads with
  //   no ordering between them, so whenever r and floor(r/2) land in
  //   different chunks (e.g. r'=63 in one thread's chunk, r=127 in another's,
  //   with no happens-before edge), row 127's write can race row 63's read.
  //   Measured effect: rows immediately after such a boundary came back
  //   catastrophically wrong (rel err up to 1.23, i.e. wrong sign / wrong
  //   magnitude, not bf16 noise) while every other row matched the oracle to
  //   ~1e-3 -- found via reference/encoder_nomic.py's own L0.fc11/fc12/gated
  //   taps, which isolated the corruption to exactly this function on a
  //   4-sentence batch after `--embed` (the golden gate never caught it
  //   because it tiles ONE 4-sentence batch 32x, so every "different" row is
  //   actually identical content and a wrong-row read is indistinguishable
  //   from a right-row read).
  void swiglu_cpu(const std::vector<float> &x, std::vector<float> &out) {
    double t0 = now_s();
    const int64_t inter = g_ffn;
    const int64_t n_rows = rows;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo_r = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi_r = std::min<int64_t>(n_rows, lo_r + chunk);
      for (int64_t r = lo_r; r < hi_r; ++r) {
        const float *lo = x.data() + r * 2 * inter;
        const float *hi = lo + inter;
        float *dst = out.data() + r * inter;
        // arch=3 (tasks/0136): exact-erf GELU on the gate half, same halves
        // order. The SiLU arm below is byte-for-byte what arch=2 always ran.
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
    // Reuses gelu_cpu's bucket -- it is the same "FFN activation, on the
    // host, in place of an NPU dispatch" cost this timer already names.
    t_hostgelu += now_s() - t0;
  }

  void layer_norm(std::vector<float> &x, size_t slot) {
    if (host_ln) { layer_norm_cpu(x, slot - 1); return; }
    double t0 = now_s();
    layernorm.bind(1, slot);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(layernorm.host_ptr(0)) + lo,
                x.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    layernorm.sync_to_device(0);
    t0 = lap(t0, t_in);
    layernorm.dispatch_only();
    t0 = lap(t0, t_disp);
    layernorm.sync_from_device(2);
    t0 = lap(t0, t_out);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_read(x.data() + lo,
                static_cast<const uint16_t *>(layernorm.host_ptr(2)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // Per-row activation scales for the int8 path, [rows]. Filled by gemm()'s
  // quantisation pass and consumed by its dequantisation pass in the same
  // call, so it never has to be threaded anywhere.
  std::vector<float> a_scale;
  // ...except when the ffn_up epilogue is fused (T37), where one pass produces
  // BOTH this GEMM's dequantised result and the next one's quantised operand,
  // so the two sets of row scales are live at once.
  std::vector<float> a_scale_next;
  std::vector<float> inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  // 1/asmooth for the GEMM currently being run, rebuilt only when the pointer
  // changes (i.e. per op, not per layer-iteration, and never per row).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;

  // Set when gemm() should not write `out` at all, but instead run the
  // activation and quantise straight into the NEXT GEMM's operand.
  struct FusedNext {
    const float *asmooth;   // the next GEMM's per-input-channel divisor
    int8_t *dst;            // the next GEMM's device A slot
    float *scale;           // filled with the next GEMM's per-row scales
    bool gated;             // SwiGLU narrows 2*inter -> inter; GELU does not
  };

  // T37-BF16 (tasks/0108): as FusedNext, for the bf16/bfp16 datapath. No
  // quantisation on this path, so no smoothing divisor and no per-row scale
  // -- the only thing that survives is the narrow-to-bf16 write, straight
  // into the next GEMM's device A slot.
  struct FusedNextBf16 {
    uint16_t *dst;           // the next GEMM's device A slot (bf16)
    bool gated;              // SwiGLU/GeGLU narrows 2*inter -> inter
  };

  // nullptr for a bf16 container, where these vectors are empty and the int8
  // arms of gemm() never run.
  static const float *i8w(const std::vector<const float *> &v, int64_t L) {
    return v.empty() ? nullptr : v[static_cast<size_t>(L)];
  }

  // T37-BF16 (tasks/0108): as dequant_act_quant (T37, tasks/0082), for the
  // bf16/bfp16 datapath. STEP 1 of this task measured the three passes this
  // collapses -- ffn_up's C-readback+bias (into `up`), the activation in
  // place, and ffn_down's A-conversion (out of `up`) -- at 17.7-28.3% of
  // wall clock across the shipped catalogue; `up` is never materialised.
  // No quantisation step exists on this path, so unlike dequant_act_quant
  // there is no per-row scale and no smoothing divisor to carry.
  //
  // BIT-IDENTICAL to the unfused path BY CONSTRUCTION, not by review: the
  // C-readback+bias arm below is copied verbatim from gemm()'s own unfused
  // bf16-C and fp32-C branches (the ones just below this function), the
  // activation arms are copied verbatim from the int8 fused epilogue's own
  // lambda (itself verified bit-identical to gelu_cpu/swiglu_cpu, tasks/0082
  // sec 2), and the final narrowing calls the SAME bf16_fill() the unfused
  // A-conversion calls. Calling bf16_fill per row rather than once over the
  // whole buffer changes nothing: it is round-to-nearest-even, purely
  // elementwise -- every output bit depends only on its own input float, not
  // on its neighbours or its position in the array.
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
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
        // Activation, in place -- COPIED VERBATIM from the int8 fused
        // epilogue's own lambda just below, not re-derived.
        if (!gated) {
          int64_t k = 0;
#if defined(__AVX2__)
          for (; k + 8 <= N; k += 8)
            _mm256_storeu_ps(v + k, gelu8(_mm256_loadu_ps(v + k)));
#endif
          for (; k < N; ++k) v[k] = gelu8(v[k]);
        } else if (g_gated_act == GatedAct::GeluErf) {
          // arch=3 (tasks/0136): exact-erf GELU on the gate half. The SiLU
          // arm below is byte-for-byte what arch=2 always ran.
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

  void gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
            size_t wslot, const float *bias, std::vector<float> &out,
            int64_t N, const float *wscale = nullptr,
            const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or a pipeline lane was constructed without "
          "copying ws_*/as_* from lane 0");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A was written straight into the device slot by the PREVIOUS gemm's
      // fused epilogue (T37), and a_scale already holds its row scales. There
      // is nothing to convert: the whole point is that this tensor is never
      // materialised in fp32 at all.
    } else if (i8) {
      // QUANTISE A, per row, with the SmoothQuant divisor folded into the same
      // pass (tasks/0078). Two reads of each row would cost a second sweep of
      // 12.6 MB at batch 128; one pass computes max|x/s| and the second writes
      // the rounded quotient.
      //
      // The smoothing divisor is NOT folded into LayerNorm: BERT is post-LN,
      // so each LayerNorm's output feeds the residual as well as this GEMM,
      // and scaling gamma/beta would scale the residual too (tasks/0078 4a).
      const int64_t K = static_cast<int64_t>(a.size()) / rows;
      a_scale.resize(static_cast<size_t>(rows));
      // RECIPROCATE THE SMOOTHING VECTOR ONCE PER GEMM, not twice per element.
      // The first version divided by asmooth[j] in both the max pass and the
      // quantise pass -- two divisions per element, and `conv` went 36.2 ms
      // (bf16) to 109.9 ms, eating 74 ms of the 126 ms the array had saved.
      // K floats of setup replaces 2*rows*K divisions.
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
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue --
      // the bf16 analogue of the i8-and-a_ready arm above. Nothing to do.
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
      d.bind(1, wslot);        // weights are already on the device
      d.bind(2, slot_c);
      d.sync_to_device(0, a.size() * d.info().a_elem_bytes);
      t0 = lap(t0, t_in);
      d.dispatch_only();
      t0 = lap(t0, t_disp);
      // NPUE-M9 (tasks/0045): with --c-bf16 the design narrows C on the core
      // after the fp32 K reduction, so this moves half the bytes. The size
      // comes from the design, never from an assumption about the datatype.
      const size_t cb = d.info().c_elem_bytes;
      d.sync_from_device(2, static_cast<size_t>(rows) * N * cb);
      t0 = lap(t0, t_out);
      // The pointer survives the unlock -- it is THIS pipeline's own bo; the
      // other pipeline binds its own slots and never touches this memory.
      c = static_cast<const float *>(d.slot_ptr(2, slot_c));
    }
    // The bias add reads the result buffer directly. It used to be a memcpy
    // out followed by a second pass over the same 21 MB; this is one pass.
    //
    // STREAMING loads in both arms: the C buffer is an XRT host bo, and
    // ordinary loads from it measured ~80 ms per encode (~2 GB/s) -- the
    // signature of uncached/write-combined memory, where each load stalls the
    // core. movntdqa reads a whole WC line per transaction. Alignment holds:
    // the bo map is page-aligned and N is a multiple of 16.
    if (i8 && fuse) {
      // FUSED EPILOGUE (T37): dequantise, apply GELU, and quantise into the
      // next GEMM's operand in ONE pass over the widest tensor in the model.
      // `out` is deliberately never written -- see dequant_gelu_quant.
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
            // arch=3 (tasks/0136): no int8 gte container exists yet
            // (pack_npue refuses --int8 for arch=3), but if one arrives this
            // arm must not silently run SiLU over a GELU model.
            if (g_gated_act == GatedAct::GeluErf) {
              const int64_t inter = n / 2;
              const float *hi = v + inter;
              for (int64_t j = 0; j < inter; ++j)
                v[j] = v[j] * gelu_erf_exact(hi[j]);
              return;
            }
            // SwiGLU, narrowing 2*inter -> inter in place. Identical
            // intrinsics to swiglu_cpu, including its -120 argument floor.
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
              __m256 e = Encoder::exp2_avx2(a);
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
      // DEQUANTISE: y = int32_acc * sa[row] * wscale[col] + bias[col], folded
      // into the pass that already reads C and adds the bias -- one extra
      // multiply per output element, no extra sweep of the 679 MB tasks/0044
      // measured this readback at. The helper picks the transport width from
      // the design, so a narrowed-C set (tasks/0080) needs nothing here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); }, sim_c_bf16);
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
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
          // One 32-byte streaming load carries 16 bf16, against 8 fp32 --
          // which is the whole point: same instruction count, half the traffic.
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

  // THE OTHER MULTI-PASS CHAIN (T37, tasks/0082 section 5).
  //
  //   add_into(x, y)      read y, read residual, write x
  //   layer_norm(x)       read x, write x
  //   memcpy(residual, x) read x, write residual
  //   quantise for next   read x, write int8
  //
  // Eight streaming passes over a rows x hidden tensor, TWICE per layer --
  // 33.5 MB at bge-large's batch 128. Every one of them touches the same row,
  // and a row is 4 KB, so all four kernels can share one L1-resident copy:
  // read y and residual once, write x, residual and the int8 operand once.
  //
  // `dst` may be null (the last layer's LN feeds pooling, not a GEMM), in
  // which case this is add + norm + residual with no quantisation.
  //
  // BIT-IDENTICAL to the four kernels it replaces, and that is not automatic:
  // it holds only because the intrinsics and the accumulation ORDER match
  // theirs exactly. tasks/0082 measured a scalar rewrite of the same algebra
  // landing on a different number, because fmadd rounds once where a*b+c
  // rounds twice.
  void add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
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
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
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
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
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
        float mx = 0.f;                                  // == quantise_a_int8
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

  // T37-BF16 (tasks/0108): as add_norm_quant, for the bf16/bfp16 datapath --
  // add + LayerNorm + residual copy + the NEXT gemm's A-conversion, one
  // L1-resident pass instead of the unfused chain's four streaming ones
  // (add_into, layer_norm_cpu, memcpy(residual), and the next gemm()'s own
  // bf16_fill). No quantisation on this path, so this is a strict subset of
  // add_norm_quant's work -- the add/LN/residual arithmetic below is copied
  // verbatim from it, with the int8 absmax+quantise tail replaced by a
  // single bf16_fill call. `dst` may be null (the last layer's LN feeds
  // pooling, not a GEMM), in which case this is add + norm + residual only,
  // exactly like the unfused path's own unconditional residual write.
  void add_norm_bf16(std::vector<float> &x, const std::vector<float> &y,
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
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
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
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
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
        if (dst) bf16_fill(dst + r * H, row, static_cast<size_t>(H));
      }
    });
    t_hostln += now_s() - t0;
  }

  // x += y, elementwise. The residual adds move 12.6 MB per layer at batch 128.
  void add_into(std::vector<float> &x, const std::vector<float> &y) {
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

  // scores[b,h,i,j] = dot(Q[b,i,h], K[b,j,h]) + mask[b,j]
  // scores[b,h,i,j] = Q[b,i,h] . K[b,j,h]. NO mask: this is the operation an
  // array kernel would perform, and the mask is a property of the batch rather
  // than of the matmul. add_additive_mask() below applies it.
  // NV is head_dim/8 as a COMPILE-TIME constant where we have one, so the
  // inner loop unrolls and qv[]/acc[] stay in registers. NV == 0 keeps the
  // fully generic path for a width we have not met yet.
  template <int NV>
  void qk_impl(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    const int64_t pairs = batch * g_heads;
    // __restrict, because these are members now: the compiler could prove two
    // fresh local allocations did not overlap and cannot prove it for two
    // fields of the same object, and without the proof every store to dst[j]
    // re-issues the loads. Measured at 2x on this loop.
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict sc_p = scores.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *q = &qkv_p[(b * g_seq + i) * 3 * g_hidden + h * g_head_dim];
          float *dst = &sc_p[(p * g_seq + i) * g_seq];
          // head_dim / 8 vectors, held across the j loop. head_dim is 32 for
          // MiniLM and bge-small and 64 for bge-large; kMaxHeadVecs bounds the
          // stack array and set_model_shape() refuses anything larger.
#if defined(__AVX512F__)
          // QK^T AT 512 BITS. Unlike A.V (which gained nothing, tasks/0086),
          // this one pays -- and not because the arithmetic is wider. The win
          // is that `_mm512_reduce_add_ps` replaces `hsum256`'s four-
          // instruction shuffle chain, and QK^T does one reduction per (i, j)
          // pair. Microbenchmarked at bge-large's geometry: 1.33x on the inner
          // loop, against 1.00x for the same widening applied to A.V.
          //
          // NOT BIT-IDENTICAL, and it cannot be: summing head_dim floats as
          // lanes of 16 associates differently from lanes of 8. That is a
          // legitimate reassociation, not an error, but it means the byte
          // comparison the fusions used does not apply here -- this is gated on
          // 1-cos instead.
          //
          // head_dim is a multiple of 8 but not necessarily of 16, so the
          // 512-bit part takes what it can and a 256-bit tail finishes.
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
            // Accumulate in the same order the unrolled version did, so the
            // floating-point result is unchanged for head_dim 32.
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

  // Dispatch on the width the container reported. head_dim 32 is MiniLM and
  // bge-small, 64 is bge-large; anything else still works, just generically.
  void qk(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    switch (g_head_dim) {
      case 32: qk_impl<4>(qkvbuf, scores); break;
      case 64: qk_impl<8>(qkvbuf, scores); break;
      default: qk_impl<0>(qkvbuf, scores); break;
    }
  }

  // ctx[b,i,h] = sum_j scores[b,h,i,j] * V[b,j,h]
  template <int NV>
  void av_impl(const std::vector<float> &scores,
               const std::vector<float> &qkvbuf, std::vector<float> &ctx) {
    const int64_t pairs = batch * g_heads;
    const float *__restrict sc_p = scores.data();
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict ctx_p = ctx.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *a = &sc_p[(p * g_seq + i) * g_seq];
          float *o = &ctx_p[(b * g_seq + i) * g_hidden + h * g_head_dim];
#if defined(__AVX2__)
#if defined(__AVX512F__)
          // A.V AT 512 BITS IS BIT-IDENTICAL TO THE 256-BIT FORM, and that is
          // not luck -- it is why this half was done first (tasks/0086).
          //
          // Each accumulator lane owns ONE output element and sums over j in
          // the same order either way; widening only changes how many lanes
          // ride in a register, never which numbers are added or when. So the
          // byte-comparison harness applies here exactly as it did to the
          // fusions.
          //
          // QK^T is the opposite case and is NOT converted: it ends in a
          // horizontal reduction (`hsum256`), and reducing head_dim floats as
          // 2 lanes of 16 sums them in a different ORDER than 4 lanes of 8.
          // That is a reassociation, so it would change the result -- small,
          // legitimate, and no longer checkable by byte comparison. Left for a
          // measurement that uses the 1-cos gate instead.
          //
          // head_dim is a multiple of 8 (set_model_shape refuses otherwise) but
          // not necessarily of 16, so the 512-bit path takes the multiple-of-16
          // part and a 256-bit tail finishes it.
          const int64_t nv = NV ? NV : g_head_dim / 8;
          const int64_t nz = nv / 2;                  // 512-bit accumulators
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

  void av(const std::vector<float> &scores, const std::vector<float> &qkvbuf,
          std::vector<float> &ctx) {
    switch (g_head_dim) {
      case 32: av_impl<4>(scores, qkvbuf, ctx); break;
      case 64: av_impl<8>(scores, qkvbuf, ctx); break;
      default: av_impl<0>(scores, qkvbuf, ctx); break;
    }
  }

  // arch=2: rotate Q and K IN PLACE inside the fused qkv buffer -- Q at
  // column offset 0, K at `hidden`, V at `2*hidden` -- rather than repacking
  // into [B,H,S,D] the way gemma_encode.cpp does. Repacking would be a
  // ~200 MB shuffle per layer at production batch; rotating the 64 floats of
  // each head in place needs no extra buffer at all.
  //
  // For each (b, s) row and each head h, at `row_base + q_off + h*head_dim`
  // (`half = head_dim/2`):
  //   for d in [0, half):
  //     x1 = v[d]; x2 = v[d + half]
  //     v[d]        = x1*cos[s][d] - x2*sin[s][d]
  //     v[d + half] = x2*cos[s][d] + x1*sin[s][d]
  // Both halves read cos[s][d]/sin[s][d] for the SAME d -- that is what
  // NeoX's concat(freqs, freqs) means (gemma_kernels.hpp's own
  // gemma_rope_tables() already duplicates the table this way), so only the
  // first `half` columns of the table are ever read here. Applied to Q
  // (offset 0) and K (offset g_hidden). NEVER V.
  void apply_rope_qkv(std::vector<float> &qkv) {
    if (!rope_ready) {
      // Built ONCE per Encoder: g_seq/g_head_dim/g_rope_theta are fixed for
      // the whole design (set_design_seq() runs once, before any Encoder
      // exists), so every layer of every call shares this table.
      rope_cos.resize(static_cast<size_t>(g_seq * g_head_dim));
      rope_sin.resize(static_cast<size_t>(g_seq * g_head_dim));
      if (!g_rope_inv_freq.empty()) {
        // arch=3: the frequencies come from the container (tasks/0134 --
        // not derivable from any single theta). Same NeoX
        // concat(freqs, freqs) table layout gemma_rope_tables() emits, and
        // the same double-angle, round-once-at-the-end arithmetic; the
        // rotation below only ever reads the first half of each row.
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

    pool->run([&](int w, int nw) {
      for (int64_t row = w; row < n_rows; row += nw) {
        const int64_t s = row % g_seq;    // [b][s] row order -> s = row % seq
        const float *cs = cos_p + s * g_head_dim;
        const float *sn = sin_p + s * g_head_dim;
        float *row_base = p + row * row_stride;
        for (int64_t h = 0; h < g_heads; ++h) {
          rotate_pair(row_base + h * g_head_dim, cs, sn);              // Q
          rotate_pair(row_base + g_hidden + h * g_head_dim, cs, sn);   // K
        }
      }
    });
  }

  std::vector<float> run(const std::vector<float> &emb_in) {
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
    // Set by the PREVIOUS iteration's fused LayerNorm, which already wrote
    // this layer's qkv operand and its row scales into the device slot.
    bool qkv_a_ready = false;
    for (int64_t L = 0; L < g_layers; ++L) {
      if (!qkv_a_ready)
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

      gemm(qkv, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden,
           i8w(ws_qkv, L), i8w(as_qkv, L), nullptr, /*a_ready=*/qkv_a_ready);
      qkv_a_ready = false;
      // arch=2: rotate Q and K in place, strictly after the GEMM (RoPE is a
      // per-position rotation of the projected q/k, not of the input) and
      // strictly before qk() reads them. No-op (false) for every arch=0
      // container.
      if (g_rope) apply_rope_qkv(qkvbuf);

      double ta = now_s();
      // QK^T per head, on the host: [64,32]x[32,64] does not tile (head_dim 32
      // fails the whole-array design's M % (m*4) == 0).
      // 1/sqrt(head_dim) is already folded into Q by the .npue.
      //
      // head_dim is 32 = four AVX2 vectors, and the 32 floats of one head ARE
      // contiguous even though consecutive rows are 3*hidden apart. So the dot
      // product vectorises without any repacking.
      qk(qkvbuf, scores);
      t_attn += now_s() - ta; t_qk += now_s() - ta;

      if (host_sm) {
        softmax_cpu(scores);  // applies add_mask itself
      } else {
        add_additive_mask(scores);
        eltwise(softmax, scores.data(), scores.size());
      }

      ta = now_s();
      // A.V. Each (b, h, i) owns its own 32 output floats, so this accumulates
      // in registers and stores once -- no zero-fill of ctx needed, and no
      // sharing between threads.
      av(scores, qkvbuf, ctx);
      t_attn += now_s() - ta; t_av += now_s() - ta;

      gemm(attn_out, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden,
           i8w(ws_ao, L), i8w(as_ao, L));
      // T37 site 1: add + LayerNorm + residual copy + ffn_up's quantisation,
      // one L1-resident pass instead of four streaming ones.
      const bool fuse_ln = fuse_ffn_epilogue && host_ln &&
                           ffn_up.info().a_elem_bytes == 1;
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue -- same fused pass,
      // narrowing straight to bf16 instead of quantising (no scale, no
      // smoothing divisor).
      const bool fuse_ln_bf16 = fuse_ffn_epilogue && host_ln &&
                                ffn_up.info().a_elem_bytes == 2;
      if (fuse_ln) {
        const float *asf = i8w(as_fu, L);
        if (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
            inv_smooth_src != asf) {
          inv_smooth.resize(static_cast<size_t>(g_hidden));
          for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asf[j];
          inv_smooth_src = asf;
        }
        a_scale.resize(static_cast<size_t>(rows));
        // `s_ln[...]` is a design SLOT; h_gamma/h_beta are indexed by
        // site, and layer_norm() converts with `slot - 1`. Passing the
        // slot straight through segfaulted on the last layer.
        add_norm_quant(x, proj, s_ln[1 + 2 * L] - 1, inv_smooth.data(),
                       static_cast<int8_t *>(ffn_up.slot_ptr(0, slot_a)),
                       a_scale.data());
      } else if (fuse_ln_bf16) {
        add_norm_bf16(x, proj, s_ln[1 + 2 * L] - 1,
                     static_cast<uint16_t *>(ffn_up.slot_ptr(0, slot_a)));
      } else {
        add_into(x, proj);
        layer_norm(x, s_ln[1 + 2 * L]);
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
      }
      // T37: when the whole ffn_up -> GELU -> ffn_down chain is int8 and GELU
      // is on the host, the epilogue is fused and `up` is never materialised.
      // Gated FFNs fuse too: SwiGLU narrows 2*inter -> inter inside the same
      // L1-resident row, which is if anything a bigger saving because their
      // ffn_up output is twice as wide.
      FusedNext fn{i8w(as_fd, L),
                   static_cast<int8_t *>(ffn_down.slot_ptr(0, slot_a)),
                   nullptr, g_gated_ffn};
      const bool fuse_ffn = fuse_ffn_epilogue && host_gelu &&
                            ffn_up.info().a_elem_bytes == 1 &&
                            ffn_down.info().a_elem_bytes == 1;
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- STEP 1 measured this chain (ffn_up's C-readback+bias, the
      // activation, ffn_down's A-convert) at 17.7-28.3% of wall clock across
      // the shipped catalogue before this fusion existed.
      FusedNextBf16 fn_bf16{
          static_cast<uint16_t *>(ffn_down.slot_ptr(0, slot_a)), g_gated_ffn};
      const bool fuse_ffn_bf16 = fuse_ffn_epilogue && host_gelu &&
                                 ffn_up.info().a_elem_bytes == 2 &&
                                 ffn_down.info().a_elem_bytes == 2;
      gemm(ffn_up, is_fu, x, s_fu[L], b_fu[L], up,
           g_gated_ffn ? 2 * g_ffn : g_ffn, i8w(ws_fu, L), i8w(as_fu, L),
           fuse_ffn ? &fn : nullptr, /*a_ready=*/fuse_ln || fuse_ln_bf16,
           fuse_ffn_bf16 ? &fn_bf16 : nullptr);

      // arch=2: SwiGLU (fc11 * silu(fc12), fused as one [hidden, 2*inter]
      // ffn_up) in place of plain GELU over a [hidden, inter] ffn_up --
      // writes into the separate `gated` buffer (see swiglu_cpu's own
      // comment for why NOT in place). Every arch=0 container has
      // g_gated_ffn == false and takes the untouched branch below.
      if (fuse_ffn) {
        // T37: the activation's output was never written. The ffn_up epilogue
        // already produced ffn_down's int8 operand in the device slot and its
        // row scales, so this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (fuse_ffn_bf16) {
        // T37-BF16: `up`/`gated` was never written -- the ffn_up epilogue
        // already produced ffn_down's bf16 operand in the device slot.
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (g_gated_ffn) {
        swiglu_cpu(up, gated);
        gemm(ffn_down, is_fd, gated, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      } else if (false) {
        // T37: `up` was never written. The ffn_up epilogue already produced
        // ffn_down's int8 operand in the device slot and its row scales, so
        // this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
             i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else {
        if (host_gelu)
          gelu_cpu(up);
        else
          eltwise(gelu, up.data(), up.size());
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      }
      // T37 site 2: the same fusion at the layer's second LayerNorm. Its
      // consumer is the NEXT layer's qkv, so the loop's own
      // `memcpy(residual, x)` is what this replaces -- and on the last layer
      // there is no next GEMM, only pooling, so `dst` is null there.
      const bool last = (L + 1 == g_layers);
      if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 1) {
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
                       asq ? static_cast<int8_t *>(qkv.slot_ptr(0, slot_a))
                           : nullptr,
                       asq ? a_scale.data() : nullptr);
        qkv_a_ready = !last;
      } else if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 2) {
        // T37-BF16 (tasks/0108): as the int8 branch above, narrowing to bf16
        // instead of quantising -- no scale to compute.
        add_norm_bf16(x, down, s_ln[2 + 2 * L] - 1,
                      last ? nullptr
                           : static_cast<uint16_t *>(qkv.slot_ptr(0, slot_a)));
        qkv_a_ready = !last;
      } else {
        add_into(x, down);
        layer_norm(x, s_ln[2 + 2 * L]);
      }
    }
    return x;
  }
};

}  // namespace app
