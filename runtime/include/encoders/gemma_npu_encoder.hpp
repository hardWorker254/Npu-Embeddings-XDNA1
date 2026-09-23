//===- gemma_npu_encoder.hpp ------------------*- C++ -*-===//
//
// EmbeddingGemma-300M on the array (arch=1): MQA, per-layer RoPE,
// four RMSNorms, GeGLU. Moved from runtime/include/gemma_npu_encoder.hpp.
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

#include "runtime/encoder.hpp"
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/pool.hpp"
#include "runtime/types.hpp"
#include "runtime/model.hpp"
#include "encoders/gemma_kernels.hpp"
#include "common/host_kernels.hpp"
#include "tokenizers/gemma.hpp"

namespace npue {

class GemmaNpuEncoder : public Encoder {
public:
  GemmaNpuEncoder(npue::File &model, npu::Design &design, app::Pool &pool);

  std::vector<float> encode(const std::vector<std::string> &texts,
                                    const std::string &prefix,
                                    int64_t *tokens) override;
  int64_t hidden() const override;
  int64_t seq() const override;

  // Text-in entry point gemma_mode.hpp drives directly. `index_base` and
  // `n_real` exist so a truncation error can name the CALLER's input: the
  // caller pads a tier by repeating its last real text, and a padded row's
  // index does not exist upstream. `tokens` accumulates only the real texts'.
  std::vector<float> encode_batch(const std::vector<std::string> &texts,
                                  const std::string &prefix,
                                  size_t index_base = 0,
                                  size_t n_real = static_cast<size_t>(-1),
                                  int64_t *tokens = nullptr);

  size_t stage_all();
  int64_t use_tier(int64_t want);
  void reset_timers();

  npu::Design &d;

  npue::GemmaTokenizer tok;

  int64_t hidden_ = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0,
          layers = 0, dense_hidden = 0, qkv_n = 0, swp = 6;
  int64_t q_off = 0, k_off = 0, v_off = 0, kv_w = 0;
  double eps = 1e-6, rope_theta = 0.0, rope_theta_local = 0.0,
         attn_scale = 1.0;

  int64_t seq_ = 0, batch = 0, rows = 0;
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  size_t slot_a = 0, slot_c = 0;
  std::mutex *npu_mu = nullptr;

  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  std::vector<float> a_scale;
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

  std::vector<float> x, hbuf, qkvbuf, ctx, proj, upbuf, gatedbuf, down,
      scores, add_mask, cos_g, sin_g, cos_l, sin_l;
  std::vector<int32_t> ids;
  std::vector<uint8_t> mask;

  double t_conv = 0, t_in = 0, t_disp = 0, t_out = 0, t_bias = 0,
         t_norm = 0, t_attn = 0, t_rope = 0, t_geglu = 0, t_tok = 0;
  int n_dispatch = 0;

  // Inline helpers.
  template <typename F> void par(size_t n, F &&f) const;
  template <typename F> void par_rows(int64_t n, F &&f) const;
  double lap(double t0, double &bucket);
  static const float *at(const std::vector<const float *> &v, int64_t L);

  // Fused epilogue types.
  struct FusedNext {
    const float *asmooth;
    int8_t *dst;
    float *scale;
  };
  struct FusedNextBf16 {
    uint16_t *dst;
  };

  // NPU dispatch helpers.
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                            int64_t out_n, const float *bias, uint16_t *dst);
  void gemm(size_t islot, const float *a, size_t a_len, size_t wslot,
             const float *bias, std::vector<float> &out, int64_t N,
             const float *wscale = nullptr, const float *asmooth = nullptr,
             FusedNext *fuse = nullptr, bool a_ready = false,
             FusedNextBf16 *fuse_bf16 = nullptr);

  // Host-side passes.
  void rms_norm(const float *xin, int64_t in_stride, float *xout,
                 int64_t out_stride, int64_t n_rows, int64_t dim,
                 const float *w);
  void apply_rope(std::vector<float> &qkv, const float *cs_t, const float *sn_t);
  void geglu(const std::vector<float> &fused, std::vector<float> &out);
  void attention(const std::vector<float> &qkv, std::vector<float> &out);
  void gemm_host(const float *a, int64_t M, int64_t K, const float *b,
                  int64_t N, float *c) const;
  void ensure_tables();

private:
  npue::File &model_;
  app::Pool &pool_;
};

}  // namespace npue