//===- bert_encoder.hpp ----------------------------------*- C++ -*-===//
//
// BERT-family NPU encoder (absolute positions, post-LN; arch=0, and
// arch=2/3 via the data-driven RoPE/gated-FFN switches in the container
// config). Moved from runtime/include/bert_encoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/encoder.hpp"
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/pool.hpp"
#include "runtime/types.hpp"
#include "runtime/model.hpp"
#include "common/host_kernels.hpp"
#include "tokenizers/wordpiece.hpp"

namespace app {
struct AnyTokenizer;
}

namespace npue {

class BertEncoder : public Encoder {
public:
  BertEncoder(npue::File &model,
              npu::Design &qkv_d, npu::Design &ao_d,
              npu::Design &fu_d, npu::Design &fd_d,
              npu::Design &gelu_d, npu::Design &ln_d, npu::Design &sm_d,
              app::Pool &pool);
  ~BertEncoder();

  std::vector<float> encode(const std::vector<std::string> &texts,
                                  const std::string &prefix,
                                  int64_t *tokens) override;
  int64_t hidden() const override;
  int64_t seq() const override;

  // Raw forward pass over already-embedded [batch*seq, hidden] rows.
  std::vector<float> run(const std::vector<float> &emb_in);

  size_t stage_all();
  int64_t use_tier(int64_t want);
  void reset_timers();

  // Active fields set by setup_encoder() or design selection.
  bool unified = false;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;

  // The additive attention mask in the form softmax consumes, [batch, seq].
  // Public because setup_encoder(), --encode-file and the embedding service
  // all install it after construction (it depends on the active tier).
  std::vector<float> add_mask;
  std::mutex *npu_mu = nullptr;
  size_t slot_a = 0, slot_c = 0;
  int64_t batch = 0, rows = 0;
  bool host_ln = false;
  bool host_sm = false;
  bool host_gelu = false;
  bool sim_c_bf16 = false;
  bool fuse_ffn_epilogue = true;

  // Pipeline lane fields.
  size_t lane_id = 0;
  size_t n_lanes = 1;

  // Per-lane input/output buffers on the three eltwise designs.
  //
  // A Design owns ONE A buffer and ONE C buffer, and --pipeline runs several
  // BertEncoders against the SAME Design objects (setup_encoder). Lane 0 uses
  // the base slots (0); every extra lane gets its own, exactly as d_qkv() got
  // one per lane. Without this the lanes write each other's rows and read each
  // other's results, and which lane wins the race is thread-scheduling
  // dependent -- which is why the answer changed between two identical
  // requests (bert_encoder.cpp::layer_norm, T-eltwise-shared-buffer).
  struct EltSlots {
    size_t a = 0;
    size_t c = 0;
  };
  EltSlots slots_gelu, slots_ln, slots_sm;

  // Members below are public because setup_encoder() stages the weights and
  // the pipeline lanes copy them, and run_bench_or_check() reads the timers --
  // exactly the access the old aggregate Encoder offered. There is no owner
  // other than the runtime.
  npue::File &model_;
  npu::Design &qkv_, &attn_out_, &ffn_up_, &ffn_down_, &gelu_, &layernorm_, &softmax_;
  app::Pool &pool_;

  // Staged weight slots and bias pointers.
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  std::vector<size_t> s_ln;
  std::vector<const float *> h_gamma, h_beta;

  // Device-resident weights, one slot per layer per design.
  std::vector<float> residual;
  std::vector<float> qkvbuf, ctx, proj, up, down, scores;
  std::vector<float> gated;
  std::vector<float> rope_cos, rope_sin;
  bool rope_ready = false;

  // Per-row activation scales and smoothing divisors for the fused int8/bf16
  // epilogues (T37). Memoised against the source pointer they were built for.
  std::vector<float> a_scale, a_scale_next;
  std::vector<float> inv_smooth, inv_smooth_next;
  const float *inv_smooth_src = nullptr;
  const float *inv_smooth_next_src = nullptr;

  // Timers.
  double t_npu = 0.0, t_attn = 0.0;
  double t_qk = 0.0, t_av = 0.0;
  double t_conv = 0.0, t_in = 0.0, t_disp = 0.0, t_out = 0.0, t_bias = 0.0;
  double t_hostln = 0.0, t_hostsm = 0.0, t_hostgelu = 0.0;
  int n_dispatch = 0;

  // Tokenizer for the text-in `encode()` entry point. Held opaque so this
  // header need not pull in the facade; built lazily, once.
  std::unique_ptr<app::AnyTokenizer> tok_;

  // Inline helpers.
  template <typename F> void par(size_t n, F &&f) const;
  template <typename F> void par_rows(int64_t n, F &&f) const;
  double lap(double t0, double &bucket);

  // NPU dispatch helpers.
  //
  // Both of these run on a design that --pipeline lanes SHARE, so both take
  // npu_mu around the bind/dispatch window. The host-side conversion is
  // outside that window, which is only sound because the A and C buffers are
  // per-lane (EltSlots above) -- the same arrangement gemm() already has.
  void eltwise(npu::Design &d, const EltSlots &slots, float *x, size_t n);
  void layer_norm(std::vector<float> &x, size_t slot);

  // Host-side passes.
  void layer_norm_cpu(std::vector<float> &x, size_t site);
  void softmax_cpu(std::vector<float> &scores);
  void gelu_cpu(std::vector<float> &x);
  void swiglu_cpu(const std::vector<float> &x, std::vector<float> &out);
  void add_additive_mask(std::vector<float> &scores);

  // GEMM and fused epilogues.
  struct FusedNext {
    const float *asmooth;
    int8_t *dst;
    float *scale;
    bool gated;
  };
  struct FusedNextBf16 {
    uint16_t *dst;
    bool gated;
  };
  static const float *i8w(const std::vector<const float *> &v, int64_t L);
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, bool gated, const float *bias,
                        uint16_t *dst);
  void gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
            size_t wslot, const float *bias, std::vector<float> &out,
            int64_t N, const float *wscale = nullptr,
            const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr);

  // Attention helpers.
  template <int NV>
  void qk_impl(const std::vector<float> &qkv, std::vector<float> &scores);
  template <int NV>
  void av_impl(const std::vector<float> &scores, const std::vector<float> &qkv,
               std::vector<float> &ctx);
  void apply_rope_qkv(std::vector<float> &qkv);
  void qk(const std::vector<float> &qkv, std::vector<float> &scores);
  void av(const std::vector<float> &scores, const std::vector<float> &qkv,
          std::vector<float> &ctx);

  // Layer-norm fusion helpers.
  void add_into(std::vector<float> &x, const std::vector<float> &y);
  void add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
                      size_t site, const float *ias_next, int8_t *dst,
                      float *scale_next);
  void add_norm_bf16(std::vector<float> &x, const std::vector<float> &y,
                     size_t site, uint16_t *dst);

  // Embedding + pooling + normalize (used by encode()).
  std::vector<float> embed_and_pool(const std::vector<int32_t> &ids,
                                    const std::vector<uint8_t> &mask,
                                    int64_t n_real);
};

}  // namespace npue