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
#include "whisper/attention_npu.hpp"
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

  // Where the time went, per site, since reset_timers().
  //
  // Exists because every optimisation decision in this area is a question
  // about WHICH pass is expensive, and the timers were already being filled in
  // (bert_encoder.cpp) but had no way out. Guessing costs more than printing.
  struct Timings {
    double qk = 0, av = 0;            // attention: QK^T and A*V
    double npu = 0, attn = 0;         // the whole attention block, and its share
    double conv = 0, in = 0, disp = 0, out = 0, bias = 0;
    double hostln = 0, hostsm = 0, hostgelu = 0;   // host-side elementwise
    int dispatch = 0;
  };
  Timings timings() const {
    Timings t;
    t.qk = t_qk; t.av = t_av; t.npu = t_npu; t.attn = t_attn;
    t.conv = t_conv; t.in = t_in; t.disp = t_disp; t.out = t_out;
    t.bias = t_bias;
    t.hostln = t_hostln; t.hostsm = t_hostsm; t.hostgelu = t_hostgelu;
    t.dispatch = n_dispatch;
    return t;
  }

  // Active fields set by setup_encoder() or design selection.
  bool unified = false;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  std::vector<int64_t> tiers;
  // qkv, attn_out, ffn_up, ffn_down -- and, when the set carries them and
  // --npu-ops attn named them, attn_qk and attn_av in slots 4 and 5. They ride
  // in the SAME per-tier table rather than in a second one because a stream
  // slot is only meaningful together with the M its tier was built at: a tier
  // whose attn slot came from a different tier would gather query rows at one
  // row count and hand them to a GEMM that computes another.
  std::vector<std::array<size_t, 6>> tier_slots;
  // One NpuAttention per tier, owned by RunContext and handed over as raw
  // pointers -- use_tier() picks among them the same way it picks among the
  // slots, so a request that changes tier changes its attention with it.
  std::vector<npue::whisper::NpuAttention *> attns;

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

  // Attention as two GEMMs on the set's own attn_qk/attn_av streams. The host
  // path is a real path and not a fallback; which one runs is decided by
  // use_tier(), because the stream slots and the row count a dispatch computes
  // are both properties of the tier. setup_encoder() fills `attns` and the
  // slots in `tier_slots` and refuses by name a set that was exported without
  // the code.
  bool attention_on_array() const { return attn_ != nullptr; }

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
  npue::whisper::NpuAttention *attn_ = nullptr;
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
  // The row stride of `scores`, which is g_seq when the host takes the softmax
  // and the softmax design's own `cols` when the array takes it -- see run().
  // It is a separate number from g_seq because the two consumers of a score row
  // were written against different widths: the host's own qk/av against the
  // real key count and the kernel against the padded one NpuAttention hands it.
  int64_t sc_cols = 0;
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
  // All three of qk, the softmax and av as ONE array pass, when a design set
  // carries the attn_qk/attn_av streams. Null -- the default, and what every
  // model gets unless --npu-ops attn names it -- is the host path above.
  void attention_npu(const std::vector<float> &qkv, std::vector<float> &ctx);

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