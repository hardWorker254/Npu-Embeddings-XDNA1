//===- decoder.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper decoder stack: greedy autoregression over the
// encoder's output, with a KV cache per layer.
//
// THE SHAPE OF THE COMPUTE
// ------------------------
// One step is ONE token, so the useful work is a single row -- and the design
// set's smallest tier is 64 rows (M 64, tile m 16). Every per-step dispatch
// therefore pads one real row up to the tier's row count and uses one of the
// results. That is wasteful by a factor of 64 and it is still the right
// schedule: the alternative, recomputing the whole prefix every step, is
// quadratic in the sequence length and re-derives K/V the cache already holds.
//
// The cache is exact, not approximate: a cached K/V row is bit-identical to
// what recomputing it from the same state and the same weights would produce,
// because the GEMM, the bf16 rounding of the operand and the bias are the same
// operations either way.
//
// Cross-attention is different from self-attention in exactly one respect: its
// K|V comes from the ENCODER output, which does not change during a
// transcription. So it is computed once per layer, in the largest available
// tier, and every step reads the same cache. 1500 positions at 256 rows is six
// dispatches per layer instead of 1500.
//
// Positions are structural, not masked: the cache only ever holds positions at
// or before the current one, so causality cannot be violated by a caller and
// there is no mask to get wrong. The one thing the caller CAN get wrong is
// feeding tokens out of order, and that is refused by name.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "whisper/attention_npu.hpp"
#include "whisper/eltwise.hpp"
#include "whisper/geometry.hpp"
#include "whisper/logits_npu.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

// One instruction-stream slot per op, at one batch tier. The seven names are
// the `stt` kind's stream list in tools/data/npu_targets.json.
struct DecoderStreams {
  size_t self_qkv = 0, self_attn_out = 0, cross_q = 0, cross_kv = 0,
         cross_attn_out = 0, ffn_up = 0, ffn_down = 0;
};

struct DecoderTier {
  int64_t batch = 0;
  int64_t rows = 0;   // what ONE dispatch of this tier computes
  DecoderStreams streams;
};

class WhisperDecoder {
public:
  WhisperDecoder(npue::File &model, npu::Design &design, app::Pool &pool,
                 const Geometry &geom);

  void add_tier(const DecoderTier &t) { tiers_.push_back(t); }
  // Where LayerNorm and GELU run. Null is the host pass, which is the
  // measured-faster path at this width; a non-null design is one
  // --npu-ops code's worth of xclbin, shared with the encoder's.
  void set_layernorm(NpuEltwise *ln) { ln_ = ln; }
  void set_gelu(NpuEltwise *gelu) { gelu_ = gelu; }
  // Where attention runs. Null is the host pass; non-null is the design set's
  // attn_qk/attn_av streams at the STEP tier, which is the only place the
  // decoder attends (the cross-attention prefill is a GEMM, not an attention).
  void set_attention(NpuAttention *attn) { attn_ = attn; }
  // Where the vocabulary projection runs. Null is the host's fp32 matvec against
  // the tied embedding; non-null is the design set's logits_i streams, one
  // dispatch per chunk of the vocabulary.
  void set_logits(NpuLogits *lg) { logit_ = lg; }
  // Which tier a decode step dispatches at, and which one the cross-attention
  // K|V prefill uses. The prefill wants the widest tier available: it is the
  // only place in the decoder that touches all 1500 positions, and it happens
  // once per transcription.
  void set_step_tier(int64_t batch);
  void set_prefill_tier(int64_t batch);

  size_t stage_all();

  // Build the per-layer cross-attention K|V cache from the encoder's output.
  // Must be called before the first step, and again whenever the source changes.
  void set_source(const std::vector<float> &enc, int64_t n_src);

  // Forget every cached K/V row. A new transcription starts here.
  void reset();

  // One step: run `token` at `position` and return the greedy next id.
  // `hidden`, when given, receives the final-LayerNorm state; `logits`, when
  // given, the tied-embedding projection of it.
  int32_t step(int32_t token, int64_t position,
               std::vector<float> *hidden = nullptr,
               std::vector<float> *logits = nullptr);

  // The checkpoint's decoding policy, from generation_config.json via the
  // packer: `suppress` is forbidden at every step, `begin` only at the first
  // generated token. Without this a greedy step is NOT the reference
  // implementation's step -- on a 3 s tone with whisper-tiny, unconstrained
  // argmax picks 522 (" (") and the reference picks 509 (" You"), because 522
  // is on the list. `begin` also contains <|endoftext|>, so without it a model
  // may answer an empty transcript on the first step.
  //
  // Both lists are REQUIRED: a container packed before they were stored is
  // refused rather than run with a policy of its own, because the difference
  // shows up only in the text and nowhere else.
  void set_suppression(std::vector<int32_t> suppress,
                       std::vector<int32_t> begin);
  bool has_suppression() const {
    return !suppress_.empty() || !begin_suppress_.empty();
  }

  // Greedy continuation of `prompt` (fed at positions 0..n-1), up to stop_id,
  // `max_new` generated tokens, or the checkpoint's own max_target_positions --
  // whichever comes first. Returns the generated ids only.
  //
  // The cap matters for more than tidiness: HF's generate() is compared against
  // this with ITS max_new_tokens, and a 31 s tone makes the model emit the same
  // token until it runs out of positions, so an uncapped chain and a capped
  // reference are not the same string and the comparison says nothing.
  // max_new <= 0 means "no cap beyond the position bound".
  std::vector<int32_t> greedy(const std::vector<int32_t> &prompt, int32_t stop_id,
                              int64_t max_new = 0);

  int64_t n_cached() const { return n_cached_; }
  int64_t n_source() const { return n_src_; }
  const NpuGemm &gemm() const { return g_; }
  const NpuAttention *attention_npu() const { return attn_; }
  void reset_timers();

private:
  const DecoderTier &tier(int64_t batch) const;
  void logits_from(const std::vector<float> &h, std::vector<float> *out,
                   int32_t *argmax);
  // One LayerNorm over `n` rows and one GELU over `n` values, on the array when
  // a design was given and on the host otherwise. The slot is indexed by the
  // caller as an argument and ignored on the host path, so the vectors are sized
  // either way.
  void norm_rows(float *x, int64_t n, const float *gamma, const float *beta,
                 size_t slot);
  void gelu_rows(float *x, int64_t n);

  npue::File &model_;
  NpuGemm g_;
  app::Pool &pool_;
  Geometry geom_;
  NpuEltwise *ln_ = nullptr, *gelu_ = nullptr;
  NpuAttention *attn_ = nullptr;
  NpuLogits *logit_ = nullptr;
  std::vector<DecoderTier> tiers_;
  int64_t step_batch_ = 0, prefill_batch_ = 0;

  std::vector<size_t> s_sqkv, s_sao, s_cq, s_ckv, s_cao, s_fu, s_fd;
  std::vector<const float *> b_sqkv, b_sao, b_cq, b_ckv, b_cao, b_fu, b_fd;
  std::vector<const float *> ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, ln3_gamma,
      ln3_beta;
  // The staged gamma|beta of each site, when LayerNorm runs on the array: three
  // per layer plus the final norm, staged once at session start.
  std::vector<size_t> s_ln1, s_ln2, s_ln3;
  size_t s_ln_final = 0;
  const float *dec_pos_ = nullptr;
  const float *final_gamma_ = nullptr, *final_beta_ = nullptr;
  const float *embed_tokens_ = nullptr;

  // Per layer: [n_cached, 2*d] K|V rows for self-attention, [n_src, 2*d] for
  // cross-attention. Cross is the expensive one -- 32 layers x 1500 x 2 x 1280
  // floats is 492 MB on large-v3 -- and it is built once per transcription.
  std::vector<std::vector<float>> self_kv_, cross_kv_;
  int64_t n_cached_ = 0, n_src_ = 0;
  // How many tokens this decoder has GENERATED, which is what the policy's
  // "begin" list is keyed on. reset() sets it back to 0, so a second
  // transcription re-primes the first-step suppression.
  int64_t n_generated_ = 0;
  std::vector<int32_t> suppress_, begin_suppress_;
  // The vocabulary projection is split over the pool, so its argmax needs one
  // reduction across workers.
  std::mutex argmax_mu_;
};

// Read the decoding policy out of the container and install it. Refuses a
// container that carries neither list -- see set_suppression for why that is a
// refusal and not a default.
inline std::vector<int32_t> container_int_list(const npue::File &f,
                                               const std::string &name) {
  std::vector<int32_t> out;
  if (!f.has(name)) return out;
  const auto sp = f.raw(name);
  if (sp.bytes % sizeof(int32_t) != 0)
    throw std::runtime_error(name + ": " + std::to_string(sp.bytes) +
                             " bytes is not a whole number of int32");
  const auto *p = static_cast<const int32_t *>(sp.data);
  out.assign(p, p + sp.bytes / sizeof(int32_t));
  return out;
}

inline void load_suppression(WhisperDecoder &dec, const npue::File &f,
                             const std::string &name = std::string()) {
  try {
    dec.set_suppression(container_int_list(f, "decoder.suppress_tokens"),
                        container_int_list(f, "decoder.begin_suppress_tokens"));
  } catch (const std::exception &e) {
    // The container's PATH in the message, not a placeholder: a refusal that
    // says "<this file>" sends the reader looking for a file called "<this
    // file>", and the one to re-pack is the one that was just opened.
    throw std::runtime_error(name.empty() ? e.what()
                                          : std::string(e.what()) + "\n  "
                                            "container: " + name);
  }
}

}  // namespace npue::whisper
