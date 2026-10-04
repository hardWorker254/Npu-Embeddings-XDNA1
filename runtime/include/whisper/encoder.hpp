//===- encoder.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper encoder stack: log-mel -> conv1/conv2 (whisper/
// features.hpp) -> +positions -> N pre-LN layers -> final LayerNorm.
//
// THE SHAPE OF THE COMPUTE
// ------------------------
// The encoder's real input is (1500, d): conv2's stride 2 turns 3000 mel
// frames into 1500 positions, and 1500 is max_source_positions for every
// shipped size. The design set, however, was exported for a FIXED row count
// (512 for whisper-tiny) with a single batch tier, because the export tool
// tiles the array by rows and rows are what the tile geometry is built from.
//
// So the tensor is walked in row chunks of the design's own count: 512, 512,
// 476. The last chunk is short and its padded rows are zeroed (see NpuGemm) --
// they are a padded row, not a position, and they must not be attended to.
// Attention itself is a host pass over ALL positions at once, because the score
// matrix spans the whole tensor no matter how the GEMMs were chunked, and
// chunking attention would need a K/V cache this architecture has no use for.
//
// WHY PRE-LN, AND WHY IT IS NOT BERT'S ORDER
// ------------------------------------------
// Whisper is pre-LN: residual, LayerNorm, attention, add. BertEncoder is
// post-LN and also owns its LayerNorm and GELU in separate eltwise designs.
// These design sets have no eltwise streams at all -- kind stt in
// tools/data/npu_targets.json lists seven GEMM streams and nothing else -- so the
// norm and the activation are host passes here, fused into the row loop that
// feeds the next GEMM. That is a schedule change, not a numerical one: the
// arithmetic is the same fp32 LayerNorm and the same exact-erf GELU the front
// end already uses.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "whisper/attention_npu.hpp"
#include "whisper/eltwise.hpp"
#include "whisper/geometry.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

// Which instruction-stream slot runs which op, and how many rows one dispatch
// of each computes. Read off the design's own `streams` array by the caller --
// the slot numbers are the export's, and a set exported with different tiers
// has different ones.
struct EncoderStreams {
  size_t qkv = 0, attn_out = 0, ffn_up = 0, ffn_down = 0;
  int64_t rows = 0;
};

class WhisperEncoder {
public:
  WhisperEncoder(npue::File &model, npu::Design &design, app::Pool &pool,
                 const Geometry &geom);

  void set_streams(const EncoderStreams &s) { streams_ = s; }
  // Where LayerNorm and GELU run. Null is the host pass, which is what the
  // measured-faster path is at this width; a non-null design is one
  // --npu-ops code's worth of xclbin, and the numbers it computes are the
  // same operations in the same order, in bf16.
  void set_layernorm(NpuEltwise *ln) { ln_ = ln; }
  void set_gelu(NpuEltwise *gelu) { gelu_ = gelu; }
  // Where attention runs. Null is the host pass over the whole score matrix,
  // which is the measured-faster path at this width; non-null is the design
  // set's own attn_qk/attn_av streams and it computes the same function.
  void set_attention(NpuAttention *attn) { attn_ = attn; }

  // Stage every layer's operands and every LayerNorm from the container.
  // Returns the bytes staged, for the status line.
  size_t stage_all();

  // The forward pass. `conv_out` is the front end's (n_src, d_model) output
  // with no positional term yet -- added here, because the position table is
  // this stack's business and the front end has no business knowing it.
  // Returns (n_src, d_model) after the final LayerNorm, which is what
  // transformers' WhisperEncoder returns and therefore what the gate compares.
  std::vector<float> run(const std::vector<float> &conv_out, int64_t n_src);

  int64_t hidden() const { return geom_.d_model; }
  const Geometry &geometry() const { return geom_; }
  const NpuGemm &gemm() const { return g_; }
  const NpuAttention *attention_npu() const { return attn_; }
  void reset_timers();

private:
  // Walk `n` rows in chunks of the design's own row count.
  template <typename F> void chunks(int64_t n, F &&f) const {
    for (int64_t r0 = 0; r0 < n; r0 += streams_.rows)
      f(r0, std::min<int64_t>(n, r0 + streams_.rows));
  }

  // One LayerNorm over `n` rows, on the array when a design was given and on the
  // host otherwise. `slot` is the staged gamma|beta pair and is only read on the
  // array path; gamma/beta are only read on the host one, so the container's
  // pointers and the staged bytes cannot drift apart unnoticed: a design whose
  // width is not d_model is refused when it is opened.
  void norm_rows(float *x, int64_t n, const float *gamma, const float *beta,
                 size_t slot);
  void gelu_rows(float *x, int64_t n);

  npue::File &model_;
  NpuGemm g_;
  app::Pool &pool_;
  Geometry geom_;
  EncoderStreams streams_;
  NpuEltwise *ln_ = nullptr, *gelu_ = nullptr;
  NpuAttention *attn_ = nullptr;

  // Per layer: four tiled operands, four biases, two LayerNorm sites.
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  std::vector<const float *> ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
  // The staged gamma|beta of each site, when LayerNorm runs on the array. One
  // per site and per layer, staged once, for the same reason the GEMM operands
  // are: a site is 2*d floats and a dispatch does not save a transfer by
  // repeating it.
  std::vector<size_t> s_ln1, s_ln2;
  size_t s_ln_final = 0;
  const float *enc_pos_ = nullptr;
  const float *final_gamma_ = nullptr, *final_beta_ = nullptr;
};

}  // namespace npue::whisper
