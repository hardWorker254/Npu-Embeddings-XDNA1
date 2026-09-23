//===- gemma_host_encoder.hpp ------------------*- C++ -*-===//
//
// EmbeddingGemma-300M (arch=1), CPU-only forward pass.
// Moved from runtime/include/gemma_encode.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/encoder.hpp"
#include "runtime/model.hpp"
#include "tokenizers/gemma.hpp"

namespace npue {

class GemmaHostEncoder : public Encoder {
public:
  explicit GemmaHostEncoder(npue::File &model);

  std::vector<float> encode(const std::vector<std::string> &texts,
                                    const std::string &prefix,
                                    int64_t *tokens) override;
  int64_t hidden() const override;
  int64_t seq() const override;

  GemmaTokenizer tok;

  std::vector<float> encode_one(
      const std::string &text, int max_len,
      const std::string &prefix_name = GemmaTokenizer::default_prefix_name(),
      size_t index = 0, bool allow_truncation = false) const;

private:
  npue::File &model_;
  int64_t hidden_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, inter_ = 0,
          layers_ = 0, dense_hidden_ = 0, sliding_window_pattern_ = 6;
  double eps_ = 1e-6, rope_theta_ = 1e6, rope_theta_local_ = 1e4,
         query_pre_attn_scalar_ = 256.0;

  const float *w_embed_ = nullptr;
  const float *w_norm_ = nullptr;
  const float *w_dense2_ = nullptr;
  const float *w_dense3_ = nullptr;

  struct LayerPtrs {
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *q_norm, *k_norm;
    const float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn;
    const float *gate_proj, *up_proj, *down_proj;
  };
  std::vector<LayerPtrs> lp_;
};

}  // namespace npue