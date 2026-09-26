//===- geometry.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper model geometry, read from the container.
//
// One struct, read ONCE from the .npue config, and every number the two stacks
// use comes from it. Nothing here may be a compiled-in constant: d_model 384
// and d_model 1280 differ in head count, in mel bins (80 vs 128) and in the
// width of every GEMM operand, so a hardcoded 384 or 80 is not a default -- it
// is a wrong answer for every model except the one it was written for.
//
// XRT-free, like the rest of the geometry, so a host-only check can read it.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"

namespace npue::whisper {

struct Geometry {
  int64_t d_model = 0;
  int64_t heads = 0;
  int64_t head_dim = 0;
  int64_t enc_layers = 0;
  int64_t dec_layers = 0;
  int64_t enc_intermediate = 0;
  int64_t dec_intermediate = 0;
  int64_t max_seq = 0;          // encoder positions the checkpoint has weights for
  int64_t max_target = 0;       // decoder positions likewise
  int64_t vocab = 0;
  int64_t mel_bins = 0;
  int64_t n_fft = 0;
  int64_t hop_length = 0;
  int64_t sample_rate = 0;
  int64_t n_samples = 0;
  int64_t nb_max_frames = 0;
  double ln_eps = 1e-5;
  double attn_scale = 0.0;      // 1/sqrt(head_dim)
  bool qkv_scale_folded = true; // 1/sqrt(head_dim) folded into every Q operand
};

// `label` is the container's path, used only to make the refusal name the file
// it came from. A missing key is an error rather than a default, because the
// alternative is a geometry this stack invents instead of reads.
inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + e.what() +
                             " -- not a Whisper container, or one packed by an "
                             "older packer (arch " + f.config_string("arch") +
                             ")");
  }
}

inline Geometry read_geometry(const npue::File &f,
                              const std::string &label = std::string()) {
  const std::string arch = f.config_string("arch");
  if (arch != "whisper_encdec_gelu")
    throw std::runtime_error(label + ": arch is '" + arch +
                             "', not 'whisper_encdec_gelu'");
  Geometry g;
  g.d_model = need_int(f, "d_model", label);
  g.heads = need_int(f, "num_heads", label);
  g.head_dim = need_int(f, "head_dim", label);
  g.enc_layers = need_int(f, "num_encoder_layers", label);
  g.dec_layers = need_int(f, "num_decoder_layers", label);
  g.enc_intermediate = need_int(f, "encoder_intermediate", label);
  g.dec_intermediate = need_int(f, "decoder_intermediate", label);
  g.max_seq = need_int(f, "max_seq_len", label);
  g.max_target = need_int(f, "max_target_positions", label);
  g.vocab = need_int(f, "vocab_size", label);
  g.mel_bins = need_int(f, "num_mel_bins", label);
  g.n_fft = need_int(f, "n_fft", label);
  g.hop_length = need_int(f, "hop_length", label);
  g.sample_rate = need_int(f, "sample_rate", label);
  g.n_samples = need_int(f, "n_samples", label);
  g.nb_max_frames = need_int(f, "nb_max_frames", label);
  g.ln_eps = f.config_double("layer_norm_eps");
  g.attn_scale = f.config_double("attention_scale");
  g.qkv_scale_folded = f.config_string("qkv_scale_folded") != "false";

  if (g.heads <= 0 || g.head_dim <= 0 || g.heads * g.head_dim != g.d_model)
    throw std::runtime_error(
        label + ": " + std::to_string(g.heads) + " heads x " +
        std::to_string(g.head_dim) + " is not d_model " +
        std::to_string(g.d_model) +
        ". HF uses head_dim = d_model // n_heads with integer division, so a "
        "pair that does not multiply back is a real truncated head and every "
        "attention scale in the model would be wrong.");
  if (std::abs(g.attn_scale - 1.0 / std::sqrt(static_cast<double>(g.head_dim))) >
      1e-6)
    throw std::runtime_error(label + ": attention_scale " +
                             std::to_string(g.attn_scale) + " is not " +
                             "1/sqrt(head_dim) for head_dim " +
                             std::to_string(g.head_dim));
  return g;
}

}  // namespace npue::whisper
