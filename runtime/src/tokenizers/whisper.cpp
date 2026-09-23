//===- whisper.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper tokenizer stub. See tokenizers/whisper.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "tokenizers/whisper.hpp"

namespace npue {

Encoded WhisperTokenizer::encode(const std::string &text, int max_len) const {
  // WhisperTokenizer is a stub: no vocabulary, so every slot is the
  // conventional 0/[PAD] id rather than a real token. The loader never selects
  // it (handles() returns false), so this is compile-only until Whisper lands.
  Encoded e;
  e.input_ids.assign(static_cast<size_t>(max_len), 0);
  e.attention_mask.assign(static_cast<size_t>(max_len), 0);
  e.token_type_ids.assign(static_cast<size_t>(max_len), 0);
  e.n_tokens = 0;
  e.n_tokens_full = 0;
  e.truncated = false;
  return e;
}

std::vector<std::string> WhisperTokenizer::tokenize(
    const std::string &text) const {
  return {};
}

}  // namespace npue
