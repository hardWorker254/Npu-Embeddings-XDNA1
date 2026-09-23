//===- whisper.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper tokenizer stub. Placeholder for future Whisper
// integration. SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizers/tokenizer.hpp"

namespace npue {

class WhisperTokenizer : public Tokenizer {
public:
  Encoded encode(const std::string &text, int max_len) const override;
  size_t vocab_size() const override { return 0; }
  std::vector<std::string> tokenize(const std::string &text) const override;
};

}  // namespace npue
