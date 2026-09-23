//===- tokenizer.hpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- base tokenizer interface.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/types.hpp"

namespace npue {

class Tokenizer {
public:
  virtual ~Tokenizer() = default;
  virtual Encoded encode(const std::string &text, int max_len) const = 0;
  virtual size_t vocab_size() const = 0;
  virtual std::vector<std::string> tokenize(const std::string &text) const = 0;
};

}  // namespace npue