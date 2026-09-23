//===- encoder.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- base encoder interface.
// SPDX-License-Identifier: Apache-2.0
//
// All encoder types (BERT, Gemma NPU, Gemma host, STT) implement this
// interface. The runtime holds a unique_ptr<Encoder> and calls encode()
// to get [batch][hidden] vectors.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue {

class Encoder {
public:
  virtual ~Encoder() = default;
  virtual std::vector<float> encode(const std::vector<std::string> &texts,
                                    const std::string &prefix,
                                    int64_t *tokens) = 0;
  virtual int64_t hidden() const = 0;
  virtual int64_t seq() const = 0;
};

}  // namespace npue