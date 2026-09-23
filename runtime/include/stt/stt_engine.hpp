//===- stt_engine.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- STT engine interface.
// SPDX-License-Identifier: Apache-2.0
//
// Base class for speech-to-text engines. Adding a new STT engine
// means subclassing STTEngine and registering the model loader.
// No existing code is modified.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue {

class STTEngine {
public:
  virtual ~STTEngine() = default;
  virtual std::vector<float> transcribe(const std::string &audio_path) = 0;
};

}  // namespace npue