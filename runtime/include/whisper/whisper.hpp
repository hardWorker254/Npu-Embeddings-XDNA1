//===- whisper.hpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper STT engine stub.
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder for future Whisper integration.
// transcribe() throws std::runtime_error until implemented.
//===----------------------------------------------------------------------===//

#pragma once

#include "whisper/stt_engine.hpp"

namespace npue {

class WhisperEngine : public STTEngine {
public:
  std::vector<float> transcribe(const std::string &audio_path) override;
};

}  // namespace npue