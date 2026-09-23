//===- whisper.cpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper STT engine stub.
// SPDX-License-Identifier: Apache-2.0
//
// transcribe() throws until Whisper integration is implemented.
//===----------------------------------------------------------------------===//

#include "stt/whisper.hpp"

#include <stdexcept>

namespace npue {

std::vector<float> WhisperEngine::transcribe(const std::string &audio_path) {
  throw std::runtime_error("STT not yet implemented");
}

}  // namespace npue