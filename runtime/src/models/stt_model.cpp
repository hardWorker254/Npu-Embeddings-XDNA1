//===- stt_model.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- STT model stub. Placeholder for future Whisper
// integration. See stt_model.hpp for details.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "models/stt_model.hpp"

#include <stdexcept>
#include <string>

namespace npue {

STTModel::STTModel(const std::string &path) : file_(path) {}

std::unique_ptr<Tokenizer> STTModel::make_tokenizer() {
  return nullptr;
}

int64_t STTModel::hidden() const { return 0; }

int64_t STTModel::seq() const { return 0; }

const std::string &STTModel::name() const {
  static const std::string kName = "whisper";
  return kName;
}

bool STTModelLoader::handles(const std::string &arch) const {
  return false;
}

std::unique_ptr<Model> STTModelLoader::load(const std::string &path) {
  return std::make_unique<STTModel>(path);
}

}  // namespace npue