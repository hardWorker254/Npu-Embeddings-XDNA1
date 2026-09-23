//===- whisper_model.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper model implementation. Placeholder for future
// Whisper integration; mirrors the embedding model wrappers so the real
// loader can drop in without touching the registry.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/whisper_model.hpp"

#include <memory>
#include <string>

#include "runtime/model.hpp"

namespace npue {

WhisperModel::WhisperModel(const std::string &path) : file_(path), path_(path) {
  name_ = "whisper";
}

std::unique_ptr<Tokenizer> WhisperModel::make_tokenizer() {
  return nullptr;
}

int64_t WhisperModel::hidden() const { return hidden_; }

int64_t WhisperModel::seq() const { return seq_; }

const std::string &WhisperModel::name() const { return name_; }

bool WhisperModelLoader::handles(const std::string &arch) const {
  // No Whisper architecture is loadable yet: the loader stays registered but
  // declines every arch until the real integration lands.
  (void)arch;
  return false;
}

std::unique_ptr<Model> WhisperModelLoader::load(const std::string &path) {
  return std::make_unique<WhisperModel>(path);
}

namespace {

// Static-init registration, same rationale as BertModelLoader.
const bool kWhisperLoaderRegistered = [] {
  register_model_loader(std::make_unique<WhisperModelLoader>());
  return true;
}();

}  // namespace

}  // namespace npue
