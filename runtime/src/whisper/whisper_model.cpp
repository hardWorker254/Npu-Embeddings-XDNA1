//===- whisper_model.cpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper model wrapper. It reads the container and hands out
// the tokenizer; the NPU forward pass belongs to the session, because it needs
// live npu::Design references that only exist after the device is open (the
// same ownership rule the BERT path follows -- see embed_models/model_loader.hpp).
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/whisper_model.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"
#include "tokenizers/whisper.hpp"
#include "whisper/geometry.hpp"

namespace npue {

WhisperModel::WhisperModel(const std::string &path) : file_(path), path_(path) {
  name_ = std::filesystem::path(path).stem().string();
  // The geometry comes from the CONTAINER, once, here: every later decision --
  // how many mel bins the front end builds, how wide the first operand is, how
  // many positions the encoder may see -- is read from these numbers rather
  // than from a constant that is right for exactly one model.
  geom_ = whisper::read_geometry(file_, path);
  hidden_ = geom_.d_model;
  seq_ = geom_.max_seq;
}

std::unique_ptr<Tokenizer> WhisperModel::make_tokenizer() {
  const auto t = file_.raw("tokenizer.whisper_table");
  return std::make_unique<WhisperTokenizer>(t.data, t.bytes);
}

int64_t WhisperModel::hidden() const { return hidden_; }

int64_t WhisperModel::seq() const { return seq_; }

const std::string &WhisperModel::name() const { return name_; }

bool WhisperModelLoader::handles(const std::string &arch) const {
  return arch == "whisper_encdec_gelu";
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
