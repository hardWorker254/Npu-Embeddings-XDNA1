//===- gemma_model.cpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Gemma model implementation. Loads a .npue container
// with arch "gemma3_mqa_rope_geglu" and provides factory methods for
// the Gemma encoder and Gemma tokenizer.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "models/gemma_model.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"
#include "tokenizers/gemma.hpp"
#include "tokenizers/tokenizer_facade.hpp"

namespace npue {

GemmaModel::GemmaModel(const std::string &path) : file_(path), path_(path) {
  const std::string arch = file_.config_string("arch");
  if (arch != "gemma3_mqa_rope_geglu") {
    throw std::runtime_error(
        "GemmaModel given arch '" + arch + "', expected "
        "'gemma3_mqa_rope_geglu'");
  }
  hidden_ = file_.config_int("hidden");
  seq_ = file_.config_int("max_seq_len");
  name_ = arch;
}

std::unique_ptr<Tokenizer> GemmaModel::make_tokenizer() {
  // Same single selector as the BERT family: the facade reads the arch and
  // loads tokenizer.gemma_table out of the container.
  return std::make_unique<app::AnyTokenizer>(
      app::load_tokenizer(file_, path_));
}

int64_t GemmaModel::hidden() const { return hidden_; }

int64_t GemmaModel::seq() const { return seq_; }

const std::string &GemmaModel::name() const { return name_; }

bool GemmaModelLoader::handles(const std::string &arch) const {
  return arch == "gemma3_mqa_rope_geglu";
}

std::unique_ptr<Model> GemmaModelLoader::load(const std::string &path) {
  return std::make_unique<GemmaModel>(path);
}

namespace {

// Static-init registration, same rationale as BertModelLoader.
const bool kGemmaLoaderRegistered = [] {
  register_model_loader(std::make_unique<GemmaModelLoader>());
  return true;
}();

}  // namespace

}  // namespace npue