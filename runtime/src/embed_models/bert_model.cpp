//===- bert_model.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- BERT model implementation. Loads a .npue container
// with arch "bert_abs_gelu_postln", "nomic_bert_rope_swiglu" or
// "gte_new_rope_geglu" and provides a factory method for its tokenizer.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "embed_models/bert_model.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"
#include "tokenizers/tokenizer_facade.hpp"
#include "tokenizers/wordpiece.hpp"

namespace npue {

namespace {

bool bert_arch(const std::string &arch) {
  return arch == "bert_abs_gelu_postln" ||
         arch == "nomic_bert_rope_swiglu" ||
         arch == "gte_new_rope_geglu";
}

}  // namespace

BertModel::BertModel(const std::string &path) : file_(path), path_(path) {
  const std::string arch = file_.config_string("arch");
  if (!bert_arch(arch)) {
    throw std::runtime_error(
        "BertModel given arch '" + arch + "', expected "
        "'bert_abs_gelu_postln', 'nomic_bert_rope_swiglu' or "
        "'gte_new_rope_geglu'");
  }
  hidden_ = file_.config_int("hidden");
  seq_ = file_.config_int("max_seq_len");
  name_ = arch;
}

std::unique_ptr<Tokenizer> BertModel::make_tokenizer() {
  // The facade is the single tokenizer selector: it reads the container's
  // arch and picks WordPiece, XLM-R Unigram (gte_new_rope_geglu) or Gemma,
  // and falls back to the loose pre-0036 vocab.txt beside the container
  // rather than throwing on an older .npue.
  return std::make_unique<app::AnyTokenizer>(
      app::load_tokenizer(file_, path_));
}

int64_t BertModel::hidden() const { return hidden_; }

int64_t BertModel::seq() const { return seq_; }

const std::string &BertModel::name() const { return name_; }

bool BertModelLoader::handles(const std::string &arch) const {
  return bert_arch(arch);
}

std::unique_ptr<Model> BertModelLoader::load(const std::string &path) {
  return std::make_unique<BertModel>(path);
}

namespace {

// Register at static-init time so `npue::load_model()` finds the BERT-family
// loader without main.cpp having to enumerate loaders. The registry lives in
// a function-local static, so this cannot race the model_loaders() body.
const bool kBertLoaderRegistered = [] {
  register_model_loader(std::make_unique<BertModelLoader>());
  return true;
}();

}  // namespace

}  // namespace npue