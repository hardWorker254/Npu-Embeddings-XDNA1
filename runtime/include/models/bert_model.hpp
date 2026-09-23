//===- bert_model.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- BERT model wrapper. Loads a .npue container with
// arch "bert_abs_gelu_postln" or "nomic_bert_rope_swiglu" and
// provides factory methods for the BERT encoder and WordPiece tokenizer.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "models/model_loader.hpp"
#include "runtime/model.hpp"
#include "tokenizers/wordpiece.hpp"
#include "encoders/bert_encoder.hpp"

namespace npue {

class BertModel : public Model {
public:
  explicit BertModel(const std::string &path);

  std::unique_ptr<Tokenizer> make_tokenizer() override;
  int64_t hidden() const override;
  int64_t seq() const override;
  const std::string &name() const override;

private:
  npue::File file_;
  std::string path_;
  int64_t hidden_ = 0;
  int64_t seq_ = 0;
  std::string name_;
};

class BertModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue