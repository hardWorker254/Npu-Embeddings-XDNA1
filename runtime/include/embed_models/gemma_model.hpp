//===- gemma_model.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Gemma model wrapper. Loads a .npue container with
// arch "gemma3_mqa_rope_geglu" and provides factory methods for the
// Gemma encoder and Gemma tokenizer.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "embed_models/model_loader.hpp"
#include "runtime/model.hpp"
#include "tokenizers/gemma.hpp"
#include "encoders/gemma_npu_encoder.hpp"
#include "encoders/gemma_host_encoder.hpp"

namespace npue {

class GemmaModel : public Model {
public:
  explicit GemmaModel(const std::string &path);

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

class GemmaModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue