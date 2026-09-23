//===- stt_model.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- STT model stub. Placeholder for future Whisper
// integration. Returns false from handles() until implemented.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "models/model_loader.hpp"
#include "runtime/model.hpp"

namespace npue {

class STTModel : public Model {
public:
  explicit STTModel(const std::string &path);

  std::unique_ptr<Tokenizer> make_tokenizer() override;
  int64_t hidden() const override;
  int64_t seq() const override;
  const std::string &name() const override;

private:
  npue::File file_;
};

class STTModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue