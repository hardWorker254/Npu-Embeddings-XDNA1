//===- whisper_model.hpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper model wrapper. Placeholder for future Whisper
// integration. Loads a .npue container and provides factory methods for the
// Whisper tokenizer and encoder.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "embed_models/model_loader.hpp"
#include "runtime/model.hpp"

namespace npue {

class WhisperModel : public Model {
public:
  explicit WhisperModel(const std::string &path);

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

class WhisperModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue
