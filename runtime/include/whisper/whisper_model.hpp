//===- whisper_model.hpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper model wrapper. Loads a .npue container, exposes its
// geometry and builds its tokenizer; the NPU forward pass lives in
// whisper/transcribe.hpp, which owns the device and the two design sets.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "embed_models/model_loader.hpp"
#include "runtime/model.hpp"
#include "whisper/geometry.hpp"

namespace npue {

class WhisperModel : public Model {
public:
  explicit WhisperModel(const std::string &path);

  std::unique_ptr<Tokenizer> make_tokenizer() override;
  int64_t hidden() const override;
  int64_t seq() const override;
  const std::string &name() const override;

  // d_model, heads, mel bins, layer counts, the attention scale and the
  // epsilon -- read once from the container. The encoder, the decoder and the
  // audio front end all take their numbers from this one struct.
  const whisper::Geometry &geometry() const { return geom_; }

private:
  npue::File file_;
  std::string path_;
  whisper::Geometry geom_;
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
