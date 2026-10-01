//===- vit_model.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=5 model wrapper. Loads a .npue container with arch
// "vit_patch16_prenorm_gelu" and exposes its geometry.
//
// WHY IT EXISTS AT ALL, GIVEN THAT npue::vit::Session IS THE REAL LOADER
// -----------------------------------------------------------------------
// Because Model is the interface everything that can HOLD a container
// understands, and an arch whose Model::make_tokenizer() throws is a statement
// worth making in one place rather than implied by its absence. A ViT has no
// tokenizer, no vocabulary and no pooling mode a caller chooses: its "tokenizer
// slot" is the image front end and its "pooling" is the CLS row, and both are
// stated here rather than left to a caller who would otherwise try to embed
// text with a classifier.
//
// The forward pass is NOT here. It belongs to npue::vit::Session, which needs a
// live npu::Design reference that only exists after the device is open -- the
// same ownership rule BertModel and WhisperModel both follow.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "embed_models/model_loader.hpp"
#include "runtime/model.hpp"
#include "vit/geometry.hpp"

namespace npue {

class VitModel : public Model {
public:
  explicit VitModel(const std::string &path);

  // REFUSES, by name. There is no text to tokenize for a model that takes
  // pixels, and a caller that reached here has a string and a container whose
  // every tensor is a weight -- so the honest answer is which one it wants,
  // not an empty vocabulary.
  std::unique_ptr<Tokenizer> make_tokenizer() override;

  int64_t hidden() const override;
  // The POSITION count, 197, not a design's sequence length. The Model
  // interface's `seq()` is "how many rows one forward pass has", and for this
  // architecture that is a property of the IMAGE (a 224px image at patch 16 is
  // 196 patches plus a CLS row) rather than of the caller's input or the
  // design. get_model_shape()'s comment about max_seq_len not being the right
  // answer for the embedding path does not apply here, and the difference is
  // recorded in the container's own config: n_patches and max_seq_len agree.
  int64_t seq() const override;
  const std::string &name() const override;

  const npue::vit::Geometry &geometry() const { return geom_; }

private:
  npue::File file_;
  std::string path_;
  npue::vit::Geometry geom_;
  int64_t hidden_ = 0;
  int64_t seq_ = 0;
  std::string name_;
};

class VitModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue
