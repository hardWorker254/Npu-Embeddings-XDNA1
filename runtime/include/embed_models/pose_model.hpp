//===- pose_model.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 model wrapper. Loads a .npue container with arch
// "yolov8_pose_c2f_silu_dfl" and exposes its geometry.
//
// THE FOURTH ARCH THAT IS NOT AN EMBEDDING
// ----------------------------------------
// Same reason as VitModel, one step further: this model has no tokenizer, no
// vocabulary, and its forward pass produces a variable number of people, not a
// vector. What it DOES have that ViT does not is a variable-length output, so
// `hidden()` is refused outright rather than given a number -- there is no
// correct value to return, and a caller that got one would index a vector that
// does not exist.
//
// The forward pass is not here either: it belongs to npue::pose::Session, which
// needs a live npu::Design for the array backend and does not for the host one.
// Which is the reason arch=6 is the first architecture in this tree whose
// default backend needs no device at all.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "embed_models/model_loader.hpp"
#include "pose/geometry.hpp"
#include "runtime/model.hpp"

namespace npue {

class PoseModel : public Model {
public:
  explicit PoseModel(const std::string &path);

  std::unique_ptr<Tokenizer> make_tokenizer() override;

  // REFUSED. A pose model's output is a list of people, so there is no width to
  // report; see the note above.
  int64_t hidden() const override;
  // The DETECTION CELLS one forward pass produces: 8400 for a 640px input.
  // The Model interface's `seq()` is "how many rows one forward pass has", and
  // for this architecture that is a property of the pyramid, so the same
  // reasoning as VitModel's seq() applies -- but unlike a patch count this one
  // is a property of the IMAGE SIZE, which is why the container is expected to
  // say input_size 640 rather than leaving it open.
  int64_t seq() const override;
  const std::string &name() const override;

  const npue::pose::Geometry &geometry() const { return geom_; }
  npue::File &file() { return file_; }

private:
  npue::File file_;
  std::string path_;
  npue::pose::Geometry geom_;
  std::string name_;
};

class PoseModelLoader : public ModelLoader {
public:
  bool handles(const std::string &arch) const override;
  std::unique_ptr<Model> load(const std::string &path) override;
};

}  // namespace npue