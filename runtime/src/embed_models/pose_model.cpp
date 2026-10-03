//===- pose_model.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 model wrapper. See pose_model.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "embed_models/pose_model.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"

namespace npue {

PoseModel::PoseModel(const std::string &path) : file_(path), path_(path) {
  name_ = std::filesystem::path(path).stem().string();
  geom_ = npue::pose::read_geometry(file_, path);
}

std::unique_ptr<Tokenizer> PoseModel::make_tokenizer() {
  throw std::runtime_error(
      path_ + ": this is a pose model (arch 6, " +
      std::string(npue::pose::kArch) +
      "), so it has no tokenizer, no vocabulary and no text input. Its front "
      "end is an image, and its output is a list of people with 17 keypoints "
      "each rather than one vector. Use `pose`: npuembeddings pose <model> "
      "<image.png>");
}

int64_t PoseModel::hidden() const {
  throw std::runtime_error(
      path_ + ": a pose model has no embedding width. Its output is a variable "
              "number of people, each with a box and " +
      std::to_string(npue::pose::kNumKeypoints) +
      " keypoints, so there is no single vector size to report and returning "
              "one (the head's channel count, say) would be a number that looks "
              "like a width and is not one. seq() is the related and "
              "meaningful number: the detection cells.");
}

int64_t PoseModel::seq() const { return geom_.n_anchors(); }

const std::string &PoseModel::name() const { return name_; }

bool PoseModelLoader::handles(const std::string &arch) const {
  return arch == npue::pose::kArch;
}

std::unique_ptr<Model> PoseModelLoader::load(const std::string &path) {
  return std::make_unique<PoseModel>(path);
}

namespace {

// Static-init registration, same rationale as BertModelLoader.
const bool kPoseLoaderRegistered = [] {
  register_model_loader(std::make_unique<PoseModelLoader>());
  return true;
}();

}  // namespace

}  // namespace npue