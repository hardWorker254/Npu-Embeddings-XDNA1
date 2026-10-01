//===- vit_model.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=5 model wrapper. See vit_model.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "embed_models/vit_model.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/model.hpp"

namespace npue {

VitModel::VitModel(const std::string &path) : file_(path), path_(path) {
  name_ = std::filesystem::path(path).stem().string();
  geom_ = npue::vit::read_geometry(file_, path);
  hidden_ = geom_.d_model;
  seq_ = geom_.n_pos;
}

std::unique_ptr<Tokenizer> VitModel::make_tokenizer() {
  throw std::runtime_error(
      path_ + ": this is an image classifier (arch 5, vit_patch16_prenorm_"
              "gelu), so it has no tokenizer, no vocabulary and no text "
              "input. Its front end is a " +
      std::to_string(geom_.image_size) + "px image and its classifier head "
              "takes the CLS row of the encoder. Use `classify`: "
              "npuembeddings classify <model> <image.png>");
}

int64_t VitModel::hidden() const { return hidden_; }

int64_t VitModel::seq() const { return seq_; }

const std::string &VitModel::name() const { return name_; }

bool VitModelLoader::handles(const std::string &arch) const {
  return arch == "vit_patch16_prenorm_gelu";
}

std::unique_ptr<Model> VitModelLoader::load(const std::string &path) {
  return std::make_unique<VitModel>(path);
}

namespace {

// Static-init registration, same rationale as BertModelLoader.
const bool kVitLoaderRegistered = [] {
  register_model_loader(std::make_unique<VitModelLoader>());
  return true;
}();

}  // namespace

}  // namespace npue
