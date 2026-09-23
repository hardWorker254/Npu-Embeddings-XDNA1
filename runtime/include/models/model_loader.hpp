//===- model_loader.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Model factory/registry. Maps architecture strings
// to the right ModelLoader at runtime. Adding a new model type means
// registering a new ModelLoader -- no existing code touched.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/encoder.hpp"
#include "runtime/model.hpp"
#include "tokenizers/tokenizer.hpp"

namespace npue {

// Model wraps a loaded .npue container and exposes its geometry and its
// tokenizer. It does NOT build encoders: a BertEncoder needs the live
// npu::Design references that only exist after the Runtime has opened the
// device and selected a design set, so encoder construction is owned by
// Runtime, not by Model. See the ownership rule in NEW_ARCHITECTURE.md.
class Model {
public:
  virtual ~Model() = default;
  virtual std::unique_ptr<Tokenizer> make_tokenizer() = 0;
  virtual int64_t hidden() const = 0;
  virtual int64_t seq() const = 0;
  virtual const std::string &name() const = 0;
};

// ModelLoader loads a Model from a .npue file.
class ModelLoader {
public:
  virtual ~ModelLoader() = default;
  virtual std::unique_ptr<Model> load(const std::string &path) = 0;
  virtual bool handles(const std::string &arch) const = 0;
};

// Registry: register ModelLoaders at static init time.
inline std::vector<std::unique_ptr<ModelLoader>> &model_loaders() {
  static std::vector<std::unique_ptr<ModelLoader>> loaders;
  return loaders;
}

inline void register_model_loader(std::unique_ptr<ModelLoader> loader) {
  model_loaders().push_back(std::move(loader));
}

inline std::unique_ptr<Model> load_model(const std::string &path) {
  npue::File f(path);
  const std::string arch = f.config_string("arch");
  for (auto &loader : model_loaders()) {
    if (loader->handles(arch)) return loader->load(path);
  }
  throw std::runtime_error("no model loader for arch '" + arch + "'");
}

}  // namespace npue