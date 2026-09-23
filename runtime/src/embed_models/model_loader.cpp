//===- model_loader.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Model loader registry implementation.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "embed_models/model_loader.hpp"

// The registry functions and inline methods are defined in the header.
// This translation unit exists so that model_loader.hpp can be included
// without triggering inline definitions in every translation unit that
// uses it, and so that the registry is linkable from main.cpp.