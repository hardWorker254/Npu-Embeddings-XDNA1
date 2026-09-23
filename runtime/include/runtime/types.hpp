//===- types.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- shared runtime types.
// SPDX-License-Identifier: Apache-2.0
//
// Aggregator header: pulls in the shared value types (Encoded) and the
// geometry globals. The geometry globals are OWNED by app_state.hpp, which
// is the single source of truth: set_model_shape() writes them, all encoders
// read them. This header re-exports app_state.hpp and device.hpp (BoMode)
// so one include gives an encoder everything it needs.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/app_state.hpp"    // geometry globals + pooling policy
#include "runtime/device.hpp"      // npu::BoMode

namespace npue {

struct Encoded {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;
  std::vector<int32_t> token_type_ids;
  int32_t n_tokens = 0;
  int32_t n_tokens_full = 0;
  bool truncated = false;
};

}  // namespace npue