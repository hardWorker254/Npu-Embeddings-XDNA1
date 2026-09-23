//===- device.cpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- XRT device open/close. See device.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/device.hpp"

#include <cstdint>

#include "device_impl.hpp"

namespace npu {

namespace {
BoMode g_bo_mode = BoMode::host_only;
size_t g_last_align = 0;
}  // namespace

void set_bo_mode(BoMode m) { g_bo_mode = m; }
BoMode bo_mode() { return g_bo_mode; }
size_t last_bo_alignment() { return g_last_align; }
void set_last_bo_alignment(size_t align) { g_last_align = align; }
const char *bo_mode_name() {
  switch (g_bo_mode) {
    case BoMode::host_only:    return "host_only";
    case BoMode::host_only_1m: return "host_only_1m";
    case BoMode::ext:          return "ext";
    case BoMode::ext_1m:       return "ext_1m";
  }
  return "?";
}

Device::Device() : impl_(std::make_unique<Impl>()) {}
Device::~Device() = default;

}  // namespace npu