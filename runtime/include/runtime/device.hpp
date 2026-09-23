//===- device.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- XRT device wrapper.
// SPDX-License-Identifier: Apache-2.0
//
// One Device owns the XRT device handle. Design objects reference it.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace npu {

// How data buffers are allocated. Set before any Design is constructed.
// Defined here, with the device that owns it; runtime/types.hpp re-exports
// it for the shared-types consumers.
enum class BoMode { host_only, host_only_1m, ext, ext_1m };

void set_bo_mode(BoMode m);
BoMode bo_mode();
const char *bo_mode_name();
size_t last_bo_alignment();
void set_last_bo_alignment(size_t align);

class Device {
public:
  Device();
  ~Device();
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  struct Impl;
  Impl *impl() const { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace npu