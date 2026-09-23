//===- device_impl.hpp --------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the private npu::Device::Impl definition.
// SPDX-License-Identifier: Apache-2.0
//
// Shared by device.cpp (which constructs it) and design.cpp (which needs the
// XRT device handle to register an xclbin). It lives here rather than in
// runtime/device.hpp so the public header stays XRT-free: design.cpp is the
// only core translation unit that needs the handle, and device.cpp is the
// only one that builds it.
//===----------------------------------------------------------------------===//

#pragma once

#include "runtime/device.hpp"

#include "xrt/xrt_device.h"

namespace npu {

struct Device::Impl {
  xrt::device device{0};
};

}  // namespace npu
