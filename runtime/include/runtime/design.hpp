//===- design.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- xclbin design and instruction-stream management.
// SPDX-License-Identifier: Apache-2.0
//
// One Design owns one xclbin, its instruction stream, and its buffers.
// The xclbin is loaded once and kept: F1 prescribes one resident xclbin,
// and tasks/0010 measured ~150 us of fixed cost per dispatch.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "device.hpp"

namespace npu {

struct DesignInfo {
  std::string name;
  std::string kind;
  int64_t M = 0, K = 0, N = 0;
  int64_t seq = 0;
  std::vector<size_t> buffer_bytes;
  std::string b_layout_hash;
  size_t c_elem_bytes = 4;
  size_t a_elem_bytes = 2;
  bool c_is_int = false;
  int64_t tile_n = 0;
  int64_t cols = 0;
  bool emulate_bfp16 = false;
  bool datapath_recorded = false;
  // Which NPU generation this design was built for. `arch` is 1 or 2 and
  // `device` is the toolchain's device name ("npu1"/"npu2"); both are written
  // by tools/export_gemm_rtp.py. The *_recorded flags separate "the design
  // says npu1" from "the design predates the field", so the status line can
  // report UNRECORDED instead of guessing.
  int64_t arch = 0;
  std::string device;
  bool arch_recorded = false;
  bool device_recorded = false;
  std::string mlir_aie_version = "unavailable";
  std::string peano_version = "unavailable";
  std::string mlir_aie_git_head = "unavailable";
  bool toolchain_recorded = false;
};

class Device;

class Design {
public:
  Design(Device &dev, const std::string &dir);
  ~Design();
  Design(const Design &) = delete;
  Design &operator=(const Design &) = delete;

  const DesignInfo &info() const { return info_; }

  double t_submit = 0.0, t_wait = 0.0;
  int n_dispatch = 0;

  void run(const std::vector<const void *> &inputs, void *output);
  void dispatch_only();
  size_t load_instr(const std::string &path);
  void bind_instr(size_t slot);
  size_t stage(size_t arg_index, const void *data, size_t bytes);
  size_t stage_alloc(size_t arg_index, size_t bytes);
  size_t probe_alloc(size_t chunk_bytes, size_t count, bool verbose);
  void *slot_ptr(size_t arg_index, size_t slot);
  void bind(size_t arg_index, size_t slot);
  void *host_ptr(size_t index);
  void sync_to_device(size_t index, size_t bytes = 0);
  void sync_from_device(size_t index, size_t bytes = 0);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Device *dev_ = nullptr;
  DesignInfo info_;
  size_t output_index_ = 0;
};

}  // namespace npu