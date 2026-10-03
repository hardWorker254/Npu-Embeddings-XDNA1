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

namespace xrt {
class bo;
}

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
  // An ELTWISE design's own shape, read from the two keys only the eltwise
  // exporter writes: `row_capacity` is how many rows one dispatch computes and
  // `ln_eps` is the epsilon compiled into the LayerNorm kernel. They are named
  // apart from `cols` and `tile.rows` on purpose -- the GEMM exporter writes a
  // "rows" of its own inside `tile`, and a field that meant AIE rows on one
  // design set and matrix rows on another is a field nobody can check.
  int64_t row_capacity = 0;
  double ln_eps = 0.0;
  // The B panel's own geometry, out of `b_layout`: the tile and the MMAC
  // sub-tile. Read here rather than re-parsed by whoever builds an operand at
  // run time, because the sub-tile is PER GENERATION (npu1 s8/t4, npu2 s8/t8)
  // and a host-side operand that guesses it is right for one board and wrong on
  // the other -- same byte count, same hash, wrong products. `b_mac_*` are 0 on
  // a design.json that predates the fields, and the operand builders refuse
  // rather than default them.
  int64_t b_tile_k = 0;
  int64_t b_mac_s = 0, b_mac_t = 0;
  int64_t cols = 0;
  bool emulate_bfp16 = false;
  bool datapath_recorded = false;
  // THE MMAC THE DESIGN'S OWN VALUES SAY IT HAS, in the two words the status
  // line prints. `a_dtype` names the operand width and `emulate_bfp16` the
  // bfloat16 emulation on top of the bf16 path -- two different questions, and
  // only the first one decides whether the MMAC is int8.
  //
  // Reading the name off `emulate_bfp16` alone, as the three call sites used
  // to, calls an int8 design "bf16 MMAC": `--int8` writes
  // emulate_bfp16=false because int8 and bfp16 are mutually exclusive
  // datapath choices (exporters/gemm_rtp/geometry.py), so the flag's false is
  // an ABSENCE, not a bf16 claim. That is the reports-the-intention failure
  // these status lines exist to prevent (tasks/0042, 0081), pointed the other
  // way: the line named a datapath the loaded design did not have. A label
  // built from a value the loader already parsed cannot drift like that.
  const char *datapath_name() const {
    if (a_elem_bytes == 1) return "int8";
    return emulate_bfp16 ? "bfp16-emulated" : "bf16";
  }
  // Which NPU generation this design was built for. `arch` is 1 or 2 and
  // `device` is the toolchain's device name ("npu1"/"npu2"); both are written
  // by tools/export/export_gemm_rtp.py. The *_recorded flags separate "the design
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

  // Index of the output (C) argument: the last buffer, for every design in
  // this project. Public because a --pipeline lane has to bind its OWN output
  // buffer on a shared design, which means naming that argument (setup_encoder).
  size_t output_index() const { return output_index_; }

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

  // Sync a NAMED slot rather than whichever one is currently bound.
  //
  // sync_to_device()/sync_from_device() resolve their BO through the current
  // bind, so they are only safe to call while holding npu_mu -- and holding
  // npu_mu across a DMA is what serialises --pipeline lanes (measured: the
  // same A upload costs 75 us uncontended and 805 us at 4 lanes, i.e. 90% of
  // the cost was queueing, not the copy).
  //
  // These two address a lane's own xrt::bo directly, so they need no bind and
  // no mutex. Sound because dispatch_only() blocks on r.wait(): once a
  // dispatch has returned, the array is provably done with that lane's A and C
  // buffers, and no other lane ever names them.
  void sync_slot_to_device(size_t arg_index, size_t slot, size_t bytes = 0);
  void sync_slot_from_device(size_t arg_index, size_t slot, size_t bytes = 0);

private:
  xrt::bo &slot_bo(size_t arg_index, size_t slot);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Device *dev_ = nullptr;
  DesignInfo info_;
  size_t output_index_ = 0;
};

}  // namespace npu