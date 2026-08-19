//===- npu_device.hpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- dispatch a compiled design on the NPU through XRT.
// SPDX-License-Identifier: Apache-2.0
//
// One `Design` owns one xclbin, its instruction stream, and its buffers. The
// xclbin is loaded once and kept: F1 prescribes one resident xclbin, and
// tasks/0010 measured ~150 us of fixed cost per dispatch, so anything that
// reloads per call has already lost.
//
// The kernel signature comes from main_kernels.json in the build cache:
//   MLIR_AIE(opcode, instr*, ninstr, bo0, bo1, bo2, bo3, bo4)
// Buffer COUNT varies by design -- GEMM and GELU take (in, in, out) and
// (in, out); LayerNorm takes (in, params, out), because a core tile has only
// two input DMA channels and gamma+beta had to be packed into one buffer
// (tasks/0020). So buffers are a vector, sized from design.json.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace npu {

// How data buffers are allocated. Set before any Design is constructed.
// See npu_device.cpp -- this exists to make "does allocation flavour affect
// DMA throughput" a measurement rather than an argument.
enum class BoMode { host_only, host_only_1m, ext, ext_1m };
void set_bo_mode(BoMode m);
BoMode bo_mode();
const char *bo_mode_name();
// Alignment of the last data buffer actually handed back, in bytes. Reported
// so the mode's effect on addresses is visible rather than assumed.
size_t last_bo_alignment();

// Metadata the build step wrote alongside the xclbin. The runtime asserts on it
// rather than trusting that the binary on disk matches what it is asked for --
// tools/export_xclbin.py once handed the same xclbin to four different designs,
// and only a check like this makes that loud.
struct DesignInfo {
  std::string name;
  std::string kind;                  // gemm | eltwise | layernorm | softmax
  int64_t M = 0, K = 0, N = 0;       // gemm only
  // The sequence length the design was compiled for. Distinct from the
  // container's max_seq_len, which is how many position embeddings were
  // packed; 0 means an export that predates the field.
  int64_t seq = 0;
  std::vector<size_t> buffer_bytes;  // in declaration order

  // What layout this design's B operand must be in, as the same sha256 that
  // tools/npue.py stamps into every tiled tensor. Empty means the design.json
  // did not say -- which the runtime treats as a failure, not as permission.
  std::string b_layout_hash;

  // Element size of C as the design actually emits it: 4 for fp32, 2 when the
  // design was exported with `--c-bf16` and narrows on the core after the fp32
  // K reduction (tasks/0045). READ, never assumed -- a bf16 artifact and an
  // fp32 one differ only here, and guessing wrong reads every result at the
  // wrong stride.
  //
  // A design.json with no "c_dtype" is an export that predates the field, and
  // every one of those IS fp32, so 4 is the correct reading of silence rather
  // than a default papering over a missing value.
  size_t c_elem_bytes = 4;
};

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

class Design {
public:
  // `dir` holds final.xclbin, insts.bin and design.json from
  // tools/export_xclbin.py.
  Design(Device &dev, const std::string &dir);
  ~Design();
  Design(const Design &) = delete;
  Design &operator=(const Design &) = delete;

  const DesignInfo &info() const { return info_; }

  // Submit and wait charged separately, summed across all dispatches. Public
  // because the answer decides what to optimise and hiding it would mean
  // guessing again.
  double t_submit = 0.0, t_wait = 0.0;
  int n_dispatch = 0;

  // Host pointers in declaration order. Inputs are copied in and synced to the
  // device; outputs are synced back and copied out. Which is which is decided
  // by `output_index` -- everything else is an input.
  void run(const std::vector<const void *> &inputs, void *output);

  // Dispatches only, reusing whatever is already in the device buffers. This is
  // what makes a benchmark measure the NPU rather than the memcpy around it.
  void dispatch_only();

  // Load an ADDITIONAL instruction stream into this design's context and
  // return its slot id; slot 0 is the stream the design was constructed with.
  // bind_instr() selects which stream the next dispatch_only() replays.
  //
  // This is the mechanism behind the one-xclbin hypothesis (research/notes/
  // 0004 step 0): final.xclbin is the STATIC configuration and insts.bin is
  // the per-dispatch runtime sequence, so two operations whose static design
  // is identical are just two instruction streams over one hw_context -- and
  // switching between them should cost a dispatch, not a context switch.
  size_t load_instr(const std::string &path);
  void bind_instr(size_t slot);

  // Stage a buffer on the device once and keep it there; returns a slot id for
  // bind(). Slot 0 is always the design's own buffer.
  //
  // This exists for weights. Four GEMM designs serve six layers, so a design's
  // B buffer held a different layer's weights on every call and was refilled
  // from the mapped .npue each time -- 21 MB of memcpy per encode, of data that
  // never changes. Staging all 24 weight sets costs 21 MB of device buffers
  // once and removes that copy entirely (tasks/0024).
  size_t stage(size_t arg_index, const void *data, size_t bytes);

  // Allocate an EMPTY staged slot -- same slot semantics as stage(), no data.
  // This is what gives each pipeline of the two-encode overlap (tasks/0033)
  // its own A and C buffers on the shared design.
  size_t stage_alloc(size_t arg_index, size_t bytes);

  // Allocate `count` buffers of `chunk_bytes` through the same path the
  // encoder uses, touch each so the pages actually commit, and return how
  // many succeeded. bge-large needs ~1 GB of XRT buffers against MiniLM's
  // 175 MB, and finding the ceiling costs one command instead of a
  // four-xclbin build. The buffers are freed when `Design` is destroyed.
  size_t probe_alloc(size_t chunk_bytes, size_t count, bool verbose);

  // Host pointer of a SPECIFIC slot, independent of what is currently bound.
  // A pipeline converts into its own slot outside the dispatch lock; the
  // bind happens inside it.
  void *slot_ptr(size_t arg_index, size_t slot);

  // Choose which staged buffer argument `arg_index` dispatches with.
  void bind(size_t arg_index, size_t slot);

  // Direct access for callers that want to stage once and dispatch many times.
  // These act on whatever is currently bound to `index`.
  //
  // `bytes` = 0 syncs the whole buffer. The unified gemm_rtp design sizes its
  // buffers for the LARGEST stream (ffn_up's C is 50 MB), so a partial sync
  // of what the current stream actually touches is the difference between
  // syncing 6 MB and 50 MB on every qkv call.
  void *host_ptr(size_t index);
  void sync_to_device(size_t index, size_t bytes = 0);
  void sync_from_device(size_t index, size_t bytes = 0);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Device *dev_ = nullptr;      // kept so stage() can allocate more buffers
  DesignInfo info_;
  size_t output_index_ = 0;
};

}  // namespace npu
