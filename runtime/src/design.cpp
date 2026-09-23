//===- design.cpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- xclbin design, instruction stream, buffer management.
// See design.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/design.hpp"

#include "device_impl.hpp"
#include "runtime/npu_contention.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace npu {
namespace {

size_t alignment_of(const void *p) {
  auto v = reinterpret_cast<uintptr_t>(p);
  if (!v) return 0;
  size_t a = 1;
  while (a < (1u << 21) && (v & a) == 0) a <<= 1;
  return a;
}

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (!f.read(reinterpret_cast<char *>(buf.data()), n))
    throw std::runtime_error("cannot read " + path);
  return buf;
}

[[maybe_unused]] size_t find_key(const std::string &t, const std::string &key) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos)
    throw std::runtime_error("design.json: missing " + key);
  return t.find(':', i) + 1;
}

int64_t json_int(const std::string &t, const std::string &key,
                 int64_t fallback, bool required = false) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) {
    if (required) throw std::runtime_error("design.json: missing " + key);
    return fallback;
  }
  return std::stoll(t.substr(t.find(':', i) + 1));
}

std::string json_str(const std::string &t, const std::string &key,
                     const std::string &fallback) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) return fallback;
  i = t.find('"', t.find(':', i) + 1) + 1;
  return t.substr(i, t.find('"', i) - i);
}

bool json_bool(const std::string &t, const std::string &key, bool fallback) {
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) return fallback;
  i = t.find(':', i) + 1;
  while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\t')) ++i;
  return t.compare(i, 4, "true") == 0;
}

std::vector<size_t> json_size_array(const std::string &t,
                                    const std::string &key) {
  std::vector<size_t> out;
  size_t i = t.find("\"" + key + "\"");
  if (i == std::string::npos) return out;
  i = t.find('[', i) + 1;
  while (true) {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == ',')) ++i;
    if (i >= t.size() || t[i] == ']') break;
    out.push_back(static_cast<size_t>(std::stoull(t.substr(i))));
    while (i < t.size() && t[i] != ',' && t[i] != ']') ++i;
  }
  return out;
}

xrt::bo make_data_bo(const xrt::device &dev, const xrt::kernel &k, size_t bytes,
                     int group) {
  constexpr size_t kMB = 1024 * 1024;
  const size_t padded = (bytes + kMB - 1) / kMB * kMB;
  xrt::bo b = [&]() -> xrt::bo {
    switch (npu::bo_mode()) {
      case BoMode::host_only:
        return xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, group);
      case BoMode::host_only_1m:
        return xrt::bo(dev, padded, XRT_BO_FLAGS_HOST_ONLY, group);
      case BoMode::ext:
        return xrt::ext::bo(dev, bytes);
      case BoMode::ext_1m:
        return xrt::ext::bo(dev, padded);
    }
    return xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, group);
  }();
  (void)k;
  set_last_bo_alignment(alignment_of(b.map<void *>()));
  return b;
}

}  // namespace

struct Design::Impl {
  xrt::hw_context ctx;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  std::vector<xrt::bo> alt_instr;
  std::vector<size_t> alt_instr_words;
  size_t active_instr = 0;
  std::vector<xrt::bo> bos;
  std::vector<std::vector<xrt::bo>> alt;
  std::vector<size_t> active;
  size_t n_instr_words = 0;

  xrt::bo &live(size_t i) {
    size_t s = active.at(i);
    return s == 0 ? bos.at(i) : alt.at(i).at(s - 1);
  }
};

Design::Design(Device &dev, const std::string &dir)
    : impl_(std::make_unique<Impl>()) {
  std::ifstream jf(dir + "/design.json");
  if (!jf) throw std::runtime_error("cannot open " + dir + "/design.json");
  std::stringstream ss;
  ss << jf.rdbuf();
  const std::string js = ss.str();

  info_.name = json_str(js, "name", dir);
  info_.kind = json_str(js, "kind", "gemm");
  info_.M = json_int(js, "M", 0);
  info_.K = json_int(js, "K", 0);
  info_.N = json_int(js, "N", 0);
  info_.seq = json_int(js, "seq", 0);
  info_.b_layout_hash = json_str(js, "b_layout_hash", "");

  {
    const std::string cd = json_str(js, "c_dtype", "f32");
    if (cd == "bf16") info_.c_elem_bytes = 2;
    else if (cd == "f32") info_.c_elem_bytes = 4;
    else if (cd == "i32") info_.c_elem_bytes = 4;
    else throw std::runtime_error(dir + ": unknown c_dtype '" + cd + "'");
    info_.c_is_int = (cd == "i32");

    const std::string ad = json_str(js, "a_dtype", "bf16");
    if (ad == "bf16") info_.a_elem_bytes = 2;
    else if (ad == "i8") info_.a_elem_bytes = 1;
    else throw std::runtime_error(dir + ": unknown a_dtype '" + ad + "'");

    info_.emulate_bfp16 = json_bool(js, "emulate_bfp16", false);
    info_.datapath_recorded =
        js.find("\"emulate_bfp16\"") != std::string::npos;

    info_.tile_n = json_int(js, "tile_n", 0);
    info_.cols = json_int(js, "cols", 0);

    // Which generation built this set (subtask 3). Top-level `arch`/`device`
    // appear before the per-stream `arch` fields, so the first hit is the
    // design's own.
    info_.arch_recorded = js.find("\"arch\"") != std::string::npos;
    info_.arch = json_int(js, "arch", 0);
    info_.device_recorded = js.find("\"device\"") != std::string::npos;
    info_.device = json_str(js, "device", "");
  }

  {
    std::ifstream tf(dir + "/toolchain.json");
    if (tf) {
      std::stringstream tss;
      tss << tf.rdbuf();
      const std::string ts = tss.str();
      info_.mlir_aie_version = json_str(ts, "mlir_aie_version", "unavailable");
      info_.peano_version = json_str(ts, "peano_version", "unavailable");
      info_.mlir_aie_git_head = json_str(ts, "mlir_aie_git_head", "unavailable");
      info_.toolchain_recorded = true;
    }
  }

  info_.buffer_bytes = json_size_array(js, "buffers");
  if (info_.buffer_bytes.empty()) {
    info_.buffer_bytes = {
        static_cast<size_t>(json_int(js, "bytes_a", 0, true)),
        static_cast<size_t>(json_int(js, "bytes_b", 0, true)),
        static_cast<size_t>(json_int(js, "bytes_c", 0, true))};
  }
  if (info_.buffer_bytes.size() < 2)
    throw std::runtime_error(dir + ": fewer than two buffers");
  output_index_ = info_.buffer_bytes.size() - 1;

  auto &d = *dev.impl();
  auto xclbin = xrt::xclbin(dir + "/final.xclbin");
  try {
    auto uuid = d.device.register_xclbin(xclbin);
    impl_->ctx = xrt::hw_context(d.device, uuid);
  } catch (const std::exception &e) {
    // NAME THE OPERATION, not the bare driver string (subtask 7). The usual
    // cause is the device hw_context budget being full -- another NPU process,
    // or an earlier run of ours that was killed before it could release its
    // contexts -- and the raw `err=-22` says none of that.
    const ContextBudget b = context_budget();
    const std::string limit =
        b.limit ? std::to_string(b.limit) + " (" + b.source + ")"
                : std::string("unknown (") + b.source + ")";
    throw std::runtime_error(
        "failed to open a hardware context for " + dir + ": " + e.what() +
        "\n  The device allows " + limit +
        " concurrent hw_contexts. A leftover NPU process still holding one is "
        "the usual cause -- `xrt-smi examine -r all` lists the holders. Close "
        "them, or pass --allow-contention if the contention is intended.");
  }

  std::string kname;
  for (const auto &k : xclbin.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kname = k.get_name(); break; }
  if (kname.empty())
    throw std::runtime_error(dir + ": no MLIR_AIE kernel in the xclbin");
  impl_->kernel = xrt::kernel(impl_->ctx, kname);

  auto instr = read_file(dir + "/insts.bin");
  impl_->n_instr_words = instr.size() / sizeof(uint32_t);
  impl_->bo_instr = xrt::bo(d.device, instr.size(), XCL_BO_FLAGS_CACHEABLE,
                                impl_->kernel.group_id(1));
  std::memcpy(impl_->bo_instr.map<void *>(), instr.data(), instr.size());
  impl_->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (size_t i = 0; i < info_.buffer_bytes.size(); ++i)
    impl_->bos.push_back(make_data_bo(
        d.device, impl_->kernel, info_.buffer_bytes[i],
        impl_->kernel.group_id(static_cast<int>(3 + i))));
  impl_->alt.resize(info_.buffer_bytes.size());
  impl_->active.assign(info_.buffer_bytes.size(), 0);
  dev_ = &dev;
}

Design::~Design() = default;

size_t Design::load_instr(const std::string &path) {
  auto &d = *dev_->impl();
  auto instr = read_file(path);
  xrt::bo bo(d.device, instr.size(), XCL_BO_FLAGS_CACHEABLE,
             impl_->kernel.group_id(1));
  std::memcpy(bo.map<void *>(), instr.data(), instr.size());
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  impl_->alt_instr.push_back(std::move(bo));
  impl_->alt_instr_words.push_back(instr.size() / sizeof(uint32_t));
  return impl_->alt_instr.size();
}

void Design::bind_instr(size_t slot) {
  if (slot > impl_->alt_instr.size())
    throw std::runtime_error(info_.name + ": no instr slot " +
                             std::to_string(slot));
  impl_->active_instr = slot;
}

size_t Design::stage(size_t arg_index, const void *data, size_t bytes) {
  if (bytes > info_.buffer_bytes.at(arg_index))
    throw std::runtime_error(info_.name + ": staged buffer for argument " +
                             std::to_string(arg_index) + " is " +
                             std::to_string(bytes) + " bytes, design allows " +
                             std::to_string(info_.buffer_bytes[arg_index]));
  auto &d = *dev_->impl();
  auto &slots = impl_->alt.at(arg_index);
  slots.push_back(make_data_bo(
      d.device, impl_->kernel, bytes,
      impl_->kernel.group_id(static_cast<int>(3 + arg_index))));
  std::memcpy(slots.back().map<void *>(), data, bytes);
  slots.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return slots.size();
}

size_t Design::probe_alloc(size_t chunk_bytes, size_t count, bool verbose) {
  auto &d = *dev_->impl();
  const int group = impl_->kernel.group_id(3);
  std::vector<xrt::bo> held;
  held.reserve(count);
  size_t ok = 0;
  for (size_t i = 0; i < count; ++i) {
    try {
      const double t0 = now_s();
      xrt::bo b = make_data_bo(d.device, impl_->kernel, chunk_bytes, group);
      std::memset(b.map<void *>(), 0, chunk_bytes);
      b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const double ms = (now_s() - t0) * 1e3;
      held.push_back(std::move(b));
      ++ok;
      if (verbose)
        std::printf("    %3zu: %6.1f MB cumulative %7.1f MB  (%6.1f ms)\n",
                    i + 1, chunk_bytes / 1e6, (i + 1) * chunk_bytes / 1e6, ms);
    } catch (const std::exception &e) {
      std::printf("    FAILED at buffer %zu (%.1f MB cumulative): %s\n",
                  i + 1, (i + 1) * chunk_bytes / 1e6, e.what());
      break;
    }
  }
  return ok;
}

size_t Design::stage_alloc(size_t arg_index, size_t bytes) {
  if (bytes > info_.buffer_bytes.at(arg_index))
    throw std::runtime_error(info_.name + ": stage_alloc for argument " +
                             std::to_string(arg_index) + " is " +
                             std::to_string(bytes) + " bytes, design allows " +
                             std::to_string(info_.buffer_bytes[arg_index]));
  auto &d = *dev_->impl();
  auto &slots = impl_->alt.at(arg_index);
  slots.push_back(make_data_bo(
      d.device, impl_->kernel, bytes,
      impl_->kernel.group_id(static_cast<int>(3 + arg_index))));
  return slots.size();
}

void *Design::slot_ptr(size_t arg_index, size_t slot) {
  return (slot == 0 ? impl_->bos.at(arg_index)
                    : impl_->alt.at(arg_index).at(slot - 1))
      .map<void *>();
}

void Design::bind(size_t arg_index, size_t slot) {
  if (slot > impl_->alt.at(arg_index).size())
    throw std::runtime_error(info_.name + ": no staged slot " +
                             std::to_string(slot));
  impl_->active.at(arg_index) = slot;
}

void *Design::host_ptr(size_t index) {
  return impl_->live(index).map<void *>();
}

void Design::sync_to_device(size_t index, size_t bytes) {
  xrt::bo &b = impl_->live(index);
  if (bytes == 0) bytes = b.size();
  b.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
}

void Design::sync_from_device(size_t index, size_t bytes) {
  xrt::bo &b = impl_->live(index);
  if (bytes == 0) bytes = b.size();
  b.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
}

void Design::dispatch_only() {
  const double t_enter = now_s();
  const size_t ai = impl_->active_instr;
  xrt::bo &instr = ai == 0 ? impl_->bo_instr : impl_->alt_instr[ai - 1];
  const uint32_t n_words = static_cast<uint32_t>(
      ai == 0 ? impl_->n_instr_words : impl_->alt_instr_words[ai - 1]);
  xrt::run r;
  switch (impl_->bos.size()) {
    case 2:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1));
      break;
    case 3:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1),
                        impl_->live(2));
      break;
    case 4:
      r = impl_->kernel(3, instr, n_words, impl_->live(0), impl_->live(1),
                        impl_->live(2), impl_->live(3));
      break;
    default:
      throw std::runtime_error(info_.name + ": unsupported buffer count");
  }
  t_submit += now_s() - t_enter;
  double t_wait0 = now_s();
  auto state = r.wait();
  t_wait += now_s() - t_wait0;
  ++n_dispatch;
  if (state != ERT_CMD_STATE_COMPLETED)
    throw std::runtime_error(info_.name + ": kernel state " +
                             std::to_string(static_cast<int>(state)));
}

void Design::run(const std::vector<const void *> &inputs, void *output) {
  if (inputs.size() != output_index_)
    throw std::runtime_error(info_.name + ": expected " +
                             std::to_string(output_index_) + " inputs");
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::memcpy(host_ptr(i), inputs[i], info_.buffer_bytes[i]);
    sync_to_device(i);
  }
  dispatch_only();
  sync_from_device(output_index_);
  std::memcpy(output, host_ptr(output_index_),
              info_.buffer_bytes[output_index_]);
}

}  // namespace npu