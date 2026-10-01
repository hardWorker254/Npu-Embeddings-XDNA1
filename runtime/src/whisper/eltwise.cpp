//===- eltwise.cpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's elementwise passes on the array. See eltwise.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/eltwise.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/host_kernels.hpp"   // bf16_fill, bf16_read, now_s

namespace npue::whisper {

void NpuEltwise::alloc_buffers() {
  const auto &in = d_.info();
  if (in.a_elem_bytes != 2 || in.c_elem_bytes != 2)
    throw std::runtime_error(
        in.name + ": this op reads bf16 in and writes bf16 out, and the design "
        "says " + std::to_string(in.a_elem_bytes * 8) + "/" +
        std::to_string(in.c_elem_bytes * 8) +
        " bits. Re-export it with tools/export/export_eltwise.py.");
  cols_ = in.cols;
  rows_ = in.row_capacity;
  if (cols_ <= 0 || rows_ <= 0)
    throw std::runtime_error(in.name +
                             "/design.json records no row width or row "
                             "capacity (`cols`, `row_capacity`): re-export "
                             "with tools/export/export_eltwise.py, which writes both");
  if (in.buffer_bytes.size() < 2)
    throw std::runtime_error(in.name + ": fewer than two buffers");
  if (in.buffer_bytes[0] < static_cast<size_t>(rows_) * cols_ * 2 ||
      in.buffer_bytes[in.buffer_bytes.size() - 1] <
          static_cast<size_t>(rows_) * cols_ * 2)
    throw std::runtime_error(
        in.name + ": " + std::to_string(rows_) + " rows of " +
        std::to_string(cols_) + " do not fit the design's own buffers (" +
        std::to_string(in.buffer_bytes[0]) + " and " +
        std::to_string(in.buffer_bytes[in.buffer_bytes.size() - 1]) +
        " bytes) -- the design and its design.json disagree");
  slot_a = d_.stage_alloc(0, in.buffer_bytes[0]);
  slot_c = d_.stage_alloc(d_.output_index(), in.buffer_bytes[d_.output_index()]);
}

// Both of these work in ELEMENTS, not rows, because the three designs disagree
// about what a row is: a LayerNorm row is d_model wide, a softmax row is a score
// row, and a GELU "row" is the whole flat span of activations (row_capacity 1).
// The capacity is rows_ * cols_ either way.
void NpuEltwise::fill_input(const float *x, int64_t n) {
  const int64_t cap = rows_ * cols_;
  if (n < 0 || n > cap)
    throw std::runtime_error(d_.info().name + ": " + std::to_string(n) +
                             " elements into a " + std::to_string(cap) +
                             "-element dispatch");
  auto *abuf = static_cast<uint16_t *>(d_.slot_ptr(0, slot_a));
  app::bf16_fill(abuf, x, static_cast<size_t>(n));
  // The tail, zeroed rather than left stale: the program fills its WHOLE input
  // buffer, so a row still holding the previous call's activations is read as
  // this call's data. Its result is discarded, but it has to be a number.
  if (n < cap)
    std::memset(abuf + n, 0, static_cast<size_t>(cap - n) * sizeof(uint16_t));
}

void NpuEltwise::read_back(float *x, int64_t n) const {
  const auto *cbuf = static_cast<const uint16_t *>(
      d_.slot_ptr(d_.output_index(), slot_c));
  app::bf16_read(x, cbuf, static_cast<size_t>(n));
}

void NpuEltwise::dispatch(size_t bytes, bool with_params, size_t param_slot) {
  const double t0 = app::now_s();
  {
    // bind -> sync-to -> dispatch -> sync-from, one window under the lock: the
    // same mutex the GEMM layer takes, because `active` on a Design is shared
    // mutable state and the decoder and the encoder can both want this design.
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d_.bind(0, slot_a);
    if (with_params) d_.bind(1, param_slot);
    d_.bind(d_.output_index(), slot_c);
    // The A buffer was written through map(); without this the device reads
    // whatever the host cache last flushed, which looks like a design that
    // computed nothing.
    d_.sync_to_device(0, bytes);
    d_.dispatch_only();
    d_.sync_from_device(d_.output_index(), bytes);
  }
  t_dispatch += app::now_s() - t0;
  ++n_dispatch;
}

size_t NpuEltwise::stage_params(const std::vector<float> &gamma_beta) {
  if (kind_ != EltwiseKind::LayerNorm)
    throw std::runtime_error("whisper eltwise: " + d_.info().name +
                             " takes no parameter operand");
  if (static_cast<int64_t>(gamma_beta.size()) != 2 * cols_)
    throw std::runtime_error(
        d_.info().name + ": gamma|beta is " +
        std::to_string(gamma_beta.size()) + " floats, this design's row is " +
        std::to_string(cols_) + " wide and wants 2*" + std::to_string(cols_) +
        ". The container's d_model is not the one this design was exported "
        "for.");
  return d_.stage(1, gamma_beta.data(),
                  gamma_beta.size() * sizeof(float));
}

void NpuEltwise::layernorm(float *x, int64_t n_real, size_t param_slot) {
  if (kind_ != EltwiseKind::LayerNorm)
    throw std::runtime_error("whisper eltwise: layernorm() on a " +
                             d_.info().name + " design");
  for (int64_t r0 = 0; r0 < n_real; r0 += rows_) {
    const int64_t n = std::min<int64_t>(rows_, n_real - r0) * cols_;
    fill_input(x + r0 * cols_, n);
    dispatch(static_cast<size_t>(rows_) * cols_ * 2, true, param_slot);
    read_back(x + r0 * cols_, n);
  }
}

void NpuEltwise::softmax(float *x, int64_t n_rows) {
  if (kind_ != EltwiseKind::Softmax)
    throw std::runtime_error("whisper eltwise: softmax() on a " +
                             d_.info().name + " design");
  for (int64_t r0 = 0; r0 < n_rows; r0 += rows_) {
    const int64_t n = std::min<int64_t>(rows_, n_rows - r0) * cols_;
    fill_input(x + r0 * cols_, n);
    // softmax's tail row is all zeros, which is a UNIFORM distribution over
    // cols elements, and a uniform row of a matrix that is never read back is
    // harmless -- unlike LayerNorm's tail, which would be beta. It is discarded
    // either way.
    dispatch(static_cast<size_t>(rows_) * cols_ * 2, false, 0);
    read_back(x + r0 * cols_, n);
  }
}

void NpuEltwise::gelu(float *x, int64_t n_real) {
  if (kind_ != EltwiseKind::Gelu)
    throw std::runtime_error("whisper eltwise: gelu() on a " + d_.info().name +
                             " design");
  const int64_t per = rows_ * cols_;   // elements one dispatch covers
  for (int64_t r0 = 0; r0 < n_real; r0 += per) {
    const int64_t n = std::min<int64_t>(per, n_real - r0);
    fill_input(x + r0, n);
    dispatch(static_cast<size_t>(per) * 2, false, 0);
    read_back(x + r0, n);
  }
}

}  // namespace npue::whisper
