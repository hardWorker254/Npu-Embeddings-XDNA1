//===- mel_proj.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the slaney mel filter bank as a GEMM. See mel_proj.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/mel_proj.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/host_kernels.hpp"   // now_s

namespace npue::whisper {

void NpuMelProj::set_streams(size_t slot, int64_t rows) {
  instr_ = slot;
  rows_ = rows;
  if (rows <= 0)
    throw std::runtime_error("whisper mel projection: the design set's "
                             "mel_proj rows are zero, so no dispatch has a row "
                             "count");
}

void NpuMelProj::alloc_buffers(const std::vector<double> &bank) {
  const auto &in = g_.design().info();
  k_ = in.b_tile_k * ((n_bins_ + in.b_tile_k - 1) / in.b_tile_k);
  // N is padded to the DESIGN's granularity, not to tile_n: a design's N must be
  // a multiple of tile_n * AIE columns (128 on npu1), and rounding 80 up to 96 --
  // the tile multiple -- is not a shape any design can have. Rounding to 96 and
  // then to 128 collapses to 32 under a ceiling done in the wrong order, and the
  // panel that follows is a different matrix: the same class of bug the design
  // library's own asserts exist to catch, on the host side of the fence.
  const int64_t gran = in.tile_n * in.cols;
  n_ = gran * ((n_mels_ + gran - 1) / gran);
  if (static_cast<int64_t>(bank.size()) != n_bins_ * n_mels_)
    throw std::runtime_error(
        "whisper mel projection: the slaney bank is " +
        std::to_string(bank.size()) + " values, and this front end's is " +
        std::to_string(n_bins_) + " x " + std::to_string(n_mels_));
  g_.alloc_buffers();
  // Zeroed past n_bins and past n_mels, so a padded column contributes nothing
  // instead of whatever the last dispatch left in the staging matrix.
  std::vector<float> m(static_cast<size_t>(k_) * n_, 0.0f);
  for (int64_t b = 0; b < n_bins_; ++b)
    for (int64_t j = 0; j < n_mels_; ++j)
      m[static_cast<size_t>(b) * n_ + j] =
          static_cast<float>(bank[static_cast<size_t>(b) * n_mels_ + j]);
  const std::vector<uint16_t> tiled =
      tile_b_panel(m.data(), k_, n_, in.b_tile_k, in.tile_n, in.b_mac_s,
                   in.b_mac_t);
  wslot_ = g_.design().stage(1, tiled.data(), tiled.size() * sizeof(uint16_t));
  arow_.assign(static_cast<size_t>(rows_) * k_, 0.0f);
  zero_.assign(static_cast<size_t>(n_), 0.0f);
}

void NpuMelProj::run(const std::vector<float> &power, int64_t frames,
                     float *mel_out) {
  if (static_cast<int64_t>(power.size()) < n_bins_ * frames)
    throw std::runtime_error("whisper mel projection: the power spectrum holds " +
                             std::to_string(power.size() / n_bins_) +
                             " frames, " + std::to_string(frames) + " given");
  // One row of the A operand is one frame's 201 bins followed by zeros up to the
  // padded K. The transposition from the (n_bins, frames) layout is the whole
  // cost of this op on the host, and it is a copy, not arithmetic.
  std::vector<float> c(static_cast<size_t>(rows_) * n_, 0.0f);
  for (int64_t r0 = 0; r0 < frames; r0 += rows_) {
    const int64_t n = std::min<int64_t>(rows_, frames - r0);
    for (int64_t i = 0; i < n; ++i) {
      float *arow = arow_.data() + i * k_;
      for (int64_t b = 0; b < n_bins_; ++b)
        arow[b] = power[static_cast<size_t>(b) * frames + r0 + i];
      for (int64_t b = n_bins_; b < k_; ++b) arow[b] = 0.0f;
    }
    g_.run(instr_, arow_.data(), n, rows_, k_, wslot_, zero_.data(), n_, c.data());
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < n_mels_; ++j)
        mel_out[static_cast<size_t>(j) * frames + r0 + i] =
            c[static_cast<size_t>(i) * n_ + j];
    ++n_dispatch;
  }
}

std::string NpuMelProj::note(int64_t frames) const {
  return std::to_string(rows_) + " frames per dispatch, K = " + std::to_string(k_) +
         ", N = " + std::to_string(n_) + " (padded from " +
         std::to_string(n_bins_) + " x " + std::to_string(n_mels_) + "), " +
         std::to_string((frames + rows_ - 1) / rows_) +
         " dispatches per window";
}

}  // namespace npue::whisper
