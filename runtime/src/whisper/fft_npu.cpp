//===- fft_npu.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the 400-point transform as a GEMM. See fft_npu.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/fft_npu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace npue::whisper {
namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}  // namespace

void NpuFft::set_streams(size_t slot, int64_t rows) {
  instr_ = slot;
  rows_ = rows;
  if (rows <= 0)
    throw std::runtime_error("whisper fft: the design set's dft400 rows are "
                             "zero, so no dispatch has a row count");
}

void NpuFft::alloc_buffers() {
  const auto &in = g_.design().info();
  k_ = in.b_tile_k * ((n_fft_ + in.b_tile_k - 1) / in.b_tile_k);
  // One half is the bins the power spectrum needs, padded up to what a
  // four-column design accepts; the two halves sit side by side in the panel, so
  // N is twice that.
  half_ = in.tile_n * in.cols * ((n_bins_ + in.tile_n * in.cols - 1) /
                                 (in.tile_n * in.cols));
  n_ = 2 * half_;
  g_.alloc_buffers();
  // W[k][n] = exp(-2*pi*i*k*n/N), real half in columns [0, half_), imaginary in
  // [half_, 2*half_). Zero past n_fft input samples and past n_bins bins, so a
  // padded column contributes nothing instead of a stale value.
  std::vector<float> m(static_cast<size_t>(k_) * n_, 0.0f);
  for (int64_t k = 0; k < n_bins_; ++k)
    for (int64_t n = 0; n < n_fft_; ++n) {
      const double ph = -kTwoPi * static_cast<double>(k) *
                        static_cast<double>(n) / static_cast<double>(n_fft_);
      m[static_cast<size_t>(n) * n_ + k] = static_cast<float>(std::cos(ph));
      m[static_cast<size_t>(n) * n_ + half_ + k] =
          static_cast<float>(std::sin(ph));
    }
  const std::vector<uint16_t> tiled =
      tile_b_panel(m.data(), k_, n_, in.b_tile_k, in.tile_n, in.b_mac_s,
                   in.b_mac_t);
  wslot_ = g_.design().stage(1, tiled.data(), tiled.size() * sizeof(uint16_t));
  zero_.assign(static_cast<size_t>(n_), 0.0f);
  c_.assign(static_cast<size_t>(rows_) * n_, 0.0f);
  arow_.assign(static_cast<size_t>(rows_) * k_, 0.0f);
}

void NpuFft::run(const std::vector<float> &frames, int64_t n_frames,
                 float *power_out) {
  if (static_cast<int64_t>(frames.size()) < n_frames * n_fft_)
    throw std::runtime_error("whisper fft: the windowed frames hold " +
                             std::to_string(frames.size() / n_fft_) +
                             " frames, " + std::to_string(n_frames) + " given");
  for (int64_t r0 = 0; r0 < n_frames; r0 += rows_) {
    const int64_t n = std::min<int64_t>(rows_, n_frames - r0);
    // A row of the operand is a frame's 400 samples followed by ZEROS up to the
    // padded K, so the staging buffer is [rows, k] and not the frames themselves:
    // the GEMM reads n_real*k elements, and handing it a 400-wide row buffer makes
    // it read 48 floats past the end of the caller's tensor on the first chunk.
    // That is a segfault, not a wrong number, and it is why windowed_frames_30s
    // being (frames, 400) is not enough on its own.
    for (int64_t i = 0; i < n; ++i) {
      float *arow = arow_.data() + i * k_;
      std::memcpy(arow, frames.data() + (r0 + i) * n_fft_,
                  static_cast<size_t>(n_fft_) * sizeof(float));
      for (int64_t j = n_fft_; j < k_; ++j) arow[j] = 0.0f;
    }
    g_.run(instr_, arow_.data(), n, rows_, k_, wslot_, zero_.data(), n_,
           c_.data());
    // re^2 + im^2 on the host: a square is not a matrix, and the MMAC has no
    // epilogue to put one in. 201 values per frame, one pass.
    for (int64_t i = 0; i < n; ++i) {
      const float *row = c_.data() + i * n_;
      float *dst = power_out + r0 + i;
      for (int64_t b = 0; b < n_bins_; ++b) {
        const float re = row[b], im = row[half_ + b];
        dst[static_cast<size_t>(b) * n_frames] = re * re + im * im;
      }
    }
    ++n_dispatch;
  }
}

std::string NpuFft::note(int64_t n_frames) const {
  return std::to_string(rows_) + " frames per dispatch, K = " +
         std::to_string(k_) + " (padded from " + std::to_string(n_fft_) +
         "), N = " + std::to_string(n_) + " (two halves of " +
         std::to_string(half_) + "), " +
         std::to_string((n_frames + rows_ - 1) / rows_) +
         " dispatches per window";
}

}  // namespace npue::whisper
