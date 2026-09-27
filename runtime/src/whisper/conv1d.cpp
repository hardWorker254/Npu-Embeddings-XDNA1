//===- conv1d.cpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's two audio convolutions on the NPU. See
// whisper/conv1d.hpp for the schedule and why it needs no new design.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/conv1d.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "common/host_kernels.hpp"   // now_s
#include "common/app_state.hpp"     // gelu_erf_exact

namespace npue::whisper {

void NpuConv1d::stage(const std::string &label, const float *w,
                      const float *bias, int64_t in_ch, int64_t out_ch,
                      int64_t taps) {
  if (!w || !bias)
    throw std::runtime_error(label + ": the container has no conv weights");
  if (in_ch <= 0 || out_ch <= 0 || taps <= 0)
    throw std::runtime_error(label + ": conv with a zero dimension (" +
                             std::to_string(in_ch) + "x" +
                             std::to_string(out_ch) + ", " +
                             std::to_string(taps) + " taps)");
  if (out_ch != n_)
    throw std::runtime_error(
        label + ": " + std::to_string(out_ch) + " output channels, but the "
        "stream this conv runs on has N = " + std::to_string(n_) +
        ". The output width IS the GEMM's N here, so it cannot be padded or "
        "split: export a stream with N = " + std::to_string(out_ch) + " (see "
        "tools/npu_targets.json kinds.stt.streams).");
  if (k_ <= 0 || k_ % d_.info().b_tile_k || n_ % d_.info().tile_n)
    throw std::runtime_error(
        label + ": the stream's " + std::to_string(k_) + "x" +
        std::to_string(n_) + " does not tile by the design's (" +
        std::to_string(d_.info().b_tile_k) + "," +
        std::to_string(d_.info().tile_n) + ") tile");
  if (!b_slots_.empty())
    throw std::runtime_error(label + ": weights already staged");

  in_ch_ = in_ch;
  out_ch_ = out_ch;
  taps_ = taps;
  bias_ = bias;

  // ceil(in_ch * taps / K) blocks of K, and the LAST one is narrower than the
  // others: a conv2 at 3*d splits into three exact d-wide blocks, a conv1 at
  // 240 columns against K = 384 is one block with 144 columns of padding.
  const int64_t used = in_ch * taps;
  const int64_t n_blocks = (used + k_ - 1) / k_;
  b_slots_.reserve(static_cast<size_t>(n_blocks));
  std::vector<float> m(static_cast<size_t>(k_) * n_, 0.0f);
  for (int64_t b = 0; b < n_blocks; ++b) {
    const int64_t lo = b * k_;
    const int64_t hi = std::min(used, lo + k_);
    // Zeroed at the TOP of every block, including the last: the tail past `hi`
    // has to be zero for the same reason the A side's tail is, since a
    // previous block's values left there would be multiplied by the padding
    // columns of A and added into the result.
    std::fill(m.begin(), m.end(), 0.0f);
    for (int64_t j = lo; j < hi; ++j) {
      const int64_t c = j / taps_, tap = j - c * taps_;
      float *dst = m.data() + (j - lo) * n_;
      for (int64_t o = 0; o < n_; ++o)
        dst[o] = w[(o * in_ch_ + c) * taps_ + tap];
    }
    const std::vector<uint16_t> tiled =
        tile_b_panel(m.data(), k_, n_, d_.info().b_tile_k, d_.info().tile_n,
                     d_.info().b_mac_s, d_.info().b_mac_t);
    b_slots_.push_back(d_.stage(1, tiled.data(), tiled.size() * sizeof(uint16_t)));
  }
}

void NpuConv1d::run(const float *in, int64_t in_ch, Conv1dLayout layout,
                    int64_t t_in, int64_t t_out, int64_t stride, int64_t pad,
                    float *out) {
  if (b_slots_.empty())
    throw std::runtime_error("whisper conv: stage() before run()");
  if (in_ch != in_ch_)
    throw std::runtime_error("whisper conv: run() was given " +
                             std::to_string(in_ch) + " input channels but "
                             "stage() was told " + std::to_string(in_ch_));
  if (stride <= 0 || taps_ <= 0)
    throw std::runtime_error("whisper conv: stride " + std::to_string(stride) +
                             ", " + std::to_string(taps_) + " taps");
  // The output length is the CONVOLUTION'S, not the caller's: pass a t_out that
  // disagrees with padding and stride by one and every row after the first is
  // shifted, which is a transcript of the wrong audio rather than an error.
  const int64_t want_out =
      (t_in + 2 * pad - taps_) / stride + 1;
  if (t_out != want_out)
    throw std::runtime_error(
        "whisper conv: " + std::to_string(t_in) + " input positions, " +
        std::to_string(taps_) + " taps, stride " + std::to_string(stride) +
        ", padding " + std::to_string(pad) + " give " +
        std::to_string(want_out) + " output positions, not " +
        std::to_string(t_out));
  if (t_out <= 0)
    return;

  if (a_.size() != static_cast<size_t>(rows_) * k_)
    a_.assign(static_cast<size_t>(rows_) * k_, 0.0f);
  if (acc_.size() != static_cast<size_t>(rows_) * n_)
    acc_.assign(static_cast<size_t>(rows_) * n_, 0.0f);

  const int64_t used = in_ch_ * taps_;
  // The gather's two strides, derived from the layout ONCE here. Deriving them
  // at the call site is what let a swapped pair through: the two cases have the
  // same element count and the same shape, and the result is a convolution of
  // the wrong tensor with the right weights.
  const bool chans_first = layout == Conv1dLayout::channels_first;
  for (int64_t t0 = 0; t0 < t_out; t0 += rows_) {
    const int64_t n = std::min(rows_, t_out - t0);
    // The accumulator is zeroed ONCE per row chunk, not per K block: the
    // blocks accumulate into it, which is the whole point of the split.
    std::fill(acc_.begin(), acc_.begin() + static_cast<long>(n * n_), 0.0f);

    for (size_t bi = 0; bi < b_slots_.size(); ++bi) {
      const int64_t lo = static_cast<int64_t>(bi) * k_;
      const int64_t hi = std::min(used, lo + k_);
      // im2col into A. Every column past `hi` stays zero, which is what makes
      // the zero-padded B rows contribute nothing: the A side and the B side
      // are padded together or not at all.
      pool_.run([&](int w, int nw) {
        for (int64_t r = w; r < n; r += nw) {
          float *arow = a_.data() + r * k_;
          const int64_t centre = (t0 + r) * stride - pad;
          for (int64_t j = lo; j < hi; ++j) {
            const int64_t c = j / taps_, tap = j - c * taps_;
            const int64_t ti = centre + tap;
            arow[j - lo] =
                (ti >= 0 && ti < t_in)
                    ? (chans_first ? in[c * t_in + ti] : in[ti * in_ch + c])
                    : 0.0f;
          }
          for (int64_t j = hi - lo; j < k_; ++j) arow[j] = 0.0f;
        }
      });
      g_.run_accum(instr_, a_.data(), n, rows_, k_, b_slots_[bi], n_,
                   acc_.data());
    }

    // bias and the activation, in fp32 on the host, after the last block: the
    // same two operations in the same order as the CPU front end's
    // `acc + b` then gelu, so the only difference between the paths is bf16.
    pool_.run([&](int w, int nw) {
      for (int64_t r = w; r < n; r += nw) {
        const float *acc = acc_.data() + r * n_;
        float *o = out + (t0 + r) * n_;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= n_; j += 8)
          _mm256_storeu_ps(o + j,
                           _mm256_add_ps(_mm256_loadu_ps(acc + j),
                                         _mm256_loadu_ps(bias_ + j)));
#endif
        for (; j < n_; ++j) o[j] = acc[j] + bias_[j];
        for (int64_t x = 0; x < n_; ++x) o[x] = app::gelu_erf_exact(o[x]);
      }
    });
  }
}

std::vector<float> conv_front_end_npu(const MelSpec &mel, NpuConv1d &conv1,
                                      NpuConv1d &conv2, int hidden) {
  if (mel.frames != kMelFrames)
    throw std::runtime_error("whisper conv: expected " +
                             std::to_string(kMelFrames) + " mel frames, got " +
                             std::to_string(mel.frames));
  if (conv1.out_ch() != hidden || conv2.out_ch() != hidden)
    throw std::runtime_error("whisper conv: the staged convolutions produce " +
                             std::to_string(conv1.out_ch()) + " and " +
                             std::to_string(conv2.out_ch()) +
                             " channels, not d_model " +
                             std::to_string(hidden));
  if (conv1.in_ch() != mel.n_mels)
    throw std::runtime_error(
        "whisper conv: conv1 was staged for " + std::to_string(conv1.in_ch()) +
        " input channels and the mel has " + std::to_string(mel.n_mels));

  // conv1: the mel is (n_mels, 3000) channels-first, stride 1, and its output
  // is (3000, hidden) time-major because the GEMM's row is a position.
  std::vector<float> h1(static_cast<size_t>(kMelFrames) * hidden);
  conv1.run(mel.data.data(), mel.n_mels, Conv1dLayout::channels_first,
            mel.frames, mel.frames, 1, 1, h1.data());
  // conv2: reads that time-major output, so ITS input is time-major too, and it
  // halves the positions with stride 2. Its own output is (1500, hidden)
  // time-major, which is the tensor the encoder wants -- no permute.
  std::vector<float> out(static_cast<size_t>(kEncoderPositions) * hidden);
  conv2.run(h1.data(), hidden, Conv1dLayout::time_major, kMelFrames,
            kEncoderPositions, 2, 1, out.data());
  return out;
}

}  // namespace npue::whisper
