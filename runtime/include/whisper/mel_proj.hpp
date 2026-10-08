//===- mel_proj.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the slaney mel filter bank as a GEMM on the encoder set's
// own mel_proj stream. SPDX-License-Identifier: Apache-2.0
//
// WHAT IT IS, AND WHAT IT IS NOT
// -----------------------------
// The front end's mel step is `power @ bank`: a (201, n_mels) matrix times a
// (n_bins, frames) tensor, once per frame. That is a GEMM with a weight, so it
// runs on the array like any other -- one dispatch per 512 frames, six per
// window, 48M MAC in total. It is the CHEAPEST operation in this tree and the
// one with the least to show for it, and it is here because the question was
// which operations CAN run there.
//
// The log10 and the floor stay on the host: a logarithm is not a matrix, and
// the floor's maximum is over the FINISHED tensor, so neither can be a GEMM
// without turning the whole front end into something else.
//
// ORDER, WHICH IS THE MODEL'S
// ---------------------------
// features.cpp projects the POWER spectrum and takes the log afterwards. A
// projection written as `log_spec @ bank` is a different spectrogram, and
// putting this GEMM in the pipeline does not get to choose: `project_and_log`
// on the host is the same order, and the gate that compares against
// transformers is what says the two agree.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/pool.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

class NpuMelProj {
public:
  NpuMelProj(npu::Design &design, app::Pool &pool, int64_t n_bins,
             int64_t n_mels)
      : g_(design, pool, "mproj"), pool_(pool), n_bins_(n_bins), n_mels_(n_mels) {}

  // The stream's slot and row count, read off the loaded set by the caller.
  void set_streams(size_t slot, int64_t rows);

  // The A/C buffers, and the bank staged as the design's B operand. The bank is
  // a WEIGHT, not an activation: it is built once by mel_filter_bank() and
  // tiled once, the same way the encoder's GEMM operands are staged once.
  void alloc_buffers(const std::vector<double> &bank);

  // `power` is (n_bins, frames) channels-first, fp32; `mel_out` is
  // (n_mels, frames) channels-first and receives the first n_mels columns of
  // every C row.
  void run(const std::vector<float> &power, int64_t frames, float *mel_out);

  int64_t n_dispatch = 0;
  std::string note(int64_t frames) const;

private:
  NpuGemm g_;
  app::Pool &pool_;
  int64_t n_bins_ = 0, n_mels_ = 0;
  size_t instr_ = 0;
  int64_t rows_ = 0, k_ = 0, n_ = 0;
  size_t wslot_ = 0;
  std::vector<float> arow_, zero_;
};

}  // namespace npue::whisper
