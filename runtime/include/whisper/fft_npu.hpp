//===- fft_npu.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's 400-point transform as one GEMM against a
// precomputed DFT matrix. SPDX-License-Identifier: Apache-2.0
//
// WHY A MATRIX AND NOT A KERNEL
// ----------------------------
// A direct transform IS a matrix: X[k] = sum_n x[n] exp(-2*pi*i*k*n/N). The
// host computes it the other way round -- an exact-size mixed-radix
// Cooley-Tukey over the factors 2 and 5, in fp64, because N = 400 is not a power
// of two and a 512-point transform of a zero-padded frame is a different
// spectrum (runtime/src/whisper/features.cpp says so, at length).
//
// The array has no FFT kernel in this project and writing one is a research
// task, not a wiring change: 400 points means 2^4 * 5^2, so a vectorised
// butterfly needs mixed-radix stages, a twiddle table in L1, and a dataflow the
// eltwise programs do not have. The GEMM needs none of that, because the MMAC
// already does the reduction: one dispatch multiplies a chunk of frames by the
// twiddle matrix and every bin of every frame comes out.
//
// WHAT IT COSTS, STATED PLAINLY
// -----------------------------
// The host's transform is accurate to about 1e-15 relative. This one multiplies
// bf16 twiddles, so a bin carries roughly 6e-3 relative error and a power
// spectrum about 1.2e-2 -- which reaches the mel as ~4e-4 after the projection's
// averaging, and the log as ~2e-4. That is inside the front end's own tolerance
// and it is NOT the same arithmetic as the host's, so the gate measures it
// against transformers rather than against our own fp64 path.
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

class NpuFft {
public:
  NpuFft(npu::Design &design, app::Pool &pool, int64_t n_fft, int64_t n_bins)
      : g_(design, pool, "fft"), pool_(pool), n_fft_(n_fft), n_bins_(n_bins) {}

  void set_streams(size_t slot, int64_t rows);

  // The twiddle matrix, staged once: it is a WEIGHT, exactly as the encoder's
  // GEMM operands are, and it is built here from the transform's own definition
  // rather than shipped in the container -- the container stores the model's
  // weights, and a DFT matrix is not one of them.
  void alloc_buffers();

  // `frames` is (frames, n_fft) row-major windowed samples; `power_out` is
  // (n_bins, frames) channels-first and receives re^2 + im^2.
  void run(const std::vector<float> &frames, int64_t n_frames, float *power_out);

  int64_t n_dispatch = 0;
  std::string note(int64_t n_frames) const;

private:
  NpuGemm g_;
  app::Pool &pool_;
  int64_t n_fft_ = 0, n_bins_ = 0;
  size_t instr_ = 0;
  int64_t rows_ = 0, k_ = 0, half_ = 0, n_ = 0;
  size_t wslot_ = 0;
  // The A operand is [rows, k] with the 48 padded input columns zeroed, so it
  // is a staging buffer and not the caller's frames: the GEMM reads n_real*k
  // elements and a 400-wide row is 48 short per row.
  std::vector<float> zero_, c_, arow_;
};

}  // namespace npue::whisper
