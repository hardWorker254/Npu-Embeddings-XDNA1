//===- features.hpp ----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper audio front end: log-mel spectrogram and the two
// convolutions that follow it.
// SPDX-License-Identifier: Apache-2.0
//
// Everything here is DEFINED BY HUGGINGFACE'S FEATURE EXTRACTOR, not chosen.
// The container stores the convolutions' weights; the spectrogram is a
// fixed function of the samples, and a container that computed a different one
// would be a model that transcribes nothing while looking perfectly healthy.
// So the constants are Whisper's own, they are named here, and the gate
// (tools/verify_whisper_features.py) holds the output against
// transformers' WhisperFeatureExtractor plus the checkpoint's own conv weights.
//
// THE PIPELINE, IN THE ORDER THAT MATTERS
// ---------------------------------------
//   samples            float32 mono, 16 kHz
//   zero-pad           to 30 s = 480000 samples (a 5 s file is NOT 3000 frames
//                      of its own audio followed by nothing; it is 5 s of audio
//                      and 25 s of silence, and the encoder always sees 1500
//                      positions)
//   reflect-pad        200 samples each side, WITHOUT repeating the edge sample
//                      (numpy's "reflect", not "symmetric")
//   frames             400 samples, hop 160, 3001 of them
//   window             periodic Hann, w[n] = 0.5 - 0.5cos(2*pi*n/400)
//                      -- w[399] is 6.17e-05 and NOT zero, which is what makes
//                      it periodic; a symmetric window is a different filter
//   FFT                exactly 400 points, NOT zero-padded to 512: a longer
//                      transform is a different spectrum
//   power              |X|^2, 201 bins
//   mel                201 x n_mels slaney filter bank, area-normalised
//   log10              of max(mel, 1e-10)
//   drop               the LAST frame -> 3000
//   floor              maximum(log_spec, global_max - 8.0)
//   scale              (x + 4.0) / 4.0
//
// The two convolutions are the model's, not ours: conv1 is
// Conv1d(n_mels, d, 3, padding=1), conv2 is Conv1d(d, d, 3, stride=2,
// padding=1), GELU (exact erf) after each, and the encoder consumes the result
// as (1500, d) -- frames by hidden -- which is the permute HF does before its
// first attention.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/pool.hpp"

namespace npue::whisper {

// Whisper's own numbers. 30 s at 16 kHz is 480000 samples; / 160 is 3000
// frames; conv2's stride 2 makes 1500 encoder positions; and 1500 is
// max_source_positions for every shipped size.
constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 30 * kSampleRate;   // 480000
constexpr int kHopLength = 160;
constexpr int kNfft = 400;
constexpr int kMelBins = kNfft / 2 + 1;        // 201, the bank's row count
constexpr int kMelFrames = 3000;                  // after dropping the last
constexpr int kEncoderPositions = kMelFrames / 2; // 1500

// (n_mels, frames), row-major, frame index fastest -- the same layout
// transformers returns, so a diff against it is a diff, not a transpose.
struct MelSpec {
  int n_mels = 0;
  int frames = 0;
  std::vector<float> data;

  float at(int mel, int frame) const {
    return data[static_cast<size_t>(mel) * frames + frame];
  }
};

// `samples` must already be mono at kSampleRate. Anything shorter than 30 s is
// zero-padded; anything longer is CUT at 30 s, and `frames_used` (when given)
// receives how many samples were real, so a caller can tell a 5 s file from a
// 30 s one instead of assuming.
// `pool` splits the frames across workers; see conv_front_end for why that is
// worth passing.
MelSpec log_mel_30s(const std::vector<float> &samples, int n_mels,
                    int64_t *samples_used = nullptr,
                    app::Pool *pool = nullptr);

//   windowed_frames_30s  (frames, 400) windowed samples, last frame dropped
//   power_30s            (n_bins, frames) power spectrum -- the host transform
//   project_and_log      bank @ power, then log10 of max(x, 1e-10)
//   mel_floor_scale      maximum over the finished tensor, floor at max - 8,
//                        then (x + 4) / 4
//
// The ORDER is the model's: the bank multiplies the power spectrum and the log
// comes after it. Writing the projection as `log_spec @ bank` instead is HF's
// formulation and produces a different spectrogram.
//
// power_30s is TWO steps on the array -- windowed_frames_30s, then a transform
// that is a GEMM rather than a recursive FFT -- so the tensor between them is
// exposed. See whisper/fft_npu.hpp for what the array's transform is and what it
// costs in precision.
std::vector<float> windowed_frames_30s(const std::vector<float> &samples,
                                       int n_mels, int64_t *samples_used = nullptr,
                                       app::Pool *pool = nullptr);
std::vector<float> power_30s(const std::vector<float> &samples, int n_mels,
                             int64_t *samples_used = nullptr,
                             app::Pool *pool = nullptr);
void project_and_log(const std::vector<float> &power, int n_mels,
                     app::Pool *pool, std::vector<float> &mel_out);
// log10 of max(x, 1e-10), in place. Its own entry point because the array's mel
// path takes the PROJECTION from a GEMM and the logarithm from here: a
// logarithm is not a matrix, and the two halves of the host's project_and_log
// have to be separately reachable for that to be true.
void mel_log_inplace(std::vector<float> &v, app::Pool *pool = nullptr);
void mel_floor_scale(std::vector<float> &mel);

// The slaney mel filter bank, (201, n_mels) row-major. Exposed because the
// encoder's first layer wants to reason about the band count, and because a
// gate that can compare the matrix itself localises a numerical divergence
// faster than one that only sees the spectrogram.
std::vector<double> mel_filter_bank(int n_mels, int sample_rate = kSampleRate,
                                   int n_fft = kNfft);

// conv1 -> GELU -> conv2 -> GELU -> permute, i.e. Whisper's whole audio front
// end. Weights are the container's, in Conv1d order: w1 is
// [d, n_mels, 3], w2 is [d, d, 3]. Returns (kEncoderPositions, d) row-major,
// which is the M x K the encoder's first GEMM wants.
//
// `pool` splits the output channels across workers. It is worth passing, and
// the reason is arithmetic: conv2 is d*d*3*1500 MACs, so 384 hidden costs
// 0.66 s per 30 s chunk single-threaded and large-v3's 1280 would cost about
// 7 s -- twenty seconds of arithmetic per minute of audio, before the first
// dispatch. It is NOT foldable into the first GEMM instead: a 3-tap
// convolution over a sliding window is a banded matmul, and the design's
// operand is a dense [M, K], so there is no arrangement of this that the
// exported GEMM can express.
std::vector<float> conv_front_end(const MelSpec &mel, const float *w1,
                                  const float *b1, const float *w2,
                                  const float *b2, int hidden,
                                  app::Pool *pool = nullptr);

}  // namespace npue::whisper
