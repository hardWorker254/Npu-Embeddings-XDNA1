//===- features.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper audio front end. See whisper/features.hpp for the
// pipeline and why each step is the one it is.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/features.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

#include "common/app_state.hpp"  // gelu_erf_exact

namespace npue::whisper {
namespace {

constexpr double kPi = 3.14159265358979323846;

// -- an exact-size FFT -------------------------------------------------------
//
// 400 = 2^4 * 5^2, and it has to be EXACTLY 400: zero-padding the frame to 512
// and running a power-of-two FFT is a different transform of a different
// signal, and the resulting spectrogram is wrong in a way no tolerance
// argument survives. So this is a mixed-radix Cooley-Tukey over the factors 2
// and 5, not a radix-2 with a pad.
//
// Recursive rather than iterative because the permutation falls out of the
// decimation and a 400-point transform is small enough that the recursion is
// not the bottleneck; the arithmetic is.
void fft_rec(std::vector<std::complex<double>> &a, size_t n) {
  if (n <= 1) return;
  // The radix is the SMALLEST prime factor, so: 2 if even, else 5. Growing p
  // multiplicatively instead (2, 5, 10, 20, ...) walks off the end of size_t
  // on a size this does not factor, and the division by zero that follows is a
  // SIGFPE with no message -- hence the explicit refusal.
  size_t p = (n % 2 == 0) ? 2 : 5;
  if (n % p != 0)
    throw std::runtime_error(
        "whisper features: FFT length " + std::to_string(n) +
        " is not 2^a*5^b; this transform only covers the sizes Whisper's "
        "n_fft produces");
  const size_t m = n / p;

  std::vector<std::vector<std::complex<double>>> sub(p);
  for (size_t q = 0; q < p; ++q) {
    sub[q].resize(m);
    for (size_t k = 0; k < m; ++k) sub[q][k] = a[k * p + q];
    fft_rec(sub[q], m);
  }
  // X[k + m*t] = sum_q A_q[k] * exp(-2*pi*i*q*(k + m*t)/n), and the exponent
  // splits: exp(-2*pi*i*qk/n) * exp(-2*pi*i*qmt/n). The first factor is the
  // usual per-level twiddle, the second depends only on (q, t) and is hoisted.
  // Evaluating the phase inside the loop instead costs four trig calls per
  // butterfly and made a 30 s chunk take a second -- which is a second per
  // chunk, on the request path, for a long recording.
  const double theta = -2.0 * kPi / static_cast<double>(n);
  for (size_t q = 0; q < p; ++q)
    for (size_t k = 0; k < m; ++k) {
      const double ph = theta * static_cast<double>(q * k);
      sub[q][k] *= std::complex<double>(std::cos(ph), std::sin(ph));
    }
  std::vector<std::vector<std::complex<double>>> v(p, std::vector<std::complex<double>>(p));
  for (size_t q = 0; q < p; ++q)
    for (size_t t = 0; t < p; ++t) {
      const double ph = theta * static_cast<double>(q * m * t);
      v[q][t] = std::complex<double>(std::cos(ph), std::sin(ph));
    }
  std::vector<std::complex<double>> out(n);
  for (size_t k = 0; k < m; ++k)
    for (size_t t = 0; t < p; ++t) {
      std::complex<double> acc(0.0, 0.0);
      for (size_t q = 0; q < p; ++q) acc += sub[q][k] * v[q][t];
      out[k + m * t] = acc;
    }
  a.swap(out);
}

std::vector<double> power_spectrum(const double *frame, int n) {
  std::vector<std::complex<double>> buf(n);
  for (int i = 0; i < n; ++i) buf[i] = std::complex<double>(frame[i], 0.0);
  fft_rec(buf, static_cast<size_t>(n));
  std::vector<double> out(static_cast<size_t>(n / 2 + 1));
  for (size_t i = 0; i < out.size(); ++i) out[i] = std::norm(buf[i]);
  return out;
}

// numpy's pad(..., mode="reflect"): the edge sample is NOT repeated, so for
// [a, b, c] with one sample of padding the result is [b, a, b, c, b]. Getting
// this wrong shifts the first two frames and nothing else, which is exactly the
// kind of error a "close enough" check waves through.
std::vector<double> reflect_pad(const std::vector<double> &x, size_t pad) {
  if (pad == 0) return x;
  if (x.size() < pad)
    throw std::runtime_error(
        "whisper features: " + std::to_string(x.size()) +
        " samples is too short to reflect-pad by " + std::to_string(pad) +
        "; the 30 s zero-padding runs first, so this cannot happen for a "
        "padded input and means the caller skipped it");
  std::vector<double> out(x.size() + 2 * pad);
  // out[i] = x[pad - i], NOT x[pad - 1 - i]: numpy's reflect for [0,1,2,3]
  // with pad 2 is [2,1,0,1,2,3,2,1], so the leftmost padded sample is the
  // one TWO back. The off-by-one version is wrong in the first two frames
  // only, in the top mel bands, by up to 0.48 in the final features -- and
  // cosine similarity stays at 1e-6 throughout, so a similarity gate
  // waves it through and only a max-abs comparison sees it.
  for (size_t i = 0; i < pad; ++i) out[i] = x[pad - i];
  std::copy(x.begin(), x.end(), out.begin() + static_cast<long>(pad));
  // The right side is the mirror of the left with the direction FLIPPED, and it
  // starts AFTER the data: out[pad + n + i] = x[n - 2 - i]. Two ways to get
  // this wrong, both of which look fine until the audio fills all 30 s: writing
  // to out[last - i] reverses the run, and writing to out[n + i] overwrites the
  // last `pad` real samples. A fixture shorter than 30 s cannot see either --
  // its last frames are silence -- so the corpus needs a full-length case.
  for (size_t i = 0; i < pad; ++i)
    out[pad + x.size() + i] = x[x.size() - 2 - i];
  return out;
}

// -- the mel scale -----------------------------------------------------------
// Slaney's, as transformers computes it: linear below 1000 Hz, logarithmic
// above, with the two constants that make the junction continuous. HTK's
// 2595*log10(1+f/700) is a different function and produces a different bank.
double hz_to_mel_slaney(double hz) {
  const double min_log_hz = 1000.0, min_log_mel = 15.0;
  const double logstep = 27.0 / std::log(6.4);
  const double linear = 3.0 * hz / 200.0;
  if (hz < min_log_hz) return linear;
  return min_log_mel + std::log(hz / min_log_hz) * logstep;
}

double mel_to_hz_slaney(double mel) {
  const double min_log_hz = 1000.0, min_log_mel = 15.0;
  const double logstep = std::log(6.4) / 27.0;
  const double linear = 200.0 * mel / 3.0;
  if (mel < min_log_mel) return linear;
  return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

}  // namespace

std::vector<double> mel_filter_bank(int n_mels, int sample_rate, int n_fft) {
  if (n_mels <= 0) throw std::runtime_error("whisper features: n_mels must be > 0");
  const int n_bins = n_fft / 2 + 1;
  // n_mels + 2 centre frequencies: the two extra ones are the outer edges the
  // first and last triangles climb to.
  std::vector<double> centre(static_cast<size_t>(n_mels) + 2);
  const double mel_min = hz_to_mel_slaney(0.0);
  const double mel_max = hz_to_mel_slaney(sample_rate / 2.0);
  for (int i = 0; i < n_mels + 2; ++i)
    centre[static_cast<size_t>(i)] =
        mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i / (n_mels + 1));

  // Triangular in FREQUENCY space (transformers' default:
  // triangularize_in_mel_space=False), each filter zero outside its own band.
  std::vector<double> bank(static_cast<size_t>(n_bins) * n_mels, 0.0);
  for (int j = 0; j < n_mels; ++j) {
    const double lo = centre[static_cast<size_t>(j)];
    const double mid = centre[static_cast<size_t>(j) + 1];
    const double hi = centre[static_cast<size_t>(j) + 2];
    // Slaney area normalisation: constant energy per channel, so a narrow
    // low band is not quieter than a wide high one purely by being narrow.
    const double enorm = 2.0 / (hi - lo);
    for (int k = 0; k < n_bins; ++k) {
      const double hz = sample_rate / 2.0 * k / (n_bins - 1);
      double w = 0.0;
      if (hz > lo && hz < hi)
        w = (hz <= mid) ? (hz - lo) / (mid - lo) : (hi - hz) / (hi - mid);
      bank[static_cast<size_t>(k) * n_mels + j] = w * enorm;
    }
  }
  return bank;
}

// The WINDOWED FRAMES, (frames, kNfft) row-major, one frame per row, last
// frame dropped. Row-major by frame because that is the GEMM's A operand: the
// 400-point transform reads a frame as one contiguous row, and transposing it
// here would be the only data movement the front end needs on either path.
//
// Split out of power_30s for the same reason log_mel_30s is split three ways: the
// transform is a separable step, and on the array it is a different algorithm
// (a direct transform against a precomputed matrix) rather than a different
// place to run the same one.
std::vector<float> windowed_frames_30s(const std::vector<float> &samples,
                                       int n_mels, int64_t *samples_used,
                                       app::Pool *pool) {
  // 1. 30 s of audio, zero-padded. The encoder is fixed at 1500 positions, so
  //    a short file is padded rather than given a shorter feature tensor.
  const size_t n_used = std::min(samples.size(),
                                 static_cast<size_t>(kChunkSamples));
  if (samples_used) *samples_used = static_cast<int64_t>(n_used);
  std::vector<double> wave(static_cast<size_t>(kChunkSamples), 0.0);
  for (size_t i = 0; i < n_used; ++i) wave[i] = samples[i];

  // 2. reflect-pad, 200 each side -> 480400, which is 3001 frames of 400 with
  //    hop 160. The last of them is dropped after the mel projection.
  const std::vector<double> padded =
      reflect_pad(wave, static_cast<size_t>(kNfft / 2));
  const int num_frames =
      1 + static_cast<int>((padded.size() - static_cast<size_t>(kNfft)) /
                           static_cast<size_t>(kHopLength));
  if (num_frames != kMelFrames + 1)
    throw std::runtime_error(
        "whisper features: " + std::to_string(num_frames) +
        " frames from a 30 s chunk, expected " +
        std::to_string(kMelFrames + 1) + " (the +1 is the frame Whisper drops)");

  // 3. periodic Hann. w[399] = 6.17e-05, not 0: that is what "periodic" means
  //    and a symmetric window is a different filter.
  std::vector<double> window(static_cast<size_t>(kNfft));
  for (int n = 0; n < kNfft; ++n)
    window[static_cast<size_t>(n)] =
        0.5 - 0.5 * std::cos(2.0 * kPi * n / kNfft);

  // 4. frames -> windowed samples, one frame at a time so the whole (frames,
  //    400) matrix is never resident twice.
  std::vector<float> fr(static_cast<size_t>(kMelFrames) * kNfft, 0.0f);

  // Frames are independent too, and each writes a disjoint span, so the same
  // channel split applies. The floor's global maximum is a separate pass over
  // the finished tensor rather than an accumulation here: that keeps the
  // workers from needing a reduction, and it is also the order transformers
  // uses (max AFTER the last frame is dropped).
  auto frames_range = [&](const std::function<void(int, int)> &body) {
    if (!pool || pool->size() <= 1) {
      body(0, num_frames);
      return;
    }
    pool->run([&](int w, int nworkers) {
      const int chunk = (num_frames + nworkers - 1) / nworkers;
      body(std::min(num_frames, chunk * w),
           std::min(num_frames, chunk * (w + 1)));
    });
  };
  frames_range([&](int t_lo, int t_hi) {
    // Per-CALL scratch, not per-frame and NOT shared: hoisting these out of
    // the lambda made two workers write the same frame buffer, which is heap
    // corruption rather than a wrong number, so the shape of the scratch is
    // part of the contract here.
    std::vector<double> frame(static_cast<size_t>(kNfft));
    for (int t = t_lo; t < t_hi; ++t) {
      const double *src = padded.data() + static_cast<size_t>(t) * kHopLength;
      for (int i = 0; i < kNfft; ++i)
        frame[static_cast<size_t>(i)] = src[i] * window[static_cast<size_t>(i)];
      if (t == num_frames - 1) continue;  // the frame Whisper drops
      float *dst = fr.data() + static_cast<size_t>(t) * kNfft;
      for (int i = 0; i < kNfft; ++i) dst[i] = static_cast<float>(frame[i]);
    }
  });
  (void)n_mels;
  return fr;
}

// The (n_bins, frames) POWER spectrum, channels-first, one frame per column.
//
// The host's transform: the exact-size mixed-radix Cooley-Tukey above, in fp64.
// The array's version of this step is a GEMM against a precomputed DFT matrix,
// which computes the same function to bf16 -- see whisper/fft_npu.hpp.
std::vector<float> power_30s(const std::vector<float> &samples, int n_mels,
                             int64_t *samples_used, app::Pool *pool) {
  const std::vector<float> fr =
      windowed_frames_30s(samples, n_mels, samples_used, pool);
  const int n_bins = kNfft / 2 + 1;
  std::vector<float> power(static_cast<size_t>(n_bins) * kMelFrames, 0.0f);
  auto frames_range = [&](const std::function<void(int, int)> &body) {
    if (!pool || pool->size() <= 1) {
      body(0, kMelFrames);
      return;
    }
    pool->run([&](int w, int nworkers) {
      const int chunk = (kMelFrames + nworkers - 1) / nworkers;
      body(std::min(kMelFrames, chunk * w),
           std::min(kMelFrames, chunk * (w + 1)));
    });
  };
  frames_range([&](int t_lo, int t_hi) {
    std::vector<double> frame(static_cast<size_t>(kNfft));
    std::vector<double> spec(static_cast<size_t>(n_bins));
    for (int t = t_lo; t < t_hi; ++t) {
      for (int i = 0; i < kNfft; ++i)
        frame[static_cast<size_t>(i)] =
            static_cast<double>(fr[static_cast<size_t>(t) * kNfft + i]);
      spec = power_spectrum(frame.data(), kNfft);
      for (int k = 0; k < n_bins; ++k)
        power[static_cast<size_t>(k) * kMelFrames + t] =
            static_cast<float>(spec[static_cast<size_t>(k)]);
    }
  });
  return power;
}

// log10 of max(x, 1e-10), in place.
//
// Its own function because the array's mel path needs it: the bank projection
// is a GEMM there and the logarithm is not a matrix, so the two halves of
// project_and_log are taken from different places -- and the log is a
// PER-ELEMENT map, so it is the same numbers on both paths by construction
// rather than by two implementations agreeing.
void mel_log_inplace(std::vector<float> &v, app::Pool *pool) {
  auto range = [&](const std::function<void(size_t, size_t)> &body) {
    if (!pool || pool->size() <= 1) {
      body(0, v.size());
      return;
    }
    pool->run([&](int w, int nworkers) {
      const size_t chunk = (v.size() + nworkers - 1) / nworkers;
      body(std::min(v.size(), chunk * static_cast<size_t>(w)),
           std::min(v.size(), chunk * static_cast<size_t>(w + 1)));
    });
  };
  range([&](size_t lo, size_t hi) {
    // The 1e-10 floor BEFORE the log, not after: log10(0) is -inf, and an -inf in
    // the tensor poisons the global max on the way to mel_floor_scale.
    for (size_t i = lo; i < hi; ++i)
      v[i] = static_cast<float>(
          std::log10(std::max(static_cast<double>(v[i]), 1e-10)));
  });
}

// The slaney bank over the power spectrum, then log10, in that order. This is
// the HOST projection; the array's is a GEMM over the same bank matrix, and the
// two are interchangeable only because the order is written down here once.
void project_and_log(const std::vector<float> &power, int n_mels,
                     app::Pool *pool, std::vector<float> &mel_out) {
  const int n_bins = kNfft / 2 + 1;
  const std::vector<double> bank = mel_filter_bank(n_mels);
  mel_out.assign(static_cast<size_t>(n_mels) * kMelFrames, 0.0f);
  auto frames_range = [&](const std::function<void(int, int)> &body) {
    if (!pool || pool->size() <= 1) {
      body(0, kMelFrames);
      return;
    }
    pool->run([&](int w, int nworkers) {
      const int chunk = (kMelFrames + nworkers - 1) / nworkers;
      body(std::min(kMelFrames, chunk * w),
           std::min(kMelFrames, chunk * (w + 1)));
    });
  };
  frames_range([&](int t_lo, int t_hi) {
    // Per-CALL scratch, not per-frame and NOT shared: hoisting this out of the
    // lambda made two workers write the same buffer, which is heap corruption
    // rather than a wrong number, so the shape of the scratch is part of the
    // contract here.
    std::vector<double> mel(static_cast<size_t>(n_mels));
    for (int t = t_lo; t < t_hi; ++t) {
      for (int j = 0; j < n_mels; ++j) {
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k)
          acc += bank[static_cast<size_t>(k) * n_mels + j] *
                 static_cast<double>(
                     power[static_cast<size_t>(k) * kMelFrames + t]);
        mel[static_cast<size_t>(j)] = acc;
        mel_out[static_cast<size_t>(j) * kMelFrames + t] =
            static_cast<float>(acc);
      }
    }
  });
  mel_log_inplace(mel_out, pool);
}

// transformers' floor and scale, in place, over a FINISHED (n_mels, frames)
// tensor. The maximum is taken after the last frame is dropped: taking it before
// would let a dropped frame raise the floor of every frame that stayed.
void mel_floor_scale(std::vector<float> &mel) {
  double global_max = -1e300;
  for (float v : mel) global_max = std::max(global_max, static_cast<double>(v));
  const double floor_v = global_max - 8.0;
  for (float &v : mel)
    v = static_cast<float>((std::max(static_cast<double>(v), floor_v) + 4.0) / 4.0);
}

MelSpec log_mel_30s(const std::vector<float> &samples, int n_mels,
                    int64_t *samples_used, app::Pool *pool) {
  const std::vector<float> power =
      power_30s(samples, n_mels, samples_used, pool);
  std::vector<float> mel;
  project_and_log(power, n_mels, pool, mel);
  mel_floor_scale(mel);
  MelSpec out;
  out.n_mels = n_mels;
  out.frames = kMelFrames;
  out.data = std::move(mel);
  return out;
}

std::vector<float> conv_front_end(const MelSpec &mel, const float *w1,
                                  const float *b1, const float *w2,
                                  const float *b2, int hidden,
                                  app::Pool *pool) {
  if (mel.frames != kMelFrames)
    throw std::runtime_error("whisper features: expected " +
                             std::to_string(kMelFrames) + " frames, got " +
                             std::to_string(mel.frames));
  const int T = mel.frames, M = mel.n_mels;
  const int T2 = T / 2;

  // Split by OUTPUT CHANNEL, the only axis with no dependency: every o is an
  // independent dot product over the whole input, so the split needs no
  // barrier -- nothing is shared but the read-only input, and each worker
  // writes a disjoint span of the output.
  const int workers = (pool && pool->size() > 1) ? pool->size() : 1;
  auto over_channels = [&](const std::function<void(int, int)> &body) {
    if (workers == 1) {
      body(0, hidden);
      return;
    }
    pool->run([&](int w, int nw) {
      const int chunk = (hidden + nw - 1) / nw;
      body(std::min(hidden, chunk * w), std::min(hidden, chunk * (w + 1)));
    });
  };

  // conv1: (M, T) -> (hidden, T), kernel 3, padding 1, GELU.
  std::vector<float> h1(static_cast<size_t>(hidden) * T);
  over_channels([&](int o_lo, int o_hi) {
    for (int o = o_lo; o < o_hi; ++o) {
      const float *w = w1 + static_cast<size_t>(o) * M * 3;
      for (int t = 0; t < T; ++t) {
        float acc = b1[o];
        for (int c = 0; c < M; ++c) {
          const float *wc = w + static_cast<size_t>(c) * 3;
          for (int k = 0; k < 3; ++k) {
            const int ti = t + k - 1;              // padding 1
            if (ti < 0 || ti >= T) continue;
            acc += wc[k] * mel.at(c, ti);
          }
        }
        h1[static_cast<size_t>(o) * T + t] = app::gelu_erf_exact(acc);
      }
    }
  });

  // conv2: (hidden, T) -> (hidden, T/2), kernel 3, stride 2, padding 1, GELU.
  std::vector<float> h2(static_cast<size_t>(hidden) * T2);
  over_channels([&](int o_lo, int o_hi) {
    for (int o = o_lo; o < o_hi; ++o) {
      const float *w = w2 + static_cast<size_t>(o) * hidden * 3;
      for (int t = 0; t < T2; ++t) {
        const int centre = t * 2;
        float acc = b2[o];
        for (int c = 0; c < hidden; ++c) {
          const float *wc = w + static_cast<size_t>(c) * 3;
          for (int k = 0; k < 3; ++k) {
            const int ti = centre + k - 1;
            if (ti < 0 || ti >= T) continue;
            acc += wc[k] * h1[static_cast<size_t>(c) * T + ti];
          }
        }
        h2[static_cast<size_t>(o) * T2 + t] = app::gelu_erf_exact(acc);
      }
    }
  });

  // permute to (T2, hidden): Conv1d is channels-first, the encoder's GEMM is
  // rows-are-tokens. Doing it here means the encoder never sees a transposed
  // operand and cannot get the stride wrong by accident.
  std::vector<float> out(static_cast<size_t>(T2) * hidden);
  for (int o = 0; o < hidden; ++o)
    for (int t = 0; t < T2; ++t)
      out[static_cast<size_t>(t) * hidden + o] =
          h2[static_cast<size_t>(o) * T2 + t];
  return out;
}

}  // namespace npue::whisper
