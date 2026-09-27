//===- test_whisper_features.cpp -------------------------------*- C++ -*-===//
//
// Runs the C++ audio front end so a gate can hold it against transformers.
//
// Same arrangement as the tokenizer's gate: a test binary rather than a CLI,
// because nothing in the shipped runtime reaches arch=4 yet, and because the
// front end takes a file and the model's conv weights and nothing else -- so
// the gate can drive it without a model registry in the way.
//
//   argv[1] <model.npue>     for the conv weights
//   argv[2] <audio>          a 16 kHz mono WAV
//   --convert                ingest through ffmpeg instead of the WAV reader,
//                            which is how the gate exercises the conversion
//                            path on a file the WAV reader must refuse
//   --threads N              split the spectrogram and the convolutions over
//                            N workers. The output must be byte-identical to
//                            the single-threaded run, and the gate checks that
//                            rather than assuming it
//
// Output, all on stdout, so the gate parses one thing:
//   mel <n_mels> <frames> <hex of the float32 mel tensor>
//   conv <rows> <cols> <hex of the float32 (rows, cols) tensor>
//   samples <n> <hex of the float32 samples actually ingested>
//
// Hex rather than text: the tensors are 80x3000 and 1500x384, and a decimal
// round trip through a pipe is a rounding step the gate would then have to
// reason about. The samples are printed too, because "the spectrogram is
// wrong" and "the WAV reader returned the wrong samples" are different bugs
// and the gate should not have to guess which one it is looking at.
//
// Build:
//   g++ -std=c++17 -O2 -I runtime/include runtime/tests/test_whisper_features.cpp \
//       runtime/src/whisper/features.cpp runtime/src/whisper/audio.cpp \
//       runtime/src/model.cpp runtime/src/pool.cpp -o /tmp/test_whisper_features
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <string>
#include <vector>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "whisper/audio.hpp"
#include "whisper/fft_npu.hpp"
#include "whisper/features.hpp"
#include "whisper/mel_proj.hpp"

namespace {

std::string to_hex(const void *data, size_t bytes) {
  static const char *d = "0123456789abcdef";
  const auto *p = static_cast<const unsigned char *>(data);
  std::string out;
  out.reserve(bytes * 2);
  for (size_t i = 0; i < bytes; ++i) {
    out.push_back(d[p[i] >> 4]);
    out.push_back(d[p[i] & 0xF]);
  }
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.npue> <audio.wav>\n", argv[0]);
    return 2;
  }
  using namespace npue::whisper;

  npue::File file(argv[1]);
  const int d = static_cast<int>(file.config_int("d_model"));
  const int n_mels = static_cast<int>(file.config_int("num_mel_bins"));
  const auto conv1 = file.raw("frontend.conv1.weight");
  const auto conv1b = file.raw("frontend.conv1.bias");
  const auto conv2 = file.raw("frontend.conv2.weight");
  const auto conv2b = file.raw("frontend.conv2.bias");
  if (!file.has("frontend.conv1.weight") || !file.has("frontend.conv2.weight")) {
    std::fprintf(stderr, "%s has no frontend.conv* -- not a Whisper container\n",
                 argv[1]);
    return 2;
  }

  bool convert = false;
  int threads = 1;
  std::string art;
  // The two front-end steps that can move, INDEPENDENTLY, so a divergence can be
  // attributed to one of them instead of to "the array front end": --fft npu
  // sends the transform, --mproj npu sends the bank projection.
  bool fft_npu = false, mproj_npu = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--convert") convert = true;
    if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[i + 1]);
    if (a == "--artifacts" && i + 1 < argc) art = argv[i + 1];
    if (a == "--fft" && i + 1 < argc) fft_npu = std::string(argv[i + 1]) == "npu";
    if (a == "--mproj" && i + 1 < argc) mproj_npu = std::string(argv[i + 1]) == "npu";
  }
  // The pool is what makes the front end affordable (conv2 is quadratic in
  // hidden), and a parallel split that changed the answer would be worse than
  // no split at all -- so the gate runs the same input both ways and demands
  // byte equality, not closeness.
  app::Pool pool(threads);
  app::Pool *use = threads > 1 ? &pool : nullptr;

  Audio a;
  try {
    a = ingest(argv[2], convert);
  } catch (const std::exception &e) {
    // Refusals are OUTPUT for this gate, not a crash: "a 44.1 kHz file is
    // refused with this message" is the property being checked, and a
    // non-zero exit would hide the message behind the harness.
    std::printf("refused %s\n", e.what());
    return 0;
  }
  std::printf("samples %zu %s\n", a.samples.size(),
              to_hex(a.samples.data(), a.samples.size() * 4).c_str());

  // The two designs, opened only for the step that was asked for. The transform
  // is a DIFFERENT algorithm on the array (a direct product against bf16
  // twiddles, not an fp64 mixed-radix FFT), which is why this gate measures it
  // rather than assuming it -- and why the two flags are separate: one of them
  // can be equivalent and the other not, and "the array front end" cannot say
  // which.
  std::unique_ptr<npu::Device> dev;
  std::unique_ptr<npu::Design> design;
  std::unique_ptr<NpuFft> fft;
  std::unique_ptr<NpuMelProj> mproj;
  if (fft_npu || mproj_npu) {
    if (art.empty())
      throw std::runtime_error("--front npu needs --artifacts <dir>");
    dev = std::make_unique<npu::Device>();
    design = std::make_unique<npu::Design>(*dev, art + "/gemm_rtp");
    std::ifstream jf(art + "/gemm_rtp/design.json");
    std::stringstream js;
    js << jf.rdbuf();
    auto all = app::parse_streams(js.str());
    std::sort(all.begin(), all.end(), [](const app::StreamEntry &x,
                                         const app::StreamEntry &y) {
      return x.slot < y.slot;
    });
    for (const auto &e : all) design->load_instr(art + "/gemm_rtp/" + e.file);
    const app::StreamEntry *df = nullptr, *mp = nullptr;
    for (const auto &e : all) {
      if (e.op == "dft400" && !df) df = &e;
      if (e.op == "mel_proj" && !mp) mp = &e;
    }
    if (fft_npu) {
      if (!df)
        throw std::runtime_error(
            art + "/gemm_rtp has no dft400 stream; re-export with "
            "--npu-extra-ops fft");
      fft = std::make_unique<NpuFft>(*design, pool, kNfft, kMelBins);
      fft->set_streams((size_t)df->slot, df->M);
      fft->alloc_buffers();
      std::fprintf(stderr, "fft: %s\n", fft->note(kMelFrames).c_str());
    }
    if (mproj_npu) {
      if (!mp)
        throw std::runtime_error(
            art + "/gemm_rtp has no mel_proj stream; re-export with "
            "--npu-extra-ops mproj");
      mproj = std::make_unique<NpuMelProj>(*design, pool, kMelBins, n_mels);
      mproj->set_streams((size_t)mp->slot, mp->M);
      mproj->alloc_buffers(mel_filter_bank(n_mels));
      std::fprintf(stderr, "mproj: %s\n", mproj->note(kMelFrames).c_str());
    }
  }

  int64_t used = 0;
  MelSpec mel;
  if (fft_npu || mproj_npu) {
    const std::vector<float> fr = windowed_frames_30s(a.samples, n_mels, &used, use);
    std::vector<float> power(static_cast<size_t>(kMelBins) * kMelFrames, 0.0f);
    if (fft_npu)
      fft->run(fr, kMelFrames, power.data());
    else
      power = power_30s(a.samples, n_mels, &used, use);
    mel.n_mels = n_mels;
    mel.frames = kMelFrames;
    mel.data.assign(static_cast<size_t>(n_mels) * kMelFrames, 0.0f);
    if (mproj_npu) {
      mproj->run(power, kMelFrames, mel.data.data());
      // The bank product, then the log: the GEMM gives the first and a
      // per-element map gives the second, exactly as the host's project_and_log
      // does them in that order.
      mel_log_inplace(mel.data, use);
    } else {
      std::vector<float> host_mel;
      project_and_log(power, n_mels, use, host_mel);
      mel.data = std::move(host_mel);
    }
    mel_floor_scale(mel.data);
  } else {
    mel = log_mel_30s(a.samples, n_mels, &used, use);
  }
  std::printf("mel %d %d %s\n", mel.n_mels, mel.frames,
              to_hex(mel.data.data(), mel.data.size() * 4).c_str());

  const std::vector<float> emb = conv_front_end(
      mel, conv1.as<float>(), conv1b.as<float>(), conv2.as<float>(),
      conv2b.as<float>(), d, use);
  const int rows = kEncoderPositions, cols = d;
  std::printf("conv %d %d %s\n", rows, cols,
              to_hex(emb.data(), emb.size() * 4).c_str());
  return 0;
}
