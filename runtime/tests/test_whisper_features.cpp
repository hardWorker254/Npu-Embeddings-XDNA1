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

#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "whisper/audio.hpp"
#include "whisper/features.hpp"

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
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--convert") convert = true;
    if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[i + 1]);
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

  int64_t used = 0;
  const MelSpec mel = log_mel_30s(a.samples, n_mels, &used, use);
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
