//===- stt_mode.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the speech-to-text entry point: a whole transcription mode,
// dispatched from Runtime::run next to the arch=1 Gemma mode, for an arch=4
// Whisper container.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// A Whisper container shares nothing with the embedding pipeline: no
// design_fits(), no pooling, no golden fixtures, no lanes, and two design sets
// instead of one. Putting it in the middle of run_setup.hpp/run_execute.hpp
// would mean every condition there learns about a second architecture, and the
// arch=1 mode already shows what that costs. So this file owns the whole STT
// path and the embedding path never learns it exists. Returns -1 when the
// container is not Whisper, exactly like maybe_gemma_mode().
//
// WHERE THE OUTPUT GOES
// ---------------------
// Diagnostics go to stderr and the TRANSCRIPT goes to stdout, alone. A
// transcribe whose answer is buried under a status block is a transcribe nobody
// can pipe into anything, and the embeddings path made the same split when it
// moved its numbers out of the prose. --json puts the OpenAI-shaped object on
// stdout instead, with the per-window segments and their offsets.
//
// TWO DESIGN SETS, BOTH REQUIRED
// -----------------------------
// gemm_rtp for the encoder stack, gemm_rtp_dec for the decoder's seven streams.
// A set with only the encoder half is refused by name (resolve_stt_artifacts),
// because "transcribe" with no decoder is not a degraded mode, it is nothing.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "runtime/model.hpp"
#include "server/stt_backend.hpp"
#include "whisper/transcribe.hpp"

namespace app {

// Returns the process exit code, or -1 when this container is not a Whisper and
// the caller should carry on with the embedding path.
inline int maybe_stt_mode(const std::string &root, int argc, char **argv,
                         const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != "whisper_encdec_gelu") return -1;

  // The same removal check the embedding path runs: a stale --npu-eltwise must
  // not be silently ignored just because this container never used it.
  refuse_removed_op_flags(argc, argv);
  auto flag = [&](const char *name) {
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == name) return std::string(argv[i + 1]);
    return std::string();
  };
  auto has = [&](const char *name) {
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == name) return true;
    return false;
  };

  // --bo-mode has to be set before any Design exists, and the Designs are built
  // inside the Session below.
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) != "--bo-mode") continue;
    const std::string m = argv[i + 1];
    if (m == "host_only") npu::set_bo_mode(npu::BoMode::host_only);
    else if (m == "host_only_1m") npu::set_bo_mode(npu::BoMode::host_only_1m);
    else if (m == "ext") npu::set_bo_mode(npu::BoMode::ext);
    else if (m == "ext_1m") npu::set_bo_mode(npu::BoMode::ext_1m);
    else
      throw std::runtime_error(
          "--bo-mode " + m + ": expected host_only, host_only_1m, ext or "
          "ext_1m");
  }

  const std::string model_name = fs::path(model_path).stem().string();
  const std::string art = npue::whisper::resolve_stt_artifacts(
      root, flag("--artifacts"), model_name);
  const int threads = std::max(1, std::atoi(flag("--threads").empty()
                                                 ? "16"
                                                 : flag("--threads").c_str()));
  const bool serve = has("--serve");
  const std::string audio = has("--transcribe") ? flag("--transcribe")
                                                : flag("--audio");
  if (!serve && audio.empty())
    throw std::runtime_error(
        "this is a speech-to-text model, so say what to transcribe:\n"
        "    npuembeddings transcribe <model> <audio.wav> [--language en]\n"
        "  or serve the OpenAI-shaped endpoint:\n"
        "    npuembeddings serve <model>      (POST /v1/audio/transcriptions)\n"
        "  (neither --transcribe nor --serve was given)");

  npue::whisper::TranscribeOptions opts;
  if (!flag("--language").empty()) opts.language = flag("--language");
  if (!flag("--task").empty()) opts.task = flag("--task");
  opts.convert = has("--convert");
  if (has("--max-new")) opts.max_new = std::atoll(flag("--max-new").c_str());
  if (has("--chunk-seconds"))
    opts.chunk_seconds = std::atoi(flag("--chunk-seconds").c_str());
  if (has("--stride-seconds"))
    opts.stride_seconds = std::atoi(flag("--stride-seconds").c_str());
  const bool json = has("--json");

  const double t0 = app::now_s();
  npue::whisper::Session session(probe, model_name, art, threads);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  std::fprintf(stderr, "NpuEmbeddings -- Whisper speech to text on the AMD NPU\n");
  std::fprintf(stderr,
               "  model      %s: d_model %lld, %lld+%lld layers, %lld heads x "
               "%lld, %lld mel bins, vocab %lld\n",
               model_name.c_str(), static_cast<long long>(g.d_model),
               static_cast<long long>(g.enc_layers),
               static_cast<long long>(g.dec_layers),
               static_cast<long long>(g.heads),
               static_cast<long long>(g.head_dim),
               static_cast<long long>(g.mel_bins),
               static_cast<long long>(g.vocab));
  std::fprintf(stderr, "  designs    %s\n", art.c_str());
  std::fprintf(stderr, "  datapath   %s\n", session.datapath_note().c_str());
  std::fprintf(stderr, "  setup      %.2f s (device, two design sets, weights "
                       "staged on it)\n", t_setup);
  std::fprintf(stderr, "  language   %s%s\n", opts.language.c_str(),
               has("--language") ? "" : "   ASSUMED: this build does not "
                                          "detect languages, so --language "
                                          "defaults to en and says so here");
  std::fprintf(stderr, "  task       %s\n", opts.task.c_str());
  // The step between window starts is printed only when it is a real step.
  // `--chunk-seconds 5 --stride-seconds 5` is two strides across one window --
  // refused below, but a status block that printed "windows start -5 s apart"
  // before the refusal landed is a line nobody can act on.
  const int step_s = opts.chunk_seconds - 2 * opts.stride_seconds;
  if (step_s > 0)
    std::fprintf(stderr,
                 "  long form  %d s windows, %d s stride each side, so "
                 "windows start %d s apart\n",
                 opts.chunk_seconds, opts.stride_seconds, step_s);
  else
    std::fprintf(stderr,
                 "  long form  %d s windows, %d s stride each side\n",
                 opts.chunk_seconds, opts.stride_seconds);

  if (serve) {
    const int port = std::atoi(flag("--serve").c_str());
    return app::serve_stt(session, model_name, port,
                          flag("--bind").empty() ? "127.0.0.1" : flag("--bind"),
                          opts);
  }

  const auto r = session.transcribe_file(audio, opts);
  std::fprintf(stderr, "  chunks     %lld of %.1f s of audio\n",
               static_cast<long long>(r.n_chunks),
               static_cast<double>(r.n_samples) / npue::whisper::kSampleRate);
  std::fprintf(stderr, "  time       mel %.2f s, conv %.2f s, encoder %.2f s, "
                       "decoder %.2f s, %lld dispatches\n",
               r.mel_seconds, r.conv_seconds, r.encoder_seconds, r.decoder_seconds,
               static_cast<long long>(r.n_dispatches));
  if (r.n_chunks > 1) {
    std::fprintf(stderr, "  windows    (each on its own, before the merge):\n");
    for (const auto &s : r.segments)
      std::fprintf(stderr, "    %6.1f s - %6.1f s  %s\n", s.start_s, s.end_s,
                   s.text.c_str());
  }
  if (json)
    std::fputs(npue::transcript_json(r, true).c_str(), stdout);
  else
    std::fputs((r.text + "\n").c_str(), stdout);
  return 0;
}

}  // namespace app
