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

#include "cli/flags.hpp"          // read_serve
#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, the one --npu-ops parser
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
  refuse_exporter_only_flags(argc, argv);
  // The LAST occurrence wins, not the first. `npuembeddings serve <model>` puts
  // its own --threads 24 in the store BEFORE forward_common() appends whatever
  // the user typed, so the reader that took the first match silently answered
  // `serve <whisper> --threads 8` with 24 threads. Nothing on the command line
  // said 24, so the run looked exactly like the one that was asked for and was
  // half again as slow.
  //
  // Last-wins is the right rule for these modes specifically because they are the
  // ones the `serve` verb injects defaults into; every other flag here is either
  // absent or typed once, where first and last are the same entry. The REPEATED
  // flags of this binary -- --classify and --pose -- do not come through this
  // lambda at all: they are accumulated by the loops below, which is why they can
  // legitimately repeat.
  auto flag = [&](const char *name) {
    std::string v;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == name) v = argv[i + 1];
    return v;
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
  // The container's own B-operand layout, which is the one fact that separates
  // this model's int8 design set from its bf16 one -- identical directory
  // shape, identical stream list, identical everything except the element type
  // the B panels hold. Passing it is what stops an int8 container resolving to
  // the bf16 set and dying on the first encoder GEMM.
  std::string want_layout;
  try {
    want_layout = probe.info("encoder.layers.0.qkv").layout_hash;
  } catch (const std::exception &) {
    // A container that does not name it is not a container this resolver can
    // filter on; an empty answer means "cannot tell", not "does not match".
  }
  const std::string art = npue::whisper::resolve_stt_artifacts(
      root, flag("--artifacts"), model_name, want_layout);

  // WHISPER ON AN INT8 DESIGN SET. The GEMM streams run: NpuGemm reads each
  // operand's .wscale and .asmooth at stage time and quantises the activation
  // panel per row on the way into the A buffer, which is the same staging the
  // BERT encoder has used since int8 landed (runtime/src/whisper/npu_ops.cpp).
  //
  // The ops that ride a stream of their own -- conv1/conv2, the array attention,
  // the mel projection and the DFT -- do NOT, and the reason is a property of
  // what the container holds, not of the runtime's willingness: conv weights
  // are packed as F32 with no .wscale/.asmooth sidecars, so there is nothing to
  // quantise them against, and an int8 design's B panel is I8 with its own MAC
  // sub-tile, which the host-side bf16 tiler cannot fill. An int8 whisper design
  // set therefore carries only the four (encoder) / seven (decoder) GEMM
  // streams, and `npu_ops.count("conv")` -- which asks the DESIGN SET, not the
  // user's flag -- is already 0 for it. The check below is what turns that
  // absence into a stated decision rather than a silent host fallback, and what
  // would refuse a design set that claimed a conv stream it cannot honour.
  //
  // Nothing here is refused any more: before this was wired the whole container
  // was refused, with a message that named the missing step rather than a
  // disagreement that did not exist.
  bool int8_design = false;
  {
    const std::string a_dtype = app::design_field_string(
        art + "/gemm_rtp/design.json", "a_dtype");
    int8_design = !a_dtype.empty() && a_dtype != "bf16";
  }
  const int threads = std::max(1, std::atoi(flag("--threads").empty()
                                                 ? "16"
                                                 : flag("--threads").c_str()));
  // --serve, read through the SHARED reader so this endpoint, the embedding one
  // and the pose one cannot drift on what `--serve` with no port means. It used
  // to be read as flag("--serve") and atoi'd, which is 0 when the value is
  // absent -- and 0 is a legal request to bind(), so `npuembeddings <root>
  // --model m.npue --serve` started and listened on a port it never printed.
  int serve_port = 8080;
  std::string serve_bind = "127.0.0.1";
  const bool serve = app::read_serve(argc, argv, serve_port, serve_bind);
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
  // conv1/conv2 on the array or on the host, from the one flag that already
  // answers "which ops go on the array". The host is the fp32 reference the
  // features gate holds against, so it stays reachable with `--npu-ops ""`.
  // The whole SET is parsed once, here, and handed to the session: the codes
  // that name a design directory decide which xclbins exist, and a design cannot
  // be opened after the stacks have been built.
  const std::set<std::string> npu_ops = [&] {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    return app::parse_npu_ops(listing);
  }();
  opts.conv_npu = npu_ops.count("conv") > 0;
  const bool json = has("--json");

  // The context budget, BEFORE the session builds anything. Two design sets are
  // the floor here, and every elementwise op on the array is one more xclbin and
  // one more hw_context: the npu1 driver measured six, so a run that asks for
  // four elementwise ops on top of the two GEMM sets cannot be served and has to
  // be told so while there is still nothing to unwind. The embedding path has
  // always done this; the STT path did not, and the first symptom was a
  // hw_context failing to open instead of a sentence saying which of the two it
  // was.
  {
    int want = 2;   // gemm_rtp and gemm_rtp_dec
    for (const auto &op : app::npu_op_table())
      if (op.design[0] && npu_ops.count(op.code)) ++want;
    if (want > 1 &&
        !npu::require_context_budget(npu::survey_contexts(), want,
                                     has("--allow-contention"), stderr))
      throw std::runtime_error(
          "NPU context budget: refusing to load " + std::to_string(want) +
          " concurrent hw_context(s) -- see the report above (close the other "
          "process, or pass --allow-contention)");
  }

  const double t0 = app::now_s();
  npue::whisper::Session session(probe, model_name, art, threads, npu_ops);
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
  // What ran, not what could: the design set's capability is one thing and this
  // request's --npu-ops is another, and a status line that reports the
  // capability is a status line that says "npu" to a run that used the host.
  // Asking for the array on a set that cannot provide it is refused, because
  // answering from the host instead is the "the flag was there and nothing
  // happened" failure this project treats as worst.
  if (opts.conv_npu && !session.conv_available())
    throw std::runtime_error(
        "--npu-ops conv asked for conv1/conv2 on the array, but " + art +
        "/gemm_rtp cannot run them: " + session.conv_device() +
        ". For an int8 design set that is the operand type, not a missing "
        "stream -- attn_out is still there and still the right shape, but "
        "NpuConv1d::stage() tiles the conv panel from fp32 weights with a bf16 "
        "tiler and an int8 panel is a different element type with a different "
        "MAC sub-tile. Re-export the encoder set with "
        "tools/export/export_gemm_rtp.py (no --int8), or drop conv from "
        "--npu-ops and run the host front end.");

  // WHERE EVERY OPERATION RUNS, and why. One row per op, the device it runs on,
  // and the thing that decides it -- the stream that exists, or the reason one
  // does not. A status block that only says "conv: npu" answers a question
  // nobody asked and hides the six ops that did not move; this one is the
  // README's "what runs where" table, computed rather than transcribed.
  //
  // The NPU rows are read off the LOADED design set (session.enc_stream_ops()
  // and friends), so the table cannot describe an export other than the one
  // running. The host rows are this build's own schedule: there is no design
  // for them to be read from, and a design set that had one would say so in
  // --npu-ops, which is the same flag that moved conv.
  {
    auto joined = [](const std::vector<std::string> &v) {
      std::string s;
      for (const auto &x : v) {
        if (!s.empty()) s += ", ";
        s += x;
      }
      return s;
    };
    const int64_t d = g.d_model;
    const int64_t n_pos = g.max_seq;
    // The score matrix attention allocates per window, in MB. It is the one
    // host cost that scales with the model's WIDTH as well as its length, and
    // it is why attention is a host pass at this size.
    const double score_mb = static_cast<double>(n_pos) * g.heads * n_pos *
                            sizeof(float) / (1024.0 * 1024.0);
    std::fprintf(stderr, "  ops        (npu = dispatched, host = this process)\n");
    // The front end in two rows, because its two halves are separable: the
    // transform is one code and the bank another, and a combined "log-mel +
    // slaney bank: host" line is what hid two of the ops that HAD moved.
    {
      const bool on = session.on_array("fft");
      std::fprintf(stderr, "             %-26s %-5s %s\n", "transform, 400 points",
                   on ? "npu" : "host",
                   on ? session.fft_note().c_str()
                      : "3001 mixed-radix transforms of 400 points, fp64");
    }
    {
      const bool on = session.on_array("mproj");
      std::fprintf(stderr, "             %-26s %-5s %s\n", "slaney mel bank",
                   on ? "npu" : "host",
                   on ? session.mel_note().c_str()
                      : "the checkpoint's own filter bank, 201 x n_mels");
    }
    if (opts.conv_npu) {
      std::fprintf(stderr, "             %-26s %-5s %s\n", "conv1, conv2", "npu",
                   session.conv_device().c_str());
    } else if (int8_design) {
      std::fprintf(stderr,
                   "             %-26s %-5s %s\n", "conv1, conv2", "host",
                   "fp32; an int8 design set has no conv stream to bind");
    } else {
      std::fprintf(stderr,
                   "             %-26s %-5s %s\n", "conv1, conv2", "host",
                   "fp32, d^2 MACs; --npu-ops conv sends it to the array");
    }
    // The operand type of every GEMM row below it, named once rather than
    // repeated on four lines: an int8 container on an int8 design set is the
    // one configuration where "on the array" and "in bf16" are different
    // claims, and a status block that cannot tell them apart is the block that
    // let an unrunnable pairing look runnable.
    if (int8_design)
      std::fprintf(stderr,
                   "             %-26s %-5s %s\n", "GEMM datapath", "npu",
                   "int8 (W8A8), row-scaled activations, bf16 C");
    std::fprintf(stderr, "             %-26s %-5s %s\n",
                 ("encoder GEMMs, " + std::to_string(g.enc_layers) + " layers")
                     .c_str(),
                 "npu", (joined(session.enc_stream_ops()) + ", M=" +
                         std::to_string(session.enc_rows()) + " rows")
                            .c_str());
    std::string tiers;
    for (int64_t r : session.dec_tier_rows()) {
      if (!tiers.empty()) tiers += "/";
      tiers += std::to_string(r);
    }
    std::fprintf(stderr, "             %-26s %-5s %s\n",
                 ("decoder GEMMs, " + std::to_string(g.dec_layers) + " layers")
                     .c_str(),
                 "npu", (joined(session.dec_stream_ops()) + ", M=" + tiers +
                         " rows per tier")
                            .c_str());
    // LayerNorm, GELU and softmax, one row each: a combined "layer_norm, gelu,
    // softmax  host" line is what hid two of the three that HAD moved, and the
    // whole point of the block is that every op is accounted for.
    auto elt_row = [&](const char *label, const char *code,
                       const char *dir) {
      const bool on = session.on_array(code);
      std::string why;
      if (on) {
        for (const auto &n : session.elt_notes())
          if (n.rfind(dir, 0) == 0) why = n;
      } else {
        why = std::string("fp32 on this process; --npu-ops ") + code +
              " sends it to the array";
      }
      std::fprintf(stderr, "             %-26s %-5s %s\n", label,
                   on ? "npu" : "host", why.c_str());
    };
    elt_row("layer_norm", "layn", "layernorm");
    elt_row("gelu", "gelu", "gelu");
    elt_row("softmax", "softm", "softmax");
    {
      const std::string attn_why =
          session.on_array("attn")
              ? "QK^T and softmax.V as GEMMs on the attn_qk/attn_av streams"
              : "one score row at a time: " + std::to_string(n_pos) + " x " +
                    std::to_string(g.heads) + " x " + std::to_string(n_pos) +
                    " = " + std::to_string(static_cast<int>(score_mb)) +
                    " MB at full length";
      std::fprintf(stderr, "             %-26s %-5s %s\n", "attention",
                   session.on_array("attn") ? "npu" : "host", attn_why.c_str());
    }
    {
      const bool on = session.on_array("logit");
      std::fprintf(stderr, "             %-26s %-5s %s\n", "logits + argmax",
                   on ? "npu" : "host",
                   on ? session.logit_note().c_str()
                      : ("tied embedding, " + std::to_string(g.vocab) + " x " +
                         std::to_string(d) + " on the last row only")
                            .c_str());
    }
    std::fprintf(stderr, "             %-26s %-5s %s\n", "decode + merge", "host",
                 "the container's own BPE table; the window merge is "
                 "transformers' longest-common-sequence over ids");
  }
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
    return app::serve_stt(session, model_name, serve_port, serve_bind, opts);
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
