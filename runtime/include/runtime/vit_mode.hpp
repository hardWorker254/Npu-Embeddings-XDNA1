//===- vit_mode.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image-classification entry point, dispatched from
// Runtime::run beside the arch=1 Gemma mode and the arch=4 STT mode, for an
// arch=5 container.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// For the same reason stt_mode.hpp gives: a classifier shares nothing with the
// embedding pipeline -- no design_fits() batch tiers, no pooling, no golden
// fixtures, no lanes, no text -- and putting a second architecture into
// run_setup.hpp/run_execute.hpp would teach every condition there about it.
// This file owns the whole path and the embedding path never learns it exists.
// Returns -1 when the container is not a ViT, exactly like maybe_stt_mode().
//
// ONE DESIGN SET
// --------------
// gemm_rtp, and the same one bge-base-en-v1.5 uses: this architecture's four
// GEMM shapes are that model's, and its patch embedding is [n_patches, 768] x
// [768, 768], which IS attn_out's shape and therefore rides attn_out's
// instruction slot. So the set is resolved with pick_artifacts() -- the same
// call the embedding path makes, with the container's own layout hash so an
// int8 container cannot be served by a bf16 set -- and a container that is
// ready prints no export instruction at all.
//
// WHERE THE OUTPUT GOES
// ---------------------
// Diagnostics to stderr, the LABEL to stdout, alone, like transcribe's
// transcript. `classify foo.png | xargs rm` has to work.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/flags.hpp"          // read_serve
#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "common/hub.hpp"
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, refuse_removed_op_flags
#include "runtime/device.hpp"        // npu::set_bo_mode, survey_contexts
#include "server/vit_backend.hpp"    // serve_vit -- POST /v1/classify
#include "vit/classify.hpp"

namespace app {

// Returns the process exit code, or -1 when this container is not a ViT and the
// caller should carry on with the embedding path.
inline int maybe_vit_mode(const std::string &root, int argc, char **argv,
                          const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != "vit_patch16_prenorm_gelu") return -1;

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

  // --bo-mode has to be set before any Design exists, and the Design is built
  // inside the Session below. Same rule, same reason as the STT mode's.
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

  // --npu-extra-ops is parsed below, BEFORE any Design exists, because the
  // answer decides how many hw_contexts this run will open: gemm_rtp always,
  // plus one per honoured eltwise code. The budget check follows the parse for
  // that reason -- a machine already busy with another process gets a sentence
  // naming a count that includes the eltwise designs, rather than one that says
  // "1" and then trips over the second load.
  std::set<std::string> npu_ops;

  // --npu-extra-ops: TWO CODES ARE HONOURED AND SIX ARE REFUSED, each by name.
  //
  // `layn` and `gelu` are genuine per-op host/array choices here, and were not
  // for a while: the refusal below once said this architecture's design set
  // carries no eltwise designs, which described the export rather than the
  // model. A ViT has the same two pre-LN LayerNorms and the same ungated
  // exact-erf FFN Whisper has, NpuEltwise already implements both kernels, and
  // --npu-extra-ops layn,gelu now builds the sibling design sets for kinds.cls
  // exactly as for kinds.stt. Each costs one extra xclbin and one extra
  // hw_context.
  //
  // The other six are refused, and the reasons are per code because they are not
  // one reason:
  //
  //   softm  softmax is INSIDE attention. Moving it alone means shipping the
  //          whole 197x197 score matrix to the array and back for one
  //          elementwise pass. The two GEMMs that bracket it are what would be
  //          worth sending, and those are `attn`.
  //   attn   QK^T and softmax.V as GEMMs -- the `attn_qk`/`attn_av` instruction
  //          streams Whisper's gemm_rtp carries and kinds.cls's does not. This is
  //          a design change, not a flag.
  //   conv   Whisper's audio front end. A ViT's front end is decode, resize,
  //          normalise and im2col -- all host image work, not convolutions over
  //          a feature map, so there is nothing this code names.
  //   mproj  the mel filter bank;   fft the 400-point transform;
  //   logit  the vocabulary projection. Three audio/decoder operations. The
  //          nearest thing here is the classification head, and THAT is not a
  //          code at all and never will be -- 1000 is not a multiple of
  //          tile_n*cols = 48*4 = 192, so no legal B panel of that width exists
  //          on this array (tools/pack/packers/vit.py's header has the long form).
  //
  // And one measurement caveat that belongs here rather than only in the README:
  // npu_targets.json says outright that this repository has no attention
  // measurement above seq 64, and a ViT has 197 positions. `attn` would be a
  // first.
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-extra-ops") listing = argv[i + 1];
    const std::set<std::string> codes = parse_npu_ops(listing);
    const std::set<std::string> dead = {
        "softm", "attn", "conv", "mproj", "fft", "logit"};
    std::vector<std::string> refused;
    for (const auto &c : codes)
      if (dead.count(c)) refused.push_back(c);
    if (!refused.empty()) {
      std::string seen;
      for (const auto &c : refused) { if (!seen.empty()) seen += ", "; seen += c; }
      std::string keep;
      for (const auto &c : {"layn", "gelu"})
        if (codes.count(c)) { keep += keep.empty() ? c : ", " + std::string(c); }
      throw std::runtime_error(
          "--npu-extra-ops " + seen + ": this architecture has no such "
          "host/array choice to make. softm is softmax inside attention, and "
          "moving it alone would ship the whole 197x197 score matrix to the "
          "array and back for one elementwise pass; attn is the two GEMMs that "
          "bracket it (attn_qk/attn_av instruction streams), which kinds.cls "
          "does not carry and which npu_targets.json records as UNMEASURED "
          "above seq 64 -- a ViT has 197 positions. conv, mproj and fft name "
          "Whisper's audio front end, mel bank and transform. The one "
          "convolution this architecture has -- the patch embedding, "
          "Conv2d(3, d, kernel, stride=kernel) -- is ALREADY on the array: "
          "its [n_pos, patch_dim] x [patch_dim, d] shape is attn_out's own, so "
          "it is dispatched on attn_out's instruction slot with no flag at all, "
          "which is why asking for conv would add no dispatch. mproj and fft "
          "have no counterpart because this front end is "
          "decode/resize/normalise/im2col. logit names the vocabulary "
          "projection; the nearest thing here is the classification head, which "
          "cannot run on this array at all because 1000 is not a multiple of "
          "tile_n*cols = 192 and no legal B panel of that width exists." +
          (keep.empty() ? std::string(" Drop it.")
                        : " Keep --npu-extra-ops " + keep +
                              ", which this architecture does honour."));
    }
    npu_ops = codes;
  }

  // One hw_context per design set: gemm_rtp always, plus one each for `layn`
  // and `gelu`. The two eltwise designs are siblings -- separate xclbins,
  // separate contexts -- which is the cost this check exists to make visible,
  // because it is the only price the flag has and the flag itself prints none.
  {
    const int want = 1 + static_cast<int>(npu_ops.count("layn")) +
                     static_cast<int>(npu_ops.count("gelu"));
    if (want > 1 &&
        !npu::require_context_budget(npu::survey_contexts(), want,
                                     has("--allow-contention"), stderr))
      throw std::runtime_error(
          "NPU context budget: refusing to load " + std::to_string(want) +
          " concurrent hw_context(s) (gemm_rtp plus one per honoured "
          "--npu-extra-ops code) -- see the report above (close the other "
          "process, or pass --allow-contention)");
  }

  const std::string model_name = fs::path(model_path).stem().string();
  const bool json = has("--json");
  const bool topk = has("--top-k");

  // --serve PORT: the HTTP endpoint, read through the SHARED reader with the
  // other three. It used to be refused outright -- "there is no endpoint for it"
  // -- which was true and also an inconsistency: `serve` is the verb, the
  // container's arch picks the endpoint, and an arch that refuses the verb is
  // one a user discovers by typing. It answers POST /v1/classify, and the answer
  // is the same object `classify --json` prints (npue::vit::prediction_json).
  int serve_port = 8080;
  std::string serve_bind = "127.0.0.1";
  const bool serving = app::read_serve(argc, argv, serve_port, serve_bind);

  // -- the images, all of them given as --classify. run_classify() writes one
  // --classify per path, so a multi-image request and a single-image one are the
  // same command with a repeated flag -- and there is no positional archaeology
  // here to disagree with it about which argv slot is the model. The check is
  // conditional only because a server has no image on the command line: it is a
  // different way of NAMING the same model, not a different mode.
  std::vector<std::string> images;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--classify") images.push_back(argv[i + 1]);
  if (images.empty() && !serving)
    throw std::runtime_error(
        "this is an image classifier, so say what to classify:\n"
        "    npuembeddings classify <model> <image.png> [more.png ...]\n"
        "    npuembeddings serve <model>          (POST /v1/classify)\n"
        "  (it reads PNG and JPEG; anything else is refused rather than "
        "guessed at)");

  const int threads = std::max(1, std::atoi(flag("--threads").empty()
                                                 ? "16"
                                                 : flag("--threads").c_str()));

  // -- the design set, resolved the same way the embedding path resolves it:
  // from the container's own geometry, paired by its layout hash so an int8
  // container cannot land on a bf16 set.
  std::string art = flag("--artifacts");
  // Read the container's B-operand layout once, for the explicit branch below.
  // The implicit branch already had it; this is the fourth resolver that needed
  // it and the second that did not have it.
  std::string want_layout;
  try {
    want_layout = probe.info("layer.0.qkv").layout_hash;
  } catch (const std::exception &) {
  }
  if (!art.empty()) {
    // The same candidate list and the same "only gemm_rtp/design.json counts"
    // rule gemma_mode uses, because the same reason applies: this arch loads
    // gemm_rtp and nothing else. Taking the first candidate instead of the
    // first one that EXISTS would accept `--artifacts bge-base-en-v1.5` and
    // then fail on the bare name, naming a path that never held a design.
    const std::vector<std::string> cands = artifacts_candidates(root, art);
    auto usable = [](const std::string &c) {
      return std::ifstream(c + "/gemm_rtp/design.json").good();
    };
    // AND FILTERED ON THE LAYOUT, which is the fourth resolver to have to learn
    // it and the second to lack it. Taking the first candidate holding a
    // design.json meant an int8 ViT container naming --artifacts
    // vit-base-patch16-224 was handed the bf16 set, and the run died in the
    // pairing check with a message that named the directory but not the
    // mismatch as its subject.
    //
    // The note compares against the first USABLE candidate, not cands.front():
    // the candidate list holds several spellings of one name and front() is the
    // one that usually holds nothing, so comparing against it prints a
    // correction on every run including the bf16 ones that need none.
    std::string first_usable;
    for (const auto &c : cands)
      if (usable(c)) { first_usable = c; break; }
    const std::string found = select_set_for_layout(cands, usable, want_layout);
    if (found.empty()) {
      std::string looked;
      for (size_t i = 0; i < cands.size(); ++i)
        looked += (i ? ", " : "") + cands[i];
      bool saw_usable = false;
      for (const auto &c : cands) saw_usable = saw_usable || usable(c);
      if (saw_usable && !want_layout.empty())
        throw std::runtime_error(
            "--artifacts '" + art +
            "' resolves only to design sets whose B-operand layout is not this "
            "container's (" + want_layout.substr(0, 12) +
            "...). They are the same shapes in different element types, so "
            "nothing in them can execute this file; looked under " + looked +
            ". Export a set for this datapath, or run a bf16 container.");
      throw std::runtime_error(
          "no design set found for --artifacts '" + art + "'; looked for "
          "gemm_rtp/design.json under " + looked + ". An image classifier "
          "needs exactly one design set -- gemm_rtp -- because its four "
          "streams ARE gemm_rtp's (tools/data/npu_targets.json "
          "kinds.cls.streams), so a directory with only a qkv/ set, or with "
          "gemm_rtp_dec/ and no gemm_rtp/, cannot serve it.");
    }
    if (found != first_usable && !want_layout.empty())
      std::fprintf(stderr,
                   "note: --artifacts %s resolves to a design set whose B "
                   "layout is not this container's (%s vs %s); using %s "
                   "instead.\n",
                   art.c_str(),
                   design_b_layout_hash(first_usable).substr(0, 12).c_str(),
                   want_layout.substr(0, 12).c_str(), found.c_str());
    art = found;
  } else {
    std::string layout;
    try {
      layout = probe.info("layer.0.qkv").layout_hash;
    } catch (const std::exception &) {
    }
    const auto *ce = npue::hub::find(model_name);
    const std::string want_datapath = ce ? ce->datapath : "bf16";
    art = pick_artifacts(root, probe.config_int("hidden"),
                         probe.config_int("intermediate"), false,
                         probe.config_int("qkv_n"), layout, want_datapath,
                         model_name);
    if (art.empty())
      throw std::runtime_error(
          "no NPU design set matches " + model_name + " (hidden " +
          std::to_string(probe.config_int("hidden")) + ", datapath " +
          want_datapath + ", device " + running_device() + ") under " + root +
          ". This container's four GEMM shapes are bge-base-en-v1.5's, so the "
          "set that serves it is the one already exported for that model: name "
          "it with --artifacts, or export one with "
          "python tools/export/export_gemm_rtp.py --target " + model_name +
          " --arch 1");
  }
  if (!std::ifstream(art + "/gemm_rtp/design.json").good())
    throw std::runtime_error(
        art + " has no gemm_rtp/design.json. An image classifier needs "
        "gemm_rtp because its four streams ARE gemm_rtp's "
        "(tools/data/npu_targets.json kinds.cls.streams), plus one sibling "
        "directory per honoured --npu-extra-ops code (layernorm/, gelu/). "
        "Export with tools/export/export_gemm_rtp.py --target " + model_name +
        " --arch 1 --npu-extra-ops " +
        (npu_ops.empty() ? std::string("(none)")
                         : [&] {
                             std::string s;
                             for (const auto &c : npu_ops) {
                               if (!s.empty()) s += ",";
                               s += c;
                             }
                             return s;
                           }()));

  const double t0 = app::now_s();
  npue::vit::Session session(probe, model_name, art, threads, npu_ops);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  std::fprintf(stderr, "NpuEmbeddings -- ViT image classification on the AMD NPU\n");
  std::fprintf(stderr,
               "  model      %s: hidden %lld, %lld pre-LN layers, %lld heads x "
               "%lld, %lldpx at patch %lld -> %lld positions, %lld labels\n",
               model_name.c_str(), static_cast<long long>(g.d_model),
               static_cast<long long>(g.layers),
               static_cast<long long>(g.heads),
               static_cast<long long>(g.head_dim),
               static_cast<long long>(g.image_size),
               static_cast<long long>(g.patch_size),
               static_cast<long long>(g.n_pos),
               static_cast<long long>(g.num_labels));
  std::fprintf(stderr, "  designs    %s\n", art.c_str());
  std::fprintf(stderr, "  weights    %.1f MB staged on the array, %s\n",
               static_cast<double>(session.staged_bytes()) / (1024.0 * 1024.0),
               session.int8() ? "int8 operands (SmoothQuant, per-row A)"
                              : "bf16 operands");
  // WHAT RAN WHERE, one row per op, so a status block cannot hide the ops that
  // did not move. Four GEMM sites on the array, everything else on the host,
  // and each host row says why.
  {
    auto joined = [](const std::vector<std::string> &v) {
      std::string s;
      for (const auto &x : v) {
        if (!s.empty()) s += ", ";
        s += x;
      }
      return s;
    };
    std::fprintf(stderr, "  ops        (npu = dispatched, host = this process)\n");
    std::fprintf(stderr, "             %-26s %-5s %s\n", "patch_embed, 5th GEMM",
                 "npu",
                 "rides attn_out's stream: [n_patches, 768] x [768, 768] IS "
                 "attn_out's shape, so it costs no new stream and no new design");
    std::fprintf(stderr, "             %-26s %-5s %s\n",
                 (std::to_string(g.layers) + " x qkv/attn_out/ffn").c_str(),
                 "npu", (joined(session.stream_ops()) + ", M=" +
                           std::to_string(session.rows_per_dispatch()) +
                           " rows per dispatch")
                             .c_str());
    std::fprintf(stderr, "             %-26s %-5s %s\n", "image front end",
                 "host",
                 ("decode, PIL-equivalent resize to " +
                  std::to_string(g.image_size) + "px, (x/255 - mean)/std, im2col")
                     .c_str());
    // ONE ROW PER PASS, and WHERE IT RAN is read from the SESSION rather than
    // from the flag. That direction of error is the one that matters: a block
    // claiming the host was doing the work when the array was doing it would
    // send a reader to profile the wrong process, and the reverse would hide a
    // cost the user asked for.
    //
    // The design's geometry comes from session.eltwise_notes() -- the width is
    // compiled into its kernel and the epsilon is baked in, so both are worth
    // printing next to the op.
    //
    // The epsilon is printed with %.6g and NOT with std::to_string, which
    // formats a double with six DECIMAL places and turns ViT's 1e-12 into
    // "0.000000". That is the wrong number in the exact case the value is worth
    // printing for, and the same mistake is in whisper's eps-mismatch refusal,
    // which both were fixed to use this.
    char eps[32];
    std::snprintf(eps, sizeof eps, "%.6g", g.ln_eps);
    auto elt_row = [&](const char *name, bool on_array,
                       const char *host_reason) {
      std::string why = host_reason;
      if (on_array)
        for (const auto &note : session.eltwise_notes()) {
          const std::string tag = note.substr(0, note.find(':'));
          if (tag == name) {
            why = note.substr(note.find(':') + 1) +
                  "; bf16 in and out, fp32 accumulate -- the same two kernels "
                  "Whisper's --npu-extra-ops layn,gelu runs";
            break;
          }
        }
      std::fprintf(stderr, "             %-26s %-5s %s\n", name,
                   on_array ? "npu" : "host", why.c_str());
    };
    elt_row("layernorm", session.layernorm_on_array(),
            ((std::string("fp32 with this container's layer_norm_eps ") + eps +
              "; --npu-extra-ops layn moves it"))
                .c_str());
    elt_row("gelu", session.gelu_on_array(),
            "exact erf, fp32; --npu-extra-ops gelu moves it");
    std::fprintf(stderr, "             %-26s %-5s %s\n", "softmax", "host",
                 ("O(seq^2) attention on the host: " +
                  std::to_string(g.n_pos) + "x" + std::to_string(g.n_pos) +
                  " scores x " + std::to_string(g.heads) +
                  " heads; moving softmax alone would ship the whole score "
                  "matrix to the array and back for one elementwise pass, and "
                  "the two GEMMs that bracket it (--npu-extra-ops attn) are "
                  "unmeasured above seq 64 in this repository")
                     .c_str());
    std::fprintf(stderr, "             %-26s %-5s %s\n", "classifier head", "host",
                 (std::to_string(g.d_model) + "x" + std::to_string(g.num_labels) +
                  " matvec: " + std::to_string(g.num_labels) +
                  " is not a multiple of tile_n*cols, so no legal B panel of "
                  "that width exists for this array")
                     .c_str());
  }
  std::fprintf(stderr, "  setup      %.2f s (device, %zu design set%s, weights "
                       "staged on it)\n",
               t_setup, 1 + session.eltwise_notes().size(),
               session.eltwise_notes().empty() ? "" : "s");
  std::fprintf(stderr,
               "  unmeasured no timing above seq 64 has been taken in this "
               "repository, and this container has %lld positions. The "
               "per-image figures below are real; the throughput is not "
               "predicted.\n",
               static_cast<long long>(g.n_pos));

  // -- the HTTP endpoint --------------------------------------------------
  //
  // Handed the session it built and its own default top-k, which is 1 unless
  // --top-k was given: the CLI's default JSON prints the argmax alone and the
  // runners-up go to stderr, and the endpoint keeps that split rather than
  // putting three numbers where one was being read.
  if (serving)
    return app::serve_vit(session, model_name, serve_port, serve_bind,
                          topk ? session.geometry().num_labels : 1);

  // -- classify -------------------------------------------------------------
  int exit = 0;
  if (json) {
    std::string out = "[\n";
    for (size_t i = 0; i < images.size(); ++i) {
      npue::vit::Prediction p = session.classify_file(images[i]);
      // top_k is 1 here even under --top-k: --top-k prints the ranking to STDERR
      // and this branch prints the object, and a `--json` consumer that suddenly
      // found a top_k array would be a shape change nobody asked for. The
      // endpoint's default is 1 for the same reason; only a request that names
      // top_k gets the array.
      out += "  " + npue::vit::prediction_json(
                       p, session.labels()[static_cast<size_t>(p.label)],
                       images[i], 1) +
             (i + 1 < images.size() ? ",\n" : "\n");
    }
    out += "]\n";
    std::fputs(out.c_str(), stdout);
  } else {
    for (const auto &path : images) {
      const npue::vit::Prediction p = session.classify_file(path);
      std::fprintf(stderr,
                   "  %-40s %.2f s  %lld dispatches\n", path.c_str(),
                   p.front_end_s + p.encoder_s + p.head_s,
                   static_cast<long long>(p.n_dispatch));
      if (topk) {
        // A ranking of the whole row, not the top two: the argmax is on stdout
        // alone and the runners-up are a status detail, so the full sort is
        // printed rather than a truncated one that would read as "there is
        // nothing else close".
        std::vector<size_t> order(p.logits.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() +
                                          std::min<size_t>(10, order.size()),
                          order.end(), [&](size_t a, size_t b) {
                            return p.logits[a] > p.logits[b];
                          });
        for (size_t k = 0; k < std::min<size_t>(10, order.size()); ++k)
          std::fprintf(stderr, "    %2zu  %-40s logit %+.4f\n", k + 1,
                       session.labels()[order[k]].c_str(),
                       static_cast<double>(p.logits[order[k]]));
      }
      std::printf("%s\n", session.labels()[static_cast<size_t>(p.label)].c_str());
      std::fflush(stdout);
    }
  }
  return exit;
}

}  // namespace app
