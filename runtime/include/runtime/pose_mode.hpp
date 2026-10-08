//===- pose_mode.hpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the body-pose entry point, dispatched from Runtime::run
// beside the arch=1 Gemma mode, the arch=4 STT mode and the arch=5 classify
// mode, for an arch=6 container.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// The same reason stt_mode.hpp and vit_mode.hpp give: a pose model shares
// nothing with the embedding pipeline -- no batch tiers, no pooling, no golden
// fixtures, no text -- and this file owns the whole path. Returns -1 for any
// other container, so the embedding path never learns it exists.
//
// THE ONE ARCH WHOSE DEFAULT PATH OPENS NO DEVICE
// ------------------------------------------------
// Whisper and ViT both construct a Device unconditionally: their GEMMs go to
// the array or the model does not run. This one does not, because it was
// measured and the array lost:
//
//     72 convolutions, 640x640, 16 CPU threads   150 ms
//     the same arithmetic as dispatched GEMMs    290 ms
//
// for three structural reasons -- N must be a multiple of tile_n*cols = 128 and
// most of this network's convolutions have N <= 64 (2.6x arithmetic padding
// waste before anything runs); a dispatch is M <= 1024 rows and the stem has M =
// 102400, so that one layer alone is 100 dispatches; and those 436 dispatches
// come to 660 us each inside conv(), of which only 140 us is the device's own
// GEMM. Per layer, 19 of the 72 are faster on the array, and an oracle splitting
// each layer onto its faster side measured 146 ms against the host's 150 -- 2.6%.
//
// So `--npu-ops conv` is HONOURED here rather than refused, which is the
// same relationship the flag has to Whisper's ops: an op is on the array when
// it is listed. What it buys is a measured comparison, printed per run -- and
// the status block prints both sides' numbers so the claim above is checkable
// rather than asserted.
//
// WHERE THE OUTPUT GOES
// ---------------------
// Diagnostics to stderr. stdout is the result: a JSON object shaped like
// MediaPipe's PoseLandmarker, because that is what a caller of this mode is
// most likely to be porting from. `--text` switches to a human summary and
// `--json` is accepted as an explicit spelling of the default rather than
// refused, because "I want JSON" is a reasonable thing to type next to a flag
// that has a text mode.
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
#include "runtime/design.hpp"        // prefer_embedded: the container's own set
#include "common/host_kernels.hpp"
#include "common/hub.hpp"
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, refuse_removed_op_flags
#include "pose/decode.hpp"
#include "pose/result_json.hpp"
#include "pose/session.hpp"
#include "runtime/device.hpp"        // npu::set_bo_mode, survey_contexts
#include "server/pose_backend.hpp"

namespace app {

// Returns the process exit code, or -1 when this container is not a pose model.
inline int maybe_pose_mode(const std::string &root, int argc, char **argv,
                           const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != npue::pose::kArch) return -1;

  refuse_removed_op_flags(argc, argv);
  refuse_exporter_only_flags(argc, argv);
  // The LAST occurrence wins, not the first. `npuimage serve <model>` puts
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

  // --bo-mode, before any Design exists. Same rule and reason as the others.
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) != "--bo-mode") continue;
    const std::string m = argv[i + 1];
    if (m == "host_only") npu::set_bo_mode(npu::BoMode::host_only);
    else if (m == "host_only_1m") npu::set_bo_mode(npu::BoMode::host_only_1m);
    else if (m == "ext") npu::set_bo_mode(npu::BoMode::ext);
    else if (m == "ext_1m") npu::set_bo_mode(npu::BoMode::ext_1m);
    else
      throw std::runtime_error(
          "--bo-mode " + m + ": expected host_only, host_only_1m, ext or ext_1m");
  }

  // --npu-ops conv is ACCEPTED and every other code is REFUSED.
  //
  // This is the opposite of vit_mode.hpp, which refuses the flag outright, and
  // the difference is the measurement in the header: a ViT's design set carries
  // no eltwise streams so there is nothing to move, whereas here there IS
  // something to move and the numbers for both sides are known.
  //
  // The codes refused are named with the reason, because "gelu is a code and I
  // asked for it and nothing happened" is the failure this project treats as
  // the worst one, and a conv net's SiLU is a different activation from
  // Whisper's GELU even though both are eltwise.
  bool conv_on_array = false;
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    const std::set<std::string> codes = parse_npu_ops(listing);
    if (!codes.empty()) {
      if (codes.size() != 1 || !codes.count("conv"))
        throw std::runtime_error(
            "--npu-ops " + listing +
            ": a pose model has exactly one array-able op, `conv`, which moves "
            "the 72 dispatched convolutions of the network onto the array as "
            "dispatched GEMMs (the 73rd weight, the head's [1,16,1,1] DFL, is "
            "folded analytically into the decoder and never dispatched). The "
            "codes " +
            app::npu_op_names(codes, "conv") +
            " name Whisper's or BERT's eltwise and projection passes, and this "
            "architecture has none of them: the activations here are SiLU, "
            "which is fused into the convolution's epilogue because it is "
            "elementwise on the GEMM's own output, and there is no LayerNorm, "
            "no attention and no vocabulary. Re-run with `--npu-ops conv`.");
      conv_on_array = true;
    }
  }

  const std::string model_name = fs::path(model_path).stem().string();

  // --serve PORT: the HTTP endpoint. Read through the SHARED reader, before the
  // images are demanded, because a server has no image on the command line -- it
  // is a different way of NAMING the same model, not a different mode.
  //
  // `npuimage serve <model>` is the one verb, and the container's arch picks
  // which endpoint answers: /v1/embeddings for a BERT-family model,
  // /v1/audio/transcriptions for Whisper, /v1/classify for a ViT, /v1/pose for
  // this one. That is already how the speech mode works off the same flag, and it
  // is the reason there is no `pose-server` subcommand: a second verb would mean a
  // second thing to learn for the same model, and the URL space is already
  // unambiguous because each arch owns a different path.
  int serve_port = 8080;
  std::string serve_bind = "127.0.0.1";
  const bool serving = app::read_serve(argc, argv, serve_port, serve_bind);

  // -- the images, given as --pose (run_pose() rewrites positionals into it).
  std::vector<std::string> images;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--pose") images.push_back(argv[i + 1]);
  if (images.empty() && !serving)
    throw std::runtime_error(
        "this is a pose model, so say whose pose to find:\n"
        "    npuimage pose <model> <image.png> [more.png ...]\n"
        "    npuimage serve <model>        (POST /v1/pose)\n"
        "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
        "  --conf 0.25  --iou 0.70  --kpt 0.50  --max-det 300\n"
        "  --npu-ops conv   run the convolutions on the array instead "
        "(slower here; measured both ways)");

  // -- thresholds. Parsed and RANGED here rather than in DecodeParams, because a
  // NaN threshold that silently keeps nothing is a threshold nobody asked for.
  npue::pose::DecodeParams params;
  auto number = [&](const char *name, double lo, double hi, double *out,
                    const char *what) {
    const std::string v = flag(name);
    if (v.empty()) return;
    try {
      size_t used = 0;
      const double d = std::stod(v, &used);
      if (used != v.size() || !(d >= lo && d <= hi))
        throw std::runtime_error("range");
      *out = d;
    } catch (const std::exception &) {
      throw std::runtime_error(std::string(name) + " '" + v +
                               "': expected a number in [" +
                               std::to_string(lo) + ", " + std::to_string(hi) +
                               "] for " + what);
    }
  };
  number("--conf", 0.0, 1.0, &params.conf, "the class score threshold");
  number("--iou", 0.0, 1.0, &params.iou, "the NMS overlap threshold");
  number("--kpt", 0.0, 1.0, &params.keypoint,
         "the keypoint visibility threshold");
  {
    const std::string v = flag("--max-det");
    if (!v.empty()) {
      const long n = std::strtol(v.c_str(), nullptr, 10);
      if (n <= 0 || n > 100000)
        throw std::runtime_error("--max-det '" + v +
                                 "': expected 1..100000 people");
      params.max_det = n;
    }
  }

  const bool text = has("--text");
  (void)has("--json");   // the default; accepted so it is not mistaken for junk

  const int threads = std::max(1, std::atoi(flag("--threads").empty()
                                                 ? "16"
                                                 : flag("--threads").c_str()));

  // -- the artifacts. EMPTY BY DEFAULT, and that is the whole point: the host
  // session needs no device, no design and no directory, so `pose` runs on a
  // machine with no /dev/accel0. Only --npu-ops conv (or an explicit
  // --artifacts) opens one.
  std::string art;
  // A CONTAINER THAT CARRIES ITS OWN SET IS ALREADY RESOLVED, and this is the
  // gate in front of that: the refusal below used to fire first, telling an
  // operator with a self-sufficient container to build a design set for a model
  // whose set was inside the file they were holding.
  //
  // `art` stays EMPTY in that case, and that is load-bearing rather than
  // convenient. prefer_embedded(container, art, set) asks the container first and
  // only builds `art + "/" + set` as a fallback, so an empty `art` is never used
  // as a path -- and the one place that DID treat it as a path is the source of
  // the leading-slash error "cannot open /gemm_rtp/insts_qkv_b4.bin" this feature
  // produced on its first run.
  const bool from_cli = !flag("--artifacts").empty();
  const bool self_sufficient =
      npu::prefer_embedded(&probe, "", "gemm_rtp").has("design.json");
  if (!from_cli && self_sufficient) {
    // Nothing to resolve. The Design comes out of the container below.
  } else if (conv_on_array || from_cli) {
    const std::string named = flag("--artifacts");
    if (named.empty()) {
      // TWO CAUSES, AND THE MESSAGE SAYS WHICH, because they have different
      // fixes and one of them is not "export a design set".
      //
      // The design COUNT is read out of the container's own `npu_streams`
      // rather than written here. An earlier version of this message said "14
      // distinct designs"; the packer measures 21 on this checkpoint, so the
      // number was wrong, and a refusal that states a wrong count is worse than
      // one that states none -- an operator who checks it finds the tool lying
      // about the very thing it is refusing over.
      std::string streams;
      try {
        streams = probe.config_string("npu_streams");
      } catch (const std::exception &) {
      }
      if (streams.empty())
        throw std::runtime_error(
            "--npu-ops conv needs the array panels, and this container "
            "does not carry any: it was packed without --npu. Repack with "
            "`--pose-onnx FILE --npu`, which stages the pre-tiled bf16 B panel "
            "for every convolution. (A design set is the SECOND requirement; "
            "this container would still need one afterwards, because a pose "
            "network's padded (K, N) shapes are its own and belong to no "
            "existing model.)");
      size_t n = 0;
      for (size_t i = 0; i < streams.size(); ++i)
        if (streams[i] == '{') ++n;
      throw std::runtime_error(
          "--npu-ops conv needs a design set on disk: pass --artifacts "
          "<dir> naming one, or build it with "
          "`python tools/export/export_gemm_rtp.py --target yolov8n-pose "
          "--arch 1 -n 32 --batch 1`. The flag has no default set to fall back "
          "on, because this container's " +
          std::to_string(n) +
          " padded (K, N) designs are its own and belong to no existing model, "
          "so there is nothing already on disk that could serve it. MEASURED, "
          "for what it is worth before you build one: the array path runs this "
          "network's 72 convolutions in 436 dispatches at 640x640 and takes "
          "290 ms against the host's 150, so it is 1.4x SLOWER here -- see "
          "runtime/include/pose/net.hpp for the three structural reasons.");
    }
    const std::vector<std::string> cands = artifacts_candidates(root, named);
    auto usable = [](const std::string &c) {
      return std::ifstream(c + "/gemm_rtp/design.json").good();
    };
    art = select_set_for_layout(cands, usable, "");
    if (art.empty()) {
      std::string looked;
      for (size_t i = 0; i < cands.size(); ++i)
        looked += (i ? ", " : "") + cands[i];
      throw std::runtime_error(
          "no pose design set found for --artifacts '" + named +
          "'; looked for gemm_rtp/design.json under " + looked +
          ". Export one with: python tools/export/export_gemm_rtp.py "
          "--target pose --arch 1");
    }
    // One hw_context, counted before anything opens, so a busy machine gets a
    // sentence rather than a failed context.
    if (!npu::require_context_budget(npu::survey_contexts(), 1,
                                     has("--allow-contention"), stderr))
      throw std::runtime_error(
          "NPU context budget: refusing to load 1 hw_context -- see the report "
          "above (close the other process, or pass --allow-contention)");
  }

  const double t0 = app::now_s();
  // The request, and ONLY the request: `conv_on_array` is --npu-ops conv and
  // `from_cli` is an explicit --artifacts. A carried set is neither, and
  // folding it in here is what made a plain `pose` run take the array --
  // 0.334 s against 0.150 s, a 2.2x slowdown nobody asked for.
  npue::pose::Session session(probe, model_name, art, threads,
                                conv_on_array || from_cli, params);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  // -- the status block --------------------------------------------------------
  std::fprintf(stderr,
               "NpuEmbeddings -- body pose (YOLOv8-pose) on the %s\n",
               session.array() ? "AMD NPU" : "CPU");
  std::fprintf(stderr,
               "  model      %s: %lldpx letterbox, %lld levels (strides "
               "%lld/%lld/%lld), %lld cells, head %lld = 4 box + %lld class + "
               "%lld keypoints x 3\n",
               model_name.c_str(),
               static_cast<long long>(g.input_size),
               static_cast<long long>(g.n_levels()),
               static_cast<long long>(g.strides[0]),
               static_cast<long long>(g.strides[1]),
               static_cast<long long>(g.strides[2]),
               static_cast<long long>(g.n_anchors()),
               static_cast<long long>(g.detect_cout),
               static_cast<long long>(g.num_classes),
               static_cast<long long>(g.num_keypoints));
  // A std::string built FIRST and then handed to fprintf's %s. The ternary that
  // produced this inline did not compile to a std::string at all: `std::string +
  // const char*` and `const char*` have a common type of `const char*`, and the
  // only way to get there is the string's own c_str() on a temporary that is
  // dead before fprintf runs -- so the status block printed the object's
  // internal bytes ("conv on `\xE4\x8C\x8CV") while the run itself was fine.
  // Anything non-trivial is built into a named std::string and c_str()'d at the
  // call, which is the only way to see this at the call site.
  const std::string backend_where =
      session.array()
          ? "the array (" + std::to_string(session.stream_ops().size()) +
                " streams)"
          : std::string("the host (default)");
  std::fprintf(stderr,
               "  backend    conv on %s, everything else on the host\n",
               backend_where.c_str());
  if (!session.array())
    std::fprintf(stderr,
                 "             pass --npu-ops conv to measure the array "
                 "instead; see runtime/include/pose/net.hpp for why the host "
                 "wins here (2.6x arithmetic padding waste on N <= 64, 100 of "
                 "the network's 436 dispatches for the stem alone, and 660 us "
                 "per dispatch of which 140 us is the device's GEMM).\n");
  if (session.array())
    std::fprintf(stderr,
                 "             designs   %s\n"
                 "             %lld rows per dispatch; compare against a host "
                 "run, the two numbers are printed per image below.\n",
                 art.c_str(),
                 static_cast<long long>(session.rows_per_dispatch()));
  std::fprintf(stderr,
               "  decode     %s\n", npue::pose::decode_defaults_line().c_str());
  if (has("--conf") || has("--iou") || has("--kpt") || has("--max-det"))
    std::fprintf(stderr, "             (thresholds overridden on the command "
                         "line, as above)\n");
  std::fprintf(stderr, "  setup      %.2f s\n", t_setup);

  // WHAT RAN WHERE, one row per op -- the same discipline vit_mode.hpp follows,
  // because a status block that only names what moved hides what did not.
  //
  // The convolution count is convs.size() MINUS ONE, and on BOTH paths. The
  // container carries 73 convolution weights and the graph runs 72: the 73rd is
  // the head's [1,16,1,1] DFL projection, whose input is a Softmax that no op in
  // the list can consume, so head() folds it in analytically as w . E[bin] + b.
  // Printing 73 in the host row and 72 in the array row would make the array
  // look like it skipped a layer, and printing 73 in both would claim a
  // convolution neither path runs.
  const size_t n_convs = session.geometry().convs.empty()
                             ? 0
                             : session.geometry().convs.size() - 1;
  std::fprintf(stderr, "  ops        (npu = dispatched, host = this process)\n");
  if (session.array())
    std::fprintf(stderr, "             %-26s %-5s %s\n",
                 (std::to_string(n_convs) + " x conv").c_str(), "npu",
                 "im2col on the host, GEMM on the array, bias by the "
                 "design, SiLU in the host epilogue");
  else
    std::fprintf(stderr, "             %-26s %-5s %s\n",
                 (std::to_string(n_convs) + " x conv").c_str(), "host",
                 "im2col + a blocked fp32 GEMM over rows, 16 threads");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "silu", "host",
               "fused into the convolution's epilogue; it is elementwise on the "
               "GEMM's output");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "concat, add", "host",
               "channel joins and the C2f residual, memory-bound");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "maxpool (SPPF)", "host",
               "size-preserving 5x5/1, pad 2");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "upsample", "host",
               "nearest x2, the neck's up path");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "image front end", "host",
               ("decode, bilinear letterbox to " +
                std::to_string(g.input_size) +
                "px, pad 114, (x/255 - mean)/std")
                   .c_str());
  std::fprintf(stderr, "             %-26s %-5s %s\n", "dfl", "folded",
               "the exported graph softmaxes its 16 bins and sums them, and the "
               "1x1 projection above it is applied as w . E[bin] + b on that "
               "mean -- which is why it is not in the 72");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "threshold, nms, "
                                                               "inverse "
                                                               "letterbox",
               "host", "host-side by construction: a list of survivors, not a "
                       "tensor");

  // -- the HTTP endpoint --------------------------------------------------
  //
  // Handed the session it built, the SAME session the CLI would have run the
  // images through, and its own `defaults` rather than a copy of the session's
  // params: a request that names no threshold gets the defaults, a request that
  // names them gets its own for that request only.
  if (serving) {
    return app::serve_pose(session, model_name, serve_port, serve_bind, params);
  }

  // -- run ---------------------------------------------------------------------
  //
  // --pose-dump FILE writes every graph node's output, in graph order, as raw
  // fp32 NCHW preceded by a fixed-size JSON header. It exists so that
  // tools/verify/verify_pose.py can say WHICH node this runtime and an
  // independent reading of the ONNX graph first disagree at: end to end, a wrong
  // first convolution and a wrong sixtieth are the same sentence, and the
  // difference between finding them and not is one re-run of the packer against a
  // bisect by hand.
  //
  // It is refused unless a path is given, so it can never write somewhere by
  // accident, and it is a diagnostic -- the detections printed below come from
  // the same run either way.
  //
  // THE HEADER IS RESERVED, NOT STREAMED. The tensors are transient: Network::run
  // returns the head and drops them, so the writer sees each one exactly once and
  // cannot go back. JSON with unknown node count cannot be written first, so the
  // file opens with 64 KiB of spaces that are overwritten by the real header once
  // the count is known. A header that does not fit is a refusal, not a truncated
  // list -- a short header would read as a short graph, and a graph that quietly
  // lost its last sixty nodes is exactly the failure this dump exists to catch.
  constexpr size_t kDumpHeader = 1u << 16;
  const std::string dump_path = flag("--pose-dump");
  std::ofstream dump;
  std::vector<std::string> dump_index;   // the header lines, built as we go
  size_t dump_img = 0;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary | std::ios::trunc);
    if (!dump)
      throw std::runtime_error("--pose-dump " + dump_path +
                               ": cannot open it for writing");
    dump.seekp(0, std::ios::end);
    dump.write(std::string(kDumpHeader, ' ').data(),
               static_cast<std::streamsize>(kDumpHeader));
    std::fprintf(stderr,
                 "  dump       %zu graph nodes per image -> %s (fp32 NCHW, "
                 "JSON header in the first 64 KiB)\n",
                 session.geometry().graph.size(), dump_path.c_str());
    session.network().on_node = [&](size_t i, const npue::pose::Tensor &t) {
      char line[320];
      const int n = std::snprintf(
          line, sizeof(line),
          "    {\"image\": %zu, \"i\": %lld, \"c\": %lld, \"h\": %lld, \"w\": %lld, "
          "\"n\": %zu, \"off\": %llu}",
          dump_img,
          // kGraphInput is SIZE_MAX and arrives here as a graph node index, so
          // it is printed as the negative the reader looks for. Casting through
          // long long rather than printing the size_t directly, which would be
          // 18446744073709551615 and would not match the Python side.
          static_cast<long long>(static_cast<std::ptrdiff_t>(i)),
          static_cast<long long>(t.c), static_cast<long long>(t.h),
          static_cast<long long>(t.w), t.d.size(),
          static_cast<unsigned long long>(dump.tellp()));
      if (n <= 0 || static_cast<size_t>(n) >= sizeof(line))
        throw std::runtime_error(
            "--pose-dump: a node's header line does not fit in 320 bytes. Three "
            "extents, an index and an offset cannot need that, so the writer is "
            "wrong rather than the graph large.");
      dump_index.emplace_back(line);
      dump.write(reinterpret_cast<const char *>(t.d.data()),
                 static_cast<std::streamsize>(t.d.size() * sizeof(float)));
    };
  }

  int exit = 0;
  for (const auto &path : images) {
    if (!dump_path.empty()) ++dump_img;
    npue::pose::Result r;
    try {
      r = session.detect_file(path);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %-40s FAILED: %s\n", path.c_str(), e.what());
      exit = 1;
      continue;
    }

    std::fprintf(stderr, "  %-40s %.2f s  %zu people  (%lld convs %s, %lld "
                         "dispatches)\n",
                 path.c_str(), r.total_s, r.people.size(),
                 static_cast<long long>(r.convs_host + r.convs_array),
                 session.array() ? "array" : "host",
                 static_cast<long long>(r.dispatches));
    std::fprintf(stderr,
                 "             %.1f ms front end, %.1f ms network, %.1f ms "
                 "decode, %lldx%lld px source\n",
                 r.front_end_s * 1e3, r.network_s * 1e3, r.decode_s * 1e3,
                 static_cast<long long>(r.width),
                 static_cast<long long>(r.height));
    if (r.convs_host > 0)
      std::fprintf(stderr,
                   "             host conv split: %.1f ms gemm, %.1f ms im2col, "
                   "%.1f ms [M,N] transpose, %.1f ms weight transpose, "
                   "%.1f ms elementwise "
                   "(%.1fx torch's %.1f)\n",
                   r.t_gemm_s * 1e3, r.t_im2col_s * 1e3,
                   r.t_transpose_s * 1e3, r.t_wmat_s * 1e3,
                   r.t_elementwise_s * 1e3,
                   r.network_s / 2.05e-2, 20.5);

    // The array path's counterpart to the host split above, and the same span:
    // every conv() call on the array, from the im2col that fills A to the NCHW
    // transpose of C. MEASURED, not the device-busy figure -- the array's own
    // counters say the device is idle for most of it, and a status line that
    // printed only the busy time would report a speedup the picture does not
    // have. The two numbers to put side by side are this and network_s.
    if (r.convs_array > 0)
      std::fprintf(stderr,
                   "             array conv: %.1f ms in %lld dispatches "
                   "(%.0f us each) -- %.1f ms GEMM on the device, %.1f ms the "
                   "host's own A repack, %.1f ms the C transpose. The other "
                   "%.1f ms of the %.1f ms network is im2col and the "
                   "elementwise ops, on the host either way, of which "
                   "%.1f ms is the non-convolution span. im2col itself, which "
                   "t_array also spans because t0 is taken before it, is "
                   "%.1f ms.\n",
                   r.t_array_s * 1e3, static_cast<long long>(r.dispatches),
                   r.dispatches > 0
                       ? r.t_array_s * 1e6 / static_cast<double>(r.dispatches)
                       : 0.0,
                   r.t_array_gemm_s * 1e3,
                   r.t_array_repack_s * 1e3, r.t_array_transpose_s * 1e3,
                   (r.network_s - r.t_array_s) * 1e3, r.network_s * 1e3,
                   r.t_elementwise_s * 1e3, r.t_im2col_s * 1e3);

    if (text) {
      // A human summary. One line per person, then the skeleton by name, which
      // is the thing a person looking at this wants and the thing the JSON
      // makes them assemble.
      std::printf("%s: %zu %s\n", path.c_str(), r.people.size(),
                  r.people.size() == 1 ? "person" : "people");
      for (size_t i = 0; i < r.people.size(); ++i) {
        const auto &p = r.people[i];
        std::printf("  person %zu  score %.3f  box %.0f,%.0f %.0fx%.0f  %zu/%d "
                    "keypoints visible\n",
                    i, p.score, p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1,
                    static_cast<size_t>(std::count_if(
                        p.keypoints.begin(), p.keypoints.end(),
                        [](const npue::pose::Keypoint &k) { return k.visible; })),
                    static_cast<int>(npue::pose::kNumKeypoints));
        for (int e = 0; e < npue::pose::Skeleton::kEdges; ++e) {
          int a = 0, b = 0;
          if (!npue::pose::Skeleton::pair(e, a, b)) continue;
          const auto &ka = p.keypoints[static_cast<size_t>(a)];
          const auto &kb = p.keypoints[static_cast<size_t>(b)];
          if (!ka.visible || !kb.visible) continue;
          std::printf("    %-34s (%.0f,%.0f) -> (%.0f,%.0f)\n",
                      (std::string(npue::pose::Skeleton::name(a)) + "-" +
                       npue::pose::Skeleton::name(b))
                          .c_str(),
                      ka.x, ka.y, kb.x, kb.y);
        }
      }
      std::fflush(stdout);
      continue;
    }

    // -- the default: a MediaPipe-shaped result on stdout, so a port from
    // PoseLandmarker reads the same names and finds the same joints.
    //
    // `result_json` is the same function the HTTP endpoint answers with. The
    // CLI's output is what somebody compares against MediaPipe by eye and the
    // endpoint's is what a program parses; two emitters would drift, and the
    // drift would surface as "the server and the CLI disagree".
    npue::pose::JsonContext jctx;
    jctx.geom = &g;
    jctx.params = &params;
    jctx.image_label = path;
    jctx.array = session.array();
    const std::string out = npue::pose::result_json(r, jctx);
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
  }
  if (!dump_path.empty()) {
    std::string h = "{\n  \"nodes\": [\n";
    for (size_t i = 0; i < dump_index.size(); ++i)
      h += (i ? ",\n" : "") + dump_index[i];
    h += "\n  ]\n}\n";
    if (h.size() > kDumpHeader)
      throw std::runtime_error(
          "--pose-dump " + dump_path + ": the header needs " +
          std::to_string(h.size()) + " bytes for " +
          std::to_string(dump_index.size()) +
          " nodes and the reserved area is 65536. Raise kDumpHeader in "
          "runtime/include/runtime/pose_mode.hpp; do NOT let it overflow, "
          "because the reader would then take tensor bytes for header text.");
    h.append(kDumpHeader - h.size(), ' ');
    dump.seekp(0);
    dump.write(h.data(), static_cast<std::streamsize>(kDumpHeader));
    dump.close();
    std::fprintf(stderr, "             %zu node tensors written\n",
                 dump_index.size());
  }
  return exit;
}

}  // namespace app