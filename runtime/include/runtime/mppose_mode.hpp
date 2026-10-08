//===- mppose_mode.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the body-pose entry point, dispatched from Runtime::run beside
// the arch=1 Gemma mode, the arch=4 STT mode, the arch=5 classify mode, the
// arch=6 YOLOv8-pose mode and the arch=7 hand-landmark mode.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// The same reason the other five give: an arch=8 container shares nothing with the
// embedding pipeline -- no batch tiers, no pooling, no golden fixtures, no text --
// and this file owns the whole path. Returns -1 for any other container, so the
// embedding path never learns it exists.
//
// WHY IT IS NOT ARCH=6 WITH A FLAG
// --------------------------------
// It is dispatched BEFORE arch=6 and refuses arch=6's flags by name, because
// "pose" here means two different models with two different vocabularies and a
// caller who typed `--conf` on an arch=8 container has said something false about
// what they are asking for. The two also differ in what they are: arch=6 runs one
// network over one frame and returns boxes and keypoints; this runs a detector
// and then a separate landmark network once per detected person, and the crop
// between them is rotated by an angle the first network predicted.
//
// THE ARRAY PATH, AND WHY IT WAS REFUSED AND IS NOT NOW
// ------------------------------------------------------
// This mode used to refuse --npu-ops conv and --artifacts with one reason: no
// design set in this tree carries arch=8's dense (K, N) pairs, so there is no
// array number to report and accepting the flag would print host numbers under a
// flag that says otherwise. That reasoning was sound and it has expired --
// geometry.py now carries MPPOSE_CONV_SHAPES, a design set was built from it, and
// the flag dispatches.
//
// --npu-ops conv moves the 99 DENSE convolutions. The other 62 are depthwise and
// do not move: their reduction is within one channel -- K is kh*kw and N is 1 --
// so there is no [M, N] GEMM in them to dispatch, and they carry 12.3 % of this
// container's MACs on their own. That is a property of the arithmetic, not of the
// design set.
//
// TWO GRAPHS, TWO SLOT RANGES
// ---------------------------
// The container holds a person detector AND a landmark network, and their
// convolution indices are SEPARATE spaces -- both start at zero, both have a conv
// 0 -- while the array backend has one panel table. So the two networks own
// disjoint slot ranges, assigned in one vector in mppose/session.cpp and read by
// both networks from Placement::slot_base. Keying panels by the bare index puts
// the landmark network's conv 0 on the detector's, and the failure is invisible
// by construction: both are the stem of a similarly-wide network, so the panel is
// the right shape, the layout_hash matches, every load succeeds, and the boxes
// come out of a pose network's weights.
//
// TWO STAGES, TWO COSTS
// ----------------------
// The status block reports the detector and the landmark network SEPARATELY,
// including how many of each one's convolutions are depthwise, because the split
// is the interesting number here: 28 of the detector's 73 convolutions and 34 of
// the landmark network's 88 are depthwise, and no array takes those. A single
// "161 convolutions" line would hide exactly the fact that decides whether the
// array is worth building a design set for.
//
// WHERE THE OUTPUT GOES
// ---------------------
// Diagnostics to stderr. stdout is the result: a JSON object shaped like
// MediaPipe's PoseLandmarker, because that is what a caller of this mode is most
// likely to be porting from.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/flags.hpp"
#include "common/host_kernels.hpp"
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, refuse_removed_op_flags
#include "common/design_selection.hpp"
#include "runtime/design.hpp"        // prefer_embedded: the container's own set   // artifacts_candidates
#include "mppose/session.hpp"
#include "runtime/device.hpp"             // require_context_budget, survey_contexts

namespace app {

// Returns the process exit code, or -1 when this container is not an arch=8 model.
inline int maybe_mppose_mode(const std::string &root, int argc, char **argv,
                             const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != npue::mppose::kArch) return -1;

  const std::string model_name = fs::path(model_path).stem().string();

  refuse_removed_op_flags(argc, argv);
  refuse_exporter_only_flags(argc, argv);

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

  // --npu-ops: `conv` MOVES THE 99 DENSE CONVOLUTIONS, the 62 depthwise ones
  // stay on the host whatever it says, and any OTHER code is refused by name
  // because there is no operation by that name in either graph.
  //
  // What changed, and when: this used to refuse every code with the reason that
  // no design set carried arch=8's dense (K, N) pairs. That was true when it was
  // written and stopped being true when
  // tools/export/exporters/gemm_rtp/geometry.py gained MPPOSE_CONV_SHAPES and a
  // design set was built from it -- 22 streams, one final.xclbin, verified
  // against the packed container in both directions by
  // tools/verify/verify_pose_streamset.py. A refusal whose reason has expired is
  // worse than no refusal: it is a claim about the state of this tree, and a
  // reader who has just built the design set would be told there is none.
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    const std::set<std::string> codes = parse_npu_ops(listing);
    const bool empty_code = listing == "none" || listing == "";
    if (!codes.empty() && !empty_code) {
      const std::set<std::string> here = {"conv"};
      for (const std::string &c : codes)
        if (!here.count(c))
          throw std::runtime_error(
              "--npu-ops " + c + ": this architecture has no operation named '" +
              c + "'. Its two graphs are {Conv, DwConv, Add, MaxPool, Resize, "
              "DepthToSpace} and the graph already FUSES the activations into the "
              "convolutions' epilogues, so 'gelu' has nothing to take over and "
              "'layn' has nothing to normalise -- there is no BatchNormalization "
              "or InstanceNormalization in either graph to name. Only 'conv' "
              "moves anything: the 99 dense convolutions, onto a design set built "
              "for their 22 padded (K, N) pairs.");
    }
  }
  // --artifacts is now READ, not refused: it names the design set the array path
  // dispatches into, and a directory that is silently ignored while a flag says
  // --npu-ops conv would report host numbers as array ones.
  //
  // What it needs is a container PACKED WITH --npu, and that is checked where
  // the session is built rather than here: a container without the pre-tiled
  // panels has nothing to stage, and the refusal it gets names the packing
  // command, which is the useful one.

  // --serve is REFUSED BY NAME, and read through the SHARED reader so that
  // `--serve 9000` and a bare `--serve` are caught the same way they are in the
  // pose and hands modes. Without this the flag falls through to the images check
  // below and a server -- which has no image on its command line by definition --
  // gets told to say whose body to look for.
  {
    int serve_port = 0;
    std::string serve_bind;
    if (read_serve(argc, argv, serve_port, serve_bind)) {
      throw std::runtime_error(
          std::string("this architecture has no HTTP endpoint yet, so `serve` "
                      "cannot be honoured for a MediaPipe Pose container. The "
                      "four that exist are /v1/embeddings (arch 1,2,3), "
                      "/v1/audio/transcriptions (arch 4), /v1/classify (arch 5) "
                      "and /v1/pose (arch 6, YOLOv8n-pose -- a DIFFERENT pose "
                      "model); arch 8 is the second container whose arch the "
                      "dispatcher routes here without an endpoint to land on. Not "
                      "refused because a body pose cannot be found over HTTP -- it "
                      "can -- but because a server that answers from a DIFFERENT "
                      "code path than `npuimage mppose` is exactly the drift "
                      "the pose endpoint documents as a contract. Until the "
                      "endpoint exists, the CLI is the whole answer. The fix is a "
                      "server/mppose_backend.cpp beside server/pose_backend.cpp: "
                      "the session, the JSON emitter and the multipart reader are "
                      "already written, so it is a binding and not a network."));
    }
  }

  // arch=6's own flags are REFUSED rather than ignored, and arch=7's with them.
  // They read as "thresholds for the detection" on a mode that has a detector with
  // a threshold in its CONTAINER, and silently dropping them would produce a run
  // whose people differ from the one that was asked for.
  for (const char *f : {"--conf", "--iou", "--kpt", "--max-det", "--pose",
                        "--pose-dump", "--bo-mode", "--max-hands", "--hands-dump"})
    if (has(f))
      throw std::runtime_error(
          std::string(f) + ": not a flag of `mppose`. The detector's score and NMS "
          "thresholds and the landmark net's confidence threshold are read from "
          "the container (score_threshold, nms_threshold, top_k, "
          "pose_conf_threshold, person_box_pre_enlarge, person_box_enlarge), "
          "because they are part of how this checkpoint was trained and the "
          "OpenCV zoo demo that recorded the golden used the container's values, "
          "not a command line's. For --pose: use `pose`, which is a different "
          "model with different flags. The three dumps are separate because they "
          "record different graphs, and one reader cannot read another's file.");

  // -- the images, given as --mppose (run_mppose() rewrites positionals into it).
  std::vector<std::string> images;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--mppose") images.push_back(argv[i + 1]);
  if (images.empty())
    throw std::runtime_error(
        "this is a MediaPipe Pose model, so say whose body to find:\n"
        "    npuimage mppose <model> <image.png> [more.png ...]\n"
        "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
        "  --max-people 1    run the landmark network on at most N detections\n"
        "  --text            a human summary instead of JSON\n"
        "  --mppose-dump FILE  every graph node's output, for the gate\n"
        "  (every threshold comes from the container, not from a flag: they are\n"
        "   part of the checkpoint -- see the refusal above)");

  // --max-people: a CAP on how many detections reach the second stage, not a score
  // threshold. Every person is an independent rotated crop of the original frame,
  // so there is no way to batch them and the only lever is how many to run.
  int64_t max_people = -1;
  {
    const std::string v = flag("--max-people");
    if (!v.empty()) {
      char *end = nullptr;
      const long n = std::strtol(v.c_str(), &end, 10);
      if (end == v.c_str() || *end != '\0' || n < 1 || n > 10000)
        throw std::runtime_error("--max-people '" + v +
                                 "': expected an integer in 1..10000, or omit it "
                                 "to run the landmark network on every detection");
      max_people = n;
    }
  }
  const bool text = has("--text");
  (void)has("--json");   // the default; accepted so it is not mistaken for junk
  (void)root;

  const int threads =
      std::max(1, std::atoi(flag("--threads").empty() ? "16"
                                                     : flag("--threads").c_str()));

  // -- the artifacts. EMPTY BY DEFAULT, and that is the point: the host session
  // needs no device, no design and no directory, so `mppose` runs on a machine
  // with no /dev/accel0. Only --npu-ops conv (or an explicit --artifacts) opens
  // one.
  //
  // STRUCTURED LIKE arch=6's AND NOT LIKE IT, in one respect: arch=6 counts the
  // design's streams out of the container's `npu_streams` and prints that count
  // in its refusal, and its count is wrong -- the packer measures 14 padded
  // (K, N) designs on that checkpoint against the "21" the message used to claim.
  // So no number is written here at all for arch=8; both facts are read out of the
  // container and the design when they exist, and where they do not, the message
  // says which one is missing. A refusal that states a wrong count is worse than
  // one that states none: an operator who checks it finds the tool lying about the
  // very thing it is refusing over.
  std::string art;
  const bool conv_on_array =
      parse_npu_ops(flag("--npu-ops")).count("conv") != 0;
  // A CONTAINER THAT CARRIES ITS OWN SET IS ALREADY RESOLVED, and this is the
  // gate in front of that: the refusal below used to fire first, telling an
  // operator with a self-sufficient container to build a design set for a model
  // whose set was inside the file they were holding.
  //
  // `art` stays EMPTY in that case, and that is load-bearing rather than
  // convenient. prefer_embedded(container, art, set) asks the container first
  // and only builds `art + "/" + set` as a fallback, so an empty `art` is never
  // used as a path. An explicit --artifacts still wins, which is why the guard
  // is `!from_cli` and not plain `self_sufficient`.
  const bool from_cli = !flag("--artifacts").empty();
  const bool self_sufficient =
      npu::prefer_embedded(&probe, "", "gemm_rtp").has("design.json");
  if (!from_cli && self_sufficient) {
    // Nothing to resolve. The Design comes out of the container below.
  } else if (conv_on_array || from_cli) {
    const std::string named = flag("--artifacts");
    std::string streams;
    try {
      streams = probe.config_string("npu_streams");
    } catch (const std::exception &) {
    }
    if (streams.empty())
      throw std::runtime_error(
          "--npu-ops conv needs the array panels, and this container does not "
          "carry any: it was packed without --npu. Repack with "
          "`--mppose-onnx models/mediapipe-pose --npu --device npu1`, which "
          "stages a pre-tiled bf16 B panel for every DENSE convolution of BOTH "
          "graphs. (A design set is the SECOND requirement, and a separate one.)");
    if (named.empty()) {
      size_t n = 0;
      for (size_t i = 0; i < streams.size(); ++i)
        if (streams[i] == '{') ++n;
      throw std::runtime_error(
          "--npu-ops conv needs a design set on disk: pass --artifacts <dir> "
          "naming one, or build it with `python tools/export/export_gemm_rtp.py "
          "--target mediapipe-pose --arch 1 -n 32`. The flag has no default to "
          "fall back on, because this container's " + std::to_string(n) +
          " padded (K, N) designs are its own. If you built one, name it: the "
          "runtime will not go looking for a directory that has no default, "
          "because picking one silently is how a run reports array numbers from "
          "a design set belonging to another model.");
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
          "no mppose design set found for --artifacts '" + named +
          "'; looked for gemm_rtp/design.json under " + looked +
          ". Export one with: python tools/export/export_gemm_rtp.py --target "
          "mediapipe-pose --arch 1 -n 32");
    }
    if (!npu::require_context_budget(npu::survey_contexts(), 1,
                                     has("--allow-contention"), stderr))
      throw std::runtime_error(
          "NPU context budget: refusing to load 1 hw_context -- see the report "
          "above (close the other process, or pass --allow-contention)");
  }

  const double t0 = app::now_s();
  npue::mppose::Session session(probe, model_name, art, threads, max_people,
                                  conv_on_array || from_cli);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  // -- the status block --------------------------------------------------------
  std::fprintf(stderr,
               "NpuEmbeddings -- mppose (MediaPipe person det + pose landmarks) on the CPU\n");
  std::fprintf(stderr,
               "  model      %s: detector %lldpx, landmarks %lldpx, %lld anchors "
               "over %zu levels, %lld rows of %lld (of which %lld are keypoints)\n",
               model_name.c_str(),
               static_cast<long long>(g.det_input_size),
               static_cast<long long>(g.pose_input_size),
               static_cast<long long>(g.det_num_anchors), g.det_levels.size(),
               static_cast<long long>(g.num_landmarks),
               static_cast<long long>(g.lm_cols),
               static_cast<long long>(g.num_keypoints));
  std::fprintf(stderr,
               "  rotation   between the detector's keypoints %lld (mid-hip) and "
               "%lld (full body), both read from the container; the RoI is the "
               "square on the hip whose half-side is that distance\n",
               static_cast<long long>(g.person_lm_mid_hip),
               static_cast<long long>(g.person_lm_full_body));
  std::fprintf(stderr,
               "  thresholds score %.3f, nms IoU %.3f, top_k %lld, pose conf %.3f, "
               "box enlarge %.2f then %.2f -- from the container, not a flag\n",
               g.score_threshold, g.nms_threshold,
               static_cast<long long>(g.top_k), g.pose_conf_threshold,
               g.person_box_pre_enlarge, g.person_box_enlarge);
  std::fprintf(stderr,
               "  normalise  detector (x/255 - %.1f)/%.1f, border 0.0 in TENSOR "
               "space; landmarks (x/255 - %.1f)/%.1f, border 0.0 in the raster\n",
               g.det_mean.empty() ? 0.0 : g.det_mean[0],
               g.det_std.empty() ? 1.0 : g.det_std[0],
               g.pose_mean.empty() ? 0.0 : g.pose_mean[0],
               g.pose_std.empty() ? 1.0 : g.pose_std[0]);
  int64_t det_dense = 0, det_dw = 0, pose_dense = 0, pose_dw = 0;
  for (const auto &c : g.det_convs) (c.depthwise() ? det_dw : det_dense)++;
  for (const auto &c : g.pose_convs) (c.depthwise() ? pose_dw : pose_dense)++;
  // Whether the dense convolutions actually went to the array, read from the
  // SESSION rather than from the flag. A line that printed "npu" because
  // --npu-ops conv was on the command line would say so for a run where the
  // design set was missing and everything fell back -- which is precisely what a
  // status block exists to prevent, and why this asks the object that did the
  // work.
  const bool on_array = session.array_placement();
  std::fprintf(stderr, "  ops        (npu = dispatched, host = this process)\n");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(det_dense) + " x conv + " +
                 std::to_string(det_dw) + " x dwconv")
                    .c_str()),
               on_array ? "npu" : "host",
               on_array
                   ? "person detector, 93 nodes, on the array's 22 padded "
                     "(K, N) slots; dwconv has no [M,N] GEMM to dispatch, so it "
                     "stays here"
                   : "person detector, 93 nodes; dwconv has no [M,N] GEMM to "
                     "dispatch, so it would not move even with a design set");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(pose_dense) + " x conv + " +
                 std::to_string(pose_dw) + " x dwconv")
                    .c_str()),
               on_array ? "npu" : "host",
               on_array ? "landmark network, 120 nodes, once per person, on the "
                          "same 22 slots under a disjoint range"
                        : "landmark network, 120 nodes, once per person");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               "add, maxpool, resize", "host",
               "elementwise and memory passes");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "d2s (DepthToSpace)", "host",
               "this architecture's own op: three of them build the detector's "
               "28/14/7 pyramid out of one 7x7 map, DCR order");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "image front end", "host",
               ("bilinear letterbox to " + std::to_string(g.det_input_size) +
                "px, normalise in float and pad with 0 in TENSOR space; per "
                "person a rotated INTER_AREA crop to " +
                std::to_string(g.pose_input_size) + "px")
                   .c_str());
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               "threshold, nms, inverse letterbox", "host",
               "host-side by construction: a list of survivors, not a tensor");
  std::fprintf(stderr,
               "  setup      %.3f s (both graphs read, weights transposed once)\n",
               t_setup);
  // THREE SPANS AND NOT ONE, because one number would invite the wrong
  // conclusion in either direction. The device's own time is one of them; the two
  // host spans are what it COSTS to feed the device and to read back, and they
  // are the honest answer to "is the array worth it" -- a line printing only the
  // device time would make a convolution look free, and one printing only the
  // total would make the array look like it was the host shuffling. Per-image
  // numbers are printed with the frame; this is the one-off staging cost.
  if (session.array_placement())
    std::fprintf(stderr, "  array      %.1f MB of bf16 panels staged, once, at "
                         "load\n",
                 session.array_staged_bytes() / (1024.0 * 1024.0));

  // -- the dump ----------------------------------------------------------------
  //
  // --mppose-dump FILE writes every graph node's output of BOTH networks, in graph
  // order, as raw fp32 NCHW preceded by a reserved JSON header. It exists so the
  // gate can say which node this walk and an independent reading of the ONNX graph
  // first disagree at.
  constexpr size_t kDumpHeader = 1u << 16;
  const std::string dump_path = flag("--mppose-dump");
  std::ofstream dump;
  std::vector<std::string> dump_index;
  size_t dump_img = 0;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary | std::ios::trunc);
    if (!dump)
      throw std::runtime_error("--mppose-dump " + dump_path +
                               ": cannot open it for writing");
    dump.seekp(0, std::ios::end);
    dump.write(std::string(kDumpHeader, ' ').data(),
               static_cast<std::streamsize>(kDumpHeader));
    std::fprintf(stderr,
                 "  dump       %zu + %zu graph nodes per image -> %s (fp32 NCHW, "
                 "JSON header in the first 64 KiB)\n",
                 g.det_graph.size(), g.pose_graph.size(), dump_path.c_str());
    auto hook = [&](const char *which) {
      return [&, which](size_t i, const npue::mppose::Tensor &t) {
        char line[352];
        const int n = std::snprintf(
            line, sizeof(line),
            "    {\"image\": %zu, \"net\": \"%s\", \"i\": %lld, \"c\": %lld, "
            "\"h\": %lld, \"w\": %lld, \"n\": %zu, \"off\": %llu}",
            dump_img, which,
            static_cast<long long>(static_cast<std::ptrdiff_t>(i)),
            static_cast<long long>(t.c), static_cast<long long>(t.h),
            static_cast<long long>(t.w), t.d.size(),
            static_cast<unsigned long long>(dump.tellp()));
        (void)n;
        dump.write(reinterpret_cast<const char *>(t.d.data()),
                   static_cast<std::streamsize>(t.d.size() * sizeof(float)));
        dump_index.push_back(line);
      };
    };
    session.det_network().on_node = hook("det");
    session.pose_network().on_node = hook("pose");
  }

  // -- run ---------------------------------------------------------------------
  int exit = 0;
  for (const std::string &path : images) {
    npue::mppose::Result r;
    try {
      r = session.detect_file(path);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %s: %s\n", path.c_str(), e.what());
      exit = 1;
      continue;
    }
    ++dump_img;
    std::fprintf(stderr,
                 "  %-40s %lldx%lld  %lld detection%s -> %lld pose%s  %.1f ms "
                 "(front %.1f / det %.1f / nms %.1f / crop %.1f / pose %.1f / "
                 "post %.1f)\n",
                 path.c_str(), static_cast<long long>(r.width),
                 static_cast<long long>(r.height),
                 static_cast<long long>(r.detections.size()),
                 r.detections.size() == 1 ? "" : "s",
                 static_cast<long long>(r.people.size()),
                 r.people.size() == 1 ? "" : "s", r.total_s * 1e3,
                 r.front_end_s * 1e3, r.det_s * 1e3, r.nms_s * 1e3,
                 r.crop_s * 1e3, r.pose_s * 1e3, r.post_s * 1e3);
    // THE ARRAY LINE IS PER IMAGE AND NOT IN THE HEADER, because every number in
    // it is a property of a run and the header is printed before there has been
    // one. It read "0 dispatches, 0.0 ms on the device" there, which is true and
    // useless -- and a status line that reports zero work for work that has been
    // done is worse than one that omits the row.
    if (session.array_placement())
      std::fprintf(stderr,
                   "             array: %lld dispatches, %.1f ms on the device, "
                   "%.1f ms repacking A and %.1f ms transposing C on the host\n",
                   static_cast<long long>(session.array_dispatches()),
                   1000.0 * session.array_seconds(),
                   1000.0 * session.array_repack_seconds(),
                   1000.0 * session.array_transpose_seconds());
    if (text) {
      if (r.people.empty()) {
        std::printf("%s: no person\n", path.c_str());
        continue;
      }
      for (size_t i = 0; i < r.people.size(); ++i) {
        const npue::mppose::Person &p = r.people[i];
        std::printf(
            "%s: pose %zu  score %.3f  conf %.3f  box %.0f,%.0f %.0f,%.0f  "
            "nose %.1f,%.1f  left wrist %.1f,%.1f\n",
            path.c_str(), i, r.detections[i].score, p.conf, p.bbox[0], p.bbox[1],
            p.bbox[2], p.bbox[3], p.landmarks[0][0], p.landmarks[0][1],
            p.landmarks[15][0], p.landmarks[15][1]);
      }
      continue;
    }
    std::fprintf(stderr,
                 "             crop: angle %.4f deg, square %.0fx%.0f, "
                 "pad_bias %.1f %.1f, crop mean %.4f\n",
                 r.angle_deg, r.rot_bbox[2], r.rot_bbox[3], r.crop_pad_bias[0],
                 r.crop_pad_bias[1], r.crop_mean);
    std::printf("{\n  \"image\": \"%s\",\n  \"width\": %lld,\n  \"height\": %lld,\n",
                path.c_str(), static_cast<long long>(r.width),
                static_cast<long long>(r.height));
    std::printf(
        "  \"letterbox\": {\"scale\": %.9g, \"pad_x\": %lld, \"pad_y\": %lld},\n",
        r.scale, static_cast<long long>(r.pad_x), static_cast<long long>(r.pad_y));
    std::printf(
        "  \"timings_ms\": {\"total\": %.3f, \"front_end\": %.3f, \"detector\": "
        "%.3f, \"nms\": %.3f, \"crop\": %.3f, \"landmarks\": %.3f, "
        "\"postprocess\": %.3f},\n",
        r.total_s * 1e3, r.front_end_s * 1e3, r.det_s * 1e3, r.nms_s * 1e3,
        r.crop_s * 1e3, r.pose_s * 1e3, r.post_s * 1e3);
    std::printf("  \"detections\": [");
    for (size_t i = 0; i < r.detections.size(); ++i) {
      const npue::mppose::Detection &d = r.detections[i];
      std::printf("%s\n    {\"score\": %.6f, \"box\": [%.3f, %.3f, %.3f, %.3f], "
                  "\"keypoints\": [",
                  i ? "," : "", d.score, d.box[0], d.box[1], d.box[2], d.box[3]);
      for (int64_t k = 0; k < npue::mppose::kDetLandmarks; ++k)
        std::printf("%s[%.3f, %.3f]", k ? ", " : "", d.kps[k][0], d.kps[k][1]);
      std::printf("]}");
    }
    std::printf("%s],\n  \"poses\": [", r.detections.empty() ? "" : "\n  ");
    for (size_t i = 0; i < r.people.size(); ++i) {
      const npue::mppose::Person &p = r.people[i];
      std::printf("%s\n    {\"score\": %.6f, \"pose_confidence\": %.6f, \"bbox\": "
                  "[%.3f, %.3f, %.3f, %.3f],\n     \"landmarks\": [",
                  i ? "," : "", r.detections[i].score, p.conf, p.bbox[0], p.bbox[1],
                  p.bbox[2], p.bbox[3]);
      for (int64_t k = 0; k < npue::mppose::kNumLandmarks; ++k)
        std::printf("%s[%.3f, %.3f, %.3f, %.5f, %.5f]", k ? ", " : "",
                    p.landmarks[k][0], p.landmarks[k][1], p.landmarks[k][2],
                    p.landmarks[k][3], p.landmarks[k][4]);
      std::printf("],\n     \"world_landmarks\": [");
      for (int64_t k = 0; k < npue::mppose::kNumLandmarks; ++k)
        std::printf("%s[%.5f, %.5f, %.5f]", k ? ", " : "", p.world[k][0],
                    p.world[k][1], p.world[k][2]);
      std::printf("],\n     \"segmentation_mask\": {\"width\": %lld, \"height\": "
                  "%lld, \"nonzero\": %llu}}",
                  static_cast<long long>(p.mask_w),
                  static_cast<long long>(p.mask_h),
                  static_cast<unsigned long long>(std::count_if(
                      p.mask.begin(), p.mask.end(),
                      [](uint8_t v) { return v != 0; })));
    }
    std::printf("%s]}\n", r.people.empty() ? "" : "\n  ");
  }

  if (!dump_path.empty()) {
    std::string h = "{\n  \"nodes\": [\n";
    for (size_t i = 0; i < dump_index.size(); ++i)
      h += (i ? ",\n" : "") + dump_index[i];
    h += "\n  ]\n}\n";
    if (h.size() > kDumpHeader)
      throw std::runtime_error(
          "--mppose-dump " + dump_path + ": the header needs " +
          std::to_string(h.size()) + " bytes for " +
          std::to_string(dump_index.size()) +
          " node records and the reserved area is 65536. Raise kDumpHeader in "
          "runtime/include/runtime/mppose_mode.hpp; do NOT let it overflow, "
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
