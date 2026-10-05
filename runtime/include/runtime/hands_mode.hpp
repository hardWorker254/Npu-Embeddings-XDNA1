//===- hands_mode.hpp --------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the hand-landmark entry point, dispatched from Runtime::run
// beside the arch=1 Gemma mode, the arch=4 STT mode, the arch=5 classify mode
// and the arch=6 pose mode, for an arch=7 container.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// The same reason the other four give: a hands container shares nothing with the
// embedding pipeline -- no batch tiers, no pooling, no golden fixtures, no text
// -- and this file owns the whole path. Returns -1 for any other container, so
// the embedding path never learns it exists.
//
// WHY THERE IS NO ARRAY PATH HERE
// -------------------------------
// pose_mode.hpp honours `--npu-ops conv` because it can: the design set loads,
// both sides' numbers are known, and the answer turned out to be that the array
// is 1.4x SLOWER at 640x640 -- a measured loss, which is a result rather than an
// absence.
//
// Here the flag is REFUSED, and the reason is not "there is nothing to move":
// the palm detector's 232.0M dense MACs and the landmark network's 123.9M are
// exactly the arithmetic a GEMM array is for, and 39 of this container's 100
// convolutions are depthwise and would stay on the host either way. What is
// missing is a design set carrying the 31 distinct dense (K, N) pairs these two
// graphs use, and without one there is no array number to report -- so a session
// that accepted the flag and ran on the host would print host timings under a
// flag that says the opposite. That is the one thing this project treats as the
// worst outcome, so the flag is refused with the reason spelled out.
//
// TWO STAGES, TWO COSTS
// ----------------------
// The status block reports the palm detector and the landmark network
// SEPARATELY, including how many of each one's convolutions are depthwise,
// because the split is the interesting number on this architecture: 23 of the
// palm's 53 convolutions and 16 of the landmark network's 47 are depthwise, and
// no array takes those. A single "145 convolutions" line would hide exactly the
// fact that decides whether the array is worth building a design set for.
//
// WHERE THE OUTPUT GOES
// ---------------------
// Diagnostics to stderr. stdout is the result: a JSON object shaped like
// MediaPipe's HandLandmarker, because that is what a caller of this mode is most
// likely to be porting from. `--text` switches to a human summary and `--json`
// is accepted as an explicit spelling of the default rather than refused.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
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
#include "hands/session.hpp"

namespace app {

// Returns the process exit code, or -1 when this container is not a hands model.
inline int maybe_hands_mode(const std::string &root, int argc, char **argv,
                            const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != npue::hands::kArch) return -1;

  // The same derivation pose_mode uses, and for the same reason: the name is
  // what the status line and the JSON's `image` sibling are labelled with, and
  // it is the FILE's stem rather than anything the user typed, so two containers
  // cannot report each other's name.
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

  // --npu-ops: REFUSED, every code, with the reason. See the header.
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    const std::set<std::string> codes = parse_npu_ops(listing);
    if (!codes.empty()) {
      const bool empty_code = listing == "none" || listing == "";
      if (empty_code) {
        // `--npu-ops none` is not a request to move anything, so it is accepted
        // and ignored rather than refused. Refusing it would be pedantry about
        // a flag whose meaning -- "no op goes on the array" -- is already what
        // this architecture does.
      } else {
        throw std::runtime_error(
            "--npu-ops " + listing +
            ": this architecture has no array path, so there is nothing for "
            "the flag to move. Not because there is no work -- the palm "
            "detector's 232.0M dense MACs and the landmark network's 123.9M are "
            "exactly a GEMM array's arithmetic -- but because no design set in "
            "this tree carries the 31 distinct dense (K, N) pairs these two "
            "graphs use, so there is no array timing to compare the host's "
            "against. Accepting the flag and running on the host would print "
            "host numbers under a flag that says otherwise, which is the one "
            "outcome worth refusing. 39 of the container's 100 convolutions are "
            "depthwise and would stay on the host either way. What this costs: "
            "the host split, measured per run below.");
      }
    }
  }
  // --artifacts is refused for the same reason and named separately, because a
  // user who typed it was told about a design set somewhere and is asking
  // whether one is needed -- the answer is that it would not be used.
  if (!flag("--artifacts").empty())
    throw std::runtime_error(
        "--artifacts " + flag("--artifacts") +
        ": not used by this architecture. There is no array path (see "
        "--npu-ops above), so a design set on disk would be loaded and never "
        "read. Both networks run on the host.");

  // pose_mode's own flags are REFUSED rather than ignored. They read as
  // "thresholds for the detection" on a mode that also has a detector, and
  // silently dropping them would produce a run whose boxes differ from the one
  // that was asked for with nothing in the output saying so. This mode's
  // thresholds come from the CONTAINER, which is what the zoo used.
  for (const char *f : {"--conf", "--iou", "--kpt", "--max-det", "--pose",
                        "--pose-dump", "--bo-mode"})
    if (has(f))
      throw std::runtime_error(
          std::string(f) + ": not a flag of `hands`. The score, NMS and crop "
          "thresholds are read from the container (score_threshold, "
          "nms_threshold, the two enlarge factors and the two shifts), because "
          "they are part of how this checkpoint was trained and the OpenCV zoo "
          "demo that recorded the golden used the container's values, not a "
          "command line's. For --pose: use `pose`, which is a different model "
          "with different flags. The two dumps are separate because they "
          "record different graphs, and one reader cannot read the other's "
          "file.");

  // -- the images, given as --hands (run_hands() rewrites positionals into it).
  std::vector<std::string> images;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--hands") images.push_back(argv[i + 1]);
  if (images.empty())
    throw std::runtime_error(
        "this is a hands model, so say whose hands to find:\n"
        "    npuembeddings hands <model> <image.png> [more.png ...]\n"
        "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
        "  --max-hands 1     run the second stage on at most N detections\n"
        "  --text            a human summary instead of JSON\n"
        "  --hands-dump FILE every graph node's output, for the gate\n"
        "  (the score and NMS thresholds come from the container, not from a\n"
        "   flag: they are part of the checkpoint -- see the refusal above)");

  // --max-hands: a CAP on how many detections reach the second stage, not a
  // score threshold. Every hand is an independent crop of the original frame, so
  // there is no way to batch them and the only lever is how many to run. Ranged
  // here rather than in the session, because a cap of 0 is a typo and a cap of
  // -1 is the session's own "all of them" -- two spellings of one thing.
  int64_t max_hands = -1;
  {
    const std::string v = flag("--max-hands");
    if (!v.empty()) {
      char *end = nullptr;
      const long n = std::strtol(v.c_str(), &end, 10);
      if (end == v.c_str() || *end != '\0' || n < 1 || n > 10000)
        throw std::runtime_error("--max-hands '" + v +
                                 "': expected an integer in 1..10000, or omit "
                                 "it to run the second stage on every "
                                 "detection");
      max_hands = n;
    }
  }
  const bool text = has("--text");
  (void)has("--json");   // the default; accepted so it is not mistaken for junk
  (void)root;

  const int threads =
      std::max(1, std::atoi(flag("--threads").empty() ? "16"
                                                     : flag("--threads").c_str()));

  const double t0 = app::now_s();
  npue::hands::Session session(probe, model_name, "", threads, max_hands);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  // -- the status block --------------------------------------------------------
  std::fprintf(stderr, "NpuEmbeddings -- hands (MediaPipe palm + landmark) on the CPU\n");
  std::fprintf(stderr,
               "  model      %s: palm %lldpx, landmark %lldpx, %lld palm "
               "anchors over %zu levels, %lld landmarks\n",
               model_name.c_str(),
               static_cast<long long>(g.palm_input_size),
               static_cast<long long>(g.lm_input_size),
               static_cast<long long>(g.palm_num_anchors), g.palm_levels.size(),
               static_cast<long long>(g.num_landmarks));
  std::fprintf(stderr,
               "  rotation   from palm keypoints %lld (wrist) and %lld "
               "(middle base), both read from the container\n",
               static_cast<long long>(g.palm_lm_wrist),
               static_cast<long long>(g.palm_lm_middle_base));
  std::fprintf(stderr,
               "  thresholds score %.3f, nms IoU %.3f, top_k %lld, crop enlarge "
               "%.2f then %.2f -- from the container, not from a flag\n",
               g.score_threshold, g.nms_threshold,
               static_cast<long long>(g.top_k), g.palm_pre.enlarge,
               g.palm_post.enlarge);
  int64_t palm_dense = 0, palm_dw = 0, lm_dense = 0, lm_dw = 0;
  for (const auto &c : g.palm_convs)
    (c.depthwise() ? palm_dw : palm_dense)++;
  for (const auto &c : g.lm_convs)
    (c.depthwise() ? lm_dw : lm_dense)++;
  std::fprintf(stderr,
               "  ops        (npu = dispatched, host = this process)\n");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(palm_dense) + " x conv + " +
                 std::to_string(palm_dw) + " x dwconv")
                    .c_str()),
               "host",
               "palm detector, 87 nodes; dwconv has no [M,N] GEMM to dispatch, "
               "so it would not move even with a design set");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(lm_dense) + " x conv + " + std::to_string(lm_dw) +
                 " x dwconv")
                    .c_str()),
               "host", "landmark network, 58 nodes, once per hand");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "add, maxpool, pad_c, resize",
               "host", "elementwise and memory passes");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "image front end", "host",
               ("decode, bilinear letterbox to " +
                std::to_string(g.palm_input_size) +
                "px padded with 0, (x/255 - mean)/std; per hand a "
                "rotated INTER_AREA crop to " +
                std::to_string(g.lm_input_size) + "px")
                   .c_str());
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               "threshold, nms, inverse letterbox", "host",
               "host-side by construction: a list of survivors, not a tensor");
  std::fprintf(stderr, "  setup      %.3f s (both graphs read, weights transposed once)\n",
               t_setup);

  // -- the dump ----------------------------------------------------------------
  //
  // --hands-dump FILE writes every graph node's output of BOTH networks, in
  // graph order, as raw fp32 NCHW preceded by a reserved JSON header. It exists
  // so the gate can say which node this walk and an independent reading of the
  // ONNX graph first disagree at: end to end, a wrong first convolution and a
  // wrong eightieth are the same sentence.
  //
  // The header is RESERVED, not streamed, for the reason pose_mode.hpp gives: the
  // tensors are transient, so the writer sees each exactly once, and a header
  // that does not fit the reservation is REFUSED rather than truncated -- a short
  // header reads as a short graph, and a graph that quietly lost its last sixty
  // nodes is the failure this dump exists to catch.
  constexpr size_t kDumpHeader = 1u << 16;
  const std::string dump_path = flag("--hands-dump");
  std::ofstream dump;
  std::vector<std::string> dump_index;
  size_t dump_img = 0;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary | std::ios::trunc);
    if (!dump)
      throw std::runtime_error("--hands-dump " + dump_path +
                               ": cannot open it for writing");
    dump.seekp(0, std::ios::end);
    dump.write(std::string(kDumpHeader, ' ').data(),
               static_cast<std::streamsize>(kDumpHeader));
    std::fprintf(stderr,
                 "  dump       %zu + %zu graph nodes per image -> %s (fp32 "
                 "NCHW, JSON header in the first 64 KiB)\n",
                 g.palm_graph.size(), g.lm_graph.size(), dump_path.c_str());
    // ONE hook on each network, and it is installed for the lifetime of the
    // process: it is a std::function call per node, 145 of them, which is a
    // rounding error against a 6 ms frame, and re-installing it per image would
    // be a per-frame allocation to avoid.
    auto hook = [&](const char *which) {
      return [&, which](size_t i, const npue::hands::Tensor &t) {
        char line[352];
        const int n = std::snprintf(
            line, sizeof(line),
            "    {\"image\": %zu, \"net\": \"%s\", \"i\": %lld, \"c\": %lld, "
            "\"h\": %lld, \"w\": %lld, \"n\": %zu, \"off\": %llu}",
            dump_img, which,
            // SIZE_MAX is the graph-input marker and prints as the -1 the
            // reader looks for; casting through long long rather than printing
            // the size_t, which would be 18446744073709551615.
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
    session.palm_network().on_node = hook("palm");
    session.lm_network().on_node = hook("lm");
  }

  // -- run ---------------------------------------------------------------------
  int exit = 0;
  for (const std::string &path : images) {
    npue::hands::Result r;
    try {
      r = session.detect_file(path);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %s: %s\n", path.c_str(), e.what());
      exit = 1;
      continue;
    }
    ++dump_img;
    std::fprintf(stderr,
                 "  %-40s %lldx%lld  %lld detection%s -> %lld hand%s  "
                 "%.1f ms (front %.1f / palm %.1f / nms %.1f / crop %.1f / lm "
                 "%.1f / post %.1f)\n",
                 path.c_str(), static_cast<long long>(r.width),
                 static_cast<long long>(r.height),
                 static_cast<long long>(r.detections.size()),
                 r.detections.size() == 1 ? "" : "s",
                 static_cast<long long>(r.hands.size()),
                 r.hands.size() == 1 ? "" : "s", r.total_s * 1e3,
                 r.front_end_s * 1e3, r.palm_s * 1e3, r.nms_s * 1e3,
                 r.crop_s * 1e3, r.lm_s * 1e3, r.post_s * 1e3);
    if (text) {
      if (r.hands.empty()) {
        std::printf("%s: no hands\n", path.c_str());
        continue;
      }
      for (size_t i = 0; i < r.hands.size(); ++i) {
        const npue::hands::Hand &h = r.hands[i];
        std::printf(
            "%s: hand %zu  score %.3f  %s  presence %.3f  box %.0f,%.0f "
            "%.0f,%.0f  wrist %.1f,%.1f  middle tip %.1f,%.1f\n",
            path.c_str(), i, r.detections[i].score,
            h.handedness > 0.5 ? "right" : "left", h.presence, h.bbox[0],
            h.bbox[1], h.bbox[2], h.bbox[3], h.landmarks[0][0],
            h.landmarks[0][1], h.landmarks[12][0], h.landmarks[12][1]);
      }
      continue;
    }
    // -- the default: a MediaPipe-shaped result on stdout, so a port from
    // HandLandmarker reads the same names and finds the same joints.
    std::fprintf(stderr,
                 "             crop: angle %.4f deg, palm kp box %.1f %.1f %.1f "
                 "%.1f, rot_bbox %.1f %.1f %.1f %.1f, pad_bias %.1f %.1f, "
                 "crop mean %.4f\n",
                 r.angle_deg, r.palm_kp_box[0], r.palm_kp_box[1],
                 r.palm_kp_box[2], r.palm_kp_box[3], r.rot_bbox[0],
                 r.rot_bbox[1], r.rot_bbox[2], r.rot_bbox[3],
                 r.crop_pad_bias[0], r.crop_pad_bias[1], r.crop_mean);
    std::printf("{\n  \"image\": \"%s\",\n  \"width\": %lld,\n  \"height\": %lld,\n",
                path.c_str(), static_cast<long long>(r.width),
                static_cast<long long>(r.height));
    std::printf(
        "  \"letterbox\": {\"scale\": %.9g, \"pad_x\": %lld, \"pad_y\": %lld},\n",
        r.scale, static_cast<long long>(r.pad_x), static_cast<long long>(r.pad_y));
    std::printf("  \"timings_ms\": {\"total\": %.3f, \"front_end\": %.3f, "
                "\"palm\": %.3f, \"nms\": %.3f, \"crop\": %.3f, \"landmarks\": "
                "%.3f, \"postprocess\": %.3f},\n",
                r.total_s * 1e3, r.front_end_s * 1e3, r.palm_s * 1e3,
                r.nms_s * 1e3, r.crop_s * 1e3, r.lm_s * 1e3, r.post_s * 1e3);
    std::printf("  \"detections\": [");
    for (size_t i = 0; i < r.detections.size(); ++i) {
      const npue::hands::Detection &d = r.detections[i];
      std::printf("%s\n    {\"score\": %.6f, \"box\": [%.3f, %.3f, %.3f, %.3f], "
                  "\"keypoints\": [",
                  i ? "," : "", d.score, d.box[0], d.box[1], d.box[2], d.box[3]);
      for (int64_t k = 0; k < npue::hands::kPalmLandmarks; ++k)
        std::printf("%s[%.3f, %.3f]", k ? ", " : "", d.kps[k][0], d.kps[k][1]);
      std::printf("]}");
    }
    std::printf("%s],\n  \"hands\": [", r.detections.empty() ? "" : "\n  ");
    for (size_t i = 0; i < r.hands.size(); ++i) {
      const npue::hands::Hand &h = r.hands[i];
      std::printf("%s\n    {\"score\": %.6f, \"handedness\": %.6f, \"presence\": "
                  "%.6f, \"handedness_category\": \"%s\", \"bbox\": [%.3f, "
                  "%.3f, %.3f, %.3f],\n     \"landmarks\": [",
                  i ? "," : "", r.detections[i].score, h.handedness, h.presence,
                  h.handedness > 0.5 ? "Right" : "Left", h.bbox[0], h.bbox[1],
                  h.bbox[2], h.bbox[3]);
      for (int64_t k = 0; k < npue::hands::kHandLandmarks; ++k)
        std::printf("%s[%.3f, %.3f, %.3f]", k ? ", " : "", h.landmarks[k][0],
                    h.landmarks[k][1], h.landmarks[k][2]);
      std::printf("],\n     \"world_landmarks\": [");
      for (int64_t k = 0; k < npue::hands::kHandLandmarks; ++k)
        std::printf("%s[%.5f, %.5f, %.5f]", k ? ", " : "", h.world[k][0],
                    h.world[k][1], h.world[k][2]);
      std::printf("]}");
    }
    std::printf("%s]}\n", r.hands.empty() ? "" : "\n  ");
  }

  if (!dump_path.empty()) {
    std::string h = "{\n  \"nodes\": [\n";
    for (size_t i = 0; i < dump_index.size(); ++i)
      h += (i ? ",\n" : "") + dump_index[i];
    h += "\n  ]\n}\n";
    if (h.size() > kDumpHeader)
      throw std::runtime_error(
          "--hands-dump " + dump_path + ": the header needs " +
          std::to_string(h.size()) + " bytes for " +
          std::to_string(dump_index.size()) +
          " node records and the reserved area is 65536. Raise kDumpHeader in "
          "runtime/include/runtime/hands_mode.hpp; do NOT let it overflow, "
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
