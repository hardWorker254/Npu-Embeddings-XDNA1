//===- geometry.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=8 body-pose geometry, read from the container.
//
// WHAT IS IN THIS CONTAINER
// -------------------------
// One .npue holds TWO networks and a front end, for the same reason arch=7
// holds a palm detector and a hand-landmark net: the landmark net is handed a
// person's REGION OF INTEREST by the detector and cannot find a person in a
// frame, and the detector cannot put a coordinate on anything. A container per
// network could not express that dependency.
//
//   det   MediaPipe's person detector: a 224x224 SSD-style detector over three
//         pyramid levels (28x28 at 2 anchors, 14x14 at 2, 7x7 at 6), 45 dense
//         convolutions, 28 depthwise ones, three DepthToSpace, and a head that
//         emits 2254 rows of 12 numbers plus 2254 scores.
//   pose  MediaPipe's pose-landmark network: a 256x256 mobile classifier-ish
//         regressor, 54 dense convolutions, 34 depthwise ones, and a head of
//         five single convolutions -- 39x5 screen landmarks, 1 confidence,
//         a 256x256 segmentation mask, a 64x64x39 heatmap and 39x3 world
//         points.
//
// WHY THIS IS NOT ARCH=6 WITH A FIELD TURNED OFF, AND NOT ARCH=7 WITH A NAME
// ------------------------------------------------------------------------
// arch=6 is one network, one head of a different shape again, and no anchors.
// arch=7 is the closest of the two and differs in four places that all matter:
//
//   * DepthToSpace. arch=7's op list is conv, dwconv, add, maxpool, pad_c,
//     resize. arch=8's is conv, dwconv, add, maxpool, resize, d2s. One op is
//     new and one is gone. The new one builds the detector's three pyramid
//     levels out of a single 7x7 feature map, so it is load-bearing rather than
//     an optimisation.
//   * THE BOX CONVENTION. arch=7 decodes cx,cy,w,h against an anchor CENTRE;
//     arch=8's detector decodes xy1,xy2, and reading one convention as the
//     other is a factor of two on the position error for every detection.
//   * TWO NORMALISATIONS. The detector is (u8/255 - 0.5) * 2 and the landmark
//     net is u8/255. arch=7's two networks agree, which is why its container
//     carries ONE image_mean/image_std pair and this one carries two.
//   * WHERE THE ZERO PAD LANDS. The detector divides by 255 and rescales
//     FIRST and pads with 0 in the [-1,1] TENSOR, so its border is 0.0 -- the
//     middle of the range, not the bottom. A reader that letterboxes a uint8
//     raster with 0 and normalises afterwards puts -1.0 there, which to this
//     stem is the most saturated value the input can take. The landmark net
//     pads in the uint8 raster before the rotation and divides at the end, so
//     its border is 0.0 too -- and for the opposite reason.
//
// THE GRAPH IS A LIST, NOT A SHAPE
// -------------------------------
// Which convolutions there are, in which order, what feeds what, and where the
// activations are is the model. The container carries an explicit op list per
// network and the runtime walks it.
//
// FOUR PADDING TERMS, NOT TWO
// ---------------------------
// Same as arch=7 and for the same reason: MediaPipe evaluates "SAME" padding at
// a stride, and these stems are [top,left,bottom,right] = [1,1,1,1] for a 3x3
// stride-2 kernel. The detector's is exactly symmetric and the landmark net's
// stem is [1,1,1,1] too, so a symmetric transcription happens to be RIGHT on
// this checkpoint -- which is a worse place to be than wrong, because the
// convention would then never be exercised. read_graph() recomputes each op's
// output extent from its four terms and refuses a disagreement, so the terms
// are read rather than assumed either way.
//
// THE ANCHORS ARE STORED, AND THE LEVELS ARE READ TWICE
// ------------------------------------------------------
// arch=7 generates its anchor table from the head's levels and cross-checks it
// against a second spelling. arch=8 STORES the 2254 x 2 table, because
// tools/pack/packers/mppose.py generates it and reproduces the OpenCV zoo's
// literal table bit for bit (0 of 2254 points differ), and a table that has
// been checked against a reference once should be shipped rather than
// regenerated on a second machine. det_anchor_levels is still read and still
// compared against the head's own levels: two spellings of one pyramid, so a
// disagreement is a refusal rather than a misaligned table.
//
// XRT-free, like the rest of the geometry, so a host-only check can read it.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/conv_host.hpp"      // npue::hostconv::Act
#include "common/json_min.hpp"       // npue::json::parse
#include "runtime/model.hpp"

namespace npue::mppose {

// The arch=8 container's `arch` string. Refused by name rather than run with the
// wrong head geometry, which is the whole point of carrying one.
inline constexpr const char *kArch = "mediapipe_pose_det_ssd_lm_regress";

// MediaPipe's pose network emits THIRTY-NINE rows: the thirty-three keypoints a
// skeleton is drawn from, plus six auxiliary ones. Both are returned and both are
// recorded, because a decoder that dropped the six would emit 33 rows and a
// caller indexing the thirty-fourth landmark would read past the end.
inline constexpr int64_t kNumLandmarks = 39;
inline constexpr int64_t kNumKeypoints = 33;

// Each landmark is [x, y, z, visibility, presence] and each world point is
// [x, y, z]. The FIVE matters: the zoo applies a sigmoid to columns THREE AND
// FOUR only, and applying it to the coordinates as well turns a pose into noise
// while every shape stays right.
inline constexpr int64_t kLmCols = 5;
inline constexpr int64_t kWorldCols = 3;

// The detector's box row: xy1x2 then 4 x 2 keypoint offsets, all as FRACTIONS
// of the input side and all relative to the anchor.
inline constexpr int64_t kDetRowTerms = 12;
inline constexpr int64_t kDetLandmarks = 4;

// The activation, as the container records it. Distinct from hostconv::Act on
// purpose: that enum is what the epilogue can COMPUTE, this is what the graph
// SAYS, and a packer that recorded one thing while the runtime believed another
// is the failure -- so the string is parsed here, refused if it is not one of
// four, and mapped across at the single call site.
enum class Act { None, Relu, Relu6, PreLU };

inline const char *act_name(Act a) {
  switch (a) {
    case Act::None: return "none";
    case Act::Relu: return "relu";
    case Act::Relu6: return "relu6";
    case Act::PreLU: return "prelu";
  }
  return "?";
}

enum class Op {
  Conv,      // dense, one [K,N] GEMM per output pixel
  DwConv,    // ONE filter per channel -- the op the array backend cannot
             // express, and 62 of this container's 161 convolutions are it
  Add,       // the mobilenet residual, with the activation possibly fused
  MaxPool,
  Resize,    // ONNX Resize, linear, half_pixel, exclude_outside=0. Its grid is
             // half-pixel and each of the other three coordinate transforms is
             // half a source pixel out, which is a pyramid shifted by half a
             // cell at every level rather than an error anything would report.
  D2S,       // DepthToSpace -- see the note on the two orders below. This is
             // how the detector builds its 28/14/7 pyramid: one 7x7x1152 map
             // becomes three levels, so it is the op this architecture exists
             // partly to express.
};

struct Layer {
  Op op = Op::Conv;
  int conv = -1;        // index into Geometry::det_convs / pose_convs
  Act act = Act::None;
  int prelu = -1;       // which PReLU slope vector, for Act::PreLU
  int64_t stride_h = 1, stride_w = 1;
  int64_t pad[4] = {0, 0, 0, 0};   // top, left, bottom, right
  int64_t kernel_h = 1, kernel_w = 1;   // MaxPool
  double scale_h = 1.0, scale_w = 1.0;   // Resize
  int64_t block = 0;   // D2S: blocksize
  bool crd = false;    // D2S: true for CRD, false for DCR (ONNX's default)
  int64_t out_c = 0, out_h = 0, out_w = 0;   // the graph's own declared output
  std::vector<int> inputs;
  std::string stream;
};

// One convolution's weights, as read from the container. [Cout, Cin, kh, kw]
// row-major, which is what the backend consumes. `group` is 1 for dense and C
// for depthwise: it is recorded rather than inferred from the weight's shape so
// that a container claiming group=C on a Cin=1 filter is refused below rather
// than running as a dense convolution with one input channel.
struct ConvW {
  int index = -1;
  int64_t cout = 0, cin = 0, kh = 0, kw = 0, group = 1;
  const float *w = nullptr;   // cout*cin*kh*kw, or cout*kh*kw when depthwise
  const float *b = nullptr;   // cout, or null
  bool depthwise() const { return group > 1; }
};

// One PReLU's slope vector. Neither network here has one -- both fuse ReLU or
// ReLU6 -- so this is read for the same reason arch=7 reads it: a container that
// named a slope must be able to supply it, and the check is the only way to know
// that without running it.
struct Slope {
  const float *v = nullptr;
  int64_t n = 0;
};

// The detector's head, per pyramid level. `box_op` and `score_op` are GRAPH NODE
// INDICES, not tensor indices: the head consumes two 1x1 convolution outputs that
// live in the middle of the graph, and a head that had to be told which tensor by
// position would break the moment an op was inserted.
struct DetLevel {
  int box_op = -1, score_op = -1;
  int64_t h = 0, w = 0;              // the level's cells per side
  int64_t per = 0;                   // anchors per cell
  int64_t rows = 0;                  // h*w*per, recorded and checked
  int64_t terms = 0;                 // 12 for every level here
};

// The landmark network's five outputs, each read out of ONE graph node.
//
// `decl` is the graph's declared shape and `conv_out` the convolution's, both
// recorded. Their RANKS DIFFER and that is not an error: MediaPipe declares the
// landmarks as [1,195] and the mask as [1,256,256,1], and a Reshape is what
// connects them. What must hold -- and what the reader checks -- is that the two
// shapes carry the same NUMBER of values, because a Reshape and a Transpose each
// preserve the element count. That single invariant covers all five without
// assuming a layout, and it is what catches a re-export that changed 39 rows into
// something else while every shape still parsed.
//
// `transposed` is the one that matters for the ORDER, since the heatmap's
// [0,2,3,1] genuinely reorders it while the mask's [1,256,256,1] reshape does not.
struct PoseOut {
  std::string name;             // landmarks, conf, mask, heatmap, world
  int op = -1;                  // graph node index
  std::vector<int64_t> decl;    // the graph's declared shape, any rank
  int64_t conv_out[4] = {1, 0, 0, 0};   // always [1, C, H, W]: the node is NCHW
  bool transposed = false;
  bool sigmoid = false;
};

// The five names, in the order the zoo returns them. Named rather than indexed
// so that a container missing one is a refusal that can say which.
inline constexpr const char *kPoseOutNames[5] = {"landmarks", "conf", "mask",
                                                 "heatmap", "world"};

struct Geometry {
  // -- both networks
  int64_t det_input_size = 0;
  int64_t pose_input_size = 0;
  int64_t num_landmarks = 0;
  int64_t num_keypoints = 0;
  int64_t lm_cols = 0, world_cols = 0;
  // TWO normalisations, because the two front ends are two models' worth of
  // preprocessing. A single pair would be a detector fed the wrong range and a
  // confidence that happened to look plausible.
  std::vector<float> det_mean, det_std;
  std::vector<float> pose_mean, pose_std;

  // -- the detector
  int64_t det_num_anchors = 0;
  int64_t det_row_terms = 0;
  int64_t det_landmarks = 0;   // the 4 keypoints its box row carries
  double score_threshold = 0.5;
  double nms_threshold = 0.3;
  int64_t top_k = 0;
  std::vector<DetLevel> det_levels;

  // -- the landmark network's geometry. The rotation is between the mid-hip and
  // the full-body point, and WHICH two is read from the container rather than
  // assumed to be (0, 1): the wrong pair rotates by some other angle, the network
  // still returns 39 rows, and every one of them is somewhere plausible.
  int64_t person_lm_mid_hip = 0;
  int64_t person_lm_full_body = 1;
  double person_box_pre_enlarge = 1.0;   // applied before the RoI is squared
  double person_box_enlarge = 1.25;      // applied to the REPORTED box
  double pose_conf_threshold = 0.5;
  std::vector<PoseOut> pose_head;
  // The outputs this front end does not read. MediaPipe's heatmap exists to
  // refine the landmarks -- the zoo's own comment says so and never calls it --
  // so it is computed because it is in the graph and carried here so that a
  // reader does not go looking for a consumer that isn't there.
  std::vector<std::string> pose_unused;

  // -- the two graphs
  std::vector<Layer> det_graph, pose_graph;
  std::vector<ConvW> det_convs, pose_convs;
  std::vector<std::string> det_nodes, pose_nodes;   // node names, for errors
  std::vector<Slope> det_slopes, pose_slopes;
  // Widened/dequantised weight storage, owned here for the lifetime of the
  // Geometry: ConvW::w is a `const float *`, and a bf16 or int8/int4 container
  // holds bytes that do not exist as floats until someone widens them or
  // multiplies an int by a scale, once, at load. Detector convs first, then
  // landmark convs, each in its graph's own index order -- the pointer is what
  // ConvW::w keeps, so the ordering is for a reader of this file rather than
  // for the convolution. Empty on an f32 container, whose ConvW::w points
  // straight into the mapping. (pose/geometry.hpp says the same thing at
  // greater length; this is the same field.)
  std::vector<std::vector<float>> w_storage;
  // The 2254 SSD anchor centres, STORED. The packer generates them and reproduces
  // the zoo's literal table bit for bit, so shipping the table keeps that a fact
  // about one machine's packing rather than about every machine's arithmetic.
  std::vector<double> anchors;

  const ConvW &conv_of(const std::vector<ConvW> &tab, const Layer &l,
                       const std::string &label) const;
  const PoseOut &pose_out(const std::string &name,
                          const std::string &label) const;
};

inline const char *op_to_string(Op o) {
  switch (o) {
    case Op::Conv: return "conv";
    case Op::DwConv: return "dwconv";
    case Op::Add: return "add";
    case Op::MaxPool: return "maxpool";
    case Op::Resize: return "resize";
    case Op::D2S: return "d2s";
  }
  return "?";
}

// Read both graphs, their weights and both heads. `f` must outlive the returned
// Geometry, whose ConvW::w pointers point into it.
Geometry read_geometry(npue::File &f, const std::string &label = std::string());

// -- refusals, in the order they are cheap to check --------------------------------

inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        label + ": " + e.what() +
        " -- not a body-pose container, or one packed by an older packer (arch " +
        f.config_string("arch") + ")");
  }
}

inline double need_double(const npue::File &f, const std::string &key,
                          const std::string &label) {
  try {
    return f.config_double(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + e.what() + " -- " + key +
                             " is not in the container");
  }
}

inline std::vector<double> need_doubles(const npue::File &f,
                                        const std::string &key, int64_t want,
                                        const std::string &label) {
  const std::string raw = f.config_string(key);
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": " + key + " is not readable JSON: " + raw);
  }
  std::vector<double> out;
  for (const auto &e : v.as_array())
    out.push_back(static_cast<double>(e.as_number()));
  if (want > 0 && static_cast<int64_t>(out.size()) != want)
    throw std::runtime_error(label + ": " + key + " has " +
                             std::to_string(out.size()) + " entries, expected " +
                             std::to_string(want));
  return out;
}

inline npue::json::Value need_json(const npue::File &f, const std::string &key,
                                   const std::string &label) {
  const std::string raw = f.config_string(key);
  try {
    return npue::json::parse(raw);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + key +
                             " is not readable JSON (" + e.what() + "): " + raw);
  }
}

inline Op op_from_string(const std::string &s, const std::string &label) {
  if (s == "conv") return Op::Conv;
  if (s == "dwconv") return Op::DwConv;
  if (s == "add") return Op::Add;
  if (s == "maxpool") return Op::MaxPool;
  if (s == "resize") return Op::Resize;
  if (s == "d2s") return Op::D2S;
  throw std::runtime_error(
      label + ": unknown op '" + s +
      "' in the packed graph. This build walks conv, dwconv, add, maxpool, "
      "resize and d2s. d2s is arch=8's own and is not a spelling of anything "
      "arch=6 or arch=7 has -- it is the detector's DepthToSpace, which builds "
      "the three pyramid levels out of one map and cannot be folded away -- "
      "while this build has no pad_c, which is arch=7's channel-widening step "
      "and which neither of these two graphs contains.");
}

inline Act act_from_string(const std::string &s, const std::string &label) {
  if (s == "none" || s == "null") return Act::None;
  if (s == "relu") return Act::Relu;
  if (s == "relu6") return Act::Relu6;
  if (s == "prelu") return Act::PreLU;
  throw std::runtime_error(label + ": activation '" + s +
                           "' is not one of none/relu/relu6/prelu. Both of these "
                           "networks fuse their activation into the convolution's "
                           "epilogue or into the residual add, and an unknown one "
                           "is not a pass this runtime can invent -- it would be a "
                           "different network.");
}

}  // namespace npue::mppose
