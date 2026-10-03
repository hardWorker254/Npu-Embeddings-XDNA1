//===- geometry.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 body-pose geometry, read from the container. The
// third modality (pixels like arch=5, but a CNN over a feature pyramid rather
// than a transformer over patches) and the first with a DETECTION head.
//
// WHY A THIRD GEOMETRY AND NOT ViT's WITH FIELDS TURNED OFF
// ---------------------------------------------------------
// arch=5's geometry is a transcript of a ViT: one image size, one patch, one
// CLS row, one label row. A pose model has none of those and every field of
// ViT's would be a lie that reads as a number: there is no single token
// sequence to pad (there are 8400 of them, at three scales), no CLS row (there
// is a grid), and the head's output is not one label per image but 4 boxes x
// (1 class + 17 keypoints x 3) per scale. So this is its own struct, read from
// its own keys, refused when a key is missing.
//
// THE GRAPH IS A LIST, NOT A SHAPE
// --------------------------------
// An embedder's geometry is three numbers because the runtime knows what a
// transformer layer is. A conv net's is not: which convolutions there are, in
// which order, what feeds what, and where the SiLUs are is the model. So the
// container carries an explicit op LIST (see Layer) and the runtime walks it. A
// packer that could emit a list that does not execute would be a packer that
// invented a network, so read_graph() refuses a list whose operand counts and
// arities do not typecheck.
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

#include "common/json_min.hpp"   // npue::json::parse
#include "runtime/model.hpp"

namespace npue::pose {

// The arch=6 container's `arch` string. One string, so a container packed for a
// different YOLOv8 variant is refused by name rather than run with the wrong
// head geometry.
inline constexpr const char *kArch = "yolov8_pose_c2f_silu_dfl";

// 8400 anchors = 80x80 + 40x40 + 20x20, i.e. the three strides 8/16/32 over a
// 640px letterboxed input. Recorded, and REFUSED rather than recomputed: the
// decode step divides by strides[level] and multiplies by grid[level], and a
// mismatch between those two and the tensor the network actually produced is a
// set of plausible keypoints in the wrong places.
inline constexpr int64_t kNumKeypoints = 17;   // COCO
inline constexpr int64_t kDflBins = 16;        // the head's 1x1 conv, /2

enum class Op {
  Conv,       // one convolution, with an optional fused SiLU
  Concat,     // channel-wise, axis 1, any number of operands -- the PAN neck's
              // joins and SPPF's four-way one
  Slice,      // a contiguous run of CHANNELS out of a tensor -- how the C2f
              // block's channel split is expressed, one node per half
  Add,        // the C2f residual
  MaxPool,    // SPPF's pooling
  Upsample,   // nearest-neighbour, the neck's up path
  // The head. Consumes, in order, two tensors per level: the box+class head's
  // output and the keypoint head's output. Emits the canonical
  // [4 + nc + nk*3, cells] tensor -- xyxy boxes in letterbox pixels, scores and
  // visibilities as probabilities, keypoints in letterbox pixels -- which is
  // the ONLY shape decode.cpp reads. See HeadInfo for what the two inputs mean.
  Detect,
};

// WHAT THE PACKED GRAPH'S `detect` NODE IS FUELED, AND WHAT IT EMITS
// ----------------------------------------------------------------
// The exported graph's last dozen nodes -- the DFL softmax, the grid anchor
// addition, the stride multiply, the box and keypoint transforms -- are NOT
// convolutions, and this runtime's graph engine walks convolutions. So the
// boundary is drawn at that arithmetic and it is drawn EXPLICITLY, in the
// container, in three fields:
//
//   head_box_format   how to read the 4*16 channels each level's box head emits
//   head_score        whether the class head's output is a logit or a probability
//   head_keypoint_visibility   likewise for the keypoint head
//
// `detect` consumes those per-level tensors and emits ONE canonical tensor:
// [4 + nc + nk*3, cells], boxes as xyxy in LETTERBOX PIXELS, the class score and
// each keypoint's visibility already as probabilities, keypoints in letterbox
// pixels. decode.cpp then reads a canonical tensor and knows nothing about the
// exporter -- which is what stops the two from having to agree about which of
// two conventions the exporter happened to choose.
//
// These are recorded by the packer, which TRACED them out of the ONNX, rather
// than detected at run time. A heuristic that guesses "the boxes look like
// corners" is a heuristic that will be wrong on the one model where it matters,
// and the difference between two conventions is not noise: xyxy read as cxcywh
// puts every box inside the top-left quadrant and NMS then deletes most of them,
// so the failure presents as an empty result rather than a visibly wrong one.
enum class BoxFormat {
  LtrbDfl,   // 4 * dfl_bins channels of logits; the head softmaxes, takes the
             // expectation over the bins, adds the grid anchor and scales by
             // the stride. YOLOv8's own export.
  Xyxy,      // 4 channels, already corners, in pixels
  CxCyWh,    // 4 channels, already centre + full width/height, in pixels
};

enum class ScoreKind {
  Sigmoid,   // already squashed to 0..1 by the packed graph
  Logit,     // raw; the head applies the sigmoid
};

struct HeadInfo {
  BoxFormat box = BoxFormat::Xyxy;
  ScoreKind score = ScoreKind::Sigmoid;
  ScoreKind kpt_visibility = ScoreKind::Sigmoid;

  // Which convolution is the DFL's projection, and whether it carries a bias.
  //
  // It is a CONV INDEX and not a graph node because the graph engine cannot run
  // it: its input is a Softmax, and no op in the op list consumes a
  // distribution. Network::head folds it in analytically instead --
  //
  //   sum_bin w[0][bin] * p[bin] + b  ==  w[0] . E[bin] + b   (sum p == 1)
  //
  // -- so the head needs the WEIGHTS, not a place in the list, and -1 means the
  // container states none. A default of 0 would silently apply the FIRST
  // convolution's weights to every distance, which for this architecture are
  // the stem's: the boxes would be wrong by a factor and nothing downstream
  // could tell.
  //
  // The bias is a separate flag rather than a pointer because the container
  // does not STORE one for this convolution -- read_geometry refuses a bias-free
  // convolution that does not say so, and the fold above needs to know which of
  // the two cases it is in. dfl_bias is true only when the tensor exists.
  int dfl_conv = -1;
  bool dfl_bias = false;

  // The two grids' half-cell offsets, in CELLS, added to (column, row) before
  // the stride multiply.
  //
  // They are not the same number and that is not a typo in the runtime. This
  // exporter anchors its BOXES at cell centres ((j, i) + 0.5) and its
  // KEYPOINTS at cell corners ((j, i) + 0), from two different constants in the
  // graph. Assuming they agreed puts every joint of every person a half a cell
  // out -- 4, 8 and 16 pixels at strides 8, 16 and 32 -- while the boxes stay
  // perfect, so a pose looks broadly right and every limb is misplaced by a
  // distance that grows with the pyramid level rather than with anything in the
  // picture. read_geometry requires both to be finite and Network::head adds
  // exactly what the container recorded; the packer reads them out of the ONNX's
  // own grid constants rather than out of a table of what exporters do.
  float box_grid_offset = 0.5f;
  float kpt_grid_offset = 0.0f;
};

// One node of the packed graph.
struct Layer {
  Op op = Op::Conv;
  int conv = -1;        // index into Geometry::convs, for Op::Conv
  bool silu = false;    // the activation, fused into the conv's epilogue
  int64_t k = 0;        // MaxPool kernel
  // Conv stride, and the padding added on each side before the window. Both are
  // PER NODE and not global defaults: this network's stem is a 3x3 with stride
  // 2, its 1x1s pad nothing, and its 3x3s pad one -- so "stride 1, pad k/2"
  // would be right for 70 of the 72 graph convolutions and wrong for the two that
  // decide the input's size.
  int64_t stride = 0;   // Conv stride, and MaxPool stride
  int64_t pad_h = 0;    // Conv: rows added above and below
  int64_t pad_w = 0;    // Conv: columns added left and right
  int64_t scale = 0;    // Upsample factor
  int64_t chan = 0;     // Slice: first channel taken
  int64_t nch = 0;      // Slice: how many
  std::vector<int> inputs;
  // For Op::Conv, the design stream this convolution dispatches on when the
  // array backend is selected -- empty when the container was packed without an
  // array panel for it.
  //
  // It is a NAME and not a (K, N) pair because the padded shape is the design's
  // business: the exporter chose tile_n, and a runtime that re-derived the
  // padded N from the conv's own cout would disagree with the xclbin whenever
  // those two disagreed, and the result would be a correct GEMM over a wrong
  // width. Reading the name the packer wrote keeps the pair on one side of the
  // fence. A conv with no stream still runs, on the host -- see net.hpp for why
  // that is the default.
  std::string stream;
};

// One convolution's weights, as read from the container. The layout is the
// checkpoint's own, [Cout, Cin, kh, kw] row-major, because that is what both
// backends consume: the host path indexes it directly and the NPU path's B
// panel is built from it by the same im2col-as-reshape the host path uses.
struct ConvW {
  int index = -1;    // its own position in Geometry::convs, for the backend
  int64_t cout = 0, cin = 0, kh = 0, kw = 0;
  const float *w = nullptr;   // cout*cin*kh*kw
  const float *b = nullptr;   // cout
};

struct Geometry {
  int64_t input_size = 0;
  int64_t num_keypoints = 0;
  int64_t num_classes = 0;
  int64_t dfl_bins = 0;

  // One entry per detection level. strides[i] is the level's stride in pixels
  // and grid[i] its cells per side; anchors[i] == grid[i]^2.
  std::vector<int64_t> strides;
  std::vector<int64_t> grid;

  // The letterbox front end. YOLOv8 normalises with mean 0 and std 1 and takes
  // RGB, so these are the identity -- but they are RECORDED, because a container
  // that says otherwise has been packed from a checkpoint that was normalised
  // differently and every keypoint in it would be wrong by an amount no later
  // stage could detect.
  std::vector<float> mean, std_dev;
  bool pad_value_centre = true;   // letterbox fills with the mean, i.e. 114/255

  std::vector<Layer> graph;
  std::vector<ConvW> convs;

  // Dequantised weight storage, one entry per convolution, owned here for the
  // lifetime of the Geometry.
  //
  // IT IS HERE AND NOT A VIEW INTO THE CONTAINER because ConvW::w is a
  // `const float *`, and on a quantised container the floats do not exist until
  // someone multiplies an int8 byte by a scale. The alternative -- keeping the
  // int8 bytes and dequantising inside the convolution -- would do that multiply
  // once per weight per image, on 3.28 M weights, to save a 3.3 MB copy made
  // once. Quantisation here buys bytes on disk and nothing else on the host
  // path, and this is the line where that is decided.
  //
  // Empty for an fp32 container, whose ConvW::w points straight into the
  // mapping. It is sized to num_convs, not to the number of quantised
  // convolutions, so `convs[i]`'s storage is at a fixed index.
  std::vector<std::vector<float>> w_storage;

  // Channels the tensor leaving Detect has to have, which is what the decode
  // step reads: four box values, then the class score, then the keypoints.
  //
  // FOUR, not zero. An earlier version of this check computed the class and
  // keypoint widths alone and so rejected the correct 56 for a 17-point head,
  // which is the width this architecture produces. The four are not optional:
  // decode.cpp reads row[0..3] for the box, so a tensor without them would be
  // read as if the first keypoint were the box.
  int64_t out_channels() const {
    return 4 + num_classes + num_keypoints * 3;
  }
  int64_t n_anchors() const {
    int64_t n = 0;
    for (int64_t g : grid) n += g * g;
    return n;
  }
  int64_t n_levels() const { return static_cast<int64_t>(strides.size()); }
  // Channels the tensor leaving Detect has to have, which is what the decode
  // step reads. Checked against the graph's last conv so a head whose weights
  // do not match the recorded geometry is refused before a single pixel moves.
  int64_t detect_cout = 0;
  int64_t detect_h = 0, detect_w = 0;
  HeadInfo head;
};

// The COCO 17-point skeleton, as NAMES and as the edges that join them.
//
// This lives in the RUNTIME, not in the container, and that asymmetry is
// deliberate: index 5 of the head tensor means the same thing for every COCO
// pose model ever packed, and the number of keypoints is already refused
// against it in read_geometry. A container that named its own joints would move
// a fixed index-space agreement into per-file data, where two models of the
// same skeleton could disagree about it and nothing would compare them.
struct Skeleton {
  static constexpr int kPoints = 17;
  // COCO's 19 bones, counted over both sides of the body, are these 12
  // undirected pairs (10 limb segments plus the two that close the torso).
  // The count is recorded rather than left implicit because the JSON and the
  // Python facade both emit the edge list, and a length that disagrees between
  // them shows up as a client drawing eleven bones and calling it a skeleton.
  static constexpr int kEdges = 12;
  static const char *name(int i);
  static bool pair(int edge, int &a, int &b);
};

// -- refusals, in the order they are cheap to check --------------------------------

inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        label + ": " + e.what() +
        " -- not a pose container, or one packed by an older packer (arch " +
        f.config_string("arch") + ")");
  }
}

inline std::vector<int64_t> need_ints(const npue::File &f,
                                      const std::string &key, int64_t want,
                                      const std::string &label) {
  const std::string raw = f.config_string(key);
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": " + key + " is not readable JSON: " + raw);
  }
  std::vector<int64_t> out;
  for (const auto &e : v.as_array()) out.push_back(static_cast<int64_t>(e.as_number()));
  if (want > 0 && static_cast<int64_t>(out.size()) != want)
    throw std::runtime_error(label + ": " + key + " has " +
                             std::to_string(out.size()) + " entries, expected " +
                             std::to_string(want));
  return out;
}

inline std::vector<float> need_floats(const npue::File &f,
                                      const std::string &key, int64_t want,
                                      const std::string &label) {
  const std::string raw = f.config_string(key);
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": " + key + " is not readable JSON: " + raw);
  }
  std::vector<float> out;
  for (const auto &e : v.as_array()) out.push_back(static_cast<float>(e.as_number()));
  if (static_cast<int64_t>(out.size()) != want)
    throw std::runtime_error(
        label + ": " + key + " has " + std::to_string(out.size()) +
        " entries and this image has " + std::to_string(want) +
        " channels. The normalisation is per channel, so a length that does "
        "not match has no per-pixel meaning.");
  return out;
}

inline Op op_from_string(const std::string &s, const std::string &label) {
  if (s == "conv") return Op::Conv;
  if (s == "concat") return Op::Concat;
  if (s == "slice") return Op::Slice;
  if (s == "add") return Op::Add;
  if (s == "maxpool") return Op::MaxPool;
  if (s == "upsample") return Op::Upsample;
  if (s == "detect") return Op::Detect;
  throw std::runtime_error(label + ": unknown op '" + s +
                           "' in the packed graph. This build walks conv, "
                           "concat, add, maxpool, upsample and detect; a "
                           "packer that emits anything else produced a network "
                           "this runtime would have to guess at.");
}

inline std::string op_to_string(Op o) {
  switch (o) {
    case Op::Conv: return "conv";
    case Op::Concat: return "concat";
    case Op::Slice: return "slice";
    case Op::Add: return "add";
    case Op::MaxPool: return "maxpool";
    case Op::Upsample: return "upsample";
    case Op::Detect: return "detect";
  }
  return "?";
}

inline BoxFormat box_format_from_string(const std::string &s,
                                        const std::string &label) {
  if (s == "ltrb_dfl") return BoxFormat::LtrbDfl;
  if (s == "xyxy") return BoxFormat::Xyxy;
  if (s == "cxcywh") return BoxFormat::CxCyWh;
  throw std::runtime_error(
      label + ": head_box_format '" + s +
      "' is not one this build implements (ltrb_dfl, xyxy, cxcywh). The "
      "convention is what the exporter baked into the graph, and reading one "
      "as another moves every box by half its size -- which NMS then mostly "
      "deletes, so the failure looks like an empty result rather than a wrong "
      "one.");
}

inline ScoreKind score_kind_from_string(const std::string &s,
                                        const std::string &label) {
  if (s == "sigmoid") return ScoreKind::Sigmoid;
  if (s == "logit") return ScoreKind::Logit;
  throw std::runtime_error(label + ": score convention '" + s +
                           "' is not one this build implements (sigmoid, "
                           "logit). The head's activation is part of the "
                           "exported graph and has to be recorded, not "
                           "assumed: a score compared against 0.25 that was "
                           "never squashed is a threshold on a different "
                           "scale, so the two models would return different "
                           "people at the same setting.");
}

// Read the graph and its weights. `load` fills every ConvW pointer from the
// container; the pointers stay valid as long as `f` is.
Geometry read_geometry(npue::File &f, const std::string &label = std::string());

}  // namespace npue::pose
