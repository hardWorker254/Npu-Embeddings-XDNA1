//===- geometry.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=7 hand geometry, read from the container.
//
// WHAT IS IN THIS CONTAINER
// -------------------------
// One .npue holds TWO networks and a front end, where every other container
// holds one:
//
//   palm  MediaPipe's palm detector, a 192x192 SSD-style detector with a 2-level
//         feature pyramid, 30 dense convolutions, 23 depthwise ones, and a head
//         that emits 2016 rows of 18 numbers plus 2016 scores.
//   lm    MediaPipe's hand-landmark network, a 224x224 mobilenet-shaped
//         classifier-ish regressor, 31 dense convolutions, 16 depthwise ones,
//         and a head that GLOBAL-AVERAGE-POOLS a 672-channel 7x7 map and
//         projects it to 63 + 1 + 1 + 63.
//
// Both are in the file because a hand is not a palm and a palm is not a hand:
// the second consumes a crop rotated by a rotation the FIRST predicted. A
// container per network could not express that dependency, and a runtime handed
// two containers would have to trust the caller to have paired a detector with
// a landmark net trained for its output -- which is the one pairing fact that
// cannot be checked at run time.
//
// WHY NOT arch=6's STRUCT WITH FIELDS TURNED OFF
// -----------------------------------------------
// It already could not (arch=6's own header says why: a CNN has no CLS row and
// no single token sequence). What is new here is that the two heads are
// DIFFERENT SHAPES of thing rather than one shape with a flag:
//
//   palm head   a DETECTION head: per-anchor, four NCHW feature maps reshaped
//               into a [2016, 18] table, with the row order being the graph's
//               own and not a C-major flatten.
//   lm head     a REGRESSION head: one global average pool, then four Gemms.
//
// So there are two head descriptions, two graphs, and two convolution tables,
// and a shape that assumed one of each would be wrong in a way that typechecks.
//
// THE GRAPH IS A LIST, NOT A SHAPE
// -------------------------------
// As in arch=6: which convolutions there are, in which order, what feeds what,
// and where the activations are is the model. The container carries an explicit
// op list per network and the runtime walks it.
//
// FOUR PADDING TERMS, NOT TWO
// ---------------------------
// arch=6 pads symmetrically. arch=7 does not: MediaPipe evaluates "SAME"
// padding at a stride, and the palm detector's stem is [top,left,bottom,right]
// = [1,1,2,2] for a 3x3 stride-2 kernel. A symmetric transcription shifts the
// whole pyramid by one pixel, which is invisible in the output shape and moves
// every detection. The packer writes four terms for that reason and read_graph()
// refuses an op whose four terms do not describe a whole-number output extent.
//
// THE ANCHORS ARE GENERATED, NOT STORED
// --------------------------------------
// palm_anchor_levels records [(h, w, anchors_per_cell)] and the table is
// derived from it here and in the packer, rather than shipped as 2016 x 2
// floats of opaque numbers. The ORDER is the graph's: h outer, w inner, and
// therefore the flat row index advances w -- X -- fastest, which is why the
// first component of a row is x. Transposing the two components mirrors the
// whole table and still leaves 2016 well-shaped rows and a confident detector.
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

namespace npue::hands {

// The arch=7 container's `arch` string. Refused by name rather than run with
// the wrong head geometry, which is the whole point of carrying one.
inline constexpr const char *kArch = "mediapipe_hands_palm_ssd_lm_heatmap";

// MediaPipe's palm detector emits SEVEN keypoints and the landmark network
// twenty-one. Recorded rather than assumed, and the two are refused against
// each other in read_geometry: a landmark head of the wrong width would produce
// a hand with a plausible number of joints and the wrong ones.
inline constexpr int64_t kPalmLandmarks = 7;
inline constexpr int64_t kHandLandmarks = 21;

// The palm detector's box row: cx, cy, w, h, then 7 x 2 keypoint offsets, all
// as FRACTIONS of the input side and all relative to the anchor.
inline constexpr int64_t kPalmRowTerms = 18;

// The activation, as the container records it. Distinct from hostconv::Act on
// purpose: that enum is what the TRANSPOSE can compute, and this is what the
// graph says, and a packer that recorded one thing while the runtime believed
// another is the failure -- so the string is parsed here, refused if it is not
// one of three, and mapped across at the single call site.
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
             // express, and the reason 39 of this container's 100 convolutions
             // stay on the host by name rather than silently becoming dense
  Add,       // the mobilenet residual, with the activation possibly fused
  MaxPool,
  PadC,      // append `count` ZERO channels -- the mobilenet "pad to 2x" step
             // that is a channel-wise concat with a constant, not a pixel pad
  Resize,    // ONNX Resize, linear, half_pixel, exclude_outside=0 -- the neck's
             // upsampling. Its grid is half-pixel and putting the other three
             // coordinate transforms here would produce a plausible detector
             // displaced by half a source pixel per level.
};

// One node of one network's packed graph.
struct Layer {
  Op op = Op::Conv;
  int conv = -1;        // index into Geometry::convs, for Conv and DwConv
  Act act = Act::None;
  int prelu = -1;       // which PReLU slope vector, for Act::PreLU
  int64_t stride_h = 1, stride_w = 1;
  int64_t pad[4] = {0, 0, 0, 0};   // top, left, bottom, right
  int64_t kernel_h = 1, kernel_w = 1;   // MaxPool
  double scale_h = 1.0, scale_w = 1.0;   // Resize
  int64_t count = 0;   // PadC: how many zero channels to append
  int64_t out_c = 0, out_h = 0, out_w = 0;   // the graph's own declared output
  std::vector<int> inputs;
  // Which design stream a DENSE convolution dispatches on when the array
  // backend is selected -- empty when the container was packed without array
  // panels. It is a NAME and not a (K, N) pair because the padded shape is the
  // design's business; see pose/geometry.hpp's Layer::stream for why.
  std::string stream;
};

// One convolution's weights, as read from the container. [Cout, Cin, kh, kw]
// row-major, which is what both backends consume. `group` is 1 for dense and C
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

// One PReLU's slope vector, one per fused activation.
struct Slope {
  const float *v = nullptr;
  int64_t n = 0;
};

// The palm detector's head, per pyramid level. `box` and `score` are GRAPH NODE
// INDICES, not tensor indices: the head consumes four feature maps that live
// somewhere in the middle of the graph, and a head that had to be told which
// tensor by position would break the moment an op was inserted.
struct PalmLevel {
  int box = -1, score = -1;
  int64_t h = 0, w = 0;              // the level's cells per side
  int64_t anchors_per_cell = 0;
  int64_t rows = 0;                  // h*w*a, recorded and checked
};

// The landmark network's head: a global average pool of one graph node, then
// Gemms whose names are TENSOR NAMES because they are not convolutions and do
// not appear in any graph node list.
struct LmProj {
  std::string w, b;
  int64_t n = 0;
  const float *wp = nullptr;
  const float *bp = nullptr;
};

struct LmOutSpec {
  int64_t n = 0;
  bool sigmoid = false;
  int proj = 0;
};

struct LmHead {
  int pool = -1;              // graph node whose output is pooled
  int64_t features = 0;       // the pooled width
  int64_t height = 0, width = 0;   // the map it pools, recorded
  std::vector<LmProj> projs;
  std::vector<LmOutSpec> outputs;
};

// One network's front-end and decode constants. Both networks share the image
// normalisation but not the crop geometry, so the two are separate structs
// rather than one with a flag -- a shared struct would need a `which` and every
// reader would have to check it.
struct Crop {
  double shift[2] = {0.0, 0.0};
  double enlarge = 1.0;
};

struct Geometry {
  // -- both networks
  int64_t palm_input_size = 0;
  int64_t lm_input_size = 0;
  int64_t num_landmarks = 0;
  std::vector<float> mean, std_dev;   // per channel; length 3, refused otherwise

  // -- the palm detector
  int64_t palm_num_anchors = 0;
  int64_t palm_row_terms = 0;
  int64_t palm_lm_wrist = 0;          // which of the 7 chooses the rotation
  int64_t palm_lm_middle_base = 0;    // and which the other end of it
  Crop palm_pre;                      // for the ROTATION crop (enlarged by 4)
  Crop palm_post;                     // for the FINAL crop (enlarged by 3)
  Crop hand;                          // for the reported box (enlarged by 1.65)
  double score_threshold = 0.5;
  double nms_threshold = 0.3;
  int64_t top_k = 0;
  std::vector<PalmLevel> palm_levels;

  // -- the landmark network
  LmHead lm_head;

  // -- the two graphs
  std::vector<Layer> palm_graph, lm_graph;
  std::vector<ConvW> palm_convs, lm_convs;
  std::vector<std::string> palm_nodes, lm_nodes;   // graph node names, for errors
  std::vector<Slope> palm_slopes, lm_slopes;
  // The 2016 SSD anchor centres, GENERATED from palm_levels rather than read,
  // in the graph's own row order (see this file's header).
  std::vector<double> anchors;

  const ConvW &conv_of(const std::vector<ConvW> &tab, const Layer &l,
                       const std::string &label) const;
};

inline const char *op_to_string(Op o) {
  switch (o) {
    case Op::Conv: return "conv";
    case Op::DwConv: return "dwconv";
    case Op::Add: return "add";
    case Op::MaxPool: return "maxpool";
    case Op::PadC: return "pad_c";
    case Op::Resize: return "resize";
  }
  return "?";
}

// Read both graphs and their weights. `f` must outlive the returned Geometry,
// whose ConvW::w pointers point into it.
Geometry read_geometry(npue::File &f, const std::string &label = std::string());

// -- refusals, in the order they are cheap to check --------------------------------

inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        label + ": " + e.what() +
        " -- not a hand container, or one packed by an older packer (arch " +
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
  if (s == "pad_c") return Op::PadC;
  if (s == "resize") return Op::Resize;
  throw std::runtime_error(
      label + ": unknown op '" + s +
      "' in the packed graph. This build walks conv, dwconv, add, maxpool, "
      "pad_c and resize; a packer that emits anything else produced a network "
      "this runtime would have to guess at. The two networks' op "
      "vocabularies are their own -- arch=7 has no upsample and no concat "
      "because neither network has one -- so this list is not arch=6's.");
}

inline Act act_from_string(const std::string &s, const std::string &label) {
  if (s == "none" || s == "null") return Act::None;
  if (s == "relu") return Act::Relu;
  if (s == "relu6") return Act::Relu6;
  if (s == "prelu") return Act::PreLU;
  throw std::runtime_error(label + ": activation '" + s +
                           "' is not one of none/relu/relu6/prelu. These two "
                           "networks fuse their activation into the "
                           "convolution's epilogue or into the residual add, "
                           "and an unknown one is not a pass this runtime can "
                           "invent -- it would be a different network.");
}

}  // namespace npue::hands