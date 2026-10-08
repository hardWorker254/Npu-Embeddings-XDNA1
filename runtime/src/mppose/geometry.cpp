//===- geometry.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read the arch=8 container's two graphs, their weights and
// both heads. See geometry.hpp for what is in one of these and why.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/geometry.hpp"

#include "common/host_kernels.hpp"   // bf16_read

#include <algorithm>
#include <sstream>

namespace npue::mppose {

namespace {

int64_t jint(const npue::json::Value &o, const std::string &key,
             const std::string &label) {
  const npue::json::Value *const v = o.find(key);
  if (!v)
    throw std::runtime_error(label + ": no '" + key +
                             "'. The packed graph and this runtime have to "
                             "agree about the model's own fields, and a key "
                             "that is not there is a packer and a runtime from "
                             "different days.");
  return static_cast<int64_t>(v->as_number());
}

int64_t jopt_int(const npue::json::Value &o, const std::string &key, int64_t dflt) {
  const npue::json::Value *const v = o.find(key);
  // An explicit JSON null is an ABSENT value, not a zero. `prelu` is written as
  // null by every op whose activation is not a PReLU, and reading that as 0 would
  // make all of them into the first one's slopes.
  return (v && !v->is_null()) ? static_cast<int64_t>(v->as_number()) : dflt;
}

std::string jopt_str(const npue::json::Value &o, const std::string &key,
                     const std::string &dflt) {
  const npue::json::Value *const v = o.find(key);
  return (v && !v->is_null()) ? v->as_string() : dflt;
}

std::string jstr(const npue::json::Value &o, const std::string &key,
                 const std::string &label) {
  const npue::json::Value *const v = o.find(key);
  if (!v)
    throw std::runtime_error(label + ": no '" + key + "'");
  return v->as_string();
}

bool jbool_or(const npue::json::Value &o, const std::string &key, bool dflt) {
  const npue::json::Value *const v = o.find(key);
  return (v && !v->is_null()) ? v->as_bool() : dflt;
}

void jpair(const npue::json::Value &o, const std::string &key,
           const std::string &label, int64_t *a, int64_t *b) {
  const npue::json::Value *const v = o.find(key);
  if (!v || v->as_array().size() != 2)
    throw std::runtime_error(label + ": '" + key +
                             "' is not a pair. Every stride, kernel and scale "
                             "in these graphs is per axis and both axes have to "
                             "be stated: these two networks are square so far, "
                             "but 'square so far' is not something to assume "
                             "about a packed graph.");
  *a = static_cast<int64_t>((*v).as_array()[0].as_number());
  *b = static_cast<int64_t>((*v).as_array()[1].as_number());
}

void jpad4(const npue::json::Value &o, const std::string &label, int64_t *pad) {
  const npue::json::Value *const v = o.find("pad");
  if (!v || v->as_array().size() != 4)
    throw std::runtime_error(
        label + ": 'pad' is not four terms [top, left, bottom, right]. It is "
        "four and not two because MediaPipe evaluates SAME padding at a stride, "
        "so a 3x3 stride-2 stem is [1,1,1,1] here and not [1,1] -- and the two "
        "stems in this container happen to be symmetric, which means a "
        "transcription error here would not show up in a shape and would move "
        "every detection instead.");
  for (int i = 0; i < 4; ++i)
    pad[i] = static_cast<int64_t>((*v).as_array()[i].as_number());
}

// `wdtype` is the container's conv_weight_dtype -- "f32" when the key is
// ABSENT, which is how every container packed before the key existed reads and
// must keep reading. Whatever it says, the byte count is checked against what
// that dtype implies for this shape BEFORE anything is read: `Span::as<T>()`
// casts blindly, so a payload that contradicts its declared precision used to
// be read as a plausible fp32 array of the wrong length rather than refused.
// `w_storage` receives the widened/dequantised fp32 weight (bf16 and the two
// int dtypes) and must outlive `c`; on f32 it stays empty and `c.w` points
// straight into the mapping, exactly as before.
ConvW read_conv(npue::File &f, const std::string &pfx, int i, bool want_depthwise,
                const std::string &wdtype, int64_t i4_group,
                std::vector<float> &w_storage, const std::string &label) {
  ConvW c;
  c.index = i;
  const std::string base = pfx + ".conv." + std::to_string(i);
  const auto &wi = f.info(base + ".w");
  if (wi.logical_shape.size() != 4)
    throw std::runtime_error(label + ": " + base + ".w is not a 4-D filter");
  c.cout = wi.logical_shape[0];
  c.cin = wi.logical_shape[1];
  c.kh = wi.logical_shape[2];
  c.kw = wi.logical_shape[3];
  if (want_depthwise) {
    c.group = c.cout;
    if (c.cin != 1)
      throw std::runtime_error(
          label + ": " + base + ".w is a dwconv with " +
          std::to_string(c.cin) +
          " input channels. A depthwise filter is ONE filter per channel; a "
          "dense GEMM over this one would compute a different network, and the "
          "difference would be this architecture's accuracy rather than a "
          "crash.");
  }
  const int64_t want = c.cout * c.cin * c.kh * c.kw;
  // [Cout, K] -- the view the scale rule and the payload are both indexed in.
  const int64_t N = c.cout, K = c.cin * c.kh * c.kw;
  if (wdtype == "f32") {
    if (static_cast<int64_t>(wi.nbytes) != want * 4)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(want * 4) +
          " fp32 values for a [" + std::to_string(c.cout) + ", " +
          std::to_string(c.cin) + ", " + std::to_string(c.kh) + ", " +
          std::to_string(c.kw) + "] weight. A weight whose byte count "
          "disagrees with its shape is not a weight at a different precision; "
          "it is a number of the wrong size.");
    c.w = f.raw(base + ".w").as<float>();
  } else if (wdtype == "bf16") {
    // Two bytes per weight, the source's own bits rounded RNE, no sidecars.
    if (static_cast<int64_t>(wi.nbytes) != want * 2)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(want * 2) + " for " +
          std::to_string(want) +
          " bf16 values. A weight whose byte count disagrees with its shape is "
          "not a weight at a different precision; it is a number of the wrong "
          "size.");
    w_storage.resize(static_cast<size_t>(want));
    app::bf16_read(w_storage.data(), f.raw(base + ".w").data,
                   static_cast<size_t>(want));
    c.w = w_storage.data();
  } else {
    // int8: one byte per weight. int4: one byte per TWO weights, and K may be
    // ODD -- these stems are 3x3xC -- so the payload is ceil(K/2) bytes per
    // row with a zero pad nibble at the end. ceil, not K/2: computing it the
    // other way makes the last nibble of the last row a phantom weight.
    const int64_t packed = wdtype == "i8" ? K : (K + 1) / 2;
    if (static_cast<int64_t>(wi.nbytes) != N * packed)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(wi.nbytes) +
          " bytes, which is not " + std::to_string(N * packed) + " for a [" +
          std::to_string(N) + ", " + std::to_string(K) + "] " + wdtype +
          " weight. The scale tensors are read next and are sized against the "
          "shape, so a payload of the wrong length is caught here rather than "
          "reading past the end of it.");
    if (!f.has(base + ".wscale"))
      throw std::runtime_error(
          label + ": " + base + ".wscale is not in the container, and " +
          base + ".w says its weights are " + wdtype +
          ". An integer payload without its scale is bytes whose meaning is "
          "one multiply away and not present in the file: the reader would "
          "either invent a factor of 1 or -- if it looked the other way -- "
          "leave raw quantisation levels in a convolution that expects "
          "weights.");
    const auto &si = f.info(base + ".wscale");
    if (static_cast<int64_t>(si.nbytes) != N * 4)
      throw std::runtime_error(
          label + ": " + base + ".wscale is " + std::to_string(si.nbytes) +
          " bytes for " + std::to_string(N) +
          " output channels. The scale is per output channel -- one per row of "
          "this weight -- and a scale count that disagrees with the row count "
          "puts every channel after the last one on another channel's factor.");
    const float *const s = f.raw(base + ".wscale").as<float>();
    w_storage.resize(static_cast<size_t>(want));
    if (wdtype == "i8") {
      const int8_t *const q = f.raw(base + ".w").as<int8_t>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
          w_storage[static_cast<size_t>(n * K + k)] =
              static_cast<float>(q[n * K + k]) * s[n];
    } else {
      // The group scales are [G, N] -- indexed by group along the INPUT axis
      // first -- and the dequantisation is the rank-1 product the packer
      // measured its error against (tools/lib/conv_quant.py's
      // dequantise_payload, transcribed here exactly as pose's reader has it).
      // Zero groups are held at 1 by the packer, so a dead input channel stays
      // exactly zero instead of becoming NaN.
      const int64_t G = i4_group > 0 ? (K + i4_group - 1) / i4_group : 1;
      if (!f.has(base + ".gscale"))
        throw std::runtime_error(
            label + ": " + base + ".gscale is not in the container, and " +
            base + ".w says its weights are i4 with a group of " +
            std::to_string(i4_group) + ". The group scale is what places each "
            "weight against its own group's range; without it there is no "
            "dequantisation, only a per-channel one applied to a payload "
            "quantised against another factor.");
      const auto &gi = f.info(base + ".gscale");
      if (static_cast<int64_t>(gi.nbytes) != G * N * 4)
        throw std::runtime_error(
            label + ": " + base + ".gscale is " + std::to_string(gi.nbytes) +
            " bytes, not " + std::to_string(G * N * 4) + " -- [" +
            std::to_string(G) + ", " + std::to_string(N) + "] for a group of " +
            std::to_string(i4_group ? i4_group : K) + " over K=" +
            std::to_string(K) + ". Read with any other stride, every weight "
            "past the first group is wrong and every shape still matches.");
      const float *const gs = f.raw(base + ".gscale").as<float>();
      const uint8_t *const p = f.raw(base + ".w").as<uint8_t>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
          // Four bits, low nibble first, SIGN-EXTENDED. A nibble of 0xF is
          // -1 and not 15: reading them unsigned maps the most negative
          // weight onto the most positive one, in every group, which looks
          // like a network whose activations are too warm.
          const uint8_t byte = p[n * ((K + 1) / 2) + k / 2];
          int v = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
          if (v >= 8) v -= 16;
          const int64_t gsel = i4_group > 0 ? k / i4_group : 0;
          w_storage[static_cast<size_t>(n * K + k)] =
              static_cast<float>(v) * s[n] * gs[gsel * N + n];
        }
    }
    c.w = w_storage.data();
  }
  if (f.has(base + ".b"))
    c.b = f.raw(base + ".b").as<float>();
  return c;
}

struct GraphRead {
  std::vector<Layer> layers;
  std::vector<ConvW> convs;
  std::vector<std::string> nodes;
  std::vector<Slope> slopes;
  // Widened/dequantised fp32 weights, one entry per convolution, moved into
  // Geometry::w_storage below. Empty entries on an f32 container, whose
  // ConvW::w points into the mapping instead.
  std::vector<std::vector<float>> w_storage;
};

// Read one network. `pfx` namespaces its tensors, and it is not decoration: both
// networks number their convolutions from zero, so an unprefixed lookup reads the
// OTHER network's weights and returns a tensor of the wrong width -- which for the
// detector means a 224-input graph running on the landmark net's first layer.
GraphRead read_one_graph(npue::File &f, const std::string &pfx,
                         const std::string &graph_key,
                         const std::string &nodes_key, int64_t num_convs,
                         const std::string &wdtype, int64_t i4_group,
                         const std::string &label) {
  GraphRead g;
  const npue::json::Value ops = need_json(f, graph_key, label);
  const npue::json::Value names = need_json(f, nodes_key, label);

  const std::string nlabel = label + "/" + graph_key + ": node '" +
                             (names.as_array().empty()
                                  ? std::string("?")
                                  : names.as_array().front().as_string()) + "'";
  if (ops.as_array().size() != names.as_array().size())
    throw std::runtime_error(
        label + ": " + graph_key + " has " +
        std::to_string(ops.as_array().size()) + " nodes and " + nodes_key +
        " has " + std::to_string(names.as_array().size()) +
        ". The second list is what a refusal quotes by name, and a refusal "
        "that quotes node 7 when node 7 does not exist is worse than none.");

  int64_t max_conv = -1, max_prelu = -1;
  for (size_t n = 0; n < ops.as_array().size(); ++n) {
    const npue::json::Value &o = ops.as_array()[n];
    const std::string nl =
        nlabel + " [" + std::to_string(n) + "] '" + names.as_array()[n].as_string() +
        "'";
    Layer l;
    l.op = op_from_string(jstr(o, "op", nl), nl);
    // WHICH OPS CARRY AN ACTIVATION, AND WHY IT IS NOT OPTIONAL FOR THREE OF
    // THEM. conv, dwconv and add all have an epilogue to fuse into, so the key is
    // REQUIRED there: a packer that omitted it would be saying "and no
    // activation", and this runtime cannot tell that apart from a packer that
    // forgot. maxpool, resize and d2s have no epilogue -- they are not
    // convolutions and there is nothing to fuse into -- so the key is absent on
    // them and its absence is read as `none` rather than refused.
    if (l.op == Op::Conv || l.op == Op::DwConv || l.op == Op::Add)
      l.act = act_from_string(jstr(o, "act", nl), nl);
    else
      l.act = act_from_string(jopt_str(o, "act", "none"), nl);
    l.prelu = jopt_int(o, "prelu", -1);
    if (l.act == Act::PreLU) {
      if (l.prelu < 0)
        throw std::runtime_error(nl +
                                 ": the op records a prelu activation and no "
                                 "slope index");
      max_prelu = std::max<int64_t>(max_prelu, l.prelu);
    }
    if (l.act != Act::PreLU && l.prelu >= 0)
      throw std::runtime_error(nl + ": carries prelu slope " +
                               std::to_string(l.prelu) + " and the activation " +
                               act_name(l.act) +
                               ". Both are written by the same packer pass over the "
                               "same fused-activation node, so this is a "
                               "half-transcribed op rather than a network whose "
                               "activation happens to be unused.");
    if (l.op == Op::Conv || l.op == Op::DwConv) {
      l.conv = static_cast<int>(jint(o, "conv", nl));
      if (l.conv < 0)
        throw std::runtime_error(nl + ": no 'conv' index");
      if (l.conv >= num_convs)
        throw std::runtime_error(nl + ": conv " + std::to_string(l.conv) +
                                 " is past the container's " +
                                 std::to_string(num_convs) +
                                 " convolutions for this network");
      max_conv = std::max<int64_t>(max_conv, l.conv);
      jpad4(o, nl, l.pad);
      jpair(o, "stride", nl, &l.stride_h, &l.stride_w);
      // The design stream this convolution runs on, read from the graph the
      // packer wrote. Absent on a container packed WITHOUT --npu, which is the
      // default and is not an error here: the host path has no use for it, and
      // the array path refuses later with the reason that names the packing
      // command. Reading it and dropping it would have been the same code minus
      // the field.
      l.stream = jopt_str(o, "stream", "");
    } else if (l.op == Op::MaxPool) {
      jpair(o, "kernel", nl, &l.kernel_h, &l.kernel_w);
      jpair(o, "stride", nl, &l.stride_h, &l.stride_w);
      jpad4(o, nl, l.pad);
    } else if (l.op == Op::Resize) {
      const npue::json::Value *const s = o.find("scale");
      if (!s || s->as_array().size() != 2)
        throw std::runtime_error(nl + ": 'scale' is not a pair");
      l.scale_h = s->as_array()[0].as_number();
      l.scale_w = s->as_array()[1].as_number();
      if (l.scale_h <= 0.0 || l.scale_w <= 0.0)
        throw std::runtime_error(nl + ": resize has a non-positive scale");
    } else if (l.op == Op::D2S) {
      l.block = jint(o, "block", nl);
      if (l.block < 1)
        throw std::runtime_error(nl + ": d2s block is " +
                                 std::to_string(l.block) + ", and a blocksize "
                                 "below one is not a blocksize");
      // ONNX NAMES THE TWO ORDERS DCR AND CRD, and DCR is the default when the
      // attribute is absent -- which is this model's case: its DepthToSpace
      // nodes carry `blocksize` and nothing else. The default has to be the
      // schema's own, because taking the other one moves every channel to a
      // different pixel while still producing a tensor of the right shape and a
      // plausible range, and nothing downstream notices.
      const std::string mode = jopt_str(o, "mode", "dcr");
      if (mode != "dcr" && mode != "crd")
        throw std::runtime_error(
            nl + ": d2s mode is '" + mode +
            "'. The ONNX schema for DepthToSpace names exactly two -- DCR and "
            "CRD -- and the default when the attribute is absent is DCR. They "
            "differ only in how the input channel axis is split across the "
            "block, and the wrong one is a rearrangement that still fits.");
      l.crd = mode == "crd";
    }
    {
      const npue::json::Value &out = o.at("out");
      if (out.as_array().size() != 4 ||
          static_cast<int64_t>(out.as_array()[0].as_number()) != 1)
        throw std::runtime_error(nl + ": 'out' is not a [1, C, H, W] shape");
      l.out_c = static_cast<int64_t>(out.as_array()[1].as_number());
      l.out_h = static_cast<int64_t>(out.as_array()[2].as_number());
      l.out_w = static_cast<int64_t>(out.as_array()[3].as_number());
    }
    for (const auto &e : o.at("inputs").as_array())
      l.inputs.push_back(static_cast<int>(e.as_number()));
    if (l.inputs.empty())
      throw std::runtime_error(nl + ": no 'inputs'");
    for (int in : l.inputs) {
      if (in != -1 && static_cast<size_t>(in) >= n)
        throw std::runtime_error(
            nl + ": takes operand " + std::to_string(in) +
            ", which is not an earlier op and not the image (-1). This graph is "
            "evaluated in order, so a forward reference is not a cycle the "
            "runtime could resolve -- it is a graph whose outputs it cannot "
            "compute.");
    }
    g.layers.push_back(std::move(l));
    g.nodes.push_back(names.as_array()[n].as_string());
  }

  const size_t n_conv = static_cast<size_t>(std::max<int64_t>(num_convs, max_conv + 1));
  g.convs.resize(n_conv);
  g.w_storage.resize(n_conv);
  for (size_t i = 0; i < n_conv; ++i) {
    const std::string base = pfx + ".conv." + std::to_string(i);
    if (!f.has(base + ".w"))
      continue;   // a sparse index; conv_of() refuses an op that names one.
    bool dw = false;
    for (const Layer &l : g.layers)
      if (l.conv == static_cast<int>(i) &&
          (l.op == Op::Conv || l.op == Op::DwConv)) {
        dw = l.op == Op::DwConv;
        break;
      }
    g.convs[i] = read_conv(f, pfx, static_cast<int>(i), dw, wdtype, i4_group,
                           g.w_storage[i], label);
  }

  // The slope count is DERIVED from the op list rather than declared, because a
  // container that listed a count and a list that disagreed would be one more
  // thing to keep in step. Neither network here has a PReLU, so this loop is
  // normally empty -- and that is the point: a container that fused one and
  // shipped no slope is caught here rather than at the first negative pixel.
  g.slopes.resize(static_cast<size_t>(max_prelu + 1));
  for (int64_t i = 0; i <= max_prelu; ++i) {
    const std::string base = pfx + ".prelu." + std::to_string(i) + ".slope";
    if (!f.has(base))
      throw std::runtime_error(label + ": the graph fuses prelu " +
                               std::to_string(i) + " and " + base +
                               " is not in the container");
    int64_t n = 1;
    for (int64_t d : f.info(base).logical_shape) n *= d;
    g.slopes[static_cast<size_t>(i)].v = f.raw(base).as<float>();
    g.slopes[static_cast<size_t>(i)].n = n;
  }
  return g;
}

}  // namespace

const ConvW &Geometry::conv_of(const std::vector<ConvW> &tab, const Layer &l,
                               const std::string &label) const {
  if (l.conv < 0 || static_cast<size_t>(l.conv) >= tab.size() ||
      tab[static_cast<size_t>(l.conv)].w == nullptr)
    throw std::runtime_error(label + ": op '" + op_to_string(l.op) +
                             "' names conv " + std::to_string(l.conv) +
                             ", which the container has no weights for. Both "
                             "lists are written by the same packer pass, so a "
                             "disagreement here means the container was edited "
                             "or half-written -- and a convolution with no "
                             "weights is a graph that would run on whatever was "
                             "in the buffer.");
  return tab[static_cast<size_t>(l.conv)];
}

const PoseOut &Geometry::pose_out(const std::string &name,
                                  const std::string &label) const {
  for (const PoseOut &p : pose_head)
    if (p.name == name) return p;
  std::string have;
  for (const PoseOut &p : pose_head) have += (have.empty() ? "" : ", ") + p.name;
  throw std::runtime_error(label + ": the landmark head has no '" + name +
                           "'; it has {" + have +
                           "}. MediaPipe Pose declares five outputs and this "
                           "front end reads all five but the heatmap, so one of "
                           "them being absent is a different network rather than "
                           "a head with fewer parts.");
}

Geometry read_geometry(npue::File &f, const std::string &label) {
  const std::string arch = f.config_string("arch");
  if (arch != kArch)
    throw std::runtime_error(
        label + ": this is an arch=" + arch + " container and this build reads "
        "arch=" + kArch +
        ". The two networks here carry depthwise convolutions the array cannot "
        "express, a DepthToSpace the other architectures do not have, two "
        "different normalisations, and a detector that decodes xy1/xy2 rather "
        "than cx/cy/w/h.");
  Geometry g;
  g.det_input_size = need_int(f, "det_input_size", label);
  g.pose_input_size = need_int(f, "pose_input_size", label);
  g.num_landmarks = need_int(f, "num_landmarks", label);
  g.num_keypoints = need_int(f, "num_keypoints", label);
  g.lm_cols = need_int(f, "lm_cols", label);
  g.world_cols = need_int(f, "world_cols", label);
  if (g.num_landmarks != kNumLandmarks || g.num_keypoints != kNumKeypoints ||
      g.lm_cols != kLmCols || g.world_cols != kWorldCols)
    throw std::runtime_error(
        label + ": the container declares " + std::to_string(g.num_landmarks) +
        " landmarks of " + std::to_string(g.lm_cols) + " columns, " +
        std::to_string(g.num_keypoints) + " keypoints and " +
        std::to_string(g.world_cols) +
        " world columns; MediaPipe Pose emits 39 rows of 5 (33 keypoints plus "
        "six auxiliary, each x/y/z/visibility/presence) and 39 of 3. A landmark "
        "head of the wrong width produces a skeleton with a plausible number of "
        "joints and the wrong ones.");
  if (g.det_input_size != 224 || g.pose_input_size != 256)
    throw std::runtime_error(
        label + ": the two input sizes are " + std::to_string(g.det_input_size) +
        " and " + std::to_string(g.pose_input_size) +
        ". This architecture's detector is 224x224 and its landmark net 256x256 "
        "-- the second is larger because the region of interest is already a "
        "rotated square of the right shape and only has to be resampled.");

  // -- the two normalisations
  {
    const auto norm = [&](const char *mkey, const char *skey,
                          std::vector<float> &mean, std::vector<float> &sd,
                          const char *who) {
      // ONE call, held in a named local. `assign(f(key).begin(), f(key).end())`
      // makes two calls, and the two iterators then belong to two different
      // temporaries -- which reads the end pointer of an object that has already
      // been destroyed and asks a vector for a length of several hundred million
      // floats. It is written this way in one place and nowhere else.
      const std::vector<double> m = need_doubles(f, mkey, 3, label);
      const std::vector<double> s = need_doubles(f, skey, 3, label);
      mean.assign(m.begin(), m.end());
      sd.assign(s.begin(), s.end());
      for (double v : sd)
        if (v == 0.0)
          throw std::runtime_error(label + ": " + skey +
                                   " has a zero. Both front ends normalise by "
                                   "it, so that is a division by zero on every "
                                   "pixel of every frame. (" + who + ")");
    };
    norm("det_image_mean", "det_image_std", g.det_mean, g.det_std, "detector");
    norm("pose_image_mean", "pose_image_std", g.pose_mean, g.pose_std,
         "landmark net");
  }

  // -- the detector's constants
  g.det_num_anchors = need_int(f, "det_num_anchors", label);
  g.det_row_terms = need_int(f, "det_row_terms", label);
  g.det_landmarks = need_int(f, "det_landmarks", label);
  if (g.det_row_terms != kDetRowTerms || g.det_landmarks != kDetLandmarks)
    throw std::runtime_error(
        label + ": det_row_terms is " + std::to_string(g.det_row_terms) +
        " with " + std::to_string(g.det_landmarks) +
        " landmarks, expected 12 with 4 (xy1, xy2, then four keypoints x 2). "
        "This detector decodes CORNERS against an anchor centre; reading that as "
        "arch=7's cx,cy,w,h doubles every box's position error.");
  g.score_threshold = need_double(f, "score_threshold", label);
  g.nms_threshold = need_double(f, "nms_threshold", label);
  g.top_k = need_int(f, "top_k", label);

  // -- the landmark network's geometry
  g.person_lm_mid_hip = need_int(f, "person_lm_mid_hip", label);
  g.person_lm_full_body = need_int(f, "person_lm_full_body", label);
  if (g.person_lm_mid_hip < 0 || g.person_lm_mid_hip >= kDetLandmarks ||
      g.person_lm_full_body < 0 || g.person_lm_full_body >= kDetLandmarks)
    throw std::runtime_error(
        label + ": the rotation is taken between the detector's keypoints " +
        std::to_string(g.person_lm_mid_hip) + " and " +
        std::to_string(g.person_lm_full_body) + ", and the detector emits " +
        std::to_string(kDetLandmarks) +
        ". WHICH two choose the rotation decides where the landmark network "
        "looks; a different pair shows it a tilted person and it returns 39 "
        "rows that are all slightly wrong.");
  g.person_box_pre_enlarge = need_double(f, "person_box_pre_enlarge", label);
  g.person_box_enlarge = need_double(f, "person_box_enlarge", label);
  g.pose_conf_threshold = need_double(f, "pose_conf_threshold", label);

  // -- the detector's head, one entry per pyramid level
  {
    const npue::json::Value lv = need_json(f, "det_head", label);
    if (lv.as_array().empty())
      throw std::runtime_error(label + ": det_head lists no levels");
    for (size_t i = 0; i < lv.as_array().size(); ++i) {
      const std::string el = label + "/det_head[" + std::to_string(i) + "]";
      DetLevel L;
      L.box_op = static_cast<int>(jint(lv.as_array()[i], "box_op", el));
      L.score_op = static_cast<int>(jint(lv.as_array()[i], "score_op", el));
      L.h = jint(lv.as_array()[i], "h", el);
      L.w = jint(lv.as_array()[i], "w", el);
      L.per = jint(lv.as_array()[i], "per", el);
      L.rows = jint(lv.as_array()[i], "rows", el);
      L.terms = jint(lv.as_array()[i], "terms", el);
      if (L.rows != L.h * L.w * L.per)
        throw std::runtime_error(el + ": rows=" + std::to_string(L.rows) +
                                 " against h*w*per = " +
                                 std::to_string(L.h * L.w * L.per) +
                                 ". `rows` is the level's share of the anchor "
                                 "table and the decode reads it, so a "
                                 "disagreement here is a table read at the wrong "
                                 "stride.");
      if (L.terms != g.det_row_terms)
        throw std::runtime_error(el + ": terms=" + std::to_string(L.terms) +
                                 " against det_row_terms=" +
                                 std::to_string(g.det_row_terms));
      if (L.per < 1 || L.h < 1 || L.w < 1)
        throw std::runtime_error(el + ": a level with no cells or no anchors");
      g.det_levels.push_back(L);
    }
  }

  // -- the landmark head: five outputs, each from one graph node
  {
    const std::string hl = label + "/pose_head";
    const npue::json::Value h = need_json(f, "pose_head", label);
    for (const char *name : kPoseOutNames) {
      const npue::json::Value *const e = h.find(name);
      if (!e)
        throw std::runtime_error(hl + ": no '" + name +
                                 "'. MediaPipe Pose declares five outputs and "
                                 "the container must describe all five, so a "
                                 "head with fewer parts is a different network.");
      const std::string ol = hl + "/" + name;
      PoseOut p;
      p.name = name;
      p.op = static_cast<int>(jint(*e, "op", ol));
      p.transposed = jbool_or(*e, "transposed", false);
      p.sigmoid = jbool_or(*e, "sigmoid", false);
      // `decl` is ANY rank -- MediaPipe declares the landmarks as [1,195] and the
      // mask as [1,256,256,1], and a Reshape is what connects them. Only
      // `conv_out` is pinned to [1,C,H,W], because the node the walk leaves behind
      // is NCHW by construction and that is what the runtime holds.
      const npue::json::Value *const d = e->find("decl");
      if (!d || d->as_array().empty())
        throw std::runtime_error(ol + ": 'decl' is missing or empty");
      for (const auto &v : d->as_array())
        p.decl.push_back(static_cast<int64_t>(v.as_number()));
      const npue::json::Value *const c = e->find("conv_out");
      if (!c || c->as_array().size() != 4)
        throw std::runtime_error(ol + ": 'conv_out' is not a [1, C, H, W] shape");
      for (int k = 0; k < 4; ++k)
        p.conv_out[k] = static_cast<int64_t>(c->as_array()[k].as_number());
      // A Reshape and a Transpose each preserve the element count, so equality
      // of the two products is the invariant that covers all five without
      // assuming a layout -- and it is what catches a re-export that changed 39
      // rows into something else while every shape still parsed.
      int64_t dn = 1, cn = 1;
      for (int64_t v : p.decl) dn *= v;
      for (int k = 0; k < 4; ++k) cn *= p.conv_out[k];
      if (dn != cn)
        throw std::runtime_error(ol + ": declares " + std::to_string(dn) +
                                 " values and its convolution writes " +
                                 std::to_string(cn) +
                                 ". One of them is a transcription.");
      if (p.decl[0] != 1 || p.conv_out[0] != 1)
        throw std::runtime_error(ol + ": these graphs run one image at a time, "
                                 "so neither shape has a batch axis to carry "
                                 "anything but 1");
      g.pose_head.push_back(std::move(p));
    }
    if (g.pose_out("conf", hl).sigmoid == false)
      throw std::runtime_error(
          hl + ": the confidence carries no sigmoid. The zoo compares this "
          "number against confThreshold=" + std::to_string(g.pose_conf_threshold) +
          " and returns it, and both assume a probability rather than a logit; "
          "an unsquashed one would read about -9.7 and silently reject every "
          "person.");
  }
  {
    const npue::json::Value un = need_json(f, "pose_unused", label);
    for (const auto &e : un.as_array())
      g.pose_unused.push_back(e.as_string());
  }

  // -- the weight precision, and the int4 group
  //
  // BOTH KEYS ARE OPTIONAL, and their ABSENCE means f32 with no group. Four
  // containers were packed before either key existed -- mediapipe-hands.npue,
  // mediapipe-pose.npue and the two .f32 variants under models/variants -- and
  // they must keep reading exactly as they did, which is plain fp32. What is
  // NOT optional any more is the byte count against the declared precision: a
  // container whose payload contradicts its own dtype is refused below rather
  // than read as a plausible fp32 array of the wrong length.
  std::string wdtype = "f32";
  try {
    wdtype = f.config_string("conv_weight_dtype");
  } catch (const std::exception &) {
    wdtype = "f32";
  }
  if (wdtype != "f32" && wdtype != "bf16" && wdtype != "i8" && wdtype != "i4")
    throw std::runtime_error(
        label + ": conv_weight_dtype is '" + wdtype +
        "', and this runtime reads f32, bf16, i8 and i4. There is no fp16 or "
        "fp8 convolution weight here, and reading either as one of the four "
        "produces a network with the right shapes and no relationship to the "
        "checkpoint.");
  int64_t i4_group = 0;
  try {
    i4_group = f.config_int("conv_int4_group");
  } catch (const std::exception &) {
    // Absent along with conv_weight_dtype on the pre-key containers, where the
    // dtype is f32 and the group means nothing. Absent while the dtype says i4
    // is different: the group decides which scale each weight multiplies by,
    // and defaulting it would be picking a stride for the packer.
    if (wdtype == "i4")
      throw std::runtime_error(
          label + ": conv_weight_dtype is 'i4' and conv_int4_group is not in "
          "the container. The group size is what places each weight against "
          "its own group's scale, so a reader that had to default it would "
          "choose the stride of the dequantisation itself.");
    i4_group = 0;
  }
  if (i4_group < 0)
    throw std::runtime_error(
        label + ": conv_int4_group is " + std::to_string(i4_group) +
        ". Zero means one group over the whole input axis -- per-channel int4 "
        "-- and is the only value below 1 with a meaning.");
  if (i4_group > 0 && wdtype != "i4")
    throw std::runtime_error(
        label + ": conv_int4_group is " + std::to_string(i4_group) + " and "
        "conv_weight_dtype is '" + wdtype +
        "'. The group only means anything for four-bit weights, so a container "
        "that states both has a scale the reader cannot place.");

  // -- the two graphs and their weights
  {
    GraphRead det = read_one_graph(f, "det", "det_graph", "det_graph_nodes",
                                   need_int(f, "det_num_convs", label),
                                   wdtype, i4_group, label + "/det");
    GraphRead pose = read_one_graph(f, "pose", "pose_graph", "pose_graph_nodes",
                                    need_int(f, "pose_num_convs", label),
                                    wdtype, i4_group, label + "/pose");
    g.det_graph = std::move(det.layers);
    g.det_nodes = std::move(det.nodes);
    g.det_convs = std::move(det.convs);
    g.det_slopes = std::move(det.slopes);
    g.pose_graph = std::move(pose.layers);
    g.pose_nodes = std::move(pose.nodes);
    g.pose_convs = std::move(pose.convs);
    g.pose_slopes = std::move(pose.slopes);
    // The storage moves too, and the ConvW::w pointers into it stay valid:
    // moving a std::vector transfers its buffer without touching it, so the
    // detector's entries land first and the landmark net's after them, in each
    // conv's original order within its own graph.
    g.w_storage.reserve(det.w_storage.size() + pose.w_storage.size());
    for (std::vector<float> &v : det.w_storage)
      g.w_storage.push_back(std::move(v));
    for (std::vector<float> &v : pose.w_storage)
      g.w_storage.push_back(std::move(v));
  }

  // -- the anchor table, STORED, and cross-checked against the levels
  //
  // The table is read rather than generated, because the packer generated it and
  // checked it against the OpenCV zoo's literal 2254 x 2 table bit for bit; a
  // table that has been verified once should be shipped rather than recomputed
  // on every reader. det_anchor_levels is still read and still compared against
  // the head's own levels: two hand-independent spellings of one pyramid, so a
  // disagreement is a refusal rather than a misaligned table.
  {
    const std::string an = "det_anchors";
    if (!f.has(an))
      throw std::runtime_error(label + ": " + an +
                               " is not in the container. The 2254 SSD anchor "
                               "centres are STORED, not regenerated here: the "
                               "packer derives them from the graph's three "
                               "DepthToSpace levels and reproduces the zoo's "
                               "literal table bit for bit, and re-deriving them "
                               "on a second machine would make that a fact about "
                               "every machine's arithmetic rather than about "
                               "this one.");
    const auto &ai = f.info(an);
    if (ai.logical_shape.size() != 2 ||
        static_cast<int64_t>(ai.logical_shape[0]) != g.det_num_anchors ||
        ai.logical_shape[1] != 2)
      throw std::runtime_error(
          label + ": " + an + " is " + std::to_string(ai.logical_shape[0]) + "x" +
          std::to_string(ai.logical_shape.size() > 1 ? ai.logical_shape[1] : 0) +
          " and the container declares " + std::to_string(g.det_num_anchors) +
          " anchors of 2.");
    const float *const ap = f.raw(an).as<float>();
    g.anchors.assign(ap, ap + 2 * g.det_num_anchors);

    const npue::json::Value lv = need_json(f, "det_anchor_levels", label);
    if (static_cast<int64_t>(lv.as_array().size()) !=
        static_cast<int64_t>(g.det_levels.size()))
      throw std::runtime_error(label + ": det_anchor_levels lists " +
                               std::to_string(lv.as_array().size()) +
                               " levels and the head has " +
                               std::to_string(g.det_levels.size()) +
                               ". These are two spellings of one pyramid and they "
                               "are compared here rather than derived, so a "
                               "disagreement is a refusal and not a 2254-row "
                               "table read at the wrong stride.");
    for (size_t i = 0; i < g.det_levels.size(); ++i) {
      const auto &e = lv.as_array()[i];
      if (e.as_array().size() != 3)
        throw std::runtime_error(label + ": det_anchor_levels[" +
                                 std::to_string(i) +
                                 "] is not [h, w, anchors_per_cell]");
      const int64_t h = static_cast<int64_t>(e.as_array()[0].as_number());
      const int64_t w = static_cast<int64_t>(e.as_array()[1].as_number());
      const int64_t a = static_cast<int64_t>(e.as_array()[2].as_number());
      const DetLevel &L = g.det_levels[i];
      if (h != L.h || w != L.w || a != L.per)
        throw std::runtime_error(
            label + ": det_anchor_levels[" + std::to_string(i) + "] is " +
            std::to_string(h) + "x" + std::to_string(w) + " with " +
            std::to_string(a) + " anchors; the head's level is " +
            std::to_string(L.h) + "x" + std::to_string(L.w) + " with " +
            std::to_string(L.per));
    }
  }

  return g;
}

}  // namespace npue::mppose
