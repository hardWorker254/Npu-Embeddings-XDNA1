//===- geometry.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read the arch=8 container's two graphs, their weights and
// both heads. See geometry.hpp for what is in one of these and why.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/geometry.hpp"

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

ConvW read_conv(npue::File &f, const std::string &pfx, int i, bool want_depthwise,
                const std::string &label) {
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
  c.w = f.raw(base + ".w").as<float>();
  if (f.has(base + ".b"))
    c.b = f.raw(base + ".b").as<float>();
  return c;
}

struct GraphRead {
  std::vector<Layer> layers;
  std::vector<ConvW> convs;
  std::vector<std::string> nodes;
  std::vector<Slope> slopes;
};

// Read one network. `pfx` namespaces its tensors, and it is not decoration: both
// networks number their convolutions from zero, so an unprefixed lookup reads the
// OTHER network's weights and returns a tensor of the wrong width -- which for the
// detector means a 224-input graph running on the landmark net's first layer.
GraphRead read_one_graph(npue::File &f, const std::string &pfx,
                         const std::string &graph_key,
                         const std::string &nodes_key, int64_t num_convs,
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
    g.convs[i] = read_conv(f, pfx, static_cast<int>(i), dw, label);
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

  // -- the two graphs and their weights
  {
    GraphRead det = read_one_graph(f, "det", "det_graph", "det_graph_nodes",
                                   need_int(f, "det_num_convs", label),
                                   label + "/det");
    GraphRead pose = read_one_graph(f, "pose", "pose_graph", "pose_graph_nodes",
                                    need_int(f, "pose_num_convs", label),
                                    label + "/pose");
    g.det_graph = std::move(det.layers);
    g.det_nodes = std::move(det.nodes);
    g.det_convs = std::move(det.convs);
    g.det_slopes = std::move(det.slopes);
    g.pose_graph = std::move(pose.layers);
    g.pose_nodes = std::move(pose.nodes);
    g.pose_convs = std::move(pose.convs);
    g.pose_slopes = std::move(pose.slopes);
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
