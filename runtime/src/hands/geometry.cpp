//===- geometry.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read the arch=7 container's two graphs, their weights and
// their heads. See geometry.hpp for what is in one of these and why.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "hands/geometry.hpp"

#include <algorithm>
#include <sstream>

namespace npue::hands {

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
  // null by every op whose activation is not a PReLU, and reading that as 0
  // would make all twenty-six of arch=7's PReLUs into the first one's slopes.
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

void jpair(const npue::json::Value &o, const std::string &key,
           const std::string &label, int64_t *a, int64_t *b) {
  const npue::json::Value *const v = o.find(key);
  if (!v || v->as_array().size() != 2)
    throw std::runtime_error(label + ": '" + key +
                             "' is not a pair. Every stride, kernel and scale "
                             "in these graphs is per axis and both axes have "
                             "to be stated: these two networks are square so "
                             "far, but 'square so far' is not something to "
                             "assume about a packed graph.");
  *a = static_cast<int64_t>((*v).as_array()[0].as_number());
  *b = static_cast<int64_t>((*v).as_array()[1].as_number());
}

void jpad4(const npue::json::Value &o, const std::string &label, int64_t *pad) {
  const npue::json::Value *const v = o.find("pad");
  if (!v || v->as_array().size() != 4)
    throw std::runtime_error(
        label + ": 'pad' is not four terms [top, left, bottom, right]. It is "
        "four and not two because arch=7 pads ASYMMETRICALLY: MediaPipe "
        "evaluates SAME padding at a stride, so the palm detector's 3x3 "
        "stride-2 stem is [1,1,2,2]. Transcribing that as a symmetric pad "
        "shifts the whole pyramid by one pixel, which is invisible in the "
        "output's shape and moves every detection by a pixel of its own size.");
  for (int i = 0; i < 4; ++i)
    pad[i] = static_cast<int64_t>((*v).as_array()[i].as_number());
}

// A convolution's weights, plus the two refusals that matter. Neither is a
// shape error at run time:
//
//   * a "dwconv" whose filter has more than one input channel is not a depthwise
//     convolution, and running it as a dense one would change the network
//     rather than fail;
//   * a "conv" carrying group > 1 IS a depthwise convolution under another
//     name, and the array backend counts the two separately, so accepting it
//     would let a container disagree with itself about how many of its
//     convolutions the array can take.
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
          "dense GEMM over this one would compute a different network, and "
          "the difference would be this architecture's accuracy rather than a "
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

// Read one network. `pfx` namespaces its tensors, and it is not decoration:
// both networks number their convolutions from zero and their PReLUs from zero,
// so an unprefixed lookup reads the OTHER network's weights and returns a
// tensor of the wrong width -- which for the palm detector means a 192-input
// graph running on the landmark net's first layer.
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
    // THEM. conv, dwconv and add all have an epilogue to fuse into, so the key
    // is REQUIRED there: a packer that omitted it would be saying "and no
    // activation", and this runtime cannot tell that apart from a packer that
    // forgot. maxpool, pad_c and resize have no epilogue -- they are not
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
                               ". Both are written by the same packer pass over "
                               "the same fused-activation node, so this is a "
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
      // The design stream this convolution runs on, read from the graph the packer
      // wrote. ABSENT on a container packed without --npu, which is the default
      // and is not an error here: the host path has no use for it and the array
      // path refuses later with the reason that names the packing command.
      // Reading it and dropping it would have been the same code minus the field,
      // and the array path then refused a container that had the answer in it.
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
    } else if (l.op == Op::PadC) {
      l.count = jint(o, "count", nl);
      if (l.count < 0)
        throw std::runtime_error(nl + ": pad_c count is negative");
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
            ", which is not an earlier op and not the image (-1). This graph "
            "is evaluated in order, so a forward reference is not a cycle the "
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
      continue;   // a sparse index. Recorded as absent rather than refused: the
                  // container numbered its convolutions from the ONNX's own
                  // order and a gap in it is a fact about the export, and the
                  // op list says which indices are real. conv_of() refuses an
                  // op that names one of the gaps.
    bool dw = false;
    for (const Layer &l : g.layers)
      if (l.conv == static_cast<int>(i) &&
          (l.op == Op::Conv || l.op == Op::DwConv)) {
        dw = l.op == Op::DwConv;
        break;
      }
    g.convs[i] = read_conv(f, pfx, static_cast<int>(i), dw, label);
  }

  // The slope count is DERIVED from the op list rather than declared in the
  // config, because a container that listed a count and a list that disagreed
  // would be one more thing to keep in step, and the op list is the thing that
  // has to be right anyway: an activation naming a slope index is the only
  // consumer of these vectors.
  g.slopes.resize(static_cast<size_t>(max_prelu + 1));
  for (int64_t i = 0; i <= max_prelu; ++i) {
    const std::string base = pfx + ".prelu." + std::to_string(i) + ".slope";
    if (!f.has(base))
      throw std::runtime_error(label + ": the graph fuses prelu " +
                               std::to_string(i) + " and " + base +
                               " is not in the container");
    // The slope is stored with the checkpoint's own [1, C, 1, 1] shape rather
    // than flattened to C, because that is what the ONNX initialiser is and
    // stripping the shape at pack time would have moved the decision to every
    // reader. The LENGTH is the product, not shape[0]: reading shape[0] of a
    // 4-D slope vector gives 1, which would look like a PReLU with one slope
    // for a 32-channel tensor -- and would be refused below, correctly, but for
    // the wrong reason.
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
                             "disagreement here means the container was "
                             "edited or half-written -- and a convolution with "
                             "no weights is a graph that would run on whatever "
                             "was in the buffer.");
  return tab[static_cast<size_t>(l.conv)];
}

Geometry read_geometry(npue::File &f, const std::string &label) {
  const std::string arch = f.config_string("arch");
  if (arch != kArch)
    throw std::runtime_error(
        label + ": this is an arch=" + arch + " container and this build reads "
        "arch=" + kArch +
        ". The two hand networks are not a pose model with its head turned "
        "off: they carry depthwise convolutions the array cannot express, an "
        "asymmetric padding convention, two different heads, and no "
        "normalisation at all.");
  Geometry g;
  g.palm_input_size = need_int(f, "palm_input_size", label);
  g.lm_input_size = need_int(f, "lm_input_size", label);
  g.num_landmarks = need_int(f, "num_landmarks", label);
  if (g.num_landmarks != kHandLandmarks)
    throw std::runtime_error(label + ": num_landmarks is " +
                             std::to_string(g.num_landmarks) +
                             ", this architecture emits " +
                             std::to_string(kHandLandmarks) +
                             " (MediaPipe's twenty-one: a wrist, four chains "
                             "of four, and four across the palm)");
  g.palm_num_anchors = need_int(f, "palm_num_anchors", label);
  g.palm_row_terms = need_int(f, "palm_row_terms", label);
  if (g.palm_row_terms != kPalmRowTerms)
    throw std::runtime_error(label + ": palm_row_terms is " +
                             std::to_string(g.palm_row_terms) + ", expected " +
                             std::to_string(kPalmRowTerms) +
                             " (4 box terms + 7 keypoints x 2)");
  g.palm_lm_wrist = need_int(f, "palm_lm_wrist", label);
  g.palm_lm_middle_base = need_int(f, "palm_lm_middle_base", label);
  if (g.palm_lm_wrist < 0 || g.palm_lm_wrist >= kPalmLandmarks ||
      g.palm_lm_middle_base < 0 || g.palm_lm_middle_base >= kPalmLandmarks)
    throw std::runtime_error(
        label + ": the rotation is taken between palm keypoints " +
        std::to_string(g.palm_lm_wrist) + " and " +
        std::to_string(g.palm_lm_middle_base) + ", and the palm detector emits " +
        std::to_string(kPalmLandmarks) +
        ". WHICH two choose the rotation decides where the landmark network "
        "looks; a different pair gives a plausible hand pointing sideways.");

  const auto crop = [&](const char *skey, const char *ekey, Crop &c) {
    const std::vector<double> v = need_doubles(f, skey, 2, label);
    c.shift[0] = v[0];
    c.shift[1] = v[1];
    c.enlarge = need_double(f, ekey, label);
  };
  crop("palm_pre_shift", "palm_pre_enlarge", g.palm_pre);
  crop("palm_shift", "palm_enlarge", g.palm_post);
  crop("hand_shift", "hand_enlarge", g.hand);

  g.score_threshold = need_double(f, "score_threshold", label);
  g.nms_threshold = need_double(f, "nms_threshold", label);
  g.top_k = need_int(f, "top_k", label);

  {
    const std::vector<double> m = need_doubles(f, "image_mean", 3, label);
    const std::vector<double> s = need_doubles(f, "image_std", 3, label);
    g.mean.assign(m.begin(), m.end());
    g.std_dev.assign(s.begin(), s.end());
    for (double v : g.std_dev)
      if (v == 0.0)
        throw std::runtime_error(label +
                                 ": image_std has a zero. Both networks "
                                 "normalise by it, so that is a division by "
                                 "zero on every pixel of every frame.");
  }

  // -- the palm head, one entry per pyramid level
  {
    const npue::json::Value lv = need_json(f, "palm_head", label);
    if (lv.as_array().empty())
      throw std::runtime_error(label + ": palm_head lists no levels");
    for (size_t i = 0; i < lv.as_array().size(); ++i) {
      const std::string el = label + "/palm_head[" + std::to_string(i) + "]";
      PalmLevel L;
      L.box = static_cast<int>(jint(lv.as_array()[i], "box", el));
      L.score = static_cast<int>(jint(lv.as_array()[i], "score", el));
      L.h = jint(lv.as_array()[i], "h", el);
      L.w = jint(lv.as_array()[i], "w", el);
      L.anchors_per_cell = jint(lv.as_array()[i], "anchors_per_cell", el);
      L.rows = jint(lv.as_array()[i], "rows", el);
      if (L.rows != L.h * L.w * L.anchors_per_cell)
        throw std::runtime_error(el + ": rows=" + std::to_string(L.rows) +
                                 " against h*w*a = " +
                                 std::to_string(L.h * L.w * L.anchors_per_cell) +
                                 ". `rows` is the level's share of the anchor "
                                 "table and the decode reads it, so a "
                                 "disagreement here is a table read at the "
                                 "wrong stride.");
      g.palm_levels.push_back(L);
    }
  }

  // -- the landmark head: one pool, then four Gemms
  {
    const std::string hl = label + "/lm_head";
    const npue::json::Value h = need_json(f, "lm_head", label);
    g.lm_head.pool = static_cast<int>(jint(h, "pool", hl));
    g.lm_head.features = jint(h, "features", hl);
    g.lm_head.height = jint(h, "height", hl);
    g.lm_head.width = jint(h, "width", hl);
    for (const auto &e : h.at("projs").as_array()) {
      LmProj p;
      p.w = jstr(e, "w", hl);
      p.b = jstr(e, "b", hl);
      p.n = jint(e, "n", hl);
      const auto &wi = f.info(p.w);
      if (wi.logical_shape.size() != 2)
        throw std::runtime_error(hl + ": " + p.w + " is not a 2-D projection");
      // Stored [in, out] and ONNX's Gemm with transB=0 is A @ W, so the
      // projection is W.T @ pooled. The other way round is a shape error here
      // and a silent TRANSPOSE in a runtime, and 63 against 1 is far enough
      // apart to be noticed -- which is the only reason that makes it a crash
      // rather than a wrong hand.
      if (wi.logical_shape[1] != p.n)
        throw std::runtime_error(hl + ": " + p.w + " has " +
                                 std::to_string(wi.logical_shape[1]) +
                                 " outputs, the config says " +
                                 std::to_string(p.n));
      if (wi.logical_shape[0] != g.lm_head.features)
        throw std::runtime_error(hl + ": " + p.w + " takes " +
                                 std::to_string(wi.logical_shape[0]) +
                                 " inputs, the pooled map is " +
                                 std::to_string(g.lm_head.features) + " wide");
      p.wp = f.raw(p.w).as<float>();
      if (f.has(p.b))
        p.bp = f.raw(p.b).as<float>();
      g.lm_head.projs.push_back(std::move(p));
    }
    for (const auto &e : h.at("outputs").as_array()) {
      const std::string ol = hl + ": one output";
      LmOutSpec o;
      o.n = jint(e, "n", ol);
      const npue::json::Value *const sg = e.find("sigmoid");
      o.sigmoid = sg && sg->as_bool();
      o.proj = static_cast<int>(jint(e, "proj", ol));
      if (o.proj < 0 || static_cast<size_t>(o.proj) >= g.lm_head.projs.size())
        throw std::runtime_error(ol + ": names projection " +
                                 std::to_string(o.proj) + " of " +
                                 std::to_string(g.lm_head.projs.size()));
      if (o.n != g.lm_head.projs[static_cast<size_t>(o.proj)].n)
        throw std::runtime_error(
            ol + ": n=" + std::to_string(o.n) + " against projection " +
            std::to_string(o.proj) + "'s " +
            std::to_string(g.lm_head.projs[static_cast<size_t>(o.proj)].n) +
            ". The two are read from one JSON and disagreeing means one of "
            "them is a transcription -- and the failure would be a hand with "
            "a plausible number of joints and the wrong ones.");
      g.lm_head.outputs.push_back(o);
    }
  }

  // -- the two graphs and their weights
  {
    GraphRead palm = read_one_graph(f, "palm", "palm_graph", "palm_graph_nodes",
                                    need_int(f, "palm_num_convs", label),
                                    label + "/palm");
    GraphRead lm = read_one_graph(f, "lm", "lm_graph", "lm_graph_nodes",
                                  need_int(f, "lm_num_convs", label),
                                  label + "/lm");
    g.palm_graph = std::move(palm.layers);
    g.palm_nodes = std::move(palm.nodes);
    g.palm_convs = std::move(palm.convs);
    g.palm_slopes = std::move(palm.slopes);
    g.lm_graph = std::move(lm.layers);
    g.lm_nodes = std::move(lm.nodes);
    g.lm_convs = std::move(lm.convs);
    g.lm_slopes = std::move(lm.slopes);
  }

  // -- the anchor table, GENERATED, in the graph's own row order
  //
  // The order is h outer and w inner, so the flat row index advances w -- the X
  // coordinate -- fastest, which is why the first component of a row is x and
  // not y. The head reshapes its [H, W, a*18] map to [H*W*a, 18] that way; a
  // C-major flatten instead would permute every row while leaving 2016
  // well-shaped rows and a confident detector behind.
  //
  // Cross-checked against palm_anchor_levels, which is a SECOND spelling of the
  // same pyramid. Neither is derived from the other at run time and they are
  // compared here: two hand-written spellings of one list catch each other's
  // typos, and what they cannot catch is a formula wrong in both. That limit is
  // why tools/verify/verify_hands.py carries a dedicated anchor-order step.
  {
    const npue::json::Value lv = need_json(f, "palm_anchor_levels", label);
    if (static_cast<int64_t>(lv.as_array().size()) !=
        static_cast<int64_t>(g.palm_levels.size()))
      throw std::runtime_error(label + ": palm_anchor_levels lists " +
                               std::to_string(lv.as_array().size()) +
                               " levels and the head has " +
                               std::to_string(g.palm_levels.size()) +
                               ". These are two spellings of one pyramid and "
                               "they are compared here rather than derived, so "
                               "a disagreement is a refusal and not a 2016-row "
                               "table of the wrong shape.");
    for (size_t i = 0; i < g.palm_levels.size(); ++i) {
      const auto &e = lv.as_array()[i];
      if (e.as_array().size() != 3)
        throw std::runtime_error(label + ": palm_anchor_levels[" +
                                 std::to_string(i) +
                                 "] is not [h, w, anchors_per_cell]");
      const int64_t h = static_cast<int64_t>(e.as_array()[0].as_number());
      const int64_t w = static_cast<int64_t>(e.as_array()[1].as_number());
      const int64_t a = static_cast<int64_t>(e.as_array()[2].as_number());
      const PalmLevel &L = g.palm_levels[i];
      if (h != L.h || w != L.w || a != L.anchors_per_cell)
        throw std::runtime_error(
            label + ": palm_anchor_levels[" + std::to_string(i) + "] is " +
            std::to_string(h) + "x" + std::to_string(w) + " with " +
            std::to_string(a) + " anchors; the head's level is " +
            std::to_string(L.h) + "x" + std::to_string(L.w) + " with " +
            std::to_string(L.anchors_per_cell));
      for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x)
          for (int64_t k = 0; k < a; ++k) {
            g.anchors.push_back((static_cast<double>(x) + 0.5) / w);
            g.anchors.push_back((static_cast<double>(y) + 0.5) / h);
          }
    }
    if (static_cast<int64_t>(g.anchors.size()) / 2 != g.palm_num_anchors)
      throw std::runtime_error(label + ": the levels generate " +
                               std::to_string(g.anchors.size() / 2) +
                               " anchors and the container declares " +
                               std::to_string(g.palm_num_anchors));
  }

  return g;
}

}  // namespace npue::hands