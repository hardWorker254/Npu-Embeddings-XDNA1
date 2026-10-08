//===- geometry.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 geometry reader. See geometry.hpp for why the graph
// is a list and why each refusal exists.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "pose/geometry.hpp"

#include "common/host_kernels.hpp"   // bf16_read

#include <algorithm>
#include <cmath>
#include <string>

namespace npue::pose {

namespace {

// The packed graph's shape at one node, so the arity and channel checks below
// have something to compare against. Kept as three numbers because that is all
// any op here can disagree about: a channel count, and a 2-D spatial extent.
struct Tensor {
  int64_t c = 0, h = 0, w = 0;
};

std::vector<Layer> read_graph(const npue::File &f, int64_t n_convs,
                              const std::string &label) {
  const std::string raw = f.config_string("graph");
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": graph is not readable JSON (" +
                             std::to_string(raw.size()) + " bytes)");
  }
  const auto &arr = v.as_array();
  std::vector<Layer> g;
  g.reserve(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    // The Value, not the object it holds: json::Value::find() returns a
    // pointer for an absent key while std::unordered_map::find() returns an
    // iterator, and every optional field below is checked against nullptr.
    const auto &node = arr[i];
    const std::string where =
        label + ": graph node " + std::to_string(i);
    Layer L;
    L.op = op_from_string(node.at("op").as_string(), where);

    auto ints = [&](const char *key) {
      std::vector<int> out;
      const auto *it = node.find(key);
      if (!it) return out;
      for (const auto &e : it->as_array())
        out.push_back(static_cast<int>(e.as_number()));
      return out;
    };

    // EVERY op reads "inputs", before any per-op branch. It used to be read
    // inside each branch, and four of the seven branches did not read it -- so
    // conv, maxpool, upsample and detect reached the typecheck with an EMPTY
    // operand list. `inputs[0]` on an empty vector is an out_of_range, not a
    // diagnosis, so the container that caused it printed a C++ library message
    // and named nothing.
    L.inputs = ints("inputs");
    if (L.op == Op::Conv) {
      const auto *ci = node.find("conv");
      if (!ci)
        throw std::runtime_error(where + ": a conv node carries no \"conv\" index");
      L.conv = static_cast<int>(ci->as_number());
      const auto *si = node.find("silu");
      L.silu = si && si->as_number() != 0;
      const auto *st = node.find("stream");
      if (st) L.stream = st->as_string();
      // The stride and the padding are READ, not assumed. See Layer's note: the
      // stem is stride 2 and the 1x1s pad nothing, so a default would be right
      // for most of the network and would silently resize the two layers that
      // set every resolution below them.
      const auto *stv = node.find("stride");
      if (!stv)
        throw std::runtime_error(
            where + ": conv carries no \"stride\". A default of 1 is right for "
            "most of this network and wrong for the stem, which downsamples by "
            "2; and a container that omits it cannot be checked against the "
            "kernel it was packed with.");
      L.stride = static_cast<int64_t>(stv->as_number());
      if (L.stride < 1)
        throw std::runtime_error(
            where + ": conv stride " + std::to_string(L.stride) +
            " is not a forward step. There is no sub-sampling form of this op; "
            "a pooling or a strided slice is what halves a pyramid level.");
      const auto *pv = node.find("pad");
      if (!pv)
        throw std::runtime_error(
            where + ": conv carries no \"pad\". The padding is zero on a 1x1 "
            "and one on a 3x3, and which is which is the difference between a "
            "642x642 first layer and a 640x640 one.");
      const auto &pa = pv->as_array();
      if (pa.size() != 2)
        throw std::runtime_error(
            where + ": conv pad has " + std::to_string(pa.size()) +
            " entries, not 2. It is the rows and the columns added on each "
            "side; the window is [Cout, Cin, kh, kw] and a single value would "
            "have to assume a square kernel.");
      L.pad_h = static_cast<int64_t>(pa[0].as_number());
      L.pad_w = static_cast<int64_t>(pa[1].as_number());
      if (L.pad_h < 0 || L.pad_w < 0)
        throw std::runtime_error(
            where + ": conv pad is [" + std::to_string(L.pad_h) + ", " +
            std::to_string(L.pad_w) +
            "]. Negative padding CROPS, and a convolution whose window is "
            "smaller than its stride crops asymmetrically; this runtime pads "
            "with zeros and never crops.");
      if (L.conv < 0 || L.conv >= n_convs)
        throw std::runtime_error(
            where + ": conv index " + std::to_string(L.conv) + " is outside the " +
            std::to_string(n_convs) + " convolutions this container holds");
      if (L.inputs.size() != 1)
        throw std::runtime_error(
            where + ": conv takes 1 input and this node lists " +
            std::to_string(L.inputs.size()) +
            ". A convolution reads one activation tensor plus its weights, which "
            "are named by the conv index rather than listed here.");
    } else if (L.op == Op::Slice) {
      if (L.inputs.size() != 1)
        throw std::runtime_error(
            where + ": slice takes 1 input and this node lists " +
            std::to_string(L.inputs.size()) +
            ". A slice reads one tensor; anything else would be a gather.");
      const auto *ci = node.find("chan");
      const auto *ni = node.find("nch");
      if (!ci || !ni)
        throw std::runtime_error(
            where + ": slice needs \"chan\" and \"nch\". The C2f block's "
            "channel split is expressed as one slice per half precisely so the "
            "graph's operand references stay single indices; without the extent "
            "a slice has no output shape.");
      L.chan = static_cast<int64_t>(ci->as_number());
      L.nch = static_cast<int64_t>(ni->as_number());
      if (L.nch <= 0 || L.chan < 0)
        throw std::runtime_error(
            where + ": slice chan=" + std::to_string(L.chan) +
            " nch=" + std::to_string(L.nch) +
            " -- a non-positive extent has no channels to take.");
    } else if (L.op == Op::Concat || L.op == Op::Add) {
      // Concat is variadic because SPPF's join has FOUR operands and the neck's
      // has two; the minimum of two is still enforced, because a one-operand
      // join is a copy and would pass every other check here.
      const bool concat = L.op == Op::Concat;
      if ((concat && L.inputs.size() < 2) || (!concat && L.inputs.size() != 2))
        throw std::runtime_error(
            where + ": " + op_to_string(L.op) + " takes " +
            (concat ? "at least 2 inputs" : "2 inputs") +
            " and this node lists " + std::to_string(L.inputs.size()) +
            (concat
                 ? ". The join is variadic -- SPPF joins four -- but never one, "
                   "since a single operand is a copy."
                 : "."));
    } else if (L.op == Op::MaxPool) {
      if (L.inputs.size() != 1)
        throw std::runtime_error(
            where + ": maxpool takes 1 input and this node lists " +
            std::to_string(L.inputs.size()) +
            ". A pooling reads one tensor.");
      const auto *ki = node.find("k");
      const auto *si2 = node.find("stride");
      if (!ki || !si2)
        throw std::runtime_error(where + ": maxpool needs \"k\" and \"stride\"");
      L.k = static_cast<int64_t>(ki->as_number());
      L.stride = static_cast<int64_t>(si2->as_number());
      if (L.k <= 0 || L.stride <= 0)
        throw std::runtime_error(where + ": maxpool k=" + std::to_string(L.k) +
                                 " stride=" + std::to_string(L.stride) +
                                 " -- a non-positive extent has no window");
      // Padding is NOT a field. SPPF is maxpool(k=5, stride=1) with padding 2 on
      // both sides, and that padding is what keeps the output the same size as
      // the input. A pooler that cropped instead would silently change every
      // downstream grid, so the shape is pinned here rather than inferred.
      if (2 * L.k - 1 < L.k)
        throw std::runtime_error(where + ": maxpool kernel does not tile");
    } else if (L.op == Op::Upsample) {
      if (L.inputs.size() != 1)
        throw std::runtime_error(
            where + ": upsample takes 1 input and this node lists " +
            std::to_string(L.inputs.size()) +
            ". A resampling reads one tensor.");
      const auto *si2 = node.find("scale");
      if (!si2)
        throw std::runtime_error(where + ": upsample needs \"scale\"");
      L.scale = static_cast<int64_t>(si2->as_number());
      if (L.scale <= 1)
        throw std::runtime_error(
            where + ": upsample scale " + std::to_string(L.scale) +
            " is not an enlargement. The neck's up path is nearest-neighbour by "
            "2; anything else is a different resample and is not inferred.");
    } else if (L.op == Op::Detect) {
      // TWO PER LEVEL, paired boxes-then-keypoints, so the count is even and at
      // least two. The exact number needs num_levels, which this function is not
      // given; read_geometry checks it, where the grid is in hand.
      if (L.inputs.size() < 2 || (L.inputs.size() % 2) != 0)
        throw std::runtime_error(
            where + ": detect takes two inputs per pyramid level (the level's "
            "box+class tensor and its keypoint tensor) and this node lists " +
            std::to_string(L.inputs.size()) +
            ". An odd count is the signature of a head whose tensors were paired "
            "by order rather than by resolution, and reading them in that order "
            "puts every joint on the neighbouring cell's pixel.");
    }
    g.push_back(std::move(L));
  }
  return g;
}

}  // namespace

Geometry read_geometry(npue::File &f, const std::string &label) {
  const std::string arch = f.config_string("arch");
  if (arch != kArch)
    throw std::runtime_error(label + ": arch is '" + arch + "', not '" +
                             std::string(kArch) + "'");

  Geometry g;
  g.input_size = need_int(f, "input_size", label);
  g.num_keypoints = need_int(f, "num_keypoints", label);
  g.num_classes = need_int(f, "num_classes", label);
  g.dfl_bins = need_int(f, "dfl_bins", label);
  g.detect_cout = need_int(f, "detect_cout", label);
  g.detect_h = need_int(f, "detect_h", label);
  g.detect_w = need_int(f, "detect_w", label);

  const int64_t n_levels = need_int(f, "num_levels", label);
  g.strides = need_ints(f, "strides", n_levels, label);
  g.grid = need_ints(f, "grid", n_levels, label);
  g.mean = need_floats(f, "image_mean", 3, label);
  g.std_dev = need_floats(f, "image_std", 3, label);

  // The head convention, required rather than defaulted. See BoxFormat's note:
  // every value here changes the answer, so a container that does not state it
  // is not one this runtime may guess about.
  g.head.box = box_format_from_string(f.config_string("head_box_format"), label);
  g.head.score = score_kind_from_string(f.config_string("head_score"), label);
  g.head.kpt_visibility =
      score_kind_from_string(f.config_string("head_keypoint_visibility"), label);

  // -- the refusals. Each of these is something a default would turn into a
  // confidently wrong pose rather than an error.
  if (g.input_size <= 0)
    throw std::runtime_error(label + ": input_size " +
                             std::to_string(g.input_size));
  if (g.num_keypoints != kNumKeypoints)
    throw std::runtime_error(
        label + ": num_keypoints " + std::to_string(g.num_keypoints) +
        " is not this build's " + std::to_string(kNumKeypoints) +
        ". The COCO skeleton is a NAMED graph: index 5 is a left eye and index 6 "
        "a right one, and a client that reads index 5 as 'nose' because this "
        "build says 16 keypoints would draw a face that is not there.");
  if (g.num_classes != 1)
    throw std::runtime_error(
        label + ": num_classes " + std::to_string(g.num_classes) +
        ". A pose checkpoint's head emits one class (person) per anchor; a "
        "detector head with more would need its class labels, and this build "
        "has none to name them with.");
  if (g.dfl_bins != kDflBins)
    throw std::runtime_error(
        label + ": dfl_bins " + std::to_string(g.dfl_bins) + " is not " +
        std::to_string(kDflBins) + ". The head's distribution is softmaxed over "
        "this many bins and its expectation is taken over the same width; a "
        "different width is a different head, and decoding one with the other's "
        "bin count produces keypoints spread across the image.");
  if (g.detect_cout != g.out_channels())
    throw std::runtime_error(
        label + ": detect_cout " + std::to_string(g.detect_cout) + " is not " +
        "4 + " + std::to_string(g.num_classes) + " + " +
        std::to_string(g.num_keypoints) + "*3 = " +
        std::to_string(g.out_channels()) + ". The decode step reads four box "
        "values, then the class score, then " +
        std::to_string(g.num_keypoints) + " keypoints x 3 out of this tensor, in "
        "that order, and a tensor of another width means it would read past the "
        "end of a row into the next anchor's.");
  for (int64_t c = 0; c < 3; ++c)
    if (g.std_dev[static_cast<size_t>(c)] == 0.f)
      throw std::runtime_error(label + ": image_std[" + std::to_string(c) +
                               "] is zero; the normalise step would divide by "
                               "zero and make a NaN of every pixel.");
  if (g.n_levels() <= 0)
    throw std::runtime_error(label + ": num_levels " + std::to_string(g.n_levels()));

  // The levels have to tile the letterboxed input exactly, because the decode
  // maps a cell (i, j) at stride s to pixel (i*s, j*s) and a grid that does not
  // divide the input leaves the last row of the pyramid off the image.
  for (int64_t i = 0; i < g.n_levels(); ++i) {
    if (g.strides[static_cast<size_t>(i)] <= 0 ||
        g.grid[static_cast<size_t>(i)] <= 0)
      throw std::runtime_error(label + ": level " + std::to_string(i) +
                               " has stride " + std::to_string(g.strides[static_cast<size_t>(i)]) +
                               " and grid " + std::to_string(g.grid[static_cast<size_t>(i)]) +
                               "; the decode divides by the stride and multiplies "
                               "by the grid.");
    if (g.input_size % g.strides[static_cast<size_t>(i)])
      throw std::runtime_error(
          label + ": input_size " + std::to_string(g.input_size) +
          " is not a whole number of stride-" +
          std::to_string(g.strides[static_cast<size_t>(i)]) +
          " cells. This build letterboxes to a square and does not pad the "
          "pyramid to a stride multiple, because a padded level produces cells "
          "whose centres are outside the image.");
    if (g.input_size / g.strides[static_cast<size_t>(i)] != g.grid[static_cast<size_t>(i)])
      throw std::runtime_error(
          label + ": level " + std::to_string(i) + " declares grid " +
          std::to_string(g.grid[static_cast<size_t>(i)]) + " but input_size " +
          std::to_string(g.input_size) + " over stride " +
          std::to_string(g.strides[static_cast<size_t>(i)]) + " is " +
          std::to_string(g.input_size / g.strides[static_cast<size_t>(i)]) +
          " cells. The network produced one and the decode would walk another.");
  }

  // -- weights -------------------------------------------------------------
  const int64_t n_convs = need_int(f, "num_convs", label);
  if (n_convs <= 0)
    throw std::runtime_error(label + ": num_convs " + std::to_string(n_convs));

  // Which convolutions declare no bias. A list, read before the loop so the
  // loop can consult it, and checked for shape rather than for content: an
  // index outside [0, num_convs) names a convolution that does not exist, and
  // an out-of-range bias-free entry would otherwise make the reader skip the
  // bias check for a convolution it never reaches.
  std::vector<int64_t> bias_free;
  {
    std::string raw;
    try {
      raw = f.config_string("bias_free");
    } catch (const std::exception &) {
      throw std::runtime_error(
          label + ": bias_free is absent. The container has to say which of "
          "its convolutions carry no bias, because a missing tensor and a "
          "deliberately absent one are the same bytes and only the list tells "
          "them apart.");
    }
    npue::json::Value v;
    try {
      v = npue::json::parse(raw);
    } catch (const std::exception &) {
      throw std::runtime_error(label + ": bias_free (" + raw +
                               ") is not readable JSON. It is a list of "
                               "convolution indices, not a scalar.");
    }
    for (const auto &e : v.as_array()) {
      const int64_t idx = static_cast<int64_t>(e.as_number());
      if (idx < 0 || idx >= n_convs)
        throw std::runtime_error(
            label + ": bias_free lists conv " + std::to_string(idx) +
            ", which is outside the " + std::to_string(n_convs) +
            " convolutions this container holds. An entry that names no "
            "convolution would make the reader skip a bias check it never "
            "reached.");
      if (std::find(bias_free.begin(), bias_free.end(), idx) != bias_free.end())
        throw std::runtime_error(label + ": bias_free lists conv " +
                                 std::to_string(idx) + " twice");
      bias_free.push_back(idx);
    }
  }

  g.convs.resize(static_cast<size_t>(n_convs));
  g.w_storage.resize(static_cast<size_t>(n_convs));

  // The weight precision, and the int4 group. Both REQUIRED, for the same reason
  // head_box_format is: a reader that had to guess would read an int8 byte as
  // four fp32 ones -- the same 73 shapes, the same 116 nodes, and a network
  // whose output is a plausible pose.
  std::string wdtype;
  try {
    wdtype = f.config_string("conv_weight_dtype");
  } catch (const std::exception &e) {
    throw std::runtime_error(
        label + ": conv_weight_dtype -- " + e.what() +
        ". A container has to say what its weights are stored at; see "
        "tools/lib/conv_quant.py for the four spellings and what each costs.");
  }
  if (wdtype != "f32" && wdtype != "bf16" && wdtype != "i8" && wdtype != "i4")
    throw std::runtime_error(
        label + ": conv_weight_dtype is '" + wdtype +
        "', and this runtime reads f32, bf16, i8 and i4. There is no fp16 or "
        "fp8 convolution weight here, and reading either as one of the four "
        "produces a network with the right shapes and no relationship to the "
        "checkpoint.");
  const int64_t i4_group = need_int(f, "conv_int4_group", label);
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

  for (int64_t i = 0; i < n_convs; ++i) {
    const std::string base = "conv." + std::to_string(i);
    const auto &info = f.info(base + ".w");
    if (info.logical_shape.size() != 4)
      throw std::runtime_error(
          label + ": " + base + ".w has " + std::to_string(info.logical_shape.size()) +
          " dimensions, not 4. A convolution's weight is [Cout, Cin, kh, kw] "
          "and every index in the graph depends on that order.");
    ConvW &c = g.convs[static_cast<size_t>(i)];
    c.index = static_cast<int>(i);
    c.cout = info.logical_shape[0];
    c.cin = info.logical_shape[1];
    c.kh = info.logical_shape[2];
    c.kw = info.logical_shape[3];
    const int64_t want = c.cout * c.cin * c.kh * c.kw;
    // [Cout, K] -- the view the scale rule and the payload are both indexed in.
    const int64_t N = c.cout, K = c.cin * c.kh * c.kw;
    if (wdtype == "f32") {
      if (static_cast<int64_t>(info.nbytes) != want * 4)
        throw std::runtime_error(
            label + ": " + base + ".w is " + std::to_string(info.nbytes) +
            " bytes, which is not " + std::to_string(want) + " fp32 values. A "
            "weight whose byte count disagrees with its shape is not a weight at a "
            "different precision; it is a number of the wrong size.");
      c.w = f.raw(base + ".w").as<float>();
    } else if (wdtype == "bf16") {
      // bf16: two bytes per weight, the source's own bits rounded RNE. There is
      // no fp32 view into the payload to point at, so the bits widen once, at
      // load, into the same per-convolution storage the int8/int4 paths use --
      // and the pointer swap at the end of that path is the only difference
      // between the three. Reusing `from_bf16`/`bf16_read` rather than writing
      // a decode here is the same rule as everywhere else these bits are read:
      // one widening rule, or two rules that disagree in the last place.
      if (static_cast<int64_t>(info.nbytes) != want * 2)
        throw std::runtime_error(
            label + ": " + base + ".w is " + std::to_string(info.nbytes) +
            " bytes, which is not " + std::to_string(want * 2) + " for " +
            std::to_string(want) + " bf16 values. A weight whose byte count "
            "disagrees with its shape is not a weight at a different precision; "
            "it is a number of the wrong size.");
      std::vector<float> &dst = g.w_storage[static_cast<size_t>(i)];
      dst.resize(static_cast<size_t>(want));
      app::bf16_read(dst.data(), f.raw(base + ".w").data,
                     static_cast<size_t>(want));
      c.w = dst.data();
    } else {
      // int8: one byte per weight. int4: one byte per TWO weights, and K may be
      // ODD -- YOLOv8's stem is 3x3x3 = 27 -- so the payload is ceil(K/2) bytes
      // per row with a zero pad nibble at the end. ceil, not K/2: computing it
      // the other way makes the last nibble of the last row a phantom weight.
      const int64_t packed = wdtype == "i8" ? K : (K + 1) / 2;
      if (static_cast<int64_t>(info.nbytes) != N * packed)
        throw std::runtime_error(
            label + ": " + base + ".w is " + std::to_string(info.nbytes) +
            " bytes, which is not " + std::to_string(N * packed) + " for a [" +
            std::to_string(N) + ", " + std::to_string(K) + "] " + wdtype +
            " weight. The scale tensors are read next and are sized against the "
            "shape, so a payload of the wrong length is caught here rather "
            "than reading past the end of it.");
      const auto &si = f.info(base + ".wscale");
      if (static_cast<int64_t>(si.nbytes) != N * 4)
        throw std::runtime_error(
            label + ": " + base + ".wscale is " + std::to_string(si.nbytes) +
            " bytes for " + std::to_string(N) + " output channels. The scale is "
            "per output channel -- one per row of this weight -- and a scale "
            "count that disagrees with the row count puts every channel after "
            "the last one on another channel's factor.");
      const float *const s = f.raw(base + ".wscale").as<float>();
      std::vector<float> &dst = g.w_storage[static_cast<size_t>(i)];
      dst.resize(static_cast<size_t>(want));
      if (wdtype == "i8") {
        const int8_t *const q = f.raw(base + ".w").as<int8_t>();
        for (int64_t n = 0; n < N; ++n)
          for (int64_t k = 0; k < K; ++k)
            dst[static_cast<size_t>(n * K + k)] =
                static_cast<float>(q[n * K + k]) * s[n];
      } else {
        // The group scales are [G, N] -- indexed by group along the INPUT axis
        // first -- and the dequantisation is the rank-1 product the packer
        // measured its error against. Zero groups are held at 1 by the packer,
        // so a dead input channel stays exactly zero instead of becoming NaN.
        const int64_t G = i4_group > 0 ? (K + i4_group - 1) / i4_group : 1;
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
            const int64_t gsel =
                i4_group > 0 ? k / i4_group : 0;
            dst[static_cast<size_t>(n * K + k)] =
                static_cast<float>(v) * s[n] * gs[gsel * N + n];
          }
      }
      c.w = dst.data();
    }
    // A bias is REQUIRED, not defaulted to zero -- but a convolution may
    // declare itself bias-free, and the list is what says which. The
    // alternative (infer it from the absence of the tensor) cannot work,
    // because a container that is missing a tensor it should have and one that
    // deliberately has no bias are the same bytes on disk.
    //
    // Which convolutions may be bias-free is therefore NOT decided here. It is
    // decided once, below, by the only convolution that has a reason to be: the
    // DFL's projection, whose analytic fold reads
    //     sum_bin w[0][bin] * p[bin] + b
    // and is exact with b = 0. Anything else bias-free means a layer's offset
    // has been dropped, and the refusal names it.
    const bool bias_free_here =
        std::find(bias_free.begin(), bias_free.end(), i) != bias_free.end();
    if (bias_free_here) {
      if (f.has(base + ".b"))
        throw std::runtime_error(
            label + ": conv " + std::to_string(i) + " is listed in bias_free "
            "and also carries a .b tensor. The list is what the head's analytic "
            "fold reads to decide whether the offset term is zero, so a "
            "convolution cannot be in both states at once.");
      c.b = nullptr;
      continue;
    }
    if (!f.has(base + ".b"))
      throw std::runtime_error(
          label + ": " + base + ".b is absent and conv " + std::to_string(i) +
          " is not listed in bias_free. Every other convolution in this "
          "architecture carries a bias; one without it shifts every output "
          "channel by the difference, which is a different network.");
    const auto &bi = f.info(base + ".b");
    if (static_cast<int64_t>(bi.nbytes) != c.cout * 4)
      throw std::runtime_error(label + ": " + base + ".b is " +
                               std::to_string(bi.nbytes) + " bytes for " +
                               std::to_string(c.cout) + " output channels");
    c.b = f.raw(base + ".b").as<float>();
    if (c.cout <= 0 || c.cin <= 0 || c.kh <= 0 || c.kw <= 0)
      throw std::runtime_error(label + ": " + base + ".w has a zero extent");
    if (c.kh != c.kw)
      throw std::runtime_error(
          label + ": " + base + ".w is " + std::to_string(c.kh) + "x" +
          std::to_string(c.kw) + ". This build's im2col assumes square kernels; "
          "an anisotropic one needs its own column order and is refused rather "
          "than transposed.");
  }

  // -- the DFL's projection, by index
  //
  // The LAST thing checked, because it is the only check that needs the weights
  // to have been read: the fold in Network::head is
  //
  //   d'[q] = sum_bin w[0][bin] * p[bin] + b  ==  w[0] . E[bin] + b
  //
  // and it is only that identity because the softmax sums to one. A projection
  // of any other shape is a different piece of arithmetic, and applying this
  // fold to it would return a distance in the wrong units by whatever factor the
  // weights should have carried -- a box that still looks like a box.
  g.head.dfl_conv = static_cast<int>(need_int(f, "head_dfl_conv", label));
  for (const char *key : {"head_box_grid_offset", "head_kpt_grid_offset"}) {
    double v = 0.0;
    try {
      v = f.config_double(key);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          label + ": " + key + " -- " + e.what() +
          ". Every exporter puts a grid in this graph and they are not all the "
          "same grid: this checkpoint anchors its boxes at cell centres and its "
          "keypoints at cell corners, which is half a cell of pose error at "
          "every pyramid level. A container that does not say which grid each "
          "branch used has no way to reproduce it.");
    }
    if (!std::isfinite(v) || std::fabs(v) > 1.0)
      throw std::runtime_error(
          label + ": " + key + " is " + std::to_string(v) +
          ", which is not an offset in cells. It is added to the cell's column "
          "and row before the stride multiply, so it has to be finite and within "
          "a cell; anything else is a container whose grid was packed wrong.");
  }
  g.head.box_grid_offset = static_cast<float>(f.config_double("head_box_grid_offset"));
  g.head.kpt_grid_offset = static_cast<float>(f.config_double("head_kpt_grid_offset"));
  if (g.head.dfl_conv < 0 || g.head.dfl_conv >= n_convs)
    throw std::runtime_error(
        label + ": head_dfl_conv " + std::to_string(g.head.dfl_conv) +
        " is outside the " + std::to_string(n_convs) +
        " convolutions this container holds. The head folds this one into the "
        "distribution's expectation, and conv 0 would mean the STEM's weights, "
        "which would scale every distance by the stem's first output channel.");
  {
    const ConvW &d = g.convs[static_cast<size_t>(g.head.dfl_conv)];
    const int64_t want_shape[4] = {1, g.dfl_bins, 1, 1};
    const int64_t got_shape[4] = {d.cout, d.cin, d.kh, d.kw};
    if (d.cout != 1 || d.cin != g.dfl_bins || d.kh != 1 || d.kw != 1)
      throw std::runtime_error(
          label + ": head_dfl_conv " + std::to_string(g.head.dfl_conv) +
          " is [" + std::to_string(d.cout) + ", " + std::to_string(d.cin) + ", " +
          std::to_string(d.kh) + ", " + std::to_string(d.kw) + "], not [" +
          std::to_string(want_shape[0]) + ", " + std::to_string(want_shape[1]) +
          ", 1, 1]. This head op folds a projection from the " +
          std::to_string(g.dfl_bins) + " distribution bins to one distance; a "
          "projection of another shape is arithmetic this runtime does not "
          "implement, and scaling the expectation by the wrong weights returns "
          "boxes in the wrong units.");
    g.head.dfl_bias = d.b != nullptr;
    if (g.head.dfl_bias)
      throw std::runtime_error(
          label + ": head_dfl_conv " + std::to_string(g.head.dfl_conv) +
          " carries a bias, and the head's fold adds it as a constant after "
          "taking the expectation. That is correct only because the fold's "
          "identity -- sum w*b*p == w.E[b] -- was derived for the projection "
          "alone; with an offset the fold would need a per-bin "
          "w[1][bin]*p[bin] term this runtime does not evaluate. A head with a "
          "biased projection is a different head.");
    if (std::find(bias_free.begin(), bias_free.end(), g.head.dfl_conv) ==
        bias_free.end())
      throw std::runtime_error(
          label + ": head_dfl_conv " + std::to_string(g.head.dfl_conv) +
          " is not listed in bias_free. The fold's identity holds because the "
          "softmax sums to one and the projection's offset is zero; a container "
          "that says neither cannot have that property.");
  }

  g.graph = read_graph(f, n_convs, label);
  if (g.graph.empty())
    throw std::runtime_error(label + ": the packed graph is empty");
  if (g.graph.back().op != Op::Detect)
    throw std::runtime_error(
        label + ": the packed graph ends in " + op_to_string(g.graph.back().op) +
        " and not \"detect\". The graph must terminate at the head, because the "
        "decode step is what turns the head's tensor into boxes and keypoints.");
  for (size_t i = 0; i + 1 < g.graph.size(); ++i)
    if (g.graph[i].op == Op::Detect)
      throw std::runtime_error(
          label + ": graph node " + std::to_string(i) +
          " is a second \"detect\". There is one head.");

  // -- typecheck the list, so a packer cannot emit a graph that walks off the
  // end of its own value stack. Node i may read node j < i.
  std::vector<Tensor> shape(g.graph.size());
  const Tensor input_shape{3, g.input_size, g.input_size};
  for (size_t i = 0; i < g.graph.size(); ++i) {
    const Layer &L = g.graph[i];
    const std::string where =
        label + ": graph node " + std::to_string(i) + " (" + op_to_string(L.op) + ")";
    // Operand -1 IS the graph's input tensor. It has to be a distinguished
    // index rather than node 0's slot, because node 0's slot is node 0's
    // OUTPUT: an offset-by-one here would make the second node in the graph read
    // the image where it should read the first convolution, and every channel
    // count after that would still add up.
    auto operand = [&](int idx, const std::string &role) -> const Tensor & {
      if (idx == -1) return input_shape;
      if (idx < 0 || static_cast<size_t>(idx) >= i)
        throw std::runtime_error(
            where + ": " + role + " operand " + std::to_string(idx) +
            " is not an earlier node (or the input, which is -1). The graph is "
            "a straight-line list, so every operand has to be something already "
            "computed; -1 is the image and 0..i-1 are the nodes before this "
            "one.");
      return shape[static_cast<size_t>(idx)];
    };

    switch (L.op) {
      case Op::Conv: {
        const ConvW &c = g.convs[static_cast<size_t>(L.conv)];
        const Tensor &in = operand(L.inputs.at(0), "input");
        if (in.c != c.cin)
          throw std::runtime_error(
              where + ": input has " + std::to_string(in.c) +
              " channels and the weight wants " + std::to_string(c.cin) +
              ". This is the check that catches a graph whose node order drifted "
              "from its weights: every shape still agrees with something, and "
              "only the join is wrong.");
        if (in.h + 2 * L.pad_h < c.kh || in.w + 2 * L.pad_w < c.kw)
          throw std::runtime_error(where + ": input is " +
                                   std::to_string(in.h) + "x" +
                                   std::to_string(in.w) + " and the kernel is " +
                                   std::to_string(c.kh) + "x" + std::to_string(c.kw));
        // floor((H + 2*pad - k)/stride) + 1 -- the standard extent, which is
        // what keeps the SAME case (pad = k/2, stride 1) preserving H while the
        // stem's (pad 1, stride 2) halves it. Written out rather than
        // specialised to one of those two, because a typecheck that hardcodes
        // "same, stride 1" agrees with an im2col that hardcodes it about a
        // network with a stem, and the disagreement never surfaces: the shapes
        // are self-consistent and the picture is 642x642.
        shape[i] = {c.cout, (in.h + 2 * L.pad_h - c.kh) / L.stride + 1,
                    (in.w + 2 * L.pad_w - c.kw) / L.stride + 1};
        break;
      }
      case Op::Concat: {
        const Tensor &a = operand(L.inputs[0], "first");
        int64_t c = a.c;
        for (size_t t = 1; t < L.inputs.size(); ++t) {
          const Tensor &b = operand(L.inputs[t], "operand " + std::to_string(t));
          if (a.h != b.h || a.w != b.w)
            throw std::runtime_error(
                where + ": operand 0 is " + std::to_string(a.h) + "x" +
                std::to_string(a.w) + " and operand " + std::to_string(t) +
                " is " + std::to_string(b.h) + "x" + std::to_string(b.w) +
                ". The join is channel-wise, so every operand has to be the "
                "same size; they do not, and padding one to the other would "
                "invent pixels.");
          c += b.c;
        }
        shape[i] = {c, a.h, a.w};
        break;
      }
      case Op::Slice: {
        const Tensor &a = operand(L.inputs[0], "input");
        // Checked here as well as in Network::slice: this is where the container
        // is validated against ITSELF, and a graph whose slices do not tile its
        // own channel counts is refused before a single pixel moves.
        if (L.chan + L.nch > a.c)
          throw std::runtime_error(
              where + ": takes channels [" + std::to_string(L.chan) + ", " +
              std::to_string(L.chan + L.nch) + ") out of a " +
              std::to_string(a.c) +
              "-channel tensor. Clamping here would put different values in "
              "different channels than the checkpoint put there, and every "
              "downstream channel count would still add up.");
        shape[i] = {L.nch, a.h, a.w};
        break;
      }
      case Op::Add: {
        const Tensor &a = operand(L.inputs[0], "left");
        const Tensor &b = operand(L.inputs[1], "right");
        if (a.c != b.c || a.h != b.h || a.w != b.w)
          throw std::runtime_error(
              where + ": adds [" + std::to_string(a.c) + "," + std::to_string(a.h) +
              "," + std::to_string(a.w) + "] to [" + std::to_string(b.c) + "," +
              std::to_string(b.h) + "," + std::to_string(b.w) +
              "]. The residual is elementwise and shape-exact.");
        shape[i] = a;
        break;
      }
      case Op::MaxPool: {
        const Tensor &in = operand(L.inputs.at(0), "input");
        if (in.h < L.k || in.w < L.k)
          throw std::runtime_error(where + ": pooling a " + std::to_string(in.h) +
                                   "x" + std::to_string(in.w) + " tensor with a " +
                                   std::to_string(L.k) + "x" + std::to_string(L.k) +
                                   " window");
        // same-padded by k/2, stride 1: the extent is preserved, which is the
        // property the SPPF relies on and the one asserted here.
        const int64_t oh = (in.h + 2 * (L.k / 2) - L.k) / L.stride + 1;
        const int64_t ow = (in.w + 2 * (L.k / 2) - L.k) / L.stride + 1;
        if (oh != in.h || ow != in.w)
          throw std::runtime_error(
              where + ": pooling " + std::to_string(L.k) + "/" +
              std::to_string(L.stride) + " takes " + std::to_string(in.h) + "x" +
              std::to_string(in.w) + " to " + std::to_string(oh) + "x" +
              std::to_string(ow) + ". This architecture's pooling is size-"
              "preserving (k=5, stride=1, pad=2); a node that shrinks the map is "
              "not that op, and the spatial pyramid's output is concatenated "
              "with the input, which needs the extents to agree.");
        shape[i] = in;
        break;
      }
      case Op::Upsample: {
        const Tensor &in = operand(L.inputs.at(0), "input");
        if (in.h * L.scale > g.input_size || in.w * L.scale > g.input_size)
          throw std::runtime_error(
              where + ": upsampling " + std::to_string(in.h) + "x" +
              std::to_string(in.w) + " by " + std::to_string(L.scale) +
              " would reach " + std::to_string(in.h * L.scale) + "x" +
              std::to_string(in.w * L.scale) + ", outside the " +
              std::to_string(g.input_size) + "px letterbox");
        shape[i] = {in.c, in.h * L.scale, in.w * L.scale};
        break;
      }
      case Op::Detect: {
        // TWO inputs per level -- the box+class head's output and the keypoint
        // head's -- and the count has to match the level list exactly. This is
        // the check that catches a head wired to the wrong feature map: both
        // tensors are plausibly shaped, so nothing downstream would complain.
        if (L.inputs.size() != static_cast<size_t>(2 * g.n_levels()))
          throw std::runtime_error(
              where + ": detect lists " + std::to_string(L.inputs.size()) +
              " inputs and the container declares " +
              std::to_string(g.n_levels()) +
              " levels. The head takes two per level -- the box+class head's "
              "output and the keypoint head's -- so this is " +
              std::to_string(2 * g.n_levels()) + ".");
        const int64_t box_ch =
            g.head.box == BoxFormat::LtrbDfl ? 4 * g.dfl_bins : 4;
        const int64_t want_bc = box_ch + g.num_classes;
        const int64_t want_kp = g.num_keypoints * 3;
        for (size_t t = 0; t < L.inputs.size(); ++t) {
          const int64_t lv = static_cast<int64_t>(t / 2);
          const bool is_kpt = (t % 2) == 1;
          const Tensor &in = operand(L.inputs[t], is_kpt ? "keypoints" : "boxes");
          const int64_t s = g.grid[static_cast<size_t>(lv)];
          const int64_t want_c = is_kpt ? want_kp : want_bc;
          if (in.c != want_c)
            throw std::runtime_error(
                where + ": level " + std::to_string(lv) + "'s " +
                (is_kpt ? "keypoint" : "box+class") + " tensor has " +
                std::to_string(in.c) + " channels and head_box_format '" +
                (g.head.box == BoxFormat::LtrbDfl ? "ltrb_dfl" : "xyxy") +
                "' needs " + std::to_string(want_c) + " (" +
                std::to_string(box_ch) + " box + " +
                std::to_string(g.num_classes) + " class, or " +
                std::to_string(g.num_keypoints) + " x 3 keypoints). Every "
                "channel is a float, so a count that is wrong here would be read "
                "as the wrong quantities and returned as a plausible pose.");
          if (in.h != s || in.w != s)
            throw std::runtime_error(
                where + ": level " + std::to_string(lv) + " declares grid " +
                std::to_string(s) + " and its " +
                (is_kpt ? "keypoint" : "box+class") + " tensor is " +
                std::to_string(in.h) + "x" + std::to_string(in.w) +
                ". The head maps cell (i, j) to pixel (i*s, j*s), so a tensor at "
                "another resolution would put every joint on the neighbouring "
                "cell's pixel.");
        }
        // The head's OUTPUT is fixed by the architecture, and it is the only
        // shape in this container that is not the input's own.
        shape[i] = {g.out_channels(), g.n_anchors(), 1};
        break;
      }
    }
  }

  // The head tensor's spatial extent has to be the concatenated pyramid, and the
  // decode step indexes it as (level, cell) using `grid`. If they disagree the
  // keypoints are real numbers attached to the wrong cells.
  {
    int64_t cells = 0;
    for (int64_t gg : g.grid) cells += gg * gg;
    if (g.detect_h * g.detect_w != cells)
      throw std::runtime_error(
          label + ": detect_h * detect_w = " + std::to_string(g.detect_h) + " * " +
          std::to_string(g.detect_w) + " = " +
          std::to_string(g.detect_h * g.detect_w) + ", but the levels hold " +
          std::to_string(cells) + " cells (" + std::to_string(g.n_levels()) +
          " grids squared). The decode walks the levels in order and reads " +
          std::to_string(g.out_channels()) + " values per cell, so a tensor that "
          "does not hold exactly the levels' cells would run off its end.");
    const Tensor &head = shape[shape.size() - 1];
    if (head.h * head.w != cells)
      throw std::runtime_error(
          label + ": the head tensor is " + std::to_string(head.h) + "x" +
          std::to_string(head.w) + " = " + std::to_string(head.h * head.w) +
          " cells and the pyramid holds " + std::to_string(cells) +
          ". The graph and the recorded geometry describe different networks.");
  }

  return g;
}

}  // namespace npue::pose
