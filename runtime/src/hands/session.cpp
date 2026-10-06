//===- session.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=7 session. See session.hpp for the contracts.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "hands/session.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "common/host_kernels.hpp"   // app::now_s

namespace npue::hands {

namespace {

Tensor as_tensor(const std::vector<float> &d, int64_t c, int64_t h, int64_t w) {
  Tensor t;
  t.c = c;
  t.h = h;
  t.w = w;
  t.d = d;
  return t;
}

}  // namespace

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads, int64_t max_hands)
    : model_(model), art_(artifacts), name_(model_name), max_hands_(max_hands) {
  geom_ = read_geometry(model_, name_);
  pool_ = std::make_unique<app::Pool>(threads);
  Placement place;   // the default: everything on the host
  if (!art_.empty()) {
    // THE SLOT RANGES, ASSIGNED HERE AND FROM ONE VECTOR. This container holds
    // the palm detector AND the landmark network, with SEPARATE convolution
    // index spaces that both start at zero, while the array backend has ONE
    // panel table. So the two networks get disjoint slot ranges and each
    // translates its own conv index through a TABLE -- see Placement::slots for
    // why that is not `base + conv`, which is a bug this architecture inherited
    // from arch=8's first version.
    place.conv_on_array = true;
    place.design_dir = art_;
    std::vector<conv::ConvSlot> slots;
    palm_slots_.assign(geom_.palm_convs.size(), -1);
    for (size_t i = 0; i < geom_.palm_convs.size(); ++i) {
      const ConvW &cw = geom_.palm_convs[i];
      if (cw.depthwise()) continue;
      palm_slots_[i] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "palm",
                                     "palm.", static_cast<int64_t>(i),
                                     stream_of(geom_.palm_graph, i), cw.cin,
                                     cw.cout, cw.kh, cw.kw, cw.b});
    }
    lm_slots_.assign(geom_.lm_convs.size(), -1);
    for (size_t i = 0; i < geom_.lm_convs.size(); ++i) {
      const ConvW &cw = geom_.lm_convs[i];
      if (cw.depthwise()) continue;
      lm_slots_[i] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "lm",
                                     "lm.", static_cast<int64_t>(i),
                                     stream_of(geom_.lm_graph, i), cw.cin,
                                     cw.cout, cw.kh, cw.kw, cw.b});
    }
    if (slots.empty())
      throw std::runtime_error(
          name_ + ": --npu-ops conv was asked for and neither graph has a dense "
          "convolution. A design set with no slots behind it is not a design set.");
    array_ = conv::make_array_backend(model_, slots, name_, art_, *pool_);
  }
  Placement palm_place = place, lm_place = place;
  palm_place.group = "palm";
  palm_place.slots = &palm_slots_;
  lm_place.group = "lm";
  lm_place.slots = &lm_slots_;
  palm_net_ = std::make_unique<Network>(geom_, palm_place, *pool_,
                                       array_ ? array_.get() : nullptr);
  lm_net_ = std::make_unique<Network>(geom_, lm_place, *pool_,
                                      array_ ? array_.get() : nullptr);
}

// The design stream the graph node calling convolution `conv` names, or "".
//
// The graph is SEARCHED rather than indexed by convolution order: the two are
// equal in every container here, and if that stops being true the wrong answer is
// a panel staged for another layer -- right shape, wrong weights, no error.
std::string Session::stream_of(const std::vector<Layer> &layers, int64_t conv) {
  const Layer *found = nullptr;
  for (const Layer &l : layers)
    if (l.conv == conv && (l.op == Op::Conv || l.op == Op::DwConv)) {
      if (found)
        throw std::runtime_error(
            name_ + ": convolution " + std::to_string(conv) +
            " is called by two graph nodes. The array path keys panels by which "
            "node calls a convolution, and two nodes sharing one would stage one "
            "panel for two different layers.");
      found = &l;
    }
  return found ? found->stream : std::string();
}

Session::~Session() = default;

std::vector<float> Session::preprocess_file(const std::string &path,
                                            const Geometry &g,
                                            Letterbox &lb) {
  const vit::Image src = vit::decode_image(path);
  const Image im{src};
  const Image padded = letterbox(im, g.palm_input_size, &lb);
  return to_nchw_normalised(padded, g.mean, g.std_dev);
}

Result Session::detect(const vit::Image &src) {
  const Image im{src};
  if (im.empty())
    throw std::runtime_error(name_ + ": the image decoded to nothing");
  Result r;
  r.width = im.width;
  r.height = im.height;
  palm_net_->reset_cost();
  lm_net_->reset_cost();

  // -- stage 1 ------------------------------------------------------------
  const double t0 = app::now_s();
  PalmStage ps = palm_stage(im, geom_);
  r.scale = ps.lb.scale;
  r.pad_x = ps.lb.pad_orig_x;
  r.pad_y = ps.lb.pad_orig_y;
  const int64_t ps_size = geom_.palm_input_size;
  const std::vector<float> pd = to_nchw_normalised(ps.padded, geom_.mean,
                                                   geom_.std_dev);
  palm_in_ = as_tensor(pd, 3, ps_size, ps_size);
  r.front_end_s = app::now_s() - t0;

  // The INPUT is dumped too, as node -1, and it is the first thing a gate
  // should look at: a front-end difference shows up here as a difference in
  // 150k pixels rather than as an amplified one in 58 nodes' worth of
  // convolutions, and separating the two turns "the C++ runtime disagrees" into
  // "the crop" or "the walk". SIZE_MAX arrives as the graph-node index -1 that
  // the reader looks for; the same convention pose_mode.hpp uses.
  if (palm_net_->on_node) palm_net_->on_node(SIZE_MAX, palm_in_);

  const double t1 = app::now_s();
  const std::vector<Tensor> &pnodes =
      palm_net_->run_body(palm_in_, geom_.palm_graph, geom_.palm_convs,
                          palm_net_->wt_palm(), geom_.palm_slopes,
                          geom_.palm_nodes, "palm");
  r.palm_s = app::now_s() - t1;
  const PalmOut ph = palm_net_->palm_head(pnodes, "palm");

  const double t2 = app::now_s();
  const double long_side = static_cast<double>(std::max(im.height, im.width));
  r.detections = decode_palm(geom_, ph, ps.lb, long_side);
  r.nms_s = app::now_s() - t2;

  // -- stage 2, once per surviving detection -------------------------------
  const int64_t run = (max_hands_ < 0 || static_cast<int64_t>(r.detections.size()) < max_hands_)
                          ? static_cast<int64_t>(r.detections.size())
                          : max_hands_;
  r.hands_run = run;
  const int64_t ls = geom_.lm_input_size;
  for (int64_t i = 0; i < run; ++i) {
    const double t3 = app::now_s();
    LandmarkStage st = landmark_stage(im, geom_, r.detections[i]);
    const std::vector<float> ld =
        to_nchw_normalised(st.blob, geom_.mean, geom_.std_dev);
    lm_in_ = as_tensor(ld, 3, ls, ls);
    r.crop_s += app::now_s() - t3;

    if (lm_net_->on_node) lm_net_->on_node(SIZE_MAX, lm_in_);

    const double t4 = app::now_s();
    const std::vector<Tensor> &lnodes =
        lm_net_->run_body(lm_in_, geom_.lm_graph, geom_.lm_convs,
                          lm_net_->wt_lm(), geom_.lm_slopes, geom_.lm_nodes,
                          "lm");
    const LmOut lo = lm_net_->lm_head(lnodes, "lm");
    r.lm_s += app::now_s() - t4;

    if (i == 0) {
      r.angle_deg = st.angle;
      for (int k = 0; k < 4; ++k) r.rot_bbox[k] = st.rot_bbox[k];
      for (int k = 0; k < 4; ++k) r.palm_kp_box[k] = st.palm_kp_box[k];
      for (int k = 0; k < 2; ++k) r.crop_pad_bias[k] = st.pad_bias[k];
      double acc = 0.0;
      for (float v : st.blob.rgb) acc += v;
      r.crop_mean = st.blob.rgb.empty()
                        ? 0.0
                        : acc / (255.0 * static_cast<double>(st.blob.rgb.size()));
    }
    const double t5 = app::now_s();
    r.hands.push_back(postprocess(geom_, lo, st, r.detections[i]));
    r.post_s += app::now_s() - t5;
  }
  r.palm_cost = palm_net_->cost();
  r.lm_cost = lm_net_->cost();
  r.total_s = app::now_s() - t0;
  return r;
}

Result Session::detect_file(const std::string &path) {
  return detect(vit::decode_image(path));
}

}  // namespace npue::hands
