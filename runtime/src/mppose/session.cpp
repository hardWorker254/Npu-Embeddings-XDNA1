//===- session.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=8 session. See session.hpp for the contracts.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/session.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "common/host_kernels.hpp"   // app::now_s

namespace npue::mppose {

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
                 const std::string &artifacts, int threads, int64_t max_people)
    : model_(model), art_(artifacts), name_(model_name), max_people_(max_people) {
  geom_ = read_geometry(model_, name_);
  pool_ = std::make_unique<app::Pool>(threads);
  // THE SLOT RANGES, ASSIGNED HERE AND FROM ONE VECTOR.
  //
  // This container holds two graphs with SEPARATE convolution index spaces, both
  // starting at zero, while the array backend has ONE panel table. So the
  // detector's convolutions and the landmark network's get disjoint slot ranges,
  // in that order, and the same vector is what stages the panels -- there is one
  // number per convolution and it is written once.
  //
  // The alternative -- keying panels by the bare convolution index -- puts the
  // landmark network's conv 0 on the detector's conv 0 panel. Both are the stem
  // of a network of similar width, so the panel is the right shape, the
  // layout_hash matches, every load succeeds, and the detector's boxes come out
  // of a pose network's weights. Nothing reports that.
  Placement place;   // the default: everything on the host
  if (!art_.empty()) {
    place.conv_on_array = true;
    place.design_dir = art_;
    const int64_t n_det = static_cast<int64_t>(geom_.det_convs.size());
    std::vector<conv::ConvSlot> slots;
    // Depthwise convolutions are NOT given slots, and not by accident: a
    // depthwise filter reduces within one channel -- K is kh*kw and N is 1 --
    // so there is no [M, N] GEMM in it to dispatch. They are 62 of this
    // container's 161 convolutions and carry 12.3 % of its MACs, and they stay
    // on the host whatever the design set says.
    //
    // THE SLOT TABLES ARE FILLED AT THE SAME TIME AND FROM THE SAME LOOP, which
    // is the whole point: slot numbers are handed out over the DENSE
    // convolutions only, so `slot = base + conv` is WRONG here -- depthwise ones
    // are interleaved through both graphs, and the detector's conv 8 is not slot
    // 8. Each table is indexed by the network's OWN conv index and holds -1 for
    // the depthwise ones, so the graph walk and the staging read the same number.
    det_slots_.assign(static_cast<size_t>(n_det), -1);
    for (int64_t i = 0; i < n_det; ++i) {
      const ConvW &cw = geom_.det_convs[static_cast<size_t>(i)];
      if (cw.depthwise()) continue;
      det_slots_[static_cast<size_t>(i)] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "det",
                                     "det.", i,
                                     stream_of(geom_.det_graph, i), cw.cin,
                                     cw.cout, cw.kh, cw.kw, cw.b});
    }
    pose_slots_.assign(geom_.pose_convs.size(), -1);
    for (size_t i = 0; i < geom_.pose_convs.size(); ++i) {
      const ConvW &cw = geom_.pose_convs[i];
      if (cw.depthwise()) continue;
      pose_slots_[i] = static_cast<int64_t>(slots.size());
      slots.push_back(conv::ConvSlot{static_cast<int64_t>(slots.size()), "pose",
                                     "pose.", static_cast<int64_t>(i),
                                     stream_of(geom_.pose_graph, i), cw.cin,
                                     cw.cout, cw.kh, cw.kw, cw.b});
    }
    det_base_ = 0;
    pose_base_ = det_slots_.size() == 0 ? 0 : 0;
    if (slots.empty())
      throw std::runtime_error(
          name_ + ": --npu-ops conv was asked for and neither graph has a dense "
          "convolution. A design set with no slots behind it is not a design set.");
    array_ = conv::make_array_backend(model_, slots, name_, art_, *pool_);
  }
  Placement det_place = place, pose_place = place;
  det_place.group = "det";
  det_place.slots = &det_slots_;
  pose_place.group = "pose";
  pose_place.slots = &pose_slots_;
  det_net_ = std::make_unique<Network>(geom_, det_place, *pool_,
                                       array_ ? array_.get() : nullptr);
  pose_net_ = std::make_unique<Network>(geom_, pose_place, *pool_,
                                        array_ ? array_.get() : nullptr);
}

// The design stream one convolution runs on, or "" if the graph does not say.
//
// The graph is searched for the node that CALLS this convolution rather than
// assumed to have one entry per convolution: the two are equal in every
// container in this repository, and if that stops being true the wrong answer
// here is a panel staged for another layer -- right shape, wrong weights, no
// error -- so it is checked rather than indexed.
std::string Session::stream_of(const std::vector<Layer> &layers, int64_t conv) {
  const Layer *found = nullptr;
  for (const Layer &l : layers)
    if (l.op == Op::Conv && l.conv == conv) {
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
                                            const Geometry &g, DetLetterbox &lb) {
  const vit::Image src = vit::decode_image(path);
  const Image im{src};
  return det_input(im, g, &lb);
}

Result Session::detect(const vit::Image &src) {
  const Image im{src};
  if (im.empty())
    throw std::runtime_error(name_ + ": the image decoded to nothing");
  Result r;
  r.width = im.width;
  r.height = im.height;
  det_net_->reset_cost();
  pose_net_->reset_cost();

  // -- stage 1 ------------------------------------------------------------
  const double t0 = app::now_s();
  DetStage ds = det_stage(im, geom_);
  r.scale = ds.lb.scale;
  r.pad_x = ds.lb.pad_bias_x;
  r.pad_y = ds.lb.pad_bias_y;
  const int64_t det_size = geom_.det_input_size;
  det_in_ = as_tensor(ds.blob, 3, det_size, det_size);
  r.front_end_s = app::now_s() - t0;

  // The INPUT is dumped too, as node -1, and it is the first thing a gate should
  // look at: a front-end difference shows up here as a difference in 150k pixels
  // rather than as an amplified one in 93 nodes' worth of convolutions, and
  // separating the two turns "the C++ runtime disagrees" into "the letterbox" or
  // "the walk". SIZE_MAX arrives as the graph-node index -1 that the reader looks
  // for; the same convention hands_mode.hpp uses.
  if (det_net_->on_node) det_net_->on_node(SIZE_MAX, det_in_);

  const double t1 = app::now_s();
  const std::vector<Tensor> &dnodes =
      det_net_->run_body(det_in_, geom_.det_graph, geom_.det_convs,
                         det_net_->wt_det(), geom_.det_slopes, geom_.det_nodes,
                         "det");
  r.det_s = app::now_s() - t1;
  const DetOut dh = det_net_->det_head(dnodes, "det");

  const double t2 = app::now_s();
  r.detections = decode_det(geom_, dh, ds.lb, im.width, im.height);
  r.nms_s = app::now_s() - t2;

  // -- stage 2, once per surviving detection -------------------------------
  const int64_t n_det = static_cast<int64_t>(r.detections.size());
  const int64_t run = (max_people_ < 0 || n_det < max_people_) ? n_det : max_people_;
  r.people_run = run;
  const int64_t ps_size = geom_.pose_input_size;
  for (int64_t i = 0; i < run; ++i) {
    const double t3 = app::now_s();
    PoseStage st = pose_stage(im, geom_, r.detections[i]);
    // The landmark net's own normalisation, which is u8/255 and NOT the
    // detector's: its two front ends are different models' worth of
    // preprocessing and the container carries both.
    pose_in_ = as_tensor(npue::raster::to_nchw_normalised(st.blob, geom_.pose_mean,
                                                          geom_.pose_std),
                         3, ps_size, ps_size);
    r.crop_s += app::now_s() - t3;

    if (pose_net_->on_node) pose_net_->on_node(SIZE_MAX, pose_in_);

    const double t4 = app::now_s();
    const std::vector<Tensor> &pnodes =
        pose_net_->run_body(pose_in_, geom_.pose_graph, geom_.pose_convs,
                            pose_net_->wt_pose(), geom_.pose_slopes,
                            geom_.pose_nodes, "pose");
    const PoseOuts po = pose_net_->pose_head(pnodes, "pose");
    r.pose_s += app::now_s() - t4;

    // The confidence gate is HERE, after the network, exactly as the demo places
    // it: `if conf < confThreshold: return None`. Applying it earlier would need
    // the network to have run first, and applying it later would report a person
    // the reference dropped. Both are visible in `people_run` versus
    // `detections`.
    if (po.conf < geom_.pose_conf_threshold) {
      // Drop this one and everything after it, the demo's way: `for person in
      // persons: pose = infer(...); if pose is not None: poses.append(pose)` does
      // not break, it continues, and on a frame whose confidences fall the rest of
      // the way the difference is visible in which people are reported.
      r.people_run = i;
      continue;
    }

    if (i == 0) {
      r.angle_deg = st.angle;
      r.rot_bbox[0] = 0.0;
      r.rot_bbox[1] = 0.0;
      r.rot_bbox[2] = static_cast<double>(st.square_w);
      r.rot_bbox[3] = static_cast<double>(st.square_h);
      r.crop_pad_bias[0] = st.pad_bias[0];
      r.crop_pad_bias[1] = st.pad_bias[1];
      double acc = 0.0;
      for (float v : st.blob.rgb) acc += v;
      r.crop_mean = st.blob.rgb.empty()
                        ? 0.0
                        : acc / (255.0 * static_cast<double>(st.blob.rgb.size()));
    }
    const double t5 = app::now_s();
    r.people.push_back(postprocess(geom_, po, st, im.width, im.height));
    r.post_s += app::now_s() - t5;
  }
  r.det_cost = det_net_->cost();
  r.pose_cost = pose_net_->cost();
  r.total_s = app::now_s() - t0;
  return r;
}

Result Session::detect_file(const std::string &path) {
  return detect(vit::decode_image(path));
}

}  // namespace npue::mppose
