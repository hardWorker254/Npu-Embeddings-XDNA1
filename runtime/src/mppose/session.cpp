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
  if (!art_.empty())
    throw std::runtime_error(
        name_ + ": this session was handed the design set at " + art_ +
        ", but arch=8 has no array path. There is no design set carrying the 99 "
        "distinct dense (K, N) pairs these two graphs use, so the array path "
        "cannot be measured here, and a session that accepted the directory and "
        "then ran every convolution on the host would report timings under a flag "
        "that says the opposite. Run it on the host: --npu-ops conv is not "
        "accepted for an arch=8 container.");
  pool_ = std::make_unique<app::Pool>(threads);
  Placement place;   // the default, and the only honest one here
  det_net_ = std::make_unique<Network>(geom_, place, *pool_);
  pose_net_ = std::make_unique<Network>(geom_, place, *pool_);
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
