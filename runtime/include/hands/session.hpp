//===- session.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one hands session: two networks, one image in, N hands out.
// And, only when asked, an array backend -- which today it refuses to do.
//
// THE DEFAULT PATH OPENS NO DEVICE, AND THE OTHER ONE DOES NOT EXIST
// ---------------------------------------------------------------
// Same contract as arch=6: Session(...) with no artifacts is a host-only
// session, and `npuembeddings hands` and the gate both work on a machine with no
// /dev/accel0 at all.
//
// What differs is the second half of the contract. arch=6's --npu-ops conv
// dispatches 72 convolutions onto a real design set that this tree can load and
// that was measured at 290 ms against the host's 150 ms -- a measured LOSS, on
// this panel, for that model. arch=7 has no such measurement: there is no
// design set carrying the 31 distinct dense (K, N) pairs these two graphs use,
// so asking for the array here would produce a number with no denominator.
// --npu-ops conv is therefore REFUSED, by name, with the reason, rather than
// quietly running on the host and printing host timings under a flag that says
// otherwise. Hands is the only session where that flag does not exist yet and
// the reason is a missing measurement rather than a missing feature.
//
// TWO NETWORKS, ONE CONTAINER
// ---------------------------
// The palm detector is 87 graph nodes and the landmark network 58, and they run
// in that order with the whole decode between them. One Session holds one
// Network for each, because the two graphs' weights are the two graphs' and
// sharing a Network would mean sharing the im2col buffer and the cost counters
// across a call that is not concurrent anyway.
//
// EVERY HAND IS AN INDEPENDENT CROP
// ---------------------------------
// The landmark network runs once per surviving detection, on a crop of the
// ORIGINAL frame. It is not a batched call and there is no way to make it one
// without changing the model's arithmetic, so `max_hands` is a cap on how many
// of the NMS survivors are run through the second stage and the default is all
// of them.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hands/decode.hpp"
#include "hands/geometry.hpp"
#include "hands/image.hpp"
#include "hands/net.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/image.hpp"

namespace npue::hands {

// One image's answer, with the timings split so a client can see where the frame
// went. Same shape as pose::Result and for the same reason: the CLI prints it
// and the server returns it.
struct Result {
  std::vector<Detection> detections;   // what the palm stage found, in score order
  std::vector<Hand> hands;             // the same subset, run through stage 2
  double front_end_s = 0.0;   // decode + letterbox + normalise, stage 1
  double palm_s = 0.0;        // the palm network only
  double nms_s = 0.0;         // the anchor decode, the sigmoid and NMS
  double crop_s = 0.0;        // every crop, rotate and area-resize
  double lm_s = 0.0;          // the landmark network, summed over the hands
  double post_s = 0.0;
  double total_s = 0.0;
  int64_t width = 0, height = 0;   // the SOURCE image, so a client can scale
  // The front end's transform, so a client can map its own annotations through
  // the pipeline the model actually saw rather than guessing at the letterbox.
  double scale = 1.0;
  int64_t pad_x = 0, pad_y = 0;
  int64_t hands_run = 0;     // how many of `detections` reached stage 2
  Cost palm_cost, lm_cost;
  // The SECOND stage's geometry, for the FIRST hand, in the caller's pixels. Not
  // part of a hand's answer -- the landmarks are the answer -- but it is the one
  // intermediate whose being wrong produces plausible landmarks, so the status
  // line prints it: an angle off by a degree, or a pad_bias off by the crop's
  // own padding, moves every joint by a fraction of its distance from the wrist
  // and nothing else in the output says so.
  double angle_deg = 0.0;
  double rot_bbox[4] = {0, 0, 0, 0};
  double palm_kp_box[4] = {0, 0, 0, 0};
  double crop_pad_bias[2] = {0, 0};
  double crop_mean = 0.0;   // the crop's mean luminance in [0, 1]
};

class Session {
public:
  // `artifacts` may be empty -- the host-only session, and the default. A
  // non-empty one is REFUSED rather than accepted-and-ignored; see the header.
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads, int64_t max_hands = -1);

  // Out of line, not defaulted, so it is defined in session.cpp where the
  // forward-declared array backend is a complete type.
  ~Session();

  const Geometry &geometry() const { return geom_; }
  Network &palm_network() { return *palm_net_; }
  Network &lm_network() { return *lm_net_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  int64_t max_hands() const { return max_hands_; }
  void set_max_hands(int64_t n) { max_hands_ = n; }

  Result detect(const vit::Image &im);
  Result detect_file(const std::string &path);

  // The letterbox only, for a gate that wants the front end without the 87
  // convolutions. Same call detect() makes.
  static std::vector<float> preprocess_file(const std::string &path,
                                            const Geometry &g, Letterbox &lb);

private:
  npue::File &model_;
  Geometry geom_;
  std::string art_, name_;
  int64_t max_hands_;

  // Declaration order is construction order and it matters: each network holds
  // REFERENCES to a Pool, so it has to exist first.
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<Network> palm_net_, lm_net_;
  // The two networks' input tensors, kept between calls so the front end writes
  // into memory that is already faulted in -- the same measured reason arch=6's
  // Session keeps its own (2 ms of a ~180 ms frame, every frame, for a buffer
  // whose contents never survive it). The landmark one is reused across the
  // hands of a frame for the same reason.
  //
  // It also makes detect() non-reentrant, which is not a new restriction: it
  // already writes the networks' cost_ counters.
  Tensor palm_in_, lm_in_;
};

}  // namespace npue::hands
