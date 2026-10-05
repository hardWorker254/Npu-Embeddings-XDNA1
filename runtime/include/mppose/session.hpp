//===- session.hpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one body-pose session: two networks, one image in, N people out.
// And, only when asked, an array backend -- which today it refuses to do.
//
// THE DEFAULT PATH OPENS NO DEVICE, AND THE OTHER ONE DOES NOT EXIST
// ---------------------------------------------------------------
// Same contract as arch=6 and arch=7: Session(...) with no artifacts is a
// host-only session, and `npuembeddings pose` on an arch=8 container and the gate
// both work on a machine with no /dev/accel0 at all.
//
// What differs is the second half. arch=6's --npu-ops conv dispatches 72
// convolutions onto a design set this tree can load, measured at 290 ms against
// the host's 150 ms -- a measured LOSS on this panel, for that model. Neither
// arch=7 nor arch=8 has that measurement: no design set here carries the 99
// distinct dense (K, N) pairs these two graphs use, so asking for the array would
// produce a number with no denominator. --npu-ops conv is therefore REFUSED, by
// name, with the reason, rather than quietly running on the host and printing host
// timings under a flag that says otherwise.
//
// TWO NETWORKS, ONE CONTAINER
// ---------------------------
// The detector is 93 graph nodes and the landmark network 120, and they run in
// that order with the whole decode between them. One Session holds one Network
// for each, because the two graphs' weights are the two graphs' and sharing one
// would mean sharing the im2col buffer and the cost counters across a call that is
// not concurrent anyway.
//
// EVERY PERSON IS AN INDEPENDENT CROP
// ----------------------------------
// The landmark network runs once per surviving detection, on a crop of the
// ORIGINAL frame, rotated and area-resampled. It is not a batched call and there
// is no way to make it one without changing the model's arithmetic, so
// `max_people` caps how many of the NMS survivors are run through the second stage
// and the default is all of them.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mppose/decode.hpp"
#include "mppose/geometry.hpp"
#include "mppose/image.hpp"
#include "mppose/net.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/image.hpp"

namespace npue::mppose {

// One image's answer, with the timings split so a client can see where the frame
// went. Same shape as pose::Result and hands::Result, and for the same reason: the
// CLI prints it and the server returns it.
struct Result {
  std::vector<Detection> detections;   // what the detector found, in score order
  std::vector<Person> people;           // the same subset, run through stage 2
  double front_end_s = 0.0;   // the letterbox, stage 1
  double det_s = 0.0;         // the detector network only
  double nms_s = 0.0;         // the anchor decode, the sigmoid and NMS
  double crop_s = 0.0;        // every crop, square, rotate and area-resize
  double pose_s = 0.0;        // the landmark network, summed over the people
  double post_s = 0.0;
  double total_s = 0.0;
  int64_t width = 0, height = 0;   // the SOURCE image, so a client can scale
  double scale = 1.0;
  int64_t pad_x = 0, pad_y = 0;    // the letterbox's offset, in original pixels
  int64_t people_run = 0;          // how many of `detections` reached stage 2
  Cost det_cost, pose_cost;
  // The SECOND stage's geometry, for the FIRST person, in the caller's pixels. Not
  // part of a person's answer -- the landmarks are the answer -- but it is the one
  // intermediate whose being wrong produces plausible landmarks, so the status line
  // prints it: an angle off by a degree, or a pad_bias off by the square's own
  // padding, moves every joint by a fraction of its distance from the hip and
  // nothing else in the output says so.
  double angle_deg = 0.0;
  double rot_bbox[4] = {0, 0, 0, 0};     // the padded square the crop became
  double crop_pad_bias[2] = {0, 0};      // where that square sits in the frame
  double crop_mean = 0.0;                // the crop's mean luminance in [0, 1]
};

class Session {
public:
  // `artifacts` may be empty -- the host-only session, and the default. A non-empty
  // one is REFUSED rather than accepted-and-ignored; see the header.
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads, int64_t max_people = -1);

  // Out of line, not defaulted, so it is defined in session.cpp where the
  // forward-declared array backend is a complete type.
  ~Session();

  const Geometry &geometry() const { return geom_; }
  Network &det_network() { return *det_net_; }
  Network &pose_network() { return *pose_net_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  int64_t max_people() const { return max_people_; }
  void set_max_people(int64_t n) { max_people_ = n; }

  Result detect(const vit::Image &im);
  Result detect_file(const std::string &path);

  // The detector's front end only, for a gate that wants the letterbox without the
  // 93 convolutions. Same call detect() makes.
  static std::vector<float> preprocess_file(const std::string &path,
                                            const Geometry &g, DetLetterbox &lb);

private:
  npue::File &model_;
  Geometry geom_;
  std::string art_, name_;
  int64_t max_people_;

  // Declaration order is construction order and it matters: each network holds a
  // REFERENCE to a Pool, so it has to exist first.
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<Network> det_net_, pose_net_;
  // The two networks' input tensors, kept between calls so the front end writes
  // into memory that is already faulted in. The pose one is reused across the
  // people of a frame for the same reason.
  //
  // It also makes detect() non-reentrant, which is not a new restriction: it
  // already writes the networks' cost_ counters.
  Tensor det_in_, pose_in_;
};

}  // namespace npue::mppose
