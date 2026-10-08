//===- decode.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 head decode: threshold, keypoint score, NMS, and the
// letterbox inverse. Turns the head tensor into people in the caller's pixels.
//
// WHAT IS AND IS NOT IN HERE
// ---------------------------
// NOT here: the DFL, the grid anchor addition, the box decoding. All of that is
// baked into the exported graph (it is the graph's one Softmax and its output
// is 56 channels, not 116), so the network already produces decoded boxes and
// this step starts from them. Putting a second DFL here would square the
// distribution over bins that have already been summed out.
//
// HERE, and each for a reason:
//   * the class threshold, applied BEFORE the keypoints are read. A person that
//     will be thrown away by the threshold should not cost 51 float reads, and
//     8400 anchors times 56 values is most of the head tensor.
//   * the keypoint visibility threshold, kept as a flag rather than a deletion.
//     A low-visibility keypoint is still a position, and a client drawing a
//     skeleton wants to know the limb was inferred rather than to find a gap
//     that reads as missing data. So the point is retained with visible=false.
//   * NMS, per class, in LETTERBOX pixels. Doing it before the inverse would be
//     cheaper and it is not done, because IoU is not invariant under the letter
//     box's uniform scale + translate -- actually it is, being a ratio of areas
//     under one affine map -- but the CLIP is not, and clipping before NMS is
//     what makes the invariance exact. It is done after, on the caller's pixels,
//     where the numbers are the ones the threshold was chosen against.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pose/geometry.hpp"
#include "pose/image.hpp"
#include "pose/net.hpp"

namespace npue::pose {

// One joint, in the caller's pixel coordinates.
struct Keypoint {
  double x = 0.0;
  double y = 0.0;
  double v = 0.0;          // the head's visibility score, as the model gave it
  bool visible = false;    // v >= DecodeParams::keypoint
  int index = 0;           // 0..16; see Skeleton::name
};

// One person, in the caller's pixel coordinates.
struct Person {
  double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;   // xyxy, clipped to the image
  double score = 0.0;
  int64_t anchor = 0;      // which of the 8400 produced it
  int64_t level = 0;       // which stride
  std::vector<Keypoint> keypoints;
};

// The knobs. Their DEFAULTS are Ultralytics' predict defaults and are recorded
// as the values they are, because a pose client that reports a person count is
// reporting the product of these three numbers.
struct DecodeParams {
  double conf = 0.25;          // class score
  double iou = 0.70;           // NMS overlap
  double keypoint = 0.50;      // below this a joint is reported not visible
  int64_t max_det = 300;       // cap on the returned list
  int64_t max_nms = 30000;     // cap on the pre-NMS candidate count
  // COCO models are single-class, so class-aware NMS degenerates to plain NMS
  // and the offset-by-class trick that separates classes does nothing. It is
  // not implemented, and read_geometry refuses a multi-class head, so the two
  // cannot drift apart.
};

// `head` is the Detect tensor: [out_channels, cells], channels fastest, cells
// in the level-by-level order Geometry::strides/grid record. `lb` is the letterbox
// the front end recorded for this image.
std::vector<Person> decode(const Tensor &head, const Geometry &g,
                           const Letterbox &lb, const DecodeParams &p);

// The 17 COCO joint names, index 0..16, as a NUL-separated table.
const char *const *skeleton_names();

// "nose,left_eye,...,right_ankle" -- the skeleton as one line, for --json and for
// the Python facade, so the two never spell the indices differently.
std::string skeleton_names_csv();

// The 19 COCO bone edges as "a-b" pairs, in the same order for every client.
std::vector<std::string> skeleton_edges();

// One person as one JSON object, no enclosing array. Used by both the CLI and
// the server so that the two answer identically -- the point of one function
// here is that `npuimage pose` and the HTTP endpoint cannot disagree about
// what a keypoint looks like.
std::string person_json(const Person &p, int64_t id);

}  // namespace npue::pose