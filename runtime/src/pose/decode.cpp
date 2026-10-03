//===- decode.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 head decode. See decode.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "pose/decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace npue::pose {

namespace {

// The COCO 17-point skeleton. Index is the head tensor's keypoint index, and it
// is FIXED: these seventeen names in this order are what every COCO pose
// checkpoint was trained to emit, which is why read_geometry refuses a
// num_keypoints other than 17 rather than accepting a different number and
// naming it with these.
const char *const kNames[] = {
    "nose",     "left_eye",  "right_eye", "left_ear",   "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"};
static_assert(sizeof(kNames) / sizeof(kNames[0]) == 17, "COCO is 17 points");

// The 19 bones, as COCO defines them. order matters only in that it is the same
// order for every client -- a client that draws them by name pairs is not
// affected, and one that indexes them is affected by agreement rather than by
// which order this happens to be.
const char *const kEdgeNames[] = {
    "left_shoulder-left_elbow", "left_elbow-left_wrist",
    "left_shoulder-left_hip",   "left_hip-left_knee",
    "left_knee-left_ankle",     "right_shoulder-right_elbow",
    "right_elbow-right_wrist",  "right_shoulder-right_hip",
    "right_hip-right_knee",     "right_knee-right_ankle",
    "left_shoulder-right_shoulder", "left_hip-right_hip"};
static_assert(sizeof(kEdgeNames) / sizeof(kEdgeNames[0]) ==
                  Skeleton::kEdges,
              "the edge table and Skeleton::kEdges disagree, so a client "
              "iterating kEdges would index past the table");

// The letterbox inverse: source = (model - pad) / scale, clipped to the source.
//
// The ORDER of those two operations is the whole function. The front end built
// the canvas as
//
//     canvas = source * scale + pad
//
// so the inverse subtracts the pad in CANVAS pixels and only then undoes the
// scale. Dividing first and subtracting after gives
//
//     v / scale - pad    instead of    (v - pad) / scale
//
// which differs by exactly pad * (1/scale - 1) -- 55 pixels on an 810x1080
// source letterboxed to 640, since pad_x is 80 and scale is 0.5926. And the
// error is not visible on a square image, where there is no pad, nor on y for a
// portrait one, where pad_y happens to be 0: it appears only on the axis that
// was padded. A detector whose every box is the right size and the right place
// on one axis, and uniformly displaced on the other, is a letterbox inverse
// that has the two operations in the wrong order.
//
// `src_len` is the SOURCE extent in pixels, and clipping against it is the
// point: a model coordinate that falls inside the letterbox PAD maps to a
// negative source pixel, and a joint reported at x = -83 is the pad showing
// through, not an estimate of a limb. So the clip is in SOURCE coordinates
// against the source's own width -- not against the pad offset, which would
// collapse the whole image onto x = 0.
inline double invert(double v, const Letterbox &lb, int64_t src_len) {
  const double s = (v - static_cast<double>(lb.pad_x)) / lb.scale;
  return std::min(std::max(s, 0.0), static_cast<double>(src_len) - 1.0);
}
inline double invert_y(double v, const Letterbox &lb, int64_t src_len) {
  const double s = (v - static_cast<double>(lb.pad_y)) / lb.scale;
  return std::min(std::max(s, 0.0), static_cast<double>(src_len) - 1.0);
}

// Intersection over union of two xyxy boxes. Straight, and the boxes are few
// (at most max_det * survivors), so the O(n^2) sweep is not the cost here --
// reading the head tensor is.
inline double iou(const Person &a, const Person &b) {
  const double w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const double h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0.0 || h <= 0.0) return 0.0;
  const double inter = w * h;
  const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  const double uni = area_a + area_b - inter;
  // Two degenerate boxes (zero area, from a keypoint clipped onto the margin)
  // give uni == 0. IoU is undefined there and the only non-NaN answer that does
  // not depend on the order of the two boxes is 0, i.e. "they do not suppress
  // each other" -- which is also the honest one: a zero-area box is not
  // evidence that anything overlaps it.
  return uni > 0.0 ? inter / uni : 0.0;
}

}  // namespace

const char *Skeleton::name(int i) {
  if (i < 0 || i >= kPoints)
    throw std::runtime_error("skeleton index " + std::to_string(i) +
                             " is outside 0..16");
  return kNames[i];
}

bool Skeleton::pair(int edge, int &a, int &b) {
  if (edge < 0 || edge >= Skeleton::kEdges) return false;
  const std::string e = kEdgeNames[edge];
  const size_t dash = e.find('-');
  for (int j = 0; j < kPoints; ++j) {
    if (kNames[j] != e.substr(dash + 1)) continue;
    for (int i = 0; i < kPoints; ++i) {
      if (e.compare(0, dash, kNames[i], dash) != 0) continue;
      a = i;
      b = j;
      return true;
    }
  }
  return false;
}

const char *const *skeleton_names() { return kNames; }

std::string skeleton_names_csv() {
  std::string s;
  for (int i = 0; i < 17; ++i) {
    if (i) s += ',';
    s += kNames[i];
  }
  return s;
}

std::vector<std::string> skeleton_edges() {
  std::vector<std::string> v;
  v.reserve(Skeleton::kEdges);
  for (int i = 0; i < Skeleton::kEdges; ++i)
    v.emplace_back(kEdgeNames[i]);
  return v;
}

std::vector<Person> decode(const Tensor &head, const Geometry &g,
                           const Letterbox &lb, const DecodeParams &p) {
  const int64_t C = g.out_channels();      // 4 + nc + nk*3
  const int64_t cells = g.n_anchors();
  if (head.c != C || head.h * head.w != cells)
    throw std::runtime_error(
        "pose decode: the head tensor is [" + std::to_string(head.c) + ", " +
        std::to_string(head.h) + ", " + std::to_string(head.w) +
        "] and the geometry describes " + std::to_string(C) + " values for " +
        std::to_string(cells) + " cells. read_geometry checked this pair "
        "already, so reaching here means the tensor and the geometry were built "
        "from different containers.");

  // The class score sits at index 4 -- after the four box values -- and the
  // keypoints after it. That order is the architecture's, and read_geometry
  // refuses a detect_cout that does not agree with it, so the indices below are
  // checked rather than trusted.
  const int64_t kCls = 4;
  const int64_t kKpt0 = 4 + g.num_classes;

  // Everything read below is CANONICAL: xyxy corners in letterbox pixels, the
  // score and the visibility already probabilities, keypoints in letterbox
  // pixels. The head (Network::head) is what makes it so, and the point of
  // putting the exporter's conventions there and not here is that this function
  // has nothing to be wrong about: there is no "which convention was this
  // exporter's" branch left in the reader, so there is no way for the packer and
  // the reader to disagree about one.

  std::vector<Person> cand;
  cand.reserve(256);

  int64_t offset = 0;
  for (int64_t lv = 0; lv < g.n_levels(); ++lv) {
    const int64_t side = g.grid[static_cast<size_t>(lv)];
    for (int64_t cell = 0; cell < side * side; ++cell, ++offset) {
      // The canonical tensor is [C, cells] -- channels first, because Network::head
      // transposes into it -- so one cell's C values are STRIDED by `cells`, not
      // adjacent. `at(ch)` is that gather and every read below goes through it.
      // Treating the tensor as [cells, C] because that is the order the head's
      // loop produces the rows in is a bug that survives every shape check:
      // nothing about the tensor's extents changes.
      const auto at = [&](int64_t ch) -> float {
        return head.chw(ch)[static_cast<size_t>(offset)];
      };
      const double score = at(kCls);
      if (score < p.conf) continue;
      if (static_cast<int64_t>(cand.size()) >= p.max_nms) break;

      Person per;
      per.score = score;
      per.anchor = offset;
      per.level = lv;

      // xyxy, already in letterbox pixels.
      per.x1 = invert(at(0), lb, lb.src_w);
      per.y1 = invert_y(at(1), lb, lb.src_h);
      per.x2 = invert(at(2), lb, lb.src_w);
      per.y2 = invert_y(at(3), lb, lb.src_h);

      per.keypoints.reserve(static_cast<size_t>(g.num_keypoints));
      for (int64_t k = 0; k < g.num_keypoints; ++k) {
        Keypoint pt;
        pt.index = static_cast<int>(k);
        pt.v = at(kKpt0 + k * 3 + 2);
        pt.x = invert(at(kKpt0 + k * 3 + 0), lb, lb.src_w);
        pt.y = invert_y(at(kKpt0 + k * 3 + 1), lb, lb.src_h);
        pt.visible = pt.v >= p.keypoint;
        per.keypoints.push_back(pt);
      }
      cand.push_back(std::move(per));
    }
  }

  // -- NMS, greedy by score ------------------------------------------------
  //
  // Score-descending greedy suppression, which is what every detector here
  // means by NMS. The sort is stable on the score so that two anchors with the
  // same score suppress each other in ANCHOR order rather than in whatever
  // order std::sort happened to produce -- an unstable sort would make the
  // returned list differ run to run on a tie, and a pose client that shows
  // "person 1 of 2" would flicker between the two.
  std::stable_sort(cand.begin(), cand.end(),
                   [](const Person &a, const Person &b) { return a.score > b.score; });

  std::vector<Person> kept;
  kept.reserve(std::min<int64_t>(p.max_det, static_cast<int64_t>(cand.size())));
  std::vector<bool> dead(cand.size(), false);
  for (size_t i = 0; i < cand.size() && static_cast<int64_t>(kept.size()) < p.max_det;
       ++i) {
    if (dead[i]) continue;
    kept.push_back(cand[i]);
    for (size_t j = i + 1; j < cand.size(); ++j) {
      if (dead[j]) continue;
      if (iou(cand[i], cand[j]) > p.iou) dead[j] = true;
    }
  }
  return kept;
}

std::string person_json(const Person &p, int64_t id) {
  // %.1f on a pixel coordinate: at 640px the letterbox's own quantisation is
  // already one model pixel, so a tenth of a source pixel is finer than the
  // input and saying more of it would be a claim about a precision the model
  // does not have. The score keeps three decimals, which is finer than the
  // threshold resolution a caller can act on.
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"id\":%lld,\"score\":%.3f,\"box\":{\"x1\":%.1f,\"y1\":%.1f,"
                "\"x2\":%.1f,\"y2\":%.1f},\"level\":%lld,\"anchor\":%lld,"
                "\"keypoints\":[",
                static_cast<long long>(id), p.score, p.x1, p.y1, p.x2, p.y2,
                static_cast<long long>(p.level), static_cast<long long>(p.anchor));
  std::string s = buf;
  for (size_t i = 0; i < p.keypoints.size(); ++i) {
    const Keypoint &k = p.keypoints[i];
    if (i) s += ',';
    std::snprintf(buf, sizeof(buf),
                  "{\"x\":%.1f,\"y\":%.1f,\"score\":%.3f,\"visible\":%s,"
                  "\"index\":%d,\"name\":\"%s\"}",
                  k.x, k.y, k.v, k.visible ? "true" : "false", k.index,
                  Skeleton::name(k.index));
    s += buf;
  }
  s += "]}";
  return s;
}

}  // namespace npue::pose