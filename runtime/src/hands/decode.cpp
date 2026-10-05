//===- decode.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=7 decode. See decode.hpp for the four steps and why the
// rotation is predicted rather than searched.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "hands/decode.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace npue::hands {

namespace {

// cv2's 2x3 rotation about a centre: [[cos, sin, tx], [-sin, cos, ty]].
//
// THE SIGN OF THE SECOND ROW IS NOT A CONVENTION. MediaPipe's rotation puts the
// wrist BELOW the middle-finger base, which in a frame whose y grows downward is
// the matrix written here. Its transpose is also a rotation, also by the same
// angle, and also produces 21 joints -- so the choice is caught by the
// landmarks' pixel positions and by nothing else about the output.
void rotation_matrix(double cx, double cy, double angle_deg, double m[6]) {
  const double a = angle_deg * M_PI / 180.0;
  const double ca = std::cos(a), sa = std::sin(a);
  m[0] = ca;   m[1] = sa;   m[2] = cx - ca * cx - sa * cy;
  m[3] = -sa;  m[4] = ca;   m[5] = cy + sa * cx - ca * cy;
}

// numpy's astype(np.int32) on a float: TRUNCATION toward zero. Then the clip,
// in that order -- see crop_pad.
inline int32_t trunc_i32(double v) {
  return static_cast<int32_t>(v < 0.0 ? std::ceil(v) : std::floor(v));
}

struct CropPad {
  Image img;
  double bbox[4] = {0, 0, 0, 0};   // the CLIPPED rectangle, unmoved by the pad
  double bias[2] = {0, 0};         // frame pixels the padding and crop moved by
};

// The demo's crop: shift by a fraction of the box, enlarge about the shifted
// centre, truncate and clip to the frame, then square.
//
// SIDE IS THE DIAGONAL FOR THE ROTATION CROP AND THE LONG SIDE FOR THE OTHER,
// and that asymmetry is not a simplification. The rotation crop is squared by
// its diagonal because a hand is rotated about its centre inside it and a
// long-side square clips the fingertips off a wide palm -- so the crop handed to
// the rotation has to hold the whole swept shape. The second crop is a tight box
// around the already-rotated keypoints, where the long side is the right square.
CropPad crop_pad(const Image &src, const double bbox_in[4], bool for_rotation,
                 const Crop &geom) {
  double bbox[4] = {bbox_in[0], bbox_in[1], bbox_in[2], bbox_in[3]};
  const double wh0[2] = {bbox[2] - bbox[0], bbox[3] - bbox[1]};
  // The shift moves BOTH CORNERS of the box, not just the low one. The demo
  // writes it as `bbox + shift * wh` on a (2, 2) array, and a (2,) added to a
  // (2, 2) broadcasts along the LAST axis -- so x1 and x2 both move by
  // shift_x * w and y1 and y2 both move by shift_y * h, and the box's SIZE is
  // unchanged. Only the centre moves, and the shift's job is to move the centre
  // down the palm.
  //
  // Applying it to the low corner alone -- which is what a loop over `k` in
  // {0, 1} writes naturally, since bbox[0] and bbox[1] ARE the low corners --
  // GROWS the box instead of moving it, by |shift| * size on the shifted axis.
  // With this container's shift of [0, -0.4] the second crop's height grows
  // from 206 to 289 pixels, the enlarge-then-clip step therefore runs off both
  // ends of the 729-pixel raster, and the crop comes back pinned to the frame:
  // the landmark network is shown a hand at 1.4x its intended scale with its
  // fingertips cut off, and returns 21 joints that are all slightly wrong. The
  // symptom is uniform -- no landmark is wild, the whole hand is small -- which
  // is why it is worth a comment rather than a test that only checks the worst
  // joint.
  for (int j = 0; j < 2; ++j)      // j: which corner, 0 = low, 1 = high
    for (int k = 0; k < 2; ++k)    // k: which axis
      bbox[2 * j + k] += geom.shift[k] * wh0[k];
  const double ctr[2] = {(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5};
  const double wh[2] = {bbox[2] - bbox[0], bbox[3] - bbox[1]};
  for (int k = 0; k < 2; ++k) {
    const double half = wh[k] * geom.enlarge * 0.5;
    bbox[k] = ctr[k] - half;
    bbox[2 + k] = ctr[k] + half;
  }
  // x clips to the WIDTH and y to the HEIGHT, which is what the demo's
  // [0, 0] .. [shape[1], shape[0]] bound says. Swapping the two is invisible on
  // a square frame and clips the whole right edge of a landscape one.
  const int64_t hi[2] = {src.width, src.height};
  int64_t q[4];
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      q[2 * k + j] = std::min<int64_t>(
          std::max<int64_t>(trunc_i32(bbox[2 * k + j]), 0), hi[j]);
  const int64_t cw = q[2] - q[0], ch = q[3] - q[1];
  if (cw <= 0 || ch <= 0)
    throw std::runtime_error(
        "the palm crop came out " + std::to_string(cw) + "x" +
        std::to_string(ch) + " after enlarging by " +
        std::to_string(geom.enlarge) +
        " and clipping to the " + std::to_string(src.width) + "x" +
        std::to_string(src.height) +
        " frame. That is a DETECTION problem rather than a crop one: the palm "
        "detector's largest anchor is 6 cells across at level 1, and a box that "
        "survives NMS is not supposed to be bigger than the picture or to sit "
        "entirely outside it.");
  Image sub = crop(src, q[0], q[1], q[2], q[3]);

  const int64_t side =
      for_rotation ? static_cast<int64_t>(std::sqrt(
                        static_cast<double>(ch) * ch +
                        static_cast<double>(cw) * cw))
                   : std::max(ch, cw);
  const int64_t ph = side - ch, pw = side - cw;
  const int64_t left = pw / 2, top = ph / 2;
  Image out;
  out.height = side;
  out.width = side;
  out.rgb.assign(static_cast<size_t>(side * side * 3), 0);
  for (int64_t y = 0; y < ch; ++y)
    for (int64_t x = 0; x < cw; ++x)
      std::copy_n(sub.at(y, x), 3, out.at(y + top, x + left));

  CropPad r;
  r.img = std::move(out);
  r.bias[0] = static_cast<double>(q[0]) - static_cast<double>(left);
  r.bias[1] = static_cast<double>(q[1]) - static_cast<double>(top);
  // THE RETURNED BOX IS THE CLIPPED ONE, UNMOVED BY THE PADDING. The caller
  // decides what to do about the padding: the FIRST crop subtracts `bias` to get
  // into padded-crop coordinates, and the SECOND crop discards the bias
  // entirely and reports the box as it stands in the rotated crop -- which is
  // the demo's asymmetry, and it is visible: it makes the second crop's
  // reported box SMALLER than the square it was padded into, and `sc = wh /
  // lm_size` in the postprocess is then that smaller scale.
  //
  // Returning the padded box here instead -- [left, top, left+cw, top+ch],
  // which is the same rectangle moved into the raster's coordinates and is the
  // more "obviously correct" of the two readings -- shifts the rotation's
  // centre by exactly the padding (104, 108) pixels on this frame. The
  // landmarks then land a hand's width off and everything about the OUTPUT
  // still looks like a hand: 21 joints, plausible depths, a sensible box, and
  // a score the detector was happy with. The rotation's centre is only
  // observable through the landmarks, so it is worth stating which of the two
  // conventions is the one.
  for (int k = 0; k < 4; ++k) r.bbox[k] = static_cast<double>(q[k]);
  return r;
}

}  // namespace

PalmStage palm_stage(const Image &img, const Geometry &g) {
  PalmStage p;
  p.padded = letterbox(img, g.palm_input_size, &p.lb);
  return p;
}

std::vector<Detection> decode_palm(const Geometry &g, const PalmOut &head,
                                   const Letterbox &lb,
                                   double orig_long_side) {
  if (head.n != g.palm_num_anchors)
    throw std::runtime_error("decode_palm: the head produced " +
                             std::to_string(head.n) +
                             " rows and the container declares " +
                             std::to_string(g.palm_num_anchors));
  const int64_t n = head.n;
  const double inv = 1.0 / static_cast<double>(g.palm_input_size);
  std::vector<double> bx(static_cast<size_t>(n) * 4);
  std::vector<double> sc(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double *row = &head.boxes[static_cast<size_t>(i) * g.palm_row_terms];
    const double *A = &g.anchors[static_cast<size_t>(i) * 2];
    // The score is a LOGIT: the graph's last node is the sigmoid's input and the
    // zoo applies 1/(1+exp(-x)) HERE, not in the graph. A head that folded the
    // sigmoid in would then be double-activated, and a score of 0.89 becomes
    // 0.71 -- a detector that still finds hands and ranks them slightly wrong.
    sc[static_cast<size_t>(i)] =
        1.0 / (1.0 + std::exp(-head.scores[static_cast<size_t>(i)]));
    // The anchor is the box's CENTRE, so it is added after the half-width is
    // subtracted. Treating it as a corner is the other common SSD convention
    // and it doubles the box's position error.
    const double cx = row[0] * inv * orig_long_side + A[0] * orig_long_side;
    const double cy = row[1] * inv * orig_long_side + A[1] * orig_long_side;
    const double bw = row[2] * inv * orig_long_side;
    const double bh = row[3] * inv * orig_long_side;
    bx[static_cast<size_t>(i) * 4 + 0] = cx - bw * 0.5 - lb.pad_orig_x;
    bx[static_cast<size_t>(i) * 4 + 1] = cy - bh * 0.5 - lb.pad_orig_y;
    bx[static_cast<size_t>(i) * 4 + 2] = cx + bw * 0.5 - lb.pad_orig_x;
    bx[static_cast<size_t>(i) * 4 + 3] = cy + bh * 0.5 - lb.pad_orig_y;
  }

  // NMS: greedy, on the score order, dropping anything at IoU above the
  // threshold with the survivor. ONE box is kept unconditionally even at the end
  // of the list -- that is the demo's loop and it matters only in the
  // single-detection case, where dropping it would return nothing.
  std::vector<int64_t> order(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](int64_t a, int64_t b) {
                     return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)];
                   });
  std::vector<int64_t> keep;
  keep.reserve(16);
  while (!order.empty() && static_cast<int64_t>(keep.size()) < g.top_k) {
    const int64_t i = order.front();
    if (sc[static_cast<size_t>(i)] < g.score_threshold) break;
    keep.push_back(i);
    order.erase(order.begin());   // O(n) on a 2016-entry list, once per survivor
    if (order.empty()) break;
    const double *bi = &bx[static_cast<size_t>(i) * 4];
    const double ai = (bi[2] - bi[0]) * (bi[3] - bi[1]);
    std::vector<int64_t> rest;
    rest.reserve(order.size());
    for (int64_t j : order) {
      const double *bj = &bx[static_cast<size_t>(j) * 4];
      const double ix =
          std::max(0.0, std::min(bi[2], bj[2]) - std::max(bi[0], bj[0]));
      const double iy =
          std::max(0.0, std::min(bi[3], bj[3]) - std::max(bi[1], bj[1]));
      const double inter = ix * iy;
      const double aj = (bj[2] - bj[0]) * (bj[3] - bj[1]);
      const double den = ai + aj - inter;
      // A zero-area box against another zero-area box gives 0/0, and the demo's
      // comparison is then FALSE, so the candidate is DROPPED rather than kept.
      // Mirroring that rather than substituting an IoU of zero matters only for
      // degenerate boxes, and the two differ in WHICH of them survives -- so it
      // is a choice, and it is recorded as the demo's.
      if (den > 0.0 && inter / den <= g.nms_threshold) rest.push_back(j);
    }
    order.swap(rest);
  }

  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int64_t i : keep) {
    Detection d;
    d.score = sc[static_cast<size_t>(i)];
    for (int k = 0; k < 4; ++k) d.box[k] = bx[static_cast<size_t>(i) * 4 + k];
    const double *row = &head.boxes[static_cast<size_t>(i) * g.palm_row_terms];
    const double *A = &g.anchors[static_cast<size_t>(i) * 2];
    for (int64_t k = 0; k < kPalmLandmarks; ++k)
      for (int j = 0; j < 2; ++j)
        d.kps[k][j] =
            (row[4 + 2 * k + j] * inv + A[j]) * orig_long_side -
            (j == 0 ? lb.pad_orig_x : lb.pad_orig_y);
    out.push_back(d);
  }
  return out;
}

LandmarkStage landmark_stage(const Image &img, const Geometry &g,
                             const Detection &det) {
  LandmarkStage s;
  CropPad a = crop_pad(img, det.box, /*for_rotation=*/true, g.palm_pre);
  s.pad_bias[0] = a.bias[0];
  s.pad_bias[1] = a.bias[1];

  // Everything below is in the padded rotation crop's own pixels. The keypoints
  // move by the crop's bias along with the raster: a landmark left in frame
  // pixels would put the rotation pivot off the hand by however far the crop was
  // shifted, which on a detection near the left margin is most of the hand.
  double pl[kPalmLandmarks][2];
  for (int64_t k = 0; k < kPalmLandmarks; ++k)
    for (int j = 0; j < 2; ++j) pl[k][j] = det.kps[k][j] - s.pad_bias[j];

  // The angle that puts the wrist BELOW the middle-finger base, wrapped into
  // [-pi, pi). The wrap is not optional: without it the same angle near pi from
  // the atan2 branch and near 0 from the other differs by a full turn, and the
  // landmark network is shown a hand upside down relative to itself.
  const double p1[2] = {pl[g.palm_lm_wrist][0], pl[g.palm_lm_wrist][1]};
  const double p2[2] = {pl[g.palm_lm_middle_base][0],
                        pl[g.palm_lm_middle_base][1]};
  double rad = M_PI / 2.0 - std::atan2(-(p2[1] - p1[1]), p2[0] - p1[0]);
  rad -= 2.0 * M_PI * std::floor((rad + M_PI) / (2.0 * M_PI));
  s.angle = rad * 180.0 / M_PI;

  const double b2[4] = {a.bbox[0] - s.pad_bias[0], a.bbox[1] - s.pad_bias[1],
                        a.bbox[2] - s.pad_bias[0], a.bbox[3] - s.pad_bias[1]};
  rotation_matrix((b2[0] + b2[2]) * 0.5, (b2[1] + b2[3]) * 0.5, s.angle,
                  s.rot_m);
  Image rot = warp_affine(a.img, s.rot_m);

  // The palm keypoints under the SAME rotation -- not re-predicted. The
  // detector already said where they are and the rotation is a rigid motion, so
  // transforming them is exact and predicting them again would be a second
  // source of disagreement with no new information.
  double rb[4] = {0, 0, 0, 0};
  bool first = true;
  for (int64_t k = 0; k < kPalmLandmarks; ++k) {
    const double rx = pl[k][0] * s.rot_m[0] + pl[k][1] * s.rot_m[1] + s.rot_m[2];
    const double ry = pl[k][0] * s.rot_m[3] + pl[k][1] * s.rot_m[4] + s.rot_m[5];
    if (first) {
      rb[0] = rb[2] = rx;
      rb[1] = rb[3] = ry;
      first = false;
    } else {
      rb[0] = std::min(rb[0], rx);
      rb[1] = std::min(rb[1], ry);
      rb[2] = std::max(rb[2], rx);
      rb[3] = std::max(rb[3], ry);
    }
  }
  for (int k = 0; k < 4; ++k) s.palm_kp_box[k] = rb[k];
  CropPad b = crop_pad(rot, rb, /*for_rotation=*/false, g.palm_post);
  s.blob = resize_area(b.img, g.lm_input_size, g.lm_input_size);
  for (int k = 0; k < 4; ++k) s.rot_bbox[k] = b.bbox[k];
  return s;
}

Hand postprocess(const Geometry &g, const LmOut &lm_out,
                 const LandmarkStage &s, const Detection &det) {
  (void)det;
  // The container lists four outputs and the landmark network is not a
  // heatmap net: the argmax is already folded into the graph, so the screen
  // landmarks arrive as 63 plain numbers. Reshaping is to 21 x 3 and taking the
  // roles by POSITION, which is why the container records the count per output
  // and refuses a set that does not add up: 63 is 21 x 3 and 1 is a scalar, so
  // a transposition is a shape error here -- but a runtime that reshaped the
  // screen output as (3, 21) would be a shape error nowhere.
  if (lm_out.outs.size() != 4)
    throw std::runtime_error("postprocess: the landmark network returned " +
                             std::to_string(lm_out.outs.size()) +
                             " outputs, this architecture has four (63 screen, "
                             "1 presence, 1 handedness, 63 world)");
  const int64_t want = kHandLandmarks * 3;
  if (static_cast<int64_t>(lm_out.outs[0].size()) != want ||
      static_cast<int64_t>(lm_out.outs[3].size()) != want)
    throw std::runtime_error(
        "postprocess: the screen output is " +
        std::to_string(lm_out.outs[0].size()) + " and the world output is " +
        std::to_string(lm_out.outs[3].size()) + ", both expected " +
        std::to_string(want) + " (21 landmarks x 3)");
  if (lm_out.outs[1].size() != 1 || lm_out.outs[2].size() != 1)
    throw std::runtime_error("postprocess: presence and handedness are scalars, "
                             "not vectors");

  // One scale for x, y AND z, and it is the LARGER of the crop's two axis
  // scales. The crop is square only when the detector's keypoints are, and they
  // usually are not by a few pixels; taking the mean would shrink the x and y
  // by a fraction of a pixel per landmark and the depth by a different one, so
  // the hand comes back slightly flattened in depth and none of the individual
  // numbers look wrong.
  const double sc_x = (s.rot_bbox[2] - s.rot_bbox[0]) / g.lm_input_size;
  const double sc_y = (s.rot_bbox[3] - s.rot_bbox[1]) / g.lm_input_size;
  const double sc = std::max(sc_x, sc_y);
  const double half = g.lm_input_size / 2.0;

  const double a = s.angle * M_PI / 180.0;
  const double ca = std::cos(a), sa = std::sin(a);
  // The rotation applied to the CROP, about the origin, which is the rotation
  // to undo. Its 2x2 is the crop rotation's, and the crop rotation's was built
  // about a centre -- so this is that same matrix without the translation, and
  // the translation is undone separately below. Composing the two into one
  // matrix here would be shorter and would hide which of the two is the
  // detector's and which is the postprocess's.
  const double r00 = ca, r01 = sa, r10 = -sa, r11 = ca;
  // The inverse affine of the crop rotation, assembled by hand rather than
  // inverted numerically: the matrix is orthonormal up to float64 error, so an
  // analytic inverse is exact and a 2x2 inversion would add a rounding step
  // whose only purpose is to be different.
  const double i00 = r00, i01 = r10, i10 = r01, i11 = r11;
  const double it0 = -(i00 * s.rot_m[2] + i01 * s.rot_m[5]);
  const double it1 = -(i10 * s.rot_m[2] + i11 * s.rot_m[5]);
  const double cx = (s.rot_bbox[0] + s.rot_bbox[2]) * 0.5;
  const double cy = (s.rot_bbox[1] + s.rot_bbox[3]) * 0.5;
  const double ox = cx * i00 + cy * i01 + it0;
  const double oy = cx * i10 + cy * i11 + it1;

  Hand h;
  for (int64_t k = 0; k < kHandLandmarks; ++k) {
    const double *L = &lm_out.outs[0][static_cast<size_t>(k) * 3];
    const double *W = &lm_out.outs[3][static_cast<size_t>(k) * 3];
    const double lx = (L[0] - half) * sc;
    const double ly = (L[1] - half) * sc;
    const double lz = L[2] * sc;
    const double wx = W[0], wy = W[1], wz = W[2];
    // Rotate the crop's own coordinates by the same angle, and then take the
    // rotation's inverse plus the frame offset. The world points take the
    // ROTATION but not the translation: their origin is the wrist, in metres,
    // and adding a pixel offset to them would put a hand-sized translation on a
    // measurement whose scale is metres.
    h.landmarks[k][0] = lx * r00 + ly * r10 + ox + s.pad_bias[0];
    h.landmarks[k][1] = lx * r01 + ly * r11 + oy + s.pad_bias[1];
    h.landmarks[k][2] = lz;
    h.world[k][0] = wx * r00 + wy * r10;
    h.world[k][1] = wx * r01 + wy * r11;
    h.world[k][2] = wz;
  }
  h.presence = lm_out.outs[1][0];
  h.handedness = lm_out.outs[2][0];

  // The reported box is fitted to the LANDMARKS, not to the detection: it is
  // the hand as the network sees it, which is a different rectangle from the
  // palm detector's -- larger, and shifted down the wrist. Then it is shifted
  // and enlarged by the container's own factors, which is the demo's way of
  // turning a tight bounding box into the box a caller wants to draw.
  double lo[2] = {h.landmarks[0][0], h.landmarks[0][1]};
  double hi[2] = {lo[0], lo[1]};
  for (int64_t k = 1; k < kHandLandmarks; ++k)
    for (int j = 0; j < 2; ++j) {
      lo[j] = std::min(lo[j], h.landmarks[k][j]);
      hi[j] = std::max(hi[j], h.landmarks[k][j]);
    }
  double b[2] = {lo[0], lo[1]}, e[2] = {hi[0], hi[1]};
  const double wh[2] = {e[0] - b[0], e[1] - b[1]};
  for (int j = 0; j < 2; ++j) b[j] += g.hand.shift[j] * wh[j];
  for (int j = 0; j < 2; ++j) e[j] = b[j] + wh[j];
  const double en[2] = {(e[0] - b[0]) * g.hand.enlarge * 0.5,
                        (e[1] - b[1]) * g.hand.enlarge * 0.5};
  const double ctr[2] = {(b[0] + e[0]) * 0.5, (b[1] + e[1]) * 0.5};
  for (int j = 0; j < 2; ++j) {
    h.bbox[j] = ctr[j] - en[j];
    h.bbox[2 + j] = ctr[j] + en[j];
  }
  return h;
}

}  // namespace npue::hands
