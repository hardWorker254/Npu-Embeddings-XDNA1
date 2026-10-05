//===- decode.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=8 decode. See decode.hpp for the four steps and for why
// the rotation is between two predicted points rather than between the box and the
// axes.
//
// EVERY INTEGER CONVERSION HERE IS numpy's, NOT C++'s
// -----------------------------------------------------
// The zoo's region-of-interest arithmetic is four `astype(np.int32)` calls and two
// `.astype(np.int32)` on a pad offset, and numpy's float-to-int cast truncates
// TOWARD ZERO -- which is not C++'s conversion, which truncates toward zero only
// for the same reason but whose static_cast<double>(v) on a negative value is a
// trap when written as an integer literal first. trunc_i32() below is the one
// definition and it is called at every one of those sites.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/decode.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace npue::mppose {

namespace {

// numpy's astype(np.int32) on a float: truncation toward zero. NOT floor: the
// region-of-interest square's left edge is negative on a person near the margin,
// and floor there moves the crop a whole pixel further out than the zoo does.
inline int64_t trunc_i32(double v) {
  return static_cast<int64_t>(v < 0.0 ? std::ceil(v) : std::floor(v));
}

// cv2's 2x3 rotation about a centre: [[alpha, beta, tx], [-beta, alpha, ty]] with
// alpha = cos(angle), beta = sin(angle).
//
// THE SIGN OF THE SECOND ROW IS NOT A CONVENTION. MediaPipe's rotation puts the
// full-body point ABOVE the hip, which in a frame whose y grows downward is the
// matrix written here. Its transpose is also a rotation, also by the same angle,
// and also produces 39 rows -- so the choice is caught by the landmarks' pixel
// positions and by nothing else about the output.
void rotation_matrix(double cx, double cy, double angle_deg, double m[6]) {
  const double a = angle_deg * M_PI / 180.0;
  const double ca = std::cos(a), sa = std::sin(a);
  m[0] = ca;   m[1] = sa;   m[2] = cx - ca * cx - sa * cy;
  m[3] = -sa;  m[4] = ca;   m[5] = cy + sa * cx - ca * cy;
}

}  // namespace

// -- stage 1, before the detector ---------------------------------------------

DetStage det_stage(const Image &img, const Geometry &g) {
  DetStage s;
  s.blob = det_input(img, g, &s.lb);
  return s;
}

// -- stage 1, after the detector ----------------------------------------------

std::vector<Detection> decode_det(const Geometry &g, const DetOut &head,
                                  const DetLetterbox &lb, int64_t img_w,
                                  int64_t img_h) {
  if (head.n != g.det_num_anchors)
    throw std::runtime_error("decode_det: the head produced " +
                             std::to_string(head.n) +
                             " rows and the container declares " +
                             std::to_string(g.det_num_anchors));
  const int64_t n = head.n;
  const double inv = 1.0 / static_cast<double>(g.det_input_size);
  // The zoo's `scale = max(original_shape)` where original_shape is [w, h]: the
  // LONG side, not the width. On a portrait frame that is the height, and using
  // the width instead scales every box by 1080/810 -- a factor of 1.33 on every
  // detection, with the shape still right.
  const double scale = static_cast<double>(std::max(img_w, img_h));
  std::vector<double> bx(static_cast<size_t>(n) * 4);
  std::vector<double> sc(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double *row = &head.boxes[static_cast<size_t>(i) * g.det_row_terms];
    const double *A = &g.anchors[static_cast<size_t>(i) * 2];
    // The score is a LOGIT: the graph's last node before this head is the
    // classifier's 1x1 convolution and the zoo applies 1/(1+exp(-x)) HERE, not in
    // the graph. A head that folded the sigmoid in would be double-activated, and a
    // score of 0.89 becomes 0.71 -- a detector that still finds people and ranks
    // them slightly wrong.
    //
    // The clip to [-100, 100] is the zoo's too, and it is not cosmetic: exp(100)
    // overflows a float64 to infinity and an infinite score is neither above nor
    // below the threshold in a way anybody chose.
    const double raw = std::max(-100.0, std::min(100.0, head.scores[static_cast<size_t>(i)]));
    sc[static_cast<size_t>(i)] = 1.0 / (1.0 + std::exp(-raw));
    // cx,cy,w,h against an anchor CENTRE. The row is NOT xy1,xy2: the zoo builds
    // the corners from a centre and a width, and reading the first two columns as
    // the top-left corner doubles the box's position error on every detection.
    const double cx = row[0] * inv * scale + A[0] * scale;
    const double cy = row[1] * inv * scale + A[1] * scale;
    const double bw = row[2] * inv * scale;
    const double bh = row[3] * inv * scale;
    bx[static_cast<size_t>(i) * 4 + 0] = cx - bw * 0.5 - static_cast<double>(lb.pad_bias_x);
    bx[static_cast<size_t>(i) * 4 + 1] = cy - bh * 0.5 - static_cast<double>(lb.pad_bias_y);
    bx[static_cast<size_t>(i) * 4 + 2] = cx + bw * 0.5 - static_cast<double>(lb.pad_bias_x);
    bx[static_cast<size_t>(i) * 4 + 3] = cy + bh * 0.5 - static_cast<double>(lb.pad_bias_y);
  }

  // NMS: greedy, on the score order, dropping anything at IoU above the threshold
  // with the survivor, stopping at top_k.
  //
  // The zoo calls cv.dnn.NMSBoxes, which takes [x,y,w,h] and this passes
  // [x1,y1,x2,y2]. That is not a bug in the zoo: with the corner form read as
  // x/y/w/h the AREA is w*h = (x2-x1)*(y2-y1) and the INTERSECTION is
  // min(x2a,x2b) - max(x1a,x1b), so the IoU is the same number either way. This
  // is the same greedy loop, on the same sorted order, with the same `> threshold`
  // suppression, and the top_k here is larger than the candidate count so no
  // pre-filter is involved.
  std::vector<int64_t> order(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)];
  });
  std::vector<int64_t> keep;
  keep.reserve(16);
  while (!order.empty() && static_cast<int64_t>(keep.size()) < g.top_k) {
    const int64_t i = order.front();
    if (sc[static_cast<size_t>(i)] < g.score_threshold) break;
    keep.push_back(i);
    order.erase(order.begin());
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
    const double *row = &head.boxes[static_cast<size_t>(i) * g.det_row_terms];
    const double *A = &g.anchors[static_cast<size_t>(i) * 2];
    // The four keypoints are OFFSETS from the anchor, in units of the input side,
    // and they get the same anchor and the same long-side scale the box does. They
    // are NOT corners and NOT absolute: reading them as absolute puts the
    // rotation's pivot at the frame's origin for the anchors whose offset is
    // small, which rotates the crop by an unrelated angle.
    for (int64_t k = 0; k < kDetLandmarks; ++k)
      for (int j = 0; j < 2; ++j)
        d.kps[k][j] = (row[4 + 2 * k + j] * inv + A[j]) * scale -
                      (j == 0 ? static_cast<double>(lb.pad_bias_x)
                              : static_cast<double>(lb.pad_bias_y));
    out.push_back(d);
  }
  return out;
}

// -- stage 2, before the landmark network --------------------------------------

PoseStage pose_stage(const Image &img, const Geometry &g, const Detection &det) {
  PoseStage s;
  const double hip[2] = {det.kps[g.person_lm_mid_hip][0],
                         det.kps[g.person_lm_mid_hip][1]};
  const double body[2] = {det.kps[g.person_lm_full_body][0],
                          det.kps[g.person_lm_full_body][1]};
  const double dist = std::hypot(body[0] - hip[0], body[1] - hip[1]);

  // FIRST TRUNCATION. `np.array([hip - d, hip + d], np.int32)`.
  double f0[2] = {static_cast<double>(trunc_i32(hip[0] - dist)),
                  static_cast<double>(trunc_i32(hip[1] - dist))};
  double f1[2] = {static_cast<double>(trunc_i32(hip[0] + dist)),
                  static_cast<double>(trunc_i32(hip[1] + dist))};
  // Enlarged about its own centre BY THE TRUNCATED WIDTH, which is an integer
  // number of pixels rather than the float 2d. Enlarging the exact square instead
  // is a sub-pixel difference that the truncation then rounds, and the two
  // orderings disagree by a pixel on some frames.
  const double ctr[2] = {(f0[0] + f1[0]) * 0.5, (f0[1] + f1[1]) * 0.5};
  const double wh[2] = {f1[0] - f0[0], f1[1] - f0[1]};
  const double half[2] = {wh[0] * g.person_box_pre_enlarge * 0.5,
                          wh[1] * g.person_box_pre_enlarge * 0.5};
  // SECOND TRUNCATION, and a fresh pair: from here on the square is the integer
  // one and every coordinate below is measured against it.
  f0[0] = static_cast<double>(trunc_i32(ctr[0] - half[0]));
  f0[1] = static_cast<double>(trunc_i32(ctr[1] - half[1]));
  f1[0] = static_cast<double>(trunc_i32(ctr[0] + half[0]));
  f1[1] = static_cast<double>(trunc_i32(ctr[1] + half[1]));

  // Refine: clip to the frame, PER AXIS. x clips to the WIDTH and y to the
  // HEIGHT; swapping the two is invisible on a square frame and clips a whole edge
  // of a landscape one.
  double p0[2] = {std::max(0.0, std::min(f0[0], static_cast<double>(img.width))),
                  std::max(0.0, std::min(f0[1], static_cast<double>(img.height)))};
  double p1[2] = {std::max(0.0, std::min(f1[0], static_cast<double>(img.width))),
                  std::max(0.0, std::min(f1[1], static_cast<double>(img.height)))};
  if (p1[0] <= p0[0] || p1[1] <= p0[1])
    throw std::runtime_error(
        "pose_stage: the region of interest clipped to nothing -- " +
        std::to_string(p0[0]) + "," + std::to_string(p0[1]) + " to " +
        std::to_string(p1[0]) + "," + std::to_string(p1[1]) + " in a " +
        std::to_string(img.width) + "x" + std::to_string(img.height) +
        " frame. That is a DETECTION problem rather than a crop one: the square is "
        "centred on the detector's hip point and has the distance from the hip to "
        "the full-body point as its half-side, so a square that lands entirely "
        "outside the frame means the box's four keypoints do not describe a body.");

  Image sub = npue::raster::crop(img, static_cast<int64_t>(p0[0]),
                                 static_cast<int64_t>(p0[1]),
                                 static_cast<int64_t>(p1[0]),
                                 static_cast<int64_t>(p1[1]));
  // Pad back to the unclipped square. `left, top = person_bbox[0] -
  // full_bbox[0]` and `right, bottom = full_bbox[1] - person_bbox[1]`, so the
  // result is exactly full_bbox's extent -- which is what the rest of the geometry
  // is measured against.
  const int64_t pad_left = static_cast<int64_t>(p0[0] - f0[0]);
  const int64_t pad_top = static_cast<int64_t>(p0[1] - f0[1]);
  const int64_t pad_right = static_cast<int64_t>(f1[0] - p1[0]);
  const int64_t pad_bottom = static_cast<int64_t>(f1[1] - p1[1]);
  Image square = npue::raster::pad_zeros(sub, pad_top, pad_bottom, pad_left, pad_right);
  s.square_w = square.width;
  s.square_h = square.height;

  // `pad_bias = np.array([0, 0]); pad_bias += person_bbox[0] - [left, top]` --
  // which is full_bbox[0] again, and it is an INTEGER and it may be NEGATIVE when
  // the square reaches past the frame's edge. Both of those matter: the landmarks
  // are offset by it in frame pixels and the mask is placed at it.
  s.pad_bias[0] = f0[0];
  s.pad_bias[1] = f0[1];

  // The rotation reads the two points in the SQUARE's own coordinates. The zoo
  // mutates its copies with `-= pad_bias`; they are locals here, so the subtraction
  // is written out rather than being a side effect on the caller's detection.
  const double hx = hip[0] - s.pad_bias[0], hy = hip[1] - s.pad_bias[1];
  const double bx = body[0] - s.pad_bias[0], by = body[1] - s.pad_bias[1];

  // The angle that puts the full-body point ABOVE the hip, wrapped into
  // (-pi, pi]. The wrap is not optional: without it the same angle near pi from the
  // atan2 branch and near 0 from the other differs by a full turn and the network
  // is shown an upside-down person.
  double rad = M_PI / 2.0 - std::atan2(-(by - hy), bx - hx);
  rad -= 2.0 * M_PI * std::floor((rad + M_PI) / (2.0 * M_PI));
  s.angle = rad * 180.0 / M_PI;

  rotation_matrix(hx, hy, s.angle, s.rot_m);
  Image rotated = npue::raster::warp_affine(square, s.rot_m);
  // INTER_AREA, an exact box average, and not bilinear: the square is normally
  // several times larger than 256 and bilinear there moves every landmark by a
  // pixel and a half.
  s.blob = npue::raster::resize_area(rotated, g.pose_input_size, g.pose_input_size);
  return s;
}

// -- stage 2, after the landmark network ----------------------------------------

Person postprocess(const Geometry &g, const PoseOuts &out, const PoseStage &s,
                   int64_t img_w, int64_t img_h) {
  const size_t nl = static_cast<size_t>(g.num_landmarks);
  const size_t want_lm = nl * static_cast<size_t>(g.lm_cols);
  const size_t want_w = nl * static_cast<size_t>(g.world_cols);
  if (out.landmarks.size() != want_lm || out.world.size() != want_w)
    throw std::runtime_error(
        "postprocess: the screen output is " + std::to_string(out.landmarks.size()) +
        " and the world output is " + std::to_string(out.world.size()) + ", " +
        "expected " + std::to_string(want_lm) + " (" + std::to_string(nl) +
        "x" + std::to_string(g.lm_cols) + ") and " + std::to_string(want_w) + " (" +
        std::to_string(nl) + "x" + std::to_string(g.world_cols) + ")");

  const double in_sz = static_cast<double>(g.pose_input_size);
  const double half = in_sz / 2.0;
  // One scale PER AXIS for x and y, and the LARGER of the two for z. The square is
  // not exactly square -- two truncations are between it and a square -- so taking
  // one scale for all three flattens the depth or the plane by a fraction of a
  // pixel per landmark, and none of the individual numbers look wrong.
  const double sf_x = static_cast<double>(s.square_w) / in_sz;
  const double sf_y = static_cast<double>(s.square_h) / in_sz;
  const double sf_z = std::max(sf_x, sf_y);

  Person p;
  p.conf = out.conf;

  // The rotation applied to the CROP, about the origin -- which is the rotation to
  // undo on the way out. Its 2x2 is the crop rotation's own, built about a centre,
  // so this is that matrix without the translation and the translation is undone
  // separately below.
  const double a = s.angle * M_PI / 180.0;
  const double ca = std::cos(a), sa = std::sin(a);
  // `coords_rotation_matrix[:, :2]` is [[cos, sin], [-sin, cos]] and the zoo writes
  // `landmarks[:, :2] @ that`, i.e. out.x = x*cos - y*sin and out.y = x*sin + y*cos.
  const double m00 = ca, m01 = -sa, m10 = sa, m11 = ca;

  // The inverse of the crop's rotation, assembled by hand rather than inverted
  // numerically: the matrix is orthonormal up to float64 error, so an analytic
  // inverse is exact and a 2x2 inversion would add a rounding step whose only
  // purpose is to be different.
  const double i00 = s.rot_m[0], i01 = s.rot_m[3];
  const double i10 = s.rot_m[1], i11 = s.rot_m[4];
  const double it0 = -(i00 * s.rot_m[2] + i01 * s.rot_m[5]);
  const double it1 = -(i10 * s.rot_m[2] + i11 * s.rot_m[5]);
  // The square's centre, mapped through that inverse. This is the translation the
  // landmarks need and the one number in the file that is easiest to get wrong by
  // a plausible amount.
  const double ccx = static_cast<double>(s.square_w) * 0.5;
  const double ccy = static_cast<double>(s.square_h) * 0.5;
  const double ocx = ccx * i00 + ccy * i01 + it0;
  const double ocy = ccx * i10 + ccy * i11 + it1;

  for (size_t k = 0; k < nl; ++k) {
    const double *L = &out.landmarks[k * static_cast<size_t>(g.lm_cols)];
    const double *W = &out.world[k * static_cast<size_t>(g.world_cols)];
    const double lx = (L[0] - half) * sf_x;
    const double ly = (L[1] - half) * sf_y;
    const double lz = L[2] * sf_z;
    // The visibility and presence columns ARE squashed, and only those two. The
    // zoo applies the sigmoid to columns three and four of the 5, and squashing
    // the coordinates as well would turn a pose into noise with every shape right.
    p.landmarks[k][0] = lx * m00 + ly * m01 + ocx + s.pad_bias[0];
    p.landmarks[k][1] = lx * m10 + ly * m11 + ocy + s.pad_bias[1];
    p.landmarks[k][2] = lz;
    p.landmarks[k][3] = 1.0 / (1.0 + std::exp(-L[3]));
    p.landmarks[k][4] = 1.0 / (1.0 + std::exp(-L[4]));
    // The world points take the ROTATION and not the translation: their origin is
    // the hip, in metres, and adding a pixel offset would put a person-sized
    // translation on a measurement whose scale is metres.
    p.world[k][0] = W[0] * m00 + W[1] * m01;
    p.world[k][1] = W[0] * m10 + W[1] * m11;
    p.world[k][2] = W[2];
  }

  // The reported box is fitted to the LANDMARKS -- all thirty-nine of them, the six
  // auxiliary rows included, which is the zoo's amin/amax over the whole array --
  // then enlarged about its own centre by the container's factor. It is not the
  // detector's box.
  double lo[2] = {p.landmarks[0][0], p.landmarks[0][1]};
  double hi[2] = {lo[0], lo[1]};
  for (size_t k = 1; k < nl; ++k)
    for (int j = 0; j < 2; ++j) {
      lo[j] = std::min(lo[j], p.landmarks[k][j]);
      hi[j] = std::max(hi[j], p.landmarks[k][j]);
    }
  const double bctr[2] = {(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5};
  const double bhalf[2] = {(hi[0] - lo[0]) * g.person_box_enlarge * 0.5,
                           (hi[1] - lo[1]) * g.person_box_enlarge * 0.5};
  for (int j = 0; j < 2; ++j) {
    p.bbox[j] = bctr[j] - bhalf[j];
    p.bbox[2 + j] = bctr[j] + bhalf[j];
  }

  // -- the mask, back into the frame ------------------------------------------
  //
  // Four steps, all of them silent: a rotation by -angle about the 256 square's
  // centre, a resize to the square's size, a crop of whatever hangs outside the
  // frame, and a pad that places the result at the square's offset.
  const int64_t ms = g.pose_input_size;
  if (static_cast<int64_t>(out.mask.size()) != ms * ms)
    throw std::runtime_error("postprocess: the mask is " +
                             std::to_string(out.mask.size()) + " values and a " +
                             std::to_string(ms) + "x" + std::to_string(ms) +
                             " mask is " + std::to_string(ms * ms));
  double inv[6];
  rotation_matrix(static_cast<double>(ms) / 2.0, static_cast<double>(ms) / 2.0,
                  -s.angle, inv);
  std::vector<float> warped(static_cast<size_t>(ms) * static_cast<size_t>(ms));
  npue::raster::warp_affine_f32(out.mask.data(), ms, ms, warped.data(), ms, ms,
                                inv, 1);
  // cv2.resize with NO interpolation argument, which is INTER_LINEAR -- not the
  // INTER_AREA the crop itself used. Using area here would be a "better" resample
  // of a different checkpoint's front end.
  std::vector<float> placed(static_cast<size_t>(s.square_w) *
                            static_cast<size_t>(s.square_h));
  npue::raster::resize_bilinear_f32(warped.data(), ms, ms, placed.data(),
                                    s.square_h, s.square_w, 1);

  // The crop is whatever lies inside the frame and the pad is the rest. pad_bias
  // may be NEGATIVE, in which case there is no left pad and the square is cropped
  // there instead -- which is why the low and high bounds are separate terms rather
  // than one offset.
  const double over[2] = {static_cast<double>(img_w) - static_cast<double>(s.square_w) -
                              s.pad_bias[0],
                          static_cast<double>(img_h) - static_cast<double>(s.square_h) -
                              s.pad_bias[1]};
  const int64_t lo_x = static_cast<int64_t>(-std::min(0.0, s.pad_bias[0]));
  const int64_t lo_y = static_cast<int64_t>(-std::min(0.0, s.pad_bias[1]));
  const int64_t hi_x = s.square_w + static_cast<int64_t>(std::min(0.0, over[0]));
  const int64_t hi_y = s.square_h + static_cast<int64_t>(std::min(0.0, over[1]));
  const int64_t pad_l = static_cast<int64_t>(std::max(0.0, s.pad_bias[0]));
  const int64_t pad_t = static_cast<int64_t>(std::max(0.0, s.pad_bias[1]));
  const int64_t pad_r = static_cast<int64_t>(std::max(0.0, over[0]));
  const int64_t pad_b = static_cast<int64_t>(std::max(0.0, over[1]));
  // THE COMPOSED EXTENT MUST BE THE FRAME'S, and this is where that is checked
  // rather than assumed. The zoo's crop-and-pad arithmetic guarantees it -- the
  // crop takes exactly what hangs outside and the pad takes exactly what does not
  // -- and the guarantee is worth having in the runtime: a pad_bias computed with
  // a rounding instead of a truncation, or a square built from the exact box rather
  // than the truncated one, makes the two disagree by a row or a column, and the
  // symptom would be a mask of the wrong height that nothing else reports.
  if ((hi_x - lo_x) + pad_l + pad_r != img_w ||
      (hi_y - lo_y) + pad_t + pad_b != img_h)
    throw std::runtime_error(
        "postprocess: the mask's crop and pad compose to " +
        std::to_string((hi_x - lo_x) + pad_l + pad_r) + "x" +
        std::to_string((hi_y - lo_y) + pad_t + pad_b) + " and the frame is " +
        std::to_string(img_w) + "x" + std::to_string(img_h) +
        ". pad_bias is (" + std::to_string(s.pad_bias[0]) + ", " +
        std::to_string(s.pad_bias[1]) + "), the square is " +
        std::to_string(s.square_w) + "x" + std::to_string(s.square_h) +
        ". One of those three is not what the zoo's arithmetic produces.");
  if (hi_x <= lo_x || hi_y <= lo_y)
    throw std::runtime_error(
        "postprocess: the mask's square lies entirely outside the frame -- " +
        std::to_string(lo_x) + "," + std::to_string(lo_y) + " to " +
        std::to_string(hi_x) + "," + std::to_string(hi_y) + " in a " +
        std::to_string(img_w) + "x" + std::to_string(img_h) +
        " frame. The landmarks would be reported anyway, so without this the "
        "failure is an empty mask with nothing saying why.");

  p.mask_w = img_w;
  p.mask_h = img_h;
  p.mask.assign(static_cast<size_t>(img_w) * static_cast<size_t>(img_h), 0);
  for (int64_t y = 0; y < hi_y - lo_y; ++y)
    for (int64_t x = 0; x < hi_x - lo_x; ++x) {
      // `np.where(mask > 0, 255, 0)` -- a BINARY mask. The network's mask is a raw
      // float with values well past 255 and the zoo thresholds it at zero, so the
      // result carries no magnitude at all and cannot.
      if (placed[static_cast<size_t>(y) * static_cast<size_t>(s.square_w) +
                 static_cast<size_t>(x)] > 0.f)
        p.mask[static_cast<size_t>(y + pad_t) * static_cast<size_t>(img_w) +
               static_cast<size_t>(x + pad_l)] = 255;
    }
  return p;
}

}  // namespace npue::mppose
