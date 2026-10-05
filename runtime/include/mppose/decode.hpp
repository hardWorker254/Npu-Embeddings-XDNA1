//===- decode.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=8 decode: 2254 anchor rows become boxes, then ONE box
// plus TWO of its four keypoints become a rotated 256x256 crop, then the
// landmark network's 195 + 1 + 117 numbers become 39 rows in the caller's pixels
// and a frame-sized mask.
//
// FOUR FUNCTIONS, AND THE SPLITS ARE THE NETWORKS
// ------------------------------------------------
//   det_stage     letterbox, into the detector's own normalisation
//   decode_det    anchors, sigmoid, NMS, inverse letterbox
//   pose_stage    crop, square, rotate, area-resize
//   postprocess   the rotation's inverse, the reported box, and the mask
//
// Between det_stage and decode_det the DETECTOR runs; between pose_stage and
// postprocess the LANDMARK NETWORK runs. So this header cannot hand the networks
// in -- it takes their outputs, which is what lets a caller compare the two
// stages separately.
//
// THE ROTATION IS BETWEEN TWO PREDICTED POINTS, NOT BETWEEN THE BOX AND THE AXES
// ---------------------------------------------------------------------------
// The detector's box row is cx,cy,w,h plus FOUR keypoints, and MediaPipe names
// them: a hip-centre, a full-body point, a shoulder-centre and an upper-body
// point. The region of interest is the SQUARE centred on the hip whose half-side
// is the distance from the hip to the full-body point -- so the crop is scaled by
// the person's own anatomy rather than by the detection box, and a box the
// detector made loose does not produce a loose crop. The rotation then puts the
// full-body point directly above the hip, wrapped into (-pi, pi].
//
// WHICH TWO points matters as much as anything else in this file, which is why
// the pair is read from the container rather than assumed to be (0, 1). The wrong
// pair rotates by some other angle, the network still returns 39 rows, and every
// one of them is somewhere plausible -- the failure is only visible through the
// landmarks' own positions.
//
// WHY THE SQUARE IS BUILT BY TRUNCATING, THEN ENLARGING, THEN TRUNCATING AGAIN
// ---------------------------------------------------------------------------
// The zoo does three integer conversions in sequence and the order is the model:
//   full_bbox = (int32)[hip - d, hip + d]          a truncation
//   enlarged about its own centre by person_box_pre_enlarge
//   full_bbox = (int32)[...]                        another truncation
// Each one moves the square by up to a pixel against the exact value, and the
// second one is applied to the ALREADY TRUNCATED box, so its width is an integer
// number of pixels rather than the float 2d. Rounding instead of truncating, or
// enlarging the exact box rather than the truncated one, is a sub-pixel error on
// every landmark and a full pixel on the crop's own edge.
//
// THE MASK IS THE LAST THREE STEPS AND THEY ARE NOT INTERCHANGEABLE
// ------------------------------------------------------------------
// The network emits a 256x256 mask in the ROTATED square's coordinates. It comes
// back through: a rotation by -angle about the square's centre, a resize to the
// square's size, a crop, and a pad into the frame at exactly this square's
// position. All four are needed and each is silent: a mask that is rotated the
// wrong way is still a plausible silhouette, and one that is placed at the wrong
// offset is a silhouette of the right person in the wrong place.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mppose/geometry.hpp"
#include "mppose/image.hpp"
#include "mppose/net.hpp"

namespace npue::mppose {

// One person detection, in the CALLER's pixels: the box, the four keypoints the
// region of interest and the rotation are read from, and the score.
struct Detection {
  double box[4] = {0, 0, 0, 0};          // x1, y1, x2, y2
  double kps[kDetLandmarks][2] = {};    // hip, full body, shoulder, upper body
  double score = 0.0;
};

// One person, fully decoded. `landmarks` and `world` are 39 rows.
//
// `landmarks` is in the CALLER's pixels: x and y absolute, and z a DEPTH RELATIVE
// TO THE MID-HIP in the same pixel units as x and y -- it comes out of the network
// in the crop's own units and is multiplied by the crop's larger axis scale here,
// so a z of 34 is thirty-four pixels nearer than the hip and not thirty-four
// metres. `world` is in METRES, hip-relative, and takes the rotation but not the
// translation, because its origin is the hip.
//
// The reported `bbox` is fitted to the LANDMARKS and then enlarged by the
// container's own factor. It is NOT the detector's box: that one bounds a
// detection, this one bounds a skeleton.
//
// `mask` is the frame-sized binary silhouette -- frame width x height, row-major,
// 255 for body and 0 for background, exactly as the zoo returns it.
struct Person {
  double bbox[4] = {0, 0, 0, 0};
  double landmarks[kNumLandmarks][kLmCols] = {};
  double world[kNumLandmarks][kWorldCols] = {};
  double conf = 0.0;
  int64_t mask_w = 0, mask_h = 0;
  std::vector<uint8_t> mask;
};

// -- stage 1, before the detector ---------------------------------------------

// The detector's input and the letterbox it came from. See mppose/image.hpp for
// why this normalisation is not arch=7's.
struct DetStage {
  std::vector<float> blob;    // 3 x S x S, NCHW
  DetLetterbox lb;
};

DetStage det_stage(const Image &img, const Geometry &g);

// -- stage 1, after the detector ----------------------------------------------
//
// The detections come back in DESCENDING SCORE order, which is the order NMS
// produces them in. Every person is an independent crop, so the caller picks which
// ones to run through stage 2 and there is no "best" argument here.
std::vector<Detection> decode_det(const Geometry &g, const DetOut &head,
                                  const DetLetterbox &lb, int64_t img_w,
                                  int64_t img_h);

// -- stage 2, before the landmark network --------------------------------------

// The crop and the four things postprocess needs to undo the geometry. Kept
// rather than recomputed: recomputing the rotation is re-deriving an atan2 and a
// wrap, and a half-degree disagreement there moves every landmark by its own
// distance from the hip.
struct PoseStage {
  Image blob;                 // pose_input_size square, 8-bit RGB
  int64_t square_w = 0, square_h = 0;   // the padded square the crop became
  double angle = 0.0;         // degrees, the rotation that was applied
  double rot_m[6] = {};       // the same rotation as a 2x3, source -> destination
  double pad_bias[2] = {0, 0};   // where the square sits in the frame
};

PoseStage pose_stage(const Image &img, const Geometry &g, const Detection &det);

// -- stage 2, after the landmark network ----------------------------------------

// `img_w` and `img_h` are the ORIGINAL frame's size: the mask is returned in
// frame coordinates and its placement is the square's offset within the frame.
Person postprocess(const Geometry &g, const PoseOuts &out, const PoseStage &s,
                   int64_t img_w, int64_t img_h);

}  // namespace npue::mppose
