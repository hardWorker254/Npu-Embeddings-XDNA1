//===- decode.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=7 decode: the palm detector's 2016 rows become boxes,
// then ONE box plus TWO palm keypoints become a rotated 224x224 crop, then the
// landmark network's 63 numbers become 21 joints in the caller's pixels.
//
// FOUR FUNCTIONS, AND THE SPLITS ARE THE NETWORKS
// ------------------------------------------------
//   palm_stage     letterbox, and record the pad
//   decode_palm    anchors, sigmoid, NMS, inverse letterbox
//   landmark_stage crop, rotate, resize
//   postprocess    the rotation's inverse, and the hand's reported box
//
// Between palm_stage and decode_palm the PALM NETWORK runs; between
// landmark_stage and postprocess the LANDMARK NETWORK runs. So this header
// cannot hand the networks in -- it takes their outputs. That is what lets a
// caller compare the two stages separately, and it is what the gate does: it
// runs the same four steps around the same two network outputs that the
// container gate used.
//
// THE ROTATION IS PREDICTED, NOT CHOSEN
// -------------------------------------
// MediaPipe does not try every angle. The palm detector emits SEVEN keypoints,
// two of which (recorded in the container as palm_lm_wrist and
// palm_lm_middle_base) name the wrist and the base of the middle finger, and the
// rotation is the one that puts THAT PAIR vertical. The landmark network is then
// shown an upright hand, and the whole of postprocess exists to put its answer
// back in the frame the rotation took it from.
//
// WHICH TWO matters as much as anything else in this file, and it is why the
// pair is read from the container rather than assumed to be (0, 2). The wrong
// pair rotates by some other angle -- usually not zero -- and the network still
// returns 21 joints and a plausible hand, because a rotated hand is still a
// hand. On this checkpoint the wrong pair moved the wrist by 35 px in the
// earlier int8 measurement; from the float one it is less and still not nothing,
// and nothing about the OUTPUT distinguishes the two cases except the
// landmarks' own positions.
//
// WHY A SECOND RESAMPLER FOR THE CROP AND NOT THE LETTERBOX'S
// -------------------------------------------------------------
// The crop is a rotated region downscaled by a factor well above one, so it
// uses cv2.INTER_AREA (an exact box average) while the letterbox uses
// INTER_LINEAR. See image.hpp: either one on the wrong stage is a different
// checkpoint's front end.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hands/geometry.hpp"
#include "hands/image.hpp"
#include "hands/net.hpp"

namespace npue::hands {

// One palm detection, in the CALLER's pixels: the box, the seven palm
// keypoints the rotation is read from, and the score.
struct Detection {
  double box[4] = {0, 0, 0, 0};        // x1, y1, x2, y2
  double kps[kPalmLandmarks][2] = {};  // seven, x/y
  double score = 0.0;
};

// One hand. `landmarks` and `world` are 21 x 3.
//
// `landmarks` is in the CALLER's pixels: x and y absolute, and z a DEPTH
// RELATIVE TO THE WRIST with the same pixel scale as x and y -- it comes out of
// the network in the crop's own units and is multiplied by the crop's scale here,
// so a z of 30 is thirty pixels nearer than the wrist, not thirty pixels of
// real distance. `world` is in METRES and wrist-relative, and is NOT in the
// frame's pixels at all: it is the metric frame the checkpoint regresses, and
// the demo returns it rotated into the frame's orientation but not translated
// into its position, because its origin is the wrist.
struct Hand {
  double bbox[4] = {0, 0, 0, 0};
  double landmarks[kHandLandmarks][3] = {};
  double world[kHandLandmarks][3] = {};
  double handedness = 0.0;   // sigmoid; 0.5 is the boundary
  double presence = 0.0;     // sigmoid
};

// -- stage 1, before the palm network -----------------------------------------

// The letterboxed raster and the pad that decode_palm has to undo.
struct PalmStage {
  Letterbox lb;
  Image padded;      // palm_size x palm_size, 8-bit RGB
};

PalmStage palm_stage(const Image &img, const Geometry &g);

// -- stage 1, after the palm network ------------------------------------------
//
// `head` is Network::palm_head's output and `lb` the letterbox that produced
// the input it saw -- handed in rather than recomputed, so the correction below
// is against the SAME scale the raster was built with. `orig_long_side` is
// max(img.height, img.width) of the ORIGINAL frame.
//
// The detections come back in DESCENDING SCORE order, which is the order NMS
// produces them in. Every hand is an independent crop, so the caller picks which
// ones to run through stage 2 and there is no "best" argument here.
std::vector<Detection> decode_palm(const Geometry &g, const PalmOut &head,
                                   const Letterbox &lb,
                                   double orig_long_side);

// -- stage 2, before the landmark network --------------------------------------

// The crop, and the four things postprocess needs to undo the geometry. Kept
// rather than recomputed: recomputing the rotation is re-deriving an atan2 and
// a wrap, and a half-degree disagreement there moves every joint by its own
// distance from the wrist.
struct LandmarkStage {
  Image blob;          // lm_size x lm_size, 8-bit RGB
  double rot_bbox[4] = {0, 0, 0, 0};   // the SECOND crop's clipped box
  // The rotated palm keypoints' bounding box BEFORE the second crop grows it.
  // Kept because the second crop's output box is clipped to the raster and a
  // clip is silent: a box that should have been [97, 58, 649, 677] and came
  // back [98, 0, 649, 729] says only "it touched the edges", and the two
  // candidate faults -- a wrong rotation of the keypoints, or a wrong enlarge
  // factor -- are told apart by this number alone.
  double palm_kp_box[4] = {0, 0, 0, 0};
  double angle = 0.0;   // degrees, the rotation that was applied
  double rot_m[6] = {}; // the same rotation as a 2x3, source -> destination
  double pad_bias[2] = {0, 0};   // what the first crop's padding cost
};

LandmarkStage landmark_stage(const Image &img, const Geometry &g,
                             const Detection &det);

// -- stage 2, after the landmark network ----------------------------------------

Hand postprocess(const Geometry &g, const LmOut &lm_out,
                 const LandmarkStage &s, const Detection &det);

}  // namespace npue::hands
