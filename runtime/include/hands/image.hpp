//===- image.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=7 image front end: letterbox, and the three
// resamplers the zoo demo's two stages need.
//
// THREE RESAMPLERS, AND CV2 NAMES ALL THREE
// -----------------------------------------
// The palm stage letterboxes with cv2.INTER_LINEAR and no antialiasing. The
// landmark stage crops a rotated region and then resizes it with cv2.INTER_AREA,
// which is an exact box average. The rotation between them is
// cv2.warpAffine, also INTER_LINEAR, border=0. So:
//
//   letterbox        INTER_LINEAR, half-pixel grid, NOT area. Using area for
//                    the letterbox is the tempting "better" choice and it is a
//                    different detector: the zoo was shipped with a bilinear
//                    letterbox and the checkpoint's edge responses were
//                    measured against it.
//   resize_area      the exact box average. Bilinear here instead moves the
//                    fingertips by a pixel and a half on a downscaled crop.
//   warp_affine      source-to-destination, so the resampling uses the
//                    INVERSE. Using M directly rotates the hand the wrong way,
//                    and the landmark network absorbs it well enough to still
//                    return a plausible hand -- the worst kind of bug, because
//                    it does not look wrong.
//
// WHY THE RASTER IS 8-BIT AND THE ACCUMULATION IS NOT
// --------------------------------------------------
// Every one of these returns a vit::Image, so the letterboxed palm input is
// rounded back to 8 bits before the network sees it -- which is what cv2 does,
// and what the zoo does. The Python reference in tools/verify/verify_hands.py
// keeps that stage in float64 instead, and the two therefore disagree by about
// one part in a hundred in the crop's scale. That is a documented property of
// being a second implementation of a resampler and it is the reason the front
// end is measured in PIXELS (1.09 on a 520-pixel frame) rather than at the
// 1e-06 of tensor scale the network itself holds to.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/raster.hpp"
#include "vit/image.hpp"

namespace npue::hands {

using vit::Image;

// THE FOUR RESAMPLERS BELOW LIVE IN common/raster.hpp AND ARE RE-EXPORTED HERE.
//
// arch=8 needed all four of them and they were, at the time, arch=7's own
// definitions. A second copy of a box-average is a copy whose OUTPUT IS THE
// MODEL'S ACCURACY, so they were moved rather than duplicated. Re-exported under
// this namespace so that every arch=7 call site and the gates' front end are
// untouched by the move -- which is what makes it a refactor rather than a change.
using npue::raster::crop;
using npue::raster::resize_area;
using npue::raster::resize_bilinear;
using npue::raster::to_nchw_normalised;
using npue::raster::warp_affine;

// The uniform scale and the symmetric zero padding the letterbox applied, so
// the decode can take a detection back to the pixels the caller handed in.
//
// `scale` is the same number on both axes, which is what "uniform" means and is
// why a keypoint's aspect ratio survives the round trip. `pad` is the
// (left, top) offset of the resized image inside the square canvas, in CANVAS
// pixels; `pad_orig` is the same offset expressed in ORIGINAL pixels, which is
// what a detection is corrected by -- the detectors' outputs are fractions of
// the input side, so they are scaled back by the LONG side of the original and
// then have this subtracted.
struct Letterbox {
  double scale = 1.0;
  int left = 0, top = 0;
  int pad_orig_x = 0, pad_orig_y = 0;
};

// Short side to `side` x `side`, uniform scale, then pad with ZEROS.
//
// The pad value is 0 and not the grey 114 that arch=6 uses, because this is not
// the same model: MediaPipe's detectors were trained on letterboxes filled with
// black, and a front end that filled them with grey feeds the network a border
// it has never seen, which moves the pyramid's edge responses and with them the
// keypoints nearest the margin.
//
// `r` is min(side/h, side/w) and the resized extent is (h*r, w*r) TRUNCATED, the
// same truncation cv2's and numpy's int casts do -- floor for the positive
// numbers both can produce here.
Image letterbox(const Image &src, int64_t side, Letterbox *out);

// resize_bilinear, resize_area, warp_affine and crop are DECLARED IN
// common/raster.hpp and re-exported at the top of this file. They were arch=7's
// own; arch=8 needed the same four and a second copy of a box-average is a copy
// whose output is the model's accuracy, so they were moved rather than
// duplicated. Their comments -- the coordinate transforms, why there is no
// upscaling special case, and why the border is 0 rather than clamped -- moved
// with them.

// HWC uint8 -> [3, H, W] float32 as (u8/255 - mean) / std. Declared in
// common/raster.hpp and re-exported at the top of this file: arch=8's landmark
// net uses exactly this, and what a checkpoint was trained with is the
// checkpoint's own business rather than either architecture's.

}  // namespace npue::hands
