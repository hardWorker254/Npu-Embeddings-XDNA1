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

#include "vit/image.hpp"

namespace npue::hands {

using vit::Image;

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

// cv2.INTER_LINEAR, no antialiasing. dst(x) reads src at (x + 0.5)*scale - 0.5.
// Off by half a source pixel it is not, and the result looks like a bad model
// rather than a bad resample.
Image resize_bilinear(const Image &src, int64_t nh, int64_t nw);

// cv2.INTER_AREA -- an exact, separable box average.
//
// NO SPECIAL CASE FOR UPSCALING, and that is deliberate. When the axis grows the
// box [i*s, (i+1)*s) is narrower than one source cell, and the same loop
// degrades on its own into a linear blend of the two cells it straddles -- which
// is what cv2 does and NOT the nearest-neighbour its documentation claims. A
// branch here would be a guess about which of the two the checkpoint was
// measured with, and the loop below does not have to guess.
Image resize_area(const Image &src, int64_t nh, int64_t nw);

// cv2.warpAffine(src, m, INTER_LINEAR, border=0). `m` is 2x3 row-major and maps
// SOURCE to DESTINATION. The output is `src`'s size, which is what the demo
// wants: the rotation is about the crop's centre and the caller keeps the
// geometry it computed rather than discovering a resized canvas.
//
// Destination pixels sampling outside the source are ZERO, not clamped. They are
// two different answers and the zoo's border is 0; clamping fills the corners
// with edge pixels and the landmark network sees a frame that is not there.
Image warp_affine(const Image &src, const double m[6]);

// A copy of the half-open rectangle, clipped to the image. The clip is done by
// the CALLER and the coordinates are integers by the time they arrive here --
// see decode.hpp, which reproduces the truncation order the demo uses.
Image crop(const Image &src, int64_t x0, int64_t y0, int64_t x1, int64_t y1);

// HWC uint8 -> [3, H, W] float32 as (u8/255 - mean) / std, the same
// normalisation arch=6 uses and for the same reason: the container carries the
// mean and the std, and a front end that ignored them would be a second model.
// For this checkpoint both are the identity (0 and 1), which is why the two
// networks may be fed the same raster as long as nothing divides by them.
//
// The zero pad is normalised too, not left at zero in tensor space: the padded
// pixel becomes (0 - mean) / std rather than 0. With mean 0 that is 0 and the
// two readings coincide, and they stop coinciding the moment a checkpoint with
// a non-zero mean is packed -- which is exactly when a front end that hard-codes
// the identity quietly becomes a different model.
std::vector<float> to_nchw_normalised(const Image &src,
                                      const std::vector<float> &mean,
                                      const std::vector<float> &std_dev);

}  // namespace npue::hands
