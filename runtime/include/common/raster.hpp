//===- raster.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the four resamplers two architectures both need.
//
// WHY THESE ARE HERE AND NOT IN EITHER ARCHITECTURE
// -------------------------------------------------
// arch=7 (MediaPipe hands) and arch=8 (MediaPipe pose) each letterbox a frame,
// then crop a rotated region and resample it. Between them they need four
// operations that are cv2 calls and not model arithmetic:
//
//   resize_bilinear   cv2.INTER_LINEAR, no antialiasing
//   resize_area       cv2.INTER_AREA, an exact box average
//   warp_affine       cv2.warpAffine, source-to-destination, border 0
//   crop              a half-open rectangle copy
//
// The first version of each lived in hands/image.cpp, and copying them for arch=8
// would have been the second copy of a resampler whose OUTPUT IS THE MODEL'S
// ACCURACY. Two copies of a box-average drift apart by a half-pixel some day and
// neither reports it: the landmarks simply move. So they live here, once, and
// hands/image.hpp re-exports them under its own namespace so that arch=7's own
// call sites are untouched -- which is what makes this a refactor and not a
// change.
//
// WHAT IS *NOT* HERE, AND WHY
// --------------------------
// The letterbox is NOT. arch=7 pads with 0 in an 8-bit raster and normalises
// afterwards; arch=8's detector pads with 0 in the [-1,1] TENSOR and its
// landmark net pads in the raster for a different reason. One letterbox would
// have to be told which, and the pad's value is the difference between 0.0 and
// -1.0 at this stem -- the middle of the range against its most saturated value.
// Same for the normalisation: arch=7's two networks share a mean/std and arch=8's
// two do not.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

#include "vit/image.hpp"

namespace npue::raster {

using vit::Image;

// cv2's saturate_cast<uchar>(double): round to nearest (ties to even, like
// cvRound), then clamp to 0..255. The rounding is half-to-even rather than
// half-away-from-zero, which differs by one 8-bit level on exact halves and by
// nothing anywhere else; it is stated here because the front end is measured in
// PIXELS and a claim of "the same resampler" should say what "same" is.
inline uint8_t to_u8(double v);

// A zero-filled image of this size.
Image like(int64_t h, int64_t w);

// cv2.INTER_LINEAR, no antialiasing. dst(x) reads src at (x + 0.5)*scale - 0.5,
// clamping at the border. Off by half a source pixel it is not, and the result
// looks like a bad model rather than a bad resample.
Image resize_bilinear(const Image &src, int64_t nh, int64_t nw);

// cv2.INTER_AREA -- an exact, separable box average.
//
// NO SPECIAL CASE FOR UPSCALING, and that is deliberate. When the axis grows the
// box [i*s, (i+1)*s) is narrower than one source cell, and the same loop
// degrades on its own into a linear blend of the two cells it straddles -- which
// is what cv2 does and NOT the nearest-neighbour its documentation claims. A
// branch here would be a guess about which of the two the checkpoint was
// measured with, and the loop does not have to guess.
Image resize_area(const Image &src, int64_t nh, int64_t nw);

// cv2.warpAffine(src, m, INTER_LINEAR, border=0). `m` is 2x3 row-major and maps
// SOURCE to DESTINATION, so the resampling uses the INVERSE; using M directly
// rotates the subject the wrong way and the network absorbs it well enough to
// still return a plausible answer -- the worst kind of bug, because it does not
// look wrong.
//
// The output is `src`'s size. Destination pixels sampling outside the source are
// ZERO, not clamped: two different answers, and the zoo's border is 0. Clamping
// fills the corners with edge pixels and the network sees a frame that is not
// there.
Image warp_affine(const Image &src, const double m[6]);

// A copy of the half-open rectangle. The clip is the CALLER's job -- the
// coordinates are integers by the time they arrive, and the truncation order that
// produces those integers is part of the model.
Image crop(const Image &src, int64_t x0, int64_t y0, int64_t x1, int64_t y1);

// cv2.copyMakeBorder(..., BORDER_CONSTANT, 0) on an 8-bit RGB raster.
//
// Not `letterbox`: that one decides the scale, this one pads a rectangle the
// caller has already cut, and the zoo uses both. `top`/`bottom` are ROWS and
// `left`/`right` are COLUMNS -- cv2's argument order is (top, bottom, left,
// right) and transposing the last pair pads the wrong axis, which on a square
// crop is invisible and on a rectangular one moves the subject.
Image pad_zeros(const Image &src, int64_t top, int64_t bottom, int64_t left,
                int64_t right);

// Bilinear on a CHANNEL-INTERLEAVED float buffer at cv2's half-pixel grid, with
// the index clamp at the border and NO 8-bit rounding.
//
// This is a separate function from resize_bilinear because the ORDER matters and
// it differs by architecture. arch=7 resizes an 8-bit raster and normalises
// after; arch=8's DETECTOR normalises first and then resizes a float image,
// because its zoo's _preprocess divides by 255 and rescales before cv2.resize
// sees the pixels. Resizing a uint8 raster and normalising afterwards is not the
// same pipeline: it rounds to 8 bits at every tap rather than accumulating in
// float, which at this stem's border is the difference between a padded 0.0 and a
// saturated -1.0. Doing it the other way round would be a quantised front end
// that reports the same shapes.
void resize_bilinear_f32(const float *src, int64_t sh, int64_t sw, float *dst,
                         int64_t dh, int64_t dw, int64_t ch = 3);

// cv2.warpAffine on a float buffer, at cv2's own grid, border 0, no 8-bit
// rounding.
//
// The float twin of warp_affine, and it exists for the same reason: the
// segmentation mask is a float32 tensor with values well past 255, and rounding
// it to 8 bits -- or clamping it -- would throw away exactly the part of the
// signal the threshold is applied to. `ch` is the channel count; the mask is one.
void warp_affine_f32(const float *src, int64_t sh, int64_t sw, float *dst,
                     int64_t dh, int64_t dw, const double m[6], int64_t ch = 1);

// HWC uint8 -> [3, H, W] float32 as (u8/255 - mean) / std, per channel.
//
// `mean` and `std_dev` come from the CONTAINER and not from a constant here: what
// a checkpoint was trained with is part of the checkpoint, and arch=8's two
// networks do not even agree with each other (the detector is [-1,1] and the
// landmark net is [0,1]).
std::vector<float> to_nchw_normalised(const Image &src,
                                      const std::vector<float> &mean,
                                      const std::vector<float> &std_dev);

}  // namespace npue::raster
