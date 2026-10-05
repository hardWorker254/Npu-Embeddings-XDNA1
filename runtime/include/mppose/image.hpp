//===- image.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=8 detector front end, and why it is not arch=7's.
//
// THE ORDER IS NORMALISE, THEN RESIZE, AND THE ORDER IS THE MODEL
// ---------------------------------------------------------------
// arch=7's palm detector letterboxes an 8-BIT raster and normalises afterwards.
// arch=8's person detector does not: mp_persondet._preprocess divides by 255,
// rescales to [-1,1], and only then calls cv2.resize and copyMakeBorder. The
// difference is not cosmetic:
//
//   * cv2.resize on a uint8 raster ROUNDS TO 8 BITS AT EVERY TAP. On a float
//     image it accumulates in float. On a 1080x810 frame downscaled by 4.8x a tap
//     is a quarter-pixel box and the rounding is a per-tap quantisation of the
//     whole input.
//   * THE PAD IS A TENSOR VALUE, NOT A RASTER VALUE. The detector pads with 0
//     AFTER the rescale, so its border is 0.0 -- the middle of [-1,1], not the
//     bottom. A front end that letterboxes a uint8 raster with 0 and normalises
//     afterwards puts (0 - 0.5)/0.5 = -1.0 in that border, which to this stem is
//     the most saturated value the input can take, all the way round the frame.
//
// The landmark net's own front end pads in the 8-bit raster and divides at the
// very end, so ITS border is 0.0 too -- for the opposite reason. Both are 0.0 and
// only one of them is a bug, which is exactly why neither is shared with the
// other.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/raster.hpp"
#include "mppose/geometry.hpp"
#include "vit/image.hpp"

namespace npue::mppose {

using vit::Image;

// The uniform scale and the symmetric zero padding the detector's letterbox
// applied, so decode_det can take a detection back to the caller's pixels.
//
// `scale` is r = min(side/h, side/w), the same number on both axes, which is what
// "uniform" means and why an aspect ratio survives the round trip.
//
// `pad_bias_x/y` is the (left, top) offset in ORIGINAL pixels -- and it is an
// INTEGER, because the zoo's final line is
// `(pad_bias / ratio).astype(np.int32)`. That truncation is part of the decode:
// a detection is corrected by this offset in original pixels, and a
// round()-instead difference is a fraction of a pixel on every box.
struct DetLetterbox {
  double scale = 1.0;
  int64_t left = 0, top = 0;              // canvas pixels
  int64_t pad_bias_x = 0, pad_bias_y = 0;  // ORIGINAL pixels, truncated
};

// The detector's input, NCHW [3, S, S] float32, together with the letterbox it
// came from.
//
// `src` is the caller's frame in RGB -- the decoders produce RGB and cv2's
// BGR2RGB in the zoo's _preprocess is the same operation already done. Doing it
// again here would swap the red and blue channels, which is a measured mistake
// rather than a hypothetical one: on this architecture's own earlier demo it
// moved the pose joints by up to 110.8 px.
std::vector<float> det_input(const Image &src, const Geometry &g,
                             DetLetterbox *lb);

}  // namespace npue::mppose
