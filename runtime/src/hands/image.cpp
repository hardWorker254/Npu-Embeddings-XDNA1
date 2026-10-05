//===- image.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=7 front end: a letterbox and a normalisation.
//
// WHAT USED TO BE HERE AND WHY IT IS NOT ANY MORE
// ------------------------------------------------
// resize_bilinear, resize_area, warp_affine and crop were defined in this file
// until arch=8 needed the same four of them. Two copies of a box-average is a
// copy whose OUTPUT IS THE MODEL'S ACCURACY, and two copies drift apart by a
// half-pixel some day with nothing reporting it -- the landmarks simply move.
// They now live in common/raster.hpp with their coordinate transforms and their
// reasons, and hands/image.hpp re-exports them so that every call site below and
// in decode.cpp is unchanged. verify_hands.py's section 4 is the proof that this
// file still produces the numbers it produced before the move.
//
// WHAT IS STILL ARCH=7'S
// ---------------------
// The letterbox and the normalisation, and they are here rather than in raster.hpp
// because arch=8's are different in a way that is a MODEL difference:
//
//   arch=7   pads with 0 in an 8-bit raster and normalises afterwards, so the
//            border is 0.0 in tensor space, and its two networks share one mean
//            and one standard deviation.
//   arch=8   its DETECTOR divides by 255 and rescales FIRST and pads with 0 in
//            the [-1,1] tensor, so the border is 0.0 too -- the middle of the
//            range, not the bottom. A reader that letterboxed a uint8 raster with
//            0 and normalised afterwards would put -1.0 there, which to that stem
//            is the most saturated value the input can take.
//
// Same numeric border value, opposite reason, and a letterbox shared between them
// would have to be told which reason applies.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "hands/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace npue::hands {

Image letterbox(const Image &src, int64_t side, Letterbox *out) {
  if (src.empty()) throw std::runtime_error("letterbox: the image is empty");
  const int64_t h = src.height, w = src.width;
  const double r = std::min(static_cast<double>(side) / h,
                            static_cast<double>(side) / w);
  // Truncated toward zero, which for these positive values is a floor, and is
  // what cv2 and numpy both do when they take an int of a float. Rounding
  // instead makes the padded canvas one row taller on some inputs and shifts
  // every detection by a fraction of its own size.
  const int64_t rs_h = static_cast<int64_t>(h * r);
  const int64_t rs_w = static_cast<int64_t>(w * r);
  if (rs_h < 1 || rs_w < 1)
    throw std::runtime_error("letterbox: a " + std::to_string(h) + "x" +
                             std::to_string(w) + " image scales to " +
                             std::to_string(rs_h) + "x" + std::to_string(rs_w) +
                             " at side " + std::to_string(side));
  Image small = npue::raster::resize_bilinear(src, rs_h, rs_w);
  const int64_t ph = side - rs_h, pw = side - rs_w;
  const int64_t left = pw / 2, top = ph / 2;
  Image out_img = npue::raster::like(side, side);
  for (int64_t y = 0; y < side; ++y) {
    const int64_t sy = y - top;
    if (sy < 0 || sy >= rs_h) continue;   // the zero pad
    for (int64_t x = 0; x < side; ++x) {
      const int64_t sx = x - left;
      if (sx < 0 || sx >= rs_w) continue;
      std::copy_n(small.at(sy, sx), 3, out_img.at(y, x));
    }
  }
  if (out) {
    out->scale = r;
    out->left = static_cast<int>(left);
    out->top = static_cast<int>(top);
    // The same offset in ORIGINAL pixels, which is the unit a detection is
    // corrected in: the head's outputs are fractions of the 192-pixel input side
    // and are scaled back by the ORIGINAL long side, so they land in original
    // pixels minus this.
    out->pad_orig_x = static_cast<int>(static_cast<double>(left) / r);
    out->pad_orig_y = static_cast<int>(static_cast<double>(top) / r);
  }
  return out_img;
}

}  // namespace npue::hands
