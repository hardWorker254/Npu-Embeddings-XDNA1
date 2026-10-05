//===- image.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=8 detector front end. See image.hpp for why the order
// is normalise-then-resize and why the pad is a tensor value.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace npue::mppose {

std::vector<float> det_input(const Image &src, const Geometry &g,
                             DetLetterbox *lb) {
  if (src.empty())
    throw std::runtime_error("det_input: the image is empty");
  const int64_t side = g.det_input_size;
  const int64_t h = src.height, w = src.width;
  if (side <= 0)
    throw std::runtime_error(
        "det_input: the container declares a detector input size of " +
        std::to_string(side));
  // r = min(side/h, side/w), the LONG side onto the canvas -- the same reduction
  // arch=7's letterbox makes, and the same one the zoo makes with
  // min(input_size / image.shape[:2]).
  const double r = std::min(static_cast<double>(side) / h,
                            static_cast<double>(side) / w);
  // Truncated toward zero, which for these positive values is a floor, and is
  // what `(h * ratio).astype(np.int32)` does. Rounding instead makes the padded
  // canvas a row taller on some inputs and shifts every detection by a fraction of
  // its own size.
  const int64_t rs_h = static_cast<int64_t>(h * r);
  const int64_t rs_w = static_cast<int64_t>(w * r);
  if (rs_h < 1 || rs_w < 1 || rs_h > side || rs_w > side)
    throw std::runtime_error("det_input: a " + std::to_string(h) + "x" +
                             std::to_string(w) + " image scales to " +
                             std::to_string(rs_h) + "x" + std::to_string(rs_w) +
                             " at side " + std::to_string(side));
  // The three per-channel constants, fetched once. This front end is written per
  // pixel and an unchecked index here is a NaN channel rather than an error.
  if (g.det_mean.size() != 3 || g.det_std.size() != 3)
    throw std::runtime_error(
        "det_input: the container declares " + std::to_string(g.det_mean.size()) +
        " means and " + std::to_string(g.det_std.size()) +
        " standard deviations for the detector; three of each are needed.");
  float mu[3], sd[3];
  for (int c = 0; c < 3; ++c) {
    mu[c] = static_cast<float>(g.det_mean[static_cast<size_t>(c)]);
    sd[c] = static_cast<float>(g.det_std[static_cast<size_t>(c)]);
    if (sd[c] == 0.0f)
      throw std::runtime_error("det_input: det_image_std[" + std::to_string(c) +
                               "] is zero, so the normalisation is not "
                               "invertible and no input can be recovered");
  }

  // STEP 1: NORMALISE THE WHOLE FRAME, into an interleaved float buffer. This is
  // where the zoo's `/255.0` and `(x - 0.5) * 2` happen, and it happens over the
  // ENTIRE image and before any resampling -- which is the whole reason this is not
  // hands' letterbox, which letterboxes the 8-bit raster first.
  const size_t whole = static_cast<size_t>(h) * static_cast<size_t>(w) * 3;
  std::vector<float> lin(whole);
  for (int64_t y = 0; y < h; ++y) {
    const uint8_t *const srow = src.at(y, 0);
    float *const drow = lin.data() + static_cast<size_t>(y) * w * 3;
    for (int64_t p = 0; p < w * 3; ++p) {
      const int c = static_cast<int>(p % 3);
      drow[p] = (static_cast<float>(srow[p]) * (1.0f / 255.0f) - mu[c]) / sd[c];
    }
  }

  // STEP 2: cv2.resize to (rs_w, rs_h), which is a REAL reduction -- a 810x1080
  // frame becomes 168x224 here, a factor of about 4.8. It is worth naming because
  // `ratio_size` is `input_size` on the long axis, which makes the target look
  // like the frame's own size at a glance, and reading it that way -- normalising
  // a 168x224 window of the frame's top-left corner instead of resampling the
  // frame -- produced a tensor with the right shape, the right range and the wrong
  // picture, and a detector that returned 20 people at score 1.000 on a frame with
  // one.
  std::vector<float> resized(static_cast<size_t>(rs_h) *
                             static_cast<size_t>(rs_w) * 3);
  npue::raster::resize_bilinear_f32(lin.data(), h, w, resized.data(), rs_h, rs_w,
                                    3);

  // STEP 3: the canvas, with the border written as 0.0 IN TENSOR SPACE.
  //
  // The output is zero-initialised and only the image's own rectangle is written,
  // so the border is exactly +0.0 and never a normalised zero. With this
  // container's mean 0.5 and std 0.5 those are very different numbers: -1.0 is the
  // most saturated value the stem can see, and a frame ringed with it moves the
  // pyramid's edge responses and with them the keypoints nearest the margin.
  const int64_t plane = side * side;
  std::vector<float> out(static_cast<size_t>(3) * static_cast<size_t>(plane), 0.f);
  const int64_t left = (side - rs_w) / 2, top = (side - rs_h) / 2;
  for (int64_t y = 0; y < rs_h; ++y) {
    const float *const srow = resized.data() + static_cast<size_t>(y) * rs_w * 3;
    const int64_t dy = y + top;
    for (int64_t x = 0; x < rs_w; ++x) {
      const int64_t dx = x + left;
      const size_t d = static_cast<size_t>(dy) * side + static_cast<size_t>(dx);
      out[static_cast<size_t>(0) * plane + d] = srow[static_cast<size_t>(x) * 3 + 0];
      out[static_cast<size_t>(1) * plane + d] = srow[static_cast<size_t>(x) * 3 + 1];
      out[static_cast<size_t>(2) * plane + d] = srow[static_cast<size_t>(x) * 3 + 2];
    }
  }

  if (lb) {
    lb->scale = r;
    lb->left = left;
    lb->top = top;
    // `(pad_bias / ratio).astype(np.int32)` -- a truncation, and an INTEGER one.
    // Both offsets are non-negative, so truncation is a floor here.
    lb->pad_bias_x = static_cast<int64_t>(static_cast<double>(left) / r);
    lb->pad_bias_y = static_cast<int64_t>(static_cast<double>(top) / r);
  }
  return out;
}

}  // namespace npue::mppose
