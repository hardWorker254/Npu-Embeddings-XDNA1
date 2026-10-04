//===- image.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- arch=6 letterbox front end. See image.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "pose/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "runtime/pool.hpp"

namespace npue::pose {

namespace {

// Split [0, n) over the pool, or run it here when there is no pool or the span is
// too small to be worth a barrier. Local to this file rather than shared with the
// resampler's copy in vit/image.cpp because the two count work differently -- one
// in output pixels, one in source rows -- and a single shared threshold would be
// the wrong number for one of them and nobody would be able to tell which.
constexpr int64_t kFrontEndMinElems = 1 << 16;

template <typename F>
void par_range(app::Pool *pool, int64_t n, int64_t per_item, F &&f) {
  if (pool == nullptr || pool->size() == 1 || n * per_item < kFrontEndMinElems) {
    for (int64_t i = 0; i < n; ++i) f(i);
    return;
  }
  pool->run([&](int w, int nw) {
    for (int64_t i = w; i < n; i += nw) f(i);
  });
}

}  // namespace

void letterbox_normalise_into(float *dst, const npue::vit::Image &src,
                              const Geometry &g, Letterbox &lb,
                              app::Pool *pool) {
  if (src.empty())
    throw std::runtime_error("pose front end: the image decoded to nothing");

  const int64_t side = g.input_size;
  lb.src_w = src.width;
  lb.src_h = src.height;
  lb.dst = side;

  // ONE uniform scale, from the LONG side, so the aspect ratio survives:
  //
  //   scale = side / max(src_w, src_h)
  //
  // and the shorter side is padded. This is the whole reason the keypoints come
  // back where the user drew them: a single scale factor and a single offset are
  // invertible, where a per-axis squash would need two and the network would have
  // been trained on people of one shape.
  const int64_t longest = std::max(src.width, src.height);
  const double scale = static_cast<double>(side) / static_cast<double>(longest);
  lb.scale = scale;

  // The resized extent, rounded. round-half-away-from-zero so that an image
  // which is exactly twice the side comes out at exactly half, and the padded
  // region is symmetric to the pixel -- a 1px asymmetry here is 1px of keypoint
  // error on one side of the body only, which reads as a bad estimate rather
  // than a systematic offset.
  auto rh = [&](double v) -> int64_t {
    const double r = v >= 0 ? v + 0.5 : v - 0.5;
    return static_cast<int64_t>(r);
  };
  int64_t rw = rh(src.width * scale);
  int64_t rhgt = rh(src.height * scale);
  if (rw < 1) rw = 1;
  if (rhgt < 1) rhgt = 1;
  // A rounding artefact can push the long side one pixel past the canvas; clamp
  // it, because the padding below would then be negative and the decode would
  // invert to an offset outside the source image.
  if (rw > side) rw = side;
  if (rhgt > side) rhgt = side;

  lb.pad_x = (side - rw) / 2;
  lb.pad_y = (side - rhgt) / 2;

  npue::vit::Image scaled;
  if (src.width == rw && src.height == rhgt) {
    scaled = src;   // resize_to is the identity here and says so itself
  } else {
    // BILINEAR, and only BILINEAR: the pose pipeline's letterbox is a bilinear
    // resize, and BICUBIC/BOX/LANCZOS are the ViT's choices, made for a
    // classifier that does not care about coordinates. Claiming a resample here
    // would be claiming a model that was not trained this way.
    scaled = npue::vit::resize_to(src, rw, rhgt, npue::vit::Resample::Bilinear,
                                   pool);
  }

  // [3, S, S], padded with kPadValue/255 and normalised per channel.
  //
  // The pad is applied as a PIXEL VALUE and then normalised like any other, so
  // it becomes (114/255 - mean)/std -- which is 0 for YOLOv8's mean 0 / std 1.
  // Filling the normalised tensor with zeros instead would be a different pad for
  // any checkpoint whose mean is not zero, and this build does not assume one.
  const size_t plane = static_cast<size_t>(side) * static_cast<size_t>(side);
  // The std check is HERE, before the parallel region and not inside it. A worker
  // that threw would unwind through Pool::run's generation counter and take the
  // other 15 threads down with it, and the message names one channel -- so it is
  // checked for all three up front, where a throw is a plain throw.
  for (int64_t c = 0; c < 3; ++c) {
    if (g.std_dev[static_cast<size_t>(c)] == 0.0)
      throw std::runtime_error("pose front end: image_std[" + std::to_string(c) +
                               "] is zero");
  }
  const double pad_n =
      (static_cast<double>(kPadValue) / 255.0 - g.mean[0]) / g.std_dev[0];
  par_range(pool, static_cast<int64_t>(3 * plane), 1,
            [&](int64_t i) { dst[i] = static_cast<float>(pad_n); });

  // One task per (channel, row). Rows are the unit because the output is
  // [3, S, S]: a row of one channel is `side` contiguous floats, so a worker
  // writes whole cache lines, and no two workers write the same line.
  const int64_t rows = 3 * rhgt;
  par_range(pool, rows, rw, [&](int64_t k) {
    const int64_t c = k / rhgt;
    const int64_t y = k - c * rhgt;
    const double mu = g.mean[static_cast<size_t>(c)];
    const double sd = g.std_dev[static_cast<size_t>(c)];
    float *out = dst + static_cast<size_t>(c) * plane;
    const int64_t py = y + lb.pad_y;
    const uint8_t *row = scaled.at(y, 0);
    for (int64_t x = 0; x < rw; ++x) {
      const double v = static_cast<double>(row[static_cast<size_t>(x) * 3 +
                                                static_cast<size_t>(c)]) / 255.0;
      out[static_cast<size_t>(py) * side + static_cast<size_t>(x + lb.pad_x)] =
          static_cast<float>((v - mu) / sd);
    }
  });
}

std::vector<float> letterbox_normalise(const npue::vit::Image &src,
                                       const Geometry &g, Letterbox &lb) {
  const int64_t side = g.input_size;
  std::vector<float> out(3 * static_cast<size_t>(side) * static_cast<size_t>(side));
  letterbox_normalise_into(out.data(), src, g, lb, nullptr);
  return out;
}

std::vector<float> preprocess_file(const std::string &path, const Geometry &g,
                                   Letterbox &lb) {
  // vit::decode_image decides PNG/JPEG BY MAGIC BYTES and refuses everything
  // else by name. Reusing it verbatim is the point: a pose front end that
  // accepted a format the classifier's would refuse would be a second, weaker
  // rule about the same bytes, and the two would drift.
  const npue::vit::Image im = npue::vit::decode_image(path);
  return letterbox_normalise(im, g, lb);
}

}  // namespace npue::pose
