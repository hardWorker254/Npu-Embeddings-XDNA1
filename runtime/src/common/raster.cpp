//===- raster.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the four resamplers arch=7 and arch=8 share. See raster.hpp
// for why they are here and for what is deliberately NOT.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/raster.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace npue::raster {

inline uint8_t to_u8_impl(double v) {
  const double r = std::nearbyint(v);   // ties to even, like cvRound
  if (r <= 0.0) return 0;
  if (r >= 255.0) return 255;
  return static_cast<uint8_t>(r);
}

Image like(int64_t h, int64_t w) {
  Image im;
  im.height = h;
  im.width = w;
  im.rgb.assign(static_cast<size_t>(h * w * 3), 0);
  return im;
}

namespace {

// Bilinear sample at float coordinates, CLAMPING at the border. warp_affine
// masks the outside itself, because clamping and zeroing are different answers
// and the zoo's border is 0.
//
// The four taps are gathered first and blended once per channel. Gathering inside
// the innermost loop instead would re-read the same two rows for every output
// pixel, which at 224x224x3 is the difference between a resampler and a memcpy
// storm.
inline void blend_at(const Image &src, double fy, double fx, uint8_t *dst) {
  const int64_t h = src.height, w = src.width;
  const double fy0f = std::floor(fy), fx0f = std::floor(fx);
  const int64_t y0 = static_cast<int64_t>(fy0f), x0 = static_cast<int64_t>(fx0f);
  const double wy = fy - fy0f, wx = fx - fx0f;
  const int64_t y0c = std::min<int64_t>(std::max<int64_t>(y0, 0), h - 1);
  const int64_t y1c = std::min<int64_t>(std::max<int64_t>(y0 + 1, 0), h - 1);
  const int64_t x0c = std::min<int64_t>(std::max<int64_t>(x0, 0), w - 1);
  const int64_t x1c = std::min<int64_t>(std::max<int64_t>(x0 + 1, 0), w - 1);
  const uint8_t *a = src.at(y0c, x0c), *b = src.at(y0c, x1c);
  const uint8_t *c = src.at(y1c, x0c), *d = src.at(y1c, x1c);
  for (int k = 0; k < 3; ++k)
    dst[k] = to_u8_impl((a[k] * (1.0 - wx) + b[k] * wx) * (1.0 - wy) +
                        (c[k] * (1.0 - wx) + d[k] * wx) * wy);
}

// Exact INTER_AREA weights for ONE axis, as (dst, src, weight) triples that cover
// every destination index exactly once and sum to 1.
//
// The coverage sum is corrected back onto the last tap. Without that the weights
// carry a float64 residue of a few 1e-17, which on a box of 49 pixels is a few
// 1e-19 -- far below the 8-bit output. It is corrected anyway because the
// alternative is a resampler whose weights provably do not sum to one, and the
// next person to change the output dtype inherits a silent brightness shift.
struct AreaAxis {
  std::vector<int64_t> dst, src;
  std::vector<double> w;
};

AreaAxis area_axis(int64_t n_src, int64_t n_dst) {
  AreaAxis ax;
  ax.dst.reserve(static_cast<size_t>(n_dst) * 2);
  ax.src.reserve(static_cast<size_t>(n_dst) * 2);
  ax.w.reserve(static_cast<size_t>(n_dst) * 2);
  const double s = static_cast<double>(n_src) / static_cast<double>(n_dst);
  for (int64_t i = 0; i < n_dst; ++i) {
    const double y0 = i * s, y1 = (i + 1) * s;
    double acc = 0.0;
    const int64_t j_end = std::min<int64_t>(
        static_cast<int64_t>(std::ceil(y1)), n_src);
    for (int64_t j = static_cast<int64_t>(std::floor(y0)); j < j_end; ++j) {
      const double lo = std::max<double>(j, y0);
      const double hi = std::min<double>(j + 1, y1);
      if (hi <= lo) continue;
      ax.dst.push_back(i);
      ax.src.push_back(j);
      ax.w.push_back((hi - lo) / s);
      acc += (hi - lo) / s;
    }
    if (!ax.w.empty() && std::abs(acc - 1.0) > 1e-12) {
      // Onto the LAST tap of this destination, which the loop above appended only
      // if it contributed -- so the index is recomputed rather than assumed,
      // because a destination whose box lies wholly outside the source (possible
      // only on a degenerate scale) has no taps at all.
      ax.w[ax.w.size() - 1] += 1.0 - acc;
    }
  }
  return ax;
}

}  // namespace

uint8_t to_u8(double v) { return to_u8_impl(v); }

Image resize_bilinear(const Image &src, int64_t nh, int64_t nw) {
  if (src.empty() || nh <= 0 || nw <= 0)
    throw std::runtime_error("resize_bilinear: " + std::to_string(nh) + "x" +
                             std::to_string(nw) + " from an empty or degenerate "
                             "source");
  const double sy = static_cast<double>(src.height) / nh;
  const double sx = static_cast<double>(src.width) / nw;
  Image out = like(nh, nw);
  for (int64_t y = 0; y < nh; ++y) {
    const double fy = (y + 0.5) * sy - 0.5;
    for (int64_t x = 0; x < nw; ++x)
      blend_at(src, fy, (x + 0.5) * sx - 0.5, out.at(y, x));
  }
  return out;
}

Image resize_area(const Image &src, int64_t nh, int64_t nw) {
  if (src.empty() || nh <= 0 || nw <= 0)
    throw std::runtime_error("resize_area: " + std::to_string(nh) + "x" +
                             std::to_string(nw) + " from an empty or degenerate "
                             "source");
  const AreaAxis ay = area_axis(src.height, nh);
  const AreaAxis ax = area_axis(src.width, nw);
  Image out = like(nh, nw);
  // Separable, and the intermediate is kept in double at the DESTINATION width
  // rather than the source width: a full-width intermediate would be a second copy
  // of the image, and the tap list along the columns is what the second pass
  // walks.
  std::vector<double> rows(static_cast<size_t>(nh) * static_cast<size_t>(src.width) * 3);
  for (size_t t = 0; t < ay.w.size(); ++t) {
    const int64_t di = ay.dst[t], sj = ay.src[t];
    const double wgt = ay.w[t];
    const uint8_t *s = src.at(sj, 0);
    double *d = rows.data() + static_cast<size_t>(di) * src.width * 3;
    for (int64_t x = 0; x < src.width * 3; ++x) d[x] += s[x] * wgt;
  }
  // The second pass accumulates in double and converts ONCE, at the end. Rounding
  // each tap's partial sum to 8 bits as it arrives would quantise a box average to
  // 8 bits per tap rather than per pixel: with four taps it is a brightness bias,
  // and it is a bias that only appears on downscaled crops -- which is every crop
  // both landmark stages see.
  std::vector<double> acc(static_cast<size_t>(nh) * static_cast<size_t>(nw) * 3, 0.0);
  for (int64_t y = 0; y < nh; ++y) {
    const double *r = rows.data() + static_cast<size_t>(y) * src.width * 3;
    for (size_t t = 0; t < ax.w.size(); ++t) {
      const int64_t di = ax.dst[t], sj = ax.src[t];
      const double wgt = ax.w[t];
      const double *s = r + static_cast<size_t>(sj) * 3;
      double *d = acc.data() + (static_cast<size_t>(y) * nw + di) * 3;
      for (int k = 0; k < 3; ++k) d[k] += s[k] * wgt;
    }
  }
  for (int64_t y = 0; y < nh; ++y)
    for (int64_t x = 0; x < nw; ++x) {
      const double *s = acc.data() + (static_cast<size_t>(y) * nw + x) * 3;
      uint8_t *d = out.at(y, x);
      for (int k = 0; k < 3; ++k) d[k] = to_u8_impl(s[k]);
    }
  return out;
}

Image warp_affine(const Image &src, const double m[6]) {
  if (src.empty())
    throw std::runtime_error("warp_affine: the image is empty");
  const int64_t h = src.height, w = src.width;
  const double a = m[0], b = m[1], tx = m[2];
  const double c = m[3], d = m[4], ty = m[5];
  const double det = a * d - b * c;
  if (det == 0.0)
    throw std::runtime_error(
        "warp_affine: the affine is singular, so it has no inverse and no "
        "destination pixel has a source. The rotation here is built from a point "
        "pair and a box, and a degenerate one is a zero-length box, not a "
        "rounding accident.");
  const double ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  // The inverse translation pairs ROW 0 of M against the ROW-0 inverse entries.
  // Pairing it the other way round -- itx against ic/id instead -- is a translation
  // in the wrong direction, which at a rotation of about pi/2 puts the sample two
  // crop widths away and returns a black crop. The network then reports a subject
  // in the middle of it.
  const double itx = -(ia * tx + ib * ty);
  const double ity = -(ic * tx + id * ty);
  Image out = like(h, w);
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      const double fx = ia * x + ib * y + itx;
      const double fy = ic * x + id * y + ity;
      uint8_t *d = out.at(y, x);
      if (fx < 0.0 || fx > w - 1 || fy < 0.0 || fy > h - 1) continue;  // border=0
      blend_at(src, fy, fx, d);
    }
  }
  return out;
}

Image crop(const Image &src, int64_t x0, int64_t y0, int64_t x1, int64_t y1) {
  if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0)
    throw std::runtime_error("crop: " + std::to_string(x0) + "," +
                             std::to_string(y0) + " to " + std::to_string(x1) +
                             "," + std::to_string(y1) +
                             " is not a non-empty rectangle inside an image");
  if (x1 > src.width || y1 > src.height)
    throw std::runtime_error("crop: the rectangle runs past the image's " +
                             std::to_string(src.width) + "x" +
                             std::to_string(src.height));
  Image out = like(y1 - y0, x1 - x0);
  for (int64_t y = 0; y < out.height; ++y)
    std::copy_n(src.at(y0 + y, x0), static_cast<size_t>(out.width) * 3, out.at(y, 0));
  return out;
}

void resize_bilinear_f32(const float *src, int64_t sh, int64_t sw, float *dst,
                         int64_t dh, int64_t dw, int64_t ch) {
  if (!src || !dst || sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 || ch <= 0)
    throw std::runtime_error("resize_bilinear_f32: " + std::to_string(dh) + "x" +
                             std::to_string(dw) + " from a " +
                             std::to_string(sh) + "x" + std::to_string(sw) +
                             " source");
  const double fy_scale = static_cast<double>(sh) / dh;
  const double fx_scale = static_cast<double>(sw) / dw;
  const int64_t cn = ch;
  for (int64_t y = 0; y < dh; ++y) {
    const double fy = (y + 0.5) * fy_scale - 0.5;
    const double fy0f = std::floor(fy);
    const int64_t y0 = static_cast<int64_t>(fy0f);
    const double wy = fy - fy0f;
    const int64_t y0c = std::min<int64_t>(std::max<int64_t>(y0, 0), sh - 1);
    const int64_t y1c = std::min<int64_t>(std::max<int64_t>(y0 + 1, 0), sh - 1);
    const float *const ra = src + static_cast<size_t>(y0c) * sw * cn;
    const float *const rb = src + static_cast<size_t>(y1c) * sw * cn;
    float *const d = dst + static_cast<size_t>(y) * dw * cn;
    for (int64_t x = 0; x < dw; ++x) {
      const double fx = (x + 0.5) * fx_scale - 0.5;
      const double fx0f = std::floor(fx);
      const int64_t x0 = static_cast<int64_t>(fx0f);
      const double wx = fx - fx0f;
      const int64_t x0c = std::min<int64_t>(std::max<int64_t>(x0, 0), sw - 1);
      const int64_t x1c = std::min<int64_t>(std::max<int64_t>(x0 + 1, 0), sw - 1);
      const float *const a0 = ra + static_cast<size_t>(x0c) * cn;
      const float *const a1 = ra + static_cast<size_t>(x1c) * cn;
      const float *const b0 = rb + static_cast<size_t>(x0c) * cn;
      const float *const b1 = rb + static_cast<size_t>(x1c) * cn;
      float *const o = d + static_cast<size_t>(x) * cn;
      for (int64_t k = 0; k < cn; ++k)
        o[k] = static_cast<float>(
            (static_cast<double>(a0[k]) * (1.0 - wx) + a1[k] * wx) * (1.0 - wy) +
            (static_cast<double>(b0[k]) * (1.0 - wx) + b1[k] * wx) * wy);
    }
  }
}

void warp_affine_f32(const float *src, int64_t sh, int64_t sw, float *dst,
                     int64_t dh, int64_t dw, const double m[6], int64_t ch) {
  if (!src || !dst || sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 || ch <= 0)
    throw std::runtime_error("warp_affine_f32: degenerate sizes");
  const double a = m[0], b = m[1], tx = m[2];
  const double c = m[3], d = m[4], ty = m[5];
  const double det = a * d - b * c;
  if (det == 0.0)
    throw std::runtime_error("warp_affine_f32: the affine is singular");
  const double ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
  const double itx = -(ia * tx + ib * ty);
  const double ity = -(ic * tx + id * ty);
  const int64_t cn = ch;
  for (int64_t y = 0; y < dh; ++y) {
    float *const row = dst + static_cast<size_t>(y) * dw * cn;
    for (int64_t x = 0; x < dw; ++x) {
      const double fx = ia * x + ib * y + itx;
      const double fy = ic * x + id * y + ity;
      float *const o = row + static_cast<size_t>(x) * cn;
      if (fx < 0.0 || fx > sw - 1 || fy < 0.0 || fy > sh - 1) {
        // border=0, and the zoo's border IS 0 rather than a clamp: clamping
        // would carry edge values of the rotated square out to the corners.
        for (int64_t k = 0; k < cn; ++k) o[k] = 0.f;
        continue;
      }
      const double fy0f = std::floor(fy), fx0f = std::floor(fx);
      const int64_t y0 = static_cast<int64_t>(fy0f), x0 = static_cast<int64_t>(fx0f);
      const double wy = fy - fy0f, wx = fx - fx0f;
      const int64_t y0c = std::min<int64_t>(std::max<int64_t>(y0, 0), sh - 1);
      const int64_t y1c = std::min<int64_t>(std::max<int64_t>(y0 + 1, 0), sh - 1);
      const int64_t x0c = std::min<int64_t>(std::max<int64_t>(x0, 0), sw - 1);
      const int64_t x1c = std::min<int64_t>(std::max<int64_t>(x0 + 1, 0), sw - 1);
      const float *const pa = src + static_cast<size_t>(y0c) * sw * cn + static_cast<size_t>(x0c) * cn;
      const float *const pb = src + static_cast<size_t>(y0c) * sw * cn + static_cast<size_t>(x1c) * cn;
      const float *const pc = src + static_cast<size_t>(y1c) * sw * cn + static_cast<size_t>(x0c) * cn;
      const float *const pd = src + static_cast<size_t>(y1c) * sw * cn + static_cast<size_t>(x1c) * cn;
      for (int64_t k = 0; k < cn; ++k)
        o[k] = static_cast<float>(
            (static_cast<double>(pa[k]) * (1.0 - wx) + pb[k] * wx) * (1.0 - wy) +
            (static_cast<double>(pc[k]) * (1.0 - wx) + pd[k] * wx) * wy);
    }
  }
}

Image pad_zeros(const Image &src, int64_t top, int64_t bottom, int64_t left,
                int64_t right) {
  if (top < 0 || bottom < 0 || left < 0 || right < 0)
    throw std::runtime_error("pad_zeros: negative border " + std::to_string(top) +
                             "," + std::to_string(bottom) + "," +
                             std::to_string(left) + "," + std::to_string(right));
  Image out = like(src.height + top + bottom, src.width + left + right);
  for (int64_t y = 0; y < src.height; ++y)
    std::copy_n(src.at(y, 0), static_cast<size_t>(src.width) * 3,
                out.at(y + top, left));
  return out;
}

std::vector<float> to_nchw_normalised(const Image &src,
                                      const std::vector<float> &mean,
                                      const std::vector<float> &std_dev) {
  if (src.empty())
    throw std::runtime_error("to_nchw_normalised: the image is empty");
  if (mean.size() != 3 || std_dev.size() != 3)
    throw std::runtime_error("to_nchw_normalised: the container declares " +
                             std::to_string(mean.size()) + " means and " +
                             std::to_string(std_dev.size()) + " standard "
                             "deviations, and this architecture has three "
                             "channels");
  // The std check is HERE, once, rather than inside the loop: with three channels
  // a worker reaching it is three divisions per pixel of a silent infinity, and a
  // divide by zero here is a channel that reaches the first convolution as a NaN
  // rather than as an error.
  for (size_t c = 0; c < 3; ++c)
    if (std_dev[c] == 0.0)
      throw std::runtime_error("to_nchw_normalised: image_std[" +
                               std::to_string(c) +
                               "] is zero, so the normalisation is not invertible "
                               "and no input can be recovered");
  const int64_t plane = src.height * src.width;
  std::vector<float> out(static_cast<size_t>(3) * static_cast<size_t>(plane));
  for (int c = 0; c < 3; ++c) {
    const double mu = static_cast<double>(mean[static_cast<size_t>(c)]);
    const double sd = static_cast<double>(std_dev[static_cast<size_t>(c)]);
    float *const d = out.data() + static_cast<size_t>(c) * plane;
    for (int64_t p = 0; p < plane; ++p) {
      const double v = static_cast<double>(src.rgb[static_cast<size_t>(p) * 3 + c]) / 255.0;
      d[p] = static_cast<float>((v - mu) / sd);
    }
  }
  return out;
}

}  // namespace npue::raster
