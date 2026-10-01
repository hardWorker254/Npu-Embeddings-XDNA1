//===- image.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image front end for an arch=5 container: decode, resize,
// normalise, im2col. It produces the [n_patches, patch_dim] matrix that the
// patch-embedding GEMM consumes.
//
// THE FOUR STEPS ARE SEPARATE FUNCTIONS ON PURPOSE
// -----------------------------------------------
// Each one is a place where a "reasonable" shortcut produces a correctly shaped
// wrong answer, so each is separately callable and separately testable:
//
//   decode_image    format from the BYTES, not the extension
//   resize_square   PIL's resample, downscale antialiased
//   normalise       (x/255 - mean) / std, per channel, from the container
//   im2col_patches  [3,S,S] -> [n_patches, 3*patch*patch], the twin of
//                   packers.vit.patch_embed_operand
//
// im2col_patches is the PIXEL half of the patch layout and patch_embed_operand
// is the WEIGHT half. They live in different modules in different languages and
// that is deliberate: the composition
//
//     im2col_patches(px) @ patch_embed_operand(W) == Conv2d(W)(px)
//
// is what tools/verify/verify_vit.py holds against HF's own convolution, which
// is the only thing in the tree that knows both halves at once. A drift between
// them would still be a well-formed GEMM, so it would not fail anywhere -- the
// model would confidently classify noise.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vit/geometry.hpp"

namespace npue::vit {

// An 8-bit interleaved RGB raster. `rgb` is width*height*3, row-major.
struct Image {
  int64_t width = 0, height = 0;
  std::vector<uint8_t> rgb;

  bool empty() const { return width <= 0 || height <= 0 || rgb.empty(); }
  const uint8_t *at(int64_t y, int64_t x) const {
    return rgb.data() + (y * width + x) * 3;
  }
};

// Decode a PNG or a JPEG into RGB.
//
// PNG and JPEG only, and both are DECIDED BY MAGIC BYTES: a `.png` that holds a
// JPEG is common enough in the wild (a browser's <img src> renamed, an
// exporter's bug) that a decoder keyed on the extension returns an error the
// user cannot act on, or worse, succeeds on the wrong bytes. Grayscale, palette,
// grayscale+alpha and 16-bit-per-channel PNGs and a grayscale JPEG are all
// handled: the first three become RGB by replication, which is what PIL's
// convert("RGB") does and therefore what the model expects; the fourth is
// REFUSED BY NAME, because two correct-looking 16->8 reductions disagree and one
// of them is what transformers feeds the model. A four-channel PNG is
// composited onto white rather than dropped, because a transparent photograph
// pasted onto a transparent background is white paper in every viewer the user
// has.
//
// REFUSES on anything else. Not "best effort": a runtime that guesses at an
// image format reports a label for a picture it may never have seen.
Image decode_image(const std::string &path);
Image decode_png(const std::vector<uint8_t> &bytes, const std::string &name);
Image decode_jpeg(const std::vector<uint8_t> &bytes, const std::string &name);

// PIL.Image.resize((side, side), resample), i.e. the resize transformers' ViT
// image processor asks for, computed in float and rounded back to 8 bits.
//
// WHY THIS IS NOT "bilinear, it is close enough". PIL's filter is a SEQUENTIAL
// resample with a scale-adaptive support -- when downscaling, the kernel is
// widened by the shrink factor so it low-passes instead of point-sampling. That
// widening is the whole difference between a correct downscale and a moire of
// aliased pixels, and 1/std then multiplies the error by 2. So this implements
// the widening, for the codes PIL names and this build can MATCH, and refuses the
// ones it cannot.
//
// WHICH CODES, AND WHY THE LIST IS SHORTER THAN PIL'S
// --------------------------------------------------
//   LANCZOS (1)  BILINEAR (2)  BICUBIC (3)  BOX (4)   implemented
//   NEAREST (0)  HAMMING (5)                 refused by name
//
// Both refusals are the same shape and the same reason: this function claims
// PIL-EQUIVALENCE, not "a smooth interpolation", and a claim that is not
// measured is not made. NEAREST has no widening to speak of and point-samples a
// downscale. HAMMING is implemented-and-measured-and-WRONG: it is named here,
// not here-and-approximated, because the first version of this file spelled all
// five codes as one Keys cubic with a `support` and an `a`, which was correct
// for BICUBIC alone and off by up to 65 levels for the others. Fixing that was
// easy for four of them -- BOX is a rectangle, BILINEAR a triangle, LANCZOS a
// windowed sinc, and all three are now transcribed from PIL's own Filters.c.
// HAMMING is not one of those shapes, and PIL's measured impulse responses for
// it match no raised cosine this build could fit. So it is refused, and the
// refusal says which four are held to what. A resampler that is CLOSE to PIL is
// the failure this project treats as worst: it produces a confident label about
// a picture the model was never shown.
//
// An image already side x side is returned UNCHANGED, because PIL skips the
// filter entirely in that case and running one would be a difference of about
// half an LSB on a pipeline that did not ask for one.
//
// The claim is measured, not assumed: runtime/tests/test_vit_image.cpp renders
// the same rasters through this and through PIL and reports the worst channel
// difference and the fraction of channels that differ at all.
// tools/verify/verify_vit_image.py section 2 gates all four codes: worst
// difference at or below one 8-bit LSB, and EXACTLY zero on a constant raster,
// which every one of these filters is the identity on. It is not claimed to be
// bit-identical elsewhere, because PIL's 8-bit path is fixed-point with 22 bits
// of fractional precision and this one accumulates in float64.
Image resize_square(const Image &src, int64_t side, Resample how);

// [0,255] RGB -> [3, S, S] fp32, (x/255 - mean[c]) / std[c].
//
// The mean and std are the CONTAINER's, read from the checkpoint's own
// preprocessor_config.json at pack time. A caller cannot pass its own, because
// there is no such thing as a correct one that is not the checkpoint's.
std::vector<float> normalise(const Image &im, const Geometry &g);

// [3,S,S] fp32 -> [n_patches, patch_dim] fp32.
//
// Channel-major outside, row-major inside: (c, i, j) maps to (c*patch + i)*patch
// + j, which is the k index packers.vit.patch_embed_operand builds. Rows come
// out in (patch_row, patch_col) order. Written as an explicit loop rather than
// a memcpy because the opposite ordering is a well-formed matrix of the right
// shape containing a permuted image.
std::vector<float> im2col_patches(const float *chw, const Geometry &g);

// decode -> resize -> normalise -> im2col, the whole front end for one file.
std::vector<float> preprocess_file(const std::string &path, const Geometry &g);
std::vector<float> preprocess(const Image &im, const Geometry &g);

}  // namespace npue::vit
