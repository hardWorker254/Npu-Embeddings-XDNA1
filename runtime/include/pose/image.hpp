//===- image.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=6 image front end: decode, letterbox, normalise.
// It produces the [3, S, S] tensor the first convolution consumes, AND the
// affine that takes a keypoint back to the pixels the user handed in.
//
// WHY LETTERBOX AND NOT ViT's SQUARE RESIZE
// ------------------------------------------
// ViT resizes to a square because classification does not care where the person
// is. Pose is not classification: every keypoint is a PIXEL COORDINATE in the
// original frame, so the front end has to remember exactly how it moved them.
// Squashing a 16:9 photograph into 640x640 does not scale uniformly, and a
// decode that assumes it did would place a nose at 1.78x its true offset -- a
// pose that looks almost right, which is the worst kind of wrong. So this
// letterboxes: one uniform scale, then padding, and both are recorded here and
// inverted in decode.hpp.
//
// The pad VALUE is also part of the model, not a style choice: Ultralytics fills
// with the grey 114, and a front end that filled with 0 would feed the network a
// border it was never trained on, which moves the pyramid's edge responses and
// with them the keypoints nearest the margin.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

#include "pose/geometry.hpp"
#include "vit/image.hpp"   // npue::vit::Image, decode_image

namespace npue::pose {

// The uniform scale and padding that letterbox applied, so decode can invert it.
//
//   src_pixel = model_pixel / scale - pad
//
// `scale` is the same number in both axes -- that is what "uniform" means here,
// and it is why a keypoint's aspect ratio survives the round trip. `pad` is the
// top-left offset of the letterboxed image inside the padded canvas, in model
// pixels, and is the SAME on both axes because the target is square.
struct Letterbox {
  double scale = 1.0;   // model px per source px
  int64_t pad_x = 0;    // model px
  int64_t pad_y = 0;    // model px
  int64_t src_w = 0;
  int64_t src_h = 0;
  int64_t dst = 0;      // the padded canvas side
};

// The grey Ultralytics pads with, as a 0..1 value. Recorded rather than assumed
// only in the sense that the packer can disagree; 114/255 is the value.
inline constexpr float kPadValue = 114.f;

// Resize `src` by `scale` with BILINEAR, then pad to side x side with kPadValue.
//
// The resize is PIL-equivalent BILINEAR, reusing vit::resize's resampler
// through the same scale-adaptive support, and it is NOT cv2.resize: those
// differ by up to one 8-bit level on a downscale, and after /255 and 200
// convolutions that is a pixel of keypoint drift nobody would attribute to the
// resampler. What is claimed here is equivalence to the resize the reference
// pipeline performs, not to a particular imaging library's name.
//
// ALREADY SQUARE AND ALREADY THE RIGHT SIZE: returned unscaled, still padded
// (so pad_x/pad_y are 0 and scale 1). A canvas of the target size is the
// identity, and running a filter over it would be a difference of about half an
// LSB on a pipeline that did not ask for one -- the same rule vit::resize_square
// follows.
std::vector<float> letterbox_normalise(const npue::vit::Image &src,
                                       const Geometry &g, Letterbox &lb);

// decode -> letterbox -> normalise, for one file. Refuses a format
// vit::decode_image refuses, by the same rules and for the same reason.
std::vector<float> preprocess_file(const std::string &path, const Geometry &g,
                                   Letterbox &lb);

}  // namespace npue::pose
