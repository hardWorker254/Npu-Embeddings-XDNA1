//===- geometry.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- ViT model geometry, read from the container. The second
// modality in this tree (pixels, not tokens) and the first pre-LN one.
//
// WHY A SEPARATE GEOMETRY AND NOT A BERT ONE WITH FIELDS TURNED OFF
// ------------------------------------------------------------------
// A ViT's encoder and BERT's differ in three places that are all silent when
// you get one of them wrong: the normalisation ORDER (pre-LN against BERT's
// post-LN), the presence of a CLS row that is a prepended VECTOR rather than a
// token id, and the fact that the input is [n_patches, patch_dim] produced by
// im2col instead of [batch, seq] looked up in a table. Every one of those
// produces a correctly shaped, entirely wrong answer. So this is its own struct,
// read from its own keys, refused when a key is missing -- and the encoder that
// consumes it is npue::vit::VitEncoder, not BertEncoder.
//
// THE IMAGE FRONT END IS PART OF THE GEOMETRY
// -------------------------------------------
// image_size, patch_size, num_channels, the resize geometry and the mean/std
// are all here because they are all part of the MODEL: a container normalised
// with another checkpoint's mean/std classifies noise, and nothing downstream
// can tell. The packer reads them out of the checkpoint's own
// preprocessor_config.json and refuses rather than defaulting (tools/pack/
// packers/vit.py), so this struct is a transcription of a decision that was
// already made -- it never invents one. A missing key throws.
//
// XRT-free, like the rest of the geometry, so a host-only check can read it.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/model.hpp"
#include "common/json_min.hpp"   // npue::json::parse, for image_mean / image_std

namespace npue::vit {

// PIL.Image's resample codes, as transformers stores them. Named because the
// container records a NUMBER and a consumer that implements a subset has to be
// able to say which ones it has -- and which it does not.
//
// FOUR OF THE SIX ARE IMPLEMENTED, and NEAREST and HAMMING are REFUSED BY NAME
// in vit/image.cpp's resize_square(). The enum lists all six because refusing to
// parse code 5 would be a different failure from refusing to implement it: a
// checkpoint asking for something this build cannot match must be named, not
// misread. See vit/image.hpp for which four are measured against PIL and to
// what.
enum class Resample {
  Nearest = 0,
  Lanczos = 1,
  Bilinear = 2,
  Bicubic = 3,
  Box = 4,
  Hamming = 5,
};

struct Geometry {
  int64_t d_model = 0;
  int64_t heads = 0;
  int64_t head_dim = 0;
  int64_t layers = 0;
  int64_t intermediate = 0;
  int64_t num_labels = 0;

  // The image front end, verbatim from preprocessor_config.json.
  int64_t image_size = 0;
  int64_t crop_size = 0;
  int64_t patch_size = 0;
  int64_t num_channels = 0;
  int64_t patch_dim = 0;
  int64_t n_patches = 0;
  std::vector<float> mean, std_dev;
  Resample resample = Resample::Bicubic;

  int64_t n_pos = 0;             // n_patches + 1: the CLS row is a position
  double ln_eps = 1e-12;
  double attn_scale = 0.0;       // 1/sqrt(head_dim)
  bool qkv_scale_folded = true;  // 1/sqrt(head_dim) folded into every Q operand
};

// `label` is the container's path, used only to make the refusal name the file
// it came from.
inline int64_t need_int(const npue::File &f, const std::string &key,
                        const std::string &label) {
  try {
    return f.config_int(key);
  } catch (const std::exception &e) {
    throw std::runtime_error(label + ": " + e.what() +
                             " -- not a ViT container, or one packed by an older "
                             "packer (arch " + f.config_string("arch") + ")");
  }
}

inline std::vector<double> need_floats(const npue::File &f,
                                       const std::string &key, int64_t want,
                                       const std::string &label) {
  const std::string raw = f.config_string(key);
  npue::json::Value v;
  try {
    v = npue::json::parse(raw);
  } catch (const std::exception &) {
    throw std::runtime_error(label + ": " + key + " is not readable JSON: " +
                             raw);
  }
  std::vector<double> out;
  for (const auto &e : v.as_array()) out.push_back(e.as_number());
  if (static_cast<int64_t>(out.size()) != want)
    throw std::runtime_error(
        label + ": " + key + " has " + std::to_string(out.size()) +
        " entries and this image has " + std::to_string(want) +
        " channels. The normalisation is per channel, so a length that does not "
        "match has no per-pixel meaning.");
  return out;
}

inline Resample need_resample(const npue::File &f, const std::string &label) {
  const int64_t code = need_int(f, "resample", label);
  if (code < 0 || code > 5)
    throw std::runtime_error(label + ": resample " + std::to_string(code) +
                             " is not a PIL.Image code (0 NEAREST, 1 LANCZOS, "
                             "2 BILINEAR, 3 BICUBIC, 4 BOX, 5 HAMMING). "
                             "Refusing rather than guessing which filter the "
                             "checkpoint asked for.");
  return static_cast<Resample>(code);
}

inline Geometry read_geometry(const npue::File &f,
                              const std::string &label = std::string()) {
  const std::string arch = f.config_string("arch");
  if (arch != "vit_patch16_prenorm_gelu")
    throw std::runtime_error(label + ": arch is '" + arch +
                             "', not 'vit_patch16_prenorm_gelu'");
  Geometry g;
  g.d_model = need_int(f, "hidden", label);
  g.heads = need_int(f, "num_heads", label);
  g.head_dim = need_int(f, "head_dim", label);
  g.layers = need_int(f, "num_layers", label);
  g.intermediate = need_int(f, "intermediate", label);
  g.num_labels = need_int(f, "num_labels", label);
  g.image_size = need_int(f, "image_size", label);
  g.crop_size = need_int(f, "crop_size", label);
  g.patch_size = need_int(f, "patch_size", label);
  g.num_channels = need_int(f, "num_channels", label);
  g.patch_dim = need_int(f, "patch_dim", label);
  g.n_patches = need_int(f, "n_patches", label);
  g.n_pos = need_int(f, "max_seq_len", label);
  g.resample = need_resample(f, label);

  // The resize GEOMETRY, refused rather than approximated. "square" is a resize
  // to image_size x image_size with nothing after it; HF's other form, a
  // shortest-edge resize plus a centre crop, produces a DIFFERENT tensor out of
  // the same photograph, and this runtime does not implement it. A container
  // that named it would be a container this build must decline, not one it may
  // resize "well enough".
  const std::string geom = f.config_string("image_resize");
  if (geom != "square")
    throw std::runtime_error(label + ": image_resize is '" + geom +
                             "', and this build implements 'square' only. A "
                             "shortest-edge resize followed by a centre crop is "
                             "a different pixel tensor, not a different quality, "
                             "so it is refused rather than approximated. Repack "
                             "from a checkpoint whose preprocessor_config.json "
                             "has a square size.");

  for (double m : need_floats(f, "image_mean", g.num_channels, label))
    g.mean.push_back(static_cast<float>(m));
  for (double s : need_floats(f, "image_std", g.num_channels, label))
    g.std_dev.push_back(static_cast<float>(s));
  for (int64_t c = 0; c < g.num_channels; ++c)
    if (g.std_dev[static_cast<size_t>(c)] == 0.f)
      throw std::runtime_error(label + ": image_std[" + std::to_string(c) +
                               "] is zero; the normalise step would divide by "
                               "zero and make a NaN of every pixel of that "
                               "channel.");

  g.ln_eps = f.config_double("layer_norm_eps");
  g.attn_scale = f.config_double("attention_scale");
  g.qkv_scale_folded = f.config_string("qkv_scale_folded") != "false";

  // -- the refusals, all of which are things a "reasonable default" would get
  // wrong rather than merely approximate.
  if (g.heads <= 0 || g.head_dim <= 0 || g.heads * g.head_dim != g.d_model)
    throw std::runtime_error(
        label + ": " + std::to_string(g.heads) + " heads x " +
        std::to_string(g.head_dim) + " is not hidden " +
        std::to_string(g.d_model) +
        ". HF uses head_dim = hidden // n_heads with integer division, so a "
        "pair that does not multiply back is a real truncated head and every "
        "attention scale in the model would be wrong.");
  if (std::abs(g.attn_scale - 1.0 / std::sqrt(static_cast<double>(g.head_dim))) >
      1e-6)
    throw std::runtime_error(label + ": attention_scale " +
                             std::to_string(g.attn_scale) + " is not "
                             "1/sqrt(head_dim) for head_dim " +
                             std::to_string(g.head_dim));
  if (g.patch_size <= 0 || g.num_channels <= 0 || g.image_size <= 0)
    throw std::runtime_error(label + ": image_size " +
                             std::to_string(g.image_size) + ", patch_size " +
                             std::to_string(g.patch_size) + ", num_channels " +
                             std::to_string(g.num_channels) +
                             " -- the patch count below divides by these.");
  if (g.image_size % g.patch_size)
    throw std::runtime_error(label + ": image_size " +
                             std::to_string(g.image_size) +
                             " is not a whole number of " +
                             std::to_string(g.patch_size) +
                             "-pixel patches. The convolution's stride equals "
                             "its kernel, so a partial patch at the right or "
                             "bottom edge is a different tensor, not a "
                             "truncated one.");
  if (g.patch_dim != g.num_channels * g.patch_size * g.patch_size)
    throw std::runtime_error(label + ": patch_dim " +
                             std::to_string(g.patch_dim) + " is not " +
                             std::to_string(g.num_channels) + " channels x " +
                             std::to_string(g.patch_size) + "^2. The GEMM's K "
                             "comes from this number, and a K that disagrees "
                             "with the image front end's row length dispatches "
                             "a matrix multiply over the wrong extent.");
  const int64_t side = g.image_size / g.patch_size;
  if (side * side != g.n_patches)
    throw std::runtime_error(label + ": n_patches " +
                             std::to_string(g.n_patches) + " is not (" +
                             std::to_string(g.image_size) + "/" +
                             std::to_string(g.patch_size) + ")^2");
  // The CLS row is a PREPENDED ROW, not an offset into the patch list, and it
  // has a position of its own. That is why n_pos is n_patches + 1 and why the
  // 196-vs-197 confusion is worth a refusal: the position table has n_pos rows
  // and a runtime that indexes it with a patch row number silently gives the
  // last patch no position.
  if (g.n_pos != g.n_patches + 1)
    throw std::runtime_error(label + ": max_seq_len " +
                             std::to_string(g.n_pos) + " is not n_patches + 1 = " +
                             std::to_string(g.n_patches + 1) +
                             ". A ViT has no [CLS] TOKEN: the CLS row is a "
                             "learned vector prepended to the patch rows, and "
                             "it holds a position like any of them.");
  if (g.crop_size != g.image_size)
    throw std::runtime_error(label + ": crop_size " +
                             std::to_string(g.crop_size) + " != image_size " +
                             std::to_string(g.image_size) +
                             ". image_resize is 'square', which has no crop.");
  if (g.num_labels <= 0)
    throw std::runtime_error(label + ": num_labels " +
                             std::to_string(g.num_labels) +
                             " -- the head's output range.");
  if (g.patch_dim != g.d_model)
    throw std::runtime_error(
        label + ": patch_dim " + std::to_string(g.patch_dim) +
        " is not hidden " + std::to_string(g.d_model) +
        ". The patch embedding is a [n_patches, patch_dim] x [patch_dim, hidden] "
        "GEMM and rides the attn_out stream, whose K is hidden; a patch_dim that "
        "disagrees is a design whose stream does not exist.");
  return g;
}

}  // namespace npue::vit
