//===- classify.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one image-classification session: the device, ONE design set
// (gemm_rtp, the same one bge-base-en-v1.5 uses), the staged weights, and the
// host head.
//
// WHY ONE DESIGN SET AND NOT TWO
// ------------------------------
// Whisper needs gemm_rtp and gemm_rtp_dec because its decoder has seven more
// streams than its encoder. A ViT has neither: the four streams it needs are
// bge-base's, byte for byte, and the patch embedding RIDES attn_out's. So
// `kinds.cls` in tools/data/npu_targets.json lists gemm_rtp's stream list
// verbatim, and this session resolves artifacts the same way the embeddings path
// does rather than through a "which two sets" search that would have nothing to
// search for. That is also why a classifier costs one xclbin and one
// hw_context, and why `list` needs no second export instruction for it.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/encoder.hpp"
#include "vit/geometry.hpp"
#include "vit/image.hpp"   // npue::vit::Image -- what classify() takes

namespace npue::vit {

// One image's answer.
struct Prediction {
  int64_t label = 0;             // the argmax
  float top1 = 0.f;              // softmax probability of the argmax
  std::vector<float> logits;     // every label, in id order
  double front_end_s = 0.0;      // decode + resize + normalise + im2col
  double encoder_s = 0.0;
  double head_s = 0.0;
  int64_t n_dispatch = 0;
};

class Session {
public:
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads);

  const Geometry &geometry() const { return geom_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  // The label names, in id order, from `labels.table` -- one newline-separated
  // name per label, so a caller can print "tabby cat" instead of 281.
  const std::vector<std::string> &labels() const { return labels_; }
  int64_t n_dispatch() const { return enc_.gemm().n_dispatch; }
  bool int8() const { return enc_.int8(); }
  // The stage names the loaded set carries, in slot order, for the status line.
  const std::vector<std::string> &stream_ops() const { return ops_; }
  int64_t rows_per_dispatch() const { return rows_; }
  size_t staged_bytes() const { return staged_; }

  // One image, already decoded and resized to the container's own geometry.
  // This is the function a test drives with a raster it built itself, which is
  // why it is separated from classify_file().
  Prediction classify(const npue::vit::Image &im);

  Prediction classify_file(const std::string &path);

  // The patches only, for the host-side gates that do not want a device.
  static std::vector<float> preprocess_file(const std::string &path,
                                            const Geometry &g);

private:
  npue::File &model_;
  Geometry geom_;
  std::string art_, name_;
  std::vector<std::string> labels_;
  // Declaration order is construction order, and it matters: the encoder holds
  // REFERENCES to a Design and a Pool, so the device has to exist before the
  // design and the design before the encoder.
  std::unique_ptr<npu::Device> dev_;
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<npu::Design> design_;
  VitEncoder enc_;
  std::vector<std::string> ops_;
  int64_t rows_ = 0;
  size_t staged_ = 0;
};

// ONE JSON spelling of a classification result, for both callers.
//
// `npuembeddings classify ... --json` and POST /v1/classify answer the same
// object, byte for byte, because they call the same function. Same reasoning as
// pose/result_json.hpp: the CLI's output is what a reader compares by eye and the
// endpoint's is what a program parses, and two emitters would surface as "the
// server and the CLI disagree" -- a question nobody can answer without diffing two
// documents by hand.
//
// `top_k` is the number of RUNNERS-UP to include, so 1 means the argmax alone. 0 is
// refused by the caller rather than here, because "the top 0 labels" is a
// malformed request and an empty array is a valid answer to it.
//
// `image_label` is a label, not a path: the server has no path for an upload.
std::string prediction_json(const Prediction &p, const std::string &label_name,
                            const std::string &image_label, int64_t top_k);

}  // namespace npue::vit
