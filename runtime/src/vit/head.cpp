//===- head.cpp ----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the ViT classifier head, on the host. arch=5.
// The contract, and why the stride is load-bearing, is in vit/head.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "vit/head.hpp"

#include <stdexcept>
#include <string>

namespace npue::vit {

void head_matvec(const float *head, const float *bias, const float *cls_row,
                 int64_t d_model, int64_t num_labels, int64_t j0, int64_t j1,
                 float *logits) {
  if (head == nullptr || cls_row == nullptr || logits == nullptr)
    throw std::runtime_error(
        "vit head: a null pointer reached head_matvec. `classifier.weight` is "
        "the one operand that never reaches the device, so nothing else would "
        "have noticed it was not read -- and reading label 0's weight as if it "
        "were all 1000 of them is a finite, confident, wrong answer.");
  if (d_model <= 0 || num_labels <= 0)
    throw std::runtime_error("vit head: d_model " + std::to_string(d_model) +
                             " and num_labels " + std::to_string(num_labels) +
                             " must both be positive");
  if (j0 < 0 || j1 > num_labels || j0 > j1)
    throw std::runtime_error(
        "vit head: label range [" + std::to_string(j0) + ", " +
        std::to_string(j1) + ") is not inside [0, " +
        std::to_string(num_labels) +
        "). Refused rather than clamped: clamping would silently leave the "
        "labels outside the block unwritten, and a logits buffer with holes in "
        "it is another confident wrong answer.");
  for (int64_t j = j0; j < j1; ++j) {
    double acc = bias != nullptr ? bias[j] : 0.0;
    // head[t * num_labels + j]: LABEL IS THE FAST AXIS. See vit/head.hpp.
    for (int64_t t = 0; t < d_model; ++t)
      acc += static_cast<double>(cls_row[t]) *
             static_cast<double>(head[t * num_labels + j]);
    logits[j] = static_cast<float>(acc);
  }
}

}  // namespace npue::vit
