//===- logits_npu.cpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the vocabulary projection as a GEMM. See logits_npu.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/logits_npu.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "common/host_kernels.hpp"   // now_s

namespace npue::whisper {

void NpuLogits::set_streams(const std::vector<size_t> &slots, int64_t rows,
                             int64_t chunk_n) {
  slots_ = slots;
  rows_ = rows;
  chunk_n_ = chunk_n;
  if (rows <= 0)
    throw std::runtime_error("whisper logits: the design set's logits rows are "
                             "zero, so no dispatch has a row count");
  if (static_cast<int>(slots_.size()) != n_chunks_)
    throw std::runtime_error("whisper logits: the design set has " +
                             std::to_string(slots_.size()) +
                             " logits streams and this run wants " +
                             std::to_string(n_chunks_) +
                             ". The chunk count comes from tools/npu_ops.py; "
                             "re-export the set so the two agree.");
}

void NpuLogits::alloc_buffers(const float *embed) {
  if (!d_)
    throw std::runtime_error("whisper logits: set_design() before alloc_buffers()");
  if (!embed)
    throw std::runtime_error("whisper logits: the container carries no tied "
                             "token embedding");
  const auto &in = d_->info();
  if (chunk_n_ <= 0)
    throw std::runtime_error("whisper logits: the chunk width is zero; the "
                             "design set's logits_i streams record it");
  // One GEMM per chunk, because a Design binds ONE B operand per dispatch and
  // the eight panels are eight different pieces of the vocabulary.
  gems_.clear();
  wslots_.clear();
  for (int c = 0; c < n_chunks_; ++c) {
    gems_.push_back(std::make_unique<NpuGemm>(*d_, pool_));
    gems_.back()->alloc_buffers();
  }
  bias_.assign(static_cast<size_t>(chunk_n_), 0.0f);
  scratch_.assign(static_cast<size_t>(rows_) * chunk_n_, 0.0f);
  mat_.assign(static_cast<size_t>(hidden_) * chunk_n_, 0.0f);

  // The tied embedding, transposed and tiled once per chunk. Zeroed past this
  // chunk's share of the vocabulary, so the last chunk of a vocabulary that is
  // not a multiple of the chunk produces no ids past its end.
  for (int c = 0; c < n_chunks_; ++c) {
    const int64_t lo = static_cast<int64_t>(c) * chunk_n_;
    const int64_t hi = std::min<int64_t>(vocab_, lo + chunk_n_);
    std::fill(mat_.begin(), mat_.begin() + static_cast<long>(hidden_ * chunk_n_),
              0.0f);
    for (int64_t k = 0; k < hidden_; ++k)
      for (int64_t v = lo; v < hi; ++v)
        mat_[static_cast<size_t>(k) * chunk_n_ + (v - lo)] =
            embed[static_cast<size_t>(v) * hidden_ + k];
    const std::vector<uint16_t> tiled =
        tile_b_panel(mat_.data(), hidden_, chunk_n_, in.b_tile_k, in.tile_n,
                     in.b_mac_s, in.b_mac_t);
    wslots_.push_back(d_->stage(1, tiled.data(), tiled.size() * sizeof(uint16_t)));
  }
}

void NpuLogits::run(const float *h, float *out) {
  for (int c = 0; c < n_chunks_; ++c) {
    const int64_t lo = static_cast<int64_t>(c) * chunk_n_;
    const int64_t hi = std::min<int64_t>(vocab_, lo + chunk_n_);
    if (lo >= vocab_) break;
    gems_[c]->run(slots_[c], h, 1, rows_, hidden_, wslots_[c], bias_.data(),
                  chunk_n_, scratch_.data());
    for (int64_t v = lo; v < hi; ++v)
      out[v] = scratch_[v - lo];
    ++n_dispatch;
  }
}

std::string NpuLogits::note() const {
  return std::to_string(n_chunks_) + " chunks of " + std::to_string(chunk_n_) +
         " columns (" + std::to_string(vocab_) +
         " ids), K = " + std::to_string(hidden_) + ", " +
         std::to_string(n_chunks_) +
         " dispatches per token, " +
         std::to_string(static_cast<int64_t>(n_chunks_) * hidden_ * chunk_n_ * 2 /
                        (1024 * 1024)) +
         " MB of staged bf16 panels";
}

}  // namespace npue::whisper
