//===- logits_npu.hpp --------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the vocabulary projection as a GEMM, in chunks. See
// logits_npu.cpp for the chunking arithmetic.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/pool.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

// logits = h @ embed_tokens^T, one dispatch per chunk of the vocabulary.
//
// The whole point of this class is that the operation it performs is NOT worth
// moving: 66 MB of tied embedding are read per generated token to do 66 MFLOP of
// work, which is a bandwidth problem wearing a GEMM's clothes. It exists because
// the question was which operations CAN run on the array, and this one can, at
// the cost of 41 MB of staged bf16 panels that the host's fp32 table does not
// need.
class NpuLogits {
public:
  NpuLogits(npu::Design &design, app::Pool &pool, int64_t hidden, int64_t vocab,
            int n_chunks)
      : pool_(pool), hidden_(hidden), vocab_(vocab), n_chunks_(n_chunks) {}

  // The chunk streams' instruction slots, the row count one dispatch covers and
  // the chunk width, all read off the loaded design set -- one slot per chunk,
  // in id order. The width is the design's own N: the runtime does not get to
  // choose how wide a chunk is, because a chunk is a stream and a stream's N is
  // its shape.
  void set_streams(const std::vector<size_t> &slots, int64_t rows,
                   int64_t chunk_n);
  void set_design(npu::Design *d) { d_ = d; }

  // One NpuGemm per chunk, each with its own A/C slots, and the tied embedding
  // staged as one B panel per chunk. `embed` is the container's fp32
  // [vocab, hidden] table, transposed and tiled here for the same reason the
  // convolutions' panels are: the panel order is the MMAC's, not the tensor's.
  void alloc_buffers(const float *embed);

  // `h` is one row of `hidden` floats; `out` receives all `vocab_` logits. The
  // suppression policy and the argmax are the CALLER's, and stay on the host:
  // they are a search over the vector, not a matrix.
  void run(const float *h, float *out);

  int64_t n_dispatch = 0;
  std::string note() const;

private:
  npu::Design *d_ = nullptr;
  app::Pool &pool_;
  int64_t hidden_ = 0, vocab_ = 0, rows_ = 0, chunk_n_ = 0;
  int n_chunks_ = 0;
  std::vector<size_t> slots_;
  std::vector<std::unique_ptr<NpuGemm>> gems_;
  std::vector<size_t> wslots_;
  std::vector<float> bias_, scratch_;
  std::vector<float> mat_;
  std::vector<uint16_t> tiled_;
};

}  // namespace npue::whisper
