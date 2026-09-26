//===- npu_ops.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the shared Whisper NPU layer: one GEMM wrapper and the three
// host passes (LayerNorm, GELU, attention).
//
// WHY THIS EXISTS INSTEAD OF AN INHERITANCE
// ------------------------------------------
// BertEncoder already solves "dispatch a GEMM, add a bias, read bf16 back" --
// and the two stacks here are the same arithmetic with a different schedule
// around it. What BertEncoder must not be inherited is its SCHEDULE: it is
// post-LN, batched over `batch * seq` rows, takes its LayerNorm and GELU from
// shared eltwise designs, and its attention is one fixed [seq, seq] score
// matrix. Whisper is pre-LN, runs the encoder in fixed-row chunks of a
// 1500-position tensor, has no eltwise designs in its design sets at all, and
// its decoder attends one new row against a KV cache of growing length. So the
// arithmetic is shared and the schedule is not, and BertEncoder is left alone.
//
// Every number here is read from the container or from the design. A tile that
// is not a multiple of the design's row count is refused, not padded silently.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"

namespace npue::whisper {

// One NPU GEMM, plus the two buffers that belong to it.
//
// A Design owns one A buffer and one C buffer. The weights are the DESIGN's
// (staged once, shared); the A and C buffers are this instance's, so two
// instances on one design -- two decoder lanes, or an encoder and a decoder on
// the same xclbin -- cannot write each other's rows.
class NpuGemm {
public:
  NpuGemm(npu::Design &design, app::Pool &pool) : d_(design), pool_(pool) {}

  // Must be called once, before the first run(). Allocates the A and C slots
  // at the design's own buffer sizes rather than at some width this class
  // guessed: the design is what decides how many rows one dispatch computes.
  void alloc_buffers();

  // Stage a [K, N] bf16 pre-tiled operand from the container. The recorded
  // layout_hash must be the one the design consumes -- the cross-generation
  // packing failure is invisible everywhere else (same byte count, same shapes,
  // matching hash on both sides, plausible numbers), and this is the only
  // place both sides are in the same expression.
  size_t stage_operand(const npue::File &model, const std::string &name);

  // out[n_real, N] = A[n_real, k] @ B + bias, computed at the stream's own row
  // count `rows`.
  //
  // Rows [n_real, rows) of A are zero-filled rather than left holding the
  // previous dispatch's values: the design computes all `rows` rows whatever A
  // holds there, and a stale row would put another call's activations into this
  // one's output. The tail is a padded row, so zero is the only value that
  // cannot be mistaken for a real position.
  void run(size_t instr, const float *a, int64_t n_real, int64_t rows, int64_t k,
           size_t wslot, const float *bias, int64_t n, float *out);

  npu::Design &design() { return d_; }
  size_t slot_a = 0, slot_c = 0;
  std::mutex *npu_mu = nullptr;
  int64_t n_dispatch = 0;
  double t_dispatch = 0.0;
  double t_convert = 0.0;

private:
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool_.size() == 1) { f(int64_t(0), n); return; }
    pool_.run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  npu::Design &d_;
  app::Pool &pool_;
};

// LayerNorm over rows of `d` columns, in place, with the container's epsilon.
// Whisper's is 1e-5, not BERT's 1e-12, and the difference is visible in the
// last bits of a 32-layer stack.
void layernorm_rows(float *x, int64_t n_rows, int64_t d, const float *gamma,
                    const float *beta, double eps, app::Pool &pool);

// exact-erf GELU, in place. The activation is the model's, not a choice: the
// container records `gelu` and tools/packers/whisper.py refuses a checkpoint
// whose activation is not this one.
void gelu_erf_inplace(float *x, size_t n, app::Pool &pool);

// Attention over a K|V block, with the queries in a separate operand.
//
// `q` is [n_q, d_model] with row stride `q_stride`; `kv` is [n_kv, 2*d_model]
// (K first, then V) with row stride `kv_stride`. The strides are what let one
// function serve all three of Whisper's attentions: the encoder's fused
// [Q|K|V] operand is a q block and a kv block with stride 3*d_model, the
// decoder's self-attention cache is [K|V] with stride 2*d_model, and the
// cross-attention cache is the same shape built once from the encoder output.
//
// `scale` is 1.0 when the packer folded 1/sqrt(head_dim) into the Q weight and
// the Q bias, and the real attention scale when it did not. It is applied here
// rather than folded again, because folding it twice is a different model.
//
// `scores` is [n_q, heads, n_kv] scratch the caller owns -- the encoder's is
// 6 x 1500 x 1500 floats for whisper-tiny and 20 x 1500 x 1500 for large-v3,
// which is why it is passed in rather than allocated per call. `out` is
// [n_q, d_model].
void attention(const float *q, int64_t q_stride, const float *kv,
               int64_t kv_stride, int64_t n_q, int64_t n_kv, int64_t d_model,
               int64_t heads, int64_t head_dim, float scale, float *out,
               float *scores, app::Pool &pool);

}  // namespace npue::whisper
