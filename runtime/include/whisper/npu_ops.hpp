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
#include <unordered_map>
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

  // Stage a [K, N] pre-tiled operand from the container. The recorded
  // layout_hash must be the one the design consumes -- the cross-generation
  // packing failure is invisible everywhere else (same byte count, same shapes,
  // matching hash on both sides, plausible numbers), and this is the only
  // place both sides are in the same expression.
  //
  // On an int8 design this is ALSO where the operand's quantisation scales are
  // read, and it is the only place that can: `.wscale` and `.asmooth` are
  // per-operand tensors in the container, the runtime has no other source for
  // them, and a GEMM that staged its A panel without them would have no way to
  // put the right numbers back. They are kept against the B slot this call
  // returns, which is what every GEMM call site already carries.
  size_t stage_operand(const npue::File &model, const std::string &name);

  // out[n_real, N] = A[n_real, k] @ B + bias, computed at the stream's own row
  // count `rows`.
  //
  // Rows [n_real, rows) of A are zero-filled rather than left holding the
  // previous dispatch's values: the design computes all `rows` rows whatever A
  // holds there, and a stale row would put another call's activations into this
  // one's output. The tail is a padded row, so zero is the only value that
  // cannot be mistaken for a real position.
  //
  // ON AN INT8 DESIGN THIS DISPATCHES TO run_i8. That is the whole of it: the
  // caller does not know or care which datapath it is on, the design does, and
  // the eleven call sites that make up Whisper's encoder and decoder are exactly
  // the ones that must not each grow a branch on it.
  void run(size_t instr, const float *a, int64_t n_real, int64_t rows, int64_t k,
           size_t wslot, const float *bias, int64_t n, float *out);

  // INT8, when the design and the container agree that they are: A is
  // quantised per ROW (per token) here, on the way into the buffer the dispatch
  // reads, and C comes back dequantised by the rank-1 product of that row's scale
  // and the weight's per-column scale. Both halves are the shared helpers the
  // BERT path uses -- `quantise_a_int8` and `dequantise_c` in
  // common/host_kernels.hpp -- so the arithmetic is written once and a
  // divergence between the two architectures would have to be introduced, not
  // inherited.
  //
  // `inv_smooth` is 1/asmooth of the operand (per INPUT channel), applied to the
  // activation before its row maximum is taken, which is the whole point of
  // SmoothQuant: the row scale is then set by the smoothed values, and the
  // weight was pre-multiplied by asmooth at pack time. Null means no smoothing,
  // which is what a container with all-ones asmooth carries.
  //
  // Called from run() for an int8 design. Reachable directly only by a test.
  void run_i8(size_t instr, const float *a, int64_t n_real, int64_t rows,
              int64_t k, size_t wslot, const float *bias, int64_t n, float *out,
              const float *wscale, const float *inv_smooth);

  // acc[n_real, n] += A[n_real, k] @ B, with no bias and no epilogue.
  //
  // This is the accumulate form, and it exists because of Whisper's audio front
  // end: a 3-tap convolution is three partial dot products over the same output
  // row, and C comes back in bf16, so summing the taps in fp32 on the host is
  // both cheaper and more accurate than one dispatch over a fused K. The bias is
  // deliberately not added here -- a caller adding it per dispatch would add it
  // once per tap.
  //
  // bf16 DESIGNS ONLY, and the reason is the B panel rather than the A side:
  // NpuConv1d::stage() builds this operand's B panel on the host, from the
  // container's F32 conv weights, through a bf16 tiler. An int8 design's panel
  // is I8 with a different MAC sub-tile, so that tiler would produce the right
  // byte count in the wrong element type and the wrong order -- which is the
  // silent-wrong-answer shape this tree treats as the worst outcome, and the
  // reason an int8 design set carries no conv stream and stt_mode.hpp sends the
  // front end to the host.
  void run_accum(size_t instr, const float *a, int64_t n_real, int64_t rows,
                 int64_t k, size_t wslot, int64_t n, float *acc);

  npu::Design &design() { return d_; }
  const npu::Design &design() const { return d_; }
  size_t slot_a = 0, slot_c = 0;
  std::mutex *npu_mu = nullptr;
  int64_t n_dispatch = 0;
  double t_dispatch = 0.0;
  double t_convert = 0.0;

private:
  // ONE OPERAND'S QUANTISATION SCALES, held against the B slot stage_operand
  // returned for it.
  //
  // The key is a SLOT rather than an operand NAME because that is what every
  // GEMM call site carries and the name is not in scope at any of them: keying
  // by name would mean threading a string through eleven call sites to look up
  // something they already hold. `inv_smooth` is stored by VALUE, not as a
  // pointer into the container plus a one-deep cache of its reciprocal, because
  // eleven operands of eleven different K lengths all live in one instance and
  // a one-deep cache thrashes on every layer boundary -- which would recompute
  // the reciprocal on every dispatch, exactly the cost the cache existed to
  // avoid. Empty means the operand carries no asmooth.
  struct OpScale {
    const float *wscale = nullptr;
    std::vector<float> inv_smooth;
  };

  // The scales for one staged operand, or a refusal naming the call that wanted
  // them. Reaching the refusal means a GEMM was dispatched against a B slot that
  // stage_operand never produced -- a slot from another design, or a hand-built
  // one -- so the message says that rather than talking about packing.
  const OpScale &scale_for(size_t wslot, const char *call) const;

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
  std::unordered_map<size_t, OpScale> scales_;
  std::vector<float> a_scale_;
};

// LayerNorm over rows of `d` columns, in place, with the container's epsilon.
// Whisper's is 1e-5, not BERT's 1e-12, and the difference is visible in the
// last bits of a 32-layer stack.
void layernorm_rows(float *x, int64_t n_rows, int64_t d, const float *gamma,
                    const float *beta, double eps, app::Pool &pool);

// exact-erf GELU, in place. The activation is the model's, not a choice: the
// container records `gelu` and tools/pack/packers/whisper.py refuses a checkpoint
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

// The pre-tiled B panel, built on the host from an [K, N] fp32 matrix.
//
// Same layout and the same order as tools/lib/npue.py's tile_b(order="k,n") and
// tools/export/exporters/... the C++ packer's tile_b, which is what the container's
// own GEMM operands are stored in:
//
//   [K,N] -> [K/tile_k][N/tile_n][tile_k/mac_s][tile_n/mac_t][s][t]
//
// The four numbers come from the DESIGN's own b_layout, never from constants
// here: mac_s/mac_t differ per generation (npu1 8/4, npu2 8/8) and tile_n is
// per model, so a hard-coded pair is right on exactly one board of one model.
// This is the one function in the tree that turns a matrix into an operand the
// MMAC can DMA, and a mistake in it is invisible -- the byte count, the shapes
// and the layout_hash all agree, and only the products are wrong.
std::vector<uint16_t> tile_b_panel(const float *mat, int64_t K, int64_t N,
                                  int64_t tile_k, int64_t tile_n, int64_t mac_s,
                                  int64_t mac_t);

}  // namespace npue::whisper
