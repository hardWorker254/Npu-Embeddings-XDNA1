//===- conv1d.hpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's two audio convolutions on the NPU.
//
// WHAT THIS IS
// ------------
// conv1 and conv2 are the only arithmetic in the Whisper path that the host
// still does per window, and they are not cheap: conv2 is d*d*3*1500 MACs, so
// whisper-tiny spends 0.66 s of CPU per 30 s window and large-v3 spends about
// 7 s, which is more than the whole 32-layer encoder stack costs on the array.
// The existing note in features.hpp is right that the convolution cannot be
// folded into the encoder's first GEMM -- a 3-tap convolution over a sliding
// window is a banded matmul, and the design's operand is a dense [M, K] -- but
// "not foldable into the first GEMM" is not "not expressible as a GEMM".
//
// HOW A CONVOLUTION BECOMES A GEMM HERE
// -------------------------------------
// im2col, without the copy: the K axis of the GEMM is the flattened
// (input channel, tap) pair, so
//
//   out[t, o] = bias[o] + sum_{c, j} w[o, c, j] * in[c, t*stride + j - pad]
//
// is A[t, c*taps + j] @ B[c*taps + j, o] with A's column index naming the tap.
// Three taps of conv1 therefore ride in ONE dispatch (240 columns of K), and
// conv2 -- whose 3*d columns do not fit one d-wide K -- is split into
// ceil(3*d / K) = 3 dispatches accumulated on the host in fp32.
//
// WHY THE EXISTING DESIGNS, AND WHAT THAT COSTS
// ---------------------------------------------
// No new stream is exported: the convolutions run on the encoder set's own
// [rows, d, d] stream, which is exactly attn_out's shape. That is free, and it
// is also the price. The design's K is d_model, so an operand narrower than
// that is zero-padded up to it: conv1's 240 (or 384) useful columns of K are
// computed against d_model of them. At whisper-tiny that is 1.6x the useful
// arithmetic on conv1 and none at all on conv2 (3*d splits exactly into three
// d-wide blocks); at large-v3 conv1 pays 3.3x, which is 0.5% of one window's
// encoder work. A design exported with K = taps*in_ch would remove the padding
// AND collapse conv1 to a single dispatch, and that is the real fix -- it needs
// one more stream in tools/data/npu_targets.json's stt set and a re-export.
//
// The GELU stays on the host, in fp32, exactly as in the CPU front end: the
// design set has no eltwise stream for it (kinds.stt lists GEMM streams only),
// and the activation between a bf16 C and the next GEMM's bf16 A is the one
// place where an extra rounding is free to remove.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "whisper/features.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::whisper {

// Where an input tensor's channels sit, which is the one thing about the
// gather that cannot be inferred from a shape.
//
// The front end has both kinds and they are not interchangeable: the mel is
// (n_mels, frames) channels-first, and conv1's own output is (frames, d)
// time-major because the GEMM's row is a position. The gather needs the
// difference, and expressing it as a pair of strides made it swappable -- a
// swapped pair is not a wrong number in one place, it is a convolution of the
// wrong tensor with the right weights, which is why it is an enum here.
enum class Conv1dLayout {
  channels_first,   // in(c, ti) == in[c * t_in + ti]
  time_major,       // in(c, ti) == in[ti * in_ch + c]
};

// One Conv1d on the array, as GEMMs over an im2col'ed K axis.
//
// It shares the design and the instruction stream with the encoder -- [rows, d,
// d] is attn_out's own shape -- but it holds its OWN NpuGemm, so its A and C
// buffers and its staged weights are the convolution's and the encoder's rows
// cannot be written under it. conv1 and conv2 share one NpuGemm between them:
// they never run at the same time, and a second A/C pair is buffer for nothing.
class NpuConv1d {
public:
  // `k` and `n` are the STREAM's K and N, which must be the design's, and
  // `n` is this conv's output width: out channels are the GEMM's N because each
  // output channel is an independent dot product.
  NpuConv1d(npu::Design &design, app::Pool &pool, NpuGemm &gemm, size_t instr,
            int64_t rows, int64_t k, int64_t n)
      : d_(design), pool_(pool), g_(gemm), instr_(instr), rows_(rows), k_(k),
        n_(n) {}

  // Stage one convolution's weights. `w` is the container's Conv1d tensor,
  // [out_ch, in_ch, taps] row-major, and `bias` is [out_ch]. The staged
  // operand is built here rather than read from the container because the
  // container keeps the convolutions in fp32 as the REFERENCE for
  // tools/verify/verify_whisper_features.py; the pre-tiled bf16 copy is derived from
  // those exact bytes with the design's own b_layout.
  void stage(const std::string &label, const float *w, const float *bias,
             int64_t in_ch, int64_t out_ch, int64_t taps);

  // out[t, o] = gelu(bias[o] + sum_{c,j} w[o, c, j] * in(c, t*stride + j - pad))
  //
  // `out` is (t_out, out_ch) time-major, which is the layout the encoder's
  // first GEMM wants, so the CPU front end's permute pass has no counterpart
  // here. The activation is applied here, in fp32, for the same reason it is
  // not folded into the GEMM: there is no eltwise stream for it.
  void run(const float *in, int64_t in_ch, Conv1dLayout layout, int64_t t_in,
           int64_t t_out, int64_t stride, int64_t pad, float *out);

  int64_t n_dispatch() const { return g_.n_dispatch; }
  const NpuGemm &gemm() const { return g_; }
  int64_t out_ch() const { return out_ch_; }
  int64_t in_ch() const { return in_ch_; }
  int64_t taps() const { return taps_; }
  int64_t k_blocks() const { return static_cast<int64_t>(b_slots_.size()); }

private:
  npu::Design &d_;
  app::Pool &pool_;
  NpuGemm &g_;
  size_t instr_;
  int64_t rows_, k_, n_;
  int64_t in_ch_ = 0, out_ch_ = 0, taps_ = 0;
  const float *bias_ = nullptr;
  std::vector<size_t> b_slots_;
  // Scratch, allocated once per conv and reused for every row chunk: the A
  // chunk is rows_*k_ floats and the accumulator is rows_*n_.
  std::vector<float> a_, acc_;
};

// conv1 -> GELU -> conv2 -> GELU, on the array, and no permute: conv2's rows are
// already time-major. `conv1` reads the mel (channels-first) and `conv2` reads
// conv1's output, so the two are staged and called in that order. Returns
// (kEncoderPositions, d_model) row-major, the same tensor conv_front_end returns.
std::vector<float> conv_front_end_npu(const MelSpec &mel, NpuConv1d &conv1,
                                      NpuConv1d &conv2, int hidden);

}  // namespace npue::whisper
