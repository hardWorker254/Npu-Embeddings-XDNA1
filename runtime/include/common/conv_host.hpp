//===- conv_host.hpp --------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the host convolution, shared by every architecture that runs
// one on the CPU: arch=6 (body pose) and arch=7 (hand palm detector + hand
// landmark network).
//
// WHY THIS FILE EXISTS
// --------------------
// The im2col, the blocked GEMM and the [M,N] -> NCHW transpose were written
// once inside runtime/src/pose/net.cpp. arch=7 needs the same three, with the
// same NCHW layout and the same threading, and a second copy would be 300 lines
// that drift: the second one gets tuned for its own network and then the two
// disagree about what "the host convolution" measures, which is exactly the
// number tools/verify/verify_hands.py and tools/verify/diff_pose_dump.py compare.
//
// So there is ONE implementation and both architectures call it. The per-file
// -O3 in runtime/CMakeLists.txt matters for this header's reason: the inner axpy
// is the whole of the host convolution's arithmetic and -O2 does not vectorise
// it.
//
// WHAT IS AND IS NOT HERE
// ----------------------
// The KERNELS are here -- Nchw, par_items, gemm_nt, transpose_mn_to_nchw,
// im2col -- because they are pure functions of a shape and know nothing about
// either network. The OPS are not: concat, maxpool, slice and the C2f residual
// stay in pose/net.cpp, and dwconv, pad_c and the nearest resize stay in
// hands/net.cpp, because each of those exists for one architecture only and
// moving it here would make the header a list of both networks' vocabularies
// rather than the arithmetic they share.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/pool.hpp"

namespace npue::hostconv {

// NCHW, the checkpoint's own layout, because both backends want it: the host
// kernel indexes it per element and the array's A panel is built from it by the
// same im2col. A channels-last buffer would be a faster host kernel and a second
// set of bugs, and the stride maths for the array's DMA wants rows contiguous
// over K, which NCHW's im2col produces.
struct Nchw {
  int64_t c = 0, h = 0, w = 0;
  std::vector<float> d;

  Nchw() = default;
  Nchw(int64_t C, int64_t H, int64_t W)
      : c(C), h(H), w(W),
        d(static_cast<size_t>(C * H * W), 0.f) {}
  bool empty() const { return c <= 0 || h <= 0 || w <= 0; }
  size_t plane() const { return static_cast<size_t>(h) * static_cast<size_t>(w); }
  size_t elems() const { return static_cast<size_t>(c) * plane(); }
  float *chw(int64_t ci) { return d.data() + static_cast<size_t>(ci) * plane(); }
  const float *chw(int64_t ci) const {
    return d.data() + static_cast<size_t>(ci) * plane();
  }
};

// The activations the two networks fuse into the convolution's epilogue.
//
// A MODEL'S activations, not a library's: arch=6's convs all record SiLU and a
// container whose convs record a different one is a different network, and
// arch=7 records none / relu / relu6 / prelu. Where each is APPLIED is this
// file's choice, made once, in transpose_mn_to_nchw -- it was a separate pass
// over the finished NCHW tensor (read 15.5 M floats, write 15.5 M, one pool
// barrier, for a function of each element and nothing else) and the fused form
// is bit-identical rather than merely close, because the value is stored and
// reloaded unchanged in between.
enum class Act {
  kNone,
  kSiLU,   // x * sigmoid(x), arch=6
  kRelu,   // max(0, x), arch=7
  kRelu6,  // min(6, max(0, x)), arch=7 -- the mobilenet family
  kPreLU,  // x >= 0 ? x : slope[c] * x, arch=7's PReLU after the first conv
};

constexpr int64_t kParMinItems = 65536;

// Parallelise a strided loop over n items of `work` elements each -- but only
// when it is worth the barrier. Pool::run is a generation counter with two
// condition variables, so waking sixteen workers costs something; for a few
// thousand elements that costs more than the loop. 65536 is the same threshold
// the encoders use (BertEncoder::par, GemmaNpuEncoder::par), and it is a
// THRESHOLD rather than a correctness argument: the loop body must be
// independent per item either way, and every caller below strides so that it is.
//
// `work` is separate from `n` because the callers stride over DIFFERENT things
// and only one of them is the right unit to measure. A residual add strides
// over a flat element run, so work is 1. Concat, slice, maxpool, dwconv and the
// up-path stride over CHANNELS -- so that each worker's memcpy is one contiguous
// plane rather than a set of interleaved offsets -- and pass the plane's element
// count as `work`. Thresholding on `n` there measured 32 to 512 channels against
// a 65536-element bar, so all of them stayed serial and the first cut of this
// helper parallelised only some of them.
//
// AND THE THRESHOLD IS NOT THE LEVER ANY MORE. Raising it to 2^20 so that only
// the largest joins stay parallel measured 172 ms of network against this
// value's 166, and lowering the ceiling entirely is not reachable either: with
// one thread this span is 18 ms and with sixteen it is 20. It does not scale in
// EITHER direction, which is what a bandwidth-bound span looks like and not what
// a barrier-bound one looks like. Its traffic is about 14 M elements of copy with
// one pool barrier in front of it, and that is the whole of what it costs.
template <typename F>
inline void par_items(app::Pool &pool, int64_t n, int64_t work, F &&f) {
  if (pool.size() == 1 || n * work < kParMinItems) {
    for (int64_t i = 0; i < n; ++i) f(i);
    return;
  }
  pool.run([&](int w, int nw) {
    for (int64_t i = w; i < n; i += nw) f(i);
  });
}

// C[M,N] = A[M,K] @ B[K,N] + bias[N], with A's rows independent.
//
// BLOCKED OVER ROWS, VECTORISED OVER N
// -------------------------------------
// The two axes are wildly unequal in both networks: M is the output pixel count
// (arch=6: 400..102400, arch=7: 49..9216) and N is the channel count
// (arch=6: 16..256, arch=7: 6..108). So the reuse that matters is of B, and the
// loop order is chosen to get it:
//
//   for each block of MR rows:
//     zero acc[MR][N]
//     for k in K:                      <- B[k, 0..N) is read ONCE
//       for mr in MR:
//         a = A[row+k]
//         for n in N: acc[mr][n] += a * B[k][n]
//
// B's row is loaded into cache once and used MR times, and the inner loop over n
// is a plain axpy that the compiler vectorises under -O3 (see the per-source
// option in CMakeLists.txt -- the file-level default is -O2, which does not).
//
// MR is 8 because the accumulator block is MR*N floats and it has to stay in L1:
// 8*256*4 = 8 KB for the widest layer of either network. Larger MR buys more B
// reuse and costs cache, and the wide-N layers are exactly the ones whose B is
// already resident.
constexpr int64_t kMR = 8;

inline void gemm_nt(const float *A, int64_t M, const float *B, int64_t K,
                    int64_t N, const float *bias, float *C, app::Pool &pool) {
  std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(N), 0.f);

  // Rows are independent -- there is no reduction across M -- so the parallel
  // axis is M and every worker writes its own rows. That is why this is safe to
  // split without a reduction, and why arch=6's stem (M = 102400) scales with
  // cores while its head (M = 400) does not: at 400 rows there are 50 blocks and
  // the 16 workers get 3 each.
  pool.run([&](int w, int nw) {
    std::vector<float> acc(static_cast<size_t>(kMR) * static_cast<size_t>(N));
    for (int64_t m0 = static_cast<int64_t>(w) * kMR; m0 < M;
         m0 += static_cast<int64_t>(nw) * kMR) {
      const int64_t mr = std::min<int64_t>(kMR, M - m0);
      std::fill(acc.begin(), acc.begin() + static_cast<size_t>(mr) * N, 0.f);
      const float *const abase = A + static_cast<size_t>(m0) * K;
      for (int64_t k = 0; k < K; ++k) {
        const float *const brow = B + static_cast<size_t>(k) * N;
        for (int64_t r = 0; r < mr; ++r) {
          const float a = abase[static_cast<size_t>(r) * K + k];
          float *const acc_r = acc.data() + static_cast<size_t>(r) * N;
          for (int64_t n = 0; n < N; ++n) acc_r[n] += a * brow[n];
        }
      }
      for (int64_t r = 0; r < mr; ++r) {
        float *const crow = C + static_cast<size_t>(m0 + r) * N;
        const float *const acc_r = acc.data() + static_cast<size_t>(r) * N;
        for (int64_t n = 0; n < N; ++n) crow[n] = acc_r[n] + bias[n];
      }
    }
  });
}

// src[M, src_cols] -> dst, NCHW over M CONSECUTIVE pixels of a tensor whose
// channel stride is `dst_ch`, with the activation applied on the way.
//
// `src_cols` and `dst_ch` exist because the array path's rows and the output
// tensor's channels no longer share extents, and the two used to be the same
// number for different reasons:
//
//   * `src_cols` (pn) is the DESIGN's N and `N` is the convolution's. The design
//     writes a padded N columns per row; the extra ones are channels the tensor
//     does not have.
//   * `dst_ch` is the tensor's channel stride, OH*OW. The array path hands over
//     ONE DISPATCH WINDOW of rows at a time -- 1024 of a 102400-pixel output --
//     and its destination is therefore `dst + n * OH*OW + done`, NOT
//     `dst + n * 1024`. Deriving the channel stride from M is what a single
//     whole-image transpose does and it is right there, and it wrote the second
//     channel of the first chunk 102398 elements early and the fifteenth one on
//     top of the first -- so the convolution's output was right for channel 0
//     and wrong for the other fifteen, at every chunk, which reads as a network
//     that has lost its depth rather than as an indexing slip.
//
// Defaults: `src_cols` 0 means N and `dst_ch` 0 means M, which is every CPU-path
// call and every whole-image transpose.
//
// WHY THE TRANSPOSE EXISTS AT ALL: the GEMM's natural output is [M, N] -- M
// pixels by N channels, N contiguous -- because that is the only layout in which
// both operands stay contiguous along the reduction axis K. The Tensor is NCHW:
// chw(n) is d + n * plane, so its memory order is [N, M]. Those are transposes
// of each other, and getting it wrong is INVISIBLE in shape: a 16x320x320 filled
// in the wrong order is still a 16x320x320 of plausible floats, and every later
// node reads a permuted image. It was wrong everywhere exactly once, in
// pose/net.cpp, before tools/verify/diff_pose_dump.py existed to say so.
//
// The copy is TILED 32 rows at a time. A naive n-outer / m-inner copy writes M
// runs of one float each for every channel and streams the destination as N
// simultaneous write pointers; tiling makes each destination run 32 floats long,
// and walking m in blocks means each channel's row is still written front to
// back across blocks.
inline void transpose_mn_to_nchw(const float *src, int64_t M, int64_t N,
                                 float *dst, app::Pool &pool,
                                 int64_t src_cols = 0, int64_t dst_ch = 0,
                                 Act act = Act::kNone,
                                 const float *slope = nullptr) {
  const int64_t sc = src_cols > 0 ? src_cols : N;
  const int64_t dc = dst_ch > 0 ? dst_ch : M;
  if (sc < N)
    throw std::runtime_error(
        "host transpose: the source rows are " + std::to_string(sc) +
        " wide and the destination needs " + std::to_string(N) +
        ", so the transpose would read past the end of a row");
  if (dc < M)
    throw std::runtime_error(
        "host transpose: the destination's channel stride is " +
        std::to_string(dc) + " for " + std::to_string(M) +
        " pixels, so writing this window would run into the next channel");
  // A PReLU with no slope vector would silently be kNone, which is a wrong
  // network rather than a crash, so it is refused here where the reason is.
  if (act == Act::kPreLU && !slope)
    throw std::runtime_error(
        "host transpose: the op records a prelu and carries no slope vector");
  constexpr int64_t kTile = 32;
  pool.run([&](int w, int nw) {
    for (int64_t m0 = static_cast<int64_t>(w) * kTile; m0 < M;
         m0 += static_cast<int64_t>(nw) * kTile) {
      const int64_t m1 = std::min<int64_t>(m0 + kTile, M);
      for (int64_t n = 0; n < N; ++n) {
        float *const d = dst + n * dc + m0;
        switch (act) {
          case Act::kNone:
            for (int64_t m = m0; m < m1; ++m) d[m - m0] = src[m * sc + n];
            break;
          case Act::kRelu:
            for (int64_t m = m0; m < m1; ++m) {
              const float v = src[m * sc + n];
              d[m - m0] = v > 0.f ? v : 0.f;
            }
            break;
          case Act::kRelu6:
            for (int64_t m = m0; m < m1; ++m) {
              const float v = src[m * sc + n];
              d[m - m0] = v < 0.f ? 0.f : (v > 6.f ? 6.f : v);
            }
            break;
          case Act::kPreLU:
            // PReLU's slope is PER CHANNEL, so n indexes it and this is the one
            // epilogue that cannot be a function of the element alone.
            for (int64_t m = m0; m < m1; ++m) {
              const float v = src[m * sc + n];
              d[m - m0] = v >= 0.f ? v : slope[n] * v;
            }
            break;
          case Act::kSiLU:
            // THE ACTIVATION IS FUSED HERE, into the pass that writes the tensor.
            // The bias is already in the value -- both backends apply it inside
            // their GEMM -- so this is the same number the separate pass read.
            for (int64_t m = m0; m < m1; ++m) {
              const float v = src[m * sc + n];
              d[m - m0] = v / (1.0f + std::exp(-v));
            }
            break;
        }
      }
    }
  });
}

// B is [K, N] -- one row per reduction step, N output columns CONTIGUOUS -- and
// the container stores the filter as [Cout, Cin, kh, kw], which is [N, K].
//
// The transpose happens ONCE, at load, in both architectures' constructors, and
// never per call. It is a startup cost, not a per-frame one, because the filter
// is the same on every frame; paying it inside conv() would be a cost the
// network does not have and one that grows with the number of convolutions
// rather than with the number of images.
//
// The index order is the whole of this function and it is the thing that is easy
// to get wrong: gemm_nt's inner axpy reads B[k*N + n], so [K,N] means k is the
// OUTER index. Passing the container's [N,K] unchanged -- a shape-compatible
// mistake, since both are K*N floats -- produces a convolution with a
// TRANSPOSED filter, which is not a wrong answer but a different network: the
// output is the right shape, every channel is a plausible mixture of the
// right ones, and the error is this architecture's accuracy rather than a crash.
// It was written twice, as two hand-transcribed index loops, and the second one
// was wrong on its first run.
//
// dst must hold N*K floats. It is NOT a bias and NOT a layout change to the
// tensor: the activation still sees [M, N] out of gemm_nt, unchanged.
inline void transpose_nk_to_kn(const float *src, int64_t N, int64_t K,
                               float *dst) {
  for (int64_t k = 0; k < K; ++k)
    for (int64_t n = 0; n < N; ++n)
      dst[static_cast<size_t>(k) * static_cast<size_t>(N) + n] =
          src[static_cast<size_t>(n) * static_cast<size_t>(K) + k];
}

// im2col for a stride/padding generalisation of the SAME case.
//
// The k index is (ci*kh + i)*kw + j and the row index is y*W + x, which is the
// order the weights are read in too, so the GEMM's K reduction walks the kernel
// exactly as the checkpoint stores it. Changing either index produces a
// well-formed matrix of the right shape containing a permuted image, and the
// network would then detect a person made of the photograph's own frequencies.
//
// The output extent is  floor((H + ph + pw - kh)/sh) + 1  and each row gathers
// the kh x kw window at (y*sh + i - ph, x*sw + j - pw), reading zero outside.
// Zero rather than clamp: a clamped edge replicates the border pixel, which is a
// different network, and it is the failure that is hardest to see -- the tensor
// has the right shape and every interior pixel is right.
//
// PADDING IS FOUR TERMS HERE, NOT TWO, and that is arch=7's doing. arch=6 pads
// symmetrically (`ph`, `pw`) and calls this with the defaults, so for that caller
// it is the same function it has always been. arch=7's palm detector carries the
// MediaPipe "SAME" padding evaluated at a stride, which is asymmetric: its first
// convolution is [top,left,bottom,right] = [1,1,2,2] for a 3x3 stride-2 kernel,
// and a symmetric transcription of that shifts the whole pyramid by one pixel --
// invisible in the output's shape and a pixel of every detection's own size.
// `pb` and `pr` default to -1, meaning "the same as `ph` and `pw`", so the
// asymmetric case is opt-in and no symmetric call site has to change.
//
// The zero border is written ONCE per tensor -- A.assign(M*K, 0) -- rather than
// tested per element: the interior of the window is the common case and a bounds
// test inside the innermost loop is the difference between a memory-bound kernel
// and a branch-bound one, and arch=6's stem has 102400 rows.
//
// BUT ONLY WHERE THERE IS A BORDER TO WRITE. With no padding the gather can
// never fall outside the image -- y*sh + i is at most (OH-1)*sh + kh - 1,
// which is H - 1 -- so every one of the K columns of every row is written below
// and the fill is pure waste. That is 28 of arch=6's 73 convolutions: the 1x1
// ones, 19.1% of that network's arithmetic and 8.3% of its im2col rows, and they
// were zeroing a buffer they then wrote end to end. resize() is a no-op when the
// buffer is already big enough, and A is reused across convolutions precisely
// so that it usually is.
inline void im2col(const Nchw &in, int64_t kh, int64_t kw, int64_t ph,
                    int64_t pw, int64_t sh, int64_t sw, std::vector<float> &A,
                    app::Pool &pool, int64_t pb = -1, int64_t pr = -1) {
  if (pb < 0) pb = ph;
  if (pr < 0) pr = pw;
  const int64_t H = in.h, W = in.w, C = in.c, K = C * kh * kw;
  const int64_t OH = (H + ph + pb - kh) / sh + 1;
  const int64_t OW = (W + pw + pr - kw) / sw + 1;
  const int64_t M = OH * OW;
  if (M <= 0 || K <= 0) {
    A.clear();
    return;
  }
  if (ph == 0 && pb == 0 && pw == 0 && pr == 0)
    A.resize(static_cast<size_t>(M) * static_cast<size_t>(K));
  else
    A.assign(static_cast<size_t>(M) * static_cast<size_t>(K), 0.f);
  // Parallel over output image ROWS, not over the flat M: rows are independent
  // here too, and each worker owns a disjoint run of A, so there is no
  // reduction. The rows are handed out in strides rather than in blocks because
  // a worker's rows are then OW*K apart, which spreads the workers across the
  // tensor instead of clustering them in one corner of it.
  pool.run([&](int w, int nw) {
    for (int64_t y = static_cast<int64_t>(w); y < OH; y += nw) {
      for (int64_t x = 0; x < OW; ++x) {
        float *const row = A.data() + (static_cast<size_t>(y) * OW + x) * K;
        for (int64_t i = 0; i < kh; ++i) {
          const int64_t sy = y * sh + i - ph;
          if (sy < 0 || sy >= H) continue;
          for (int64_t j = 0; j < kw; ++j) {
            const int64_t sx = x * sw + j - pw;
            if (sx < 0 || sx >= W) continue;
            for (int64_t c = 0; c < C; ++c)
              row[(c * kh + i) * kw + j] =
                  in.chw(c)[static_cast<size_t>(sy) * W + sx];
          }
        }
      }
    }
  });
}

}  // namespace npue::hostconv