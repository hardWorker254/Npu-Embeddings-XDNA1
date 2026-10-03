//===- net.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=6 graph walk and its host GEMM.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "pose/net.hpp"

#include "common/host_kernels.hpp"   // app::now_s

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace npue::pose {

namespace {

// SiLU, x * sigmoid(x), in place. The activation is the MODEL's: YOLOv8's
// C2f/SPPF/Detect blocks all use SiLU, and a container whose convs record a
// different one is a different network. Fused into the conv's epilogue rather
// than run as a separate pass because it is elementwise on the GEMM's output --
// a second sweep over the tensor would double the memory traffic of the widest
// layers for no arithmetic.
inline void silu_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const float v = x[i];
    x[i] = v / (1.0f + std::exp(-v));
  }
}

// C[M,N] = A[M,K] @ B[K,N] + bias[N], with A's rows independent.
//
// BLOCKED OVER ROWS, VECTORISED OVER N
// -------------------------------------
// The two axes are wildly unequal in this network: M is 400..102400 rows (the
// spatial extent) and N is 16..256 columns (the channel count). So the reuse
// that matters is of B, and the loop order is chosen to get it:
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
// 8*256*4 = 8 KB, while 8*64*4 = 2 KB for the narrow layers. Larger MR buys
// more B reuse and costs cache, and the wide-N layers are exactly the ones whose
// B is already resident.
constexpr int64_t kMR = 8;

void gemm_nt(const float *A, int64_t M, const float *B, int64_t K, int64_t N,
             const float *bias, float *C, app::Pool &pool) {
  std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(N),
            0.f);

  // Rows are independent -- there is no reduction across M -- so the parallel
  // axis is M and every worker writes its own rows. That is why this is safe to
  // split without a reduction, and why the stem (M = 102400) scales with cores
  // while the head (M = 400) does not: at 400 rows there are 50 blocks and the
  // 16 workers get 3 each.
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

// [M, N] -> [N, M], which is the whole point of this function.
//
// WHY IT EXISTS AT ALL
// -------------------
// The GEMM's natural output is [M, N] -- M pixels by N channels, N contiguous --
// because that is the only layout in which both operands stay contiguous along
// the reduction axis K: A is [M, K] and B is [K, N], so the epilogue writes
// C[m * N + n]. The Tensor, however, is NCHW: chw(n) is d + n * plane, so its
// memory order is [N, M].
//
// Those are transposes of each other, and getting it wrong is invisible in
// shape: a Tensor of 16x320x320 filled in the wrong order is still a
// 16x320x320 of plausible floats, and every later node reads a permuted image
// and produces a network that is wrong everywhere. It was wrong everywhere
// exactly once, in this file, before tools/verify/diff_pose_dump.py existed to
// say so.
//
// The copy is TILED 32 rows at a time. A naive n-outer / m-inner copy writes M
// runs of one float each for every channel and streams the destination as N
// simultaneous write pointers; tiling makes each destination run 32 floats long,
// and walking m in blocks means each channel's row is still written front to
// back across blocks.
//
// Parallel over m blocks rather than n blocks: M is the large axis (up to 102400
// rows) and N is at most 256, so splitting on m uses every worker and splitting on
// n would leave half of them idle on most of this network's convolutions.
// src[M, src_cols] -> dst, NCHW over M CONSECUTIVE pixels of a tensor whose
// channel stride is `dst_ch`.
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
void transpose_mn_to_nchw(const float *src, int64_t M, int64_t N, float *dst,
                          app::Pool &pool, int64_t src_cols = 0,
                          int64_t dst_ch = 0) {
  const int64_t sc = src_cols > 0 ? src_cols : N;
  const int64_t dc = dst_ch > 0 ? dst_ch : M;
  if (sc < N)
    throw std::runtime_error(
        "pose transpose: the source rows are " + std::to_string(sc) +
        " wide and the destination needs " + std::to_string(N) +
        ", so the transpose would read past the end of a row");
  if (dc < M)
    throw std::runtime_error(
        "pose transpose: the destination's channel stride is " +
        std::to_string(dc) + " for " + std::to_string(M) +
        " pixels, so writing this window would run into the next channel");
  constexpr int64_t kTile = 32;
  pool.run([&](int w, int nw) {
    for (int64_t m0 = static_cast<int64_t>(w) * kTile; m0 < M;
         m0 += static_cast<int64_t>(nw) * kTile) {
      const int64_t m1 = std::min<int64_t>(m0 + kTile, M);
      for (int64_t n = 0; n < N; ++n) {
        float *const d = dst + n * dc + m0;
        for (int64_t m = m0; m < m1; ++m) d[m - m0] = src[m * sc + n];
      }
    }
  });
}

// im2col for a stride/padding generalisation of the SAME case.
//
// The k index is (ci*kh + i)*kw + j and the row index is y*W + x, which is the
// order the weights are read in too, so the GEMM's K reduction walks the kernel
// exactly as the checkpoint stores it. Changing either index produces a
// well-formed matrix of the right shape containing a permuted image, and the
// network would then detect a person made of the photograph's own frequencies.
//
// The zero border is written ONCE per tensor -- A.assign(M*K, 0) -- rather than
// tested per element: the interior of the window is the common case and a bounds
// test inside the innermost loop is the difference between a memory-bound kernel
// and a branch-bound one, and there are 102400 rows at the stem.
//
// The output extent is  floor((H + ph + pw - kh)/sh) + 1  and each row gathers
// the kh x kw window at (y*sh + i - ph, x*sw + j - pw), reading zero outside.
// Zero rather than clamp: a clamped edge replicates the border pixel, which is a
// different network, and it is the failure that is hardest to see -- the tensor
// has the right shape and every interior pixel is right.
void im2col(const Tensor &in, int64_t kh, int64_t kw, int64_t ph, int64_t pw,
            int64_t sh, int64_t sw, std::vector<float> &A, app::Pool &pool) {
  const int64_t H = in.h, W = in.w, C = in.c, K = C * kh * kw;
  const int64_t OH = (H + 2 * ph - kh) / sh + 1;
  const int64_t OW = (W + 2 * pw - kw) / sw + 1;
  const int64_t M = OH * OW;
  A.assign(static_cast<size_t>(M) * static_cast<size_t>(K), 0.f);
  if (M <= 0 || K <= 0) return;
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

}  // namespace

Network::Network(const Geometry &g, const Placement &place, app::Pool &pool,
                 NpuConvs *design)
    : g_(g), place_(place), pool_(pool), npu_(design) {}

Tensor Network::conv(const Tensor &in, const ConvW &w, bool silu,
                     int64_t stride, int64_t pad_h, int64_t pad_w,
                     std::vector<float> &a_buf) {
  if (in.c != w.cin)
    throw std::runtime_error("pose conv: input " + std::to_string(in.c) +
                             " channels, weight wants " + std::to_string(w.cin));
  if (stride <= 0)
    throw std::runtime_error("pose conv: stride " + std::to_string(stride) +
                             "; a non-positive step has no output position");
  if (in.h + 2 * pad_h < w.kh || in.w + 2 * pad_w < w.kw)
    throw std::runtime_error(
        "pose conv: a " + std::to_string(in.h) + "x" + std::to_string(in.w) +
        " input padded by " + std::to_string(pad_h) + "/" + std::to_string(pad_w) +
        " is smaller than the " + std::to_string(w.kh) + "x" +
        std::to_string(w.kw) + " window, so the output extent would be "
        "negative");
  const int64_t K = w.cin * w.kh * w.kw;
  // 2 * pad_h, not pad_h + pad_w. The pads are equal for every convolution in
  // this network, so the two spellings compute the same extent here and the wrong
  // one would not show up in a test on THIS checkpoint -- it would show up on the
  // next one that pads asymmetrically, as an output one row or column too large,
  // which is again a plausible tensor.
  const int64_t OH = (in.h + 2 * pad_h - w.kh) / stride + 1;
  const int64_t OW = (in.w + 2 * pad_w - w.kw) / stride + 1;
  const int64_t M = OH * OW;
  const int64_t N = w.cout;
  Tensor out(N, OH, OW);

  // The GEMM's [M, N] result, held apart from the Tensor and transposed into it
  // afterwards -- see transpose_mn_to_nchw's header for why the two cannot be the
  // same buffer. Reused across convolutions for the same reason a_buf is: the
  // stem's is 102400 x 16 floats = 6.6 MB and allocating that 73 times per image
  // would be churn for a value that is dead as soon as the transpose is done.
  const size_t cn = static_cast<size_t>(M) * static_cast<size_t>(N);
  if (c_scratch_.size() < cn) c_scratch_.resize(cn);

  const double t0 = app::now_s();

  // The weight matrix the GEMM wants is [K, N] -- k-major, n-contiguous -- and
  // the container holds [N, K]. Transposing it once per convolution, at load
  // time rather than per run, is what keeps the inner loop above a contiguous
  // axpy; the transpose is 3.28 M elements over the whole model, against 4.59 G
  // of GEMM per image.
  //
  // It is HERE, per run, not where that comment says it should be. The comment
  // describes the fix; the code is the problem the fix exists for, and it is
  // 3.28 M elements of strided write plus a fresh allocation per convolution per
  // image. That is not the reason the network takes 250 ms -- the GEMM is -- but
  // it is real and it is measured separately now so that it cannot be mistaken
  // for part of the GEMM.
  const double tw = app::now_s();
  std::vector<float> wmat(static_cast<size_t>(K) * static_cast<size_t>(N));
  for (int64_t k = 0; k < K; ++k)
    for (int64_t n = 0; n < N; ++n)
      wmat[static_cast<size_t>(k) * N + n] =
          w.w[static_cast<size_t>(n) * K + k];
  cost_.t_wmat += app::now_s() - tw;

  const double ti = app::now_s();
  im2col(in, w.kh, w.kw, pad_h, pad_w, stride, stride, a_buf, pool_);
  cost_.t_im2col += app::now_s() - ti;

  if (place_.conv_on_array && npu_ != nullptr) {
    // The array path takes the SAME im2col rows and the SAME weight panel, so
    // the two backends cannot differ in the part that is easy to get wrong (the
    // column order); they differ only in who multiplies. It also produces [M, N]
    // -- a GEMM writes pixels by channels -- so it lands in the same scratch and
    // takes the same transpose.
    //
    // EVERY WIDTH HERE IS THE DESIGN'S, NOT THE CONVOLUTION'S. This is the whole
    // difference between the two backends and it is not a detail:
    //
    //   * K is padded up to a multiple of tile_k (27 becomes 64 for the stem),
    //     and the design reads `rows * k` columns of A as one contiguous run --
    //     so A has to be REPACKED at the padded stride and its tail zeroed. Left
    //     at the real stride, row r's tail columns read row r+1's data, which is
    //     a plausible network that finds 271 people where there are 3.
    //   * N is padded up to a multiple of tile_n * cols (16 becomes 128), and the
    //     design writes `rows * n` columns of C whatever n was asked for.
    //   * The bias is read for all n columns, so it has to be n wide -- which is
    //     why the backend stages a zero-padded copy rather than handing over the
    //     convolution's own.
    //
    // The repack is per CHUNK, not per image: rows*pk is 9.4 MiB at this
    // geometry while the whole convolution's A is 900 MiB, and the array never
    // needs more than one dispatch window at a time. For the same reason each
    // chunk is transposed as soon as it lands -- the padded C is then 1 MiB
    // instead of 100 MiB -- at `out.d + done` with the tensor's OWN channel
    // stride, which is OH*OW and not the chunk's row count.
    const int64_t rows = npu_->rows_per_dispatch();
    const int64_t pk = npu_->padded_k(w.index);
    const int64_t pn = npu_->padded_n(w.index);
    if (pk < K || pn < N)
      throw std::runtime_error(
          "pose conv: the design's padded shape " + std::to_string(pk) + "x" +
          std::to_string(pn) + " is smaller than the convolution's own " +
          std::to_string(K) + "x" + std::to_string(N) +
          ", so the panel would be truncated");
    const size_t a_need = static_cast<size_t>(rows) * static_cast<size_t>(pk);
    if (a_pad_.size() < a_need) a_pad_.resize(a_need);
    const size_t c_need = static_cast<size_t>(rows) * static_cast<size_t>(pn);
    if (c_pad_.size() < c_need) c_pad_.resize(c_need);

    int64_t done = 0;
    while (done < M) {
      const int64_t chunk = std::min<int64_t>(rows, M - done);
      const double tr0 = app::now_s();
      // Zero first, copy second: the padded columns are not written by the
      // memcpy and must not hold the previous dispatch's activations.
      std::memset(a_pad_.data(), 0,
                  static_cast<size_t>(chunk) * static_cast<size_t>(pk) *
                      sizeof(float));
      pool_.run([&](int w2, int n2) {
        for (int64_t r = static_cast<int64_t>(w2); r < chunk;
             r += static_cast<int64_t>(n2)) {
          std::memcpy(a_pad_.data() + static_cast<size_t>(r) * pk,
                      a_buf.data() + static_cast<size_t>(done + r) * K,
                      static_cast<size_t>(K) * sizeof(float));
        }
      });
      const double tr1 = app::now_s();
      npu_->gemm(w.index, a_pad_.data(), chunk, pk, pn, w.b, c_pad_.data());
      transpose_mn_to_nchw(c_pad_.data(), chunk, N,
                           out.d.data() + static_cast<size_t>(done), pool_,
                           pn, M);
      // The two host-side spans of the array path are timed separately, and
      // they are the reason a status line that printed only the device time
      // would lie: neither of them is the array's work.
      cost_.t_array_repack += tr1 - tr0;
      cost_.t_array_transpose += app::now_s() - tr1;
      done += chunk;
      cost_.dispatches++;
    }
    cost_.convs_array++;
    cost_.t_array += app::now_s() - t0;
    if (silu) silu_inplace(out.d.data(), out.d.size());
    return out;
  }

  const double tg = app::now_s();
  gemm_nt(a_buf.data(), M, wmat.data(), K, N, w.b, c_scratch_.data(), pool_);
  cost_.t_gemm += app::now_s() - tg;
  cost_.convs_host++;
  const double tt = app::now_s();
  transpose_mn_to_nchw(c_scratch_.data(), M, N, out.d.data(), pool_);
  cost_.t_transpose += app::now_s() - tt;
  cost_.t_host += app::now_s() - t0;
  if (silu) silu_inplace(out.d.data(), out.d.size());
  return out;
}

Tensor Network::concat(const std::vector<const Tensor *> &in) {
  if (in.size() < 2)
    throw std::runtime_error("pose concat: " + std::to_string(in.size()) +
                             " operand(s). A join needs at least two, and the "
                             "graph's operand list is the only place that "
                             "count is visible -- a one-operand join would "
                             "silently be a copy, and a copy reads as a "
                             "plausible layer.");
  int64_t c = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    const Tensor &t = *in[i];
    if (i && (t.h != in[0]->h || t.w != in[0]->w))
      throw std::runtime_error(
          "pose concat: operand 0 is " + std::to_string(in[0]->h) + "x" +
          std::to_string(in[0]->w) + " and operand " + std::to_string(i) +
          " is " + std::to_string(t.h) + "x" + std::to_string(t.w) +
          ". The join is channel-wise, so every operand has to be the same "
          "size; two different ones would produce a tensor of one of the two "
          "shapes holding half the channels of each.");
    c += t.c;
  }
  Tensor out(c, in[0]->h, in[0]->w);
  int64_t at = 0;
  for (const Tensor *t : in) {
    // d.size(), NOT plane(). plane() is h*w, one channel; the operand is
    // t->c of them. Copying a plane per operand wrote 16 channels' worth of
    // offsets but only a quarter of each one's data, and the join that came out
    // had the right shape and the right first row: the three surviving planes
    // landed at channels 0, 16 and 32 -- the correct channels for the FIRST
    // channel of each operand -- and every other channel of the join was zero.
    // Every channel count downstream still added up.
    std::memcpy(out.chw(at), t->d.data(), t->d.size() * sizeof(float));
    at += t->c;
  }
  return out;
}

// A contiguous run of channels. This is how the C2f block's channel split is
// expressed: one Slice node per half, rather than a multi-output Split node.
//
// The alternative -- a Split op kind with N outputs -- would make the graph's
// node-to-operand reference a pair instead of an index, and every operand in
// the op list would become "[node, which output]" -- a second dimension of
// addressing across the whole format, checked in every reader, for the sake of
// an op that has exactly one use in this architecture and is trivially two
// slices. The refusal below keeps the width honest: a Split whose halves are
// not contiguous would need a strided variant, and it is not approximated here.
Tensor Network::slice(const Tensor &in, int64_t chan, int64_t nch) {
  if (nch <= 0 || chan < 0 || chan + nch > in.c)
    throw std::runtime_error(
        "pose slice: channels [" + std::to_string(chan) + ", " +
        std::to_string(chan + nch) + ") out of a " + std::to_string(in.c) +
        "-channel tensor. A split that is not a contiguous run of channels is a "
        "different op and is not approximated by clamping, because the two "
        "would put different values in different channels and every downstream "
        "channel count would still add up.");
  Tensor out(nch, in.h, in.w);
  std::memcpy(out.d.data(), in.chw(chan), out.d.size() * sizeof(float));
  return out;
}

Tensor Network::add(const Tensor &a, const Tensor &b) {
  if (a.c != b.c || a.h != b.h || a.w != b.w)
    throw std::runtime_error("pose add: shapes differ");
  Tensor out = a;
  const size_t n = out.d.size();
  for (size_t i = 0; i < n; ++i) out.d[i] = a.d[i] + b.d[i];
  return out;
}

Tensor Network::maxpool(const Tensor &in, int64_t k) {
  const int64_t pad = k / 2;
  Tensor out(in.c, in.h, in.w);
  for (int64_t c = 0; c < in.c; ++c) {
    const float *src = in.chw(c);
    float *dst = out.chw(c);
    for (int64_t y = 0; y < in.h; ++y) {
      for (int64_t x = 0; x < in.w; ++x) {
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < k; ++i) {
          const int64_t sy = y + i - pad;
          if (sy < 0 || sy >= in.h) continue;
          for (int64_t j = 0; j < k; ++j) {
            const int64_t sx = x + j - pad;
            if (sx < 0 || sx >= in.w) continue;
            best = std::max(best, src[static_cast<size_t>(sy) * in.w + sx]);
          }
        }
        // A window entirely in the padding cannot happen with pad = k/2 and an
        // in.h >= k image (geometry.cpp refuses a tensor smaller than the
        // window), so `best` is always a real value here. It is initialised to
        // -inf rather than to 0 because a pad-averaged zero would let the
        // padding raise a maximum, which is not what max-pooling a padded
        // tensor means.
        dst[static_cast<size_t>(y) * in.w + x] = best;
      }
    }
  }
  return out;
}

Tensor Network::upsample(const Tensor &in, int64_t scale) {
  Tensor out(in.c, in.h * scale, in.w * scale);
  for (int64_t c = 0; c < in.c; ++c) {
    const float *src = in.chw(c);
    float *dst = out.chw(c);
    for (int64_t y = 0; y < in.h; ++y)
      for (int64_t x = 0; x < in.w; ++x) {
        const float v = src[static_cast<size_t>(y) * in.w + x];
        for (int64_t dy = 0; dy < scale; ++dy) {
          float *row = dst + static_cast<size_t>(y * scale + dy) * out.w;
          for (int64_t dx = 0; dx < scale; ++dx) row[x * scale + dx] = v;
        }
      }
  }
  return out;
}

// -- the head ------------------------------------------------------------------
//
// Assembles the packed graph's per-level head outputs into the ONE canonical
// tensor decode.cpp reads: [4 + nc + nk*3, cells], xyxy boxes in letterbox
// pixels, class score and keypoint visibility as probabilities, keypoints in
// letterbox pixels. `in` is (box+class, keypoints) per level, in the order
// geometry.strides/grid record.
//
// WHY THE MATHS LIVES HERE AND NOT IN THE PACKED GRAPH
// -----------------------------------------------------
// The DFL softmax, the expectation over the bins, the grid anchor addition, the
// stride multiply and the keypoint affine are the exported graph's own
// arithmetic and not convolutions, so the conv engine cannot run them -- and
// adding op kinds for them would mean a softmax and a weighted sum per level
// with nothing to check either against but a careful reading. Here they are one
// function with the two magic constants named where they appear, and
// tools/verify/verify_pose.py checks the whole assembly against an independent
// NumPy reference transcribed from the ONNX graph itself.
//
// THE THREE ORDERING FACTS, which are the whole of the risk:
//
//  1. (anchor + offset) * s, and the offset belongs to the ANCHOR. Multiplying
//     the sum instead differs by offset * s, which at stride 32 is 16 pixels of
//     offset -- a shift, not a scale, so nothing about the output looks wrong
//     except that the poses do not line up with the photograph. The offset is
//     RECORDED (HeadInfo::box_grid_offset, kpt_grid_offset) because the
//     exporter's two grids are not the same grid: see HeadInfo.
//  2. The corner order is [l, t, r, b], so the x corners are q = 0 and q = 2
//     and the y corners are q = 1 and q = 3. Interleaving them instead
//     transposes every box, which for a roughly symmetric person is nearly
//     invisible and for an outstretched arm is not.
//  3. The tensors arriving here are NCHW, like every other tensor in this
//     network, so corner q's bin b is at element (q*bins + b) * cells + cell --
//     CHANNEL first, cell second. Reading it as (q*bins + b) + cell, which is
//     what channels-last would mean, addresses the first few cells of channel 0
//     sixteen times over: the same 16 logits for every cell of every level, and
//     therefore the same box for every candidate. That is not a small error. It
//     is a detector whose scores are exactly right -- the class channel is read
//     with the cell offset and comes from the right place -- attached to boxes
//     and keypoints that are constant, which reads as "the model is bad" rather
//     than as "this function is wrong".
Tensor Network::head(const std::vector<const Tensor *> &in) {
  const int64_t lv = g_.n_levels();
  if (static_cast<int64_t>(in.size()) != 2 * lv)
    throw std::runtime_error(
        "pose head: the detect node lists " + std::to_string(in.size()) +
        " inputs and the container declares " + std::to_string(lv) +
        " levels. The head takes two tensors per level -- the box+class head's "
        "output and the keypoint head's -- and a count that disagrees with the "
        "level list means the graph and the geometry describe different "
        "networks.");

  const int64_t box_ch = g_.head.box == BoxFormat::LtrbDfl ? 4 * g_.dfl_bins : 4;
  const int64_t cls_ch = g_.num_classes;
  const int64_t kpt_ch = g_.num_keypoints * 3;
  const int64_t C = g_.out_channels();
  const int64_t bins = g_.dfl_bins;

  int64_t cells = 0;
  for (int64_t gg : g_.grid) cells += gg * gg;
  Tensor out(C, cells, 1);
  // The loop below produces one cell's worth of the canonical tensor at a time,
  // which is [cell][C]; `out` is [C][cells]. Rather than declare a Tensor whose
  // shape does not describe its memory -- which is exactly how the DFL and the
  // keypoints above came to be indexed channels-last, with decode.cpp agreeing
  // with the mistake -- the rows are built in the order they are produced and
  // transposed through the same helper the convolutions use.
  std::vector<float> rows(static_cast<size_t>(cells) * static_cast<size_t>(C));
  float *const dst = rows.data();

  int64_t offset = 0;
  for (int64_t l = 0; l < lv; ++l) {
    const Tensor &bc = *in[static_cast<size_t>(2 * l)];
    const Tensor &kp = *in[static_cast<size_t>(2 * l + 1)];
    const int64_t s = g_.grid[static_cast<size_t>(l)];
    const float stride = static_cast<float>(g_.strides[static_cast<size_t>(l)]);

    // Checked here rather than trusted. This is the node where a channel-order
    // mistake becomes invisible downstream: every channel is a float and the
    // decode will happily read the wrong ones and return a plausible pose.
    if (bc.c != box_ch + cls_ch)
      throw std::runtime_error(
          "pose head: level " + std::to_string(l) + " feeds the head a " +
          std::to_string(bc.c) + "-channel box+class tensor; head_box_format '" +
          (g_.head.box == BoxFormat::LtrbDfl
               ? "ltrb_dfl' needs 4 * " + std::to_string(bins) + " + " +
                     std::to_string(cls_ch)
               : std::string("xyxy'/'cxcywh' needs 4 + ") +
                     std::to_string(cls_ch)) +
          " = " + std::to_string(box_ch + cls_ch) + " channels");
    if (kp.c != kpt_ch)
      throw std::runtime_error(
          "pose head: level " + std::to_string(l) + " feeds the head a " +
          std::to_string(kp.c) + "-channel keypoint tensor, and " +
          std::to_string(g_.num_keypoints) + " keypoints x 3 is " +
          std::to_string(kpt_ch));
    if (bc.h != s || bc.w != s || kp.h != s || kp.w != s)
      throw std::runtime_error(
          "pose head: level " + std::to_string(l) + " declares grid " +
          std::to_string(s) + " but the head's tensors are " +
          std::to_string(bc.h) + "x" + std::to_string(bc.w) + " and " +
          std::to_string(kp.h) + "x" + std::to_string(kp.w) +
          ". The head maps cell (i, j) to pixel (i*s, j*s), so the grid and the "
          "tensor have to be the same grid or every keypoint belongs to a "
          "neighbouring cell.");

    // NCHW, so one base pointer per tensor and (channel, cell) as the index. The
    // inner loop below is over cells and the outer over channels, which means
    // the 16 bins of a corner are 16 DIFFERENT planes -- the opposite of the
    // adjacency the softmax would like. The softmax is 4 corners x 16 gathers of
    // one float each per cell, 8400 cells, so it is 500k loads and not the cost
    // of this function either way.
    const float *const box = bc.chw(0);
    const float *const cls = bc.chw(box_ch);
    const float *const kpc = kp.chw(0);
    const size_t plane = static_cast<size_t>(s) * static_cast<size_t>(s);
    // corner q, bin b -> its element in the channels-first tensor
    const auto bin = [&](int64_t q, int64_t b, size_t cell) -> float {
      return box[(static_cast<size_t>(q) * static_cast<size_t>(bins) +
                  static_cast<size_t>(b)) *
                     plane +
                 cell];
    };

    for (int64_t i = 0; i < s; ++i) {
      for (int64_t j = 0; j < s; ++j) {
        const int64_t cell = i * s + j;
        float *const row = dst + (offset + cell) * C;

        // -- the box. Four corners, `bins` each, corner-major: corner q's bins
        // are channels q*bins .. (q+1)*bins-1.
        if (g_.head.box == BoxFormat::LtrbDfl) {
          // d[q] = the expectation over the softmaxed bins, in CELLS.
          // The DFL's PROJECTION, folded in analytically rather than run.
          //
          //     d'[q] = sum_bin w[0][bin] * p[bin] + b
          //
          // and because the softmax sums to one, that IS
          //
          //     d'[q] = w[0] . E[bin] + b
          //
          // -- one dot product against the expectation, not a pass over the
          // distribution. That identity is why the projection can live in the
          // head instead of the graph: its input is a Softmax, which no op in
          // the op list consumes. read_geometry checks that the projection is
          // [1, bins, 1, 1] and bias-free, which are exactly the two conditions
          // the identity needs.
          //
          // The weights are APPLIED, not assumed to be the bin index. They
          // happen to be 0..15 in this checkpoint, so reading w[b] as b would
          // give the same number -- for this checkpoint. Using the bin index
          // would be a fold valid for one weight file and silently wrong for
          // the next one, and the error would be a uniform scale factor on
          // every distance: boxes the right shape in the wrong units.
          const float *const dfl_w =
              g_.convs[static_cast<size_t>(g_.head.dfl_conv)].w;
          double d[4];
          for (int64_t q = 0; q < 4; ++q) {
            // The max is subtracted before the exp. It cancels in the ratio, so
            // this is not an approximation -- it is the difference between a
            // correct answer and an inf. A DFL logit of 40 is not unusual on a
            // confident box.
            float mx = bin(q, 0, cell);
            for (int64_t b = 1; b < bins; ++b)
              mx = std::max(mx, bin(q, b, cell));
            double num = 0.0, den = 0.0;
            for (int64_t b = 0; b < bins; ++b) {
              const double e = std::exp(static_cast<double>(bin(q, b, cell) - mx));
              num += e * static_cast<double>(dfl_w[b]);
              den += e;
            }
            d[q] = num / den;
          }
          const double gx = static_cast<double>(j) + g_.head.box_grid_offset;
          const double gy = static_cast<double>(i) + g_.head.box_grid_offset;
          row[0] = static_cast<float>((gx - d[0]) * stride);
          row[1] = static_cast<float>((gy - d[1]) * stride);
          row[2] = static_cast<float>((gx + d[2]) * stride);
          row[3] = static_cast<float>((gy + d[3]) * stride);
        } else {
          // Channels-first again: corner q of this cell, not element q of this
          // cell's 65 floats.
          const auto corner = [&](int64_t q) -> float {
            return box[static_cast<size_t>(q) * plane + static_cast<size_t>(cell)];
          };
          if (g_.head.box == BoxFormat::Xyxy) {
            row[0] = corner(0);
            row[1] = corner(1);
            row[2] = corner(2);
            row[3] = corner(3);
          } else {
            const double hw = static_cast<double>(corner(2)) * 0.5;
            const double hh = static_cast<double>(corner(3)) * 0.5;
            row[0] = static_cast<float>(corner(0) - hw);
            row[1] = static_cast<float>(corner(1) - hh);
            row[2] = static_cast<float>(corner(0) + hw);
            row[3] = static_cast<float>(corner(1) + hh);
          }
        }

        // -- the class score, at index 4: after the four box values, which is
        // the architecture's order and is what read_geometry's detect_cout check
        // exists to pin down.
        float sc = cls[cell];
        if (g_.head.score == ScoreKind::Logit)
          sc = 1.0f / (1.0f + std::exp(-sc));
        row[4] = sc;

        // -- the keypoints: (anchor + 2 * offset) * stride, and the 2 is the
        // exporter's -- the keypoint branch predicts offsets twice the size the
        // box branch's are, which is why a factor of 1 would place every joint
        // half way out toward the grid centre.
        for (int64_t k = 0; k < g_.num_keypoints; ++k) {
          float *const o = row + 4 + g_.num_classes + k * 3;
          // Element (k*3 + c) of the keypoint tensor: channel k*3+c, cell
          // `cell`. base is the channel-group's plane offset and the per-value
          // stride is the plane, not 1 -- adding 1 here reads the neighbouring
          // CELL's value of the same channel, which for a visibility logit means
          // reporting another candidate's confidence.
          const size_t base = static_cast<size_t>(k) * 3 * plane +
                              static_cast<size_t>(cell);
          o[0] = (static_cast<float>(j) + g_.head.kpt_grid_offset +
                  2.0f * kpc[base + 0 * plane]) *
                 stride;
          o[1] = (static_cast<float>(i) + g_.head.kpt_grid_offset +
                  2.0f * kpc[base + 1 * plane]) *
                 stride;
          float v = kpc[base + 2 * plane];
          if (g_.head.kpt_visibility == ScoreKind::Logit)
            v = 1.0f / (1.0f + std::exp(-v));
          o[2] = v;
        }
      }
    }
    offset += s * s;
  }
  transpose_mn_to_nchw(rows.data(), cells, C, out.d.data(), pool_);
  return out;
}

Tensor Network::run(const Tensor &input) {
  cost_.reset();
  if (input.c != 3)
    throw std::runtime_error("pose net: the input has " +
                             std::to_string(input.c) +
                             " channels; this front end produces 3 (RGB)");
  if (input.h != g_.input_size || input.w != g_.input_size)
    throw std::runtime_error("pose net: the input is " + std::to_string(input.h) +
                             "x" + std::to_string(input.w) + " and the model's "
                             "input_size is " + std::to_string(g_.input_size));

  std::vector<Tensor> vals(g_.graph.size());
  // Operand -1 is the image, matching geometry.cpp. vals[0] is node 0's OUTPUT,
  // so the image cannot live there: an offset-by-one would make node 1 read the
  // photograph where it should read the first convolution, and every channel
  // count after it would still add up.
  auto operand = [&](int idx) -> const Tensor & {
    return idx == -1 ? input : vals[static_cast<size_t>(idx)];
  };
  std::vector<float> a_buf;   // reused across convolutions: the widest im2col
                              // in the network is 6400x1152 floats = 29 MB and
                              // allocating that 73 times per image would be
                              // 2 GB of churn for one frame.
  Tensor head_out;
  bool have_head = false;

  if (on_node) on_node(kGraphInput, input);
  for (size_t i = 0; i < g_.graph.size(); ++i) {
    const Layer &L = g_.graph[i];
    switch (L.op) {
      case Op::Conv:
        vals[i] = conv(operand(L.inputs.at(0)),
                       g_.convs[static_cast<size_t>(L.conv)], L.silu,
                       L.stride, L.pad_h, L.pad_w, a_buf);
        break;
      case Op::Concat: {
        std::vector<const Tensor *> ins;
        ins.reserve(L.inputs.size());
        for (int idx : L.inputs)
          ins.push_back(&vals[static_cast<size_t>(idx)]);
        vals[i] = concat(ins);
        break;
      }
      case Op::Slice:
        vals[i] = slice(operand(L.inputs[0]), L.chan, L.nch);
        break;
      case Op::Add:
        vals[i] = add(operand(L.inputs[0]),
                      operand(L.inputs[1]));
        break;
      case Op::MaxPool:
        vals[i] = maxpool(operand(L.inputs.at(0)), L.k);
        break;
      case Op::Upsample:
        vals[i] = upsample(operand(L.inputs.at(0)), L.scale);
        break;
      case Op::Detect: {
        // The head consumes several earlier nodes and produces ONE tensor, so
        // it is the one op that reads a list rather than a single operand. The
        // pointers are into `vals`, which is not resized inside the loop, so
        // they stay valid for the duration of the call.
        std::vector<const Tensor *> ins;
        ins.reserve(L.inputs.size());
        for (int idx : L.inputs) ins.push_back(&operand(idx));
        // Into `vals` as well as `head_out`: `vals[i]` is the output of node i
        // for every node, and the detect node not filling its own slot left the
        // per-node dump showing a 0x0x0 tensor for the one node whose output is
        // the whole answer. head_out is kept separately because it is returned
        // and a return-by-value of vals[i] would copy 1.9 MB.
        vals[i] = head_out = this->head(ins);
        have_head = true;
        break;
      }
    }
    if (on_node) on_node(i, vals[i]);
  }
  if (!have_head)
    throw std::runtime_error("pose net: the graph produced no head tensor");
  return head_out;
}

}  // namespace npue::pose
