//===- net.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=6 network: one graph walk, two backends.
//
// ONE WALK, TWO BACKENDS
// ----------------------
// The graph in geometry.hpp is executed ONCE, and the only thing a backend
// choice changes is where a convolution's GEMM runs. That is the whole reason
// the graph is a list of ops rather than a call into per-layer code: a schedule
// that branched per LAYER would need the branch written twice, and the two
// copies would drift, and the drift would be a plausible pose.
//
//   host  (default)   im2col + a blocked fp32 GEMM, multi-threaded over rows
//   array (--npu-ops conv)   the same A rows into NpuGemm::run, chunked to
//                     the design's row count
//
// WHY CPU IS THE DEFAULT, WITH THE NUMBER ATTACHED
// -------------------------------------------------
// This is measured, not preferred. At 640x640 on this machine (Ryzen 7 PRO
// 8845HS, 16 threads) the network's 72 convolutions take 150 ms on the CPU.
// Dispatched to the array they take 290 ms, and the reason is structural rather
// than a tuning failure:
//
//   * a design's N must be a multiple of tile_n * AIE columns = 128 on npu1, and
//     most of this network's convolutions have N <= 64 -- so 4.59 GMAC of useful
//     work becomes 11.95 GMAC of dispatched work, 2.6x waste, before anything runs
//   * a dispatch is M <= 1024 rows, and the stem has M = 102400, so that one
//     layer is 100 of the network's 436 dispatches
//   * those 436 dispatches come to 660 us each inside conv(), of which only
//     140 us is the device's own GEMM; the rest is the host widening A to the
//     panel's K and transposing C back into NCHW
//
// Per layer, 19 of the 72 are faster on the array than on the host, and an
// oracle that put each layer on its faster side measured 146 ms against the
// host's 150 -- 2.6%, for choosing per layer. So `--npu-ops conv` exists as
// an explicit, measurable choice -- the same relationship the flag has to
// Whisper's ops -- and the status block prints both numbers so the trade is
// visible per run rather than argued in a comment.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/conv_host.hpp"   // npue::hostconv::Nchw, gemm_nt, im2col
#include "pose/geometry.hpp"
#include "runtime/pool.hpp"

namespace npue::pose {

// NCHW, defined once in runtime/include/common/conv_host.hpp because arch=7
// runs the same host convolution through the same im2col and the same blocked
// GEMM -- a second copy would be 300 lines that drift, and the drift would show
// up as the two architectures disagreeing about what the host convolution costs.
using Tensor = npue::hostconv::Nchw;

// Which ops the array is asked to run. An EMPTY placement is the default and
// means every op is on the host; that is the normal case, and it is why
// Network::run() needs no Design at all on the default path.
struct Placement {
  bool conv_on_array = false;
  // The design set's directory. Only consulted when conv_on_array, and its
  // absence there is a refusal rather than a silent fall back to the host: a flag
  // whose whole point is "put this on the array" must not quietly not do it.
  std::string design_dir;
};

// What one run cost, per backend. Printed by the status block, because the
// interesting number on this architecture is the SPLIT between them and a single
// wall-clock figure would hide it.
struct Cost {
  int64_t convs_host = 0;
  int64_t convs_array = 0;
  int64_t dispatches = 0;
  double t_host = 0.0;
  double t_array = 0.0;
  // The array path's own split. t_array is the sum of these and of the GEMM
  // calls; the two host-side spans are reported separately because on this
  // network they are most of it, and a reader told only "the array path takes
  // X" cannot tell an array that is too slow from a host that is too busy.
  double t_array_repack = 0.0;      // zero-pad + widen A to the design's K
  double t_array_transpose = 0.0;   // padded C -> NCHW at the design's stride
  // The host convolution's own breakdown. t_host is the sum of the three, so a
  // reader can tell a 13x gap against a reference GEMM from which of them is
  // responsible -- and NOT, which is why the split exists: measured only as one
  // number, "the network is slow" has three unrelated possible causes and
  // fixing the wrong one costs a day.
  double t_wmat = 0.0;       // [N, K] -> [K, N] weight transpose
  double t_im2col = 0.0;     // the A matrix, including its zero-fill
  double t_gemm = 0.0;       // the blocked fp32 multiply
  double t_transpose = 0.0;  // [M, N] -> NCHW [N, H, W]
  // Everything in run() that is NOT one of the four above: the SiLU epilogue,
  // and the 43 nodes that are not convolutions (join, split, residual add,
  // max-pool, upsample). It was the largest unattributed span in the network
  // -- network_s minus the four buckets, which came to 101 ms of 253 -- and it
  // was unattributed only because nothing timed it, not because it was small.
  double t_elementwise = 0.0;
  // The device call itself, MEASURED around npu_->gemm rather than recovered by
  // subtracting the host spans from t_array. The subtraction is wrong: t_array
  // starts at t0, which is before the weight transpose and before im2col, so a
  // residual labelled "GEMM on the device" silently carries both of those and
  // reported 63-66 ms where the call is a small fraction of that.
  double t_array_gemm = 0.0;
  void reset() { *this = Cost{}; }
};

// Executes Geometry::graph. Owns nothing the caller has to keep alive except
// the container the geometry's ConvW pointers point into.
class Network {
public:
  // `pool` is borrowed and must outlive the Network. `design` is only needed
  // when `place.conv_on_array` is set, and is borrowed the same way.
  Network(const Geometry &g, const Placement &place, app::Pool &pool,
          class NpuConvs *design);

  // One image: [3, S, S] in, the head tensor out. The head tensor is
  // [out_channels, cells] with the cells concatenated level by level in the
  // order geometry.strides/grid record, which is the order decode.cpp reads.
  Tensor run(const Tensor &input);

  // Called once per graph node, right after that node's output is computed, with
  // (node index, tensor) -- and ONCE BEFORE the graph, with (kGraphInput, the
  // input tensor). VERIFICATION ONLY: tools/verify/verify_pose.py uses it to find
  // which node the runtime and an independent reference of the ONNX first
  // disagree at, which end-to-end comparison cannot say -- a wrong first
  // convolution and a wrong twentieth are both "the detections differ".
  //
  // The input is included rather than left to the caller because the front end is
  // a plausible place for two implementations to disagree and it is invisible
  // from node 0 onwards: a wrong pad colour or a channel swap gives a node 0 that
  // is wrong and every later node wrong in a way that reads as the network's
  // fault.
  //
  // A std::function rather than a flag so that the normal path costs one branch
  // per node and the caller owns the memory and the format.
  static constexpr size_t kGraphInput = static_cast<size_t>(-1);
  std::function<void(size_t, const Tensor &)> on_node;

  const Cost &cost() const { return cost_; }

private:
  // The stride and the padding are PARAMETERS, not an assumption baked into
  // im2col. Two of this network's 72 convolutions are not "3x3 same stride 1":
  // the stem is a 3x3 with stride 2, and every 1x1 has zero padding. An im2col
  // that assumed k/2 padding and stride 1 would compute the right thing for 70
  // of them and a 642x642 pyramid for the other two -- and the typecheck would
  // have to model the same wrong rule, so the two would agree and the picture
  // would be the wrong size rather than an error.
  Tensor conv(const Tensor &in, const ConvW &w, bool silu, int64_t stride,
              int64_t pad_h, int64_t pad_w, std::vector<float> &a_buf);
  // Variadic because SPPF joins FOUR tensors (its own input and three pooled
  // copies) where the neck joins two. Two overloads would be the same body
  // twice; a vector is one body and the refusal below names the offending pair.
  Tensor concat(const std::vector<const Tensor *> &in);
  Tensor slice(const Tensor &in, int64_t chan, int64_t nch);
  Tensor add(const Tensor &a, const Tensor &b);
  Tensor maxpool(const Tensor &in, int64_t k);
  Tensor upsample(const Tensor &in, int64_t scale);
  Tensor head(const std::vector<const Tensor *> &in);

  const Geometry &g_;
  const Placement &place_;
  app::Pool &pool_;
  NpuConvs *npu_ = nullptr;
  // The GEMM's [M, N] result for the convolution in flight, transposed into the
  // output Tensor after every convolution. See transpose_mn_to_nchw in net.cpp for
  // why the two layouts cannot share a buffer. A member rather than a local
  // because it is the widest allocation in a run (6.6 MB at the stem) and there
  // are 72 convolutions; growing it once and keeping it is the difference between
  // 6.6 MB of allocation and 480 MB of churn per image.
  std::vector<float> c_scratch_;
  // The array path's TWO buffers, sized by the DESIGN's widths and not by the
  // convolution's. They are separate from c_scratch_ rather than shared with it
  // because their rows are wider: c_scratch_ is the convolution's real N and
  // these are the design's padded n, and a buffer sized for one cannot hold the
  // other without a stride the other does not use.
  //
  // Both are ONE DISPATCH WINDOW, not one convolution, so they stay small where a
  // whole-convolution buffer would not: at this geometry rows*pk is 9.4 MiB and
  // rows*pn is 1.0 MiB, against 900 MiB and 100 MiB for the unpadded whole thing.
  // That ratio is why the array path repacks per chunk instead of once per
  // convolution.
  std::vector<float> a_pad_;
  std::vector<float> c_pad_;
  // Every convolution's weights, already transposed from the container's [N, K]
  // into the [K, N] the GEMM reads, built ONCE in the constructor. It was
  // rebuilt per convolution per image, which is 3.28 M elements of strided write
  // and a fresh allocation 72 times a frame to produce a constant -- and on the
  // array path it was built and then never read, because the array takes the
  // panel the packer already laid out.
  //
  // A member rather than a local in conv() because that is the whole point: conv()
  // is per image and this is not. One flat buffer with per-convolution offsets,
  // not a vector of vectors, so the whole table is a single allocation and the
  // transposes happen in one pass in one order.
  std::vector<float> wmat_;
  std::vector<size_t> wmat_at_;
  // Buffers of tensors whose LAST consumer has already run, kept for reuse.
  // run() used to hold all 116 node outputs alive until the frame ended, which
  // is 116 allocations of up to 26 MB per image; the large ones go through
  // mmap, so free() gives the pages back and the next image re-faults every one
  // of them. That is the "neither" row in the README's table.
  std::vector<std::vector<float>> spare_;
  // release_at_[i] lists the node indices whose buffer is dead once node i has
  // been computed. Computed ONCE in the constructor from the graph's operand
  // lists, because deriving it per run would be per-run work to save per-run
  // allocations -- and a liveness bug here does not crash, it returns a buffer
  // that is about to be written by something else.
  std::vector<std::vector<size_t>> release_at_;
  // A Tensor whose storage comes from spare_ when one is big enough. Every
  // operator still ZEROES what it returns: reuse must not turn "allocated
  // zeroed" into "allocated dirty", because an operator that ever fails to
  // write all of its output would then read another tensor's leftovers -- a
  // wrong number, not a crash.
  Tensor take(int64_t c, int64_t h, int64_t w);
  Cost cost_;
};

// The array side of a convolution, owned by the Session and borrowed here.
//
// Separate from Network so that the default path links and runs with no XRT
// design at all: Network calls into it only when Placement says so, and the
// Session builds one only when --npu-ops named conv.
class NpuConvs {
public:
  virtual ~NpuConvs() = default;
  // out[m, n] = a[m, k] @ b[k, n] + bias[n], over m rows of k, n columns.
  // `conv` is the convolution's index in Geometry::convs, which is what the
  // backend keyed its staged panel by -- so the two backends cannot pick
  // different weights for the same layer, because there is only one lookup and
  // it is made once, at load.
  virtual int64_t gemm(int64_t conv, const float *a, int64_t m, int64_t k,
                       int64_t n, const float *bias, float *out) = 0;
  // The widths the DISPATCH must use, which are the design's and NOT the
  // convolution's. A design's K is rounded up to a multiple of tile_k and its N
  // to a multiple of tile_n * cols, and the compiled core runs exactly those
  // widths whatever the caller asks for: NpuGemm::run packs `m * k` A columns as
  // one contiguous run, adds `bias[j]` for every j < n, and reads back n columns
  // per C row. Passing the convolution's real widths instead makes the A row
  // stride wrong -- row r's padded tail reads row r+1's data -- and returns a
  // network that finds hundreds of people where there are three. That is not a
  // wrong number, it is a wrong shape, and nothing downstream can see it.
  //
  // The A rows the caller hands over must therefore already be `padded_k` wide
  // with columns [k, padded_k) zeroed, and the C rows it receives back are
  // `padded_n` wide.
  virtual int64_t padded_k(int64_t conv) const = 0;
  virtual int64_t padded_n(int64_t conv) const = 0;
  // The design's row count, which is the chunk size conv() must not exceed.
  virtual int64_t rows_per_dispatch() const = 0;
  virtual int64_t dispatches() const = 0;
  virtual double t_array() const = 0;
};

}  // namespace npue::pose
