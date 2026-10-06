//===- net.cpp -----------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=8 graph walk and its two heads.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "mppose/net.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/host_kernels.hpp"   // app::now_s

namespace npue::mppose {

using npue::hostconv::gemm_nt;
using npue::hostconv::im2col;
using npue::hostconv::transpose_mn_to_nchw;

namespace {

// The graph's declared output extent, recomputed from the op and REFUSED if it
// disagrees. Every op in the list carries its own `out`, which the packer wrote by
// running the ONNX; recomputing it here and comparing means a container whose
// padding or stride does not produce the extent it declares is refused at load
// rather than producing a tensor of the declared shape filled with whatever the
// kernel happened to write.
//
// FLOOR, NOT EXACT DIVISION, AND THAT IS THE WHO POINT ABOUT THE PADDING HERE.
// arch=6's and arch=7's check requires the padded extent to divide by the stride,
// which their checkpoints' "SAME padding evaluated at a stride" happens to satisfy.
// This one's does NOT: the detector's stem is a 3x3 at stride 2 with a SYMMETRIC
// pad of [1,1,1,1], so 224 + 1 + 1 - 3 = 223 and 223 / 2 is 111.5. ONNX takes
// floor, the declared output is 112, and a check demanding an exact division
// refuses the model's own first convolution.
//
// The four terms are still carried and still compared, because the extent is where
// a wrong pad bites: [1,1,2,2] on this same stem gives 224, hence 113, and is
// caught here. The stems on THIS checkpoint are symmetric, which is worth saying
// because arch=7's header claims MediaPipe's are not and that claim is true there
// (the palm detector's is [1,1,2,2]) -- a convention that happens to hold is not
// one to assume, and it is not one this runtime assumes.
void check_extent(const Layer &l, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  const std::string &label) {
  const int64_t eh = H + l.pad[0] + l.pad[2] - kh;
  const int64_t ew = W + l.pad[1] + l.pad[3] - kw;
  if (eh < 0 || ew < 0)
    throw std::runtime_error(
        label + ": op '" + op_to_string(l.op) + "' has pad [" +
        std::to_string(l.pad[0]) + "," + std::to_string(l.pad[1]) + "," +
        std::to_string(l.pad[2]) + "," + std::to_string(l.pad[3]) +
        "] with a " + std::to_string(kh) + "x" + std::to_string(kw) +
        " kernel on a " + std::to_string(H) + "x" + std::to_string(W) +
        " input, so the kernel is larger than the padded input and there is no "
        "output at all.");
  const int64_t oh = eh / l.stride_h + 1, ow = ew / l.stride_w + 1;
  if (oh != l.out_h || ow != l.out_w)
    throw std::runtime_error(
        label + ": op '" + op_to_string(l.op) + "' computes " +
        std::to_string(oh) + "x" + std::to_string(ow) +
        " and the container says its output is " + std::to_string(l.out_h) +
        "x" + std::to_string(l.out_w) +
        ". The two are written by the packer and here from the same op, so a "
        "disagreement is a container whose op list was edited -- and the usual "
        "cause is a four-term pad transcribed as a symmetric one.");
}

npue::hostconv::Act host_act(Act a, const float *slope) {
  // hostconv::Act and mppose::Act are two enums on purpose -- one is what the
  // graph says, the other is what the epilogue can compute -- and this is the one
  // place they are mapped.
  using H = npue::hostconv::Act;
  switch (a) {
    case Act::None: return H::kNone;
    case Act::Relu: return H::kRelu;
    case Act::Relu6: return H::kRelu6;
    case Act::PreLU:
      if (!slope)
        throw std::runtime_error(
            "the graph fuses a prelu and the container has no slope vector for "
            "it");
      return H::kPreLU;
  }
  return H::kNone;
}

// The sigmoid the landmark head's confidence passes through. Written here rather
// than taken from hostconv's epilogue because this one is NOT fused into a
// convolution: the graph's Sigmoid is a node of its own, the packer absorbed it
// into the head spec as a flag, and so it is the front end's turn. Same function,
// one definition.
inline float sigmoid1(float v) {
  return 1.0f / (1.0f + std::exp(-v));
}

}  // namespace

Network::Network(const Geometry &g, const Placement &place, app::Pool &pool,
                 conv::Convs *array)
    : g_(g), place_(place), pool_(pool), array_(array) {
  // THE WEIGHT TRANSPOSE HAPPENS HERE, ONCE, AND NOT PER CALL.
  //
  // gemm_nt's B is [K, N] -- one row per reduction step, N output columns
  // contiguous -- and the container stores the filter as [Cout, Cin, kh, kw],
  // which is [N, K]. Passing it unchanged is a shape-compatible mistake (both are
  // K*N floats) that yields a convolution with a TRANSPOSED filter: the output
  // has the right shape, every channel is a plausible mixture of the right ones,
  // and the failure is this architecture's accuracy rather than a crash.
  //
  // It is a STARTUP cost because the filter is the same on every frame. Paying it
  // inside conv() would grow with the number of convolutions rather than with the
  // number of images, which is the wrong way round.
  //
  // DEPTHWISE FILTERS ARE NOT TRANSPOSED, and that is not an oversight. Their
  // reduction is over kh*kw WITHIN one channel, so dwconv() indexes
  // w[(c*kh + i)*kw + j] straight off the container's own layout and there is no
  // [K, N] view of one filter to build. The entry is left nullptr, and conv()
  // refuses a dense convolution without one rather than reading the untransposed
  // filter.
  const auto build = [](const std::vector<ConvW> &tab,
                        std::vector<std::vector<float>> &store,
                        std::vector<const float *> &ptrs,
                        const std::string &label) {
    ptrs.assign(tab.size(), nullptr);
    store.assign(tab.size(), {});
    for (size_t i = 0; i < tab.size(); ++i) {
      const ConvW &cw = tab[i];
      if (!cw.w || cw.depthwise()) continue;
      const int64_t K = cw.cin * cw.kh * cw.kw, N = cw.cout;
      store[i].assign(static_cast<size_t>(K * N), 0.f);
      npue::hostconv::transpose_nk_to_kn(cw.w, N, K, store[i].data());
      ptrs[i] = store[i].data();
    }
    (void)label;
  };
  build(g.det_convs, wmat_det_, wt_det_, "det");
  build(g.pose_convs, wmat_pose_, wt_pose_, "pose");
}

// -- dense convolution -------------------------------------------------------

// The array half of one dense convolution, and arch=6's second implementation of
// the same idea. It is written out rather than shared because the two
// architectures disagree about something structural: arch=6 has one graph and
// addresses panels by its own convolution index, arch=8 has TWO and must offset
// that index into the slot range its network owns (Placement::slot_base). A
// shared body would take a slot number as a parameter, and the parameter would
// then be computed twice -- once here and once by the caller -- which is the same
// two-spellings-of-one-mapping this file already got wrong once.
//
// EVERY WIDTH BELOW IS THE DESIGN'S, NOT THE CONVOLUTION'S, and that is the whole
// difference between the two backends:
//
//   * K pads up to a multiple of tile_k. The design reads `rows * k` columns of A
//     as ONE contiguous run, so A must be REPACKED at the padded stride and its
//     tail zeroed. Left at the real stride, row r's tail columns read row r+1's
//     data -- a network that is well shaped and confidently wrong.
//   * N pads up to a multiple of tile_n * cols. The design writes `rows * n`
//     columns of C whatever N was asked for, and the host narrows back to the
//     real N on the way out.
//   * The bias is read for all n columns, which is why the backend holds a
//     zero-padded copy rather than this convolution's own.
//
// The repack is per CHUNK, not per convolution: rows*pk is 256 KiB at this
// geometry while the whole convolution's A can be tens of megabytes, and the
// array never needs more than one dispatch window at a time.
Tensor Network::conv_array(const Tensor &in, Tensor &out, const Layer &l,
                           const ConvW &w, int64_t M, int64_t K, int64_t N,
                           const float *slope, double t0,
                           const std::string &label) {
  // THE SLOT COMES FROM A TABLE, NEVER FROM ARITHMETIC ON THE CONVOLUTION INDEX.
  // See Placement::slots: the two disagree because slot numbers are handed out
  // over the DENSE convolutions only, and depthwise ones are interleaved through
  // both graphs. `-1` means there is no panel, which is the depthwise case and
  // which conv() never reaches -- dwconv has its own path -- so it is refused
  // rather than dispatched against whatever occupies that row.
  if (!place_.slots || w.index < 0 ||
      static_cast<size_t>(w.index) >= place_.slots->size())
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " is outside this network's slot table, which has " +
                             std::to_string(place_.slots ? place_.slots->size() : 0) +
                             " entries");
  const int64_t slot = (*place_.slots)[static_cast<size_t>(w.index)];
  if (slot < 0)
    throw std::runtime_error(
        label + ": conv " + std::to_string(w.index) +
        " has no array panel (slot -1). Only the DENSE convolutions are staged: "
        "a depthwise filter reduces within one channel, so it has no [M, N] "
        "GEMM to dispatch. This layer reached the dense path, so the graph and "
        "the weight table disagree about whether it is depthwise.");

  // The dispatch itself is conv::conv_array_dispatch, shared with arch=6 and
  // arch=7. Written out here it would have been the third copy of a loop whose
  // three silent failure modes are the padded-K repack, the narrowed C and the
  // per-chunk accounting.
  const double t_before = app::now_s();
  conv::conv_array_dispatch(*array_, slot, a_.data(), M, K, N, out.d.data(),
                            out.plane(), host_act(l.act, slope), slope, pool_,
                            apad_, cpad_, &cost_.t_array_repack,
                            &cost_.t_array_gemm, &cost_.t_array_transpose,
                            &cost_.dispatches);
  cost_.convs_array++;
  cost_.t_array += app::now_s() - t_before;
  (void)t0;
  return out;
}

Tensor Network::conv(const Tensor &in, const Layer &l, const ConvW &w,
                     const float *wkn, const float *slope,
                     const std::string &label) {
  check_extent(l, in.h, in.w, w.kh, w.kw, label);
  if (w.cin != in.c)
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " takes " + std::to_string(w.cin) +
                             " channels and the tensor has " +
                             std::to_string(in.c) +
                             ". Reading the other one would index past the end "
                             "of the im2col row.");
  if (w.cout != l.out_c)
    throw std::runtime_error(label + ": conv " + std::to_string(w.index) +
                             " has " + std::to_string(w.cout) +
                             " outputs and the graph says " +
                             std::to_string(l.out_c));
  if (!wkn)
    throw std::runtime_error(
        label + ": conv " + std::to_string(w.index) +
        " has no [K,N] weight table. The transpose is built in the constructor "
        "for every DENSE filter, so a missing one means this convolution is "
        "depthwise and was routed here by name -- which would compute a different "
        "network rather than fail.");

  Tensor out(l.out_c, l.out_h, l.out_w);
  const double t0 = app::now_s();
  im2col(in, w.kh, w.kw, l.pad[0], l.pad[1], l.stride_h, l.stride_w, a_, pool_,
         l.pad[2], l.pad[3]);
  const int64_t M = l.out_h * l.out_w, K = w.cin * w.kh * w.kw, N = w.cout;
  cost_.t_im2col += app::now_s() - t0;

  if (array_) return conv_array(in, out, l, w, M, K, N, slope, t0, label);

  static const std::vector<float> kZero;
  const double t1 = app::now_s();
  c_.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
  gemm_nt(a_.data(), M, wkn, K, N, w.b ? w.b : kZero.data(), c_.data(), pool_);
  const double t2 = app::now_s();
  cost_.t_gemm += t2 - t1;

  transpose_mn_to_nchw(c_.data(), M, N, out.d.data(), pool_, 0, 0,
                       host_act(l.act, slope), slope);
  cost_.t_transpose += app::now_s() - t2;
  cost_.convs_host++;
  return out;
}

// -- depthwise convolution ---------------------------------------------------

Tensor Network::dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                       const float *slope, const std::string &label) {
  check_extent(l, in.h, in.w, w.kh, w.kw, label);
  if (w.cout != in.c)
    throw std::runtime_error(
        label + ": dwconv " + std::to_string(w.index) + " has " +
        std::to_string(w.cout) + " filters and the tensor has " +
        std::to_string(in.c) + " channels. One filter per channel means they are "
        "the same number, and reading it as a dense convolution would compute a "
        "different network rather than fail.");
  if (l.out_c != in.c)
    throw std::runtime_error(label + ": dwconv " + std::to_string(w.index) +
                             " changes the channel count from " +
                             std::to_string(in.c) + " to " +
                             std::to_string(l.out_c) +
                             ". A depthwise convolution cannot: it would need a "
                             "projection, which is what the `conv` after it is "
                             "for.");
  Tensor out(l.out_c, l.out_h, l.out_w);
  const int64_t H = in.h, W = in.w, kh = w.kh, kw = w.kw;
  const int64_t pt = l.pad[0], pl = l.pad[1];
  const Act act = l.act;
  const int64_t KH = l.out_h, KW = l.out_w;
  const double t0 = app::now_s();
  par_items(pool_, in.c, out.plane(), [&](int64_t c) {
    const float *const src = in.chw(c);
    const float *const wf = w.w + static_cast<size_t>(c) * kh * kw;
    float *const dst = out.chw(c);
    const float b = w.b ? w.b[c] : 0.f;
    const float sl = slope ? slope[c] : 0.f;
    for (int64_t y = 0; y < KH; ++y) {
      for (int64_t x = 0; x < KW; ++x) {
        float acc = b;
        for (int64_t i = 0; i < kh; ++i) {
          const int64_t sy = y * l.stride_h + i - pt;
          if (sy < 0 || sy >= H) continue;
          for (int64_t j = 0; j < kw; ++j) {
            const int64_t sx = x * l.stride_w + j - pl;
            if (sx < 0 || sx >= W) continue;
            acc += wf[i * kw + j] * src[static_cast<size_t>(sy) * W + sx];
          }
        }
        dst[static_cast<size_t>(y) * KW + x] = act1(act, acc, sl);
      }
    }
  });
  cost_.t_dwconv += app::now_s() - t0;
  cost_.dwconvs_host++;
  return out;
}

// -- residual add, with the activation possibly fused ------------------------

Tensor Network::add(const Tensor &a, const Tensor &b, const Layer &l,
                    const float *slope, const std::string &label) {
  if (a.c != b.c || a.h != b.h || a.w != b.w)
    throw std::runtime_error(
        label + ": add joins a " + std::to_string(a.c) + "x" +
        std::to_string(a.h) + "x" + std::to_string(a.w) + " and a " +
        std::to_string(b.c) + "x" + std::to_string(b.h) + "x" +
        std::to_string(b.w) + ". Neither network has a broadcast add in its "
        "graph, so the two shapes are a transcription error rather than "
        "something to broadcast: broadcasting here would silently accept a block "
        "that resized instead of adding.");
  Tensor out(l.out_c, l.out_h, l.out_w);
  if (out.c != a.c || out.h != a.h || out.w != a.w)
    throw std::runtime_error(label + ": add's declared output is " +
                             std::to_string(l.out_c) + "x" +
                             std::to_string(l.out_h) + "x" +
                             std::to_string(l.out_w) + " and its operands are " +
                             std::to_string(a.c) + "x" + std::to_string(a.h) +
                             "x" + std::to_string(a.w));
  const Act act = l.act;
  const size_t n = a.d.size();
  const double t0 = app::now_s();
  par_items(pool_, static_cast<int64_t>(n), 1, [&](int64_t i) {
    out.d[static_cast<size_t>(i)] = act1(act, a.d[static_cast<size_t>(i)] +
                                                    b.d[static_cast<size_t>(i)],
                                        slope ? slope[(i / a.plane()) % a.c] : 0.f);
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- max pool, with four padding terms for the same reason --------------------

Tensor Network::maxpool(const Tensor &in, const Layer &l,
                        const std::string &label) {
  check_extent(l, in.h, in.w, l.kernel_h, l.kernel_w, label);
  if (l.out_c != in.c)
    throw std::runtime_error(label + ": maxpool changes the channel count");
  Tensor out(l.out_c, l.out_h, l.out_w);
  const int64_t H = in.h, W = in.w, kh = l.kernel_h, kw = l.kernel_w;
  const int64_t pt = l.pad[0], pl = l.pad[1];
  const double t0 = app::now_s();
  par_items(pool_, in.c, out.plane(), [&](int64_t c) {
    const float *const src = in.chw(c);
    float *const dst = out.chw(c);
    for (int64_t y = 0; y < l.out_h; ++y) {
      for (int64_t x = 0; x < l.out_w; ++x) {
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < kh; ++i) {
          const int64_t sy = y * l.stride_h + i - pt;
          if (sy < 0 || sy >= H) continue;
          for (int64_t j = 0; j < kw; ++j) {
            const int64_t sx = x * l.stride_w + j - pl;
            if (sx < 0 || sx >= W) continue;
            best = std::max(best, src[static_cast<size_t>(sy) * W + sx]);
          }
        }
        // A window entirely in the padding has no samples. -inf is the honest
        // value for a max over nothing; a 0 default would be a different tensor.
        dst[static_cast<size_t>(y) * l.out_w + x] = best;
      }
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- ONNX Resize, linear, half_pixel, exclude_outside=0 ----------------------

Tensor Network::resize(const Tensor &in, const Layer &l,
                       const std::string &label) {
  if (l.out_c != in.c)
    throw std::runtime_error(label + ": resize changes the channel count");
  const int64_t sh = in.h, sw = in.w, dh = l.out_h, dw = l.out_w;
  Tensor out(l.out_c, l.out_h, l.out_w);
  const double t0 = app::now_s();
  // The column indices and weights are the same for every channel, so they are
  // computed once and the work is a separable two-pass blend per plane.
  std::vector<int64_t> y0(dh), y1(dh), x0(dw), x1(dw);
  std::vector<float> wy(dh), wx(dw);
  for (int64_t i = 0; i < dh; ++i) {
    const double s = (i + 0.5) / l.scale_h - 0.5;
    const double f = std::floor(s);
    y0[i] = std::min<int64_t>(std::max<int64_t>(f, 0), sh - 1);
    y1[i] = std::min<int64_t>(std::max<int64_t>(f + 1, 0), sh - 1);
    wy[i] = static_cast<float>(s - f);
  }
  for (int64_t i = 0; i < dw; ++i) {
    const double s = (i + 0.5) / l.scale_w - 0.5;
    const double f = std::floor(s);
    x0[i] = std::min<int64_t>(std::max<int64_t>(f, 0), sw - 1);
    x1[i] = std::min<int64_t>(std::max<int64_t>(f + 1, 0), sw - 1);
    wx[i] = static_cast<float>(s - f);
  }
  // The row scratch is PER WORKER, and that is a correctness matter rather than a
  // tidiness one: a shared one is 16 threads writing one buffer. It went unnoticed
  // in arch=7 because par_items only parallelises above 65536 elements, so the
  // first resize ran serially and was right and the second went parallel and was
  // wrong. A race that only shows up above a threshold the same file chose is
  // exactly the kind that survives a review.
  par_items(pool_, in.c, out.plane(), [&](int64_t c) {
    std::vector<float> row(static_cast<size_t>(dw));
    const float *const src = in.chw(c);
    float *const dst = out.chw(c);
    for (int64_t i = 0; i < dh; ++i) {
      const float *const a = src + static_cast<size_t>(y0[i]) * sw;
      const float *const b = src + static_cast<size_t>(y1[i]) * sw;
      const float w = wy[i];
      for (int64_t j = 0; j < dw; ++j)
        row[static_cast<size_t>(j)] = a[x0[j]] * (1.f - w) + b[x0[j]] * w;
      for (int64_t j = 0; j < dw; ++j) {
        const float l0 = row[static_cast<size_t>(j)];
        const float l1 = a[x1[j]] * (1.f - w) + b[x1[j]] * w;
        dst[static_cast<size_t>(i) * dw + j] = l0 * (1.f - wx[j]) + l1 * wx[j];
      }
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- DepthToSpace ------------------------------------------------------------
//
// THE ONE OP THIS ARCHITECTURE EXISTS PARTLY TO EXPRESS. out[co, h*b + d1,
// w*b + d2] = in[ci, h, w], where the input channel index ci is the only thing
// the two ONNX orders disagree about:
//
//   CRD  ci = co*b*b + d1*b + d2      the input channel axis splits as
//                                    (co, d1, d2) -- row-major
//   DCR  ci = (d1*b + d2)*cout + co  the axis splits as (d1, d2, co) --
//                                    column-major, and ONNX's DEFAULT when the
//                                    attribute is absent, which is this model's
//                                    case
//
// Everything else is a copy: one input plane lands at one b x b tile of one
// output channel. So the work is b*b plane copies per output channel and no
// arithmetic at all, which is why this is timed as its own span rather than
// folded into the elementwise total -- it is the one op here whose cost is
// bandwidth and not a reduction.
//
// WHY THE ORDER IS NOT ASSUMED. Taking the wrong one is not a shape error and
// not a range error: every channel lands on a different pixel of a map that has
// the right shape and a plausible distribution, the next convolution consumes it
// happily, and the result is a detector that finds people in the wrong places.
// That was measured, not imagined: the packed op list executed against
// onnxruntime with CRD in place of DCR diverged at this op by 0.99 in
// peak-relative error while every convolution above it matched to 1e-06.
Tensor Network::d2s(const Tensor &in, const Layer &l, const std::string &label) {
  const int64_t b = l.block;
  if (b < 1)
    throw std::runtime_error(label + ": d2s has blocksize " +
                             std::to_string(b));
  const int64_t bb = b * b;
  if (in.c % bb)
    throw std::runtime_error(
        label + ": d2s has " + std::to_string(in.c) + " input channels and a "
        "blocksize of " + std::to_string(b) + ", and " + std::to_string(in.c) +
        " is not a multiple of b^2. DepthToSpace divides the CHANNEL count by "
        "the block area; there is no other reading of an indivisible one.");
  const int64_t cout = in.c / bb;
  if (l.out_c != cout || l.out_h != in.h * b || l.out_w != in.w * b)
    throw std::runtime_error(
        label + ": d2s goes " + std::to_string(in.c) + "x" + std::to_string(in.h) +
        "x" + std::to_string(in.w) + " -> " + std::to_string(l.out_c) + "x" +
        std::to_string(l.out_h) + "x" + std::to_string(l.out_w) +
        " with blocksize " + std::to_string(b) + ", which is not the "
        "[C/b^2, H*b, W*b] this operator means.");
  // The two orders are the SAME loop with a different address for the source
  // plane, so they are one function and not two -- a second implementation here
  // would be a second thing to keep correct for a difference of one expression.
  const bool crd = l.crd;
  Tensor out(l.out_c, l.out_h, l.out_w);
  const int64_t H = in.h, W = in.w;
  const double t0 = app::now_s();
  par_items(pool_, cout, out.plane(), [&](int64_t co) {
    float *const dst = out.chw(co);
    for (int64_t d1 = 0; d1 < b; ++d1) {
      for (int64_t d2 = 0; d2 < b; ++d2) {
        const int64_t ci = crd ? (co * b + d1) * b + d2
                               : (d1 * b + d2) * cout + co;
        const float *const src = in.chw(ci);
        // OUT ROW = y*b + d1 AND OUT COLUMN = x*b + d2. BOTH put the WITHIN-BLOCK
        // position in the low part and the block index in the high part -- ONNX's
        // out[co, h*b + d1, w*b + d2] = in[ci, h, w]. The version written first put
        // d2 in the high part of the column and the source column in the low one,
        // which is the transpose of that axis: same shape, same range, every
        // channel on a different pixel.
        //
        // d1 AND d2 ARE BOTH OUTER LOOPS, because each of them SELECTS THE SOURCE
        // PLANE -- ci depends on both -- and neither of them is a broadcast. An
        // earlier revision moved d2 inside and wrote srow[x] into both slots,
        // which fills every pair with one plane's value and is wrong in a way no
        // shape check sees. For a given d1, each d2 fills the columns congruent to
        // d2, and over all b*b pairs every output element is written once.
        for (int64_t y = 0; y < H; ++y) {
          float *const drow = dst + static_cast<size_t>(y * b + d1) * l.out_w;
          const float *const srow = src + static_cast<size_t>(y) * W;
          for (int64_t x = 0; x < W; ++x)
            drow[x * b + d2] = srow[x];
        }
      }
    }
  });
  cost_.t_d2s += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- the walk ----------------------------------------------------------------

const std::vector<Tensor> &Network::run_body(
    const Tensor &input, const std::vector<Layer> &graph,
    const std::vector<ConvW> &convs, const std::vector<const float *> &wkn,
    const std::vector<Slope> &slopes, const std::vector<std::string> &names,
    const std::string &label) {
  const size_t n = graph.size();
  // Index 0 is the image and node i is at i+1, so an op's operand -1 reads the
  // image and every other operand is its node index plus one. See the header.
  body_.assign(n + 1, Tensor());
  body_[0] = input;
  for (size_t i = 0; i < n; ++i) {
    const Layer &l = graph[i];
    const std::string nl = label + " node " + std::to_string(i) + " '" +
                           (i < names.size() ? names[i] : std::string("?")) +
                           "'";
    const float *slope = nullptr;
    if (l.act == Act::PreLU) {
      if (l.prelu < 0 || static_cast<size_t>(l.prelu) >= slopes.size())
        throw std::runtime_error(nl + ": prelu " + std::to_string(l.prelu) +
                                 " is past the network's " +
                                 std::to_string(slopes.size()) +
                                 " slope vectors");
      slope = slopes[static_cast<size_t>(l.prelu)].v;
      if (slopes[static_cast<size_t>(l.prelu)].n != l.out_c)
        throw std::runtime_error(
            nl + ": prelu " + std::to_string(l.prelu) + " has " +
            std::to_string(slopes[static_cast<size_t>(l.prelu)].n) +
            " slopes for a " + std::to_string(l.out_c) +
            "-channel tensor. PReLU's slope is PER CHANNEL, so a length "
            "mismatch has no per-pixel meaning.");
    }
    switch (l.op) {
      case Op::Conv: {
        if (l.conv < 0 || static_cast<size_t>(l.conv) >= wkn.size())
          throw std::runtime_error(nl + ": conv " + std::to_string(l.conv) +
                                   " is past this network's " +
                                   std::to_string(wkn.size()) +
                                   " weight tables");
        body_[i + 1] =
            conv(body_[static_cast<size_t>(l.inputs[0]) + 1], l,
                 g_.conv_of(convs, l, nl),
                 wkn[static_cast<size_t>(l.conv)], slope, nl);
        break;
      }
      case Op::DwConv:
        body_[i + 1] =
            dwconv(body_[static_cast<size_t>(l.inputs[0]) + 1], l,
                   g_.conv_of(convs, l, nl), slope, nl);
        break;
      case Op::Add: {
        if (l.inputs.size() != 2)
          throw std::runtime_error(nl + ": add has " +
                                   std::to_string(l.inputs.size()) +
                                   " operands, not two");
        body_[i + 1] = add(body_[static_cast<size_t>(l.inputs[0]) + 1],
                           body_[static_cast<size_t>(l.inputs[1]) + 1], l, slope,
                           nl);
        break;
      }
      case Op::MaxPool:
        body_[i + 1] = maxpool(body_[static_cast<size_t>(l.inputs[0]) + 1], l, nl);
        break;
      case Op::Resize:
        body_[i + 1] = resize(body_[static_cast<size_t>(l.inputs[0]) + 1], l, nl);
        break;
      case Op::D2S:
        body_[i + 1] = d2s(body_[static_cast<size_t>(l.inputs[0]) + 1], l, nl);
        break;
    }
    // AFTER the switch, so the hook sees the activation too -- it fires on the
    // node's OUTPUT, and an activation fused into the convolution is part of what
    // the node produced.
    if (on_node) on_node(i, body_[i + 1]);
  }
  return body_;
}

// -- the detector head -------------------------------------------------------

DetOut Network::det_head(const std::vector<Tensor> &nodes,
                         const std::string &label) const {
  DetOut o;
  std::vector<double> boxes, scores;
  const size_t n_nodes = nodes.empty() ? 0 : nodes.size() - 1;
  for (size_t li = 0; li < g_.det_levels.size(); ++li) {
    const DetLevel &L = g_.det_levels[li];
    const std::string el = label + " level " + std::to_string(li);
    if (L.box_op < 0 || static_cast<size_t>(L.box_op) >= n_nodes ||
        L.score_op < 0 || static_cast<size_t>(L.score_op) >= n_nodes)
      throw std::runtime_error(el + ": names graph node " +
                               std::to_string(L.box_op) + " / " +
                               std::to_string(L.score_op) +
                               ", which is not one of the detector's " +
                               std::to_string(n_nodes) + " nodes");
    const Tensor &b = node(static_cast<size_t>(L.box_op));
    const Tensor &s = node(static_cast<size_t>(L.score_op));
    if (b.c != L.per * g_.det_row_terms || b.h != L.h || b.w != L.w)
      throw std::runtime_error(
          el + ": the box feature is " + std::to_string(b.c) + "x" +
          std::to_string(b.h) + "x" + std::to_string(b.w) + " and the head says " +
          std::to_string(L.per * g_.det_row_terms) + "x" + std::to_string(L.h) +
          "x" + std::to_string(L.w) +
          ". Both are written by the packer from the ONNX's own output names, so "
          "a disagreement is one of them a transcription.");
    if (s.c != L.per || s.h != L.h || s.w != L.w)
      throw std::runtime_error(el + ": the score feature is " +
                               std::to_string(s.c) + "x" + std::to_string(s.h) +
                               "x" + std::to_string(s.w) + " against " +
                               std::to_string(L.per) + "x" + std::to_string(L.h) +
                               "x" + std::to_string(L.w));
    boxes.reserve(boxes.size() +
                  static_cast<size_t>(L.rows) * static_cast<size_t>(g_.det_row_terms));
    scores.reserve(scores.size() + static_cast<size_t>(L.rows));
    for (int64_t y = 0; y < L.h; ++y) {
      for (int64_t x = 0; x < L.w; ++x) {
        const size_t cell = static_cast<size_t>(y * L.w + x);
        for (int64_t a = 0; a < L.per; ++a) {
          for (int64_t t = 0; t < g_.det_row_terms; ++t)
            boxes.push_back(b.chw(a * g_.det_row_terms + t)[cell]);
          scores.push_back(s.chw(a)[cell]);
        }
      }
    }
  }
  o.boxes = std::move(boxes);
  o.scores = std::move(scores);
  o.n = static_cast<int64_t>(o.scores.size());
  if (o.n != g_.det_num_anchors)
    throw std::runtime_error(label + ": the levels produced " +
                             std::to_string(o.n) + " anchors and the container "
                             "declares " + std::to_string(g_.det_num_anchors) +
                             ". The decode reads a table of the declared width, "
                             "so this is a refusal rather than a read past the "
                             "end.");
  return o;
}

// -- the landmark head -------------------------------------------------------

namespace {

// One output's graph node, checked against the shape the container recorded.
const Tensor &head_node(const Network &net, const PoseOut &p, size_t n_nodes,
                        const std::string &label) {
  const std::string ol = label + "/" + p.name;
  if (p.op < 0 || static_cast<size_t>(p.op) >= n_nodes)
    throw std::runtime_error(ol + ": names graph node " + std::to_string(p.op) +
                             ", which is not one of the landmark network's " +
                             std::to_string(n_nodes) + " nodes");
  const Tensor &t = net.node(static_cast<size_t>(p.op));
  // The node is NCHW and the recorded conv_out is [1, C, H, W].
  if (t.c != p.conv_out[1] || t.h != p.conv_out[2] || t.w != p.conv_out[3])
    throw std::runtime_error(
        ol + ": its convolution wrote " + std::to_string(p.conv_out[1]) + "x" +
        std::to_string(p.conv_out[2]) + "x" + std::to_string(p.conv_out[3]) +
        " and node " + std::to_string(p.op) + " holds " + std::to_string(t.c) +
        "x" + std::to_string(t.h) + "x" + std::to_string(t.w) +
        ". Both come from the same packer pass over the same graph.");
  return t;
}

}  // namespace

PoseOuts Network::pose_head(const std::vector<Tensor> &nodes,
                            const std::string &label) const {
  const std::string hl = label + " landmark head";
  const size_t n_nodes = nodes.empty() ? 0 : nodes.size() - 1;
  PoseOuts o;

  // landmarks: declared [1, 195] and the convolution writes [1, 195, 1, 1], so
  // the flat order is the graph's own and nothing is reordered. Read as float
  // and widened: the world output is the arithmetic that follows and it is done
  // in double on both sides of the comparison.
  {
    const PoseOut &p = g_.pose_out("landmarks", hl);
    const Tensor &t = head_node(*this, p, n_nodes, hl);
    if (p.transposed)
      throw std::runtime_error(
          hl + "/landmarks: the container records a transpose on the way out, "
          "and this output is the screen landmark table -- 39 rows of "
          "x/y/z/visibility/presence -- which a transpose would turn into 195 "
          "coordinates that are individually plausible.");
    const size_t want = static_cast<size_t>(g_.num_landmarks * g_.lm_cols);
    if (t.d.size() != want)
      throw std::runtime_error(
          hl + "/landmarks: node " + std::to_string(p.op) + " holds " +
          std::to_string(t.d.size()) + " values and " +
          std::to_string(g_.num_landmarks) + "x" + std::to_string(g_.lm_cols) +
          " = " + std::to_string(want) + " were expected. The zoo's _postprocess "
          "reshapes this to 39 rows of 5, so a different count is a different "
          "model.");
    o.landmarks.assign(t.d.begin(), t.d.end());
  }

  // conf: one value, and the SIGMOID IS THIS HEAD'S JOB. The graph's Sigmoid is a
  // node of its own; the packer absorbed it and recorded that it did, and this is
  // where it is applied. Reading the node raw would compare a logit against
  // 0.5 and reject every person in the frame.
  {
    const PoseOut &p = g_.pose_out("conf", hl);
    const Tensor &t = head_node(*this, p, n_nodes, hl);
    if (t.d.size() != 1)
      throw std::runtime_error(hl + "/conf: node " + std::to_string(p.op) +
                               " holds " + std::to_string(t.d.size()) +
                               " values; the confidence is one number");
    if (!p.sigmoid)
      throw std::runtime_error(
          hl + "/conf: the container does not record a sigmoid on this output. "
          "It is compared against " + std::to_string(g_.pose_conf_threshold) +
          " and returned to the caller, both of which assume a probability; an "
          "unsquashed logit reads about -9.7 and silently rejects every person.");
    o.conf = sigmoid1(t.d[0]);
  }

  // mask: declared [1, 256, 256, 1] and the convolution writes [1, 1, 256, 256].
  // The single channel is dropped and the flat order is unchanged -- it is a
  // Reshape, not a Transpose -- so the result is row-major [y, x], which is what
  // the postprocess resamples.
  {
    const PoseOut &p = g_.pose_out("mask", hl);
    const Tensor &t = head_node(*this, p, n_nodes, hl);
    if (p.transposed)
      throw std::runtime_error(
          hl + "/mask: the container records a transpose on the way out, and a "
          "transposed segmentation mask is still a segmentation mask -- of the "
          "wrong body.");
    const size_t want = static_cast<size_t>(g_.pose_input_size) *
                        static_cast<size_t>(g_.pose_input_size);
    if (t.d.size() != want)
      throw std::runtime_error(hl + "/mask: node " + std::to_string(p.op) +
                               " holds " + std::to_string(t.d.size()) +
                               " values and a " +
                               std::to_string(g_.pose_input_size) + "x" +
                               std::to_string(g_.pose_input_size) + " mask is " +
                               std::to_string(want));
    o.mask.assign(t.d.begin(), t.d.end());
  }

  // world: declared [1, 117] over 39 rows of 3, in metres.
  {
    const PoseOut &p = g_.pose_out("world", hl);
    const Tensor &t = head_node(*this, p, n_nodes, hl);
    if (p.transposed)
      throw std::runtime_error(
          hl + "/world: the container records a transpose on the way out, and the "
          "world output is 39 rows of a metric x/y/z -- transposed it is 117 "
          "numbers that are individually plausible and jointly wrong.");
    const size_t want = static_cast<size_t>(g_.num_landmarks * g_.world_cols);
    if (t.d.size() != want)
      throw std::runtime_error(hl + "/world: node " + std::to_string(p.op) +
                               " holds " + std::to_string(t.d.size()) +
                               " values and " + std::to_string(g_.num_landmarks) +
                               "x" + std::to_string(g_.world_cols) + " = " +
                               std::to_string(want) + " were expected");
    o.world.assign(t.d.begin(), t.d.end());
  }

  // heatmap: declared [1, 64, 64, 39] and the convolution writes [1, 39, 64, 64].
  // This one IS transposed, so decl[y][x][c] = node[c][y][x] and the copy has to
  // be strided rather than linear. It is read so that a gate can compare it, and
  // NOT USED by anything: MediaPipe emits it to refine the landmarks and the
  // reference never calls that function, which is why the container lists it in
  // pose_unused.
  {
    const PoseOut &p = g_.pose_out("heatmap", hl);
    const Tensor &t = head_node(*this, p, n_nodes, hl);
    // The ONE output whose declared axes are not the convolution's, because it is
    // transposed on the way out: declared [1,64,64,39] over a convolution that
    // writes [1,39,64,64]. So the declared shape has to be read, and it has to be
    // four-dimensional: this is the one output whose meaning is in WHICH axis is
    // which, and a rank-2 spelling of it would leave the order undefined.
    if (p.decl.size() != 4)
      throw std::runtime_error(
          hl + "/heatmap: declared " + std::to_string(p.decl.size()) +
          " axes and it is the one output whose ORDER is the content -- it is "
          "transposed [0,2,3,1] on the way out of a [1,39,64,64] convolution, so "
          "a rank-2 or rank-3 spelling would leave [h,w,c] and [c,h,w] "
          "indistinguishable.");
    const int64_t cc = p.decl[3], hh = p.decl[1], ww = p.decl[2];
    if (t.c != cc || t.h != hh || t.w != ww)
      throw std::runtime_error(
          hl + "/heatmap: declared [" + std::to_string(cc) + ", " +
          std::to_string(hh) + ", " + std::to_string(ww) +
          "] and the node holds " + std::to_string(t.c) + "x" + std::to_string(t.h) +
          "x" + std::to_string(t.w) +
          ". The heatmap is the one output where the declared axes and the "
          "convolution's are different -- it is transposed on the way out -- so "
          "they are compared here rather than assumed equal.");
    o.heatmap.resize(t.d.size());
    for (int64_t y = 0; y < hh; ++y)
      for (int64_t x = 0; x < ww; ++x)
        for (int64_t c = 0; c < cc; ++c)
          o.heatmap[static_cast<size_t>((y * ww + x) * cc + c)] =
              t.chw(c)[static_cast<size_t>(y * ww + x)];
  }

  return o;
}

}  // namespace npue::mppose
