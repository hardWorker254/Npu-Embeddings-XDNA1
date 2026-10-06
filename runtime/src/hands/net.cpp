//===- net.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=7 graph walk and its two heads.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "hands/net.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/host_kernels.hpp"   // app::now_s

namespace npue::hands {

using npue::hostconv::gemm_nt;
using npue::hostconv::im2col;
using npue::hostconv::transpose_mn_to_nchw;

namespace {

// The graph's declared output extent, recomputed from the op and REFUSED if it
// disagrees. Every op in the list carries its own `out`, which the packer wrote
// by running the ONNX; recomputing it here and comparing means a container whose
// padding or stride does not divide its input is refused at load rather than
// producing a tensor of the declared shape filled with whatever the kernel
// happened to write.
void check_extent(const Layer &l, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  const std::string &label) {
  const int64_t eh = H + l.pad[0] + l.pad[2] - kh;
  const int64_t ew = W + l.pad[1] + l.pad[3] - kw;
  if (eh < 0 || ew < 0 || eh % l.stride_h || ew % l.stride_w)
    throw std::runtime_error(
        label + ": op '" + op_to_string(l.op) + "' has pad [" +
        std::to_string(l.pad[0]) + "," + std::to_string(l.pad[1]) + "," +
        std::to_string(l.pad[2]) + "," + std::to_string(l.pad[3]) +
        "] on a " + std::to_string(H) + "x" + std::to_string(W) +
        " input with a " + std::to_string(kh) + "x" + std::to_string(kw) +
        " kernel at stride " + std::to_string(l.stride_h) + "," +
        std::to_string(l.stride_w) + ", which does not give a whole-number "
        "output. Every convolution in these graphs pads asymmetrically on "
        "purpose -- MediaPipe's SAME padding evaluated at a stride -- so this "
        "is the check that keeps a symmetric transcription of it out.");
  const int64_t oh = eh / l.stride_h + 1, ow = ew / l.stride_w + 1;
  if (oh != l.out_h || ow != l.out_w)
    throw std::runtime_error(
        label + ": op '" + op_to_string(l.op) + "' computes " +
        std::to_string(oh) + "x" + std::to_string(ow) +
        " and the container says its output is " + std::to_string(l.out_h) +
        "x" + std::to_string(l.out_w) +
        ". The two are written by the packer and here from the same op, so a "
        "disagreement is a container whose op list was edited.");
}

npue::hostconv::Act host_act(Act a, const float *slope) {
  // hostconv::Act and hands::Act are two enums on purpose -- one is what the
  // graph says, the other is what the epilogue can compute -- and this is the
  // one place they are mapped. A default is not provided for kPreLU without a
  // slope: it would be a wrong network rather than a crash.
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

}  // namespace

Network::Network(const Geometry &g, const Placement &place, app::Pool &pool,
                 conv::Convs *array)
    : g_(g), place_(place), pool_(pool), array_(array) {
  // THE WEIGHT TRANSPOSE HAPPENS HERE, ONCE, AND NOT PER CALL.
  //
  // gemm_nt's B is [K, N] -- one row per reduction step, N output columns
  // contiguous -- and the container stores the filter as [Cout, Cin, kh, kw],
  // which is [N, K]. Passing it unchanged is a shape-compatible mistake (both
  // are K*N floats) that yields a convolution with a TRANSPOSED filter: the
  // output has the right shape, every channel is a plausible mixture of the
  // right ones, and the failure is this architecture's accuracy rather than a
  // crash. It cost one debugging session on the first run of this file.
  //
  // It is a STARTUP cost because the filter is the same on every frame. Paying
  // it inside conv() would grow with the number of convolutions rather than
  // with the number of images, which is the wrong way round for a cost that a
  // video pipeline runs once and then never again.
  //
  // DEPTHWISE FILTERS ARE NOT TRANSPOSED, and that is not an oversight. Their
  // reduction is over kh*kw WITHIN one channel, so dwconv() indexes
  // w[(c*kh + i)*kw + j] straight off the container's own layout and there is
  // no [K, N] view of one filter to build: a depthwise filter of one input
  // channel has K = kh*kw and N = 1 per channel, and 100 such transposes would
  // be 100 copies of a 3x3. The entry is left nullptr, and conv() refuses a
  // dense convolution without one rather than reading the untransposed filter.
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
  build(g.palm_convs, wmat_palm_, wt_palm_, "palm");
  build(g.lm_convs, wmat_lm_, wt_lm_, "lm");
}

// -- dense convolution -------------------------------------------------------
//
// im2col + the shared blocked GEMM + the shared transpose-with-epilogue. The
// three kernels are in include/common/conv_host.hpp and are the SAME ones
// arch=6's host path runs, which is why the two architectures can be compared
// on "what does the host convolution cost" at all.
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
        " has no [K,N] weight table. The transpose is built in the "
        "constructor for every DENSE filter, so a missing one means this "
        "convolution is depthwise and was routed here by name -- which would "
        "compute a different network rather than fail.");

  Tensor out(l.out_c, l.out_h, l.out_w);
  const double t0 = app::now_s();
  im2col(in, w.kh, w.kw, l.pad[0], l.pad[1], l.stride_h, l.stride_w, a_, pool_,
         l.pad[2], l.pad[3]);
  const int64_t M = l.out_h * l.out_w, K = w.cin * w.kh * w.kw, N = w.cout;
  cost_.t_im2col += app::now_s() - t0;

  if (array_) {
    if (!place_.slots || w.index < 0 ||
        static_cast<size_t>(w.index) >= place_.slots->size())
      throw std::runtime_error(
          label + ": conv " + std::to_string(w.index) +
          " is outside this network's slot table, which has " +
          std::to_string(place_.slots ? place_.slots->size() : 0) + " entries");
    const int64_t slot = (*place_.slots)[static_cast<size_t>(w.index)];
    if (slot < 0)
      throw std::runtime_error(
          label + ": conv " + std::to_string(w.index) +
          " has no array panel (slot -1). Only the DENSE convolutions are "
          "staged: a depthwise filter reduces within one channel, so it has no "
          "[M, N] GEMM to dispatch. This layer reached the dense path, so the "
          "graph and the weight table disagree about whether it is depthwise.");
    conv::conv_array_dispatch(*array_, slot, a_.data(), M, K, N, out.d.data(),
                              out.plane(), host_act(l.act, slope), slope, pool_,
                              apad_, cpad_, &cost_.t_array_repack,
                              &cost_.t_array_gemm, &cost_.t_array_transpose,
                              &cost_.dispatches);
    cost_.convs_array++;
    cost_.t_array += app::now_s() - t0;
    return out;
  }

  // `wkn` is the filter as [K, N], transposed once in the constructor from the
  // container's [N, K]. t_wmat stays at zero rather than carrying a cost that
  // is not paid on this path -- the number would be a startup cost counted
  // against a per-frame total.
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
//
// ONE filter per channel, so there is no [K,N] matrix and no single GEMM: the
// reduction is over kh*kw within one channel and the output channel index is
// the input channel index. That is exactly the shape the array's instruction
// stream cannot express -- it wants a K-reduction across N output columns -- and
// it is why 39 of this container's 100 convolutions stay on the host by name.
//
// Looping per channel rather than building one im2col for the whole tensor and
// doing a block-diagonal GEMM is deliberate: the block-diagonal form computes
// K*C times the useful arithmetic and spends the difference on multiplying by
// zeros, which for the 3x3 blocks is 9x the work to get the same answer.
//
// Parallel over CHANNELS, so each worker's run is one contiguous plane.
Tensor Network::dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                       const float *slope, const std::string &label) {
  check_extent(l, in.h, in.w, w.kh, w.kw, label);
  if (w.cout != in.c)
    throw std::runtime_error(
        label + ": dwconv " + std::to_string(w.index) + " has " +
        std::to_string(w.cout) + " filters and the tensor has " +
        std::to_string(in.c) + " channels. One filter per channel means they "
        "are the same number, and reading it as a dense convolution would "
        "compute a different network rather than fail.");
  if (l.out_c != in.c)
    throw std::runtime_error(label + ": dwconv " + std::to_string(w.index) +
                             " changes the channel count from " +
                             std::to_string(in.c) + " to " +
                             std::to_string(l.out_c) +
                             ". A depthwise convolution cannot: it would need "
                             "a projection, which is what the `conv` after it "
                             "is for.");
  Tensor out(l.out_c, l.out_h, l.out_w);
  const int64_t H = in.h, W = in.w, kh = w.kh, kw = w.kw;
  // Only the TOP and LEFT terms are read here: they are the offsets of
  // the gather. The bottom and right terms are checked by check_extent()
  // against the op's declared output, which is what makes them matter.
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
//
// The mobilenet block's `x + f(x)`, and where PReLU/ReLU6 lives when the export
// fused it onto the sum rather than onto the last convolution. Both of these
// networks do fuse onto the add, so this is not a rare shape.
Tensor Network::add(const Tensor &a, const Tensor &b, const Layer &l,
                    const float *slope, const std::string &label) {
  if (a.c != b.c || a.h != b.h || a.w != b.w)
    throw std::runtime_error(
        label + ": add joins a " + std::to_string(a.c) + "x" +
        std::to_string(a.h) + "x" + std::to_string(a.w) + " and a " +
        std::to_string(b.c) + "x" + std::to_string(b.h) + "x" +
        std::to_string(b.w) + ". Neither network has a broadcast add in its "
        "graph, so the two shapes are a transcription error rather than "
        "something to broadcast: broadcasting here would silently accept a "
        "block that resized instead of adding.");
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
        // value for a max over nothing, and it is what a following convolution
        // sees; a 0 default would be a different tensor and it would only ever
        // show up as a slightly wrong pixel.
        dst[static_cast<size_t>(y) * l.out_w + x] = best;
      }
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- pad_c: append `count` ZERO CHANNELS -------------------------------------
//
// The mobilenet "pad to 2x" step. It is NOT a pixel pad and NOT a zero pad: it
// concatenates a tensor of zero CHANNELS at the channel axis, which is what makes
// the following 1x1 convolution a widening projection. Reading it as a pixel pad
// would give the right total channel count and a wrong tensor, and the
// difference would be one network's accuracy rather than a shape error.
Tensor Network::pad_c(const Tensor &in, const Layer &l,
                      const std::string &label) {
  if (l.out_c != in.c + l.count)
    throw std::runtime_error(label + ": pad_c appends " +
                             std::to_string(l.count) + " channels to " +
                             std::to_string(in.c) + " and the graph says the "
                             "output is " + std::to_string(l.out_c) +
                             " wide. `count` is the number of ZERO CHANNELS, "
                             "appended at the channel axis.");
  if (l.out_h != in.h || l.out_w != in.w)
    throw std::runtime_error(label + ": pad_c changes the spatial extent");
  Tensor out(l.out_c, l.out_h, l.out_w);
  const size_t plane = out.plane();
  const double t0 = app::now_s();
  par_items(pool_, l.out_c, plane, [&](int64_t c) {
    float *const dst = out.chw(c);
    if (c < in.c)
      std::memcpy(dst, in.chw(c), plane * sizeof(float));
    else
      std::memset(dst, 0, plane * sizeof(float));
  });
  cost_.t_elementwise += app::now_s() - t0;
  cost_.elementwise_ops++;
  return out;
}

// -- ONNX Resize, linear, half_pixel, exclude_outside=0 ----------------------
//
// Two passes, and each weight belongs to the axis its pass moves along: `bot`
// differs from `top` in the ROW index so it is blended by wy, and `right`
// differs from `left` in the COLUMN index so it is blended by wx. Putting them
// the other way round is a TRANSPOSITION -- still bilinear, still the right
// shape, still within a factor of two of right, and still a detector that finds
// a hand. It is caught by comparing against a reference, not by looking.
//
// half_pixel, and the alternatives are all off by half a source pixel:
//   half_pixel        src = (i + 0.5) * scale - 0.5      <- this one
//   align_corners     src = i * scale
//   asymmetric         src = i * scale
//   pytorch_half      src = (i + 0.5) * scale          (falls_back off)
Tensor Network::resize(const Tensor &in, const Layer &l,
                       const std::string &label) {
  if (l.out_c != in.c)
    throw std::runtime_error(label + ": resize changes the channel count");
  const int64_t sh = in.h, sw = in.w, dh = l.out_h, dw = l.out_w;
  if (l.scale_h <= 0 || l.scale_w <= 0)
    throw std::runtime_error(label + ": resize has a non-positive scale");
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
  // The row scratch is PER WORKER, and that is a correctness matter rather than
  // a tidiness one. It used to be declared here, outside the loop below, which
  // is correct for arch=6's upsample because that stride-parallelises over
  // CHANNELS with a work unit of a whole plane and every worker therefore owns
  // its own slice. resize also strides over channels -- but each worker needs a
  // scratch row for the two separable passes, and a shared one is 16 threads
  // writing one buffer.
  //
  // It went unnoticed because par_items only parallelises above 65536 elements,
  // and the palm detector's FIRST resize is 256 channels of 12x12 = 36864 -- just
  // under the bar, so it ran serially and was right. The second is 256 channels
  // of 24x24 = 147456, went parallel, and was wrong by 16.9 absolute. A race
  // that only shows up above a threshold the same file chose is exactly the kind
  // that survives a review.
  par_items(pool_, in.c, out.plane(), [&](int64_t c) {
    std::vector<float> row(static_cast<size_t>(dw));
    const float *const src = in.chw(c);
    float *const dst = out.chw(c);
    for (int64_t i = 0; i < dh; ++i) {
      const float *const a = src + static_cast<size_t>(y0[i]) * sw;
      const float *const b = src + static_cast<size_t>(y1[i]) * sw;
      const float w = wy[i];
      for (int64_t j = 0; j < dw; ++j)
        row[static_cast<size_t>(j)] =
            a[x0[j]] * (1.f - w) + b[x0[j]] * w;
      for (int64_t j = 0; j < dw; ++j) {
        const float l0 = row[static_cast<size_t>(j)];
        const float l1 = a[x1[j]] * (1.f - w) + b[x1[j]] * w;
        dst[static_cast<size_t>(i) * dw + j] =
            l0 * (1.f - wx[j]) + l1 * wx[j];
      }
    }
  });
  cost_.t_elementwise += app::now_s() - t0;
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
      case Op::PadC:
        body_[i + 1] = pad_c(body_[static_cast<size_t>(l.inputs[0]) + 1], l, nl);
        break;
      case Op::Resize:
        body_[i + 1] = resize(body_[static_cast<size_t>(l.inputs[0]) + 1], l, nl);
        break;
    }
    // AFTER the switch, so the hook sees the activation too -- it fires on the
    // node's OUTPUT, and an activation that is fused into the convolution is
    // part of what the node produced. A hook placed before it would report the
    // pre-activation tensor for every node that has one and the gate would then
    // measure the epilogue twice and blame the wrong thing.
    if (on_node) on_node(i, body_[i + 1]);
  }
  return body_;
}

// -- the palm head -----------------------------------------------------------
//
// Four NCHW feature maps -> the graph's [1, 2016, 18] and [1, 2016, 1].
//
// The reshape is the delicate part and it is done in the graph's own axis order:
// the checkpoint transposes NCHW to NHWC and reshapes [H,W,C] to [H*W*A, terms],
// so row (h*W + w)*A + a, term t, is channel a*terms + t of cell (h, w). The
// flat row index therefore advances w -- the X coordinate -- fastest, which is
// the same order the anchor table is generated in, and a mismatch between the
// two is a detector whose best cell is a mirror image away.
PalmOut Network::palm_head(const std::vector<Tensor> &nodes,
                           const std::string &label) const {
  PalmOut o;
  o.n = 0;
  std::vector<double> boxes, scores;
  for (size_t li = 0; li < g_.palm_levels.size(); ++li) {
    const PalmLevel &L = g_.palm_levels[li];
    const std::string el = label + " level " + std::to_string(li);
    const size_t n_nodes = nodes.empty() ? 0 : nodes.size() - 1;
    if (L.box < 0 || static_cast<size_t>(L.box) >= n_nodes ||
        L.score < 0 || static_cast<size_t>(L.score) >= n_nodes)
      throw std::runtime_error(el + ": names graph node " +
                               std::to_string(L.box) + " / " +
                               std::to_string(L.score) +
                               ", which is not one of the network's " +
                               std::to_string(n_nodes) + " nodes");
    const Tensor &b = node(static_cast<size_t>(L.box));
    const Tensor &s = node(static_cast<size_t>(L.score));
    if (b.c != L.anchors_per_cell * g_.palm_row_terms || b.h != L.h ||
        b.w != L.w)
      throw std::runtime_error(
          el + ": the box feature is " + std::to_string(b.c) + "x" +
          std::to_string(b.h) + "x" + std::to_string(b.w) + " and the head says " +
          std::to_string(L.anchors_per_cell * g_.palm_row_terms) + "x" +
          std::to_string(L.h) + "x" + std::to_string(L.w) +
          ". Both are written by the packer from the ONNX's own output names, "
          "so a disagreement is one of them a transcription.");
    if (s.c != L.anchors_per_cell || s.h != L.h || s.w != L.w)
      throw std::runtime_error(el + ": the score feature is " +
                               std::to_string(s.c) + "x" + std::to_string(s.h) +
                               "x" + std::to_string(s.w) + " against " +
                               std::to_string(L.anchors_per_cell) + "x" +
                               std::to_string(L.h) + "x" + std::to_string(L.w));
    boxes.reserve(boxes.size() + static_cast<size_t>(L.rows) * g_.palm_row_terms);
    scores.reserve(scores.size() + static_cast<size_t>(L.rows));
    for (int64_t y = 0; y < L.h; ++y) {
      for (int64_t x = 0; x < L.w; ++x) {
        const size_t cell = static_cast<size_t>(y * L.w + x);
        for (int64_t a = 0; a < L.anchors_per_cell; ++a) {
          for (int64_t t = 0; t < g_.palm_row_terms; ++t)
            boxes.push_back(b.chw(a * g_.palm_row_terms + t)[cell]);
          scores.push_back(s.chw(a)[cell]);
        }
      }
    }
  }
  o.boxes = std::move(boxes);
  o.scores = std::move(scores);
  o.n = static_cast<int64_t>(o.scores.size());
  if (o.n != g_.palm_num_anchors)
    throw std::runtime_error(label + ": the levels produced " +
                             std::to_string(o.n) + " anchors and the container "
                             "declares " + std::to_string(g_.palm_num_anchors) +
                             ". The decode reads a table of the declared "
                             "width, so this is a refusal rather than a read "
                             "past the end.");
  return o;
}

// -- the landmark head -------------------------------------------------------
//
// One global average pool, then four Gemms. The pool is a MEAN over the 7x7 map
// and not a max and not a stride-7 conv: MediaPipe's hand-landmark head is a
// squeeze-th Gemm, and the shapes are kept with their batch axis so a squeeze
// here would compare equal in value and unequal in shape against a reference.
//
// The projections are stored [in, out] and ONNX's Gemm with transB=0 is A @ W,
// so this is W.T @ pooled. Reading it the other way round is a shape error in
// the config and a silent TRANSPOSE in a runtime -- and 63 against 1 is far
// enough apart to be noticed, which is the only reason that makes it a crash
// rather than a wrong hand.
LmOut Network::lm_head(const std::vector<Tensor> &nodes,
                       const std::string &label) {
  const std::string hl = label + " landmark head";
  const size_t n_nodes = nodes.empty() ? 0 : nodes.size() - 1;
  if (g_.lm_head.pool < 0 || static_cast<size_t>(g_.lm_head.pool) >= n_nodes)
    throw std::runtime_error(hl + ": names graph node " +
                             std::to_string(g_.lm_head.pool) +
                             ", which is not one of the network's " +
                             std::to_string(n_nodes) + " nodes");
  const Tensor &x = node(static_cast<size_t>(g_.lm_head.pool));
  if (x.c != g_.lm_head.features || x.h != g_.lm_head.height ||
      x.w != g_.lm_head.width)
    throw std::runtime_error(
        hl + ": the pooled node is " + std::to_string(x.c) + "x" +
        std::to_string(x.h) + "x" + std::to_string(x.w) + " and the head says " +
        std::to_string(g_.lm_head.features) + "x" +
        std::to_string(g_.lm_head.height) + "x" + std::to_string(g_.lm_head.width) +
        ". The width is the projection's input size, so a disagreement here is a "
        "projection that cannot be applied at all.");
  const double t0 = app::now_s();
  std::vector<float> pooled(static_cast<size_t>(x.c), 0.f);
  const size_t plane = x.plane();
  for (int64_t c = 0; c < x.c; ++c) {
    const float *const p = x.chw(c);
    float acc = 0.f;
    for (size_t i = 0; i < plane; ++i) acc += p[i];
    pooled[static_cast<size_t>(c)] = acc / static_cast<float>(plane);
  }
  cost_.t_pool += app::now_s() - t0;

  std::vector<std::vector<double>> proj(g_.lm_head.projs.size());
  for (size_t p = 0; p < g_.lm_head.projs.size(); ++p) {
    const LmProj &P = g_.lm_head.projs[p];
    proj[p].assign(static_cast<size_t>(P.n), 0.0);
    for (int64_t j = 0; j < P.n; ++j) {
      double acc = P.bp ? P.bp[j] : 0.0;
      for (int64_t c = 0; c < x.c; ++c)
        acc += static_cast<double>(pooled[static_cast<size_t>(c)]) *
               static_cast<double>(P.wp[static_cast<size_t>(c) * P.n + j]);
      proj[p][static_cast<size_t>(j)] = acc;
    }
  }
  LmOut o;
  o.outs.resize(g_.lm_head.outputs.size());
  for (size_t i = 0; i < g_.lm_head.outputs.size(); ++i) {
    const LmOutSpec &decl = g_.lm_head.outputs[i];
    o.outs[i] = proj[static_cast<size_t>(decl.proj)];
    if (!decl.sigmoid) continue;
    for (double &v : o.outs[i]) v = 1.0 / (1.0 + std::exp(-v));
  }
  return o;
}

}  // namespace npue::hands