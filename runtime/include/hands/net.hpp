//===- net.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=7 graph walk: one network, one walk, two heads.
//
// THE VOCABULARY IS SIX OPS AND NOT ARCH=6's SEVEN
// -------------------------------------------------
// arch=6 walks conv, concat, slice, add, maxpool, upsample, detect. arch=7 walks
// conv, dwconv, add, maxpool, pad_c, resize. None of arch=6's extra ops appears
// here and three of these do not appear there:
//
//   dwconv   ONE filter per channel. This is the op the array backend cannot
//            express -- there is no [K,N] matrix to hand a panel -- and 39 of
//            this container's 100 convolutions are it. It is kept as its own
//            op and refused FROM THE ARRAY BY NAME rather than quietly turned
//            into a dense convolution, because a gate that cannot tell the two
//            apart has stopped measuring anything.
//   pad_c    the mobilenet "pad to 2x" step: append N ZERO CHANNELS. It is a
//            channel-wise concat with a constant, and reading it as a PIXEL
//            pad would give the right channel count and a wrong tensor.
//   resize   ONNX Resize, linear, half_pixel, exclude_outside=0 -- the neck's
//            upsampling. Its coordinate transform is half-pixel, and the other
//            three are each off by half a source pixel, which produces a
//            detector that finds a hand in the right place by a fraction of a
//            pixel out at every level.
//
// THE ACTIVATION IS THE MODEL'S, AND IT IS FUSED
// ----------------------------------------------
// PReLU in the palm detector (26 of them), ReLU6 in the landmark network (32),
// and both networks fuse theirs into the convolution's epilogue or into the
// residual `add`. Fused because a separate pass over the finished NCHW tensor
// is a read and a write of the whole tensor plus a pool barrier, for a function
// of each element and nothing else -- and because where it is applied is a
// decision that must be made ONCE. The epilogue lives in
// include/common/conv_host.hpp with the rest of the host convolution.
//
// DWCONV HAS NO EPILOGUE TO FUSE INTO AND GETS ITS OWN
// -----------------------------------------------------
// There is no [M,N] GEMM to write the output through, so the depthwise kernel
// writes NCHW directly and applies the activation in the inner loop. That is a
// second implementation of the same three activations, and it is deliberate
// rather than a duplication: the dense epilogue exists only because a GEMM's
// natural output is [M,N] and the tensor is NCHW, and the depthwise kernel never
// has that intermediate to transpose. Both call the same `act1()` below, so the
// ARITHMETIC has one definition.
//
// TWO HEADS, AND NEITHER IS ARCH=6's
// ---------------------------------
// palm: four NCHW feature maps -> [2016, 18] of boxes and keypoints and
//      [2016, 1] of scores. The reshape is the delicate part and it is done in
//      the graph's own axis order -- see Network::palm_head.
// lm:   one global average pool -> four Gemms -> [63], [1], [1], [63].
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>
#include <functional>
#include <string>
#include <vector>

#include "common/conv_host.hpp"
#include "hands/geometry.hpp"
#include "runtime/pool.hpp"

namespace npue::hands {

using Tensor = npue::hostconv::Nchw;
using npue::hostconv::par_items;
// NOTE: `Act` is NOT imported from hostconv. There are two enums with that name
// in this namespace's reach -- hands::Act (what the graph records) and
// hostconv::Act (what the epilogue can compute) -- and a using-declaration here
// would silently merge them into one, which is the mistake the mapping function
// in net.cpp exists to make impossible.

// The three activations, as functions of one element -- except PReLU, whose
// slope is per channel and which therefore takes it. One definition, called
// from both the dense epilogue and the depthwise kernel. arch=6's SiLU is
// NOT here: it is not one of these networks' activations, and a switch that
// carried it would be an architecture borrowing the other's vocabulary.
inline float act1(Act a, float v, float slope) {
  switch (a) {
    case Act::None: return v;
    case Act::Relu: return v > 0.f ? v : 0.f;
    case Act::Relu6: return v < 0.f ? 0.f : (v > 6.f ? 6.f : v);
    case Act::PreLU: return v >= 0.f ? v : slope * v;
  }
  return v;
}

// Which ops the array is asked to run. An EMPTY placement is the default and
// means everything is on the host; that is also the only honest placement today,
// because --npu-ops conv needs a design set whose streams this container does
// not carry (see Placement).
struct Placement {
  bool conv_on_array = false;
  std::string design_dir;
};

// What one run cost. Same fields as arch=6's Cost, and for the same reason: the
// interesting number on this architecture is the SPLIT between the dense
// convolutions a panel could take and the depthwise ones it cannot.
struct Cost {
  int64_t convs_host = 0;
  int64_t dwconvs_host = 0;
  int64_t convs_array = 0;
  int64_t dispatches = 0;
  int64_t elementwise_ops = 0;
  double t_host = 0.0;
  double t_array = 0.0;
  double t_dwconv = 0.0;      // the depthwise convolutions, which the array
                              // cannot take at all
  double t_wmat = 0.0;        // [N, K] -> [K, N] weight transpose
  double t_im2col = 0.0;
  double t_gemm = 0.0;
  double t_transpose = 0.0;
  double t_elementwise = 0.0;   // add, pad_c, resize, maxpool
  double t_pool = 0.0;          // the landmark network's global average
  void reset() { *this = Cost{}; }
};

// What the palm detector's head produces, in its own shape: `boxes` is
// [n, row_terms] and `scores` is [n], both row-major, row i being anchor i.
struct PalmOut {
  std::vector<double> boxes;   // n * row_terms: cx, cy, w, h, then 7 x 2
  std::vector<double> scores;  // n
  int64_t n = 0;
};

// What the landmark network produces, in the order the container lists.
struct LmOut {
  std::vector<std::vector<double>> outs;   // [63], [1], [1], [63]
};

// Executes one packed graph. Owns nothing the caller has to keep alive except
// the container the geometry's ConvW pointers point into.
class Network {
public:
  Network(const Geometry &g, const Placement &place, app::Pool &pool);

  // One image, NCHW [3, S, S] in. Returns a vector of every node's output.
  //
  // Operand -1 in an op's `inputs` is the image, so the returned vector is one
  // longer than the graph and node i is at index i + 1 -- index 0 is the image.
  // Shifting by one rather than keeping a side map because an op's operand list
  // is read by array index at six call sites and a special case at each of them
  // is six chances to forget it.
  //
  // EVERY NODE IS KEPT, not the live ones, because the heads read nodes in the
  // MIDDLE of the graph -- the palm head's two levels are nodes 74/75 and 85/86
  // of 87. A walk that freed them would have to be told which to keep, and
  // being told is being wrong: a head that read a freed buffer would get a
  // plausible 2016 rows.
  //
  // The returned reference is to storage owned by this Network and is valid
  // until the next call. Two networks cannot be run alternately through one
  // Network; the Session owns one each.
  const std::vector<Tensor> &run_body(const Tensor &input,
                                      const std::vector<Layer> &graph,
                                      const std::vector<ConvW> &convs,
                                      const std::vector<const float *> &wkn,
                                      const std::vector<Slope> &slopes,
                                      const std::vector<std::string> &names,
                                      const std::string &label);

  // The two graphs' [K,N] weight tables, one per network. They are separate
  // because the graphs are separate and run_body is handed one table at a time;
  // `wkn` is parallel to the `convs` it belongs to, and nullptr at a depthwise
  // filter -- see the constructor for why those are not transposed.
  const std::vector<const float *> &wt_palm() const { return wt_palm_; }
  const std::vector<const float *> &wt_lm() const { return wt_lm_; }

  // Node i of the graph, whatever the operand bookkeeping in run_body did with
  // the indices. The heads go through THIS and not through the returned
  // vector, because the container's `palm_head` and `lm_head` name GRAPH NODES
  // while the returned vector is offset by one to hold the image -- reading
  // them out of the vector directly is off by one on every node of both
  // networks, and it surfaces as a shape mismatch about a tenth of the way in
  // rather than as a crash.
  const Tensor &node(size_t i) const { return body_[i + 1]; }

  // -- the two heads
  //
  // palm_head is the reshape, and it is the one place in this file where getting
  // an index order wrong is invisible in every shape: the checkpoint transposes
  // NCHW to NHWC and reshapes [H,W,C] to [H*W*A, terms], so row (h*W + w)*A + a,
  // term t, is channel a*terms + t of cell (h, w). The flat row index therefore
  // advances w -- X -- fastest. A C-major flatten instead would permute every
  // row and leave 2016 well-shaped rows and a confident detector behind.
  PalmOut palm_head(const std::vector<Tensor> &nodes,
                    const std::string &label) const;
  // Not const, unlike palm_head: the global average pool is timed, and the
  // landmark network's pool is one of the few spans in the whole container that
  // is neither a convolution nor a memory pass.
  LmOut lm_head(const std::vector<Tensor> &nodes, const std::string &label);

  // Called once per graph node, in graph order, with the node's index and its
  // output. Nothing in the runtime sets it; it exists so the gate can say WHICH
  // node this walk and an independent reading of the ONNX graph first disagree
  // at -- end to end, a wrong first convolution and a wrong fiftieth are the
  // same sentence. A hook that fires on every node is also a per-node cost, so
  // it is std::function and empty by default rather than a null check at six
  // call sites inside the timed region.
  std::function<void(size_t, const Tensor &)> on_node;

  const Cost &cost() const { return cost_; }
  // Zero the counters for the next frame. Not `cost().reset()`: cost() hands
  // out a const reference on purpose, so a caller reaching through it to clear
  // one is writing through a const. The Session resets before every frame
  // because a run that APPENDS to the previous frame's counts reports a total
  // that grows with the number of frames seen, which looks like a leak and is
  // an accounting bug.
  void reset_cost() { cost_.reset(); }

private:
  Tensor conv(const Tensor &in, const Layer &l, const ConvW &w,
               const float *wkn, const float *slope, const std::string &label);
  Tensor dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                const float *slope, const std::string &label);
  Tensor add(const Tensor &a, const Tensor &b, const Layer &l,
             const float *slope, const std::string &label);
  Tensor maxpool(const Tensor &in, const Layer &l, const std::string &label);
  Tensor pad_c(const Tensor &in, const Layer &l, const std::string &label);
  Tensor resize(const Tensor &in, const Layer &l, const std::string &label);

  const Geometry &g_;
  const Placement place_;
  app::Pool &pool_;
  Cost cost_;
  std::vector<float> a_;       // the im2col buffer, reused across convolutions
  std::vector<float> c_;       // the [M,N] GEMM output, reused
  std::vector<Tensor> body_;   // every node's output; index 0 is the image
  // The [K,N] weight tables. Storage first, then the pointers into it, so that
  // the pointers stay valid when the outer vectors are moved or resized.
  std::vector<std::vector<float>> wmat_palm_, wmat_lm_;
  std::vector<const float *> wt_palm_, wt_lm_;
};

}  // namespace npue::hands