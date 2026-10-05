//===- net.hpp ------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=8 graph walk: two networks, two walks, two heads.
//
// THE VOCABULARY IS SIX OPS, ONE OF THEM NEW, ONE OF THEM GONE
// --------------------------------------------------------------
// arch=6 walks conv, concat, slice, add, maxpool, upsample, detect. arch=7 walks
// conv, dwconv, add, maxpool, pad_c, resize. arch=8 walks conv, dwconv, add,
// maxpool, resize, d2s. Against arch=7 exactly one op arrives and one leaves:
//
//   d2s     DepthToSpace. It is how the detector builds its three pyramid
//           levels -- a single 7x7x1152 map becomes 28x28, 14x14 and 7x7 at
//           three resolutions -- so there is no rearrangement of conv, dwconv
//           and add that expresses it. Three of them, one per level.
//   pad_c   gone. arch=7's channel-widening step, which neither of these two
//           graphs contains.
//
// WHY THE D2S ORDER IS CARRIED IN THE OP AND NOT DECIDED HERE
// ------------------------------------------------------------
// ONNX names DepthToSpace's two orders DCR (depth, column, row) and CRD, and they
// differ only in how the input CHANNEL axis is split across the block. The
// default when the attribute is absent is DCR, and this model's nodes carry
// `blocksize` and nothing else -- so the graph states DCR by saying nothing.
//
// Taking the other order is not a shape error and not a range error: every
// channel lands on a different pixel of a feature map that has the right shape
// and a plausible distribution, and the next convolution consumes it happily.
// It shows up as a detector that finds people in the wrong places, which is the
// failure mode this op's comment exists to prevent. So `mode` is read into the
// op, both orders are implemented, and the semantic check below is the same one
// the packer makes: out[co, h*b + d1, w*b + d2] = in[ci, h, w].
//
// THE ACTIVATION IS THE MODEL'S, AND IT IS FUSED
// ----------------------------------------------
// Both networks fuse ReLU6 into the convolution's epilogue or into the residual
// `add`, and this checkpoint has no PReLU at all. Fused because a separate pass
// over the finished NCHW tensor is a read and a write of the whole tensor plus a
// pool barrier, for a function of each element and nothing else -- and because
// where it is applied must be decided ONCE. The epilogue lives in
// include/common/conv_host.hpp with the rest of the host convolution.
//
// DWCONV HAS NO EPILOGUE TO FUSE INTO AND GETS ITS OWN
// -----------------------------------------------------
// There is no [M,N] GEMM to write the output through, so the depthwise kernel
// writes NCHW directly and applies the activation in the inner loop. That is a
// second implementation of the same activations, deliberately: the dense epilogue
// exists only because a GEMM's natural output is [M,N] and the tensor is NCHW.
// Both call the same `act1()` below, so the ARITHMETIC has one definition.
// 62 of this container's 161 convolutions are depthwise -- 28 in the detector and
// 34 in the landmark net -- and it is 20.8 % of the landmark net's MACs against
// 8.4 % of the detector's, which is the split the array backend would have to
// take or refuse.
//
// TWO HEADS, AND NEITHER IS ARCH=6's OR ARCH=7's
// ---------------------------------------------
// det:  three levels of two 1x1 convolution outputs -> [2254, 12] of boxes and
//      keypoints and [2254, 1] of scores, concatenated in the Concat's own order.
// pose: five single convolutions, of which one is transposed on the way out and
//      one has a sigmoid in front of it -- and both facts come out of the
//      container rather than out of the output's NAME.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/conv_host.hpp"
#include "conv/array.hpp"
#include "mppose/geometry.hpp"
#include "runtime/pool.hpp"

namespace npue::mppose {

using Tensor = npue::hostconv::Nchw;
using npue::hostconv::par_items;
// NOTE: `Act` is NOT imported from hostconv. Two enums carry that name in this
// namespace's reach -- mppose::Act (what the graph records) and hostconv::Act
// (what the epilogue can compute) -- and a using-declaration here would silently
// merge them into one, which is the mistake the mapping function in net.cpp
// exists to make impossible.

// The activations, as functions of one element -- except PReLU, whose slope is
// per channel and which therefore takes it. One definition, called from both the
// dense epilogue and the depthwise kernel. arch=6's SiLU is NOT here: it is not
// one of these networks' activations, and a switch that carried it would be an
// architecture borrowing the other's vocabulary.
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
// means everything is on the host, which is also what a run gets when the
// container was packed without --npu.
struct Placement {
  bool conv_on_array = false;
  std::string design_dir;
  // WHERE THIS NETWORK'S CONVOLUTIONS LAND in the array backend's panel table.
  //
  // This container packs TWO graphs with SEPARATE convolution index spaces, and
  // the table is ONE, so every dense convolution gets a slot number assigned
  // across both graphs in one run. The mapping is NOT `slot = base + conv`,
  // because slot numbers are handed out over the DENSE convolutions only and
  // depthwise ones are interleaved throughout both graphs -- the detector's conv
  // 0, 2, 3, 6, 8 are dense and 1, 4, 5, 7 are not, so conv 8 is not slot 8.
  //
  // That arithmetic was WRONG here first, as `slot_base + conv`, and the
  // backend's own check caught it on the first run with:
  //
  //     the design's padded shape 64x256 is smaller than the convolution's own
  //     144x24, so the panel would be truncated
  //
  // which is the check doing the only thing it can: a slot that is another
  // convolution's is a panel of the RIGHT SHAPE in the common case, so most of
  // the run succeeds and one layer disagrees. Had the widths happened to line up
  // it would have produced confident wrong landmarks instead.
  //
  // So the table is passed rather than computed. The session builds it once, from
  // the same vector it hands the backend, and a -1 in it means "this
  // convolution has no panel", which is what a depthwise convolution gets.
  const std::vector<int64_t> *slots = nullptr;
  // For error messages: "det" or "pose".
  std::string group = "det";
};

// What one run cost. Same fields as arch=7's Cost, and for the same reason: the
// interesting number here is the SPLIT between the dense convolutions a panel
// could take and the depthwise ones it cannot.
struct Cost {
  int64_t convs_host = 0;
  int64_t dwconvs_host = 0;
  int64_t convs_array = 0;
  int64_t dispatches = 0;
  int64_t elementwise_ops = 0;
  double t_host = 0.0;
  double t_array = 0.0;
  double t_dwconv = 0.0;
  double t_d2s = 0.0;      // this architecture's own op
  double t_wmat = 0.0;
  double t_im2col = 0.0;
  double t_gemm = 0.0;
  double t_transpose = 0.0;
  // The array path's own three spans, kept apart for arch=6's reason: none of
  // them is the device's work, and a line printing only t_array would read as
  // though the host shuffle around the dispatch were free.
  double t_array_gemm = 0.0;
  double t_array_repack = 0.0;
  double t_array_transpose = 0.0;
  double t_elementwise = 0.0;   // add, maxpool, resize
  void reset() { *this = Cost{}; }
};

// What the detector's head produces, in its own shape: `boxes` is [n, 12] of
// xy1, xy2 and four keypoints x 2, and `scores` is [n], both row-major, row i
// being anchor i.
struct DetOut {
  std::vector<double> boxes;   // n * row_terms
  std::vector<double> scores;  // n
  int64_t n = 0;
};

// What the landmark network produces, in the five shapes the container declares.
//
// `mask` is row-major [256, 256] -- out[h, w] -- which is the declared
// [1,256,256,1] read with its single channel dropped. `heatmap` is [64, 64, 39]
// in the declared NHWC order and is carried for a dump only: MediaPipe emits it
// to refine the landmarks and the reference never calls it, so nothing in the
// front end reads it and it is in the container's `pose_unused` for that reason.
struct PoseOuts {
  std::vector<double> landmarks;   // 39 x 5: x, y, z, visibility, presence
  double conf = 0.0;               // already sigmoid'd by the graph
  std::vector<float> mask;         // 256 x 256
  std::vector<double> world;       // 39 x 3, metres
  std::vector<float> heatmap;      // 64 x 64 x 39, unused by the front end
};

// Executes one packed graph. Owns nothing the caller has to keep alive except the
// container the geometry's ConvW pointers point into.
class Network {
public:
  // `array` is borrowed and may be null, which is the default and the only thing
  // a container packed without --npu can use. Passing one while the container
  // carries no panels fails at the FIRST CONVOLUTION rather than at load, so the
  // session builds the backend only when it has already confirmed the panels are
  // there.
  Network(const Geometry &g, const Placement &place, app::Pool &pool,
          conv::Convs *array = nullptr);

  // One image in NCHW [3, S, S], returning every node's output.
  //
  // Operand -1 in an op's `inputs` is the image, so the returned vector is one
  // longer than the graph and node i is at index i + 1 -- index 0 is the image.
  // Shifting by one rather than keeping a side map because an op's operand list
  // is read by array index at seven call sites and a special case at each of them
  // is seven chances to forget it.
  //
  // EVERY NODE IS KEPT, not the live ones, because the heads read nodes in the
  // MIDDLE of the graph: the detector's three box convolutions are nodes 90, 74
  // and 58 of 93, and its three score convolutions are 92, 76 and 60. A walk that
  // freed them would have to be told which to keep, and being told is being
  // wrong: a head that read a freed buffer would get 2254 well-shaped rows.
  //
  // The returned reference is to storage owned by this Network and is valid until
  // the next call. The Session owns one Network per graph.
  const std::vector<Tensor> &run_body(const Tensor &input,
                                      const std::vector<Layer> &graph,
                                      const std::vector<ConvW> &convs,
                                      const std::vector<const float *> &wkn,
                                      const std::vector<Slope> &slopes,
                                      const std::vector<std::string> &names,
                                      const std::string &label);

  // The two graphs' [K,N] weight tables, one per network. `wkn` is parallel to
  // the `convs` it belongs to and nullptr at a depthwise filter -- see the
  // constructor for why those are not transposed.
  const std::vector<const float *> &wt_det() const { return wt_det_; }
  const std::vector<const float *> &wt_pose() const { return wt_pose_; }

  // Node i of the graph, whatever the operand bookkeeping in run_body did with
  // the indices. The heads go through THIS: the container's `det_head` and
  // `pose_head` name GRAPH NODES while the returned vector is offset by one to
  // hold the image, and reading them out of the vector directly is off by one on
  // every node -- which surfaces as a shape mismatch about a tenth of the way in
  // rather than as a crash.
  const Tensor &node(size_t i) const { return body_[i + 1]; }

  // -- the two heads
  //
  // det_head is the reshape, and it is the one place where getting an index order
  // wrong is invisible in every shape: the checkpoint transposes NCHW to NHWC and
  // reshapes [H,W,C] to [H*W*A, terms], so row (h*W + w)*A + a, term t, is
  // channel a*terms + t of cell (h, w). The flat row index therefore advances w
  // -- the X coordinate -- fastest, which is the order the stored anchor table is
  // in. A C-major flatten instead would permute every row and leave 2254
  // well-shaped rows and a confident detector behind.
  DetOut det_head(const std::vector<Tensor> &nodes,
                  const std::string &label) const;
  PoseOuts pose_head(const std::vector<Tensor> &nodes,
                     const std::string &label) const;

  // Called once per graph node, in graph order, with the node's index and its
  // output. Nothing in the runtime sets it; it exists so the gate can say WHICH
  // node this walk and an independent reading of the ONNX graph first disagree
  // at -- end to end, a wrong first convolution and a wrong fiftieth are the same
  // sentence.
  std::function<void(size_t, const Tensor &)> on_node;

  const Cost &cost() const { return cost_; }
  void reset_cost() { cost_.reset(); }

  // Whether the dense convolutions went to the array, and the array's own
  // numbers if they did. The status block asks THIS rather than reading the
  // command line, because a line that reported "npu" because --npu-ops conv was
  // on argv would also report it for a run where the design set was missing and
  // everything fell back -- which is the one thing a status block must not do.
  bool on_array() const { return array_ != nullptr; }
  int64_t array_dispatches() const {
    return array_ ? array_->dispatches() : 0;
  }
  double array_seconds() const {
    // The DEVICE's time, deliberately not including the host-side repack and
    // transpose: neither is the array's work, and adding them in would make a
    // number that is mostly host shuffling look like array cost.
    return array_ ? array_->t_array() : 0.0;
  }
  size_t array_staged_bytes() const {
    return array_ ? array_->staged_bytes() : 0;
  }

private:
  Tensor conv(const Tensor &in, const Layer &l, const ConvW &w,
              const float *wkn, const float *slope, const std::string &label);
  // The array half of the same convolution. Writes into `out` (already shaped)
  // and returns it, so the caller allocates the tensor once either way.
  Tensor conv_array(const Tensor &in, Tensor &out, const Layer &l,
                    const ConvW &w, int64_t M, int64_t K, int64_t N,
                    const float *slope, double t0, const std::string &label);
  Tensor dwconv(const Tensor &in, const Layer &l, const ConvW &w,
                const float *slope, const std::string &label);
  Tensor add(const Tensor &a, const Tensor &b, const Layer &l,
             const float *slope, const std::string &label);
  Tensor maxpool(const Tensor &in, const Layer &l, const std::string &label);
  Tensor resize(const Tensor &in, const Layer &l, const std::string &label);
  Tensor d2s(const Tensor &in, const Layer &l, const std::string &label);

  const Geometry &g_;
  const Placement place_;
  app::Pool &pool_;
  Cost cost_;
  // Borrowed, and null unless Placement::conv_on_array is set. The type names no
  // XRT class, which is what lets this header -- and every build without a
  // design set -- link with the device code absent.
  conv::Convs *array_ = nullptr;
  std::vector<float> a_;       // the im2col buffer, reused across convolutions
  std::vector<float> c_;       // the [M,N] GEMM output, reused
  // The array path's own buffers, and they are separate from a_/c_ rather than
  // reused: the repack reads a_ and the device writes apad_, and a chunk that
  // overwrote its own input would be a plausible wrong network. Sized on first
  // use, so a host-only run allocates neither.
  std::vector<float> apad_;
  std::vector<float> cpad_;
  std::vector<Tensor> body_;   // every node's output; index 0 is the image
  // The [K,N] weight tables. Storage first, then the pointers into it, so that
  // the pointers stay valid when the outer vectors are moved or resized.
  std::vector<std::vector<float>> wmat_det_, wmat_pose_;
  std::vector<const float *> wt_det_, wt_pose_;
};

}  // namespace npue::mppose
