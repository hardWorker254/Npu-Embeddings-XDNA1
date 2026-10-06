//===- array.hpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the array side of a CONVOLUTION, shared by every conv-only
// architecture.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//
#ifndef NPUE_CONV_ARRAY_HPP
#define NPUE_CONV_ARRAY_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/conv_host.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"

namespace npue::conv {

// What a network calls when a convolution is placed on the array.
//
// SEPARATE FROM THE NETWORK, and that separation is the reason this header holds
// no XRT type: a network holds a pointer to one of these and names none of
// npu::Device, npu::Design or NpuGemm, so a build with no design set -- which is
// every arch=7 and arch=8 run on a machine that has no array -- links and runs
// without the device code at all. The backend that implements it is in
// conv/array.cpp and is only constructed when a flag actually asks for it.
//
// WHY IT IS NOT ARCH=6's OWN CLASS ANY MORE. It was `npue::pose::NpuConvs`, and
// arch=8 needs the identical three methods against the identical design, so a
// second copy would have been the same 90 lines twice with the two copies free
// to drift on exactly the details that are dangerous -- the bias width, the
// padded-K repack and the "one panel per CONVOLUTION, never per stream name"
// rule, each of which has a wrong answer that is a plausible number rather than
// an error. One copy cannot drift.
class Convs {
public:
  virtual ~Convs() = default;

  // out[m, n] = a[m, k] @ b[k, n] + bias[n], over m rows of k, n columns.
  //
  // `conv` is a SLOT NUMBER from ConvSlot -- NOT the architecture's own
  // convolution index, and for a two-graph container definitely not, because both
  // graphs have a conv 0. The backend keyed its staged panel by exactly this, and
  // one lookup made once at load is what stops the two backends picking different
  // weights for the same layer.
  virtual int64_t gemm(int64_t conv, const float *a, int64_t m, int64_t k,
                       int64_t n, const float *bias, float *out) = 0;

  // The widths the DISPATCH must use, which are the DESIGN's and not the
  // convolution's. A design's K is rounded up to a multiple of tile_k and its N
  // to a multiple of tile_n * cols, and the compiled core runs exactly those
  // widths whatever the caller asks: NpuGemm::run packs `m * k` A columns as one
  // contiguous run, adds bias[j] for every j < n, and reads back n columns per C
  // row. Passing the convolution's real widths makes row r's padded tail read
  // row r+1's data -- a network that finds hundreds of people where there are
  // three, which is a wrong SHAPE and not a wrong number.
  //
  // So the A rows the caller hands over must already be `padded_k` wide with
  // columns [k, padded_k) zeroed, and the C rows it receives back are
  // `padded_n` wide.
  virtual int64_t padded_k(int64_t conv) const = 0;
  virtual int64_t padded_n(int64_t conv) const = 0;

  // The design's row count, which is the chunk size a caller must not exceed.
  virtual int64_t rows_per_dispatch() const = 0;

  // What the status line reports. The array's own time is `t_array` and it
  // deliberately does NOT include the host-side repack and transpose: neither is
  // the array's work, and a line that printed only device time under a flag that
  // moves the whole convolution would read as though the array were free.
  virtual int64_t dispatches() const = 0;
  virtual double t_array() const = 0;
  virtual size_t staged_bytes() const = 0;
  // The distinct stream names this backend staged, in graph order, for the
  // status line. Distinct only -- one convolution out of 99 sharing another's
  // panel would otherwise print the same name dozens of times.
  virtual std::vector<std::string> stream_ops() const = 0;
};

// One dense convolution, dispatched. SHARED BY arch=6, arch=7 and arch=8, and
// for the same reason the panel builder is shared: the three details that make
// this wrong do not announce it.
//
//   * A must be REPACKED at the design's padded K. The compiled core reads
//     `rows * k` columns as ONE contiguous run, so row r's padded tail would
//     otherwise be row r+1's activations.
//   * C comes back `padded_n` wide and the host narrows it to the real N.
//   * The repack is per CHUNK, not per convolution: rows*pk is 256 KiB here while
//     the whole convolution's A can be tens of megabytes, and the array never
//     needs more than one dispatch window.
//
// `slot` is a number from ConvSlot, NOT the architecture's own convolution index:
// arch=7 and arch=8 each pack TWO graphs whose index spaces both start at zero.
//
// `dst_ch` is the destination tensor's OWN pixel count, which is what
// transpose_mn_to_nchw needs to address the right plane -- and it is the channel
// plane count, so `dst + done` is correct only because a chunk never straddles a
// plane boundary, which it cannot when the tensor is [C, H, W] and the chunk runs
// along H*W.
void conv_array_dispatch(Convs &array, int64_t slot, const float *a_im2col,
                         int64_t M, int64_t K, int64_t N, float *dst,
                         int64_t dst_ch, hostconv::Act act, const float *slope,
                         app::Pool &pool, std::vector<float> &apad,
                         std::vector<float> &cpad, double *t_repack,
                         double *t_gemm, double *t_transpose,
                         int64_t *dispatches);

// One convolution the backend must be able to run, flattened.
//
// FLATTENED RATHER THAN A Layer&, because every architecture spells its own
// Layer and ConvW -- `npue::pose::Layer`, `npue::mppose::Layer` -- and this file
// deliberately depends on none of them. It needs six numbers and two strings out
// of each, and asking for exactly those is what lets one backend serve arch=6
// and arch=8 without either learning about the other.
struct ConvSlot {
  // THE SLOT NUMBER, EXPLICIT, AND IT IS NOT THE CONVOLUTION'S OWN INDEX.
  //
  // arch=8 packs a person detector AND a landmark network into one container, and
  // those two graphs have SEPARATE index spaces -- both start at zero, both have a
  // conv 0. So the network's own index cannot address a panel: the landmark
  // network's conv 0 and the detector's conv 0 are different weights. The caller
  // assigns slot numbers across all of its graphs' dense convolutions in one run
  // (arch=8: the detector's 0..72, then the landmark network's 73..160) and
  // passes the same run to its graph walks, which is what makes the two agree by
  // construction rather than by coincidence.
  int64_t slot = -1;
  // Which network this convolution belongs to, for error messages only.
  std::string group;
  // The weight-name prefix: "det." / "pose." / "".
  std::string prefix;
  // That architecture's own convolution index, which is what its weight table is
  // named with.
  int64_t conv = -1;
  // The design stream this convolution runs on, as the packer recorded it.
  std::string stream;
  int64_t cin = 0, cout = 0, kh = 0, kw = 0;
  // The convolution's own bias, or nullptr. The backend builds a padded copy, so
  // this is only read during construction.
  const float *bias = nullptr;
  // The weight table's name, i.e. prefix + "conv." + conv + ".btile".
  std::string panel_name() const {
    return prefix + "conv." + std::to_string(conv) + ".btile";
  }
  // K is the reduction width and N the output width, unpadded.
  int64_t k() const { return cin * kh * kw; }
};

// `slots` must list every DENSE convolution of every graph that will run, and it
// must be in a fixed order: panels are staged in that order and the status line
// prints them in it.
std::unique_ptr<Convs> make_array_backend(
    npue::File &model, const std::vector<ConvSlot> &slots,
    const std::string &container_name, const std::string &artifacts_dir,
    app::Pool &pool);

}  // namespace npue::conv

#endif  // NPUE_CONV_ARRAY_HPP