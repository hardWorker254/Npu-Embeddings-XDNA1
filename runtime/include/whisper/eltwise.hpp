//===- eltwise.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper's elementwise passes on the array: LayerNorm,
// softmax and GELU, one sibling xclbin each.
//
// WHY A SEPARATE XCLBIN AND NOT ANOTHER STREAM
// --------------------------------------------
// The GEMM set's streams are GEMMs: the RTP program's shim and its NOC carry
// operand panels and an accumulator. An elementwise kernel carries a row of
// activations and a gamma|beta vector, so it is a different program on the same
// device -- and one resident xclbin is the rule this project has never broken
// (runtime/include/runtime/design.hpp). So each op is its own design directory,
// exactly as BertEncoder's are, and each costs one hw_context: the npu1 driver
// allows six and Whisper already spends two on its two GEMM sets.
//
// WHAT THE DESIGN OWNS, AND WHAT THE CONTAINER OWNS
// -------------------------------------------------
// The kernel's ROW WIDTH and its EPSILON are compiled in (kernels/layernorm.cc,
// -DLN_COLS / -DLN_EPS) and recorded in design.json, because the container's
// d_model and layer_norm_eps are the model's own and a design built for another
// model would normalise with the wrong constant inside a square root. Both are
// checked here before the first dispatch: an eps of 1e-12 against a container
// that says 1e-5 is a refusal, not a rounding difference.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/pool.hpp"

namespace npue::whisper {

// Which op a design directory holds. The set of binds below differs per op --
// LayerNorm takes a parameter vector, softmax and GELU do not -- and binding the
// wrong argument count is the same class of mistake as running the wrong
// instruction stream, so it is named once here and carried in the object.
enum class EltwiseKind { LayerNorm, Softmax, Gelu };

// One elementwise design, its A/C buffers, and the three entry points.
//
// Every op is dispatched over the design's OWN row capacity and no less: the
// program fills its whole input buffer, so a dispatch that sent fewer rows would
// leave the rest holding the previous call's data. The tail is zeroed and the
// result's tail rows are discarded, which is exact for all three ops: LayerNorm
// of a zero row is beta, softmax of a row of -1e30 is uniform, GELU of zero is
// zero.
class NpuEltwise {
public:
  NpuEltwise(npu::Design &design, app::Pool &pool, EltwiseKind kind)
      : d_(design), pool_(pool), kind_(kind) {}

  // Must be called once, before the first run(): the A and C slots come from the
  // design's own buffer sizes, not from a width this class guessed.
  void alloc_buffers();

  // The design's row capacity and width, read from design.json. `rows` is how
  // many rows one dispatch computes and `cols` how wide a row is; both are the
  // design's, and both are what the caller has to fit its tensor into.
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }

  // LayerNorm over `n_real` rows of `cols_` columns, in place, with the
  // gamma|beta pair staged at `param_slot` (one fp32 buffer of 2*cols, gamma
  // first -- a core tile has two input DMA channels and three arguments would
  // need three). The container's epsilon is not an argument: it is compiled into
  // this design and checked against the container in alloc_buffers().
  void layernorm(float *x, int64_t n_real, size_t param_slot);

  // Softmax over `n_rows` rows of `cols_` columns, in place. No mask: Whisper's
  // encoder attends over every position it was given and the decoder's cache
  // holds only positions at or before the current one.
  void softmax(float *x, int64_t n_rows);

  // GELU on `n_real` elements, in place. The kernel is the model's own
  // exact-erf activation, not the tanh polynomials the shipped one uses: the
  // packer refuses a checkpoint whose activation is not `gelu`, and 2.5e-3
  // relative is larger than the bf16 datapath's own error.
  void gelu(float *x, int64_t n_real);

  // Stage a gamma|beta pair as the design's parameter operand and return its
  // slot. The pair is ONE buffer of 2*cols fp32, gamma first: a core tile has
  // two input DMA channels, and gamma and beta as two arguments need three
  // ("tile (0,3) requires 3 input/1 output DMA channels").
  //
  // Per site, once, at session start -- the same reason the GEMM operands are
  // staged once: a LayerNorm site is 2*d floats, and re-staging it per dispatch
  // would be a per-layer transfer to save nothing.
  size_t stage_params(const std::vector<float> &gamma_beta);

  int64_t n_dispatch = 0;
  double t_dispatch = 0.0;
  std::mutex *npu_mu = nullptr;

private:
  // fill the A buffer with `n` values and zeros to the design's capacity
  void fill_input(const float *x, int64_t n);
  // read the first `n` values of the C buffer back into x
  void read_back(float *x, int64_t n) const;
  void dispatch(size_t bytes, bool with_params, size_t param_slot);

  npu::Design &d_;
  app::Pool &pool_;
  EltwiseKind kind_;
  size_t slot_a = 0, slot_c = 0;
  int64_t rows_ = 0, cols_ = 0;
};

}  // namespace npue::whisper
