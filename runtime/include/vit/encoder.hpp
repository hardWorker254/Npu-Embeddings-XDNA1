//===- encoder.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the ViT classifier stack, arch=5: patches -> patch embedding
// -> +CLS/positions -> L pre-LN layers -> final LayerNorm -> host head.
//
// WHY A MODE AND NOT AN EXTENSION OF BertEncoder
// ----------------------------------------------
// BertEncoder is post-LN and reads BERT's tensor names. This is pre-LN and
// reads `layer.i.*` / `frontend.*`. Both facts are silent when you get them
// wrong -- every shape, every byte count and every layout hash agrees, and only
// the ANSWER is a different model's. So the arithmetic that is genuinely shared
// (one NPU GEMM wrapper, the host LayerNorm/GELU/attention) comes from
// npue::whisper::npu_ops.hpp, which was extracted for exactly this reason, and
// only the SCHEDULE is written here.
//
// THE SCHEDULE
// ------------
//   patches [n_patches, patch_dim] @ patch_embed -> + bias          NPU, attn_out
//   row 0 := cls_token, row i+1 := row i, then + position_embeddings
//   per layer, twice:
//     x += attention(ln1(x))                                    pre-LN
//     x += ffn_down(gelu(ffn_up(ln2(x))))                        pre-LN
//   x := layernorm(x)                                            the final one
//   logits := x[0] @ classifier.weight + classifier.bias         HOST
//
// WHAT RIDES WHICH STREAM, AND WHY IT IS NOT A FIFTH STREAM
// ---------------------------------------------------------
// The patch embedding is [n_patches, 768] x [768, 768] -- the SAME K and N as
// attn_out, which is why it is dispatched on attn_out's instruction slot with
// attn_out's weights staged once alongside. It is a [768,768] operand, so the
// GEMM is not merely similar to attn_out's, it IS attn_out's shape, and tools/
// data/npu_targets.json's `kinds.cls` lists the gemm_rtp streams verbatim
// because nothing new is needed to export one.
//
// WHAT IS ON THE HOST AND WHY
// ---------------------------
// LayerNorm, GELU, softmax and the head. The first three for the same reason
// Whisper's are: a kind's stream list has no eltwise designs in it, and one
// xclbin plus one hw_context apiece is not free. The HEAD for a harder reason,
// which is in tools/pack/packers/vit.py's header: 1000 is not a multiple of
// tile_n * cols = 48 * 4 = 192, so there is no legal B panel of that width for
// this array at all, and a 768x1000 matvec per image costs far less than the
// ~150 us a dispatch does.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/geometry.hpp"
#include "whisper/npu_ops.hpp"   // NpuGemm, layernorm_rows, gelu_erf_inplace, attention

namespace npue::vit {

// Which instruction-stream slot runs which op, and how many rows one dispatch
// of each computes. Read off the loaded design set's own `streams` array --
// the slot numbers are the export's, and a set exported with different tiers
// has different ones.
struct EncoderStreams {
  size_t attn_out = 0, qkv = 0, ffn_up = 0, ffn_down = 0;
  int64_t rows = 0;
};

// One operand's staging state: the device slot for its pre-tiled panel, its fp32
// bias, and -- on an int8 container -- the two host-side sidecars the epilogue
// needs. A plain struct rather than four parallel vectors because the four
// vectors were four ways to have one of them out of step with the others, and
// every use site indexed all four by the same layer number.
struct Operand {
  size_t slot = 0;
  const float *bias = nullptr;
  const float *wscale = nullptr;   // int8 only: per OUTPUT channel
  const float *asmooth = nullptr;  // int8 only: per INPUT channel
};

class VitEncoder {
public:
  VitEncoder(npue::File &model, npu::Design &design, app::Pool &pool,
             const Geometry &geom);

  void set_streams(const EncoderStreams &s) { streams_ = s; }

  // Stage every operand, every LayerNorm and the two front-end tables. Returns
  // the bytes staged, for the status line.
  size_t stage_all();

  // patches [n_patches, patch_dim] -> the CLS row, [n_pos, d_model], after the
  // final LayerNorm. That is what transformers' ViTModel hands the head, so it
  // is what the gate compares.
  std::vector<float> run(const std::vector<float> &patches);

  // The head, on the host, spread over the pool by label. The arithmetic is
  // head_matvec() below -- one function, one implementation -- so this is a
  // slice of it and not a second matvec.
  void classify(const std::vector<float> &cls_row, std::vector<float> &logits);

  int64_t hidden() const { return geom_.d_model; }
  const Geometry &geometry() const { return geom_; }
  const npue::whisper::NpuGemm &gemm() const { return g_; }
  void reset_timers();

  // True when this container's GEMM operands are int8. READ from the entries,
  // never from `a_dtype`: the design's own a_elem_bytes is what the dispatch
  // depends on and the container's string is what a packer wrote, and a
  // container whose two disagree is a container this must refuse rather than
  // branch on the wrong one.
  bool int8() const { return int8_; }

private:
  // Walk `n` rows in chunks of the design's own row count.
  template <typename F> void chunks(int64_t n, F &&f) const {
    for (int64_t r0 = 0; r0 < n; r0 += streams_.rows)
      f(r0, std::min<int64_t>(n, r0 + streams_.rows));
  }

  // One GEMM, routed on the container's operand dtype. The two branches are the
  // two functions in npue::whisper::NpuGemm and nothing else -- a third call
  // site for "dispatch a GEMM" is a third place for the operand dtype to be
  // decided, which is exactly what went wrong when the packer and the design
  // disagreed about int8.
  void gemm1(size_t instr, const float *a, int64_t n_real, int64_t k,
             const Operand &w, int64_t n, float *out);

  npue::File &model_;
  npue::whisper::NpuGemm g_;
  app::Pool &pool_;
  Geometry geom_;
  EncoderStreams streams_;
  bool int8_ = false;

  // The front end's own two tables, plus the patch-embedding operand.
  Operand patch_;
  const float *pos_ = nullptr;     // [n_pos, d_model]
  const float *cls_token_ = nullptr;  // [d_model]

  // Per layer: the four operands and the two LayerNorm sites.
  std::vector<Operand> qkv_, ao_, fu_, fd_;
  std::vector<const float *> ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
  const float *final_gamma_ = nullptr, *final_beta_ = nullptr;

  // The head, host-side, row-major F32 [d_model, num_labels]. The arithmetic
  // is head_matvec() in vit/head.hpp -- its own translation unit, because it
  // is the one piece of arch=5 a host-only box can falsify and this file
  // drags in the whole AIE stack.
  const float *head_ = nullptr;
  const float *head_bias_ = nullptr;
};

}  // namespace npue::vit
