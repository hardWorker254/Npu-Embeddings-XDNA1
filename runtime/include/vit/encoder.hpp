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
// The head, for a reason that is in tools/pack/packers/vit.py's header: 1000 is
// not a multiple of tile_n * cols = 48 * 4 = 192, so there is no legal B panel
// of that width for this array at all, and a 768x1000 matvec per image costs far
// less than the ~150 us a dispatch does. The head is therefore NOT a --npu-
// extra-ops code and never will be -- it is not missing, it does not exist on
// this board.
//
// LayerNorm, GELU and softmax WERE host-only for a while, with the reason "a
// kind's stream list has no eltwise designs in it". That was true and it was
// the wrong kind of answer: it described the export, not the model. A ViT has
// the same two pre-LN LayerNorms and the same ungated exact-erf FFN that
// Whisper has, npue::whisper::NpuEltwise already implements both kernels, and
// `--npu-ops layn` / `gelu` now build the sibling design sets for
// kinds.cls exactly as they do for kinds.stt. So the two codes below are real
// per-op host/array choices and the flag says so.
//
// SOFTMAX IS A PER-OP CHOICE AS WELL, and it was not always one. The sentence
// that used to live here refused it by name on the grounds that shipping the
// seq x seq score matrix to the array and back for one elementwise pass could
// only lose against a host softmax that never crosses a bus at all. That
// argument predicted the SIGN and got the MAGNITUDE wrong in a way that had to
// be measured rather than argued: on vit-base-patch16-224 (197 positions, 12
// layers, 12 heads) the host encoder takes 0.244 s and --npu-ops softm takes
// 0.822 s for exactly 12 dispatches, because the design fills its whole row
// capacity on every one of them whether this model has 2364 score rows to
// normalise or the kernel was built for 12288. The array is 3.4x slower here,
// the same direction as the argument and not the same size as it.
//
// What the argument also got wrong was that this is a *choice* with nothing to
// move: kinds.cls's attn_qk/attn_av streams exist now (resolve.py consults the
// registry for every kind, not only stt), so the two GEMMs bracketing the
// softmax are the separate `attn` code, and a run may take one, the other, or
// both. NpuAttention runs the three phases when both are asked for and hands
// the score row to this operator between its own two dispatches; with `softm`
// alone the GEMMs stay on the host and only the score row crosses. A score row
// is padded to the design's width with -1.0e30f, a value the exp() underflows
// to zero and the row max never picks up, so the padding is not a term in any
// normalisation this pass produces.
//
// The throughput claim above seq 64 is a measurement now rather than a hole:
// the array's attention came out SLOWER than the host at 197 positions (0.349 s
// against 0.244 s of encoder), which is the same story the embedders told at
// their own sequence lengths. These codes make the model RUN on the array, not
// run faster on it.
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
#include "whisper/eltwise.hpp"    // NpuEltwise, the two elementwise designs
#include "whisper/npu_ops.hpp"   // NpuGemm, layernorm_rows, gelu_erf_inplace, attention
#include "whisper/attention_npu.hpp"  // NpuAttention -- attn as two GEMMs

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
};

class VitEncoder {
public:
  VitEncoder(npue::File &model, npu::Design &design, app::Pool &pool,
             const Geometry &geom);

  void set_streams(const EncoderStreams &s) { streams_ = s; }

  // The two elementwise passes, host by default and on the array when these are
  // set. Null means host, and the HOST path is not a fallback: it is the
  // measured-faster one here for the same reason it is in Whisper, and the two
  // agree to bf16 rather than exactly, which is the datapath's own error and
  // not this schedule's.
  //
  // NpuEltwise rather than a ViT-owned copy, for the reason the GEMM wrapper is
  // shared too: the kernels and their dispatch are one implementation, so a
  // divergence between the two architectures would have to be typed in rather
  // than inherited.
  void set_layernorm(npue::whisper::NpuEltwise *ln) { ln_ = ln; }
  void set_gelu(npue::whisper::NpuEltwise *gelu) { gelu_ = gelu; }
  // Attention as two GEMMs on the set's own attn_qk/attn_av slots. Null is the
  // host pass -- a real path and not a fallback. Installed by the classify
  // wrapper only after the same geometry checks Whisper makes, so a set that
  // cannot carry this shape never reaches the loop below.
  void set_attention(npue::whisper::NpuAttention *a) { attn_ = a; }
  // The softmax design, for `softm` WITHOUT `attn`. When attention itself is on
  // the array the two GEMMs carry the softmax with them inside NpuAttention and
  // this pointer is unused there; it is the host attention below that reads it,
  // and it takes the score row wide enough for the kernel.
  void set_softmax(npue::whisper::NpuEltwise *s) { softm_ = s; }

  // True when that pass is on the array. Read by the status block so the block
  // reports where a thing actually ran rather than what was requested.
  bool layernorm_on_array() const { return ln_ != nullptr; }
  bool gelu_on_array() const { return gelu_ != nullptr; }
  // True when THIS encoder will put the softmax on the array -- which it does
  // only when attention itself is on the host, because NpuAttention takes the
  // same operator through its own set_softmax() and runs it between its two
  // GEMMs. `attn` alone therefore still normalises on the host and is not
  // reported as a softm run; `attn,softm` puts it on the array here too, so
  // the pointer is the single source for both questions.
  bool softmax_on_array() const { return softm_ != nullptr; }

  // The elementwise dispatches this encoder made. Reported as its own number
  // rather than added into the GEMM's, because they answer a different question:
  // 25 LayerNorm sites and 12 GELU blocks are 37 on top of the GEMM's 49, and
  // the two together would be a number that cannot be attributed to either.
  //
  // The wall time of those dispatches is deliberately NOT exposed here. The GEMM
  // wrapper's timer is not exposed either, because the status block prints
  // before any image has been classified and per-image timings already come back
  // in the result -- so an accessor would be a public number with no reader.
  int64_t elt_dispatch() const {
    return (ln_ ? ln_->n_dispatch : 0) + (gelu_ ? gelu_->n_dispatch : 0) +
           (softm_ ? softm_->n_dispatch : 0);
  }

  // The two attention GEMMs, counted apart from the 49 and added back by the
  // caller. Reading it here rather than inside elt_dispatch() is deliberate:
  // attn_qk and attn_av are GEMMs on gemm_rtp's own slots, so they belong in
  // `dispatches` and not in `elt_dispatches` -- a softmax design is a separate
  // xclbin and these two are not. `reset_timers()` drains it, so the number is
  // this classify's and not this process's.
  int64_t attn_dispatch() const { return attn_ ? attn_->n_dispatch : 0; }

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

  // There is deliberately NO gemm1() here. A ViT-specific wrapper around
  // NpuGemm::run() existed once, branching on the container's operand dtype and
  // calling run_i8() itself; it was the one call site outside NpuGemm::run()
  // that had to know about int8, and it got the SmoothQuant direction wrong --
  // it passed the container's asmooth where run_i8 documents 1/asmooth, so every
  // ViT activation was divided by s instead of by 1/s, the product came out as
  // X @ W * asmooth^2, and the classifier answered confidently and wrongly
  // (bus.jpg: 'water jug' at p=0.019 where fp32 says 'minibus' at p=0.629).
  // run() already routes on the datapath and already holds the per-operand
  // scales against the B slot, so the five call sites below say run() and the
  // dtype is decided in exactly one place.

  npue::File &model_;
  npue::whisper::NpuGemm g_;
  // The array's two elementwise designs, or null for host. See set_layernorm.
  npue::whisper::NpuEltwise *ln_ = nullptr, *gelu_ = nullptr;
  npue::whisper::NpuAttention *attn_ = nullptr;
  npue::whisper::NpuEltwise *softm_ = nullptr;
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

  // The staged gamma|beta slot per LayerNorm site, in the order run() visits
  // them: 2*layers pre-LN sites then the final one. Empty when LayerNorm is on
  // the host, and stage_all() fills it only then -- staging 25 parameter pairs
  // that nothing reads would be 25 transfers of nothing.
  //
  // Site order is a list, not an index computed at the call site, because a
  // site_index(L, which) helper would be a second way to say the same thing as
  // the loop that stages it, and those two would disagree the first time a layer
  // was added a third norm to.
  std::vector<size_t> ln_slot_;

  // The head, host-side, row-major F32 [d_model, num_labels]. The arithmetic
  // is head_matvec() in vit/head.hpp -- its own translation unit, because it
  // is the one piece of arch=5 a host-only box can falsify and this file
  // drags in the whole AIE stack.
  const float *head_ = nullptr;
  const float *head_bias_ = nullptr;
};

}  // namespace npue::vit
