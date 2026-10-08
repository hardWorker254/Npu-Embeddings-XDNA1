//===- encoder.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=5 classifier stack. See vit/encoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "vit/encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/host_kernels.hpp"
#include "common/int4_panel.hpp"
#include "vit/head.hpp"

namespace npue::vit {

VitEncoder::VitEncoder(npue::File &model, npu::Design &design, app::Pool &pool,
                       const Geometry &geom)
    : model_(model), g_(design, pool), pool_(pool), geom_(geom) {}

size_t VitEncoder::stage_all() {
  if (streams_.rows <= 0)
    throw std::runtime_error("vit encoder: set_streams() before stage_all()");
  const int64_t d = geom_.d_model;
  size_t bytes = 0;

  // THE OPERAND DTYPE, decided ONCE and from the entries rather than from the
  // config string. `a_dtype` is what a packer wrote; the tensor directory's
  // dtypes are what the bytes are. A container whose two disagree -- a packer
  // that wrote I8 panels and left "a_dtype": "BF16" -- is a container that
  // would otherwise be read as bf16 and produce numbers 100x too small, which
  // looks like a model that does not work rather than a disagreement.
  //
  // Two operands of different dtypes in one container is refused rather than
  // resolved per operand: the array's MMAC has ONE A element width, so a mixed
  // container has no legal dispatch schedule at all.
  {
    const std::string a = model_.info("layer.0.qkv").dtype;
    const std::string b = model_.info("layer.0.ffn_down").dtype;
    const std::string c = model_.info("frontend.patch_embed").dtype;
    if (a != b || a != c)
      throw std::runtime_error(
          "vit encoder: layer.0.qkv is " + a + ", layer.0.ffn_down is " + b +
          " and frontend.patch_embed is " + c +
          ". The array's MMAC has one A element width, so a container with "
          "operands of mixed dtype has no legal dispatch at all. Repack with "
          "tools/pack/pack_npue.py.");
    if (a == "BF16") {
      int8_ = false;
    } else if (a == "I8" || a == "I4") {
      // I4 is the SAME datapath with a narrower weight: the payload is
      // nibbles, but gemm_b_panel widens them to an int8 panel before stage(),
      // so the MMAC sees an I8 operand and needs the same scales. Admitting it
      // here rather than in a branch of its own is the whole point of the
      // format -- there is no fourth dispatch schedule.
      int8_ = true;
    } else {
      throw std::runtime_error("vit encoder: GEMM operands are " + a +
                               ", and this build dispatches BF16, I8 or I4. "
                               "Refusing rather than reinterpreting the bytes.");
    }
    const std::string cfg = model_.config_string("a_dtype");
    const std::string want = int8_ ? "I8" : "BF16";
    if (cfg != want)
      throw std::runtime_error(
          "vit encoder: a_dtype is '" + cfg + "' but the packed operands are " +
          want + ". Refusing rather than picking one -- the design set's A "
          "element width has to agree with the container's, and this container "
          "does not even agree with itself.");
  }

  auto operand = [&](const std::string &name, Operand &out) {
    out.slot = g_.stage_operand(model_, name);
    out.bias = model_.raw(name + ".bias").as<float>();
    if (int8_) {
      // The sidecars are NOT optional, and this check stays even though the
      // scales themselves are no longer read here: stage_operand() takes them
      // for its own OpScale, and an I8 panel with no per-channel scale is not a
      // weight at a different precision -- it is a number 1e-4 of the right
      // size, and the failure is a plausible classifier rather than a crash.
      for (const char *suffix : {".wscale", ".asmooth"})
        if (!model_.has(name + suffix))
          throw std::runtime_error(
              name + " is I8 but " + name + suffix +
              " is not in the container -- its bytes carry no scale, so they "
              "are not a weight. Repack with tools/pack/pack_npue.py --int8.");
    }
    // The STAGED size, not the stored one: an I4 payload is half-width, so
    // raw().bytes would under-count it by two and the total this function
    // returns would no longer be what the slots hold.
    bytes += staged_bytes(model_, name);
  };
  auto norm = [&](const std::string &name, std::vector<const float *> &gamma,
                  std::vector<const float *> &beta) {
    gamma.push_back(model_.raw(name + ".weight").as<float>());
    beta.push_back(model_.raw(name + ".bias").as<float>());
    bytes += 2 * static_cast<size_t>(d) * sizeof(float);
  };

  operand("frontend.patch_embed", patch_);
  for (int64_t L = 0; L < geom_.layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    operand(p + "qkv", qkv_.emplace_back());
    operand(p + "attn_out", ao_.emplace_back());
    operand(p + "ffn_up", fu_.emplace_back());
    operand(p + "ffn_down", fd_.emplace_back());
    norm(p + "ln1", ln1_gamma, ln1_beta);
    norm(p + "ln2", ln2_gamma, ln2_beta);
  }
  pos_ = model_.raw("frontend.position_embeddings").as<float>();
  cls_token_ = model_.raw("frontend.cls_token").as<float>();
  bytes += static_cast<size_t>(geom_.n_pos) * d * sizeof(float) +
           static_cast<size_t>(d) * sizeof(float);
  final_gamma_ = model_.raw("layernorm.weight").as<float>();
  final_beta_ = model_.raw("layernorm.bias").as<float>();
  bytes += 2 * static_cast<size_t>(d) * sizeof(float);

  // The gamma|beta pairs, staged as the LayerNorm design's parameter operand --
  // but only when that design exists. This is 2*12+1 = 25 slots of 2*768 floats,
  // staged once for the life of the session because a LayerNorm site is a fixed
  // weight and re-staging it per dispatch would be a per-layer transfer to save
  // nothing. The order here IS the order run() indexes ln_slot_ by (all the
  // pre-LN sites in layer order, then the final one), and it is written as a
  // loop over the same two collections the host path reads rather than as 25
  // separate calls, so a site cannot be staged in one order and visited in
  // another.
  if (ln_) {
    // gamma|beta in ONE buffer, gamma first: the core tile's two input DMA
    // channels take the parameters as a pair, so there is one staging call per
    // site and not two. That pairing is why a site is staged once and read by
    // index, rather than the two halves being staged separately.
    const int64_t d = geom_.d_model;
    std::vector<float> gb(static_cast<size_t>(2 * d));
    auto site = [&](const float *gamma, const float *beta) {
      std::copy(gamma, gamma + d, gb.begin());
      std::copy(beta, beta + d, gb.begin() + d);
      return ln_->stage_params(gb);
    };
    for (size_t i = 0; i < ln1_gamma.size(); ++i) {
      ln_slot_.push_back(site(ln1_gamma[i], ln1_beta[i]));
      ln_slot_.push_back(site(ln2_gamma[i], ln2_beta[i]));
    }
    ln_slot_.push_back(site(final_gamma_, final_beta_));
  }

  // The head. Stored plain row-major F32 under the role `gemm_b_host`, with NO
  // layout and therefore no layout_hash, so there is nothing for it to
  // disagree with a design about -- it is the container's own statement that
  // this one operand is not the array's. It is declared [d_model, num_labels]
  // and the packer transposes torch's [labels, hidden] to match, so the shape
  // in the directory is the shape of the bytes.
  {
    const auto &wi = model_.info("classifier.weight");
    if (wi.logical_shape.size() != 2 ||
        wi.logical_shape[0] != geom_.d_model ||
        wi.logical_shape[1] != geom_.num_labels)
      throw std::runtime_error(
          "classifier.weight is declared [" +
          std::to_string(wi.logical_shape.size() == 2 ? wi.logical_shape[0] : -1) +
          "," +
          std::to_string(wi.logical_shape.size() == 2 ? wi.logical_shape[1] : -1) +
          "], expected [" + std::to_string(geom_.d_model) + "," +
          std::to_string(geom_.num_labels) +
          "]. The head is stored K-major like every other operand in this "
          "format; a checkpoint's own [labels, hidden] layout has to be "
          "transposed by the packer, and a container that stored it straight "
          "would be a well-formed matrix of the right shape holding a "
          "transposed model.");
    if (wi.dtype != "F32")
      throw std::runtime_error("classifier.weight is " + wi.dtype +
                               ". This head is a host matvec in fp32 and the "
                               "packer refuses to store anything else; "
                               "refusing rather than converting here, because a "
                               "conversion would not be the same tensor.");
    head_ = model_.raw("classifier.weight").as<float>();
    head_bias_ = model_.raw("classifier.bias").as<float>();
    bytes += wi.nbytes +
             static_cast<size_t>(geom_.num_labels) * sizeof(float);
  }

  g_.alloc_buffers();
  return bytes;
}

void VitEncoder::reset_timers() {
  g_.n_dispatch = 0;
  g_.t_dispatch = 0.0;
  g_.t_convert = 0.0;
  // The attention's too, by draining rather than assigning: NpuAttention
  // accumulates across images and take_timers() is its read-and-zero. Leaving
  // it undrained would print the FIRST image's attention dispatches after the
  // fifth image and the sum of all five after the fifth -- a count that grows
  // while the work it describes does not.
  if (attn_) attn_->take_timers();
}

std::vector<float> VitEncoder::run(const std::vector<float> &patches) {
  const int64_t d = geom_.d_model, inter = geom_.intermediate;
  const int64_t rows = streams_.rows;
  if (rows <= 0)
    throw std::runtime_error("vit encoder: set_streams() before run()");

  // The two elementwise passes, one call site each. Both are the same choice
  // Whisper's encoder makes with the same two classes, and both read a pointer
  // that is null unless --npu-ops asked for the design -- so a run that
  // asked for neither takes the host path without a second flag, and a run that
  // asked cannot reach the host one by forgetting something.
  //
  // THREE helpers rather than one indexed by site number: the three call sites
  // below already hold gamma and beta for their own site, and an index that
  // turns (layer, which) back into that pair is a second, easier-to-misread
  // spelling of what they already say.
  auto ln1 = [&](float *x, int64_t n, int64_t L) {
    if (ln_)
      ln_->layernorm(x, n, ln_slot_[static_cast<size_t>(2 * L)]);
    else
      npue::whisper::layernorm_rows(x, n, d, ln1_gamma[static_cast<size_t>(L)],
                                    ln1_beta[static_cast<size_t>(L)],
                                    geom_.ln_eps, pool_);
  };
  auto ln2 = [&](float *x, int64_t n, int64_t L) {
    if (ln_)
      ln_->layernorm(x, n, ln_slot_[static_cast<size_t>(2 * L + 1)]);
    else
      npue::whisper::layernorm_rows(x, n, d, ln2_gamma[static_cast<size_t>(L)],
                                    ln2_beta[static_cast<size_t>(L)],
                                    geom_.ln_eps, pool_);
  };
  // The final one is over the whole finished stack rather than a chunk of a
  // layer, and it has no layer index; NpuEltwise walks it in chunks of the
  // design's own row capacity, so the row count handed over is n_pos either way.
  auto ln_final = [&](float *x, int64_t n) {
    if (ln_)
      ln_->layernorm(x, n, ln_slot_[static_cast<size_t>(2 * geom_.layers)]);
    else
      npue::whisper::layernorm_rows(x, n, d, final_gamma_, final_beta_,
                                    geom_.ln_eps, pool_);
  };
  // GELU is one flat span, not rows: the kernel's row capacity is 1 and the
  // runtime walks the whole activation block in chunks of whatever the design
  // holds, which is why the call takes an ELEMENT count here and a row count
  // for the LayerNorms above.
  auto gelu = [&](float *x, int64_t n_elems) {
    if (gelu_)
      gelu_->gelu(x, n_elems);
    else
      npue::whisper::gelu_erf_inplace(x, static_cast<size_t>(n_elems), pool_);
  };
  if (static_cast<int64_t>(patches.size()) <
      geom_.n_patches * geom_.patch_dim)
    throw std::runtime_error(
        "vit encoder: the patch matrix is " +
        std::to_string(patches.size() / geom_.patch_dim) + " rows, " +
        std::to_string(geom_.n_patches) + " wanted (" +
        std::to_string(geom_.image_size) + "px at patch " +
        std::to_string(geom_.patch_size) + ")");

  // -- the patch embedding, on the attn_out stream ---------------------------
  //
  // [n_patches, patch_dim] x [patch_dim, hidden] with patch_dim == hidden, so
  // this is attn_out's own shape and it runs on attn_out's instruction slot.
  // The result has n_patches rows and no CLS row yet: the CLS vector is
  // PREPENDED, which is not the same as offsetting the patch rows by one.
  //
  // `conv` is passed per RUN, not per instance: this operand shares this
  // instance's `gemm` calls (and its staging) but is named by `conv`, so a run
  // with `--npu-ops gemm` and none with `--npu-ops conv` puts it on the host
  // while the four per-layer GEMMs go to the array, and the other way round.
  // It is the one call site in the tree that does not inherit its instance's
  // code, and the reason is that this operand is a convolution that happens to
  // fit a GEMM slot.
  std::vector<float> patch_out(static_cast<size_t>(rows) * d);
  {
    int64_t done = 0;
    while (done < geom_.n_patches) {
      const int64_t n = std::min<int64_t>(rows, geom_.n_patches - done);
      g_.run(streams_.attn_out, patches.data() + done * geom_.patch_dim, n, rows,
             geom_.patch_dim, patch_.slot, patch_.bias, d, patch_out.data(),
             "conv");
      done += n;
    }
  }

  // -- the CLS row and the position table ------------------------------------
  std::vector<float> x(static_cast<size_t>(geom_.n_pos) * d);
  {
    float *row0 = x.data();
    for (int64_t j = 0; j < d; ++j) row0[j] = cls_token_[j];
    for (int64_t p = 0; p < geom_.n_patches; ++p)
      std::copy(patch_out.begin() + static_cast<size_t>(p) * d,
                patch_out.begin() + static_cast<size_t>(p + 1) * d,
                x.begin() + static_cast<size_t>(p + 1) * d);
    // Positions are added AFTER the concatenation and BEFORE the first
    // LayerNorm, which is HF's order: pos_embeddings[:, 1:] + patch, then
    // cat((cls, patches)), then + pos_embeddings[:, :1]. Same sum, same tensor,
    // and the CLS row therefore holds position 0 like any other.
    for (int64_t t = 0; t < geom_.n_pos; ++t) {
      const float *pe = pos_ + t * d;
      float *row = x.data() + t * d;
      for (int64_t j = 0; j < d; ++j) row[j] += pe[j];
    }
  }

  // Per-chunk scratch, sized by the DESIGN's row count rather than by n_pos,
  // because the design computes all `rows` rows whatever the chunk holds.
  std::vector<float> norm(static_cast<size_t>(rows) * d);
  std::vector<float> qkv_chunk(static_cast<size_t>(rows) * 3 * d);
  std::vector<float> qkv_all(static_cast<size_t>(geom_.n_pos) * 3 * d);
  std::vector<float> ctx(static_cast<size_t>(geom_.n_pos) * d);
  std::vector<float> proj(static_cast<size_t>(rows) * d);
  std::vector<float> up(static_cast<size_t>(rows) * inter);
  std::vector<float> down(static_cast<size_t>(rows) * d);
  // The host score matrix. Only when the host path will READ it: on the array
  // QK^T writes its own scratch, so this buffer would be allocated, sized and
  // never touched. An earlier version of this comment said attention was a host
  // pass because kinds.cls listed no attn streams -- that described the export
  // rather than the model, and the streams are built for kinds.cls now.
  //
  // The ROW STRIDE is the softmax design's width and not n_pos, when `softm`
  // asked for the array: that kernel reduces over its whole row, so the row it
  // is handed has to BE its whole width. n_pos is 197 and the design is built
  // at n_kv = 384 (197 padded to tile_n*cols = 192), which makes this buffer
  // 197*12*384 = 908 k floats, 3.6 MB rather than 1.9 MB -- and the columns
  // past 197 are filled with -1e30 inside attention() so they contribute
  // nothing to the denominator. Without `softm` there is no such kernel and the
  // rows are n_pos wide, which is what the host pass has always used.
  const int64_t score_stride = softm_ ? softm_->cols() : geom_.n_pos;
  std::vector<float> scores(
      attn_ ? 0
            : static_cast<size_t>(geom_.n_pos) * geom_.heads * score_stride,
      0.f);

  // The attention scale is inside the Q weight and the Q bias unless the
  // container was packed with --no-fold-scale. Applying it here as well would
  // be applying it twice, which is a different model -- see the qkv_scale_folded
  // key and the bug it was added for.
  const float scale = geom_.qkv_scale_folded
                          ? 1.0f
                          : static_cast<float>(geom_.attn_scale);

  for (int64_t L = 0; L < geom_.layers; ++L) {
    // ---- self-attention, pre-LN -------------------------------------------
    chunks(geom_.n_pos, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      std::copy(x.begin() + r0 * d, x.begin() + r1 * d, norm.begin());
      ln1(norm.data(), n, L);
      g_.run(streams_.qkv, norm.data(), n, rows, d, qkv_[L].slot, qkv_[L].bias,
           3 * d, qkv_chunk.data());
      std::copy(qkv_chunk.begin(), qkv_chunk.begin() + n * 3 * d,
                qkv_all.begin() + r0 * 3 * d);
    });
    // The fused [Q|K|V] row is Q at offset 0 and a K|V block at offset d, both
    // with row stride 3*d_model -- the same layout the Whisper encoder reads.
    // There is NO mask: a ViT's encoder attends over every position it was
    // given, and every position is a real one, so set_additive_mask is never
    // called and the class's null default is the right answer.
    //
    // The two paths write the same `ctx`, and the host one below is not a
    // fallback for the array one: it is the reference the array one is measured
    // against. On the array the score matrix below is not used at all -- QK^T
    // writes its own buffer and NpuAttention reads it back -- which is why it
    // is only allocated when the host path will read it.
    if (attn_) {
      attn_->run(qkv_all.data(), 3 * d, qkv_all.data() + d, 3 * d,
                 geom_.n_pos, geom_.n_pos, d, scale, ctx.data());
    } else {
      // `softm_` only when it was asked for: it splits this into QK^T, the
      // softmax on the array, and softmax.V, and moves exactly that one op --
      // the two GEMMs stay on the host, because moving them is `attn`.
      npue::whisper::attention(qkv_all.data(), 3 * d, qkv_all.data() + d, 3 * d,
                               geom_.n_pos, geom_.n_pos, d, geom_.heads,
                               geom_.head_dim, scale, ctx.data(), scores.data(),
                               pool_, softm_, score_stride);
    }
    chunks(geom_.n_pos, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      g_.run(streams_.attn_out, ctx.data() + r0 * d, n, rows, d, ao_[L].slot,
           ao_[L].bias, d, proj.data());
      for (int64_t i = 0; i < n * d; ++i)
        x[r0 * d + i] += proj[static_cast<size_t>(i)];
    });

    // ---- feed-forward, pre-LN ---------------------------------------------
    chunks(geom_.n_pos, [&](int64_t r0, int64_t r1) {
      const int64_t n = r1 - r0;
      std::copy(x.begin() + r0 * d, x.begin() + r1 * d, norm.begin());
      ln2(norm.data(), n, L);
      g_.run(streams_.ffn_up, norm.data(), n, rows, d, fu_[L].slot, fu_[L].bias,
           inter, up.data());
      gelu(up.data(), n * inter);
      g_.run(streams_.ffn_down, up.data(), n, rows, inter, fd_[L].slot,
           fd_[L].bias, d, down.data());
      for (int64_t i = 0; i < n * d; ++i)
        x[r0 * d + i] += down[static_cast<size_t>(i)];
    });
  }

  // transformers' ViTModel applies a final LayerNorm after the encoder, and
  // that normalised CLS row is what the head consumes. Skipping it changes the
  // logits by a scale, which changes which label wins, without changing any
  // shape -- so it is not a detail.
  // The whole n_pos in one call, not per chunk: this one is over the finished
  // stack rather than over a chunk of a layer.
  ln_final(x.data(), geom_.n_pos);
  return x;
}

void VitEncoder::classify(const std::vector<float> &cls_row,
                          std::vector<float> &logits) {
  const int64_t d = geom_.d_model, nlab = geom_.num_labels;
  if (static_cast<int64_t>(cls_row.size()) < d)
    throw std::runtime_error("vit head: the CLS row is " +
                             std::to_string(cls_row.size() / std::max<int64_t>(d, 1)) +
                             " rows, one wanted");
  logits.assign(static_cast<size_t>(nlab), 0.f);
  // One 768 x 1000 matvec, on the host, over the container's own F32 head. The
  // arithmetic is head_matvec() in vit/head.hpp -- this is a slice of it, not a
  // second matvec, because two matvecs would be two strides to keep equal.
  //
  // The labels are CONTIGUOUS blocks, not interleaved. head_matvec walks a
  // stride-num_labels run of `head` per label, so an interleaved split gives
  // every worker the same long strided walk over the whole 3 MB table and
  // defeats the cache; contiguous blocks give each worker its own slab. The
  // boundaries are integer divisions rather than a ceil, so the blocks are
  // exact: no worker writes a label another owns, and none is left unwritten.
  pool_.run([&](int w, int nw) {
    const int64_t j0 = nlab * w / nw, j1 = nlab * (w + 1) / nw;
    head_matvec(head_, head_bias_, cls_row.data(), d, nlab, j0, j1,
                logits.data());
  });
}

}  // namespace npue::vit
