//===- bert_encoder.cpp ------------------------------------------------*- C++ -*-===//
//
// BERT-family NPU encoder implementation. Moved from
// runtime/include/bert_encoder.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "encoders/bert_encoder.hpp"

#include "common/app_state.hpp"
#include "common/conv_host.hpp"  // hostconv::gemm_nt, the host multiply
#include "common/host_b.hpp"     // hostb::host_b_kn, the untile
#include "common/int4_panel.hpp"
#include "common/npu_ops_flag.hpp"  // op_on_array -- array or host, one place
#include "encoders/gemma_kernels.hpp"
#include "runtime/model.hpp"
#include "tokenizers/tokenizer_facade.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace app;

namespace npue {

BertEncoder::BertEncoder(npue::File &model,
                           npu::Design &qkv_d, npu::Design &ao_d,
                           npu::Design &fu_d, npu::Design &fd_d,
                           npu::Design &gelu_d, npu::Design &ln_d,
                           npu::Design &sm_d, Pool &pool)
    : model_(model), qkv_(qkv_d), attn_out_(ao_d), ffn_up_(fu_d),
      ffn_down_(fd_d), gelu_(gelu_d), layernorm_(ln_d), softmax_(sm_d),
      pool_(pool) {}

int64_t BertEncoder::hidden() const { return g_hidden; }

int64_t BertEncoder::seq() const { return g_seq; }

size_t BertEncoder::stage_all() {
  size_t bytes = 0;
  const bool i8 = qkv_.info().a_elem_bytes == 1;

  // A HOST-LAYOUT CONTAINER (`--dtype f32`, gemm_layout "host") carries plain
  // row-major F32 operands: no panel, no layout_hash, and nothing for the DMA
  // to read. It still has to RUN -- the default for every one of the nine
  // codes is the host, and common/host_b.hpp's F32 branch reads exactly these
  // bytes with no untile -- so the per-operand staging below is skipped rather
  // than failed, the same way `host_ln` skips the layernorm design's staging.
  //
  // Absent means PRETILED, not broken: this key was written only by arch=1
  // until tasks/0140, and every other arch's containers predate it, so a
  // silent `== "host"` here would read all 498 older containers as host ones.
  // Only the spelled-out value flips the path.
  bool host_layout = false;
  try {
    host_layout = model_.config_string("gemm_layout") == "host";
  } catch (const std::exception &) {
    host_layout = false;
  }

  // The cache key for an unstaged weight. stage() numbers slots from zero
  // within one design's buffer, so anything this far out is a number it cannot
  // produce -- and it is never bound (`d.bind(1, wslot)` is the array path,
  // which refuses a host container above rather than reaching it).
  static constexpr size_t kHostSlot = static_cast<size_t>(1) << 40;
  size_t host_seq = 0;  // one key space for all four per-layer designs

  auto one = [&](npu::Design &d, const std::string &name,
                 std::vector<size_t> &slots,
                 std::vector<const float *> &bias,
                 std::vector<const float *> *wsc = nullptr,
                 std::vector<const float *> *asm_ = nullptr) {
    if (host_layout) {
      // Asked for the array by name, on a container that has no panel to
      // hand it: refuse with the reason rather than stage nothing and let
      // the bind fail later on a slot number it was never told about.
      if (app::op_on_array("gemm"))
        throw std::runtime_error(
            name + ": this container is `--dtype f32` (gemm_layout \"host\") "
            "-- its GEMM operands are row-major F32 for the CPU, and the MMAC "
            "multiplies bf16 and int8 only. --npu-ops gemm is refused rather "
            "than silently ignored; drop the flag, or repack with "
            "--dtype bf16/i8 for the array.");
      // ONE counter for all four vectors, not one per vector: in unified mode
      // qkv_/attn_out_/ffn_up_/ffn_down_ are the SAME Design object, so the
      // cache key is (that pointer, slot) and four vectors each starting at
      // zero would collapse 4L operands onto L keys -- which is what made the
      // host path multiply ffn_down at qkv's [K, N] on the first attempt.
      slots.push_back(kHostSlot + host_seq++);
      host_w_->record(&d, slots.back(), name);
      bias.push_back(model_.raw(name + ".bias").as<float>());
      // No wscale/asmooth: the packer writes neither for an f32 operand, and
      // i8 is false here anyway (the design it embeds is the bf16 one).
      return;
    }
    const std::string &want = d.info().b_layout_hash;
    const std::string &got = model_.info(name).layout_hash;
    if (want.empty())
      throw std::runtime_error(d.info().name +
                               "/design.json has no "
                               "b_layout_hash -- re-export with "
                               "tools/export_xclbin.py");
    if (got.empty())
      throw std::runtime_error(name + ": .npue tensor carries no "
                               "layout_hash -- repack with "
                               "tools/pack/pack_npue.py");
    if (want != got)
      throw std::runtime_error(
          name + ": layout mismatch -- design " + d.info().name +
          " wants " + want.substr(0, 16) + "..., file has " +
          got.substr(0, 16) + "... The bytes would be the right size and the "
          "wrong order.");
    // One accessor for BOTH schemes (common/int4_panel.hpp): bf16 and I8 come
    // straight out of the mapping with no copy, and an I4 payload is widened
    // to the int8 panel the array consumes before stage() ever sees it. The
    // bytes counted are the STAGED ones -- K*N for int8 and int4, K*N*2 for
    // bf16 -- which is what the slot sizes this function returns add up to.
    const Panel panel = gemm_b_panel(model_, name, d.info().a_elem_bytes);
    slots.push_back(d.stage(1, panel.bytes.data, panel.bytes.bytes));
    // Where the host path finds this weight again: gemm() is handed `wslot`,
    // and common/host_b.hpp needs the CONTAINER tensor name to untile from.
    // Written here because this is the only place that knows both halves.
    host_w_->record(&d, slots.back(), name);
    bias.push_back(model_.raw(name + ".bias").as<float>());
    bytes += panel.bytes.bytes;
    if (i8 && wsc) {
      wsc->push_back(model_.raw(name + ".wscale").as<float>());
      asm_->push_back(model_.raw(name + ".asmooth").as<float>());
    }
  };
  for (int64_t L = 0; L < g_layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    one(qkv_, p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
    one(attn_out_, p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
    one(ffn_up_, p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
    one(ffn_down_, p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
  }

  std::vector<float> gb(2 * g_hidden);
  auto ln_one = [&](const std::string &g, const std::string &b) {
    std::memcpy(gb.data(), model_.raw(g).data, g_hidden * sizeof(float));
    std::memcpy(gb.data() + g_hidden, model_.raw(b).data,
                g_hidden * sizeof(float));
    s_ln.push_back(layernorm_.stage(1, gb.data(), gb.size() * sizeof(float)));
    bytes += gb.size() * sizeof(float);
    h_gamma.push_back(model_.raw(g).as<float>());
    h_beta.push_back(model_.raw(b).as<float>());
  };
  if (host_ln) {
    auto ln_host = [&](const std::string &g, const std::string &b) {
      s_ln.push_back(s_ln.size() + 1);
      h_gamma.push_back(model_.raw(g).as<float>());
      h_beta.push_back(model_.raw(b).as<float>());
    };
    ln_host("embeddings.ln.weight", "embeddings.ln.bias");
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      ln_host(p + "ln1.weight", p + "ln1.bias");
      ln_host(p + "ln2.weight", p + "ln2.bias");
    }
    return bytes;
  }
  ln_one("embeddings.ln.weight", "embeddings.ln.bias");
  for (int64_t L = 0; L < g_layers; ++L) {
    const std::string p = "layer." + std::to_string(L) + ".";
    ln_one(p + "ln1.weight", p + "ln1.bias");
    ln_one(p + "ln2.weight", p + "ln2.bias");
  }
  return bytes;
}

int64_t BertEncoder::use_tier(int64_t want) {
  if (tiers.empty()) return batch;
  size_t pick = tiers.size() - 1;
  for (size_t i = 0; i < tiers.size(); ++i)
    if (tiers[i] >= want) { pick = i; break; }
  batch = tiers[pick];
  rows = batch * g_seq;
  is_qkv = tier_slots[pick][0];
  is_ao = tier_slots[pick][1];
  is_fu = tier_slots[pick][2];
  is_fd = tier_slots[pick][3];
  // Attention follows the tier. Null here is the host path -- not an oversight:
  // a tier whose set carries no attn_qk/attn_av is a tier that was exported
  // without the code, and load-time refuses that combination by name rather
  // than letting it silently drop to the host per tier.
  attn_ = pick < attns.size() ? attns[pick] : nullptr;
  return batch;
}

void BertEncoder::reset_timers() {
  t_qk = t_av = 0.0;
  t_npu = t_attn = 0.0;
  t_hostln = t_hostsm = t_hostgelu = 0.0;
  t_conv = t_in = t_disp = t_out = t_bias = 0.0;
  n_dispatch = 0;
  // NpuAttention times itself, and those counters are ADDED into t_qk/t_av
  // above -- so they are accumulators across every request since the last
  // reset, not per-request values. Leaving them running makes the status block
  // print a QK^T larger than the attention total it is a sub-item of, which is
  // the one inconsistency this block is not allowed to have.
  for (auto *a : attns) {
    a->t_qk = 0.0;
    a->t_av = 0.0;
    a->n_dispatch = 0;
  }
}

template <typename F>
void BertEncoder::par(size_t n, F &&f) const {
  if (pool_.size() == 1 || n < 65536) { f(size_t(0), n); return; }
  pool_.run([&](int w, int nw) {
    const size_t chunk = ((n / nw) + 63) & ~size_t(63);
    const size_t lo = std::min(n, chunk * size_t(w));
    const size_t hi = std::min(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

template <typename F>
void BertEncoder::par_rows(int64_t n, F &&f) const {
  if (pool_.size() == 1) { f(int64_t(0), n); return; }
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n, chunk * w);
    const int64_t hi = std::min<int64_t>(n, lo + chunk);
    if (lo < hi) f(lo, hi);
  });
}

double BertEncoder::lap(double t0, double &bucket) {
  double t = now_s();
  bucket += t - t0;
  return t;
}

// HOW MUCH OF A TENSOR ONE DISPATCH OF AN ELTWISE DESIGN COMPUTES, as the
// (offset, count) chunks that add up to the whole thing.
//
// These designs compute [rows, cols] tiles, and this file used to assume a
// tensor always fits one of them: `eltwise` and `layer_norm` converted the WHOLE
// tensor into the A buffer with no capacity check anywhere. For most models that
// assumption holds, because the exporter sizes each design to hold exactly one
// batch tier of its own tensor at seq 64 and nothing asks for a second.
//
// bge-large asks for a second, and it does so exactly where the sizing is
// tightest. The softmax design is ONE program reused for every model -- a fixed
// 12288 rows x 64 cols -- and 12288 rows is precisely batch 16 * seq 64 * 12
// heads. bge-large has 16 heads, so at tier 16 it needs 16*16*64*64 = 1,048,576
// elements against a 786,432-element buffer: a third over. `bf16_fill` then wrote
// 262,144 bf16 past the end of a device buffer and the process died with SIGSEGV
// (exit 139, no message at all) from a flag whose entire contract is "move this
// op to the array". bge-base sits on the boundary at exactly 786,432, which is
// why it passed and its 33%-larger sibling did not -- a test that only ever ran
// the 12-head models would have kept passing.
//
// Every design states what it holds, in two fields: `cols` is the row width and
// `row_capacity` the rows one dispatch computes. Sets exported before
// `row_capacity` existed still record `cols`, and dividing the A buffer's
// element capacity by it recovers exactly the number the newer ones state:
// all-MiniLM's three sets work out at 1572864, 1024 and 12288 rows, the same
// values bge-base records explicitly. The plan is therefore read from the design
// in both cases, never assumed -- which is also why it is checked rather than
// trusted: a design whose `cols` does not divide the tensor it is handed is a
// design exported for a different shape, and that is a refusal, not a rewrite.
static std::vector<std::pair<size_t, size_t>> elt_chunks(const npu::DesignInfo &in,
                                                         size_t n) {
  std::vector<std::pair<size_t, size_t>> out;
  if (in.buffer_bytes.size() < 2)
    throw std::runtime_error(in.name +
                             ": fewer than two buffers, so there is no input "
                             "buffer to stage into");
  const size_t cap = in.buffer_bytes[0] / sizeof(uint16_t);
  if (n == 0) return out;

  // GELU. `kind` is "eltwise" and it declares `cols` as the WHOLE flat span with
  // row_capacity 1, so for this op a row IS a dispatch and the walk advances in
  // ELEMENTS. That is not a shortcut: an element-wise function is correct at
  // every split point, whereas softmax and LayerNorm are only correct at row
  // boundaries -- and a GELU call is routinely smaller than the design, so its
  // length is not a multiple of anything and must not be asked to be.
  if (in.kind == "eltwise") {
    for (size_t off = 0; off < n; off += cap)
      out.emplace_back(off, std::min(cap, n - off));
    return out;
  }

  // LayerNorm and softmax are ROW-wise: softmax normalises one score row and
  // LayerNorm one hidden row, so a chunk boundary inside a row would compute a
  // row that does not exist in the tensor. `cols` is load-bearing here, not
  // descriptive, which is why a missing one has to be refused rather than
  // defaulted -- there is no safe unit to fall back to.
  if (in.cols <= 0)
    throw std::runtime_error(
        in.name + "/design.json records no row width (`cols`), and this op is "
        "row-wise, so a tensor larger than one dispatch cannot be split without "
        "cutting a row in half. Re-export with tools/export/export_eltwise.py, "
        "which writes `cols`.");
  const size_t cols = static_cast<size_t>(in.cols);
  if (n % cols != 0)
    throw std::runtime_error(
        in.name + ": " + std::to_string(n) + " elements is not a whole number of " +
        std::to_string(cols) + "-wide rows, so this design cannot compute this "
        "tensor without splitting a row. It was exported for a different shape "
        "than the one it is being handed -- a softmax whose cols is the sequence "
        "length, a LayerNorm whose cols is d_model. Re-export it for this model.");
  const size_t rows_total = n / cols;
  const size_t rows_per =
      in.row_capacity > 0 ? static_cast<size_t>(in.row_capacity) : cap / cols;
  if (rows_per == 0)
    throw std::runtime_error(in.name + ": a " + std::to_string(cap) +
                             "-element buffer holds no whole row of " +
                             std::to_string(cols));
  for (size_t r = 0; r < rows_total; r += rows_per)
    out.emplace_back(r * cols,
                     std::min(rows_per, rows_total - r) * cols);
  return out;
}

void BertEncoder::eltwise(npu::Design &d, const EltSlots &slots, float *x,
                          size_t n) {
  const size_t cap_elems = d.info().buffer_bytes[0] / sizeof(uint16_t);
  // One chunk for every case that already fitted, so this changes no result that
  // used to work -- it stops the cases that did not fit from writing off the end.
  for (const auto &chunk : elt_chunks(d.info(), n)) {
  const size_t off = chunk.first, cnt = chunk.second;
  double t0 = now_s();
  float *xb = x + off;
  // The A buffer is this lane's own, so the conversion runs unlocked and in
  // parallel. It is the dispatch window below that has to be exclusive.
  auto *in_bf16 = static_cast<uint16_t *>(d.slot_ptr(0, slots.a));
  par(cnt, [&](size_t lo, size_t hi) {
    bf16_fill(in_bf16 + lo, xb + lo, hi - lo);
  });
  if (cnt < cap_elems)
      std::memset(in_bf16 + cnt, 0, (cap_elems - cnt) * sizeof(uint16_t));
  t0 = lap(t0, t_conv);
  // ONE mutex, exactly as gemm() takes it, because the thing being protected is
  // the same thing gemm() protects: Design's `active` slot table is shared
  // mutable state, so two lanes binding and dispatching one Design concurrently
  // read a torn binding and each other's buffers. Previously only gemm() took
  // it -- and gelu/softmax/layernorm, which have no per-lane weight slot to
  // hide behind, silently raced here.
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d.bind(0, slots.a);
    d.bind(1, slots.c);
    d.sync_to_device(0, cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(d.output_index(), cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_out);
  }
  // Read back through slot_ptr, not host_ptr: an explicit slot touches neither
  // the shared binding table nor the lock, and the C buffer is this lane's own.
  const auto *out_bf16 = static_cast<const uint16_t *>(
      d.slot_ptr(d.output_index(), slots.c));
  par(cnt, [&](size_t lo, size_t hi) {
    bf16_read(xb + lo, out_bf16 + lo, hi - lo);
  });
  lap(t0, t_conv);
  ++n_dispatch;
  }
}

void BertEncoder::layer_norm(std::vector<float> &x, size_t slot) {
  if (host_ln) { layer_norm_cpu(x, slot - 1); return; }
  const size_t cap_elems = layernorm_.info().buffer_bytes[0] / sizeof(uint16_t);
  // Same chunking and the same reason as eltwise(): this design is sized to one
  // batch tier of B*seq*hidden, which is an exact fit for every model that has
  // one -- and a buffer overrun for one that needs two.
  for (const auto &chunk : elt_chunks(layernorm_.info(), x.size())) {
  const size_t off = chunk.first, cnt = chunk.second;
  double t0 = now_s();
  auto *in_bf16 = static_cast<uint16_t *>(
      layernorm_.slot_ptr(0, slots_ln.a));
  par(cnt, [&](size_t lo, size_t hi) {
    bf16_fill(in_bf16 + lo, x.data() + off + lo, hi - lo);
  });
  if (cnt < cap_elems)
      std::memset(in_bf16 + cnt, 0, (cap_elems - cnt) * sizeof(uint16_t));
  t0 = lap(t0, t_conv);
  // Same window, same reason, same mutex as eltwise() and gemm(). `slot` is the
  // gamma|beta site, and lanes sit at DIFFERENT sites at the same instant -- so
  // an unlocked bind(1, slot) handed this dispatch another layer's parameters,
  // which is the larger half of the corruption. It is the same site for every
  // chunk of this call, because a chunk is a piece of one layer, not a layer.
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    layernorm_.bind(0, slots_ln.a);
    layernorm_.bind(1, slot);
    layernorm_.bind(2, slots_ln.c);
    layernorm_.sync_to_device(0, cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_in);
    layernorm_.dispatch_only();
    t0 = lap(t0, t_disp);
    layernorm_.sync_from_device(layernorm_.output_index(),
                                cap_elems * sizeof(uint16_t));
    t0 = lap(t0, t_out);
  }
  const auto *out_bf16 = static_cast<const uint16_t *>(
      layernorm_.slot_ptr(layernorm_.output_index(), slots_ln.c));
  par(cnt, [&](size_t lo, size_t hi) {
    bf16_read(x.data() + off + lo, out_bf16 + lo, hi - lo);
  });
  lap(t0, t_conv);
  ++n_dispatch;
  }
}

void BertEncoder::layer_norm_cpu(std::vector<float> &x, size_t site) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t n_rows = static_cast<int64_t>(x.size()) / g_hidden;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
    for (int64_t r = lo; r < hi; ++r) {
      float *row = x.data() + r * g_hidden;
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_hidden; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / g_hidden;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_hidden; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / g_hidden;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (int64_t j = 0; j < g_hidden; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                    _mm256_loadu_ps(g + j),
                                    _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, y);
      }
#else
      double sm = 0.0;
      for (int64_t j = 0; j < g_hidden; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / g_hidden);
      double sv = 0.0;
      for (int64_t j = 0; j < g_hidden; ++j) {
        const float d = row[j] - mean;
        sv += static_cast<double>(d) * d;
      }
      const float var = static_cast<float>(sv / g_hidden);
      const float is = 1.0f / std::sqrt(var + 1e-12f);
      for (int64_t j = 0; j < g_hidden; ++j)
        row[j] = (row[j] - mean) * is * g[j] + b[j];
#endif
    }
  });
  t_hostln += now_s() - t0;
}

void BertEncoder::add_additive_mask(std::vector<float> &scores) {
  const int64_t rows_per_seq = g_heads * g_seq;
  const int64_t n_rows = static_cast<int64_t>(scores.size()) / sc_cols;
  pool_.run([&](int w, int nw) {
    for (int64_t r = w; r < n_rows; r += nw) {
      float *row = scores.data() + r * sc_cols;
      const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
      for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
    }
  });
}

// Only ever called with host_sm set, and run() makes sc_cols == g_seq exactly
// then, so every stride below is the row's true width. The array branch uses
// add_additive_mask() + eltwise() instead, at sc_cols, with the padded tail
// already in the row.
void BertEncoder::softmax_cpu(std::vector<float> &scores) {
  double t0 = now_s();
  const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
    const int64_t rows_per_seq = g_heads * g_seq;
    for (int64_t r = lo; r < hi; ++r) {
      float *row = scores.data() + r * g_seq;
      const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
      for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
#if defined(__AVX2__)
      __m256 mx = _mm256_loadu_ps(row);
      for (int64_t j = 8; j < g_seq; j += 8)
        mx = _mm256_max_ps(mx, _mm256_loadu_ps(row + j));
      __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx),
                             _mm256_extractf128_ps(mx, 1));
      m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
      m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
      const __m256 mv = _mm256_set1_ps(_mm_cvtss_f32(m4));
      const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
      const __m256 argfloor = _mm256_set1_ps(-120.0f);
      __m256 sum = _mm256_setzero_ps();
      for (int64_t j = 0; j < g_seq; j += 8) {
        __m256 a = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv),
                                    log2e);
        __m256 e = exp2_avx2(_mm256_max_ps(a, argfloor));
        _mm256_storeu_ps(row + j, e);
        sum = _mm256_add_ps(sum, e);
      }
      const __m256 inv = _mm256_set1_ps(1.0f / hsum256(sum));
      for (int64_t j = 0; j < g_seq; j += 8)
        _mm256_storeu_ps(row + j,
                            _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
#else
      float m = row[0];
      for (int64_t j = 1; j < g_seq; ++j) m = std::max(m, row[j]);
      float sum = 0.f;
      for (int64_t j = 0; j < g_seq; ++j) {
        row[j] = std::exp(row[j] - m);
        sum += row[j];
      }
      const float inv = 1.0f / sum;
      for (int64_t j = 0; j < g_seq; ++j) row[j] *= inv;
#endif
    }
  });
  t_hostsm += now_s() - t0;
}

void BertEncoder::gelu_cpu(std::vector<float> &x) {
  double t0 = now_s();
  par(x.size(), [&](size_t lo, size_t hi) {
    size_t i = lo;
#if defined(__AVX2__)
    const __m256 vR = _mm256_set1_ps(4.0f);
    const __m256 vz = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 c0 = _mm256_set1_ps(-7.2340282171e-05f);
    const __m256 c1 = _mm256_set1_ps(1.8179518005e-03f);
    const __m256 c2 = _mm256_set1_ps(-1.7707383379e-02f);
    const __m256 c3 = _mm256_set1_ps(8.4577147641e-02f);
    const __m256 c4 = _mm256_set1_ps(-1.9228671834e-01f);
    const __m256 c5 = _mm256_set1_ps(9.8431124458e-02f);
    const __m256 c6 = _mm256_set1_ps(3.6137852062e-01f);
    const __m256 c7 = _mm256_set1_ps(-4.9454128936e-01f);
    const __m256 c8 = _mm256_set1_ps(-1.3007010117e-04f);
    for (; i + 8 <= hi; i += 8) {
      __m256 v = _mm256_loadu_ps(x.data() + i);
      __m256 u = _mm256_min_ps(_mm256_andnot_ps(sign, v), vR);
      __m256 pl = _mm256_fmadd_ps(c0, u, c1);
      pl = _mm256_fmadd_ps(pl, u, c2);
      pl = _mm256_fmadd_ps(pl, u, c3);
      pl = _mm256_fmadd_ps(pl, u, c4);
      pl = _mm256_fmadd_ps(pl, u, c5);
      pl = _mm256_fmadd_ps(pl, u, c6);
      pl = _mm256_fmadd_ps(pl, u, c7);
      pl = _mm256_fmadd_ps(pl, u, c8);
      _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_max_ps(v, vz), pl));
    }
#endif
    for (; i < hi; ++i) {
      const float v = x[i];
      const float u = std::min(std::fabs(v), 4.0f);
      float pl = -7.2340282171e-05f;
      pl = pl * u + 1.8179518005e-03f;
      pl = pl * u + -1.7707383379e-02f;
      pl = pl * u + 8.4577147641e-02f;
      pl = pl * u + -1.9228671834e-01f;
      pl = pl * u + 9.8431124458e-02f;
      pl = pl * u + 3.6137852062e-01f;
      pl = pl * u + -4.9454128936e-01f;
      pl = pl * u + -1.3007010117e-04f;
      x[i] = std::max(v, 0.0f) + pl;
    }
  });
  t_hostgelu += now_s() - t0;
}

void BertEncoder::swiglu_cpu(const std::vector<float> &x,
                              std::vector<float> &out) {
  double t0 = now_s();
  const int64_t inter = g_ffn;
  const int64_t n_rows = rows;
  pool_.run([&](int w, int nw) {
    const int64_t chunk = (n_rows + nw - 1) / nw;
    const int64_t lo_r = std::min<int64_t>(n_rows, chunk * w);
    const int64_t hi_r = std::min<int64_t>(n_rows, lo_r + chunk);
    for (int64_t r = lo_r; r < hi_r; ++r) {
      const float *lo = x.data() + r * 2 * inter;
      const float *hi = lo + inter;
      float *dst = out.data() + r * inter;
      if (g_gated_act == GatedAct::GeluErf) {
        for (int64_t j = 0; j < inter; ++j)
          dst[j] = lo[j] * gelu_erf_exact(hi[j]);
        continue;
      }
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
      const __m256 argfloor = _mm256_set1_ps(-120.0f);
      const __m256 one = _mm256_set1_ps(1.0f);
      for (; j + 8 <= inter; j += 8) {
        __m256 xv = _mm256_loadu_ps(hi + j);
        __m256 a = _mm256_max_ps(
            _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
            argfloor);
        __m256 e = exp2_avx2(a);
        __m256 s = _mm256_div_ps(xv, _mm256_add_ps(one, e));
        __m256 loV = _mm256_loadu_ps(lo + j);
        _mm256_storeu_ps(dst + j, _mm256_mul_ps(loV, s));
      }
#endif
      for (; j < inter; ++j) {
        const float xv = hi[j];
        float a = -xv * 1.4426950408889634f;
        if (a < -120.0f) a = -120.0f;
        const float e = std::exp2(a);
        const float s = xv / (1.0f + e);
        dst[j] = lo[j] * s;
      }
    }
  });
  t_hostgelu += now_s() - t0;
}

const float *BertEncoder::i8w(const std::vector<const float *> &v, int64_t L) {
  return v.empty() ? nullptr : v[static_cast<size_t>(L)];
}

void BertEncoder::dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                                    int64_t out_n, bool gated, const float *bias,
                                    uint16_t *dst) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      float *v = row.data();
      int64_t j = 0;
#if defined(__AVX2__)
      if (c_bytes == 2) {
        const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256i lo = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
          __m256i hi = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
          _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                _mm256_loadu_ps(bias + j + 8)));
        }
      } else {
        const float *cr = static_cast<const float *>(c) + r * N;
        for (; j + 8 <= N; j += 8) {
          __m256i raw = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                _mm256_loadu_ps(bias + j)));
        }
      }
#endif
      for (; j < N; ++j) {
        const float cf = c_bytes == 2
            ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
            : static_cast<const float *>(c)[r * N + j];
        v[j] = cf + bias[j];
      }
      if (!gated) {
        int64_t k = 0;
#if defined(__AVX2__)
        for (; k + 8 <= N; k += 8)
          _mm256_storeu_ps(v + k, gelu8(_mm256_loadu_ps(v + k)));
#endif
        for (; k < N; ++k) v[k] = gelu8(v[k]);
      } else if (g_gated_act == GatedAct::GeluErf) {
        const int64_t inter = N / 2;
        const float *hi = v + inter;
        for (int64_t k = 0; k < inter; ++k)
          v[k] = v[k] * gelu_erf_exact(hi[k]);
      } else {
        const int64_t inter = N / 2;
        const float *hi = v + inter;
        int64_t k = 0;
#if defined(__AVX2__)
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; k + 8 <= inter; k += 8) {
          __m256 xv = _mm256_loadu_ps(hi + k);
          __m256 a = _mm256_max_ps(
              _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
              argfloor);
          __m256 e = exp2_avx2(a);
          __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
          _mm256_storeu_ps(v + k, _mm256_mul_ps(_mm256_loadu_ps(v + k), sg));
        }
#endif
        for (; k < inter; ++k) {
          const float xv = hi[k];
          float a = -xv * 1.4426950408889634f;
          if (a < -120.0f) a = -120.0f;
          v[k] = v[k] * (xv / (1.0f + std::exp2f(a)));
        }
      }
      bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
    }
  });
}

void BertEncoder::host_gemm(const std::vector<float> &a,
                            std::vector<float> &out, int64_t N,
                            const float *bias, const float *asmooth,
                            npu::Design &d, size_t wslot) {
  if (a.empty() || rows <= 0 || a.size() % static_cast<size_t>(rows))
    throw std::runtime_error(
        "bert: host GEMM input of " + std::to_string(a.size()) +
        " elements is not a whole number of the " + std::to_string(rows) +
        " rows this encoder's shapes are built for");
  const int64_t K = static_cast<int64_t>(a.size()) / rows;
  if (K <= 0 || N <= 0)
    throw std::runtime_error("bert: host GEMM with a zero dimension");
  const std::vector<float> &w = host_w_->get(model_, &d, wslot, K, N,
                                            d.info().name.c_str());
  std::vector<float> scratch;
  const float *A =
      npue::hostb::apply_inv_smooth(a.data(), static_cast<size_t>(rows), K,
                                    asmooth, scratch);
  // NOTE, and it is the honest half of this path: no a_scale, no int8 rounding.
  // The array quantises A per ROW to int8 and dequantises C with that row's
  // scale; the host never quantises, so the number it produces is the fp32
  // product of the same operands rather than an int8 approximation of it. The
  // two therefore agree to int8 rounding and not bit for bit -- which is what
  // every comparison of host against array in this project measures.
  hostconv::gemm_nt(A, rows, w.data(), K, N, bias, out.data(), pool_);
}

void BertEncoder::gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
                         size_t wslot, const float *bias, std::vector<float> &out,
                         int64_t N, const float *wscale,
                         const float *asmooth,
                         FusedNext *fuse, bool a_ready,
                         FusedNextBf16 *fuse_bf16) {
  const bool i8 = d.info().a_elem_bytes == 1;
  if (i8 && (wscale == nullptr || asmooth == nullptr))
    throw std::runtime_error(
        "int8 design but this encoder has no quantisation scales -- the "
        "container is bf16, or a pipeline lane was constructed without "
        "copying ws_*/as_* from lane 0");

  // ARRAY OR HOST, decided by the command line alone, before anything that
  // belongs to a dispatch: `a_ready` and the two fuse* pointers are all about
  // the design's A buffer, and a host run reads neither.
  if (!gemm_on_array()) {
    host_gemm(a, out, N, bias, asmooth, d, wslot);
    ++n_host;
    return;
  }

  double t0 = now_s();
  if (i8 && a_ready) {
  } else if (i8) {
    const int64_t K = static_cast<int64_t>(a.size()) / rows;
    a_scale.resize(static_cast<size_t>(rows));
    if (inv_smooth.size() != static_cast<size_t>(K) ||
        inv_smooth_src != asmooth) {
      inv_smooth.resize(static_cast<size_t>(K));
      for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
      inv_smooth_src = asmooth;
    }
    const float *ias = inv_smooth.data();
    auto *abuf = static_cast<int8_t *>(d.slot_ptr(0, slot_a));
    quantise_a_int8(a.data(), rows, K, ias, abuf, a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
    t0 = lap(t0, t_conv);
  } else if (a_ready) {
  } else {
  auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
  par(a.size(), [&](size_t lo, size_t hi) {
    bf16_fill(abuf + lo, a.data() + lo, hi - lo);
  });
  t0 = lap(t0, t_conv);
  }
  const float *c;
  // The A DMA is OUTSIDE npu_mu on purpose. It addresses this lane's own slot
  // by name (sync_slot_to_device), so it touches no state another lane reads,
  // and dispatch_only() blocks on r.wait(), so the array finished with this
  // lane's slot before we began filling it again. Holding the mutex across a
  // 75 us copy is what cost 805 us per lane at --pipeline 4.
  d.sync_slot_to_device(0, slot_a, a.size() * d.info().a_elem_bytes);
  t0 = lap(t0, t_in);
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    if (unified) d.bind_instr(islot);
    d.bind(0, slot_a);
    d.bind(1, wslot);
    d.bind(2, slot_c);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
  }
  const size_t cb = d.info().c_elem_bytes;
  // Likewise for C: r.wait() has returned, so this lane's C slot is final and
  // private. Another lane dispatching concurrently writes its OWN slot.
  d.sync_slot_from_device(2, slot_c, static_cast<size_t>(rows) * N * cb);
  t0 = lap(t0, t_out);
  c = static_cast<const float *>(d.slot_ptr(2, slot_c));
  if (i8 && fuse) {
    const int64_t Kn = fuse->gated ? N / 2 : N;
    if (inv_smooth_next.size() != static_cast<size_t>(Kn) ||
        inv_smooth_next_src != fuse->asmooth) {
      inv_smooth_next.resize(static_cast<size_t>(Kn));
      for (int64_t j = 0; j < Kn; ++j)
        inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
      inv_smooth_next_src = fuse->asmooth;
    }
    const int64_t out_n = fuse->gated ? N / 2 : N;
    dequant_act_quant(
        c, d.info().c_elem_bytes, rows, N, out_n,
        [gated = fuse->gated](float *v, int64_t n) {
          if (!gated) {
            int64_t j = 0;
#if defined(__AVX2__)
            for (; j + 8 <= n; j += 8)
              _mm256_storeu_ps(v + j, gelu8(_mm256_loadu_ps(v + j)));
#endif
            for (; j < n; ++j) v[j] = gelu8(v[j]);
            return;
          }
          if (g_gated_act == GatedAct::GeluErf) {
            const int64_t inter = n / 2;
            const float *hi = v + inter;
            for (int64_t j = 0; j < inter; ++j)
              v[j] = v[j] * gelu_erf_exact(hi[j]);
            return;
          }
          const int64_t inter = n / 2;
          const float *hi = v + inter;
          int64_t j = 0;
#if defined(__AVX2__)
          const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
          const __m256 argfloor = _mm256_set1_ps(-120.0f);
          const __m256 one = _mm256_set1_ps(1.0f);
          for (; j + 8 <= inter; j += 8) {
            __m256 xv = _mm256_loadu_ps(hi + j);
            __m256 a = _mm256_max_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                argfloor);
            __m256 e = exp2_avx2(a);
            __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
            _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(v + j), sg));
          }
#endif
          for (; j < inter; ++j) {
            const float xv = hi[j];
            float a = -xv * 1.4426950408889634f;
            if (a < -120.0f) a = -120.0f;
            v[j] = v[j] * (xv / (1.0f + std::exp2f(a)));
          }
        },
        a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
        fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
  } else if (i8) {
    dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); }, sim_c_bf16);
  } else if (fuse_bf16) {
    const int64_t out_n = fuse_bf16->gated ? N / 2 : N;
    dequant_act_bf16(c, d.info().c_elem_bytes, N, out_n, fuse_bf16->gated,
                       bias, fuse_bf16->dst);
  } else if (d.info().c_elem_bytes == 2) {
    const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
    // par_rows, not par: par()'s 65536 threshold is calibrated in ELEMENTS and
    // this call passes a ROW count, so rows=1024 always took par()'s serial
    // fallback -- the single largest pass in the profile ran on one thread.
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb16 + r * N;
        float *o = out.data() + r * N;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256i lo = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
          __m256i hi = _mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
          _mm256_storeu_ps(o + j,
                           _mm256_add_ps(_mm256_castsi256_ps(lo),
                                         _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(o + j + 8,
                           _mm256_add_ps(_mm256_castsi256_ps(hi),
                                         _mm256_loadu_ps(bias + j + 8)));
        }
#endif
        for (; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
      }
    });
  } else {
    // Same unit mismatch as the bf16 branch above -- rows, not elements.
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *cr = c + r * N;
        float *o = out.data() + r * N;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= N; j += 8) {
          __m256i raw = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          _mm256_storeu_ps(o + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                _mm256_loadu_ps(bias + j)));
        }
#endif
        for (; j < N; ++j) o[j] = cr[j] + bias[j];
      }
    });
  }
  lap(t0, t_bias);
  ++n_dispatch;
}

void BertEncoder::apply_rope_qkv(std::vector<float> &qkv) {
  if (!rope_ready) {
    rope_cos.resize(static_cast<size_t>(g_seq * g_head_dim));
    rope_sin.resize(static_cast<size_t>(g_seq * g_head_dim));
    if (!g_rope_inv_freq.empty()) {
      const int64_t half = g_head_dim / 2;
      for (int64_t s = 0; s < g_seq; ++s) {
        float *cs = rope_cos.data() + s * g_head_dim;
        float *sn = rope_sin.data() + s * g_head_dim;
        for (int64_t j = 0; j < half; ++j) {
          const double ang =
              static_cast<double>(s) *
              static_cast<double>(g_rope_inv_freq[static_cast<size_t>(j)]);
          const float c = static_cast<float>(std::cos(ang));
          const float si = static_cast<float>(std::sin(ang));
          cs[j] = c;
          cs[half + j] = c;
          sn[j] = si;
          sn[half + j] = si;
        }
      }
    } else {
      npue::gemma_rope_tables(g_seq, g_head_dim, g_rope_theta,
                              rope_cos.data(), rope_sin.data());
    }
    rope_ready = true;
  }
  const int64_t half = g_head_dim / 2;
  const int64_t row_stride = 3 * g_hidden;
  const int64_t n_rows = batch * g_seq;
  float *__restrict p = qkv.data();
  const float *__restrict cos_p = rope_cos.data();
  const float *__restrict sin_p = rope_sin.data();

  auto rotate_pair = [half](float *v, const float *cs, const float *sn) {
    int64_t d = 0;
#if defined(__AVX2__)
    for (; d + 8 <= half; d += 8) {
      __m256 x1 = _mm256_loadu_ps(v + d);
      __m256 x2 = _mm256_loadu_ps(v + d + half);
      __m256 c = _mm256_loadu_ps(cs + d);
      __m256 s = _mm256_loadu_ps(sn + d);
      __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s));
      __m256 o2 = _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s));
      _mm256_storeu_ps(v + d, o1);
      _mm256_storeu_ps(v + d + half, o2);
    }
#endif
    for (; d < half; ++d) {
      const float x1 = v[d], x2 = v[d + half];
      v[d] = x1 * cs[d] - x2 * sn[d];
      v[d + half] = x2 * cs[d] + x1 * sn[d];
    }
  };

  pool_.run([&](int w, int nw) {
    for (int64_t row = w; row < n_rows; row += nw) {
      const int64_t s = row % g_seq;
      const float *cs = cos_p + s * g_head_dim;
      const float *sn = sin_p + s * g_head_dim;
      float *row_base = p + row * row_stride;
      for (int64_t h = 0; h < g_heads; ++h) {
        rotate_pair(row_base + h * g_head_dim, cs, sn);
        rotate_pair(row_base + g_hidden + h * g_head_dim, cs, sn);
      }
    }
  });
}

template <int NV>
void BertEncoder::qk_impl(const std::vector<float> &qkvbuf,
                          std::vector<float> &scores) {
  const int64_t pairs = batch * g_heads;
  const float *__restrict qkv_p = qkvbuf.data();
  float *__restrict sc_p = scores.data();
  pool_.run([&](int w, int nw) {
    for (int64_t p = w; p < pairs; p += nw) {
      const int64_t b = p / g_heads, h = p % g_heads;
      for (int64_t i = 0; i < g_seq; ++i) {
        const float *q = &qkv_p[(b * g_seq + i) * 3 * g_hidden + h * g_head_dim];
        // At `sc_cols`, not g_seq: see run() for the two widths. The columns
        // past g_seq are filled with -1.0e30 below rather than left stale, and
        // std::fill of an empty range when the two are equal costs nothing, so
        // the host-softmax run is byte for byte what it was.
        float *dst = &sc_p[(p * g_seq + i) * sc_cols];
#if defined(__AVX512F__)
        const int64_t nv = NV ? NV : g_head_dim / 8;
        const int64_t nz = nv / 2;
        const bool zt = (nv & 1) != 0;
        __m512 zq[(NV ? NV : kMaxHeadVecs) / 2 + 1];
        __m256 yq;
        for (int64_t v = 0; v < nz; ++v) zq[v] = _mm512_loadu_ps(q + v * 16);
        if (zt) yq = _mm256_loadu_ps(q + nz * 16);
#else
        __m256 qv[NV ? NV : kMaxHeadVecs];
        const int64_t nv = NV ? NV : g_head_dim / 8;
        for (int64_t v = 0; v < nv; ++v) qv[v] = _mm256_loadu_ps(q + v * 8);
#endif
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *k = &qkv_p[(b * g_seq + j) * 3 * g_hidden + g_hidden +
                                  h * g_head_dim];
#if defined(__AVX512F__)
          __m512 zs = _mm512_mul_ps(zq[0], _mm512_loadu_ps(k));
          for (int64_t v = 1; v < nz; ++v)
            zs = _mm512_fmadd_ps(zq[v], _mm512_loadu_ps(k + v * 16), zs);
          float acc = _mm512_reduce_add_ps(zs);
          if (zt) acc += hsum256(_mm256_mul_ps(yq,
                                               _mm256_loadu_ps(k + nz * 16)));
          dst[j] = acc;
#elif defined(__AVX2__)
          __m256 s = _mm256_mul_ps(qv[0], _mm256_loadu_ps(k));
          for (int64_t v = 1; v < nv; ++v)
            s = _mm256_fmadd_ps(qv[v], _mm256_loadu_ps(k + v * 8), s);
          dst[j] = hsum256(s);
#else
          float s = 0.f;
          for (int64_t d = 0; d < g_head_dim; ++d) s += q[d] * k[d];
          dst[j] = s;
#endif
        }
        // The padding column, only when the array's softmax is what will read
        // this row and it was compiled wider than the sequence.
        std::fill(dst + g_seq, dst + sc_cols, -1.0e30f);
      }
    }
  });
}

void BertEncoder::qk(const std::vector<float> &qkv, std::vector<float> &scores) {
  switch (g_head_dim) {
    case 32: qk_impl<4>(qkv, scores); break;
    case 64: qk_impl<8>(qkv, scores); break;
    default: qk_impl<0>(qkv, scores); break;
  }
}

template <int NV>
void BertEncoder::av_impl(const std::vector<float> &scores,
                          const std::vector<float> &qkvbuf,
                          std::vector<float> &ctx) {
  const int64_t pairs = batch * g_heads;
  const float *__restrict sc_p = scores.data();
  const float *__restrict qkv_p = qkvbuf.data();
  float *__restrict ctx_p = ctx.data();
  pool_.run([&](int w, int nw) {
    for (int64_t p = w; p < pairs; p += nw) {
      const int64_t b = p / g_heads, h = p % g_heads;
      for (int64_t i = 0; i < g_seq; ++i) {
        // At the score row's own stride -- the padded tail is not a key and is
        // not read; av() stops at g_seq regardless of how wide the row is.
        const float *a = &sc_p[(p * g_seq + i) * sc_cols];
        float *o = &ctx_p[(b * g_seq + i) * g_hidden + h * g_head_dim];
#if defined(__AVX2__)
#if defined(__AVX512F__)
        const int64_t nv = NV ? NV : g_head_dim / 8;
        const int64_t nz = nv / 2;
        __m512 zacc[(NV ? NV : kMaxHeadVecs) / 2 + 1];
        __m256 yacc{};
        for (int64_t v = 0; v < nz; ++v) zacc[v] = _mm512_setzero_ps();
        const bool tail = (nv & 1) != 0;
        if (tail) yacc = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          const __m512 zaj = _mm512_set1_ps(a[j]);
          for (int64_t k = 0; k < nz; ++k)
            zacc[k] = _mm512_fmadd_ps(zaj, _mm512_loadu_ps(v + k * 16),
                                      zacc[k]);
          if (tail)
            yacc = _mm256_fmadd_ps(_mm256_set1_ps(a[j]),
                                   _mm256_loadu_ps(v + nz * 16), yacc);
        }
        for (int64_t v = 0; v < nz; ++v)
          _mm512_storeu_ps(o + v * 16, zacc[v]);
        if (tail) _mm256_storeu_ps(o + nz * 16, yacc);
#else
        __m256 acc[NV ? NV : kMaxHeadVecs];
        const int64_t nv = NV ? NV : g_head_dim / 8;
        for (int64_t v = 0; v < nv; ++v) acc[v] = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          const __m256 aj = _mm256_set1_ps(a[j]);
          for (int64_t k = 0; k < nv; ++k)
            acc[k] = _mm256_fmadd_ps(aj, _mm256_loadu_ps(v + k * 8), acc[k]);
        }
        for (int64_t v = 0; v < nv; ++v)
          _mm256_storeu_ps(o + v * 8, acc[v]);
#endif
#else
        for (int64_t d = 0; d < g_head_dim; ++d) o[d] = 0.f;
        for (int64_t j = 0; j < g_seq; ++j) {
          const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                  2 * g_hidden + h * g_head_dim];
          for (int64_t d = 0; d < g_head_dim; ++d) o[d] += a[j] * v[d];
        }
#endif
      }
    }
  });
}

void BertEncoder::av(const std::vector<float> &scores,
                     const std::vector<float> &qkv, std::vector<float> &ctx) {
  switch (g_head_dim) {
    case 32: av_impl<4>(scores, qkv, ctx); break;
    case 64: av_impl<8>(scores, qkv, ctx); break;
    default: av_impl<0>(scores, qkv, ctx); break;
  }
}

// QK^T, softmax and softmax.V as two GEMMs on the set's attn_qk/attn_av
// streams, ONE CALL PER SEQUENCE.
//
// Per sequence, and that is forced by the hardware rather than chosen: attn_qk's
// B operand is ONE K panel of [head_dim, n_kv], shared by every query row the
// dispatch computes. A batched call would therefore hand every sequence the
// keys of whichever sequence's rows the panel was built from -- sequence 0's --
// and produce plausible scores for a model that never existed. There is no
// arrangement of one [head_dim, n_kv] panel that serves sixteen sequences, so
// the loop is the contract and not an inconvenience.
//
// The per-sequence mask rides along: set_additive_mask() is given this
// sequence's own row, so a padded position attends to nothing exactly as the
// host path's add_additive_mask() makes it.
void BertEncoder::attention_npu(const std::vector<float> &qkv,
                                std::vector<float> &ctx) {
  const int64_t stride = 3 * g_hidden;
  const int64_t row = stride * g_seq;
  const float *qkvp = qkv.data();
  const bool masked = add_mask.size() >= static_cast<size_t>(batch * g_seq);
  for (int64_t b = 0; b < batch; ++b) {
    const float *base = qkvp + b * row;
    // Scale 1.0: the packer folded 1/sqrt(head_dim) into Q's weight and bias
    // (the container records qk_scale_folded_into_q), which is why the host qk()
    // above applies no scale either. Passing a real 1/sqrt(head_dim) here would
    // fold it twice and compute a different model -- the failure
    // whisper::attention documents for the same flag.
    attn_->set_additive_mask(masked ? add_mask.data() + b * g_seq : nullptr);
    // kv = the row plus d_model, so K lands at 0 and V at d_model -- the layout
    // NpuAttention's panels read, and the same one whisper::attention's contract
    // states. BERT's qkv is Q|K|V with each part g_hidden wide.
    attn_->run(base, stride, base + g_hidden, stride, g_seq, g_seq, g_hidden,
               1.0f, ctx.data() + b * g_seq * g_hidden);
  }
  // QK^T, A*V and the dispatch count are TAKEN, not read: the class accumulates
  // across the layers while this method is called once per layer, so adding the
  // current value each time sums 1+2+...+N calls' worth. Measured: the status
  // block printed 293 ms of QK^T for 87 ms of attention that actually ran,
  // under a 192 ms attention total -- two sub-items of a block, larger than the
  // block, in the one display not allowed to be self-contradictory.
  const auto tu = attn_->take_timers();
  n_dispatch += static_cast<int>(tu.dispatch);
  t_qk += tu.qk;
  t_av += tu.av;
}

void BertEncoder::add_into(std::vector<float> &x, const std::vector<float> &y) {
  par(x.size(), [&](size_t lo, size_t hi) {
    size_t i = lo;
#if defined(__AVX2__)
    for (; i + 8 <= hi; i += 8)
      _mm256_storeu_ps(x.data() + i,
                       _mm256_add_ps(_mm256_loadu_ps(y.data() + i),
                                     _mm256_loadu_ps(residual.data() + i)));
#endif
    for (; i < hi; ++i) x[i] = y[i] + residual[i];
  });
}

void BertEncoder::add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
                                   size_t site, const float *ias_next, int8_t *dst,
                                   float *scale_next) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t H = g_hidden;
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float *row = x.data() + r * H;
      const float *yr = y.data() + r * H;
      float *res = residual.data() + r * H;
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= H; j += 8)
        _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
      for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / H;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / H;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, yv);
        _mm256_storeu_ps(res + j, yv);
      }
#else
      double sm = 0.0;
      for (j = 0; j < H; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / H);
      double vs = 0.0;
      for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
      const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
      for (j = 0; j < H; ++j) {
        row[j] = (row[j] - mean) * is * g[j] + b[j];
        res[j] = row[j];
      }
#endif
      if (!dst) continue;
      float mx = 0.f;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= H; j += 8)
          acc = _mm256_max_ps(acc, _mm256_and_ps(
              _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        mx = _mm_cvtss_f32(h);
      }
#endif
      for (; j < H; ++j) {
        const float a = std::fabs(row[j] * ias_next[j]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      scale_next[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      int8_t *q = dst + r * H;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 invv = _mm256_set1_ps(inv);
        const __m256 vhi = _mm256_set1_ps(127.0f);
        const __m256 vlo = _mm256_set1_ps(-127.0f);
        for (; j + 8 <= H; j += 8) {
          __m256 t = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps(row + j),
                            _mm256_loadu_ps(ias_next + j)), invv);
          t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
          __m256i i32 = _mm256_cvtps_epi32(t);
          __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                        _mm256_extracti128_si256(i32, 1));
          _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                           _mm_packs_epi16(p16, p16));
        }
      }
#endif
      for (; j < H; ++j) {
        float t = std::nearbyintf(row[j] * ias_next[j] * inv);
        if (t > 127.f) t = 127.f;
        if (t < -127.f) t = -127.f;
        q[j] = static_cast<int8_t>(t);
      }
    }
  });
  t_hostln += now_s() - t0;
}

void BertEncoder::add_norm_bf16(std::vector<float> &x,
                                 const std::vector<float> &y,
                                 size_t site, uint16_t *dst) {
  double t0 = now_s();
  const float *g = h_gamma[site], *b = h_beta[site];
  const int64_t H = g_hidden;
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float *row = x.data() + r * H;
      const float *yr = y.data() + r * H;
      float *res = residual.data() + r * H;
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= H; j += 8)
        _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
      for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
      __m256 s = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8)
        s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
      const float mean = hsum256(s) / H;
      const __m256 mv = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        v = _mm256_fmadd_ps(d, d, v);
      }
      const float var = hsum256(v) / H;
      const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
      for (j = 0; j + 8 <= H; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
        __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
        _mm256_storeu_ps(row + j, yv);
        _mm256_storeu_ps(res + j, yv);
      }
#else
      double sm = 0.0;
      for (j = 0; j < H; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / H);
      double vs = 0.0;
      for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
      const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
      for (j = 0; j < H; ++j) {
        row[j] = (row[j] - mean) * is * g[j] + b[j];
        res[j] = row[j];
      }
#endif
      if (!dst) continue;
      for (j = 0; j < H; ++j)
        dst[r * H + j] = to_bf16(row[j]);
    }
  });
  t_hostln += now_s() - t0;
}
BertEncoder::~BertEncoder() = default;

// Text-in entry point: tokenize, gather embeddings, run the stack, pool and
// L2-normalise -- the same path EmbedService::chunk walks in the runtime, kept
// here so the abstract Encoder interface is self-contained.
std::vector<float> BertEncoder::encode(const std::vector<std::string> &texts,
                                       const std::string &prefix,
                                       int64_t *tokens) {
  if (!tok_)
    tok_ = std::make_unique<app::AnyTokenizer>(
        app::load_tokenizer(model_, std::string()));
  const float *w_word = model_.raw("embeddings.word").as<float>();
  const float *w_pos = model_.raw("embeddings.position").as<float>();
  const float *w_typ = model_.raw("embeddings.token_type").as<float>();
  const int64_t n = static_cast<int64_t>(texts.size());
  if (batch <= 0)
    throw std::runtime_error(
        "BertEncoder::encode: no batch tier configured -- call use_tier() "
        "during setup before the text-in path");
  std::vector<float> out(static_cast<size_t>(n) * g_hidden, 0.f);
  int64_t base = 0;
  while (base < n) {
    const int64_t take = std::min<int64_t>(batch, n - base);
    const int64_t bt = use_tier(take);
    std::vector<float> buf(static_cast<size_t>(bt) * g_seq * g_hidden, 0.f);
    std::vector<float> cmask(static_cast<size_t>(bt) * g_seq, -1.0e30f);
    std::vector<float> cam(static_cast<size_t>(bt) * g_seq, 0.f);
    int64_t ntok = 0;
    for (int64_t b = 0; b < take; ++b) {
      const std::string &raw = texts[static_cast<size_t>(base + b)];
      const auto en = prefix.empty()
          ? tok_->encode(raw, static_cast<int>(g_seq))
          : tok_->encode(prefix + raw, static_cast<int>(g_seq));
      check_truncation(en.truncated, en.n_tokens_full,
                       static_cast<size_t>(base + b), g_seq);
      ntok += en.n_tokens;
      for (int64_t s = 0; s < g_seq; ++s) {
        const int32_t id = en.input_ids[static_cast<size_t>(s)];
        const float m = static_cast<float>(en.attention_mask[static_cast<size_t>(s)]);
        cam[static_cast<size_t>(b * g_seq + s)] = m;
        cmask[static_cast<size_t>(b * g_seq + s)] = m > 0 ? 0.f : -1.0e30f;
        float *dst = buf.data() + static_cast<size_t>((b * g_seq + s) * g_hidden);
        const float *wv = w_word + static_cast<size_t>(id) * g_hidden;
        const float *pv = w_pos + static_cast<size_t>(s) * g_hidden;
        for (int64_t c = 0; c < g_hidden; ++c)
          dst[c] = wv[c] + pv[c] + w_typ[c];
      }
    }
    add_mask = cmask;
    auto h = run(buf);
    pool_rows(h.data(), cam.data(), take, out.data() + base * g_hidden);
    if (tokens) *tokens += ntok;
    base += take;
  }
  return out;
}

std::vector<float> BertEncoder::run(const std::vector<float> &emb_in) {
  std::vector<float> x = emb_in;
  // Read ONCE for the whole pass: whether the four per-layer GEMMs dispatch is
  // a property of this run, not of the layer, and the four fused epilogues
  // below are each gated on it as well as on their own condition -- see the
  // note by gemm_on_array() in the header. With it false the arithmetic is the
  // add_into + layer_norm/gelu_cpu pair, which is the same computation with the
  // next A operand not written, and the host branch in gemm() does not read it.
  const bool gemm_on = gemm_on_array();
  layer_norm(x, s_ln[0]);

  qkvbuf.resize(rows * 3 * g_hidden);
  ctx.resize(rows * g_hidden);
  proj.resize(rows * g_hidden);
  up.resize(rows * (g_gated_ffn ? 2 : 1) * g_ffn);
  if (g_gated_ffn) gated.resize(rows * g_ffn);
  down.resize(rows * g_hidden);
  // The row stride of `scores`, and why it is not simply g_seq.
  //
  // The host's own qk_impl writes one score row per (sequence, head, query
  // position) and that row is g_seq wide -- the real key count, unpadded. The
  // softmax design is built for the PADDED key count (resolve.py sets sm_cols =
  // n_kv; 64 -> 384 for bge-base), because NpuAttention, the design's other
  // consumer, pads its score rows to the kernel's own width. So when the array
  // takes the softmax but NOT the attention, the two widths disagree -- and
  // elt_chunks does not catch it, because its only guard is `n % cols != 0` and
  // batch*heads*g_seq*g_seq is a multiple of 384 anyway. Measured on bge-base:
  // the kernel normalised 384-element windows, six real score rows at a time,
  // and the embedding came back at relfro 7.08e-01 / cos 0.749237895 against the
  // host. With `attn,softm` the same design is correct (relfro 1.9e-02), because
  // NpuAttention lays the rows out at the width it was built for.
  //
  // The pad is -1.0e30 and not zero, for the same reason NpuAttention pads the
  // way it does: exp(-1e30 - rowmax) underflows to0, so a padded column is a 0
  // term in the row's sum, and av() stops at g_seq so it never reads one. A zero
  // would be a real key attending to everything.
  sc_cols = g_seq;
  if (!host_sm) {
    const int64_t want = softmax_.info().cols;
    if (want < g_seq)
      throw std::runtime_error(
          "softmax/ was built with cols " + std::to_string(want) +
          " but this container's sequence is " + std::to_string(g_seq) +
          " wide. The design normalises whatever row width it was compiled for "
          "and cannot be handed a longer one, so the scores would be computed "
          "at the wrong length. Re-export it with sm_cols >= seq.");
    sc_cols = want;
  }
  scores.resize(batch * g_heads * g_seq * sc_cols);

  residual.resize(x.size());
  bool qkv_a_ready = false;
  for (int64_t L = 0; L < g_layers; ++L) {
    if (!qkv_a_ready)
      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

    gemm(qkv_, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden,
         i8w(ws_qkv, L), i8w(as_qkv, L), nullptr, /*a_ready=*/qkv_a_ready);
    qkv_a_ready = false;
    if (g_rope) apply_rope_qkv(qkvbuf);

    // Attention, on whichever path --npu-ops attn named. Both paths write the
    // same `ctx`, and the host one below is not a fallback for the array one:
    // it is the reference the array one is measured against.
    if (attn_) {
      double ta = now_s();
      attention_npu(qkvbuf, ctx);
      t_attn += now_s() - ta;
    } else {
      double ta = now_s();
      qk(qkvbuf, scores);
      t_attn += now_s() - ta; t_qk += now_s() - ta;

      if (host_sm) {
        softmax_cpu(scores);
      } else {
        add_additive_mask(scores);
        eltwise(softmax_, slots_sm, scores.data(), scores.size());
      }

      ta = now_s();
      av(scores, qkvbuf, ctx);
      t_attn += now_s() - ta; t_av += now_s() - ta;
    }

    gemm(attn_out_, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden,
         i8w(ws_ao, L), i8w(as_ao, L));
    const bool fuse_ln = gemm_on && fuse_ffn_epilogue && host_ln &&
                         ffn_up_.info().a_elem_bytes == 1;
    const bool fuse_ln_bf16 = gemm_on && fuse_ffn_epilogue && host_ln &&
                              ffn_up_.info().a_elem_bytes == 2;
    if (fuse_ln) {
      const float *asf = i8w(as_fu, L);
      if (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
          inv_smooth_src != asf) {
        inv_smooth.resize(static_cast<size_t>(g_hidden));
        for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asf[j];
        inv_smooth_src = asf;
      }
      a_scale.resize(static_cast<size_t>(rows));
      add_norm_quant(x, proj, s_ln[1 + 2 * L] - 1, inv_smooth.data(),
                     static_cast<int8_t *>(ffn_up_.slot_ptr(0, slot_a)),
                     a_scale.data());
    } else if (fuse_ln_bf16) {
      add_norm_bf16(x, proj, s_ln[1 + 2 * L] - 1,
                    static_cast<uint16_t *>(ffn_up_.slot_ptr(0, slot_a)));
    } else {
      add_into(x, proj);
      layer_norm(x, s_ln[1 + 2 * L]);
      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
    }
    FusedNext fn{i8w(as_fd, L),
                 static_cast<int8_t *>(ffn_down_.slot_ptr(0, slot_a)),
                 nullptr, g_gated_ffn};
    const bool fuse_ffn = gemm_on && fuse_ffn_epilogue && host_gelu &&
                          ffn_up_.info().a_elem_bytes == 1 &&
                          ffn_down_.info().a_elem_bytes == 1;
    if (fuse_ffn) {
      a_scale_next.resize(static_cast<size_t>(rows));
      fn.scale = a_scale_next.data();
    }
    FusedNextBf16 fn_bf16{
        static_cast<uint16_t *>(ffn_down_.slot_ptr(0, slot_a)), g_gated_ffn};
    const bool fuse_ffn_bf16 = gemm_on && fuse_ffn_epilogue && host_gelu &&
                               ffn_up_.info().a_elem_bytes == 2 &&
                               ffn_down_.info().a_elem_bytes == 2;
    gemm(ffn_up_, is_fu, x, s_fu[L], b_fu[L], up,
         g_gated_ffn ? 2 * g_ffn : g_ffn, i8w(ws_fu, L), i8w(as_fu, L),
         fuse_ffn ? &fn : nullptr, /*a_ready=*/fuse_ln || fuse_ln_bf16,
         fuse_ffn_bf16 ? &fn_bf16 : nullptr);

    if (fuse_ffn) {
      a_scale.swap(a_scale_next);
      gemm(ffn_down_, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
           g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
    } else if (fuse_ffn_bf16) {
      gemm(ffn_down_, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
           g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
    } else if (g_gated_ffn) {
      swiglu_cpu(up, gated);
      gemm(ffn_down_, is_fd, gated, s_fd[L], b_fd[L], down, g_hidden,
           i8w(ws_fd, L), i8w(as_fd, L));
    } else {
      if (host_gelu)
        gelu_cpu(up);
      else
        eltwise(gelu_, slots_gelu, up.data(), up.size());
      gemm(ffn_down_, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
           i8w(ws_fd, L), i8w(as_fd, L));
    }
    const bool last = (L + 1 == g_layers);
    if (gemm_on && fuse_ffn_epilogue && host_ln &&
        qkv_.info().a_elem_bytes == 1) {
      const float *asq = last ? nullptr : i8w(as_qkv, L + 1);
      if (asq && (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
                  inv_smooth_src != asq)) {
        inv_smooth.resize(static_cast<size_t>(g_hidden));
        for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asq[j];
        inv_smooth_src = asq;
      }
      if (asq) a_scale.resize(static_cast<size_t>(rows));
      add_norm_quant(x, down, s_ln[2 + 2 * L] - 1,
                     asq ? inv_smooth.data() : nullptr,
                     asq ? static_cast<int8_t *>(qkv_.slot_ptr(0, slot_a))
                         : nullptr,
                     asq ? a_scale.data() : nullptr);
      qkv_a_ready = !last;
    } else if (gemm_on && fuse_ffn_epilogue && host_ln &&
               qkv_.info().a_elem_bytes == 2) {
      add_norm_bf16(x, down, s_ln[2 + 2 * L] - 1,
                    last ? nullptr
                         : static_cast<uint16_t *>(qkv_.slot_ptr(0, slot_a)));
      qkv_a_ready = !last;
    } else {
      add_into(x, down);
      layer_norm(x, s_ln[2 + 2 * L]);
    }
  }
  return x;
}

}  // namespace npue
