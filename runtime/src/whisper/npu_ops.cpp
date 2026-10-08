//===- npu_ops.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- shared Whisper NPU layer. See whisper/npu_ops.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/npu_ops.hpp"
#include "whisper/eltwise.hpp"   // NpuEltwise::softmax, for the array path

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/app_state.hpp"   // gelu_erf_exact
#include "common/conv_host.hpp"   // hostconv::gemm_nt, the host multiply
#include "common/host_b.hpp"      // hostb::host_b_kn, the untile
#include "common/host_kernels.hpp"  // bf16_fill, from_bf16, now_s
#include "common/int4_panel.hpp"     // gemm_b_panel
#include "common/npu_ops_flag.hpp"   // op_on_array -- array or host, one place

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace npue::whisper {

void NpuGemm::alloc_buffers() {
  slot_a = d_.stage_alloc(0, d_.info().buffer_bytes[0]);
  slot_c = d_.stage_alloc(2, d_.info().buffer_bytes[2]);
}

size_t NpuGemm::stage_operand(const npue::File &model,
                              const std::string &name) {
  // A HOST-LAYOUT CONTAINER (`--dtype f32`, gemm_layout "host"): row-major
  // F32 operands, no panel and no layout_hash, so there is nothing for the DMA
  // to read. The host half of this call needs only the TENSOR NAME (host_w_ is
  // a recipe, not bytes), so that is all that is written -- and the array half
  // is refused by name instead of falling through to a layout check whose
  // message ("repack with tools/pack/pack_npue.py") would blame the packer for
  // a container that was packed exactly as asked.
  //
  // Absent means PRETILED: the key is arch=1's alone in older containers and
  // only the spelled-out value flips the path.
  bool host_layout = false;
  try {
    host_layout = model.config_string("gemm_layout") == "host";
  } catch (const std::exception &) {
    host_layout = false;
  }
  if (host_layout) {
    if (app::op_on_array("gemm"))
      throw std::runtime_error(
          name + ": this container is `--dtype f32` (gemm_layout \"host\") -- "
          "its GEMM operands are row-major F32 for the CPU, and the MMAC "
          "multiplies bf16 and int8 only. --npu-ops gemm is refused rather "
          "than silently ignored; drop the flag, or repack with --dtype bf16/"
          "i8 for the array.");
    // stage() numbers slots from zero within this design's buffer, so a value
    // this far out is one it cannot produce; nothing binds it (the bind is the
    // array path, which has already refused above).
    static constexpr size_t kHostSlot = static_cast<size_t>(1) << 40;
    const size_t wslot = kHostSlot + host_w_.size();
    HostWeight hw;
    hw.model = &model;
    hw.name = name;
    host_w_[wslot] = std::move(hw);
    return wslot;
  }
  const std::string &want = d_.info().b_layout_hash;
  const std::string &got = model.info(name).layout_hash;
  if (want.empty())
    throw std::runtime_error(d_.info().name +
                             "/design.json has no b_layout_hash -- re-export "
                             "with tools/export/export_gemm_rtp.py");
  if (got.empty())
    throw std::runtime_error(name + ": .npue tensor carries no layout_hash -- "
                             "repack with tools/pack/pack_npue.py");
  if (want != got)
    throw std::runtime_error(
        name + ": layout mismatch -- design " + d_.info().name + " wants " +
        want.substr(0, 16) + "..., file has " + got.substr(0, 16) +
        "... The bytes would be the right size and the wrong order.");
  // bf16/I8 hand the mapping's own bytes to stage() with no copy; an I4
  // payload is widened to the int8 panel the array consumes first
  // (common/int4_panel.hpp). Whisper has no encoder- or decoder-local staging
  // of its own -- vit and both whisper paths funnel through here -- so this one
  // call site covers them.
  const Panel panel = gemm_b_panel(model, name, d_.info().a_elem_bytes);
  const size_t wslot = d_.stage(1, panel.bytes.data, panel.bytes.bytes);

  // The scales ride with the slot, not with the design: they are per OPERAND,
  // and one design carries up to seven of them (the decoder) or four (the
  // encoder). Reading them only on an int8 design is deliberate -- a bf16
  // container has no `.wscale` and `File::raw` throws on a missing tensor, so
  // asking unconditionally would break every bf16 pack.
  if (d_.info().a_elem_bytes == 1) {
    OpScale sc;
    sc.wscale = model.raw(name + ".wscale").as<float>();
    // asmooth is OPTIONAL and its absence is meaningful: `add_gemm_b_int8`
    // writes an all-ones vector when the calibration found nothing to smooth,
    // but a container packed by a scheme that has no smoothing at all carries
    // no such tensor. An empty inv_smooth is the same instruction either way --
    // no smoothing -- so the two cases do not have to be told apart here.
    if (model.has(name + ".asmooth")) {
      const TensorInfo &ti = model.info(name);
      const int64_t K = ti.logical_shape.empty() ? ti.padded_shape[0]
                                                 : ti.logical_shape[0];
      const float *as = model.raw(name + ".asmooth").as<float>();
      sc.inv_smooth.resize(static_cast<size_t>(K));
      for (int64_t j = 0; j < K; ++j)
        sc.inv_smooth[static_cast<size_t>(j)] =
            as[j] > 0.f ? 1.0f / as[j] : 1.0f;
    }
    scales_[wslot] = std::move(sc);
  }
  // The HOST half of the same record, for every dtype rather than only int8:
  // an fp32 copy of the operand is what a run that did not dispatch multiplies
  // with, and run() has no model pointer to ask. Built on demand there (see
  // NpuGemm::host_weight) -- only the recipe is written here, three words.
  {
    HostWeight hw;
    hw.model = &model;
    hw.name = name;
    host_w_[wslot] = std::move(hw);
  }
  return wslot;
}

const std::vector<float> &NpuGemm::host_weight(size_t wslot, int64_t k,
                                               int64_t n) {
  auto it = host_w_.find(wslot);
  if (it == host_w_.end())
    throw std::runtime_error(
        std::string(d_.info().name) + ": a host GEMM asked for B slot " +
        std::to_string(wslot) +
        ", which stage_operand() did not produce for this instance. A host run "
        "reads its weights from the CONTAINER rather than from the design, so "
        "it cannot proceed without the operand's name.");
  HostWeight &hw = it->second;
  if (hw.kn.empty()) {
    hw.kn = hostb::host_b_kn(*hw.model, hw.name);
    const TensorInfo &t = hw.model->info(hw.name);
    if (t.padded_shape[0] != k || t.padded_shape[1] != n)
      throw std::runtime_error(
          std::string(d_.info().name) + ": " + hw.name + " is [" +
          std::to_string(t.padded_shape[0]) + "," +
          std::to_string(t.padded_shape[1]) + "] but this GEMM multiplies at K=" +
          std::to_string(k) + ", N=" + std::to_string(n) +
          ". The host reads the operand's own shape rather than trusting the "
          "call site's, so the two disagreeing is a shape bug and not something "
          "to pad away.");
  }
  return hw.kn;
}

void NpuGemm::run_host(const float *a, int64_t n_real, int64_t k, size_t wslot,
                       const float *bias, int64_t n, float *out) {
  const double t0 = app::now_s();
  const std::vector<float> &w = host_weight(wslot, k, n);

  // SmoothQuant, once and only on this side: the packer pre-multiplied W by
  // asmooth, so the activation carries 1/asmooth and the product is the
  // unsmoothed one. Scaled into a scratch row-block rather than in place --
  // `a` belongs to the caller and to the next dispatch too.
  const bool i8 = d_.info().a_elem_bytes == 1;
  const float *inv_smooth = nullptr;
  if (i8) {
    const OpScale &sc = scale_for(wslot, "run_host");
    if (!sc.inv_smooth.empty()) inv_smooth = sc.inv_smooth.data();
  }
  const float *A = a;
  std::vector<float> scratch;
  if (inv_smooth) {
    scratch.assign(static_cast<size_t>(n_real) * static_cast<size_t>(k), 0.f);
    for (int64_t r = 0; r < n_real; ++r)
      for (int64_t c = 0; c < k; ++c)
        scratch[static_cast<size_t>(r) * k + c] = a[r * k + c] * inv_smooth[c];
    A = scratch.data();
  }

  hostconv::gemm_nt(A, n_real, w.data(), k, n, bias, out, pool_);
  ++n_host;
  t_host += app::now_s() - t0;
}

const NpuGemm::OpScale &NpuGemm::scale_for(size_t wslot,
                                           const char *call) const {
  const auto it = scales_.find(wslot);
  if (it == scales_.end())
    throw std::runtime_error(
        std::string(d_.info().name) + ": " + call + " dispatched against B "
        "slot " + std::to_string(wslot) + ", which stage_operand() did not "
        "produce for this design. An int8 GEMM has no other source for that "
        "operand's .wscale/.asmooth, so it cannot proceed without them.");
  return it->second;
}

void NpuGemm::run(size_t instr, const float *a, int64_t n_real, int64_t rows,
                  int64_t k, size_t wslot, const float *bias, int64_t n,
                  float *out, const char *code) {
  if (n_real <= 0 || n_real > rows)
    throw std::runtime_error(d_.info().name + ": " + std::to_string(n_real) +
                             " real rows into a " + std::to_string(rows) +
                             "-row dispatch");
  if (rows <= 0 || k <= 0 || n <= 0)
    throw std::runtime_error(d_.info().name + ": GEMM with a zero dimension");

  // ARRAY OR HOST, decided once and by the command line alone.
  //
  // It sits before the buffer-fit check below because that check is about the
  // DESIGN's A and C slots, and a host run touches neither: `out` is the
  // caller's own buffer at the width the caller asked for. Keeping the two
  // dimension checks above this line means a shape bug is still a shape bug on
  // both paths -- what is skipped is only the machinery of dispatching.
  if (!app::op_on_array(code ? code : op_code_)) {
    run_host(a, n_real, k, wslot, bias, n, out);
    return;
  }

  const size_t ab = d_.info().a_elem_bytes, cb = d_.info().c_elem_bytes;
  if (static_cast<size_t>(rows) * k * ab > d_.info().buffer_bytes[0] ||
      static_cast<size_t>(rows) * n * cb > d_.info().buffer_bytes[2])
    throw std::runtime_error(
        d_.info().name + ": " + std::to_string(rows) + "x" +
        std::to_string(k) + " A or " + std::to_string(rows) + "x" +
        std::to_string(n) + " C does not fit the design's buffers -- the "
        "design was exported for a different shape (seq " +
        std::to_string(d_.info().seq) + ", M " + std::to_string(d_.info().M) +
        "). Re-export it for this model.");

  const double t0 = app::now_s();
  if (ab == 1) {
    // The ONE branch on the datapath, here rather than at each of the eleven
    // call sites. Everything after it -- the bounds check, the dispatch window,
    // the row-scaled read-back -- is the int8 path's own, and it re-checks
    // `a_elem_bytes` so a direct run_i8() on a bf16 design is still refused.
    const OpScale &sc = scale_for(wslot, "run");
    run_i8(instr, a, n_real, rows, k, wslot, bias, n, out, sc.wscale,
           sc.inv_smooth.empty() ? nullptr : sc.inv_smooth.data());
    return;
  }
  auto *abuf = static_cast<uint16_t *>(d_.slot_ptr(0, slot_a));
  app::bf16_fill(abuf, a, static_cast<size_t>(n_real * k));
  // The padded tail, zeroed rather than left stale -- see the header.
  if (n_real < rows)
    std::memset(abuf + n_real * k, 0,
                static_cast<size_t>(rows - n_real) * k * sizeof(uint16_t));
  t_convert += app::now_s() - t0;

  {
    // bind -> sync-to -> dispatch -> sync-from, whole window under the lock.
    // The mutex is the same one the rest of the runtime takes: `active` on a
    // Design is shared mutable state, so two threads binding concurrently read a
    // torn binding and each other's buffers.
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d_.bind_instr(instr);
    d_.bind(0, slot_a);
    d_.bind(1, wslot);
    d_.bind(2, slot_c);
    // The A buffer was written through map(); without this the device reads
    // whatever the host cache last flushed, which looks like a design that
    // computed nothing.
    d_.sync_to_device(0, static_cast<size_t>(rows) * k * ab);
    d_.dispatch_only();
    d_.sync_from_device(d_.output_index(), static_cast<size_t>(rows) * n * cb);
  }
  t_dispatch += app::now_s() - t0;
  ++n_dispatch;

  const auto *c = static_cast<const uint16_t *>(
      d_.slot_ptr(d_.output_index(), slot_c));
  par_rows(n_real, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float *o = out + r * n;
      app::bf16_read(o, c + r * n, static_cast<size_t>(n));
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= n; j += 8)
        _mm256_storeu_ps(o + j, _mm256_add_ps(_mm256_loadu_ps(o + j),
                                              _mm256_loadu_ps(bias + j)));
#endif
      for (; j < n; ++j) o[j] += bias[j];
    }
  });
}

void NpuGemm::run_i8(size_t instr, const float *a, int64_t n_real,
                     int64_t rows, int64_t k, size_t wslot, const float *bias,
                     int64_t n, float *out, const float *wscale,
                     const float *inv_smooth) {
  const auto &in = d_.info();
  if (in.a_elem_bytes != 1)
    throw std::runtime_error(d_.info().name +
                             ": run_i8 on a design whose A operand is " +
                             std::to_string(in.a_elem_bytes * 8) +
                             " bits. The container and the design must agree, "
                             "and they do not.");
  if (n_real <= 0 || n_real > rows)
    throw std::runtime_error(d_.info().name + ": " + std::to_string(n_real) +
                             " rows into a " + std::to_string(rows) +
                             "-row dispatch");
  if (rows <= 0 || k <= 0 || n <= 0)
    throw std::runtime_error(d_.info().name + ": GEMM with a zero dimension");
  if (static_cast<size_t>(rows) * k > d_.info().buffer_bytes[0] ||
      static_cast<size_t>(rows) * n * in.c_elem_bytes >
          d_.info().buffer_bytes[d_.output_index()])
    throw std::runtime_error(
        d_.info().name + ": " + std::to_string(rows) + "x" +
        std::to_string(k) + " A or " + std::to_string(rows) + "x" +
        std::to_string(n) + " C does not fit the design's buffers. Re-export "
        "the design set with --int8 for this model.");
  if (!wscale)
    throw std::runtime_error(
        d_.info().name + ": an int8 GEMM with no per-output-channel weight "
        "scale. The container carries one for every int8 operand ("
        "tools/lib/gemm_i8.py), so a missing one means the operand was not packed "
        "as int8 while the design says it is.");

  const double t0 = app::now_s();
  a_scale_.assign(static_cast<size_t>(rows), 0.0f);
  auto *abuf = static_cast<int8_t *>(d_.slot_ptr(0, slot_a));
  // The helpers take a `par_rows(n, body)` callable and this class HAS one, so
  // the pool is the same in the int8 path as in the fp32 one. Passing a serial
  // stand-in instead would be correct and would put the whole A quantisation on
  // one core, which is the cost this datapath is supposed to avoid.
  auto par = [this](int64_t n, auto &&body) { par_rows(n, body); };
  // `inv_smooth` arrives ALREADY reciprocalled, from stage_operand's OpScale,
  // which holds it by value for the operand's whole life. There is no pointer
  // comparison and no per-dispatch K divisions here: the caller pays the
  // reciprocal once per operand, at stage time, and this reads it.
  app::quantise_a_int8(a, n_real, k, inv_smooth, abuf, a_scale_.data(), par);
  // The padded tail, zeroed exactly as the bf16 path above zeroes it. quantise_a
  // only sees the `n_real` rows the host actually has, and the design computes
  // every one of `rows`; leaving the rest holding the previous dispatch's
  // activations would put another call's tokens into this one's output.
  if (n_real < rows)
    std::memset(abuf + n_real * k, 0,
                static_cast<size_t>(rows - n_real) * k);
  t_convert += app::now_s() - t0;

  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d_.bind_instr(instr);
    d_.bind(0, slot_a);
    d_.bind(1, wslot);
    d_.bind(d_.output_index(), slot_c);
    d_.sync_to_device(0, static_cast<size_t>(rows) * k);
    d_.dispatch_only();
    d_.sync_from_device(d_.output_index(),
                        static_cast<size_t>(rows) * n * in.c_elem_bytes);
  }
  t_dispatch += app::now_s() - t0;
  ++n_dispatch;

  const void *c = d_.slot_ptr(d_.output_index(), slot_c);
  app::dequantise_c(c, in.c_elem_bytes, n_real, n, a_scale_.data(), wscale, bias,
               out, par);
}

void NpuGemm::run_accum(size_t instr, const float *a, int64_t n_real,
                        int64_t rows, int64_t k, size_t wslot, int64_t n,
                        float *acc) {
  if (n_real <= 0 || n_real > rows)
    throw std::runtime_error(d_.info().name + ": " + std::to_string(n_real) +
                             " real rows into a " + std::to_string(rows) +
                             "-row dispatch");
  if (rows <= 0 || k <= 0 || n <= 0)
    throw std::runtime_error(d_.info().name + ": GEMM with a zero dimension");
  const size_t ab = d_.info().a_elem_bytes, cb = d_.info().c_elem_bytes;
  if (static_cast<size_t>(rows) * k * ab > d_.info().buffer_bytes[0] ||
      static_cast<size_t>(rows) * n * cb > d_.info().buffer_bytes[2])
    throw std::runtime_error(
        d_.info().name + ": " + std::to_string(rows) + "x" +
        std::to_string(k) + " A or " + std::to_string(rows) + "x" +
        std::to_string(n) + " C does not fit the design's buffers -- the "
        "design was exported for a different shape (seq " +
        std::to_string(d_.info().seq) + ", M " + std::to_string(d_.info().M) +
        "). Re-export it for this model.");
  if (ab != 2)
    throw std::runtime_error(
        d_.info().name + ": run_accum is the CONVOLUTION's accumulate form and "
        "its B panel is built on the host from the container's F32 conv weights "
        "through a bf16 tiler (NpuConv1d::stage), which cannot fill an int8 "
        "design's I8 panel with its own MAC sub-tile. This design is int8, so "
        "the front end must run on the host -- which is what stt_mode.hpp does "
        "for an int8 design set. Reaching this means that decision was "
        "bypassed.");
  // Checked BEFORE the dispatch, not after reading C: a design that emits fp32
  // would already have run by the time a wrong-width read refused.
  if (cb != 2)
    throw std::runtime_error(d_.info().name + ": run_accum reads C as bf16; this "
                             "design emits " + std::to_string(cb * 8) +
                             "-bit C. The encoder's own run() handles fp32 C, "
                             "and the convolution needs the accumulate path, so "
                             "re-export with --c-bf16.");

  const double t0 = app::now_s();
  auto *abuf = static_cast<uint16_t *>(d_.slot_ptr(0, slot_a));
  app::bf16_fill(abuf, a, static_cast<size_t>(n_real * k));
  if (n_real < rows)
    std::memset(abuf + n_real * k, 0,
                static_cast<size_t>(rows - n_real) * k * sizeof(uint16_t));
  t_convert += app::now_s() - t0;

  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    d_.bind_instr(instr);
    d_.bind(0, slot_a);
    d_.bind(1, wslot);
    d_.bind(2, slot_c);
    d_.sync_to_device(0, static_cast<size_t>(rows) * k * ab);
    d_.dispatch_only();
    d_.sync_from_device(d_.output_index(), static_cast<size_t>(rows) * n * cb);
  }
  t_dispatch += app::now_s() - t0;
  ++n_dispatch;

  const auto *c = static_cast<const uint16_t *>(
      d_.slot_ptr(d_.output_index(), slot_c));
  // The scratch row is allocated INSIDE the pool lambda, so each worker has its
  // own: a single buffer shared by the workers is a race that returns a
  // different answer on every run, and only on a host with more than one core.
  // bf16_read is reused rather than a second bf16 unpack written here, because
  // two unpack paths in one file is one more thing to keep bit-identical.
  pool_.run([&](int w, int nw) {
    std::vector<float> tmp(static_cast<size_t>(n));
    for (int64_t r = w; r < n_real; r += nw) {
      // += , not =: this is the accumulate form, and acc already holds the
      // previous K block's contribution to this row.
      app::bf16_read(tmp.data(), c + r * n, static_cast<size_t>(n));
      float *o = acc + r * n;
      int64_t j = 0;
#if defined(__AVX2__)
      for (; j + 8 <= n; j += 8)
        _mm256_storeu_ps(o + j,
                         _mm256_add_ps(_mm256_loadu_ps(o + j),
                                       _mm256_loadu_ps(tmp.data() + j)));
#endif
      for (; j < n; ++j) o[j] += tmp[static_cast<size_t>(j)];
    }
  });
}

void layernorm_rows(float *x, int64_t n_rows, int64_t d, const float *gamma,
                    const float *beta, double eps, app::Pool &pool) {
  pool.run([&](int w, int nw) {
    for (int64_t r = w; r < n_rows; r += nw) {
      float *row = x + r * d;
      double sm = 0.0;
      for (int64_t j = 0; j < d; ++j) sm += row[j];
      const float mean = static_cast<float>(sm / d);
      double sv = 0.0;
      for (int64_t j = 0; j < d; ++j) {
        const double dd = static_cast<double>(row[j]) - mean;
        sv += dd * dd;
      }
      const float var = static_cast<float>(sv / d);
      const float is = 1.0f / std::sqrt(var + static_cast<float>(eps));
      for (int64_t j = 0; j < d; ++j)
        row[j] = (row[j] - mean) * is * gamma[j] + beta[j];
    }
  });
}

void gelu_erf_inplace(float *x, size_t n, app::Pool &pool) {
  pool.run([&](int w, int nw) {
    for (size_t i = static_cast<size_t>(w); i < n;
         i += static_cast<size_t>(nw))
      x[i] = app::gelu_erf_exact(x[i]);
  });
}

void attention(const float *q, int64_t q_stride, const float *kv,
               int64_t kv_stride, int64_t n_q, int64_t n_kv, int64_t d_model,
               int64_t heads, int64_t head_dim, float scale, float *out,
               float *scores, app::Pool &pool, NpuEltwise *softm,
               int64_t score_stride) {
  if (heads * head_dim != d_model)
    throw std::runtime_error("whisper attention: " + std::to_string(heads) +
                             " heads x " + std::to_string(head_dim) +
                             " is not d_model " + std::to_string(d_model));
  if (kv_stride < 2 * d_model)
    throw std::runtime_error("whisper attention: K|V row stride " +
                             std::to_string(kv_stride) + " cannot hold " +
                             std::to_string(2 * d_model) + " values");
  // A stride narrower than the keys each row carries overlaps the next row, and
  // the score row is exactly the buffer that decides whether an array softmax
  // computed the right denominator. Checked rather than assumed: the caller
  // passes this number, and a wrong one reads as plausible garbage.
  if (score_stride <= 0) score_stride = n_kv;
  if (score_stride < n_kv)
    throw std::runtime_error("whisper attention: score row stride " +
                             std::to_string(score_stride) + " is narrower than " +
                             std::to_string(n_kv) + " keys");
  const int64_t st = score_stride;
  // The kernel reduces over its WHOLE row, so a row wider than the design's
  // own width would run off the end of the buffer and a row narrower than it
  // would softmax columns that are not there. Equal is the only answer.
  if (softm && st != softm->cols())
    throw std::runtime_error(
        "whisper attention: score rows are " + std::to_string(st) +
        " wide and the softmax design was built for " +
        std::to_string(softm->cols()) + ". Re-export the softmax design at this "
        "model's n_kv -- tools/export/export_gemm_rtp.py takes it from the "
        "target's own window, not from a flag.");

  // The decoder's steps pass no scratch: one query row against up to 1500 keys
  // is heads * score_stride floats, which is worth allocating rather than
  // making every caller own a buffer it will not look at again. Every caller
  // that DOES pass scratch sized it at n_kv rows, which is why score_stride
  // defaults to n_kv -- see the header.
  std::vector<float> local_scores;
  if (!scores) {
    local_scores.resize(static_cast<size_t>(n_q) * heads * st);
    scores = local_scores.data();
  }

  // The three parts of one attention row, named once and used in both shapes
  // below.
  //
  // Splitting them out is what keeps the default path exactly what it was: the
  // same three operations, in the same order, over the same rows, in the same
  // vectorized loops. A caller passing no softmax operator gets the arithmetic
  // this function has always done, so Whisper's three call sites -- and every
  // number measured from them -- cannot move because a parameter was appended
  // to a signature.
  const auto qk_row = [&](int64_t i, int64_t h, float *sc) {
    const float *qh = q + i * q_stride + h * head_dim;
    // q @ K^T, one row of scores at a time. head_dim is a multiple of 8 on
    // every shipped size (64 for all of them), but the tail loop is scalar
    // so a head_dim that is not still computes the right answer.
    for (int64_t j = 0; j < n_kv; ++j) {
      const float *kj = kv + j * kv_stride + h * head_dim;
      float acc = 0.f;
      int64_t t = 0;
#if defined(__AVX2__)
      __m256 a = _mm256_setzero_ps();
      for (; t + 8 <= head_dim; t += 8)
        a = _mm256_fmadd_ps(_mm256_loadu_ps(qh + t), _mm256_loadu_ps(kj + t), a);
      acc = app::hsum256(a);
#endif
      for (; t < head_dim; ++t) acc += qh[t] * kj[t];
      sc[j] = acc * scale;
    }
  };
  // Softmax with the maximum subtracted, the same reduction HF's own attention
  // does. There is no mask: the encoder attends over every position it was
  // given, and the decoder's cache holds only positions at or before the
  // current one, so causality is structural rather than applied. The loop runs
  // to n_kv and not to st, so a padded row stride is invisible here.
  // Returns the reciprocal sum rather than folding it into the row: the caller
  // applies it inside its own walk over the row, which is the shape this
  // function has always had and the shape that reads each score exactly once.
  // Folding it in here would add a full pass over a buffer that is 13.5 MB for
  // whisper-tiny and 54 MB for large-v3, and a second read of every score for
  // no change in any value is still a change to what Whisper's numbers measure.
  const auto softmax_row = [&](float *sc) -> float {
    float mx = sc[0];
    for (int64_t j = 1; j < n_kv; ++j) mx = std::max(mx, sc[j]);
    float sum = 0.f;
    for (int64_t j = 0; j < n_kv; ++j) {
      sc[j] = std::exp(sc[j] - mx);
      sum += sc[j];
    }
    return sum > 0.f ? 1.0f / sum : 0.f;
  };
  const auto av_row = [&](int64_t i, int64_t h, const float *sc, float inv) {
    // softmax @ V, into this head's slice of the output row. `inv` is 1.0f on
    // the array path -- the design already normalized the row -- and a multiply
    // by one is exact, so the two paths differ in no bit of it.
    float *o = out + i * d_model + h * head_dim;
    std::fill(o, o + head_dim, 0.f);
    for (int64_t j = 0; j < n_kv; ++j) {
      const float a = sc[j] * inv;
      const float *vj = kv + j * kv_stride + d_model + h * head_dim;
      int64_t t = 0;
#if defined(__AVX2__)
      const __m256 av = _mm256_set1_ps(a);
      for (; t + 8 <= head_dim; t += 8)
        _mm256_storeu_ps(o + t, _mm256_fmadd_ps(av, _mm256_loadu_ps(vj + t),
                                                _mm256_loadu_ps(o + t)));
#endif
      for (; t < head_dim; ++t) o[t] += a * vj[t];
    }
  };

  if (!softm) {
    // One pass, three operations per row, in place -- what this function has
    // always done, and the path every existing caller takes.
    pool.run([&](int w, int nw) {
      for (int64_t i = w; i < n_q; i += nw)
        for (int64_t h = 0; h < heads; ++h) {
          float *sc = scores + (i * heads + h) * st;
          qk_row(i, h, sc);
          av_row(i, h, sc, softmax_row(sc));
        }
    });
    return;
  }

  // -- three phases, with the softmax in the middle and on the array --------
  //
  // This is the only shape in which `softm` can mean anything for a model
  // whose softmax has no pass of its own: the softmax is INSIDE the attention,
  // so the attention has to step aside between its two GEMMs. What does not
  // move is those two GEMMs -- they stay on the host, because `softm` named one
  // op and moving QK^T and softmax.V is the `attn` code.
  //
  // Phase 1: QK^T, still parallel, and the score rows PADDED to the design's
  // width so that the kernel's whole row is a real row. The columns past n_kv
  // are -1e30 rather than zero for the reason that decides correctness: the
  // kernel normalizes over every column it is given, so a padded column left at
  // zero would add exp(0 - max) to the denominator and shrink every weight in
  // the row. exp(-1e30 - max) is zero, so the real weights are untouched.
  pool.run([&](int w, int nw) {
    for (int64_t i = w; i < n_q; i += nw)
      for (int64_t h = 0; h < heads; ++h) {
        float *sc = scores + (i * heads + h) * st;
        qk_row(i, h, sc);
        for (int64_t j = n_kv; j < st; ++j) sc[j] = -1.0e30f;
      }
  });
  // Phase 2: the softmax itself, serially, over every row at once.
  //
  // It cannot be inside the parallel pass above: the design has ONE input and
  // ONE output buffer, so two workers dispatching into it would hand each
  // other's rows back -- the same failure the eltwise slots comment describes.
  // NpuEltwise::softmax walks the whole range in chunks of the design's own row
  // capacity, so any row count fits and the dispatch count is that count
  // divided by the capacity, not the count itself.
  softm->softmax(scores, n_q * heads);
  // Phase 3: softmax.V, parallel again, reading the first n_kv columns of each
  // row. The padded tail was never a key; it is not read.
  pool.run([&](int w, int nw) {
    for (int64_t i = w; i < n_q; i += nw)
      for (int64_t h = 0; h < heads; ++h)
        av_row(i, h, scores + (i * heads + h) * st, 1.0f);
  });
}

std::vector<uint16_t> tile_b_panel(const float *mat, int64_t K, int64_t N,
                                  int64_t tile_k, int64_t tile_n, int64_t mac_s,
                                  int64_t mac_t) {
  if (tile_k <= 0 || tile_n <= 0 || mac_s <= 0 || mac_t <= 0)
    throw std::runtime_error(
        "tile_b_panel: the layout has a zero extent (tile " +
        std::to_string(tile_k) + "x" + std::to_string(tile_n) + ", mac " +
        std::to_string(mac_s) + "x" + std::to_string(mac_t) +
        ") -- read it from the design's b_layout, not from a default");
  if (K % tile_k || N % tile_n)
    throw std::runtime_error(
        "tile_b_panel: operand [" + std::to_string(K) + "," +
        std::to_string(N) + "] does not tile evenly by (" +
        std::to_string(tile_k) + "," + std::to_string(tile_n) +
        "): K%tile_k=" + std::to_string(K % tile_k) +
        ", N%tile_n=" + std::to_string(N % tile_n));
  if (tile_k % mac_s || tile_n % mac_t)
    throw std::runtime_error(
        "tile_b_panel: tile " + std::to_string(tile_k) + "x" +
        std::to_string(tile_n) + " does not split into the MMAC sub-tile " +
        std::to_string(mac_s) + "x" + std::to_string(mac_t));

  // The order, spelled out: k block, then n block, then the sub-tile's k step,
  // then its n step, then s, then t. Same traversal as tools/lib/npue.py's
  // tile_b(order="k,n") and the C++ packer's tile_b, and the byte order inside
  // the panel is the MMAC's (s fastest after t), which is why mac_s/mac_t have
  // to be the design's and not a constant.
  const int64_t kb_n = K / tile_k, nb_n = N / tile_n;
  std::vector<uint16_t> out(static_cast<size_t>(K) * N);
  size_t w = 0;
  for (int64_t kb = 0; kb < kb_n; ++kb)
    for (int64_t nb = 0; nb < nb_n; ++nb)
      for (int64_t si = 0; si < tile_k / mac_s; ++si)
        for (int64_t ti = 0; ti < tile_n / mac_t; ++ti)
          for (int64_t s = 0; s < mac_s; ++s)
            for (int64_t t = 0; t < mac_t; ++t) {
              const int64_t r = kb * tile_k + si * mac_s + s;
              const int64_t c = nb * tile_n + ti * mac_t + t;
              out[w++] = app::to_bf16(mat[r * N + c]);
            }
  return out;
}

}  // namespace npue::whisper
