//===- npu_ops.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- shared Whisper NPU layer. See whisper/npu_ops.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/npu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/app_state.hpp"   // gelu_erf_exact
#include "common/host_kernels.hpp"  // bf16_fill, from_bf16, now_s

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
  const std::string &want = d_.info().b_layout_hash;
  const std::string &got = model.info(name).layout_hash;
  if (want.empty())
    throw std::runtime_error(d_.info().name +
                             "/design.json has no b_layout_hash -- re-export "
                             "with tools/export_gemm_rtp.py");
  if (got.empty())
    throw std::runtime_error(name + ": .npue tensor carries no layout_hash -- "
                             "repack with tools/pack_npue.py");
  if (want != got)
    throw std::runtime_error(
        name + ": layout mismatch -- design " + d_.info().name + " wants " +
        want.substr(0, 16) + "..., file has " + got.substr(0, 16) +
        "... The bytes would be the right size and the wrong order.");
  const Span w = model.raw(name);
  return d_.stage(1, w.data, w.bytes);
}

void NpuGemm::run(size_t instr, const float *a, int64_t n_real, int64_t rows,
                  int64_t k, size_t wslot, const float *bias, int64_t n,
                  float *out) {
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

  const double t0 = app::now_s();
  if (ab != 2)
    throw std::runtime_error(d_.info().name + ": a_dtype is int8, which the "
                             "Whisper packer does not produce -- the container "
                             "and the design disagree about the operand type");
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
               float *scores, app::Pool &pool) {
  if (heads * head_dim != d_model)
    throw std::runtime_error("whisper attention: " + std::to_string(heads) +
                             " heads x " + std::to_string(head_dim) +
                             " is not d_model " + std::to_string(d_model));
  if (kv_stride < 2 * d_model)
    throw std::runtime_error("whisper attention: K|V row stride " +
                             std::to_string(kv_stride) + " cannot hold " +
                             std::to_string(2 * d_model) + " values");

  // The decoder's steps pass no scratch: one query row against up to 1500 keys
  // is heads * n_kv floats, which is worth allocating rather than making every
  // caller own a buffer it will not look at again.
  std::vector<float> local_scores;
  if (!scores) {
    local_scores.resize(static_cast<size_t>(n_q) * heads * n_kv);
    scores = local_scores.data();
  }

  pool.run([&](int w, int nw) {
    for (int64_t i = w; i < n_q; i += nw) {
      const float *qr = q + i * q_stride;
      for (int64_t h = 0; h < heads; ++h) {
        const float *qh = qr + h * head_dim;
        float *sc = scores + (i * heads + h) * n_kv;
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
        // Softmax with the maximum subtracted, the same reduction HF's own
        // attention does. There is no mask: the encoder attends over every
        // position it was given, and the decoder's cache holds only positions
        // at or before the current one, so causality is structural rather than
        // applied.
        float mx = sc[0];
        for (int64_t j = 1; j < n_kv; ++j) mx = std::max(mx, sc[j]);
        float sum = 0.f;
        for (int64_t j = 0; j < n_kv; ++j) {
          sc[j] = std::exp(sc[j] - mx);
          sum += sc[j];
        }
        const float inv = sum > 0.f ? 1.0f / sum : 0.f;
        // softmax @ V, into this head's slice of the output row.
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
      }
    }
  });
}

}  // namespace npue::whisper
