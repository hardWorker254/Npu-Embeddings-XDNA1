//===- host_kernels.hpp --------------------------------------*- C++ -*-===//
//
// Host-side numerics shared by every encoder, split out of main.cpp (code
// verbatim): bf16 conversion, int8 quantisation/dequantisation, GELU, the
// small AVX2 reductions, timing, and the fixture readers.
//
// THE AVX2 PATHS ARE *NOT* ALL BIT-IDENTICAL TO THEIR SCALAR FALLBACKS, and
// this line used to claim they were. Measured, by tools/verify/verify_i8_kernels.py,
// which builds one probe three ways from this header and diffs the bytes:
//
//   quantise_a_int8            bit-identical. The vector head clips in float
//                               and rounds once, the scalar tail rounds then
//                               clips, and for a value already inside +/-127
//                               the two orders give the same integer.
//   dequantise_c               NOT bit-identical. 112 of 448 elements (25%)
//                               differ, every one of them by exactly 1 ULP
//                               (max relative 1.19e-07). The vector path
//                               contracts `cf*sa*wscale + bias` into an FMA,
//                               which rounds the final add once; the scalar
//                               tail at the bottom of the loop rounds it
//                               twice. Both were reproduced exactly: AVX2
//                               matches round-then-FMA 448/448, scalar matches
//                               round-round-round 448/448.
//   dequant_act_quant          same cause, and it PROPAGATES: its sa_next is
//                               an absmax over the dequantised values, so a
//                               1-ULP difference upstream moves the scale and
//                               therefore the int8 payload of the NEXT
//                               operand. An int8 embedding is not bit-
//                               reproducible across host CPUs.
//
// `dequantise_c`'s `bias` IS NOT NULLABLE, and that is not an oversight left
// for the reader: `_mm256_loadu_ps(bias + j)` in the vector path and `bias[j]`
// in the scalar tail both fault on a null, so a caller that passes one gets a
// SIGSEGV rather than a refusal. Every call site in the tree loads the bias out
// of the container (`name + ".bias"`, and File::raw throws when it is absent),
// so it is unreachable today -- but it was reached once while writing
// runtime/tests/test_vit_model.cpp, which is why it is written down. The
// accumulate form that genuinely has no bias is NpuGemm::run_accum, a
// different function.
//
// That is a reproducibility statement, not an accuracy one: 1 ULP of fp32 is
// four orders below the 2e-03 gate. It matters because a golden captured on an
// AVX2 host cannot be compared bit-for-bit against a non-AVX2 one, and because
// an A/B between two builds is confounded by which host ran it. If you want
// the two paths to agree, the fix is `std::fma` in the scalar tails -- a
// numerics change to the fallback, deliberately not made here.
//
// The alignment precondition the vector paths rely on is stated, and checked,
// at the top of this file. It is a crash, not a wrong number, and it only
// crashes on the hosts that have the fast path.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace app {

// THE STREAMING-LOAD PRECONDITION, stated once and checked at every entry point
// that needs it.
//
// `_mm256_stream_load_si256` is `vmovntdqa`, which requires a 32-byte aligned
// address and FAULTS otherwise -- it is not a slow path, it is a SIGSEGV. The
// scalar fallbacks load element by element and do not care, so the same call
// crashes on a host with AVX2 and returns the right answer on one without. A
// bug that only reproduces on the machines that have the fast path is the worst
// shape a bug can have, so the condition is checked rather than assumed.
//
// TWO conditions, not one, and the second is the one that is easy to miss:
//
//   1. the base pointer is 32-byte aligned, and
//   2. the ROW STRIDE `N * c_bytes` is a multiple of 32 -- because the kernels
//      address row r at `c + r * N`, so every row has to land on a 32-byte
//      boundary too. Equivalently N % 8 == 0 for an int32 C and N % 16 == 0
//      for a bf16 C.
//
// MEASURED, not reasoned: a C buffer skewed by one int32 faults under AVX2
// and returns a correct result without it; and N=53 with an int32 C gives a
// row stride of 212 bytes, so row 1 starts 20 bytes into a 32-byte block and
// faults on the very first row past row 0. tools/verify/verify_i8_kernels.py reproduces
// both. See the ASan backtrace in that gate's header comment.
//
// Cost: two integer operations per GEMM, not per row. Every current call site
// passes an XRT host BO slot straight through (page-aligned, and every N in this
// project is a multiple of 16), so this never fires today -- which is exactly
// why it needs to be a refusal rather than a comment.
inline void require_stream_aligned(const void *c, int64_t N, size_t c_bytes,
                                   const char *who) {
#if defined(__AVX2__)
  const uintptr_t base = reinterpret_cast<uintptr_t>(c);
  const uintptr_t stride = static_cast<uintptr_t>(N) * c_bytes;
  if ((base & 31u) || (stride & 31u))
    throw std::runtime_error(
        std::string(who) + ": C is not stream-load addressable -- base " +
        std::to_string(base) + " is " + std::to_string(base & 31u) +
        " mod 32 bytes, row stride N=" + std::to_string(N) + " * " +
        std::to_string(c_bytes) + " = " + std::to_string(stride) + " is " +
        std::to_string(stride & 31u) + " mod 32. Both must be 0 (N % 8 for an "
        "int32 C, N % 16 for a bf16 C). The AVX2 path faults on this and the "
        "scalar path does not, so it would only ever reproduce on the hosts "
        "that have the fast path.");
#else
  (void)c; (void)N; (void)c_bytes; (void)who;
#endif
}

inline std::vector<float> read_f32(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes != count * sizeof(float))
    throw std::runtime_error(path + ": expected " +
                             std::to_string(count * sizeof(float)) +
                             " bytes, found " + std::to_string(bytes));
  f.seekg(0);
  std::vector<float> v(count);
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  return v;
}

inline std::vector<int32_t> read_i32(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0);
  std::vector<int32_t> v(count);
  f.read(reinterpret_cast<char *>(v.data()), count * sizeof(int32_t));
  return v;
}

// fp32 -> bf16, round-to-nearest-even. The rounding tools/lib/npue.py uses when
// packing; truncation would bias every value toward zero.
inline uint16_t to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// The vectorised forms below are BIT-IDENTICAL to the scalar ones above, which
// is the only reason they are safe to swap in: every integer op used has the
// same semantics on uint32 as on __m256i lanes, and after the >> 16 the values
// are in [0, 65535] so packus never actually saturates. The scalar tail keeps
// the two paths agreeing on any n.
//
// 13.8 M elements per encode go through these (tasks/0024), which is why they
// are worth writing out.
#if defined(__AVX2__)

inline void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kone = _mm256_set1_epi32(1);
  auto rne = [&](__m256i u) {
    __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone);
    return _mm256_srli_epi32(
        _mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
  };
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i a = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i)));
    __m256i b = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i + 8)));
    // packus interleaves the two 128-bit lanes; 0xD8 puts them back in order.
    __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), p);
  }
  for (; i < n; ++i) d[i] = to_bf16(src[i]);
}

inline void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i));
    __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(u));
  }
  for (; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#else
inline void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < n; ++i) d[i] = to_bf16(src[i]);
}
inline void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  for (size_t i = 0; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#endif

inline double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A double that is usually tiny, for a diagnostic.
//
// std::to_string is the wrong function for this and was the wrong function
// here: it formats with six DECIMAL places, so a layer_norm_eps of 1e-12 --
// what vit-base-patch16-224's config.json actually says -- prints as
// "0.000000". That is the wrong number in precisely the case the value is worth
// printing for, because an epsilon-mismatch refusal quotes both sides so a
// reader can see which one is off. %.6g prints 1e-12 as "1e-12" and 1e-05 as
// "1e-05", and a value like 0.001 as "0.001".
inline std::string eps_text(double eps) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.6g", eps);
  return buf;
}

// ---------------------------------------------------------------------------
// int8 quantisation, shared by BOTH encoders.
//
// These were inline in Encoder::gemm() until tasks/0081 gave arch=1 an int8
// path too. GemmaNpuEncoder is a separate encoder (RMSNorm x4, MQA, per-layer
// RoPE, GeGLU), and copying ~120 lines of hand-vectorised quantisation into it
// would have created exactly the kind of duplicate that drifts: the reciprocal
// hoist below was a 63 ms fix found once (tasks/0080), and a second copy would
// not have it.
// ---------------------------------------------------------------------------

// A -> int8, per row, with the SmoothQuant divisor folded into the same pass
// (tasks/0078). `ias` is 1/asmooth, reciprocated ONCE by the caller: dividing
// by asmooth[j] in both the max pass and the quantise pass was two divisions
// per element and cost 74 ms of the 126 ms the array had saved.
//
// The divisor is NOT folded into the preceding norm. BERT is post-LN, so the
// norm's output feeds the residual as well as this GEMM (tasks/0078 4a); on
// Gemma the same holds for a different reason -- `pre_feedforward_layernorm`'s
// output is consumed by the GeGLU pair only, but `input_layernorm`'s feeds the
// residual, and one code path is worth more than one folded multiply.
template <typename ParRows>
void quantise_a_int8(const float *a, int64_t rows, int64_t K, const float *ias,
                     int8_t *q_base, float *a_scale, ParRows par_rows) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *x = a + r * K;
      int8_t *q = q_base + r * K;
      float mx = 0.f;
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
      __m256 acc = _mm256_setzero_ps();
      for (; j + 8 <= K; j += 8)
        acc = _mm256_max_ps(acc, _mm256_and_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            absmask));
      __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                            _mm256_extractf128_ps(acc, 1));
      h = _mm_max_ps(h, _mm_movehl_ps(h, h));
      h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
      mx = _mm_cvtss_f32(h);
#endif
      for (; j < K; ++j) {
        const float v = std::fabs(x[j] * ias[j]);
        if (v > mx) mx = v;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      a_scale[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      j = 0;
#if defined(__AVX2__)
      const __m256 invv = _mm256_set1_ps(inv);
      const __m256 hi = _mm256_set1_ps(127.0f);
      const __m256 lo = _mm256_set1_ps(-127.0f);
      for (; j + 8 <= K; j += 8) {
        __m256 v = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            invv);
        v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);
        // cvtps_epi32 rounds per MXCSR, i.e. nearest-even by default -- the
        // same rule tools/pack/pack_npue.py's np.rint uses on the weights, so the
        // two halves of the product round the same way.
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                      _mm256_extracti128_si256(i32, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                         _mm_packs_epi16(p16, p16));
      }
#endif
      for (; j < K; ++j) {
        float v = std::nearbyintf(x[j] * ias[j] * inv);
        if (v > 127.f) v = 127.f;
        if (v < -127.f) v = -127.f;
        q[j] = static_cast<int8_t>(v);
      }
    }
  });
}

// GELU, the same degree-8 minimax polynomial gelu_cpu() uses, so the fused and
// unfused paths are bit-identical rather than merely close.
#if defined(__AVX2__)
inline __m256 gelu8(__m256 v) {
  const __m256 u = _mm256_min_ps(
      _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v), _mm256_set1_ps(4.0f));
  __m256 pl = _mm256_fmadd_ps(_mm256_set1_ps(-7.2340282171e-05f), u,
                              _mm256_set1_ps(1.8179518005e-03f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.7707383379e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(8.4577147641e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.9228671834e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(9.8431124458e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(3.6137852062e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-4.9454128936e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.3007010117e-04f));
  return _mm256_add_ps(_mm256_max_ps(v, _mm256_setzero_ps()), pl);
}
#endif
inline float gelu8(float v) {
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
  return std::max(v, 0.0f) + pl;
}

#if defined(__AVX2__)
// Vectorised 2^x, shared by the BERT softmax/SwiGLU and the Gemma tanh GeGLU.
// Same degree-7 polynomial on the fractional part the scalar std::exp2 path
// approximates, so both encoders round identically.
inline __m256 exp2_avx2(__m256 x) {
  const __m256 c0 = _mm256_set1_ps(1.5483275463e-05f);
  const __m256 c1 = _mm256_set1_ps(1.5669833174e-04f);
  const __m256 c2 = _mm256_set1_ps(1.3331825236e-03f);
  const __m256 c3 = _mm256_set1_ps(9.6164605538e-03f);
  const __m256 c4 = _mm256_set1_ps(5.5504156855e-02f);
  const __m256 c5 = _mm256_set1_ps(2.4022684109e-01f);
  const __m256 c6 = _mm256_set1_ps(6.9314717694e-01f);
  const __m256 c7 = _mm256_set1_ps(9.9999998955e-01f);
  __m256i k = _mm256_cvttps_epi32(x);
  __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(k));
  __m256 pl = _mm256_fmadd_ps(c0, f, c1);
  pl = _mm256_fmadd_ps(pl, f, c2);
  pl = _mm256_fmadd_ps(pl, f, c3);
  pl = _mm256_fmadd_ps(pl, f, c4);
  pl = _mm256_fmadd_ps(pl, f, c5);
  pl = _mm256_fmadd_ps(pl, f, c6);
  pl = _mm256_fmadd_ps(pl, f, c7);
  __m256i bits = _mm256_slli_epi32(
      _mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(pl, _mm256_castsi256_ps(bits));
}
#endif

// THE ffn_up EPILOGUE IN ONE PASS (tasks/0081 T37).
//
// The unfused chain walks the widest tensor in the model six times: the
// dequantiser writes fp32 `up`, GELU reads and writes it, and the next GEMM's
// quantiser reads it again and writes int8. At bge-large's batch 128 that
// tensor is 134 MB, so the chain is ~636 MB per layer and ~15 GB per encode --
// and tasks/0081 section 3 measured the host at 69.6% of the encode, nearly
// all of it memory traffic rather than arithmetic.
//
// Fused it is ~100 MB per layer: read C, and write ffn_down's int8 operand.
// Nothing else is materialised. The row (16 KB at most) stays in L1 across the
// three sub-passes, so the absmax GELU's output needs costs a cache hit rather
// than a DRAM sweep -- which is the whole reason a per-row activation scale is
// affordable at all (tasks/0079 measured the static alternative at 57x worse).
//
// Bit-identical to the unfused path by construction: same polynomial, same
// rounding, same order.
// `act` transforms the dequantised row IN PLACE and leaves `out_n` values at
// row[0, out_n) -- N for a plain GELU, N/2 for a gated FFN that combines two
// halves into one. It is a parameter rather than a branch so the caller keeps
// ownership of the exact intrinsics, which is what makes the fused and
// unfused paths bit-identical rather than merely close.
template <typename ParRows, typename Act>
inline void dequant_act_quant(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                       int64_t out_n, Act act,
                       const float *sa_up, const float *wscale,
                       const float *bias, const float *ias_next,
                       int8_t *dst, float *sa_next, ParRows par_rows) {
  require_stream_aligned(c, N, c_bytes, "dequant_act_quant");
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      const float sa = sa_up[static_cast<size_t>(r)];
      float *v = row.data();
      int64_t j = 0;
      // 1. dequantise + bias + GELU, into L1. THE INTRINSICS MUST MATCH the
      // unfused path's exactly -- `_mm256_fmadd_ps` rounds once where
      // `a*b + c` rounds twice, so a scalar rewrite of the same formula is
      // NOT the same number. Measured: 1.161e-03 unfused against 1.180e-03
      // for a scalar fused pass. Both pass the gate, but a fused path that
      // silently changes the result is a fused path nobody can A/B.
#if defined(__AVX2__)
      const __m256 sav = _mm256_set1_ps(sa);
      if (c_bytes == 2) {
        const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(lo, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(v + j + 8, _mm256_fmadd_ps(
              _mm256_mul_ps(hi, sav), _mm256_loadu_ps(wscale + j + 8),
              _mm256_loadu_ps(bias + j + 8)));
        }
      } else {
        const int32_t *cr = static_cast<const int32_t *>(c) + r * N;
        for (; j + 8 <= N; j += 8) {
          __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j)));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(cf, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
        }
      }
#endif
      for (; j < N; ++j) {
        const float cf = c_bytes == 2
            ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
            : static_cast<float>(static_cast<const int32_t *>(c)[r * N + j]);
        // NOT bit-identical to the FMA above, on purpose and for the record:
        // this rounds the final add TWICE where `_mm256_fmadd_ps` rounds it
        // once. 1 ULP, 25% of elements, measured by tools/verify/verify_i8_kernels.py,
        // and it moves sa_next -- see the file header. Do not "fix" this into
        // an fma or a plain rewrite without updating that gate: the two paths
        // are a reproducibility fact, and changing either one silently is
        // exactly what the gate exists to catch.
        v[j] = cf * sa * wscale[j] + bias[j];
      }
      // 2. the activation, in place, narrowing to out_n.
      act(v, N);
      // 3. row absmax of the smoothed value, and 4. quantise. Both over a row
      // that is now hot in L1, which is what makes this worth doing at all.
      float mx = 0.f;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= out_n; j += 8)
          acc = _mm256_max_ps(acc, _mm256_and_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
                            _mm256_loadu_ps(ias_next + j)), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        mx = _mm_cvtss_f32(h);
      }
#endif
      for (; j < out_n; ++j) {
        const float a = std::fabs(v[j] * ias_next[j]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      sa_next[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      int8_t *q = dst + r * out_n;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 invv = _mm256_set1_ps(inv);
        const __m256 vhi = _mm256_set1_ps(127.0f);
        const __m256 vlo = _mm256_set1_ps(-127.0f);
        for (; j + 8 <= out_n; j += 8) {
          __m256 t = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
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
      for (; j < out_n; ++j) {
        float t = std::nearbyintf(v[j] * ias_next[j] * inv);
        if (t > 127.f) t = 127.f;
        if (t < -127.f) t = -127.f;
        q[j] = static_cast<int8_t>(t);
      }
    }
  });
}

// C -> fp32: y = acc * sa[row] * wscale[col] + bias[col]. A rank-1
// outer-product scaling folded into the pass that already reads C and adds the
// bias. `c_bytes` selects the transport width the design chose: 4 = int32
// accumulator straight out, 2 = narrowed to bf16 on the core (tasks/0080).
template <typename ParRows>
inline void dequantise_c(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                  const float *a_scale, const float *wscale, const float *bias,
                  float *out, ParRows par_rows, bool sim_bf16 = false) {
  require_stream_aligned(c, N, c_bytes, "dequantise_c");
  if (c_bytes == 2) {
    const uint16_t *cb = static_cast<const uint16_t *>(c);
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb + r * N;
        float *o = out + r * N;
        const float sa = a_scale[static_cast<size_t>(r)];
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 sav = _mm256_set1_ps(sa);
        for (; j + 16 <= N; j += 16) {
          // One 32-byte streaming load carries 16 bf16 against 8 int32 -- the
          // same instruction count for twice the elements, which is the whole
          // point of narrowing C.
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(o + j,
              _mm256_fmadd_ps(_mm256_mul_ps(lo, sav),
                              _mm256_loadu_ps(wscale + j),
                              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(o + j + 8,
              _mm256_fmadd_ps(_mm256_mul_ps(hi, sav),
                              _mm256_loadu_ps(wscale + j + 8),
                              _mm256_loadu_ps(bias + j + 8)));
        }
#endif
        for (; j < N; ++j)
          o[j] = from_bf16(cr[j]) * sa * wscale[j] + bias[j];   // 2 roundings; see the file header
      }
    });
    return;
  }
  const int32_t *ci = static_cast<const int32_t *>(c);
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int32_t *cr = ci + r * N;
      float *o = out + r * N;
      const float sa = a_scale[static_cast<size_t>(r)];
      int64_t j = 0;
#if defined(__AVX2__)
      // Streaming loads: C is a write-combined XRT host bo and ordinary loads
      // from it stall per line (tasks/0024).
      const __m256 sav = _mm256_set1_ps(sa);
      for (; j + 8 <= N; j += 8) {
        __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
            reinterpret_cast<const __m256i *>(cr + j)));
        if (sim_bf16) {
          // --sim-c-bf16: round exactly as a narrowed-C design would, to price
          // one before building it (tasks/0080). RNE to bf16 = add half an ulp
          // plus the tie-break bit, then truncate.
          __m256i u = _mm256_castps_si256(cf);
          u = _mm256_add_epi32(
              u, _mm256_add_epi32(
                     _mm256_set1_epi32(0x7FFF),
                     _mm256_and_si256(_mm256_srli_epi32(u, 16),
                                      _mm256_set1_epi32(1))));
          cf = _mm256_castsi256_ps(
              _mm256_and_si256(u, _mm256_set1_epi32(int(0xFFFF0000u))));
        }
        _mm256_storeu_ps(o + j,
            _mm256_fmadd_ps(_mm256_mul_ps(cf, sav),
                            _mm256_loadu_ps(wscale + j),
                            _mm256_loadu_ps(bias + j)));
      }
#endif
      for (; j < N; ++j) {
        float cf = static_cast<float>(cr[j]);
        if (sim_bf16) {
          uint32_t u;
          std::memcpy(&u, &cf, 4);
          u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
          std::memcpy(&cf, &u, 4);
        }
        o[j] = cf * sa * wscale[j] + bias[j];   // 2 roundings; see the file header
      }
    }
  });
}

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v),
                         _mm256_extractf128_ps(v, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
#endif

inline double cpu_seconds() {
#ifdef _WIN32
  FILETIME c, e, k, u;
  GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
  auto to_s = [](FILETIME f) {
    return ((static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime) *
           1e-7;
  };
  return to_s(k) + to_s(u);
#else
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
#endif
}

}  // namespace app
