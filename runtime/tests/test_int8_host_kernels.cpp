//===- test_int8_host_kernels.cpp ----------------------------- C++ -*-===//
//
// Do the int8 host kernels in runtime/include/common/host_kernels.hpp mean the
// same thing on a host with AVX2 and on one without?
//
// The file header used to assert that they did -- "The AVX2 paths are
// bit-identical to their scalar fallbacks" -- and it was wrong. This probe is
// how that was found, and it is the thing that keeps the answer pinned.
//
// NOT a self-contained gate, on purpose, for the same reason
// test_pack_mac.cpp is not: the reference here is the OTHER BUILD of this same
// source, so the comparison belongs where both are visible.
// tools/verify/verify_i8_kernels.py builds this file three times -- AVX2+FMA, scalar
// with contraction off, scalar with the compiler's default contraction -- and
// diffs the sections below.
//
// It also does the one thing a diff cannot: it CALLS the kernels with a
// deliberately unaligned C and a deliberately odd row stride, and reports
// whether the runtime refused. Before require_stream_aligned() those two calls
// were a SIGSEGV on an AVX2 host and a correct answer on a non-AVX2 one, which
// is the worst possible shape for a bug -- it reproduces only on the machines
// that have the fast path. The ASan backtrace that pinned it down:
//
//   #0 _mm256_stream_load_si256 .../avx2intrin.h:922
//   #1 operator() runtime/include/common/host_kernels.hpp:463
//   #3 dequantise_c<...> runtime/include/common/host_kernels.hpp:452
//   ==12610==The signal is caused by a READ memory access.
//   rdi = 0x00007bba2ebe0074        <- 0x74, so 20 mod 32
//
// 20 is 212 mod 32, and 212 is 53 * 4: an int32 C with N=53 puts row 1 twenty
// bytes into a 32-byte block. The base pointer was perfectly aligned. The
// precondition is on the row STRIDE, which is the half that is easy to miss.
//
// Build:
//   g++ -std=c++17 -O2 -mavx2 -mfma -I runtime/include \
//       runtime/tests/test_int8_host_kernels.cpp -o /tmp/probe_avx2
//   g++ -std=c++17 -O2 -mno-avx2 -ffp-contract=off -I runtime/include \
//       runtime/tests/test_int8_host_kernels.cpp -o /tmp/probe_scalar
// Run:
//   /tmp/probe_avx2 <blob.bin> <index.txt>
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/host_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace app;

namespace {

// The storage behind an aligned_copy must OUTLIVE the pointer, so it lives in a
// function-static that only grows. A local vector is a use-after-free, not a
// leak -- which is its own lesson about the kind of bug this file hunts.
std::vector<std::vector<char>> g_keep;

template <typename T>
T *aligned_copy(const std::vector<T> &src) {
  g_keep.emplace_back(src.size() * sizeof(T) + 64);
  std::vector<char> &raw = g_keep.back();
  char *base = raw.data();
  const size_t off = ((32 - ((uintptr_t)base % 32)) % 32) / sizeof(T);
  T *p = reinterpret_cast<T *>(base + off * sizeof(T));
  for (size_t i = 0; i < src.size(); ++i) p[i] = src[i];
  return p;
}

// One LCG, so all three builds see identical inputs with no input file to
// disagree about. A gate whose inputs are generated differently per build would
// be measuring its own generator.
uint32_t g_s = 987654321u;
float rnd() {
  g_s = g_s * 1664525u + 1013904223u;
  return (float)((g_s >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

// rows, K and N are chosen to be unrepresentative of a GEMM on purpose, and the
// reasons are different in each case:
//   K = 45  is NOT a multiple of 8, so quantise_a_int8's vector head and scalar
//           tail run inside one call. If the two orders of clip-then-round vs
//           round-then-clip ever disagreed, this is where it would show.
//   N = 64  IS a multiple of 16, because a production N always is, and an odd N
//           would trip the alignment precondition instead of testing arithmetic.
// A second N = 72 appears below, legal for an int32 C and illegal for a bf16
// one, which is the only way to show the check distinguishes the two.
constexpr int64_t kRows = 7, kK = 45, kN = 64, kOutN = 32;
constexpr int64_t kN72 = 72;   // 72*4 = 288 = 9*32 legal; 72*2 = 144 = 4*32+16 not

struct Sections {
  std::vector<char> blob;
  std::map<std::string, std::pair<size_t, size_t>> index;   // name -> (off, bytes)

  template <typename T>
  void put(const std::string &name, const T *data, size_t n) {
    const size_t off = blob.size();
    const size_t bytes = n * sizeof(T);
    const char *p = reinterpret_cast<const char *>(data);
    blob.insert(blob.end(), p, p + bytes);
    index[name] = {off, bytes};
  }
  void put_text(const std::string &name, const std::string &s) {
    put(name, s.data(), s.size());
  }
};

auto serial = [](int64_t n, auto &&body) { body(int64_t(0), n); };

// Run something that is EXPECTED to throw, and report what it said. "returned a
// value" is the answer that matters just as much as the message: a precondition
// that quietly does nothing is the failure mode this whole file exists to
// avoid, and it looks exactly like a pass if nobody checks.
template <typename F>
std::string refusal(F &&f) {
  try {
    f();
    return "RETURNED";
  } catch (const std::exception &e) {
    return std::string("threw: ") + e.what();
  }
}

}   // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <blob.bin> <index.txt>\n", argv[0]);
    return 2;
  }
  Sections s;

  // ---- inputs, dumped so a divergence can be attributed without a rebuild ----
  std::vector<float> a(kRows * kK), ias(kK), ws(kN), bias(kN), iasn(kN);
  for (auto &v : a) v = rnd() * 3.0f;
  for (auto &v : ias) v = 0.25f + std::fabs(rnd()) * 2.0f;
  for (auto &v : ws) v = std::fabs(rnd()) * 0.08f;
  for (auto &v : bias) v = rnd() * 0.7f;
  for (auto &v : iasn) v = 0.3f + std::fabs(rnd());
  std::vector<int32_t> acc(kRows * kN);
  for (auto &v : acc) v = (int32_t)(rnd() * 200000.0f);
  std::vector<uint16_t> cbf(kRows * kN);
  for (size_t i = 0; i < cbf.size(); ++i)
    cbf[i] = to_bf16((float)acc[i] * 1e-4f);

  float *ap = aligned_copy(a), *iasp = aligned_copy(ias);
  float *wsp = aligned_copy(ws), *bp = aligned_copy(bias), *iasnp = aligned_copy(iasn);
  int32_t *accp = aligned_copy(acc);
  uint16_t *cbfp = aligned_copy(cbf);

  s.put("in.a", a.data(), a.size());
  s.put("in.ias", ias.data(), ias.size());
  s.put("in.ws", ws.data(), ws.size());
  s.put("in.bias", bias.data(), bias.size());
  s.put("in.acc", acc.data(), acc.size());
  s.put("in.cbf16", cbf.data(), cbf.size());

  // ---- 1. quantise_a_int8 ----
  std::vector<int8_t> q(kRows * kK);
  std::vector<float> sa(kRows);
  quantise_a_int8(ap, kRows, kK, iasp, q.data(), sa.data(), serial);
  s.put("quantise.q", q.data(), q.size());
  s.put("quantise.sa", sa.data(), sa.size());

  // ---- 2. dequantise_c, three transport widths ----
  std::vector<float> out(kRows * kN);
  dequantise_c(accp, 4, kRows, kN, sa.data(), wsp, bp, out.data(), serial);
  s.put("dequantise_c.i32", out.data(), out.size());
  dequantise_c(cbfp, 2, kRows, kN, sa.data(), wsp, bp, out.data(), serial);
  s.put("dequantise_c.bf16", out.data(), out.size());
  dequantise_c(accp, 4, kRows, kN, sa.data(), wsp, bp, out.data(), serial, true);
  s.put("dequantise_c.sim_bf16", out.data(), out.size());

  // ---- 3. dequant_act_quant: the fused FFN epilogue ----
  std::vector<int8_t> q2(kRows * kOutN);
  std::vector<float> sa2(kRows);
  dequant_act_quant(
      accp, 4, kRows, kN, kOutN,
      [](float *v, int64_t n) {   // a gated narrowing, the shape it is built for
        for (int64_t j = 0; j < n / 2; ++j) v[j] = std::fabs(v[j]) * 1.5f - 0.3f;
      },
      sa.data(), wsp, bp, iasnp, q2.data(), sa2.data(), serial);
  s.put("dequant_act_quant.q", q2.data(), q2.size());
  s.put("dequant_act_quant.sa", sa2.data(), sa2.size());

  // ---- 4. the alignment precondition, exercised ----
  // 4a. a base pointer off by one int32. The subtle half is 4b; this half is
  //     the obvious one, included so a fix that only handles the stride shows up.
  s.put_text("align.skewed_base",
             refusal([&] {
               dequantise_c(accp + 1, 4, kRows, kN, sa.data(), wsp, bp,
                            out.data(), serial);
             }));
  s.put_text("align.skewed_base.fused",
             refusal([&] {
               dequant_act_quant(
                   accp + 1, 4, kRows, kN, kOutN,
                   [](float *, int64_t) {}, sa.data(), wsp, bp, iasnp,
                   q2.data(), sa2.data(), serial);
             }));

  // 4b. a perfectly aligned base with a row stride that is not a multiple of 32.
  //     N=72 over an int32 C is legal (288 = 9*32); the SAME N over a bf16 C is
  //     not (144 = 4*32+16). Both are called on the same aligned buffer, so the
  //     only difference is the width the stride is computed from.
  std::vector<int32_t> wide(kRows * kN72);
  for (auto &v : wide) v = (int32_t)(rnd() * 200000.0f);
  std::vector<float> ws72(kN72), b72(kN72), out72(kRows * kN72);
  for (int64_t j = 0; j < kN72; ++j) { ws72[j] = wsp[j % kN]; b72[j] = bp[j % kN]; }
  int32_t *widep = aligned_copy(wide);
  s.put_text("align.stride_ok_i32_n72",
             refusal([&] {
               dequantise_c(widep, 4, kRows, kN72, sa.data(), ws72.data(),
                            b72.data(), out72.data(), serial);
             }));
  s.put_text("align.stride_bad_bf16_n72",
             refusal([&] {
               dequantise_c(cbfp, 2, kRows, kN72, sa.data(), ws72.data(),
                            b72.data(), out72.data(), serial);
             }));

  // ---- write ----
  FILE *fb = fopen(argv[1], "wb");
  if (!fb) { perror(argv[1]); return 2; }
  fwrite(s.blob.data(), 1, s.blob.size(), fb);
  fclose(fb);

  FILE *fi = fopen(argv[2], "w");
  if (!fi) { perror(argv[2]); return 2; }
  for (const auto &kv : s.index)
    fprintf(fi, "%s\t%zu\t%zu\n", kv.first.c_str(), kv.second.first,
            kv.second.second);
  fclose(fi);

  fprintf(stderr, "probe: %zu bytes, %zu sections%s\n", s.blob.size(),
          s.index.size(),
#if defined(__AVX2__)
          " [AVX2+FMA]"
#else
          " [scalar]"
#endif
  );
  return 0;
}
