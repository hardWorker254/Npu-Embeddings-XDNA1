//===- gemma_kernels_test.cpp ----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- standalone verification CLI for the EmbeddingGemma host
// kernels (runtime/src/gemma_kernels.cpp). SPDX-License-Identifier: Apache-2.0
//
// Deliberately a SEPARATE tiny executable, not a mode of npuembed.exe, same
// reasoning as tokenizer_gemma_cli.cpp: gemma_kernels.cpp/.hpp have no XRT
// dependency, so this builds and runs without the NPU runtime and does not
// touch main.cpp or hub.cpp -- this task is the kernels alone, not the
// arch=1 integration (see tasks/0063-m12-embeddinggemma-kernels/TASK.md).
//
// Reads the flat binary tools/dump_gemma_kernel_vectors.py writes -- a
// sequence of (name, kind, before..., after) records built from REAL
// intermediate tensors tapped out of reference/encoder_gemma.py running the
// real checkpoint on real sentences (reference/goldens_gemma/
// embeddinggemma-300m_l24_s64_taps.safetensors) -- runs the matching C++
// kernel on each "before" block, and reports max/mean absolute difference
// against the tapped "after" block. Exits nonzero if anything exceeds the
// per-kind tolerance.
//
// Usage:
//   gemma_kernels_test.exe <vectors.bin> [tolerance]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gemma_kernels.hpp"

namespace {

struct Record {
  std::string name;
  uint32_t kind = 0, rows = 0, dim = 0, seq_len = 0;
  float eps = 0.0f;
  double theta = 0.0;
  std::vector<float> a, b, expected;   // a/b: weight+before (kind 0), before
                                       // (kind 1), gate+up (kind 2)
};

bool read_u32(std::ifstream &f, uint32_t &v) {
  return static_cast<bool>(f.read(reinterpret_cast<char *>(&v), sizeof(v)));
}

bool read_block(std::ifstream &f, std::vector<float> &v, size_t n) {
  v.resize(n);
  if (n == 0) return true;
  return static_cast<bool>(
      f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float)));
}

// Returns false at clean EOF (no more records), throws on a truncated file.
bool read_record(std::ifstream &f, Record &r) {
  char magic[4];
  if (!f.read(magic, 4)) return false;   // clean EOF
  if (std::memcmp(magic, "GKK1", 4) != 0)
    throw std::runtime_error("bad magic");
  uint32_t name_len = 0;
  if (!read_u32(f, name_len)) throw std::runtime_error("truncated (name_len)");
  std::string name(name_len, '\0');
  if (name_len && !f.read(name.data(), name_len))
    throw std::runtime_error("truncated (name)");
  uint32_t kind = 0, rows = 0, dim = 0, seq_len = 0;
  float eps = 0.0f;
  double theta = 0.0;
  if (!read_u32(f, kind) || !read_u32(f, rows) || !read_u32(f, dim))
    throw std::runtime_error("truncated (header)");
  if (!f.read(reinterpret_cast<char *>(&eps), sizeof(eps)))
    throw std::runtime_error("truncated (eps)");
  if (!read_u32(f, seq_len)) throw std::runtime_error("truncated (seq_len)");
  if (!f.read(reinterpret_cast<char *>(&theta), sizeof(theta)))
    throw std::runtime_error("truncated (theta)");

  r.name = name;
  r.kind = kind;
  r.rows = rows;
  r.dim = dim;
  r.seq_len = seq_len;
  r.eps = eps;
  r.theta = theta;

  const size_t rd = static_cast<size_t>(rows) * dim;
  if (kind == 0) {
    if (!read_block(f, r.a, dim)) throw std::runtime_error("truncated (weight)");
    if (!read_block(f, r.b, rd)) throw std::runtime_error("truncated (before)");
    if (!read_block(f, r.expected, rd)) throw std::runtime_error("truncated (after)");
  } else if (kind == 1) {
    if (!read_block(f, r.b, rd)) throw std::runtime_error("truncated (before)");
    if (!read_block(f, r.expected, rd)) throw std::runtime_error("truncated (after)");
  } else if (kind == 2) {
    if (!read_block(f, r.a, rd)) throw std::runtime_error("truncated (gate)");
    if (!read_block(f, r.b, rd)) throw std::runtime_error("truncated (up)");
    if (!read_block(f, r.expected, rd)) throw std::runtime_error("truncated (after)");
  } else {
    throw std::runtime_error("unknown kind " + std::to_string(kind));
  }
  return true;
}

struct Stats {
  double max_abs = 0.0, mean_abs = 0.0, max_rel = 0.0, rel_fro = 0.0;
};

// rel_fro = ||got-want||_F / ||want||_F -- this project's own standard
// per-tensor accuracy metric (used throughout tasks/ for GEMM/kernel
// validation), reported here alongside max_abs/mean_abs/max_rel because it
// is scale-aware in a way a fixed absolute tolerance is not: EmbeddingGemma's
// deep-layer activations range from O(1) to O(1e4) (confirmed: L23's
// post-feedforward-norm output reaches |x|~28,357), so a single global
// absolute-diff threshold would either be meaninglessly loose on small
// tensors or falsely fail on large ones purely from float32 ULP scaling --
// exactly what happened during development here before this metric was
// added (max_abs 9.8e-4 on a tensor whose values reach 2.8e4 is one ULP,
// not a bug). max_rel (per-element, denominator floored at 1e-8) is also
// reported for diagnostics -- it is expected to spike near RoPE's cos/sin
// zero-crossings, where a machine-epsilon absolute difference divides by a
// near-zero reference value; that is a property of the relative-error
// metric near zero, not evidence of a wrong computation (max_abs stays at
// the float32 ULP floor there too).
Stats compare(const std::vector<float> &got, const std::vector<float> &want) {
  Stats s;
  double sum_abs = 0.0, sum_sq_diff = 0.0, sum_sq_want = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = static_cast<double>(want[i]);
    const double d = std::fabs(g - w);
    sum_abs += d;
    sum_sq_diff += (g - w) * (g - w);
    sum_sq_want += w * w;
    s.max_abs = std::max(s.max_abs, d);
    const double denom = std::max(1e-8, std::fabs(w));
    s.max_rel = std::max(s.max_rel, d / denom);
  }
  s.mean_abs = want.empty() ? 0.0 : sum_abs / static_cast<double>(want.size());
  s.rel_fro = sum_sq_want > 0.0 ? std::sqrt(sum_sq_diff / sum_sq_want)
                                : std::sqrt(sum_sq_diff);
  return s;
}

const char *kind_name(uint32_t k) {
  switch (k) {
    case 0: return "rmsnorm";
    case 1: return "rope";
    case 2: return "geglu";
    default: return "?";
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vectors.bin> [tolerance=1e-4]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  // Gate on rel_fro (scale-aware), not max_abs -- see the comment on
  // compare() for why a fixed absolute tolerance is wrong across tensors
  // whose magnitude spans O(1) to O(1e4) in this model. 1e-5 is generous
  // relative to the ~1e-7 (float32 machine epsilon) actually observed on
  // every case in tasks/0063 -- a real formula bug measures orders of
  // magnitude larger (this project's own "1-cos" gates elsewhere sit at
  // 1e-3 to 2e-3 for legitimately-passing BF16 NPU kernels; this is fp32
  // host code with no NPU/bf16 involved, so the bar is far tighter).
  const double tol = argc >= 3 ? std::atof(argv[2]) : 1e-5;

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return 2;
  }

  // Unit-test the per-layer full/sliding attention selection itself (no
  // tensor comparison needed -- it is a pure deterministic function of the
  // layer index). Known-correct set for sliding_window_pattern=6, 24
  // layers, per reference/encoder_gemma.py's is_full_attention_layer():
  // layers 5, 11, 17, 23 (0-indexed) are full_attention.
  {
    bool ok = true;
    for (int i = 0; i < 24; ++i) {
      const bool want = (i == 5 || i == 11 || i == 17 || i == 23);
      const bool got = npue::gemma_is_full_attention_layer(i, 6);
      if (got != want) {
        std::fprintf(stderr, "FAIL gemma_is_full_attention_layer(%d): got %d want %d\n",
                     i, got, want);
        ok = false;
      }
    }
    std::printf("%-40s %s\n", "layer_selection", ok ? "PASS" : "FAIL");
    if (!ok) return 1;
  }

  int n_pass = 0, n_fail = 0;
  Record r;
  try {
    while (read_record(f, r)) {
      std::vector<float> got(r.expected.size());
      Stats s;

      if (r.kind == 0) {
        npue::rms_norm_cpu(r.b.data(), r.a.data(), got.data(), r.rows, r.dim, r.eps);
        s = compare(got, r.expected);
      } else if (r.kind == 1) {
        std::vector<float> cos(static_cast<size_t>(r.seq_len) * r.dim);
        std::vector<float> sin(static_cast<size_t>(r.seq_len) * r.dim);
        npue::gemma_rope_tables(r.seq_len, r.dim, r.theta, cos.data(), sin.data());
        npue::apply_rope_cpu(r.b.data(), cos.data(), sin.data(), got.data(),
                             r.rows, r.seq_len, r.dim);
        s = compare(got, r.expected);
      } else {  // kind == 2, geglu
        npue::geglu_cpu(r.a.data(), r.b.data(), got.data(), r.a.size());
        s = compare(got, r.expected);
      }

      const bool pass = s.rel_fro < tol;
      std::printf("%-40s %-8s rows=%-6u dim=%-6u rel_fro=%.3e max_abs=%.3e "
                 "mean_abs=%.3e max_rel=%.3e  %s\n",
                 r.name.c_str(), kind_name(r.kind), r.rows, r.dim, s.rel_fro,
                 s.max_abs, s.mean_abs, s.max_rel, pass ? "PASS" : "FAIL");
      if (pass) ++n_pass; else ++n_fail;
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s (after %d records)\n", e.what(), n_pass + n_fail);
    return 1;
  }

  std::printf("\n%d/%d records passed (tolerance %.1e)\n", n_pass,
             n_pass + n_fail, tol);
  return n_fail == 0 ? 0 : 1;
}
