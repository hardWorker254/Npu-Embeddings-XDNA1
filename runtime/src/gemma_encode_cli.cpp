//===- gemma_encode_cli.cpp -----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- standalone verification CLI for GemmaEncoder (arch=1).
// SPDX-License-Identifier: Apache-2.0
//
// Same role as tokenizer_gemma_cli.cpp (tasks/0061) and
// gemma_kernels_test.cpp (tasks/0063): a small, XRT-free executable built by
// hand with cl.exe, isolated from runtime/build/ and from any concurrent
// NPU session, so the CPU-only encode can be iterated on and verified fast.
// See tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md for the build command.
//
// Usage: gemma_encode_cli.exe <model.npue> <texts.txt> <out.f32> [max_len] [prefix]
//   texts.txt: one text per line.
//   out.f32:   raw float32, [n_texts, hidden] row-major -- compared against
//              reference/encoder_gemma.py's output for the SAME texts and
//              prefix by tools/verify_gemma_cpu_encode.py.

#include "gemma_encode.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <model.npue> <texts.txt> <out.f32> [max_len] "
                 "[prefix]\n",
                 argv[0]);
    return 2;
  }
  const std::string model_path = argv[1], texts_path = argv[2], out_path = argv[3];
  const int max_len = argc > 4 ? std::atoi(argv[4]) : 64;
  const std::string prefix = argc > 5 ? argv[5] : "document";

  try {
    npue::File model(model_path);
    npue::GemmaEncoder enc(model);
    std::printf("loaded %s: hidden=%lld layers=%lld heads=%lld kv_heads=%lld "
               "head_dim=%lld, tokenizer vocab %zu\n",
               model_path.c_str(), (long long)enc.hidden(), 0LL, 0LL, 0LL, 0LL,
               enc.tok.vocab_size());

    std::vector<std::string> texts;
    {
      std::ifstream in(texts_path, std::ios::binary);
      if (!in) throw std::runtime_error("cannot open " + texts_path);
      std::string line;
      while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        texts.push_back(line);
      }
    }
    std::printf("encoding %zu texts, max_len=%d, prefix='%s'\n", texts.size(),
               max_len, prefix.c_str());

    std::ofstream of(out_path, std::ios::binary);
    if (!of) throw std::runtime_error("cannot open " + out_path);

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < texts.size(); ++i) {
      const auto v = enc.encode_one(texts[i], max_len, prefix);
      of.write(reinterpret_cast<const char *>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(float)));
      std::printf("  [%zu] \"%.60s\"  first6: %.4f %.4f %.4f %.4f %.4f %.4f\n",
                 i, texts[i].c_str(), v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double el = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wrote %s: %zu x %lld fp32, %.2f s (%.2f s/text)\n",
               out_path.c_str(), texts.size(), (long long)enc.hidden(), el,
               el / std::max<size_t>(1, texts.size()));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
