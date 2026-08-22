//===- tokenizer_gemma_cli.cpp -------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- standalone verification CLI for the EmbeddingGemma
// tokenizer (runtime/src/tokenizer_gemma.cpp). SPDX-License-Identifier:
// Apache-2.0
//
// Deliberately a SEPARATE tiny executable, not a mode of npuembed.exe: it
// has no XRT dependency at all (tokenizer_gemma.cpp/hpp are pure STL), so it
// builds and runs without the NPU runtime, and it does not touch main.cpp
// or the model catalogue -- this task is the tokenizer alone, not the
// arch=1 integration (see tasks/00XX-gemma-tokenizer/TASK.md).
//
// Usage:
//   tokenizer_gemma_cli.exe <table.bin> <corpus.txt> <max_len> [prefix_name]
//
// Reads one text per line from corpus.txt, encodes each with
// GemmaTokenizer::encode(text, max_len, prefix_name), and prints one line of
// space-separated input_ids per input line -- the same contract
// tools/verify_tokenizer.py already uses against npuembed.exe's --tokenize
// mode for the BERT tokenizer, so tools/verify_tokenizer_gemma.py can diff
// this output against HuggingFace's directly.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "tokenizer_gemma.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <table.bin> <corpus.txt> <max_len> [prefix_name]\n",
                 argv[0]);
    return 2;
  }
  const std::string table_path = argv[1];
  const std::string corpus_path = argv[2];
  const int max_len = std::atoi(argv[3]);
  const std::string prefix_name = argc >= 5 ? argv[4] : "document";

  try {
    npue::GemmaTokenizer tok = npue::GemmaTokenizer::from_table_file(table_path);

    std::ifstream f(corpus_path, std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", corpus_path.c_str());
      return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const std::string use_prefix = prefix_name == "-" ? std::string() : prefix_name;
      npue::GemmaEncoded e = tok.encode(line, max_len, use_prefix);
      std::string out;
      out.reserve(e.input_ids.size() * 7);
      for (size_t i = 0; i < e.input_ids.size(); ++i) {
        if (i) out.push_back(' ');
        out += std::to_string(e.input_ids[i]);
      }
      std::puts(out.c_str());
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
