//===- tokenizer_xlmr_cli.cpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- standalone verification CLI for the XLM-R Unigram
// tokenizer (runtime/src/tokenizer_xlmr.cpp). SPDX-License-Identifier:
// Apache-2.0
//
// Deliberately a SEPARATE tiny executable, not a mode of npuembed.exe --
// same reasoning as tokenizer_gemma_cli.cpp: no XRT dependency at all
// (tokenizer_xlmr.cpp/hpp and xlmr_tokenizer_gen.cpp are pure STL), so it
// builds and runs without the NPU runtime, and it does not touch main.cpp
// or the model catalogue. The arch-3 integration is a later task.
//
// Usage:
//   tokenizer_xlmr_cli.exe <table.bin> <texts.txt>
//       Reads one text per line, encodes each with XlmrTokenizer::encode()
//       (<s> ... </s>, no padding, no truncation -- the same contract as
//       tools/xlmr_tokenizer_ref.py's CLI), prints one line of
//       space-separated ids per input line.
//   tokenizer_xlmr_cli.exe --gen <tokenizer.json> <out.bin>
//       Runs the C++ generator port (xlmr_tokenizer_gen.cpp) and writes the
//       XLMRTOK1 table -- so tasks/0133's byte-identity check against the
//       Python generator needs no throwaway program (0067 used one).
//
// INPUT ESCAPING (encode mode only): two corpus entries carry embedded
// newline codepoints, which a one-text-per-line protocol cannot ship raw.
// Lines are therefore minimally backslash-escaped: "\\n" -> LF, "\\r" ->
// CR, "\\\\" -> backslash; everything else passes through verbatim. The
// driver (tools/verify_tokenizer_xlmr.py --cli) applies the mirror-image
// escape. The Python reference CLI predates this and reads raw lines; the
// verifier drives it in-process, so nothing diffs the two CLIs' framing.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "tokenizer_xlmr.hpp"
#include "xlmr_tokenizer_gen.hpp"

namespace {

std::string unescape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      const char c = s[i + 1];
      if (c == 'n') { out.push_back('\n'); ++i; continue; }
      if (c == 'r') { out.push_back('\r'); ++i; continue; }
      if (c == '\\') { out.push_back('\\'); ++i; continue; }
    }
    out.push_back(s[i]);
  }
  return out;
}

int run_gen(const char *tokenizer_json, const char *out_path) {
  const std::vector<uint8_t> table =
      npue::generate_xlmr_tokenizer_table(tokenizer_json);
  std::ofstream f(out_path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s for writing\n", out_path);
    return 2;
  }
  f.write(reinterpret_cast<const char *>(table.data()),
          static_cast<std::streamsize>(table.size()));
  f.close();
  if (!f) {
    std::fprintf(stderr, "error writing %s\n", out_path);
    return 2;
  }
  std::fprintf(stderr, "wrote %s (%zu bytes)\n", out_path, table.size());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--gen")
      return run_gen(argv[2], argv[3]);
    if (argc != 3) {
      std::fprintf(stderr,
                   "usage: %s <table.bin> <texts.txt>\n"
                   "       %s --gen <tokenizer.json> <out.bin>\n",
                   argv[0], argv[0]);
      return 2;
    }

    npue::XlmrTokenizer tok = npue::XlmrTokenizer::from_table_file(argv[1]);

    std::ifstream f(argv[2], std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", argv[2]);
      return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
      while (!line.empty() && line.back() == '\r') line.pop_back();
      const std::vector<int32_t> ids = tok.encode(unescape(line));
      std::string out;
      out.reserve(ids.size() * 7);
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out.push_back(' ');
        out += std::to_string(ids[i]);
      }
      std::puts(out.c_str());
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
