//===- test_whisper_tokenizer.cpp -------------------------------*- C++ -*-===//
//
// Runs the C++ Whisper tokenizer over stdin so a gate can hold it against the
// reference and against HuggingFace.
//
// This is a test binary, not a CLI: `npuembeddings tokenize` needs a model the
// loader accepts, and the loader does not accept arch=4 yet (that is the
// encoder's phase, not the tokenizer's). The tokenizer reads nothing but the
// packed table, so a test binary reaches it without dragging the model
// registry in front of it -- and the gate it feeds is the thing that decides
// whether the encoder may rely on this code.
//
// Protocol: one input PER LINE, hex-encoded, because a corpus for this
// tokenizer necessarily contains newlines and CRLF and a line-framed plain
// text protocol would silently shift every case after the first one it splits:
//   stdin  <hex of the text>
//   stdout <n> ids...            the token ids for that input
//   stderr one line per input: the decoded round trip, as hex, so a mismatch
//          is comparable without quoting
//
// Build:
//   g++ -std=c++17 -O1 -I runtime/include runtime/tests/test_whisper_tokenizer.cpp \
//       runtime/src/tokenizers/whisper.cpp runtime/src/model.cpp \
//       -o /tmp/test_whisper_tokenizer
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "runtime/model.hpp"
#include "tokenizers/whisper.hpp"

namespace {

int from_hex(const std::string &s, std::string &out) {
  if (s.size() % 2) return -1;
  out.clear();
  out.reserve(s.size() / 2);
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i < s.size(); i += 2) {
    const int hi = nib(s[i]), lo = nib(s[i + 1]);
    if (hi < 0 || lo < 0) return -1;
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  return 0;
}

std::string to_hex(const std::string &s) {
  static const char *d = "0123456789abcdef";
  std::string out;
  out.reserve(s.size() * 2);
  for (unsigned char c : s) {
    out.push_back(d[c >> 4]);
    out.push_back(d[c & 0xF]);
  }
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.npue>\n", argv[0]);
    return 2;
  }
  npue::File file(argv[1]);
  if (!file.has("tokenizer.whisper_table")) {
    std::fprintf(stderr,
                 "%s has no tokenizer.whisper_table -- it is not a Whisper "
                 "container, or it was packed without a tokenizer\n",
                 argv[1]);
    return 2;
  }
  const npue::Span table = file.raw("tokenizer.whisper_table");

  npue::WhisperTokenizer tok(table.data, table.bytes);
  // Named lookups, so a missing control token is a refusal at construction
  // rather than a transcript that cannot stop.
  for (const char *t : {"<|startoftranscript|>", "<|notimestamps|>",
                        "<|transcribe|>", "<|translate|>", "<|endoftext|>"})
    std::fprintf(stderr, "control %-26s id %d\n", t, tok.token_id(t));
  std::fprintf(stderr, "vocab %zu  merges %zu\n", tok.vocab_size(),
               tok.n_merges());

  std::string line, text;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (from_hex(line, text) != 0) {
      std::fprintf(stderr, "input is not hex: %s\n", line.c_str());
      return 2;
    }
    const npue::Encoded e = tok.encode(text, 0);
    std::printf("%zu", e.input_ids.size());
    for (int32_t id : e.input_ids) std::printf(" %d", id);
    std::printf("\n");
    std::fflush(stdout);
    // The round trip is the direction a transcription needs, so it is reported
    // whether or not the caller asked for ids.
    std::fprintf(stderr, "back %s\n",
                 to_hex(tok.decode(e.input_ids)).c_str());
  }
  return 0;
}
