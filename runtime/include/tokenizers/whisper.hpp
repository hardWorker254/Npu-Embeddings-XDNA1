//===- whisper.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper (GPT-2 byte-level BPE) tokenizer.
// SPDX-License-Identifier: Apache-2.0
//
// Whisper's vocabulary is NOT in vocab.json alone. vocab.json is GPT-2's
// 50258 entries and nothing else; every control token that makes a
// transcription possible -- <|startoftranscript|>, <|notimestamps|>,
// <|transcribe|>, <|translate|>, <|endoftext|>, and the ~1600 timestamp and
// language tokens -- lives in added_tokens.json with ids from 50258 up. Both
// are packed into ONE binary table at container build time
// (tools/lib/whisper_bpe.py is the single definition of its layout), and this
// class is its only reader.
//
// So the class is constructed from a table blob, not from a checkpoint
// directory: the container is the only place both halves exist, and a reader
// that could be pointed at vocab.json alone is a reader that will be.
//
// WHAT IS VERIFIED, AND HOW
// -------------------------
// tools/lib/whisper_tokenizer_ref.py is the executable specification for this
// file, byte for byte, over the same blob; tools/verify/verify_whisper_tokenizer.py
// holds this implementation against it AND against HuggingFace's own
// tokenizer over an adversarial corpus (Cyrillic, CJK, emoji, CRLF, digit
// runs, contractions, long repeats). The gate is exact token ids, because a
// transcription is either the right text or it is not: there is no tolerance
// to spend here.
//
// encode() exists for the `tokenize` CLI and for scoring a hypothesis in a
// test; decode() is the direction an STT container actually needs, since it
// receives ids from the decoder and owes the caller a string.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers/tokenizer.hpp"

namespace npue {

class WhisperTokenizer : public Tokenizer {
public:
  // `table` is the packed blob: `tokenizer.whisper_table` out of the
  // container, byte for byte. Throws if it is not a table this build
  // understands -- a wrong magic or version means the container was packed by
  // something else, and guessing at the layout is how a transcript comes out
  // as plausible garbage.
  explicit WhisperTokenizer(const void *table, size_t bytes);

  Encoded encode(const std::string &text, int max_len) const override;
  size_t vocab_size() const override { return tokens_.size(); }
  std::vector<std::string> tokenize(const std::string &text) const override;

  // Ids -> text, with the byte-level alphabet undone. This is the direction
  // that matters: an STT container is handed ids and owes a string.
  std::string decode(const std::vector<int32_t> &ids) const;

  // The control tokens, looked up BY NAME and refused when absent. A Whisper
  // that cannot emit <|endoftext|> has no stopping condition, and one that
  // cannot find <|notimestamps|> produces timestamp soup -- so the ids are
  // fetched once, here, where the failure can be a throw.
  int32_t token_id(const std::string &token) const;

  size_t n_merges() const { return ranks_.size(); }

private:
  // One code point of the input, decoded from UTF-8. `len` is the byte count
  // consumed. An invalid byte decodes as U+FFFD with len 1: the pattern's
  // three classes all reject it, so it lands in the punctuation branch, which
  // is what a decoder that emitted a stray byte wants.
  struct Cp {
    uint32_t cp;
    size_t len;
  };
  static Cp next_cp(const std::string &s, size_t i);

  // The GPT-2 pattern, hand-rolled. std::regex cannot run it: libstdc++ has no
  // \p{L} and mis-handles the lookahead. See
  // tools/gen/gen_whisper_unicode_tables.py. Order of the alternatives is the
  // pattern's, and it is load-bearing.
  std::vector<std::string> pretokenize(const std::string &text) const;

  // Entry indices for one pretoken, BPE'd. `piece` is ORIGINAL text; the
  // byte-level encoding happens here, not before the split.
  std::vector<uint32_t> bpe(const std::string &piece) const;

  std::string token_bytes(uint32_t entry) const;

  std::vector<std::string> tokens_;              // entry order, not id order
  std::vector<int32_t> ids_;                     // dense, 0..n-1
  std::unordered_map<std::string, uint32_t> index_;
  // rank -> merged entry, and the same keyed by the pair of entries, which is
  // what the inner loop asks. Keyed by entry rather than by string so the hot
  // loop is a hash of two integers.
  std::vector<std::pair<uint32_t, uint32_t>> merges_;
  std::unordered_map<uint64_t, uint32_t> ranks_;
  // GPT-2's reversible byte <-> printable-codepoint map, built the same way
  // whisper_bpe.bytes_to_unicode() builds it. The decoder is indexed by CODE
  // POINT, not by byte: the alphabet's characters above U+007F are stored in
  // the vocabulary as multi-byte UTF-8, so indexing it with a raw byte would
  // map 'Ġ' to two wrong bytes instead of one right one.
  uint32_t byte_encoder_[256]{};                 // byte -> codepoint
  int32_t byte_decoder_[512]{};                  // codepoint -> byte, -1 absent
};

}  // namespace npue
