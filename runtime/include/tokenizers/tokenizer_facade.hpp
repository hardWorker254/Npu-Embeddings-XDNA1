//===- tokenizer_facade.hpp --------------------------------------------*- C++ -*-===//
//
// One tokenizer interface, three tokenizer families behind it (WordPiece,
// XLM-R Unigram and Gemma SentencePiece BPE). For the BERT family the
// max_len / padding / truncation semantics here are this runtime's policy,
// not a property of the algorithms; Gemma carries its own <bos>/<eos>
// semantics inside GemmaTokenizer and the base encode() just forwards to it.
//
// Split out of main.cpp verbatim.
//
// This facade IS the tokenizer selector: `load_tokenizer()` chooses by the
// container's `arch` string and returns the one `AnyTokenizer` the call sites
// use. There is deliberately no separate tokenizer registry -- an earlier
// `tokenizer_registry.hpp` was dropped because every construction takes a
// `npue::File &` / raw bytes, not a path, so a path-keyed factory registry
// could not express the two call shapes here. Add a new family by extending
// `AnyTokenizer` and `load_tokenizer()`.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "runtime/model.hpp"
#include "tokenizers/gemma.hpp"
#include "tokenizers/tokenizer.hpp"
#include "tokenizers/wordpiece.hpp"
#include "tokenizers/xlmr.hpp"

namespace app {

// The vocabulary lives inside the .npue as of 0036, so a deployed model is
// ONE file. A model packed before that still works: fall back to the loose
// vocab.txt and say so, rather than failing on a file that is merely older.
// One tokenizer interface for the encode path, three tokenizer families
// behind it (tasks/0136). The facade lives HERE rather than giving
// XlmrTokenizer a WordPiece-shaped encode(), because the max_len /
// padding / truncation semantics are this runtime's policy (0110's
// refuse-on-overflow contract runs on the Encoded fields), not a property
// of the Unigram algorithm -- tokenizer_xlmr.cpp stays the line-for-line
// port of its Python reference, diffable function by function. Chosen over
// branching at the call sites because the encode() calls sit inside
// EmbedService::chunk() and the --tokenize loop, and a branch at each
// would be the drift-prone shape encoder_implemented() exists to prevent.
struct AnyTokenizer : npue::Tokenizer {
  std::unique_ptr<npue::WordPiece> wordpiece;
  std::unique_ptr<npue::XlmrTokenizer> xlmr;
  std::unique_ptr<npue::GemmaTokenizer> gemma;

  size_t vocab_size() const override {
    if (wordpiece) return wordpiece->vocab_size();
    if (xlmr) return xlmr->vocab_size();
    return gemma->vocab_size();
  }

  std::vector<std::string> tokenize(const std::string &text) const override {
    if (wordpiece) return wordpiece->tokenize(text);
    if (xlmr) return xlmr->tokenize(text);
    return gemma->tokenize(text);
  }

  npue::Encoded encode(const std::string &text, int max_len) const override {
    if (wordpiece) return wordpiece->encode(text, max_len);
    // Gemma BPE: the concrete GemmaTokenizer::encode already applies its own
    // <bos> + prefix + text + <eos> padding, so forward through the base
    // reference -- the (text, max_len) and (text, max_len, prefix) overloads
    // would otherwise make a two-argument call ambiguous.
    if (gemma)
      return static_cast<const npue::Tokenizer &>(*gemma).encode(text, max_len);
    // XLM-R Unigram. XlmrTokenizer::encode_raw() returns the FULL <s>...</s>
    // sequence, unpadded and untruncated (its header: an input that does
    // not fit is the caller's error to raise, not the tokenizer's to
    // hide). This adds the WordPiece path's exact max_len semantics on
    // top: truncation keeps <s> + the first (max_len - 2) pieces + </s>,
    // which is HuggingFace's longest_first truncation under the
    // "<s> A </s>" post-processor, so --tokenize stays diffable against
    // AutoTokenizer. n_tokens_full/truncated feed check_truncation()
    // unchanged -- 0110's refuse-on-overflow applies to arch=3 exactly as
    // to arch=0/2.
    std::vector<int32_t> full = xlmr->encode_raw(text);
    npue::Encoded e;
    e.n_tokens_full = static_cast<int32_t>(full.size());
    e.truncated = e.n_tokens_full > max_len;
    if (e.truncated) {
      full.resize(static_cast<size_t>(max_len));
      full.back() = xlmr->eos_id;
    }
    e.n_tokens = static_cast<int32_t>(full.size());
    e.input_ids = std::move(full);
    e.input_ids.resize(static_cast<size_t>(max_len), xlmr->pad_id);
    e.attention_mask.assign(static_cast<size_t>(max_len), 0);
    for (int32_t s = 0; s < e.n_tokens; ++s) e.attention_mask[s] = 1;
    e.token_type_ids.assign(static_cast<size_t>(max_len), 0);
    return e;
  }
};

inline AnyTokenizer load_tokenizer(npue::File &model,
                               const std::string &model_path) {
  // The ARCH decides, not the presence of a tokenizer key -- the Gemma and
  // arch=3 (XLMRTOK1) blobs below are stored whole in the container
  // (tasks/0135) and consumed in place, while a missing tokenizer.vocab
  // already means something else (the pre-0036 fallback).
  std::string arch;
  try {
    arch = model.config_string("arch");
  } catch (const std::exception &) {
    // Pre-arch container: WordPiece, like everything else that old.
  }
  if (arch == "gemma3_mqa_rope_geglu") {
    auto v = model.raw("tokenizer.gemma_table");
    AnyTokenizer t;
    t.gemma = std::make_unique<npue::GemmaTokenizer>(
        npue::GemmaTokenizer::from_table_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    return t;
  }
  if (arch == "gte_new_rope_geglu") {
    auto v = model.raw("tokenizer.xlmr_table");
    AnyTokenizer t;
    t.xlmr = std::make_unique<npue::XlmrTokenizer>(
        npue::XlmrTokenizer::from_table_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    // The facade's padding and the sequence template lean on XLM-R's
    // specials -- check the blob rather than assume it.
    if (t.xlmr->bos_id != 0 || t.xlmr->eos_id != 2 || t.xlmr->pad_id != 1)
      throw std::runtime_error(
          "tokenizer.xlmr_table specials are not XLM-R's <s>=0, </s>=2, "
          "<pad>=1 -- refusing rather than padding with the wrong id");
    return t;
  }
  try {
    auto v = model.raw("tokenizer.vocab");
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::WordPiece>(
        npue::WordPiece::from_vocab_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    return t;
  } catch (const std::exception &) {
    // Pre-0036 container: the loose checkpoint directory beside it, derived
    // from the container's own name rather than assumed to be MiniLM's.
    const std::string p =
        std::filesystem::path(model_path).replace_extension().string() +
        "/vocab.txt";
    std::printf("  tokenizer  .npue has no vocabulary; using %s\n", p.c_str());
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::WordPiece>(
        npue::WordPiece::from_vocab_file(p));
    return t;
  }
}

}  // namespace app