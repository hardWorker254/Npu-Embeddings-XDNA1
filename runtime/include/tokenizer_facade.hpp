//===- tokenizer_facade.hpp -----------------------------------------------*- C++ -*-===//
//
// One tokenizer interface for the BERT-family encode path, two tokenizer
// families behind it (WordPiece and XLM-R Unigram). The max_len / padding /
// truncation semantics here are this runtime's policy, not a property of
// the algorithms.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "npue.hpp"
#include "tokenizer.hpp"
#include "tokenizer_xlmr.hpp"

namespace app {

// The vocabulary lives inside the .npue as of 0036, so a deployed model is
// ONE file. A model packed before that still works: fall back to the loose
// vocab.txt and say so, rather than failing on a file that is merely older.
// One tokenizer interface for the BERT-family encode path, two tokenizer
// families behind it (tasks/0136). The facade lives HERE rather than giving
// XlmrTokenizer a WordPiece-shaped encode(), because the max_len /
// padding / truncation semantics are this runtime's policy (0110's
// refuse-on-overflow contract runs on the Encoded fields), not a property
// of the Unigram algorithm -- tokenizer_xlmr.cpp stays the line-for-line
// port of its Python reference, diffable function by function. Chosen over
// branching at the call sites because the encode() calls sit inside
// EmbedService::chunk() and the --tokenize loop, and a branch at each
// would be the drift-prone shape encoder_implemented() exists to prevent.
struct AnyTokenizer {
  std::unique_ptr<npue::Tokenizer> wordpiece;
  std::unique_ptr<npue::XlmrTokenizer> xlmr;

  size_t vocab_size() const {
    return wordpiece ? wordpiece->vocab_size() : xlmr->vocab_size();
  }

  npue::Encoded encode(const std::string &text, int max_len) const {
    if (wordpiece) return wordpiece->encode(text, max_len);
    // XLM-R Unigram. XlmrTokenizer::encode() returns the FULL <s>...</s>
    // sequence, unpadded and untruncated (its header: an input that does
    // not fit is the caller's error to raise, not the tokenizer's to
    // hide). This adds the WordPiece path's exact max_len semantics on
    // top: truncation keeps <s> + the first (max_len - 2) pieces + </s>,
    // which is HuggingFace's longest_first truncation under the
    // "<s> A </s>" post-processor, so --tokenize stays diffable against
    // AutoTokenizer. n_tokens_full/truncated feed check_truncation()
    // unchanged -- 0110's refuse-on-overflow applies to arch=3 exactly as
    // to arch=0/2.
    std::vector<int32_t> full = xlmr->encode(text);
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
  // arch=3: the XLMRTOK1 Unigram blob, stored whole in the container
  // (tasks/0135) and consumed in place. The ARCH decides, not the absence
  // of a tokenizer.vocab key -- absence already means something else below.
  std::string arch;
  try {
    arch = model.config_string("arch");
  } catch (const std::exception &) {
    // Pre-arch container: WordPiece, like everything else that old.
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
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_bytes(
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
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_file(p));
    return t;
  }
}

}  // namespace app
