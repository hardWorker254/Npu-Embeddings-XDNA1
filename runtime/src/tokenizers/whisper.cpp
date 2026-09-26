//===- whisper.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Whisper tokenizer. See tokenizers/whisper.hpp for what this
// is and how it is verified; tools/whisper_tokenizer_ref.py is the executable
// specification it is written against.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "tokenizers/whisper.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "tokenizers/whisper_unicode_tables.hpp"

namespace npue {
namespace {

using whisper_unicode::is_letter;
using whisper_unicode::is_number;
using whisper_unicode::is_space;

// tools/whisper_bpe.py's header is the single definition of this layout; the
// field offsets below are transcribed from it and checked against the magic and
// version, because a reader that trusts a length field it has not verified
// reads whatever else is in the file.
constexpr char kMagic[4] = {'W', 'B', 'P', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 24;

uint32_t rd_u32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

const char *const kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

// The alphabet's codepoints top out just above U+0130, so a flat array beats
// a map here; the decode side indexes it with the codepoint it decoded.
constexpr size_t kMaxByteCp = 0x200;

std::string utf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

}  // namespace

WhisperTokenizer::WhisperTokenizer(const void *table, size_t bytes) {
  const auto *blob = static_cast<const uint8_t *>(table);
  if (bytes < kHeaderSize || std::memcmp(blob, kMagic, 4) != 0)
    throw std::runtime_error(
        "tokenizer.whisper_table: not a Whisper tokenizer table (magic)");
  if (rd_u32(blob + 4) != kVersion)
    throw std::runtime_error("tokenizer.whisper_table: version " +
                             std::to_string(rd_u32(blob + 4)) +
                             ", this build reads " + std::to_string(kVersion));
  const uint32_t n_vocab = rd_u32(blob + 8);
  const uint32_t n_merges = rd_u32(blob + 12);
  const uint32_t pool_bytes = rd_u32(blob + 16);

  // The size the header claims must fit the blob. Every field below is a
  // consequence of these offsets, so this one check is what makes the rest
  // safe to read.
  const uint64_t need = kHeaderSize + pool_bytes + 12ull * n_vocab +
                        8ull * n_merges;
  if (need > bytes)
    throw std::runtime_error(
        "tokenizer.whisper_table: header describes " + std::to_string(need) +
        " bytes, tensor holds " + std::to_string(bytes));

  const uint8_t *pool = blob + kHeaderSize;
  const uint8_t *offs = pool + pool_bytes;
  const uint8_t *lens = offs + 4ull * n_vocab;
  const uint8_t *ids = lens + 4ull * n_vocab;
  const uint8_t *merges = ids + 4ull * n_vocab;

  tokens_.reserve(n_vocab);
  ids_.reserve(n_vocab);
  for (uint32_t i = 0; i < n_vocab; ++i) {
    const uint32_t o = rd_u32(offs + 4ull * i), l = rd_u32(lens + 4ull * i);
    if (static_cast<uint64_t>(o) + l > pool_bytes)
      throw std::runtime_error("tokenizer.whisper_table: entry " +
                               std::to_string(i) + " runs past the pool");
    tokens_.emplace_back(reinterpret_cast<const char *>(pool + o), l);
    ids_.push_back(static_cast<int32_t>(rd_u32(ids + 4ull * i)));
  }
  for (uint32_t i = 0; i < n_vocab; ++i) index_.emplace(tokens_[i], i);
  // A duplicate string would make one of the two ids unreachable, and which one
  // wins would depend on iteration order. Refused rather than resolved.
  if (index_.size() != n_vocab)
    throw std::runtime_error(
        "tokenizer.whisper_table: " + std::to_string(n_vocab - index_.size()) +
        " duplicate token strings; the table cannot represent this vocabulary");

  merges_.reserve(n_merges);
  ranks_.reserve(n_merges * 2);
  for (uint32_t r = 0; r < n_merges; ++r) {
    const uint32_t a = rd_u32(merges + 8ull * r);
    const uint32_t b = rd_u32(merges + 8ull * r + 4);
    if (a >= n_vocab || b >= n_vocab)
      throw std::runtime_error("tokenizer.whisper_table: merge " +
                               std::to_string(r) + " names a missing entry");
    merges_.emplace_back(a, b);
    ranks_.emplace((static_cast<uint64_t>(a) << 32) | b, r);
  }

  // GPT-2's byte <-> printable-codepoint map, built as whisper_bpe.py builds
  // it: the bytes that are already printable map to themselves, the rest are
  // pushed above 0xFF so no two bytes collide.
  {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }
    for (size_t i = 0; i < kMaxByteCp; ++i) byte_decoder_[i] = -1;
    for (size_t i = 0; i < 256; ++i) {
      byte_encoder_[static_cast<size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
      byte_decoder_[static_cast<size_t>(cs[i])] = static_cast<int32_t>(bs[i]);
    }
  }
}

int32_t WhisperTokenizer::token_id(const std::string &token) const {
  const auto it = index_.find(token);
  if (it == index_.end())
    throw std::runtime_error(
        "this checkpoint's tokenizer has no " + token +
        ". A Whisper container without it cannot start or finish a "
        "transcription, and the id is not invented here because a wrong "
        "control token is a wrong transcript rather than an error.");
  return ids_[it->second];
}

WhisperTokenizer::Cp WhisperTokenizer::next_cp(const std::string &s, size_t i) {
  const auto b0 = static_cast<uint8_t>(s[i]);
  const size_t left = s.size() - i;
  auto cont = [&](size_t n) { return left >= n; };
  if (b0 < 0x80) return {b0, 1};
  auto is_cont = [&](uint8_t c) { return (c & 0xC0) == 0x80; };
  if ((b0 & 0xE0) == 0xC0 && cont(2) && is_cont(s[i + 1]))
    return {static_cast<uint32_t>(((b0 & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu)), 2};
  if ((b0 & 0xF0) == 0xE0 && cont(3) && is_cont(s[i + 1]) &&
      is_cont(s[i + 2]))
    return {static_cast<uint32_t>(((b0 & 0x0Fu) << 12) |
                                 ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu)),
            3};
  if ((b0 & 0xF8) == 0xF0 && cont(4) && is_cont(s[i + 1]) &&
      is_cont(s[i + 2]) && is_cont(s[i + 3]))
    return {static_cast<uint32_t>(((b0 & 0x07u) << 18) |
                                 ((s[i + 1] & 0x3Fu) << 12) |
                                 ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu)),
            4};
  return {0xFFFD, 1};  // not valid UTF-8; see the header
}

std::vector<std::string> WhisperTokenizer::pretokenize(
    const std::string &text) const {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < text.size()) {
    const size_t start = i;

    // 1. 's 't 're 've 'm 'll 'd -- no boundary requirement, exactly as the
    //    pattern has it. Checked in the pattern's own order.
    if (text[i] == '\'') {
      for (const char *c : kContractions) {
        const size_t n = std::strlen(c);
        if (text.compare(i, n, c) == 0) {
          out.emplace_back(text.substr(i, n));
          i += n;
          break;
        }
      }
      if (i != start) continue;
    }

    // A single leading SPACE, and only a space: the pattern's ` ?` is a
    // literal, not \s. Claimed only when a letter or a digit follows, which is
    // what makes the alternative fail and fall through when it does not.
    size_t j = i;
    if (text[j] == ' ') ++j;

    if (j < text.size()) {
      const Cp c = next_cp(text, j);
      if (is_letter(c.cp)) {  // 2. ?\p{L}+
        j += c.len;
        while (j < text.size()) {
          const Cp d = next_cp(text, j);
          if (!is_letter(d.cp)) break;
          j += d.len;
        }
        out.emplace_back(text.substr(start, j - start));
        i = j;
        continue;
      }
      if (is_number(c.cp)) {  // 3. ?\p{N}+
        j += c.len;
        while (j < text.size()) {
          const Cp d = next_cp(text, j);
          if (!is_number(d.cp)) break;
          j += d.len;
        }
        out.emplace_back(text.substr(start, j - start));
        i = j;
        continue;
      }
      if (!is_space(c.cp)) {  // 4. ?[^\s\p{L}\p{N}]+[\r\n]*
        j += c.len;
        while (j < text.size()) {
          const Cp d = next_cp(text, j);
          if (is_space(d.cp) || is_letter(d.cp) || is_number(d.cp)) break;
          j += d.len;
        }
        while (j < text.size() && (text[j] == '\r' || text[j] == '\n')) ++j;
        out.emplace_back(text.substr(start, j - start));
        i = j;
        continue;
      }
    }

    // Neither a letter, a digit nor punctuation, and the optional space did
    // not help: whitespace. 5. \s+(?!\S) then 6. \s+.
    // The first takes the whole run only when nothing non-space follows it;
    // otherwise it gives the last space back, which is the one the following
    // word claims as its own ` ?`.
    size_t k = i;
    for (;;) {
      if (k >= text.size()) break;
      const Cp d = next_cp(text, k);
      if (!is_space(d.cp)) break;
      k += d.len;
    }
    size_t take = k - i;
    if (take > 0 && k < text.size() && !is_space(next_cp(text, k).cp)) --take;
    if (take == 0) take = k - i;  // a single space before a word: branch 6
    out.emplace_back(text.substr(i, take));
    i += take;
  }
  return out;
}

std::vector<uint32_t> WhisperTokenizer::bpe(const std::string &piece) const {
  // Byte-encoding happens HERE, on the pretoken, never before the split: the
  // byte-level alphabet is not the original one ('Ġ' is a letter and '²' is
  // not), so splitting first and encoding after is the only order that gives
  // the ids the reference does.
  std::string form;
  form.reserve(piece.size());
  for (unsigned char c : piece) form += utf8(byte_encoder_[c]);

  if (const auto whole = index_.find(form); whole != index_.end())
    return {whole->second};

  std::vector<uint32_t> syms;
  syms.reserve(form.size());
  for (unsigned char c : piece) {
    // Every single byte is in a GPT-2 vocabulary; a miss means the table is not
    // the one this was written for, and a wrong id is worse than a throw.
    const auto it = index_.find(utf8(byte_encoder_[c]));
    if (it == index_.end())
      throw std::runtime_error("tokenizer.whisper_table: no entry for byte " +
                               std::to_string(static_cast<int>(c)));
    syms.push_back(it->second);
  }

  // Repeatedly merge the adjacent pair with the LOWEST rank -- not the first
  // one found, and not the longest: rank order is the whole of BPE.
  std::string merged;
  while (syms.size() > 1) {
    size_t best = syms.size();
    uint32_t best_rank = 0;
    bool found = false;
    for (size_t k = 0; k + 1 < syms.size(); ++k) {
      const auto it = ranks_.find((static_cast<uint64_t>(syms[k]) << 32) |
                                  syms[k + 1]);
      if (it == ranks_.end()) continue;
      if (!found || it->second < best_rank) {
        found = true;
        best = k;
        best_rank = it->second;
      }
    }
    if (!found) break;
    merged = tokens_[syms[best]] + tokens_[syms[best + 1]];
    const auto it = index_.find(merged);
    if (it == index_.end())
      throw std::runtime_error("tokenizer.whisper_table: merge produced '" +
                               merged + "', which is not an entry");
    syms[best] = it->second;
    syms.erase(syms.begin() + static_cast<long>(best) + 1);
  }
  return syms;
}

Encoded WhisperTokenizer::encode(const std::string &text, int max_len) const {
  std::vector<int32_t> ids;
  for (const std::string &piece : pretokenize(text))
    for (uint32_t e : bpe(piece)) ids.push_back(ids_[e]);

  Encoded out;
  out.n_tokens_full = static_cast<int64_t>(ids.size());
  if (max_len > 0 && ids.size() > static_cast<size_t>(max_len)) {
    // Reported, never silent. A truncated transcript is not a shorter
    // transcript, it is a different one, and this is the only place the caller
    // can find out.
    out.truncated = true;
    ids.resize(static_cast<size_t>(max_len));
  }
  out.n_tokens = static_cast<int64_t>(ids.size());
  out.input_ids = ids;
  out.attention_mask.assign(ids.size(), 1);
  // Whisper's decoder is decoder-only: there is no segment pairing, so the
  // third stream does not exist. Emitted as zeros of the same length because
  // the Encoded contract has three vectors and a caller that sizes them
  // differently per model is a caller that will read one model's padding as
  // another's mask.
  out.token_type_ids.assign(ids.size(), 0);
  return out;
}

std::vector<std::string> WhisperTokenizer::tokenize(
    const std::string &text) const {
  std::vector<std::string> out;
  for (const std::string &piece : pretokenize(text))
    for (uint32_t e : bpe(piece)) out.push_back(tokens_[e]);
  return out;
}

std::string WhisperTokenizer::token_bytes(uint32_t entry) const {
  std::string raw;
  const std::string &tok = tokens_[entry];
  raw.reserve(tok.size());
  for (size_t i = 0; i < tok.size();) {
    const Cp c = next_cp(tok, i);
    if (c.cp >= kMaxByteCp || byte_decoder_[c.cp] < 0)
      throw std::runtime_error(
          "tokenizer.whisper_table: token '" + tok +
          "' contains a character outside the byte-level alphabet");
    raw.push_back(static_cast<char>(byte_decoder_[c.cp]));
    i += c.len;
  }
  return raw;
}

std::string WhisperTokenizer::decode(const std::vector<int32_t> &ids) const {
  // ids_ is dense by construction -- the packer refuses a vocabulary with a
  // gap, because the decoder's logit indexing assumes one -- so the id IS the
  // entry index. A lookup table here would hide a violated invariant instead
  // of failing on it.
  std::string raw;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= ids_.size())
      throw std::runtime_error("decode: id " + std::to_string(id) +
                               " is outside the vocabulary");
    raw += token_bytes(static_cast<uint32_t>(id));
  }
  return raw;
}

}  // namespace npue
