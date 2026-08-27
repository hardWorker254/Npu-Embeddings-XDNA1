# NpuEmbeddings -- pure-Python reference encoder for the XLM-R SentencePiece
# Unigram tokenizer, reading the XLMRTOK1 blob that
# tools/gen_xlmr_tokenizer_table.py writes.
# SPDX-License-Identifier: Apache-2.0
#
# THIS FILE IS THE EXECUTABLE SPEC for the future C++ port (T52 step 2).
# It consumes the *blob*, not tokenizer.json, so it exercises the exact
# bytes the C++ tokenizer will read, and tools/verify_tokenizer_xlmr.py
# holds it byte-exact against HuggingFace over a multilingual adversarial
# corpus. Every semantic choice below was confirmed against the live
# `tokenizers` 0.22.2 backend, not inferred from documentation; where
# HuggingFace deviates from upstream sentencepiece, HuggingFace wins,
# because that is what the golden embeddings were produced with.
#
# PIPELINE (mirrors tokenizers-rs, quirks included, in order):
#   1. Precompiled normalizer (normalizers/precompiled.rs + the
#      spm_precompiled crate): iterate UAX #29 extended grapheme clusters;
#      a grapheme < 6 UTF-8 bytes is first tried WHOLE against the Darts
#      double-array trie via common-prefix search, and the FIRST (i.e.
#      shortest) match's replacement substitutes the ENTIRE grapheme --
#      even when the match covers only a prefix of it (confirmed live:
#      'ẛ̣' U+1E9B+U+0323 -> 'ṡ', the dot below dropped). If the grapheme
#      is >= 6 bytes or has no match, each char is tried independently.
#   2. WhitespaceSplit: split on Unicode White_Space (Rust
#      char::is_whitespace -- NOT Python str.isspace(), which also claims
#      U+001C..U+001F), delimiters removed.
#   3. Metaspace(replacement '▁', prepend_scheme always, split true):
#      per word, replace ' ' with '▁' (a no-op after WhitespaceSplit),
#      prepend '▁' unless the word already starts with it, then split at
#      every '▁' with the '▁' merged into what follows.
#   4. Unigram Viterbi per pre-token (models/unigram/model.rs
#      encode_optimized): forward DP over char positions, f64 sums,
#      strict > relaxation (first-seen wins ties; starts ascending, piece
#      lengths ascending). Where no single-char piece matches, an <unk>
#      node covering exactly one char scores min_score - 10.0. Runs of
#      <unk> ids are fused into one (fuse_unk, confirmed live).
#   5. Post-processor: <s> ids </s>.
#
# KNOWN LIMITATION (deliberate, shared with the future C++ port): the
# added-tokens splitter is NOT implemented, so an input containing a
# literal special-token string ("<s>", "<mask>", ...) will be tokenized
# through the Unigram model instead of being extracted verbatim. The
# verifier's corpus therefore excludes literal special-token strings;
# embedding server inputs are plain text, never templated.
#
# Env: any Python 3, stdlib only (struct + unicodedata).
# Usage (CLI, mirrors what the C++ CLI will do):
#   python tools\xlmr_tokenizer_ref.py <table.bin> <texts.txt>
#     -- one text per line in, one space-separated id line out per text.

from __future__ import annotations

import struct
import sys
import unicodedata

MAGIC = b"XLMRTOK1"
VERSION = 1


# --------------------------------------------------------------------------
# UAX #29 extended grapheme clusters -- approximation sufficient for the
# Precompiled normalizer. Only multi-char graphemes SHORTER than 6 UTF-8
# bytes change behaviour (longer ones fall back to per-char anyway), so the
# properties that matter are combining marks, Hangul jamo, regional-indicator
# pairs and ZWJ emoji joins. unicodedata has no Grapheme_Cluster_Break table;
# the classes below are built from general categories plus the small explicit
# ranges, biased toward OVER-joining (which is behaviour-neutral: an
# over-long grapheme >= 6 bytes takes the same per-char path).
# --------------------------------------------------------------------------

_PREPEND = frozenset(
    list(range(0x0600, 0x0606)) + [0x06DD, 0x070F, 0x0890, 0x0891, 0x08E2,
                                   0x0D4E, 0x110BD, 0x110CD]
)
_ZWJ = 0x200D


def _gcb_class(ch: str) -> str:
    cp = ord(ch)
    if cp == 0x000D:
        return "CR"
    if cp == 0x000A:
        return "LF"
    if cp == _ZWJ:
        return "ZWJ"
    if cp == 0x200C:
        return "Extend"  # ZWNJ
    if cp in _PREPEND:
        return "Prepend"
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return "RI"
    # Hangul
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LV" if (cp - 0xAC00) % 28 == 0 else "LVT"
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me"):
        return "Extend"
    if cat == "Mc":
        return "SpacingMark"
    if cat in ("Cc", "Zl", "Zp") or cat == "Cf":
        return "Control"
    return "Other"


def _is_ext_pict(ch: str) -> bool:
    # Extended_Pictographic approximation for GB11 (ZWJ emoji sequences).
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FFFD
        or 0x2600 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or cp in (0x00A9, 0x00AE, 0x2122, 0x203C, 0x2049, 0x2139)
        or 0x2190 <= cp <= 0x21FF
        or 0x2300 <= cp <= 0x23FF
        or 0x25A0 <= cp <= 0x25FF
        or 0x2900 <= cp <= 0x297F
        or 0x3030 <= cp <= 0x303D
        or cp in (0x3297, 0x3299)
        or 0xFE00 <= cp <= 0xFE0F  # variation selectors ride along anyway (Mn)
    )


def graphemes(s: str):
    """Yield extended grapheme clusters of s (UAX #29 approximation)."""
    if not s:
        return
    cls = [_gcb_class(c) for c in s]
    start = 0
    ri_run = 0
    for i in range(1, len(s)):
        left, right = cls[i - 1], cls[i]
        if left == "RI":
            ri_run += 1
        else:
            ri_run = 0
        brk: bool
        if left == "CR" and right == "LF":
            brk = False                                     # GB3
        elif left in ("CR", "LF", "Control"):
            brk = True                                      # GB4
        elif right in ("CR", "LF", "Control"):
            brk = True                                      # GB5
        elif left == "L" and right in ("L", "V", "LV", "LVT"):
            brk = False                                     # GB6
        elif left in ("LV", "V") and right in ("V", "T"):
            brk = False                                     # GB7
        elif left in ("LVT", "T") and right == "T":
            brk = False                                     # GB8
        elif right in ("Extend", "ZWJ"):
            brk = False                                     # GB9
        elif right == "SpacingMark":
            brk = False                                     # GB9a
        elif left == "Prepend":
            brk = False                                     # GB9b
        elif left == "ZWJ" and _is_ext_pict(s[i]):
            brk = False                                     # GB11 (approx)
        elif left == "RI" and right == "RI" and ri_run % 2 == 1:
            brk = False                                     # GB12/13
        else:
            brk = True                                      # GB999
        if brk:
            yield s[start:i]
            start = i
    yield s[start:]


# --------------------------------------------------------------------------
# Unicode White_Space, as Rust char::is_whitespace sees it. Python's
# str.isspace() additionally claims U+001C..U+001F, which Rust does not.
# --------------------------------------------------------------------------
_WHITE_SPACE = frozenset(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)


def _is_whitespace(ch: str) -> bool:
    return ord(ch) in _WHITE_SPACE


# --------------------------------------------------------------------------
# Darts double-array trie (darts_clone as embedded in sentencepiece and the
# spm_precompiled crate). Units are u32 LE:
#   has_leaf(u) = (u >> 8) & 1
#   value(u)    = u & 0x7FFFFFFF          (on the leaf unit)
#   label(u)    = u & 0x800000FF          (compared against the key byte)
#   offset(u)   = (u >> 10) << ((u & 0x200) >> 6)
# --------------------------------------------------------------------------
class DartsTrie:
    def __init__(self, blob: bytes):
        if len(blob) % 4 != 0:
            raise ValueError("trie blob not a multiple of 4 bytes")
        self.units = struct.unpack(f"<{len(blob) // 4}I", blob)

    def common_prefix_search(self, key: bytes) -> list[int]:
        units = self.units
        results: list[int] = []
        node_pos = 0
        unit = units[node_pos]
        node_pos ^= (unit >> 10) << ((unit & 0x200) >> 6)
        for c in key:
            node_pos ^= c
            unit = units[node_pos]
            if (unit & 0x800000FF) != c:
                return results
            node_pos ^= (unit >> 10) << ((unit & 0x200) >> 6)
            if (unit >> 8) & 1:
                results.append(units[node_pos] & 0x7FFFFFFF)
        return results


class XlmrTokenizer:
    """Reference implementation over the XLMRTOK1 blob."""

    def __init__(self, table_path: str):
        with open(table_path, "rb") as f:
            blob = f.read()
        if blob[:8] != MAGIC:
            raise ValueError(f"bad magic {blob[:8]!r}, want {MAGIC!r}")
        off = 8
        (version, self.vocab_size, self.unk_id, self.bos_id, self.eos_id,
         self.pad_id, self.mask_id, self.add_bos, self.add_eos,
         metaspace_cp, self.prepend_scheme, self.metaspace_split,
         self.max_piece_bytes, self.max_piece_chars) = \
            struct.unpack_from("<14I", blob, off)
        off += 14 * 4
        if version != VERSION:
            raise ValueError(f"table version {version}, want {VERSION}")
        self.min_score, self.unk_score = struct.unpack_from("<dd", blob, off)
        off += 16
        trie_size, norm_size = struct.unpack_from("<II", blob, off)
        off += 8
        self.trie = DartsTrie(blob[off:off + trie_size])
        off += trie_size
        self.normalized_blob = blob[off:off + norm_size]
        off += norm_size
        self.scores = struct.unpack_from(f"<{self.vocab_size}d", blob, off)
        off += self.vocab_size * 8
        pieces: list[str] = []
        for _ in range(self.vocab_size):
            (n,) = struct.unpack_from("<H", blob, off)
            off += 2
            pieces.append(blob[off:off + n].decode("utf-8"))
            off += n
        if off != len(blob):
            raise ValueError(f"trailing bytes: read {off} of {len(blob)}")
        self.pieces = pieces
        self.piece_to_id = {p: i for i, p in enumerate(pieces)}
        self.metaspace = chr(metaspace_cp)

    # -- 1. Precompiled normalizer ------------------------------------
    def _transform(self, chunk: str) -> str | None:
        results = self.trie.common_prefix_search(chunk.encode("utf-8"))
        if not results:
            return None
        index = results[0]  # FIRST match, as spm_precompiled does
        end = self.normalized_blob.index(b"\x00", index)
        return self.normalized_blob[index:end].decode("utf-8")

    def normalize(self, text: str) -> str:
        out: list[str] = []
        for g in graphemes(text):
            if len(g.encode("utf-8")) < 6:
                norm = self._transform(g)
                if norm is not None:
                    out.append(norm)  # replaces the WHOLE grapheme
                    continue
            for ch in g:
                norm = self._transform(ch)
                out.append(ch if norm is None else norm)
        return "".join(out)

    # -- 2 + 3. WhitespaceSplit then Metaspace ------------------------
    def pre_tokenize(self, normalized: str) -> list[str]:
        words: list[str] = []
        cur: list[str] = []
        for ch in normalized:
            if _is_whitespace(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))

        rep = self.metaspace
        pretokens: list[str] = []
        for word in words:
            # replace(' ', rep) is a no-op here: WhitespaceSplit removed them
            if self.prepend_scheme == 2 and not word.startswith(rep):
                word = rep + word
            elif self.prepend_scheme == 1 and not pretokens and \
                    not word.startswith(rep):
                word = rep + word
            if self.metaspace_split:
                # split on rep, delimiter merged with what FOLLOWS it
                parts: list[str] = []
                cur_part = ""
                for ch in word:
                    if ch == rep and cur_part:
                        parts.append(cur_part)
                        cur_part = ch
                    else:
                        cur_part += ch
                if cur_part:
                    parts.append(cur_part)
                pretokens.extend(parts)
            else:
                pretokens.append(word)
        return pretokens

    # -- 4. Unigram Viterbi (encode_optimized mirror) ------------------
    def viterbi(self, pretoken: str) -> list[int]:
        n = len(pretoken)
        if n == 0:
            return []
        NEG = float("-inf")
        best_score = [NEG] * (n + 1)
        best_start = [-1] * (n + 1)
        best_id = [-1] * (n + 1)
        best_score[0] = 0.0
        vocab = self.piece_to_id
        scores = self.scores
        maxc = self.max_piece_chars
        for start in range(n):
            here = best_score[start]
            if best_start[start] == -1 and start != 0:
                continue  # unreachable (cannot happen: unk covers every char)
            has_single = False
            top = min(n, start + maxc)
            for end in range(start + 1, top + 1):
                piece = pretoken[start:end]
                tid = vocab.get(piece)
                if tid is None:
                    continue
                cand = here + scores[tid]
                if best_start[end] == -1 or cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = start
                    best_id[end] = tid
                if end - start == 1:
                    has_single = True
            if not has_single:
                end = start + 1
                cand = here + self.unk_score
                if best_start[end] == -1 or cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = start
                    best_id[end] = self.unk_id
        # backtrack
        ids: list[int] = []
        pos = n
        while pos > 0:
            ids.append(best_id[pos])
            pos = best_start[pos]
        ids.reverse()
        # fuse consecutive <unk> (fuse_unk = true in the HF Unigram model)
        fused: list[int] = []
        for tid in ids:
            if tid == self.unk_id and fused and fused[-1] == self.unk_id:
                continue
            fused.append(tid)
        return fused

    # -- full pipeline -------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in self.pre_tokenize(self.normalize(text)):
            ids.extend(self.viterbi(pretoken))
        if self.add_bos:
            ids.insert(0, self.bos_id)
        if self.add_eos:
            ids.append(self.eos_id)
        return ids


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__ or "usage: xlmr_tokenizer_ref.py <table.bin> <texts.txt>",
              file=sys.stderr)
        return 2
    tok = XlmrTokenizer(sys.argv[1])
    with open(sys.argv[2], encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").rstrip("\r")
            print(" ".join(str(i) for i in tok.encode(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
