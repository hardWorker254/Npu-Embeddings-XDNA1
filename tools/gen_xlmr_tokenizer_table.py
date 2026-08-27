# NpuEmbeddings -- generate the binary SentencePiece-Unigram table the future
# C++ XLM-R tokenizer (T52 step 2) will read at load time.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS CHECKPOINT'S TOKENIZER ACTUALLY IS
# ---------------------------------------------
# gte-multilingual-base (Alibaba-NLP) is an XLM-R-family checkpoint. Reading
# models/gte-multilingual-base/tokenizer.json directly (rule learned in T29:
# read the file before writing code) shows `model.type` is **"Unigram"** --
# 250,002 (piece, log-prob) pairs, unk_id 3, byte_fallback false. This is the
# third tokenizer family in this repo, sharing no algorithm with WordPiece
# (runtime/src/tokenizer.cpp) or SentencePiece BPE
# (runtime/src/tokenizer_gemma.cpp).
#
# THE FULL PIPELINE, CONFIRMED EMPIRICALLY (probed the live `tokenizers`
# 0.22.2 backend in .venv-ref, not inferred from docs)
# --------------------------------------------------------------------
#   normalizer     type "Precompiled": sentencepiece's serialized nmt_nfkc
#                  normalization, shipped as base64 `precompiled_charsmap`.
#                  Layout: u32 LE trie-block size N, then N bytes of
#                  double-array trie (Darts, u32 LE units), then a blob of
#                  NUL-terminated replacement strings; a trie hit's value is
#                  a byte offset into that blob. HuggingFace's semantics
#                  (which the verifier holds us to) are NOT sentencepiece's
#                  longest-match walk: tokenizers' precompiled.rs iterates
#                  UAX #29 extended grapheme clusters; a grapheme < 6 bytes
#                  whose *prefix* hits the trie is replaced WHOLE by the
#                  first (shortest) match -- confirmed live: 'ẛ̣' (U+1E9B
#                  U+0323, 5 bytes) normalizes to 'ṡ', the combining dot
#                  below silently dropped. Otherwise each char is transformed
#                  independently. The blob stores the decoded trie +
#                  strings verbatim; semantics live in the reference
#                  encoder (tools/xlmr_tokenizer_ref.py) and later the C++.
#   pre_tokenizer  Sequence [WhitespaceSplit, Metaspace('▁',
#                  prepend_scheme=always, add_prefix_space=true)]. Confirmed:
#                  each whitespace-separated word gets ▁ prepended UNLESS it
#                  already starts with ▁, then is split at every literal ▁
#                  with the ▁ merged into what follows ('a▁b' pre-tokenizes
#                  to ['▁a', '▁b']).
#   model          Unigram Viterbi, f64 log-prob sums. A char at which no
#                  single-char piece matches contributes an <unk> node with
#                  score min_score - 10.0 (tokenizers' kUnkPenalty), and
#                  consecutive <unk> outputs are FUSED into one id --
#                  confirmed live: 'x𒀱𒀲y' -> ['▁x', '<unk>', 'y'], one
#                  <unk> for two unknown codepoints.
#   post_processor TemplateProcessing single = [<s>, A, </s>], ids 0 and 2.
#                  ids: <s>=0 <pad>=1 </s>=2 <unk>=3 <mask>=250001.
#
# WHY THE SCORES ARE STORED AS f64, NOT f32
# ------------------------------------------
# 65,856 of the 250,002 log-probs in tokenizer.json do not round-trip
# through float32 (e.g. 'k' = -7.471577644348144). HuggingFace parses JSON
# numbers as f64 and sums f64 in Viterbi; storing f32 would perturb
# near-tie segmentations and break byte-exactness. 2 MB of f64 is the
# price of an honest comparison. Checked below; if a future checkpoint's
# scores all become f32-exact this stays f64 anyway -- the C++ reader is
# simpler with one code path.
#
# WHY A BINARY TABLE, NOT JSON AT RUNTIME
# ----------------------------------------
# CLAUDE.md rule 5: Python is build-time only; the shipped runtime is C++
# with no JSON parser. This script reads tokenizer.json (17 MB) ONCE,
# offline, and writes a flat binary the C++ side reads with ifstream +
# memcpy -- the exact role tools/gen_gemma_tokenizer_table.py plays for
# EmbeddingGemma.
#
# XLMRTOK1 FORMAT (all little-endian)
# ------------------------------------
#   magic            8 bytes  "XLMRTOK1"
#   version          u32      1
#   vocab_size       u32
#   unk_id           u32
#   bos_id           u32      <s>
#   eos_id           u32      </s>
#   pad_id           u32
#   mask_id          u32
#   add_bos          u32      1 iff post_processor prepends <s>
#   add_eos          u32      1 iff post_processor appends </s>
#   metaspace_char   u32      codepoint of the replacement (0x2581 '▁')
#   prepend_scheme   u32      0=never 1=first 2=always
#   metaspace_split  u32      1 iff Metaspace splits on the replacement
#   max_piece_bytes  u32      longest vocab piece in UTF-8 bytes
#   max_piece_chars  u32      longest vocab piece in codepoints
#   min_score        f64      min over all vocab scores
#   unk_score        f64      min_score - 10.0
#   trie_size        u32      charsmap double-array size in bytes
#   norm_size        u32      charsmap normalized-strings blob size in bytes
#   trie             trie_size bytes, verbatim (u32 LE Darts units)
#   normalized       norm_size bytes, verbatim (NUL-terminated strings)
#   scores           vocab_size x f64, id order
#   pieces           vocab_size x (u16 byte-length + UTF-8 bytes), id order
#
# Env: any Python 3 (json + struct + base64 only; transformers is only
# needed by the verify script that checks this table against ground truth).
# Usage:
#   python tools\gen_xlmr_tokenizer_table.py
#     [--tokenizer-json PATH] [--out PATH]

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_JSON = REPO / "models" / "gte-multilingual-base" / "tokenizer.json"
DEFAULT_OUT = REPO / "models" / "gte-multilingual-base" / "xlmr_tokenizer.bin"

FORMAT_MAGIC = b"XLMRTOK1"  # 8 bytes, on disk verbatim
VERSION = 1
UNK_PENALTY = 10.0  # tokenizers' unigram kUnkPenalty, model.rs

PREPEND_SCHEMES = {"never": 0, "first": 1, "always": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-json", default=str(DEFAULT_TOKENIZER_JSON))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tok_path = Path(args.tokenizer_json)
    out_path = Path(args.out)

    with tok_path.open(encoding="utf-8") as f:
        tok = json.load(f)

    # --- model: must be Unigram, and shaped the way this generator expects
    model = tok["model"]
    if model.get("type") != "Unigram":
        raise SystemExit(
            f"expected model.type == 'Unigram', got {model.get('type')!r} -- "
            "this generator implements SentencePiece Unigram specifically "
            "(the BPE generator is tools/gen_gemma_tokenizer_table.py); "
            "re-check tokenizer.json before proceeding"
        )
    if model.get("byte_fallback"):
        raise SystemExit(
            "expected byte_fallback: false -- the Viterbi <unk> path below "
            "assumes no byte pieces; a checkpoint with byte_fallback needs "
            "new code, not this table"
        )
    unk_id = model.get("unk_id")
    vocab: list = model["vocab"]
    vocab_size = len(vocab)
    if not isinstance(unk_id, int) or not (0 <= unk_id < vocab_size):
        raise SystemExit(f"unk_id {unk_id!r} not an id in [0, {vocab_size})")

    pieces: list[str] = []
    scores: list[float] = []
    seen: set[str] = set()
    for row in vocab:
        if not (isinstance(row, list) and len(row) == 2):
            raise SystemExit(f"vocab row not a [piece, score] pair: {row!r}")
        piece, score = row
        if piece in seen:
            raise SystemExit(
                f"duplicate vocab piece {piece!r} -- the C++ hash lookup "
                "would silently keep one of the two scores"
            )
        seen.add(piece)
        pieces.append(piece)
        scores.append(float(score))

    n_not_f32 = sum(
        1 for s in scores if struct.unpack("<f", struct.pack("<f", s))[0] != s
    )
    min_score = min(scores)
    unk_score = min_score - UNK_PENALTY

    # --- normalizer: Precompiled charsmap, decoded and sanity-checked
    norm = tok.get("normalizer") or {}
    if norm.get("type") != "Precompiled" or "precompiled_charsmap" not in norm:
        raise SystemExit(
            f"expected normalizer type 'Precompiled' with a "
            f"precompiled_charsmap, got {norm.get('type')!r} -- the charsmap "
            "trie below would not apply"
        )
    charsmap = base64.b64decode(norm["precompiled_charsmap"])
    if len(charsmap) < 4:
        raise SystemExit("precompiled_charsmap shorter than its own size field")
    (trie_size,) = struct.unpack_from("<I", charsmap, 0)
    if trie_size % 4 != 0 or 4 + trie_size > len(charsmap):
        raise SystemExit(
            f"charsmap trie size {trie_size} inconsistent with blob length "
            f"{len(charsmap)} -- the u32-LE-size-then-trie-then-strings "
            "layout assumption is wrong for this checkpoint"
        )
    trie = charsmap[4 : 4 + trie_size]
    normalized = charsmap[4 + trie_size :]
    if normalized and normalized[-1] != 0:
        raise SystemExit(
            "charsmap normalized-strings blob does not end in NUL -- the "
            "'replacement runs to the next NUL' walk would read past the end"
        )

    # --- pre_tokenizer: exactly [WhitespaceSplit, Metaspace]
    pre = tok.get("pre_tokenizer") or {}
    if pre.get("type") != "Sequence":
        raise SystemExit(f"expected pre_tokenizer Sequence, got {pre.get('type')!r}")
    subs = pre.get("pretokenizers") or []
    if len(subs) != 2 or subs[0].get("type") != "WhitespaceSplit" or \
            subs[1].get("type") != "Metaspace":
        raise SystemExit(
            f"expected pre_tokenizers [WhitespaceSplit, Metaspace], got "
            f"{[s.get('type') for s in subs]!r} -- the pre-tokenization in "
            "the reference encoder and the C++ port would be wrong"
        )
    meta = subs[1]
    replacement = meta.get("replacement")
    if replacement != "▁" or len(replacement) != 1:
        raise SystemExit(f"expected Metaspace replacement '▁', got {replacement!r}")
    scheme_name = meta.get("prepend_scheme")
    if scheme_name is None:
        # legacy serialization: add_prefix_space bool only
        scheme_name = "always" if meta.get("add_prefix_space") else "never"
    if scheme_name not in PREPEND_SCHEMES:
        raise SystemExit(f"unknown Metaspace prepend_scheme {scheme_name!r}")
    if "add_prefix_space" in meta and \
            bool(meta["add_prefix_space"]) != (scheme_name != "never"):
        raise SystemExit(
            f"Metaspace add_prefix_space {meta['add_prefix_space']!r} "
            f"contradicts prepend_scheme {scheme_name!r}"
        )
    metaspace_split = bool(meta.get("split", True))

    # --- specials: read from added_tokens, cross-checked against the vocab
    added = {t["content"]: t for t in tok.get("added_tokens") or []}
    expect_specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    ids: dict[str, int] = {}
    for name in expect_specials:
        if name not in added:
            raise SystemExit(f"special token {name!r} missing from added_tokens")
        if not added[name].get("special"):
            raise SystemExit(f"{name!r} present but not marked special")
        if added[name].get("normalized"):
            raise SystemExit(f"{name!r} is normalized:true -- unexpected for XLM-R")
        ids[name] = added[name]["id"]
        # every special must also be a vocab row at the same id, or the
        # C++ id->piece table would disagree with the added_tokens ids
        if not (0 <= ids[name] < vocab_size) or pieces[ids[name]] != name:
            raise SystemExit(
                f"added token {name!r} id {ids[name]} does not match vocab "
                f"row {pieces[ids[name]] if ids[name] < vocab_size else '<oob>'!r}"
            )
    if ids["<unk>"] != unk_id:
        raise SystemExit(f"added <unk> id {ids['<unk>']} != model unk_id {unk_id}")

    # --- post_processor: single-sequence template must be <s> A </s>
    post = tok.get("post_processor") or {}
    if post.get("type") != "TemplateProcessing":
        raise SystemExit(f"expected TemplateProcessing, got {post.get('type')!r}")
    single = post.get("single") or []
    want_single = [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ]
    if single != want_single:
        raise SystemExit(
            f"expected single template [<s>, A, </s>], got {single!r} -- "
            "the fixed add_bos/add_eos flags below would lie"
        )
    post_ids = post.get("special_tokens") or {}
    for name in ("<s>", "</s>"):
        got = (post_ids.get(name) or {}).get("ids")
        if got != [ids[name]]:
            raise SystemExit(
                f"post_processor maps {name!r} to {got!r}, added_tokens to "
                f"[{ids[name]}]"
            )
    add_bos = add_eos = 1

    if tok.get("truncation") is not None or tok.get("padding") is not None:
        raise SystemExit(
            "tokenizer.json carries baked-in truncation/padding -- this "
            "project refuses silent truncation (tasks/0110); re-check"
        )

    piece_bytes = [p.encode("utf-8") for p in pieces]
    max_piece_bytes = max(len(b) for b in piece_bytes)
    max_piece_chars = max(len(p) for p in pieces)
    if max_piece_bytes > 0xFFFF:
        raise SystemExit("a vocab piece exceeds 65535 bytes")

    # --- write the binary table ----------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(FORMAT_MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<II", vocab_size, unk_id))
        f.write(struct.pack("<IIII", ids["<s>"], ids["</s>"], ids["<pad>"],
                            ids["<mask>"]))
        f.write(struct.pack("<II", add_bos, add_eos))
        f.write(struct.pack("<III", ord(replacement),
                            PREPEND_SCHEMES[scheme_name],
                            1 if metaspace_split else 0))
        f.write(struct.pack("<II", max_piece_bytes, max_piece_chars))
        f.write(struct.pack("<dd", min_score, unk_score))
        f.write(struct.pack("<II", len(trie), len(normalized)))
        f.write(trie)
        f.write(normalized)
        f.write(struct.pack(f"<{vocab_size}d", *scores))
        for b in piece_bytes:
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.2f} MB)")
    print(f"  vocab_size={vocab_size} unk_id={unk_id} "
          f"max_piece_bytes={max_piece_bytes} max_piece_chars={max_piece_chars}")
    print(f"  bos={ids['<s>']} eos={ids['</s>']} pad={ids['<pad>']} "
          f"mask={ids['<mask>']}")
    print(f"  min_score={min_score!r} unk_score={unk_score!r} "
          f"(scores stored f64; {n_not_f32} of {vocab_size} are not f32-exact)")
    print(f"  charsmap: trie {len(trie)} B, normalized strings {len(normalized)} B")
    print(f"  metaspace: replacement U+{ord(replacement):04X}, "
          f"prepend_scheme={scheme_name}, split={metaspace_split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
