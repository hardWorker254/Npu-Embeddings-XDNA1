# NpuEmbeddings -- generate the binary SentencePiece-BPE table the C++
# EmbeddingGemma tokenizer (runtime/src/tokenizer_gemma.cpp) reads at load
# time.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS CHECKPOINT'S TOKENIZER ACTUALLY IS
# ---------------------------------------------
# The task that asked for this file assumed SentencePiece **Unigram**, by
# analogy with other sentence-embedding models. Reading
# models/embeddinggemma-300m/tokenizer.json directly (not assuming) shows the
# HuggingFace `model.type` field is **"BPE"**, with `byte_fallback: true` and
# a 514,906-row `merges` list -- not a Unigram `vocab` of (piece, log-prob)
# pairs. This is the real Gemma/Llama-family SentencePiece BPE tokenizer,
# 262,144-entry vocabulary. Everything below implements BPE, not Unigram; see
# tasks/00XX-gemma-tokenizer/TASK.md for how this was confirmed (probed the
# live `transformers` tokenizer, not inferred from the plan).
#
# THE FULL PIPELINE, CONFIRMED EMPIRICALLY (not just read from JSON)
# --------------------------------------------------------------------
#   normalizer     ONE rule: literal ' ' (U+0020) -> '▁' (metaspace).
#                  No NFKC, no case folding, no accent stripping, and -- the
#                  one surprising bit -- NO automatic leading metaspace.
#                  encode("Hello world") != encode(" Hello world"); the
#                  former's first token is "Hello" (no leading ▁), the
#                  ONLY normalization is turning literal spaces into
#                  metaspace, verbatim.
#   pre_tokenizer  Split on literal ' ' -- but the normalizer already
#                  consumed every space, so by the time this runs there is
#                  nothing left to split on. Confirmed via
#                  backend_tokenizer.pre_tokenizer.pre_tokenize_str(): a
#                  4-word sentence comes back as ONE segment. So: the whole
#                  normalized string is one BPE input, unlike BERT's
#                  whitespace-pre-split WordPiece.
#   BPE            Standard merge algorithm: start from one symbol per
#                  Unicode codepoint (each codepoint's UTF-8 string looked up
#                  in vocab directly); repeatedly merge the adjacent pair
#                  with the lowest merge rank until none remain.
#   byte_fallback  A codepoint whose own string is not a vocab entry is
#                  decomposed into its UTF-8 bytes, each looked up as
#                  "<0xXX>" (uppercase hex, confirmed -- '<0xF0>' not
#                  '<0xf0>'). All 256 such entries exist (ids 238-493, found
#                  by regex over the vocab). Confirmed to actually fire: an
#                  obscure Cuneiform codepoint (U+12031) tokenizes to
#                  ['<0xF0>','<0x92>','<0x80>','<0xB1>'].
#   post_processor TemplateProcessing, single-sequence template
#                  [<bos>, A, <eos>] -- always both, unconditionally
#                  (add_bos_token/add_eos_token both true in
#                  tokenizer_config.json). ids: <pad>=0 <eos>=1 <bos>=2
#                  <unk>=3 <mask>=4.
#
# WHY A BINARY TABLE, NOT JSON/PROTOBUF AT RUNTIME
# --------------------------------------------------
# CLAUDE.md rule 5: Python is for build-time table generation only; the
# shipped runtime is C++ with no Python and no JSON/protobuf parser. This
# script reads tokenizer.json (33 MB of JSON) ONCE, offline, and writes a
# flat binary the C++ side reads with plain ifstream + memcpy -- exactly the
# role tools/gen_tokenizer_tables.py plays for BERT's Unicode tables, and
# tools/npue_pack.py plays for weights. Merges are stored as (id_a, id_b,
# merged_id) triples, keyed by vocabulary ID rather than by string, so the
# runtime never has to hash a substring during the merge loop -- it hashes
# only once per input codepoint to seed the initial symbol.
#
# TASK-PREFIX TABLE
# ------------------
# Read verbatim from the checkpoint's own
# models/embeddinggemma-300m/config_sentence_transformers.json "prompts"
# dict -- the authoritative source (not FastFlowLM's copy, not retyped).
# `default_prompt_name` in that file is `null`, meaning sentence-transformers
# applies NO prefix unless the caller names one. This project bakes in
# "document" ("title: none | text: ") as ITS OWN default -- a decision, not a
# fact from the checkpoint -- because it matches how tokens 0055's spike
# validated goldens and how this project's own `--embed` CLI is used
# (embedding arbitrary text into a corpus, the "document" side of retrieval).
# The binary format stores every prompt so a future caller can override it.
#
# Env: any Python 3 (json + struct only, no transformers needed for the
# generator itself -- transformers is only needed for the verify script that
# checks this table against ground truth).
# Usage:
#   python tools\gen_gemma_tokenizer_table.py
#     [--tokenizer-json PATH] [--sbert-config PATH] [--out PATH]

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_JSON = REPO / "models" / "embeddinggemma-300m" / "tokenizer.json"
DEFAULT_SBERT_CONFIG = (
    REPO / "models" / "embeddinggemma-300m" / "config_sentence_transformers.json"
)
DEFAULT_OUT = REPO / "models" / "embeddinggemma-300m" / "gemma_tokenizer.bin"

MAGIC = 0x314B_544D_47  # not used directly; see below for the real on-disk magic
FORMAT_MAGIC = b"GEMATOK1"  # 8 bytes, on disk verbatim
VERSION = 1

# This project's chosen default task prefix -- see the module docstring.
DEFAULT_PREFIX_NAME = "document"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-json", default=str(DEFAULT_TOKENIZER_JSON))
    ap.add_argument("--sbert-config", default=str(DEFAULT_SBERT_CONFIG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tok_path = Path(args.tokenizer_json)
    sbert_path = Path(args.sbert_config)
    out_path = Path(args.out)

    with tok_path.open(encoding="utf-8") as f:
        tok = json.load(f)

    model = tok["model"]
    if model.get("type") != "BPE":
        raise SystemExit(
            f"expected model.type == 'BPE', got {model.get('type')!r} -- "
            "this generator implements SentencePiece BPE specifically, not "
            "Unigram; re-check tokenizer.json before proceeding"
        )
    if not model.get("byte_fallback"):
        raise SystemExit("expected byte_fallback: true -- this generator assumes it")
    if model.get("dropout") is not None:
        raise SystemExit("expected dropout: null (deterministic BPE) -- got a value")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise SystemExit(
            "expected no continuing_subword_prefix/end_of_word_suffix -- "
            "the merge-is-plain-concatenation assumption below would be wrong"
        )

    normalizer = tok.get("normalizer") or {}
    if normalizer.get("type") != "Replace" or normalizer.get("pattern") != {
        "String": " "
    } or normalizer.get("content") != "▁":
        raise SystemExit(
            f"unexpected normalizer, re-verify pipeline assumptions: {normalizer!r}"
        )

    vocab: dict[str, int] = model["vocab"]
    vocab_size = len(vocab)
    id_to_token: list[str | None] = [None] * vocab_size
    for tok_str, tid in vocab.items():
        if not (0 <= tid < vocab_size):
            raise SystemExit(f"vocab id {tid} for {tok_str!r} out of [0, {vocab_size})")
        if id_to_token[tid] is not None:
            raise SystemExit(f"duplicate vocab id {tid}")
        id_to_token[tid] = tok_str
    if any(t is None for t in id_to_token):
        missing = [i for i, t in enumerate(id_to_token) if t is None]
        raise SystemExit(f"vocab has {len(missing)} unused ids, e.g. {missing[:5]}")

    merges_raw: list[list[str]] = model["merges"]
    merges: list[tuple[int, int, int]] = []
    for rank, (a, b) in enumerate(merges_raw):
        ida = vocab.get(a)
        idb = vocab.get(b)
        if ida is None or idb is None:
            raise SystemExit(f"merge rank {rank} references unknown piece {a!r}/{b!r}")
        merged_str = a + b
        merged_id = vocab.get(merged_str)
        if merged_id is None:
            raise SystemExit(
                f"merge rank {rank} ({a!r}+{b!r}) has no vocab entry for "
                f"{merged_str!r} -- the plain-concatenation assumption is wrong"
            )
        merges.append((ida, idb, merged_id))

    # Special ids -- read from the checkpoint's own added_tokens_decoder
    # rather than hardcoded, but cross-checked against the known constants
    # documented in the module header so a future checkpoint revision that
    # changes them fails loudly instead of silently shipping wrong ids.
    special_by_name = {t: i for i, t in enumerate(id_to_token[:16]) if t}
    expect = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4}
    for name, want_id in expect.items():
        got = vocab.get(name)
        if got != want_id:
            raise SystemExit(f"expected {name}={want_id}, checkpoint has {got}")

    post = tok.get("post_processor") or {}
    single = post.get("single") or []
    add_bos = any(
        step.get("SpecialToken", {}).get("id") == "<bos>" for step in single
    )
    add_eos = any(
        step.get("SpecialToken", {}).get("id") == "<eos>" for step in single
    )
    if not (add_bos and add_eos):
        raise SystemExit(
            f"expected post_processor to add both <bos> and <eos>, got {single!r}"
        )

    # --- task prefixes ------------------------------------------------
    with sbert_path.open(encoding="utf-8") as f:
        sbert = json.load(f)
    prompts: dict[str, str] = sbert["prompts"]
    if DEFAULT_PREFIX_NAME not in prompts:
        raise SystemExit(
            f"chosen default prefix {DEFAULT_PREFIX_NAME!r} not in checkpoint's "
            f"prompts dict: {sorted(prompts)}"
        )
    prefix_names = sorted(prompts)  # deterministic order
    default_prefix_index = prefix_names.index(DEFAULT_PREFIX_NAME)

    # --- write the binary table ----------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(FORMAT_MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", len(merges)))
        f.write(
            struct.pack(
                "<IIIII",
                vocab["<pad>"],
                vocab["<eos>"],
                vocab["<bos>"],
                vocab["<unk>"],
                vocab["<mask>"],
            )
        )
        f.write(struct.pack("<II", 1 if add_bos else 0, 1 if add_eos else 0))
        f.write(struct.pack("<I", len(prefix_names)))
        f.write(struct.pack("<I", default_prefix_index))

        # vocab: id order, u16-length-prefixed UTF-8 bytes
        for tok_str in id_to_token:
            b = tok_str.encode("utf-8")
            if len(b) > 0xFFFF:
                raise SystemExit(f"token {tok_str!r} exceeds 65535 bytes")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        # merges: rank order, (id_a, id_b, merged_id) as u32 triples
        for ida, idb, merged_id in merges:
            f.write(struct.pack("<III", ida, idb, merged_id))

        # task prefixes: name then prefix text, both u16-length-prefixed
        for name in prefix_names:
            nb = name.encode("utf-8")
            pb = prompts[name].encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<H", len(pb)))
            f.write(pb)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.2f} MB)")
    print(f"  vocab_size={vocab_size} merges={len(merges)}")
    print(f"  bos={vocab['<bos>']} eos={vocab['<eos>']} pad={vocab['<pad>']} "
          f"unk={vocab['<unk>']} mask={vocab['<mask>']}")
    print(f"  task prefixes: {prefix_names}")
    print(f"  default prefix: {DEFAULT_PREFIX_NAME!r} -> {prompts[DEFAULT_PREFIX_NAME]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
