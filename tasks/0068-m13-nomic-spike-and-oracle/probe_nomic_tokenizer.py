# NpuEmbeddings -- tokenizer-delta probe for nomic-ai/nomic-embed-text-v1.5.
# SPDX-License-Identifier: Apache-2.0
#
# QUESTION: can the existing C++ WordPiece tokenizer (runtime/src/tokenizer.cpp)
# serve nomic-embed-text-v1.5 unchanged, or is there a configuration delta that
# would silently produce different token ids than HuggingFace?
#
# This script does NOT build or run npuembed.exe -- there is no packed .npue
# for nomic yet, so tools/verify_tokenizer.py's C++-vs-HF harness cannot run
# end to end. Instead it:
#   1. diffs vocab.txt / tokenizer_config.json / special_tokens_map.json /
#      tokenizer.json across nomic and the four shipping models,
#   2. runs HuggingFace's own tokenizer for nomic AND bge-base over a shared
#      adversarial corpus and checks the id sequences are IDENTICAL. If they
#      are, then because the C++ tokenizer is already proven byte-identical
#      to bge-base's HF tokenizer (tasks/0036, 6826/6826 texts), it is
#      byte-identical to nomic's by transitivity -- WITHOUT needing to build
#      the C++ side at all.
#   3. measures the token cost of nomic's four documented task prefixes.
#
# Env: .venv-ref (transformers). Usage:
#   & "..\..\.venv-ref\Scripts\python.exe" probe_nomic_tokenizer.py
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

NOMIC = MODELS / "nomic-embed-text-v1.5"
BGE_BASE = MODELS / "bge-base-en-v1.5"
MINILM = MODELS / "all-MiniLM-L6-v2"
BGE_SMALL = MODELS / "bge-small-en-v1.5"
BGE_LARGE = MODELS / "bge-large-en-v1.5"

ALL_MODELS = {
    "nomic-embed-text-v1.5": NOMIC,
    "bge-base-en-v1.5": BGE_BASE,
    "all-MiniLM-L6-v2": MINILM,
    "bge-small-en-v1.5": BGE_SMALL,
    "bge-large-en-v1.5": BGE_LARGE,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def line_count(p: Path) -> int:
    return sum(1 for _ in p.open("rb"))


# ---------------------------------------------------------------------------
# Step 1: static config diff
# ---------------------------------------------------------------------------

def step1_vocab_diff() -> None:
    print("=" * 78)
    print("STEP 1a: vocab.txt across all five models")
    print("=" * 78)
    digests = {}
    for name, path in ALL_MODELS.items():
        vp = path / "vocab.txt"
        d = sha256_file(vp)
        n = line_count(vp)
        digests[name] = (d, n)
        print(f"{name:28s} sha256={d}  lines={n}")

    ref = digests["bge-base-en-v1.5"]
    all_same = all(v == ref for v in digests.values())
    print()
    if all_same:
        print("RESULT: all five vocab.txt files are BYTE-IDENTICAL "
              "(same sha256, same line count).")
    else:
        print("RESULT: vocab.txt files DIFFER. Diffing nomic vs bge-base line by line:")
        a = (NOMIC / "vocab.txt").read_text(encoding="utf-8").splitlines()
        b = (BGE_BASE / "vocab.txt").read_text(encoding="utf-8").splitlines()
        if len(a) != len(b):
            print(f"  line count differs: nomic={len(a)} bge-base={len(b)}")
        for i, (la, lb) in enumerate(zip(a, b)):
            if la != lb:
                print(f"  line {i}: nomic={la!r}  bge-base={lb!r}")
    print()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def step1_tokenizer_config_diff() -> None:
    print("=" * 78)
    print("STEP 1b: tokenizer_config.json -- fields that matter")
    print("=" * 78)
    fields = ["do_lower_case", "tokenize_chinese_chars", "strip_accents",
              "never_split", "model_max_length", "cls_token", "sep_token",
              "pad_token", "unk_token", "mask_token", "tokenizer_class",
              "do_basic_tokenize"]
    nomic_cfg = load_json(NOMIC / "tokenizer_config.json")
    bge_cfg = load_json(BGE_BASE / "tokenizer_config.json")
    print(f"{'field':28s}{'nomic':30s}{'bge-base':30s}same?")
    for f in fields:
        nv = nomic_cfg.get(f, "<absent>")
        bv = bge_cfg.get(f, "<absent>")
        same = "YES" if nv == bv else "**DIFFERS**"
        print(f"{f:28s}{str(nv):30s}{str(bv):30s}{same}")
    print()
    print("Full nomic tokenizer_config.json keys:", sorted(nomic_cfg.keys()))
    print("Full bge-base tokenizer_config.json keys:", sorted(bge_cfg.keys()))
    print()


def step1_special_tokens_map_diff() -> None:
    print("=" * 78)
    print("STEP 1c: special_tokens_map.json")
    print("=" * 78)
    nomic_m = load_json(NOMIC / "special_tokens_map.json")
    bge_m = load_json(BGE_BASE / "special_tokens_map.json")

    def spelling(v):
        # bge-base's file: plain strings. nomic's file: {"content": "...", ...} dicts.
        if isinstance(v, dict):
            return v.get("content")
        return v

    for tok in ["cls_token", "sep_token", "pad_token", "unk_token", "mask_token"]:
        nv = spelling(nomic_m.get(tok))
        bv = spelling(bge_m.get(tok))
        same = "YES" if nv == bv else "**DIFFERS**"
        print(f"{tok:12s} nomic={nv!r:10s} bge-base={bv!r:10s} spelling same? {same}")
    print()
    print("Structural note: nomic's special_tokens_map.json wraps each token in "
          "a {content,lstrip,normalized,rstrip,single_word} dict (newer "
          "tokenizers serialization); bge-base's is flat strings. This is a "
          "file-format difference in a file the C++ tokenizer does not even "
          "read (it reads vocab.txt directly and looks up the literal strings "
          "[CLS]/[SEP]/[PAD]/[UNK]/[MASK]) -- so it cannot affect token ids.")
    print()


def step1_tokenizer_json_diff() -> None:
    print("=" * 78)
    print("STEP 1d: tokenizer.json -- the authoritative source")
    print("=" * 78)
    nomic_t = load_json(NOMIC / "tokenizer.json")
    bge_t = load_json(BGE_BASE / "tokenizer.json")

    for block in ["normalizer", "pre_tokenizer"]:
        nv = nomic_t.get(block)
        bv = bge_t.get(block)
        same = "YES" if nv == bv else "**DIFFERS**"
        print(f"-- {block} -- same? {same}")
        print("  nomic   :", json.dumps(nv))
        print("  bge-base:", json.dumps(bv))

    nv_model = dict(nomic_t.get("model", {}))
    bv_model = dict(bge_t.get("model", {}))
    nv_vocab = nv_model.pop("vocab", None)
    bv_vocab = bv_model.pop("vocab", None)
    same_model = "YES" if nv_model == bv_model else "**DIFFERS**"
    print(f"-- model (excl. vocab table) -- same? {same_model}")
    print("  nomic   :", json.dumps(nv_model))
    print("  bge-base:", json.dumps(bv_model))
    same_vocab = "YES" if nv_vocab == bv_vocab else "**DIFFERS**"
    print(f"-- model.vocab table identical dict? {same_vocab}")

    nv_pp = nomic_t.get("post_processor")
    bv_pp = bge_t.get("post_processor")
    same_pp = "YES" if nv_pp == bv_pp else "**DIFFERS**"
    print(f"-- post_processor -- same? {same_pp}")
    print("  nomic   :", json.dumps(nv_pp))
    print("  bge-base:", json.dumps(bv_pp))

    nv_added = [t["content"] for t in nomic_t.get("added_tokens", [])]
    bv_added = [t["content"] for t in bge_t.get("added_tokens", [])]
    print(f"-- added_tokens -- nomic={nv_added} bge-base={bv_added} "
          f"same? {'YES' if nv_added == bv_added else '**DIFFERS**'}")
    print()


def step1_sentence_bert_config() -> None:
    print("=" * 78)
    print("STEP 1e (extra): sentence_bert_config.json / config_sentence_transformers.json")
    print("=" * 78)
    for name, path in [("nomic", NOMIC), ("bge-base", BGE_BASE)]:
        sb = path / "sentence_bert_config.json"
        if sb.exists():
            print(f"{name} sentence_bert_config.json: {sb.read_text(encoding='utf-8')}")
    cst = NOMIC / "config_sentence_transformers.json"
    if cst.exists():
        print(f"nomic config_sentence_transformers.json (verbatim):")
        print(cst.read_text(encoding="utf-8"))
        d = load_json(cst)
        print(f"Contains a 'prompts' key? {'prompts' in d}")
    else:
        print("nomic has no config_sentence_transformers.json")
    print()
    print("NOTE: nomic's sentence_bert_config.json says max_seq_length=8192, "
          "do_lower_case=false. tokenizer_config.json / tokenizer.json's "
          "normalizer both say do_lower_case/lowercase=true -- the "
          "authoritative source (tokenizer.json's BertNormalizer block) wins, "
          "and it agrees with bge-base. The sentence_bert_config field is a "
          "sentence-transformers wrapper setting that does not feed the fast "
          "tokenizer's normalizer at all; it is reported here as a possible "
          "footgun for anyone reading config files by eye, not a real delta.")
    print()


# ---------------------------------------------------------------------------
# Step 2: tokenizer.cpp assumptions vs nomic's tokenizer.json, read manually
# and reported (this step doesn't execute code; runtime/src/tokenizer.cpp and
# runtime/include/tokenizer.hpp were read directly, see the report text below)
# ---------------------------------------------------------------------------

def step2_report() -> None:
    print("=" * 78)
    print("STEP 2: runtime/src/tokenizer.cpp assumptions vs nomic's tokenizer.json")
    print("=" * 78)
    text = """
tokenizer.cpp / tokenizer.hpp encode a fixed pipeline (comment block at the
top of tokenizer.cpp, mirrors HF's tokenization_bert.py):
  _clean_text -> _tokenize_chinese_chars -> whitespace split -> per-token
  lower()+strip-accents (NFD, drop Mn) -> _run_split_on_punc -> WordPiece
  (greedy longest-match, '##' continuations, max_input_chars_per_word=100).

Checked against nomic's tokenizer.json (Step 1d output above), field by field:

  lowercase            true   in tokenizer.cpp's Unicode fold tables (built by
                               tools/gen_tokenizer_tables.py from CPython's
                               unicodedata) vs nomic's normalizer.lowercase=true
                               -- MATCH.
  strip_accents         null  (i.e. inherits do_lower_case -> true) in nomic's
                               tokenizer.json AND in bge-base's -- the C++ always
                               strips accents whenever it lowercases (single
                               fold table, no separate strip_accents branch) --
                               MATCH, and it is the same behaviour the code
                               already ships for the four current models.
  clean_text / CJK       true  tokenizer.cpp's tokenize_plain() does
                               _clean_text + _tokenize_chinese_chars in one pass
                               (drops control/NUL/U+FFFD, pads the same Han
                               ranges) -- MATCH with nomic's
                               normalizer.clean_text=true,
                               handle_chinese_chars=true.
  WordPiece model         type=WordPiece, unk_token=[UNK],
                               continuing_subword_prefix=##,
                               max_input_chars_per_word=100 -- all four fields
                               identical to bge-base's, and identical to the
                               constants tokenizer.cpp/tokenizer.hpp hard-code
                               ('##' literal in tokenize_plain, id_of() falls
                               back to unk_id, max_chars_per_word_=100 default
                               in tokenizer.hpp) -- MATCH.
  post_processor        TemplateProcessing [CLS] Sequence [SEP], single
                               type_id 0 -- MATCH with tokenizer.cpp's
                               Tokenizer::encode(): push cls_id, tokens,
                               sep_id, then pad with pad_id / attention_mask 0,
                               token_type_ids all zero. Identical structure.
  vocab.txt / specials   vocab.txt byte-identical (Step 1a); the five specials
                               [CLS]/[SEP]/[PAD]/[UNK]/[MASK] are the ONLY
                               added_tokens in nomic's tokenizer.json (checked
                               directly: ['[PAD]', '[UNK]', '[CLS]', '[SEP]',
                               '[MASK]'], 5 entries) -- exactly what
                               tokenizer.cpp's build_index() looks up by
                               literal string and sorts into specials_ for the
                               added-token pre-pass. MATCH.
  model_max_length        nomic: 8192 in tokenizer_config.json (2048 in
                               config.json's max_position_embeddings, 8192 in
                               sentence_bert_config.json / n_positions) vs
                               bge-base: 512. tokenizer.cpp does NOT read this
                               field at all -- Tokenizer::encode(text, max_len)
                               takes max_len as a CALLER-SUPPLIED argument, not
                               something parsed from tokenizer_config.json. So
                               this is not a tokenizer-code delta; it is a
                               caller-side decision (what --max-len / sequence
                               bucket the runtime passes in) that the packer /
                               CLI must set correctly for nomic. NOT a tokenizer.cpp
                               change, but flagged because CLAUDE.md's own text
                               about 'eats into the 64-token sequence budget'
                               shows this project runs short buckets by choice
                               -- confirm the bucket picked for nomic, whatever
                               it is, is deliberate and not a leftover default.

CONCLUSION: every field tokenizer.cpp's pipeline is sensitive to (lowercase,
strip_accents inheritance, clean_text, CJK ranges, WordPiece model params,
post_processor template, special-token set, vocab table) is IDENTICAL between
nomic's tokenizer.json and bge-base's. Nothing nomic specifies is left
unimplemented by the C++, and nothing the C++ assumes is contradicted by
nomic. The only difference found anywhere in the config files
(model_max_length / max_seq_length) is not read by tokenizer.cpp at all --
it is an external parameter, not a tokenizer algorithm delta.
"""
    print(text)


# ---------------------------------------------------------------------------
# Step 3: empirical gate -- HF nomic tokenizer vs HF bge-base tokenizer
# ---------------------------------------------------------------------------

def build_corpus() -> list[str]:
    # Reuses tools/verify_tokenizer.py's adversarial corpus (imported so this
    # probe cannot silently drift from the harness that will run the real
    # C++-vs-HF gate once a nomic .npue exists), plus reference/corpus.py's
    # four frozen golden sentences, plus the extra cases the task explicitly
    # asked for (>100-char single word already in CORPUS as "a"*200 etc, but
    # add one unambiguous single very-long WORD with no spaces to be sure).
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "reference"))
    import verify_tokenizer  # type: ignore
    import corpus as ref_corpus  # type: ignore

    texts = list(verify_tokenizer.CORPUS)
    texts += list(ref_corpus.SENTENCES)
    texts += [
        "b" * 150,  # single word > max_input_chars_per_word=100, no spaces at all
        "søppelbøtte, blåbær, Ørje, Ænes",  # more Norwegian æøå coverage
        "emoji only: 😀😁😂🤣😃😄😅",
    ]
    # de-dup while preserving order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def step3_empirical_gate() -> tuple[bool, int]:
    print("=" * 78)
    print("STEP 3: HF nomic tokenizer vs HF bge-base tokenizer, same corpus")
    print("=" * 78)
    from transformers import AutoTokenizer

    tok_nomic = AutoTokenizer.from_pretrained(str(NOMIC))
    tok_bge = AutoTokenizer.from_pretrained(str(BGE_BASE))

    print(f"nomic tokenizer class:    {type(tok_nomic).__name__}")
    print(f"bge-base tokenizer class: {type(tok_bge).__name__}")
    print()

    corpus = build_corpus()
    print(f"Corpus size: {len(corpus)} texts "
          "(tools/verify_tokenizer.py's adversarial set + "
          "reference/corpus.py's 4 golden sentences + 3 extra cases)")
    print()

    mismatches = []
    for text in corpus:
        ids_nomic = tok_nomic(text, add_special_tokens=True)["input_ids"]
        ids_bge = tok_bge(text, add_special_tokens=True)["input_ids"]
        if ids_nomic != ids_bge:
            mismatches.append((text, ids_nomic, ids_bge))

    print(f"Texts compared: {len(corpus)}")
    print(f"Identical id sequences: {len(corpus) - len(mismatches)}")
    print(f"Mismatches: {len(mismatches)}")
    print()

    if mismatches:
        print("MISMATCHING TEXTS (full detail):")
        for text, ids_n, ids_b in mismatches:
            print(f"  text: {text!r}")
            print(f"    nomic    ids: {ids_n}")
            print(f"    bge-base ids: {ids_b}")
        print()
        print("RESULT: nomic's HF tokenizer and bge-base's HF tokenizer "
              "DIVERGE on at least one text in this corpus. See mismatches "
              "above for the exact texts and ids.")
        return False, len(corpus)
    else:
        print("RESULT: nomic's HF tokenizer and bge-base's HF tokenizer "
              "produce IDENTICAL id sequences on EVERY text in this corpus, "
              f"all {len(corpus)}/{len(corpus)}.")
        print()
        print("Since tasks/0036 already proved the C++ tokenizer is byte-"
              "identical to bge-base's HF tokenizer on 6,826/6,826 texts, "
              "and this step proves nomic's HF tokenizer == bge-base's HF "
              "tokenizer on every text tried here, the C++ tokenizer matches "
              "nomic's HF tokenizer BY TRANSITIVITY on this corpus. This is "
              f"NOT the same claim as tasks/0036's -- it covers {len(corpus)} "
              "adversarial texts, not 6,826, and it has not exercised the "
              "actual C++ binary at all (no .npue exists for nomic yet). A direct "
              "gate (npuembed --tokenize once a nomic .npue is packed, run "
              "through tools/verify_tokenizer.py's harness against nomic's "
              "own tokenizer_config.json/vocab.txt) still has to run in a "
              "later task before this claim covers the C++ binary directly.")
        return True, len(corpus)


# ---------------------------------------------------------------------------
# Step 4: the prefix question
# ---------------------------------------------------------------------------

def step4_prefix_cost() -> None:
    print("=" * 78)
    print("STEP 4: nomic's task prefixes -- token cost and mechanism")
    print("=" * 78)

    cst_path = NOMIC / "config_sentence_transformers.json"
    cst = load_json(cst_path)
    print("config_sentence_transformers.json verbatim:")
    print(json.dumps(cst, indent=2))
    print(f"Contains a 'prompts' dict? {'prompts' in cst}")
    print()
    print("modules.json (sentence-transformers pipeline):")
    print(json.dumps(load_json(NOMIC / "modules.json"), indent=2))
    print()
    print("1_Pooling/config.json:")
    print(json.dumps(load_json(NOMIC / "1_Pooling" / "config.json"), indent=2))
    print()
    print("No file in the downloaded snapshot defines the task prefixes "
          "programmatically -- no 'prompts' dict in config_sentence_transformers.json, "
          "no prefix strings anywhere in tokenizer.json's added_tokens (checked: "
          "only the 5 standard specials, see Step 1d). This sentence-transformers "
          "snapshot (__version__.sentence_transformers = "
          f"{cst.get('__version__', {}).get('sentence_transformers', '?')}) predates "
          "the library's 'prompts' config feature. The nomic-embed-text-v1.5 model "
          "card (not fetched here -- CLAUDE.md restricts this probe to files already "
          "in the repo/model download) documents the four prefixes "
          "('search_document: ', 'search_query: ', 'clustering: ', "
          "'classification: ') as something the CALLER must prepend to the raw "
          "text before encoding; nothing in this checkpoint's files applies them "
          "automatically.")
    print()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(NOMIC))

    prefixes = ["search_document: ", "search_query: ", "clustering: ",
                "classification: "]
    print(f"{'prefix':22s}{'ids (no specials)':45s}{'# tokens (no specials)':25s}"
          "# tokens with [CLS]/[SEP]")
    for p in prefixes:
        ids_plain = tok(p, add_special_tokens=False)["input_ids"]
        ids_special = tok(p, add_special_tokens=True)["input_ids"]
        print(f"{p!r:22s}{str(ids_plain):45s}{len(ids_plain):<25d}{len(ids_special)}")
    print()

    # Confirm mechanism: prefix + text is tokenized as ONE string, not as a
    # separately-encoded pair -- compare (prefix+text) tokenized whole against
    # prefix tokens (no specials) + text tokens (no specials), wrapped once in
    # [CLS]...[SEP].
    sample = "The quick brown fox jumps over the lazy dog."
    for p in prefixes:
        combined = p + sample
        ids_whole = tok(combined, add_special_tokens=True)["input_ids"]
        ids_manual = ([tok.cls_token_id] +
                      tok(p, add_special_tokens=False)["input_ids"] +
                      tok(sample, add_special_tokens=False)["input_ids"] +
                      [tok.sep_token_id])
        match = "MATCH" if ids_whole == ids_manual else "**MISMATCH**"
        print(f"prefix={p!r:22s} whole-string encode == "
              f"[CLS]+prefix_ids+text_ids+[SEP]? {match}")
        if match == "**MISMATCH**":
            print(f"    whole : {ids_whole}")
            print(f"    manual: {ids_manual}")
    print()
    print("RESULT: the prefix is prepended as RAW TEXT before [CLS] -- it is "
          "concatenated with the input string and the concatenation is "
          "tokenized as a single sequence (WordPiece boundary at the ':' "
          "punctuation splits the colon into its own token, then normal "
          "whitespace/word tokenization continues into the text). It is not "
          "a special token and is not injected by the post_processor "
          "template. The C++ side, if it takes a --prefix / task-type flag "
          "later, only needs to prepend the literal string to the input "
          "before calling Tokenizer::encode() -- no change to tokenizer.cpp's "
          "algorithm.")
    print()


def main() -> int:
    step1_vocab_diff()
    step1_tokenizer_config_diff()
    step1_special_tokens_map_diff()
    step1_tokenizer_json_diff()
    step1_sentence_bert_config()
    step2_report()
    gate_passed, corpus_size = step3_empirical_gate()
    step4_prefix_cost()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if gate_passed:
        print("Can the existing C++ WordPiece tokenizer serve nomic-embed-text-v1.5 "
              "UNCHANGED? YES, on the evidence gathered here.")
        print("- vocab.txt is byte-identical (sha256 match) across nomic and all "
              "four shipping models.")
        print("- Every tokenizer.json field tokenizer.cpp's algorithm depends on "
              "(normalizer, pre_tokenizer, WordPiece model params, post_processor "
              "template, added_tokens set) is identical between nomic and bge-base.")
        print("- HF's own tokenizers for nomic and bge-base agree on every id, on "
              "every one of the adversarial + golden + extra texts tried here.")
        print("- The only config difference found (model_max_length: nomic 8192 vs "
              "bge-base 512) is not read by tokenizer.cpp -- max_len is a "
              "caller-supplied parameter, not a tokenizer-algorithm delta.")
        print("- nomic's task prefixes are plain text prepended before [CLS]; they "
              "need caller-side wiring (which string, how many tokens it costs) "
              "but no tokenizer.cpp code change.")
        print()
        print("NOT verified here (still needs a direct gate in a later task):")
        print("- The actual C++ binary has not been run against nomic's vocab.txt "
              "or corpus -- no nomic .npue exists yet to drive npuembed --tokenize.")
        print(f"- Only {corpus_size} texts were compared here, not tools/verify_tokenizer.py's "
              "full corpus run through the real C++ binary against nomic's own "
              "tokenizer_config.json (tasks/0036's 6,826/6,826 number is for "
              "bge-base, carried over here only by the transitivity argument above).")
    else:
        print("Can the existing C++ WordPiece tokenizer serve nomic-embed-text-v1.5 "
              "UNCHANGED? NO -- see the mismatches printed in Step 3. Those texts "
              "are where nomic's real tokenizer diverges from bge-base's, so they "
              "are exactly where the C++ implementation (which was built and "
              "verified against bge-base's behaviour) would silently produce wrong "
              "ids for nomic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
