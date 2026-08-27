# 0127 — T52 step 1: the SentencePiece Unigram generator + byte-exact reference for gte-multilingual-base

**Goal**: the Python half of [T52](../../research/OPEN-THREADS.md#t52) — download
the `Alibaba-NLP/gte-multilingual-base` tokenizer files, write the build-time
generator that turns the HF fast `tokenizer.json` into a flat binary table
(`XLMRTOK1`, sibling of `GEMATOK1`), write a pure-Python reference encoder that
consumes that table (the executable spec the C++ port will be diffed against),
and hold the reference byte-exact against HuggingFace over a genuinely
multilingual adversarial corpus. **No C++ in this task** — the port is T52
step 2.

**Result: 343/343 sequences byte-exact against `transformers` 5.15.0 /
`tokenizers` 0.22.2, across 19 categories, special tokens included, no
truncation on either side.**

```
  english 30/30      norwegian 20/20        chinese 20/20     japanese 20/20
  korean  15/15      cyrillic  15/15        arabic  15/15     thai     12/12
  hindi   10/10      hebrew_greek 10/10     vietnamese_turkish 13/13
  emoji   15/15      mixed_script 15/15     nfkc    30/30     whitespace 18/18
  long     6/6       unk       12/12        punct_numbers 15/15
  nfd_decomposed 52/52
```

Full report with per-mismatch detail slots (empty):
[`verify_tokenizer_xlmr.json`](verify_tokenizer_xlmr.json).

**Corpus-hygiene note**: this session was interrupted once by an API content
filter, most likely on raw multilingual corpus text echoed into commentary.
Per the coordinator's instruction the corpus lives **only** in
`tools/verify_tokenizer_xlmr.py`; this log and the terminal transcripts carry
category names, codepoints and id sequences, never the raw strings. Where a
specific input matters below it is named by codepoints.

---

## 1. What the tokenizer actually is (read first, per T29's lesson)

`tokenizer.json` (17 MB), read before writing any code:

| component | value |
|---|---|
| `model.type` | **`Unigram`** — 250,002 `[piece, log_prob]` rows, `unk_id` 3, `byte_fallback` false |
| `normalizer` | `Precompiled`, base64 `precompiled_charsmap` (237,539 B decoded: u32 LE trie size, 177,152 B Darts double-array trie, 60,383 B NUL-terminated replacement strings) |
| `pre_tokenizer` | `Sequence` [`WhitespaceSplit`, `Metaspace(replacement=U+2581, add_prefix_space=true, prepend_scheme="always")`] |
| `post_processor` | `TemplateProcessing`, single = `<s> A </s>` |
| specials | `<s>`=0 `<pad>`=1 `</s>`=2 `<unk>`=3 `<mask>`=250001 (also vocab rows at those ids) |
| truncation / padding | both `null`; `model_max_length` 32768 |

So the T52 filing was right where the Gemma plan had been wrong-then-corrected:
this really is Unigram, the third tokenizer family, and
`gen_gemma_tokenizer_table.py`'s BPE machinery shares nothing with it.

## 2. Files downloaded

`models/gte-multilingual-base/`, from
`https://huggingface.co/Alibaba-NLP/gte-multilingual-base/resolve/main/<file>`,
unauthenticated:

| file | bytes | sha256 |
|---|---:|---|
| `tokenizer.json` | 17,082,756 | `f59925fcb90c92b894cb93e51bb9b4a6105c5c249fe54ce1c704420ac39b81af` |
| `tokenizer_config.json` | 1,149 | `24cebbf2ef20fc317256e03e52ac7b2ca326586f946a8427ecac036332bf0933` |
| `special_tokens_map.json` | 964 | `8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835` |
| `config.json` | 1,429 | `711bdc81365fc25d30533cf05b9fdf588e5ba01f18540fbbb1307d787597a313` |

Generated: `xlmr_tokenizer.bin`, 5,318,988 B,
`15bc6dd63d6b3c54e2b186993697661a2823f468eeb350bbfdb3f10ba73ebde9`.

## 3. Semantics probed on the live backend BEFORE writing code

All probed via `tokenizers` 0.22.2's own `normalize_str` /
`pre_tokenize_str` / encode in `.venv-ref` — the observable behaviour is the
spec, not the docs, and HuggingFace (not upstream sentencepiece) is the
authority because it produced the golden embeddings. Four findings shaped the
implementation:

1. **The Precompiled normalizer is grapheme-based with a deliberate quirk.**
   `tokenizers`' `precompiled.rs` walks UAX #29 extended grapheme clusters; a
   cluster shorter than 6 UTF-8 bytes is first tried WHOLE against the trie by
   common-prefix search, and the **first (shortest) match replaces the entire
   cluster** — even when the match covers only a prefix. Confirmed live: the
   two-codepoint cluster U+1E9B U+0323 (5 bytes) normalizes to the single
   codepoint U+1E61; the combining mark U+0323 is silently dropped. Upstream
   sentencepiece would longest-match the whole string instead. The reference
   mirrors HF's quirk exactly (`precompiled.rs`'s own comment: "Yes, this
   seems broken").
2. **`fuse_unk` is on.** Input `x` U+12031 U+12032 `y` (two adjacent
   codepoints with no vocab piece) encodes to `[0, 1022, 3, 53, 2]` — the two
   unknown codepoints yield **one** id 3, not two. The Viterbi backtrack
   therefore fuses consecutive unk ids.
3. **Metaspace prepends only when the pre-token does not already start with
   U+2581, and splits on it merged-with-next.** A pre-token containing a
   literal U+2581 mid-word is split there. `prepend_scheme="always"` applies
   per whitespace-separated word, not once per text.
4. **Empty and whitespace-only inputs encode to `[0, 2]`** — no pre-tokens,
   template only.

Two more facts found by probing that changed the design:

* **65,856 of 250,002 log-probs do not round-trip through float32** (first
  example: the piece at id for single `k`, −7.471577644348144). HF parses
  JSON into f64 and sums f64 in Viterbi, so an f32 table would perturb
  near-tie segmentations. **The blob stores f64 scores** — the task brief
  said "float log-probs", the data said otherwise, and the data won (2 MB
  instead of 1 MB; the byte-exact bar is the point of the whole exercise).
* **The private-use codepoints U+E000/U+E001 are real vocab entries**
  (ids 244343 / 243967), so PUA is *not* a route to `<unk>` on this vocab.
  Genuine unknowns in the corpus use Cuneiform / Byzantine music / tag
  characters instead.

## 4. What was built

### `tools/gen_xlmr_tokenizer_table.py` (build-time, stdlib only)

Mirrors `gen_gemma_tokenizer_table.py`'s discipline: every assumption about
`tokenizer.json` is a hard `SystemExit` naming what changed — refuses
non-Unigram (pointing BPE checkpoints at the Gemma generator, the mirror
image of that file's refusal), refuses `byte_fallback: true`, duplicate
pieces, a malformed charsmap (size field vs blob length, missing trailing
NUL), a pre-tokenizer that is not exactly [WhitespaceSplit, Metaspace],
an unexpected Metaspace replacement or prepend scheme, added_tokens whose
ids disagree with their own vocab rows, a post-processor that is not exactly
`<s> A </s>`, and baked-in truncation/padding (0110's rule).

**`XLMRTOK1` format** (all little-endian):

```
magic "XLMRTOK1" · u32 version=1
u32 vocab_size, unk_id, bos_id, eos_id, pad_id, mask_id
u32 add_bos, add_eos
u32 metaspace_char (0x2581), prepend_scheme (0/1/2 = never/first/always), metaspace_split
u32 max_piece_bytes (48), max_piece_chars (16)
f64 min_score (−20.3647518157959), unk_score (min_score − 10.0)
u32 trie_size (177,152), norm_size (60,383)
trie bytes verbatim (Darts u32 LE units) · normalized-strings blob verbatim
scores: vocab_size × f64, id order
pieces: vocab_size × (u16 byte-length + UTF-8 bytes), id order
```

The charsmap trie and strings blob are stored **verbatim** (decoded from
base64 once, at build time) — the C++ side walks the same double-array the
Rust and C++ sentencepiece implementations do, no re-derivation. The
`unk_score` bakes in `tokenizers`' `kUnkPenalty = 10.0` so the constant
lives in one place.

### `tools/xlmr_tokenizer_ref.py` (executable spec, stdlib only)

Consumes the **blob**, not `tokenizer.json` — so the verification exercises
the exact bytes the C++ port will read, and validates the format itself.
Implements, in pipeline order:

1. Darts double-array common-prefix search (`has_leaf`/`value`/`label`/
   `offset` bit layout as in darts_clone);
2. the Precompiled normalizer with HF's grapheme semantics (quirk included),
   over a UAX #29 extended-grapheme-cluster approximation built from
   `unicodedata` categories + explicit Hangul/RI/Prepend ranges, biased
   toward **over-joining** — provably behaviour-neutral here, because any
   cluster ≥ 6 bytes takes the same per-char path regardless of where its
   boundaries fall;
3. WhitespaceSplit against an explicit Unicode `White_Space` set — **not**
   Python's `str.isspace()`, which also claims U+001C–U+001F and would
   diverge from Rust's `char::is_whitespace` (a corpus entry with an
   embedded U+001C pins this);
4. Metaspace with the probed prepend/split semantics;
5. Unigram Viterbi mirroring `unigram/model.rs`'s `encode_optimized`: f64
   sums, forward DP with starts ascending and piece lengths ascending,
   strict `>` relaxation (first-seen wins ties), an unk node covering
   exactly one char added only where no single-char piece matches, and
   consecutive unk ids fused;
6. the `<s> … </s>` template.

CLI: `python tools\xlmr_tokenizer_ref.py <table.bin> <texts.txt>` — one text
per line in, one space-separated id line out, the same shape the future C++
CLI will have (`verify_tokenizer_gemma.py` precedent).

**Known limitation, deliberate and shared with the future C++ port**: no
added-tokens splitter, so a text containing a literal special-token string
(e.g. `<mask>`) goes through the Unigram model instead of being extracted
verbatim. Embedding inputs are plain text; the corpus excludes these; the
C++ task should decide whether to close or keep this.

### `tools/verify_tokenizer_xlmr.py` (`.venv-ref`)

Modeled on `verify_tokenizer_gemma.py`: both implementations run over the
same corpus, every id compared, per-category counts printed, any mismatch
detail (including the offending text) written **only** to the JSON report.
The corpus is 343 sequences in 19 categories — the 18 authored ones listed
at the top plus `nfd_decomposed`, which is *derived programmatically*
(`unicodedata.normalize("NFD", …)` over the Korean, Vietnamese/Turkish,
Norwegian, Hindi, Hebrew/Greek and Arabic categories, keeping the 52 that
actually change) so decomposed jamo and split diacritics are guaranteed
present rather than hoped for. Audited the authored entries' codepoints
after writing: the three "decomposed/stacked combining" NFKC entries do
carry combining marks, and the NBSP / thin-space / ZWSP / U+001C entries
carry exactly those codepoints.

## 5. Exact commands

```powershell
# 1. download (from repo root; ~17 MB total)
cd models\gte-multilingual-base
foreach ($f in "tokenizer.json","tokenizer_config.json","special_tokens_map.json","config.json") {
  curl.exe -sL -o $f "https://huggingface.co/Alibaba-NLP/gte-multilingual-base/resolve/main/$f" }
Get-FileHash -Algorithm SHA256 tokenizer.json,tokenizer_config.json,special_tokens_map.json,config.json

# 2. generate the table (stdlib-only python)
C:\Users\vegar\.conda\envs\iron\python.exe tools\gen_xlmr_tokenizer_table.py
#   -> models\gte-multilingual-base\xlmr_tokenizer.bin (5.07 MB)

# 3. verify byte-exact against HuggingFace (343 sequences, 19 categories)
.venv-ref\Scripts\python.exe tools\verify_tokenizer_xlmr.py
#   -> 343/343 exact, PASS; report in tasks\0127-t52-unigram-generator\verify_tokenizer_xlmr.json

# 4. reference CLI smoke (the shape the C++ CLI will be diffed through)
C:\Users\vegar\.conda\envs\iron\python.exe tools\xlmr_tokenizer_ref.py `
  models\gte-multilingual-base\xlmr_tokenizer.bin <texts.txt>
```

(Console note: on this machine the conda interpreter defaults stdout to
cp1252; every command above that prints non-ASCII was run with
`PYTHONIOENCODING=utf-8`. The C++ port owes nothing here — the blob and all
file I/O are byte-oriented.)

## 6. Problems hit, honestly

* **The task brief's f32 assumption was wrong for this checkpoint** —
  §3's 65,856 non-f32-exact scores. Caught by an explicit round-trip check
  in the generator before the format was frozen; the blob went f64 and the
  generator prints the count every run.
* **A content filter ended the first session run** during verification
  bring-up, apparently on raw corpus text in the transcript. Cost: one
  restart; fix: the hygiene rule at the top of this log. Worth keeping for
  every future multilingual-corpus task.
* **Two candidate "unknown codepoint" probes turned out to be vocab
  entries** (U+E000, U+E001) — the unk category was rebuilt around
  codepoints verified to actually miss the vocab. A reminder that on a
  250k-piece vocab, "surely this is unk" is an assumption like any other.
* **The Precompiled quirk (§3.1) would have been unfindable from
  sentencepiece's documentation** — it contradicts upstream's longest-match
  semantics. It was found by probing `normalize_str` on a 5-byte two-
  codepoint cluster before writing the normalizer, which is T29's
  "read/probe before writing code" lesson paying out a second time.
* **No wrong turns in the Viterbi itself**: the first full corpus run after
  the 25-case smoke set passed 343/343. The smoke set had already flushed
  out the ordering details (starts ascending, strict `>`), which were
  transcribed from `model.rs` rather than guessed.

## 7. What T52 step 2 (the C++ port) inherits

* `xlmr_tokenizer.bin` + the format table in §4 — everything needed, no
  JSON at runtime (rule 5).
* `tools/xlmr_tokenizer_ref.py` as the line-for-line spec: Darts walk,
  grapheme approximation (over-joining bias note included), White_Space
  set, Metaspace, Viterbi relaxation order, unk fusing.
* `tools/verify_tokenizer_xlmr.py` as the gate: point it at the C++ CLI the
  way `verify_tokenizer_gemma.py` points at `gemma_tok_cli.exe`, same
  343-sequence bar.
* Open decisions recorded above: added-tokens splitter (or not), and
  `gemma_tokenizer_gen.cpp:90-95`'s refusal of Unigram is still T52's TODO
  marker for the fresh-clone packing path.
