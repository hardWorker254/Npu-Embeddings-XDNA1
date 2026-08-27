# 0133 — T52 step 2: the C++ side of the XLM-R Unigram tokenizer

**Goal**: the C++ half of [T52](../../research/OPEN-THREADS.md#t52), mirroring
the gemma two-implementation discipline — a runtime tokenizer that reads the
`XLMRTOK1` blob ([`0127`](../0127-t52-unigram-generator/TASK.md)), a C++ port
of the table generator (fresh clones must pack without Python,
`gemma_tokenizer_gen.hpp`'s rationale), a verification CLI, and the three-way
byte-exactness proof: C++ generator vs Python generator (sha256), C++ encoder
vs the Python reference, C++ encoder vs HuggingFace. No wiring into
`main.cpp`/`hub.cpp` — the arch-3 integration is a later task.

**Result, all three on the FIRST full run:**

| comparison | result |
|---|---|
| C++ generator blob vs Python generator blob | **byte-identical** — sha256 `15bc6dd63d6b3c54e2b186993697661a2823f468eeb350bbfdb3f10ba73ebde9` both sides, `fc /b` no differences (5,318,988 B) |
| C++ CLI vs HuggingFace (`transformers` 5.15.0 / `tokenizers` 0.22.2) | **343/343** byte-exact, all 19 categories |
| C++ CLI vs `tools/xlmr_tokenizer_ref.py` | **343/343** byte-exact |
| (control) Python reference vs HuggingFace, re-run | 343/343, unchanged from 0127 |

Report: [`verify_tokenizer_xlmr.json`](verify_tokenizer_xlmr.json)
(mismatch detail slots empty).

**Corpus hygiene** (0127's rule, kept): raw multilingual corpus text appears
only in `tools/verify_tokenizer_xlmr.py` and the JSON report; this log and
the terminal transcripts carry category names, counts and ids only.

---

## 1. What was built

### `runtime/src/tokenizer_xlmr.cpp` + `runtime/include/tokenizer_xlmr.hpp` (runtime tokenizer)

Line-for-line port of `tools/xlmr_tokenizer_ref.py`, same section order and
method boundaries (`normalize` / `pre_tokenize` / `viterbi` / `encode`) so
the two diff function by function. Loads the blob from bytes
(`from_table_bytes`, for a future `.npue` tensor) or a file. Honours all
three of 0127's findings:

* **f64 end to end** — scores stored `double`, Viterbi sums `double`;
  no `float` anywhere on the scoring path.
* **HF's Precompiled grapheme quirk** — a cluster < 6 UTF-8 bytes whose
  prefix hits the charsmap trie is replaced whole by the first (shortest)
  match; otherwise per-char. The Darts double-array trie and the
  normalized-strings blob are walked **in place** from the blob (same bit
  layout as the reference's `DartsTrie`; the walk stops at the first leaf
  because only `results[0]` is ever consumed).
* **fuse_unk on; White_Space is the Rust set** — explicit codepoint set,
  not `isspace()`; consecutive unk ids fused in the backtrack.

Also ported verbatim: the UAX #29 extended-grapheme approximation (GB3–GB999
in the reference's exact rule order, over-joining bias included), Metaspace
prepend/split semantics, the strict-`>` first-seen-wins Viterbi relaxation
with starts ascending / piece lengths ascending, and the unk node covering
exactly one char where no single-char piece matches. Known limitation shared
with the reference, deliberate: no added-tokens splitter (recorded in 0127
§"Known limitation"; embedding inputs are plain text).

**One new generated header**: the reference classifies codepoints via
Python's `unicodedata.category()` (Mn/Me → Extend, Mc → SpacingMark,
Cc/Cf/Zl/Zp → Control), which C++ does not have. `tools/gen_xlmr_unicode_tables.py`
(new, mirrors `gen_tokenizer_tables.py` → `bert_unicode_tables.hpp`) emits
`runtime/include/xlmr_unicode_tables.hpp` — 343/182/23 inclusive ranges from
the same interpreter's unicodedata (UCD 15.1.0, stamped in the header), so
the two implementations agree by construction. Everything else the
classifier needs (Prepend set, Hangul, RI, Extended_Pictographic,
White_Space) is explicit ranges in the reference and is transcribed
directly.

### `runtime/src/xlmr_tokenizer_gen.cpp` + `runtime/include/xlmr_tokenizer_gen.hpp` (generator port)

Direct port of `tools/gen_xlmr_tokenizer_table.py`, every `SystemExit` guard
becoming a `std::runtime_error` in the same order against the same fields
(the `gemma_tokenizer_gen.cpp` discipline): refuses non-Unigram (pointing at
the BPE generator, the mirror image of that file's refusal),
`byte_fallback: true`, a bad `unk_id`, malformed/duplicate vocab rows, a
non-Precompiled normalizer, an inconsistent charsmap (size field vs blob
length, missing trailing NUL), a pre-tokenizer that is not exactly
[WhitespaceSplit, Metaspace], a wrong Metaspace replacement, an unknown or
self-contradictory prepend scheme, missing/unmarked/normalized specials,
added-token ids disagreeing with their own vocab rows, a post-processor that
is not exactly `<s> A </s>` (full shape check, extra keys fail — Python
compares dict equality), post-processor ids disagreeing with added_tokens,
and baked-in truncation/padding (0110's rule). Single input file
(`tokenizer.json`); reuses `json_min` unchanged; adds a strict base64
decoder (fail-closed where Python's `b64decode` would silently discard).

**json_min's f64 handling needed no fix.** `parse_number` delegates to
`strtod`, which on MSVC's UCRT is correctly-rounded decimal→double — the
same contract as Python's `float()`. Verified empirically, not assumed: the
blob's 250,002 raw-f64 scores (65,856 of them not f32-exact, 0127 §3) are
inside the sha256 that matched byte-for-byte.

### `runtime/src/tokenizer_xlmr_cli.cpp` (verification CLI)

Mirrors `tokenizer_gemma_cli.cpp`: XRT-free, one text per line in, one
space-separated id line out. Two deltas, both recorded in its header:

* `--gen <tokenizer.json> <out.bin>` mode runs the generator port, so the
  byte-identity check needs no throwaway program (0067 used one).
* Minimal input escaping (`\n`, `\r`, `\\`) — two corpus entries carry
  embedded newline codepoints, which a line protocol cannot ship raw. The
  verifier applies the mirror-image escape.

### `runtime/CMakeLists.txt`

`tokenizer_xlmr.cpp` + `xlmr_tokenizer_gen.cpp` added to `npuembed`'s source
list (same rationale as the gemma pair: XRT-free, ships the packer-side
generator; nothing in `main.cpp` calls them yet), and a new **`xlmr_tok_cli`**
target (the four XRT-free sources, `/W3 /EHsc /Zc:__cplusplus /O2`). Note
the gemma CLI never got a target — 0061 compiled `gemma_tok_cli.exe` by hand
with `cl.exe`; this one gets a real target so the verifier's `--cli` build
is reproducible.

### `tools/verify_tokenizer_xlmr.py` (extended, not forked)

New `--cli` argument. When given, the C++ CLI runs once over the full
corpus (escaped temp file) and every text is compared three ways: ref vs
HF (as before), **C++ vs HF**, **C++ vs ref**. Per-category `cpp` counts on
stdout, all three totals in the JSON, and PASS now requires all three at
100%. Without `--cli` the behaviour is 0127's, unchanged.

## 2. Exact commands

```powershell
# 0. regenerate the unicode-category header (committed; deterministic)
C:\Users\vegar\.conda\envs\iron\python.exe tools\gen_xlmr_unicode_tables.py
#   -> runtime\include\xlmr_unicode_tables.hpp (11.4 KB, UCD 15.1.0)

# 1. build (existing Ninja build dir; CMake re-ran itself on the edited list)
cd runtime
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake --build build --config Release'
#   -> [8/9] npuembed.exe, [7/9] xlmr_tok_cli.exe, zero warnings at /W3

# 2. verification (a): C++ generator blob vs the committed Python blob
#    (output to the session scratchpad -- models\gte-multilingual-base is
#     read-only for this task)
.\runtime\build\xlmr_tok_cli.exe --gen models\gte-multilingual-base\tokenizer.json <scratchpad>\xlmr_tokenizer.cpp_generated.bin
Get-FileHash -Algorithm SHA256 models\gte-multilingual-base\xlmr_tokenizer.bin, <scratchpad>\xlmr_tokenizer.cpp_generated.bin
cmd /c "fc /b models\gte-multilingual-base\xlmr_tokenizer.bin <scratchpad>\xlmr_tokenizer.cpp_generated.bin"
#   -> both 15bc6dd63d6b3c54e2b186993697661a2823f468eeb350bbfdb3f10ba73ebde9
#   -> FC: no differences encountered

# 3. verifications (b) + (c): C++ CLI vs Python reference vs HuggingFace
.venv-ref\Scripts\python.exe tools\verify_tokenizer_xlmr.py --cli runtime\build\xlmr_tok_cli.exe --out tasks\0133-t52-cpp-port\verify_tokenizer_xlmr.json
#   -> ref-vs-HF 343/343, C++ vs HF 343/343, C++ vs ref 343/343, PASS
```

## 3. Problems hit, honestly

* **The corpus cannot ride a raw line protocol.** Two `whitespace`-category
  entries embed U+000A / U+000D (checked by codepoint scan before writing
  the CLI, not discovered by a failing diff). Fixed with the three-escape
  scheme above rather than by flattening the texts (the gemma verifier
  flattens; that weakens exactly the category these entries exist to
  stress).
* **The reference leans on `unicodedata`, which the C++ side does not
  have.** Solved with the generated-ranges precedent
  (`bert_unicode_tables.hpp`) rather than hand-transcribed ranges — 548
  ranges across three merged classes is exactly the kind of table a human
  copies wrong. The UCD version is stamped in the header; if the Python
  ever moves to a different UCD, regenerate before trusting the diff.
* **The task brief expected a `tokenizer_gemma_cli` CMake target to mirror;
  there is none** — 0061 built it by hand. Decision: give the new CLI a
  real target (`xlmr_tok_cli`) instead of repeating the hand-build, so the
  verifier's `--cli` path is reproducible from a clean checkout.
* **What did NOT go wrong, for once**: no mismatch was ever observed — the
  generator was byte-identical and the encoder 343/343 on their first full
  runs. Credit belongs to 0127's executable-spec discipline: every
  behavioural trap (grapheme quirk, tie-breaking order, unk fusing,
  White_Space set, f64 scores) was already nailed down in the reference,
  and the port only had to follow it line by line. The `strtod` f64
  concern raised in the brief dissolved the same way — UCRT `strtod` is
  correctly rounded, and the sha256 identity is the proof it matched
  Python's `float()` on all 250,002 scores.

## 4. What T52 still needs (not this task)

* The arch-3 wiring: `encoder_implemented()` / `set_model_shape()`
  whitelist, `npue_pack` calling `generate_xlmr_tokenizer_table()` for a
  fresh fetch, the forward-pass deltas (GELU-on-gate, biases, NTK theta —
  [T44](../../research/OPEN-THREADS.md#t44)'s note), and `main.cpp`
  integration. Deliberately untouched here.
* The added-tokens-splitter decision, still open and still recorded in both
  implementations' headers.
