# 0061 — EmbeddingGemma-300M: C2 tokenizer (SentencePiece BPE, verified byte-identical)

- **Date** 2026-08-20
- **Milestone** M12 (research plan Del C, C2 only — first of C2-C4)
- **Status** done, scoped exactly as asked: tokenizer only, not wired in

## Goal

Build C2 from the approved plan (`~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`,
Del C) and from `tasks/0055-m10-embeddinggemma-spike/TASK.md`'s go-ahead: a
from-scratch C++ SentencePiece tokenizer for EmbeddingGemma-300M, structurally
parallel to `runtime/src/tokenizer.cpp` (BERT WordPiece) but implementing
whatever algorithm this checkpoint actually needs, verified byte-identical to
HuggingFace's own tokenizer over a real corpus including the task-prefix
protocol. **Explicitly out of scope, not attempted**: wiring into
`main.cpp`'s `Encoder::run()`, `hub.cpp`'s model catalogue, `arch=1`, RMSNorm/
RoPE/GeGLU, or any packer change. No NPU, no IRON, no Peano — a concurrent
session was doing NPU-heavy phase-fusion work tonight and this task never
touched `C:\dev\mlir-aie` or ran `npuembed.exe` against hardware.

## The load-bearing finding: this checkpoint is BPE, not Unigram

The task brief (and the plan it was scoped from) assumed **SentencePiece
Unigram** — reasonable by analogy with other sentence-embedding tokenizers,
and Unigram is what the brief asked to implement (Viterbi/forward-DP search
over (piece, log-prob) pairs). **Reading `models/embeddinggemma-300m/
tokenizer.json` directly, rather than trusting that assumption, shows
`model.type == "BPE"`**, with `byte_fallback: true`, `dropout: null`,
`continuing_subword_prefix`/`end_of_word_suffix` both null, and a
**514,906-row `merges` list** — not a Unigram `vocab` of scored pieces. This
is the standard Gemma/Llama-family SentencePiece **BPE** tokenizer (262,144
vocabulary), confirmed independently by loading the live checkpoint through
`transformers.AutoTokenizer` and inspecting `tok.backend_tokenizer` — its
class is literally `GemmaTokenizer`. Per this project's own precedent (M5's
"pre-tiling is a wash" correcting an earlier assumption), the assumption is
recorded and corrected rather than silently followed: **everything below
implements BPE**, and `tools/gen_gemma_tokenizer_table.py`'s module docstring
asserts `model.type == "BPE"` and refuses to run otherwise, so a future
checkpoint revision that actually ships Unigram fails loudly instead of
silently mis-tokenizing.

## The full pipeline, confirmed empirically (not read off the JSON alone)

Every claim below was checked by calling the real `transformers` fast
tokenizer (`.venv-ref`, `transformers 5.15.0`) and inspecting
`backend_tokenizer.normalizer` / `.pre_tokenizer` / `.encode()` directly —
not inferred from the plan's summary or from FastFlowLM's copy of the
prefix table.

- **Normalizer is exactly ONE rule**: literal `' '` (U+0020) → `'▁'`
  (U+2581, metaspace). No NFKC, no case folding, no accent stripping —
  confirmed by normalizing `"the quick brown fox"` and getting back
  `"the▁quick▁brown▁fox"` verbatim, case and all.
- **No automatic leading metaspace**, unlike the more common SentencePiece
  convention: `encode("Hello world")` → first token `"Hello"` (no leading
  `▁`); `encode(" Hello world")` → first token `"▁Hello"`. This is a
  configuration choice this checkpoint's normalizer does NOT make, and
  getting it backwards would silently shift every non-prefixed encode by one
  token's worth of leading-space ambiguity.
- **The pre_tokenizer is a no-op in practice.** It is configured to split on
  literal `' '`, but the normalizer already consumed every space by the time
  it runs, so `pre_tokenize_str()` on a 4-word sentence returns **one
  segment** for the whole string. Unlike BERT's per-whitespace-word
  WordPiece, **the entire normalized+prefixed text is one BPE input** — no
  artificial word boundary blocks a merge. This simplified the C++
  implementation materially: no per-word segmentation loop is needed at all.
- **BPE proper**: one symbol per Unicode codepoint initially (each
  codepoint's own UTF-8 string looked up in vocab); repeatedly merge the
  adjacent pair with the lowest merge rank until none remain. Standard
  linked-list + lazy-invalidation-heap implementation
  (`GemmaTokenizer::tokenize` in `runtime/src/tokenizer_gemma.cpp`).
- **byte_fallback, confirmed to actually fire**: a codepoint whose own string
  is not a vocab entry decomposes into its raw UTF-8 bytes, each becoming a
  `"<0xXX>"` symbol (uppercase hex — confirmed `'<0xF0>'` not `'<0xf0>'`; all
  256 entries exist, ids 238–493). Probed with an obscure Cuneiform codepoint
  (U+12031): tokenizes to `['<0xF0>','<0x92>','<0x80>','<0xB1>']`, four
  separate byte tokens, matching the C++ output exactly (verified below on
  300 such codepoints, not just this one).
- **post_processor always wraps `<bos> ... <eos>`** (`add_bos_token` /
  `add_eos_token` both `true` in `tokenizer_config.json`) — `<pad>=0
  <eos>=1 <bos>=2 <unk>=3 <mask>=4`.
- **truncation_side and padding_side are both `"right"`**, matching this
  project's existing BERT convention exactly (`room = max_len - 2`, take the
  first `room` body tokens, pad on the right with `<pad>`) — confirmed by
  truncating a 100-word input to `max_length=16` through the real tokenizer
  and reproducing it exactly in C++.

## What was built

1. **`tools/gen_gemma_tokenizer_table.py`** (263 lines) — build-time-only
   Python (CLAUDE.md rule 5: Python for design generation, never at
   runtime). Reads the checkpoint's `tokenizer.json` (33 MB of JSON, 262,144
   vocab entries, 514,906 merges) plus `config_sentence_transformers.json`'s
   `"prompts"` dict, asserts every pipeline assumption above (fails loudly if
   the checkpoint ever changes shape), and writes a flat binary table
   (`GEMATOK1` magic, version 1) the C++ side reads with `ifstream` +
   `memcpy` — no JSON, no protobuf, at runtime, exactly the role
   `tools/gen_tokenizer_tables.py` plays for BERT's Unicode tables. Merges
   are stored as `(id_a, id_b, merged_id)` triples in rank order, so the
   runtime never string-hashes a merge candidate — only the initial
   per-codepoint symbolization does a string lookup.
   Output: `models/embeddinggemma-300m/gemma_tokenizer.bin`, **8.60 MB**
   (gitignored — `models/**` already excludes everything but
   `CHECKPOINT.json`; regenerate with the command below).
2. **`runtime/include/tokenizer_gemma.hpp` + `runtime/src/tokenizer_gemma.cpp`**
   (107 + 393 lines) — the tokenizer itself. `GemmaTokenizer::from_table_file`
   / `from_table_bytes` load the binary table; `tokenize()` runs metaspace +
   BPE merge + byte-fallback; `encode()` wraps `<bos>` + task-prefix + text +
   `<eos>`, truncates/pads to `max_len`. **Zero XRT dependency** — pure STL,
   deliberately standalone (see the header's own comment) so it builds and
   runs without the NPU runtime and does not touch `main.cpp` or
   `hub.cpp`'s model catalogue.
3. **Task-prefix table**: all 14 rows of the checkpoint's own
   `config_sentence_transformers.json` `"prompts"` dict are baked into the
   binary (read directly from the checkpoint, not FastFlowLM's copy per
   `FASTFLOW_ISSUE.md`'s own caution about that repo). **Decision, recorded
   as a decision, not a checkpoint fact**: this project defaults to
   `"document"` (`"title: none | text: "`) when a caller doesn't name one —
   the checkpoint's own `default_prompt_name` is `null`, meaning
   sentence-transformers itself applies *no* prefix by default, but
   `"document"` matches how `tasks/0055`'s spike validated its goldens and
   how this project's own `--embed` CLI is used (embedding arbitrary text
   into a corpus — the retrieval "document" side). `encode()` takes a
   `prefix_name` parameter so any of the 14 can be selected, or `""` for the
   checkpoint's own raw default.
4. **`runtime/src/tokenizer_gemma_cli.cpp`** (69 lines) — a tiny **separate**
   executable (not a mode of `npuembed.exe`) that reads one text per line,
   encodes each, and prints space-separated `input_ids` — the same contract
   `tools/verify_tokenizer.py` already uses against `npuembed.exe
   --tokenize`. Built by hand with `cl.exe` directly (see commands below),
   **not through `runtime/CMakeLists.txt`** and not through the shared
   `runtime/build/` directory, specifically so this task's build could not
   collide with a concurrent session's NPU work in the same tree tonight.
5. **`tools/verify_tokenizer_gemma.py`** (152 lines) — reuses
   `tools/verify_tokenizer.py`'s adversarial corpus (accents, CJK,
   punctuation, out-of-vocab subwords, casing incl. Greek final sigma,
   degenerate inputs, emoji — 36 entries) plus `reference/corpus_gemma.py`'s
   four already-oracle-validated sentences, run through **five** task-prefix
   configurations (no prefix, `document`, `query`, `STS`, `Clustering`) against
   the live HuggingFace tokenizer.

## Commands

```powershell
# 1. Generate the binary table from the checkpoint's tokenizer.json (any Python 3)
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\gen_gemma_tokenizer_table.py

# 2. Build the standalone CLI (no CMake, no XRT, isolated from runtime/build/)
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd runtime\build_gemma_tok   # created fresh; gitignored
cmd /c "call `"$vcvars`" >nul && cl.exe /nologo /std:c++17 /EHsc /O2 /I ..\include ..\src\tokenizer_gemma.cpp ..\src\tokenizer_gemma_cli.cpp /Fe:gemma_tok_cli.exe"

# 3. Verify against HuggingFace (.venv-ref, transformers 5.15.0)
& ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer_gemma.py

# 4. Extra byte-fallback stress corpus (300 obscure codepoints -- Cuneiform,
#    Egyptian hieroglyphs, Tangut, PUA, variation selectors -- specifically
#    targeting the byte_fallback path, which the base corpus barely touches)
& ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer_gemma.py --extra-file <bytefallback_stress_corpus.txt> --out <bytefallback_result.json>
```

## Result

**210/210 exact (100.00%)** on the base run: 42 texts (36 adversarial + 4
reference/oracle sentences + 2 flattened) × 5 prefix configurations, `max_len
64`. `tasks/0061-m12-embeddinggemma-tokenizer/verify_tokenizer_gemma.json`.

**1,715/1,715 exact (100.00%)** including the byte-fallback stress corpus:
343 texts (the same 42 plus 301 sentences each built around one of 300
obscure codepoints spanning Cuneiform, Egyptian hieroglyphs (U+13000 block),
Tangut/other CJK-extension-adjacent ranges (U+18800), variation selectors
(U+E0000), and the Private Use Area (U+E000)) × 5 prefixes.
`tasks/0061-m12-embeddinggemma-tokenizer/verify_tokenizer_gemma_bytefallback_stress.json`,
corpus saved alongside as `bytefallback_stress_corpus.txt`.

**Every id, every position, every prefix, byte-for-byte identical to
HuggingFace's own tokenizer** — including truncation (tested up to a
100-word input against `max_length=16`, matches exactly), padding, the
`<bos>`/`<eos>` wrap, and the no-prefix control case. `1,925` total
sequence comparisons across both runs, zero mismatches.

## What this establishes and what remains

**Established**: the tokenizer half of C2-C4 is done and independently
verified — SentencePiece BPE with metaspace, byte-fallback, and the
task-prefix protocol, all reproduced from first principles in C++ with no
protobuf/JSON dependency at runtime. Combined with `tasks/0055`'s numpy
architecture oracle (`1-cos` 1.065e-07), **the two hardest correctness
questions for a full EmbeddingGemma integration (does the encoder math match,
does the tokenizer match) are both closed with measured evidence**, leaving
only orchestration and NPU-specific engineering (the `arch=1` runtime branch,
RMSNorm/RoPE/GeGLU host code, the MQA-aware packer, `.npue`'s vocabulary
container role for a 262k-entry table instead of a 30k-line `vocab.txt`) —
none of which was touched here.

**Not done, deliberately**: no wiring into `main.cpp`, `hub.cpp`, or any
`.npue` packer. `GemmaTokenizer` currently loads its table from a loose file
(`from_table_file`) the same way `npue::Tokenizer::from_vocab_file` did
before `tasks/0036` folded BERT's vocabulary into the `.npue` container —
that same step (embedding `gemma_tokenizer.bin`'s bytes as a raw tensor,
`from_table_bytes` already supports it) is a natural, small follow-on
whenever the packer work happens, not attempted here.

## Problems hit

- **The single biggest risk was the wrong assumption in the task brief
  itself** (Unigram vs. BPE) — caught only by reading `tokenizer.json`
  directly before writing any code, rather than trusting the plan's summary.
  Had the Unigram (Viterbi/log-prob) algorithm been implemented against this
  checkpoint's actual BPE+merge-rank vocabulary, it would have compiled,
  run, and produced *plausible-looking wrong tokenizations* — the exact
  failure mode CLAUDE.md's traps section warns about repeatedly (a wrong
  implementation that looks right until compared against real ground truth).
- **`UnicodeEncodeError` on Windows stdout with the metaspace character**
  (`▁`, U+2581) when inspecting `tokenizer.json` in Python — the same
  `cp1252`-default-console trap `research/` notes and `make_goldens.py`
  already documented; worked around with `PYTHONIOENCODING=utf-8` /
  `sys.stdout.reconfigure`, same fix as prior tasks.
- **`cl.exe`'s `/Fo`/`/Fe` path quoting broke under PowerShell's `cmd /c`
  bridging** when passing absolute output paths with a trailing backslash
  before a closing quote (`\"` is interpreted as an escaped quote, not
  path-separator-then-quote) — worked around by `cd`-ing into the output
  directory first and using bare filenames for `/Fe`, avoiding the trailing
  backslash entirely.
- **No blockers.** The checkpoint, `.venv-ref` (`transformers 5.15.0`), and
  `cl.exe` (via `vcvars64.bat`) were all already in place from tonight's
  earlier work; nothing needed installing. `ironenv` and the NPU were never
  touched, as scoped.

## Artifacts

- `tools/gen_gemma_tokenizer_table.py` — build-time table generator (committed).
- `runtime/include/tokenizer_gemma.hpp`, `runtime/src/tokenizer_gemma.cpp` —
  the tokenizer (committed).
- `runtime/src/tokenizer_gemma_cli.cpp` — standalone verification CLI (committed).
- `tools/verify_tokenizer_gemma.py` — verification harness (committed).
- `models/embeddinggemma-300m/gemma_tokenizer.bin` — generated binary table,
  8.60 MB (gitignored, regenerate with the command above).
- `runtime/build_gemma_tok/` — hand-built CLI binary (gitignored, new
  `.gitignore` entry added, regenerate with the command above).
- `tasks/0061-m12-embeddinggemma-tokenizer/verify_tokenizer_gemma.json`,
  `verify_tokenizer_gemma_bytefallback_stress.json`,
  `bytefallback_stress_corpus.txt` — verification results (committed).
- `research/OPEN-THREADS.md` — T29 updated with this task's result.

## Next

If the user wants to continue toward a running EmbeddingGemma encode: the
natural next slice is C3's runtime branch (`arch=1` in `Encoder::run()`),
which needs RMSNorm/RoPE/GeGLU host code and can be built and *numerically*
verified against `reference/encoder_gemma.py`'s goldens without touching the
NPU at all (host-only GEMM via a reference matmul, exactly how `tasks/0055`
validated the architecture) — the packer and NPU-specific work (MQA-aware
QKV tiling, `tile_n=16`) can follow once that is solid. `0055`'s performance
prior (~62–72 seq/s, 2.5–3× slower than bge-base) still stands unrevised;
nothing in this task changes it.
