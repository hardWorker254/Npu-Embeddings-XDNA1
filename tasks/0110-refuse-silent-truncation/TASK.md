# 0110 — silent truncation becomes an error, and `seq` becomes an export flag

- **Date** 2026-08-25
- **Status** done — the runtime refuses an input it cannot hold, on all four
  encode paths and over HTTP; `tools/export_gemm_rtp.py --seq` replaces the
  `SEQ = 64` module constant. Golden gate re-run and PASSES on arch 0, 1
  and 2. **No long-sequence design has been exported or measured** — see
  §"What is NOT done".

## Goal

Two things, in priority order, from a user report that a downstream consumer
was getting plausible-but-wrong embeddings:

1. **Stop failing silently.** A text longer than the design's sequence length
   was cut at `max_len` and embedded anyway. That is the failure; the cap
   itself is not.
2. **Make longer strings possible at all.** `SEQ = 64` was a module-level
   constant in the export tool with no flag.

## Context

Read: `CLAUDE.md` traps 6b / 6c / 7c / 7d (the "stale/wrong value fails open"
class this belongs to), `docs/04-model/npue-format.md`,
`tasks/0037` (batch tiers share one xclbin), `tasks/0069`/`0074`
(design.json fields that exist because inference from geometry was a
fail-open).

## The bug

`Tokenizer::encode()` and `GemmaTokenizer::encode()` both did:

```cpp
const int room = max_len - 2;                     // [CLS] ... [SEP]
const int take = std::min<int>(toks.size(), std::max(0, room));
...
e.n_tokens = static_cast<int32_t>(e.input_ids.size());
```

`n_tokens` is therefore **capped at `max_len` by construction**, so a
truncated input is byte-for-byte indistinguishable from one that happened to
fit — including to `usage.prompt_tokens` in the HTTP response, which could
only ever report the cut count. Nothing anywhere recorded that tokens were
dropped.

The blast radius is the answer, not the run. A truncated text still returns a
correctly shaped, correctly normed, deterministic vector, so every existing
guard passes.

### Demonstrated, not argued

Two real-shaped Norwegian saksframlegg sharing an administrative preamble and
differing entirely after it:

```
$ ./runtime/build/npuembed.exe . --tokenize .../two_shared_preamble.txt 64 \
      --model models/all-MiniLM-L6-v2.npue
  NOTE  2 of 2 lines exceeded max_len=64 and were truncated. ...
101 7842 5705 27843 19968 13910 2290 25312 ... 14540 25592 1010 102
101 7842 5705 27843 19968 13910 2290 25312 ... 14540 25592 1010 102
```

**Byte-identical token sequences.** And end to end, under the new
`--allow-truncation` escape hatch which reproduces the old behaviour exactly:

```
$ ... --embed two_shared_preamble.txt --cpu --allow-truncation
  WARNING  --allow-truncation: input 0 is 123 tokens and was CUT to 64. ...
  [0] +0.0282 +0.1094 +0.0005 -0.0414 -0.0311 +0.0512 ...
  [1] +0.0282 +0.1094 +0.0005 -0.0414 -0.0311 +0.0512 ...
```

Two different documents, one vector. Retrieval between them is a coin flip
wearing a similarity score.

## What was changed

### 1. The tokenizers record what they were given

`npue::Encoded` and `npue::GemmaEncoded` each gain `n_tokens_full` (what
`n_tokens` would have been with unlimited room) and `truncated`. `n_tokens`
keeps its old meaning so nothing downstream shifts. For the Gemma tokenizer
the count **includes the task prefix**, because the prefix is prepended
before tokenization and spends the same budget — a caller sizing text against
`max_len` alone can legitimately overflow by the length of
`"title: none | text: "`.

### 2. `npue::InputTooLong`, a distinct exception type

In `npue.hpp`, deriving from `std::runtime_error` but caught separately, so
the HTTP layer can answer **400 `invalid_request_error`** rather than 500.
Carries `index` / `n_tokens` / `limit` so the message names the caller's own
input and its real token count.

### 3. One policy, four enforcement points

`check_truncation()` in `main.cpp`, called from:

| path | site |
|---|---|
| BERT serve/embed | `EmbedService::chunk()` |
| Gemma NPU | `GemmaNpuEncoder::encode_batch()` |
| Gemma CPU | `GemmaEncoder::encode_one()` (policy as an argument, not a global — it is a library type) |
| `--tokenize` | **deliberately not an error**; see below |

Default is refuse. `--allow-truncation` restores the old behaviour exactly and
warns once per run on stderr.

`--tokenize` is the mode `tools/verify_tokenizer.py` drives to diff against
HuggingFace token for token. HuggingFace truncates too, so refusing would make
the two incomparable at exactly the lengths worth comparing, and an extra
stdout line would desynchronise the diff. It reports a **count on stderr**;
stdout is unchanged.

### 4. Two latent crashes fixed, both made reachable by the above

- **`EmbedService::embed()`'s multi-lane branch had no exception handling
  at all.** An exception escaping a `std::thread` entry point calls
  `std::terminate`. This was survivable only while nothing on the path threw;
  the new refusal throws on ordinary bad input. Now captures the first
  `exception_ptr`, stops handing out work, rethrows on the joining thread.
- **The Gemma lane runner flattened exceptions to `what()`** and rethrew as
  `std::runtime_error`, which would have erased `InputTooLong`'s type one
  rethrow before the HTTP layer that needs it. Now an `exception_ptr`.

### 5. `--seq` replaces `SEQ = 64`

The finding that makes this cheap: **`seq` is not compiled into anything.**
It reaches a design through exactly one expression, `M = batch * seq` in
`shapes_for()`, and the instruction streams only ever know `M`, `K`, `N`. The
runtime inverts the split at load time (`batch = design.M / design.seq`). So
`batch 128 × seq 64` and `batch 16 × seq 512` are the **same M = 8192 and the
same arithmetic** — seq is a host-side slicing convention over a fixed row
count.

**Trap 7d does NOT apply here, and this was checked rather than assumed.**
`M` is a `CompileTime[int]` *keyword argument* to `pretiled_array()`
(`gemm_pretiled.py:762`), so the JIT cache key derives from it. 7d would have
bitten had `shapes_for()` been a generator reading `SEQ` via `LOAD_GLOBAL`;
it is a plain helper that computes `M` and passes it down.

The flag validates the runtime's own rules up front (`seq % 8 == 0`;
`batch*seq % (m*4) == 0`) so a bad value costs a second rather than a full
export followed by a load-time refusal, and prints an explicit "throughput is
unknown until traced" line whenever `--seq` differs from the default.

## Commands run

```powershell
cmake --build runtime\build --config Release

# the failure, reproduced
.\runtime\build\npuembed.exe . --tokenize <two_shared_preamble.txt> 64 `
    --model models\all-MiniLM-L6-v2.npue
.\runtime\build\npuembed.exe . --embed <two_shared_preamble.txt> --cpu `
    --allow-truncation --model models\all-MiniLM-L6-v2.npue

# the refusal, CPU and NPU
.\runtime\build\npuembed.exe . --embed <two_shared_preamble.txt> --cpu `
    --model models\all-MiniLM-L6-v2.npue          # exit 2
.\runtime\build\npuembed.exe . --embed <two_shared_preamble.txt> `
    --model models\all-MiniLM-L6-v2.npue          # exit 2

# over HTTP
.\runtime\build\npuembed.exe serve all-MiniLM-L6-v2 --port 8137 --root .
curl ... -d '{"input": "<324-char text>"}'                     # HTTP 400
curl ... -d '{"input": ["short one", "<324-char text>"]}'      # HTTP 400, names input 1

# golden gate, three architectures
.\runtime\build\npuembed.exe . --model models\all-MiniLM-L6-v2.npue
.\runtime\build\npuembed.exe . --model models\bge-base-en-v1.5.npue --artifacts runtime\artifacts_base
.\runtime\build\npuembed.exe . --model models\nomic-embed-text-v1.5.npue --artifacts runtime\artifacts_nomic
.\runtime\build\npuembed.exe . --model models\embeddinggemma-300m.cpp_test.npue --embed <one_short.txt> --cpu
```

## Results

| check | result |
|---|---|
| MiniLM golden (arch 0) | **PASS** — worst 1-cos `3.397e-04`, tol 2e-03 |
| bge-base golden (arch 0) | **PASS** — worst 1-cos `1.353e-05` |
| nomic golden (arch 2) | **PASS** — worst 1-cos `2.599e-05` |
| EmbeddingGemma (arch 1) | encodes; long input refused at 106 tokens (prefix included, as documented) |
| short input, NPU, before vs after | `+0.0700 +0.1215 +0.0466 +0.0206 -0.0734 +0.0408` — **unchanged** |
| long input, CPU / NPU / HTTP | exit 2 / exit 2 / **HTTP 400 `invalid_request_error`** |
| batch [short, long] over HTTP | 400 naming **input 1**, the caller's index, not a tier-local row |

The behaviour change is confined to inputs that were previously being answered
wrong: anything that fitted before still fits and returns the bit-identical
vector.

## Problems hit

1. **`pathlib.write_text()` on Windows rewrote `main.cpp` to CRLF.** The file
   was LF in the working tree; `write_text` defaults to `newline=None`, which
   translates `\n` to `os.linesep`. The diff went to **13,337 changed lines**
   and was unreviewable. Fixed by normalising back to LF and doing all
   subsequent edits on **bytes**. The other eight files were already CRLF, so
   they were unaffected — which is why the damage showed up in exactly one
   file and looked like a real change.
2. **The Bash tool's heredoc ate one level of backslash**, three times, so
   `\n` and `\r` inside inserted C++ literals arrived as **real control
   characters** — an unterminated-string compile error, or worse, a silently
   changed literal. Do not put backslash escapes in a heredoc here; use the
   Write/Edit tools or `chr(10)`.
3. **A pre-existing `npuembeddings` server on port 8080 answered my curl.**
   The reply said `"model":"nomic-embed-text-v1.5-npu"` while I had launched
   MiniLM — Windows let both bind. This is `7c` wearing a network port:
   *never identify a build artifact by the fact that something answered.*
   Match on contents; here, the `model` field in the response. Re-ran on a
   distinct port via the `serve` subcommand (`--port` is **not** honoured in
   the `--serve` flag form — pre-existing, not fixed here).
4. `models/embeddinggemma-300m.npue` fails to load with
   `no tensor named layer.0.q_proj`. **Pre-existing and unrelated** — it fires
   during container load, before any tokenization. `embeddinggemma-300m.cpp_test.npue`
   was used instead for arch-1 coverage.

## What is NOT done

**No long-sequence design has been exported, and none has been measured.**
`--seq` makes it expressible; it does not make it fast, and nothing here
predicts what it costs. Three things a follow-up has to face:

1. **Host attention is O(seq²).** F3 prices attention at 2–5% of the work
   *at seq 64*. At 512 that is 64× the attention work per sequence and could
   plausibly dominate the encode. Unmeasured.
2. **Position tables.** `set_design_seq()` already refuses
   `seq > max_seq_len`, and the packer's `--max-seq` defaults to **256** — so
   a seq-512 BERT design needs a repack, not just a re-export. nomic is RoPE
   and has no such table. This one is already loud; it just fires at load.
3. **The batch/seq trade is not free at constant M.** `batch = M / seq`, so
   seq 512 at M 8192 means batch 16 — the tier table and the right-sizing
   logic in `plan()` would want revisiting for request shapes that no longer
   fill a tier.

Also unaddressed: `--port` is ignored in the `--serve` flag form (§Problems 3).
