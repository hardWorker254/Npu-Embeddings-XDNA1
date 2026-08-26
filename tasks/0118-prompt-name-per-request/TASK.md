# 0118 — the task prompt becomes a per-request field, and loses its default

**Date**: 2026-08-26
**Threads**: files [T43](../../research/OPEN-THREADS.md#t43) and
[T44](../../research/OPEN-THREADS.md#t44). Touches no open thread.

## Goal

Two things, from the user:

1. `--prefix` is stuck at the wrong granularity. It is resolved once per process
   (`resolve_prefix()`, `runtime/src/main.cpp`) and applied to every request for
   the life of `serve`. A RAG deployment needs `search_query` **and**
   `search_document` in the same session, so today that means two processes,
   each holding an `hw_context` on a shared NPU. Cut it, and make the prompt a
   required part of the request, answered with **HTTP 400 naming the valid
   prompts** when it is missing.
2. Assess `tencent/WeMM-Embedding-2B` for the 0.4.0 catalogue, then three more:
   `ibm-granite/granite-embedding-small-english-r2`,
   `ai-sage/Giga-Embeddings-instruct-480M-0826`,
   `nvidia/Nemotron-3-Embed-1B-BF16`.

Outcome: (1) is built and verified. (2) is priced and **filed in the register**;
nothing was adopted, on the user's decision that 0.4.0 lands the prompt change
alone.

## How the API shape was chosen

Worth recording, because "just add a field" had four defensible answers and the
one this repo picked is not the most common one.

**OpenAI itself has no such field** — `input`, `model`, `encoding_format`,
`dimensions`, `user`, and nothing else. It never needed one: `text-embedding-3-*`
are symmetric. So every server hosting asymmetric models invented something, and
they do not agree:

| API | field | where | required? |
|---|---|---|---|
| Cohere | `input_type` | native | **yes**, v3+. Values `search_document`, `search_query`, `classification`, `clustering`, `image` |
| vLLM | `input_type` | **on** `/v1/embeddings` | no; whitelist from `config_sentence_transformers.json`'s `prompts` |
| TEI | `prompt_name` | native `/embed` **only** | no; plus a server-level `--default-prompt-name` |
| Voyage | `input_type` | native | no |
| Jina | `task` | native | no |
| Ollama, llama.cpp | — | — | put the prefix in the text yourself |

TEI's split was verified against its own OpenAPI spec: `OpenAICompatRequest` is
`{dimensions, encoding_format, input, model, user}` — no `prompt_name` — while
`EmbedRequest` has it. **TEI deliberately kept it off the OpenAI route and vLLM
deliberately put it on.** So this is a judgement call, not a convention.

Three observations that decided it:

1. **nomic's four prompt names ARE Cohere's `input_type` values, verbatim.**
   nomic adopted Cohere's vocabulary.
2. **"Required, else error" is Cohere's behaviour** — the vendor that coined
   those names. Everyone else made it optional *with a server-level default*,
   which is precisely the design being removed here.
3. But **`input_type` would be a lie for EmbeddingGemma**, whose table holds 14
   sentence-transformers keys: `STS`, `BitextMining`, `Summarization`,
   `Reranking`… Those are not input types. The container already calls the table
   `prompts`, so **`prompt_name`**, on `/v1/embeddings`, required.

A fourth option was considered and rejected: encoding the prompt in the `model`
field (`nomic-embed-text-v1.5/search_query`), which would keep 100% OpenAI
compatibility and is tempting here because `serve` **ignores `model` entirely**
today. Rejected as a naming hack that would make `/v1/models` five rows for one
model.

## What was built

### The runtime

- **`EmbedBackend` carries the whitelist, not a resolved string.**
  `std::string prefix_text` → `std::vector<std::string> prompt_names`, and
  `embed` gains a `const std::string &prompt_name` parameter.
  Deliberately the **names** and not the table: arch 1 keeps its prompts in the
  `GEMATOK1` tokenizer blob and arch 0/2 keep theirs in the container config.
  Unifying those two stores was explicitly **not** attempted — each of the three
  wiring sites fills `prompt_names` from whichever store it already reads and
  resolves name → text inside its own lambda, so `serve_http()` validates
  against one vector and never learns which store answered. That is what kept
  this task small.
- **Three refusals in `serve_http()`**, each mirroring a `resolve_prefix()`
  counterpart so the CLI and the endpoint cannot drift: table + no name;
  no table + a name; a name not in the table. All three **list** what the model
  really offers.
- **`json_field_kind()`** in `runtime/include/http.hpp`. `json_field_string()`
  returns its fallback for both absent and empty, and those are now different
  answers — a missing `prompt_name` is a refusal, `""` is an explicit request
  for no prompt. `Other` is a third state rather than being folded into
  `Absent`, so a `null` prompt_name reports as malformed instead of as omitted.
- **`/health` advertises the table**: `prompt_names` + `prompt_required`,
  replacing the old `"prefix"` key. Emitted only when the table is non-empty, so
  a BERT model's `/health` is byte-identical to before.
- **The container default no longer applies.** `resolve_prefix()`'s fall-through
  to `g_prompt_default` is gone, and so is `flag_val("--prefix", "document")` on
  the Gemma path — **the last silent default in the runtime**, and the
  worst-placed one, on a checkpoint that names no default among its 14 prompts.
- **`serve` rejects `--prefix`** rather than ignoring it, naming the request
  field. `embed`/`--bench` keep the flag, required, with `--prefix ""` meaning
  none.
- `prompt_default` stays in the container and stays validated, as **advisory**
  metadata for harnesses choosing which prompt to exercise.

### Harnesses

- `tools/verify_endpoint.py` — had **zero** prefix coverage. Now reads
  `/health` first (which is also how it proves discovery works), carries the
  prompt through every `create()` via the official client's `extra_body`, passes
  the same prompt to the `--embed` cross-check, and adds the refusal cases plus
  the one check that proves the feature. Also gained `--artifacts`, which was
  hardcoded to `artifacts_b128il` and limited the harness to hidden 384.
- `experiments/m8-npu-vs-cpu/compare_three.py`, `npu_encoder.py` — stop passing
  `--prefix ""` (see Problems).
- `tools/verify_embed_e2e.py` — comment only; it already passed `--prefix`
  explicitly.

## Commands run

```powershell
# build
.\build_runtime.bat

# baseline for the bit-identity check: HEAD, built in a worktree
git worktree add /tmp/npue-base HEAD
cmake -S runtime -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ; cmake --build build

# refusals
npuembed embed nomic-embed-text-v1.5 in.txt --root <repo>
npuembed embed embeddinggemma-300m   in.txt --root <repo>
npuembed embed bge-base-en-v1.5      in.txt --prefix search_query --root <repo>
npuembed embed nomic-embed-text-v1.5 in.txt --prefix nonsense     --root <repo>
npuembed serve nomic-embed-text-v1.5        --prefix search_query --root <repo>
npuembed serve embeddinggemma-300m          --prefix document     --root <repo>

# bit-identity: OLD default vs NEW explicit
<old>\npuembed embed nomic-embed-text-v1.5 in.txt old.f32 --root <repo>
<new>\npuembed embed nomic-embed-text-v1.5 in.txt new.f32 --root <repo> --prefix search_document
cmp old.f32 new.f32

# endpoint, three models
npuembed .. --model all-MiniLM-L6-v2      --artifacts artifacts_b128il      --serve 8420
npuembed .. --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16 --serve 8420
npuembed .. --model embeddinggemma-300m   --artifacts artifacts_gemma_bfp16 --serve 8420
.venv-ref\Scripts\python.exe tools\verify_endpoint.py --port 8420 --model <m> --artifacts <a>
```

## Results

**The numerical gate — bit-identical on all three model families.** The refactor
threads a name where a string used to be captured; nothing about the arithmetic
was allowed to move.

| model | comparison | `cmp` |
|---|---|---|
| nomic-embed-text-v1.5 | OLD no flag (default `search_document`) vs NEW `--prefix search_document` | **identical** |
| bge-base-en-v1.5 | OLD no flag vs NEW no flag (no prompts table either way) | **identical** |
| embeddinggemma-300m | OLD no flag (silent `document`) vs NEW `--prefix document` | **identical** |

**Refusals**, all exit 2, all listing the real options:

| case | message |
|---|---|
| nomic `embed`, no `--prefix` | lists `[classification, clustering, search_document, search_query]` |
| gemma `embed`, no `--prefix` | lists all **14** |
| bge-base `embed`, `--prefix search_query` | "this model has no task prefixes" |
| nomic `embed`, `--prefix nonsense` | lists the four |
| `serve` + `--prefix` (both archs) | "does not apply to `serve` … send `prompt_name`" |

**Endpoint** — `tools/verify_endpoint.py`, all three PASS. Artifacts:
`verify_endpoint_nomic.json`, `verify_endpoint_gemma.json` here;
`tasks/0037-m9-tiers-endpoint/verify_endpoint.json` re-run for MiniLM.

| | MiniLM | nomic | gemma |
|---|---|---|---|
| `/health` prompts | none | 4 | 14 |
| missing `prompt_name` | 400 (field refused: no table) | 400, lists all 4 | 400, lists all 14 |
| unknown / `null` | 400 / 400 | 400 / 400 | 400 / 400 |
| **two prompts, one session** | n/a | **differ**, cos +0.9472 | **differ**, cos +0.7427 |
| endpoint vs `--embed` | 0.000e+00 | 0.000e+00 | 0.000e+00 |

That last row is worth noting on its own: with the same prompt applied on both
sides the endpoint and the batch path are **bit-identical**, not merely within
the harness's 1e-3 tolerance — so the CLI and HTTP prompt paths provably agree.

Manual cross-check on nomic over HTTP, showing `usage.prompt_tokens` tracks the
prompt as it should:

| `prompt_name` | tokens | first 3 |
|---|---|---|
| `search_query` | 17 | 0.01629, 0.04062, −0.15351 |
| `search_document` | 17 | 0.00108, 0.04689, −0.11657 |
| `""` (none) | **13** | 0.01005, 0.05085, −0.14652 |

## Problems hit

1. **`compare_three.py` has been passing `--prefix ""` to BERT models since
   0071, and that is a refusal, not a no-op.** `--prefix "" ` on a model with no
   prompts table hits `resolve_prefix()`'s first branch and throws — verified
   against the *unmodified* code path, so this is **pre-existing at HEAD**, not
   caused by this task. Fixed by omitting the flag entirely when there is no
   name (`*(["--prefix", n] if n else [])`), in both `compare_three.py` and
   `npu_encoder.py`. Whether the release sweep ever exercised that path on a
   BERT model is not established here.

2. **`"prompt_name": ""` on a model with no prompts table is a 400, not a 200.**
   Arguable: `""` asks for no prefix, which is exactly what a BERT model does
   anyway, so the request is satisfiable. Refused for consistency — the CLI
   already refuses `--prefix ""` there, and one behaviour beats two. The cost is
   that a generic client which always sends `prompt_name` must special-case the
   BERT models; the benefit is that such a client breaks loudly instead of
   silently getting unprompted vectors on four of the six models.

3. **`json_field_string()` could not express the distinction the feature needs.**
   Absent and `""` both returned the fallback. Solved with `json_field_kind()`
   rather than a sentinel default, because a sentinel is guessable input.

4. **Incidental, not fixed**: `serve <model> <port>` ignores the positional port
   on the BERT path (it listened on 8080 when told 8137) while `run_gemma_mode`
   honours it. `--port` works on both. Not touched — out of scope, and recorded
   here so the next session does not rediscover it as a bug in this change.

5. Two `.py` files in the tree are CRLF (`npu_encoder.py`,
   `verify_embed_e2e.py`, `verify_endpoint.py` is LF) — patch scripts that
   assume one or the other silently match zero occurrences. Not a code problem;
   a note for whoever scripts an edit across the tree next.

## What was NOT done

- No model was added. The four candidates are priced in
  [T44](../../research/OPEN-THREADS.md#t44), the tokenizer that gates all four
  in [T43](../../research/OPEN-THREADS.md#t43).
- The two prompt stores (GEMATOK1 blob vs container config) were **not**
  unified. They still disagree about where a model's prompts live; only their
  *names* now meet, inside `EmbedBackend`.
- `hub.cpp`'s `add`-path fail-open (`model_type` falls through to BERT with no
  `else throw`, so an unrecognised architecture is accepted and fails later at a
  tensor-name lookup in `npue_pack.cpp`) was found while mapping and is **not**
  fixed. It belongs to `add`, not to this change.
- The whole-catalogue release sweep has **not** been re-run. This task touches
  the encode path on every model, so 0.4.0 must not ship without it.
