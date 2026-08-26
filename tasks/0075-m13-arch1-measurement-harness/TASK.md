# 0075 — teach the measurement harness arch=1, and cut CLAUDE.md down to size

**Goal (user, 2026-08-22, "0.4.0 #2"):** 0075, T34, a CLAUDE.md cleanup, and
then `npuembeddings add <model>`. Research is goal number one; the repository's
biggest value now is *"enorme mengde tekst som andre kan leke seg med"*.

Status: **DONE.** The M8 gate **PASSES at mean +0.00**, worst task -0.00 — the tightest
MTEB agreement this project has recorded. Getting there meant finding that the first run's
**-1.88 FAIL was the measurement harness, not the model** (§5b).

---

## 1. Why this is one task and not three

[T34](../../research/OPEN-THREADS.md#t34) listed three gaps for arch=1: no MTEB
score, no CPU ratio, no energy figure. They read as three jobs. They are one:
`run_mteb.py` and `compare_three.py` both build their NPU side through
`experiments/m8-npu-vs-cpu/npu_encoder.py`, and `energy_compare.ps1` drives the
same executable. One bridge, three consumers.

**And the bridge could not load arch=1 for a specific, structural reason.** It
tokenizes in *Python*, does the embedding lookup in *Python*, and hands the
runtime an embedding SUM plus masks through `--encode-file`:

```
text --[HF tokenizer, npu_encoder.py]--> emb_sum.f32 + masks
     --[npuembed.exe --encode-file]--> out.f32
```

That contract is BERT-shaped in three ways at once. arch=1 has **no position
table**, its input is `embed_tokens[id] · √hidden` rather than a three-way sum,
and its tokenizer is a C++ SentencePiece BPE with no Python equivalent in this
repo. The bridge died on a `KeyError` for
`embeddings.word_embeddings.weight` — a tensor that architecture never had.

**arch=1 already had a better door.** `--embed` is text in, vectors out, in one
process. Using it is *less* machinery than the BERT bridge, not more.

One consequence must be stated rather than discovered later: on this path **the
tokenizer is also under test.** For arch 0 and 2 both sides share HuggingFace's
tokenizer, which deliberately isolates the datapath. Here the NPU side
tokenizes in C++ and the CPU side in HuggingFace, so an MTEB delta covers the
tokenizer too. That is acceptable because
[`0061`](../0061-m12-embeddinggemma-tokenizer/TASK.md) measured them
**1,925/1,925 sequences byte-identical** — a documented equality, not an
assumption — but it is a wider comparison and the number should be read as
such.

---

## 2. The prefix, and why it had to move into the container

`run_mteb.py` reads the task prefix **once, from the container**, and applies
it to *both* sides, so the two agree by construction rather than by two
literals that can drift. EmbeddingGemma's prefixes lived only inside
`gemma_tokenizer.bin`, so its container had no `prompts` key at all — meaning
`prefix` would have been `""` on the CPU side while the exe applied its own
default `"document"` on the NPU side.

**That is the one failure this comparison exists to rule out.** The measured
MTEB delta would have been the prefix, reported as a datapath result.

So the packer now writes the checkpoint's own table into the container, from
`config_sentence_transformers.json`, **sorted by key**. Sorting is not
cosmetic: `runtime/include/json_min.hpp` stores objects in an `unordered_map`
and says so in its own header (`model.vocab` is a 262,144-entry object, so
anything order-preserving is expensive), which means the C++ mirror cannot see
source order at all. Order is not semantic in a JSON object; the content is,
and the content is verbatim.

`prompt_default` is `"document"` — **this project's choice, not the
checkpoint's**, whose own `default_prompt_name` is `null`. It matches what
[`0061`](../0061-m12-embeddinggemma-tokenizer/TASK.md) already picked for the
tokenizer table, and the packer refuses if that key is missing rather than
naming a row that does not exist.

The two routes are then **cross-checked at runtime rather than trusted**: the
exe prints `prefix='<name>' -> <text>`, and the bridge asserts that text equals
the container's. If the runtime does not report a prefix at all, the bridge
**refuses to measure** — an absent data source is not a negative reading
([`0040`](../0040-m9-honest-cpu-baseline/TASK.md)).

That check earned its keep immediately, on my own code: every one of this
model's prefixes ends in a space (`"title: none | text: "`), and the space is
what separates prefix from text. The parser `.strip()`ed it and the check
failed against a runtime that was correct. Right failure, wrong reason —
`.strip()` removed.

---

## 3. CLAUDE.md: 1,345 lines → 375

`CLAUDE.md` is loaded into every session. Its *Current state* section had grown
to **1,027 lines of dated update blocks** — 76% of the file — stacked newest
first from 2026-08-18 to 2026-08-22, with several entries directly contradicted
by the one above them.

Moved **verbatim** to [`docs/history.md`](../../docs/history.md), nothing
deleted (rule 3b: failures are the valuable part, and those blocks are where
the refuted claims live — the bandwidth cost model, the "missing 4,500
cycles", B-reuse, the pre-tiling win, the parity-width projection, each with
the measurement that killed it). What replaced it is current truth plus
pointers to the three files that own the rest: `docs/CURRENT_STATUS.md` for
numbers, `research/OPEN-THREADS.md` for open questions, `tasks/` for the diary.

`## Target model` was also stale — it still described "all-MiniLM-L6-v2 and
bge-small-en-v1.5, both running" when six models run. Replaced with **what
geometry the array actually wants**, which is the question that is asked when a
*seventh* model is proposed, stated as the four assertions
`gemm_pretiled.py::_build_design` actually makes, plus the rule 0074
established: **when a width does not divide, pad it — do not lower `tile_n`.**

---

## 4. Commands run

```powershell
# the container gains the prompts table; both packers stay byte-identical
.\.venv-ref\Scripts\python.exe tools\pack_npue.py `
    --model-dir models\embeddinggemma-300m --out models\embeddinggemma-300m.npue
.\.venv-ref\Scripts\python.exe tools\verify_pack_parity.py `
    --model-dir <abs>\models\embeddinggemma-300m --exe <abs>\runtime\build\npuembed.exe --tile-n 48

# MTEB, both sides, one session
.\.venv-ref\Scripts\python.exe -u experiments\m8-npu-vs-cpu\run_mteb.py `
    --model embeddinggemma-300m --artifacts artifacts_gemma `
    --threads 24 --pipeline 4 `
    --out tasks\0075-m13-arch1-measurement-harness\mteb-embeddinggemma-300m.json
```

---

## 5. Results

**Packer parity holds with the prompts table**: both packers produce sha256
`238047ef79635ea910d1df39b13ae1bc5d7b715059e8bb9292741129283a5180`
(json 71,299 → 72,001 B), and `models/embeddinggemma-300m.npue` has that
digest.

### MTEB — the M8 gate, arch=1's first ever

Run through the fixed harness, both sides in one session, five tasks at seq 64:

| task | CPU | NPU | delta |
|---|---:|---:|---:|
| STSBenchmark | 88.18 | 88.18 | -0.00 |
| SICK-R | 81.35 | 81.35 | +0.01 |
| STS12 | 79.33 | 79.34 | +0.00 |
| Banking77Classification | 91.29 | 91.29 | -0.00 |
| TwentyNewsgroupsClustering | 49.76 | 49.78 | +0.02 |
| **MEAN** | | | **+0.00** |

**PASS** — the gate is |mean| <= 0.5 and no task worse than -0.5. Worst single task
**-0.00**.

That is the tightest agreement in the project: MiniLM +0.04, bge-small -0.03, nomic +0.09,
**EmbeddingGemma +0.00**. It is also a complete confirmation of §5b's diagnosis — the
first run's -1.88 mean and -3.49 worst were the prompt asymmetry and nothing else. Every
task moved to within 0.02 points once both sides used the same prompt.

Worth stating for what it does and does not prove: this shows the **bf16 array datapath plus
the C++ SentencePiece tokenizer** together preserve embedding quality on this architecture,
because on this bridge the tokenizer is under test too (§1). It says nothing about int8,
which is [T20](../../research/OPEN-THREADS.md#t20)/[`0077`](../0077-m13-int8-gate/TASK.md).

*(interleaved CPU ratio and energy: still pending — both now unblocked by the same harness
work, and both are timing measurements that should not share a machine with an MTEB run.)*

---

## 5b. THE M8 GATE FAILED, AND THE FAILURE WAS THE HARNESS

The first MTEB run reported a clean **FAIL**:

| task | CPU | NPU | delta |
|---|---:|---:|---:|
| STSBenchmark | 87.96 | 85.08 | **-2.88** |
| SICK-R | 82.04 | 81.30 | -0.74 |
| STS12 | 78.89 | 76.54 | **-2.36** |
| Banking77Classification | 91.04 | 91.11 | +0.07 |
| TwentyNewsgroupsClustering | 51.56 | 48.07 | **-3.49** |
| **mean** | | | **-1.88** |

**It did not survive one question: how can a model at `1-cos` 9.96e-06 lose 1.88 MTEB
points?** That fidelity is TIGHTER than nomic (2.6e-05) and bge-base (1.35e-05), both of
which pass. Two embedding matrices agreeing to 1e-05 cannot move a Spearman correlation by
2.4 points. So either the fidelity figure did not describe the MTEB corpus, or the two sides
were not being handed the same problem.

Three hypotheses, each killed by a measurement rather than an argument:

1. **The CPU side is not truncating to seq 64** (it would then see strictly more text, and
   the losses would track text length — which they appeared to: Banking77's one-line queries
   at +0.07, long newsgroup posts at -3.49). **Refuted**: `max_seq_length = 64` propagates to
   the Transformer module and `tokenize` returns `(1, 64)`.
2. **The C++ tokenizer diverges at the truncation boundary.** On this bridge the tokenizer is
   under test, unlike arch 0/2 where both sides share HuggingFace's, and
   [`0061`](../0061-m12-embeddinggemma-tokenizer/TASK.md)'s 1,925 comparisons may never have
   hit truncation. **Refuted**: 12 deliberately over-length texts (up to 20x the budget, plus
   long-word, punctuation-run, CJK and emoji stress cases) added to
   `verify_tokenizer_gemma.py` -> **270/270 exact**.
3. **The datapath degrades on long text.** **Refuted**: 64 MTEB-like texts that all truncate,
   NPU vs sentence-transformers directly -> `1-cos` worst **1.337e-05**, mean 8.083e-06.
   Same band as short texts.

Then the decisive one — run BOTH harness sides on **real STS12 sentences** and compare the
matrices they actually produce: **`1-cos` worst 5.369e-05, mean 8.448e-06, zero rows over
1e-3.** The two sides produce the same embeddings. The delta was not in the embeddings at
all.

**The cause.** `mteb` selects a prompt from the model's own prompts table by task type and
passes it to a SentenceTransformer as `prompt=`. Captured directly: for STS12 the CPU side
receives `prompt='task: sentence similarity | query: '`, and sentence-transformers prepends
it **on top of** the `'title: none | text: '` that `run_mteb.py` had already prepended. So:

```
CPU:  "title: none | text: task: sentence similarity | query: <text>"
NPU:  "title: none | text: <text>"
```

Two different strings. **The measured M8 delta was the PROMPT, not the datapath** — exactly
the failure `run_mteb.py`'s own comment says the container-read prefix exists to prevent
("so both sides agree by construction rather than by two literals that can drift"), defeated
because mteb *also* injects one, for models it recognises as SentenceTransformers and not for
ours.

**The tell was in the table the whole time.** STS and clustering — both pure cosine geometry
— lost 2.4-3.5 points. Banking77, a logistic regression over the embeddings, gained 0.07: a
linear classifier absorbs a constant shift in the space, so the one task whose metric is
*insensitive to a global prompt change* is the one that showed no delta. A datapath
regression has no reason to sort itself that way.

**The fix makes the prompt symmetric rather than suppressing it.** EmbeddingGemma ships 14
prompts named after MTEB task types precisely so they get used, so the right answer is that
BOTH sides use the task-appropriate one:

* `NpuEncoder.encode()` now performs mteb's own lookup order (task name -> task type +
  prompt type -> task type -> prompt type) against the **container's copy of the same
  table**, and re-verifies the runtime's prefix echo when it switches.
* `run_mteb.py` no longer prepends the container prefix when mteb has supplied a `prompt` /
  `prompt_name`.

Verified before re-running: STS12/STSBenchmark/SICK-R -> `'STS'`, Banking77 ->
`'Classification'`, TwentyNewsgroups -> `'Clustering'` on the NPU side, matching what mteb
hands the CPU side.

**What this costs in credibility, stated plainly.** The four BERT models and nomic have MTEB
numbers taken through the *old* code path. For them the question is whether their containers
carry a prompts table that mteb could match: the four BERT ones carry none at all, so nothing
changes. **nomic does** (`search_document` etc., written by this project rather than by the
checkpoint) — and whether mteb matched any of those keys to a task type decides whether
nomic's recorded +0.09 was measured symmetrically. That has not been checked and it is now
[T36](../../research/OPEN-THREADS.md).

## 6. Problems hit

- **A background MTEB run piped through `tail` looked hung for half an hour.**
  The pipe buffers until the process exits, so the log stayed empty at 0 bytes
  while the run was in fact working — confirmed by sampling per-process CPU
  *delta*, which showed ~12 cores busy. `release_benchmark.ps1` already learned
  this lesson the other way round (it adds `-u` to every Python it launches,
  with a comment saying progress you cannot see is indistinguishable from a
  hang); the same applies to how the output is *consumed*, not just produced.
- **The prefix echo check failed against a correct runtime** — see §2.

---

## 7. Not done

*(pending)*
