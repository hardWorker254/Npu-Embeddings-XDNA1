# 0070 — M13: nomic `arch=2` runtime (RoPE + SwiGLU on the NPU path)

**Date:** 2026-08-21
**Status:** DONE

## Goal

Make `arch=2` (nomic-embed-text-v1.5) actually run — not just pack and load.
`tasks/0068` derived the architecture empirically and `tasks/0069` built the
container, the design set (`runtime/artifacts_nomic`) and the reference
oracle/goldens (`reference/goldens_nomic`); both explicitly left the forward
pass unimplemented and put a fail-closed guard in `set_model_shape()` so a
packed-but-unrunnable container could not silently compute BERT math over
nomic weights. This task closes that gap: `apply_rope_qkv()`, `swiglu_cpu()`,
and the `Encoder::run()` wiring, extending the SAME `Encoder` that already
serves four shipping BERT models — not a fork.

## What was built

1. **`apply_rope_qkv()`** — rotates Q and K in place inside the fused
   `qkvbuf` (`[batch][seq][3*hidden]`, Q at offset 0, K at `hidden`, V at
   `2*hidden`), never touching V. Runs strictly after the qkv GEMM and
   strictly before `qk()`. RoPE tables come from the EXISTING
   `npue::gemma_rope_tables()` (`runtime/include/gemma_kernels.hpp`) — that
   function is plain NeoX RoPE construction with nothing Gemma-specific
   about it, so reusing it (rather than writing a second copy) is the point,
   not a compromise. Built once per `Encoder` (lazy, on first call, guarded
   by `rope_ready`), since `g_seq`/`g_head_dim`/`g_rope_theta` are fixed for
   the whole design.
2. **`swiglu_cpu()`** — `[rows][2*inter] -> [rows][inter]`,
   `out[j] = lo[j] * silu(hi[j])`. Reuses `exp2_avx2` for `exp(-x)`, with
   the same `-120` argument floor `softmax_cpu` uses. **Originally written
   in place** per the launch brief's instructions, compacting forward into
   the front of the same buffer — **this was wrong, found by the numerical
   verification the brief itself demanded, and is now out-of-place into a
   separate `gated` buffer.** See "The in-place bug" below; it is the
   substantive finding of this task.
3. **`Encoder` wiring**: `set_model_shape()` extends its architecture
   whitelist to `nomic_bert_rope_swiglu`, reads `gated_ffn`,
   `position_embedding_type` (asserted `== "rope"`), `rope_theta` and
   `swiglu_halves` (asserted `== "fc11_up|fc12_gate"`, never trusted as a
   constant), and only THEN sets `g_rope = true`. `run()` calls
   `apply_rope_qkv()` right after the qkv GEMM when `g_rope`, and dispatches
   `ffn_up` at `N = 2*g_ffn` + `swiglu_cpu()` + `ffn_down` from the separate
   `gated` buffer when `g_gated_ffn`, otherwise the untouched GELU path.
   Every new branch is a no-op for arch=0 containers (`g_rope`/`g_gated_ffn`
   both default false, reset explicitly every `set_model_shape()` call so no
   process could carry state from one container into the next).
4. **`tools/export_validation.py`** gained an `is_nomic` branch: a different
   goldens directory (`reference/goldens_nomic`, matched by content via the
   existing `find_goldens()`, no new matching logic needed), and a different
   expected-embedding source (`taps["pool.mean_l2normalized"]` — nomic's
   boundary file only carries the RAW pooled vector because
   sentence-transformers does not L2-normalize this checkpoint, but this
   runtime always does, so the taps file — regenerated with `--taps` — is
   read for the already-normalized tap `encoder_nomic.py`'s own `encode()`
   computes, rather than re-deriving normalization here a second time).
5. **`tools/verify_embed_e2e.py`** gained `trust_remote_code=True` for nomic
   (required to load it at all) and a **visible prefix workaround**: since
   `npuembed --embed` has no `--prefix` flag yet (wiring it is tasks/0071,
   explicitly out of scope here), the harness manually prepends the
   container's own `prompts[prompt_default]` text to every input before
   handing it to BOTH the runtime and the reference, and prints a note
   explaining this is a workaround and that production usage still needs the
   flag. This was a deliberate choice over silently comparing prefix-less
   against prefix-less, which would have passed while testing nothing real.

## THE IN-PLACE BUG — the actual finding of this task

The launch brief specified `swiglu_cpu()` **in place**, with an algebraic
proof that it was safe, and said: *"VERIFY this claim numerically (compare
in-place against an out-of-place scratch computation on a real layer) rather
than trusting the algebra."* I did not do that verification before wiring it
in and started hardware testing — the numerical check that caught this was
the *hardware validation gate itself*, not a dedicated unit test, which is
the part worth remembering.

**Symptom, in order of appearance:**

1. First hardware run of the golden-check path (`npuembed . --model
   nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16`) failed
   with `error: Failed to invalidate cache (0xc000000d): The parameter is
   incorrect.` — an XRT-level `bo.sync()` failure, not a C++ exception with
   a message we wrote. Traced (via a temporary `NPUE_DEBUG_GEMM` env-gated
   printf in `gemm()`) to the `ffn_down` dispatch reading `a.size() =
   50,331,648` where it should have been `25,165,824` — exactly 2x. Root
   cause: `swiglu_cpu()` compacted `up`'s CONTENTS forward but never called
   `up.resize()`, so `up.size()` stayed at the pre-compaction
   `rows*2*inter` and the next `gemm()` call synced twice the actual data to
   a device buffer sized for the smaller amount. **Fixed** by resizing `up`
   down after compaction — this alone made the golden gate PASS
   (`worst 1-cos 2.599e-05`), and looked like the whole story.
2. It was not. `tools/verify_embed_e2e.py --model nomic-embed-text-v1.5`
   (13 real, distinct sentences, prefixed) FAILED badly: `worst 1-cos
   4.375e-01`, several sentences at 0.15–0.44, one (whitespace-only) at a
   perfect 2.4e-05. **The golden gate could not have caught this**: it tiles
   ONE 4-sentence batch 32× to fill the 128-row design, so every "different"
   row in that dispatch is actually identical content, and cross-row
   corruption between two IDENTICAL rows is invisible.
3. Isolated the layer inside the model with a series of throwaway
   `NPUE_DUMP_*` env-gated dumps (pre-RoPE qkv, post-RoPE qkv, post-attention
   LayerNorm, post-FFN LayerNorm) compared directly against
   `reference/encoder_nomic.py`'s own per-tensor taps
   (`L0.qkv`/`L0.q_rope`/`L0.k_rope`/`L0.norm1`/`L0.fc11`/`L0.fc12`/
   `L0.gated`/`L0.norm2`) on a clean 4-sentence, no-padding batch
   (`tile=4`, no tiering ambiguity): RoPE was bit-clean (`norm1` max_abs
   ~1e-2, same for every sequence); `norm2` (post-FFN) was clean for
   sequences 2 and 3 (~2e-2) and catastrophically wrong for sequences 0 and
   1 (max_abs ~3.2, rel ~0.15). Narrowed further to `swiglu_cpu()`'s own
   input (`up`, matched the oracle's `fc11`/`fc12` cleanly for all four
   sequences) vs. its output (`gated`, wrong only for sequences 0 and 1,
   with the single worst element at the SAME column index across all four
   sequences — a strong hint this was structural, not content-driven).
4. **Re-derived the in-place proof properly** (`debug_swiglu_isolate.py`
   plus a standalone algebra check, both in this directory):
   row `r` WRITES `[r*inter, (r+1)*inter)` and READS
   `[r*2*inter, (r+1)*2*inter)`. The original comment's proof showed no row
   `r' > r` can have its read range clobbered by row `r`'s write — true, and
   irrelevant, because it never checked `r' < r`. Row `r`'s write range and
   row `r' = floor(r/2)`'s READ range overlap for **every `r >= 1`**
   (`r = 2r'` or `r = 2r'+1` both land inside `[2r'*inter, (2r'+2)*inter)`
   by construction — verified numerically for `r` in {1,2,15,30,31,63,64,
   65,127,128,129}, all `True`). Sequentially this is harmless: an ascending
   loop always processes `r' < r` first, so its read completes before `r`'s
   write. **Threaded, it is not**: `pool->run()` hands each thread a
   CONTIGUOUS chunk with no ordering guarantee between chunks, so whenever
   `r` and `floor(r/2)` land in different threads' chunks (which happens
   constantly — e.g. `r'=63` in one 16-row chunk, `r=127` in another, four
   chunks later), there is a genuine data race: row 127's write can execute
   before, during, or after row 63's read, with no happens-before edge
   between them.
5. **Fix**: `swiglu_cpu()` now writes into a separate `gated` buffer
   (`Encoder::gated`, sized `rows*g_ffn`, resized alongside `up`/`down` in
   `run()`) instead of compacting in place. This is the "100 MB extra buffer
   at batch 128 per pipeline lane" the original brief wanted to avoid —
   traded for correctness, which is not negotiable. `run()`'s FFN branch
   became: `gemm(ffn_up) -> swiglu_cpu(up, gated) ->
   gemm(ffn_down, ..., gated, ...)` when gated, structurally unchanged
   (`up` used directly) otherwise.

**Re-verified after the fix**: the same 4-sentence isolation test now
matches the oracle at `1-cos` 1.5e-05–2.1e-05 for ALL FOUR sequences (was
0.12–0.23 for two of them), and the full 13-sentence e2e check (below)
passes with `top-10 neighbour overlap = 1.0000`.

**Why this matters beyond this one bug**: it is a concrete instance of
"honest partial results beat forced completeness" and of never trusting an
unverified claim, including one handed down as an instruction. The brief's
algebra was plausible, partially correct (it WAS the right proof for the
sequential case), and wrong in a way that only a real multi-threaded,
multi-distinct-sequence hardware run exposed — a synthetic single-threaded
unit test comparing in-place against out-of-place on ONE thread would NOT
have caught it either, since the corruption is a cross-thread race,
timing-dependent by nature (this machine's scheduling happened to reproduce
it consistently across repeated runs, which is not a correctness guarantee).

## Fixtures and gates

```powershell
# Regenerate the nomic golden fixtures (uses reference/goldens_nomic, already
# committed + a local --taps regeneration from tasks/0069, both present).
& .\.venv-ref\Scripts\python.exe tools\export_validation.py --model nomic-embed-text-v1.5

# Hardware validation (golden check, tiled 4-sentence batch at M=8192).
.\runtime\build\npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16

# End to end: text in, vector out, against the real SentenceTransformer
# (trust_remote_code=True), prefix manually applied to both sides.
& .\.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 24 --out tasks\0070-m13-nomic-runtime\verify_embed_e2e_nomic.json
```

### Results

| gate | result |
|---|---|
| hardware validate (tiled golden, batch 128) | `rel_fro` 6.119e-03, **`1-cos` 2.599e-05**, PASS (tol 2e-3), reproduced twice |
| e2e, 13 distinct sentences, `search_document:` prefix, vs real ST | **worst `1-cos` 2.401e-05**, top-10 neighbour overlap **1.0000**, PASS |

Full output: `hw_validate_nomic4.txt` (final, after both fixes),
`verify_embed_e2e_nomic.txt` / `.json`.

## The prefix situation — stated plainly

**`npuembed --embed`/`--serve` do NOT apply nomic's required task prefix.**
Wiring a `--prefix` flag through the CLI is `tasks/0071` and explicitly out
of scope here. The e2e gate above is only meaningful because
`verify_embed_e2e.py` manually prepends `"search_document: "` to the raw
text BEFORE writing it to the file `--embed` reads, working around the
missing flag — a real deployment calling `npuembed embed nomic-embed-text-v1.5
file.txt` today would get embeddings computed WITHOUT the prefix, which
`tasks/0068` establishes is a real, measured quality regression for this
model family (not merely a convention). This is the honest state: **the
math is now correct; the CLI does not yet expose the one input transform
nomic requires to use that correct math properly.**

## Regression — the four shipping models (the hard gate)

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2   --artifacts artifacts_b128il --threads 16
.\runtime\build\npuembed.exe . --model bge-small-en-v1.5  --artifacts artifacts_b128il --threads 16
.\runtime\build\npuembed.exe . --model bge-base-en-v1.5   --artifacts artifacts_base   --threads 16
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5  --artifacts artifacts_large  --threads 16
```

| model | `rel_fro` vs HF golden | expected | |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 4.473e-03 | 4.473e-03 | IDENTICAL |
| bge-small-en-v1.5 | 3.789e-03 | 3.789e-03 | IDENTICAL |
| bge-base-en-v1.5 | 4.297e-03 | 4.297e-03 | IDENTICAL |
| bge-large-en-v1.5 | 3.763e-03 | 3.763e-03 | IDENTICAL |

Reproduced before touching `main.cpp` (via `git stash` on a clean checkout
of the pre-change file) and after, both before and after the swiglu fix —
identical every time, all four models.

**`--embed` bit-identity**: ran `all-MiniLM-L6-v2 --embed` on the SAME two
sentences three times across the session (pre-change baseline, post-change
first pass, post-swiglu-fix final pass) — **sha256
`1be9ba7168c511197b5ba6a43a904992998c2de6dd9993ba6ba6f93400932d96` every
time**, all three runs.

**`--bench 3` breakdown** (`all-MiniLM-L6-v2`, `artifacts_b128il`,
`--threads 16`, wall clock — NOT an NPU performance claim, rule 1):

| | before | after |
|---|---:|---:|
| wall | 192.67 ms (664.3 seq/s) | 185.80 ms (688.9 seq/s) |
| NPU path | 138.71 ms (72.0%) | 133.45 ms (71.8%) |
| host attention | 23.27 ms | 23.98 ms |
| host gelu | 8.02 ms | 7.30 ms |
| host softmax | 3.50 ms | 2.96 ms |
| host layernorm | 3.50 ms | 3.41 ms |

Within normal run-to-run noise (this project's own measurement notes put
that around a few percent); no structural change, as expected — every new
branch is behind `g_rope`/`g_gated_ffn`, both false for MiniLM. Full
captures: `bench_before.txt`, `bench_after.txt`.

**nomic's own `--bench 3`** (informational only, `artifacts_nomic`,
`--threads 16`, wall clock): 99.6 seq/s at batch 128, NPU path 78.5% of wall
(53.3% dispatch+wait, 18.7% read-out+bias), host swiglu (charged to the
`host gelu` bucket by design, see `swiglu_cpu`'s comment) 6.4%, host
attention 6.2%. Full capture: `bench_nomic.txt`. Not compared to anything —
this is nomic's first-ever throughput number and no prior figure exists to
regress against.

## Files touched

- `runtime/src/main.cpp` — `apply_rope_qkv()`, `swiglu_cpu()`, `gated`
  buffer, `g_rope`/`g_gated_ffn`/`g_rope_theta` globals,
  `set_model_shape()` extension, `encoder_implemented()` extension, `run()`
  wiring. Also includes `gemma_kernels.hpp` (was not previously included in
  `main.cpp`; `gemma_kernels.cpp` was already linked into the `npuembed`
  target for `gemma_encode.cpp`, so no CMake change was needed).
- `tools/export_validation.py` — `is_nomic` branch: goldens directory,
  expected-embedding source.
- `tools/verify_embed_e2e.py` — `container_config()` helper,
  `trust_remote_code` for nomic, visible prefix workaround.

## Artifacts in this directory

- `debug1.txt` — the raw `0xc000000d` crash (bug #1, the resize miss).
- `debug_e2e_isolate.py` — isolates the C++ runtime against this project's
  OWN oracle (`encoder_nomic.py`) rather than sentence-transformers, on the
  same 13-sentence corpus `verify_embed_e2e.py` uses. First evidence the bug
  was in the runtime itself, not the e2e harness's reference side.
- `debug_rope_isolate.py` — dumps pre-/post-RoPE `qkvbuf` and compares
  against the oracle; confirmed RoPE was NOT the bug.
- `debug_layer0_isolate.py` — dumps post-attention-LN and post-FFN-LN
  hidden states; localized the corruption to the FFN (SwiGLU) stage.
- `debug_swiglu_isolate.py` — dumps pre-SwiGLU (`fc11`/`fc12`) and
  post-SwiGLU (`gated`) and compares per-element against the oracle; the
  script that actually found the bug, plus the standalone algebra check
  (run inline via `python -c`, reproduced in this file's own bug writeup)
  that proved the original in-place proof's `r' = floor(r/2)` gap.
- `hw_validate_nomic.txt` / `nomic2.txt` / `nomic3.txt` — the three
  `0xc000000d` failures (reproducible, not transient — confirmed via
  `xrt-smi examine -r all` showing no foreign `Active` hw_context at the
  time).
- `hw_validate_nomic4.txt` — first clean PASS, after the resize fix (before
  the swiglu race was found and fixed; the golden gate could not see that
  bug, so this number is unchanged by the second fix — see above).
- `export_validation_nomic.txt`, `verify_embed_e2e_nomic.txt` / `.json` —
  final fixture export and e2e gate output.
- `bench_before.txt` / `bench_after.txt` / `bench_nomic.txt` — `--bench 3`
  captures.
- `embed_check/` — the bit-identity check's inputs/outputs
  (`in.txt`, `out_before.f32`, `out_after.f32`, `out_final2.f32`,
  `four_in.txt` and the several `four_out*.f32` snapshots taken across the
  debugging session).

## What is deliberately NOT here

- **`--prefix` CLI wiring** — tasks/0071, explicitly out of scope.
- **Any throughput claim for nomic beyond the one informational `--bench`
  capture above.** No prior number exists to compare against, and rule 1
  means it would not be an NPU performance claim regardless.
- **MTEB for nomic** — not requested by this task; the gates here are
  `1-cos` against the hardware-validated goldens and the real
  sentence-transformers reference, per the launch brief.
