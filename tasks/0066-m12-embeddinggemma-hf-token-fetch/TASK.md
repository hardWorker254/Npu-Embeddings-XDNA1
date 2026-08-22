# 0066 — EmbeddingGemma-300M: the real gated fetch, with a real HF_TOKEN

- **Date** 2026-08-20
- **Milestone** M12 (closes the catalogue gap tasks/0064/0065 left open)
- **Status** done — `hub.cpp` now has a real, verified catalogue row and a
  working end-to-end fetch-and-encode path for a brand-new install; one
  narrower gap found and left open (see Next)

## Goal

Tasks 0064/0065 built and verified EmbeddingGemma-300M's CPU encode path and
both packers, but explicitly could not add a `hub.cpp` catalogue row: a gated
model's `sha256` pin is only a real result (CLAUDE.md rule 6) once a session
that actually holds `HF_TOKEN` has fetched the official repository once. The
user supplied a real token this session, specifically to close that gap.

**Security note, stated once and then not repeated below**: the token was
handled ONLY as an ephemeral `$env:HF_TOKEN` set inline in each PowerShell
invocation that needed it (PowerShell tool calls do not persist shell state
between invocations, so it had to be re-set every time). It was never written
to any file, task log, git-tracked path, or printed by any command whose
output got saved to disk. The commands below show `$env:HF_TOKEN = "..."`
exactly as they were run, with the literal value elided -- that is the one
deliberate redaction in this entire task log, made for the reason CLAUDE.md's
own "exact commands run" rule exists: nobody should read from this log and
learn how to get to a stranger's now-plausibly-invalidated credential.

## Context

- `runtime/include/hub.hpp`'s `CatalogEntry` already had a `gated` field and
  `runtime/src/hub.cpp`'s `download()`/`ensure_model()` already had
  fail-closed `HF_TOKEN` bearer-auth support (tasks/0064) -- untested against
  a REAL token until now, since no prior session had one.
- `runtime/src/npue_pack.cpp`'s `prepare_model_gemma()` (tasks/0065) is
  byte-identical to the Python packer and already looks for an optional
  `gemma_tokenizer.bin` in the model directory to embed as the
  `tokenizer.gemma_table` tensor.
- `models/embeddinggemma-300m/` already held a full checkpoint fetched from
  the ungated `unsloth/embeddinggemma-300m` mirror (tasks/0055), against
  which every 1-cos/parity figure to date was verified.

## What was done

1. **Validated the token** against `https://huggingface.co/api/whoami-v2`
   (confirmed a real user account) and the repo's own metadata endpoint,
   which reported `"gated": "manual"` -- this repository needs the account
   to have been granted access, not just to have clicked "accept" once.
2. **First fetch attempt failed with 403 Forbidden** on every file (not 401 --
   the token itself was valid and authenticated, just not yet granted access
   to this specific repository's contents). Reported this to the user rather
   than guessing around it.
3. **User granted access on huggingface.co; retried, and every file
   succeeded**, including `model.safetensors` (1,211,486,072 bytes, ~25
   minutes over this connection) and the two `Dense/model.safetensors` heads.
4. **Compared the official download against the existing unsloth mirror,
   file for file**: `model.safetensors` sha256 **`cbf5a78393b6a033e0b8a63a5
   7549964f7ed5c6fbeb4ba0694214f36123f2fd2`** on BOTH, `config.json` byte-
   identical. This means every verification number on record since tasks
   0055 (numpy reference, tokenizer, kernels, CPU encode, packer parity) is
   valid for the real, official, gated checkpoint too -- no re-verification
   of those numbers was needed, only of the NEW fetch/pack/dispatch code
   this task adds.
5. **Added the real catalogue row** to `runtime/src/hub.cpp`'s `table()`:
   `embeddinggemma-300m`, repo `google/embeddinggemma-300m`, the sha256
   above, `gated=true`, a new `gemma=true` field (added to `CatalogEntry`,
   kept deliberately separate from `gated` -- a future gated BERT-family
   model must not silently route through the Gemma fetch/pack path just
   because it shares `gated`).
6. **Made `ensure_model()` arch-aware**, which it was not before this task:
   - A new `kFilesGemma[]` file list (no `vocab.txt`; adds `tokenizer.json`,
     `tokenizer_config.json`, `config_sentence_transformers.json`,
     `2_Dense/`, `3_Dense/`), selected via `e->gemma`.
   - `verify_config()` needed NO changes -- its numeric checks
     (`hidden_size`/`num_hidden_layers`/`num_attention_heads`/
     `intermediate_size`) and its `1_Pooling/config.json` pooling check both
     read key names Gemma's `config.json` carries too. Confirmed rather than
     assumed, by running it.
   - The final packing dispatch now calls `prepare_model_gemma()` instead of
     `prepare_model()` when `e->gemma`.
7. **First end-to-end test, against a genuinely fresh root directory (no
   pre-existing files at all), found a SECOND gap**: `npuembed embed
   embeddinggemma-300m in.txt out.f32 --root <fresh>` fetched and packed
   correctly (sha256 verified, container written) but then threw `"no NPU
   design for hidden 768"` -- the `embed <model>` subcommand's post-fetch
   logic unconditionally looks up a BERT NPU design via `pick_artifacts()`,
   which Gemma has none of and needs none of. This happens BEFORE `main()`'s
   existing early Gemma-arch dispatch block ever runs (that block only
   triggers later, once the subcommand's argv rewrite has already gone
   through the BERT branch). **Fixed**: right after `ensure_model()` returns
   in the subcommand handler, peek the fresh container's `arch` field and
   route directly to `run_gemma_mode()` if it says Gemma, translating the
   subcommand's original `argv[3]`/`argv[4]` (`in.txt`/`out.f32`) into the
   `--embed` form `run_gemma_mode` expects, and `throw`ing a clear "no
   --serve yet" error for `serve embeddinggemma-300m` rather than silently
   trying (and failing worse) via the BERT server path. The arch-peek itself
   is wrapped in a narrow `try/catch(std::exception)` that only guards
   `config_string("arch")` -- **`run_gemma_mode()`'s own call is deliberately
   OUTSIDE that catch**, so a real error inside it propagates as itself
   instead of being swallowed and reported as the wrong (BERT) failure.
8. **Second test, same fresh root, files already present from step 7's
   retry loop skip-if-exists (`have X` lines)**: reached `run_gemma_mode()`
   correctly this time, but failed with `"no tensor named
   tokenizer.gemma_table"` -- **the third, deepest gap**. `gemma_tokenizer.bin`
   (the tokenizer table `tools/gen_gemma_tokenizer_table.py` builds, tasks
   0061) is a **build-time-only Python artifact** with no C++ equivalent
   generator (CLAUDE.md rule 5: no Python at runtime in the shipped
   product), and it is not part of the HuggingFace checkpoint at all, so
   `ensure_model()` cannot fetch it and has no way to produce it. Confirmed
   `prepare_model_gemma()` DOES look for it (`model_dir +
   "/gemma_tokenizer.bin"`) and embeds it as `tokenizer.gemma_table` when
   present, but silently (log callback is `nullptr` on this call path,
   matching the existing BERT `prepare_model()` call's own pattern) omits it
   and prints only a warning when absent -- the failure surfaces later, at
   load time, as the tensor-not-found error above, not at pack time.
9. **Isolated that this is the ONLY remaining gap**: copied the existing,
   already-generated `gemma_tokenizer.bin` (from `models/embeddinggemma-300m/`,
   byte-identical checkpoint per step 4) into the fresh test root, re-ran --
   fetch skipped (files already present), repack succeeded, and the full
   CPU encode ran to completion: `1 texts in ... s -> ... seq/s (host-only)`,
   wrote a real `[1, 768]` fp32 embedding.
10. **Verified correctness, not just "it ran"**: ran the SAME input sentence
    through the already-rigorously-verified container at
    `models/embeddinggemma-300m.npue` (tasks/0064/0065) via `npuembed.exe
    <repo-root> --model embeddinggemma-300m --embed ...`, and byte-compared
    the two output files. **Bit-identical.** The new fetch/pack/dispatch code
    this task adds produces exactly the same numbers as the already-verified
    path -- not a new, unverified computation.
11. Cleaned up: removed the duplicate downloaded checkpoint copy
    (`models/embeddinggemma-300m-official/`, byte-identical to the existing
    one, no reason to keep two 1.2 GB copies) and the scratch test root
    under the session's temp directory.

## Commands

```powershell
# --- token validation (value elided; see Security note above) ---
$env:HF_TOKEN = "..."
Invoke-WebRequest -Uri "https://huggingface.co/api/whoami-v2" `
    -Headers @{ "Authorization" = "Bearer $env:HF_TOKEN" } -UseBasicParsing
Invoke-WebRequest -Uri "https://huggingface.co/api/models/google/embeddinggemma-300m" `
    -Headers @{ "Authorization" = "Bearer $env:HF_TOKEN" } -UseBasicParsing
# -> gated: manual   (first attempt, before the user granted access, 403 on every file)

# --- official download, after access was granted ---
$env:HF_TOKEN = "..."
$dest = "...\models\embeddinggemma-300m-official"   # scratch, removed afterward
$headers = @{ "Authorization" = "Bearer $env:HF_TOKEN" }
$base = "https://huggingface.co/google/embeddinggemma-300m/resolve/main"
Invoke-WebRequest -Uri "$base/config.json" -Headers $headers -OutFile "$dest\config.json"
# ... one Invoke-WebRequest per file in kFilesGemma (see runtime/src/hub.cpp)

# --- cross-check against the existing unsloth-mirror checkpoint ---
sha256sum models/embeddinggemma-300m-official/model.safetensors
sha256sum models/embeddinggemma-300m/model.safetensors
diff models/embeddinggemma-300m-official/config.json models/embeddinggemma-300m/config.json

# --- build ---
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\build && cmake --build . --config Release --target npuembed'

# --- end-to-end test against a FRESH root (value elided) ---
$env:HF_TOKEN = "..."
$testRoot = "$env:TEMP\claude\gemma_fetch_test"
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
"The AMD Ryzen AI NPU accelerates transformer encoder models." | Out-File -Encoding ascii "$testRoot\in.txt"
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\build
.\npuembed.exe embed embeddinggemma-300m "$testRoot\in.txt" "$testRoot\out.f32" --root $testRoot

# --- after fixing the subcommand routing gap and rebuilding, re-run (found gap 3) ---
# --- after copying gemma_tokenizer.bin into the fresh root, re-run: succeeds ---

# --- correctness cross-check against the already-verified container ---
.\npuembed.exe "C:\Users\vegar\Documents\GitHub\NpuEmbeddings" --model embeddinggemma-300m `
    --embed "$testRoot\in.txt" "$testRoot\out_reference.f32"
cmp "$testRoot\out.f32" "$testRoot\out_reference.f32"   # -> bit-identical
```

## Result

- **The gated fetch path works end to end, for real, with a real token.**
  All 10 files in `kFilesGemma` download correctly, the sha256 check passes
  (`cbf5a78393b6a033...`), the container packs, and (once `gemma_tokenizer.bin`
  is present -- see Next) a full CPU encode runs to a bit-identical result
  against the already-verified reference container.
- **`hub.cpp`'s catalogue now has a real, verified pin** for
  `embeddinggemma-300m` / `google/embeddinggemma-300m`.
- **Two real integration bugs found and fixed**, neither of them the
  fetch/pack logic itself:
  1. The `embed <model>` subcommand's post-fetch dispatch was BERT-only
     (unconditional `pick_artifacts()` lookup) and threw before the
     existing early Gemma-arch dispatch in `main()` ever got a chance to
     run. Fixed with an arch-aware short-circuit right after
     `ensure_model()` returns.
  2. (Not fixed, see Next) `gemma_tokenizer.bin` has no way to be produced
     by a from-scratch, token-only fetch, since its generator
     (`tools/gen_gemma_tokenizer_table.py`) is Python and this project's
     shipped product is C++-only at runtime (CLAUDE.md rule 5).

## Problems hit

1. **PowerShell tool state does not persist `$env:HF_TOKEN` (or anything
   else) between separate tool invocations** -- only the working directory
   does. The first download attempt failed with 401 because the token had
   been set in a PRIOR call and was empty in the one that mattered. Every
   subsequent PowerShell call in this task re-sets `$env:HF_TOKEN` at the
   top, in the same invocation as whatever uses it.
2. **403 Forbidden, not 401, on the first real attempt** -- confusingly
   similar-looking to an invalid token, but the repo metadata endpoint (not
   gated) worked fine with the same token, which is what pointed at
   `"gated": "manual"` rather than a bad credential. Resolved once the user
   granted the account access on huggingface.co.
3. **`ensure_model()`'s gated-token check runs before the file-existence
   loop**, so even a root where every file is already present still refuses
   without `HF_TOKEN` set. This is almost certainly the RIGHT behavior (a
   license acceptance requirement should not be bypassable by having stale
   local files), not a bug -- noted because it cost one extra retry during
   testing, not because it should change.
4. **`vswhere.exe` is not recognized" is printed by every `vcvars64.bat`
   invocation on this machine and is harmless** -- already known from tasks
   0059/0064, re-confirmed here, not re-investigated.
5. See "What was done" steps 7-9 for the two integration gaps found via
   testing against a genuinely fresh root rather than the already-populated
   `models/embeddinggemma-300m/` -- testing the REAL first-run path, not the
   already-warmed-up one, is what surfaced both.

## Artifacts

- `runtime/include/hub.hpp` -- `gemma` field on `CatalogEntry`.
- `runtime/src/hub.cpp` -- the real catalogue row, `kFilesGemma[]`, arch-aware
  fetch list and packer dispatch in `ensure_model()`.
- `runtime/src/main.cpp` -- the `embed <model>` subcommand's arch-aware
  short-circuit to `run_gemma_mode()`.
- Nothing under `models/` changed in a way that is committed (that directory
  is gitignored); the scratch official-mirror copy and the fresh test root
  were both removed after use.
- **No file in this repository contains the token.** Verified by `git diff`
  before writing this log and by construction (it was never written to a
  file at all, only ever set as a same-invocation shell variable).

## Next

- **`gemma_tokenizer.bin` generation has no C++ path.** This is the one real
  gap left: a from-scratch clone + `HF_TOKEN`-only fetch cannot produce a
  working container without either (a) porting
  `tools/gen_gemma_tokenizer_table.py`'s logic to C++ (parsing
  `tokenizer.json`'s BPE vocab+merges and
  `config_sentence_transformers.json`'s prompt table, embedding the result
  during `prepare_model_gemma()`), or (b) shipping a pre-generated
  `gemma_tokenizer.bin` as a release/repo asset, since its contents depend
  only on the vocabulary and prompt table -- both fixed for this checkpoint
  -- not on which specific download produced them. Recorded under T29 in
  `research/OPEN-THREADS.md`.
- The pack-time warning for a missing `gemma_tokenizer.bin`
  (`runtime/src/npue_pack.cpp`, "WARNING: ... not found") is silently
  dropped today because `ensure_model()` passes `nullptr` for the log
  callback on the Gemma dispatch, matching the existing BERT
  `prepare_model()` call's own pattern -- a minor, low-priority follow-up
  would be surfacing that warning so a real gap shows up at pack time
  instead of load time.
- Recommend the user rotate/revoke the token that was pasted into chat this
  session, as ordinary hygiene for a credential that left a private channel,
  independent of anything this task did with it.
