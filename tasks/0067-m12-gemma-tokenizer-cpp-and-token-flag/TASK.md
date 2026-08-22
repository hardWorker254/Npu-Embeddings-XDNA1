# 0067 — EmbeddingGemma tokenizer table in C++, and a `--token` CLI flag

- **Date** 2026-08-21
- **Milestone** M12 (closes the last gap tasks/0066 left open; T29)
- **Status** done — both deliverables built, wired into production, verified

## Goal

Two independent, self-contained C++ features:

**A.** Port `tools/gen_gemma_tokenizer_table.py` to C++ so `prepare_model_gemma()`
can produce `gemma_tokenizer.bin` itself instead of only reading an
already-generated one. Task 0066 found this was the one real gap left in the
EmbeddingGemma fetch path: a from-scratch clone that fetches the checkpoint via
`HF_TOKEN` gets a `.npue` with no `tokenizer.gemma_table` tensor, and the first
real encode throws `"no tensor named tokenizer.gemma_table"`.

**B.** Add a `--token <value>` CLI flag as the primary way to authenticate a
gated HuggingFace fetch, with `HF_TOKEN` as a fallback, so the user is not
forced to use an environment variable.

## Context

- `runtime/src/npue_pack.cpp`'s `prepare_model_gemma()` (tasks/0065) only ever
  read `<model_dir>/gemma_tokenizer.bin` from disk; nothing in the C++ build
  could generate one.
- `tools/gen_gemma_tokenizer_table.py` (tasks/0061) is the only existing
  generator, and its docstring is the authoritative record of the tokenizer
  pipeline: SentencePiece **BPE**, confirmed empirically against the real
  checkpoint, not Unigram.
- This project has zero third-party C++ dependencies (`runtime/CMakeLists.txt`:
  only XRT/ws2_32/winhttp) and no existing JSON parser capable of walking a
  33 MB, deeply-nested `tokenizer.json` (the existing `cfg_int`/`cfg_str`/
  `cfg_raw` scanners in `npue_pack.cpp` only handle flat single-object
  configs).
- `runtime/src/hub.cpp`'s `ensure_model()` already had fail-closed `HF_TOKEN`
  bearer-auth support (tasks/0064/0066) but no `--token` flag — env var only.

## What was done

### Deliverable A

1. Read `tools/gen_gemma_tokenizer_table.py` in full and confirmed the exact
   field types in the real checkpoint files (via a throwaway `python -c`
   probe against `models/embeddinggemma-300m/tokenizer.json`, conda `iron`
   env) rather than assuming: `model.type == "BPE"`, `dropout: null`,
   `continuing_subword_prefix`/`end_of_word_suffix`: `null`, `byte_fallback:
   true`, merges stored as 2-element `[a, b]` string-pair arrays (not a single
   `"a b"` string), `normalizer` = `{"type":"Replace","pattern":{"String":"
   "},"content":"▁"}`, `post_processor.single` a list of `{"SpecialToken":
   {"id": "<bos>"|"<eos>", ...}}` / `{"Sequence": ...}` steps.
2. Wrote a dependency-free JSON DOM parser from scratch:
   `runtime/include/json_min.hpp` + `runtime/src/json_min.cpp`. Recursive
   descent, `Value` as a tagged union (`Null`/`Bool`/`Number`/`String`/
   `Array`/`Object`), objects as `std::unordered_map<std::string, Value>`
   (`model.vocab` is 262,144 entries, looked up by key, never enumerated in
   file order), arrays as `std::vector<Value>` (order-preserving — required
   for `model.merges`' rank order). Full escape support (`\" \\ \/ \b \f \n
   \r \t \uXXXX` including UTF-16 surrogate pairs → UTF-8). The string
   scanner fast-paths the common run-of-plain-bytes case with one
   `append(ptr, len)` rather than per-character appends, since the file has
   ~262k vocab strings and ~515k merge-pair strings.
3. Wrote `runtime/include/gemma_tokenizer_gen.hpp` +
   `runtime/src/gemma_tokenizer_gen.cpp`:
   `generate_gemma_tokenizer_table(tokenizer_json_path, sbert_config_path)`
   → `std::vector<uint8_t>`, a line-by-line port of every validation guard in
   the Python script (type/byte_fallback/dropout/prefix-suffix/normalizer
   checks, dense `[0, vocab_size)` id range with duplicate detection, every
   merge's two source pieces and their concatenation resolving to vocab ids,
   the five special-id constants, `post_processor` adding both `<bos>` and
   `<eos>`, the checkpoint's `config_sentence_transformers.json` `"prompts"`
   dict containing `"document"`), each throwing `std::runtime_error` with a
   message that names what was expected, on the same field, in the same
   order as the Python `raise SystemExit(...)` it mirrors. Binary layout
   (magic, version, counts, 5 special ids, add_bos/add_eos, prefix count +
   default index, then vocab/merges/prefixes) written with plain
   `put_u16`/`put_u32`/`put_str16` little-endian helpers — safe as plain
   writes on this platform (x86/x64 is little-endian natively).
4. Wired it into `npue_pack.cpp`'s `prepare_model_gemma()`: unchanged
   behavior when `<model_dir>/gemma_tokenizer.bin` exists (read as before);
   when it does not, call the new generator, use the bytes for the
   `tokenizer.gemma_table` tensor, **and** write them to
   `<model_dir>/gemma_tokenizer.bin` so a second pack of the same model hits
   the fast "have" path and the table stays inspectable on disk exactly like
   the Python tool's output always was.
5. Added `src/json_min.cpp` and `src/gemma_tokenizer_gen.cpp` to
   `runtime/CMakeLists.txt`'s explicit source list (it is not a glob).
6. **Verified against real ground truth, not self-reference** (CLAUDE.md
   traps 6b/6c): moved the existing, genuinely Python-generated
   `models/embeddinggemma-300m/gemma_tokenizer.bin` aside as a reference
   copy; compiled a throwaway standalone program
   (`verify_gen.cpp`, in the session scratchpad, never added to the repo or
   to CMakeLists.txt) that calls `generate_gemma_tokenizer_table()` against
   the real `tokenizer.json` + `config_sentence_transformers.json` and
   writes the result to a new file; compared the two with both
   `Get-FileHash -Algorithm SHA256` and `cmd /c fc /b`. **Byte-identical**
   (see Result). Then ran the *real* production path end to end: renamed
   `gemma_tokenizer.bin` aside again, ran the actual shipped
   `npuembed.exe --prepare-model models/embeddinggemma-300m
   <out>.npue.verifytest`, confirmed the log line
   `"generated tokenizer.gemma_table (no cached gemma_tokenizer.bin found)"`,
   confirmed the freshly-written `gemma_tokenizer.bin` was again
   byte-identical to the saved reference (sha256 match), and used
   `tools/npue.py`'s `Reader` (Python, read-only inspection, not part of the
   verified computation) to confirm the produced `.npue` actually contains a
   `tokenizer.gemma_table` tensor of `dtype=uint8, shape=(9020206,)`. Cleaned
   up all scratch/test artifacts under `models/embeddinggemma-300m/`
   afterward (that whole directory is gitignored, so nothing here affects
   the committed tree either way) and left the freshly-C++-generated
   `gemma_tokenizer.bin` in place (byte-identical to the original, no reason
   to prefer one over the other).

### Deliverable B

1. `main.cpp`'s subcommand handler (`embed`/`serve`) gained a `--token` scan
   in the same loop that already reads `--root`, producing `cli_token`
   (empty if not given).
2. `hub.hpp`/`hub.cpp`'s `ensure_model()` gained a trailing
   `const std::string &token_override = ""` parameter (default empty, so no
   other/future call site breaks — confirmed by grep that `main.cpp`'s is
   the only call site).
3. `ensure_model()`'s gated-model check now resolves the token with explicit
   precedence: `token_override` (from `--token`) wins if non-empty;
   otherwise `std::getenv("HF_TOKEN")`; if both are empty for a gated model,
   throws `std::runtime_error` **before** the fetch-list loop / any
   `download()` call, with an updated message naming both options. Neither
   the flag value nor the env var value is ever logged.
4. `main.cpp`'s call site passes `cli_token` through as the new last
   argument.
5. Added `--token VALUE` to `print_usage()`'s options list, next to
   `--root`.

## Commands

```powershell
# --- probe real tokenizer.json field types (conda iron env, not shipped code) ---
& "C:\Users\vegar\.conda\envs\iron\python.exe" "<scratchpad>\inspect_tok.py"

# --- reconfigure + build after adding the two new source files ---
cmd /c "`"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" && cd /d C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --target npuembed"

# --- back up the real Python-generated reference table ---
cp models/embeddinggemma-300m/gemma_tokenizer.bin models/embeddinggemma-300m/gemma_tokenizer.bin.python_reference

# --- compile a throwaway standalone verifier (not part of the shipped build) ---
cmd /c "`"...\vcvars64.bat`" && cl /std:c++17 /EHsc /O2 /I C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\include /Fe:<scratchpad>\verify_gen.exe <scratchpad>\verify_gen.cpp C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\src\json_min.cpp C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\src\gemma_tokenizer_gen.cpp"

# --- run it against the real checkpoint files ---
<scratchpad>\verify_gen.exe models\embeddinggemma-300m\tokenizer.json models\embeddinggemma-300m\config_sentence_transformers.json models\embeddinggemma-300m\gemma_tokenizer.bin.cpp_generated

# --- byte-diff (both methods) ---
Get-FileHash models\embeddinggemma-300m\gemma_tokenizer.bin.python_reference -Algorithm SHA256
Get-FileHash models\embeddinggemma-300m\gemma_tokenizer.bin.cpp_generated -Algorithm SHA256
cmd /c "fc /b models\embeddinggemma-300m\gemma_tokenizer.bin.python_reference models\embeddinggemma-300m\gemma_tokenizer.bin.cpp_generated"
# -> FC: no differences encountered

# --- force the REAL production packer to generate from scratch ---
mv models/embeddinggemma-300m/gemma_tokenizer.bin models/embeddinggemma-300m/gemma_tokenizer.bin.setaside
C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\build\npuembed.exe --prepare-model `
    C:\Users\vegar\Documents\GitHub\NpuEmbeddings\models\embeddinggemma-300m `
    C:\Users\vegar\Documents\GitHub\NpuEmbeddings\models\embeddinggemma-300m\embeddinggemma-300m.npue.verifytest
# -> "  generated tokenizer.gemma_table (no cached gemma_tokenizer.bin found)  9.02021 MB"

sha256sum models/embeddinggemma-300m/gemma_tokenizer.bin models/embeddinggemma-300m/gemma_tokenizer.bin.python_reference
# -> identical: c7a03c2c35ffc2a16b5513bb11c3d04e4a19c84acb9254b36765510acbf5bc81

# --- confirm the .npue actually carries the tensor, right size ---
& "C:\Users\vegar\.conda\envs\iron\python.exe" "<scratchpad>\inspect_npue.py"
# -> tokenizer.gemma_table: dtype=uint8 shape=(9020206,) nbytes=9020206

# --- cleanup ---
rm models/embeddinggemma-300m/embeddinggemma-300m.npue.verifytest
rm models/embeddinggemma-300m/gemma_tokenizer.bin.setaside models/embeddinggemma-300m/gemma_tokenizer.bin.python_reference

# --- Deliverable B: --token precedence, verified with a TEMPORARY debug throw
# (added, tested, then reverted before the final build -- see Problems hit) ---
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
npuembed.exe embed embeddinggemma-300m in.txt --root <fresh-empty-root>
# -> exit 2, clear error naming BOTH --token and HF_TOKEN, before any network call

npuembed.exe embed embeddinggemma-300m in.txt --root <fresh-empty-root> --token faketoken123
# -> (debug build) TEMP_DEBUG_TOKEN=[faketoken123]

$env:HF_TOKEN = "envtoken456"
npuembed.exe embed embeddinggemma-300m in.txt --root <fresh-empty-root>
# -> (debug build) TEMP_DEBUG_TOKEN=[envtoken456]

$env:HF_TOKEN = "envtoken456"
npuembed.exe embed embeddinggemma-300m in.txt --root <fresh-empty-root> --token clitoken789
# -> (debug build) TEMP_DEBUG_TOKEN=[clitoken789]   -- --token wins over HF_TOKEN

# --- final clean rebuild after reverting the debug throw ---
cmd /c "`"...\vcvars64.bat`" && cd /d C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\build && cmake --build . --config Release --target npuembed"
```

## Result

- **Deliverable A: byte-identical, twice.** The standalone throwaway
  generator matched the real Python-tool output at
  `sha256 c7a03c2c35ffc2a16b5513bb11c3d04e4a19c84acb9254b36765510acbf5bc81`,
  both files **exactly 9,020,206 bytes**, `fc /b` reporting "no differences
  encountered". Then the *actual production packer*
  (`npuembed.exe --prepare-model`), run against a checkpoint directory with
  `gemma_tokenizer.bin` deliberately removed, generated a table via the new
  C++ path and wrote it back to disk — **same sha256, same 9,020,206
  bytes**, confirming the wiring (not just the generator function in
  isolation) is correct. The resulting `.npue` was independently confirmed
  (via `tools/npue.py`'s `Reader`, not the C++ path being tested) to carry
  `tokenizer.gemma_table` as `dtype=uint8, shape=(9020206,)` — matching
  exactly.
- **Deliverable B: all four required behaviors confirmed**, the precedence
  test with a temporary instrumented build that throws the resolved token
  value before any network call (so no real HuggingFace request was ever
  made — no token was available and none was needed):
  1. Neither `--token` nor `HF_TOKEN` set → fails closed with the new
     dual-option error message, exit code 2, before the fetch-list loop.
  2. `--token faketoken123` alone → resolved token is `faketoken123`.
  3. `HF_TOKEN=envtoken456` alone (no `--token`) → resolved token is
     `envtoken456` (regression: old env-var-only behavior intact).
  4. Both set (`HF_TOKEN=envtoken456`, `--token clitoken789`) → resolved
     token is `clitoken789` — `--token` wins.
- Full Release build is clean (MSVC `/W3`, zero warnings from the new files;
  the pre-existing `getenv` C4996 deprecation note on the untouched
  `std::getenv` call is unchanged from before this task).
- `git status` after cleanup: five files modified
  (`runtime/CMakeLists.txt`, `runtime/include/hub.hpp`, `runtime/src/hub.cpp`,
  `runtime/src/main.cpp`, `runtime/src/npue_pack.cpp`), four new
  (`runtime/include/gemma_tokenizer_gen.hpp`, `runtime/include/json_min.hpp`,
  `runtime/src/gemma_tokenizer_gen.cpp`, `runtime/src/json_min.cpp`) — no
  stray scratch/debug files left in the tree.

## Problems hit

1. **The `Bash` tool's `cmd /c` invocation is broken in this sandbox** — every
   attempt (`cmd /c '...'` and `cmd /c "..."` both) returned only the
   Windows/cmd banner text and a stale prompt line, never the command's real
   output, even for `cmd /c "echo hello_world_test"`. Not investigated
   further (out of scope); worked around by running every `cmd /c
   vcvars64.bat && ...` build/compile invocation through the **PowerShell
   tool** instead, which invokes `cmd.exe` correctly. All build commands in
   this log are PowerShell-tool invocations of `cmd /c "..."`, not Bash-tool
   ones, unlike the pattern shown in tasks/0058-0066 (which apparently ran
   in a session where the Bash tool's `cmd` worked). Future sessions hitting
   the same symptom should try the PowerShell tool first before debugging
   further.
2. **PowerShell here-string (`@'...'@`) passed to `python -c` via `&` lost
   its quoting** (`SyntaxError: invalid syntax` on a raw-string literal that
   was clearly present in the source). Worked around by writing the probe to
   a `.py` file and running `python script.py` instead of `-c` with an
   inline here-string — same root cause class as the `cmd /c` issue, some
   layer between the PowerShell tool and the underlying process is not
   passing multi-line/quoted arguments through unmodified.
3. **`fc` failed inside the PowerShell tool** with a
   `Format-Custom`-binding error — PowerShell aliases `fc` to
   `Format-Custom`, not `cmd.exe`'s `fc.exe`. Not a bug in the change under
   test; worked around by invoking it as `cmd /c "fc /b ..."` explicitly.
   `Get-FileHash -Algorithm SHA256` was used as the primary comparison
   method for exactly this reason; `fc /b` was a second, independent
   confirmation.
4. **Verifying `--token` precedence without a real token or real network
   access** (explicitly disallowed by the task brief) needed a different
   approach than "run it and watch the download start". Solved by adding a
   temporary `throw std::runtime_error("TEMP_DEBUG_TOKEN=[" + bearer_token +
   "]")` immediately after `ensure_model()`'s token-resolution logic and
   before the `sha256.empty()` check that follows it, rebuilding, running
   all four precedence cases (each exits with code 2 and the debug message,
   never reaching `download()` or opening a socket), then reverting the
   throw and doing one final clean Release rebuild. Confirmed by `grep` that
   no `TEMP_DEBUG` string remains in `hub.cpp` after the revert.
5. **The "vocab has N unused ids" error path's exact numeric behavior was
   not exercised** — the real checkpoint's vocab is dense (262,144/262,144
   entries, no gaps), which is the expected, un-triggerable-by-construction
   case for a real HuggingFace tokenizer.json. This guard (like several of
   the fail-closed checks ported from the Python script) is verified
   **by code review against the Python original**, not by a golden failing
   input, since no known checkpoint revision violates it. Recorded honestly
   rather than claimed as tested.

## Artifacts

- `runtime/include/json_min.hpp` / `runtime/src/json_min.cpp` — new,
  dependency-free recursive-descent JSON DOM parser.
- `runtime/include/gemma_tokenizer_gen.hpp` /
  `runtime/src/gemma_tokenizer_gen.cpp` — new, the C++ port of
  `tools/gen_gemma_tokenizer_table.py`.
- `runtime/src/npue_pack.cpp` — `prepare_model_gemma()`'s tokenizer-table
  block now generates-and-caches instead of read-only-with-a-warning.
- `runtime/CMakeLists.txt` — two new source files added to the explicit list.
- `runtime/include/hub.hpp` / `runtime/src/hub.cpp` — `ensure_model()` gained
  `token_override`, the gated-check precedence logic and message.
- `runtime/src/main.cpp` — `--token` flag scan, call-site wiring,
  `print_usage()` entry.
- `models/embeddinggemma-300m/gemma_tokenizer.bin` — regenerated by the new
  C++ path during verification, byte-identical to the pre-existing
  Python-generated one (that directory is gitignored; noted for anyone
  diffing `models/` locally, not a committed change).
- No file in this repository or its scratch outputs contains a real
  HuggingFace token — Deliverable B's precedence tests used only
  fabricated placeholder strings (`faketoken123`, `envtoken456`,
  `clitoken789`), and no real network request to huggingface.co was made
  anywhere in this task.

## Next

- The gap tasks/0066 left open under T29 (`gemma_tokenizer.bin` has no C++
  generator) is now closed. `research/OPEN-THREADS.md`'s T29 entry updated
  accordingly.
- Not done, out of scope for this task: no attempt was made to speed up the
  JSON parser beyond the run-of-bytes fast path. It is a build-time-only
  cost, run once per fresh model install (CLAUDE.md rule 5) — **measured
  at 2,793 ms wall clock** (`[System.Diagnostics.Stopwatch]`, includes
  process startup) for the standalone verifier to parse the 33 MB
  `tokenizer.json`, walk 262,144 vocab entries + 514,906 merges, parse the
  small `config_sentence_transformers.json`, and write the 9,020,206-byte
  table — comfortably inside the "a few seconds at most" the task brief
  allowed, and this is a wall-clock host-tool timing (not an NPU claim), so
  rule 1 does not apply.
