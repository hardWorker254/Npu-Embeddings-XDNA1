# 0144 — Granite 4.2-3B into Q4NX, and reading the container back

**Status: the conversion half is DONE and verified on the CPU — the fold is
bit-exact and the model generates coherently. The runtime half is BLOCKED, and
now measured rather than suspected: `llama_npu.dll` whitelists `hidden_size` to
{2048, 3072, 4096} and granite is 2560 ([T54](../../research/CLOSED-THREADS.md#t54)).**

This task is not about this repository's product. It is work in
`../q4nx-build` and `../FastFlowLM`, logged here because rule 3b wants every
unit of work recorded and because two of its findings — a second, independent
confirmation of the `q4nx` container layout, and the shape of FLM's closed
boundary — belong with this project's XDNA2 material rather than in a converter
repo's commit log.

## Goal

Add `ibm-granite/granite-4.2-3b` to FastFlowLM: convert it to Q4NX, and find out
what else the port needs. Test first, upstream later.

## What Granite actually is

Worth stating up front, because it decides everything below.
`granite-4.2-3b` is **`GraniteForCausalLM`** — dense, no Mamba, no MoE, unlike
the `granite-4.0-h-*` hybrids. `gguf`'s own tensor set for arch `granite` is
**exactly Llama's dense subset**:

```
$ python -c "from gguf.constants import MODEL_ARCH, MODEL_TENSORS, TENSOR_NAMES;
  g=set(MODEL_TENSORS[MODEL_ARCH.GRANITE]); l=set(MODEL_TENSORS[MODEL_ARCH.LLAMA]);
  print('granite-only:', sorted(TENSOR_NAMES[t] for t in g-l))"
granite-only: []
```

The difference is four scalar multipliers, carried in the GGUF metadata as
`granite.attention.scale`, `granite.embedding_scale`, `granite.residual_scale`
and `granite.logit_scale`. In this checkpoint **three of the four are 1.0**:

```
hidden 2560 · inter 8192 · 40 layers · 40 heads / 8 KV · head_dim 64 · vocab 100352
attention_multiplier 0.015625 · embedding_multiplier 1.0
residual_multiplier  1.0      · logits_scaling       1.0
```

**All four fold exactly into the weights**, which is the finding the port rests
on — nothing in the FLM runtime reads `attention_multiplier` (verified: the
string does not appear in `flm.exe` or any `*_npu.dll`, while `hidden_size`
does appear in `flm.exe`):

```
attention_multiplier -> q_proj   *= attention_multiplier * sqrt(head_dim)
embedding_multiplier -> embed_tokens *= embedding_multiplier
residual_multiplier  -> o_proj, down_proj *= residual_multiplier
logits_scaling       -> lm_head  /= logits_scaling
```

The attention fold is the only live one here: llama's implicit scale is
`64**-0.5 = 0.125`, granite wants `0.015625`, so `q_proj *= 0.125`. RoPE is a
rotation and commutes with a scalar, so folding before RoPE is safe.

**Folding into the quantized form is lossless.** A Q4_1 block stores
`w = code*d + m` with `d = (max-min)/15`, `m = min`; scaling every weight by
`c > 0` scales `max` and `min` by `c`, hence `d` and `m` by `c`, and leaves
every 4-bit code untouched. `c = 0.125` is a power of two, so the `d`/`m`
scaling is exact as well — this conversion introduces **no** error beyond Q4_1
itself.

## Why the runtime half is a different problem

FastFlowLM's orchestration is MIT and open; everything that touches the array is
not. Every per-family engine is a pimpl over a shipped binary
(`src/include/models/llama/llama_npu.hpp` is `struct Impl; Impl* _impl;`), the
sequence generators too, and `src/xclbins/<Name>/` holds prebuilt MLIR-AIE
artifacts. Adding a family is ~40 lines of C++ (commit `e7eee4d`, "feat:
nanbeige done", is the template: 17 files, ~690 lines, of which the only
unbuildable parts are the binary blobs).

**The blocker is an ABI, not a toolchain.** We can build xclbins — this project
has 253, and `LLMNpuTest/designs/lm_head` is a working 8-core kernel against
FLM's own weights. But `llama_npu.dll` generates its own instruction streams
(`gen_layer_seq`, `gen_mha_engine_seq`) and patches RTPs at the addresses
`config.json` carries as `addr_qk / addr_kv / addr_kk / addr_l_begin_mha /
addr_l_end_mha`. An xclbin we compile would be a fine xclbin the closed DLL
cannot drive.

What the shipped artifacts say is in
[T54](../../research/CLOSED-THREADS.md#t54) rather than here, per rule 3. As it
turned out, none of that analysis was reached: the engine rejects the model on
`hidden_size` before it opens an xclbin at all — see §6.

## What was built

In `../q4nx-build`, branch `feat/granite`:

| commit | |
|---|---|
| `feat: support IBM Granite dense models (granite-4.x)` | `configs/granite.json`, `q4nx/models/granite.py`, registry + detection wiring, `tests/test_granite.py` |
| `fix: install gguf from PyPI instead of cloning llama.cpp` | see below |
| `feat: read Q4NX back, and verify a conversion against its source` | `q4nx/unpack.py`, `tools/verify_q4nx.py`, `tools/oracle_granite.py`, `tests/test_unpack.py` |

`Granite` **subclasses `Llama`** and overrides only the multipliers. The house
style in that repo is a copy-pasted class per family (`nanbeige.py` is 95%
`llama.py`), so this is a deliberate divergence: the delta really is four
scalars, and a future fix to the shared GGUF path should reach Granite for free.
Flagged in the PR so a maintainer can ask for the copy-paste form instead.

Two things it **refuses rather than guesses**:

* a partial rotary factor (`rope.dimension_count != hidden/heads`), which would
  make the q/k RoPE permutation wrong;
* HF-safetensors input — see the defect below.

### Defect found: the HF path never quantizes

`model_converter.py`'s `_convert_hf` stores raw float tensors and **never calls
`_pack_q4nx`**; `_export_weights` then `save_file`s them as `model.q4nx`. Every
`_pack_q4nx` call site is in a `_convert_gguf`:

```
$ grep -n "_pack_q4nx" q4nx/models/*.py | grep -v "def "     # all in _convert_gguf
```

So `-i <hf-repo>` produces a float safetensors file with a `.q4nx` name that the
runtime cannot read. Not fixed here — it is a shared-code change well outside a
model port — but `Granite._convert_hf` raises `NotImplementedError` with the
reason rather than silently emitting one.

### Defect found: `requirements.txt` clones llama.cpp for a 100 KB package

This one cost real bandwidth before it was noticed. `requirements.txt` had:

```
git+https://github.com/ggml-org/llama.cpp.git#subdir=gguf-py
```

which makes pip clone the **entire llama.cpp repository** — multi-GB of git
history — to extract `gguf-py`. `gguf` is on PyPI. Verified the pin was not
load-bearing before changing it: `gguf 0.19.0` exposes every quantization type
the converter references, `MXFP4` included.

The symptom was misleading in a way worth recording: the household connection
was saturated for ten minutes and the obvious suspect was the model download.
It was not — the model download had landed **0.02 MB**. Measuring instead of
assuming is what found it:

```
$ find ~/.cache/huggingface/hub -name "*.incomplete" -printf "%s %p\n"
$ Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'git|pip|python' }
```

The second command also showed the *rest* of the traffic was a GitHub Desktop
`git clone --recursive` of q4nx-build, not ours at all.

## The container reader, and why it is trustworthy

`q4nx/unpack.py` inverts `_pack_q4nx`. The converter only ever wrote, so the
only check on its output was "does the model talk" — and per `LLMNpuTest`'s own
lesson, an aggregate cannot localise a fault and with two faults present cannot
even detect one.

**A third, independent confirmation of
[note 0010](../../research/notes/0010-q4nx-format.md).** That note solved the
container against ground truth — the `ssm_alpha_proj` tensor that
`Qwen3.5-0.8B-NPU2` stores twice — and `LLMNpuTest/tools/q4nx.py` reads it from
the published model file and FLM's MIT headers. Neither used a *writer*, because
none was known to be available.

There is one: `q4nx-build` is MIT, and its `_pack_q4nx` is an open-source
implementation of the same packing. Inverting it reproduces note 0010's layout
**bit-for-bit** — tile size, metadata indexing, and the nibble parity the note
records as unsettleable from the file alone. Cross-checked by packing a random
tensor with `_pack_q4nx` and dequantizing it both ways:

```
tiles compared     : 4
max abs diff       : 0.0
bit-identical      : True
```

They agree on tile size (5120 B), metadata indexing (`kb*32 + rb*16 + r`) and —
the one that could not be settled from the file alone — **nibble parity**
(`LOW_NIBBLE_IS_EVEN`). The parity measured against ground truth is the parity
the writer actually emits.

Worth carrying forward: **an open writer for this format existed the whole
time.** It does not diminish note 0010's method — ground truth is what makes a
reading *checkable*, and the writer only says what one implementation does — but
if `q4nx-build` had been read first, the parity question would not have needed
solving. Where a closed format has an open converter somewhere in its ecosystem,
that converter is worth looking for before reverse-engineering the container.

`tests/test_unpack.py` round-trips packer→reader with no model file and no
download, comparing codes **exactly** — a tiling, parity or half-block error is
a permutation, not noise, so it matches perfectly or not at all. Both boundaries
that produce plausible-looking garbage have a dedicated test: the even/odd
nibble parity, and the 16-row half-block split.

## Exact commands

Environment. `.venv-ref` already had torch, transformers, safetensors and
einops; only `gguf` was missing, and it is a pure-Python leaf package nothing
else here imports, so it went in with `--no-deps` rather than costing ~400 MB
for a fresh venv with its own torch:

```
$ ./.venv-ref/Scripts/python.exe -m pip install --no-deps --no-cache-dir gguf
gguf 0.19.0 | numpy 2.4.3 | torch 2.10.0+cpu | transformers 5.15.0
```

Source weights. `Q8_0` rather than `bf16`, at the operator's choice, on a
household connection: 3.89 GB against 7.32 GB, and Q8_0's ~0.2% error sits far
below the Q4_1 floor this converts into. **Not** `Q4_1`, despite being smallest
and already the target format — its embedding table and `lm_head` would be
Q4_1-dequantized, and FLM deliberately keeps embeddings at bf16.

`huggingface_hub` has no rate limit; `curl` does, and the CDN honours ranges
(`accept-ranges: bytes`), so the download is capped and resumable:

```
$ curl -L -C - --limit-rate 2M --retry 5 --retry-delay 10 \
    -o granite-4.2-3b-Q8_0.gguf \
    https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/resolve/main/granite-4.2-3b-Q8_0.gguf
```

`config.json`, `tokenizer.json`, `chat_template.jinja` and friends came from
`ibm-granite/granite-4.2-3b` itself (9.4 MB, small files only).

Tests:

```
$ cd ../q4nx-build && ../NpuEmbeddings/.venv-ref/Scripts/python.exe -m unittest discover -s tests -q
Ran 92 tests in 0.263s
OK
```

57 of those were already there and still pass; 35 are new.

## Housekeeping: 0143's session links came back

[`0143`](../0143-gte-cpu-energy/TASK.md) Part 3 stripped `Claude-Session:`
trailers from this repository's history and installed
`tools/git-hooks/commit-msg`, writing that "no prose here would stop the next
session doing it again."

It did not. The hook is installed in **this** repository via `core.hooksPath`;
`q4nx-build` (public, `github.com/vegah/q4nx-build`) and `FastFlowLM` have no
such hook, and three commits were written there carrying the trailer before it
was noticed. Caught before pushing and stripped:

```
$ FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
    --msg-filter 'grep -v "^Claude-Session: https://claude.ai/code/session_"' 90dcda0..HEAD
$ git log 90dcda0..HEAD --grep="Claude-Session" --oneline | wc -l
0
```

Both repositories now point at this repository's tracked hook, as **local config
only** so nothing lands in a pull request:

```
$ git -C ../q4nx-build  config core.hooksPath C:/Users/vegar/Documents/GitHub/NpuEmbeddings/tools/git-hooks
$ git -C ../FastFlowLM config core.hooksPath C:/Users/vegar/Documents/GitHub/NpuEmbeddings/tools/git-hooks
```

That is a per-clone setting and will not survive a fresh clone, which is the
same weakness 0143's own install has. Recorded, not solved.

## Results

The conversion ran in **79 s** and produced a 2.64 GB `model.q4nx`. The
converter's own line confirms the metadata read and the single live fold:

```
[INFO] Granite multipliers: attention=0.015625, embedding=1.0, residual=1.0,
       logits_scaling=1.0 (head_dim=64)
[INFO] Granite fold: self_attn.q_proj.weight *= 0.125
        Converted output.weight to lm_head.weight        <- not tied
```

### 1. Per-tensor diff against the source — PASS

```
$ python tools/verify_q4nx.py ~/.cache/openfflm/Granite-4.2-3B-NPU2 \
    --gguf ~/.cache/openfflm/granite-4.2-3b-GGUF/granite-4.2-3b-Q8_0.gguf

[INFO] 282 tensors checked, floor 0.99
[INFO] worst: model.layers.21.self_attn.v_proj.weight at 0.996150
[PASS] every tensor is at or above the quantization floor
```

Worth one line of corroboration: `LLMNpuTest`'s own tensor-by-tensor diff of
FLM's *shipped* Qwen3.5 weights reports **worst 0.996116** across 150 tensors.
Landing at 0.996150 on an independent model and an independent conversion is
good evidence that this is the Q4_1 group-of-32 floor and not a defect.

### 2. Fold equivalence — PASS, and exactly

```
$ python tools/oracle_granite.py ~/.cache/openfflm/Granite-4.2-3B-NPU2 \
    --config <upstream granite-4.2-3b config.json>

[CHECK 1] fold equivalence -- llama(folded) vs granite(unfolded)
          logits cosine   : 1.00000000
          max abs diff    : 0.000e+00
          same argmax     : yes
```

**Bit-identical, not merely close.** That is not luck: `attention_multiplier`
0.015625 and llama's `64 ** -0.5` = 0.125 are both powers of two, so the fold
and its inverse are exact in bf16 — no mantissa bit moves. The same weights
through a stock `LlamaForCausalLM` (which applies `head_dim ** -0.5`, exactly
what FLM's llama engine does) and through `GraniteForCausalLM` with the fold
divided back out produce the same logits to the last bit.

That is the claim the port rests on, and it was checked against **no external
reference** — same weights, two implementations, forced to agree.

### 3. Generation — PASS

```
'The capital city of France is Paris.\n\nSo the answer is Paris.\n\nBut the instruction says'
```

Coherent, correct, and visibly a thinking model. This is what covers what a
cosine diff cannot reach: a wrong RoPE permutation, tile layout or nibble
parity yields garbage here, not a small numeric error.

Run with `--dtype bfloat16` and the two models built and freed one at a time.
A first attempt at fp32 with both resident was killed at the 2-minute mark: a
3B model is 12.8 GB at fp32 and the state_dict another 12.8 GB, against 17.5 GB
free on a 29.6 GB machine.

### 4. A defect the results found: the config described the wrong model

The emitted `config.json` was **`granite-4.1-3b-base`'s**. The skeleton is
resolved by following the GGUF's `general.base_model.0.repo_url`, which names
the model this one was *trained from*:

```
general.base_model.0.repo_url  https://huggingface.co/ibm-granite/granite-4.1-3b-base
```

| key | skeleton said | truth |
|---|---|---|
| `embedding_multiplier` | 12.0 | 1.0 |
| `residual_multiplier` | 0.22 | 1.0 |
| `logits_scaling` | 10.0 | 1.0 |
| `tie_word_embeddings` | true | false |
| `bos_token_id` | 100257 | 100283 |

Every dimension matched, so nothing complained and the model still ran. Exactly
the shape this project keeps meeting: **shapes agree, output is plausible, and
the artifact describes something it does not contain.**

Fixed in `q4nx/model_assets.py`. `reconcile_config_with_gguf` warns for every
family; dimensions are reported but **not** rewritten, because nanbeige
deliberately publishes a padded `intermediate_size` the GGUF does not carry.
Token ids are corrected. `apply_granite_fold_to_config` makes the config
describe the folded weights — post-fold `attention_multiplier` 0.125, the other
three 1.0, originals preserved under `q4nx_folded_multipliers`.

Note the tokenizer files were already right: `ensure_runtime_tokenizer_ids`
patches `tokenizer_config.json` from GGUF metadata, and that is the file the
runtime actually reads for bos/eos. Only `config.json` was stale.

### 5. A test-harness bug worth recording

The first run of the new reconciliation tests failed with a `TypeError` deep in
`apply_granite_fold_to_config`. Cause: the fake GGUF field returned `None` from
`contents()` for `general.architecture`, because the fake was built for
`get_model_arch_from_gguf` (which reads `field.parts`) while
`model_assets._gguf_field` trusts `contents()` first. Every arch-prefixed
lookup silently missed. A fake that is wrong in the same direction as the code
under test proves nothing — the fake now returns the string from `contents()`
like a real reader does.

## 6. The NPU experiment — T54, answered

The spike ran. **The engine refuses granite before it reaches an xclbin:**

```
[FLM]  Using custom model list path: C:/Users/vegar/flm-granite-spike/model_list.json
[FLM]  Loading model: C:\Users\vegar\Documents\flm\models\Granite-4.2-3B-NPU2
[ERROR]  Failed to load model: Unsupported hidden size: 2560
```

Setup, exactly as [T54](../../research/CLOSED-THREADS.md#t54) predicted it could
be — **no administrator, nothing written into `Program Files`**:

```
$ SPIKE=~/flm-granite-spike
$ cp "C:/Program Files/flm/model_list.json"  $SPIKE/
$ cp "C:/Program Files/flm/model_info.json"  $SPIKE/
$ cp "C:/Program Files/flm/xclbins/Llama-3.2-1B-NPU2/"*.xclbin \
     $SPIKE/xclbins/Granite-4.2-3B-NPU2/
$ export FLM_CONFIG_PATH=$SPIKE/model_list.json
$ export FLM_MODELINFO_PATH=$SPIKE/model_info.json
$ export FLM_XCLBIN_PATH=$SPIKE
$ flm list        # -> granite:3b ✅
$ printf 'What is the capital of France?\n/bye\n' | flm run granite:3b
```

granite was registered under the existing `"family": "llama3"`, so **no C++ was
written**, and the model's `config.json` got the donor's five `addr_*` RTP
addresses.

Then the whitelist was mapped by editing `hidden_size` and reading which error
came back — a rejected value stops at the gate, an accepted one gets past it and
fails later on `Invalid size for bytes allocation` because the weights no longer
match the declared width:

```
512 1024 1536 1792   REJECTED
2048                 accepted
2304 2560 2816       REJECTED
3072                 accepted
3584                 REJECTED
4096                 accepted
5120 8192            REJECTED
```

`{2048, 3072, 4096}` is exactly the set of hidden sizes among the three shipped
llama-family models. **The whitelist is the shipped design set, enumerated.**

### The failure worth keeping: three rounds of probes that measured nothing

The first three probe rounds all reported `Unsupported hidden size: 2560`
whatever was changed, and two wrong conclusions were drawn from that in
sequence — first that the gate derived the width from `head_dim x
num_attention_heads`, then that it read it from the weight tensors. Both were
announced before they were checked.

The cause was neither. The probe passed a **POSIX path to the Windows Python**:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/c/Users/vegar/Documents/flm/models/Granite-4.2-3B-NPU2/config.json'
```

Every config edit had failed silently, so every run re-read the same unmodified
2560. The loop *did* contain a verification print — but it was inside the same
python call that was failing, and the grep filtering the output selected only
lines matching `unsupported|error`, so the traceback was discarded and the
verification never appeared. **A check that cannot survive the failure it is
checking for is not a check.** The rerun prints the value read back from disk
*after* writing, outside the filtered stream.

Only the weight-swap probe was valid, because it used `cp` rather than Python —
and it is the one whose result (unchanged error with hidden-2048 weights in
place) was correct and, in hindsight, already told the story: the gate never
looks at the weights.

### 6b. Independent corroboration: someone already hit this wall

[ROCm/FastFlowLM#572](https://github.com/ROCm/FastFlowLM/pull/572)
("Feature/huggingface", by **Atomic-Germ** — who is also q4nx-build's upstream
author), closed unmerged, against
[#569](https://github.com/ROCm/FastFlowLM/issues/569) "Feature Request: IBM
Granite 4.1 models". Its description says:

> Partly closes #569 along with `Atomic-Germ/Granite-4.1-8B-NPU2` [...]
> **Other sizes won't work due to llama support not being quite compatible
> enough to fudge.**

That is the same wall, found independently and from the other side. Granite
**8B** works as a llama-family model because its hidden size is **4096** — on
the whitelist. "Other sizes won't work" is 2560 (and every other granite width)
failing the same gate this task measured.

Two things follow. First, granite-on-llama is not speculative: the fold-and-run
approach is already in production for one size, which is evidence that the
architecture mapping is sound and the *only* obstacle for 4.2-3B is the width.
Second, the whitelist is worth stating publicly, because "not quite compatible
enough to fudge" is a much weaker description than `{2048, 3072, 4096}` — the
latter tells you immediately which sizes work and that padding is the fix.

(`Atomic-Germ/Granite-4.1-8B-NPU2` itself returns 401, so its config could not
be read; the conventions it uses are unknown.)

## 7. Is the converter itself correct? Checked against FLM's own shipped models

`tests/test_unpack.py` only proves the reader and the writer agree with each
other — they could agree on the wrong layout. `tools/validate_against_flm.py`
checks against ground truth neither produced: read every installed FLM model,
unpack it, repack it, and require the result to be **byte-identical to the file
AMD ships**. No downloads; it reads what is already on disk.

```
$ python tools/validate_against_flm.py --all "C:/Users/vegar/Documents/flm/models"
DeepSeek-R1-0528-Qwen3-8B-NPU2   253 ok, 0 differ      Llama-3.2-1B-NPU2        113 ok, 0 differ
Gemma3-1B-NPU2                    52 ok, 0 differ      Phi4-mini-Instruct-NPU2  225 ok, 0 differ
Gemma3-4B-NPU2                   239 ok, 0 differ      Qwen3-4B-NPU2            253 ok, 0 differ
Granite-4.2-3B-NPU2              281 ok, 0 differ      Qwen3-8B-NPU2            253 ok, 0 differ
Llama-3.1-8B-NPU2                225 ok, 0 differ      Qwen3.5-{4B,9B x3}       120 ok each, 0 differ

[PASS] 2374 tensors reproduce FLM's shipped bytes exactly
```

**13 models, 6 architecture families, zero mismatches.** That is the strongest
statement available about the packer without hardware.

Two things the tool had to learn to be honest:

**Padded tensors.** `_pack_q4nx` rounds rows up to `row_block_size` and cols up
to `col_block_size` *before* tiling, so a model whose hidden size is not a whole
number of column-blocks ships wider tensors than its config implies — gemma3 is
hidden 1152 against a 256-wide column block. Deriving shapes without that gave a
wall of spurious skips.

**FLM's shipped Gemma3-1B contains NaN in its quantization metadata.** Gemma3-1B
first showed 52 tensors "differing" — but only ever in the `d`/`m` planes, never
in the codes, and the bytes decode to NaN. Counted directly:

| model | tensor | NaN in `d` | NaN in `m` | Inf in `d` |
|---|---|---:|---:|---:|
| Gemma3-1B-NPU2 | `layers.0.mlp.down_proj` | 281 | 538 | 2 |
| Gemma3-1B-NPU2 | `layers.0.self_attn.o_proj` | 33 | 76 | 0 |
| Llama-3.2-1B-NPU2 | `layers.0.self_attn.o_proj` | **0** | **0** | 0 |

A round trip preserves the *value* (still NaN) but normalises the payload bits,
so this was our artifact, not a packing error — the comparison is now NaN-aware.
But the underlying observation stands and is somebody else's bug: a shipped
model carries NaN scales, and `Gemma3-1B` is the only one of the 13 that does.
Not chased further here; it is out of scope for a granite port and belongs
upstream.

## 8. Padding hidden 2560 -> 3072

The whitelist says 3072 is accepted, and `Llama-3.2-3B-NPU2` is the only
llama-architecture design at that width — with `intermediate_size 8192`, which
granite **already matches exactly**. So a padded granite lands on a shipped
design rather than needing a new one.

### What can and cannot be padded

**Hidden can.** Every tensor that writes into the residual stream gets zero rows
and every tensor that reads from it gets zero columns, so the padded lanes stay
identically zero at every layer. In Q4_1 the added groups are whole (the widths
are multiples of 32), so `d = 0`, `m = 0`, `code = 0` and `w = 0*0 + 0 = 0`
exactly.

**Heads cannot**, and this is the reason the pad is hidden-only. Granite is
40 q-heads / 8 KV (GQA group 5). Matching Llama-3.2-3B's 3072-wide q at
head_dim 64 would need 48 heads, and its 1024-wide k/v would need 16 KV heads —
group 3. **Eight groups of five do not embed into sixteen groups of three**
without changing which q head reads which KV head. So the head structure stays
as granite has it, and the run depends on head counts being RTP-driven rather
than compiled in — which the `attn.xclbin` sharing evidence supports
(Nanbeige 20/4 and Llama-3.1-8B 32/8 ship the same byte-identical file).

### RMSNorm is the part that is not automatic

Widening H -> H' shrinks `mean(x^2)` by exactly H/H', so RMSNorm's denominator
moves. Both terms are corrected, and together they cancel it for **every** input
rather than approximately:

```
norm weights *= sqrt(H/H')      = sqrt(2560/3072) = 0.912870929
rms_norm_eps *= H/H'            = 2560/3072       -> 8.333333e-06
```

With `eps' = eps·H/H'` the radicand becomes `(H/H')(S/H + eps)`, whose square
root carries a factor `sqrt(H/H')` that the norm-weight scale cancels exactly.
Scaling only the weights leaves the epsilon term behind — small, but a real
error, and one no shape check would ever see.

**The existing `--pad-to-fit` does not do this.** `q4nx/models/qwen35.py` pads
the norm tensors (`PAD_HIDDEN_NORM_SUFFIXES`) with zeros and rescales nothing,
while the flag's help text says "padded channels are inert". That is true of
every matmul and false of RMSNorm. Either it is a latent bug in that path, or
FLM normalizes over something other than the padded width — so `--pad-hidden`
takes the mathematically correct route and `--no-pad-norm-fix` exists to build
the other one, which makes the question answerable on hardware instead of
arguable.

### A bug the CPU check caught before hardware could

The first padded build failed in the comparison harness with

```
RuntimeError: shape '[228, 5120]' is invalid for input of size 983040
```

983040/5120 = 192 tiles, which is `(512/32) x (3072/256)` — the correct
`k_proj` at 8 KV heads x head_dim 64. The expected 228 came from deriving
`head_dim = hidden_size // num_attention_heads`, which granite's config invites
because it **omits `head_dim` entirely**. That derivation is right for the
unpadded model (2560/40 = 64) and wrong the instant hidden is padded
(3072/40 = **76**).

So a padded model must carry an explicit `head_dim`, and the converter now
always writes one. Worth generalising: **any config key that can be derived
from another will be, by somebody, and padding breaks exactly those
derivations.** This one failed loudly in a reshape; the same wrong 76 inside a
closed engine would not have.

### Padding makes hidden indivisible by the head count, and that is unusual

`transformers` refuses the padded config outright:

```
ValueError: The hidden size (3072) is not a multiple of the number of
attention heads (40).
```

That is a `transformers` validator, not an FLM rule — and it only runs in
`__init__`, so the harness builds at the unpadded width and widens before the
layers are created. But it points at something real: **every model FLM ships
has `hidden_size % num_attention_heads == 0`** (Qwen3-4B 2560/32, Gemma3-1B
1152/4, Qwen3.5-4B 2560/16 …). A padded granite at 3072/40 would be the first
that does not, so whether the engine cares is an open question the run will
answer.

**If it does care, there is a fix that keeps GQA intact.** Pad the *query*
heads 40 -> 48 while leaving KV at 8, by inserting one zero head after every
five real ones. The group size becomes 48/8 = 6, and placing the five real
heads at offsets 0–4 of each group of six preserves exactly which q head reads
which KV head — the mapping that plain head-padding breaks. Then
48 x 64 = 3072 = hidden, divisibility holds, and the zero heads contribute
nothing because their `o_proj` columns are zero. (A zero q-head attends
uniformly and returns the mean of its KV group, which is *not* zero — it is
`o_proj` that discards it, so the zero columns are load-bearing.)

Not implemented; only needed if the engine rejects the indivisible form.

## 9. The padded model on hardware: it loads, then dies

The padded build **passes the gate and loads**:

```
[FLM]  Loading model: ...\Granite-4.2-3B-NPU2
[FLM]  Loading model: granite:3b
[FLM]  WebServer started on port 52625
[🟢 ]  NPU Locked!
[FLM]  Start prefill...
[FLM]  Prefill chunk 1/1 with 49 tokens
```

Then, depending on which xclbins it was given:

| `layer` | `attn` + `addr_*` | result |
|---|---|---|
| Llama-3.2-3B (hidden 3072) | Llama-3.2-1B (head_dim 64) | **hangs** in prefill — CPU flat at 3.09 s across a 10 s sample, the signature of a DMA waiting on a lock that never comes |
| Llama-3.2-3B | Llama-3.2-3B (head_dim 128) | **segfault** |

### Why: there is no shipped design with granite-3B's geometry

granite-4.2-3B needs **head_dim 64 at hidden 2560**. Enumerating the shipped
designs by `attn.xclbin` hash:

* the head_dim-64 attn design (`ed95180d…`) is used by **six** models —
  `Llama-3.2-1B`, `LFM2-1.2B`, `LFM2-2.6B`, `LFM2-2.6B-Transcript`,
  `LFM2.5-1.2B`, `LFM2.5-1.2B-Thinking` — and **every one of them is hidden
  2048**;
* every design at hidden ≥ 2560 is head_dim 128.

So the two requirements are satisfied by disjoint sets, and padding cannot
bridge them: hidden can be padded **up** (2048 is below granite's 2560, so the
head_dim-64 design cannot be reached by widening), and head_dim is intrinsic to
RoPE and cannot be padded at all.

**granite-4.2-3B cannot run on FLM's shipped designs, by any combination.**
That is stronger than T54's answer, which only established that the gate
rejects 2560: the gate can be passed by padding, and the model still cannot run.

### But granite-4.2-**8B** is a geometric match

| | granite-4.2-8b | Llama-3.1-8B design |
|---|---:|---:|
| hidden | 4096 | 4096 (whitelisted) |
| heads / KV | 32 / 8 | 32 / 8 |
| head_dim | 128 | 128 |
| intermediate | **12800** | **14336** |

Everything matches but the MLP width, and the 1536 gap is three 512-wide blocks
— the same padding nanbeige already ships (10496 -> 10752). Intermediate padding
is **exact with nothing to correct**: no norm spans that axis, so unlike
`--pad-hidden` there is not even a bf16 rounding.

This is also the geometry
[#572](https://github.com/ROCm/FastFlowLM/pull/572) already has working
(`Granite-4.1-8B-NPU2`, same shape as 4.2-8B), which is independent evidence
that the approach lands.

`--pad-intermediate` is implemented and tested for it. What is missing is the
weights: granite-4.2-8b is ~8.7 GB at Q8_0, and this is a household connection.

## 10. Direction: build the engine rather than borrow one

Faced with "granite-4.2-3B needs a design AMD has not compiled", the operator's
call was **"Vi lager xclbin og dll for den!"** — build our own. That is a
different project from everything above, and it removes the constraint that
drove all of it:

**Our own engine has no whitelist.** The `{2048, 3072, 4096}` gate lives inside
`llama_npu.dll`. FLM's `causal_lm` interface (`src/include/causal_lm.hpp`) is
open and MIT, so a `granite_npu` implementing it runs granite-4.2-3B at its
native hidden 2560 and head_dim 64 — **no padding at all**. Everything in §8
and §9 becomes unnecessary for this path (though `--pad-intermediate` remains
the right tool for granite-4.2-8B on the stock engine).

### The prerequisite that was assumed dead

[[flm-windows-build-needs-msvc]] recorded that FLM could not be built here:
no MSVC, no Boost, no XRT. **That memory is stale.** MSVC 18 (2026) is
installed, `C:/dev/XRT` and `C:/dev/vcpkg` are present, Boost resolves through
vcpkg, and `src/build/CMakeCache.txt` is configured for
`Visual Studio 18 2026`. A build had been *configured* on 2026-08-11 but never
run past `ZERO_CHECK`.

### Staged plan, smallest milestone first

| | | |
|---|---|---|
| **M1** | `granite_npu` as a **host** implementation of `causal_lm` | `flm.exe` runs granite end to end — chat, server, templates, sampling — with correct output and no NPU. This alone is "a working FastFlowLM running the model", and it is checkable against the oracle already built in §2. |
| **M2** | `lm_head` on the array, our own IRON design | 34% of the per-token MACs in one dispatch; `LLMNpuTest/designs/lm_head` is a working 8-core precedent, fp32-exact, needing only a reshape from hidden 1024 to 2560. |
| **M3** | the projections (W4A16 GEMV) | the bulk of the remaining arithmetic. |
| **M4** | a fused layer | where `LLMNpuTest`'s dispatch measurements say the win actually is: a layer matmul costs 3x more to issue than to run, so per-op kernels lose. |

M1 is the honest deliverable and M2–M4 are where the learning is. Note that M1
also gives every later stage a reference to diff against **inside FLM's own
process**, which nothing else in this task has had.

## 11. M1: FastFlowLM builds here, and runs granite through an engine we wrote

### The build was never actually blocked

[[flm-windows-build-needs-msvc]] said this was a multi-hour, 15–20 GB setup and
advised against starting it. Wrong, and worth recording as a stale-memory
failure: MSVC 18 (2026), `C:/dev/XRT` and `C:/dev/vcpkg` were **already
installed**, and `src/build/` was **already configured** for
`Visual Studio 18 2026` — it had simply never been built past `ZERO_CHECK`, so
the absent artifacts looked like an absent toolchain.

The only genuinely missing piece was **Boost.Beast** (`server.hpp` includes
`boost/beast/core.hpp`); vcpkg had only the `program_options` subset.

```
$ ./vcpkg.exe install boost-beast:x64-windows      # 1.6 minutes, 22 packages
$ cd src/build && cmake --build . --config Release --target flm
$ ./flm.exe version
FLM v1.0.0
```

**One error, one install, and the whole "wait for AMD" framing dissolved.** A
memory that says "don't start this" deserves re-checking before it is trusted,
because the world moves underneath it.

### What was written

| file | |
|---|---|
| `include/models/granite/q4nx_host.hpp` | host q4nx reader — the layout from §5, so no dependency on the closed `q4_npu_eXpress.dll` |
| `include/models/granite/granite_npu.hpp`, `common/models/granite_npu.cpp` | the engine: RMSNorm, GQA attention with KV cache, half-split RoPE, SwiGLU, threaded bf16 matvec |
| `include/AutoModel/modeling_granite.hpp`, `common/AutoModel/modeling_granite.cpp` | the family — tokenizer, chat template, reasoning sampler defaults |
| `all_models.hpp`, `automodel.hpp` | enum, name map, dispatch |

Two decisions that paid for themselves:

* **The RoPE permutation is undone once at load time**, so the forward pass uses
  the plain half-split rotation — the exact arrangement `oracle_granite.py` was
  verified against, rather than a second interpretation to re-derive.
* **The attention scale is read from `config.json`**, not hardcoded. The
  converter writes the post-fold `head_dim**-0.5` there, so it is right for a
  folded build and would stay right for an unfolded one.

### It loads at granite's native geometry

```
[FLM]  granite (host engine): hidden 2560, layers 40, heads 40/8, head_dim 64, attn_scale 0.125
[FLM]  granite: dequantizing weights to bf16 (host)...
[FLM]  granite: weights ready (40 layers)
[FLM]  WebServer started on port 52625
```

**hidden 2560, head_dim 64 — no padding.** The whitelist that made §8 and §9
necessary lives inside `llama_npu.dll`; our engine simply does not have one.
7.48 GB resident for 3.66 B bf16 parameters, as predicted.

### A build trap worth keeping

The first link failed with exactly one unresolved symbol,
`Granite::Granite(xrt::device*)`. `src/CMakeLists.txt` uses
`file(GLOB SOURCES "common/*/*.cpp")`, which is evaluated at **configure** time —
new source files are invisible to an incremental build until `cmake .` re-runs.
The symptom is misleading: it looks like a missing definition in code you just
wrote, not a build-system staleness.

### It answers correctly

```
POST /v1/chat/completions  {"messages":[{"role":"user","content":"What is the capital of France?"}]}

'&>\n</think>\nThe capital of France is Paris.\n\n<|im_system>\n> The capital of France is Paris.\n```'
```

**The right answer, from an engine we wrote, at granite's native geometry.**
The `</think>` is granite-4.2 reasoning as it should.

### Speed: a self-inflicted 1.5x, recovered

| | prefill | decode |
|---|---:|---:|
| thread pool created **per matvec** | 0.65 tok/s | 0.64 tok/s |
| **persistent pool** | 0.93 tok/s | 0.96 tok/s |

A 40-layer step issues seven matvecs per layer plus an attention pass — ~320
parallel regions, each of which was creating and joining ~23 threads. The pool
is worth 1.5x on its own, and it is not the last of it: the bf16 dot product is
still scalar, and the engine reads all 7.3 GB of weights per token.

**A race I wrote and then removed rather than shipped.** The first pool did work
stealing off a shared counter. That is wrong here: a worker which has finished
the last part of one epoch can still be looping on the counter when the caller
returns and the *next* `run()` overwrites `parts_` / `chunk_` / `body_`. Fixed
parts — worker `id` owns part `id` — make each worker read the epoch's state
exactly once under the mutex it was published with, and since the parts are
equal-sized there was nothing to steal anyway.

### Output noise: chased down, and the engine is clean

Two early runs began with a stray token (`%`, then `&>`); a later one did not,
which looked like non-determinism.

First guess — `temperature: 0` dividing logits by zero — **was wrong**.
`temperature` does reach the sampler (`rest_handler.cpp:493`), and
`sampler_temp_apply` special-cases 0 as greedy rather than dividing by it
(`sampler.cpp:277`).

Settled by measurement instead: two byte-identical requests to the same server.

```
run1: 'We need to produce a concise answer: "Paris".\n</think>'
run2: 'We need to produce a concise answer: "Paris".\n</think>'
IDENTICAL
```

**The engine is deterministic.** The earlier variation was between *different
prompts* ("What is the capital of France?" vs "…? Answer briefly."), with the
`rep_penalty = 1.05` ring buffer carrying state between requests in one process
on top. Nothing to fix in the engine; the sampler defaults for this family are
worth revisiting separately, since a fresh conversation arguably should not
start with penalty history.

Worth noting how nearly this became a third wrong conclusion. Three data points
that looked like non-determinism were three different inputs, and the honest
test — *identical* input, twice — took two minutes and settled it.

## Where this leaves the port

Granite-4.2-3B cannot run on the stock llama engine at any xclbin combination.
Two ways forward:

1. **AMD-side, the clean fix.** Add 2560 to the whitelist and ship a
   llama-architecture `layer.xclbin` at `(2560, 8192)`. This task's value is
   that the ask is now exact rather than speculative.
2. **Pad hidden 2560 -> 3072.** 3072 is on the whitelist *and* `Llama-3.2-3B` is
   `(3072, 8192)` — llama architecture, and granite's intermediate is already
   8192, so a padded granite lands exactly on a shipped design. The padding is
   exact if the RMSNorm width term is corrected (`sqrt(2560/3072)` on the norm
   weights, `2560/3072` on `rms_norm_eps`), folded the same way the Granite
   multipliers already fold. Costs ~20% wasted arithmetic in the hidden
   dimension. Not attempted; the open sub-question is `head_dim` 64 against
   Llama-3.2-3B's 128 and whether a mixed xclbin donor is coherent.

## What is NOT done

* **The padding path is untried** — option 2 above.
* **No C++ was written for FastFlowLM**, and none was needed: the experiment ran
  under the existing `llama3` family. `FastFlowLM` is still at zero commits on
  `feat/granite-4.2`.
* **`layer.xclbin`'s keying was never exercised** — the gate fires first, so
  T54's analysis of which artifact binds remains inference, not measurement.
* **No independent-checkpoint diff.** `tools/verify_q4nx.py` diffs against the
  GGUF the conversion was fed, which catches quantization, tiling and fold
  faults but shares the llama.cpp conversion step with its input. An independent
  arm would need the 6.2 GB bf16 safetensors; deferred.
* **No C++ was written for FastFlowLM**, deliberately. The T54 experiment needs
  none: granite can be registered in `model_list.json` under the existing
  `"family": "llama3"` for a correctness answer. A proper `Granite : AutoModel`
  class is only worth writing once the runtime is known to accept the shape —
  it is wanted for the chat template, the think markers and sampler defaults
  (granite-4.2-3b is a thinking + tool-calling model, `enable_thinking` default
  true, `<|im_end|>` EOS, GPT2 BPE), none of which affect whether the array
  runs it.
* **No independent-checkpoint diff.** `tools/verify_q4nx.py` diffs against the
  GGUF the conversion was fed, which catches quantization, tiling and fold
  faults but shares the llama.cpp conversion step with its input. An
  independent arm would need the 6.2 GB bf16 safetensors; deferred unless the
  oracle looks ambiguous.
