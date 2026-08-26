# 0076 — `npuembeddings add <repo> [<sha256>]`: run a finetune without recompiling

**Goal (user, 2026-08-22, "0.4.0 #2"):** *"lage en npuembeddings add <model> for å
legge til finetunes av de eksisterende modellene. Det vil ikke kreve noe annet
enn at vi flytter modellinfo inn i en json/yaml, og add kan hente ned andre
modeller fra huggingface."*

**Trust model, decided by the user mid-task:** *"add <model> <sha> — OM sha ikke
spesifiseres skal den lagres som tom og da skal den ikke valideres. MEN det skal
gis advarsel om at SHA er tom, både ved nedlasting og starting av modellen i
stdout."*

Status: **working end to end.** `TaylorAI/bge-micro-v2`, a 3-layer F16 finetune
this executable had never heard of, was added by name, fetched, packed and run
on the array at **1-cos 5.507e-06** against sentence-transformers.

---

## 1. Where the model catalogue lives, and where it does NOT

The user's framing was "move model info into a json/yaml". I did that for the
rows `add` writes and **deliberately not** for the six built-in rows.

`runtime/src/hub.cpp::table()` stays a C++ literal. Its six rows carry sha256
pins that this repository fetched once and validated against goldens — they are
the reason `serve bge-base-en-v1.5` is safe. A JSON file that could *override*
them would let one edit silently repoint `bge-base-en-v1.5` at other weights
while every table in the product still said "bge-base-en-v1.5". That is the
exact failure shape [`0051`](../0051-m9-bge-base-and-in-exe-fetch/TASK.md) had
in mind when it deleted `get-model.cmd`.

So: **`<root>/models/catalog.json` is a USER catalogue, merged AFTER the
built-ins, and may not shadow them.** `add` refuses a built-in name with a
message saying why; `load_user_catalog` also refuses (loudly, keeping the
built-in) if the file was hand-edited to do it anyway.

The user gets everything they asked for — add a model, no recompile — without
the built-in pins becoming editable.

---

## 2. Geometry is DERIVED, not matched

The existing catalogue cross-checks a download's `config.json` against the
row's hardcoded geometry (`verify_config`). For an arbitrary finetune there is
no row to check against, so `probe_repo()` inverts it:

1. Fetch **only** `config.json` and `1_Pooling/config.json` (a few KB, into a
   temp directory — `add` does not download weights).
2. Read `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
   `intermediate_size`, `model_type`, `num_key_value_heads`, `head_dim`.
3. **Compute the largest legal `tile_n`** from this project's own constraint —
   every `N` divisible by `tile_n · 8`, `K` divisible by 64, and
   `2·(64·64·2 + 64·t·2 + 64·t·4) < 64512` — trying 48, 32, 24, 16, 8 in order.
   If none is legal, refuse and say so: the array cannot serve that shape
   without a new design, and `add` does not build designs.
4. Read pooling from `1_Pooling/config.json` and refuse anything that is not
   `cls` or `mean` rather than approximating it.
5. Ask `pick_artifacts()` whether an installed design actually serves the
   result, and print the answer.

`model_type` selects the architecture exactly as both packers already do, so a
finetune of nomic or of Gemma routes correctly with no flag. MQA/GQA is
detected from `num_key_value_heads` and the fused qkv width is padded by the
same arithmetic 0074 introduced — **derived from this checkpoint**, never
copied from whatever it was finetuned from. A finetune that changed its head
layout would otherwise be handed a design built for the original, which is the
T31/0074 fail-open one more time.

---

## 3. The unpinned path, and what "loudly" means

`add <repo>` with no sha256 stores an empty pin. Nothing is verified. That is
allowed, and it is announced in **three** places:

| when | what |
|---|---|
| `add` | a five-line WARNING block, plus the exact `add <repo> <sha>` command to pin it later |
| every fetch | the digest actually received, and the same command to pin it |
| **every run** | `warn_if_unpinned()` — before any encode, on stdout |

The third is the one that matters and the one the user specified. A warning
seen once at install time is not a warning seen today, and the whole premise of
allowing an unpinned add is that the person running it *knows*. It is also the
only place in the fetch path that fails **open**, which is why it is this loud.

Two smaller honesty fixes fell out:

- The fetch banner said *"N MB of checkpoint, verified against a checksum built
  into this executable"* unconditionally. For an unpinned row that is simply
  false, and `download_mb` is 0 anyway. It now says which of the two this is.
- `CHECKPOINT.json` records the digest **actually received** for an unpinned
  row rather than the empty pin, so the container's `source_sha256` still
  identifies which bytes it was built from. The distinction between "verified
  against a pin" and "this is merely what arrived" lives in the catalogue,
  where a reader can act on it.

---

## 4. Two real bugs the first end-to-end run found

Both were found by running `add` against a **real** finetune rather than a
contrived one, and neither would have shown up on the six built-ins.

### 4a. Every built-in ships F32. HuggingFace mostly does not.

`TaylorAI/bge-micro-v2` ships **F16** safetensors, and the C++ packer read F32
only — a property of the six checkpoints this project happened to bring up, not
of the format. It refused, correctly, and `add` would have been a demo rather
than a feature.

`read_safetensors` now widens **F16 and BF16 to fp32 at read time**, once, so
no consumer below has to know. Both conversions are exact (fp32 has more
exponent range and more mantissa than either), so this cannot be the source of
any error measured downstream. The half-precision path handles subnormals and
the Inf/NaN exponent in their own arms rather than approximating them with the
fast path — getting either wrong is silent.

`Tensor` gained a `shared_ptr<vector<float>>` for the widened storage, so the
`data` pointer survives the copies and moves a `std::map` performs.

### 4b. The golden-fixture guard refused a correct run

`embed bge-micro-v2` then failed with *"the golden fixtures were made from
checkpoint 53aa5117… but this model is 792472b6…"*. The guard
([`0039`](../0039-m9-bge-small/TASK.md)) is right and stayed: never compare a
model against another model's answers.

The **fallback** was the problem. With no per-model fixture directory, the code
fell back to the flat pre-multi-model `runtime/artifacts/validation/`, which
holds MiniLM's vectors — so a model with no fixtures of its own would always
land on someone else's and then be refused.

A mismatch now means different things in the two locations, which is what it
always should have meant:

* **flat fallback** → these fixtures are not this model's; `have_val = false`,
  i.e. *this model has no fixtures*, and the encode proceeds.
* **the model's own directory** → the fixtures claim to be this model's and are
  stale. Still throws.

---

## 5. Commands run

```powershell
cmake --build runtime\build --config Release

.\runtime\build\npuembed.exe add TaylorAI/bge-micro-v2 --root .
.\runtime\build\npuembed.exe list --root .
.\runtime\build\npuembed.exe embed bge-micro-v2 tasks\0074-m13-gemma-on-npu\corpus.txt --root .
# and the accuracy check, against sentence-transformers on the same texts
```

---

## 6. Results

`add` reads the repository and reports before writing anything:

```
  repo       TaylorAI/bge-micro-v2
  name       bge-micro-v2
  geometry   hidden 384, 3 layers, 12 heads, ffn 1536
  pooling    mean
  tiling     tile_n 48
  design     .\runtime\artifacts_b128il
```

`list` then shows it as `available`; the first `embed` fetches, warns, packs and
runs:

```
  model      bge-micro-v2: 42 tensors, 58.26 MB, checkpoint 792472b64c3ca1c7
  shape      TaylorAI/bge-micro-v2: 3 layers, hidden 384, 12 heads x 32, ffn 1536, mean pooling
  weights    10.62 MB staged on the device once, not per call
  embedded   13 texts in 0.02 s  ->  652.0 seq/s
```

**Accuracy, against `sentence-transformers` on the same 13 texts at seq 64:
worst `1-cos` 5.507e-06, mean 4.736e-06** — the same band as the built-in
models (MiniLM 1.086e-05, bge-base 1.353e-05). Nothing about this model was
known to the executable an hour earlier, and no design was rebuilt: it reuses
`artifacts_b128il` because its geometry is bge-small's.

---

## 7. Problems hit

- **The two bugs in §4**, both invisible on the built-in six.
- `add`'s `download_mb` is 0 because nothing probes the file size before
  fetching. Cosmetic — the banner no longer claims a size for unpinned rows —
  but a `HEAD` on `model.safetensors` would fill it in.

---

## 8. Not done

- **No YAML.** JSON only, matching everything else in the container/design
  path, and it is what `json_min.hpp` already parses.
- **`add` does not build a design.** If no installed design serves the
  geometry, it says so at `add` time and `serve`/`embed` refuses later. That is
  the honest behaviour, but a finetune with an unusual `intermediate_size`
  will need `tools/export_gemm_rtp.py` run by hand.
- **No `remove`/`list --user`.** Editing `models/catalog.json` by hand works and
  is validated on load.
- **A finetune with ADDED TOKENS is untested.** It should only change the
  embedding table, not the GEMM designs, but that is reasoning rather than a
  measurement — filed on [T35](../../research/OPEN-THREADS.md#t35).
