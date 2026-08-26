# 0115 — T34 closed: arch=1 gets the HTTP endpoint, and it is the same endpoint

- **Date** 2026-08-26
- **Milestone** M13 (T34)
- **Status** done — T34 **ANSWERED**. `--serve` works on EmbeddingGemma, on
  both the NPU and the host-only path, through **one shared `serve_http()`**
  rather than a second implementation. Bit-identical to arch=1's own `--embed`
  over base64. BERT's endpoint re-verified unchanged, and both architectures'
  accuracy gates reproduce their recorded numbers exactly.

## Goal

Close [T34](../../research/CLOSED-THREADS.md#t34). Items 1 and 2 were already
answered ([`0075`](../0075-m13-arch1-measurement-harness/TASK.md) for MTEB,
[`0109`](../0109-fused-ratio-energy/TASK.md) for the CPU ratio and energy).
What remained was item 3 — the host side — and one line the thread called
"also unbuilt, and smaller":

> `--serve` still refuses on arch=1, so this model has no HTTP endpoint.

## Part 1 — item 3, re-measured, because two of its levers were spent

T34 filed item 3 with a single-lane breakdown: *array 48.9%, host attention
14.7%, C readback + bias 11.5%, RMSNorm 7.8%, bf16 convert 5.9%, GeGLU 4.5%*,
and named two levers — `--c-bf16` and host-attention blocking.

That was before [`0104`](../0104-adopt-bfp16-per-model/TASK.md) made bfp16 +
bf16-C gemma's shipped datapath and before
[`0108`](../0108-fuse-epilogue-bfp16/TASK.md) ported the epilogue fusion to it.
Re-measured on the shipped build, 520 texts, single lane, foreign `hw_context`
submissions identical before and after the run:

```
datapath   bfp16-emulated MMAC, C as bf16
embedded   520 texts in 3.78 s  ->  137.7 seq/s
breakdown  npu 1389 ms (in 95, dispatch 1212, out 82) | conv 237 | bias 477 |
           norm 481 | attn 777 | rope 87 | geglu 0 | tok 6  [480 dispatches]
```

| bucket | T34 as filed | now | what changed |
|---|---:|---:|---|
| array | 48.9% | **36.7%** | the array got faster; the host is now most of it |
| host attention | 14.7% | **20.6%** | untouched — now the **largest** host bucket |
| RMSNorm | 7.8% | **12.7%** | untouched — now the second largest |
| C readback | (in "11.5%") | **2.2%** (`out` 82 ms) | **`--c-bf16` is shipped: lever spent** |
| bf16 convert | 5.9% | 6.3% | unchanged |
| **GeGLU** | 4.5% | **0 ms** | **0108's fusion reached arch=1** |

So of item 3's two named levers, one is **collected** (the readback is 2.2% of
wall — there is nothing left in it) and one is **untouched** (host attention,
now the biggest single host bucket). GeGLU disappearing entirely is 0108's port
showing up where it was predicted to.

What is left is not an open question — it is two named, sized, unbuilt
optimisations: **host attention (20.6%)**, which
[T42](../../research/OPEN-THREADS.md#t42) now carries with a real price on it
after [`0113`](../0113-t40-seq512-close/TASK.md)/[`0114`](../0114-t38-pad-dimensions-probe/TASK.md);
and **gemma's four-norm RMSNorm chain (12.7%)**, which
[T37](../../research/CLOSED-THREADS.md#t37) explicitly left unfused and which
0108 left unfused for the same reason.

## Part 2 — the endpoint

**The finding that made this small.** arch=1 did not lack an HTTP endpoint
because anything about EmbeddingGemma is incompatible with one. It lacked one
because the endpoint was **written against the BERT encoder's type** — 266 lines
inline in the BERT path, reading `EmbedService` directly. The actual coupling
turned out to be **three members wide**:

```cpp
struct EmbedBackend {
  size_t vocab_size;
  std::string prefix_text;
  int64_t hidden, seq;
  std::function<std::vector<float>(const std::vector<std::string> &,
                                   int64_t *)> embed;
};
int serve_http(const EmbedBackend &, const std::string &model_id,
               int port, const std::string &bind_addr);
```

`serve_http()` is the *same* code the BERT path always ran, moved up above
`run_gemma_mode()` and handed those three things. There is deliberately **no
second implementation**: a duplicated endpoint would drift, and the OpenAI
response shape, the four 400 cases, the base64 arm and the `InputTooLong` → 400
mapping are exactly the things that must not.

**What was changed**

1. `serve_http()` + `EmbedBackend` extracted; the BERT call site now builds a
   backend from `svc` and calls it.
2. `GemmaNpuEncoder::encode_batch()` gained an optional `int64_t *tokens`,
   accumulating **real** texts' token counts (padding repeats excluded, which
   `n_real` already distinguished) — `usage.prompt_tokens` needs a real number.
3. `run_gemma_mode()`'s lane loop became `embed_texts(texts, tokens)`, a
   callable over an arbitrary batch. The body is unchanged apart from taking
   its input as a parameter. The file path now calls it too, so `--embed` and
   `--serve` cannot diverge.
4. The two `--serve` refusals removed; `serve <model> [port]` routes to
   `run_gemma_mode` like `embed <model> ...` does.
5. **The host-only (`--cpu`) path serves too.** This was not symmetry for its
   own sake: `serve_port` is parsed before that branch, so leaving it unhandled
   would have made `--serve --cpu` embed the placeholder sentence once and
   exit — a silent no-op, the exact failure shape traps 6b/6c/7c/7d are about.
   Its model id is `<model>-cpu`, not `-npu`: the id is what a client sees, and
   naming a host-only server after the array is the same silent mislabel the
   "path HOST-only" line exists to prevent.

## Commands

```powershell
# build (the MSVC env must come from vcvars64.bat -- tasks/0059's lesson)
cmd /c build_runtime.bat

# arch=1, flag form and subcommand form
.\runtime\build\npuembed.exe . --model embeddinggemma-300m --artifacts artifacts_gemma_bfp16 --serve 8422
.\runtime\build\npuembed.exe serve embeddinggemma-300m 8423 --artifacts artifacts_gemma_bfp16

# the gate: does the endpoint agree with arch=1's OWN --embed?
.\runtime\build\npuembed.exe . --model embeddinggemma-300m --artifacts artifacts_gemma_bfp16 --embed tasks\0115-t34-serve-arch1\in.txt tasks\0115-t34-serve-arch1\out_embed.f32 --threads 24
.\.venv-ref\Scripts\python.exe tasks\0115-t34-serve-arch1\crosscheck.py --port 8422

# regressions
.\.venv-ref\Scripts\python.exe tools\verify_endpoint.py --model all-MiniLM-L6-v2 --port 8424
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp16 --threads 16
.\.venv-ref\Scripts\python.exe tools\verify_gemma_npu_encode.py --npu <fresh> --cpu tasks\0074-m13-gemma-on-npu\out_cpu.f32
```

## Result

**arch=1 endpoint** (`crosscheck.py`, 8 texts):

```
model                    -> embeddinggemma-300m-npu
batch of 8               -> shape (8, 768), indices ordered: True
norms                    -> min 1.000000 max 1.000000
usage.prompt_tokens      -> 130
vs --embed               -> max abs diff 5.215e-08
base64 (decoded) vs float-> max abs diff 5.215e-08
base64 vs --embed        -> max abs diff 0.000e+00
  no input     -> HTTP 400
  empty list   -> HTTP 400
  bad format   -> HTTP 400
  token ids    -> HTTP 400
PASS
```

`/health` reports the resolved prefix, as the BERT path does:
`{"status":"ok","model":"embeddinggemma-300m-npu","backend":"amd-xdna2-npu","prefix":"title: none | text: "}`

**The 5.215e-08 is the JSON text form, not an encode difference**, and this is
worth stating because it looked like a failure first. The float arm serialises
with `%.7g` — 7 significant digits, a deliberate response-size choice in the
existing endpoint — so it can only ever agree to ~1e-7 relative. **base64
carries the raw fp32 and is bit-identical to `--embed` at 0.000e+00**, and
base64 is the arm the OpenAI client uses by default. My first gate demanded
`== 0.0` on the float arm and failed a working endpoint; the corrected gate
checks exactness where exactness exists.

**Regressions — all four reproduce recorded numbers exactly**

| check | result | against |
|---|---|---|
| BERT endpoint (`verify_endpoint.py`) | **PASS**, `vs --embed` **0.000e+00** | tasks/0037's own gate |
| MiniLM golden gate | rel_fro **2.430e-02**, 1-cos **3.406e-04** | 0104's recorded pair, to the digit |
| gemma differential gate | 1-cos mean **1.749e-04**, worst **2.315e-04** | 0104/0105's recorded pair, to the digit |
| gemma `serve` subcommand | `/health` OK | — |

The two accuracy gates reproducing bit-exactly is what says the `encode_batch`
and lane-loop edits did not touch the arithmetic.

**One consequence outside the runtime**: `tools/make_release.ps1` excluded
`artifacts_gemma_bfp16` from what a release ships, and the comment there names
the reason — *"arch=1 has no --serve/HTTP path yet"*. That reason is gone, so
gemma joins the list (645 KB, the same order as every other design set).

## Problems hit

1. **The build environment, not the build.** `cmake --build` from a plain shell
   fails with `Cannot open include file: 'algorithm'` — MSVC's `INCLUDE` is not
   set. This is [`0059`](../0059-m11-production-verify-post-migration/TASK.md)'s
   already-documented lesson; the fix is a `.bat` that calls `vcvars64.bat`
   first, invoked through the PowerShell tool. Written as `build_runtime.bat`
   so the next session does not rediscover it.
2. **A first `cmake --build` said "ninja: no work to do" and proved nothing.**
   It was only after a real compile that the environment problem appeared. A
   no-op build is not a passing build — 0059 says this too.
3. **String literals mangled by the patch script.** The generated `serve_http()`
   had real newlines inside `printf` literals where `\n` was intended, producing
   a wall of `C2001: newline in string literal`. Repaired by hand. The lesson is
   narrow and practical: generate C++ *code structure* with a script if you
   must, but write the string literals with an editor.
4. **My own gate was wrong before the code was.** See the 5.215e-08 above.

## Artifacts

* `crosscheck.py` — the arch=1 endpoint gate (the check `verify_endpoint.py`
  performs for BERT, in the form arch=1 needs: it drives `--embed` separately
  rather than assuming `artifacts_b128il`)
* `in.txt`, `out_embed.f32` — its inputs and the `--embed` reference
* `build_runtime.bat` (repo root) — the vcvars-wrapped build

## Next

T34 closes. What it leaves are two sized optimisations, not questions: host
attention (20.6% of gemma's wall clock) under
[T42](../../research/OPEN-THREADS.md#t42), and gemma's RMSNorm chain (12.7%),
which is [T37](../../research/CLOSED-THREADS.md#t37)'s explicitly-deferred
remainder.
