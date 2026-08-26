# 0082 — T37: the host's two multi-pass chains, fused

**Goal (user, 2026-08-22):** *"…deretter research på mer speed. Speed er ikke det viktigste med
prosjektet, men det er kanskje der vi har mest å hente i læring til andre og til andre
prosjekter senere."*

[`0081`](../0081-m13-int8-everywhere/TASK.md) §3 measured the array at **30.4%** of a bge-large
encode, so every remaining array lever is capped at 1.44× no matter how good it is. The other
69.6% is host work and almost all of it is **memory traffic, not arithmetic**. This is the
first cut at it, and it is the cheapest one available.

---

## 1. The FFN chain, and why it costs six passes

Between `ffn_up` and `ffn_down` the host does:

```
GEMM ffn_up ─→ dequantise C ─→ write `up` (fp32)     read C,  write up
            ─→ GELU in place                          read up, write up
            ─→ quantise for ffn_down                  read up, write int8
```

Three kernels, each a separate streaming pass, and `up` is the **widest tensor in the model**:
`rows × intermediate`, which at bge-large's batch 128 / seq 64 is **134 MB** (and 268 MB for a
gated FFN, whose `ffn_up` output is `2 × intermediate`). Per layer that is ~636 MB of host
memory traffic; over 24 layers, ~15 GB.

**None of it needs to exist.** A single row is at most 16 KB and fits L1. The dequantiser
already reads C row by row; the activation is elementwise; and the quantiser's per-row absmax
is a reduction over exactly the row that is already in cache. So the whole chain is:

```
read C row → dequantise+bias → activate → absmax → quantise → write int8
```

one read of C, one write of the int8 operand, ~100 MB per layer instead of ~636.

`up` is **never materialised**. Note this is the same argument that made a *per-row*
activation scale affordable in the first place — [`0079`](../0079-m13-int8-why-only-1.1x/TASK.md)
measured a static tensor-wide scale at 57× worse accuracy, and the reason per-row is cheap is
precisely that the reduction is over cache-resident data.

---

## 2. Bit-identical, and the first attempt was not

The point of a fusion is that it changes *when* work happens, not *what* the answer is. That
is testable, and it caught a real mistake.

**First attempt: scalar inner loops.** `1-cos` went 1.161e-03 → **1.180e-03**. Both pass the
gate, and it would have been easy to wave through as "within noise". It is not noise: the
unfused path computes the GELU polynomial with `_mm256_fmadd_ps`, which rounds **once**, while
`pl * u + c` in scalar C++ rounds **twice** under MSVC's default `/fp:precise`. Same formula,
different number.

**Fixed by using the identical intrinsics**, not merely the identical algebra — the same
`fmadd` chain for the polynomial, the same `_mm256_cvtps_epi32` for the quantiser's
round-to-nearest-even, the same `-120` argument floor in SwiGLU's `exp2`:

| model | unfused | fused |
|---|---:|---:|
| `all-MiniLM-L6-v2.int8` | 1.161e-03 | **1.161e-03** |
| `bge-base-en-v1.5.int8` | 1.778e-03 | **1.778e-03** |
| `nomic-embed-text-v1.5.int8` (gated) | 1.098e-03 | **1.098e-03** |

And the embeddings themselves are **byte-for-byte identical** (`cmp` on the `--embed` output),
which is the check that actually matters — a `1-cos` agreeing to four digits would not have
ruled out a per-row bug.

**The scalar-vs-FMA gap is the transferable lesson here**, and it is not specific to this
project: any "obviously equivalent" rewrite of vectorised float code is a different
computation unless the intrinsics match. It is why the activation is a **parameter** of the
fused helper rather than a branch inside it — the caller keeps ownership of the exact
instruction sequence its unfused twin uses.

---

## 3. Both FFN shapes fuse

* **Plain GELU** (arch 0: MiniLM, bge-small, bge-base, bge-large): row width `intermediate`
  in and out.
* **Gated SwiGLU** (arch 2: nomic): the row is `2 × intermediate` in and `intermediate` out —
  the activation *narrows in place*, combining `lo * silu(hi)` into the first half. This is
  the **bigger** saving of the two, because the tensor that is not materialised is twice as
  wide.

* **Gated GeGLU** (arch 1: EmbeddingGemma): `gelu(gate) * up`, the same shape of transform,
  through the same helper. It lives in a **separate encoder**, which makes it the case that
  proves the activation-as-parameter design paid — wiring it was a call site, not a third
  copy of the pass.

---

## 4. What it buys

**An indicative A/B taken under CPU load read 1.19×, and it was wrong by a third.** Both arms
were equally contended, so the ratio looked like a fair go/no-go — but contention compresses
exactly the term the fusion removes. On an idle machine, one session, `--threads 24
--pipeline 4 --bench 3`:

| model | unfused | fused | |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2.int8` | 1147.5 | **1599.6** | 1.394× |
| `bge-small-en-v1.5.int8` | 573.9 | **813.5** | 1.417× |
| `bge-base-en-v1.5.int8` | 275.8 | **391.2** | 1.418× |
| `bge-large-en-v1.5.int8n64` | 99.0 | **141.4** | 1.428× |
| **`nomic-embed-text-v1.5.int8`** (gated) | 210.9 | **312.6** | **1.482×** |

**nomic gains most, which is the traffic model's own ordering** — its gated `ffn_up` output is
`2 × intermediate`, so the tensor that stops being materialised is twice as wide. That the
prediction got the *ranking* right while missing the *magnitude* (§ below) is worth keeping
as a calibration on pass-counting.

Component breakdown, from the contended run (the shape is unaffected): `convert` 62.3 →
25.8 ms as `ffn_down`'s quantisation site disappears, `read out` 113.8 → 89.4 ms as
`ffn_up`'s fp32 write goes, and host GELU absorbed entirely into the pass that was already
reading the data.

> **CORRECTED by [`0084`](../0084-m13-host-isa-and-repeats/TASK.md).** Every arm above is
> **one run**. Re-run with three per arm the ratios hold to within 3% — 1.399 / 1.406 / 1.467 /
> 1.391 against the 1.394 / 1.418 / 1.482 / 1.428 printed here — because both arms drift
> together, which is what makes a same-session A/B survive being single-run. **The absolutes do
> not**: the fused figures here are ~4% low (MiniLM 1599.6 against a three-run 1672.0).
>
> And the spread claim below is wrong. It generalises from two bge-large readings, 141.4 and
> 135.7, one of which was taken under load. Within-arm spread on a quiet machine is **under
> 0.5%**.

**Run-to-run spread is ~4%** — bge-large int8 measured 141.4 and 135.7 on two runs — so these
are one-session figures, not tight ones. And [T18](../../research/OPEN-THREADS.md#t18) is a
larger caveat than that: `--probe-streams` and `--bench` disagree by **up to 10%** on array
time and nobody has explained why. Every figure here is `--bench`, so the A/B is a
within-instrument comparison and T18 explicitly says those are safe; the absolute numbers
carry T18's uncertainty. `tools/release_benchmark.ps1` (now carrying the int8
rows) is what produces release numbers.

### And against the bf16 baseline, same session

| model | bf16 | int8 + fused | total |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2` | 967.6 | 1599.6 | **1.65×** |
| `bge-small-en-v1.5` | 476.4 | 813.5 | **1.71×** |
| `bge-base-en-v1.5` | 201.7 | 391.2 | **1.94×** |
| `nomic-embed-text-v1.5` | 153.3 | 312.6 | **2.04×** |
| **`bge-large-en-v1.5`** | 53.2 | 141.4 | **2.66×** |
| `embeddinggemma-300m` ‡ | 128.6 | ~139 | **1.08×** |

‡ arch=1 has no `--bench`; measured by encoding the 520-text corpus, a different harness, and
two runs read 139.2 / 137.6.

Against [`0079`](../0079-m13-int8-why-only-1.1x/TASK.md)'s **1.10× on MiniLM and 1.44× on
bge-large**, which is where this line of work started three tasks ago. The ordering is the
same one every step has shown: **the bigger the model, the more there is to win**, because a
bigger model spends a larger share of its encode moving activations.


**And the traffic model turns out to be RIGHT — the constant was wrong.** Counting passes says
host traffic falls 3.45× on every BERT model and 4.29× on nomic. Priced at 25 GB/s that
over-predicted the win by ~2.4×, and the first draft of this task recorded it as the model
failing. It was not:

| model | traffic removed | time saved, measured | **implied bandwidth** |
|---|---:|---:|---:|
| MiniLM-L6 | 2.04 GB | 31.5 ms | 64.8 GB/s |
| bge-small | 4.08 GB | 65.7 ms | 62.1 GB/s |
| bge-base | 8.16 GB | 136.9 ms | 59.6 GB/s |
| nomic (gated) | 13.89 GB | 197.5 ms | 70.3 GB/s |
| bge-large | 21.74 GB | 387.7 ms | 56.1 GB/s |

**56–70 GB/s across five models spanning 10× in size** — a tight fit, not a coincidence. The
25 GB/s came from [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md), which measured what
reaches **the NPU through the shim**; that is the wrong constant for a host thread streaming
its own DRAM. Corrected, the pass-counting model predicts each model's saving to within ~15%.

So this task leaves a second number the project did not have: **the host side runs at
~60 GB/s**, against the ~28 GB/s [`0080`](../0080-m13-int8-traffic-bound/TASK.md) fitted for
the NPU path. Two buses, two constants, and using one for the other is what briefly made a
correct model look broken.

---

## 5. The LayerNorm chain, fused the same way

The second chain, which runs **twice per layer**:

```
add_into(x, y)        read y, read residual, write x
layer_norm(x)         read x, write x
memcpy(residual, x)   read x, write residual
quantise for next     read x, write int8
```

Eight streaming passes over a `rows × hidden` tensor — 33.5 MB at bge-large's batch 128, and
twice per layer against the FFN chain's once. Every kernel touches the same row, and a row is
4 KB, so all four share one L1-resident copy: **read y and residual once, write x, residual
and the int8 operand once.** Four passes instead of eight.

The consumers differ by site: after `ln1` it is `ffn_up`, after `ln2` it is the **next
layer's** `qkv` — which also makes the loop's own `memcpy(residual, x)` redundant, since the
fused pass has already written it. On the last layer there is no next GEMM (the output goes to
pooling), so `dst` is null there and the pass is add + norm + residual only.

**A real bug on the way, and it is worth recording because the failure was loud rather than
subtle.** `s_ln[...]` is a design *slot*; `h_gamma`/`h_beta` are indexed by *site*, and
`layer_norm()` converts between them with `slot - 1`. Passing the slot straight through
segfaulted. It could just as easily have read a valid neighbouring gamma and returned
plausible numbers — the off-by-one is the same shape as the T31 "one field to the left" class
this project keeps hitting. What caught it was that the byte-comparison harness was already in
place from §2, so there was no version of this that shipped quietly.

### Verified

| | |
|---|---|
| `1-cos`, fused vs `--no-fuse-ffn` | identical on MiniLM (1.161e-03), bge-base (1.778e-03), nomic (1.098e-03) |
| `--embed` output | **byte-for-byte identical** on all four (incl. Gemma, arch=1) |
| across lane counts | pipeline 1 vs 4 **byte-for-byte identical** |
| bf16 path untouched | MiniLM 1.086e-05, bge-base 1.353e-05, nomic 2.599e-05, bge-large 8.432e-06 — all unchanged |

### arch = 1 fused too, and its lanes had 0078's bug waiting

EmbeddingGemma's GeGLU (`gelu(gate) * up`) is the same shape of transform as SwiGLU, so the
same helper expresses it — the activation being a *parameter* is what made that a small
change rather than a third copy.

Wiring it surfaced a latent bug that had nothing to do with fusion: **`GemmaNpuEncoder`'s
extra pipeline lanes never copied the int8 scale vectors from lane 0.** This is precisely
[`0078`](../0078-m13-int8-accuracy/TASK.md)'s bug, reproduced in the encoder that was written
after it — lane 0 works, lanes 1+ have nothing. There it segfaulted; here it **threw**, with
the message 0078 added for exactly this:

```
error: int8 design but this encoder has no quantisation scales -- the container is
bf16, or it was packed before tools/pack_npue.py --int8 supported arch=1
```

A guard written for one encoder caught the same mistake in a different one two tasks later.
That is the argument for spending a line on a refusal instead of a comment.

## 6. What is still left

**Host attention, 14.6%** — nothing has touched it, and F3 priced attention folding at 1.4%
end-to-end in the **bf16 era**, when the host was a far smaller share of the encode. That
number deserves re-deriving before it is used again.

**arch = 1 (EmbeddingGemma)** uses a separate encoder and neither fusion is wired into it. Its
GeGLU is the same shape of transform (`gelu(lo) * hi`) so the helper already expresses it; the
RMSNorm chain differs (no beta, four norms per layer) and would need its own pass.

---

## 7. Commands

```powershell
# correctness: the fused and unfused paths must agree BYTE for byte
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 `
    --artifacts artifacts_int8c_mini --threads 24 --no-fuse-ffn --embed in.txt a.f32
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 `
    --artifacts artifacts_int8c_mini --threads 24 --embed in.txt b.f32
# cmp a.f32 b.f32   -> identical

# the A/B
.\runtime\build\npuembed.exe . --model bge-base-en-v1.5.int8 `
    --artifacts artifacts_int8c_base --threads 24 --bench 2 [--no-fuse-ffn]
```
