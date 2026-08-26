# Open threads

Every question this project has written down and not answered, in one place,
with a status.

**Why this exists.** [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) found
that the expert review's **§6b** had been *"deferred with cause"*, then
explicitly unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
and then sat untouched for two more tasks — because nothing re-reads a deferral
when its cause expires. [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md)
Part 3 found the same shape again: [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)
wrote down the *correct* hypothesis for a mystery and left it unchased for **28
tasks**.

A `TASK.md` is a diary entry — it is written once and never revisited. That is
the right property for a diary and the wrong one for an open question. This file
is the register that gets revisited.

**Rules.** A thread is added the moment a task says "untested", "not measured",
"deferred" or "open question". It leaves only by being **ANSWERED** (with a
pointer to where), **RETIRED** (with a reason), or **SUPERSEDED**. Nothing is
removed silently, and a stale "open" in an old task is not evidence that the
thread is still open — this file is.

Status: **OPEN** · **ANSWERED** · **RETIRED** · **BLOCKED**

---

## Live, ordered by what they would change

**Three threads, as of 2026-08-26.** The register ran to 20 threads and 765
lines when it was split on 2026-08-23; 2026-08-25/26 closed the rest, each
with a measurement or a build rather than a decision to stop caring. All
three that are left are design tasks with a price and a trigger, not
questions about the hardware — so a session that finds nothing to pick up here is reading
the file correctly.

---

<a id="t42"></a>
### T42 — Fold attention onto the array for LONG-sequence designs · **OPEN, filed 2026-08-25**
Successor to [T38](CLOSED-THREADS.md#t38), which removed the blocker, and to
[T40](CLOSED-THREADS.md#t40), which reversed the economics. Filed as a design
task with a price already on it, not as an open question about the hardware.

**What is no longer in doubt.**
[`0114`](../tasks/0114-t38-pad-dimensions-probe/TASK.md) built the probe T38
asked for: mem-tile `pad_dimensions` pads attention's 8-wide per-column slice to
16 **exactly, on all 8 columns**, the un-pad is an ordinary strided mem-tile
read, and a compute tile consumes the padded stream. Every check was bit-exact.
So `n = 16, cols = 8` is expressible and 0043's `cols ≤ 4` is a property of a
design that declines to pad, not of the hardware.

**Why the verdict reversed.** T38 inherited its cost/benefit from F3's
"attention is 2–5% of the work", and [`0113`](../tasks/0113-t40-seq512-close/TASK.md)
measured that this is a **seq-64 number**:

| | seq 64 | seq 256 | seq 512 |
|---|---:|---:|---:|
| array share of wall | **76.3%** | 62.0% | **46.6%** |
| host attention share of wall | 16.5% | 33.6% | **52.1%** |

At seq 64 the array is the bottleneck, so moving attention onto it is the wrong
direction and 0043's verdict was right. Above seq ≈ 470 the host is the
bottleneck and attention is more than half the wall clock.

**The starting price, a MODEL and labelled as one** (nomic, hidden 768,
intermediate 3072, gated; projections + FFN ≈ 9.4 M MACs/token, attention
`2·seq·hidden`):

| seq | attention as share of MACs | array if folded | host attention removed |
|---:|---:|---:|---:|
| 64 | ~1.0% | 1569 → ~1585 ms | −339 ms (host not the bottleneck) |
| 512 | **~7.7%** | 1537 → **~1656 ms** | **−1721 ms** |

which would put wall at roughly **1700–2000 ms against 3300**, i.e.
**1.65×–1.95×** — but prices **nothing** for softmax on the array, for the
per-head data movement, or for the design switch a separate attention shape
would reintroduce (note 0004: ~25 µs + 7.2 µs per lock).

**What would have to be built**: QK^T and A·V as array GEMMs at `n=16, cols=8`
with mem-tile padding, softmax either on the array or as a round trip, and the
per-head slicing. [note 0007](notes/0007-unused-iron-surface.md) §3.4's warning
stands unchanged — whisper-xdna built fused attention, it was correct, and it
still lost.

**Trigger**: only worth starting if long-sequence designs become something this
project actually ships. At seq 64 — every model in the catalogue today — the
answer remains no, and that is measured, not assumed.

---

<a id="t43"></a>
### T43 — A byte-level BPE tokenizer is the gate on every modern encoder · **OPEN, filed 2026-08-26**
Filed as a design task with a price and a trigger, not as a question about the
hardware. Discovered while pricing four candidate models (T44) during
[`0118`](../tasks/0118-prompt-name-per-request/TASK.md): none of them is blocked
by the array, and **all four are blocked by the same missing tokenizer.**

**What exists.** Two tokenizers ship, both hand-written, both with generated
tables rather than a runtime JSON parser (rule 5):

| | WordPiece | SentencePiece BPE |
|---|---|---|
| impl | `runtime/src/tokenizer.cpp` (388) | `runtime/src/tokenizer_gemma.cpp` (397) |
| tables | `bert_unicode_tables.hpp` (6009, generated) | `GEMATOK1` blob, 262k vocab + 515k merges |
| generator | `tools/gen_tokenizer_tables.py` | that **plus** a C++ port, `gemma_tokenizer_gen.cpp` (315) + `json_min.cpp` (323) |
| serves | arch 0 **and** arch 2 | arch 1 |

**What is missing.** Byte-level BPE — the GPT-2 / OLMo / tekken / Qwen family.
Granite-r2 uses OLMo's (the `|||IP_ADDRESS|||` and `|||EMAIL_ADDRESS|||` entries
in its vocab are the tell), Giga-480M has vocab 128256, Nemotron-1B has
Mistral's tekken at 131072, WeMM has Qwen's at 248078.

**The price, and why it is not a whole tokenizer.** Gemma's merge machinery
already applies BPE merges over a vocab, and that part is reusable. What is new
is the **pretokenizer**: a GPT-2 regex split and a byte↔unicode mapping, in place
of Metaspace. Note `gemma_tokenizer_gen.cpp:108` already *refuses*
`continuing_subword_prefix` / `end_of_word_suffix`, so the generator is
half-general and knows where its edges are. Estimate **400–600 lines** against
Gemma's ~1,100, plus a table generator, plus a C++ port of that generator
(`gemma_tokenizer_gen.hpp:13-23` records why: a fresh clone has to pack without
Python), plus a byte-exact HuggingFace verifier on the pattern of
`tools/verify_tokenizer_gemma.py`.

**Trigger**: the first model outside WordPiece/Gemma-SP being adopted. It is
listed as its own thread rather than as a line item inside T44 because the cost
is paid **once** and unlocks the ModernBERT, Qwen and Mistral families together
— which changes the ranking in T44, where it would otherwise be counted three
times.

---

<a id="t44"></a>
### T44 — Which encoder joins the catalogue next · **OPEN, filed 2026-08-26**
Four candidates were priced on 2026-08-26 at the user's request. None was
adopted, and none is blocked by the array. Recorded here because the geometry
work is real and would otherwise be re-derived.

**Measured against this project's own gates** — `set_model_shape()`'s asserts
(`runtime/src/main.cpp:574-586`), the `kMaxHeadVecs` ceiling (`:90`, `head_dim
<= 128`), and `hub.cpp:651-669`'s `tile_n` search:

| | granite-small-r2 | Giga-Emb-480M | Nemotron-3-1B | WeMM-2B |
|---|---|---|---|---|
| family | ModernBERT | Qwen3 bidirec | Ministral3 | Qwen3.5-VL |
| hidden / layers | 384 / 12 | 1024 / 28 | 2048 / 16 | 2048 / 24 |
| heads / kv / head_dim | 12 / — / 32 | 16 / 8 / 64 | 24 / 8 / 128 | 8 / 2 / 256 |
| `head_dim x heads == hidden` | ✓ | ✓ | ✗ 3072≠2048 | ✓ |
| `head_dim <= 128` | ✓ | ✓ | ✓ (exactly) | ✗ |
| bidirectional | ✓ | ✓ | ✓ `is_causal:false` | ✗ causal |
| qkv_n / `tile_n` | 1152 / **48** | 2048 / 32 | 5120 / 32 | — / 32 |
| GEMM weights bf16 | **57 MB** | 705 MB | 1.74 GB | ~2.9 GB + vision |
| pooling | cls ✓ | mean ✓ | mean ✓ | last-token ✗ |

Reference points from the shipping catalogue: bge-large is **604 MB** of GEMM
weights and the EmbeddingGemma container is already **1.05 GB**. Giga-480M is
therefore the size of something that already ships; only Nemotron and WeMM are
genuinely new territory on weight traffic (F2).

**All four need [T43](#t43).** That is the finding, not a footnote: the array
serves three of the four at a `tile_n` already in production, and what actually
stops them is the tokenizer.

**1. granite-embedding-small-english-r2 — cheapest.** Lands on bge-small's exact
geometry, but **not** on its design: ModernBERT's GeGLU doubles `ffn_up` to
N=3072 where bge-small's is 1536, so `design_fits()` will not match the
installed set. That is a new `export_gemm_rtp.py` run, not new code, and
`tile_n` 48 is legal so it compiles. Encoder work: pre-LN, bias-free, GeGLU, and
**two RoPE thetas** — global 80000 on every 3rd layer, local 10000 with a 128
window otherwise. The windowing is a host-side mask change, since attention
already runs on the host. English-only, so the MTEB gate applies with no new
quality machinery.
**UNVERIFIED**: its config carries `position_embedding_type:
"relative_key_query"`, which is almost certainly a vestigial `BertConfig` key —
ModernBERT uses RoPE unconditionally and both thetas are present — but this must
be read out of `modeling_modernbert.py` before anyone writes code, not assumed.

**2. Giga-Embeddings-instruct-480M — cheapest *encoder*.** Every ingredient
already exists in the tree: RMSNorm and q/k-norm from arch 1, RoPE and SwiGLU
from arch 2, and GQA sits between Gemma's MQA and nomic's MHA. Passes every
runtime assert. Being an *instruct* model it exercises
[`0118`](../tasks/0118-prompt-name-per-request/TASK.md)'s `prompt_name` directly
(`include_prompt: true` — the prompt tokens count in the mean pool). Needs its
`trust_remote_code` modeling file read to confirm it is Qwen3 minus the causal
mask.

**3. Nemotron-3-Embed-1B — one structural blocker.** `head_dim x heads` is 3072
against a hidden of 2048, so `attn_out` is (K=3072, N=2048) with K ≠ hidden —
which `tools/export_gemm_rtp.py` does not parameterise today. The only candidate
that needs a change to the geometry model rather than a new run of it.

**4. WeMM-Embedding-2B — DECLINED**, and the reasons are kept because they are
the valuable part. It is decoder-only
(`Qwen3_5ForConditionalGeneration`, causal, last-token pooling), its `head_dim`
of 256 is twice the runtime's ceiling, it carries ~2.9 GB of GEMM weights before
the vision tower, and a single 1024×1024 image at patch 16 with `spatial_merge
2` is 1024 vision tokens — where
[`0113`](../tasks/0113-t40-seq512-close/TASK.md) measured host attention passing
the **whole array** at seq ≈ 470, making [T42](#t42) mandatory rather than
optional. Rule 5 then requires Qwen byte-level BPE, JPEG/PNG decode,
smart-resize, patchify and 2D-RoPE, all in C++, in a repo with **zero** lines of
image handling anywhere. This project's own calibration: arch 2 was 3 files and
one task; arch 1 was ~10 files across two milestones. A vision tower is strictly
larger than arch 1, because arch 1 could still reuse the four-GEMM design, the
container's single geometry block and a text-only input API — and a multimodal
model reuses none of those three.

**Trigger**: pick one up when T43 is built, or when a specific retrieval need
names a model. Ranking is by cost, not by quality — no MTEB score for any of the
four has been measured here, and none should be quoted until one is.

---

## Closed

| thread | status | where |
|---|---|---|
| Is `aie::vector<float>` really IEEE fp32? | **ANSWERED** — yes, ~24 mantissa bits | [`0016`](../tasks/0016-m5-fp32-probe/TASK.md), refuting [`0015`](../tasks/0015-m5-gelu-polynomial/TASK.md) |
| What carries GELU's 3.886e-03 implementation error, if not fp32 precision? | **ANSWERED** — the default `floor` rounding mode | [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3, chasing [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)'s own hypothesis after 28 tasks |
| Can B reuse be expressed with `consumer_obj_type`? | **ANSWERED** — no; no spare DMA channel exists | [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) |
| Does cascade free the channels B-reuse needs? | **ANSWERED** — frees inputs 6/6→3/6, costs outputs 3/6→6/6 | [`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) |
| LayerNorm still opens three fifos per core | **ANSWERED** — params broadcast from the mem tile, 8 columns | [`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) |
| M6 speed not measured | **ANSWERED** — deliberately deferred to M7, then measured | [`0023`](../tasks/0023-m7-full-cpp-encode/TASK.md) onward |
| Is bge-small a byte-identical drop-in? | **ANSWERED** — no; 12 layers and CLS pooling are data, not constants | [`0039`](../tasks/0039-m9-bge-small/TASK.md) |
| `pack_npue.py` had not run in months | **ANSWERED** — broken import found and fixed | [`0036`](../tasks/0036-m8-tokenizer/TASK.md) |
| Is the centred polynomial basis worth 2.5×? | **RETIRED** — measured, worth nothing at fp32 | [note 0007](notes/0007-unused-iron-surface.md) §3.2 |
| `AIE_LOOP_UNROLL_FULL` | **RETIRED** — 14% slower on straight vector loops | [note 0007](notes/0007-unused-iron-surface.md) §1.9 |
| `burst_length` tuning | **RETIRED** — already maximal by default | [note 0007](notes/0007-unused-iron-surface.md) §2 |
| MTEB on the bf16-C datapath | **RETIRED** — datapath decided as bf16 in / fp32 out, 2026-08-19 | [`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) |
| `--emulate-bfp16` | **RETIRED** — fails accuracy; closed by the MTEB gate | [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) |
| Pre-tiling as a performance lever | **RETIRED** — a wash under isolation | [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md), [`0008`](../tasks/0008-m5-bfp16-real-data/TASK.md) |

---

## Closed threads

The 19 answered, retired and superseded threads moved to
[`CLOSED-THREADS.md`](CLOSED-THREADS.md) on 2026-08-23. They were **64% of this
file** — 1,242 lines against the 702 that were live — and rule 3 makes this file
"the authority on what is still open", which it can only be if a reader can
finish it.

They are kept in full, not summarised: rule 3b's point is that the refuted
claims and the measurements that killed them are the valuable part. Nothing was
deleted and every anchor still resolves.

* [T30](CLOSED-THREADS.md#t30) — The C-drain guard is half-wired, and every shipped model is one step from it · **ANSWERED 20
* [T31](CLOSED-THREADS.md#t31) — `design_fits()` matches on K alone, and nomic breaks it · **ANSWERED 2026-08-21 (filed and f
* [T33](CLOSED-THREADS.md#t33) — `l2_normalize` is written to the container and never read, and it costs nomic 4.5 MTEB point
* [T1](CLOSED-THREADS.md#t1) — What *is* the GEMM's 3 ms? · **ANSWERED 2026-08-19**
* [T2](CLOSED-THREADS.md#t2) — Claim B-reuse via cascade · **RETIRED 2026-08-19**
* [T29](CLOSED-THREADS.md#t29) — EmbeddingGemma-300M: does the tile_n=16 tax rule it out? · **ANSWERED 2026-08-22** · NO — an
* [T35](CLOSED-THREADS.md#t35) — `npuembeddings add <model> [<sha256>]` for finetunes · **BUILT 2026-08-22** ([`0076`](../tas
* [T36](CLOSED-THREADS.md#t36) — Was nomic's MTEB +0.09 measured symmetrically? · **ANSWERED 2026-08-22: YES, it stands**
* [T14](CLOSED-THREADS.md#t14) — Per-core: operand prep or accumulator dependency? · **ANSWERED 2026-08-19**
* [T16](CLOSED-THREADS.md#t16) — Why is a k-block iteration 4.3× the arithmetic in it? · **ANSWERED 2026-08-19**
* [T19](CLOSED-THREADS.md#t19) — Stationary-B single buffering, to make `k` bigger · **ANSWERED 2026-08-20: negative, measure
* [T20](CLOSED-THREADS.md#t20) — int8 · **ANSWERED 2026-08-22: BUILT, SHIPPED BEHIND A FLAG, BOTH GATES PASSED** ([`0077`](..
* [T22](CLOSED-THREADS.md#t22) — mlir-aie 1.4.1 · **ANSWERED 2026-08-20** · Python design surface migrated and independently 
* [T24](CLOSED-THREADS.md#t24) — 0048's iteration fit missed bge-base by 27% · **ANSWERED 2026-08-20**
* [T27](CLOSED-THREADS.md#t27) — The emulated datapath is traffic-bound: which byte-lever pays first? · **ANSWERED 2026-08-22
* [T37](CLOSED-THREADS.md#t37) — The host epilogue chain is four passes over one tensor · **BOTH CHAINS FUSED 2026-08-22** ([
* [T37](CLOSED-THREADS.md#t37) — T37 (original filing, kept for the reasoning) · 2026-08-22
* [T27](CLOSED-THREADS.md#t27) — T27 (original entry, kept for the reasoning) · conditional on T23
* [T25](CLOSED-THREADS.md#t25) — bge-base has no MTEB gate and no interleaved CPU ratio · **ANSWERED 2026-08-20**
* [T15](CLOSED-THREADS.md#t15) — `tasks/README.md` has no rows for 0038–0043 · **ANSWERED 2026-08-23** — and the debt was 21 rows, not six ([`0089`](../tasks/0089-t15-task-index-backfill/TASK.md))
* [T11](CLOSED-THREADS.md#t11) — Is the hw_context partition width settable at creation? · **RETIRED 2026-08-23** — a driver-ioctl field on Linux, unreachable from the XRT layer we call ([`0093`](../tasks/0093-t11-t12-t13-research/TASK.md))
* [T12](CLOSED-THREADS.md#t12) — Larger L2 megatiles · **ANSWERED 2026-08-23** — the paper's 5.9 → 13.7 TOPS is 2.0× reshape at constant volume and only 1.14× from size ([`0093`](../tasks/0093-t11-t12-t13-research/TASK.md))
* [T5](CLOSED-THREADS.md#t5) — Does DMA compression pay on real weights? · **ANSWERED 2026-08-23** — no: the AIE2P **shim DMA has no compression hardware**, and every byte the traffic model prices crosses it ([`0090`](../tasks/0090-t5-dma-compression/TASK.md))
* [T7](CLOSED-THREADS.md#t7) — `gelu_poly.cc` narrows through an emulated fp32 multiply · **ANSWERED 2026-08-23** — fixed, −11.0%, plus a third instance that was a dead multiply-by-one ([`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md))
* [T8](CLOSED-THREADS.md#t8) — `gelu_poly` degree 8 → 7 · **ANSWERED 2026-08-23** — −12.4%, and note 0007's error figures do not reproduce: the two degrees are 0.36% apart ([`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md))
* [T6](CLOSED-THREADS.md#t6) — Does `aie_stream=` free a DMA channel? · **ANSWERED 2026-08-23** — yes (core out 1/2 → 0/2), but only on a core endpoint, and the scarce resource is mem-tile *inputs* ([`0095`](../tasks/0095-t6-t10-probes/TASK.md))
* [T10](CLOSED-THREADS.md#t10) — `aie::exp2` never measured · **ANSWERED 2026-08-23** — one native `vexp2` and 2.6× cheaper, but 2.4× less accurate; `exp2_poly` stays ([`0095`](../tasks/0095-t6-t10-probes/TASK.md))
* [T32](CLOSED-THREADS.md#t32) — The golden gate is structurally blind to cross-row corruption · **ANSWERED 2026-08-23** — and it was also comparing 4 of 128 rows; both fixed, five known-bad artifacts ([`0094`](../tasks/0094-t32-golden-gate-rows/TASK.md))
* [T4](CLOSED-THREADS.md#t4) — Finish `0043` · **ANSWERED 2026-08-23** — attention's geometry costs the projections **2.229×** (1.289 tile × 1.729 columns); CLAUDE.md's verdict is now earned ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md))
* [T18](CLOSED-THREADS.md#t18) — probe vs bench disagree by 10% · **ANSWERED 2026-08-23** — it did not reproduce; both call the same code path, 0.2% apart ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md))
* [T21](CLOSED-THREADS.md#t21) — One tile geometry for four shapes · **ANSWERED 2026-08-23** — worth **4.3%**, only on narrow models, all in `ffn_up`; zero on bge-large ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md))
* [T39](CLOSED-THREADS.md#t39) — Nothing records which toolchain built an artifact · **ANSWERED 2026-08-25** — `toolchain.json` beside `design.json`, all six shipping designs re-exported ([`0106`](../tasks/0106-toolchain-provenance/TASK.md))
* [T9](CLOSED-THREADS.md#t9) — `xrt::runlist` · **RETIRED 2026-08-23** — its trigger ("after T3 or T4") fired when T4 closed negatively and pointed nowhere; worth 0.9%, never measured
* [T17](CLOSED-THREADS.md#t17) — Bigger L1 tiles · **ANSWERED 2026-08-23** — no on bf16 (≤1.29×), yes on int8 (**1.366×**, shipped); T21 had already answered the rest without either thread noticing ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md))
* [T23](CLOSED-THREADS.md#t23) — The GEMM datapath: bfp16 emulation · **ANSWERED 2026-08-23** — decided on MTEB evidence and **shipped per model**; five adopted, bge-small stays bf16 ([`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md))
* [T26](CLOSED-THREADS.md#t26) — Why is bfp16 + bf16-C 6.6× more accurate? · **ANSWERED 2026-08-23** — a `floor`-vs-`conv_even` leak the **1.3.4** matmul emitted no control for; 1.4.2 does, and the effect is gone ([`0099`](../tasks/0099-t26-rounding-ablation/TASK.md))
* [T13](CLOSED-THREADS.md#t13) — Explain the pre-tiled instability · **ANSWERED 2026-08-25** — real at production `M` and it does not dilute, but it is a **3.6% stall tail on a kernel that is otherwise 0.6% from row-major**, and untraced the pre-tiled path **wins** end to end (1.066× mean / 1.130× best) ([`0111`](../tasks/0111-t13-pretiled-stability-at-scale/TASK.md))
* [T40](CLOSED-THREADS.md#t40) — What does a long-sequence design actually cost? · **ANSWERED 2026-08-25** — 1.605× per token at seq 512, the array unmoved across an 8× range (−2.0%), and **host attention passes the array at seq ≈ 470**; short requests pay 5.8× ([`0112`](../tasks/0112-t40-seq256-nomic/TASK.md), [`0113`](../tasks/0113-t40-seq512-close/TASK.md))
* [T38](CLOSED-THREADS.md#t38) — Does mem-tile `pad_dimensions` make attention worth folding onto the array? · **ANSWERED 2026-08-25** — the mechanism **works, exactly, on all 8 columns**, so 0043's `cols ≤ 4` wall is removable; the value question is now sequence-dependent and its sign flips above seq ≈ 470 ([`0114`](../tasks/0114-t38-pad-dimensions-probe/TASK.md))
* [T34](CLOSED-THREADS.md#t34) — arch=1 on the array: MTEB, the ratios, the host side, and `--serve` · **ANSWERED 2026-08-26** — all four items done; the endpoint is now **one shared `serve_http()` for every architecture**, bit-identical to arch=1's own `--embed` ([`0115`](../tasks/0115-t34-serve-arch1/TASK.md))
* [T41](CLOSED-THREADS.md#t41) — The golden pipeline hardcodes the sequence length · **ANSWERED 2026-08-26** — all three makers take `--seq`; the default reproduces a committed golden bit-for-bit and the flag reproduces 0112's hand edit byte-for-byte ([`0116`](../tasks/0116-t41-golden-seq-flag/TASK.md))
* [T3](CLOSED-THREADS.md#t3) — Device-resident intermediates · **RETIRED 2026-08-26** — re-priced on the post-fusion build at **1.095×–1.160×** best case, below the 1.223×–1.265× [`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md) already banked; mechanism proven, value collected cheaper ([`0117`](../tasks/0117-t3-t28-repricing-retire/TASK.md))
* [T28](CLOSED-THREADS.md#t28) — True phase fusion (the pipelined relay) · **RETIRED 2026-08-26** — same measurement, same architecture; even an upper bound crediting the relay with the ENTIRE transport bucket is 1.242× on bge-large ([`0117`](../tasks/0117-t3-t28-repricing-retire/TASK.md))
