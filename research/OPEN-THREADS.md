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

**Five threads, as of 2026-08-27, and falling: the 0.5.0 plan is working
through them.** It was three threads two days before; five of the seven filed
on the 27th came from one task —
[`0120`](../tasks/0120-roofline-analytic/TASK.md), the analytic roofline. That
is what a new instrument does: it does not answer questions so much as make
questions askable that previously had no coordinate system. Three of the
27th's batch have already closed: [T49](CLOSED-THREADS.md#t49) (the register
mechanism, [`0123`](../tasks/0123-register-hygiene/TASK.md) — which also filed
[T52](CLOSED-THREADS.md#t52), the Unigram tokenizer — **filed and ANSWERED the
same day**: built twice, byte-exact everywhere, shipping inside the 7th
catalogue model),
[T47](CLOSED-THREADS.md#t47) (the `--probe-streams` byte accounting, fixed and
audited, [`0124`](../tasks/0124-t47-t50-runtime-fixes/TASK.md)/
[`0125`](../tasks/0125-t47-gbs-audit/TASK.md)) and
[T50](CLOSED-THREADS.md#t50) (the fail-open `--cpu` and the stale artifacts
fallback, same rebuild).

T42/T43/T44 are still design tasks with a price and a trigger. **T48 is the
roofline's one remaining debt** — the lever it re-priced against a
retirement whose premise expired ([T48](#t48)), now measured at 1.72× array
and parked behind a funded C-join re-plumb. The other roofline debts are
paid: [T46](CLOSED-THREADS.md#t46) closed with the memtile leg measured
two-thirds idle ([`0139`](../tasks/0139-t46-memtile-leg/TASK.md)). The measurement it could not take is now taken:
[T45](CLOSED-THREADS.md#t45) closed with the roof **measured at 45.5 GB/s on
the hardware timebase**, wall clock agreeing to 0.4%
([`0128`](../tasks/0128-t45-m-sweep/TASK.md),
[`0130`](../tasks/0130-t45-traced-roof/TASK.md)).

[T51](#t51) came from
[`0122`](../tasks/0122-bge-large-short-input-tail/TASK.md) and is the most
consequential of the batch: **bge-large is 100× worse at the tail than at the
median on single-word inputs**, and the reason nobody knew is that all three of
this project's accuracy gates report a central tendency. It was found because a
user brought a reference vector from outside — the first check this project has
ever run with no shared code lineage with us at all.

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

> **UPDATED 2026-08-27
> ([`0123`](../tasks/0123-register-hygiene/TASK.md)), and the ranking changed:
> a fifth candidate needs no array work at all, and it was picked for 0.5.0.**
> `Alibaba-NLP/gte-multilingual-base` (`model_type: "new"`, the NewModel
> `trust_remote_code` impl): hidden 768 / 12 layers, 12 heads (head_dim 64),
> gated MLP with **GELU on the gate half in nomic's exact `fc11_up|fc12_gate`
> ordering** (`up_gate_proj` 768→6144 bias-free, `down_proj` 3072→768 with
> bias), post-LN, RoPE base 20000 with NTK scaling `{factor: 8.0}`, cls
> pooling, vocab 250,048, `unpad_inputs: false`. Its four GEMM shapes — qkv
> N=2304, attn_out 768, ffn_up 6144, ffn_down K=3072 — are a **literal match
> to `runtime/artifacts_nomic_bfp16/gemm_rtp/design.json`, `b_layout_hash`
> included**, with seq 256/512 sets already built. Zero `export_gemm_rtp.py`
> runs: the first candidate whose array cost is *nothing*.
>
> What blocks it is also not [T43](#t43): XLM-R's tokenizer is SentencePiece
> **Unigram** (Viterbi over log-probs), a third family distinct from both
> WordPiece and Gemma's SP-BPE — filed as [T52](CLOSED-THREADS.md#t52). Its forward pass is
> arch 2 with three deltas (GELU-for-SiLU on the gate, real biases, an
> NTK-scaled theta that must be read out of `new-impl/modeling.py`, not
> assumed — a wrong theta is silently wrong, tasks/0068). **The theta is now
> derived and probe-verified bit-for-bit
> ([`0134`](../tasks/0134-gte-oracle/TASK.md)): `inv_freq_i = 160000^(-i/32)
> / 8^(1/32)` — not expressible as a single theta — and the oracle
> `reference/encoder_gte.py` matches the repaired fp32 reference per-layer
> at 1e-06** (the loaded reference itself carried two landmines, recorded
> there — and [`0136`](../tasks/0136-gte-runtime/TASK.md) found a **third**:
> v5 also garbages `embeddings.position_ids`, so a sentence-transformers
> run of gte without the repair is *silently position-scrambled* when the
> garbage is in-bounds — measured 9.0e-02 before repair, 6.7e-08 after;
> every future MTEB/ST run of gte must carry
> `make_goldens_gte.py::repair_rotary`). **The model now encodes end to end
> on the NPU** ([`0135`](../tasks/0135-gte-container/TASK.md) container,
> [`0136`](../tasks/0136-gte-runtime/TASK.md) runtime): 1-cos vs the oracle
> **2.2e-04–3.7e-04** on the shipping nomic bfp16 design set, semantic gate
> 24/24, goldens 17/17. **The gates are run and everything PASSES
> ([`0137`](../tasks/0137-gte-gates/TASK.md))**: English MTEB (0103
> protocol, repaired fp32 baseline) mean **+0.06 / worst −0.06** — the
> cleanest verdict in the catalogue, with the clustering cell *positive*;
> multilingual STS17/STS22.v2 deltas +0.00/−0.02 with all 29 per-language
> subsets in [−0.30, +0.20] (recorded as the 0.6.0 gate's baseline,
> non-gating in 0.5.0 per the user's decision); tail p99 5.2e-04,
> max/median 2.2× — no bge-large-style tail, consistent with 0129's
> depth-12 expectation; semantic corpus v6 adds lang-gated no/de/es
> clusters (gte 30/30, the six English models bit-for-bit unchanged).
> **Adoption decision taken: bfp16, on the shipping nomic design set —
> and SHIPPED as the 7th catalogue model the same day
> ([`0138`](../tasks/0138-gte-hub-adoption/TASK.md))**: built-in row with
> the weights pin, bfp16 adoption record citing the 0137 gate, `kFilesGte`
> fetch set, `probe_repo` routing for `model_type: "new"`, and a C++
> `prepare_model_gte` whose container is **whole-file byte-identical** to
> the Python packer's — `embed gte-multilingual-base` with no
> `--artifacts` runs on the NPU and reports the adopted datapath. All
> seven models pass the semantic and tail gates. **This thread stays open
> for the question it asks — which encoder joins NEXT** — and the four
> byte-BPE candidates above remain priced and blocked on [T43](#t43)
> exactly as written.

**Trigger**: pick one up when T43 is built, or when a specific retrieval need
names a model — **gte-multilingual-base named itself for 0.5.0, gated on
[T52](CLOSED-THREADS.md#t52)**. Ranking is by cost, not by quality — no MTEB score for any of
the five has been measured here, and none should be quoted until one is.

---

<a id="t48"></a>
### T48 — B-reuse, re-priced on measured numbers: **1.72× array / 1.31× e2e on bge-large** — a priced design task awaiting a funded C-join re-plumb · **OPEN, filed 2026-08-27, gates closed 2026-08-27**
Successor to [T2](CLOSED-THREADS.md#t2) (RETIRED 2026-08-19) and
[T27](CLOSED-THREADS.md#t27) (ANSWERED 2026-08-22). Neither was wrong when it
was written; both premises have since been overturned by decisions this project
itself took.

**Premise 1, expired.** T2 was retired by
[T1](CLOSED-THREADS.md#t1)/[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)
on *"B-reuse removes bytes; bytes are not the constraint"*. That is a
**plain-bf16 statement** — 0048's own discriminating pair sits on the flat
compute roof, and [`0120`](../tasks/0120-roofline-analytic/TASK.md) §3a
re-derives it from geometry alone (1.50× apart on arithmetic intensity, 1.1%
apart on throughput). Since
[`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md), **five of six shipped
models are not on that roof.** Only `bge-small-en-v1.5` still is — and it is the
one model 0120 prices this lever at exactly 1.00× for.

**Premise 2, expired.** T27 is closed *"ANSWERED for int8, **which replaced
bfp16 as the fast datapath**"*, and its reasoning turns on the sentence *"bfp16
was never adopted (T23 is an accuracy decision nobody took)."* True on
2026-08-22. [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md) and
[`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) then took the decision, and
0.4.0 ships **bfp16 on five of six models with int8 behind a flag** — the exact
reverse. The levers T27 priced were priced on int8 traffic, whose composition
differs from bfp16's.

**Where the bytes actually are now**, from `design.json` alone
([`0120`](../tasks/0120-roofline-analytic/TASK.md) §2):

| model | A | **B** | C |
|---|---:|---:|---:|
| `minilm_bfp16` | 31% | **46%** | 23% |
| `base_bfp16` | 35% | **52%** | 13% |
| `gemma_bfp16` | 34% | **51%** | 16% |
| `nomic_bfp16` | 34% | **52%** | 14% |
| `large_bfp16` | 46% | **46%** | 9% |

C is down to 9–23% because the project already narrowed it
([`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md),
[`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)). **B is now the largest
single term on every dispatch of every model**, replicated ×32 by the row
blocks.

**The price** (byte ratio on the slanted roof, × 0109's measured array share):

| model | array | end to end |
|---|---:|---:|
| bge-base | 2.02× | **1.31×** |
| nomic | 2.00× | **1.34×** |
| bge-large | 1.80× | **1.34×** |
| MiniLM | 1.81× | **1.18×** |
| bge-small | 1.00× | 1.00× |

**What has NOT changed: the wall.**
[`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) and
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) established that the
blocker is DMA **channels**, not capacity — every core tile 2/2 in, five of eight
mem tiles 6/6 in, with the **C join spending the budget**. `repeat_count` is
unavailable on shim tiles, and `forward()`/`split()` do not expose
`consumer_obj_type`. Nothing above affects any of that. **This is a price, not a
plan.**

**Two things to check before anyone builds it.**

1. **Re-census the channels.** The C join is *cheaper* since C was narrowed to
   bf16, and 0046's census predates that. `tools/count_dma_channels.py` on the
   shipped `artifacts_*_bfp16` sets is the cheap first step and **needs no
   build**. That is why this thread is filed OPEN rather than BLOCKED.
2. **The pricing assumes pure proportionality to bytes**, which holds only to
   the left of the ridge. 0120 establishes that production sits 5–7× left of it
   (AI ≈ 120 against a ridge at 741) — but on a y-axis [T45](CLOSED-THREADS.md#t45) has not
   confirmed, and with a fixed cost the model does not carry. **Expect the real
   number to be below 1.80×.**

**Trigger**: item 1, which is one `python` invocation. Do not fund the build
before [T45](CLOSED-THREADS.md#t45) — now closed, so gate 2 is unblocked.

> **Gate 2 DONE — the price is measured and the build is NOT funded,
> 2026-08-27 ([`0131`](../tasks/0131-t48-gate-decision/TASK.md)).**
> Subtracting only the saved B bytes at the measured 45.5 GB/s roof from
> the measured 8-column dispatch times: **1.715× array / 1.310× e2e on
> bge-large** (at 0109's 56.7% array share). This thread's own prediction
> — "expect the real number to be below 1.80×" — was right. But
> [`0126`](../tasks/0126-t48-channel-recensus/TASK.md)'s census failed the
> wiring gate: no spare mem-tile input exists on the current topology, so
> a build is the cascade C-collapse (vectorised kernel unfunded, 0047's
> "not a session") or a hierarchical re-join, not a fifth fifo. **What
> remains here is a priced design task, not an open question**: the prize
> is 1.72×/1.31×, the first target is bge-large at k=64 (container format
> unchanged), and the trigger is someone funding the cascade microkernel
> or an equivalent re-plumb. Re-run 0131's arithmetic if the roof, the
> array share, or the dispatch times move.

> **Gate 1 DONE, and its conjecture refuted, 2026-08-27
> ([`0126`](../tasks/0126-t48-channel-recensus/TASK.md)).** The re-census on
> the shipped sets (all three geometry classes, cache dirs matched to the
> shipped xclbins by **0-byte** content diff) is bit-for-bit 0046's: every
> core tile 2/2 in, five of eight mem tiles 6/6 in, C join spending 4 of 6.
> Narrowing C to bf16 freed nothing because **a channel is an integer
> allocation — element width changes bytes per transfer, not channel
> count.** So the "expressible wiring" gate fails on the current topology:
> a build means re-plumbing the C join (0047's cascade collapse, whose
> vectorised kernel is the unfunded part) or a hierarchical re-join, not a
> fifth fifo. The remaining open item is gate 2: the re-price on T45's
> measured roof and fixed cost. Spare inputs exist only on (0,1)/(1,1)
> (one each) and (7,1) (two) — three tiles cannot broadcast to eight
> columns without transiting the five full ones.

---

<a id="t51"></a>
### T51 — bge-large has a 100× accuracy tail on single-word inputs, and every gate we own averages it away · **OPEN, filed 2026-08-27**
Found by [`0122`](../tasks/0122-bge-large-short-input-tail/TASK.md), which
started from something this project had never had: **a reference vector with no
shared code lineage with us at all.** The user obtained embeddings for the word
`"test"` from an online service. `bge-small` matched at the quoting floor;
`bge-large` did not, and the disagreement survived being re-measured against
full precision.

**The measurement.** 224 texts — 180 single words drawn deterministically from
the model's own vocabulary, 36 corpus sentences, 8 five-word phrases — against
fp32 `AutoModel` CLS:

| `bge-large-en-v1.5` (bfp16) | n | median | max | over the 2e-03 gate |
|---|---:|---:|---:|---:|
| sentences | 36 | 2.00e-04 | **2.47e-04** | 0 |
| 5-word phrases | 8 | 1.89e-04 | 2.60e-04 | 0 |
| **single words** | 180 | 2.27e-04 | **2.18e-02** | **5** |

`p99 6.63e-03 · max/median 100.2×`. Against `bge-base-en-v1.5` on the
**identical bfp16 datapath**: max 6.13e-04, max/median **3.6×**, zero
violations.

Worst five: `newsletter` 2.183e-02, `sermons` 1.195e-02, `contact` 7.940e-03,
`cinema` 2.227e-03, `messages` 2.222e-03. `test`, the input that started this,
is only sixth.

**Deterministic and not a harness artifact.** Bit-stable across batch
compositions (alone, with 1, with 36), and the five were reproduced to four
significant figures through `tools/verify_embed_e2e.py`, which uses
sentence-transformers rather than the `AutoModel` path — two different reference
implementations.

**Two mechanisms tested and REFUTED** (0122 §3), which is why this is filed as
an open question rather than a fix:

* **Cancellation** — ‖CLS‖ is 16.5–19.1 for every word and `test` has the
  **largest**; `corr(log‖CLS‖, log 1-cos) = +0.111` where the hypothesis needs
  ≈ −1.
* **Block-FP outliers** — bfp16 shares one exponent per block of 8, so an
  outlier channel costs its neighbours mantissa bits. Measured as the hardware
  sees it: the crest factor `max|x|/rms|x|` over aligned blocks of 8 is
  **1.879–1.909** across every word and `max|h|` is 17.6–18.0. Correlations
  −0.00 / −0.20 / −0.28.

**No input statistic predicts the error.** Whatever this is, it is not visible
in the activations going in.

**THE PART WITH THE LONGEST REACH IS THE METHODOLOGY, NOT THE NUMBER.** Three
gates pass bge-large and all three are blind for the same reason:

| gate | reads | why it misses |
|---|---|---|
| `1-cos` in `CURRENT_STATUS` | 2.626e-04 PASS | a mean over a small golden corpus of **sentences** — it is the median, and the tail is not in the corpus |
| MTEB delta | +0.13 / −0.01 PASS | averages thousands of pairs; the docs already say `1-cos` is *"sensitive to what MTEB averages away"* |
| the semantic gate ([`0121`](../tasks/0121-semantic-gate/TASK.md)) | 24/24 PASS | a ranking over 36 sentences, with no absolute error term by design |

Every instrument this project owns reports a central tendency. **None looks at a
tail**, and a tail is where a user meets a bad answer.

**What is not known**: the mechanism; whether depth (24 layers against
bge-base's 12), width (1024 against 768) or `tile_n = 32` (bge-large is the only
model not at 48) is the discriminator; and whether a 12° angle actually reorders
a retrieval result — that needs an MTEB *retrieval* task, which
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md) deliberately excluded.

**Three candidate next steps, cheapest first.**

1. **Which of the three differences matters.** bge-large at `tile_n = 48` is
   geometrically illegal (1024 mod 384 ≠ 0), but an int8 bge-large at
   `tile_n = 64` already ships ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md))
   — running the same 224 texts through it separates "tiling" from "depth and
   width" for the price of one sweep and no build.
2. **Layer-by-layer divergence** on `newsletter` against `sample`. Needs
   intermediate NPU activations, which only `arch = 1` exposes today
   (`tools/dump_gemma_kernel_vectors.py`), so this is real work.
3. **Make a gate that can see a tail.** A p99 over a few hundred varied inputs
   costs one encode per model and would have caught this on the day bge-large
   was adopted. Cheaper than either diagnosis and useful regardless of the
   outcome.

**Trigger**: step 3 belongs in the next release sweep whatever happens to steps
1 and 2. Step 1 whenever someone next has the array idle.

> **Step 1 DONE, 2026-08-27
> ([`0129`](../tasks/0129-t51-int8-tail/TASK.md)) — and it excludes two of
> the three candidates.** The same 224 texts through the shipping **int8
> bge-large at `tile_n = 64`** produce the **same worst words** as bfp16 at
> `tile_n = 32` (`newsletter`/`contact`/`sermons`/`cinema`/`messages`;
> log-Pearson **0.834**, top-10 overlap 8/10). So `tile_n` is exonerated,
> and the number format is exonerated *as the cause*: two arithmetically
> unrelated quantisation schemes rank the same inputs worst. The tail is a
> property of **(model × input)** — bge-large's forward pass amplifies any
> low-precision perturbation on these single-word inputs. Left standing vs
> bge-base: depth, width, or the learned representations themselves — step
> 2's question, now better-scoped (profile depth, not tiling). Step 3's
> gate must be **per-model, per-datapath, on a p99** — int8 bge-large's
> *median* already exceeds 2e-03 (its adoption rests on MTEB, 0085), so a
> max-gate says nothing there, while bfp16's p99/median of 30× is exactly
> what the averaging instruments miss.

> **Step 3 DONE — the release sweep has a gate that sees a tail, 2026-08-27
> ([`0132`](../tasks/0132-t51-tail-gate/TASK.md)).** `tools/verify_tail.py`
> (stdlib-only, `--exe`/`--root` for cold-zip runs) gates **p99 of per-text
> `1-cos`** over 0129's 224 texts against per-model baselines in
> `reference/tail/` (`ceiling = max(2e-3, 2×p99_measured)` — a ratchet;
> bge-large's wide ceiling carries an explicit T51 waiver note). It refuses
> to compare across datapaths, per 0129. All six models baselined and
> passing; bge-large reproduces 0122/0129 to the digit (same worst five);
> the negative control (tightened ceiling → FAIL, exit 1) is proven. Wired
> into `tools/release_benchmark.ps1` as a `tail` stage — and the seventh
> baseline, gte's, is recorded in
> [`0137`](../tasks/0137-gte-gates/TASK.md): p99 5.2e-04, max/median
> **2.2×**, no bge-large-style tail, the depth-12 expectation measured. Watch item: MiniLM's
> worst input sits at 1.85e-03, 8% under its ceiling — the likely first
> alarm if anything drifts. **What remains open in this thread is only the
> mechanism** (step 2: layer-wise divergence, now scoped to profile depth,
> not tiling) — the instrument debt is paid.

**Not blocking anything today** — for phrase-length text and longer, bge-large
is within 1.24× of its own median. But `newsletter` and `contact` are exactly
what a search box receives, and in retrieval the document side is sentences
while the **query** side is where one-word inputs live.

---

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
