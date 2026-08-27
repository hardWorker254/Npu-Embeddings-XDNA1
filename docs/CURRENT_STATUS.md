# CURRENT STATUS

*Last updated: 2026-08-27, after the 0.5.0 session (tasks/0123–0141):
`gte-multilingual-base` shipped as the **seventh model** on the nomic design
set with a new SentencePiece-Unigram tokenizer family and arch 3; the
bandwidth roof **measured at 45.5 GB/s on the hardware timebase**
([`0130`](../tasks/0130-t45-traced-roof/TASK.md)); a **p99 tail gate** joined
the sweep ([`0132`](../tasks/0132-t51-tail-gate/TASK.md)); and six threads
closed (T45/T46/T47/T49/T50/T52) with B-reuse re-priced and parked
([`0131`](../tasks/0131-t48-gate-decision/TASK.md)). The 0.5.0 sweep is
[`tasks/0141`](../tasks/0141-release-sweep-050/TASK.md).*

> **Six models, and a second architecture on the array.**
> [`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md)–[`0071`](../tasks/0071-m13-nomic-shippable/TASK.md)
> added **nomic-embed-text-v1.5** (Apache-2.0) as `arch=2`: RoPE instead of
> absolute positions, a gated SwiGLU FFN instead of GELU, and the first
> genuinely new architecture that runs **on the NPU** rather than host-only the
> way `arch=1` (EmbeddingGemma) does. It needs a task prompt, and the
> runtime prints which one it applied, because a wrong prompt is a quality
> regression no `1-cos` gate can see. (Since
> [`0118`](../tasks/0118-prompt-name-per-request/TASK.md) that prompt is a
> per-request `"prompt_name"` on the endpoint and a required `--prefix` on
> `embed`; there is no default on either.)
>
> Getting there found a **silent correctness bug** in the shipping design
> generator: above `N = 4096` a design compiled and returned wrong numbers, and
> bge-large's production `N = 4096` sits at *exactly* the threshold of a strict
> `>`, so the guarded path had never once executed. See §"Known walls".

> **The shipped CLI is now subcommands**: `npuembeddings list`,
> `npuembeddings serve <model>`, `npuembeddings embed <model> <file>`, and a
> bare invocation prints help plus the model table. Models are fetched and
> checksum-verified **inside the executable** (`runtime/src/hub.cpp`); the
> `get-model.cmd` batch script is gone, because `curl` plus a hardcoded
> `certutil` digest is the behavioural signature of a dropper and antivirus
> treated it as one. The flag form below is unchanged and still carries every
> probe and benchmark.
>
> **`serve` now covers every architecture, including EmbeddingGemma**
> ([`0115`](../tasks/0115-t34-serve-arch1/TASK.md)). arch=1 used to refuse the
> HTTP endpoint — not because anything about the model was incompatible with
> one, but because the endpoint was written against the BERT encoder's *type*.
> The coupling was three members wide (vocabulary size, the prompt whitelist,
> "embed these texts" -- it was the resolved *prefix text* until
> [`0118`](../tasks/0118-prompt-name-per-request/TASK.md) moved the choice to
> per-request), so there is now **one `serve_http()` that every architecture
> hands those to**, rather than a second implementation that could drift on the
> OpenAI response shape, the four 400 cases, base64, or `InputTooLong` → 400.
> Gated on agreeing with arch=1's own `--embed`: **bit-identical over base64**
> (the float arm's 5.215e-08 is the `%.7g` JSON text form, not an encode
> difference). The `--cpu` path serves too, under a `<model>-cpu` id — naming a
> host-only server after the array would be the mislabel the "path HOST-only"
> line exists to prevent. `artifacts_gemma_bfp16` therefore joins what
> `tools/make_release.ps1` ships.

> **An int8 datapath now exists alongside bf16, behind a flag**
> ([`0077`](../tasks/0077-m13-int8-gate/TASK.md)–[`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)).
> The array's MMAC unit takes `(8,8,8)` for int8 against bf16's `(4,8,8)`, and
> the int32 accumulator is **exact and order-independent** — integer addition
> associates, so unlike every float path in this project there is no rounding
> inside the reduction. Traced at **5.5–7.7×** on all four production shapes,
> bit-exact. It needs **SmoothQuant** (α = 0.5) as container data, because the
> usual fold into LayerNorm is unavailable: BERT is **post-LN**, so each LN
> output feeds the residual as well as the GEMM.
>
> **Two things it taught that outlast it.** First, **which cost model governs
> the GEMM depends on the datapath.** Under bf16 it is iteration-bound
> ([`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)); under int8 the
> arithmetic got cheap enough that
> [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)'s **traffic** model
> governs again (R² 0.987 against the iteration model's 0.568). So *"bytes are
> free"* is a bf16 statement, and the levers retired on it — B-reuse, the
> cascade milestone — are live again for int8. Second, **the accuracy cost of
> a design can often be measured before the design exists**: `--sim-c-bf16`
> rounds the accumulator in the host dequantiser exactly as a narrowed-C design
> would, and predicted `1-cos` 1.161e-03 — the number the built design then
> measured.
>
> (An earlier throughput table lived here. It is superseded by the measured one
> two blocks below, which was taken in a single session after the host fusions
> landed — keeping both would invite quoting the stale half.)
>
> **All six models now have an int8 container, and five pass the `1-cos`
> gate** ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md)):
>
> | model | arch | int8 `1-cos` | |
> |---|---|---:|---|
> | `all-MiniLM-L6-v2` | 0 | 1.161e-03 | PASS (MTEB mean −0.04) |
> | `bge-small-en-v1.5` | 0 | 6.385e-04 | PASS |
> | `bge-base-en-v1.5` | 0 | 1.778e-03 | PASS |
> | **`bge-large-en-v1.5`** | 0 | **2.968e-03** | **1-cos FAIL, MTEB PASS** |
> | `nomic-embed-text-v1.5` | 2 | 1.098e-03 | PASS |
> | `embeddinggemma-300m` | 1 | 1.902e-03 † | PASS |
>
> † differential against the bf16 NPU encode; arch=1 has no HuggingFace golden.
>
> The error tracks **width** more than depth — bge-small is 12 layers at hidden
> 384 and lands at 6.4e-04, bge-large is 24 at hidden 1024 and lands 4.6×
> worse. Wider weight matrices give SmoothQuant more outlier range to move, and
> moving it costs the activations.
>
> **bge-large misses the `1-cos` gate by 1.48× and PASSES MTEB at mean −0.05,
> worst −0.18** ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) §5). That
> is [`0035`](../tasks/0035-m8-mteb-gate/TASK.md)'s point for the third time:
> **`1-cos` is a fidelity check on the arithmetic, not a quality gate.** It is
> still the check that caught every real bug in 0077–0082 — a UTF-8 BOM on one
> row of 520, a double-applied task prefix, two harness bugs — precisely
> because it is sensitive to what MTEB averages away. The two measure different
> things and the project uses both.
>
> **Two things to know before using it.** `npuembeddings add` cannot produce an
> int8 container (the C++ packer is bf16-only; int8 needs the calibration pass
> that runs the numpy oracle), and nomic and Gemma each needed their **own**
> calibration oracle — their reference forward passes emit 5 and 7 GEMM call
> sites per layer against the packer's 4, because the packer fuses `fc11|fc12`
> and `q|k|v|pad`. An assert refuses rather than letting the factors shift.

> **The host, not the array, is now the encode**
> ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) §3,
> [`0082`](../tasks/0082-m13-fused-ffn-epilogue/TASK.md)). Measured on
> bge-large int8: **dispatch is 30.4% of the encode**, so an infinitely fast
> array caps the whole thing at 1.44×. The other 69.6% is host work and almost
> all of it is **memory traffic rather than arithmetic** — between two GEMMs
> the host walked the same tensor four to eight times (dequantise → activate or
> add-residual → normalise → quantise), each a separate streaming pass over
> 33.5–134 MB.
>
> Both chains are now **fused into one L1-resident pass per row**, and the
> intermediate tensors are never materialised. Verified **byte-for-byte
> identical** to the unfused path, and across lane counts — with a
> `--no-fuse-ffn` flag kept so the two can be A/B'd.
>
> **The 30.4% above is an int8, UNFUSED measurement, and both halves of that
> matter now.** It was taken before the fusion existed and before
> [`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) moved five of six models
> to bfp16, so **the 1.44× ceiling it implies is not the bound for the shipped
> configuration** — [`0107`](../tasks/0107-t3-t28-pricing/TASK.md) found the
> analogous bfp16 bound is *larger* (1.36–1.87×), because emulated MACs are
> slower than native int8 ones and therefore leave more array time to remove.
> Quoting 1.44× as "the ceiling" without naming the datapath is the mistake
> that entry exists to stop.
>
> **The fusion was int8-only until 2026-08-25**
> ([`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md)): it was gated on
> `a_elem_bytes == 1`, so it never fired on the datapath five of six models had
> just adopted. Ported to bf16/bfp16 it is worth **1.153×–1.340×** end to end,
> bit-identical again (embedding vectors hash the same fused vs unfused; every
> model reproduces its recorded `1-cos` to the digit). The measured-first
> finding worth keeping: the bf16 chain has the **same three-pass structure** as
> int8's minus the quantisation step, addressable share 17.7–28.3% — so the win
> is real but smaller than int8's 1.39–1.48×, exactly as the missing step
> predicts.
>
> **The transferable part is why the first attempt was not identical**: a
> scalar rewrite of the same algebra moved `1-cos` from 1.161e-03 to 1.180e-03,
> because `_mm256_fmadd_ps` rounds once where `a*b + c` rounds twice. Matching
> the *algebra* of vectorised float code is not enough; the intrinsics have to
> match too.

> **THE 0.5.0 NUMBERS** ([`0141`](../tasks/0141-release-sweep-050/TASK.md):
> one whole-catalogue sweep, one session, stages accuracy / throughput /
> interleaved / energy / **tail**; MTEB carried forward bit-identically —
> nomic verified by hash after the one change touching its path, gte from
> [`0137`](../tasks/0137-gte-gates/TASK.md)'s own symmetric session).
> End-to-end wall clock, **not an NPU kernel performance claim** (rule 1).
>
> | model | datapath (reported) | NPU seq/s | vs 0.4.0 | NPU/best CPU | J/1k better | worst `1-cos` | p99 tail | MTEB Δ |
> |---|---|---:|---:|---:|---:|---:|---:|---:|
> | all-MiniLM-L6-v2 | bfp16 | **1461** | 1.00× | 1.855× | 3.42× | 3.406e-04 | 9.7e-04 | +0.12 / −0.07 |
> | bge-small-en-v1.5 | bf16 | **630** | 1.00× | 1.589× | 2.64× | 8.348e-06 | 1.2e-05 | −0.10 / **−0.5010** |
> | bge-base-en-v1.5 | bfp16 | **324** | 1.00× | 2.713× | 4.52× | 2.284e-04 | 3.7e-04 | −0.06 / −0.19 |
> | bge-large-en-v1.5 | bfp16 | **95.2** | 1.00× | 2.607× | 4.89× | 2.626e-04 | **6.6e-03** (T51 waiver) | +0.13 / −0.01 |
> | nomic-embed-text-v1.5 | bfp16 | **262** | 1.00× | 3.494× | 6.55× | 1.402e-03 | 1.6e-03 | +0.01 / −0.25 |
> | embeddinggemma-300m | bfp16 | **181** | 1.01× | 1.271× | 3.72× (carried) | PASS (differential) | 4.2e-04 | +0.16 / −0.02 |
> | **gte-multilingual-base** | bfp16 | **255** | new | — (see below) | — | 4.608e-04 | 5.2e-04 | **+0.06 / −0.06** |
>
> The six carried-over models reproducing 0.4.0 to ±1% **is** the regression
> check: no array code changed for them. gte's CPU-ratio and energy cells are
> **deliberately absent** — both need a CPU reference arm, and gte's
> `trust_remote_code` model is unusable without the 0134/0136 buffer repairs
> (an unrepaired run is silently position-scrambled), so the NPU figure
> stands alone and labelled. gemma's energy is carried from 0.4.0: this
> sweep's gemma NPU energy arm produced a degenerate differential (Δt ≈ 0
> between low and high — recorded in 0141). Energy ratios move with the CPU
> side (nomic 4.01× → 6.55× between sweeps): indicative, not constants. The
> **p99 tail** column is the new instrument — per-model ceilings with an
> explicit ratchet; bge-large's tail is real and carried as a
> register-linked waiver, not rounded away.

> **THE 0.4.0 NUMBERS, ON THE DATAPATHS THAT ACTUALLY SHIP, FUSED (historical)**
> ([`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md) for throughput, after
> [`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md)'s adoption and
> [`0105`](../tasks/0105-release-sweep-adopted-datapaths/TASK.md)'s sweep; int8
> columns from [`0085`](../tasks/0085-m13-release-sweep/TASK.md)).
> **Throughput is the fused arm of 0108's A/B — three runs per model, idle
> machine, one session**, and it supersedes 0105's figures, which were taken on
> the unfused build. End-to-end wall clock, **not an NPU kernel performance
> claim** (rule 1).
>
> **The `datapath` column is read from the runtime's own status line**, not
> restated from a harness table — the point of 0104's guard. Every row's
> reported datapath matched its intended one. Since
> [`0106`](../tasks/0106-toolchain-provenance/TASK.md) the runtime also prints
> the **toolchain** that built each design, so a number can be traced to a build.
>
> | model | datapath | throughput | fusion ¹ | `1-cos` | NPU/CPU ⁸ | J/1k better ⁸ | MTEB mean / worst ³ | gate ³ |
> |---|---|---:|---:|---:|---:|---:|---:|---|
> | `all-MiniLM-L6-v2` | **bfp16** | **1168.8** | 1.265× | 3.406e-04 | 1.885× | 3.10× | +0.12 / −0.07 | PASS |
> | `bge-small-en-v1.5` | **bf16** | **421.0** | 1.208× | 8.348e-06 | 1.688× | 2.73× | −0.10 / **−0.5010** | **FAIL — why it stays** |
> | `bge-base-en-v1.5` | **bfp16** | **259.1** | 1.251× | 2.284e-04 | 2.734× | 4.17× | −0.06 / −0.19 | PASS |
> | **`bge-large-en-v1.5`** | **bfp16** | **75.4** | 1.223× | 2.626e-04 ⁹ | 2.691× | **4.79×** | +0.13 / −0.01 | PASS |
> | `nomic-embed-text-v1.5` | **bfp16** | **210.2** | **1.340×** | 1.402e-03 ⁵ | **3.553×** | 4.01× | +0.01 / −0.25 | PASS |
> | `embeddinggemma-300m` | **bfp16** | **167.1** ⁶ | 1.153× | 2.315e-04 ⁷ | 1.392× | 3.72× | +0.16 / −0.02 | PASS |
>
> *(int8 throughput and `1-cos` moved out of this table — they are 0085's
> unfused figures for a datapath nothing currently ships, and are listed two
> blocks above.)*
>
> ¹ against the same build with `--no-fuse-ffn`, same session — the flag is kept
> so the two arms stay comparable. ³ from
> [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md)'s gate table, the run that
> decided adoption per model; still valid because 0108 is bit-identical. Gate is
> `|mean| ≤ 0.5` **and** no task worse than −0.5. ⁵ thinnest margin of the five adopted, still ~1.4×
> inside the 2e-03 tolerance. ⁶ arch=1 has no `--bench`; corpus-encode harness,
> and its within-arm spread is **5.2%** against ≤1.1% for the BERT-family rows.
> ⁷ differential against the host-only path; arch=1 has no HuggingFace golden.
>
> ⁹ **THE `1-cos` COLUMN IS A CENTRAL TENDENCY, NOT AN UPPER BOUND**
> ([`0122`](../tasks/0122-bge-large-short-input-tail/TASK.md), 2026-08-27). Each
> figure is a mean over a small golden corpus **of sentences**, and at least one
> model has a tail those sentences cannot see: `bge-large-en-v1.5` measures
> **2.18e-04 at the median and 2.18e-02 at the worst** over 224 varied texts —
> a **100×** spread — with **5 of 224 over the 2e-03 tolerance**, every one of
> them a **single-word** input (`newsletter`, `sermons`, `contact`, `cinema`,
> `messages`). Sentences and phrases stay within 1.24× of the median, and
> `bge-base-en-v1.5` on the identical bfp16 datapath has a max/median of 3.6×
> and zero violations. Read these numbers as "what a sentence typically costs",
> never as "the worst this model does". → [T51](../research/OPEN-THREADS.md#t51)
>
> ⁸ [`0109`](../tasks/0109-fused-ratio-energy/TASK.md), on the fused build,
> steady state, `\Energy Meter` RAPL counters per
> [`0034`](../tasks/0034-m8-energy/TASK.md) and the interleaved protocol
> [`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) requires — same
> statistic on both sides, one session. **NPU idle throughout**; the CPU carried
> light background load (an IDE and a browser, mean 3.7–4.3%), milder than
> 0105's but present and flagged. Every energy ratio improved over 0105 except
> nomic (4.38× → 4.01×), which is still a large win.
>
> **Array time is unchanged by the fusion — measured, not argued.**
> [`0109`](../tasks/0109-fused-ratio-energy/TASK.md) re-took `--bench`'s
> `wait (hardware)` line on the fused build and it moved **−0.1% to +2.2%**
> against 0105: MiniLM 1,485 µs/dispatch (was 1,486), bge-small 3,025 (3,030),
> bge-base 4,636 (4,602), bge-large 9,825 (9,734), nomic 6,352 (6,218). The
> array does the same MMAC work whichever way the host reads its output, which
> is what this doc previously *argued* and now states as a measurement.
>
> **But the balance has inverted, and that is the headline of the fusion.**
> Array time held while host time fell, so the array's **share** of wall clock
> rose on every model — bge-large **46.4% → 56.7%**, MiniLM 26.7% → 33.4%,
> bge-base 36.2% → 46.7% — and with it the array-infinite ceiling: bge-large
> **1.87× → 2.31×**. So the host lever is largely spent and **the array is once
> again the larger remaining piece**, which is the opposite of what the block
> above this table said when it was written.
>
> **All rows pass MTEB**, on the datapath each one actually runs — five on
> bfp16 per [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md), bge-small on
> plain bf16, and every int8 container per
> [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md). 0085's finding that
> throughput spread is **under 0.6%** on five of six rows still stands and still
> retires 0082's "~4% run-to-run", which was generalised from one contended
> reading.
>
> **`embeddinggemma-300m` used to lose to torch on the CPU (0.88× in 0085) and
> now wins at 1.23×.** Read that as the datapath change, not as a reversal of
> the underlying argument, which is unchanged: arch=1 runs RMSNorm ×97, RoPE and
> MQA attention on the host and only four GEMMs per layer on the array, so the
> NPU does a smaller share of the work than in any other model — which is why it
> has the *lowest* ratio of the six even now. It stays in the table rather than
> a footnote for that reason.

A single place to answer: **what works, what does not, what was tried and
failed, where everything lives, and how to build and run it.**

This file is a snapshot and will go stale. The durable documents are
[`docs/`](00-overview.md) (how things work) and [`tasks/`](../tasks/README.md)
(what happened on a given day, including the failures). Where they disagree with
this file, they are right and this file is old.

---

## 1. Where the project is

**All six models run end to end on the NPU, in C++, with no Python in the
process, all validated against HuggingFace.** EmbeddingGemma was the exception
until [`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md): its MQA geometry really
does floor `tile_n` at 16, but **zero-padding the fused Q|K|V from 1280 to
1536** removes the floor exactly (zero columns of B give zero columns of C), so
it runs at the same `tile (64,64,48)` as everything else and packs to the same
`layout_hash`. M0-M8 are done: the tokenizer ships, the MTEB gate passes, and
energy is measured. M9 made the runtime model-driven; M12 added `arch=1`; M13
added `arch=2` and put `arch=1` on the array.

Throughput below is from the **whole-catalogue sweep**
([`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)) — one session, one
machine state, one protocol, `--threads 24 --pipeline 4`, mean of three runs,
contention guard on. It is **end-to-end throughput and not an NPU kernel
performance claim** (rule 1). Earlier tables here mixed sessions and lane
counts and were not comparable to themselves.

| `--model` | arch | hidden | layers | pooling | `rel_fro` vs HF | seq/s |
|---|---|---:|---:|---|---:|---:|
| `all-MiniLM-L6-v2` | 0 | 384 | 6 | mean | 4.473e-03 | **951** |
| `bge-small-en-v1.5` | 0 | 384 | 12 | CLS | 3.789e-03 | **494** |
| `bge-base-en-v1.5` | 0 | 768 | 12 | CLS | 4.297e-03 | **211** |
| `bge-large-en-v1.5` | 0 | 1024 | 24 | CLS | 3.763e-03 | **60.8** |
| `nomic-embed-text-v1.5` | **2** | 768 | 12 | mean | 6.119e-03 | **166** |
| `embeddinggemma-300m` | **1** | 768 | 24 | mean | 9.962e-06 `1-cos` † | **~133** ‡ |

† EmbeddingGemma has **no `rel_fro` against HuggingFace** because no golden
fixture exists for `arch=1`. Its gate is differential: the NPU encode against
the host-only path, which tasks/0064-0065 tied to `reference/encoder_gemma.py`
at `1-cos` 5.496e-13. ‡ and its throughput is from tasks/0074, **not** the
0073 sweep — a separate session and a separate protocol, so it is comparable
to itself and only loosely to the rows above.

**Interleaved against the CPU**, one session: 1.85× / 1.60× / 2.49× / 2.47× /
2.26×, and energy 3.65× / 3.00× / 3.33× / 3.28× / 3.75× better per sequence.
**EmbeddingGemma now has an MTEB score, and it is the best in the project:
the M8 gate PASSES at mean +0.00, worst task -0.00** (tasks/0075) — against
MiniLM +0.04, bge-small -0.03 and nomic +0.09. Note that on this model's
bridge the **C++ tokenizer is under test too**, since the NPU side tokenizes
in C++ while the CPU side uses HuggingFace; the agreement therefore covers
both. It still has **no CPU ratio and no energy figure** — both are unblocked
by the same harness work and simply not yet run.

**A caution that applies to every MTEB number above.** The first arch=1 run
FAILED at mean -1.88, and the failure was the *harness*: `mteb` injects a
task-appropriate prompt into a SentenceTransformer, and `run_mteb.py` was
prepending the container's prefix on top of it, so the two sides encoded
different strings. Fixed symmetrically. The four BERT models are unaffected
(their containers carry no prompts table for `mteb` to match), and **nomic's
+0.09 has now been checked and stands**: `mteb` passed `prompt=None` on all
five gate tasks, because nomic's sentence-transformers config carries only
`document`/`query`, which `mteb` uses solely for retrieval-style tasks
([T36](../research/OPEN-THREADS.md#t36)). Adding a Retrieval task to the gate
would re-open that question.
The caveats are in [`06-performance.md`](06-performance.md) and they matter:
the CPU side moves ~24% between sessions, so treat a ratio as indicative to
about ±20%.

**The number that matters is not the ratio.** Those NPU figures were taken with
torch and ONNX Runtime saturating all twelve cores; measured idle the same
models give 951 / 494 / 211 / 60.8 / 166 — **within 1.5% on every model.** The
work is genuinely off the CPU.

`nomic-embed-text-v1.5` was predicted at **150–160 seq/s** before it was
measured, from one argument — its gated `ffn_up` emits both halves, so it does
~1.33× bge-base's per-layer GEMM work at the same depth. It landed at **166**,
so the prior was **4% low**: 211 / 1.33 = 159 against a measured 166. A near
miss rather than a hit, and recorded as one. `bge-base-en-v1.5` remains **the
model whose geometry fits this NPU best**.

`bge-base-en-v1.5` was added in [`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)
and is **the model whose geometry fits this NPU best**: head_dim 64, and every
layer width a multiple of 384, so `tile_n` stays 48 where bge-large is forced
down to 32. It is also the most array-bound model we run -- 74.1% of wall
clock is NPU against MiniLM's ~40%.

`--model` is **required as soon as more than one container is installed** --
picking one silently is how this project's fail-open bugs have always looked.
Geometry, depth, pooling and tile size all come from the `.npue`; none of them
is a constant in the binary.

```
worst 1 - cos vs HuggingFace          1.086e-05  (all 128 rows, 4 distinct
                                                   sentences rotated across
                                                   tile copies)  (tolerance 2e-03)
```

**The gate compares every output row, not a sample of four** — fixed in
[`0094`](../tasks/0094-t32-golden-gate-rows/TASK.md), see §4 bug #5. Before
that fix it compared only the first 4 rows regardless of batch size (4 of
128 at batch 128), and even a loop over all rows is not sufficient by
itself: the 4-sentence golden batch is tiled into every 4-row copy, and a
**plain** tile repeat makes every copy's content identical, so a bug that
reads the *wrong* row (rather than corrupting content) is invisible even
when every row is compared. The fix tiles with a **per-copy rotation**
(`(k + r) % 4`, applied to the input, mask and expected output alike) so
every physical row's content is unique — proven necessary, not just
sufficient, by demonstrating both bug classes separately against known-bad
artifacts (0094 §"Known-bad verification").

**The product is complete and every claim is measured on hardware.** One
`.npue` (69 MB, weights + vocabulary) and one `npuembed.exe` take text in and
return embeddings, with no Python in the process
([`0036`](../tasks/0036-m8-tokenizer/TASK.md)):
faithful to fp32 (`1-cos` 1.086e-05), **downstream quality preserved**
(MTEB +0.04 points, [`0035`](../tasks/0035-m8-mteb-gate/TASK.md)), **faster
than the CPU** (interleaved: 877 vs torch's 489 and ONNX Runtime's 234,
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md))
and **1.94× lower energy per sequence** ([`0034`](../tasks/0034-m8-energy/TASK.md)),
on ~5.3 cores against twelve. The tokenizer shipped in
[`0036`](../tasks/0036-m8-tokenizer/TASK.md).

**And the advantage grows with width, shrinks with depth**
([`0042`](../tasks/0042-m9-bge-large/TASK.md)). Same width, twice the layers:
1.792 -> 1.533, because our cost is per dispatch and the CPU has no such term.
Width 384 -> 1024: **2.106x**, and **4.2x per core**. That is
[`0027`](../tasks/0027-m7-width-hypothesis/TASK.md)'s prediction, made from a
synthetic sweep, confirmed on real models.

**Attention still cannot share a design with the projections, and now we know
why structurally** ([`0043`](../tasks/0043-m9-attention-geometry/TASK.md)).
Attention is `[64,64] x [64,64]`, so `N % (n * cols) == 0` forces `n * cols` to
divide 64, while the AIE microkernel asserts `n % 16 == 0`. **Therefore
cols <= 4**: any design that can express attention uses at most half the array,
with an eighth of the output tile.

Since 0030 the architecture changed shape (tasks/0031–0032): the NPU is now a
**pure GEMM engine** — four shapes as instruction streams over **one xclbin in
one hw_context, zero design switches** — while LayerNorm, softmax and GELU run
on the host in fp32 (the same polynomials the NPU kernels used, minus the bf16
round trip). Every op that moved got FASTER and MORE ACCURATE; 1.086e-05 is
exactly what M6 predicted for a host-GELU pipeline.

### Throughput, honestly

Against `sentence-transformers` on the same machine's CPU (12 threads,
[`tasks/0018`](../tasks/0018-npu-vs-cpu/TASK.md)), **at matched batch**:

| config | NPU seq/s | NPU cores | CPU seq/s | wall ratio | per core | J/1000 seq |
|---|---|---|---|---|---|---|
| single lane, batch 128 | 604–618 | ~3.5 | 710.0 | 0.86× | ~3.0× better | 53.5 (1.59×) |
| **pipelined 2 lanes** | **833** | ~5.3 | 710.0 | **1.17×** | 2.7× better | **44.0 (1.94×)** |

> **Superseded by [`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md).** That
> table compares the CPU's **best-of-5** against the NPU's **mean**, and the two
> sides were measured minutes apart. Re-measured **interleaved**, same statistic
> on every side, steady state: **NPU 877.0, torch 489.4, ONNX Runtime 234.3
> seq/s — 1.792×.** The 710 is not reproducible today and the gap is not fully
> explained by the statistic (worth ~3.4%) or by background load (~6%); what is
> defensible is the interleaved ratio. The NPU is also far the more
> reproducible side: ±1.09× against the CPU's ±1.28× within a single run.

**Both halves of the project's own goal are now measured**: faster than the CPU
in wall clock, and **1.94× less energy per sequence** ([`0034`](../tasks/0034-m8-energy/TASK.md)).

**CPU wall parity is PASSED** (tasks/0033): two encodes pipelined over the one
unified design overlap host and NPU work 1.49×. (Older per-batch rows predate
0031–0033; re-measure small batches before citing them.)

Progress within M7: **42.4 → 833 seq/s, 19.6×** — and accuracy improved 31×
along the way (3.4e-04 → 1.086e-05).

---

## 2. What works

| | status | evidence |
|---|---|---|
| Native-Windows toolchain (IRON → Peano → aiecc → xclbin → NPU → trace) | ✅ | [`0002`](../tasks/0002-m1-hello-npu/TASK.md) |
| bf16 GEMM on the whole array, traced | ✅ | [`0003`](../tasks/0003-m2-bf16-gemm/TASK.md), [`0004`](../tasks/0004-m2-multicore-gemm/TASK.md) |
| numpy fp32 oracle matching HF at every layer (≤ 9.9e-07) | ✅ | [`0005`](../tasks/0005-m3-python-reference/TASK.md) |
| `.npue` pre-tiled weight container, bit-exact round trip | ✅ | [`0006`](../tasks/0006-m4-npue-pretiling/TASK.md) |
| All four per-layer GEMMs validated on hardware | ✅ | [`0011`](../tasks/0011-m5-first-op-validated/TASK.md), [`0012`](../tasks/0012-m5-all-layer-gemms/TASK.md) |
| Own GELU kernel (no transcendental call) | ✅ | [`0015`](../tasks/0015-m5-gelu-polynomial/TASK.md) |
| Own LayerNorm and softmax kernels | ✅ | [`0020`](../tasks/0020-m5-layernorm-kernel/TASK.md), [`0021`](../tasks/0021-m5-softmax-and-full-model/TASK.md) |
| Full encode in Python on the NPU | ✅ | [`0017`](../tasks/0017-m6-full-encode/TASK.md) |
| **Full encode in C++, no Python** | ✅ | [`0022`](../tasks/0022-m7-cpp-runtime/TASK.md), [`0023`](../tasks/0023-m7-full-cpp-encode/TASK.md) |
| Batching, arbitrary batch from one flag | ✅ | [`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md) |
| Validated cost models (dispatch, switch, width crossover) | ✅ | [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md), [`0024`](../tasks/0024-m7-dispatch-cost-anatomy/TASK.md) |

**On the NPU per encode:** 24 GEMMs, 6 GELU, 13 LayerNorm, 6 softmax = 49
dispatches.
**Still on the host:** embedding gather, attention's per-head GEMMs, bias adds,
residual adds, pooling, L2 normalise.

---

## 3. What does NOT work, or was not achieved

### Not achieved

- **CPU parity.** ~2.9× short at every batch. See §5 for why and what would be
  needed. [`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)
- **Attention on the array.** `head_dim = 32` fails the whole-array design's
  `M % (m·4) == 0`. Needs padding to 64 or folding two heads into one 64-deep
  tile. Worth ~4% of the encode now that the host version is vectorised.
- ~~**A tokenizer.**~~ **DONE** ([`0036`](../tasks/0036-m8-tokenizer/TASK.md)):
  WordPiece in C++, **6,826/6,826 texts byte-identical to HuggingFace**, with
  the Unicode tables generated from `unicodedata`. `npuembed --embed file.txt`
  takes plain text and returns vectors — one process, no Python. The
  vocabulary now lives inside the `.npue`, so the product is ONE file plus one
  executable.
- ~~**M8 / MTEB.**~~ **GATE PASSED** ([`0035`](../tasks/0035-m8-mteb-gate/TASK.md)):
  five tasks, mean delta **+0.04 points** against the CPU running the same
  checkpoint at the same sequence length, worst task −0.01. bf16 + fp32
  accumulate confirmed as production; `--emulate-bfp16` closed out for good.
- ~~**Energy measurement.**~~ **DONE** ([`0034`](../tasks/0034-m8-energy/TASK.md)):
  **1.94× better energy per sequence than the CPU** (44.0 vs 85.3 J per 1000
  sequences), measured with the Windows **`\Energy Meter`** RAPL counters — the
  earlier "no instances" finding was against `\Power Meter`, the wrong counter
  set. No external instrumentation needed. A control experiment proves the
  package meter sees the NPU block (≈5.2 W under saturation).

### Known walls (things that will not build)

| wall | message | status |
|---|---|---|
| LayerNorm / softmax above 2 columns | `no ShimNOCTile has sufficient DMA capacity` | **Fixable, and the constraint is documented.** A shimNOC DMA has **≤ 6 S2MM channels**, so a flat 32-way join is inexpressible — it must be joined *hierarchically through the mem tiles* (INDEX.md constants). That is exactly what `.split()`/`.join()` does, and it is why GELU now runs at 8 columns. LayerNorm still opens **three** fifos per core and softmax two; neither has been rewritten. [`0027`](../tasks/0027-m7-width-hypothesis/TASK.md) |
| ~~`hidden ≥ 1536` in the width sweep~~ | ~~`'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range`~~ | **CLOSED, and it was worse than a wall.** tasks/0030 fixed the stride by forcing `tb_n_rows = 1` above `m·n_aie_rows·N > 2^20`, so it *builds* — but the fill/drain walk kept stepping by a hardcoded `tb_max_n_rows // 2` and never read the guarded value, so above the threshold it **compiled and returned wrong numbers** (`rel_fro` 7.07e-01, 28/32 row-bands wrong). Never observed because bge-large's N=4096 sits at **exactly** 2^20 and the guard is a strict `>`, so it had never fired in a shipped design. Found and fixed in [`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §6/§6b when nomic's N=6144 crossed it. → [T30](../research/OPEN-THREADS.md) |
| 8-column design + core trace | `Unable to find a legal routing` | Known. Trace at 2 or 4 columns, throughput at 8 |
| B reuse in L2 | `no space for this BD` | ObjectFifo depth maps 1:1 to mem-tile BDs; ceiling is 6 tiles at 4 cols, 4 at 8, against a slice needing 24–48. Needs a core-side redesign. [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) |

### A bug this section used to call unresolved — it was diagnosed and fixed in 0030

- **`exp2_poly` works standalone (6.7e-03) but corrupts when composed into
  softmax** — row sums went to zero, output `[0,-120,0,-120,...]`, which is fp32
  being read as bf16 pairs. [`0021`](../tasks/0021-m5-softmax-and-full-model/TASK.md)
  reverted to `aie::exp2<bfloat16>` and left this "never diagnosed" — but
  [`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) §5a **did** diagnose and
  fix it, nine tasks later: the cause is the **worker stack** (`0xD00`, the same
  size the 4-chain GELU independently overran), not the fp32→bf16 narrowing. A
  2×2 (stack 0xD00/0x2000 × lib/poly) reproduces the corruption at 0xD00 and
  clears it at 0x2000; `exp2_poly` **is production softmax today**, at
  NPU-vs-golden 4.278e-03 (`softmax.cc`, `softmax_poly_bf16`). This paragraph was
  stale for 0072–0094 — the exact failure mode CLAUDE.md rule 3 exists to catch,
  found while answering [T10](../research/CLOSED-THREADS.md#t10) in
  [`0095`](../tasks/0095-t6-t10-probes/TASK.md).

---

## 4. Things that were tried and did NOT work

Kept because the negative results are load-bearing — several of them stopped us
optimising the wrong thing.

| tried | result |
|---|---|
| **Pre-tiling B as a performance lever** | **Refuted.** A wash (±9%), with 9–22% run-to-run spread against row-major's 2–3%. It optimises the L3→L2 access pattern, which is not the binding constraint. Kept anyway because it lets the runtime hand mapped bytes straight to DMA. [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) |
| **Tile order `k,n` vs `n,k`** | No difference (118.9 vs 118.5). The locality hypothesis was wrong |
| **`--emulate-bfp16` (worth 5.5× on GEMM compute)** | **+2.2% end to end, and it fails accuracy** (3.470e-03 vs the 2e-03 tolerance). That it buys so little *is* the diagnosis: GEMM compute is not the bottleneck |
| **GELU tile 1024 → 4096** (4× fewer DMA transactions) | **No change at all** (16.4 ms both ways). This is what first showed the cost was not in the data movement |
| **More resident contexts / fewer resident contexts** | No effect. 2 loaded ≡ 7 loaded (1509 vs 1519 µs) |
| **Spatial partitioning across processes** | Two processes scale 1.46× *regardless of design width* (1, 2, 4 columns all the same). Width-independence rules out spatial partitioning; it is host overlap |
| **Simulated bfp16 accuracy predictions from M3** | **Refuted on hardware.** M3 predicted real activations would be 6.0× worse than uniform for block FP; measured 0.85×. Refit on hardware data gave 7 bits/element, not 5 |
| **"Two designs per process corrupt the NPU"** | **Our own bug**, publicly retracted before filing upstream. `A.numpy()[:] = x` writes the host buffer only. [`0009`](../tasks/0009-m5-sync-misdiagnosis/TASK.md), [note 0003](../research/notes/0003-two-designs-per-process.md) |
| **"AIE `float` is not real fp32"** | Wrong inference, killed by a direct probe: full 24-bit mantissa on add and multiply. [`0016`](../tasks/0016-m5-fp32-probe/TASK.md) |
| **Ceiling claim in 0026** ("eltwise is at the machine's fp32 limit, no more optimisation") | **Overreached.** The degree probe bounds *one core*; compute-bound work scales with cores. Eltwise measures 1.98× per column doubling |
| **"The placer will not do 4 or 8 columns"** | **Misattributed.** The failing frame was `ln_array`, not `gelu_array`. GELU runs at 8 columns: 3.87× alone, 1.13× end to end |
| **Parity at h ≈ 1300–2000** | **Withdrawn.** It assumed CPU time grows 4× per doubling, but the CPU runs the same encoder and has the same 1:3h structure, so its total also grows sub-quadratically |

### Five bugs that "failed open"

A recurring class worth its own list: a check that could not tell, and was read
as "fine".

1. **[`0022`](../tasks/0022-m7-cpp-runtime/TASK.md)** — mtime picked the same
   xclbin for all four designs. *A buffer-size check catches a wrong size, never
   a wrong layout* (`rel_fro` 1.186).
2. **[`0024`](../tasks/0024-m7-dispatch-cost-anatomy/TASK.md)** — column counter
   read `aie.mlir`, which is **pre-placement and has no tile coordinates**, so
   it returned 0 for every design.
3. **[`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md)** — cache matcher
   pinned symbol and width but not **size**, so a batch-4 export got the
   batch-16 GELU. `1-cos` 8.651e-02.
4. **[`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)** — `std::max(0.0, NaN)`
   returns `0.0`, so a NaN-producing kernel scored a **perfect 0.000e+00 and
   PASSED**. *A tolerance test whose failure mode is a perfect score is not a
   test.*
5. **[`0094`](../tasks/0094-t32-golden-gate-rows/TASK.md)** — the golden
   gate's comparison loop read `for (b = 0; b < kGoldenBatch; ++b)`, where
   `kGoldenBatch` is the constant 4, regardless of the design's actual batch.
   At batch 128 it silently compared 4 of 128 output rows and never read
   the other 124 — **a corruption bug in any row past index 3 was invisible
   by construction**, no matter how wrong the arithmetic. A SECOND hole sat
   alongside it (filed as T32, 2026-08-21): the four golden sequences were
   tiled identically into every copy, so even a loop that checked every row
   could not distinguish "the right row's data" from "an identical copy of
   the wrong row's data" — the class of bug 0070's threaded `swiglu_cpu()`
   race actually shipped, invisibly, until a separate distinct-text e2e tool
   caught it by accident. Both are fixed: the loop runs over all `batch`
   rows, and the tiling rotates which of the 4 golden rows lands in each
   copy, verified against five deliberately reintroduced known-bad
   artifacts — including one that shows the all-rows fix **alone** does not
   catch the second bug class (a same-slot row swap across tile copies
   passes under plain tiling even with every row compared, and is only
   caught once the rotation is added too).

All five are now fail-closed, and the fixes were verified against
**known-bad** artifacts, not only good ones.

---

## 5. Why CPU parity was not reached

At batch 128, `dispatch + wait` is **347 ms of a 509 ms encode**:

| | per call | calls | total |
|---|---|---|---|
| GELU | ~19 ms | 6 | **114 ms** |
| LayerNorm | ~4.2 ms | 13 | 55 ms |
| softmax | ~6.6 ms | 6 | 40 ms |
| 4 GEMMs | — | 24 | 77 ms |
| design switches | — | 49 | ~60 ms |

**Elementwise is 209 of 347 ms**, and it is compute bound, not data bound: a
passthrough moving the *same* bytes over the *same* columns takes **626 µs**
against GELU's **9703 µs** at batch 64.

Two structural findings explain the rest:

**(a) Switching design is the most expensive single operation in the stack.**
~25 µs + 7.2 µs per `aie.lock` — 89 µs for a trivial passthrough, 2.4 ms for an
8-column GEMM, 10–17× a dispatch. It is not reconfiguration-by-difference (the
*same* xclbin in two contexts costs the same), not residency, not eviction
(`Suspensions = 0` throughout). **Data movement and switching are one budget:
every descriptor added to feed the array better is paid again at each switch.**
[note 0004](../research/notes/0004-context-switch-cost.md)

**(b) MiniLM is structurally the wrong shape for this machine.** Per layer and
token: **4h GELU elements against 12h² MACs — a 1:3h ratio**, so the elementwise
share falls as **1/h** regardless of implementation. Measured: GEMM grows 3.72×
and GELU 1.96× when h goes 384 → 768 (predicted 4× and 2×), and the elementwise
share drops **29.7% → 18.1%**. At h = 384 we are near the worst case.

**No parity-h number is claimed.** The crossover lies somewhere inside the range
of widths people use, above h = 384; where exactly is unmeasured, and the GEMM
rate gain is nearly spent (0.639 → 0.688 TFLOP/s, already 72–88% of the
2-column-normalised ceiling).

---

## 6. Where everything lives

```
NpuEmbeddings/
├── CLAUDE.md                 ground rules, traps, current state (read first)
├── docs/                     durable truth — how things work
│   ├── 00-overview.md        start here
│   ├── 01-hardware/          XDNA2 array, limits, bandwidth
│   ├── 02-toolchain/         IRON, Peano, aiecc, XRT on native Windows
│   ├── 03-kernels/           kernel authoring
│   ├── 04-model/             MiniLM analysis, .npue format spec
│   ├── 05-measurement/       the measurement doctrine
│   └── CURRENT_STATUS.md     this file
├── tasks/                    what happened, day by day, failures included
│   └── 0001…0027/TASK.md     27 tasks; README.md is the index
├── research/notes/           0001 kernel pitfalls, 0002 arch fallback,
│                             0003 the retracted sync claim, 0004 switch cost,
│                             0005 the expert-review scoreboard
├── reference/                the ORACLE
│   ├── encoder.py            numpy fp32 MiniLM with swappable hooks
│   ├── encode_npu.py         the Python NPU path (M6)
│   ├── make_goldens.py       generates goldens/ from HuggingFace
│   └── goldens/              per-layer taps, pinned to a checkpoint sha256
├── experiments/
│   ├── m1-hello-npu/         SAXPY, first traced kernel
│   ├── m2-bf16-gemm/         GEMM bring-up
│   ├── m5-pretiled-gemm/     gemm_pretiled.py — THE production GEMM design
│   ├── m5-eltwise/           gelu_kernel.py, layernorm_kernel.py,
│   │   └── kernels/*.cc      softmax_kernel.py + the AIE C++ kernels
│   ├── m7-switch-cost/       the passthrough control design
│   └── m8-npu-vs-cpu/        the CPU baseline
├── tools/                    BUILD-TIME Python (never at runtime)
│   ├── pack_npue.py          HuggingFace → .npue
│   ├── npue.py               reader/writer + gemm_b_layout()/layout_hash()
│   ├── verify_npue.py        round-trip check
│   ├── export_xclbin.py      IRON designs → runtime/artifacts*/
│   └── export_validation.py  golden check vectors → artifacts/validation/
├── runtime/                  THE PRODUCT — C++ + XRT, no Python
│   ├── CMakeLists.txt
│   ├── include/npue.hpp      mmap reader for .npue
│   ├── include/npu_device.hpp  Design: one xclbin, staged buffers, dispatch
│   └── src/main.cpp          the encoder, the benchmark, the probes
├── models/                   checkpoint + all-MiniLM-L6-v2.npue (68.77 MB)
```

**Everything under `runtime/artifacts*/` is a build artifact and is
gitignored.** Regenerate it; never commit it.

---

## 7. How to build the whole thing

### Prerequisites (already installed — verify, do not reinstall)

| | |
|---|---|
| IRON | `C:\dev\mlir-aie` — mlir-aie **1.4.2.dev16+g7e00b57**, Peano 21.0.0.2026080301, Python 3.13.15 |
| XRT | `C:\Xilinx\XRT` (2.21.0) — **not** under Program Files |
| `xrt-smi` | `C:\Windows\System32\AMD\xrt-smi.exe` (not on PATH) |
| MSVC | VS Community 2026, toolset 14.51 |
| Boost | `C:/dev/boost_1_88_0` (fallback only) |

> **`XILINX_XRT` must stay unset** — it poisons Windows builds. Use `XRT_ROOT`.

### Step 0 — environment (every shell)

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1          # MUST be dot-sourced
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
```

### Step 1 — weights and goldens (once)

```powershell
python reference\fetch_model.py        # HuggingFace checkpoint -> models/
python reference\make_goldens.py       # per-layer goldens from HF
python tools\pack_npue.py              # -> models/all-MiniLM-L6-v2.npue
python tools\verify_npue.py            # bit-exact round trip
python tools\export_validation.py      # -> runtime/artifacts/validation/
```

### Step 2 — compile the NPU designs

One command produces all seven xclbins. Pick width and batch deliberately —
they interact (§8):

```powershell
# the current best full-encode configuration
python tools\export_xclbin.py --cols 8 --elt-cols 2 --batch 128 `
                              --out runtime\artifacts_b128

# small batch: narrow GEMMs win, because switching dominates
python tools\export_xclbin.py --cols 2 --elt-cols 1 --batch 4 `
                              --out runtime\artifacts_b4
```

Useful flags: `--gelu-tile {1024,4096}`, `--gelu-variant {poly,probe2}`,
`--hidden H` (width sweep), `--emulate-bfp16` (faster, fails accuracy).

> The validation vectors live in `runtime/artifacts/validation/` and are read
> from there regardless of `--artifacts`. If you build into a fresh directory,
> copy `validation/` into it or leave the default `artifacts/` in place.

### Step 3 — build the C++ runtime

```powershell
cd runtime
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

> **`project()` must precede `find_package(XRT)`** or linking silently
> downgrades to static (mlir-aie #3048). **`/Zc:__cplusplus` is required**, not
> cosmetic — without it XRT demands Boost. `/arch:AVX2` enables the vectorised
> host paths.

### Step 4 — validate, then measure

```powershell
# correctness against HuggingFace (do this first, always)
.\build\npuembed.exe .. --artifacts artifacts_b128 --threads 16

# throughput
.\build\npuembed.exe .. --artifacts artifacts_b128 --threads 16 --bench 5
```

`--threads` controls the **host attention and conversion pool only**; default 1
keeps the low-core-count claim intact, so turning it up is an explicit trade of
cores for wall clock.

---

## 8. How to measure, and the knobs that matter

### The probes built into the runtime

```powershell
.\build\npuembed.exe .. --probe                     # repeat one design vs alternate two
.\build\npuembed.exe .. --probe-pair                # 2 resident contexts vs 7
.\build\npuembed.exe .. --probe-ctx                 # SAME xclbin in two contexts
.\build\npuembed.exe .. --probe-design <dir>        # one design: alone, A<->A', switch
C:\Windows\System32\AMD\xrt-smi.exe examine --report aie-partitions
```

`--bench` prints a five-way split of the NPU path (convert / sync-in /
dispatch+wait / sync-out / bias) plus a per-design table. Use it before
optimising anything: **twice in M7 the obvious explanation was wrong and the
breakdown found the real one.**

### Choosing width and batch

Design switching costs ~25 µs + 7.2 µs per lock, so **wider designs compute
faster and switch slower**. The crossover moves with M:

| GEMM columns | M = 256 (batch 4) | M = 1024 (batch 16) |
|---|---|---|
| 1 | 51.9 | 62.9 |
| 2 | **53.6** | **72.2** |
| 4 | 46.1 | **72.0** |

Predicted crossovers from `t = 140 µs + 2.43·M/cols`: 2 over 1 at M ≈ 237,
**4 over 2 at M ≈ 947**, 8 over 4 at M ≈ 3793. The M ≈ 947 prediction was tested
and confirmed. Rule of thumb: **`--cols 2` below ~batch 15, `--cols 4` above,
`--cols 8` at batch 64+**. `--elt-cols 2` always (until LayerNorm and softmax
are rewritten).

### The measurement rules that are not negotiable

1. **Wall clock is never an NPU kernel claim.** Hardware traces or static
   instruction counts. Wall clock is valid only for host/dispatch cost and
   end-to-end throughput, always labelled.
2. **Every sweep needs a control with a known correct value.** A sweep that only
   measures the quantity of interest cannot detect a silent corruption.
3. **Never a single run.** Pre-tiled designs show up to 22% spread; a one-shot
   bench once suggested +12.8% that three repeats erased.
4. **Verify a check against a known-bad artifact**, not only a good one.

---

## 9. Next steps, in priority order

The authority on open work is
[`research/OPEN-THREADS.md`](../research/OPEN-THREADS.md) (CLAUDE.md rule 3),
which as of 2026-08-27 holds **five live threads** — down from ten at the
start of the 0.5.0 session, every closure carrying the condition it depends
on. The priority order below is this file's own judgement, not the
register's (the register does not rank).

1. **T51 — the bge-large tail's mechanism** (step 2: layer-wise divergence,
   now scoped to profile *depth*, since [`0129`](../tasks/0129-t51-int8-tail/TASK.md)
   exonerated tiling and the number format — the same five words top the
   tail on two unrelated datapaths). The *instrument* debt is paid: the p99
   tail gate ships in the sweep. What is left is understanding, not
   protection.
2. **T43 — the byte-level BPE tokenizer**, still the gate on the
   ModernBERT/Qwen/Mistral generation. The Unigram build
   ([T52](../research/CLOSED-THREADS.md#t52), filed and closed in a day at
   343/343 byte-exactness twice over) is the template: generator, C++ port,
   byte-exact harness. Needs no NPU.
3. **T44 — which encoder joins next.** gte moved from candidate to shipped;
   the four byte-BPE candidates stand priced, all blocked on T43.
4. **T48 — B-reuse**, measured at **1.72× array / 1.31× e2e** on bge-large
   ([`0131`](../tasks/0131-t48-gate-decision/TASK.md)) and parked: the
   channel census shows the cheap wiring does not exist, so the build is the
   cascade C-collapse (vectorised `cascade_mm`) or a hierarchical C re-join
   — fund it as a project, not a session. A build must re-run
   [`0139`](../tasks/0139-t46-memtile-leg/TASK.md)'s port-coverage analysis:
   B-reuse is exactly the change that could re-bind the mem-tile leg.
5. **T42 — attention on the array for long sequences**, priced 1.65–1.95×
   at seq 512, trigger unchanged: only if long-sequence traffic becomes
   real. gte is now the second model the seq-256/512 sets serve, which is
   the demand signal this trigger watches.

Also queued as evidence-to-gate promotions: the multilingual STS baseline
([`0137`](../tasks/0137-gte-gates/TASK.md), 29 language subsets) becomes a
gate in 0.6.0; and `verify_semantics.py`/`verify_tail.py` still default to
the six-model built-in list — gte joins their defaults at the next canonical
re-cut of those reports.

## 10. The traps that will cost you an hour each

The full list is in [`CLAUDE.md`](../CLAUDE.md); these are the ones that bite
hardest.

- **Set the device explicitly** or IRON silently compiles for NPU1 and
  `--emulate-bfp16` becomes a no-op — worth 5.5×, no error.
  `iron.set_current_device(from_name("npu2", n_cols=None))`
- **Never write device tensors through `.numpy()`.** `A[:] = x`, not
  `A.numpy()[:] = x`. The first dispatch in a process is correct either way,
  which is what makes it so easy to ship.
- **Never validate against a device read-back** — it agrees with whatever the
  device used and passes while measuring nothing.
- **Never identify a build artifact by mtime.** A JIT *cache hit* does not
  restamp the directory.
- **`aie.mlir` has no tile coordinates** — it is pre-placement. Use
  `input_with_addresses.mlir` (the same file tracing needs).
- **A JIT cache can serve a stale object after a `.cc` edit** with identical
  size. Delete the cache directory.
- **Budget L1 before compiling**: `2·(m·k·in + k·n·in + m·n·out) < 64512`. The
  limit is **63 KB, not 64** — 1 KB of DMEM is reserved for the program stack.
- **Budget ports and registers, not only bytes.** Per core: **2 in / 2 out** DMA
  streams (so a kernel can compute `C = AB` but *not* `C = AB + C`), 24 vector
  and **5** accumulator registers. Per mem tile: 6 in / 6 out. Per shimNOC:
  **≤ 6 S2MM channels** — which is the wall the eltwise designs hit.
- **Never scalar float math in a kernel** — 1,617× slower, measured.
- **Watch the worker stack.** Four interleaved chains in GELU overran
  `stack_size=0xD00`. It does not fault; it **corrupts**, and produced NaN.
