# CLOSED THREADS

Answered, retired and superseded questions, moved out of
[`OPEN-THREADS.md`](OPEN-THREADS.md) on 2026-08-23 so that file can be read to
the end. **Nothing here is a summary** — each thread is verbatim, including the
claims that turned out wrong and the measurement that killed them, because
CLAUDE.md rule 3b says that is the part worth keeping.

Read this when you want to know *how* something was settled, or before
re-deriving anything: on 2026-08-23 a session spent an hour re-deriving on paper
what `tasks/0062` had already built and passed, because the thread that would
have said so ran to 308 lines. `tools/check_register.py` now checks for exactly
that.

A thread leaves this file only by being reopened — with a pointer, and a reason.
**A closure names the condition it depends on** (T49, check 2): if the reason a
thread closed can expire — a datapath not adopted, a host share, a seq length —
the entry says so in one line, so a reader can tell whether it still holds.

<a id="t30"></a>
### T30 — The C-drain guard is half-wired, and every shipped model is one step from it · **ANSWERED 2026-08-21 (found and fixed same day)**
**A design built with `N > 4096` COMPILES and returns the wrong answer.**
[`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §6, found while asking
only "does nomic's N=6144 `ffn_up` compile?".

`gemm_pretiled.py::_build_design` guards the C-drain DMA stride
(`if m * n_aie_rows * N > 2**20: tb_n_rows = 1`, tasks/0030). `C_tiles` uses the
guarded value; the fill loop's `current_tb_n_rows` computes its own from
`tb_max_n_rows // 2` and never reads it. When the guard fires, drains cover 1
row-block while fills stream 2, and most of the output is stale. Measured
`rel_fro` **7.074e-01**, 28/32 row-bands with max abs error > 1.0 — invariant
across cols 4/8, tile_n 16/32/48 and M 1024/8192, which is what ruled out a
hardware limit (BD size, stride range, DMA channels and L1 all checked clean).

**The boundary is exact and the near-miss is the point.** The guard fires at
`N > 2**20/(m·n_aie_rows)` = **N > 4096** at m=64, rows=4. Measured: N=3840 PASS
(3.291e-07), **N=4096 PASS (3.283e-07) — bge-large's real production shape**,
N=4224 FAIL (7.083e-01), N=6144 FAIL. `64·4·4096 = 1,048,576` is **exactly 2^20**
and the test is a strict `>`, so **the guard has never fired in a shipped
design**. The tasks/0030 fix has never executed in production and half of it is
wrong. The file's own comment records the near-miss — *"measured: N=4096 at
exactly 2^20 builds"* — without drawing the conclusion, because measuring **at**
a boundary never exercises what is beyond it.

**Also corrects a doc:** `docs/CURRENT_STATUS.md`'s "known walls" still lists
`hidden >= 1536` as an open *build* wall. It is not a build wall. It builds and
returns wrong numbers, which is strictly worse.

**FIXED** in 0068 §6b. The whole row-block walk is now driven by the guarded
`tb_n_rows` (`tb_step = 2 * tb_n_rows`; `row_base = tb*tb_step + pingpong*tb_n_rows`;
`current_tb_n_rows = min([tb_n_rows, ...])`), which at the historical unguarded
`tb_n_rows = 2` reduces algebraically to the originals — provably a no-op below
the threshold. Four gates: **N=6144 7.076e-01 → 3.283e-07 PASS**; the three
below-threshold shapes reproduce (8.168e-07, 3.288e-07, 3.284e-07); the M=256
single-row-block tail still builds correct (3.289e-07); and fixed-vs-unfixed
xclbins under the same toolchain differ by **74 bytes**, inside the 0029 UUID
budget.

The fallback is therefore NOT needed, but stays on record: split a gated
`ffn_up` into two N=3072 dispatches over the existing bge-base design set
(6144 = 2×3072; `64·4·3072 = 786,432 < 2^20`), at 12 extra dispatches per encode.

**Left behind, newly written down:** the shipped `artifacts_base` xclbin
**cannot be reproduced byte-for-byte from today's toolchain** — a scratch rebuild
with the *unmodified* generator still differs by 3,457 bytes, from the mlir-aie
1.3.4 → 1.4.x upgrade (tasks/0058). tasks/0059 established this does not affect
correctness (a shipped `.xclbin` is a static binary XRT loads regardless of what
built it), but "regenerating the release produces a different binary than the one
on disk" is a real property of the repo that had not been recorded.

**Standing consequence:** any future model wider than `N = 4096` per GEMM would
have been silently wrong before this fix, and the failure mode left no trace. See
also
[T31](#t31--design_fits-matches-on-k-alone-and-nomic-breaks-it--open-filed-2026-08-21).

<a id="t31"></a>
### T31 — `design_fits()` matches on K alone, and nomic breaks it · **ANSWERED 2026-08-21 (filed and fixed same day)**
`pick_artifacts()` picks a design set by asking whether `hidden` appears as a
`"K"` in `design.json` (`main.cpp:1363`). nomic's K set is `{768, 3072}` —
**identical to bge-base's** — but its gated `ffn_up` is `N=6144` where
bge-base's is `N=3072`. So the check passes and hands nomic a design that
computes half the FFN width, silently.
[`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §7 verified this
against the real `artifacts_base/gemm_rtp/design.json`: three of four streams
match exactly, `ffn_up` does not.

The predicate is weaker than its own comment, which names precisely this danger
(*"a design built for another width has the same filenames and loads fine — it
would simply compute the wrong thing"*). It has been sound so far only because
every shipped model has `N = 4·hidden`.

**FIXED** in [`0069`](../tasks/0069-m13-nomic-arch2-container/TASK.md).
`design_fits()` now matches the **streams' `(op, K, N)`** against the
container's `hidden`, `intermediate` and `gated_ffn`, requiring every op to be
present and *every* occurrence of it to match. It reuses `parse_streams()`,
which every `design.json` has carried since 0032, so the hole closes on design
sets exported long before the geometry keys existed — **no re-export needed**.
`export_gemm_rtp.py` still writes explicit `hidden`/`intermediate`/`gated_ffn`
keys, because a design that states its own geometry beats one inferred from its
streams.

Falsification test, which is the part that matters — a predicate that only ever
says "yes" proves nothing: with `artifacts_nomic` moved aside and
`artifacts_base` still on disk, nomic reports **`no design`** while bge-base
stays **`ready`**. The four shipping models each still resolve to the set they
resolved to before, and their validation encodes reproduce their recorded
figures exactly (bge-base 4.297e-03, the number in 0051).

**It also closed a fail-open nobody had filed:** `print_catalog()` called
`pick_artifacts()` unconditionally, so `embeddinggemma-300m` reported **`ready`**
whenever any hidden-768 design happened to be present — despite arch=1 having no
NPU kernel at all and running entirely on the host. There is now a `cpu` state.

**And it exposed a third one, in the making.** With nomic's container packed and
its design built, `list` said `ready` — true about designs, wrong about
outcomes. `Encoder::run()` is a BERT forward pass, and arch=2 reuses BERT's
tensor names and shapes deliberately, so it would have read every tensor, run
the wrong model, and returned embeddings nothing downstream could question.
`set_model_shape()` now refuses on any architecture this build has no encoder
for (exit 2), written as a **whitelist of what is implemented** rather than a
blacklist — the pre-existing arch=1 diversion names the one arch it redirects,
which is exactly the mechanism by which anything new falls through to BERT.
`encoder_implemented()` is the single source both the `list` table and the
dispatch refusal read, so they cannot drift.

<a id="t33"></a>
### T33 — `l2_normalize` is written to the container and never read, and it costs nomic 4.5 MTEB points · **HALF ANSWERED 2026-08-23** · the runtime now reads it; nomic's value is still a decision

> **Item 1 is done.** `main.cpp` reads `l2_normalize` from the container
> instead of hardcoding it. This changes **nothing** today — every shipped
> container says `true`, and a container predating the key still defaults to
> `true`; verified by all three gates reproducing their exact numbers. What it
> changes is that deciding nomic's case is now a **repack**, not a code change.
>
> **Item 2 is still open and is not ours to settle**: nomic's model card calls
> `F.normalize` while its sentence-transformers pipeline has no `Normalize`
> module. The model is ambiguous, and the 4.5-point Banking77 gap is a question
> about which geometry downstream code wants.
**The runtime always L2-normalises. For one shipping model that is measurably
the wrong choice on classification tasks.**

`pack_npue.py` writes `"l2_normalize": true` into every container.
`main.cpp:90` hardcodes `g_l2_normalize = true` and **never reads the key** — the
eighth fail-open's shape, a literal that should have been data, except here it
has been harmless because every model wanted `true`.

nomic-embed-text-v1.5 is the first where it is not obviously right. Its
sentence-transformers pipeline is `Transformer + Pooling` with **no `Normalize`
module** — `encode()` returns vectors of norm ≈20.9 — while all four BERT models
end in `Normalize`. Measured in [`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)
Stage 4b, on the reference side alone:

| Banking77Classification | score |
|---|---:|
| nomic, **unnormalised** (what sentence-transformers gives you) | **83.77** |
| nomic, **normalised** (what this runtime gives you) | **79.23** |

**4.5 points, on a logistic-regression task.** STS is unaffected — cosine cannot
see a scale change — and clustering moves a little (38.29 → 38.96, the other
direction). So this is not a precision question; it is a question about which
geometry downstream code wants.

Two separate things to decide:

1. **Should `l2_normalize` be read from the container** rather than hardcoded?
   Cheap, and it turns a silent assumption into a stated one. The container
   already carries the key.
2. **Should nomic's container say `false`?** Less obvious. nomic's own
   documented usage calls `F.normalize`, and the Matryoshka recipe normalises
   after truncation — so `true` matches the model card while `false` matches
   what sentence-transformers actually returns. The model is ambiguous, not us.

Until decided, note the user-facing consequence plainly: **running nomic through
sentence-transformers and through this runtime gives vectors of different scale
for the same text.** Cosine agrees; a classifier trained on one does not
transfer to the other.

**How this surfaced is the useful part.** nomic *failed* its MTEB gate at
mean −0.68 / worst −4.54, and the number alone reads as "the new architecture
costs accuracy". The pattern said otherwise: the three cosine tasks agreed to
±0.03 while the two raw-feature tasks moved in **opposite** directions, which is
not the signature of a precision loss. The harness was comparing normalised NPU
vectors against unnormalised CPU ones. Fixed there; the underlying question is
this thread.

<a id="t1"></a>
### T1 — What *is* the GEMM's 3 ms? · **ANSWERED 2026-08-19**
**The count of tile iterations, not bytes.**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md): `ffn_up` and
`ffn_down` have identical MACs and differ **1.50× in bytes**, and measure
**4,196 vs 4,273 µs** — 1.8% apart, in the wrong direction. `GMAC/ms` is flat at
1.08–1.15; `GB/s` spreads 17.7–27.0. Reproduced on the `--c-bf16` set. Fit:
`t = 573 µs + 4.72 µs × iterations`, ≤2.3% residual, against 0010's traffic
model being 50% out on the discriminating pair.

Left a successor, [T16](#t16),
answered the same day in [`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md) — the "4.3×" compared production against the
bfp16-emulated datapath's trace. Note therefore that this thread's own framing
("compute-shaped, but not compute") is half-superseded: it IS compute, on the
fp32 vector datapath.

> **SCOPED 2026-08-22 ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)):
> this answer is about the bf16 datapath, and int8 reverses it.** The same
> discriminating pair, re-run on int8, *diverges* (1.120 in dispatch wall time
> against bf16's 0.986), and fitting both models over the four production
> shapes swaps them cleanly:
>
> | datapath | traffic (0010) | iterations (0048) |
> |---|---|---|
> | bf16 | R² 0.709, worst 20.1% | **R² 0.994, worst 2.6%** |
> | int8 | **R² 0.987, worst 5.3%** | R² 0.568, worst 36.4% |
>
> int8: `t ≈ 627 µs + traffic / ~28 GB/s`. Independently confirmed by building
> the geometry the iteration model recommends: **k=128 fits int8's L1 where it
> overflows bf16's, halves the iterations, and measures 0.989× — no change.**
> The same regime [T27](#t27) found for emulated bfp16; anything that makes the
> MACs cheap moves the constraint to transport.
>
> **So "bytes are free" is a bf16 statement, and every lever retired on it is
> live again for int8** — see [T2](#t2)
> and [T12](#t12). Acted on
> already: narrowing C from int32 to bf16 removes 27% of the traffic for
> **1.333× on the four dispatches**, at *better* accuracy.

<a id="t2"></a>
### T2 — Claim B-reuse via cascade · **RETIRED 2026-08-19**
**Retired by [T1](#t1).** B-reuse
removes *bytes*; bytes are not the constraint. The 1.26–1.68× that
[`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) priced was priced with
the model 0048 refutes, and the cascade milestone
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) scoped existed only to
free channels *for* B-reuse.

The channel census from [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) and
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) keeps its value — it is
about what the array can express, not about B.

> **REOPENED for the int8 datapath, 2026-08-22
> ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)).** The retirement
> rested entirely on "bytes are not the constraint", and on int8 they are: the
> traffic model fits at R² 0.987 and **B is 23–43% of the traffic per
> dispatch**, replicated ×32 by the row blocks. 0010's 1.26–1.68× was priced
> with a model that is correct *here*.
>
> What has **not** changed is 0046's finding that the blocker is DMA
> **channels** rather than capacity — every core tile 2/2 in, five of eight mem
> tiles 6/6 in, with the C join spending the budget. So this is now a live
> lever behind a real hardware wall, not a dead one. Note the C join is
> *cheaper* since 0080 narrowed C, which is the first thing to re-census.

> **SUPERSEDED by [T48](OPEN-THREADS.md#t48), 2026-08-27.** The reopening above
> was scoped to int8. Since [`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md)
> the *default* datapath is bfp16 on five of six models, where **B is 46–52% of
> DDR traffic** — the largest single term on every dispatch —
> and [`0120`](../tasks/0120-roofline-analytic/TASK.md) prices B-reuse at
> **1.80–2.02× array / 1.18–1.34× end to end** from geometry alone. The
> annotation above was also never acted on: this thread stayed in the *closed*
> file while reading REOPENED, which is [T49](#t49)'s first
> instance.

<a id="t29"></a>
### T29 — EmbeddingGemma-300M: does the tile_n=16 tax rule it out? · **ANSWERED 2026-08-22** · NO — and the tax itself was avoidable. It runs on the array at ~133 seq/s.

**2026-08-22 ([`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md)): answered NO,
and the question was framed on the wrong option set.** The `tile_n = 16` floor
is real arithmetic about MQA's 256-wide K/V, and
[`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md) proved it holds across
all four *fusion* strategies. What nobody tried was **zero-padding the N axis**,
which costs the packer nothing because B is pre-tiled offline anyway: append 256
zero columns to the fused Q|K|V and N goes 1280 → **1536**, a multiple of
`tile_n · n_aie_cols = 384`. Zero columns of B give exactly-zero columns of C,
so this is **exact**, not an approximation, and the host slices Q/K/V off the
front by offset. Gemma then runs at `tile (64,64,48)` — the production geometry
— and packs to the **same `layout_hash`** as MiniLM, bge-small, bge-base and
nomic.

Two corrections to this thread's own record, both load-bearing:

* **0055's table already contradicted its verdict.** Strategy C (fused QKV 1280,
  fused gate_up 2304) is listed there at `tile_n = 32` @ 8 cols, passing the L1
  budget. The verdict line, the 62–72 seq/s prior and this thread's *title* all
  used **16**, the worst strategy, with nothing justifying that choice — the
  packer picks the fusion. Padding then beats strategy C too, by 1.40×.
* **`design_fits()` derived `qkv`'s width as `3 * hidden`**, so it would have
  **rejected Gemma's own correct design**. Same bug class as
  [T31](#t31), one field to the left. `qkv_n` is now data in `design.json`, in
  the container, and in `hub.cpp`'s catalogue row.

**Measured, on hardware.** 97.7% of the model's MACs on the array (4 GEMMs per
layer × 24 layers = 96 dispatches, ONE xclbin, one hw_context). Throughput
**134.9 / 133.0 / 132.7 seq/s** over three runs behind the NPU contention guard
(1.7% spread, 4 lanes, wall clock end to end — not an NPU kernel claim) against
the host-only path's **0.20 seq/s** on the same corpus in the same session.
`1-cos` **9.962e-06** against that host-only path, which
[`0064`](../tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md)/[`0065`](../tasks/0065-m12-embeddinggemma-cpp-packer/TASK.md)
tied to the numpy reference at 5.496e-13 — in family with every other bf16 model
here. Every lane count (1/2/4/6) returns **bit-identical** output, and the 40
repeats of each sentence in a 520-row encode agree to `max |Δ| = 0`.

The array-only share of that wall clock is **176 seq/s**, inside the 140–162
prior [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s model gave
before the build — so the cost model is confirmed at a third width, not merely
un-refuted.

Both packers agree **byte-for-byte** on the new container (sha256
`4a4c6f70995601f6…`, first attempt), so tasks/0065's parity property survives.
What is left is [T34](#t34), not this thread.

---

*(history below; the tile_n question above is now closed)*

**2026-08-21 update (tasks/0067)**: **the one gap tasks/0066 left open is
closed — `gemma_tokenizer.bin` now has a C++ generator.** A from-scratch,
dependency-free JSON DOM parser (`runtime/include/json_min.hpp` +
`runtime/src/json_min.cpp`) backs a line-by-line port of
`tools/gen_gemma_tokenizer_table.py` (`runtime/include/gemma_tokenizer_gen.hpp`
+ `runtime/src/gemma_tokenizer_gen.cpp`), every validation guard preserved.
`prepare_model_gemma()` now generates-and-caches the table instead of only
reading one. Verified against the real, genuinely Python-generated reference
(not self-reference, CLAUDE.md traps 6b/6c): **byte-identical, sha256
`c7a03c2c35ffc2a16b5513bb11c3d04e4a19c84acb9254b36765510acbf5bc81`, both
exactly 9,020,206 bytes** — confirmed twice, once from a standalone verifier
and once from the real `npuembed.exe --prepare-model` with the cached file
deliberately removed. A from-scratch clone with only `HF_TOKEN`/`--token` set
can now produce a fully working `.npue` with no manual Python step. The
tile_n=16 question this thread is named for remains genuinely open and
unaffected by this update — it only concerns a future NPU-kernel version of
this model, and the shipped host-only path still needs none of that
machinery.

**2026-08-20 update (tasks/0066)**: **the gated HuggingFace fetch was tested
with a real `HF_TOKEN`, for the first time.** `hub.cpp`'s `table()` now
carries a real, verified catalogue row for `embeddinggemma-300m` /
`google/embeddinggemma-300m`, `sha256` **`cbf5a78393b6a033e0b8a63a57549964
f7ed5c6fbeb4ba0694214f36123f2fd2`** — pinned by downloading the OFFICIAL
gated repository and confirming it is **byte-identical** (same sha256, same
`config.json`) to the `unsloth/embeddinggemma-300m` mirror every prior task
verified against, so nothing already on record needed re-checking. Two
integration bugs (neither in the fetch/pack logic itself) found by testing
against a genuinely FRESH root rather than the already-populated checkpoint
directory: (1) the `embed <model>` subcommand's post-fetch dispatch was
BERT-only (`pick_artifacts()` unconditionally, throwing "no NPU design for
hidden 768" before Gemma's own dispatch ever got a chance) — fixed with an
arch-aware short-circuit right after `ensure_model()`; (2) **`gemma_tokenizer
.bin` has no way to be produced by a token-only fetch** — its generator,
`tools/gen_gemma_tokenizer_table.py`, is Python-only and this project's
shipped product is C++-only at runtime (CLAUDE.md rule 5), so a from-scratch
clone cannot self-produce it. Worked around for THIS test by copying the
already-generated table; **not fixed** — either port the generator to C++,
or ship the table (vocab-derived, checkpoint-independent) as a release/repo
asset. Full end-to-end correctness confirmed **bit-identical** against the
already-verified reference container on the same input, once the table was
present.

**2026-08-20 update (tasks/0064)**: **EmbeddingGemma-300M now runs end to end
in production C++, entirely on the host** — the arch=1 integration C3/C4
scoped for. `tools/pack_npue.py` gained `pack_gemma()` (dispatched from the
checkpoint's own `config.json["model_type"]`, never guessed): every GEMM
operand is stored PLAIN (F32, row-major, no tiling), because there is no NPU
kernel for this arch and nothing here ever becomes a DMA descriptor — which
means **the tile_n=16 tax this thread is named for DOES NOT APPLY to the
path that actually exists today**. It remains the right question for a
*future* NPU-kernel version of this model (unanswered, unchanged by this
task), but the integration that shipped needed none of the tile_n/L1-budget/
DMA-BD machinery at all. 317/317 packed tensors round-trip bit-exact against
the source checkpoint (which is confirmed ALL-F32 on disk, 314 tensors,
correcting an assumption that it shipped bf16). A new `GemmaEncoder`
(`runtime/include/gemma_encode.hpp`) runs the full 24-layer forward pass
(double-accumulated host GEMMs, `0061`'s tokenizer, `0063`'s RMSNorm/RoPE/
GeGLU kernels, MQA collapsed to direct K/V reuse since
`num_key_value_heads=1`) and is wired into `main.cpp`'s production
`Encoder::run()` as a genuinely separate early-dispatch path (not a branch
inside it — `Encoder` needs seven NPU `Design&` this arch has none of).
Checked against `reference/encoder_gemma.py` on two independent 4-sentence
corpora, real HuggingFace tokenization: worst `1-cos` **4.969e-12** and
**5.496e-13** — tighter than any other model's gate in this project (no bf16
rounding anywhere in this path). `npuembed.exe --model embeddinggemma-300m
--embed` reproduces the standalone verification CLI bit-for-bit; the BERT
path's MiniLM golden check is unaffected (`1-cos` 1.086e-05, identical to
its recorded history) after rebuild. `--serve` on this arch refuses loudly
(no HTTP endpoint built) instead of silently doing nothing.
**Not done**: the C++ packer mirror (`npue_pack.cpp` has no arch=1 support —
`--prepare-model` on a Gemma checkpoint only works through Python today);
`hub.cpp` gained a `gated` field and fail-closed `HF_TOKEN` bearer-auth
support but deliberately NO catalogue row for EmbeddingGemma (no session has
ever held `HF_TOKEN` to pin a verified sha256, and the auto-pack step is
still BERT-only C++); no batching/AVX2/threading (~7.9 s/sentence,
unoptimised by design — correctness was this task's stated priority); no
MTEB run on this arch at all yet. Full detail:
[`0064`](../tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md).

**2026-08-20 update (tasks/0065): the C++ packer mirror gap CLOSED, fully —
`--prepare-model` on a Gemma checkpoint now works through C++ too, and it is
proven byte-identical to `pack_npue.py`, not just structurally similar.**
`runtime/src/npue_pack.cpp` gained `prepare_model_gemma()`, a direct port of
`pack_gemma()`: `Writer::write()` grew one new `arch` parameter (default 0,
BERT call site unchanged), a new `add_gemm_b_host()` sits next to the
existing tiled `add_gemm_b()` and is less code than it (transpose + raw F32
copy, no tiling, no `layout_hash`), and `main.cpp`'s `--prepare-model` CLI
gained an early `config.json["model_type"]` dispatch mirroring Python's
`main()`. **`tools/verify_pack_parity.py`, extended to detect the arch by
`model_type` instead of assuming every checkpoint has `vocab.txt`, reports
byte-identical `sha256` for both packers' 1,239.65 MB / 317-tensor output** —
confirmed a second way by a direct `sha256sum` on both files outside the
harness, and the BERT path stays byte-identical too (regression check). The
correctness gate was re-run against the FRESH C++-packed container rather
than assumed from parity: worst `1-cos` **5.496e-13** / **4.969e-12** on
tasks/0064's own two corpora, identical to the digit already on record, with
the raw encode output file `cmp`-matching tasks/0064's bit-for-bit. The one
real risk in the port — C++ reformatting `rms_norm_eps`/`rope_theta`/
`rope_local_base_freq` and disagreeing with Python's float-to-string in the
last digit — was sidestepped by copying the config.json literal substring
verbatim (checked correct for this checkpoint's exact text by a direct
`json.dumps(json.loads(x))` round-trip, not assumed in general — a future
checkpoint with non-canonical float formatting would FAIL parity loudly
rather than silently packing wrong bytes). What tasks/0064 left NOT done
(HF_TOKEN-gated fetch + `hub.cpp` catalogue row, MTEB, CPU speed work) is
unchanged and was out of this task's scope. Full detail:
[`0065`](../tasks/0065-m12-embeddinggemma-cpp-packer/TASK.md).

**2026-08-20 update (tasks/0055/0061/0063, superseded in relevance by 0064
above but kept for the history)**: the user decided to proceed ("Vi kjører
Gemma") — the
~62-72 seq/s prior below is accepted, not a blocker. **C2 (tokenizer) is done
and independently verified**, [`0061`](../tasks/0061-m12-embeddinggemma-tokenizer/TASK.md):
**1,925/1,925 sequence comparisons byte-identical to HuggingFace** (base
corpus 210/210 + a 300-codepoint byte-fallback stress corpus 1,715/1,715),
across five task-prefix configurations, `max_len` 64. **Load-bearing
correction to this thread's own framing and to the plan**: this checkpoint's
`tokenizer.json` declares `model.type == "BPE"`, not Unigram — the plan and
`0055`'s "~600-900 LOC SentencePiece Unigram" estimate both assumed the wrong
algorithm family (reasonable by analogy with other sentence-embedding
tokenizers; wrong for this one). It is the standard Gemma/Llama-family
SentencePiece BPE (metaspace normalizer, no pre-split word boundaries —
confirmed the whole prefixed+normalized text is ONE BPE input, not
per-word — byte-fallback via `<0xXX>` vocabulary entries, 514,906 merge
rules). `tools/gen_gemma_tokenizer_table.py` and `runtime/src/
tokenizer_gemma.cpp` implement BPE, not Viterbi/Unigram search — a materially
different core algorithm than what was asked for, caught only by reading
`tokenizer.json` before writing code. Task-prefix table (14 rows) read
verbatim from the checkpoint's own `config_sentence_transformers.json`;
project default is `"document"` (`"title: none | text: "`) — a decision,
since the checkpoint's own `default_prompt_name` is `null` (no prefix by
default under the standard sentence-transformers API). **Not yet wired into
`main.cpp`/`hub.cpp` — standalone only**, by design (0061's explicit scope).

**C3 (RMSNorm/RoPE/GeGLU host kernels) is done and independently verified**,
[`0063`](../tasks/0063-m12-embeddinggemma-kernels/TASK.md): `runtime/src/
gemma_kernels.cpp`, checked against real tapped intermediates from `0055`'s
checkpoint run — **36/36 records PASS**. Per-head `q_norm`/`k_norm` RMSNorm
and all 4 GeGLU cases are **bit-exact** (`rel_fro` 0.0); full-hidden RMSNorm
0.0–2.75e-11; RoPE 1.7e-8–4.3e-8 (the float32 ULP floor, confirmed against
numpy's own `cos`/`sin`). One `rms_norm_cpu` (double-precision reduction,
the Gemma `x/rms*(1+w)` form — NOT the more common `x/rms*w`) serves both
the full-hidden and per-head uses. GeGLU needed the reference's exact
two-stage float32 rounding (round `act`, THEN promote and multiply by `up`)
to reach bit-exact from ~1e-8 — a mathematically-equivalent single fused
double-precision expression was NOT enough, a real finding about matching
reference rounding order, not just reference formulas. **A negative control
run against the same real tensors** (drop RMSNorm's `1+`; swap RoPE's
per-layer theta; use the existing BERT `gelu_cpu`'s exact-erf form instead
of Gemma's tanh-approximation) measures `rel_fro` 0.439 / 0.757 / 3.26e-4 —
4-6 orders of magnitude worse than the real result, proving the test
discriminates rather than just measuring noise. **Not yet wired into
`main.cpp`/`hub.cpp`/the packer — standalone only**, same discipline as C2.
**Next slice if continuing**: the `arch=1` runtime branch in `Encoder::run()`
that actually calls these kernels in sequence, plus the `hub.cpp` catalogue
entry (with `HF_TOKEN`-gated fetch per the plan's decision) and the packer
changes (both `pack_npue.py` and `npue_pack.cpp`, held byte-identical) to
get MQA-fused QKV weights and the container's `arch=1` field into a real
`.npue` — none of which touch the NPU directly, since Gemma's host-side ops
have no NPU kernel yet either (matching this project's own "eltwise lives on
the host" precedent for the BERT models).

[`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md), Del C's C1 spike
(plan: `~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`). The numpy
reference encoder (`reference/encoder_gemma.py`) validates against HuggingFace
and sentence-transformers to **1-cos 1.065e-07 / 2.110e-08** — as tight as
M3's MiniLM oracle, despite 24 RMSNorm/RoPE/GeGLU/MQA layers against MiniLM's
6 LayerNorm/GELU ones. The architecture question the plan flagged is answered
with real numbers, not the plan's estimate: MQA's K/V width (1 KV head ×
head_dim 256 = 256) floors `tile_n` at **16 at 8 columns / 32 at 4 columns**
regardless of whether QKV or gate/up are fused offline — confirmed across all
four fusion combinations, matching the plan's prior exactly. L1 budget passes
comfortably at both (28,672 B and 40,960 B of 63 KB); the DMA BD 1024-dim
limit is nowhere close (worst case 80 n-blocks). **The real cost is
iteration count, per [T1](#t1)'s
own model**: a per-shape iteration count comparison against bge-base
(same hidden=768, tile_n=48, 12 layers) gives Gemma **3.58× more total
GEMM-iteration-proxy per encode** (24 layers × 516 vs 12 × 288), of which
isolating tile_n alone (hypothetical illegal tile_n=48 for Gemma) accounts
for **~3×** and GeGLU's genuinely lighter per-layer FFN width (2×1152 vs
bge-base's 1×3072) claws back roughly 0.6× — so this is a geometry tax, not
an inherent-FLOPs one. Feeding both shape sets through
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s own fit
(`t = 573 µs + 4.72 µs × iterations`, summed per shape then per layer) gives
a **2.90× dispatch-time ratio** (the fixed 573 µs/dispatch term dilutes the
raw 3.58× iteration ratio slightly) — a **PRIOR, not a measurement** (no
hardware trace exists for this model): roughly **62-72 seq/s**, against
bge-base's measured 181.2-209.1. This ignores new host-side work this model
needs that BERT-style models do not (4× RMSNorm/layer, RoPE, GeGLU's
elementwise gate multiply — all host per the plan's C3 sketch), so it is
likely optimistic, not pessimistic. **Not answered**: whether ~62-72 seq/s
(or worse) is worth the C2-C4 cost (new SentencePiece tokenizer ~600-900 LOC,
two packer rewrites, `arch=1` runtime branch, MQA-aware weight fusion) — a
product decision for the user, laid out in
[`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md)'s go/no-go section,
not concluded here.

<a id="t35"></a>
### T35 — `npuembeddings add <model> [<sha256>]` for finetunes · **BUILT 2026-08-22** ([`0076`](../tasks/0076-m13-add-model/TASK.md)) · one question left open

**Built and working**: `TaylorAI/bge-micro-v2`, a 3-layer F16 finetune the
executable had never heard of, was added by name, fetched, packed and run on
the array at `1-cos` **5.507e-06** against sentence-transformers — the same
band as the built-in six, with no design rebuilt.

Both design points below were resolved as written. The user catalogue is
`models/catalog.json`, merged AFTER the built-ins and forbidden from shadowing
them; geometry is derived from the finetune's own `config.json` and the largest
legal `tile_n` is computed rather than assumed. Two bugs that only a REAL
finetune could expose are recorded in 0076 §4: every built-in ships F32 while
HuggingFace mostly does not (the packer now widens F16/BF16 at read time), and
the golden-fixture guard refused a correct run because a model with no fixtures
fell back to another model's.

**Still open, and it is the one thing here that is reasoning rather than
measurement:** a finetune with **added tokens** has a larger vocabulary. That
should change the embedding table and nothing else — the GEMM designs do not
see it — but no such checkpoint has been run. It is the cheapest possible test
and it has not been done.

Also not built, deliberately: `add` does not export a design, so a finetune
with an unusual `intermediate_size` is reported as having none and refuses at
`serve` rather than silently using a wrong one.

*(original filing below, kept for the reasoning)*
Planned for 0.4.0 (user, 2026-08-22). Today the model catalogue is a C++
literal in `hub.cpp::table()` with a pinned sha256 per row, so the only models
that exist are the six this repo shipped. A **finetune of an existing model is
the natural next axis** and is nearly free: identical geometry means the same
designs, the same `layout_hash` and the same container layout.

Two design points, one of them already decided.

**1. The trust model — DECIDED by the user, 2026-08-22.** `add <model> <sha>`
pins and verifies exactly as the catalogue does today. **`add <model>` with no
sha stores an EMPTY pin and does not validate** — and must WARN, loudly, on
stdout, in two places: when downloading, and again every time that model is
loaded. Not once at install time: a warning you saw last month is not a warning
you see today, and the whole point is that the person running it knows the
weights were never verified. Note the history this sits on: `get-model.cmd` was
deleted because `curl` + a hardcoded hash comparison is the behavioural
signature of a dropper (0051), so a *silent* unpinned fetch is exactly the shape
this project already refused once.

**2. Geometry should be DERIVED and validated, not matched against a row.**
The catalogue currently cross-checks a download's `config.json` against the
row's hardcoded geometry. For an arbitrary finetune there is no row, so invert
it: read `hidden`/`layers`/`heads`/`intermediate`/`model_type` from the
checkpoint's own `config.json`, and ask `design_fits()` whether an installed
design serves it. A finetune of bge-base then works with no new entry at all,
and anything else refuses with a message naming the geometry it would need.
This is strictly better than the current check for the pinned rows too.

The mechanical part -- moving `table()` from a C++ literal to a JSON/YAML file
-- is the small half. Things to assert rather than assume: a finetune with
ADDED TOKENS has a larger vocabulary, which changes the embedding table but not
the GEMM designs; pooling must keep coming from `1_Pooling/config.json` and the
prompts table from `config_sentence_transformers.json` (both packers already do
this); and `model_type` still selects the arch, so a nomic or Gemma finetune
routes correctly without a flag.

<a id="t36"></a>
### T36 — Was nomic's MTEB +0.09 measured symmetrically? · **ANSWERED 2026-08-22: YES, it stands**

**Measured, not reasoned.** Captured the kwargs `mteb` actually passes to the CPU side for
all five gate tasks (abort after the first `encode` call):

| task | what mteb passed |
|---|---|
| STSBenchmark | `prompt=None` |
| SICK-R | `prompt=None` |
| STS12 | `prompt=None` |
| Banking77Classification | `prompt=None` |
| TwentyNewsgroupsClustering | `prompt=None` |

**mteb injected nothing.** So both sides used the container's `search_document` prefix —
`run_mteb.py` prepending it on the CPU side, `NpuEncoder` on the NPU side — and nomic's
recorded **+0.09 mean was symmetric**. No correction needed to
[`docs/CURRENT_STATUS.md`](../docs/CURRENT_STATUS.md).

**Why nomic escaped and EmbeddingGemma did not.** It is the *key set*, not the architecture.
Gemma's `config_sentence_transformers.json` ships prompts named after MTEB **task types** —
`STS`, `Classification`, `Clustering` — so mteb matched one every time. nomic's
sentence-transformers config carries only `document` and `query`, which mteb uses only when
`prompt_type` is set, i.e. for retrieval-style tasks with distinct query and corpus sides.
None of the five gate tasks is one.

**One trap this leaves standing.** Add a **Retrieval** task to `DEFAULT_TASKS` and nomic's
`query`/`document` keys *would* match, and the asymmetry would reappear — for nomic, silently,
because its ST prompt text and its container prompt text are different strings. The fix
already shipped guards it (`run_mteb.py` skips its own prefix whenever mteb supplies one, and
`NpuEncoder` performs the same lookup), but the guard has only been exercised on Gemma. A
first retrieval task should re-verify both sides encode the same string before its number is
believed.

An intermediate reading nearly went in the log and was wrong: a hand-written guess at mteb's
lookup order said `query`/`document` would match on every task. The empirical capture says
they do not. **The candidate list was a guess at someone else's resolution order**, which is
not evidence about it.

*(original filing)*
[`0075`](../tasks/0075-m13-arch1-measurement-harness/TASK.md) §5b found that `mteb` passes a
task-appropriate `prompt=` to a SentenceTransformer, chosen from the model's own prompts
table — and `run_mteb.py` was *also* prepending the container's prefix, so the CPU side got
both while the NPU side got only its default. On EmbeddingGemma that asymmetry read as a
**-1.88 point M8 FAIL** while the two sides' embeddings agreed to `1-cos` 8.4e-06.

**Every MTEB number this project has published was taken through that code path.** For the
four BERT models the question closes immediately: their containers carry **no prompts table
at all**, so mteb had nothing to match and nothing to inject.

**nomic-embed-text-v1.5 is the one that has to be checked.** Its container *does* carry a
prompts table (`search_document`, `search_query`, `clustering`, `classification`) — written
by this project rather than by the checkpoint, and labelled as such. Whether
sentence-transformers loaded those keys, and whether mteb matched any of them to a task type,
decides whether nomic's recorded **+0.09 mean** was a symmetric measurement or the same
artifact at a smaller scale. Note the two plausible outcomes differ in kind: if ST never saw
the table, the number stands as taken; if it did, the number is measuring a prompt
difference.

**The test is minutes**, and it is the same probe 0075 used: monkeypatch the CPU side's
`encode`, run one cheap task, and print the kwargs `mteb` supplies. If a `prompt` /
`prompt_name` appears, re-run nomic's five tasks through the fixed harness and correct
`docs/CURRENT_STATUS.md`.

Until it is checked, treat nomic's MTEB line as **unverified rather than wrong** — the
asymmetry's direction is not predictable a priori, and its size on Gemma was set by how
different the two prompts were.

<a id="t14"></a>
### T14 — Per-core: operand prep or accumulator dependency? · **ANSWERED 2026-08-19**
**Neither — the datapath choice.** Folded into T16 and answered with it
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): the plain-bf16
inner loop is near-perfectly packed (a `vmac.f` in essentially every VLIW
bundle, operand shuffles dual-issued alongside), so operand prep costs ~nothing
and the 32-MACs/cycle ceiling is the fp32 vector datapath itself. 0003's static
prediction (~28) was right all along.

<a id="t16"></a>
### T16 — Why is a k-block iteration 4.3× the arithmetic in it? · **ANSWERED 2026-08-19**
**It is not — the 145-MACs/cycle baseline was the bfp16-EMULATED datapath.**
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md): every row of 0007's
"148.9–149.9 traced" artifact carries `emulate_bfp16: true` and the 1.04e-02
error signature; M2's 137–142 are the same path, and M2's own plain-bf16 figure
was always **25.0**. The production datapath traced today at 4 columns:
**7,813-cycle k-block window of which 6,144 = exactly 768 MMAC steps × 8
`vmac.f` — the fp32 datapath at its hard 32 MACs/cycle limit** — plus 1,669
non-vector in-window and an 84-cycle gap. LOCK/STREAM_STALL ≈ 0. At the
documented 1.808 GHz that is 4.37 µs, 93% of 0048's fitted 4.72 µs (0048's
"5,900 cycles" used an implicit wrong clock). **The GEMM is compute-bound on
the fp32 datapath at ~100% code quality; only the 22% non-vector share is
amortisable by any tile-geometry lever.** Under emulation the same design is
the opposite — 39% vector busy, lock-stall gaps, DMA-bound — which is what
M2's "starved, not slow" actually described, and why bytes measured free in
0048.

<a id="t19"></a>
### T19 — Stationary-B single buffering, to make `k` bigger · **ANSWERED 2026-08-20: negative, measured**
**Built and traced in [`0052`](../tasks/0052-m10-research-night/TASK.md) §8:
k=96 with B single-buffered is 5.2% WORSE per MAC than the shipping
(64,64,48).** The compute scales perfectly (vector cycles exactly 1.5×, still
at the 32 MACs/cyc limit) and the overhead amortises as predicted — but
single-buffering exposes B's L2→L1 fill, and the inter-window gap grows
84 → 1,193 cycles, roughly double what the amortisation saves. The thread's
own risk note called this exactly. `--b-depth` stays in `gemm_pretiled.py`
for reuse. Only the ATB form (shrink A's M, keep everything double-buffered,
[2511.16041](https://arxiv.org/abs/2511.16041)) remains open on this front, under T17's
≤1.29× ceiling. Original pricing follows as history.
Falls straight out of [T1](#t1):
iterations go as `1/(m·k·n)`, so raising `k` cuts them directly. **`k` is already
maximal under today's budget** — everything is double-buffered, giving
`4mk + 4kn + 8mn`, and at `(64, ?, 48)`:

| k | today (all double-buffered) | Stationary-B (`B` single) | 384 % k | 1536 % k |
|---:|---:|---:|---:|---:|
| **64** | 53,248 **OK** | 47,104 OK | yes | yes |
| 88 | 64,000 OK | 55,552 OK | no | no |
| **96** | 67,584 **over** | **58,368 OK** | **yes** | **yes** |
| 128 | 81,920 over | 69,632 over | yes | yes |

So `k = 96` is **illegal today and legal under the Stationary-B budget**
`2mk·T_in + kn·T_in + 2mn·T_out` that `CLAUDE.md` trap 3 records from
ICPP '25 — the paper that says Stationary-B
beats the Stationary-C algorithm our stock IRON matmul uses.

Using [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s fit, `k` 64 → 96
took `ffn_up` from **4,198 → 3,267 µs, 1.28×** — but that assumed the
per-iteration cost is fixed.

**T16 answered the condition, and it fails**
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): 78% of the
iteration is compute that scales with `m·k·n`, so `k = 96` amortises only the
~1,753 non-vector cycles — **≈1.08×, not 1.28×**. Still positive, and it
stacks with nothing else on the ledger, but it now has to beat its own costs:
single-buffering B risks losing fetch/compute overlap (which would eat the
8% directly), and `k` is part of `gemm_b_layout`, so the `.npue` repacks.

**A cheaper mechanism for the same 1.08×, 2026-08-19**: [ATB
(2511.16041)](https://arxiv.org/abs/2511.16041) reaches the L1 relief by shrinking **A's
buffered M** (its lifetime is one C-row accumulation) instead of
single-buffering B — everything stays double-buffered, so the overlap risk
above disappears. At our (64,64,48), ρ=2 on A frees 8,192 B. If this thread is
ever built, build it ATB-shaped, not Stationary-B-shaped.

<a id="t20"></a>
### T20 — int8 · **ANSWERED 2026-08-22: BUILT, SHIPPED BEHIND A FLAG, BOTH GATES PASSED** ([`0077`](../tasks/0077-m13-int8-gate/TASK.md), [`0078`](../tasks/0078-m13-int8-accuracy/TASK.md))

| | int8 | bf16 (default, unchanged) |
|---|---|---|
| array datapath, traced, 4 production shapes | **5.5–7.7×**, bit-exact | 1.00× |
| golden `1-cos` vs HuggingFace | **1.417e-03** PASS | 1.086e-05 PASS |
| **M8 MTEB gate** | **mean −0.03, worst −0.13 — PASS** | +0.04 |
| end-to-end throughput (MiniLM, 4 lanes) | **1086.7 seq/s** | 985.3 seq/s |
| container | 58.68 MB | 69.02 MB |

**The whole quantisation question costs 0.03 MTEB points.** On fidelity int8 looks 130×
worse than bf16 and only 1.4× inside its gate; on the measure that describes what embeddings
are *for* it is a wash, and the clustering task came out ahead. That is
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md)'s point about which gate is the authority,
demonstrated again.

**It needs SmoothQuant, and the standard fold is unavailable.** Naive W8A8 fails at
2.864e-03. Statically-calibrated smoothing at **α=0.5** passes. The factor cannot be folded
into LayerNorm because **BERT is post-LN** — each LN output feeds the residual as well as the
GEMM — so it ships as container data and the runtime applies it inside the quantisation pass
that already reads every element of A.

**End to end it scales with the model** ([`0079`](../tasks/0079-m13-int8-why-only-1.1x/TASK.md)):

| model | bf16 | int8 | gain |
|---|---:|---:|---:|
| MiniLM-L6 (6 layers, h=384) | 985.3 | 1090.0 | **1.10×** |
| bge-large (24 layers, h=1024) | 60.1 | 86.3 | **1.44×** |

**Why not 7×, measured at every link.** The arithmetic *is* 5.5–7.7×. The **dispatch** is
1.79× (3084 → 1727 µs on MiniLM), and 0048's model accounts for the gap exactly: applying 7×
to the variable term only predicts 961 µs, and the missing ~766 µs is **the C drain** —
679 MB per MiniLM encode, and int32 C is 4 bytes exactly as fp32 was, so int8 moves not one
byte less. **The binding constraint moved from arithmetic to transport.**

**BUILT, 2026-08-22** ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)) — and the
diagnosis above needed correcting twice on the way. The C drain is real but the *reason* is
bigger than "the drain": int8 moved the whole GEMM back into the **traffic-bound** regime, so
0010's model governs it at R² 0.987 while 0048's iteration model drops to 0.568. See
[T1](#t1).

`narrow_i32_bf16.cc` narrows C **on the core**, and the key point is that it needs **no
third input stream** — trap 3b forbids one, which is why applying `wscale[j]` on the core
(the plan recorded above) is *not* available. int32 → bf16 is a pure format conversion; the
host keeps the rank-1 `sa[i]·wscale[j]` and reads half the bytes.

| | | |
|---|---:|---:|
| four production dispatches | 9839 µs | **7383 µs = 1.333×** |
| MiniLM, 4 lanes | 1104.2 | **1204.7 seq/s** |
| bge-large, 4 lanes | 86.3 | **93.1 seq/s** |
| MiniLM `1-cos` | 1.178e-03 | **1.161e-03** (better) |

**Accuracy is free here and was not on bf16** (0045 cost 1.38–1.52×): the int32
accumulator's low bits sit under the int8 quantisation noise already in the operands.
Priced *before* the kernel existed, with `npuembed --sim-c-bf16` rounding the accumulator in
the host dequantiser — which predicted 1.161e-03, the number the built design measures.

**"Quantise earlier" was tested and half of it is impossible.** A static (calibrated)
activation scale — which would remove the per-row reduction and make dequant one per-column
FMA known at pack time — measures **6.693e-02** against per-row's 1.166e-03. At seq 64 most
rows are padding, so one tensor-wide scale is set by the largest row and quantises the rest
toward zero. What survives: the per-row absmax is a reduction over a row **LayerNorm already
reduces over**, so LN could emit a second int8 output at nearly zero cost.

**bge-large FAILS the 1-cos gate**, and α cannot save it: 0.3 → 4.475e-03, 0.4 → 3.537e-03,
**0.5 → 3.127e-03 (best)**, 0.65 → 1.892e-02, 0.8 → 1.979e-01. It misses 2e-03 by 1.56×.
24 layers accumulate what 6 absorb, and higher α hurts more because bge-large's weights are
already the harder operand. **MTEB on bge-large int8 is unrun** and is what would settle it
(0035's precedent: bfp16 failed 1-cos at 3.47e-03, which is where this sits).

*(the case as it was assembled, kept because it is what motivated the work)*


**Measured, not predicted.** Same shape [512,384,384], same 4 columns, same session, hardware
trace:

**At PRODUCTION geometry** — pre-tiled B through the same `tile_b()` that packs the `.npue`,
the four real MiniLM GEMM shapes, traced at 4 columns, one session, MACs/cycle/core:

| shape | bf16 | **int8, tile (64,64,48)** | int8, tile (64,96,48) |
|---|---:|---:|---:|
| qkv | 27.8 | **203.3 (7.30×)** | 158.8 (5.70×) |
| attn_out | 26.3 | **203.1 (7.72×)** | 249.8 (9.49×) |
| ffn_up | 27.7 | **153.2 (5.53×)** | 183.3 (6.62×) |
| ffn_down | 25.6 | **191.1 (7.46×)** | 191.5 (7.47×) |

**Every int8 run is bit-exact** (`rel_fro` 0.00e+00) — the right gate for a datapath with no
rounding in the reduction, rather than a tolerance that would pass a subtly wrong kernel. The
bf16 column (25.6–27.8) agrees with [`0003`](../tasks/0003-m2-bf16-gemm/TASK.md)'s 25.0 and
0049's 28.9.

**k=96 is NOT a uniform win** — +23% on attn_out, +20% on ffn_up, but **−22% on qkv**. That
is [T21](#t21)'s tension made concrete: the one-xclbin architecture needs ONE tile geometry
for every shape, so the production choice stays **(64,64,48)** and its 5.5–7.7×.

**Every prior estimate was low**: whisper-xdna measured 1.33×, the attainable-TOPS ratio
implied 2.58×, the `mac_dims` geometry suggested 2×. The reason is
[T16](#t16)/[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md) cashed in — the bf16
path runs at ~100% of the **fp32 vector datapath** while the **MMAC unit sits idle**, so this
comparison is not "int8 vs bf16 arithmetic" but "the MMAC unit vs the fp32 vector unit".
[note 0007](notes/0007-unused-iron-surface.md) §3.5's explanation of the 1.33× as a movement
bound **predates 0048/0049 and does not describe this machine.**

**And it compounds with a tile T19 could not afford.** int8 halves the operand bytes, so
`(64,96,48)` fits **double-buffered** at 46,080 B where bf16 needs 67,584 and does not.
[T19](#t19) closed *negative* precisely because k=96 forced single-buffered B and the exposed
fill grew the inter-window gap 84 → 1,193 cycles. It is worth a further **1.25×** here. k=96
divides every K in this project's model set (384, 768, 1152, 1536, 3072).

**What is NOT established** (0077 §5, and these matter):

- **END TO END IT IS WORTH AT MOST 1.5–2.2×, NOT 7×.** 0048's 573 µs fixed cost per dispatch
  does not shrink because the arithmetic got faster, and neither does host eltwise or the C
  readback (int32 C is 4 bytes, so [`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md)'s
  transport saving does not apply). Amdahl on this project's own measured splits: MiniLM
  40.3% array → **≤1.53×**, EmbeddingGemma 48.9% → **≤1.72×**, bge-large 60.8% → **≤2.20×**,
  and those are generous upper bounds. **The value is that the array stops being the
  bottleneck**, after which [T34](#t34) §3 (host) and [T28](#t28) (fusion) become the whole
  game — the same Amdahl curve [2608.18182](https://arxiv.org/abs/2608.18182) measured from the other
  side, where accelerating GEMM grew LayerNorm's share 7.5×.
- **Per-core cycles are not seq/s.** The encode still pays 0048's 573 µs/dispatch, host
  eltwise, and transport — and with int32 C at 4 bytes, none of
  [`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md)'s readback saving applies. A 7×
  faster array makes the host share *worse*, which is the Amdahl point
  [2608.18182](https://arxiv.org/abs/2608.18182) measures directly (LayerNorm's share grows 7.5× once
  GEMM is accelerated).
- **No accuracy claim at all.** Bit-exactness is about integer arithmetic, not about whether
  an int8-*quantized* model embeds well. SmoothQuant → int8 container → scale handling
  (power-of-two, per [2608.13756](https://arxiv.org/abs/2608.13756)) → MTEB is the entire remaining
  risk, and none of it is done.

**2026-08-22, later the same day ([`0078`](../tasks/0078-m13-int8-accuracy/TASK.md)): the
accuracy half is answered in simulation, and it needs SmoothQuant.**

End-to-end `1-cos` through the numpy oracle (no NPU, no C++ — the reference encoder takes a
`gemm=` primitive for exactly this):

| configuration | worst 1-cos | gate 2e-03 |
|---|---:|---|
| bf16 (sanity check) | 1.380e-05 | PASS |
| **naive W8A8** (per-token act, per-channel W) | **3.397e-03** | **FAIL** |
| W8A8, per-**tensor** activations | 1.794e-02 | FAIL (5.3× worse) |
| **+ SmoothQuant, deployable (qkv & ffn_up only)** | **1.38–1.54e-03** | **PASS** |
| + SmoothQuant on all four GEMMs | 1.051e-03 | PASS |

Naive W8A8 lands within 2% of bfp16's 3.47e-03 — two unrelated schemes converging, which
says the encoder's sensitivity sets the number rather than the scheme. **SmoothQuant is
therefore not optional but it is also cheap**: only `qkv` (after ln1) and `ffn_up` (after
ln2) have a LayerNorm to fold the per-channel factor into, and smoothing just those two
recovers most of the benefit at **zero runtime cost** — the factor lives in the gamma/beta
the packer already writes.

**The headroom is thin.** 1.38e-03 against a 2e-03 gate is 1.4×, and those smoothing factors
were derived from the activations of each call — an upper bound on what a static offline
calibration achieves. That is the number most likely to move when it is built for real.

**The flag is built** (0078 §6): `--int8` on the exporter, `a_dtype` in `design.json`, `I8`
in the container format, `add_gemm_b_int8()` in the packer, and an int8 design set at
`runtime/artifacts_int8_mini`. bf16 remains the default and is untouched. The int8 design's
`b_layout_hash` differs from bf16's, so a mismatched container is refused by the check that
already exists rather than read as garbage.

**Still unbuilt**: the runtime quant/dequant (per-row absmax in, `sa[i]·s[j]` folded into the
existing bias pass out), the packer's SmoothQuant fold, and **MTEB** — which
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md) makes the actual authority here, not 1-cos.

**The question has changed shape twice today**: from "is int8 fast enough to bother with"
(yes, 5.5–7.7×) to "does an int8-quantized encoder still embed well" (yes in simulation, with
SmoothQuant, with 1.4× headroom) to **"does it hold on MTEB, on more than one model, with a
static calibration"**.

---

*(the case as assembled before the measurement, kept because it is what motivated the probe)*


**2026-08-22 update.** This thread has sat as "a datapath lever, accuracy is an MTEB
question" since T16 promoted it. The August 2026 paper batch answered its three open parts at
once, and re-reading it against 0048/0049 turns up a fourth thing nobody had noticed.

**1. The datapath number, and why our own note may have priced it too low.**
`_MM_MAC_DIMS["aie2p"]` gives int8 **(8,8,8)** against bf16's **(4,8,8)** — twice the `r`,
verified in the installed wheel ([note 0007](notes/0007-unused-iron-surface.md) §3.5). The
attainable-GEMM figures this project plans against are **38 TOPS int8 against 14.71 TOPS
bf16** (2512.13282), i.e. **2.58×**; Estévez measured 56.28 of 58.98 TOPS int8 (95%) across
all 32 tiles.

whisper-xdna measured only **1.33×** on real projection and MLP GEMMs, and note 0007
explained the shortfall as *"the same movement bound we have measured everywhere else."*
**That explanation predates [T1](#t1) and [T16](#t16), and those two contradict it for our
configuration.** 0048 showed the plain-bf16 path is not bandwidth-bound (identical MACs, 1.5×
the bytes, 1.8% the time), and 0049 showed the DMA idles in the compute's shadow while the
GEMM runs at ~100% of the **fp32 vector datapath's** 32 MACs/cycle/core — *with the MMAC unit
sitting idle*. int8 moves the work onto that idle unit. So the honest range here is
**1.33× (measured elsewhere, on a machine whose bound we do not share) to ~2.6× (attainable,
this SKU)**, and our own reason for expecting the low end no longer holds.

**2. The accuracy blocker had a name, and now has an answer.** 2209.13325 (indexed) documents
the **±50–100 outlier dimensions in post-LN BERT** that force fp32 LayerNorm and **rule out
per-tensor int8 activations**. That is precisely the problem **SmoothQuant** exists to solve —
it divides activations by a per-channel smoothing factor and multiplies the weights by the
same factor, moving the outliers from the tensor that is hard to quantise into the one that
is not, leaving the linear layer mathematically identical.
[2608.18182](https://arxiv.org/abs/2608.18182) (Intel) reports it working on **BERT, DistilBERT and
XLM-RoBERTa** — our model family — with *"negligible, in some cases no measurable"* accuracy
loss, and it is **upstreamed into PyTorch and TorchAO**. Calibration needs a small dataset.
We would run it in `.venv-ref` and export int8 weights the way we already export bf16; the
quantisation science is not ours to build.

**3. The implementation bug is pre-located.** [2608.13756](https://arxiv.org/abs/2608.13756) proves that
for shared int8 operands with no INT32 overflow the accumulator is **exact and
order-independent** — so divergence between two correct-looking int8 kernels lives entirely
in **scale application and output rounding**, and is removed by **rounding scales to powers of
two** (196/196 and 252/252 layers, pre-registered). On our array that is the SRS/narrowing
step, which is exactly where CLAUDE.md trap 2b already burned this project once (`floor`
rounding was the entire implementation error of all three eltwise kernels). **Plan
power-of-two scales from the start.**

Note the pleasant consequence: int8 x int8 -> int32 has **no rounding in the reduction at
all**, unlike bf16-in/fp32-out which rounds every operand. int8's error is entirely
quantisation plus scale application — a different error *structure*, not merely a different
magnitude.

**4. THE PART NOBODY HAD NOTICED: int8 halves the operand bytes, which un-blocks a tile
geometry [T19](#t19) died for want of.** The L1 budget is
`2·(m·k·in + k·n·in + m·n·out) < 64512`. At in=1 instead of 2:

| tile (m,k,n) | bf16 (in=2) | int8 (in=1) |
|---|---:|---:|
| (64, 64, 48) — production | 53,248 ✓ | **38,912 ✓** |
| (64, 64, 64) | 65,536 ✗ (over by 1,024) | **49,152 ✓** |
| (64, 96, 48) double-buffered | 67,584 ✗ | **46,080 ✓** |
| (64, 96, 64) | — | 57,344 ✓ |
| (64, 128, 64) | — | 65,536 ✗ |

**T19 closed NEGATIVE for a capacity reason, not an arithmetic one.** It measured k=96 at
5.2% worse per MAC, and diagnosed it precisely: vector cycles scaled perfectly (exactly 1.5×,
still 8 cyc/MMAC) but **B had to be single-buffered**, and the exposed fill grew the
inter-window gap **84 → 1,193 cycles**. At in=1, k=96 fits *double-buffered*. And **k=96
divides every K in this project's model set** — 384, 768, 1152, 1536, 3072 are all multiples
of 96 — so the k-block count drops 1.5× across the board.

That is a **compounding** claim: fewer iterations (1.5×, if the fill was the whole story) on
top of more MACs per cycle (1.33–2.6×). It is also a **prior, not a measurement**, and the
1.5× half rests on an inference about why T19 failed rather than on a re-run.

`(64,64,64)` is the other newly legal point but is less useful: it needs `N % 512 == 0`, and
none of our N-sets satisfies that today — though [`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md)'s
padding trick is exactly the tool for it, at a cost that would have to be weighed.

**What is actually unknown**, and should be probed before any of this is believed:

- **Does `gemm_pretiled.py` build at all with `dtype_in_str="i8"`?** It is parameterised, but
  every design this project has ever built is bf16, and the int8 path through IRON's matmul
  kernel, the ObjectFifo dtypes and the C accumulator (int32 rather than fp32) is untested
  here. **That probe is the gate on everything above** and it is one build.
- Whether the 22% non-vector in-window share 0049 measured scales down with the vector work
  or stays fixed — if fixed, it caps the datapath win the way it caps every other lever.
- Whether SmoothQuant's per-channel activation scales survive our *pre-tiled offline* B
  layout, which is a packer question, not a kernel one.

**Do not read this as "int8 is worth 4×."** Read it as: the three reasons this thread was
parked have all been answered by other people, the geometry is verified in our own installed
toolchain, and there is now a specific one-build experiment that separates the whole story
from a fantasy.

---

*(original filing below)*

[note 0007](notes/0007-unused-iron-surface.md) §3.5. `_MM_MAC_DIMS["aie2p"]`
gives int8 **(8, 8, 8)** against bf16's **(4, 8, 8)** — twice the `r` — and
whisper-xdna measured **1.33×**, not the 2× the geometry suggests.
[T16](#t16)
sharpened why this matters: the production GEMM runs the fp32 vector datapath
at its 32-MACs/cycle limit while the MMAC unit sits idle, so **changing
datapath is the only identified multi-× lever left on array time** (the other
route is bfp16 emulation, [T23](#t23)).
int8 would cut *cycles per iteration*, not iteration count. Accuracy is an
MTEB question, and [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) is the
precedent for how to settle it.

<a id="t22"></a>
### T22 — mlir-aie 1.4.1 · **ANSWERED 2026-08-20** · Python design surface migrated and independently re-verified; the one tool-side regression is now fixed and verified
[note 0007](notes/0007-unused-iron-surface.md) §3.6. We are on 1.3.4;
phoenix-sdr-dsp pins 1.4.1 and documents that the rolling wheel channel silently
resolves back to 1.3.4. Not urgent — every feature note 0007 lists is already in
our 1.3.4 — but the gap will widen.

**The upgrade happened** ([`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md)):
`C:\dev\mlir-aie` moved to `main` (mlir-aie 1.4.2.dev16+g7e00b57), which
substantially rewrote `aie.iron`'s `Runtime`/`ObjectFifo` surface (see
[note 0008](notes/0008-iron-1.4-migration.md) for the concrete before/after
patterns) and broke all 15 of our `experiments/`/`tools/` files that
constructed a `Runtime`. A first session migrated all 15 and hardware-verified
one (`exp2_probe.py`) before being interrupted; **a second session
independently re-ran the other 14 from scratch and confirmed 14 of 15 are
bit/value-exact against the historical numbers already on record** in earlier
task logs — not just spot-checked, every printed number in 0058's table was
re-derived this session, not read off the first session's claims: 335/541,662-cycle
SAXPY (1,617×, exact), 25.0→137.2 MACs/cyc bf16→bfp16 single-core and 141.7
MACs/cyc 4-column/16-core whole-array GEMM (exact against CLAUDE.md's M2
headline), all three eltwise kernels' floor/RNE error pairs (GELU
4.312e-03→2.494e-03, LayerNorm 3.326e-03→2.059e-03 with implementation error
3.659e-03→3.967e-05, softmax 4.278e-03→3.325e-03 with the exact row-sum
mechanism — min 0.994581, max 1.000000 under floor), `gemm_pretiled.py`'s
140.9 MACs/cyc rowmajor ffn_down at 0.0% spread over 3 runs (exact against
task 0007's "140.9 (0.1%)"), the four M10 pipeline/join probes (3.550e-08 /
3.474e-08 family, `pipeline_gemm_gelu_probe.py`'s real-GELU 9.052e-04 exact),
and `build_passthrough.py`'s fwd/rev order-independence — **67 of 23,304
xclbin bytes differ, all outside configuration**, the same 67-byte count task
0029 recorded (the total xclbin size differs only because the two
measurements are from different toolchain states; the invariant that matters,
67 identity-metadata bytes and zero configuration bytes, is exact). The 15th
(`experiments/m7-unified/unified_design.py`) migrated at the code level
(including solving a genuine execution-order incompatibility — the old API's
`rt._fifos` idle-endpoint-pinning hack ran too late under the new lazy-body
model and had to move into `Runtime`'s `fn_args` instead) but **cannot be run
to completion under either toolchain version** — it compiles cleanly through
30 of 37 `aiecc` stages (every Python/IRON construction step succeeds) and
then hits the pre-existing, already-documented
([`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md)) 16 KB
program-memory overflow the file's own header says it was abandoned for,
reproduced with the *identical* `XAie_LoadElf failed with XAIE_INVALID_ELF`
error. Not a migration regression; the design has never run to completion.

**A new, DIFFERENT instance of the marker-specificity fail-open class turned
up during re-verification**, inside `@iron.jit`'s own cache lookup rather
than any script's `purge()`: running `cross_column_join_probe.py`'s N=8
column case right after its N=2 case (separate processes, no in-memory
state) returned a stale N=2-compiled kernel — `RuntimeError: Tensor argument
'A' has 32768 elements but the kernel was compiled for 8192 elements` — even
though the file's own `markers_for()`/`purge()` correctly found 0 stale
candidates for the (first-time) N=8 marker. A full `rm -rf ~/.npu/cache/*`
before the retest produced the correct, previously-documented compiler error
(`"tile (0, 1) requires 8 input/1 output DMA channels, but only 4 input/4
output available"`). Fourth documented instance of this fail-open class in
this project (0030, 0053, 0054 are the other three) — this one is upstream
of any of this project's own `purge()` implementations, so no script-level
fix is possible; the working mitigation is to purge fully rather than trust
a script's scoped marker match when switching configurations within one
session.

**FIXED and verified** ([`0060`](../tasks/0060-m11-export-gemm-rtp-marker-fix/TASK.md)):
`tools/export_gemm_rtp.py`'s `markers_for()` cache-marker string match was
broken by the unrelated MLIR pretty-printer format change described above —
the sequence-body `aie.dma_bd` op's textual form changed from bracket-tuples
(`<size = 64, stride = 48>`, still used for `ObjectFifo` `dimensionsToStream`
attributes) to a flat `sizes = [...] strides = [...]` array — so the
substring `f"<size = {k}, stride = {n}>"` the marker relied on no longer
appeared anywhere in freshly-built `aie.mlir`. Confirmed by building all four
production shapes and grepping their `aie.mlir`: B's (`%arg1`) `aie.dma_bd`
always ends its access pattern with the tile dims as the last two entries of
`sizes`, immediately before `strides` (`sizes = [.., .., 64, 48] strides =
[..]`), exactly twice per build (the ping/pong pair) and nowhere else in the
file. `markers_for()`'s second marker is now `f"{k}, {n}] strides = ["` — the
direct translation of the old marker into the new textual form, same two
numbers, same adjacency requirement, so it stays as specific as the six prior
marker-fail-open fixes demanded rather than merely permissive.
**Verified on a fully purged cache**: the small case (`--batch 4 --cols 2`)
now finds exactly 1 candidate per shape and completes with all identity
checks `OK`; the real production shape (`--batch 128 --cols 8`, the `0032`
recipe) rebuilds into a separate directory
(`runtime/artifacts_verify_t22fix/`, gitignored) and, combined with the
shipped weights/manifest/eltwise designs, **reproduces `0038`'s and `0059`'s
exact historical numbers on hardware**: golden-vs-`.npue` `1-cos` **1.086e-05**
(identical) and `verify_embed_e2e.py` worst `1-cos` **2.644e-05** (matches to
4 s.f.), both `PASS`. `runtime/artifacts_b128il/` itself was never modified.
`gemm_pretiled.py` (which `export_gemm_rtp.py` imports `pretiled_array`
from) was already confirmed correctly migrated in `0058` — the break, and
now the fix, is entirely in `export_gemm_rtp.py`'s own string-matching, not
in the Runtime migration.

<a id="t24"></a>
### T24 — 0048's iteration fit missed bge-base by 27% · **ANSWERED 2026-08-20**
**The fit was fine; the miss was the host + overlap term the prediction
ignored.** [`0052`](../tasks/0052-m10-research-night/TASK.md) §1 ran the one
command: `--probe-streams` on the h=768 design. The discriminating pair
reproduces (identical-MAC shapes 0.3% apart across 1.27× bytes), marginal
per-iteration is **4.49 µs against the h=384 fit's 4.72 (−5%)**, and the
array account closes exactly: predicted 1,053 ms of array time for two
pipelined lanes against a measured 1,047 ms NPU share. The wrong move in
0051 was predicting *end-to-end* from an *array-only* model. The array fit
is now validated at both widths; what has no model is the host side.

<a id="t27"></a>
### T27 — The emulated datapath is traffic-bound: which byte-lever pays first? · **ANSWERED 2026-08-22 for int8**, which replaced bfp16 as the fast datapath

**int8 arrived and made this thread live on a path that ships.** bfp16 was
never adopted (T23 is an accuracy decision nobody took); int8 is bit-exact and
is in the tree, and [`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)
confirms this thread's whole premise on it — 0010's traffic model fits at
R² 0.987 while 0048's iteration model drops to 0.568.

**Which byte-lever paid, in the order this thread listed them:**

| lever | verdict |
|---|---|
| host readback / **narrow C** | **BUILT. 1.333× on the four dispatches**, accuracy free (0080) |
| **bigger tiles** | **BUILT for bge-large. `tile_n = 64` is legal at int8's 1-byte operands and over budget at bf16's — 1.366×** ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md)) |
| bigger tiles, the `k` axis | **REFUTED. `k=128` fits int8's L1, halves the iterations, measures 0.989×** — it is `tile_n` that matters, because A is re-streamed `N/(n·cols)` times |
| B-reuse | still channel-blocked (0046/0047); see [T2](#t2), reopened |
| A in bf16→bfp16 offline | superseded — int8 already halves A |

**AND THIS THREAD'S LAST SENTENCE WAS THE IMPORTANT ONE.** It warned that *"at
pipeline 4 the bfp16 encode is 47% NPU busy, so array levers beyond 2.2× buy
little until T3-class host work lands"* — written 2026-08-20, and then four
tasks in a row optimised the array anyway. 0081 §3 measures the same thing on
int8 and it is worse than the warning: **bge-large is 30.4% NPU busy, so the
array's remaining headroom is capped at 1.44× no matter what is done to it.**
The host is 69.6%, and almost all of it is memory traffic rather than
arithmetic — four to six separate streaming passes over a 33.5 MB activation
tensor between two GEMMs, ≈17 GB per encode.

**Successor: [T37](#t37).** The byte-lever question has moved off the array.

> **PREMISE EXPIRED — see [T48](OPEN-THREADS.md#t48), 2026-08-27.** This thread
> is closed *"for int8, which replaced bfp16 as the fast datapath"*, and its
> reasoning rests on the sentence above: *"bfp16 was never adopted (T23 is an
> accuracy decision nobody took)."* That was true when written.
> [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md) and
> [`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) then took the decision,
> and **0.4.0 ships bfp16 on five of six models with int8 behind a flag** — the
> exact reverse. The levers priced above were priced on int8 traffic, whose
> composition differs from bfp16's. Nothing here is deleted or rewritten: the
> reasoning was sound for the configuration it was written against, and the
> class of failure is [T49](#t49).

<a id="t37"></a>
### T37 — The host epilogue chain is four passes over one tensor · **BOTH CHAINS FUSED 2026-08-22** ([`0082`](../tasks/0082-m13-fused-ffn-epilogue/TASK.md)) · attention is what is left

**Built, and byte-for-byte identical to the unfused path** on MiniLM, bge-base
and nomic, and across lane counts. `--no-fuse-ffn` keeps both paths runnable so
they can be A/B'd.

| chain | was | now |
|---|---|---|
| `ffn_up` → activation → `ffn_down` | 6 streaming passes over `rows × inter` (134 MB at bge-large; 268 MB gated) | 1 read of C, 1 write of the int8 operand |
| add → LayerNorm → residual → quantise, **twice per layer** | 8 passes over `rows × hidden` (33.5 MB) | 4 |

**Measured clean, one session, `--threads 24 --pipeline 4 --bench 3`** — an
earlier A/B under CPU load read 1.19× and was low by a third, because
contention compresses exactly the term the fusion removes:

| | fused / unfused | vs bf16 |
|---|---:|---:|
| MiniLM-L6 | 1.394× | **1.65×** |
| bge-small | 1.417× | **1.71×** |
| bge-base | 1.418× | **1.94×** |
| bge-large | 1.428× | **2.66×** |
| **nomic** (gated) | **1.482×** | **2.04×** |
| Gemma (arch=1, own harness) | — | 1.08× |

**Re-measured with three runs per arm
([`0084`](../tasks/0084-m13-host-isa-and-repeats/TASK.md)): 1.399 / 1.406 /
1.467 / 1.391.** The ratios above were single-run and hold to 3%; the
absolutes they came with are ~4% low.

nomic gains most, which is the traffic model's own ordering — its gated
`ffn_up` output is `2 × intermediate`. Against
[`0079`](../tasks/0079-m13-int8-why-only-1.1x/TASK.md)'s 1.10× / 1.44×, where
this line of work started. Run-to-run spread is ~4%, and [T18](#t18) is the
larger caveat: probe and bench disagree by up to 10% and nobody has explained
why. These are all `--bench`, so the A/B is within-instrument.

**A second constant falls out of it: the host side runs at ~60 GB/s.** The
pass-counting model predicts each model's saving to within ~15% once priced at
the host's own DRAM rate rather than [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)'s
25 GB/s, which measured what reaches the **NPU through the shim**. Implied
bandwidth from traffic-removed ÷ time-saved: 64.8, 62.1, 59.6, 70.3, 56.1 GB/s
across five models spanning 10× in size. Two buses, two constants — using one
for the other briefly made a correct model look broken.

**The transferable finding is why the first attempt was NOT identical.** Scalar
inner loops computing the same algebra moved `1-cos` from 1.161e-03 to
1.180e-03 — inside the gate, and easy to wave through as noise. It is not
noise: `_mm256_fmadd_ps` rounds **once** where `a*b + c` rounds twice under
`/fp:precise`. **Matching the algebra of vectorised float code is not enough;
the intrinsics have to match.** That is why the activation is a *parameter* of
the fused helper rather than a branch inside it — the caller keeps ownership of
the exact instruction sequence its unfused twin uses.

Also caught, by the byte-comparison harness that existed by then: `s_ln[...]`
is a design **slot**, `h_gamma`/`h_beta` are indexed by **site**, and
`layer_norm()` converts with `slot - 1`. Passing the slot through segfaulted —
but it could as easily have read a valid neighbouring gamma and returned
plausible numbers, the same "one field to the left" shape as [T31](#t31).

**What is left in this thread: host attention at 14.6%**, which nothing has
touched. Note that CLAUDE.md's F3 prices attention folding at **1.4%
end-to-end** — a *bf16-era* number from AMD, measured when the host was a far
smaller share of the encode. It should be re-derived before it is quoted again.

And **arch = 1** uses a separate encoder; its GeGLU epilogue fuses with the
same helper, its RMSNorm chain (no beta, four norms per layer) would need its
own pass.

**PORTED TO THE bf16/bfp16 DATAPATH 2026-08-25
([`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md)).** Everything above was
gated on `a_elem_bytes == 1` — **int8 only** — so from
[`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) onward this fusion never
fired on the datapath five of six models actually run.
[`0107`](../tasks/0107-t3-t28-pricing/TASK.md) is what noticed, while pricing
[T28](#t28): the cheaper lever was also the unbuilt one. 0108 removed the gate by
extending this thread's own `FusedNext`/`gemm()` mechanism rather than adding a
parallel one, having first **measured** the addressable share at **17.7–28.3%**
of wall clock instead of assuming it. Result, A/B at three runs per arm on an
idle machine: **1.265 / 1.208 / 1.251 / 1.223 / 1.340 / 1.153×** (MiniLM,
bge-small, bge-base, bge-large, nomic, gemma). Bit-identical, verified the two
ways this thread's own lesson demands — the embedding vectors hash the same
fused vs `--no-fuse-ffn`, and every model reproduces
[`0105`](../tasks/0105-release-sweep-adopted-datapaths/TASK.md)'s `1-cos` to the
digit. arch=1's RMSNorm chain is **still** unfused, exactly as the paragraph
above left it, and 0109 measured what that costs: gemma's 1.153× is the
smallest win of the six.

**And it moved the bottleneck back.**
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md) measured array time unchanged
by the fusion (`wait (hardware)` −0.1% to +2.2%) but its **share** risen on every
model — bge-large 46.4% → 56.7%, array-infinite ceiling 1.87× → **2.31×**. This
thread's line of work, which started at
[`0079`](../tasks/0079-m13-int8-why-only-1.1x/TASK.md)'s 1.10×, has now largely
spent the host-side lever.

<a id="t37-2"></a>
### T37 (original filing, kept for the reasoning) · 2026-08-22

Between two GEMMs the host does **dequantise → add residual → LayerNorm →
quantise**, each a separate streaming pass over the same tensor. At bge-large's
batch 128 / seq 64 / hidden 1024 that tensor is 33.5 MB, and the FFN
intermediate is 134 MB.

They can be **one** pass. A row is 4 KB and fits L1; LayerNorm already reduces
over exactly that row, so a third per-row statistic (the quantisation absmax)
is nearly free; and LN can write **two** outputs — fp32 for the residual and
int8 for the next GEMM. One read and two writes instead of four of each.
[`0078`](../tasks/0078-m13-int8-accuracy/TASK.md) §4a's post-LN objection does
**not** apply: it forbids rescaling the output that feeds the residual, not
adding a second one.

Two related questions this reopens:

* **Is "eltwise on the host" still right?**
  [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md) measured every eltwise
  kernel faster *and* more accurate on the host — **when each NPU dispatch cost
  a design switch**. There is now ONE xclbin in ONE hw_context, the switch is
  gone, and the array idles for 70% of an encode. `epilogue="gelu"` already
  exists in `gemm_pretiled.py`; fusing it into `ffn_up` would delete a
  dequantise, a GELU and a quantise pass over a **134 MB** tensor per layer.
* **Host attention is 14.6%** and has never been optimised, only priced
  ([F3](../CLAUDE.md) put attention folding at 1.4% end-to-end — a *bf16-era*
  number, measured when the host was a much smaller share).

<a id="t27-2"></a>
### T27 (original entry, kept for the reasoning) · conditional on T23
[`0052`](../tasks/0052-m10-research-night/TASK.md) §3. On bfp16 the array is
DMA-bound (GB/s 32–46 flat-ish, GMAC/ms spread; ffn_up slower than ffn_down
by its byte ratio) — the inverse of the plain-bf16 economy 0048 measured. If
T23 reopens, the retired levers re-price on the new path: **B-reuse** (still
channel-blocked per 0046/0047 — cascade or CascadeFlow was the route),
**ATB/bigger tiles** (now amortising DMA, not overhead), **A in bf16→bfp16
offline** (halves A bytes?), and the host readback (bf16-C already in). Also
note the host wall: at pipeline 4 the bfp16 encode is 47% NPU busy, so array
levers beyond 2.2× buy little until T3-class host work lands.

<a id="t25"></a>
### T25 — bge-base has no MTEB gate and no interleaved CPU ratio · **ANSWERED 2026-08-20**
[`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md) validated it to
`1-cos` 1.353e-05 and end-to-end 2.613e-05 with top-10 overlap 1.0000, which is
the correctness bar. It lacked the two things
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md) and
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) established as the bars
for *quality* and for *speed claims*: five MTEB tasks against the CPU on the
same checkpoint, and a round-robin interleaved throughput ratio.

**Both landed in [`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md).**
MTEB, five tasks, seq 64, both NPU datapaths against the same CPU column:

| task | CPU | NPU plain-bf16 | Δ plain | NPU bfp16+bf16C | Δ bfp16 |
|---|---:|---:|---:|---:|---:|
| STSBenchmark | 86.418 | 86.422 | +0.004 | 86.407 | −0.011 |
| SICK-R | 80.301 | 80.299 | −0.002 | 80.286 | −0.016 |
| STS12 | 78.028 | 78.027 | −0.002 | 77.984 | −0.045 |
| Banking77Classification | 83.984 | 83.981 | −0.003 | 83.925 | −0.058 |
| TwentyNewsgroupsClustering | 50.576 | 50.695 | +0.119 | 50.383 | −0.193 |
| **mean / worst** | | | **+0.023 / −0.003** | | **−0.065 / −0.193** |

Both **PASS** (gate: \|mean\| ≤ 0.5 AND worst ≥ −0.5) — bge-base is not more
fragile than MiniLM under either datapath; bfp16+bf16C's bge-base worst
(−0.193) sits well inside MiniLM's own worst (−0.06 per 0052 §7). Artifacts:
`experiments/m8-npu-vs-cpu/artifacts/mteb_bge_base.json` (plain, self-computed
gate), `mteb_bge_base_bfp_cbf16.json` + `mteb_bge_base_bfp_cbf16_gate.json`
(bfp16 NPU-only run, gate computed by merging against the plain run's already-
recorded CPU column rather than re-running CPU a second time — `run_mteb.py
--sides npu` does not self-compute a delta, since its gate logic reads
`results["cpu"]`, which is absent from a NPU-only run; noted as a real, minor
gap in the harness, not fixed this session).

**Interleaved CPU ratio (0040 protocol, mains power, `compare_three.py
--rounds 8`, artifacts_base / plain bf16 only)**: torch 111.2 seq/s (steady,
strongest CPU), ORT 55.4 seq/s, NPU 181.5 seq/s → **NPU / strongest CPU =
1.633×**. Machine state recorded: `Online / NoSystemBattery / Balanced`.
Artifact: `experiments/m8-npu-vs-cpu/artifacts/compare_bge_base.json`. (The
script crashed *after* writing the artifact, in its final cosmetic
`print(f"wrote {out.relative_to(REPO)}")` — `out` is a relative `Path`,
`REPO` absolute, so `relative_to` raises `ValueError` on Windows; a real small
bug in `compare_three.py`, not investigated further since the data was
already on disk.) The bfp16+bf16C interleaved ratio was **not** measured this
session (only its MTEB gate) — left for a follow-up night alongside T27.

<a id="t15"></a>
### T15 — `tasks/README.md` has no rows for 0038–0043 · **ANSWERED 2026-08-23**
Noted in [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md). Deliberately not
back-filled from `CLAUDE.md` summaries, because an index entry written from a
summary rather than from the task is the kind of second-hand claim this repo
does not keep. Needs reading the six tasks.

**ANSWERED 2026-08-23** ([`0089`](../tasks/0089-t15-task-index-backfill/TASK.md)):
read from the task logs, not from any summary, per the thread's own condition.
**The debt was 21 rows, not six** — 0038–0043 (T15's own count) plus
0073–0087 (fifteen more tasks that had *also* fallen out of the index and that
T15 never knew about, since it was filed against 0044's narrower observation).
`0072` is correctly not one of them — `tasks/README.md` already documents that
gap by name.

Two contradictions surfaced while reading, both left standing rather than
reconciled in either document, per this task's own instructions: (1) CLAUDE.md
cites [`0043`](../tasks/0043-m9-attention-geometry/TASK.md) for a "not worth
the fight" cost/benefit verdict that 0043's own log never reaches — its
`## Results` section is the literal placeholder `(filled in below)`, and the
task supports only the structural `cols ≤ 4` argument, not a measured
tradeoff (see [T4](#t4), which already says the same about
0043's missing measurements). (2) CLAUDE.md's "1.10× on MiniLM... the int8
quantisation pass costs more than the bf16 conversion it replaces" is the
finding of [`0079`](../tasks/0079-m13-int8-why-only-1.1x/TASK.md), which
0079 itself found is the *wrong* mechanism (quantisation is 1.04× bf16's
cost, not more expensive; the real bottleneck is the C drain) — and both the
1.10× figure and the 60.8% bge-large array-share figure CLAUDE.md quotes are
superseded by 0080–0085's fused-epilogue and int8-everywhere work (MiniLM
int8/bf16 now 1.71×, bge-large 2.39×, bge-large tile_n now legally 64 not
32). Full detail in [`0089`](../tasks/0089-t15-task-index-backfill/TASK.md).

<a id="t11"></a>
### T11 — Is the hw_context partition width settable at creation? · **RETIRED 2026-08-23**
[note 0004](notes/0004-context-switch-cost.md) §1 and
[`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md). Decided in the
driver's create-hwctx call, not at XRT level. Low value now that production is
one xclbin at a fixed width.

> **RETIRED 2026-08-23 ([`0093`](../tasks/0093-t11-t12-t13-research/TASK.md)):**
> **yes** at the driver-ioctl level, on Linux — `struct amdxdna_drm_create_hwctx`
> has an explicit `__u32 num_tiles` input field passed to
> `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` (a reverse-engineered struct, corroborated by
> five independent call sites in `externalrepos/Meeting-Ops-UC1-OSS`; Phoenix /
> XDNA1, not our SKU, and the Windows driver is unconfirmed). **No** at the layer
> this project and IRON's own hostruntime actually call: `xrt::hw_context`'s
> public `cfg_param_type` (`xrt_hw_context.h:55`, in our XRT install) is a
> `std::map<std::string, uint32_t>` documented only for QoS/scheduling keys, with
> no topology key at all, and both `runtime/src/npu_device.cpp:231` and mlir-aie's
> `xrtruntime/hostruntime.py` call the **no-QoS constructor**. Partition width is
> baked into the xclbin at compile time (`from_name("npu2", n_cols=...)`), not
> chosen at context creation. Retired as answered-as-far-as-the-sources-allow,
> which matches the thread's own stated low value.

<a id="t12"></a>
### T12 — Larger L2 megatiles · **ANSWERED 2026-08-23**
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §3, citing
[2602.06063](https://arxiv.org/abs/2602.06063)'s 5.9 → 13.7 TOPS from megatile size alone.
Was filed as a *bandwidth* lever, which T1 has now retired. If the megatile
result is real it must be acting through **iteration count** (bigger tiles) —
i.e. it is [T17](#t17)
in L2 clothing, and the paper is worth re-reading with that question.

> **UNDONE 2026-08-23**: T1's retirement is now datapath-scoped, and on int8
> the traffic model governs (R² 0.987). **This thread's ORIGINAL bandwidth
> framing is the live one there**, and 2602.06063's 5.9 → 13.7 TOPS may mean
> exactly what it said. Re-read the paper with the original question, not the
> re-pointed one.

> **ANSWERED 2026-08-23 ([`0093`](../tasks/0093-t11-t12-t13-research/TASK.md)):
> the paper conflates two levers, and "size alone" is wrong.** Its own table is
> 128×512×512 = 5.9, 256×256×512 = **12.0**, 512×512×512 = 13.7. The first step
> is a **reshape at constant volume** — both megatiles hold 33.5 M elements — and
> it is worth **2.0×**, i.e. most of the headline. Only the second step is a real
> size increase (4× the volume), and it buys a further **1.14×**. The lever is
> aspect ratio first, capacity second.
>
> It is also measured on **bf16**, the wrong dtype for the now-live question
> (int8 traffic-boundedness), though on the same `aie2p`/npu2 family (Krackan
> Point, not our Strix Point HX 370). And the mechanism — amortise each operand's
> L3→L2 fetch over a wider L2-resident block — **already exists in this project
> under two names**: `gemm_pretiled.py`'s `b_reuse="mega"` (B's megatile, blocked
> by mem-tile DMA channels per [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md),
> not by L2 capacity) and `tile_n` (A's megatile axis, reopened independently by
> [`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md) §8 from the traffic model
> alone, before this paper was re-read). Predicted payoff on `ffn_up` at a legal
> `tile_n = 96` against the shipped 48: **~11% traffic reduction, ~1.03–1.05× on
> that shape** — shape-specific (`ffn_down`'s N = 384 cannot take n = 96) and
> small next to [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) §3's finding
> that the array is only 30.4% of wall clock. Not a new discovery: it confirms a
> lever this project had already found from its own data.

<a id="t5"></a>
### T5 — Does DMA compression pay on real weights? · **ANSWERED 2026-08-23**

> This is a **byte** lever, and bytes were free on bf16 (T1) and are the whole
> cost on int8 ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)). B is
> 23–43% of an int8 dispatch's traffic. Worth more than when it was filed, and
> it is still one run.
[note 0007](notes/0007-unused-iron-surface.md) §2. Lossless, unused by
mlir-aie's own passes for the compute-tile path, and the only published ratio is
**1.39× on `arange`**. One run of `basic/dma_compression`'s `cmp_only` with a
real `.npue` tile answers it permanently.

> **ANSWERED 2026-08-23 ([`0090`](../tasks/0090-t5-dma-compression/TASK.md)): no
> — structurally, not merely empirically. The shim DMA has no compression
> hardware.** The control was reproduced first (**1.391× on `arange`**, sha256
> matched against the example's own npu2 golden), and then the vendor's own
> AIE2P register tables were read
> (`third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c`):
>
> | DMA module | `.Compression` |
> |---|---|
> | `Aie2PMemTileDmaMod` (line 1666) | `XAIE_FEATURE_AVAILABLE` |
> | `Aie2PTileDmaMod` (line 1904) | `XAIE_FEATURE_AVAILABLE` |
> | **`Aie2PShimDmaMod` (line 2157)** | **`XAIE_FEATURE_UNAVAILABLE`** |
>
> This is a hardware capability flag on this exact chip, not an mlir-aie
> omission, so there is no software workaround. And **every byte
> [`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)'s traffic-bound cost
> model prices** (`t ≈ 627 µs + traffic / 28 GB/s`, R² 0.987) **crosses the
> shim** — it is the only door from DRAM into the fabric. Compression can
> therefore only ever touch the mem-tile→core-tile leg, which is on-die and is
> not the modelled bottleneck. A perfect ratio applied inside the array moves
> zero bytes off the leg that costs.
>
> **The arithmetic, granting the lever everything it could want** — the control's
> full 1.391× applied to B, and the shim gap ignored: across 0080's four measured
> int8 production dispatches at M = 8192 / 8 columns, the four-GEMM sum falls
> 9,838 µs → 9,269 µs, i.e. **5.8%**, before subtracting the decompression the
> mechanism itself costs. That is the ceiling this lever was ever worth.
>
> **What is genuinely unmeasured**, and recorded as such: the real-weight ratio
> itself. Real B tiles (bf16 and int8, all four shapes) were extracted from
> `bge-base-en-v1.5.npue` and the hardware attempt was blocked by a **toolchain
> bug** — `iron.jit`'s cache key (`_create_function_cache_key` /
> `_generation_cache_key`) hashes call arguments and `(device, full_elf)` but
> **never a generator's own module globals**, so resizing the destination BD via
> a module-level constant silently re-ran the control's compiled config *even
> with `use_cache=False`*. `.as_mlir()` *does* see the change, which isolates the
> staleness to the compile/cache layer. This is the **sixth** instance in this
> repo of the cache-staleness fail-open class (0030, 0053, 0054, 0083, 0087 are
> the others) and the first found in IRON itself rather than in our own marker
> logic. The weakest available ratio signal is a zero-byte-fraction proxy: real
> weights carry **0.23–3.13%** zero bytes against `arange`'s **51.7%**, 17–220×
> less of the raw material this scheme exploits — so the true ratio would very
> likely have been worse than 1.391× even on the leg where it cannot help.
>
> **One free corroboration on the way past**: the same `Aie2PShimDmaMod` struct
> also carries `.Padding = XAIE_FEATURE_UNAVAILABLE`, which independently
> confirms [note 0007](notes/0007-unused-iron-surface.md) §1.1's correction that
> `pad_dimensions` is **mem-tile only** — relevant to [T4](#t4),
> which reopens 0043's `cols ≤ 4` conclusion via exactly that feature.

<a id="t7"></a>
### T7 — `gelu_poly.cc` narrows through an emulated fp32 multiply · **ANSWERED 2026-08-23** · free
[`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) measured
`aie::mul(v, 1.0f).to_vector<bfloat16>()` at **34 `vmul.f`/`vadd.f` per 64
elements** against **0** for `accum::from_vector`. `gelu_poly.cc` uses the
expensive form in two places, and
[`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md) called that kernel "at the
machine's fp32 vector limit" — a limit measured with avoidable emulated ops in
it. Dormant while eltwise runs on the host.

> **ANSWERED 2026-08-23 ([`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md)).**
> Fixed in place rather than as a `*_rne.cc`-style variant, because unlike the
> rounding-mode question the two forms are provably the **same value** — there is
> nothing to A/B. `gelu_poly_impl`'s widen and narrow together: **652 → 580
> `vmul.f`/`vadd.f`, −11.0% of the function** (per-symbol counts from the stored
> disassembly, independently re-counted).
>
> **A third instance the thread did not know about**, in `gelu_poly_f32_epilogue`,
> turned out to be a **dead multiply-by-one** — fp32 in, fp32 out, no conversion
> needed at all — and was deleted outright rather than converted: **648 → 580,
> −10.5%**.
>
> **No shipping kernel carried the pattern.** `narrow_f32_bf16.cc` and
> [`0062`](../tasks/0062-m11-t28-hierarchical-merge/TASK.md)'s
> `gelu_epilogue_3072_f32_to_bf16` were already using the correct form (0-delta,
> confirmed by compiling both before and after). The one live instance found is
> in `ffn_down_hop_matmul_g2.cc`, T28's relay hop matmul, and was **reported to
> that thread rather than fixed here** because the file was in active use.
>
> **One number corrected.** Re-measuring the narrow direction on today's
> toolchain gives **21** emulated ops per 64 elements, not 0045's 34. Same
> conclusion, different count; most likely toolchain drift (0045 predates
> `1.4.2.dev16+g7e00b57`). Recorded rather than reconciled.
>
> Exactness: the widen direction is exact **analytically** — bf16 is a strict
> bit-subset of fp32, so widening has no rounding step — verified bit-identical
> across a real activation tensor. The narrow direction inherits 0045's own
> hardware measurement (100.00% bit-exact) rather than being re-run, because the
> array was in use by another thread.

<a id="t8"></a>
### T8 — `gelu_poly` degree 8 → 7 · **ANSWERED 2026-08-23** · ~9% estimated, 12.4% measured
[note 0007](notes/0007-unused-iron-surface.md) §3.2: degree 8 sits at 3.6e-04
against a 2.465e-03 bf16 floor, degree 7 at 5.9e-04 — still 4.2× inside.
One Horner step of eight. Dormant with T7.

> **ANSWERED 2026-08-23 ([`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md)): the
> direction holds and the numbers do not.** Implemented as
> `gelu_poly_impl_deg7` — a genuine **refit** at degree 7 on the same Chebyshev
> nodes, not a truncation of the degree-8 fit — and kept as a selectable variant
> rather than made the default, following the `*_rne.cc` precedent for a kernel
> that does not execute today. Measured **580 → 508 `vmul.f`/`vadd.f`, −12.4%**,
> against 0007's ~9% timing-derived estimate.
>
> **0007 §3.2's own table does not reproduce.** Its 3.6e-04 / 5.9e-04 survived
> none of three attempted metrics (closest: max-abs error of the fit alone
> against exact `erf`, 1.303e-04 for degree 8 against 0007's 3.613e-04 — 2.8×
> off), and it appears to exclude the **bf16 output rounding step** entirely.
> Under the metric that actually gates a shipping decision — `rel_fro` on real
> `L0.ffn_up` activations, through bf16 output rounding, which reproduces the
> kernel header's own long-standing **2.494e-03** for degree 8 exactly — the two
> degrees are nearly indistinguishable:
>
> | degree | incl. bf16 rounding | fp32 design limit |
> |---:|---:|---:|
> | 6 | 4.571e-03 | 4.316e-03 |
> | **7** | **2.503e-03** | 1.934e-03 |
> | **8** (shipped) | **2.494e-03** | 1.923e-03 |
> | 10 | 2.465e-03 | 1.893e-03 |
>
> **0.36% apart**, not "4.2× inside the floor" — because at this degree the bf16
> output rounding, not the polynomial, is what sets the error. So 0007's
> conclusion (degree 7 is free) is *more* strongly true than its own numbers
> said, for a different reason than it gave. Reproduced independently from the
> task's stored script.

<a id="t6"></a>
### T6 - Does `aie_stream=` free a DMA channel? · **ANSWERED 2026-08-23** · yes, and it is the wrong channel
[`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) closed B-reuse on channel
exhaustion. `aie_stream=(end, port)` makes a producer wire-only with no L1
buffer ([note 0007](notes/0007-unused-iron-surface.md) 1.6); whether it also
costs no channel is unknown and directly relevant.

> **ANSWERED 2026-08-23 ([`0095`](../tasks/0095-t6-t10-probes/TASK.md)): it
> genuinely frees a channel, and it cannot free the one we need.** Proven two
> ways on upstream's own shipped `programming_examples/ml/magika/group2.py`,
> built compile-only with and without the flag:
>
> | | with `aie_stream=(0,0)` | without |
> |---|---|---|
> | core tile (1,3) | in 1/2 - **out 0/2** | in 1/2 - **out 1/2** |
>
> and in the raw MLIR the core's `aie.dma_start(MM2S, ...)` block is **entirely
> absent** rather than merely idle. So the channel is not reserved-and-unused;
> it is not allocated at all.
>
> **But it only ever frees a channel on the marked endpoint, and only a tile
> with an executing core can be that endpoint.** The other end of the same link
> -- a shim tile, which has no program counter -- keeps an unchanged
> `aie.shim_dma_allocation` in both builds. A follow-up probe marking a consumer
> that still used an ordinary `.acquire()` was rejected outright by the
> compiler: `'aie.objectfifo.acquire' op cannot acquire from objectfifo stream
> port`.
>
> And [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) /
> [`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) established that the
> production GEMM's exhausted resource is specifically **mem-tile input**
> channels (the four-core C join), while the cores themselves already have
> output headroom -- 1 of 2 used. `aie_stream` frees core-side channels, which
> are not scarce, and cannot touch mem-tile inputs, which are. **B-reuse stays
> closed**, on the same grounds 0046 closed it, now with the one remaining
> candidate mechanism eliminated by measurement rather than left as an unknown.

<a id="t10"></a>
### T10 - `aie::exp2` never measured · **ANSWERED 2026-08-23** · a real trade, not a clear win
[`0020`](../tasks/0020-m5-layernorm-kernel/TASK.md). Superseded in practice by
our own `exp2_poly` ([`0030`](../tasks/0030-m7-expert-review-tests/TASK.md)),
but the comparison was never made.

> **ANSWERED 2026-08-23 ([`0095`](../tasks/0095-t6-t10-probes/TASK.md)): the
> comparison is made, and it goes the opposite way from the thread's framing --
> `aie::exp2` is far CHEAPER and materially LESS accurate.** `llvm-objdump` on
> `softmax_impl<false>` and `softmax_impl<true>`, same translation unit, same
> flags:
>
> | | instructions | bytes | native op |
> |---|---:|---:|---|
> | `aie::exp2` | **291** | 1,552 | **one inlined `vexp2`**, covering the whole 64-wide row |
> | `exp2_poly` | 765 | 3,696 | none -- a fully unrolled degree-7 Horner chain plus manual IEEE exponent construction |
>
> **2.6x the instructions for the polynomial**, and neither form emits a
> function call, so trap 5's `__mulsf3` concern does not apply to either.
> (Independently re-counted from the stored disassembly: 1 `vexp2` in the
> intrinsic build, 0 in the polynomial build.)
>
> Accuracy runs the other way. A numpy model reproducing `exp2_poly.h`'s
> coefficients bit-for-bit against exact fp64 `2**x`, under AIE's actual default
> `floor` rounding, gives max relative error **7.109e-03** -- which closely
> matches [`0021`](../tasks/0021-m5-softmax-and-full-model/TASK.md)'s own
> hardware measurement of 6.7e-03, validating the model. `aie::exp2` measures
> **1.711e-02** against the same reference: **2.4x worse**.
>
> **Decision: keep `exp2_poly`**, which confirms the choice already shipped in
> 0030 rather than changing anything. The trade stated honestly: this project
> pays 2.6x the instructions to buy 2.4x the accuracy, on a kernel that does not
> execute today.
>
> **And a fifth instance of the `floor`-bias pattern (trap 2b):** under
> `conv_even` the same polynomial would measure **3.702e-03**, **1.92x better**
> than it does under the default rounding mode -- the same shape as the
> GELU / softmax / LayerNorm findings in
> [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3.

<a id="t32"></a>
### T32 — The golden gate is structurally blind to cross-row corruption · **ANSWERED 2026-08-23** (filed 2026-08-21)
**Every "different" row in the validation batch is identical content, so any bug
that reads the wrong ROW is invisible to it.**

`--model X --artifacts Y` is this project's primary on-device accuracy gate, and
it runs the M3 goldens, which are **batch 4**. For a larger design the runtime
tiles those four sequences to fill it (`reps4 = batch / 4`, so **32×** at batch
128 — `main.cpp:2855-2870`). The code is honest that it "makes no accuracy claim
beyond batch 4". What nobody wrote down is the consequence: **a row-indexing or
cross-row-aliasing bug reads identical data whichever copy it lands on**, so it
cannot change the answer.

Demonstrated, not theorised. [`0070`](../tasks/0070-m13-nomic-runtime/TASK.md)
shipped a threaded in-place `swiglu_cpu()` whose write range genuinely raced
another row's read range. The golden gate **PASSED**. A 13-distinct-sentence
end-to-end run failed at worst `1-cos` **0.44**, with rows immediately after a
thread-chunk boundary wrong by rel err up to 1.23 — wrong sign and magnitude,
not bf16 noise — while every other row matched the oracle to ~1e-3.

So the gate that this project leans on hardest has a whole bug class it cannot
see, and the gate that caught it (`verify_embed_e2e.py`) is a separate tool that
is not run on every change.

Cheap fixes, in increasing order of value:
1. **Make the tiled copies distinguishable.** Even permuting the four sentences
   per tile, or scaling copy *i* by a known factor and dividing it back out,
   turns "identical content" into "content that must match its own row".
2. **Generate goldens at a batch that is not 4** — at least one fixture with as
   many distinct sequences as the design's batch, so the tiling disappears.
3. **Run `verify_embed_e2e.py` as part of the standard gate**, since distinct
   texts is exactly the property it has and the golden check lacks.

Until then, treat a golden-gate PASS as evidence about arithmetic and **not**
about indexing, and say so wherever the two get conflated.

> **ANSWERED, closed 2026-08-23
> ([`0094`](../tasks/0094-t32-golden-gate-rows/TASK.md)) — and reading the code
> found a SECOND, more basic hole alongside the one this thread named.** The
> comparison loop only ever checked the first `kGoldenBatch = 4` rows of the
> output regardless of batch size, and `want` was read at a fixed 4 rows and
> never tiled to match. **At batch 128 the gate compared 4 of 128 rows and never
> read the other 124 at all.** That is truncation, and it is a bigger hole than
> "the copies are indistinguishable" — it means a corruption bug anywhere past
> row 3 was never observed, let alone made invisible by identical tiling. It is
> also the direct explanation for how [`0070`](../tasks/0070-m13-nomic-runtime/TASK.md)'s
> `swiglu_cpu()` race passed: the corrupted rows were never looked at.
>
> **Both holes are fixed**, and they needed different fixes:
> * the loop now runs over all `batch` rows, and the printed line says so
>   rather than being silently redefined;
> * the four golden sequences are tiled with a **per-copy rotation**
>   (`(k + r) % kGoldenBatch`, applied identically to `emb_in`, `mask`,
>   `amask_i` **and** `want`), so every physical row's content is unique —
>   this thread's option 1.
>
> **Verified against five deliberately reintroduced known-bad artifacts**, to
> [`CURRENT_STATUS.md`](../docs/CURRENT_STATUS.md) §4's stated bar (*"verified
> against known-bad artifacts, not only good ones"*):
>
> | # | injected bug | gate | result |
> |---|---|---|---|
> | 1 | row-10 negation | truncated compare | **PASS** — the old gate is blind |
> | 2 | row-10 negation | all rows | **FAIL** — rel_fro 1.768e-01, 1-cos 2.000 |
> | 3 | row 6/10 swap (same slot, different tile copy) | all rows, **plain** tiling | **PASS**, bit-identical to clean |
> | 4 | row 6/10 swap | all rows, **rotated** tiling | **FAIL** — 1-cos 1.016 |
> | 5 | forced NaN row | all rows, rotated | **FAIL** — "non-finite output" |
>
> **Row 3 is the one that justifies the rotation**: the all-rows fix *alone*
> does not catch aliasing, because identical content cannot tell which copy it
> came from. Row 5 confirms `CURRENT_STATUS.md` §4's bug #4 (`std::max(0.0,
> NaN)` scoring a perfect zero) has not regressed through the rewritten loop —
> the obvious place for it to come back.
>
> **One real finding on the way**: the first corruption hook added `+5.0f`,
> which inflated `rel_fro` but left `worst_1mcos` **unchanged**, and PASSED even
> under the fixed gate — because `1-cos` assumes unit-norm vectors and is blind
> to a shift that survives normalisation. The two metrics genuinely diverge, and
> the hook was changed to row negation (which drives 1-cos to ~2). Worth knowing
> wherever this project quotes one of the two.
>
> **No published number moved.** All four models with a HuggingFace golden
> re-ran clean under the stronger gate and every `rel_fro` matches its recorded
> value to the digit — MiniLM 4.473e-03, bge-base 4.297e-03, bge-large
> 3.763e-03, nomic 6.119e-03. So rows 4+ were always as accurate as rows 0–3.
> This is the first time anyone checked.
>
> **Item 2** ("generate goldens at a batch that is not 4") was **not** done: the
> rotation makes it unnecessary for the aliasing class item 1 addresses, and a
> genuinely larger distinct-content fixture is a bigger lift. **Item 3**
> (`verify_embed_e2e.py` in the standard gate) is **partly actioned and left
> named, not dropped**: confirmed working standalone (bge-base, worst 1-cos
> 2.613e-05, PASS) and costed — cheap for the five bf16 BERT/nomic rows, needs a
> `--cpu-model`-style flag first for the six int8 rows (the same gap
> [`0085`](../tasks/0085-m13-release-sweep/TASK.md)'s `run_mteb.py` hit), and
> already redundant for arch=1, which has its own distinct-text differential
> check. Not wired into `tools/release_benchmark.ps1` — the sweep takes hours
> and the machine was shared.

<a id="t4"></a>
### T4 — Finish [`0043`](../tasks/0043-m9-attention-geometry/TASK.md) · **ANSWERED 2026-08-23**
Its Results section is empty. The four artifact sets are exported and the
commands are recorded; only the measurements are missing.
[note 0007](notes/0007-unused-iron-surface.md) §1.1 additionally reopens its
`cols ≤ 4` conclusion via mem-tile `pad_dimensions`, and §3.4 supplies
whisper-xdna's warning that fused attention was built elsewhere, was correct,
and still lost.

> **ANSWERED 2026-08-23 ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md)):
> 0043's Results section is filled in, from its own recorded commands, and it
> SUBSTANTIATES the claim CLAUDE.md had been making without support.**
>
> The cost of adopting the only geometry that can also express attention
> decomposes cleanly and multiplicatively:
>
> | step | what changes | cost |
> |---|---|---:|
> | A -> C | tile size alone, `cols` fixed at 8 | **1.289x** |
> | C -> D | column count alone, tile fixed at (16,64,16) | **1.729x** |
> | **A -> D** | **both, the real price** | **2.229x** |
>
> 1.289 x 1.729 = 2.228, matching to three significant figures, so the two
> penalties compound rather than add. Against attention's ~2-5% host cost (F3),
> paying **2.229x on every projection GEMM** to fold it in is clearly negative.
> [`0089`](../tasks/0089-t15-task-index-backfill/TASK.md) had flagged that
> CLAUDE.md credited 0043 with a "not worth the fight" verdict its own log never
> reached; the verdict is now earned.
>
> **But note 0007 1.1's reopening is NOT closed by this, and is now better
> priced.** Reading `xaie2pgbl_reginit.c` settles where padding exists in
> silicon: `Aie2PMemTileDmaMod` line 1667 has
> `.Padding = XAIE_FEATURE_AVAILABLE`, while the compute tile (1905) and shim
> (2158) do not. So the hardware supports padding **exactly where 1.1 wants to
> use it**, which `AIEDialect.cpp`'s verifier already enforces independently.
> **If** mem-tile padding works, the projections pay set **C**'s 1.289x rather
> than set D's 2.229x, because the column halving is precisely what padding
> removes. That is a different trade - probably still negative, but no longer
> obviously so. Filed as [T38](#t38) rather than folded into this
> closure.

<a id="t18"></a>
### T18 — `--probe-streams` and `--bench` disagree by up to 10% on array time · **ANSWERED 2026-08-23**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md). Mean per-dispatch:
probe 3,328 µs vs bench 3,028 µs (fp32 C); 3,038 vs 2,999 (bf16 C). Both on an
idle array with the gate green. The probe dispatches back to back with no buffer
syncs; bench syncs around each dispatch. Until it is explained, **neither number
should be quoted as "the" array time** — though 0048's conclusion is a
within-run comparison and does not depend on it.

> **ANSWERED 2026-08-23 ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md)):
> the gap did not reproduce, and there was never a definitional difference to
> find.** Same artifact set, same commands, idle array verified by `xrt-smi`
> (never a foreign Active context), three repeats each: probe's mean
> **3,034.4 us** sits **0.2%** from bench's `wait (hardware)` **3,028.3 us**, and
> 1.6% from bench's `submit + wait` - which is the figure actually comparable to
> the probe, since the probe times the whole `dispatch_only()` call. Run-to-run
> spread was under 0.3%, and submit overhead is only ~50 us, far too small to
> account for 0048's 300 us either way.
>
> Structurally there is nothing to explain: **both harnesses call the same
> `Design::dispatch_only()` / `xrt::run::wait()`, timed the same way.** 0048's
> gap is most likely session-to-session drift. **Quote `--bench`'s
> `wait (hardware)` line** for array-only time - it comes from the production
> dispatch path and separates hardware time from submit overhead - but nothing
> here says either number was ever wrong. The bf16-C pairing was not
> independently re-confirmed: `--bench` cannot run on that artifact set, which
> predates the full-encode wiring.

<a id="t21"></a>
### T21 — One tile geometry for four shapes · **ANSWERED 2026-08-23** · worth ~4%, and only on narrow models
whisper-xdna measured **2.5× and 4.6×** swings from per-shape tile sizes
([note 0007](notes/0007-unused-iron-surface.md) §3.5). Our M2/M5 finding that
per-core cost is flat to 0.4% across the four MiniLM shapes is not contradicted —
ours are all fat, theirs were not — but the one-xclbin architecture *requires* a
single geometry, and [T1](CLOSED-THREADS.md#t1) has
just made geometry the thing that matters. Unmeasured here.

> **Partly overtaken 2026-08-23.** Geometry is now chosen **per model** rather
> than once for the project: bge-large's int8 design runs `tile_n = 64` while
> every other model runs 48, because 64 is legal only at int8's 1-byte operands
> and only bge-large's N values divide `64·8`
> ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md)). One xclbin per *model*
> was never the constraint; one geometry across the four shapes *within* a
> model still is, and that half of this thread is untouched.

> **ANSWERED 2026-08-23 ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md)):
> the remaining half - one geometry across the four shapes *within* a model - is
> now priced, on int8, where the traffic model governs.**
>
> Three of MiniLM's four shapes (`qkv`, `attn_out`, `ffn_down`) are **already at
> their legal `n` ceiling** at the shipped 48 for `cols = 8`. Only `ffn_up` has
> room to go to 64, because `intermediate = 4*hidden` always divides
> `64*8 = 512` while `qkv`'s `3*hidden` and `attn_out`/`ffn_down`'s `hidden`
> usually do not. Built as a standalone single-op design: **1,539 -> 1,351 us,
> 1.139x on that shape alone, 1.043x (4.3%) on the four-shape sum.**
>
> **On bge-large it is worth exactly zero** - at hidden 1024 all four shapes
> already divide 512, which is precisely why
> [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) could move that whole model
> to `tile_n = 64` uniformly and lose nothing.
>
> So the one-geometry-per-model constraint costs **~4% of GEMM array time, only
> on the narrow models, concentrated in a single shape** - a long way from
> whisper-xdna's 2.5x/4.6x, and consistent with this project's own M2/M5 finding
> that per-core cost is flat across our four shapes because ours are all fat.
> A further lever exists (`n = 128`, legal by L1 and blocked only by there being
> no c-bf16 kernel entry point for `m*n = 8192`) and would need new kernel work;
> not covered here.

<a id="t26"></a>
### T26 — Why is bfp16 + bf16-C 6.6× MORE accurate than bfp16 + fp32-C? · **ANSWERED 2026-08-23** · a rounding-mode leak that mlir-aie 1.3.4 did not emit control for, and 1.4.2 does
[`0052`](../tasks/0052-m10-research-night/TASK.md) §6. Adding a rounding
(bf16 C transport) to the emulated datapath should cost accuracy; it measured
1-cos 2.395e-03 → **3.615e-04**, confirmed independently by e2e text
(9.7e-04) and MTEB (worst task −0.33 → −0.06). The two workers differ only in
where the fp32 accumulator lives: the C fifo object (fp32-C path) versus a
core-local `Buffer` + `narrow_f32_bf16` (bf16-C path).

**PROBED 2026-08-20** ([`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md)):
the named hypothesis (the fp32-C fifo path re-quantises C partials at every
k-block boundary — six times at K=384) is **REFUTED on two independent
grounds**. (1) The matmul kernel object is **byte-for-byte identical**
between the fp32-C and bf16-C builds (`llvm-objdump` diff, one shared
`matmul_bf16_f32_<hash>.o` in both cache dirs) — there is only one matmul
kernel, used identically regardless of accumulator location, so no per-block
conversion exists to be the mechanism. The per-core wrapper differs only in
the accumulator's address and (bf16-C only) one `narrow_f32_bf16` call
*after* the k-loop, using `conv_even` rounding — nothing changes *inside*
the loop. (2) Numerically, a single-GEMM probe (M=256,N=192, 4 cols) run
"full" (one dispatch, K/64 k-blocks in one loop) vs "split" (K/64 SEPARATE
one-k-block dispatches, host-summed) shows the fp32-C full/split ratio is
**flat at 1.000× at BOTH K=384 (6 blocks) and K=1536 (24 blocks)** — if
boundary-crossing degraded it, the ratio should grow with block count; it
does not move at all.

**The anomaly itself also does not reproduce on isolated synthetic data**:
at full K, bf16-C is statistically tied with fp32-C (a hair *worse*) at both
K=384 (1.480e-02 vs 1.472e-02) and K=1536 (1.539e-02 vs 1.530e-02) — nothing
like production's 6.6×. A smaller, real, reproducible effect exists only in
"split" mode (single k-block, no boundary by construction): bf16-C beats
fp32-C by 27–36%, growing with block count (6→24) but present even at N=1
block, so it cannot be a boundary-count effect either.

**PROBED FURTHER 2026-08-20** ([`0056`](../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md)):
the rounding-mode-asymmetry hypothesis above is **REFUTED BY THE CODE**, and
a REFINED structural hypothesis is **CONFIRMED as a genuine compounding
mechanism** on a controlled chained probe.

1. **Refuted**: `runtime/src/main.cpp`'s `to_bf16()` (line 403) — the
   function used at the ONE point where an activation is narrowed to bf16 to
   feed the *next* GEMM's input, for BOTH the fp32-C and bf16-C paths alike,
   at every layer boundary — is explicitly round-to-nearest-even
   (`(u + 0x7FFF + ((u >> 16) & 1)) >> 16`), mirrored bit-for-bit by
   `tools/npue.py`'s `to_bf16_bits` used when packing weights offline. AIE's
   default `floor` (trap 2b) does not appear anywhere in this narrowing
   chain. There is no floor-vs-`conv_even` asymmetry in production.
2. **What the reading DID surface**: a real structural difference — the
   bf16-C path narrows its raw fp32 accumulator ONCE, EARLY (on-core,
   `conv_even`, before bias-add), *in addition to* the same late host RNE
   narrow both paths share right before the next GEMM; the fp32-C path keeps
   full fp32 precision through bias-add (and any host eltwise op) and only
   narrows once, late.
3. **Tested this refined hypothesis on a chained multi-GEMM probe**
   (`t26_chain_probe.py`, one constant shape reused for every stage so only
   2 device builds are needed for any chain length): tied at 1 stage
   (`1-cos` ratio 0.99, reproducing 0053's isolated-GEMM near-tie exactly),
   then **diverging monotonically** — 1.35×, 1.46×, **1.67× by stage 4**
   (`rel_fro`: 1.00×, 1.16×, 1.21×, 1.29×). The gap genuinely compounds with
   chain length, and extrapolating the growth rate to a 24-GEMM chain lands
   in the right order of magnitude for production's 6.6×.
4. **Still open**: the numerical *why* — does an extra early RNE-family
   narrowing reduce the growth of bfp16-emulation quantisation error because
   independent per-block noise partially cancels when narrowed-and-summed
   rather than accumulated-then-narrowed-once? 0053's own split-mode data is
   suggestive (bf16-C's split error *decreases* with more blocks: 1.073e-2 at
   6 blocks → 9.878e-3 at 24), but this was not isolated from the
   chain-compounding effect measured here — they may be the same phenomenon
   or two different ones. Also still open, carried from 0053: whether real
   weight/activation distributions (vs synthetic Gaussian, used in both the
   0053 and 0056 probes) matter — 0008 already showed real-vs-uniform data
   changes plain-bfp16 error by 0.85×, not the 6× a simulated prediction
   assumed, so distribution-dependence is an established mechanism class
   here even though it isn't confirmed as *this* mechanism. **Not
   examined**: whether today's shipping plain-bf16 (non-emulated) path
   carries any analogous, smaller effect — an explicit non-goal of 0056,
   left for a future decision.

**PROBED FURTHER 2026-08-23 ([`0096`](../tasks/0096-t26-numerical-mechanism/TASK.md)):
a host-side model built to explain the numerical *why* FAILED ITS OWN
CONTROL, and that failure is the result.** The model is numpy/fp64,
calibrated against [`0008`](../tasks/0008-m5-bfp16-real-data/TASK.md)'s
real-hardware bfp16 fit (7 bits/element, block 8 — cross-checked against
`aie_api`'s own `type_bits<bfp16ebs8> = 8`) and structurally faithful to
0056's two traced narrowing sites, using the bit-identical RNE formula at
each. **It does not reproduce 0056's chain-probe measurement.** The
fp32C/bf16C ratio stays pinned at **0.99–1.00 across four stages** — and
across eight, under 300-trial averaging — against hardware's measured
0.99 → 1.35 → 1.46 → **1.67**. That held at every mantissa calibration
tested, 3 through 9 bits.

Per the control rule, the ablations built on top of it are **not** treated
as mechanism evidence. What the failure does establish is a **negative that
narrows the thread**: *"narrow position applied to independent,
magnitude-calibrated per-block bfp16 noise"* is **not a sufficient
explanation**. The supporting ablations agree — both paths show near-zero
error mean (`NOT BIAS-DOMINATED`) and near-zero stage-to-stage correlation
(|corr| < 1.3e-03), so neither the bias hypothesis nor the noise-cancellation
hypothesis has anything to act on in this model.

Three sub-results, all from the same run:

* **Real data is a direction, but an order of magnitude short.** Real
  weights and activations through the actual 6-layer MiniLM encoder give
  **1.195×** on the embedding-level metric, against Gaussian's ~1.00 — so
  distribution-dependence is not nothing, which is consistent with 0008's
  established mechanism class. But 1.195× against production's 6.6× means it
  cannot be the missing ingredient by itself.
* **0053's split-mode shrinkage does not reproduce either** (6 blocks
  1.8028e-02, 24 blocks 1.8054e-02 — flat, not shrinking). So whether it is
  "the same phenomenon" as the chain compounding is still open; the two stay
  structurally distinct regardless, being width of one reduction against
  depth across stages.
* **Today's shipping plain-bf16 path does carry an effect, and it is the
  opposite one.** bf16-C is *worse* by 1.41× at one stage and converges
  toward parity by four (ratio 0.710 → 0.890) — opposite sign, opposite
  trend. A different artifact of the same narrow-position asymmetry, not
  "the same anomaly, smaller." This closes 0056's explicit non-goal with a
  measured answer.

**One analytic fact survives independent of the model**: RNE-narrowing to
bf16 is **idempotent**, so bf16-C's "extra" narrow is only ever
non-degenerate because it targets the **pre-bias** accumulator. Narrow-count
and narrow-position are therefore not independent factors here, and any
future hypothesis must treat them as one.

**PROBED FURTHER 2026-08-23 ([`0098`](../tasks/0098-t26-kernel-source/TASK.md)):
reading the kernel source — 0096's own prescribed next step — found a concrete
asymmetry no host model could see. It is trap 2b again, one level deeper.**

`aie_kernels/aie2p/mm.cc` wraps its emulated-bf16 k-loop in
`aie::swap_rounding(conv_even)` (line 101) / `aie::set_rounding(saved)`
(line 228), so the MAC's internal bf16→bfp16 quantisation of A and B tiles is
meant to be unbiased. **The compiled object contains no such instruction.** A
grep for `rnd`/`round`/`crrnd` across the full disassembled matmul object
returns **zero hits**, while the same disassembly plainly contains the emulated
path's `vconv.bfp16ebs8.fp32`. So the code path compiled in; only the rounding
control around it did not.

The mechanism is a missing data dependence in the API.
`mmul_bf16_bf16.hpp` (lines 78, 79, 90, 91, 112, 113) calls
`::to_v64bfp16ebs8(acc)` — the **ambient**-mode intrinsic at
`aie2p_srs.h:1308` — and not `to_v64bfp16ebs8_conf(acc, rnd)` at
`aie2p_srs.h:1331`, which is the variant that actually takes the mode. Nothing
the compiler can see connects the surrounding `swap_rounding`/`set_rounding`
pair to the quantisation between them, so the pair is dead code and is removed.
(The root LLVM-IR step is inferred rather than proven — see 0098's Problems.)
The toolchain unambiguously *can* emit the instruction:
`narrow_f32_bf16.o` carries **three `mov crrnd, #0xc`**, one per tile-size entry
point — **and never restores it.**

**Net effect: every A/B-tile bfp16 quantisation inside the emulated matmul runs
under whatever rounding mode already sits in that physical core's control
register.** A core that has never executed `narrow()` — every fp32-C core,
always — stays at AIE's default `floor`, systematically biased low, compounding
over the K reduction. A bf16-C core inherits the `conv_even` its *own* prior
`narrow()` call left behind and never undoes: unbiased, for free, as a side
effect of an unrelated kernel sharing the core.

This is exactly what CLAUDE.md trap 2b already warns about in the abstract —
*"`set_rounding` is core-wide state that leaks between kernels sharing a
core"* — showing up as an accuracy result nobody connected to it.

**It reproduces every qualitative feature the host model could not**: parity at
stage 1 (no core has poisoned itself yet), monotonic divergence with chain
length (fp32-C's bias keeps compounding, bf16-C's does not), and even 0053's
split-mode shrinkage (more dispatches dilute the single `floor`-quantised first
one against a growing majority of `conv_even` ones) — the sub-question 0096
explicitly failed to reproduce.

**NOT confirmed on hardware, and it must be before it is believed.** Two
ablations, neither run:
1. **Warm up an fp32-C core** with a throwaway `narrow()`-calling dispatch
   before the real measurement. Predicts most of the 6.6× gap closes *without
   touching C's transport dtype at all* — which would settle it.
2. **Patch `mmul_bf16_bf16.hpp` to call `to_v64bfp16ebs8_conf` explicitly.** If
   (1) confirms, this is a real upstream fix, not a local workaround.

**And one consequence for the shipping path, not just the retired one**:
`narrow_f32_bf16` sets `crrnd` and never restores it, so on `--c-bf16` every
kernel that afterwards shares that core runs under `conv_even` rather than the
default. That is probably benign and possibly good, but it is unaudited.

**So the next step is not another host model.** 0053 proved the matmul
kernel object bit-identical between the two builds and 0008/trap 2 prove the
fp32 accumulation exact, so the missing ingredient must live in
hardware/kernel-level numerical behaviour that a host bit-formula model
cannot express — the emulated bfp16 matmul's accumulator handling, or
`narrow_f32_bf16` itself. Read the kernel source directly (trap 2b's own
discovery method — that whole finding came from reading `aie_api`'s headers,
not from modelling), or run a device-level ablation that isolates narrow
position with real hardware arithmetic.

---

## ANSWERED 2026-08-23 ([`0099`](../tasks/0099-t26-rounding-ablation/TASK.md)): the mechanism is 0098's, and **the mlir-aie 1.3.4 -> 1.4.2 upgrade already fixed it**

The ablation 0098 proposed was run, and it is an **exact null** — which is what
0098's own mechanism predicts once you check *which toolchain built the binary*.

**The ablation.** Rather than warming a core with a `c_bf16` dispatch and then
switching (those are provably different xclbins, so a switch would confound the
mechanism with a context reset), 0099 added a `poison` worker variant to
`gemm_pretiled.py`: byte-for-byte the plain fp32-C worker plus one throwaway,
discarded `narrow()` call placed **after** the real output is released. C stays
fp32 throughout, and "cold" and "warmed" are literally the same compiled xclbin
dispatched twice in one process.

| | `rel_fro` |
|---|---:|
| cold fp32-C | 9.44211e-03 |
| warmed fp32-C | **9.44211e-03** — bit-identical per seed, to full float64 |
| bf16-C | 9.59035e-03 — 1.6% *worse*, not 6.6x better |

**Why the null is the confirmation, not the refutation.** 0099 objdumped a fresh
rebuild of the *same kernel config hash* `333c4d33` and found it **now contains
the `crrnd` save/set/restore triple** that 0098's checked-in artifact genuinely
lacked. Re-verified independently here, with the AIE `llvm-objdump` (the MSVC one
on PATH cannot decode `elf32-unknown` at all and fails silently to zero hits —
worth knowing):

| object | `crrnd` | `vconv.bfp16` |
|---|---:|---:|
| `objdump_*_rtp_matmul_bf16_f32_333c4d33.txt` (checked in) | **0** | 20 |
| `matmul_bf16_f32_333c4d33.o` (cache today, 5 copies, all md5-identical) | **3** | 20 |

It is not a disassembler artifact — the instruction streams genuinely differ,
and the old dump's own header names a different cache directory:

```
old 0x24:  mova r18, #0x6;   movx r17, #0x300;  mov r29, p0
new 0x36:  mova r19, #0x180;  movx r18, #0x6;    mov r22, crrnd
```

**And the dates settle it.** The artifacts were committed in `efd8c76` at
**12:15:45** on 2026-08-20; the mlir-aie 1.3.4 -> 1.4.2 migration
([`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md)) landed in `42da31d` at
**16:05:34** the same day. `git merge-base --is-ancestor` confirms the ordering.
**Every measurement that ever saw this anomaly — [`0052`](../tasks/0052-m10-research-night/TASK.md),
[`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md),
[`0056`](../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md) — was taken on
mlir-aie 1.3.4**, where the rounding control was not emitted.

So the account is complete and every piece of evidence lines up:

* **On 1.3.4** the emulated matmul emitted no `crrnd` control, so its bfp16
  quantisation of A and B ran under whatever mode the core was left in — `floor`
  on an fp32-C core, which never calls `narrow()`, and `conv_even` on a bf16-C
  core, inherited from its own epilogue and never restored. `floor` is a
  systematic downward bias, not symmetric noise (trap 2b), so it compounds over
  the K reduction and across chained GEMMs. That is the 6.6x.
* **On 1.4.2** the same config emits the triple, both paths quantise under
  `conv_even`, and the asymmetry cannot exist. Accordingly an unmodified rerun of
  `t26_chain_probe.py` **no longer reproduces 0056's 0.99 -> 1.67 divergence** —
  it is flat at 0.98–0.99 — and poisoning a core does exactly nothing, because
  the matmul now overrides whatever it finds.

It also explains, retrospectively, every qualitative feature 0096's host model
could not reproduce (parity at stage 1, monotonic divergence, 0053's split-mode
shrinkage) and why it could not: the model was faithful to the *source*, and the
source was not what 1.3.4 compiled.

**What is NOT done, and it is a verification rather than an open mechanism**: the
production `1-cos` comparison that produced the headline pair (2.395e-03 fp32-C
vs 3.615e-04 bf16-C) has not been re-run on 1.4.2 to show the two now agree. The
chain probe is the measurement that exhibited the effect and it no longer does,
which is why this closes — but the production number is the one people quote.
**That re-measurement is folded into [T23](#t23)**, which needs it anyway.

**One consequence beyond this thread.** 0053's *"the matmul kernel object is
byte-for-byte identical between the two builds"* still stands — it was a
within-session comparison. But an object is **not** stable across a toolchain
change, and nothing in the cache path or the config hash records which toolchain
built it. Any conclusion resting on a cached object's contents from a different
day is suspect, and `333c4d33` naming two different instruction streams is the
proof. That belongs with trap 7c's family, and it is the reason this thread took
four sessions.

**That consequence was audited immediately, in
[`0102`](../tasks/0102-toolchain-provenance-audit/TASK.md), and it is narrow.**
Roughly twenty published claims were checked against the boundary. Exactly two
are genuinely expired — this thread's own 6.6× and `--emulate-bfp16`'s
retirement, which share one mechanism, both re-measured in
[`0101`](../tasks/0101-t23-bfp16-gates-on-1.4.2/TASK.md). Everything else is
cleared by evidence already on record: [`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md)
re-verified SAXPY, the M2 GEMM headline and **trap 2b's whole rounding table**
bit-exact on 1.4.2 (including the softmax row-sum mechanism), and
[`0060`](../tasks/0060-m11-export-gemm-rtp-marker-fix/TASK.md) rebuilt a real
production GEMM shape post-migration and reproduced `1-cos` 1.086e-05.

The sharper rule 0102 extracted is worth carrying: **the elimination is a
property of the call, not of the toolchain version.** The risk is a
rounding-sensitive operation happening *inside an `aie_api` header via an
ambient intrinsic* — not a measurement merely being old. Kernels that set the
mode adjacent to their own vector ops kept it, which is why `narrow_f32_bf16.o`
retained its `crrnd` writes on the very build that lost `mm.cc`'s. The
provenance gap itself is now [T39](#t39).

**And the practical consequence has been carried all the way through**: this
thread invalidated [T23](#t23)'s entire accuracy case, which was
re-measured on 1.4.2 for two models in
[`0101`](../tasks/0101-t23-bfp16-gates-on-1.4.2/TASK.md) and for the remaining
four in [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md). The headline: the
6.6× this thread existed to explain **is gone**, exactly as the mechanism
predicts — fp32-C moved ~7× better while bf16-C barely moved — and five of six
models now clear the MTEB gate on the emulated datapath.

<a id="t23"></a>
### T23 — The GEMM datapath itself: bfp16 emulation is 2.9× of array GEMM time · **ANSWERED 2026-08-23** · decided by the user on MTEB evidence, and shipped per model
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md). On the same design,
same shape, the emulated path costs 2,684 cycles per k-block against plain
bf16's 7,897 — **2.9× of array GEMM time**, worth roughly **1.35× end to end
on MiniLM (39.4% wait) and 1.66× on bge-large (60.8% wait)**. `--emulate-bfp16`
was retired when it was worth +2.2% end to end
([`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)) and failed the 1-cos gate at
3.470e-03; **the pricing half of that retirement is superseded, the accuracy
half stands.** T16 also showed the emulated design is DMA-bound (39% vector
busy, lock-stall gaps), so under emulation the retired byte-levers (B-reuse,
T2) would partially revive. Reopening this means running MTEB on the bfp16
path per [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) — flagged for the user;
per the 0045 precedent, datapath accuracy decisions are made by the user on
MTEB evidence, not unilaterally. [T20](CLOSED-THREADS.md#t20)
(int8) is the same decision shape with different numbers.

**Upside re-priced 2026-08-19 by [ATB](https://arxiv.org/abs/2511.16041)** (UCLA+AMD, our
SKU, our toolchain, web-indexed): with asymmetric tile buffering and a
hand-optimised microkernel, BFP16 GEMM reaches **24.3 TFLOPS** on this silicon
— ~8× our production array rate, of which 2.88× is microkernel work alone
(stock 0.32 → 0.92 TFLOPS/core). The 2.9× above is what the *stock* emulated
path buys; it is the floor of this decision, not the ceiling.

**MEASURED 2026-08-20 ([`0052`](../tasks/0052-m10-research-night/TASK.md)) —
every gate now passes, on the bfp16+bf16C combination:**

| | bfp16, fp32 C | **bfp16 + bf16 C** |
|---|---:|---:|
| `1-cos` vs HF (validation) | 2.395e-03 FAIL | **3.615e-04 PASS** |
| e2e real text, worst `1-cos` | — | **9.701e-04 PASS** |
| top-10 neighbour overlap | — | **0.9923 PASS** |
| MTEB, 5 tasks, mean Δ vs CPU | +0.01 (worst −0.33) | **+0.16 (worst −0.06)** |
| array GEMM vs plain bf16 | 1.74× | **2.20×** |
| e2e at pipeline 4 (MiniLM) | 982.1 seq/s | **1,046.2 seq/s (+8.7%)** |

Two big caveats travel with this. (1) **The e2e gain is only +9% because the
encode is host-walled at 47% NPU busy** — the datapath decision pays its
1.35–1.66× only together with host-side work (T3, readback). (2) **The
emulated array is TRAFFIC-bound** (GB/s 32–46, ffn_up slower than ffn_down in
proportion to bytes), so the byte-levers T1 retired come back to life on this
path: B-reuse, ATB, and bf16-C (already in). Decision remains the user's; the
mechanism of the accuracy jump is T26 (below), which as of 2026-08-20 has
moved from "unknown mechanism" through "original hypothesis refuted,
isolated-GEMM effect only 27–36% against production's 6.6×" to "the
rounding-mode-asymmetry hypothesis (`conv_even` vs `floor`) is ALSO refuted
by reading `runtime/src/main.cpp` (both paths' downstream narrowing is
round-to-nearest-even), but a refined structural hypothesis — bf16-C narrows
early, before bias, in addition to a late narrow both paths share — is
CONFIRMED to compound on a chained probe: tied at 1 stage, 1.67× by 4 stages
([`0056`](../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md))." The
mechanism *class* (compounding across chained narrowing) is now evidenced;
the numerical root cause within that class is not yet nailed down.

**bge-base MTEB landed 2026-08-20 ([`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md),
T25 above): mean −0.065, worst −0.193, still comfortably PASS** — so "MiniLM is measured,
bge-base/large are not" is now half-retired: bge-base's bfp16+bf16C accuracy
is measured and passes, on a **second, independent model geometry** (h=768 vs
384, N sets that force `tile_n=48` the same way, but a different layer count
and a completely different real-weight distribution). **Still not measured**:
bge-base's bfp16+bf16C *throughput* (only the plain-bf16 interleaved ratio —
1.633× — was run this session; T27's traffic-bound question and bge-large
remain open). This strengthens rather than settles the decision: two models
now clear the accuracy bar, but the speed case for reopening bfp16 is still
argued from MiniLM's host-walled +9% alone.


> **AMENDED 2026-08-23 ([`0099`](../tasks/0099-t26-rounding-ablation/TASK.md)):
> this thread's entire accuracy case is toolchain-stale, and must be re-measured
> before any decision.** [T26](CLOSED-THREADS.md#t26) is now answered: the 6.6x
> accuracy advantage of bfp16 + bf16-C over bfp16 + fp32-C was a **mlir-aie
> 1.3.4 artifact** — the emulated matmul did not emit its own `crrnd` control, so
> the fp32-C path's bfp16 quantisation ran under `floor` and was systematically
> biased low, while the bf16-C path inherited `conv_even` from its epilogue.
> **mlir-aie 1.4.2 emits the control and the asymmetry cannot exist.**
>
> Every row of this thread's gate table — `1-cos` 2.395e-03 FAIL vs 3.615e-04
> PASS, the e2e 9.701e-04, the 0.9923 neighbour overlap, the MTEB deltas, and
> [`0053`](../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md)'s bge-base
> confirmation — was measured on 1.3.4, before `42da31d`. **None of them can be
> quoted for a decision today.**
>
> Which way it moves is genuinely unknown and worth stating both ways: the fp32-C
> path should get *better* (its bias is gone), so the *combination* that passed
> every gate may no longer be the best one, and plain bfp16 + fp32-C — retired in
> [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) at 3.470e-03 — may now pass on its
> own. The **speed** half is unaffected: 2.20x on array GEMM time and the
> host-walled +8.7% end to end are timing, not arithmetic.
>
> **So the decision is not ready to be made.** What it needs is a re-run of the
> `1-cos` and MTEB gates for both C-transport variants on 1.4.2, on at least
> MiniLM and bge-base, in one session. Until then this thread is blocked on a
> measurement, not on the user.


**RE-MEASURED ON 1.4.2, 2026-08-23 ([`0101`](../tasks/0101-t23-bfp16-gates-on-1.4.2/TASK.md)).
Every number above is superseded. The decision is now ready to be made, and it
is still the user's.**

[T26](CLOSED-THREADS.md#t26) established that this thread's whole accuracy case
was measured on mlir-aie 1.3.4, where the emulated matmul emitted no rounding
control. Re-run on 1.4.2, three configurations x two models, with the plain-bf16
production path as a control that reproduced twice to the digit (MiniLM
`rel_fro` 4.473e-03 / worst `1-cos` 1.086e-05; bge-base 4.297e-03 / 1.353e-05).

| model | config | `1-cos` | MTEB mean | MTEB worst | verdict |
|---|---|---:|---:|---:|---|
| MiniLM | bfp16, fp32-C | 3.368e-04 | +0.12 | −0.05 | **both PASS** |
| MiniLM | bfp16, bf16-C | 3.406e-04 | +0.12 | −0.07 | **both PASS** |
| bge-base | bfp16, fp32-C | 2.174e-04 | −0.14 | **−0.58** | 1-cos PASS, **MTEB FAIL** |
| bge-base | bfp16, bf16-C | 2.284e-04 | −0.06 | −0.19 | **both PASS** |

**Three things changed, and one of them settles the thread's premise.**

1. **The 6.6x is gone.** fp32-C moved 2.395e-03 -> 3.368e-04 (~7x better) while
   bf16-C barely moved (3.615e-04 -> 3.406e-04). That is exactly the asymmetry
   T26's mechanism predicts: fp32-C was the broken one, stuck at `floor` because
   its cores never call `narrow()`, and bf16-C was accidentally fine because it
   inherited `conv_even` from its own epilogue. On `1-cos` the two are now within
   1–5%, with fp32-C marginally *ahead*.
2. **[`0035`](../tasks/0035-m8-mteb-gate/TASK.md)'s retirement of
   `--emulate-bfp16` is partly overturned, and the qualifier is the point.** At
   `1-cos` it now passes comfortably on both models, so the 3.470e-03 failure
   that retired it was the `floor` bias and not the datapath. But on **bge-base
   it fails MTEB**, so the retirement stands for that model. Model-dependent, not
   settled either way.
3. **The two gates disagree in DIRECTION, which is new.** On bge-base `1-cos`
   ranks fp32-C better (2.174e-04 vs 2.284e-04) while MTEB ranks bf16-C better
   (worst −0.19 vs −0.58), on the same CPU baseline of 50.58 —
   `TwentyNewsgroupsClustering` reads 50.38 against 50.00. Previous disagreements
   were about magnitude; this one is about order. Fourth instance of
   [`0035`](../tasks/0035-m8-mteb-gate/TASK.md)'s point, and the strongest
   argument yet for keeping both gates.

**Speed, labelled by provenance.** Array GEMM time (`--bench`'s `wait
(hardware)` line, which [T18](CLOSED-THREADS.md#t18) established is the number to
quote) against plain bf16: MiniLM **1.75x** fp32-C / **2.04x** bf16-C, bge-base
**2.08x** / **2.29x** — close to 0052's pre-migration 1.74x/2.20x, so the speed
half of this thread was never toolchain-dependent. End-to-end **wall clock, not
an NPU claim** (rule 1): +19.8% / +32.1% on MiniLM, +44.5% / +56.9% on bge-base.

**The reading, for a decision that remains the user's per the 0045 precedent.**
**bfp16 + bf16-C is the only configuration clearing both gates on both models,
and it is also the faster of the two.** The reason to prefer it is no longer the
refuted "6.6x more accurate" claim — it is simply that it is the one that
passes. Plain bfp16 + fp32-C is **not** safe to adopt on `1-cos` alone; MTEB
catches a real bge-base regression that `1-cos` not only misses but ranks the
wrong way.

**Residual risk, stated so it is not rounded away.** Only **2 of 6** shipped
models were measured — bge-large is the one most likely to behave differently,
being the widest and already the catalogue's only `1-cos` failure on int8, and it
was not tested. The `1-cos` figures are post-[`0094`](../tasks/0094-t32-golden-gate-rows/TASK.md)
**128-row** numbers and are not directly comparable to the pre-migration 4-row
ones. Speed figures are single `--bench 5` runs, not repeated. And bfp16 is
genuinely less faithful than production bf16 — `1-cos` 3.4e-04 against
1.086e-05, 31x worse, though 6x inside the tolerance.

**DECISION RULE SET BY THE USER, 2026-08-23: adopt bfp16 per model wherever
MTEB passes, and measure all six.** *"Vi kjører alle. Men i utgangspunktet
kjører vi når mteb er pass."*

This sharpens [`0035`](../tasks/0035-m8-mteb-gate/TASK.md)'s doctrine from "MTEB
is the authority" into an explicit adoption criterion, and it resolves the
gate disagreement above in MTEB's favour by rule rather than case by case. Note
what it implies and what it does not:

* **`1-cos` does not gate adoption**, but it stays as the fidelity check that has
  caught every real bug in 0077–0082 — bge-base bfp16+fp32-C is now the fourth
  case where the two gates rank differently, and the third where the project
  keeps both anyway.
* **Adoption is per model**, so the catalogue may end up running different
  datapaths for different models. That is already expressible — bfp16 is a
  property of the *design* (the xclbin), and every model already has its own
  artifacts directory — but it means a release sweep must **state which datapath
  each row was measured on**, or the table becomes uninterpretable.
* **The measurement must be the real gate**, `--sides cpu,npu` in one session.
  0101's first pass reported verdicts the gate never issued; see the process note
  below.

Measurement of the remaining four models is [`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md).

**WHOLE CATALOGUE MEASURED 2026-08-23 ([`0103`](../tasks/0103-t23-bfp16-all-models/TASK.md)):
five of six models pass, and the decision rule above resolves them.**

Real gate verdicts (`--sides cpu,npu` in one invocation, verdict issued by the
tool), on the adoption candidate **bfp16 + bf16-C**:

| model | arch | mean | worst | gate | adopt |
|---|---|---:|---:|---|---|
| `all-MiniLM-L6-v2` | 0 | +0.12 | −0.07 | **PASS** | yes |
| `bge-small-en-v1.5` | 0 | −0.10 | **−0.5010** | **FAIL** | **no** |
| `bge-base-en-v1.5` | 0 | −0.06 | −0.19 | **PASS** | yes |
| `bge-large-en-v1.5` | 0 | +0.13 | −0.01 | **PASS** | yes |
| `nomic-embed-text-v1.5` | 2 | +0.01 | −0.25 | **PASS** | yes |
| `embeddinggemma-300m` | 1 | +0.16 | −0.02 | **PASS** | yes |

**Three architectures pass**, so this is not a BERT-only result. bge-large —
the widest model, and the catalogue's only int8 `1-cos` failure — passes with
the *tightest* worst-task delta of the six.

**bge-small fails by 0.001 points and it is not noise.** −0.5010 against the
−0.5 line, reproduced **bit-identically** end to end (`cpu 0.4862581899`,
`npu 0.4812477913` on both runs). Both sides are deterministic. An initial
reading of this as "a coin flip against a bright line" was wrong: the wide
swings on that task are variance *between models*, not noise *within* a
measurement.

**And bge-small is the model MTEB likes least on BOTH quantised datapaths** —
on int8 it already has the best `1-cos` of any row (6.385e-04) and the worst
MTEB (−0.09). Whatever makes it sensitive is a property of the model, showing
up under two unrelated quantisation schemes.

**A property of the gate, worth knowing separately from any of these
verdicts.** Across all ten cells measured in 0101 and 0103,
`TwentyNewsgroupsClustering` is the worst task **every time**, ranging +0.89 to
−0.58, while the other four tasks stay inside ±0.12. So the −0.5 line is in
practice a test on clustering. That is defensible — clustering amplifies small
distance changes into hard group assignments — but the gate measures one task's
sensitivity rather than an average quality loss, and which models pass is
decided there.

**bfp16 does not degrade with width the way int8 does.** int8 runs 6.385e-04
(bge-small) → 2.968e-03 (bge-large), 4.6× worse; bfp16 is flat at 3.368e-04 /
2.174e-04 / 2.273e-04 across MiniLM, bge-base, bge-large. Plausible mechanism,
not verified: int8 quantises per tensor so one outlier drags the whole scale
and SmoothQuant has more to move as matrices widen, while bfp16 keeps a shared
exponent per 8-element block. This contradicts an assumption carried since
[`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) that quantisation error
tracks width — true for int8, **not** for bfp16.

**Speed on bge-large**, array GEMM time from `--bench`'s `wait (hardware)` line
per [T18](CLOSED-THREADS.md#t18): control 19,099 µs → **10,458 µs (1.83×)**
fp32-C → **9,776 µs (1.95×)** bf16-C. The array share of wall clock falls 43.3%
→ 28.8%.

**What this leaves open**: `fp32-C` has no MTEB verdict on four of six models,
and 0101 already showed it failing where bf16-C passes, so it cannot be assumed
equivalent. Nothing has been adopted and no default changed — per the 0045
precedent that step is the user's.

**One process note worth keeping** ([`0101`](../tasks/0101-t23-bfp16-gates-on-1.4.2/TASK.md)
Problems #1): the first pass reported MTEB verdicts the gate never issued. All
four bfp16 runs printed `NO GATE -- this run has no delta to gate on: cpu side(s)
not run`, and the deltas were hand-computed against a CPU baseline from a
different invocation. Re-run with `--sides cpu,npu` in one session, the real gate
**confirmed every cell to within 0.02 points** — so the arithmetic had been
right, and it still was not a result. The guard that caught it exists because the
same fail-open shipped once before.

---

## ANSWERED and ADOPTED 2026-08-23 ([`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md))

The user's rule was applied and the result shipped: **bfp16 + bf16-C is now the
datapath for the five models that cleared the MTEB gate**, and `bge-small`
stays on plain bf16. Nothing about this thread is open any more — the pricing
was never in doubt, the accuracy question was re-measured on 1.4.2 after
[T26](#t26) invalidated the 1.3.4 evidence, and the decision was the user's to
make.

**Adoption created the hole it had to close first, and that is the durable part
of this thread.** A bfp16 design and a plain-bf16 design at the same geometry
carry the **same `b_layout_hash`** — emulation changes MMAC precision, not B's
tiling — and `design.json` recorded nothing about the datapath. So
`design_fits()` would have let bge-small, which shares MiniLM's hidden-384
geometry and **failed** the bfp16 gate, be handed the bfp16 design by
**alphabetical tie-break**. The runtime now:

* writes `emulate_bfp16` into `design.json` (`tools/export_gemm_rtp.py`);
* refuses a datapath the model was not adopted for (`want_datapath` in
  `design_fits()` / `pick_artifacts()`), proved against a known-bad artifact —
  bge-small *refuses* against the very directory MiniLM runs on;
* records each model's adopted datapath in the catalogue, as an **explicit
  allowlist** so a future built-in defaults to plain bf16 rather than inheriting
  bfp16;
* reports `UNRECORDED` rather than claiming `bf16` when a design predates the
  field, since **absent is not false**;
* breaks a tie between equally-fitting sets in favour of the one that records
  what it is, rather than on spelling.

**Speed, for the record** (`--bench`'s `wait (hardware)` line, per
[T18](CLOSED-THREADS.md#t18)): bge-large 19,099 → 9,776 µs per dispatch,
**1.95×**, array share of wall clock 43.3% → 28.8%.

**What this does NOT close.** `fp32-C` has no MTEB verdict on four of six
models and [`0101`](../tasks/0101-t23-bfp16-gates-on-1.4.2/TASK.md) showed it
failing where bf16-C passes, so it is not an equivalent alternative. And the
release sweep must now **state which datapath each row was measured on**, or
the throughput table cannot be read — five rows moved datapath and one did not.

<a id="t17"></a>
### T17 — Bigger L1 tiles, which need more L1 than a core has · **ANSWERED 2026-08-23** · answered by measurements already in hand, on both datapaths

> **SCOPED 2026-08-23.** Everything below is a **bf16** measurement, and both
> of its load-bearing claims flip on int8:
>
> * *"78% of the iteration is `vmac.f` compute"* — on int8 the core uses
>   **0.578 µs of a 1.94 µs iteration** ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)),
>   so there is far more than 22% to amortise.
> * *"every legal larger geometry overflows"* the 63 KB L1 — **false at int8's
>   1-byte operands**: `(64,64,64)` costs 49,152 B, and
>   [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) built it for bge-large
>   and measured **1.366×**, above this thread's ≤1.29× bf16 ceiling.
>
> Note the mechanism is *not* the one this thread assumes. Bigger tiles paid
> through **A re-streaming** (`N/(n·cols)` fetches of A), not through
> amortising per-iteration overhead — and correspondingly `k=128`, which
> amortises overhead but moves no fewer bytes, measured **0.989×**. On a
> traffic-bound datapath it is `tile_n` that matters and `tile_k` that does
> not.
Was "the successor lever" under the assumption that per-iteration cost is fixed.
[T16](CLOSED-THREADS.md#t16)
measured it: **78% of the iteration is `vmac.f` compute that scales with
m·k·n**, so bigger tiles amortise only the ~1,753 non-vector cycles — **≤1.29×
with all overhead gone, ~1.06× for (64,64,64)**. The L1 wall itself is
unchanged: `(64,64,48)` costs 53,248 B of 63 KB, every legal larger geometry
overflows, and cross-tile `Buffer`
([note 0007](notes/0007-unused-iron-surface.md) §1.2) remains the one way past
it — now priced as a small lever, not the lever.

---

**ANSWERED 2026-08-23. Nothing new had to be measured — T21's closure supplied
the missing half, and this thread and that one have converged.**

The question was whether bigger L1 tiles are a lever. Both datapaths now have
an answer, and they differ:

* **bf16 — no.** [T16](CLOSED-THREADS.md#t16)'s anatomy stands: 78% of the
  iteration is `vmac.f` compute that scales with `m·k·n`, so bigger tiles
  amortise only ~1,753 non-vector cycles. **≤1.29× with all overhead gone,
  ~1.06× for (64,64,64)** — and every legal larger geometry overflows the
  63 KB L1 at 2-byte operands anyway.
* **int8 — yes, and it is already shipped.**
  [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) built `(64,64,64)` for
  bge-large and measured **1.366×**, above the bf16 ceiling, because at 1-byte
  operands it costs 49,152 B and fits.

**And the remaining question — does it generalise beyond bge-large? — was
answered by [T21](CLOSED-THREADS.md#t21)/[`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md)
without either thread noticing they had merged.** Three of the four production
shapes are **already at their legal `n` ceiling** at the shipped 48; only
`ffn_up` has room to 64, because `intermediate = 4·hidden` always divides
`64·8 = 512` while `qkv`'s `3·hidden` and `attn_out`/`ffn_down`'s `hidden`
usually do not. Measured: **1.139× on that shape, 1.043× on the four-shape
sum** — and **exactly zero on bge-large**, where all four already divide 512,
which is why 0081 could move that model wholesale.

**The mechanism is not the one this thread was filed on**, and that is the
durable part. Bigger tiles pay through **A re-streaming** — `N/(n·cols)`
fetches of A — not through amortising per-iteration overhead. `k=128`
amortises overhead but moves no fewer bytes and measured **0.989×**, i.e.
nothing. So on a traffic-bound datapath `tile_n` matters and `tile_k` does
not, which is the opposite of what "bigger tiles amortise fixed cost" predicts.

**What stays unbuilt, and is now its own small thing rather than this
thread's**: cross-tile `Buffer` ([note 0007](notes/0007-unused-iron-surface.md)
§1.2) is still the only way past the L1 wall, and remains unused. It is
priced as a small lever, not the lever, and nothing here needs it.

<a id="t9"></a>
### T9 — `xrt::runlist` · **RETIRED 2026-08-23** · its own trigger fired and pointed nowhere
[note 0007](notes/0007-unused-iron-surface.md) §3.3. Present in our XRT, unused
by our runtime, and worth having the moment dispatch count rises and dispatch
size falls — i.e. after T3 or T4.

> **RETIRED 2026-08-23, on the thread's own terms.** It was filed as worth
> having *"the moment dispatch count rises and dispatch size falls — i.e. after
> [T3](#t3) or [T4](#t4)."* **T4 closed negatively**
> ([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md)): the
> attention-capable geometry costs the projections **2.229×**, so attention
> stays on the host and the dispatch profile does not change. The trigger fired
> and pointed nowhere.
>
> T3 is still open and could revive this — but T3's own architecture has **no
> trace and therefore no price** ([T28](#t28) item (c)), so
> nothing yet says its dispatch profile will move either.
>
> Meanwhile the lever is worth **0.9%** by this thread's own estimate, and
> measuring it means spending an uncontended array — this project's scarcest
> measurement resource, per rule 1 — on the smallest number in the register.
> `research/references-external.md` records that `whisper-xdna` got
> `pyxrt.runlist` working on NPU1 despite mlir-aie's examples documenting it as
> NPU2-only, so **the mechanism is not in doubt**; only its value here is, and
> its value here is small.
>
> **Retired rather than answered**, because nothing was measured. Re-open it
> when a dispatch-profile change actually lands — that is a concrete
> precondition, not a vague one.

<a id="t39"></a>
### T39 — Nothing records which toolchain built an artifact · **ANSWERED 2026-08-25**
Filed out of [`0102`](../tasks/0102-toolchain-provenance-audit/TASK.md), which
audited this project's published claims against the mlir-aie 1.3.4 → 1.4.2
boundary after [T26](CLOSED-THREADS.md#t26) found the upgrade had silently
changed emitted code.

**The gap, stated plainly**: neither the JIT cache path, nor the kernel config
hash, nor `design.json`, nor the `.npue` header records the toolchain that
produced a binary. The proof that this matters is that config hash `333c4d33`
names **two different instruction streams** — 0 `crrnd` instructions before the
migration, 3 after — and nothing in the build path distinguishes them. 0102 also
had to fall back on `.xclbin` **mtime** to establish which shipped designs
predate the upgrade, in a repo whose trap 7c says never to identify a build
artifact that way (0102 argues why that particular use is defensible; it is
still weaker than a content check, and it only exists because no better source
does).

**Proposed fix, priced and not implemented**: a `toolchain.json` sidecar written
by the export tools — `mlir_aie` version, Peano version, mlir-aie git HEAD — and
copied into the `.npue` header. **Under an hour** per 0102.

**Why it is worth an hour.** This project's habit is to make a class of error
*fail closed* rather than to remember not to make it, and the audit it forced
today cost far more than an hour. Note the failure is currently silent in both
directions: a stale artifact cannot be detected, and a re-measurement cannot
prove which toolchain it ran on.

> **ANSWERED 2026-08-25 ([`0106`](../tasks/0106-toolchain-provenance/TASK.md)),
> at the hour it was priced at.** `tools/toolchain_provenance.py` writes a
> `toolchain.json` **next to** `design.json` — three best-effort strings:
> `mlir_aie_version`, `peano_version`, `mlir_aie_git_head` — called from both
> `export_gemm_rtp.py` and `export_xclbin.py`. The runtime reports it beside the
> `datapath` line, following 0104's absent-is-not-a-value discipline: a design
> with no `toolchain.json` reads **`UNRECORDED (design predates tasks/0106)`**,
> never a guess and never silence. **All six shipping designs re-exported**, so
> the product itself now carries `mlir_aie 1.4.2.dev16+g7e00b57`,
> `peano 21.0.0.2026080301+c9c5ecb7`, `mlir-aie HEAD 7e00b57955e1`.
>
> **Deliberate departure from [note 0009](notes/0009-toolchain-provenance.md)'s
> proposal**: the note also asked `pack_npue.py` to copy the string into the
> `.npue` header. It should not, and does not. A container is packed from
> HuggingFace weights by a tool that never invokes mlir-aie, and the
> design↔container relationship is many-to-many in both directions —
> [`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) gave MiniLM and
> bge-small *different* designs at the *same* geometry. Baking a design's build
> string into a container asserts a 1:1 relationship that does not exist, and it
> is the same mistake as putting the datapath in the container, which 0104
> rejected on the grounds that the `.npue` is byte-identical for bf16 and bfp16.
> The note is right about the value and wrong about the location.
>
> Graceful degradation was proved **against the failing case**, not only the
> working one: pointed at a non-git directory, the export still succeeds and
> records `mlir_aie_git_head: "unavailable"`. A build must not fail over
> provenance.
>
> **Two failures on the way, both worth keeping.** A re-export script claimed
> `ALL SIX RE-EXPORTED` while two had failed, because
> `$ErrorActionPreference = 'Stop'` does not apply to a native command's exit
> code — an `[XRT] ERROR: ... The device has been removed` went straight past
> it. And that same transient device removal produced a subtly wrong build that
> the exporter's **own** identity guard refused (`82 differing bytes DIVERGED`
> against a threshold of 80, not reproducible — a third run gave 68–73 and
> exported cleanly). The guard did exactly its job on a build nobody would
> otherwise have re-examined, which is a positive result for a check that is
> invisible when everything works.

<a id="t13"></a>
### T13 — Explain the pre-tiled instability · **ANSWERED 2026-08-25**

*(filed and carried as **OPEN**, re-scoped by 0093; the entry below is verbatim as it stood, and the answer follows it)*
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §4: best-case pre-tiled
runs match row-major exactly, so it is an intermittent stall, not a ceiling.
Pre-tiling was refuted as a lever, so this only matters if it is a symptom of
something else.

> **RE-SCOPED 2026-08-23 ([`0093`](../tasks/0093-t11-t12-t13-research/TASK.md)):**
> confirmed to be neither an old-toolchain artifact nor still costing production
> anything, on the two pieces of evidence available. `tasks/0058` (M11, current
> mlir-aie) re-measured 0007's exact isolated config (M=512, 4 columns,
> `ffn_down`) and reproduced **spread ~10%**, unprompted, as a migration
> regression check — and **row-major in the same session measured 0.0% over 3
> runs**, which is what rules out machine contention as the explanation. The
> instability is real on today's toolchain and was not fixed by the 1.3.4→1.4.x
> upgrade. `tasks/0085` (M13, production scale: 8 columns, batch 8192, 4 lanes)
> measured end-to-end throughput spread **under 0.6% on five of six models** —
> far tighter than an unmasked 9–22% GEMM-level instability would leave after
> dilution by host work. **Neither result tests the other's geometry**, so this
> is not a contradiction and not a close: no measurement anywhere traces the
> pretiled access pattern's per-core stability AT production scale (8 columns,
> batch ≥ 1024). That is now the precise open question — a single traced
> `--repeat` run at production geometry would settle it either direction, and it
> is the cheapest experiment left on this thread.

**ANSWERED 2026-08-25 ([`0111`](../tasks/0111-t13-pretiled-stability-at-scale/TASK.md)).**
Six measurements on a verifiably quiet array (foreign `hw_context` submission
counts identical before, during and after the whole session — liveness by
growth, not by `Active`/`Idle`).

**First, a correction to the experiment as this thread specified it.** "8
columns, traced" is not expressible: trap 7, and the harness encodes it —
`TRACE_ROUTING = {2: (1, 1), 4: (0, 0)}`, everything else refuses with `NOT
TRACEABLE`. So the traced width stayed at 4 (the widest that traces) and the
axis that had actually never been varied moved instead: `M` 512 → **8192**,
production's own row count.

| | mean MACs/cyc/core | spread over 5 runs |
|---|---:|---:|
| row-major, M=512 | 140.9 | 0.1% |
| pre-tiled, M=512 | 109.6 | **13.4%** |
| row-major, M=8192 | 140.7 | 0.1% |
| pre-tiled, M=8192 | 124.7 | **15.0%** |

M=512 reproduces 0007's own figure to the digit, so the instrument is sound;
and at 16× the rows the instability **does not dilute**. That settles the
half of the question 0093 posed — and it also reconciles
[`0085`](../tasks/0085-m13-release-sweep/TASK.md)'s <0.6% end-to-end spread,
which was dilution by host work rather than absence of the effect.

**What it actually is.** Per-invocation deltas from the stored trace: the two
arms are the *same kernel* up to p90 (median 1,450 vs 1,422 cycles; p10
identical at 1,385). Row-major has **zero** invocations above 1.5× its median
and a max of 1,500. Pre-tiled has **13 of 365 (3.6%)**, spanning 2,175–19,192
cycles, carrying **19.7% of all cycles**; their excess over the median is
**16.6% of the run**. Remove that tail and pre-tiled measures **139.8 MACs/cyc
against row-major's 140.7 — 0.6% apart**. 0007 §4's "best-case pre-tiled runs
match row-major exactly" is now a number. The outliers are bursty, not
periodic (positions 77, 78, 79, 92, 118, … gaps of 1 to 98).

**What it is not.** Not one permutation's access pattern: all four
`order × inner` variants carry it (means 118.8–129.3, none near row-major's
140.7). Not B-fetch latency deeper L1 buffering can absorb: `--b-depth 3` —
the deepest the 63 KB budget allows — leaves it unchanged (121.0, spread
13.0%).

**And it costs the shipped path nothing — pre-tiled WINS.** The same two
designs with **no trace instrument at all**, 50 iterations, quiesced NPU
(end-to-end dispatch throughput, the use `docs/05-measurement/` permits wall
clock for):

| | NPU avg µs | NPU best µs | TFLOP/s |
|---|---:|---:|---:|
| row-major | 3568.0 | 3260.9 | 2.71 |
| **pre-tiled** | **3348.0** | **2884.7** | **2.89** |

**1.066× on the mean, 1.130× on the best** — on the exact shape whose per-core
cycles call it 12% slower. Both are true: at 4 columns this dispatch is **not
compute-bound**, and per-core MACs/cyc measures the compute window *including
the core's waiting*, so a design that moves data better can finish sooner while
its traced core is recorded waiting more often. That is CLAUDE.md F2's
bandwidth-bound frame, met head-on.

So the condition 0007 §4 attached to this thread — *"this only matters if it is
a symptom of something else"* — resolves **negative**, and the thread closes.
The framing it carried for 104 tasks is what was wrong: reading a
data-movement-bound dispatch through a compute-window instrument turned a
1.13× win into a 12% loss.

**Left unmeasured, deliberately**: the column axis cannot be traced past 4, so
whether the tail changes at production's 8 columns is unanswerable with this
instrument. The untraced win above is the only evidence that speaks for the
shipped width.

<a id="t40"></a>
### T40 — What does a long-sequence design actually cost? · **ANSWERED 2026-08-25**

*(filed OPEN 2026-08-25, carried through PARTLY ANSWERED at seq 256; the entry below is verbatim as it stood, and the closing answer follows it)*
**ANSWERED FOR seq 256 2026-08-25
([`0112`](../tasks/0112-t40-seq256-nomic/TASK.md))** — the experiment named at
the bottom of this entry was run exactly as specified, two runs per arm, on an
array with zero foreign submissions. Both arms encode **the same 163,840
tokens** (4×128×64 = 4×32×256), which is what makes them comparable.

| | seq 64 (batch 128) | seq 256 (batch 32) | ratio |
|---|---:|---:|---:|
| **tokens/s** | 81,015 / 78,327 | 63,555 / 66,987 | **0.819×** |
| **NPU dispatch+wait, ms** | 1555.8 / 1581.8 | 1544.7 / 1564.3 | **0.991×** |
| host attn, ms (lane p1) | 337.5 / 340.7 | 857.0 / 829.9 | **2.487×** |
| host elt, ms | 194.9 / 209.8 | 269.7 / 261.2 | 1.312× |
| host bias, ms | 504.7 / 574.6 | 499.4 / 475.6 | 0.903× |

**Three things, in order of what they change.**

1. **The array does not move: −0.9%.** 0110 established by *reading* that `seq`
   enters only as `M = batch·seq`; this is that claim on hardware — same `M`,
   same 192 dispatches, 4× the sequence. The export was fast for the same
   reason: `M` is the JIT cache key and both tiers were already compiled.
2. **Per token, seq 256 costs 1.221×, and all of it is host-side.** Attention
   moves **2.487×** where a pure O(seq²) term predicts 4× — longer rows
   amortise the AVX2 loop's per-row overheads, discounting the scaling by about
   a third but not rescuing it. `elt` (softmax over the scores) 1.31×. The
   per-token buckets are flat, as they must be: `bias` 0.90×, `conv` 1.03×.
3. **The balance inverts and attention becomes the largest host bucket.** Array
   share of wall **76.3% → 62.0%**; attention share of wall **16.5% → 33.6%**;
   at seq 64 the biggest host bucket is `bias` (539.7 ms), at 256 it is `attn`
   (843.5 ms). **F3's conclusion may survive — its premise does not.** CLAUDE.md
   prices attention at 2–5% of the work and folding it at 1.4% end-to-end; both
   are seq-64 numbers (and, per [`0109`](../tasks/0109-fused-ratio-energy/TASK.md),
   bf16-era ones). Whether folding is *worth* it is still [T38](#t38)'s
   head_dim-64 tiling question, untouched by this.

**Extrapolated, labelled as such**: if the same sublinear factor holds, seq 512
puts attention near 2,100 ms against an unchanged ~1,555 ms of array time —
attention alone exceeding the whole array. One interval, no measurement.

**A correction to item 2 below, from the same task.** The 256-position cap is
**not** a property of the position table, so "nomic has no such table" does not
exempt it:

```cpp
// runtime/src/main.cpp:474, 597
g_max_positions = m.config_int("max_seq_len");
if (seq > g_max_positions) throw ...
```

It is a **config field**, and nomic's container carries `max_seq_len: 256` like
every other — seq 256 works only because it sits exactly *at* the limit. **seq
512 needs a repack for nomic too**, though a cheap one: `pack_npue.py:970`
writes nomic's position table as `np.zeros((max_seq, hidden))`, a placeholder
RoPE never reads.

**And a friction this thread did not know about: the validation goldens are
seq-shaped.** `--bench` refuses a seq-256 design against seq-64 fixtures
(`emb_sum.f32: expected 3145728 bytes, found 786432`), and regenerating them
needs the torch oracle plus an edit to `reference/corpus_nomic.py:18`'s
`SEQ_LEN = 64` — a hardcoded sequence length of exactly the kind 0110 removed
from `export_gemm_rtp.py`, one layer further out. 0112 edited and restored it,
with the shipped fixtures backed up and copied back. The regeneration passed its
own two-oracle check at seq 256 (1.356e-06).

**What is left open on this thread**: (a) seq **512**, unmeasured and now known
to need a repack; (b) item 3 below, the tier table — 0112's export spans tiers
4–32 (8×) against the shipped 4–128 (32×), and whether real request shapes still
fill a tier is untouched; (c) no trace was taken, so nothing above is a
kernel-cycle claim.

*(original filing)*
Filed by [`0110`](../tasks/0110-refuse-silent-truncation/TASK.md), which made
long sequences *expressible* (`tools/export_gemm_rtp.py --seq`) without making
any claim about what they cost. **Nothing in this repo has ever run at a
sequence length other than 64.**

**What 0110 established, by reading rather than measuring.** `seq` is not
compiled into anything. It reaches a design through exactly one expression,
`M = batch * seq` in `shapes_for()`, and the instruction streams only ever
know `M`, `K`, `N`; the runtime inverts the split at load
(`batch = design.M / design.seq`). So `batch 128 × seq 64` and
`batch 16 × seq 512` are the **same M = 8192 and the same array arithmetic**.
Seq is a host-side slicing convention over a fixed row count, not a hardware
limit — which is why this is a re-export rather than a redesign, and why the
question is entirely about the *host* side.

Also established: **CLAUDE.md trap 7d does not apply.** `M` is a
`CompileTime[int]` keyword argument to `pretiled_array()`, so the JIT cache key
derives from it and a changed `--seq` produces a genuinely new build. 7d would
have bitten had `shapes_for()` been a generator reading a module global; it is
not. Checked in `gemm_pretiled.py:762`, not assumed.

**The three things that are not known.**
1. **Host attention is O(seq²) and runs on the host in fp32.** F3 prices
   attention at **2–5% of the work at seq 64**. At 512 that is 64× the
   attention work per sequence, against array work that is unchanged — so the
   ratio F3 rests on inverts somewhere between 64 and 512, and nobody has
   found where. This is the whole question; the rest is bookkeeping.
2. **Position tables cap BERT-family models at 256, not 512.** The packer's
   `--max-seq` defaults to 256 and `set_design_seq()` refuses
   `seq > max_seq_len`, so a seq-512 bge-* design needs a **repack**, not just
   a re-export. nomic (RoPE) has no such table. Already loud — it fires at
   load — but it means the cheap experiment is nomic, not bge.
3. **`batch = M / seq` shrinks the tier table.** Seq 512 at M 8192 gives
   batch 16, so `plan()`'s right-sizing has four tiers spanning a 16× narrower
   range. Whether request shapes still fill a tier is unexamined.

**Cheapest decisive experiment**: export nomic at `--seq 256 --batch 32`
(same M = 8192, no repack needed since nomic has no position table), trace it,
and compare *host* time per token against the shipped seq-64 build. If host
attention has not overtaken the array by 256, 512 is worth a repack; if it
has, the answer is that windowing on the caller's side is the right design and
this thread retires.

**ANSWERED 2026-08-25 ([`0113`](../tasks/0113-t40-seq512-close/TASK.md)).**
seq 512 measured, after the repack 0112 showed was necessary. All three points
encode exactly 163,840 tokens (4×128×64 = 4×32×256 = 4×16×512).

| | seq 64 | seq 256 | seq 512 |
|---|---:|---:|---:|
| **tokens/s** | 79,671 | 65,271 | **49,637** |
| **cost per token** | 1.000× | 1.221× | **1.605×** |
| **NPU dispatch+wait, ms** | 1568.8 | 1554.5 | **1537.4** |
| NPU share of wall | 76.3% | 62.0% | **46.6%** |
| **host attn, ms** | 339.1 | 843.5 | **1720.6** |
| attn share of wall | 16.5% | 33.6% | **52.1%** |
| host elt / bias / conv, ms | 202 / 540 / 32 | 266 / 488 / 33 | 338 / 404 / 22 |

**1. The array does not move across an 8× sequence range: −2.0%**, on identical
`M` and identical 192 dispatches. 0110 derived this by reading; it is now
measured at three sequence lengths.

**2. Attention scales linearly per token above 256.** 64 → 256 is 2.49× for 4×
(sublinear — longer rows amortise the AVX2 loop's per-row overheads); 256 → 512
is **2.04× for 2×**, textbook. The discount is a small-seq effect and it is
spent by 256. **This corrects 0112's own extrapolation**, which projected
~2,100 ms at 512 by assuming the discount persisted: measured **1,720.6 ms,
22% below it**.

**3. The crossover, which is what this thread was for.** Solving
`attn(seq) ≈ 3.295 · seq` ms against a flat ~1,540 ms of array time:

> **Host attention overtakes the array at seq ≈ 470**, on nomic, at constant
> `M = 8192`.

**F3's premise does not survive that.** CLAUDE.md's "2–5% of the work" is a
seq-64 figure (and a bf16-era one, per
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md)); at 512 attention alone is
52.1% of wall clock. F3's *conclusion* — leave attention on the host — is
[T38](#t38)'s head_dim-64 question and is untouched by this.

Two per-token buckets *fall* as seq grows — `bias` 539.7 → 403.7 ms and `conv`
32.0 → 22.2 ms on identical row counts — the only argument this work found in
favour of long sequences.

**Item 2, the tier table, is answered too, and it is about SMALL requests.**
`use_tier()` rounds a job up to the smallest tier that fits, so the ladder's
*span* is irrelevant; the smallest tier's token footprint is what bites. In
token slots the ladders are 256/1024/2048/8192 at seq 64 against **2048**/8192
at seq 512 — 8× the minimum. Measured, single lane, same texts:

| request | seq 64 | seq 512 | penalty |
|---|---:|---:|---:|
| 1 text | 0.05 s | 0.29 s | **5.8×** |
| 4 texts | 0.05 s | 0.30 s | **6.0×** |
| 64 texts | 0.36 s | 4.14 s | **11.5×** |

So **windowing on the caller's side is right for a mixed workload — but for the
measured reason, not the predicted one**: not because attention makes long
sequences unaffordable (1.605× per token is affordable), but because a seq-512
design charges a 1-token request for 2,048 tokens.

**Contention, stated honestly**: two of the three seq-512 runs sit in a window
where 14 foreign `WorkloadsSessionHost.exe` submissions occurred at unknown
times. A **third run bracketed by identical samples** (zero growth during it)
reproduces them to within 0.6% / 1.2% / 1.4%, which is what retires the concern.

**Leaves behind**: [T41](#t41), the golden pipeline's hardcoded
`SEQ_LEN` — tooling debt, nothing blocked.

<a id="t38"></a>
### T38 — Does mem-tile `pad_dimensions` make attention worth folding onto the array? · **ANSWERED 2026-08-25**

*(filed OPEN 2026-08-23; the entry below is verbatim as it stood, and the answer follows it)*
Split out of [T4](CLOSED-THREADS.md#t4) rather than folded into its closure,
because it is a different question with a different answer.

[note 0007](notes/0007-unused-iron-surface.md) §1.1 proposes padding attention's
real 8-wide-per-column N=64 slice to 16 **in the mem tile**, which would make
`n=16, cols=8` legal for attention and remove 0043's derived `cols ≤ 4`
restriction.

**Two things now support it that did not before.**
[`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md) measured the penalty
decomposition (see [T4](CLOSED-THREADS.md#t4)): the column halving is worth
**1.729×** of the 2.229× total, and padding is exactly the thing that would
remove it — so the trade becomes **1.289×** on the projections instead of
2.229×. And the hardware has the feature where it is needed:
`xaie2pgbl_reginit.c` line 1667 gives `Aie2PMemTileDmaMod`
`.Padding = XAIE_FEATURE_AVAILABLE`, while the compute tile (1905) and shim
(2158) do not — matching `AIEDialect.cpp`'s verifier, which already restricts
`pad_dimensions` to mem tiles.

**What is not known**: whether it actually routes and works in an IRON design at
this geometry, and whether 1.289× on every projection GEMM is worth attention's
~2–5% (F3). Probably still negative — but "probably" is a different verdict from
0043's, and this one is worth an actual build rather than an argument.
[note 0007](notes/0007-unused-iron-surface.md) §3.4 carries whisper-xdna's
warning that fused attention was built elsewhere, was correct, and still lost.

**ANSWERED 2026-08-25 ([`0114`](../tasks/0114-t38-pad-dimensions-probe/TASK.md)) —
the build this thread asked for was made, and the mechanism WORKS.**

Three stages, geometry taken from attention itself (`ROWS=64`, `N_REAL=8`,
`N_PAD=16`, bf16, `pad_value=0`):

```
PAD    cols=1: real MATCH, padded half all-0 YES (max |pad| = 0.000e+00)
UNPAD  cols=1: EXACT  max|diff| = 0.000e+00
PAD    cols=8: real MATCH, padded half all-0 YES (max |pad| = 0.000e+00)
UNPAD  cols=8: EXACT  max|diff| = 0.000e+00
compute over a padded stream: EXACT  max|diff| = 0.000e+00
```

Every check bit-exact, which is the right gate for data movement plus a
doubling — there is no rounding in it. Four findings:

1. **It routes on all 8 columns.** That is the one that mattered: 0043's
   restriction was that a design expressing attention could use at most half the
   array, and the padded form uses all of it.
2. **The padded half is exactly the pad value**, so `n=16` over a real 8-wide
   slice is *exact*, not approximate — zero columns of B give exactly-zero
   columns of C.
3. **The un-pad is an ordinary access pattern**, `dims_to_stream=[(64,16),(8,1)]`
   on the mem tile's outbound leg — precisely as note 0007 §1.1 said.
4. **A compute tile can consume a padded stream**, a distinct question from
   whether a mem tile can make one, since the verifier forbids padding *on* a
   compute tile.

**note 0007's prediction was right, and collected.** It warned that the output
side "is where a first attempt will fail". It did: the first form put pad and
un-pad on one hop and returned **MISMATCH, 2.121e+00**, because a shim DMA
consumes the stream in order and cannot skip — a drain tap says where bytes
*land*, not which stream elements to keep. Selecting 8 of 16 is a strided read
on the **mem tile**, a different BD on a different hop, which is also the real
design's topology.

**Also caught: CLAUDE.md trap 7d, live.** A second variant was served the
first's binary (`Tensor argument 'Y' has 512 elements but the kernel was
compiled for 1024`) because `cols`/`full_width` were closure variables, which
`iron.jit`'s cache key never inspects. Fixed as `CompileTime[int]` kwargs. The
**seventh** instance of this project's "stale binary fails open" class — and the
first that failed *loudly*, only because the argument sizes disagreed.

**The value half of this thread is answered too, and its sign flips with
sequence length.** T38 inherited its cost/benefit from F3's "attention is 2–5%
of the work", which [`0113`](../tasks/0113-t40-seq512-close/TASK.md) measured the
same day to be a **seq-64 number**: array share of wall 76.3% / 62.0% / 46.6%
and host attention 16.5% / 33.6% / **52.1%** at seq 64 / 256 / 512. At seq 64
the array is the bottleneck, so folding attention onto it is the wrong direction
and **0043's verdict was right**; above seq ≈ 470 it is not.

**Successor: [T42](OPEN-THREADS.md#t42)**, which carries the design task and the
model estimate (1.65×–1.95× at seq 512, pricing nothing for softmax, per-head
movement, or the reintroduced design switch).

<a id="t34"></a>
### T34 — arch=1 is on the array: MTEB, the ratios, the host side, and `--serve` · **ANSWERED 2026-08-26**

*(carried as PARTLY ANSWERED from 2026-08-22; the entry below is verbatim as it stood, and the closing answer follows it)*

**Item 1 (MTEB) is ANSWERED**: the M8 gate **PASSES at mean +0.00**, worst task -0.00, the
tightest agreement in the project ([`0075`](../tasks/0075-m13-arch1-measurement-harness/TASK.md)).
The first run FAILED at -1.88 and that failure was the **harness** — `mteb` injects a
task-appropriate prompt into a SentenceTransformer, and `run_mteb.py` was prepending the
container's prefix on top of it, so the two sides encoded different strings. See
[T36](CLOSED-THREADS.md#t36) for the debt that leaves on nomic's recorded number.

**Item 2 (CPU ratio, energy) is ANSWERED 2026-08-25
([`0109`](../tasks/0109-fused-ratio-energy/TASK.md))**: EmbeddingGemma is no longer the one
model in the catalogue with no ratio at all. **1.392× against the strongest CPU baseline**
(110.2 vs torch's 79.2 seq/s; ORT raised the same `invalid unordered_map<K, T> key` it
raised in 0085/0105, so the ratio uses torch, which
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) established as the stronger side
anyway), interleaved per 0040. And **3.72× better energy** — 262.4 J/1000 seq against the
CPU's 976.0, RAPL differential per [`0034`](../tasks/0034-m8-energy/TASK.md).

Two honest caveats travel with those numbers, and neither is a reason to keep the item
open. The model still has **no `--bench` mode**, so both arms were driven through the
ordinary encode path rather than the one the other five use. And the pair had to be re-run
**standalone** with a temporary `--allow-contention` bypass, after manually re-verifying
the array was idle: `compare_three.py`'s gemma NPU arm passes `--guard-contention`, and
that check **misreads a genuinely idle array** (zero `hw_context` rows) as an unparseable
format change — it fails closed. That tool gap is what item 2 leaves behind; it is flagged
in 0109 and not fixed.

Item 3 (the host side) remains as filed below, and is now *more* pressing, not less:
[`0077`](../tasks/0077-m13-int8-gate/TASK.md) measured the int8 datapath at 7.7-9.6x bf16,
and a faster array makes the host share worse. 0109 sharpened it from the other end too —
gemma took the **smallest** win of the six from the host epilogue fusion (**1.153×**
against nomic's 1.340×), precisely because
[`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md) left its four-norm RMSNorm chain
unfused, exactly as [T37](CLOSED-THREADS.md#t37) had left it for int8.

Also still unbuilt, and unchanged: `--serve` refuses on arch=1
(`runtime/src/main.cpp:4172`), so this model has no HTTP endpoint.

*(original filing)*
[`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md) §11. Split out of
[T29](CLOSED-THREADS.md#t29) rather than left inside an answered thread.

1. **MTEB has never been run on this architecture, and it is now practical for
   the first time.** At the host path's 0.20 seq/s a five-task MTEB run was out
   of the question; at ~133 seq/s it is minutes.
   [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) established MTEB, not `1-cos`,
   as this project's authority for accuracy decisions — so EmbeddingGemma
   currently ships with a fidelity number and no quality number. The blocker is
   tooling, not time: `experiments/m8-npu-vs-cpu/run_mteb.py` loads models
   through the BERT-only `npu_encoder.py`.
   **This is the top item.**
2. **No CPU ratio and no energy figure.** `compare_three.py` and
   `energy_compare.ps1` both drive the `--bench` path, which arch=1 does not
   have. `release_benchmark.ps1` writes UNMEASURED for both rather than
   omitting the row — but the model is the only one in the catalogue with no
   ratio at all, and 0040's interleaving rule is what makes such a number
   defensible when it is finally taken.
3. **The host side is now half the wall clock, and its shape is known.** The
   single-lane breakdown: array 48.9%, host attention 14.7%, C readback + bias
   11.5%, RMSNorm 7.8%, bf16 convert 5.9%, GeGLU 4.5%. Two named levers follow
   — `--c-bf16` ([`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md)) halves
   the readback and was worth +4.9% on a model where it was a *smaller* share
   than it is here; and host attention is a plain AVX2 loop over
   `seq × seq × head_dim` that nothing has tried to block or tile. Note this
   model is the best case in the catalogue for lane overlap precisely BECAUSE
   the two sides are balanced (lanes buy 1.54× here against 1.19–1.35× on the
   BERT models), so making the host faster partly cannibalises that.

Also unbuilt, and smaller: `--serve` still refuses on arch=1, so this model has
no HTTP endpoint.

**ANSWERED 2026-08-26 ([`0115`](../tasks/0115-t34-serve-arch1/TASK.md)) — the
last two items, and both are done.**

**Item 3, re-measured, because two of its levers had been spent without the
thread noticing.** The breakdown above dates to before
[`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md) made bfp16 + bf16-C
gemma's shipped datapath and before
[`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md) ported the epilogue fusion
to it. On the shipped build, 520 texts, single lane, array verifiably idle:

| bucket | as filed | now | what changed |
|---|---:|---:|---|
| array | 48.9% | **36.7%** | the array got faster; the host is most of it now |
| host attention | 14.7% | **20.6%** | untouched — now the **largest** host bucket |
| RMSNorm | 7.8% | **12.7%** | untouched — now the second largest |
| C readback | (inside 11.5%) | **2.2%** | **`--c-bf16` shipped: this lever is spent** |
| bf16 convert | 5.9% | 6.3% | unchanged |
| **GeGLU** | 4.5% | **0 ms** | **0108's fusion reached arch=1** |

Of the two levers item 3 named, one is **collected** — the readback is 2.2% of
wall clock, there is nothing left in it — and one is **untouched**. What
remains is not an open question but two sized, unbuilt optimisations: host
attention, which [T42](OPEN-THREADS.md#t42) now carries with a real price on
it, and gemma's four-norm RMSNorm chain, which
[T37](#t37) explicitly deferred and 0108 deferred again for the same reason.

**`--serve` on arch=1 is BUILT, and the finding is why it was small.** arch=1
did not lack an endpoint because EmbeddingGemma is incompatible with one; it
lacked one because the endpoint was **written against the BERT encoder's
type** — 266 lines inline in the BERT path reading `EmbedService` directly. The
real coupling was three members wide (vocabulary size, prefix text, "embed
these texts"), so `serve_http()` now takes exactly those and **both
architectures run the same server**. No second implementation: the OpenAI
response shape, the four 400 cases, the base64 arm and the `InputTooLong` → 400
mapping are precisely the things that must not drift.

The host-only (`--cpu`) path serves too — not for symmetry, but because
`serve_port` is parsed before that branch and leaving it unhandled would have
made `--serve --cpu` embed a placeholder sentence once and exit. Its model id
is `<model>-cpu`, since naming a host-only server after the array is the silent
mislabel the "path HOST-only" line exists to prevent.

**Gated on agreeing with arch=1's own `--embed`:**

```
base64 vs --embed        -> max abs diff 0.000e+00
vs --embed (float arm)   -> max abs diff 5.215e-08
usage.prompt_tokens      -> 130      norms -> 1.000000
no input / empty list / bad format / token ids -> HTTP 400
```

The 5.215e-08 is the **`%.7g` JSON text form**, not an encode difference —
base64 carries raw fp32 and is bit-identical. (The first gate demanded `== 0.0`
on the float arm and failed a working endpoint.)

**Regressions, all reproducing recorded numbers exactly**: BERT endpoint PASS
with `vs --embed` 0.000e+00; MiniLM golden gate **2.430e-02 / 3.406e-04** and
gemma's differential gate **1.749e-04 / 2.315e-04**, both to the digit against
0104/0105. That bit-exactness is what says the `encode_batch` and lane-loop
edits did not touch the arithmetic.

**One consequence outside the runtime**: `tools/make_release.ps1` excluded
`artifacts_gemma_bfp16` and its comment named the reason — *"arch=1 has no
--serve/HTTP path yet"*. The reason is gone, so gemma joins what a release
ships (645 KB, the same order as every other design set).

<a id="t41"></a>
### T41 — The golden pipeline hardcodes the sequence length · **ANSWERED 2026-08-26**

*(filed OPEN 2026-08-25; the entry below is verbatim as it stood, and the answer follows it)*
Split out of [T40](CLOSED-THREADS.md#t40) rather than left inside its closure,
because it is tooling debt rather than a question about the hardware.

`reference/corpus_nomic.py:18` is `SEQ_LEN = 64`, a module constant with no
flag, and `make_goldens_nomic.py` imports it for both the padding width and the
golden filename. `--bench` refuses a design whose seq does not match its
fixtures (`emb_sum.f32: expected 3145728 bytes, found 786432`), so **every new
sequence length needs an edit-and-restore around the golden regeneration**.
[`0112`](../tasks/0112-t40-seq256-nomic/TASK.md) and
[`0113`](../tasks/0113-t40-seq512-close/TASK.md) each did exactly that, by hand,
and each verified `git diff` was empty afterwards.

This is the **same class** [`0110`](../tasks/0110-refuse-silent-truncation/TASK.md)
removed from `tools/export_gemm_rtp.py` (`SEQ = 64` → `--seq`), one layer
further out in the reference pipeline. `reference/corpus.py` and
`corpus_gemma.py` follow the same pattern, so a fix should cover all three
rather than only the one that was measured.

**Why it was not fixed in 0113**: it is a change to the *reference* oracle
pipeline — the thing every accuracy number in this project is measured against
— and doing it inside a measurement task would have put an unreviewed edit to
the oracle underneath the measurement it was producing. It wants its own task.

**Not urgent**: nothing is blocked, the manual procedure works and has been run
twice, and 0113 found the safer form of it (give the long-sequence container its
own model name, so `export_validation.py` writes to its own directory and no
shipped fixture is ever touched).

**ANSWERED 2026-08-26 ([`0116`](../tasks/0116-t41-golden-seq-flag/TASK.md)).**
`reference/make_goldens{,_gemma,_nomic}.py` each take `--seq`, defaulting to
their corpus module's constant. The mechanism is small because every `SEQ_LEN`
use is inside `main()` (checked, not assumed — only the import is at module
scope), so one `SEQ_LEN = args.seq` after `parse_args()` shadows the name for
the whole function and no call site changes.

Verified two ways, and the second is the one that matters:

| check | result |
|---|---|
| default reproduces nomic's committed s64 golden | `4521b8b4…` → `4521b8b4…`, **identical** |
| `--seq 256` vs the file 0112 made by EDITING the constant | `1ed01589…` → `1ed01589…`, **identical** |

i.e. the flag and the hand edit are demonstrably the same operation. Both runs
passed the maker's own two-oracle check at 1.356e-06.

**One thing looked like a regression and was not, and the way it was settled is
the transferable part.** `make_goldens.py` at its default rewrote MiniLM's
golden and `git status` showed it modified. Three checks, in order: it was
**deterministic** (two regenerations agreed); it was **not this change** (the
pre-change script recovered via `git show HEAD:` produced the same new hash);
and it was **not the data** — all 14 tensors bit-identical, `worst
max-abs-diff 0.000e+00`, the sole difference a metadata key (`'pooling':
'mean'`) added after that file was committed. **A golden's hash is not its
contract** — compare tensors before believing a diff. The working copy was
restored rather than a re-derived golden committed.

<a id="t3"></a>
### T3 — Device-resident intermediates (expert review §6b) · **RETIRED 2026-08-26**

*(carried OPEN since the expert review; the entry below is verbatim as it stood, and the closing measurement follows T28 below)*

**RETIRED with [T28](#t28) by [`0117`](../tasks/0117-t3-t28-repricing-retire/TASK.md)** — the re-pricing is stated once, under T28, because T3 has never had a separate build-out: it activates inside that architecture and its own 33%-of-the-encode figure is superseded there.
Raised in [note 0005](notes/0005-expert-review-tests.md) §6b, deferred *with
cause*, unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
re-priced by [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 4 at
**33% of the encode** — against §6b's original ~70 ms estimate, which never
counted the C readback. Needs the one-operator-per-core design that 0032's
16 KB program-memory wall dictates. **Activates inside the same architecture
as [T28](#t28)
and does not have its own separate build-out.** **Update 2026-08-20
([`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md))**: the
prerequisite mechanism — a tile crossing from one core to another through a
mem tile with NO host DRAM round trip, in one dispatch — is no longer just
priced, it is **built and measured at small scale** (rel_fro 9.052e-04,
correct). T3's own 33%-of-the-encode number is still unclaimed at
production scale.

**Update 2026-08-23 ([`0092`](../tasks/0092-t28-relay-bf16-output/TASK.md))**:
the mechanism for keeping an FFN intermediate fully on-device — including the
bf16-narrowed final drain to host that production L1 budgets require — is now
proven correct end-to-end at **production tile width** (rel_fro 2.510e-03).
So T3's open part is no longer *"does this work at all"*; it is *"does it work
at production SCALE and SPEED"* — items (b′) and (c) under [T28](#t28), of
which (c) is the binding one: no trace exists for that design, so nothing has
yet priced the round trip against today's four dispatches.

**Update 2026-08-25 ([`0107`](../tasks/0107-t3-t28-pricing/TASK.md))**: it is
priced now, and the answer is **park**. T3's own 33% figure predates the
datapath adoption and is superseded — see the pricing block under
[T28](#t28). The addressable share on the shipped datapath is 19.7–25.9%,
best case 1.25–1.35×, which is *smaller* than the host-side fusion
[`0082`](../tasks/0082-m13-fused-ffn-epilogue/TASK.md) already measured at
1.39–1.48× and which is not yet wired for bfp16.

**Update 2026-08-25 ([`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md))**: that
cheaper lever is now built, so the comparison above is no longer hypothetical.
The host fusion was ported to the bf16/bfp16 datapath and measured at
**1.153–1.340×**, bit-identical — landing on the *ceiling* of the relay's own
1.25–1.35× estimate with none of its open risks. T3 stays parked. But what it
is parked *against* has changed:
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md) measured array time unmoved by
the fusion (`wait (hardware)` −0.1% to +2.2%) while its **share** of wall clock
rose on every model (bge-large 46.4% → 56.7%), taking the array-infinite ceiling
from 1.87× to **2.31×**. The host lever is largely spent and the array is the
larger remaining piece again — so T3 should be re-priced against the
**post-fusion** breakdown, not 0107's pre-fusion one.

<a id="t28"></a>
### T28 — True phase fusion (0030 §4, the pipelined block-fusion design) · **RETIRED 2026-08-26**

*(filed 2026-08-20; the entry below is verbatim as it stood. T3 activates inside this same architecture and retires with it.)*
[`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) §4 and
[note 0005](notes/0005-expert-review-tests.md) proved every individual
mechanism the pipelined post-attention/FFN fusion design needs (RTP-unified
GEMM shapes, a K-augmented GELU epilogue, 8-column eltwise via split/join,
heterogeneous workers all exist in IRON) and then priced the *naive* build:
spatially partitioning one static design across the 8 columns (some for
GEMM, some for eltwise) is **roughly a wash at batch 128** — squeezing
eltwise from 8 columns to 4 to make room costs **≈+75 ms/encode** (GELU +28,
LayerNorm +14, softmax +32) against the **≈+60 ms** the eliminated design
switches save. The winning form is not spatial partition but **pipeline**:
GEMM columns streaming into eltwise columns through the mem tiles, one
dispatch per layer block, so no column idles while another works — what
AMD's 15→3 dispatch reduction, STEEL's 22.8×, and ARIES' adjacent-tile
handoff all independently measured (CLAUDE.md F1). [T3](#t3)
(device-resident intermediates, ~33% of the encode per 0044) and §6a
(micro-batch pipelining) both activate inside this SAME architecture and are
not separate builds.

**Update 2026-08-20 ([`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md),
Del B): the pipeline mechanism itself now WORKS, at small scale — a real
2-op chain, not just priced.** A GEMM core's output tile now reaches a
DIFFERENT core's GELU computation through a mem tile, in ONE dispatch, with
**no C tensor at all** in the design's I/O signature (`rt.sequence(A, B,
Y)`) — the intermediate provably never leaves the array. Correct to
rel_fro **9.052e-04** (a pure-routing `--identity` variant that swaps GELU
for a copy measures **3.550e-08**, isolating that the routing itself is
exact). Dispatch-latency cost of the extra GELU stage over a bare GEMM at
the same tiny scale: **1.088×** (wall-clock-derived, not a trace — no trace
exists for this design, see below). Getting here took five wrong designs,
each a genuine new IRON trap, none previously documented in this repo:
(1) an `ObjectFifo` cannot be BOTH a join-destination and a split/forward-
source — confirmed in the library source, `ObjectFifoLink` explicitly
forbids N:M fan-in-then-fan-out through one buffer, not just this
particular attempt at it; (2) a bare point-to-point `ObjectFifo` with no
`.forward()` compiles but **hangs the hardware** at runtime with no
diagnostic; (3) **a hand-built `TensorAccessPattern` for a plain
full-region copy compiles cleanly and HANGS THE HARDWARE** — root-caused via
a from-scratch diagnostic that reproduced the hang even with NO cross-core
routing at all (ruling out row count and the pipeline mechanism itself),
fixed by using `TensorTiler2D.simple_tiler(dims)[0]` for the identical
logical access pattern instead; (4) a mem-tile JOIN of multiple producer
tiles needs an explicit `dims_to_stream` unscrambling formula even on this
tiny 2-tile case (gives a WRONG but FINITE result if omitted — not a
crash); (5) that fix appeared to do nothing on first retest because the
JIT's cache purge marker was too coarse to notice a `dims_to_stream`-only
source change, silently serving a stale pre-fix binary — a THIRD, distinct
instance of the marker-specificity fail-open class 0030 and 0053 already
found two of. Full account: [`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md)
Problems #1-6.

**NOT attempted, and the reason is architectural, not time**: the full
production ffn_up→GELU→**ffn_down** THREE-stage chain. ffn_down's
K-reduction needs ffn_up's ENTIRE N=4h output, which is split across every
GEMM COLUMN's own N-slice — feeding it to ffn_down is a many-to-many
regather across columns, not the 1:1 single-mem-tile hop this session
built. Whether that regather is even expressible given finding (1) above
(no N:M through one buffer) is itself the open question a future session
needs to answer before attempting the build. `xrt::runlist`
([T9](CLOSED-THREADS.md#t9), retired) and
`disable_synchronization`+`delegate_tile` remain unused — this session's
designs are single-dispatch probes with no concrete use for either yet.

**Upstream check, 2026-08-20 (no build, research only)**: read
`verifyObjectFifoLinks()` in `AIEObjectFifoStatefulTransform.cpp` at both our
checked-out commit (`ed23bba`, 2026-07-01, part of installed mlir-aie 1.3.4)
and the current GitHub `main` (past released v1.4.1, 2026-08-11) — **byte-for-
byte identical**. The one-`ObjectFifoLinkOp`-per-`ObjectFifo` rule (finding
(1) above) is a deliberate, tracked structural invariant, not something a
newer mlir-aie relaxes; a future session should not expect an upgrade to open
the N:M regather. What upstream DOES have that we don't: **`--aie-objectfifo-
liveness`** (PRs #3257/#3312, landed 2026-07-09/12), a new opt-in static pass
that turns one class of "compiles clean, hangs hardware, zero diagnostic"
bugs — coupled cyclic multicast under-buffering — into a compile-time error.
Whether it would have caught findings (2)/(3) above (plain point-to-point /
`TensorAccessPattern` hangs, no cycle involved) is unconfirmed; the PR
description scopes it to cyclic coupled multicasts specifically and lists
plain acquire-in-loop exhaustion as explicitly out of scope. **Not pursued**:
our installed mlir-aie is a built/installed 1.3.4, not a git checkout on a
branch — getting to 1.4.1 means sourcing or building a new wheel, a real
infrastructure undertaking CLAUDE.md already flags as high-risk ("a pip
install accident must not break the toolchain that took the most work to get
running"), and out of scope for a research session. Recorded so nobody
re-checks this before a deliberate toolchain-upgrade decision.

> **STALE, corrected 2026-08-23: we HAVE the liveness pass, and have had it
> since the 1.4.2.dev16 upgrade in [`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md).**
> `aie-opt --help` lists it in the installed build, with a description sharper
> than the PR text above:
>
> ```
> --aie-objectfifo-liveness   Flag a specific under-buffered coupled-multicast
>                             objectFIFO class that hangs at runtime
>                             (not a general deadlock detector).
> ```
>
> So the paragraph above is right that it is narrowly scoped and wrong that it
> is unavailable — the upgrade it says would be needed already happened, four
> tasks later, and nobody re-read this. Exactly the failure mode rule 3 exists
> for, found while evaluating whether to move to the tagged v1.4.2.
>
> **It is free to try and has not been tried.** The neighbouring
> `--aie-verify-runtime-rearm` is also present. Given
> [`0092`](../tasks/0092-t28-relay-bf16-output/TASK.md) spent a session on a
> hang with no diagnostic, running the relay design through both passes is a
> cheap first move for whoever picks this thread up.

**Update 2026-08-20 ([`0057`](../tasks/0057-m10-t28-cross-column-regather/TASK.md)):
the cross-column regather question 0054 left open is now ANSWERED — it is
expressible at reduced scale and PRECISELY port-budget-blocked at
production (8-column) scale, with the exact numbers coming from the
compiler itself, not a count taken after the fact.**

Two things were tested, independently, both against a real numpy reference
(never a device read-back, trap 6c):

- **A join's own `.cons()` works as an ordinary Worker input for further
  on-chip compute — no second link.** 0054's restriction (an `ObjectFifo`
  cannot be both a join-dest and a split/forward-source) turns out to be
  specifically about calling `.split()`/`.forward()`/`.join()` a SECOND
  time on an already-linked object — every one of 0054's failed attempts
  did exactly that. Simply handing the join's OWN `.cons()` to a third
  `Worker`, the same pattern `B_fwd.cons()`/`A_l2l1_fifos[row].cons()`
  already use everywhere in this codebase, is a different, unrestricted
  operation: **PASS, rel_fro 3.550e-08**, matching 0054's own `--identity`
  control number to the digit.
- **Cross-column JOIN routing works, at ANY distance across the array.**
  Two independent `[TM,K]x[K,TN]` GEMMs, each fed from ITS OWN column's
  shim, computed by cores in DIFFERENT physical columns, `.join()`-ed into
  ONE mem tile pinned at a third column: **PASS at rel_fro 3.474e-08 for
  both an adjacent-column pair (0->1) and the maximum-distance pair spanning
  the WHOLE array (0->7) -- identical number, distance was free.** (One
  probe-construction bug on the way looked exactly like a routing failure
  and was not: both columns' host-side fill taps pointed at buffer offset
  0, so both GEMM cores silently computed the SAME problem -- diagnosed by
  per-column rel_fro showing column 1's output byte-identical to column
  0's, not garbage, and fixed by tiling the full multi-column tensor
  instead of building N identical single-tile taps by hand.)

**The wall, quantified by the compiler at the exact point it bites:** an
8-source join needs 8 mem-tile input ports; no mem tile has more than 6
(CLAUDE.md trap 3b / [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) /
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md), now compiler-stated
rather than counted from placed MLIR): `"tile (0, 1) requires 8 input/1
output DMA channels, but only 4 input/4 output available"` when the
destination tile also does its own local `ffn_up` A/B feed (6-2=4, exact),
`"...requires 7 input/1 output... only 6 input/6 output available"` for a
dedicated gather-only tile (the bare 6-port ceiling), and a THIRD distinct
failure at exactly 6 sources -- the join itself fits, but relaying the
gathered result back OUT through the SAME mem tile needs a 7th port
(`"only 0 input/5 output available"`). **A single JOIN cannot express the
full 8-column ffn_down regather in one hop, full stop, on this hardware.**

> **CORRECTION 2026-08-23: the paragraph below was filed as a derivation and
> it was already BUILT AND PASSING.**
> [`0062`](../tasks/0062-m11-t28-hierarchical-merge/TASK.md) built the
> hierarchical 2-hop merge on 2026-08-20 — real GELU at every producer, two
> merge mem tiles each also serving its own column's A/B feed, a relay core
> reading both hops through its two input channels via direct `.cons()`, and a
> real second-stage matmul. **PASS on hardware**, rel_fro 1.726e-03 against a
> 3e-2 bound, bit-identical on repeat.
>
> **This thread never recorded it**, so a later session re-derived on paper what
> was sitting in the tree. That is rule 3's own failure mode, and the register
> is the thing rule 3 says is the authority — so this is the second time it has
> cost a session, and this time to the register's own reader. The derivation
> below is kept because its arithmetic is right and matches what 0062 built;
> read it as corroboration, not as news.
>
> **MEASURED 2026-08-23 ([`0083`](../tasks/0083-m13-join-port-budget/TASK.md)):
> the port limit is 5 with an out-relay, 6 without — and GROUP=4's failure was
> never a port failure at all.**
>
> 0057 read three different walls off one data point. At production tile size a
> 4-way join dies on **L1** (`Basic sequential allocation failed`, two 49,152 B
> buffers) before the port count is ever reached. Re-probed at 512-float tiles,
> where L1 cannot be the constraint:
>
> | sources | verdict |
> |---:|---|
> | 2, 3, 4, **5** | **PASS, bit-exact** |
> | 6 | `requires 1 input/1 output … only 0 input/5 output` — the join takes all six, the out-relay gets none |
> | 7, 8 | `requires 7/8 input/1 output … only 6 input/6 output` |
>
> So **8 → 5+3 → 1 fits in two tiers**, and 8 → 4+4 → 1 has a spare port each.
> 0062's GROUP=2 build had more headroom than it used.
>
> And **production geometry fits a STREAMING relay**: `ffn_down`'s K-reduction
> is a reduction, so the relay need never hold the whole K chunk — one k-block
> at a time, exactly as `gemm_pretiled`'s own K-loop does. At `(k, N_DOWN) =
> (64, 48)` — production's own tile — that is **47,104 B of 64,512, 27% spare**.
> The 49,152 B wall 0062 hit was a property of its one-shot gather, not of
> production. Arithmetic, not a measurement: nothing yet says it routes or pays.
>
> **What 0062 leaves genuinely open** (its §"What was not attempted"):
> * **Production scale.** Its probe runs `K_total=192`, `N_DOWN=16` against
>   production's `K=1536`, `N=384`. Scaling needs either **more than 2 hops** —
>   impossible into one core, which has exactly 2 input channels (trap 3b) — or
>   a different relay strategy: smaller `N_DOWN` chunks per hop, or a **third
>   tier** of hierarchy. *"Not designed or estimated this session."*
>
>   **ATTEMPTED 2026-08-23 ([`0087`](../tasks/0087-m13-relay-production-width/TASK.md)).**
>   `N_DOWN = 48` — production's own tile width — **allocates and compiles once
>   the output is narrowed to bf16**, and then **hangs the hardware**
>   (`ERT_CMD_STATE_TIMEOUT`).
>
>   Two results worth carrying forward. First, **0083's L1 budget was
>   optimistic by exactly 8,192 B** — it assumed a bf16 output (6,144, not
>   12,288) and forgot the 2,048-byte stack trap 3 names explicitly. The
>   compiler's own map reads 69,632 of 65,536 at fp32 output; narrowed it is
>   63,488. So **the relay needs C-narrowing for the same reason the GEMM did**
>   (0080), reached from the opposite direction: there it bought bandwidth,
>   here it buys the address space to exist.
>
>   Second, **the hang is the narrowing, not the width** — `N_DOWN = 16` with a
>   narrowed output hangs at a geometry that passes with fp32, which rules out
>   the port budget, the weight size and the k-block streaming in one
>   comparison. `narrow_f32_bf16.cc`'s own documented first suspect (stack, run
>   at 0xD00 in gemm_pretiled) is **refuted** — raised from 0x800 to 0xD00 and
>   it still hangs. Next suspects, cheapest first: the output fifo's element
>   type independent of the kernel (the drain tap's sizes are in *elements* and
>   nothing tells the shim they got narrower), then `aie::set_rounding` as
>   core-wide state across the relay's now-three kernels.
> * **GROUP=4 with real second-stage compute**, ruled out by L1 arithmetic
>   (49,152 B for the one-shot gather alone) but *"a version that streams
>   GROUP=4 in smaller sub-chunks might still fit — not investigated"*.
> * **Whether a second `dims_from_stream` on a join's `.cons()` composes with
>   the base object's own `dims_to_stream`** — which is what would let the relay
>   use the MMAC-accelerated `kernels.mm()` instead of a hand-written matmul.
>   Untested in any session.
>
> ---
>
> *(the original derivation, 2026-08-22, from 0057's compiler-stated numbers)* The wall
> is per-tile, and 0057 measured it at a *flat* 8-way gather. Split it:
>
> | | sources | ports needed | mem tile has |
> |---|---:|---|---|
> | flat, dedicated gather tile | 8 | 7 in / 1 out (compiler-stated) | 6 / 6 — **OVER** |
> | **hierarchical, level 1 (×2 tiles)** | 4 | **4 in / 1 out** | 6 / 6 — fits |
> | **hierarchical, level 2 (×1 tile)** | 2 | **2 in / 1 out** | 6 / 6 — fits |
>
> This is exactly the shape CLAUDE.md trap 3b already prescribes for the other
> port-limited case: *"a shimNOC DMA has ≤6 S2MM channels, so a flat 32-way
> join is inexpressible — join hierarchically through the mem tiles."* The same
> sentence applies here and nobody connected it to `ffn_down`.
>
> **The structural objection has an answer too, and it is 0057's other
> finding.** Level 1's join destination must reach level 2, and 0054 finding
> (1) forbids calling `.forward()`/`.split()` on an already-linked
> `ObjectFifo`. But 0057 established that handing a join's **own `.cons()` to a
> further Worker is unrestricted** — so the relay is a *core* doing a copy, not
> a second link on the same buffer. That costs a tile per level, and
> [`0081`](../tasks/0081-m13-int8-everywhere/TASK.md) §3 measured the array
> idle for ~70% of an encode, so tiles are the resource we have most of.
>
> **Not built, and not free**: two extra hops add latency to a chain whose
> whole purpose is removing it, and the relay cores' L1 must hold a full
> `ffn_up` N-slice. Whether the round trip beats today's four dispatches is the
> measurement. But *"inexpressible"* was the blocker, and it is only true of
> the flat form.



**What this means for T28: the full production-scale three-stage chain is
NOT proven inexpressible -- it is proven to need a HIERARCHICAL 2-hop merge**
(e.g. two 4-way joins, each exactly at the 6-port budget when its tile also
carries local A/B traffic, landing in two different mem tiles, read by a
relay/`ffn_down` core using its 2 input DMA channels, one per hop) -- an
architecture whose two load-bearing sub-mechanisms are now BOTH proven
correct on hardware, but which was not itself built this session. A
separate, non-fundamental L1 wall showed up in the small-scale probes (a
gather-consumer acquiring a whole flat joined tile at once tops out around
N=2 at this tile size) and is not a blocker -- it just means a real
`ffn_down` consumer must stream k-blocks, exactly like every other GEMM
core in this codebase already does, not gather-then-copy in one piece.
Full account, including every raw compiler error: [`0057`](../tasks/0057-m10-t28-cross-column-regather/TASK.md).

**BUILT 2026-08-20 ([`0062`](../tasks/0062-m11-t28-hierarchical-merge/TASK.md)):
the hierarchical 2-hop merge itself is no longer just budgeted -- it is
built, and PASSES on hardware, with REAL compute at every stage, not the
`identity_copy_*` diagnostic 0054/0057 both used.** 4 GEMM+GELU producer
columns (a real `kernels.mm()` GEMM, then a real GELU epilogue narrowing
straight to bf16 on the SAME core, new kernel
`gelu_epilogue_3072_f32_to_bf16`) feed two merge mem tiles (`.join()`,
GROUP=2 columns each, each mem tile ALSO carrying its own column's local
A/B feed -- the double-duty topology this thread named), read by ONE relay
core over its two input channels via direct `.cons()` (Q1's mechanism, not
`.forward()`), hop-by-hop: acquire hop0, partial-matmul-accumulate,
release; acquire hop1, partial-matmul-accumulate, release -- the
acquire/release-per-hop discipline 0057 flagged as untested. The relay's
own second-stage matmul is a hand-written (not `kernels.mm()`) vectorised
kernel producing a real, non-trivial `ffn_down`-shaped output. Checked
against an independent fp64 reference (never a device read-back, trap 6c):
**rel_fro 1.726e-03**, reproduced at a different seed (1.783e-03) and
bit-identically on repeat -- 17x inside the 3e-2 tolerance set going in.

**Scale reduction, and why**: GROUP=2 (not 0057's tightest GROUP=4
boundary case) and the merge/gather data narrowed to bf16 (not fp32) -- a
REAL second-stage matmul needs L1 for a resident weight and accumulator on
top of the gathered hop buffers, and 0057's own GROUP=4 one-shot gather
ALONE already used the entire one-shot L1 budget with nothing but a copy
kernel in it. `N_DOWN=16` (not production's 48) is a code-simplicity
choice only. Full L1 arithmetic and the port topology are in the task log.

**One new build trap found**: pulling more than one `ExternalFunction`
entry point from the SAME multi-symbol `.cc` file within ONE design
duplicate-symbol-fails the link -- every requested entry point compiles
the WHOLE file, with no per-symbol extraction, so N requested symbols from
one file link N copies of everything in it. No prior design in this
codebase had ever asked for more than one symbol per file per build
(`gelu_poly.cc` and `narrow_f32_bf16.cc` both hold several, but every
existing caller picks exactly one). Fixed by splitting the three new relay
kernels into three single-symbol files. Full error text and the fix are in
the task log.

**Still not attempted, and this is where the thread's remaining gap now
sits, precisely**: (a) whether `kernels.mm()`'s MMAC operand order composes
with a join's own `dims_to_stream` join-undo transform -- unexplored, and
the reason this session's relay matmul is hand-written rather than
MMAC-accelerated; (b) any relay design at GROUP=4 or N_DOWN=48
(production scale) -- the L1 arithmetic in the task log shows why a
one-shot version of either overflows, and no streamed/chunked alternative
was designed or estimated; (c) the third tier a full 8-column, K=1536
regather would need, since a relay core's 2-input-channel ceiling (trap
3b) is fixed regardless of how the first two tiers are built. T28 stays
**OPEN**: the hierarchical merge mechanism is now proven correct with real
compute, not just arithmetically sound, but full production scale is
neither built nor decisively ruled out.

**RESOLVED 2026-08-23 ([`0092`](../tasks/0092-t28-relay-bf16-output/TASK.md)):
the bf16-output hang is found and fixed, and `N_DOWN = 48` — production tile
width — now PASSES at rel_fro 2.510e-03.** Item (b) above is therefore
half-answered: production *width* is built and correct. The relevant control
is 0062's fp32 `N_DOWN=16` at rel_fro 1.726e-03, reproduced digit-for-digit;
narrowed to bf16 the same geometry gives 2.306e-03, and at production width
2.510e-03 — all against the probe's own independent fp64 reference, never a
device read-back (trap 6c), all inside the 3e-2 tolerance.

**The cause was outside the chip, and five on-chip suspects were refuted
first.** 0092 killed the narrowing kernel, the relay's stack, the fifo depth,
contention at the auto-placed mem tile, and — decisively — *all upstream
compute*: a kernel that ignored its inputs entirely and wrote a hardcoded
constant to the bf16 output still hung. Then a diff of the emitted
`aiex.dma_configure_task_for` / `aie.dma_bd` ops between the passing fp32
build and the hanging bf16 build of the **identical** design found it: the
design's host-facing argument type was **hardcoded**

```python
Y_ty = np.ndarray[(Y_SIZE,), np.dtype[np.float32]]     # never read NARROW_OUT
```

while `A_ty` and `B_ty` two lines above were correctly conditional. So the
shim DMA was configured to move **4,096 B against a host buffer XRT had
allocated at 2,048 B**: the mem-tile side signalled completion after 2,048 B
and the shim side waited forever for bytes that were never coming. Compiles
clean, transfers forever, no diagnostic — which is why five rounds of
looking *inside* the array found nothing. One line.

A second, independent bug surfaced immediately after, on the very next
production-width attempt: `Y_out` was needlessly double-buffered when
narrowed (`depth=2`), overflowing L1 by 5,376 B. Also one line.

**And 0087 §4's discriminator was invalid**, though its conclusion survives.
`NPUE_BF16_COPY=1` declared the relay accumulator `bfloat16` (2,048 B) while
still linking `ffn_down_hop_matmul_g2_64x48x16`, whose C++ signature takes
`float *restrict acc` — IRON's `arg_types` shapes only the MLIR call-site
declaration, never the linked object. Confirmed with the allocation map and
`llvm-objdump` (64-byte stride × 64 rows = 4,096 B written through a 2,048 B
pointer). A correctly-typed bf16-accumulator twin was built and **still
hung**, so the conclusion held — but only via the corrected test. The
original test proved nothing. 0087 carries dated corrections for both.

**What T28 still needs, and none of it is a hang:**
* **(a)** whether `kernels.mm()`'s MMAC operand order composes with a join's
  own `dims_to_stream` — 0062's original question, still unchased, and the
  reason the relay's second-stage matmul is hand-written rather than
  MMAC-accelerated. (Separately, [`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md)
  measured that hand-written kernel's inner loop carrying T7's emulated-widen
  idiom, so it is known to be inefficient as well as non-MMAC.)
* **(b′)** scaling past this probe's `GROUP=2` / `K_total=192` to production's
  full `K=1536` across 8 columns. A core has exactly 2 input channels
  (trap 3b), so this needs a third hierarchy tier or a different relay
  strategy. Not designed.
* **(c)** **no hardware trace exists for this design** — too small and too
  many mem-tile hops to route a trace flow (trap 7) — so there is still
  **no performance claim here, only correctness.** Whether the round trip
  beats today's four dispatches remains the measurement T28 was always
  about.

**PRICED 2026-08-25 ([`0107`](../tasks/0107-t3-t28-pricing/TASK.md)) — and the
recommendation is to PARK the production build.**

The relay is correct at production tile width and still has no trace, so it was
priced from the models instead of measured. On the datapath that actually ships
(bfp16-emulated MMAC + bf16-C, five of six models per
[`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md)), **best case, assuming
zero rebuild overhead**:

| model | addressable share of encode | best-case speedup |
|---|---:|---:|
| MiniLM | 25.9% | **1.35×** |
| bge-base | 23.9% | **1.31×** |
| bge-large | 19.7% | **1.25×** |

The relay removes host GELU in full plus an N-proportional share of the
transport bucket (**55.5%** — an assumption, stated as one, resting on
`qkv:attn_out:ffn_up:ffn_down = 3:1:4:1` being architecture-invariant for BERT
models). Every open item under T28 can push these numbers **down**, not up.

**Why it should be parked, in one comparison.**
[`0082`](../tasks/0082-m13-fused-ffn-epilogue/TASK.md) already measured
**1.394 / 1.417 / 1.418 / 1.428 / 1.482×** on hardware — reproduced at three
runs per arm in [`0084`](../tasks/0084-m13-host-isa-and-repeats/TASK.md) — for a
plain **C++ host-side fusion**, no array work at all. That is *larger* than the
relay's best case, for a fraction of the engineering risk. And it was built for
**int8 only**: it is **not wired for the bfp16 datapath five of six models now
run**. So the cheaper lever is both better-measured and unbuilt on the datapath
that matters.

**Recommendation: port 0082's host fusion to bfp16 first, then re-price the
relay against whatever gap is left.** Note the adaptation is not free — 0082's
chain fuses dequantise → activate → normalise → quantise, and the bfp16 path has
no quantisation step, so it is a rewrite of the same shape rather than a flag.
That cost is unestimated and should be the first thing the next session bounds.

**DONE 2026-08-25 ([`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md)) — the
recommendation was executed, and the adaptation cost it asked to have bounded is
bounded.** The bf16 FFN chain turned out to have the *same three-pass structure*
as int8's minus the quantisation step, so it extended 0082's own
`FusedNext`/`gemm()` mechanism rather than becoming a parallel one; arch=1 was
ported too, leaving Gemma's four-norm RMSNorm chain unfused exactly as 0082 left
it. Addressable share was **measured before building, not assumed**:
**17.7–28.3%** of wall clock. A/B, three runs per arm, idle machine,
bit-identical output (vectors hash the same fused vs `--no-fuse-ffn`, and every
model reproduces 0105's `1-cos` to the digit): MiniLM **1.265×**, bge-small
1.208×, bge-base 1.251×, bge-large 1.223×, **nomic 1.340×**, gemma 1.153×.

**It strengthens the park verdict rather than weakening it.** A pure C++ host
rewrite reached the relay's *best case* (1.25–1.35×), so the relay would now
compete for a smaller remaining gap than the one it was priced against. And the
gap that is left has moved **onto the array**:
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md) measured array time unchanged
by the fusion but its share risen on every model, array-infinite ceiling
1.87× → **2.31×**. When this thread is next opened, price it against that
breakdown — the relay's target is now the larger half of wall clock, not the
smaller.

**T3's own 33%-of-the-encode figure is dated** to the pre-adoption plain-bf16 /
fp32-C datapath, and is superseded here: `--emulate-bfp16`'s array speed-up and
`--c-bf16`'s byte-halving already collected part of that prize **without anyone
building the relay**. Note also that 0081's 30.4% / 1.44× ceiling is an **int8**
figure and is the wrong bound to compare against — the analogous bound on the
shipped bfp16 datapath is *larger* (1.36×–1.87×), because emulated MACs are
slower than native int8 ones, which leaves more array time to remove.

> **CORRECTION to item (b′) above, and it was mine.** (b′) says scaling to
> production needs *"a third hierarchy tier or a different relay strategy"*.
> That is wrong, and it contradicts
> [`0083`](../tasks/0083-m13-join-port-budget/TASK.md) — appended to this very
> file earlier the same day — which measured that an 8-column regather **fits in
> two tiers**: `8 → 4+4 → 1` or `8 → 5+3 → 1`, since the relay core's two input
> channels are exactly what a 2-tier topology needs. I conflated the column
> count with the K-reduction depth, and the relay streams k-blocks so `K = 1536`
> costs no extra tiers at all.
>
> The real remaining scope of (b′) is narrower: extend the already-priced 2-tier
> topology from the built `GROUP=2` to a full `4+4` or `5+3` at production
> `N_DOWN = 48` and the full `K = 1536`. Item **(a)** — whether `kernels.mm()`'s
> MMAC operand order composes with a join's `dims_to_stream` — remains the
> genuinely unexplored one, and it is why the relay's matmul is hand-written and
> [`0091`](../tasks/0091-t7-t8-gelu-poly/TASK.md)-measurably slow.

**RETIRED 2026-08-26 ([`0117`](../tasks/0117-t3-t28-repricing-retire/TASK.md)) —
re-priced on the build that actually ships, which is what this entry asked for.**

Post-fusion single-lane breakdown, both ends of the catalogue, array verifiably
idle (bge-large's 56.6% array share reproduces
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md)'s 56.7% in a separate
session):

| | MiniLM | bge-large |
|---|---:|---:|
| wall | 111.35 ms | 1652.69 ms |
| **wait (hardware)** | 35.64 ms (32.0%) | **935.07 ms (56.6%)** |
| read out + bias | 22.08 ms (19.8%) | 242.75 ms (14.7%) |
| bf16 convert + sync | 12.48 ms (11.2%) | 80.64 ms (4.9%) |
| host attention | 22.33 ms (20.1%) | 198.08 ms (12.0%) |
| host layernorm | 12.38 ms (11.1%) | 163.27 ms (9.9%) |

The relay can remove only the `ffn_up`→`ffn_down` leg's share of the transport
bucket — **44.4%** of it by N (`3:1:4:1` is exact for both models, not fitted),
since after [`0108`](../tasks/0108-fuse-epilogue-bfp16/TASK.md) the host GELU is
no longer a bucket of its own but lives inside the fused read-out pass. That is
precisely why the target shrank:

| | MiniLM | bge-large |
|---|---:|---:|
| relay-addressable share of wall | **13.8%** | **8.7%** |
| **best case, zero rebuild cost** | **1.160×** | **1.095×** |

against 0107's 1.25×–1.35×: roughly halved.

**The verdict does not rest on the 44.4% split.** Credit the relay with the
*entire* transport bucket — every byte for all four shapes, which it cannot
remove — and the bound is 1.450× / **1.242×**. On bge-large even that absurd
upper bound is barely above the **1.223× 0108 already delivered on hardware**
for a C++ rewrite with no array work at all. And three costs are priced at zero
in it: array time *rises* (GELU moves on-core, on the model where the array is
already 56.6%), the relay's matmul is hand-written and slow (item (a), never
explored), and a separate design shape can reintroduce switches
([note 0004](notes/0004-context-switch-cost.md): ~25 µs + 7.2 µs per lock).

**So: mechanism proven, value collected cheaper.** The relay is correct at
production tile width ([`0092`](../tasks/0092-t28-relay-bf16-output/TASK.md),
rel_fro 2.510e-03) and its pipeline works end to end
([`0054`](../tasks/0054-m10-phase-fusion-pipeline/TASK.md)); nothing here
refutes the engineering. It retires on the same ground
[T2](#t2)/[T9](#t9)/[T12](#t12) did — a measured "not worth it" is a complete
answer.

**T3's 33%-of-the-encode figure is three datapath generations old** (it predates
`--emulate-bfp16`, `--c-bf16` and 0108's fusion, each of which took a piece of
what it counted) and should not be quoted again.

**Trigger for reopening, so nobody re-derives it**: only if the HOST side grows
back to dominate. [`0113`](../tasks/0113-t40-seq512-close/TASK.md) measured
exactly that at seq 512 (host 75%, attention alone above the array) — but there
the dominant host work is *attention*, not the FFN leg this relay addresses, so
the lever is [T42](OPEN-THREADS.md#t42), not this one.

<a id="t49"></a>
### T49 — Nothing detects a closed thread whose closing premise expired · **ANSWERED 2026-08-27** · check 1 is in `check_register.py`; check 2 resolved editorially
The generic form of [T48](OPEN-THREADS.md#t48), and the third distinct failure mode this
register has produced.

The first two are on record: a live question sitting unread after its blocker
went away ([`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md), twice), and a
thread misleading its own reader by running to 308 lines without naming the task
that had already built the thing it still called open — which is what produced
`tools/check_register.py`. **This is the third**: a thread that left the register
*correctly*, by the rules, with a pointer — and whose closing argument stopped
being true afterwards, because this project changed its own configuration.

Two instances, both found by [`0120`](../tasks/0120-roofline-analytic/TASK.md) §4b:

* **[T27](#t27)** closed on *"bfp16 was never adopted"*.
  Adopted five tasks later
  ([`0104`](../tasks/0104-adopt-bfp16-per-model/TASK.md)).
* **[T2](#t2)** carries an in-place block reading *"REOPENED
  for the int8 datapath, 2026-08-22"* and **never moved back** to this file.
  `CLAUDE.md` has stated "three live threads" ever since.

**Why the existing check cannot catch it.** `tools/check_register.py` matches a
task's **title** against the threads and asks whether the thread links back.
That catches work done and not recorded. Premise decay is the opposite shape:
the thread is internally consistent and its *world* moved.

**Two candidate checks, cheapest first.**

1. **A closed thread whose body contains "REOPENED" with no counterpart in
   `OPEN-THREADS.md`.** Trivial, exact, no false positives by construction — and
   it would have caught T2 on the day the annotation was written.
2. **A closed thread naming a configuration `design.json` no longer matches.**
   T27's own title says *"which replaced bfp16 as the fast datapath"* while five
   of six shipped designs carry `emulate_bfp16: true`. Grepping closed threads
   for `a_dtype`/`emulate_bfp16` claims against the shipped design files is more
   work, and would have caught T27.

Check 1 is worth writing regardless. Check 2 is worth **scoping before
writing** — the honest fix may be editorial rather than mechanical: *a closed
thread should name the condition its closure depends on*, so a reader sees in
one line whether it still holds.

**Trigger**: before the next release sweep. Rule 3 already says
`check_register.py` runs then and exits non-zero.

> **ANSWERED 2026-08-27
> ([`0123`](../tasks/0123-register-hygiene/TASK.md)).** Check 1 is built:
> `tools/check_register.py` now fails on a closed thread carrying a bold
> REOPENED annotation that neither has a section back in `OPEN-THREADS.md` nor a
> later `SUPERSEDED by [Tnn]` block after the annotation. Proven on a planted
> copy of this file with T2's superseding block removed — it fires, exactly as
> it would have on the day T2's annotation was written. The bold marker is the
> trigger on purpose, so prose that merely mentions the word — this very
> paragraph — cannot false-positive.
>
> Check 2 was scoped and resolved **editorially, not mechanically**, as this
> thread itself suspected: grepping closed threads for `a_dtype` /
> `emulate_bfp16` claims against shipped `design.json` files would catch only
> the datapath flavour of premise decay while missing every other kind (T27's
> premise was a *decision* not a config field until 0104 made it one). The
> durable fix is the sentence now in this file's preamble: **a closure names
> the condition it depends on**, so a reader — and a future grep — sees in one
> line whether it still holds. Enforcement stays with the release-sweep
> register check plus the reader; a mechanical check with a false-negative
> guarantee would have been worse than none.

<a id="t47"></a>
### T47 — `--probe-streams` and the roofline disagree about bytes, and the tool is the one that is wrong · **ANSWERED 2026-08-27** · fixed, rebuilt, and every shipped set re-probed
Found by [`0120`](../tasks/0120-roofline-analytic/TASK.md) §4a while making the
two agree. `runtime/src/main.cpp:6484` computes the same traffic quantity as
`tools/roofline.py` with **two `design.json` fields hardcoded as constants**:

| hardcoded | true value | which shipped design breaks it |
|---|---|---|
| `2` as the A/B element size | 1 under int8 | every `--int8` container |
| `48.0 * 8.0` as `tile_n * cols` | 32·8 at bf16 bge-large, 64·8 at int8 bge-large ([`0081`](../tasks/0081-m13-int8-everywhere/TASK.md)) | `large_bfp16`, `bge-large-en-v1.5.int8n64.npue` |

**Measured effect on the one table audited**
([`0097`](../tasks/0097-t18-t21-t4-measurements/TASK.md) t21, MiniLM int8): the
`GB/s` column read 56.5 / 35.9 / 57.4 / **58.5** and should read
35.9 / 22.8 / 36.8 / **31.7**. That is **1.57× to 1.85×**, and *differentially*
— C is counted correctly and C's share varies by shape, so the error is largest
exactly where C is smallest. The printed column made three of four shapes look
flat, which is the tool's own stated signature for "traffic-bound".

**The conclusion t21 drew survives** — `GMAC/s` still spreads 2.1× against
bandwidth's 1.16× — so nothing needs re-deciding. The number was wrong and the
flatness it displayed was partly an artifact.

**Deliberately not fixed in 0120**, so the two implementations can be diffed
rather than silently reconciled. **Two things are open, not one**: the fix
(~5 lines, reading the operand dtype out of the design the runtime has already
loaded, exactly as it already reads `c_elem_bytes`, plus a rebuild), and the
**audit** — 0120 checked t21 line by line and nothing else. Any published
`GB/s` from an int8 or bge-large probe is suspect until someone looks.

**Trigger**: before the next time a `GB/s` figure is quoted anywhere, and before
[T45](#t45) — a traced session that re-runs `--probe-streams` should not print a
column known to be wrong.

> **ANSWERED 2026-08-27
> ([`0124`](../tasks/0124-t47-t50-runtime-fixes/TASK.md) fix,
> [`0125`](../tasks/0125-t47-gbs-audit/TASK.md) audit).** `DesignInfo` now
> carries `tile_n`/`cols` read from design.json, the probe uses
> `a_elem_bytes` and refuses a design that predates the fields, and the
> banner states the tiling it used. The audit found **no current doc quotes
> a probe-streams `GB/s`** — the inflated figures live in task logs (kept,
> rule 3b) and this thread's own text — and produced the corrected
> nine-set reference table. The fixed binary reproduces `0097` t21's
> hand-corrected MiniLM int8 numbers to ~2% (35.5/22.1/36.3/31.6 vs
> 35.9/22.8/36.8/31.7), and the bfp16 large shapes cluster at
> **43.8–46.7 GB/s** — an independent per-dispatch corroboration of the
> ~44 GB/s roof, still on the host-observed y-axis
> [T45](#t45) owns. **Closure condition**: holds while
> `--probe-streams` is the only consumer of `DesignInfo::tile_n/cols`; a
> future consumer that defaults instead of refusing on 0 reintroduces the
> class.

<a id="t50"></a>
### T50 — `embed`'s CLI has a flag that is accepted and ignored, and a form that picks no design set · **ANSWERED 2026-08-27** · both halves refuse or resolve; silence is gone
Both found by [`0121`](../tasks/0121-semantic-gate/TASK.md) while building the
semantic gate, which drives the shipped `embed` subcommand rather than the
older flag form. Filed together because they are one surface and one sitting.

**The fail-open half: `--cpu` does nothing.** On *both* CLI forms, for the whole
BERT family:

```
> npuembeddings.exe embed all-MiniLM-L6-v2 in.txt out.f32 --root . --cpu
  designs    ONE xclbin, 16 streams (4 batch tiers), one hw_context
  datapath   bfp16-emulated MMAC, C as bf16
```

`force_cpu` is parsed (`runtime/src/main.cpp:4490`) and reaches only the arch-1
path. So a caller asking for the host encoder gets the NPU, silently, and the
vectors are correct — which is what makes it the fail-open shape rather than a
bug someone would notice. It is the exact fault
[`0118`](../tasks/0118-prompt-name-per-request/TASK.md) removed elsewhere in
this same file: **refusing beats ignoring**, which is why `serve` rejects
`--prefix` outright and `resolve_prefix()` refuses rather than defaulting.

Note the status line is *not* lying — it reports the datapath it really used, and
that is how 0121 caught this. The defect is that nothing refuses the flag.

**The loud half: the legacy form selects no design set.**
`npuembed.exe <root> --model X --embed in out` falls back to
`runtime/artifacts`, a per-op set predating the unified xclbin, and dies on every
model wider than 384:

| model | legacy form, no `--artifacts` |
|---|---|
| bge-base | `error: qkv: staged buffer for argument 1 is 3538944 bytes, design allows 884736` |
| bge-large | `error: layer.0.qkv: layout mismatch -- design wants 94266693..., file has f2ab7b0d...` |

**No wrong answer is ever returned** — the staged-buffer size check and the
`b_layout_hash` refusal are the guards working exactly as designed, and CLAUDE.md
already says a mismatched pair is refused rather than read as garbage. MiniLM and
bge-small pass only because that stale directory happens to be their width. The
shipped `embed <model> <in> <out> --root <dir>` form auto-picks correctly on all
six, in the dev tree and in a dist zip, so this is a usability wart with a
workaround (`--artifacts`) that every existing harness already uses — which is
why nobody had noticed.

**Priced together** because they are the same function and the same rebuild:
route `force_cpu` through the BERT path (or refuse the flag where it cannot be
honoured), and either give the legacy form the same `pick_artifacts()` call the
subcommand gets or make it refuse rather than reach for a stale directory. Both
sides of that choice are the 0118 rule; either is acceptable, silence is not.

**What is NOT open**: which form to prefer. The subcommand is the shipped one
(`embed.cmd` in a release zip runs it) and 0121's gate drives it.

**Trigger**: the next runtime rebuild for any reason. Neither half blocks
anything today, and 0121 removed `--cpu` from its own tool rather than shipping
a flag it had just measured as inert.

> **ANSWERED 2026-08-27
> ([`0124`](../tasks/0124-t47-t50-runtime-fixes/TASK.md)).** Both halves, one
> rebuild. `--cpu` on the BERT family (both CLI forms) now **refuses** with a
> message naming why — there is no host BERT encoder to route it to — and the
> subcommand rewrite forwards the flag it used to drop, so the refusal
> actually fires on `embed <model> ... --cpu`. The flag form's
> no-`--artifacts` default no longer means the stale per-op `artifacts`
> directory: resolution defers until the container is loaded and goes through
> the same `pick_artifacts()` call the subcommands make, catalogue datapath
> included, printing what it picked. Verified: bge-base's legacy form went
> from a staged-buffer error naming the wrong problem to running on
> `artifacts_base_bfp16` with the adopted datapath in the status line; the
> semantic gate passes 6/6 models on the rebuilt binary. **Closure
> condition**: the `--cpu` refusal is correct while no host BERT encoder
> exists in the C++ runtime — if one is built, the refusal should become
> routing. Found along the way: arch-1's own `--cpu` control needs a
> host-only container (`GemmaEncoder` reads raw `q_proj` F32 tensors); on
> the shipped pretiled container it fails loudly — pre-existing, left
> as-is, recorded in 0124.

<a id="t45"></a>
### T45 — The roofline's y-axis is wall clock, and its bandwidth roof is an inference · **ANSWERED 2026-08-27** · the roof is 45.5 GB/s on the hardware timebase, and the wall clock agreed to 0.4%
Filed by [`0120`](../tasks/0120-roofline-analytic/TASK.md), which built the
analytic roofline and could only build half of it. **x is safe**: `2·M·K·N /
bytes` over `M`, `K`, `N`, `tile_n`, `cols` and two dtypes, every one of them a
`design.json` field, so no other process on the machine can move it. **y is
work ÷ time**, and time is exactly what rule 1 forbids taking from the wall
clock.

So 0120 §3c's headline — *the four shipped bfp16 models imply 41.3–46.8 GB/s, a
13.3% spread against `GMAC/s`'s 28.2%, i.e. a bandwidth roof at ~44 GB/s* — is
an **inference from four host-observed aggregates**, not a measurement. The
aggregates are `--bench`'s `wait (hardware)` line, which
[`0109`](../tasks/0109-fused-ratio-energy/TASK.md) itself describes as
*"trace-adjacent … still reported as a host-observed duration, not a hardware
trace."*

**Two measurements, in value order.**

1. **One traced dispatch on the bfp16 datapath at production tile width.**
   Trap 7 caps tracing at 4 columns, which is fine — 0049 traced the plain-bf16
   and emulated microkernels at 4 columns already, and this is the same
   instrument pointed at a shipped geometry rather than at a probe. It converts
   the ~44 GB/s roof from an inference into a number.
2. **Separate the fixed cost from the slope.** A roofline has **no fixed-cost
   term**, and this machine has a large one: three independent fits put it at
   150 µs ([`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)), 573 µs
   ([`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)) and 627 µs
   ([`0080`](../tasks/0080-m13-int8-traffic-bound/TASK.md)). It is already
   visible as one consistent artifact, which is why this is not speculative:

   | datapath | `attn_out` % of ceiling | its large sibling | ratio |
   |---|---:|---:|---:|
   | plain bf16, fp32 C | 48.2% | 65.1% (`qkv`) | 0.74 |
   | plain bf16, bf16 C | 51.1% | 65.8% | 0.78 |
   | int8 | 6.6% | 10.3% | 0.64 |

   **`attn_out` is the smallest dispatch and reads low on every datapath we have
   measured.** The same deficit in three datapaths is a per-dispatch constant,
   not a property of arithmetic intensity — so the roofline mis-reads it by
   construction, and an M-sweep intercept on one shape is what makes it
   plottable. It also explains why MiniLM is the lowest of the four bfp16
   aggregates: at 1 485 µs/dispatch a ~150 µs fixed cost is 10% of the reading,
   against 1.5% at bge-large's 9 825.

**What this does NOT need**: a new design, a new kernel, or an 8-column trace.
Both items are the existing tracing path pointed at an existing artifact set.

**Trigger**: the next session with a verified-idle array. Nothing downstream of

> **Item 2 DONE, 2026-08-27
> ([`0128`](../tasks/0128-t45-m-sweep/TASK.md)).** `--probe-streams` now
> probes all four batch tiers, giving a 16-point sweep (~0.7 → 604 MB) per
> set; least-squares `t = t0 + MB/B` fits: base_bfp16 **t0 180 µs / 44.0
> GB/s** (R² 0.979), nomic_bfp16 **278 µs / 45.5 GB/s** (0.992),
> minilm_bfp16 74 µs / 36.8, int8c_mini 85 µs / 29.6. **The ~44 GB/s roof
> survives separating the intercept** on the h=768 bfp16 sets — and the
> fixed cost is *not one number* (nomic vs base share a geometry class and
> differ 278 vs 180 µs), so the two-parameter model is incomplete; do not
> quote a single fixed cost from it. Item 1 — the traced dispatch at
> production tile width, 4 columns — is what remains, and it now also owns
> explaining the intercept's shape-dependence. (0125's corrected
> per-dispatch table independently shows the same clustering, 43.8–46.7,
> and the attn_out fixed-cost fingerprint.)
0120 should be quoted as traced until item 1 lands — including [T48](OPEN-THREADS.md#t48)'s
pricing, which assumes the shipped designs really are on the slanted roof.

> **ANSWERED 2026-08-27
> ([`0128`](../tasks/0128-t45-m-sweep/TASK.md) item 2,
> [`0130`](../tasks/0130-t45-traced-roof/TASK.md) item 1).** The M-sweep
> (probe extended to all four batch tiers) fits `t = t0 + MB/B` at R²
> 0.98–0.99 with marginal rates **44.0–45.5 GB/s** on the h=768 bfp16 sets;
> the traced dispatch (production tile, 4 columns, bge-base `ffn_up` at
> M=8192) measures kernel-start pitch 2,932 cycles against a 1,658-cycle
> window — duty 56.6%, effective 67.0 MACs/cyc/core — implying **9.97 ms on
> the 1.808 GHz core clock against 10.006 ms measured by wall (0.4%)**, i.e.
> 453 MB → **45.5 GB/s on the hardware timebase**. Four instruments now
> agree on the roof (aggregate inference ~44; per-dispatch probes
> 43.8–46.7; M-sweep marginal 44.0–45.5; trace 45.5), and the y-axis doubt
> is resolved by cross-instrument agreement: on this dispatch the wall
> clock was honest. LOCK_STALL covers ~53% of the traced span — the bfp16
> datapath is traffic-bound *on the array*, measured. **Closure
> condition**: 45.5 is a 4-column, one-shape trace; its 8-column standing
> rests on 0128's marginal rate — if future probes stop clustering at
> ~44–46 GB/s, re-trace before quoting. **Residual, not closed here**: the
> fixed cost is shape-dependent (180 vs 278 µs on one geometry class,
> 0128) — that anatomy belongs to [T46](#t46)'s ledger.

<a id="t46"></a>
### T46 — Only the DRAM leg is on the figure; there are three · **ANSWERED 2026-08-27** · measured: the memtile↔L1 leg is two-thirds idle while LOCK_STALL binds — the DRAM leg is the one that matters
[`0120`](../tasks/0120-roofline-analytic/TASK.md) §6.3, stated as a limitation
rather than discovered. The roofline it built counts **DDR bytes**, so it prices
the shim leg and nothing else. The array has at least three levels worth a
separate roofline — DRAM↔shim, memtile↔L1, and L1↔register — and the same kernel
can be bound at one while looking healthy at another. A design that is
memtile-starved appears on 0120's figure as a point below the roof with no
explanation attached.

**Why this is a second figure, not a second instrument.**
`PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0,
master=True)` is already in the tracing surface
(`docs/05-measurement/README.md`), so an L2↔L1 roofline comes off the same
trace. `tools/roofline.py` takes its byte accounting from one function
(`traffic_bytes`), so a second accounting is an argument, not a rewrite.

**What would make it urgent.** A dispatch whose measured y sits well below
*both* the compute ceiling and the DRAM bandwidth roof. On the plain-bf16 rows
that gap is 32–35% and [`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)
already accounts for it in-core (77.8% INSTR_VECTOR, 21.1% loop bookkeeping,
2.5% MEMORY_STALL, **0% LOCK and STREAM_STALL**) — **so today there is no
unexplained gap, and that is the honest reason this thread is not first.**

**Trigger**: after [T45](#t45), or the first dispatch whose gap 0049's anatomy
does not cover.

> **ANSWERED 2026-08-27
> ([`0139`](../tasks/0139-t46-memtile-leg/TASK.md)).** Closed from data
> already on disk: exact interval-union coverage over 0130's stored trace
> shows the traced core's input ports covered **30.9% + 3.7%** of the
> steady-state span while **LOCK_STALL covers 52.8%** — the L2→L1 link
> sits idle while the core waits for operands, the signature of an
> upstream (DRAM→L2) bind. The second accounting is one argument, as this
> thread predicted: `traffic_bytes(..., leg="memtile")` — the legs differ
> only in B's ×4 row broadcast (aggregate 2.0–2.5× DRAM; 14–23 GB/s per
> mem tile at measured dispatch times, against links two-thirds idle). The
> L1↔register leg is 0049's in-core anatomy plus 0130's 118.5-of-256
> window rate — a kernel roofline, already owned. **Closure condition**:
> holds at the shipped operating point; **B-reuse is exactly the change
> that re-binds this leg** (it removes B's DRAM re-streams but not its
> L2→L1 broadcasts, and speeds the dispatch 1.72×), so a T48 build must
> re-run 0139's port-coverage analysis as part of its verification. The
> T45 residual (shape-dependent fixed cost) narrows to schedule effects —
> input links idle, locks binding — but stays a recorded residual, not a
> thread.

<a id="t52"></a>
### T52 — A SentencePiece Unigram tokenizer gates the multilingual encoder family · **ANSWERED 2026-08-27, same day it was filed** · built twice, byte-exact everywhere, and shipping inside the 7th catalogue model
Filed by [`0123`](../tasks/0123-register-hygiene/TASK.md) while planning 0.5.0,
as the sibling of [T43](OPEN-THREADS.md#t43) — a *third* tokenizer family, not a discharge of
the second. The repo ships WordPiece (`runtime/src/tokenizer.cpp`, arch 0 and
2) and SentencePiece **BPE** (`runtime/src/tokenizer_gemma.cpp`, arch 1). What
`gte-multilingual-base` — and with it the whole XLM-R / multilingual-E5 / mGTE
family — needs is SentencePiece **Unigram**: a Viterbi search over per-piece
log-probs, plus XLM-R's `precompiled_charsmap` normalizer (a trie), neither of
which shares an algorithm with the merges machinery BPE runs on.

**The estimate already exists and was never spent.** The Gemma plan
([`0055`](../tasks/0055-m10-embeddinggemma-spike/TASK.md)) budgeted **~600–900
LOC** for SentencePiece Unigram before
[`0061`](../tasks/0061-m12-embeddinggemma-tokenizer/TASK.md) read
`tokenizer.json` and found Gemma3 is actually **BPE**, and built that instead
([T29](#t29) records the reversal — "caught only by reading
`tokenizer.json` before writing code", a lesson that applies verbatim here). The
generator must read the HF *fast* `tokenizer.json` (17 MB — the repo pattern
has no `sentencepiece.bpe.model` to lean on), emit a `GEMATOK1`-style blob
(call it `XLMRTOK1`: charsmap trie + vocab with fp32 log-probs), and exist
twice per the arch-1 precedent: a Python generator and a C++ port
(`gemma_tokenizer_gen.hpp` records why the port is non-optional — a fresh
clone packs without Python). `gemma_tokenizer_gen.cpp:90-95` already refuses
Unigram by name; that refusal is this thread's TODO marker.

**Reusable from the tree**: `json_min.cpp`, the table-blob container
convention, `tools/verify_tokenizer_gemma.py`'s byte-exact-vs-HF harness (0061: 1,925/1,925 byte-identical). The
part with no in-repo precedent is the charsmap trie and the Viterbi loop, and
the verification corpus must be genuinely multilingual — CJK no-space text is
where Viterbi bugs hide, and NFKC-sensitive input is where the charsmap does.

**Trigger**: 0.5.0's adoption of `gte-multilingual-base`
([T44](OPEN-THREADS.md#t44)) — the cost is paid once and unlocks the multilingual family the
same way [T43](OPEN-THREADS.md#t43) unlocks the byte-BPE family.

> **Step 1 DONE — the Python side is byte-exact, 2026-08-27
> ([`0127`](../tasks/0127-t52-unigram-generator/TASK.md)).**
> `tools/gen_xlmr_tokenizer_table.py` (stdlib, fail-closed on every
> tokenizer.json assumption) emits an `XLMRTOK1` blob (5.3 MB: Darts trie +
> normalized-strings blob verbatim, vocab scores, Metaspace/template
> metadata); `tools/xlmr_tokenizer_ref.py` consumes the **blob** and matches
> HuggingFace **343/343 byte-exact** across 19 categories (CJK no-space,
> NFD-decomposed jamo, NFKC, emoji/ZWJ, unk-fusion, whitespace edge cases).
> Three findings the C++ port must honour: **scores are f64** (65,856 of
> 250,002 log-probs do not survive f32, and HF's Viterbi sums f64 — f32
> storage would break near-tie exactness); **HF's Precompiled normalizer
> deviates from sentencepiece** (grapheme clusters <6 UTF-8 bytes replaced
> whole by the first prefix trie match — mirrored, confirmed live);
> **fuse_unk is on** and Python `str.isspace()` ≠ Rust White_Space.
> Deliberate gap, recorded: no added-tokens splitter (literal special-token
> strings in input go through the model) — decision deferred to the C++
> task. What remains: the C++ port (runtime tokenizer + generator port, the
> gemma two-implementation pattern) and the container wiring.

> **Step 2 DONE — the C++ side matches everything, first run, 2026-08-27
> ([`0133`](../tasks/0133-t52-cpp-port/TASK.md)).** Three verifications at
> the full bar: the C++ generator's blob is **byte-identical** to the
> Python generator's (sha256 equal over 5,318,988 B — which also settles
> the f64 worry: json_min's `strtod` is correctly-rounded, proven by
> identity over all 250,002 raw-f64 scores); the C++ CLI matches
> HuggingFace **343/343** and the Python reference **343/343**. New:
> `tokenizer_xlmr.{hpp,cpp}` (f64 Viterbi, grapheme quirk, Rust
> White_Space, in-place Darts walk, `from_table_bytes` ready for .npue),
> `xlmr_tokenizer_gen.{hpp,cpp}` (every fail-closed guard in the Python's
> order), `tokenizer_xlmr_cli` (a real CMake target — the gemma CLI never
> had one), and generated `xlmr_unicode_tables.hpp` (UCD 15.1.0, the
> `bert_unicode_tables` precedent). Deferred, recorded: the added-tokens
> splitter decision, and the arch-3 wiring — which is the only thing left
> before this thread closes.

> **The wiring is DONE and the tokenizer runs in production code, 2026-08-27
> ([`0136`](../tasks/0136-gte-runtime/TASK.md)).** arch 3 encodes end to end
> on the NPU through `tokenizer.xlmr_table` (8/8 byte-exact vs AutoTokenizer
> in-process, truncation refusal per 0110 included). What remains for T52's
> closure is only the C++ packer mirror (`prepare_model_gte`) so a
> no-Python clone can pack — the gemma precedent's last leg.

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

> **ANSWERED 2026-08-27
> ([`0127`](../tasks/0127-t52-unigram-generator/TASK.md) generator,
> [`0133`](../tasks/0133-t52-cpp-port/TASK.md) C++ port,
> [`0136`](../tasks/0136-gte-runtime/TASK.md) runtime wiring,
> [`0138`](../tasks/0138-gte-hub-adoption/TASK.md) packer mirror).** The
> family is built at the gemma discipline's full bar and beyond it:
> Python generator + reference **343/343** byte-exact vs HuggingFace over
> 19 adversarial categories; C++ port byte-identical blob and 343/343 on
> all three comparison axes; the runtime encodes through
> `tokenizer.xlmr_table` in production; and `prepare_model_gte`'s C++
> container is **whole-file byte-identical** to the Python packer's —
> including the from-scratch path that regenerates the blob in C++, so a
> no-Python clone self-produces a working container (the exact gap
> tasks/0067 closed for gemma). The estimate history closed the loop: the
> ~600–900 LOC budgeted in 0055 for an algorithm gemma turned out not to
> need was finally spent on the model that does. **Closure condition**:
> the added-tokens splitter remains deliberately unimplemented (literal
> special-token strings in input text go through the Unigram model —
> recorded in 0127/0133); revisit if a use case feeds such strings.
