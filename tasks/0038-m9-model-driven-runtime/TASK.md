# 0038 — M9 Phase A: the runtime reads the model instead of assuming it

**Goal.** Stop being a one-model binary. Every shape the encoder needs comes
from the `.npue` container, model selection becomes explicit, and the guards
that would catch a wrong model get added *before* a second model exists to
trip them. Prerequisite for bge-small (Phase B) and bge-large (Phase D).

**Gate.** MiniLM must be **bit-identical** before and after. A refactor of this
size that changes one float has done something nobody asked for.

Plan: `C:\Users\vegar\.claude\plans\ja-la-oss-kj-re-shimmering-dewdrop.md`.

---

## What was changed

| | |
|---|---|
| A1 | `constexpr` geometry → `set_model_shape()` reading the container |
| A2 | three near-identical pooling blocks → one `pool_rows()` with a mode |
| A3 | `qk()`/`av()` generic in `head_dim` instead of a hardcoded 4-vector unroll |
| A4 | load-time guards: `head_dim*heads == hidden`, `head_dim % 8`, `hidden % 8`, design seq ≤ packed positions |
| A5 | `--model` / `--list-models`, the table built from the containers |
| A6 | pooling read from `1_Pooling/config.json` in **both** packers |
| — | the fixture-identity guard (`source_sha256`) |
| — | scratch buffers hoisted out of `Encoder::run()` |
| — | the additive mask moved out of `qk()` |
| A9 | **the compile-time vector count A3 lost, restored as a template** |

Commits: `76dcebf` (A1–A3), `51fdd11` (A6 + fixture guard), this one (A5, A9,
the two review items, `tools/backfill_design_seq.py`).

---

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
cmake --build build --config Release

# bit-identity, the gate for the whole phase
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2 `
    --embed <scratch>\base.txt <scratch>\after.f32
Get-FileHash <scratch>\before.f32, <scratch>\after.f32 -Algorithm SHA256

# the two eltwise branches
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2
.\build\npuembed.exe .. --artifacts artifacts_b128  --threads 24

# throughput, interleaved with the pre-phase binary, 4 repeats each
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 5

cd ..
python tools\verify_pack_parity.py
python tools\backfill_design_seq.py runtime          # report
python tools\backfill_design_seq.py runtime --write
.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py
.venv-ref\Scripts\python.exe tools\verify_tokenizer.py
```

---

## Results

| check | result |
|---|---|
| `--embed` on 200 texts vs the pre-phase binary | **bit-identical** (sha256 `89F46358…`) |
| golden, host eltwise | `1-cos` **1.086e-05** |
| golden, NPU eltwise (softmax on the array) | `1-cos` **2.469e-04**, unchanged to 4 digits |
| packers still agree | **byte-identical**, same sha256 as before A6 |
| end-to-end vs sentence-transformers | mean `1-cos` 3.493e-04, top-10 overlap **1.0000** |
| tokenizer | byte-for-byte identical to HuggingFace |
| throughput, batch 128, 2 lanes, 24 threads, **balanced** | **892.2 seq/s** mean (868.5–909.8) vs **844.8** (818.8–860.0) — **1.056×** |

Throughput is wall clock and is labelled as such: it measures the host path
and dispatch, not kernel quality.

---

## The failure worth keeping: a 2× regression that every correctness gate passed

A3 was committed in `76dcebf` with the message *"the production `--embed` path
is BIT-IDENTICAL"*. It was. It was also **twice as slow in attention**, and
nothing noticed for two commits.

The old `qk()`/`av()` unrolled exactly four AVX2 vectors — 32 floats — and
never mentioned `head_dim`. A3 replaced that with a loop over `g_head_dim / 8`.
`g_head_dim` is a runtime global, so `nv` is a runtime bound: the inner loop
stopped unrolling and `qv[]`/`acc[]` stopped living in registers. Host
attention went **29 ms → 58 ms per encode** at batch 128, about 10% of host
work and 1.5 extra CPU cores.

The plan had said *"keeping a compile-time fast path for the common 32 and
64"*. That clause was skipped, and no gate we own can see the difference,
because the arithmetic is identical — which is exactly why it was verified
bit-identical and shipped.

### How it was found, and three wrong answers on the way

It surfaced only because the `--bench` breakdown was read line by line rather
than looking at the total: **wall clock barely moved** (829 → 828 seq/s) while
`attn` doubled and `bias` fell. Host work is overlapped with the NPU, so a
regression of this size hides completely in the headline number. `cores busy`
5.46 → 8.06 was the other visible symptom.

Three hypotheses were tested and killed before the real one:

1. **Aliasing.** The scratch buffers had just been hoisted to members, so the
   compiler could no longer prove `qkvbuf` and `scores` were distinct
   allocations. Plausible, and a known 2× effect. Added `__restrict` to the
   base pointers in both kernels: **no change** (attn 55.2 → 55.2). Refuted.
2. **The allocation hoist itself.** Reverted it, keeping everything else:
   attn stayed at **59.4**. Refuted — and the revert cost throughput
   (772 vs 834 seq/s), so the hoist was helping.
3. **The mask move.** Put the additive mask back inside `qk()`: attn stayed
   at **58.7**. Refuted.
4. **The machine.** Re-ran the *unchanged* pre-phase binary at the end of the
   session: attn **29.1**, i.e. it reproduced its old number. Not thermal
   drift, not contention — a real code difference, and older than the changes
   being bisected.

That last control is what pointed at A3: the saved binary predates A1–A3, so
"before" and "after" differed by more than the edits under test. **A bisection
is only as good as the assumption that the reference is one step away.**

### The fix

`qk_impl<NV>` / `av_impl<NV>` with `NV` a template parameter, dispatched on
`g_head_dim`: 4 for head_dim 32 (MiniLM, bge-small), 8 for 64 (bge-large),
and `NV == 0` keeping the fully generic loop for a width we have not met.
Genericity now costs nothing on the widths that exist.

attn **30 ms**, cores busy **5.6–6.1**, and throughput **above** the pre-phase
binary because the allocation hoist is no longer masked by the regression.

**Rule this adds:** a refactor verified bit-identical is verified for
*correctness only*. Genericity that replaces a compile-time constant with a
runtime one is a performance change by construction, and needs `--bench` run
against the previous binary in the same session, interleaved.

---

## Other things found

**The mask could not go where the plan said.** The plan said "move the
additive mask out of `qk()` into `softmax_cpu()`". Done literally that is a
**new fail-open**: the NPU-softmax branch (`host_sm == false`, still reachable
and still tested at `artifacts_b128`) would then softmax unmasked scores and
attend to padding, silently. The mask instead goes to whatever consumes the
scores — folded into `softmax_cpu`'s per-row prologue, where the row is
already in L1 and it costs nothing, and as an explicit pass on the NPU branch,
which has to pay for it because an aie kernel has no second operand here.
Verified by running the NPU branch: `2.469e-04`, identical to the pre-phase
binary on the same design set.

**Sequence length had no owner.** First read from the container's
`max_seq_len`; the banner immediately printed `batch 32 x seq 256` instead of
`batch 128 x seq 64`. `max_seq_len` is how many position embeddings were
*packed* (256), not what the designs were *compiled for* (64). Sequence length
belongs to the design. 203 design sets predate the field, so
`tools/backfill_design_seq.py` recovers it from two artifacts inside each set
that must agree — `softmax/design.json` `cols` is the row length (= seq), and
`softmax.rows / qkv.M` must come out a positive integer head count. 24 sets
stamped, **13 refused** (probe and passthrough sets with no softmax design);
refusing beats stamping a plausible 64.

**`--model` is required exactly when there is a choice.** With one container
installed there is no ambiguity and every documented command line still works;
with two, the runtime prints the table and refuses. A script written today
therefore breaks **loudly** the day bge-small is installed, instead of quietly
changing which model it measured.

**The fixture guard was pre-loaded for Phase B.** MiniLM-L6 and bge-small have
identical hidden, heads, head_dim, ffn, vocab and golden batch, so every
validation fixture file is the same *size*. `validation.json` has carried the
checkpoint's sha256 since it was written and nothing read it. Verified against
a **known-bad** artifact, not only a good one: with a corrupted sha the runtime
exits 2 and names both checkpoints.

**The three pooling copies were not the same code.** The golden path
accumulated in `float` while the other two used `double`, so the comment
claiming they matched was already wrong by one rounding. Unified on the
`double` accumulator — the one the MTEB result was measured with.

---

## Carry forward

- `tools/verify_pack_parity.py` must stay a gate: two packers, one binary
  layout, and A6 changed both.
- Phase B's landmine is unchanged and still live:
  `reference/make_goldens.py` writes a hardcoded `minilm_l6_s64_*.safetensors`
  and would **overwrite the MiniLM goldens**.
- The `--bench` breakdown, not the seq/s line, is where host regressions are
  visible. Read it.
