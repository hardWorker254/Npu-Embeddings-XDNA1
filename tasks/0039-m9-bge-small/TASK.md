# 0039 — M9 Phase B: a second model

**Goal.** Run `BAAI/bge-small-en-v1.5` end to end on the NPU, and find out what
"drop-in weight swap" actually costs. No design changes: the GEMM shapes are
identical, so every compiled xclbin and the `.npue` tiling carry over unchanged.

Plan: `C:\Users\vegar\.claude\plans\ja-la-oss-kj-re-shimmering-dewdrop.md`.
Depends on [`0038`](../0038-m9-model-driven-runtime/TASK.md), which made the
runtime read its geometry from the container.

---

## Headline

**bge-small runs on the NPU on the first attempt, at `1-cos` 8.348e-06 against
HuggingFace** — better than MiniLM's 1.086e-05 — and passes the MTEB gate at
**mean delta −0.03 points**. It costs exactly what twice the layers costs:
**450.0 seq/s against MiniLM's 888.7**, 0.506×, at 96 dispatches per group
instead of 48.

Against MiniLM, on the same five tasks, same NPU, same seq 64:

| task | MiniLM | bge-small | delta |
|---|---:|---:|---:|
| STSBenchmark | 82.04 | 85.87 | **+3.83** |
| SICK-R | 77.59 | 79.41 | +1.82 |
| STS12 | 72.36 | 77.45 | **+5.09** |
| Banking77Classification | 80.05 | 81.75 | +1.69 |
| TwentyNewsgroupsClustering | 45.98 | 48.48 | +2.50 |
| **MEAN** | **71.60** | **74.59** | **+2.99** |

Not the +5.9 the docs promised — that figure is the full 56-task MTEB at the
model's own sequence length. This is a five-task subset truncated to seq 64 on
both sides. **Half the throughput for three points**, measured rather than
quoted.

---

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

.venv-ref\Scripts\python.exe reference\fetch_model.py `
    --model BAAI/bge-small-en-v1.5 --layers 12
.venv-ref\Scripts\python.exe reference\make_goldens.py `
    --model-dir models\bge-small-en-v1.5 --taps
.venv-ref\Scripts\python.exe reference\check_reference.py `
    --model-dir models\bge-small-en-v1.5 `
    --goldens reference\goldens\bge-small-en-v1.5_l12_s64_boundary.safetensors

runtime\build\npuembed.exe .. --prepare-model ..\models\bge-small-en-v1.5 `
    ..\models\bge-small-en-v1.5.npue          # run from runtime\
python tools\export_validation.py --model bge-small-en-v1.5
.venv-ref\Scripts\python.exe tools\verify_npue.py --model bge-small-en-v1.5

runtime\build\npuembed.exe .. --model bge-small-en-v1.5 `
    --artifacts artifacts_b128il --threads 24 --pipeline 2
runtime\build\npuembed.exe .. --model bge-small-en-v1.5 `
    --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 5

.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py --model bge-small-en-v1.5
.venv-ref\Scripts\python.exe experiments\m8-npu-vs-cpu\run_mteb.py `
    --model bge-small-en-v1.5 `
    --out experiments\m8-npu-vs-cpu\artifacts\mteb_bge_small.json
```

---

## Results

| check | MiniLM | bge-small |
|---|---|---|
| oracle vs HF **and** sentence-transformers | 1-cos 2.2e-08 | **1-cos 3.925e-08** |
| `.npue` verify (bf16 activations) | 1-cos 1.17e-05 | **1-cos 9.631e-06** |
| NPU end to end vs HF golden | 1-cos 1.086e-05 | **1-cos 8.348e-06** |
| end to end vs sentence-transformers | PASS, top-10 overlap 1.0000 | **PASS, top-10 overlap 1.0000** |
| MTEB gate (CPU vs NPU) | mean +0.04 | **mean −0.03**, worst −0.14 |
| throughput, batch 128, 2 lanes | 888.7 seq/s | **450.0 seq/s** (0.506×) |
| dispatches per group | 48 | 96 |
| weights staged | 21.23 MB | 42.47 MB |

`1-cos` being *better* for a deeper model is not a mystery: it is measured
against that model's own HuggingFace output, and bge's larger activations
round proportionally less in bf16.

---

## What the published docs got wrong

`docs/00-overview.md` called bge-small a **"byte-identical drop-in swap"**,
`README.md` called it *"a drop-in weight swap by design, but untested"*, and
`docs/04-model/README.md` made it a **design constraint**. All three are now
corrected, along with `CLAUDE.md`.

It is not byte-identical. **12 layers, not 6. CLS pooling, not mean.** What is
true is the useful half: identical tensor names and identical GEMM shapes, so
the xclbins, the tiling and the whole design set carry over untouched. The two
differences are data the runtime reads, not constants it was compiled with —
which is exactly what [`0038`](../0038-m9-model-driven-runtime/TASK.md) was for.

---

## Four more places that assumed the pooling mode

The previous task made the *runtime* read pooling from
`1_Pooling/config.json`. Running a real CLS model found four more:

1. `reference/encoder.py` — `encode()` mean-pooled unconditionally.
2. `reference/make_goldens.py` — mean-pooled in torch, unconditionally.
3. `reference/check_reference.py` — built the oracle with no mode.
4. `tools/verify_npue.py` — `build_from_npue()` likewise.

All four now read the checkpoint's own metadata. **Number 4 is the instructive
one**: it reported `1-cos` **9.294e-02** — four orders of magnitude worse than
the same weights measure on the actual NPU — and still printed **PASS**.

### Why it passed: the gate only existed relative to MiniLM

`check_goldens` gated on `ratio = (1-cos) / M3_BF16_1MCOS`, a MiniLM-L6
constant. Making the baseline per-model was correct, but my first version
*skipped the check entirely* when a model had no baseline — so a new model got
no gate on `1-cos` at all. Replaced with an absolute limit of **1e-03** for
models without a measured baseline: well under the runtime's own 2e-03 gate,
far above any legitimate bf16 result, so it catches a wrong pooling or a broken
fusion without pretending to be a precision measurement. bge-small's measured
9.631e-06 is now recorded in `BF16_BASELINE`.

The fold-cost check also read **1.000× ("free")** while both sides were wrong
in the same way. It now reads 1.036×.

---

## The fixture guard fired in production, on its first real encounter

Added in [`0038`](../0038-m9-model-driven-runtime/TASK.md) a commit before it
could be needed. Pointed at bge-small with MiniLM's fixtures still in place,
the runtime refused and named both checkpoints. MiniLM-L6 and bge-small have
identical hidden, heads, head_dim, ffn, vocab and golden batch, so **every
fixture file is the same size**: without the guard this would have compared
correctly-shaped floats against the wrong answers and reported a pass.

Fixtures now live per model under `runtime/artifacts/validation/<model>/`, so
the collision cannot occur at all. A collision that cannot happen beats one
that is merely detected; the guard stays as the backstop.

---

## Three more literals that should have been data

**`source_repo`.** `npue_pack.cpp` wrote
`sentence-transformers/all-MiniLM-L6-v2` as a literal, so bge-small's container
claimed MiniLM's repository — visible immediately in the `--list-models` table.
Now read from `CHECKPOINT.json`, and it **refuses** when it cannot establish
the repo: a container that misattributes its own weights is a licensing
statement, not a cosmetic error. The `source_sha256` was never affected, since
the C++ packer computes it from the weights.

**The dispatch-saving banner** said "6 fewer" and "13 fewer" regardless of
depth. Now `g_layers` and `1 + 2*g_layers`; bge-small reports 12 and 25.

**`fetch_model.py`'s shape table.** 384 / 1536 / 30522 / 512 as literals. They
happen to be right for bge-small, so Phase B would have passed on them **by
luck**, and a check that passes by luck is not a check. Split into structural
expectations that stay literal (bert, absolute positions, gelu, eps 1e-12) and
dimensional ones read from the checkpoint's own config and then demanded of
every tensor — which still catches a config that disagrees with the weights
beside it.

---

## The endpoint gate was measuring the wrong property

`verify_endpoint.py` required `unrelated_cos < 0.4`. bge-small produces
**0.4197** and failed — a model that passes MTEB, measures 1-cos 8.348e-06
against HuggingFace, and reproduces sentence-transformers exactly.

The absolute cosine **scale** is a property of the training objective, not of
quality: bge's contrastive training leaves unrelated text near 0.42 where
MiniLM leaves it near 0.09. **Separation** is the property worth gating. The
check is now `min(paraphrase cos) − unrelated cos > 0.25`:

| | worst paraphrase | unrelated | margin |
|---|---:|---:|---:|
| MiniLM | 0.5564 | 0.0881 | **+0.4683** |
| bge-small | 0.7513 | 0.4197 | **+0.3316** |

Both pass. Loosening the threshold globally would have hidden a real
regression in MiniLM; gating on separation tests what the corpus was written
to test.

---

## The destructive bug, disarmed twice

`reference/make_goldens.py` wrote `minilm_l6_s64_boundary.safetensors` as a
**literal**, regardless of `--model-dir`. Generating bge-small goldens would
have silently overwritten the committed MiniLM contract with 12-layer
CLS-pooled data of identical shape.

Fixed twice, because naming is a convention and not a guard:

1. The slug is derived from the checkpoint — `bge-small-en-v1.5_l12_s64_*`.
   MiniLM keeps its historical slug, because `tasks/0005` and six scripts under
   `experiments/` cite the filename and renaming it would falsify a task log.
   One definition, in `tools/npue.py`, for all three callers — three copies of
   a naming rule is how the three pooling implementations started.
2. `refuse_to_clobber()` reads the existing golden's metadata and stops if it
   belongs to a different checkpoint. **This is the check that would have
   caught the bug even under the old name.**

Verified against a known-bad pair: same sha regenerates, different sha refuses,
`--force` overrides.

**And the second oracle is now asserted rather than printed.**
`make_goldens.py` computed the max abs difference between sentence-transformers
and its own manual pooling and only *printed* it — yet that number is the proof
the pooling mode is right, because sentence-transformers reads
`1_Pooling/config.json` itself. Now a hard limit of 2e-6; bge-small measures
**8.196e-08**.

---

## And the release was broken by the commit before this one

`get-model.cmd` downloads `model.safetensors`, `vocab.txt` and `config.json` —
but **not** `1_Pooling/config.json`, which packing now requires, and it never
wrote `CHECKPOINT.json`, which `source_repo` now requires. A release built
today would have fetched the weights and then failed to pack. Both added; the
script already knew the repo id and the pinned sha because it embeds them in
its own checksum check.

**Rule this adds:** making the runtime read a new file means the *release path*
must supply that file. The verification gates all run from a full checkout and
cannot see this.

---

## Carry forward to Phase C and D

- MiniLM is unchanged throughout: `1-cos` 1.086e-05, packers byte-identical,
  `verify_npue` PASS, endpoint PASS.
- The CPU baseline still uses **best-of-5** while the NPU is reported as a
  **mean**. Phase C fixes that, and it will move published numbers.
- `bge-large` needs `tile_n` 48 → 32 in three places at once, and the ~800 MB
  XRT buffer footprint should be probed before anything else.
