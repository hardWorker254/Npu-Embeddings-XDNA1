# 0121 — A semantic gate: do the vectors mean anything? All six models, ordering only

**Goal** (the user's words): an internal test routine over all the models, with
English sentences of about 100 characters that are either *very* close or *very*
far apart, validating that this is what comes out. Explicitly **not** measured to
the decimal — *"'I like fishing in the river' and 'I like fishing in the lake'
are near each other, and 'I like fishing in the river' and 'the office chair was
bought at IKEA' are some distance apart."*

**Result: built, and all six models pass.** `tools/verify_semantics.py` +
`tools/semantic_corpus.json`. No tolerance anywhere; the gate is a ranking. The
harder half of the work was not writing it — it was **discovering that four of
my own corpus revisions were measuring the corpus rather than the runtime**, and
finding where to draw the line between the two. Two runtime CLI defects fell out
on the way, one of them a fail-open flag.

```
  model                     gate    near     far    margin  probe  datapath the runtime reported
  all-MiniLM-L6-v2         24/24  +0.790  +0.057  +0.387   12/12  bfp16-emulated MMAC, C as bf16
  bge-small-en-v1.5        24/24  +0.856  +0.400  +0.224   10/12  bf16 MMAC, C as fp32
  bge-base-en-v1.5         24/24  +0.846  +0.353  +0.217   12/12  bfp16-emulated MMAC, C as bf16
  bge-large-en-v1.5        24/24  +0.830  +0.328  +0.237   12/12  bfp16-emulated MMAC, C as bf16
  nomic-embed-text-v1.5    24/24  +0.925  +0.599  +0.189   12/12  bfp16-emulated MMAC, C as bf16
  embeddinggemma-300m      24/24  +0.826  +0.357  +0.256   12/12  bfp16-emulated MMAC, C as bf16
```

Negative control: **24/24 gate failures on every model**, as required (§5).

---

## 1. Why this is not covered by what already exists

Three gates already run: `tools/verify_embed_e2e.py` (NPU vs
sentence-transformers, text in and vector out), `--bench`'s golden gate, and
MTEB. They are the right instruments for *"is the datapath correct"*, and they
share one blind spot — **they pass when both sides are wrong the same way.**

* `verify_embed_e2e` prepends the **same** task prefix to the reference that the
  runtime applies, so a wrongly-chosen prompt agrees with itself.
* A container whose `arch` was read wrong returns vectors that are correctly
  shaped, correctly normed and bit-for-bit deterministic — which is `CLAUDE.md`'s
  own stated reason for making `arch` a whitelist rather than a description.
* [`0110`](../0110-refuse-silent-truncation/TASK.md): a truncated text returned a
  perfectly ordinary vector, and two documents sharing a preamble returned
  **byte-identical** ones.

This gate needs no reference at all. That is also why it is **stdlib only** — no
numpy, no torch, no `.venv-ref`. 36 vectors of at most 1024 floats is 630 dot
products, milliseconds in pure Python, and paying that buys the ability to run it
against a **cold release zip** on a machine with nothing installed but CPython
and the exe:

```powershell
python tools\verify_semantics.py --exe dist\npuembeddings-0.4.0\npuembeddings.exe `
                                 --root dist\npuembeddings-0.4.0
```

That is exactly the situation in which "the vectors are meaningless" is most
likely and least likely to be noticed.

## 2. The gate

Twelve clusters of three sentences, 80–95 characters each, twelve mutually
unrelated topics (fishing, IKEA furniture, weather, cooking, cars, programming,
medicine, astronomy, football, banking, gardening, air travel).

| | | |
|---|---|---|
| `.a` | anchor | |
| `.b` | **lexical** paraphrase — most words shared, one or two swapped | **GATED** |
| `.c` | **semantic** paraphrase — same topic, deliberately almost no shared words | **reported only** |

**The gate**: for each of the 24 `.a`/`.b` sentences, its own cluster partner
must be the **nearest of the other 23**. That is the user's requirement stated
per anchor, and it is scale-free, corpus-size-free and threshold-free — add a
thirteenth cluster and every existing sentence's test is unchanged in kind.

**There is exactly one number in the whole file, and it is not a semantic one**:
two *different-topic* sentences coming back at cosine > 0.9999 are flagged as a
**collapsed encoder**. 0110 is why that is explicit rather than assumed — under a
pure ranking gate, byte-identical vectors surface as arbitrary tie-breaking,
which is a confusing way to be told the encoder is broken.

## 3. Four corpus revisions, and where the line is

**This is the part worth keeping.** The first corpus failed, and so did the next
three. Every failure was a defect in *my corpus*, not in any model — and the
fourth one was a defect in my *test design*, which is a different thing and took
longer to see.

| ver | what failed | the actual cause |
|---|---|---|
| v1 | MiniLM p@2 0.833; `furn.c`/`med.c` at **+0.456**, above every near pair | `furn.c` said "boxes" and `med.c` said "box". Also `fish.c` ("casting a line at dawn") named no fish and collided with `wthr.b`'s "dawn"; `prog.c`'s "Three hours" collided with the flight cluster's "three hours" |
| v2 | `cook.c`/`gard.a` at **+0.349**, above every near pair | **cooking and gardening were not unrelated topics.** Gardening grew vegetables and cooking cooked them. Gardening moved to ornamental |
| v3 | `wthr.c`/`astr.c` at **+0.344** | **weather and astronomy both talked about the night sky.** Clear-vs-stormy night *is* a shared topic; the sentences were fine, the cluster boundary was not |
| v4 | four of six models made `car.c` and `med.c` each other's nearest neighbours; gemma paired `med.c` with `ball.c` | **the twelve semantic paraphrases had become an accidental thirteenth cluster.** They were all first-person errand stories — *I went somewhere, I came back with something, on the way home*. The models were right; the corpus was measuring narrative voice |

**v5 fixed the sentences and it was still not enough**, and that is the finding.
`gard.c` ("the hedge wants cutting back, the bulbs must go into the border")
matched `med.b`; `car.c` ("a worn timing belt meant the garage kept the vehicle")
matched the flight cluster, because *mechanical failure causing a delay* is a
real shared meaning. I had spent four revisions pushing the `.c` sentences toward
"few shared words" and had pushed them into ambiguity instead.

> **The `.c` tier was mine, not the user's.** The requirement was the lexical
> pair — *river* against *lake* against *the IKEA chair* — and all six models
> hold that comfortably and always did. The semantic tier is a strictly harder
> test, and what it tests is a **checkpoint's world knowledge**, which is what
> MTEB is for. A gate that fires on *"is a hedge related to a rose"* would cry
> wolf until somebody switched it off, and a gate nobody trusts is worse than no
> gate.

So `.c` was **demoted to a probe**: reported every run, never a veto. And it is
scored **against the 24 gate sentences only, never against other probes**, which
is what dissolves the accidental thirteenth cluster. The signal is kept, the
false alarms are gone. Result: probe 12/12 on five models, **10/12 on bge-small**
(`car.c`→`fly.b`, own cluster rank 3; `gard.c`→`med.b`, own rank 2) — visible,
named, and not a failure.

**The rule I held myself to** after v2: *change a sentence only when the failure
is attributable to a corpus defect I can name* — a shared incidental token, a
shared domain, a shared narrative frame. Anything I could not attribute would be
a finding about the model and would get reported, not tuned away. v5 is where
that rule stopped permitting edits and forced the design change instead.

## 4. Two runtime CLI defects, found by driving the shipped path

**4a. `--cpu` is accepted and silently ignored by `embed`.** On both CLI forms.
The status line still reads `designs ONE xclbin ... datapath bfp16-emulated
MMAC`, i.e. it ran on the NPU:

```
> npuembeddings.exe embed all-MiniLM-L6-v2 in.txt out.f32 --root . --cpu
  designs    ONE xclbin, 16 streams (4 batch tiers), one hw_context
  datapath   bfp16-emulated MMAC, C as bf16
```

`force_cpu` is parsed (`main.cpp:4490`) but reaches only the arch-1 path. This is
the fail-open class this project catalogues, and it is the exact fault
[`0118`](../0118-prompt-name-per-request/TASK.md) removed elsewhere — *"refusing
beats ignoring"* — with `serve` rejecting `--prefix` and `resolve_prefix()`
refusing rather than defaulting. Filed as [T50](../../research/OPEN-THREADS.md#t50).

**I removed `--cpu` from this tool rather than shipping it.** Offering a flag
that silently does nothing would be this task committing the fault it is
reporting.

**4b. The legacy `--model X --embed` form does not select a design set**, and
dies on every model wider than 384:

| model | legacy form, no `--artifacts` | shipped `embed` subcommand |
|---|---|---|
| bge-base | `error: qkv: staged buffer for argument 1 is 3538944 bytes, design allows 884736` | works, picks `bfp16-emulated MMAC, C as bf16` |
| bge-large | `error: layer.0.qkv: layout mismatch -- design wants 94266693..., file has f2ab7b0d...` | works |

It falls back to `runtime/artifacts`, a per-op set predating the unified xclbin.
It **fails loudly both times** — the staged-buffer size check and the
`b_layout_hash` refusal are the guards working exactly as designed, and no wrong
answer is ever returned — but it means every harness on that form must pass
`--artifacts` by hand, which is why nobody noticed. MiniLM and bge-small pass
only because the stale directory happens to be their width.

The gate drives the **shipped** form (`embed <model> <in> <out> --root <dir>`),
which is what `embed.cmd` in a release zip runs. So it tests the thing a user
gets, and it auto-picks correctly on all six in the dev tree *and* in the dist.

**4c. And the gate now records the datapath it validated.** Scraped from the
runtime's own status line, never restated from what the tool asked for — the same
discipline [`0105`](../0105-release-sweep-adopted-datapaths/TASK.md) had to adopt
for `datapath_reported`. It matters because the catalogue is per-model since
[`0104`](../0104-adopt-bfp16-per-model/TASK.md), and the table at the top of this
log shows it working: **bge-small alone reads `bf16 MMAC, C as fp32`** and the
other five read bfp16. A gate that validates vectors without recording which
datapath produced them is one artifact-set mix-up away from certifying the wrong
build.

## 5. The negative control, which found a flaw in itself

**A gate that has never failed is not known to work.**
[`0094`](../0094-t32-golden-gate-rows/TASK.md) is this project's own precedent:
the golden gate was structurally blind to cross-row corruption *and* was
comparing 4 of 128 rows, and neither was visible from the fact that it passed.

So the scoring runs a **second time on every invocation**, over the same
embeddings, with the vectors permuted. That must FAIL — and if it does not, the
run is void whatever the real result said. It is free: no extra encode, no RNG,
no seed, identical on every machine.

**Getting the permutation right took two attempts, and the first one shipped a
false pass.**

* **Rotate the whole list by one** — nearly the identity here. The corpus is
  stored cluster-major in groups of three, so `fish.a` picks up `fish.b`'s vector
  and stays in its own cluster. It caught five models by luck and
  **`nomic-embed-text-v1.5` PASSED THE CONTROL**, which is how the flaw surfaced.
* **Permute whole clusters** — cannot work, and the reason is instructive: the
  gate asks only *"is my partner nearest"*, and relabelling clusters consistently
  preserves exactly that. A control has to break the **pairing**, not the
  labelling.
* **Rotate within the gate subset by one** — provably breaks it. Cluster *k* owns
  gate slots 2*k* and 2*k*+1; after the rotation they carry the vectors of *k*'s
  `.b` and of *k*+1's `.a` — different clusters, never a near pair — while *k.b*'s
  true best match now sits in cluster *k*−1's slot. Every one of the 24 must fail,
  on every model, by construction.

Measured: **24/24 control failures on all six models.** The control is maximally
sharp rather than marginally satisfied.

## 6. What is reported and deliberately NOT gated

**Global separation** — *"the weakest near pair beats the strongest far pair"* —
is the user's property stated verbatim, and it is the **right intuition and the
wrong gate**. It is a min over 12 near pairs against a max over 264 far ones: two
extreme-value statistics, so it gets harder as the corpus grows for unchanged
model quality. `verify_embed_e2e.py` makes exactly this argument about a fixed
epsilon over 23 M pairs — *"testing against a constant tests the corpus size"*.

It is also not a correctness question. Two clusters can be legitimately adjacent
without any model being wrong; §3 found three such pairs and fixed all three in
the **corpus**, because the cluster boundary was the thing that was wrong. But
the next such pair added would fail this gate while every model behaved
perfectly.

It happens to hold on all six right now (margins **+0.189 to +0.387**), and it is
printed on every run so a regression is visible. It just does not decide
pass/fail.

Also reported: the probe row (§3), and the **worst far pairs by shared words** —
derived by Jaccard over the corpus rather than hand-picked, so the adversarial
cases cannot be quietly chosen to pass. Currently `cook.a`/`med.b` at 0.19
overlap ("a pinch of pepper" against "prescribed painkillers ... come back in a
fortnight"), well below every near pair on every model.

## 7. Commands

No `iron_env.ps1`, no `.venv-ref`. The tool is stdlib only; the interpreter below
is just what was on PATH.

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_semantics.py
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_semantics.py --models bge-base-en-v1.5
```

## 8. Artifacts

| file | what |
|---|---|
| `tools/verify_semantics.py` | the gate; stdlib only, drives the shipped `embed` subcommand |
| `tools/semantic_corpus.json` | the corpus, v5, with the design rationale in the file |
| `semantic_gate.json` | full per-model result incl. every probe rank and the control |
| `raw.txt` | verbatim output of the passing sweep |
| `semantic_corpus_v1.json`, `corpus_v1_minilm.json`, `raw_v1_corpus.txt` | **the first corpus and its failure**, kept per rule 3b — §3 is not reconstructable without them |

## 9. Problems hit

* Four corpus revisions before the design change; §3 is the log. The temptation
  to keep editing sentences until it passed was real, and the rule stated at the
  end of §3 is what stopped it.
* The first negative control passed on one model. Recorded rather than quietly
  replaced — a control that can itself be wrong is the whole reason §5 exists.
* Bash heredocs could not carry the tool source (unterminated-quote parse error
  at an unrelated line); written with the file tool instead. Second time this
  session.
* `bge-large` takes ~2 s for 36 texts because its design ships a single batch
  tier (M = 8192), so a small request pays a full dispatch. Expected, not a
  regression — it is [T40](../../research/CLOSED-THREADS.md#t40)'s "a
  long-sequence design is a bad instrument for short requests", one axis over.

## 10. Not done

* **Not wired into the release skill.** It should be — it is the one gate that
  runs against a cold dist zip with nothing installed — but that is an edit to
  `.claude/skills/release/SKILL.md` and belongs with the next release, not with
  the tool's first commit.
* **`--cpu` and the legacy-form fallback are not fixed**, only reported and
  filed as [T50](../../research/OPEN-THREADS.md#t50). Both are C++ changes
  needing a rebuild and a re-verify.
* Only the six built-ins are swept. `npuembeddings add` finetunes are the user's
  business, not this gate's.
