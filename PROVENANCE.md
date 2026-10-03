# PROVENANCE.md

Two conventions in this source tree look like broken paths and are not. This
file says what they refer to, so that a reader who follows one finds an
explanation rather than a dead end.

Both were load-bearing before this file existed, in the sense that a reader
following `tasks/0045` or `docs/04-model/npue-format.md` would find nothing in
this repository and no way to tell whether the citation was stale, wrong, or
referring to something held elsewhere.

---

## `tasks/NNNN`

**These are entries in the project's task log, which is not in this
repository.** The number is a stable identifier; the log entry is the record of
what was tried, measured, and concluded.

75 distinct task numbers are cited from 64 files, spanning 0004 to 0140. They appear overwhelmingly in
comments next to a hard-won decision, and the comment almost always carries the
conclusion too, because the code was written to be readable without the log:

```c++
  // A timed run REFUSES to start when the array is not ours. tasks/0044 read
  // 221.4 seq/s off a device another process was using...
```

If you delete one of these comments, the decision stays and the measurement
behind it goes with it. If you keep it, the number tells you which log entry to
look up. The number is the citation; the log is the bibliography.

### What was in the repository, and what happened to it

There *was* a `tasks/` directory, holding four files:

| File | What it was | Now |
|---|---|---|
| `0074-m13-gemma-on-npu/corpus_520.txt` | **live build input** — gemma's default int8 calibration corpus, 520 lines, read by `tools/pack/pack_npue.py` | `tools/data/gemma_corpus_520.txt` |
| `0121-semantic-gate/semantic_gate.json` | gate output — `verify_semantics.py` | `gate_output/semantic_gate.json` |
| `0132-t51-tail-gate/tail_gate.json` | gate output — `verify_tail.py` | `gate_output/tail_gate.json` |
| `0037-m9-tiers-endpoint/verify_endpoint.json` | gate output — `verify_endpoint.py` | `gate_output/verify_endpoint.json` |

The directory has been removed. The reason it existed at all was this: a build
input was sitting in a directory named after a work log. That makes the file
look disposable and the directory look load-bearing, and it cost three
consequences that are worth recording because they are easy to repeat:

1. **The live data was one careless `rm -rf` from being lost**, and nothing in
   the tree said it was live — a comment in `pack_npue.py` named the path, which
   is the only place that recorded it.
2. **The gate outputs needed three `.gitignore` lines written out by full
   path**, each naming a task, so the ignore list had to be edited whenever a
   gate was added or renamed. There is now one line, `gate_output/`, and one
   default location in `tools/lib/gate_output.py`.
3. **`ls tasks/` could not tell you what the build reads.** A reader could not
   distinguish an input from last Tuesday's measurement without opening each
   file.

The citations in comments were left alone deliberately. They are 75 stable
identifiers into a log, and deleting the directory does not make the
measurements wrong — it makes the *location* of the measurement unavailable,
which is what this file is for.

---

## `docs/…`

**These paths name documents that are not in this repository either.** There is
no `docs/` directory here, and there has not been one in this tree's history as
committed. Roughly 45 places in the source cite one of nine paths:

| Cited path | What it held |
|---|---|
| `docs/CURRENT_STATUS.md` | the running status log: measured numbers, rejected hypotheses, known walls |
| `docs/00-overview.md` | system overview and design |
| `docs/01-hardware/` | hardware and toolchain setup |
| `docs/04-model/README.md` | the encoder design specification — pooling, normalisation, the numerical decisions, and the "sharp edges" named by number throughout `reference/encoder.py` and `tools/lib/npue.py` |
| `docs/04-model/npue-format.md` | the container format spec; `runtime/src/common/npue_pack.cpp` is the implementation |
| `docs/05-measurement` | measurement methodology — what may and may not be quoted as a hardware result |
| `docs/06-performance.md` | performance results |
| `docs/history.md` | chronological project history |

These carry more reasoning than the `tasks/` citations do: `docs/04-model` is
where the encoder's numerical rules come from, and a comment saying "this is
`docs/04-model`'s -inf/NaN landmine" tells you the rule *exists* without telling
you what it is. Treat those comments as pointers to a rule you will have to
re-derive, and treat the numbers they attach as measured.

Note that this convention is narrower than it looks. A great many other
`docs/…` strings appear in this repository — in comments inherited from
vendored code and in site-packages — and those are citations to *external*
documentation (PyTorch, HuggingFace, LLVM, the OpenAI API). Those are correct as
written and are not what this file is about.

---

## If you are restoring either tree

Both are recoverable in principle and neither is recoverable from this
repository. If you are reconstructing `docs/` from the source, the highest-value
targets in order are:

1. **`docs/04-model/README.md`** — 20-odd citations, and it is the authority for
   decisions the code states without explaining.
2. **`docs/CURRENT_STATUS.md`** — cited wherever a measured number's verdict
   (pass/fail, adopted/rejected) is recorded.
3. **`docs/04-model/npue-format.md`** — the container layout, one citation in
   the C++ packer and one in the header.

For `tasks/`, the numbers are dense and sequential (0004 … 0138), so the log
either exists in full or the citations have to become the record. If it does not
exist, the honest move is to convert each surviving citation into the claim it
supports — which is a real piece of work, not a mechanical one, and is why it has
not been done by script.
