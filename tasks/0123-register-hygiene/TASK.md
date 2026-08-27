# 0123 — T49 closed, T52 filed, T44 re-ranked: the register made trustworthy before 0.5.0 quotes it

**Date**: 2026-08-27
**Goal**: Phase 0 of the 0.5.0 plan. Before the measurement chain starts
quoting the register and the adoption chain starts extending it, the register
itself gets the mechanical check T49 asked for, the candidate note that
changes the next-encoder cost ranking, and its stale self-descriptions fixed.
No rebuild, no hardware.

## What was done

### 1. T49 check 1 — built, and proven on a planted copy

`tools/check_register.py` gains check **2b**: a thread in
`research/CLOSED-THREADS.md` whose body carries a **bold `REOPENED`**
annotation (`\*\*REOPENED` — the marker, not the word) must either have a
section back in `OPEN-THREADS.md` or a later `SUPERSEDED by [Tnn]` block after
the annotation. Otherwise the run FAILS (exit 1), which gates a release per
rule 3.

Two deliberate design points:

* **The bold marker is the trigger, not the word.** T49's own closed entry —
  and this file — must be able to *discuss* REOPENED annotations without
  firing the check. First draft of the closure text contained a literal
  `**REOPENED` inside backticks and would have failed the check on the very
  entry that closes the thread; caught before install by grepping the staged
  file for `\*\*REOPENED` (1 occurrence left: T2's genuine annotation).
* **`SUPERSEDED by [Tnn]` is accepted as resolution** because a reopening
  handed to a successor thread is acted on; and if that successor later goes
  stale as a *link* (the anchor moves files), the existing check 5 (dangling
  anchors) fires — the two checks compose.

Planted-case proof (the check must fire on the world as it was on 2026-08-22,
when T2's annotation was written and nothing acted on it):

```
$SP = "$env:LOCALAPPDATA\...\scratchpad"    # session scratchpad
mkdir $SP/regtest/tools, $SP/regtest/research, $SP/regtest/tasks, $SP/regtest/docs
cp tools/check_register.py $SP/regtest/tools/
cp research/OPEN-THREADS.md $SP/regtest/research/
# CLOSED-THREADS.md copied with T2's SUPERSEDED block replaced by a stub
cd $SP/regtest ; C:\Users\vegar\.conda\envs\iron\python.exe tools\check_register.py
```

```
FAIL: REOPENED annotation in CLOSED-THREADS that was never acted on:
  T2 says **REOPENED but has no section back in OPEN-THREADS.md
       and no later 'SUPERSEDED by [Tnn]' block
  (T2 sat like this for five days while CLAUDE.md said 'three live threads')
exit=1
```

On the real repo (where T2's SUPERSEDED-by-T48 block resolves it) the check
passes.

### 2. T49 check 2 — scoped, and resolved editorially

As T49 itself suspected. A grep of closed threads for `a_dtype` /
`emulate_bfp16` claims against shipped `design.json` files would catch only
the *datapath* flavour of premise decay — T27's premise ("bfp16 was never
adopted") was a **decision**, not a config field, until 0104 made it one, so
the mechanical check would have a structural false-negative on exactly the
class of premise most likely to decay. A check that guarantees false negatives
teaches readers to trust a green result it cannot deliver.

The durable fix is one sentence added to `CLOSED-THREADS.md`'s preamble:
**a closure names the condition it depends on** — if the reason a thread
closed can expire (a datapath not adopted, a host share, a seq length), the
entry says so in one line. Enforcement stays with the release-sweep register
check plus the reader. Existing closures were *not* retro-edited (rule 3b:
verbatim), but T48's eventual closure and everything after it write under the
new rule.

### 3. T49 moved to CLOSED-THREADS as ANSWERED

Section moved verbatim, status line updated, closure block appended pointing
here. Anchor `t49` now lives in `CLOSED-THREADS.md`; the two inbound links
from T2's superseded block and T27's annotation updated from
`OPEN-THREADS.md#t49` to `#t49`, and `CLAUDE.md`'s link re-pointed. Intra-body
links in the moved section re-prefixed (`#t45` → `OPEN-THREADS.md#t45` etc.),
which check 5 verifies mechanically.

### 4. T52 filed — the SentencePiece Unigram tokenizer

The 0.5.0 plan adopts `Alibaba-NLP/gte-multilingual-base`, whose XLM-R
tokenizer is SentencePiece **Unigram** — a third family, sibling of T43's
byte-BPE, not a discharge of it. The thread records what the exploration
established: the ~600–900 LOC estimate 0055 budgeted (for what it thought
Gemma needed) and 0061 never spent when Gemma turned out to be BPE; the reuse
surface (`json_min.cpp`, the `GEMATOK1` blob convention,
`verify_tokenizer_gemma.py`'s 1,925/1,925 byte-exact harness pattern); the
no-precedent parts (the `precompiled_charsmap` trie, the Viterbi loop); and
that `gemma_tokenizer_gen.cpp`'s own `model.type == "BPE"` refusal is the TODO
marker. Filed now rather than in the build task because the register's rule is
"a thread is added the moment a task says open question" — and this planning
session is that moment.

### 5. T44 re-ranked — a fifth candidate needs no array design at all

Dated annotation added to T44: `gte-multilingual-base`'s four GEMM shapes
(qkv N=2304, attn_out 768, ffn_up 6144, ffn_down K=3072) are a **literal
match to `runtime/artifacts_nomic_bfp16/gemm_rtp/design.json`, `b_layout_hash`
included**, seq 256/512 sets already built — zero `export_gemm_rtp.py` runs.
Its blocker is T52, not T43. The four existing candidates stand exactly as
priced. Verified against the design file in-tree, not assumed:

```
python - <<EOF
import json; d = json.load(open('runtime/artifacts_nomic_bfp16/gemm_rtp/design.json'))
print(d['tile'], d['cols'], [ (s['op'], s['K'], s['N']) for s in d['streams'][:4] ])
EOF
```

(The forward-pass deltas — GELU-for-SiLU on the gate half, real biases,
NTK-scaled theta — are recorded in T44 with the theta explicitly marked
*must be read out of `new-impl/modeling.py`, not assumed*, per 0068's
silently-wrong-theta precedent.)

### 6. Stale self-descriptions fixed

* `CLAUDE.md` rule 3 said `OPEN-THREADS.md` holds "**1 thread, 163 lines as of
  2026-08-26**" while its own current-state section, 530 lines later, said
  ten — the exact decay class T49 is about, in the file that defines the
  rules. Replaced with a count-free pointer: the register's own header is the
  authority.
* `CLAUDE.md`'s ten-thread narrative updated: T49 closed, T52 filed, T44
  re-ranked; "the other 43 are closed" → 44.
* `OPEN-THREADS.md` header recounted: T49 out, T52 in — **ten threads**, with
  the T45–T49 batch description now T45–T48 plus a pointer to T49's closure.

## Commands run

```
C:\Users\vegar\.conda\envs\iron\python.exe tools\check_register.py   # before: register OK, 10 open / 43 closed
# edits to tools/check_register.py, research/OPEN-THREADS.md,
# research/CLOSED-THREADS.md, CLAUDE.md (staged as .tmp, then installed)
C:\Users\vegar\.conda\envs\iron\python.exe tools\check_register.py   # planted copy: FAIL, exit 1  (transcript above)
C:\Users\vegar\.conda\envs\iron\python.exe tools\check_register.py   # after: register OK, 10 open / 44 closed
```

Final state:

```
note: 5 of 122 task logs are referenced from no thread, doc or note:
  0014 0037 0041 0086 0119
note: OPEN-THREADS.md       723 lines, 10 threads
note: CLOSED-THREADS.md    3826 lines, 44 threads
register OK
```

(The five advisory orphans are pre-existing and untouched; 0119 will gain its
reference when the 0.5.0 release task points at the 0.4.0 one.)

## Problems hit

* **The closure text nearly failed its own check** (see §1) — the check's
  trigger had to be the bold marker, not the word, and that requirement is now
  written into both the script's docstring and the check's failure message.
* **Two references in the first draft of T52 were wrong from memory**: the
  Gemma tokenizer task is `0061` (not 0057), the estimate came from `0055`,
  and the Unigram-to-BPE reversal is recorded in T29 (not T31). Caught by
  grepping before install — the same "read the file before writing code"
  lesson T29 itself records.
