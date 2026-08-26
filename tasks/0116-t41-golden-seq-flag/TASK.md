# 0116 — T41 closed: the golden pipeline takes `--seq`, and the flag reproduces the hand edits byte-for-byte

- **Date** 2026-08-26
- **Milestone** research (T41)
- **Status** done — T41 **ANSWERED**. All three golden makers take `--seq`,
  defaulting to the corpus module's constant so an invocation without it is
  unchanged. Verified two ways: the default reproduces a committed golden
  bit-for-bit, and `--seq 256` reproduces byte-for-byte the file
  [`0112`](../0112-t40-seq256-nomic/TASK.md) made by editing the constant.

## Goal

Close [T41](../../research/CLOSED-THREADS.md#t41), filed by
[`0113`](../0113-t40-seq512-close/TASK.md) after hitting the same wall twice:

> `reference/corpus_nomic.py:18` is `SEQ_LEN = 64`, a module constant with no
> flag […] **every new sequence length needs an edit-and-restore around the
> golden regeneration.** 0112 and 0113 each did exactly that, by hand.

It is the same class [`0110`](../0110-refuse-silent-truncation/TASK.md) removed
from `tools/export_gemm_rtp.py` (`SEQ = 64` → `--seq`), one layer further out in
the reference pipeline — and the reference pipeline is what every accuracy number
in this project is measured against, which is why 0113 declined to change it
inside a measurement task.

## What was done

`reference/corpus.py`, `corpus_gemma.py` and `corpus_nomic.py` all export
`SEQ_LEN = 64`, and their three makers import it and use it in 5–8 places each
(padding width, the printed banner, the `seq_len` metadata field, and the golden
**filename**). All uses are inside `main()`; only the import is at module scope.

So the change is small and total:

1. The import becomes `SEQ_LEN as DEFAULT_SEQ_LEN`.
2. `--seq` is added, `default=DEFAULT_SEQ_LEN`.
3. Immediately after `parse_args()`, a single `SEQ_LEN = args.seq` **shadows**
   the name for the rest of `main()`, so every existing use picks it up and no
   call site changes.

The shadowing is deliberate and is commented as such in each file. It was
checkable rather than assumed — every `SEQ_LEN` reference in all three makers
was confirmed to be inside `main()` before relying on it:

```
make_goldens.py       46 OUTSIDE main (the import), 109/114/164/194/219 in main
make_goldens_gemma.py 38 OUTSIDE main (the import),  69/73/140/175/205 in main
make_goldens_nomic.py 53 OUTSIDE main (the import), 128/132/137/139/140/190/226/248 in main
```

The constant stays in `corpus*.py` as the default, so nothing that does not pass
`--seq` changes at all — including `tools/release_benchmark.ps1` and the release
skill, neither of which was touched.

Also fixed, one line: `make_goldens_nomic.py`'s own error message told the
reader to *"Raise SEQ_LEN in corpus_nomic.py"*, which is now the wrong advice.
It says "Pass a larger `--seq`".

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# 1. does the DEFAULT path still produce the committed golden?
.\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --force

# 2. does the FLAG reproduce what 0112 made by editing the constant?
.\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --seq 256 --force

# 3. the other two makers, at their defaults
.\.venv-ref\Scripts\python.exe reference\make_goldens.py --force
.\.venv-ref\Scripts\python.exe reference\make_goldens_gemma.py --help
```

## Result

**1. The default is byte-identical.** nomic's committed s64 golden:

```
sha256 before : 4521b8b48ae07641b9a5e6c1ba5be34040ff9e429719033f6488d8a6d060ea89
sha256 after  : 4521b8b48ae07641b9a5e6c1ba5be34040ff9e429719033f6488d8a6d060ea89
```

**2. The flag reproduces the hand edit, byte-for-byte.** This is the check that
matters, because it says the flag and the edit are the same operation:

```
s256 golden, made by EDITING the constant (0112): 1ed01589e86bfbe7d9aa00c202db2a8a1073583a4091eb5c1fb494ee457211a0
s256 golden, made by the FLAG (this task):        1ed01589e86bfbe7d9aa00c202db2a8a1073583a4091eb5c1fb494ee457211a0
identical: True
```

Both regenerations passed the maker's own two-oracle check (native transformers
vs SentenceTransformer with `trust_remote_code`) at `max abs diff 1.356e-06`.

## Problems hit — and one thing that looked like a regression and was not

`make_goldens.py` at its default rewrote `minilm_l6_s64_boundary.safetensors`
and `git status` showed it **modified**. On a task whose whole claim is
"byte-neutral", that is the alarming outcome, and it took three checks to
resolve rather than one:

1. **Is it deterministic?** Two regenerations gave the same new hash
   `ad239fc7…` ≠ the committed `ce11b513…`. So: reproducible drift, not noise.
2. **Is it mine?** Ran the **pre-change script**, recovered with
   `git show HEAD:reference/make_goldens.py`. It produced the *same*
   `ad239fc7…`. So the drift exists without this task's edit — exonerated by
   experiment rather than by argument.
3. **Is it the data?** Compared the two files tensor by tensor:

   ```
   keys committed: 14   regenerated: 14   same set: True
   tensors differing: 0 of 14
   worst max-abs-diff: 0.000e+00
   ```

   **Every tensor is bit-identical.** The only difference is a single metadata
   key: the regenerated file carries `'pooling': 'mean'`, which the committed
   one does not. The committed golden simply predates the commit that started
   recording pooling in the metadata.

So the file's *hash* differs while its *contract* does not, and the working copy
was restored with `git checkout` rather than committing a re-derived golden. Two
things worth carrying forward: **a golden's hash is not its contract** — compare
tensors before believing a diff — and nomic's golden reproducing exactly while
MiniLM's did not is explained entirely by which file was written more recently,
not by anything about the models.

## Artifacts

Three modified files (`reference/make_goldens{,_gemma,_nomic}.py`) and the two
long-sequence nomic goldens from 0112/0113, now reproducible from the command
line without editing anything.

## Next

Nothing on T41. The remaining register is [T3](../../research/OPEN-THREADS.md#t3)
/ [T28](../../research/OPEN-THREADS.md#t28) (the device-resident relay, parked
with a price) and [T42](../../research/OPEN-THREADS.md#t42) (attention on the
array for long-sequence designs, filed with a trigger).
