# 0085 — the 0.4.0 sweep: twelve rows, five stages, one protocol

- **Date** 2026-08-23
- **Status** done — 12/12 rows complete on all five stages; three plumbing bugs
  found and fixed on the way, two of them introduced by me earlier the same day

## Goal

**User:** *"Kjør hele sveipet én gang, med repetisjoner — 12 rader, ett
protokoll. Da har 0.4.0 tall som er sanne samtidig, og de i dokumentene nå er
4% for lave."*

The catalogue is twelve rows now: six models × two datapaths. Every number in
`docs/` came from a different session, and
[`0084`](../0084-m13-host-isa-and-repeats/TASK.md) had just shown the published
absolutes were ~4% low.

---

## 1. The table

One session, one machine state, `--threads 24 --pipeline 4`, `--bench 5`, three
runs per throughput cell. **End-to-end throughput, not an NPU kernel claim**
(rule 1).

| model | bf16 | int8 | int8/bf16 | spread |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 992.6 | 1696.6 | 1.71× | 0.60% |
| `bge-small-en-v1.5` | 503.5 | 862.9 | 1.71× | 0.56% |
| `bge-base-en-v1.5` | 214.3 | 416.0 | 1.94× | 0.51% |
| **`bge-large-en-v1.5`** | 61.9 | **147.9** | **2.39×** | 0.20% |
| `nomic-embed-text-v1.5` | 166.8 | 325.6 | 1.95× | 0.58% |
| `embeddinggemma-300m` | 136.6 | 155.8 | 1.14× | 3.40% ‡ |

‡ arch=1 has no `--bench`; it encodes a corpus, a different harness with a
different noise floor.

**The spread settles 0084's correction**: under 0.6% on five of six rows, not
the "~4%" 0082 generalised from a single contended reading.

### Accuracy, and the two gates disagreeing about which model is hardest

| model | `1-cos` int8 | MTEB int8 (mean / worst) |
|---|---:|---|
| `all-MiniLM-L6-v2` | 1.161e-03 | −0.04 / −0.14 |
| **`bge-small-en-v1.5`** | **6.385e-04** (best) | **−0.09 / −0.41** (worst) |
| `bge-base-en-v1.5` | 1.778e-03 | +0.01 / −0.13 |
| `bge-large-en-v1.5` | **2.968e-03 FAIL** | −0.05 / −0.18 **PASS** |
| `nomic-embed-text-v1.5` | 1.098e-03 | +0.00 / −0.03 |
| `embeddinggemma-300m` | 1.070e-03 † | −0.04 / −0.25 |

† differential against the bf16 **NPU** encode; arch=1 has no HuggingFace
golden. `nearest-is-self 13/13` on a corpus whose closest off-diagonal pair
sits at cos 0.3885 — well separated, unlike `corpus_520.txt`.

**All twelve rows PASS MTEB**, bf16 means +0.00 to +0.09 and int8 means −0.09 to
+0.01.

**And the two gates rank the models differently.** `bge-small` has the *best*
`1-cos` of any int8 row and the *worst* MTEB. `bge-large` has the worst `1-cos`
— the only gate failure in the catalogue — and a comfortably passing MTEB that
reproduces [`0081`](../0081-m13-int8-everywhere/TASK.md)'s standalone run
digit-for-digit (−0.05 / −0.18, different session, different command line).
0081 §5 argued `1-cos` is a fidelity check and not a quality gate; this table is
that argument with six models instead of one.

### Against the CPU, interleaved, 8 rounds

| model | torch | ORT | NPU | NPU / best CPU |
|---|---:|---:|---:|---:|
| `nomic-embed-text-v1.5` | 76.7 | 46.2 | 166.9 | **2.18×** |
| `bge-base-en-v1.5` | 123.6 | 61.7 | 214.2 | 1.73× |
| `bge-large-en-v1.5` | 36.1 | 18.5 | 61.8 | 1.71× |
| `all-MiniLM-L6-v2` | 767.5 | 293.3 | 996.2 | 1.30× |
| `bge-small-en-v1.5` | 412.2 | 176.9 | 501.4 | 1.22× |
| **`embeddinggemma-300m`** | **85.5** | — | **75.5** | **0.88×** |

**EmbeddingGemma loses to torch on the CPU**, and it is not noise: arch=1 runs
RMSNorm ×97, RoPE and MQA attention on the host and only four GEMMs per layer on
the array, so the NPU does a smaller share of the work than in any other model.
It belongs in the table, not in a footnote.

### Energy, J per 1000 sequences

| model | CPU (bf16 row) | CPU (int8 row) | NPU bf16 | NPU int8 |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 74.0 | 67.5 | 2.55× | 3.12× |
| `bge-small-en-v1.5` | 144.9 | 145.5 | 2.18× | 3.38× |
| `bge-base-en-v1.5` | 469.3 | 475.1 | 2.94× | 5.14× |
| `bge-large-en-v1.5` | 1645.7 | 1619.8 | 2.84× | **5.64×** |
| **`nomic-embed-text-v1.5`** | 783.9 | 787.8 | 3.78× | **7.22×** |
| `embeddinggemma-300m` | 673.4 | 688.4 | 2.13× | 3.02× |

**The two CPU columns are the same measurement taken twice**, in different runs
hours apart, and they agree within 9%. That is the sweep's own internal control
and it only became visible once §2's bug was fixed — before that the int8 CPU
column read between −0.8 and 1.2.

---

## 2. Three plumbing bugs, two of them mine from the same day

**(a) `--cpu-model` was one argv token, not two.** I added it earlier today
after noticing by *reading* that int8 rows had no CPU baseline. The fix used a
nested subexpression inside the argument array:

```powershell
@( ..., $(if ($m.cpu) { "--cpu-model"; $m.cpu }), ... )
```

That does **not** splat. The subexpression stays one element holding an array;
`Invoke-Logged` stringifies it to `"--cpu-model all-MiniLM-L6-v2"`, sees the
space, and quotes it into a single token. `run_mteb.py` reported `unrecognized
arguments: --cpu-model all-MiniLM-L6-v2` — which reads like a missing flag and
is a quoting bug. **Cost all six int8 MTEB rows, two hours in.**

**(b) Splatting an array where a hashtable was needed.** Fixing (a) for the
energy stage, I wrote `& $script @arrayOfFlags`, which splats **positionally**
— so `"-Model"` bound to `energy_compare.ps1`'s first positional parameter,
`-Low [int]`, and died with `Cannot convert value "-Model" to type
"System.Int32"`. Named splatting needs a hashtable.

Two splatting mistakes in one file in one day, in opposite directions, each
producing an error message that named the wrong thing.

**(c) The energy stage measured nothing on every int8 row and reported it as a
negative.** `energy_compare.ps1` passed `$Model` — a `.npue` **container**
name — to `energy_cpu_load.py`, which loads a HuggingFace **checkpoint
directory**. `models\all-MiniLM-L6-v2.int8\` does not exist, so the CPU arm died
instantly and the differential came out between **−0.8 and 1.2 J/1000 seq**.

This one is the worst of the three, because **the differential method cannot
distinguish "ran and used little energy" from "died in 0.2 s"** — both are a
small delta. It had to be caught by a precondition, and `energy_compare.ps1`
now takes `-CpuModel` and *refuses* when the checkpoint is missing rather than
reporting a number of the wrong sign.

**(d) And a trap for the release itself**, unrelated to this run:
`make_release.ps1` hardcoded `tasks\0073-m13-release-benchmarks\sweep.json`. A
release cut after a newer sweep would have shipped the **older** numbers —
silently, and while passing its own freshness check, because that check reads
whichever file it is pointed at. Now a `-SweepDir` parameter, so a release
*names* the numbers it carries. ("Newest wins" would have been the other
fail-open, trap 7c.)

---

## 3. And the machine broke underneath it

The first run died three stages in with `ModuleNotFoundError: No module named
'torch'` — from `.venv-ref`, an interpreter that had run MTEB successfully two
hours earlier. `C:\Users\vegar\.conda\envs\iron\Lib\site-packages` had been
emptied to three entries, and `--system-site-packages` inherits exactly that
directory and nothing else.

**What made it hard to see:** bare `python` still imported numpy and
onnxruntime, because those live in the **user** site-packages
(`%APPDATA%\Roaming\Python\Python313`), which a venv deliberately ignores. So
`python -c "import numpy"` succeeded while `.venv-ref` had nothing.

Only torch was actually missing. Installed **into `.venv-ref`**, which now owns
it: 114 MB of duplication against a dependency on a directory nothing in this
repo owns. `ironenv` verified untouched. Written up in CLAUDE.md.

**The contention guard also fired twice, correctly**, on a transient from
Notepad/PyCharm/Spotify. Overriding it with `-AllowCpuContention` would have
reproduced exactly the error 0084 had just corrected, so the run waited instead.

---

## 4. What the sweep still does not do

* **No stage-level sanity check.** (c) produced a negative energy and the sweep
  wrote it to `sweep.json` without complaint. A stage that returns a physically
  impossible number should fail the row, not fill it.
* **`sweep.json` carries log paths, not extracted values.** Every table above
  was assembled by re-parsing the logs. That is fine for a human and wrong for
  `make_release.ps1`, which embeds `$sweep.rows` into the release manifest —
  so the manifest ships paths, not numbers.
* **No repeats outside throughput.** Interleaved, energy and MTEB each ran once
  per row. Throughput's three runs are what made §1's spread claim possible;
  the other three stages have no such evidence.

---

## 5. Commands

```powershell
.\tools\release_benchmark.ps1 -OutDir "tasks\0085-m13-release-sweep"

# the two repair passes, after fixing the plumbing above
.\tools\release_benchmark.ps1 -OutDir "tasks\0085-m13-release-sweep" `
    -Skip accuracy,throughput,interleaved,energy -Models <the six int8 rows>
.\tools\release_benchmark.ps1 -OutDir "tasks\0085-m13-release-sweep" `
    -Skip accuracy,throughput,interleaved,mteb  -Models <the six int8 rows>
```
