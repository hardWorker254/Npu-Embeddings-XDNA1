# reference/ — the oracle

The fp32 numpy implementation of all-MiniLM-L6-v2 and the golden vectors every
NPU kernel is validated against. Built in M3
([`tasks/0005`](../tasks/0005-m3-python-reference/TASK.md)).

Nothing here ships. Python is prototyping and build-time only
([`../docs/00-overview.md`](../docs/00-overview.md), ground rule 3).

## The environment boundary

Golden data crosses environments **as files, never as imports** — a `pip install`
accident must not break the toolchain that took the most work to get running.

| script | env | why |
|---|---|---|
| `fetch_model.py` | `.venv-ref` | huggingface_hub |
| `make_goldens.py` | `.venv-ref` | torch, transformers, sentence-transformers |
| `encoder.py` | either | **numpy only** |
| `check_reference.py` | **iron** | numpy only — running it there is the proof the boundary holds |
| `precision_study.py` | **iron** | numpy only |

`safetensors_io.py` is our own ~80-line reader/writer, so the consuming side
needs no `safetensors` package. M4/M7 parse this format from C++ anyway.

## Files

| | |
|---|---|
| `encoder.py` | `BertModel` + mean pool + L2 norm. Every decision from [`docs/04-model`](../docs/04-model/README.md) implemented where it matters, with the reason in a comment. QKV fused as M4 will bake it; `1/√32` deliberately *not* folded, so M4's fold stays provable. All six NPU-bound GEMMs route through a swappable `self.gemm`. |
| `corpus.py` | Four frozen sentences chosen to hit the tokenizer traps (accents, CJK, `##` decomposition, ragged padding). Changing it invalidates every golden. |
| `fetch_model.py` | Downloads the checkpoint and **asserts it against the docs** — config, all 101 architecture tensors, shapes, dtypes. Pins the sha256 into `CHECKPOINT.json`. |
| `make_goldens.py` | Writes the goldens. `--taps` adds the full 75-tensor intermediate dump. |
| `check_reference.py` | **The M3 gate.** Reference vs HuggingFace, tensor by tensor. |
| `precision_study.py` | What bf16 / bfp16 cost the embedding. Calibrated against M2's hardware measurement first — a simulation that cannot reproduce the measurement has no standing to predict. |

## Goldens

| file | size | in git |
|---|---|---|
| `goldens/minilm_l6_s64_boundary.safetensors` | 3.2 MB | **yes** — the contract |
| `goldens/precision_study.json` | small | **yes** |
| `goldens/minilm_l6_s64_taps.safetensors` | 54 MB | no — regenerate with `--taps` |

Both carry the checkpoint sha256 in their metadata. `check_reference.py` refuses
to run against a checkpoint that does not match.

## Reproduce from scratch

```powershell
& ".\.venv-ref\Scripts\python.exe" -m pip install transformers safetensors sentence-transformers huggingface_hub
& ".\.venv-ref\Scripts\python.exe" reference\fetch_model.py
& ".\.venv-ref\Scripts\python.exe" reference\make_goldens.py
& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference.py     # the gate
& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\precision_study.py
```

## Status

**Gate passed.** ≤ 9.9e-07 relative Frobenius at every layer boundary,
`1-cos = 2.2e-08` on the final embedding, against two independent oracles
(HuggingFace `BertModel` and the sentence-transformers pipeline).

The error grows monotonically with depth, which is what accumulation-order noise
looks like. A formula error would show as a flat ~1e-3 floor at every layer.
