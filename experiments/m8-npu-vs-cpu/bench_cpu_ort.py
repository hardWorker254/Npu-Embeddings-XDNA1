# NpuEmbeddings -- the CPU baseline that is actually hard to beat.
# SPDX-License-Identifier: Apache-2.0
#
# research/prior-art.md prescribes ONNX Runtime CPU as THE primary baseline,
# and until now we only measured against sentence-transformers/torch -- which
# is what a user reaches for first, but is not the strongest opponent. A claim
# of "faster than the CPU" is worth what the comparator is worth.
#
# Three things this does that bench_cpu_baseline.py did not:
#
#   1. VERIFIES BEFORE IT TIMES. A faster baseline that computes something
#      else is not a baseline. ORT output is checked against
#      sentence-transformers on the same texts before any number is quoted.
#   2. REPORTS MEAN, MEDIAN AND BEST. bench_cpu_baseline.py reports best-of-5
#      while our NPU figures are means -- an asymmetry that flatters the CPU
#      by whatever its run-to-run spread happens to be. Reporting all three
#      makes the size of that effect visible instead of arguable.
#   3. IS MODEL-DRIVEN, including the pooling mode, read from the checkpoint's
#      own 1_Pooling/config.json like every other pooling site in this repo.
#
# Wall clock on both sides, which docs/05-measurement permits for end-to-end
# throughput and for nothing else.
#
# Env: .venv-ref  (torch, transformers, onnxruntime, sentence-transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\bench_cpu_ort.py
#   ... --model bge-small-en-v1.5

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent


def read_pooling(model_dir: Path) -> str:
    """Same file, same rule, as both packers and the oracle."""
    p = model_dir / "1_Pooling" / "config.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    modes = [k for k, v in c.items()
             if k.startswith("pooling_mode_") and v is True]
    if modes == ["pooling_mode_cls_token"]:
        return "cls"
    if modes == ["pooling_mode_mean_tokens"]:
        return "mean"
    raise SystemExit(f"{p}: unsupported pooling {modes}")


def export_onnx(model_dir: Path, out: Path, seq: int) -> Path:
    """Export BertModel to ONNX with torch.onnx.export.

    `optimum` is not installed and is not needed: the graph we want is the
    plain encoder, and pooling stays outside it so the ONNX side pools with
    exactly the code the reference uses. Cached -- exporting is slow and
    deterministic.
    """
    if out.exists():
        return out
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir)).eval()

    class Encoder(torch.nn.Module):
        """Positional args no longer line up with BertModel.forward, whose
        signature has gained keyword-only parameters; passing three tensors
        positionally lands one of them on `use_cache`. Naming them also fixes
        the exported graph's input order, which is what the feeds rely on."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    model = Encoder(model).eval()
    ids = torch.ones(1, seq, dtype=torch.long)
    mask = torch.ones(1, seq, dtype=torch.long)
    tt = torch.zeros(1, seq, dtype=torch.long)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        # dynamo=False forces the TorchScript exporter. torch 2.10 defaults to
        # the dynamo one, which IGNORES dynamic_axes -- the batch dimension is
        # then baked in as 1, ORT's optimiser folds a Reshape to {seq, hidden}
        # around it, and every batch above 1 fails at run time rather than at
        # export. It fails loudly, at least.
        torch.onnx.export(
            model, (ids, mask, tt), str(out),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch"},
                          "attention_mask": {0: "batch"},
                          "token_type_ids": {0: "batch"},
                          "last_hidden_state": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False)

    # The batch axis must be SYMBOLIC in the written graph. Verified rather
    # than trusted: this is the difference between a baseline and an exception.
    import onnx
    g = onnx.load(str(out)).graph
    for vi in list(g.input) + list(g.output):
        d = vi.type.tensor_type.shape.dim[0]
        if not d.dim_param:
            out.unlink(missing_ok=True)
            raise SystemExit(
                f"ONNX export baked a fixed batch into '{vi.name}' "
                f"(dim {d.dim_value}); the graph would only serve that batch")
    return out


def pool(last_hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cls":
        pooled = last_hidden[:, 0, :]
    else:
        m = mask[:, :, None].astype(np.float32)
        pooled = (last_hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    n = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    return (pooled / n).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--intra-op", type=int, default=12,
                    help="ORT intra_op_num_threads; 0 leaves ORT's default. "
                         "12 measured fastest here (24 logical cores): "
                         "4->235, 8->281, 12->309, 16->285, 24->202 seq/s")
    ap.add_argument("--tol", type=float, default=2e-5,
                    help="max abs diff ORT vs sentence-transformers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO / "reference"))
    from corpus import SENTENCES, SEQ_LEN                     # noqa: E402

    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = REPO / "models" / args.model
    mode = read_pooling(model_dir)
    onnx_path = export_onnx(model_dir, ARTIFACTS / "onnx" / f"{args.model}.onnx",
                            SEQ_LEN)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.intra_op:
        so.intra_op_num_threads = args.intra_op
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"ONNX Runtime {ort.__version__}, CPUExecutionProvider")
    print(f"model {args.model}, seq {SEQ_LEN}, {mode} pooling")
    print(f"  intra_op {args.intra_op or 'ORT default'}, "
          f"graph opt {so.graph_optimization_level.name}\n")

    def encode(sents):
        enc = tok(sents, padding="max_length", truncation=True,
                  max_length=SEQ_LEN, return_tensors="np")
        feeds = {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64),
                 "token_type_ids": enc["token_type_ids"].astype(np.int64)}
        lh = sess.run(["last_hidden_state"], feeds)[0]
        return pool(lh, enc["attention_mask"], mode)

    # -- verify BEFORE timing -------------------------------------------------
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.max_seq_length = SEQ_LEN
    ref = st.encode(list(SENTENCES), convert_to_numpy=True,
                    normalize_embeddings=True)
    got = encode(list(SENTENCES))
    delta = float(np.abs(got - ref).max())
    cos = float((got.astype(np.float64) * ref.astype(np.float64)).sum(1).min())
    print(f"  ORT vs sentence-transformers: max abs {delta:.3e}, "
          f"worst 1-cos {1 - cos:.3e}")
    if not (delta < args.tol):
        print(f"\nFAIL -- ORT does not reproduce the reference "
              f"(limit {args.tol:.0e}). A faster baseline that computes "
              f"something else is not a baseline.")
        return 1
    print("  ok -- same embeddings, so the timings below are comparable\n")

    # -- time -----------------------------------------------------------------
    rows = []
    for batch in (4, 32, 128):
        sents = (list(SENTENCES) * ((batch // len(SENTENCES)) + 1))[:batch]
        encode(sents)                                          # warm
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            encode(sents)
            ts.append(time.perf_counter() - t0)
        best, mean, med = min(ts), statistics.fmean(ts), statistics.median(ts)
        rows.append({"batch": batch, "repeats": args.repeats,
                     "best_s": best, "mean_s": mean, "median_s": med,
                     "seq_per_s_best": batch / best,
                     "seq_per_s_mean": batch / mean,
                     "seq_per_s_median": batch / med})
        print(f"  batch {batch:>4}: "
              f"best {batch / best:8.1f}   mean {batch / mean:8.1f}   "
              f"median {batch / med:8.1f} seq/s"
              f"   (spread {100 * (max(ts) - min(ts)) / min(ts):.1f}%)")

    print("\nbest-of-N flatters the CPU by exactly the spread column. Our NPU "
          "figures are means, so compare against the MEAN column.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else ARTIFACTS / f"bench_cpu_ort_{args.model}.json"
    out.write_text(json.dumps({
        "kind": "hardware measurement",
        "what": "ONNX Runtime CPU, wall clock, end-to-end throughput",
        "model": args.model,
        "pooling": mode,
        "seq_len": SEQ_LEN,
        "onnxruntime": ort.__version__,
        "verified_vs_sentence_transformers": {"max_abs": delta,
                                              "worst_1_minus_cos": 1 - cos},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
