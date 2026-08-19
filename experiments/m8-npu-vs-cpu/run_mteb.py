# NpuEmbeddings -- the M8 accuracy gate: MTEB on the NPU path vs the CPU path.
# SPDX-License-Identifier: Apache-2.0
#
# THE QUESTION THIS ANSWERS
# -------------------------
# Every accuracy number the project has produced so far is a FIDELITY measure:
# how close the NPU embedding is to the fp32 oracle on four fixed sentences
# (1-cos 1.086e-05). That is necessary and not sufficient. The claim the whole
# project rests on is that the embeddings are still GOOD -- that a downstream
# task cannot tell the difference. Only a benchmark can say that.
#
# THE COMPARISON IS PAIRED, AND THAT IS THE POINT
# -----------------------------------------------
# Both sides run the SAME tasks through the SAME mteb version with the SAME
# sequence length (64, the compiled design's), and the CPU side is the same
# checkpoint. So the absolute scores are "MiniLM at seq 64" -- below the
# published seq-256 leaderboard numbers, and that is expected and irrelevant.
# What is being tested is the DELTA. A gate on the delta cannot be gamed by
# choosing an easy task.
#
# Env: .venv-ref  (mteb, sentence-transformers, transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py
#   ... --tasks STSBenchmark,SICK-R --limit 400

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARTIFACTS = HERE / "artifacts"
SEQ = 64

# Small, fast, and load-bearing: three STS tasks (do the embeddings preserve
# semantic similarity?) plus one classification and one clustering task (is the
# embedding SPACE still shaped the same?). Retrieval is deliberately omitted --
# it is the slowest family by an order of magnitude and STS+clustering already
# exercise the same geometry.
DEFAULT_TASKS = ["STSBenchmark", "SICK-R", "STS12",
                 "Banking77Classification", "TwentyNewsgroupsClustering"]


def score_of(result) -> dict:
    """Pull the headline score out of an mteb result, version-tolerantly."""
    out = {}
    try:
        for split, entries in result.scores.items():
            vals = []
            for e in entries:
                if isinstance(e, dict) and "main_score" in e:
                    vals.append(float(e["main_score"]))
            if vals:
                out[split] = float(np.mean(vals))
    except Exception as exc:                                # noqa: BLE001
        out["error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--sides", default="cpu,npu",
                    help="which encoders to run")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="container under models/, without .npue -- both "
                         "sides use it, so the comparison stays like for like")
    ap.add_argument("--artifacts", default="artifacts_b128il")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--pipeline", type=int, default=2)
    ap.add_argument("--out", default=str(ARTIFACTS / "mteb_results.json"))
    args = ap.parse_args()

    import mteb

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]

    def build(side):
        if side == "cpu":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(str(REPO / "models" / args.model),
                                    device="cpu")
            # MUST match the NPU design's sequence length or the comparison
            # hands the CPU strictly more information.
            m.max_seq_length = SEQ
            return m
        sys.path.insert(0, str(HERE))
        from npu_encoder import NpuEncoder
        return NpuEncoder(artifacts=args.artifacts, threads=args.threads,
                          pipeline=args.pipeline, model=args.model)

    results = {}
    for side in sides:
        print(f"\n=== {side.upper()} ===", flush=True)
        model = build(side)
        results[side] = {}
        for name in task_names:
            try:
                tasks = mteb.get_tasks(tasks=[name])
                ev = mteb.MTEB(tasks=tasks)
                t0 = time.perf_counter()
                res = ev.run(model, output_folder=None, verbosity=0)
                el = time.perf_counter() - t0
                sc = score_of(res[0]) if res else {"error": "no result"}
                main_score = next((v for k, v in sc.items()
                                   if k in ("test", "validation", "dev")),
                                  next(iter(sc.values()), float("nan")))
                results[side][name] = {"scores": sc, "main": main_score,
                                       "seconds": el}
                print(f"  {name:<32} {main_score:.4f}   ({el:.1f} s)",
                      flush=True)
            except Exception as exc:                        # noqa: BLE001
                print(f"  {name:<32} FAILED: {exc}", flush=True)
                results[side][name] = {"error": str(exc)}

    # The gate: per task and on the mean, NPU must be within 0.5 points of CPU.
    print("\n================ M8 GATE ================")
    print(f"  {'task':<32} {'CPU':>8} {'NPU':>8} {'delta':>8}")
    deltas, rows = [], []
    for name in task_names:
        c = results.get("cpu", {}).get(name, {}).get("main")
        n = results.get("npu", {}).get(name, {}).get("main")
        if c is None or n is None or not np.isfinite([c, n]).all():
            print(f"  {name:<32} {'--':>8} {'--':>8} {'--':>8}")
            continue
        d = (n - c) * 100.0
        deltas.append(d)
        rows.append({"task": name, "cpu": c, "npu": n, "delta_points": d})
        print(f"  {name:<32} {c * 100:8.2f} {n * 100:8.2f} {d:+8.2f}")
    ok = None
    if deltas:
        mean_d = float(np.mean(deltas))
        worst_d = float(np.min(deltas))
        print(f"  {'MEAN':<32} {'':>8} {'':>8} {mean_d:+8.2f}")
        print(f"  worst single task: {worst_d:+.2f} points")
        ok = abs(mean_d) <= 0.5 and worst_d >= -0.5
        print(f"\n{'PASS' if ok else 'FAIL'} -- gate is |mean| <= 0.5 points "
              f"AND no task worse than -0.5")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "kind": "hardware measurement", "task": "0035",
        "seq_len": SEQ, "tasks": task_names,
        "artifacts": args.artifacts, "pipeline": args.pipeline,
        "note": "seq 64 truncation on BOTH sides; absolute scores are below "
                "published seq-256 numbers by construction. The claim is the "
                "delta.",
        "per_side": results, "gate": {"rows": rows, "pass": ok},
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
