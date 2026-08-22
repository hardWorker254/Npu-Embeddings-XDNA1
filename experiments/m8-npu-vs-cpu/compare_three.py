# NpuEmbeddings -- NPU vs torch vs ONNX Runtime, interleaved, in one session.
# SPDX-License-Identifier: Apache-2.0
#
# docs/05-measurement's comparison protocol requires same input, same sequence
# length, same batch. It does not yet say INTERLEAVED, and it should: these are
# wall-clock numbers on a shared machine, and this session alone measured
# sentence-transformers at 710, 662.9 and 518.5 seq/s on different occasions.
# Any ratio between numbers taken minutes apart is measuring the machine's mood
# as much as the encoders.
#
# So all three run round-robin, N rounds, and the report is the MEAN of each --
# the same statistic on every side. best-of-N against a mean is the asymmetry
# this experiment exists to remove.
#
# Env: .venv-ref
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\compare_three.py
#   ... --model bge-small-en-v1.5 --rounds 5

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent


def machine_state() -> dict:
    """Power plan and mains/battery, per docs/05-measurement's protocol.

    Recorded automatically because a hand-rolled check got it wrong once:
    Win32_Battery returns NOTHING on a machine with no battery, and a naive
    "is BatteryStatus == 2" test then reports "on battery" for a desktop on
    mains. Rosti measured 145 -> 255 GFLOP/s on mains against 95 -> 111 on
    battery, so a wrong answer here invalidates every ratio below it.
    """
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Windows.Forms; "
         "$p=[System.Windows.Forms.SystemInformation]::PowerStatus; "
         "$s=(powercfg /getactivescheme); "
         "Write-Output \"$($p.PowerLineStatus)|$($p.BatteryChargeStatus)|$s\""],
        capture_output=True, text=True)
    line = ps.stdout.strip().splitlines()[-1] if ps.stdout.strip() else "?|?|?"
    parts = line.split("|")
    return {"power_line": parts[0] if parts else "?",
            "battery": parts[1] if len(parts) > 1 else "?",
            "power_plan": parts[2].strip() if len(parts) > 2 else "?"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--artifacts", default="artifacts_b128il")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--pipeline", type=int, default=2)
    ap.add_argument("--intra-op", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO / "reference"))
    sys.path.insert(0, str(HERE))
    from corpus import SENTENCES, SEQ_LEN                      # noqa: E402
    from bench_cpu_ort import export_onnx, pool, read_pooling  # noqa: E402

    import onnxruntime as ort
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model_dir = REPO / "models" / args.model
    mode = read_pooling(model_dir)
    sents = (list(SENTENCES) * ((args.batch // len(SENTENCES)) + 1))[:args.batch]

    # The task prefix is part of the WORK -- it lengthens every sequence -- so
    # a throughput comparison that omits it on one side is measuring different
    # sequences. Read it from the container so all three sides agree by
    # construction. tasks/0071.
    sys.path.insert(0, str(REPO / "tools"))
    from npue import Reader                                       # noqa: E402
    with Reader(str(REPO / "models" / f"{args.model}.npue")) as _c:
        _cfg = _c.config
    prefix = (_cfg.get("prompts") or {}).get(_cfg.get("prompt_default"), "")
    if prefix:
        print(f"  task prefix (all three sides): {prefix!r}")
        sents = [prefix + s for s in sents]

    # -- torch ---------------------------------------------------------------
    # trust_remote_code: nomic-embed-text-v1.5 needs it on older
    # transformers/sentence-transformers. Build-time .venv-ref only, never the
    # shipped runtime.
    st = SentenceTransformer(str(model_dir), device="cpu",
                             trust_remote_code=True)
    st.max_seq_length = SEQ_LEN

    def run_torch():
        st.encode(sents, batch_size=args.batch, convert_to_numpy=True,
                  normalize_embeddings=True)

    # -- onnxruntime ---------------------------------------------------------
    # DEGRADE, DO NOT CRASH. The ONNX export goes through the checkpoint's own
    # modelling code, and a custom architecture (nomic's `nomic_bert`, via
    # trust_remote_code) is not guaranteed to trace. Losing this side is
    # survivable -- tasks/0040 measured torch as the STRONGER baseline anyway
    # (489.4 vs ORT's 234.3 on MiniLM), and `prior-art.md`'s prescription of ORT
    # as *the* baseline was refuted there. What is NOT survivable is a crash
    # that throws away the whole interleaved run, or a silent drop that leaves
    # the reader thinking ORT was measured and lost.
    ort_error = None
    sess = None
    try:
        onnx_path = export_onnx(model_dir,
                                ARTIFACTS / "onnx" / f"{args.model}.onnx",
                                SEQ_LEN)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if args.intra_op:
            so.intra_op_num_threads = args.intra_op
        sess = ort.InferenceSession(str(onnx_path), so,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:                       # noqa: BLE001
        ort_error = f"{type(exc).__name__}: {exc}"
        print(f"  ONNX Runtime side UNAVAILABLE for {args.model}: {ort_error}")
        print("  -> the CPU baseline for this model is torch alone, and the "
              "ratio below says so.")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    enc = tok(sents, padding="max_length", truncation=True,
              max_length=SEQ_LEN, return_tensors="np")
    feeds = {k: enc[k].astype(np.int64)
             for k in ("input_ids", "attention_mask", "token_type_ids")
             if k in enc}

    def run_ort():
        lh = sess.run(["last_hidden_state"], feeds)[0]
        pool(lh, enc["attention_mask"], mode)

    # -- the NPU -------------------------------------------------------------
    # One process per round, because that is how the runtime is used and how
    # every previously published NPU figure was taken. Its own --bench does the
    # timing, so the numbers stay the ones the runtime reports.
    exe = REPO / "runtime" / "build" / "npuembed.exe"

    def run_npu():
        r = subprocess.run(
            [str(exe), "..", "--model", args.model,
             "--artifacts", args.artifacts, "--threads", str(args.threads),
             "--pipeline", str(args.pipeline), "--bench", "5"],
            cwd=str(REPO / "runtime"), capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"npuembed failed:\n{r.stdout[-800:]}\n{r.stderr[-400:]}")
        for line in r.stdout.splitlines():
            if "seq/s" in line:
                return float(line.split("->")[1].split("seq/s")[0])
        raise SystemExit("no seq/s line in npuembed output")

    ms = machine_state()
    print(f"model {args.model}, batch {args.batch}, seq {SEQ_LEN}, "
          f"{mode} pooling")
    print(f"  machine: {ms['power_line']} / {ms['battery']} / "
          f"{ms['power_plan']}")
    print(f"  torch threads {torch.get_num_threads()}, "
          f"ORT intra_op {args.intra_op}, "
          f"NPU {args.threads} threads / {args.pipeline} lanes")
    print(f"  interleaved, {args.rounds} rounds, MEAN reported on every side\n")

    run_torch()                           # warm
    if sess is not None:
        run_ort()

    res = {"torch": [], "onnxruntime": [], "npu": []}
    for r in range(args.rounds):
        t0 = time.perf_counter(); run_torch()
        res["torch"].append(args.batch / (time.perf_counter() - t0))
        if sess is not None:
            t0 = time.perf_counter(); run_ort()
            res["onnxruntime"].append(args.batch / (time.perf_counter() - t0))
        res["npu"].append(run_npu())
        ort_col = (f"{res['onnxruntime'][-1]:7.1f}" if sess is not None
                   else "     --")
        print(f"  round {r + 1}:  torch {res['torch'][-1]:7.1f}   "
              f"ort {ort_col}   npu {res['npu'][-1]:7.1f} seq/s")

    print()
    # The CPU RAMPS. torch measured 469 -> 686 seq/s across five rounds of one
    # run while the NPU held to +/-1%, so an all-rounds mean measures how cold
    # the machine was as much as how fast the encoder is. The second half is
    # the steady state, and using it is the choice that favours the CPU.
    half = max(1, len(res["torch"]) // 2)
    # Skip empty sides -- ONNX Runtime is absent when the export could not trace
    # this architecture, and fmean([]) raises.
    live = {k: v for k, v in res.items() if v}
    means = {k: statistics.fmean(v) for k, v in live.items()}
    steady = {k: statistics.fmean(v[-half:]) for k, v in live.items()}
    for k, v in live.items():
        print(f"  {k:<12} all {means[k]:7.1f}   steady(last {half}) "
              f"{steady[k]:7.1f}   range {min(v):7.1f}-{max(v):7.1f} seq/s")
    print()
    cpu_sides = {k: steady[k] for k in ("torch", "onnxruntime") if k in steady}
    who = max(cpu_sides, key=cpu_sides.get)
    best_cpu = cpu_sides[who]
    if ort_error:
        print(f"  ONNX Runtime NOT measured for this model ({ort_error}) -- "
              f"the baseline below is torch alone, which tasks/0040 measured "
              f"as the stronger of the two anyway.")
    print(f"  strongest CPU baseline: {who} at {best_cpu:.1f} seq/s")
    print(f"  NPU / strongest CPU   : {steady['npu'] / best_cpu:.3f}x  "
          f"(steady state, both sides)")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else ARTIFACTS / f"compare_three_{args.model}.json"
    out.write_text(json.dumps({
        "kind": "hardware measurement",
        "what": "interleaved wall-clock throughput, NPU vs torch vs ONNX Runtime",
        "machine_state": ms,
        "model": args.model, "batch": args.batch, "seq_len": SEQ_LEN,
        "pooling": mode, "rounds": args.rounds,
        "config": {"torch_threads": torch.get_num_threads(),
                   "ort_intra_op": args.intra_op,
                   "ort_version": ort.__version__,
                   "npu_threads": args.threads, "npu_lanes": args.pipeline,
                   "npu_artifacts": args.artifacts},
        "seq_per_s": res, "means": means, "steady_means": steady,
        "steady_rounds": half,
        "prefix": prefix,
        # Absent, and WHY. A side that is simply missing from the JSON reads as
        # "not run"; this says which and for what reason.
        "ort_error": ort_error,
        "strongest_cpu": who,
        "npu_over_strongest_cpu": steady["npu"] / best_cpu,
    }, indent=2), encoding="utf-8")
    # `relative_to` RAISES when --out points outside the repo, which threw away
    # a completed interleaved measurement in its very last statement (noted in
    # thread T25 and never fixed until tasks/0071). A run that has already done
    # all the work and written its artifact must not fail while announcing it.
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
