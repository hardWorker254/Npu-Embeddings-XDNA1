# NpuEmbeddings -- does XRT buffer-object flavour change DMA throughput?
# SPDX-License-Identifier: Apache-2.0
#
# FastFlowLM allocates every buffer with xrt::ext::bo(device, size) padded up to
# a 1 MB multiple (src/include/buffer.hpp); we allocate
# xrt::bo(..., XRT_BO_FLAGS_HOST_ONLY, group_id) at the exact size. That is TWO
# differences at once, and their code changes both together, so it cannot say
# which matters -- or whether either does.
#
#   host_only     what we ship
#   host_only_1m  rounding alone
#   ext           allocation class alone
#   ext_1m        exactly FastFlowLM
#
# The proposed mechanism is IOMMU page granularity: a 1 MB-multiple allocation
# might be backed by larger pages, cutting TLB pressure on the DMA path.
#
# Two numbers per mode. `seq/s` is end-to-end wall clock, which
# docs/05-measurement permits for throughput. `NPU dispatch+wait` is the
# runtime's own serialized device time -- the only part a DMA change could move,
# and therefore the one that decides the question. A mode that improves seq/s
# without moving dispatch+wait improved the HOST, not the DMA.
#
# Interleaved and steady-state, per the protocol as amended in tasks/0040.
#
# Usage:
#   & "C:\\Users\\vegar\\.conda\\envs\\iron\\python.exe" `
#       experiments\\m9-bo-mode\\bench_bo_mode.py --rounds 6

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
MODES = ["host_only", "host_only_1m", "ext", "ext_1m"]


def run(mode: str, args) -> dict:
    exe = REPO / "runtime" / "build" / "npuembed.exe"
    r = subprocess.run(
        [str(exe), "..", "--model", args.model, "--artifacts", args.artifacts,
         "--threads", str(args.threads), "--pipeline", str(args.pipeline),
         "--bo-mode", mode, "--bench", str(args.inner)],
        cwd=str(REPO / "runtime"), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{mode} failed:\n{r.stdout[-900:]}\n{r.stderr[-300:]}")
    out = {"mode": mode}
    for line in r.stdout.splitlines():
        s = line.strip()
        if "seq/s" in s and "->" in s:
            out["seq_per_s"] = float(s.split("->")[1].split("seq/s")[0])
        elif s.startswith("NPU dispatch+wait"):
            # "NPU dispatch+wait (serialized)   156.42 ms   54.3%  48 dispatches"
            out["npu_ms"] = float(s.split(")")[1].split("ms")[0])
        elif s.startswith("bo-align"):
            out["align"] = int(s.split("aligned to")[1].split("B")[0])
    if "seq_per_s" not in out:
        raise SystemExit(f"{mode}: no seq/s in output:\n{r.stdout[-600:]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--artifacts", default="artifacts_b128il")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--pipeline", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--inner", type=int, default=5,
                    help="--bench N inside each process")
    ap.add_argument("--out", default=str(HERE / "bo_mode.json"))
    args = ap.parse_args()

    print(f"model {args.model}, {args.artifacts}, {args.threads} threads / "
          f"{args.pipeline} lanes")
    print(f"interleaved, {args.rounds} rounds x --bench {args.inner}, "
          f"MEAN of the second half reported\n")

    res = {m: {"seq_per_s": [], "npu_ms": []} for m in MODES}
    align = {}
    for r in range(args.rounds):
        row = []
        for m in MODES:                       # round-robin, not mode-by-mode
            o = run(m, args)
            res[m]["seq_per_s"].append(o["seq_per_s"])
            if "npu_ms" in o:
                res[m]["npu_ms"].append(o["npu_ms"])
            align.setdefault(m, o.get("align"))
            row.append(f"{m} {o['seq_per_s']:7.1f}")
        print(f"  round {r + 1}: " + "   ".join(row))

    half = max(1, args.rounds // 2)
    print(f"\n  {'mode':<14}{'align':>9}{'seq/s':>10}{'range':>17}"
          f"{'NPU ms':>10}")
    base = None
    summary = {}
    for m in MODES:
        s = res[m]["seq_per_s"]
        steady = statistics.fmean(s[-half:])
        npu = statistics.fmean(res[m]["npu_ms"][-half:]) if res[m]["npu_ms"] else float("nan")
        if base is None:
            base = steady
        summary[m] = {"steady_seq_per_s": steady, "steady_npu_ms": npu,
                      "all": s, "npu_ms_all": res[m]["npu_ms"],
                      "alignment_bytes": align[m],
                      "vs_host_only": steady / base}
        print(f"  {m:<14}{align[m]:>7}B{steady:>10.1f}"
              f"{min(s):>8.1f}-{max(s):<8.1f}{npu:>10.1f}"
              f"   {steady / base:5.3f}x")

    print("\n  A mode that moves seq/s but NOT the NPU ms did not change DMA.")
    Path(args.out).write_text(json.dumps({
        "kind": "hardware measurement",
        "what": "XRT buffer-object allocation flavour vs throughput",
        "model": args.model, "artifacts": args.artifacts,
        "rounds": args.rounds, "inner_bench": args.inner,
        "steady_rounds": half,
        "modes": summary,
    }, indent=2), encoding="utf-8")
    print(f"wrote {Path(args.out).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
