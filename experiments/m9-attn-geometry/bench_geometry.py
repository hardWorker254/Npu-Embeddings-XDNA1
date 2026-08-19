# NpuEmbeddings -- what a unified attention geometry would cost the projections.
# SPDX-License-Identifier: Apache-2.0
#
# Attention's per-head GEMM is [64,64] x [64,64]. `N % (n * cols) == 0` forces
# n*cols to divide 64, and the AIE microkernel forces n >= 16
# (`static_assert(n % (2*t) == 0)`, t = 8) -- so a design that can express
# attention uses AT MOST 4 of the 8 columns. That is structural, not a tuning
# choice.
#
# The question this measures is therefore not "can we", but "what does the
# rest of the model pay for it". Four geometries, all running the same
# bge-large encode:
#
#   A  m=64 k=64 n=32  8 cols   production; cannot express attention
#   B  m=32 k=64 n=16  8 cols   mid
#   C  m=16 k=64 n=16  8 cols   the unified TILE at production width
#   D  m=16 k=64 n=16  4 cols   the unified geometry -- the only candidate
#
# C is the control that makes the answer decomposable: A -> C is the
# tile-size penalty alone, C -> D is the column-count penalty alone. Without it
# the two are confounded and "D is X% slower" explains nothing.
#
# Interleaved and steady-state per the protocol as amended in tasks/0040.
#
# Usage:
#   & "C:\\Users\\vegar\\.conda\\envs\\iron\\python.exe" `
#       experiments\\m9-attn-geometry\\bench_geometry.py --rounds 4

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent

# (label, artifacts dir, container, expresses attention?)
GEOMS = [
    ("A m64n32 c8", "artifacts_large",       "bge-large-en-v1.5", False),
    ("B m32n16 c8", "artifacts_large_m32",   "bge-large-n16",     False),
    ("C m16n16 c8", "artifacts_large_m16c8", "bge-large-n16",     False),
    ("D m16n16 c4", "artifacts_large_m16c4", "bge-large-n16",     True),
]


def run(art: str, model: str, args) -> dict:
    exe = REPO / "runtime" / "build" / "npuembed.exe"
    r = subprocess.run(
        [str(exe), "..", "--model", model, "--artifacts", art,
         "--threads", str(args.threads), "--pipeline", str(args.pipeline),
         "--bench", str(args.inner)],
        cwd=str(REPO / "runtime"), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{art} failed:\n{r.stdout[-900:]}\n{r.stderr[-300:]}")
    out = {}
    for line in r.stdout.splitlines():
        s = line.strip()
        if "seq/s" in s and "->" in s:
            out["seq_per_s"] = float(s.split("->")[1].split("seq/s")[0])
        elif s.startswith("NPU dispatch+wait"):
            out["npu_ms"] = float(s.split(")")[1].split("ms")[0])
    if "seq_per_s" not in out:
        raise SystemExit(f"{art}: no seq/s:\n{r.stdout[-600:]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--pipeline", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "geometry.json"))
    args = ap.parse_args()

    print("bge-large, batch 128, seq 64 -- four tile geometries, interleaved")
    print(f"{args.rounds} rounds x --bench {args.inner}, "
          f"mean of the second half\n")

    res = {g[0]: {"seq_per_s": [], "npu_ms": []} for g in GEOMS}
    for r in range(args.rounds):
        row = []
        for label, art, model, _ in GEOMS:
            o = run(art, model, args)
            res[label]["seq_per_s"].append(o["seq_per_s"])
            if "npu_ms" in o:
                res[label]["npu_ms"].append(o["npu_ms"])
            row.append(f"{label.split()[0]} {o['seq_per_s']:6.1f}")
        print(f"  round {r + 1}: " + "   ".join(row))

    half = max(1, args.rounds // 2)
    steady = {k: statistics.fmean(v["seq_per_s"][-half:]) for k, v in res.items()}
    npums = {k: (statistics.fmean(v["npu_ms"][-half:]) if v["npu_ms"] else float("nan"))
             for k, v in res.items()}

    base = steady[GEOMS[0][0]]
    print(f"\n  {'geometry':<14}{'attn?':>7}{'seq/s':>9}{'NPU ms':>10}{'vs A':>8}")
    for label, _, _, attn in GEOMS:
        print(f"  {label:<14}{'yes' if attn else 'no':>7}{steady[label]:>9.1f}"
              f"{npums[label]:>10.1f}{steady[label] / base:>7.3f}x")

    a, c, d = (steady[GEOMS[i][0]] for i in (0, 2, 3))
    print(f"\n  tile-size penalty   A -> C : {c / a:.3f}x")
    print(f"  column-count penalty C -> D : {d / c:.3f}x")
    print(f"  total cost of a unified geometry : {d / a:.3f}x")

    Path(args.out).write_text(json.dumps({
        "kind": "hardware measurement",
        "what": "tile geometry vs bge-large throughput; can attention share a design",
        "geometries": [{"label": g[0], "artifacts": g[1], "model": g[2],
                        "expresses_attention": g[3],
                        "steady_seq_per_s": steady[g[0]],
                        "steady_npu_ms": npums[g[0]],
                        "all": res[g[0]]["seq_per_s"]} for g in GEOMS],
        "tile_penalty_A_to_C": c / a,
        "column_penalty_C_to_D": d / c,
        "unified_total": d / a,
        "rounds": args.rounds, "inner_bench": args.inner,
    }, indent=2), encoding="utf-8")
    print(f"wrote {Path(args.out).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
