"""T13: where does the pretiled stall sit? One traced core, per invocation.

Reads the trace.json the harness already wrote (last repeat of each arm).
NOTE only ONE core carries a trace flow (trap 7), so this is the per-invocation
distribution of ONE core, not a per-core comparison.
Run under the IRON env -- it imports aie.utils.trace.
"""
import statistics
from aie.utils.trace.utils import get_cycles_summary

A = "experiments/m5-pretiled-gemm/artifacts"
FILES = {
    "rowmajor": f"{A}/trace_rowmajor_4c_bf16_f32_bfp16_8192x1536x384_t64x64x48.json",
    "pretiled": f"{A}/trace_pretiled_kn_st_4c_bf16_f32_bfp16_8192x1536x384_t64x64x48.json",
}

def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))]

for kind, path in FILES.items():
    d = []
    for entry in get_cycles_summary(path):
        d += [x for x in entry[1:] if x is not None]
    tot = sum(d)
    med = statistics.median(d)
    thr = 1.5 * med                      # "stalled" = 1.5x the typical iteration
    out = [x for x in d if x > thr]
    idx = [i for i, x in enumerate(d) if x > thr]
    print(f"\n=== {kind}   n={len(d)}  total={tot} cyc")
    print(f"  min {min(d)}  p10 {pct(d,10)}  median {med:.0f}  p90 {pct(d,90)}  "
          f"p99 {pct(d,99)}  max {max(d)}")
    print(f"  mean {tot/len(d):.1f}  (mean/median = {tot/len(d)/med:.3f})")
    print(f"  > {thr:.0f} cyc: {len(out)} of {len(d)} ({100*len(out)/len(d):.1f}%), "
          f"carrying {100*sum(out)/tot:.1f}% of all cycles")
    if out:
        excess = sum(x - med for x in out)
        print(f"  excess over median in those {len(out)}: {excess:.0f} cyc "
              f"= {100*excess/tot:.1f}% of the run")
        print(f"  largest: {sorted(out, reverse=True)[:8]}")
    print(f"  outlier positions (first 25 of {len(idx)}): {idx[:25]}")
    if len(idx) > 1:
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        print(f"  gaps between outliers: min {min(gaps)} median "
              f"{statistics.median(gaps):.0f} max {max(gaps)}")
    # what would the run look like with the tail removed?
    clean = [x for x in d if x <= thr]
    print(f"  if the tail were removed: mean {sum(clean)/len(clean):.1f} cyc "
          f"-> {(64*64*48)/(sum(clean)/len(clean)):.1f} MACs/cyc/core")
