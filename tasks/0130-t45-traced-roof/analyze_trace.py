"""T45 item 1: derive the dispatch's hardware-timebase duration and implied
DRAM bandwidth from the trace's own timeline.

The trace buffer (262144 B) captures ~316 of the 6144 kernel invocations on
the traced core -- a steady-state window, not the whole dispatch. What it
yields exactly is the *pitch*: the interval between consecutive kernel
starts (INSTR_EVENT_0 rising edges) on the 1.808 GHz core clock. The design
is symmetric (every core runs the same schedule), so
    dispatch ~= invocations_per_core x mean_pitch.
Validated against wall clock to 1.5% (see TASK.md).
Run from the repo root: python tasks/0130-t45-traced-roof/analyze_trace.py
"""
import json
import statistics as st

ev = json.load(open("tasks/0130-t45-traced-roof/trace.json"))
if isinstance(ev, dict):
    ev = ev["traceEvents"]

k0 = [e["ts"] for e in ev if e.get("name") == "INSTR_EVENT_0"
      and e.get("ph") == "B"]
pitch = [b - a for a, b in zip(k0, k0[1:])]
p = sorted(pitch[5:-5])          # steady state
med, mean = st.median(p), st.mean(p)
print(f"kernel starts: {len(k0)}  pitch median {med:.0f} cyc  "
      f"mean {mean:.1f}  p10 {p[len(p)//10]}  p90 {p[9*len(p)//10]}")

CLK = 1.808e9
M, K, N, cols = 8192, 768, 3072, 4
m, k, n = 64, 64, 48
inv_per_core = (M // (m * 4)) * (N // (n * cols)) * (K // k)
disp_s = inv_per_core * mean / CLK
nb_groups = N / (n * cols)
row_blocks = M / m / 4
traffic = M * K * 2 * nb_groups + K * N * 2 * row_blocks + M * N * 4
print(f"invocations/core {inv_per_core}  window avg 1658.4 cyc (get_cycles_summary)")
print(f"duty cycle {1658.4 / mean * 100:.1f}%   effective {m*k*n/mean:.1f} MACs/cyc/core")
print(f"implied dispatch {disp_s*1e3:.2f} ms (hardware timebase)")
print(f"traffic {traffic/1e6:.1f} MB  ->  {traffic/1e9/disp_s:.1f} GB/s")
