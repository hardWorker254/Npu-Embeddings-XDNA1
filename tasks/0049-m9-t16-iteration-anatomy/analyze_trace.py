# T16: decompose a traced GEMM core's time into kernel windows, stalls and gaps.
#
# Reads the Perfetto-style trace.json parse.py emits. All events are B/E pairs
# per thread (tid). INSTR_EVENT_0/1 are *instants* marking kernel entry/exit
# (parse.py emits them as B/E pairs on their own tids); the kernel window is
# INSTR_EVENT_0 start -> INSTR_EVENT_1 start. Everything else (INSTR_VECTOR,
# MEMORY_STALL, STREAM_STALL, LOCK_STALL, PORT_RUNNING_*) are duration events
# whose overlap with each window we integrate.
import json, sys
from collections import defaultdict

path = sys.argv[1]
evs = json.load(open(path))

# collect intervals per event name
tid_name = {}
for e in evs:
    if e.get('name') == 'thread_name':
        tid_name[(e['pid'], e['tid'])] = e['args']['name']

open_ev = {}
intervals = defaultdict(list)   # name -> [(t0, t1)]
for e in evs:
    ph = e.get('ph')
    if ph not in ('B', 'E'):
        continue
    key = (e['pid'], e['tid'])
    name = tid_name.get(key, e.get('name'))
    if ph == 'B':
        open_ev[key] = e['ts']
    else:
        t0 = open_ev.pop(key, None)
        if t0 is not None:
            intervals[name].append((t0, e['ts']))

for name, iv in sorted(intervals.items()):
    tot = sum(b - a for a, b in iv)
    print(f"{name:16s} n={len(iv):6d} total={tot:10d} cyc")

e0 = sorted(intervals['INSTR_EVENT_0'])
e1 = sorted(intervals['INSTR_EVENT_1'])
# kernel windows: pair each EVENT_0 start with the next EVENT_1 start
starts = [a for a, b in e0]
ends = [a for a, b in e1]
wins = []
j = 0
for s in starts:
    while j < len(ends) and ends[j] <= s:
        j += 1
    if j < len(ends):
        wins.append((s, ends[j]))
        j += 1

def overlap(iv, s, t):
    return sum(max(0, min(b, t) - max(a, s)) for a, b in iv)

print(f"\nkernel windows: {len(wins)}")
rows = []
for s, t in wins:
    d = t - s
    rows.append(dict(
        dur=d,
        vec=overlap(intervals.get('INSTR_VECTOR', []), s, t),
        mem=overlap(intervals.get('MEMORY_STALL', []), s, t),
        lock=overlap(intervals.get('LOCK_STALL', []), s, t),
        stream=overlap(intervals.get('STREAM_STALL', []), s, t),
    ))

import statistics as st
def col(k): return [r[k] for r in rows]
for k in ('dur', 'vec', 'mem', 'lock', 'stream'):
    v = col(k)
    print(f"in-window {k:7s} mean={st.mean(v):8.1f} min={min(v):6d} max={max(v):6d} median={st.median(v):8.1f}")

# gaps between consecutive windows
gaps = [wins[i+1][0] - wins[i][1] for i in range(len(wins)-1)]
if gaps:
    print(f"\ngaps: n={len(gaps)} mean={st.mean(gaps):8.1f} min={min(gaps)} max={max(gaps)} median={st.median(gaps):8.1f}")
    for s_, t_ in [(wins[i][1], wins[i+1][0]) for i in range(len(wins)-1)][:0]:
        pass
    gl = [overlap(intervals.get('LOCK_STALL', []), wins[i][1], wins[i+1][0]) for i in range(len(wins)-1)]
    print(f"gap lock-stall total={sum(gl)} of {sum(gaps)}")

# histogram of window durations
from collections import Counter
buck = Counter((r['dur'] // 500) * 500 for r in rows)
print("\nwindow duration histogram (500-cyc buckets):")
for b in sorted(buck):
    print(f"  {b:6d}-{b+499:6d}: {buck[b]:4d} {'#'*buck[b]}")
