#!/usr/bin/env python3
"""What the roofline prices, once a design is known to sit on the slanted roof.

Reads roofline.json. On a bandwidth-bound dispatch, array time is proportional
to DDR bytes and NOTHING ELSE -- so every lever's value is a byte ratio, and it
can be computed without touching hardware. That is only valid to the left of
the ridge; sec 3 of the TASK.md establishes that all five bfp16 models are
5-7x left of theirs, and this script refuses to price a compute-bound row.

  python tasks/0120-roofline-analytic/levers.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "roofline.json"), encoding="utf-8"))

# tasks/0109, `--bench` `wait (hardware)` as a share of dispatch+wait on the
# FUSED build. The array-side ceiling multiplies by this, not by 1.
ARRAY_SHARE = {
    "minilm_bfp16": 0.334, "small_bf16": 0.486, "base_bfp16": 0.467,
    "large_bfp16": 0.567, "nomic_bfp16": 0.510,
}
ROW_BLOCKS = 8192 / (64 * 4)          # 32, identical for every shipped design


def amdahl(array_gain, share):
    return 1.0 / ((1.0 - share) + share / array_gain)


by_model = {}
for r in D["dispatches"]:
    by_model.setdefault(r["model"], []).append(r)

print("Lever 1 -- B streamed ONCE instead of once per row block (B-reuse).")
print("Channel-blocked at 8 columns (tasks/0046); this is the PRICE, not a plan.\n")
hdr = ("%-14s%-12s%10s%10s%8s%8s%9s%10s"
       % ("model", "datapath", "MB now", "MB reuse", "AI now", "AI new",
          "array x", "e2e x"))
print(hdr)
print("-" * len(hdr))
for model, rows in sorted(by_model.items()):
    now = sum(r["traffic_mb"] for r in rows)
    # B is replicated ROW_BLOCKS times today; reuse leaves one copy.
    new = sum(r["traffic_mb"] - r["traffic_b_mb"] * (ROW_BLOCKS - 1) / ROW_BLOCKS
              for r in rows)
    ai_now = sum(2 * r["gmac"] for r in rows) * 1e3 / now
    ai_new = sum(2 * r["gmac"] for r in rows) * 1e3 / new
    dp = rows[0]["datapath"]
    if dp == "vector_fp32":
        gain, e2e = 1.0, 1.0            # compute-bound: bytes are free
        note = "  (compute-bound -- roofline prices this at 1.00x)"
    else:
        gain = now / new
        share = ARRAY_SHARE.get(model)
        e2e = amdahl(gain, share) if share else None
        # embeddinggemma-300m has no `--bench` mode, so tasks/0109 could not
        # measure its array share. Left blank rather than borrowed from a
        # model with a different geometry.
        note = "" if share else "  (no array share measured -- 0109 sec 2)"
    print("%-14s%-12s%10.0f%10.0f%8.1f%8.1f%9.3f%10s%s"
          % (model, dp, now, new, ai_now, ai_new, gain,
             "%.3f" % e2e if e2e else "n/a", note))

print("\n\nWhere the bytes are, per model (share of total DDR traffic):\n")
hdr = "%-14s%8s%8s%8s%10s" % ("model", "A", "B", "C", "total MB")
print(hdr)
print("-" * len(hdr))
for model, rows in sorted(by_model.items()):
    t = sum(r["traffic_mb"] for r in rows)
    a = sum(r["traffic_a_mb"] for r in rows)
    b = sum(r["traffic_b_mb"] for r in rows)
    c = sum(r["traffic_c_mb"] for r in rows)
    print("%-14s%7.0f%%%7.0f%%%7.0f%%%10.0f"
          % (model, 100 * a / t, 100 * b / t, 100 * c / t, t))

print("\n\nLever 2 -- C narrowed to bf16 on bge-small, the one model still")
print("shipping fp32 C. It is on the FLAT roof, so the roofline prices it at")
print("1.00x -- and tasks/0045 measured exactly that (~0% of array time).\n")
rows = by_model["small_bf16"]
now = sum(r["traffic_mb"] for r in rows)
new = sum(r["traffic_mb"] - r["traffic_c_mb"] / 2 for r in rows)
print("  bge-small traffic %.0f MB -> %.0f MB (%.0f%% fewer bytes), "
      "predicted array gain 1.00x" % (now, new, 100 * (1 - new / now)))

print("\n\nLever 3 -- tile_n on bge-large (A re-streams N/(tile_n*cols) times).")
print("bge-large ships tile_n=32 and has the highest A share of any model.\n")
rows = by_model["large_bfp16"]
now = sum(r["traffic_mb"] for r in rows)
for n in (48, 64):
    ok_div = all(r["N"] % (n * 8) == 0 for r in rows)
    # trap 3, in the form this design actually occupies L1 with: A and B
    # double-buffered, an fp32 accumulator Buffer single-buffered, and the
    # narrowed bf16 C tile double-buffered.
    m, k = 64, 64
    l1 = 2 * m * k * 2 + 2 * k * n * 2 + m * n * 4 + 2 * m * n * 2
    new = sum(r["traffic_mb"] - r["traffic_a_mb"]
              + r["traffic_a_mb"] * (32.0 / n) * (1 if r["N"] >= n * 8 else 0)
              for r in rows)
    print("  tile_n=%d: N divisible by n*cols=%d? %-5s | L1 %d B of 64512 %-8s "
          "| traffic %.0f -> %.0f MB (%.2fx)"
          % (n, n * 8, ok_div, l1, "OK" if l1 < 64512 else "OVER BUDGET",
             now, new, now / new))
print("\n  Both are refused, and by DIFFERENT constraints -- 48 by geometry")
print("  (1024 mod 384 != 0), 64 by the L1 budget. bf16 bge-large is stuck at 32.")
