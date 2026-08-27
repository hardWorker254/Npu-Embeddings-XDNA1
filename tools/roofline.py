#!/usr/bin/env python3
"""Analytic roofline for the shipped GEMM designs (tasks/0120).

Build-time analysis only -- reads `design.json` files and prints/plots where
every production dispatch sits against this chip's compute and bandwidth
ceilings. Runs no hardware and measures nothing: every x-coordinate is pure
arithmetic over geometry the design file already records, which is what makes
the x-axis immune to rule 1 (wall clock is never an NPU performance claim).

Measured points come in through --measured, each carrying its own provenance
string, and are plotted as a SEPARATE series so a reader can never mistake a
prediction for a measurement.

  python tools/roofline.py --designs "dist/npuembeddings-0.4.0/artifacts_*/gemm_rtp" \
      --measured tasks/0120-roofline-analytic/measured.json \
      --out tasks/0120-roofline-analytic/roofline

WHY THE TRAFFIC FORMULA LOOKS LIKE THIS. tasks/0010 fitted, and tasks/0048 and
tasks/0080 both re-used, this accounting of DDR bytes per dispatch:

    A is re-streamed once per n-block group   -> max(1, N / (tile_n * cols))
    B is re-streamed once per row block       -> M / (tile_m * 4 rows)
    C is written once

B reuse would remove the second factor. It does not build at 8 columns and is
not a modelling choice we are free to make -- tasks/0046 established that every
core tile is at 2/2 input DMA channels and five of eight mem tiles at 6/6, so
there is no channel to give B. The formula describes the design that ships.

TWO CORRECTIONS AGAINST THE RUNTIME'S OWN `--probe-streams` (main.cpp:6481).
That code computes the same quantity with `2` hardcoded as the A/B element size
and `48.0 * 8.0` hardcoded as tile_n*cols. Both are properties of a design, not
constants, and both are wrong for designs we ship -- see tasks/0120 sec 4.
"""
import argparse
import glob
import json
import math
import os

# -- machine constants -----------------------------------------------------
# docs/01-hardware/README.md. The clock is MEASURED (1.808 GHz on a Krackan
# SKU), not the marketing figure; the MACs/cycle column is AMD's for aie2p.
CLOCK_HZ = 1.808e9
N_CORES = 32
N_AIE_ROWS = 4                      # gemm_pretiled.py:99, fixed by the design

# Compute ceilings, MACs/cycle/core. THREE, not one -- which applies is a
# per-design property (`emulate_bfp16` / `a_dtype`), not a property of the chip.
#
#   vector_fp32  plain bf16 rides the fp32 VECTOR unit and the MMAC sits idle;
#                32 is its hard limit, and tasks/0049 traced the core spending
#                78% of every k-block iteration executing back-to-back vmac.f
#                against it.
#   mmac_bf16    --emulate-bf16-mmul-with-bfp16 upgrades the aie::mmul geometry
#                to 8x8x8 and reaches the MMAC. Native bfp16 (512) needs the
#                Chess compiler, which has no Windows build.
#   mmac_int8    native (8,8,8) int8.
PEAK_MACS_PER_CYCLE = {
    "vector_fp32": 32,
    "mmac_bf16": 256,
    "mmac_int8": 512,
}

# The same ceilings as this repo has actually MEASURED them, per core, in
# isolation (tasks/0049 sec 1, same script/toolchain/shape, reproduced). These
# are the honest ceilings for a design that ships: the difference from the
# table above is microkernel quality, not machine capability.
MEASURED_MACS_PER_CYCLE = {
    "vector_fp32": 28.9,            # plain bf16, traced
    "mmac_bf16": 146.7,             # --emulate-bfp16, traced
    "mmac_int8": None,              # never traced in isolation
}

# Off-chip bandwidth. NOT a measurement of one thing -- four numbers from four
# methods, and the spread IS the finding. Plotted as a band rather than a line
# because pretending to one value would be the dishonest move.
# docs/01-hardware/README.md, "Memory bandwidth".
BANDWIDTH_GBS = [
    (60.0, "docs band, upper (Zen-Attention, AMD: ~60 GB/s read+write)"),
    (40.0, "docs band, lower (Gemma3 team on Krackan: below 40 GB/s)"),
    (33.0, "tasks/0010 marginal fit, R2 0.902 -- superseded for production"),
    (28.0, "tasks/0080 int8 marginal fit, R2 0.987"),
]


def datapath_of(design):
    """Which compute ceiling this design's dispatches run under."""
    if design.get("a_dtype") == "i8":
        return "mmac_int8"
    return "mmac_bf16" if design.get("emulate_bfp16") else "vector_fp32"


def elem_bytes(dtype):
    return {"i8": 1, "bf16": 2, "f32": 4}[dtype]


def traffic_bytes(M, K, N, a_bytes, c_bytes, tile_m, tile_n, cols,
                  leg="dram"):
    """Bytes for one dispatch on the named memory leg (T46, tasks/0139).

    leg="dram" -- tasks/0010's accounting, generalised in exactly two places
    against the runtime's old hardcoded copy: `a_bytes` (1 under int8) and
    `tile_n * cols` (256 for bf16 bge-large, 512 for int8 bge-large, not the
    hardcoded 384).

    leg="memtile" -- the L2->L1 leg's aggregate bytes, which differ from the
    DRAM leg in exactly one term: **B is broadcast from each column's mem
    tile to its 4 rows**, so the mem tiles emit 4x B's DRAM bytes. A is a
    split (each row receives its own m-slice; aggregate equals DRAM) and C
    is a join (aggregate equals DRAM). The number is the ARRAY-WIDE sum
    across all `cols` mem tiles; divide by `cols` for a per-tile figure.
    Measured context (0139, from 0130's stored trace): the traced core's
    input ports are covered only 30.9% + 3.7% of the steady-state span
    while LOCK_STALL covers 52.8% -- the L2->L1 leg has headroom and the
    DRAM leg binds, which is why the roofline's default leg stays "dram".
    """
    nb_groups = max(1.0, N / (tile_n * cols))
    row_blocks = M / (tile_m * N_AIE_ROWS)
    a = M * K * a_bytes * nb_groups
    b = K * N * a_bytes * row_blocks
    c = M * N * c_bytes
    if leg == "memtile":
        b *= N_AIE_ROWS
    elif leg != "dram":
        raise ValueError(f"unknown leg {leg!r}")
    return dict(a=a, b=b, c=c, total=a + b + c)


def analyse(design, label):
    dp = datapath_of(design)
    a_bytes = elem_bytes(design["a_dtype"])
    c_bytes = elem_bytes(design["c_dtype"])
    tile_m, tile_n = design["tile"]["m"], design["tile"]["n"]
    cols = design["cols"]
    peak = PEAK_MACS_PER_CYCLE[dp] * N_CORES * CLOCK_HZ          # MAC/s
    meas = MEASURED_MACS_PER_CYCLE[dp]
    peak_meas = meas * N_CORES * CLOCK_HZ if meas else None

    rows = []
    top_batch = design["batch"]
    for st in design["streams"]:
        if st["batch"] != top_batch:
            continue                       # one point per shape, largest tier
        M, K, N = st["M"], st["K"], st["N"]
        tr = traffic_bytes(M, K, N, a_bytes, c_bytes, tile_m, tile_n, cols)
        macs = float(M) * K * N
        ai = 2.0 * macs / tr["total"]                            # FLOP/byte
        rows.append(dict(
            model=label, op=st["op"], M=M, K=K, N=N,
            datapath=dp, a_dtype=design["a_dtype"], c_dtype=design["c_dtype"],
            tile_n=tile_n, cols=cols,
            gmac=macs / 1e9, traffic_mb=tr["total"] / 1e6,
            traffic_a_mb=tr["a"] / 1e6, traffic_b_mb=tr["b"] / 1e6,
            traffic_c_mb=tr["c"] / 1e6,
            ai_flop_per_byte=ai,
            peak_gmac_s=peak / 1e9,
            peak_measured_gmac_s=peak_meas / 1e9 if peak_meas else None,
            # Where the two ceilings cross for THIS dispatch, at each candidate
            # bandwidth: below the ridge the dispatch is bandwidth-bound.
            ridge_ai={"%g" % bw: 2.0 * peak / (bw * 1e9)
                      for bw, _ in BANDWIDTH_GBS},
            bw_bound_gmac_s={"%g" % bw: ai * bw * 1e9 / 2.0 / 1e9
                             for bw, _ in BANDWIDTH_GBS},
        ))
    for r in rows:
        r["bound_by"] = {
            k: ("compute" if r["peak_gmac_s"] <= v else "bandwidth")
            for k, v in r["bw_bound_gmac_s"].items()}
    return rows


# -- SVG (no matplotlib in any env this repo owns; see CLAUDE.md) -----------

W, H = 980, 640
PAD_L, PAD_R, PAD_T, PAD_B = 78, 252, 50, 62
# Axis window. Chosen to contain the two ridge points that the analysis turns
# on (741 for bfp16, 1481 for int8 at 40 GB/s) as well as every dispatch, which
# all sit between AI 85 and 256 -- a wider window buries the whole catalogue in
# a fifth of the plot.
X0, X1 = 32.0, 2000.0           # FLOP/byte
Y0, Y1 = 500.0, 40000.0         # GMAC/s
X_TICKS = (50, 100, 200, 500, 1000, 2000)
Y_TICKS = ((500, "500"), (1000, "1 000"), (5000, "5 000"),
           (10000, "10 000"), (30000, "30 000"))


def sx(v):
    return PAD_L + (math.log10(v) - math.log10(X0)) / (
        math.log10(X1) - math.log10(X0)) * (W - PAD_L - PAD_R)


def sy(v):
    return H - PAD_B - (math.log10(v) - math.log10(Y0)) / (
        math.log10(Y1) - math.log10(Y0)) * (H - PAD_T - PAD_B)


PALETTE = {
    "vector_fp32": "#c2410c",
    "mmac_bf16": "#1d4ed8",
    "mmac_int8": "#047857",
}
DP_LABEL = {
    "vector_fp32": "plain bf16 (fp32 vector unit)",
    "mmac_bf16": "bf16 via bfp16 emulation (MMAC)",
    "mmac_int8": "int8 (MMAC)",
}


def svg(rows, measured, path):
    o = []
    o.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d" font-family="Segoe UI,Helvetica,Arial,sans-serif">'
             % (W, H, W, H))
    o.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))

    for d in range(0, 5):
        for m in range(1, 10):
            v = m * 10 ** d
            if X0 <= v <= X1:
                x = sx(v)
                o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                         'stroke="#eef1f5" stroke-width="1"/>'
                         % (x, PAD_T, x, H - PAD_B))
    for d in range(2, 6):
        for m in range(1, 10):
            v = m * 10 ** d
            if Y0 <= v <= Y1:
                y = sy(v)
                o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                         'stroke="#eef1f5" stroke-width="1"/>'
                         % (PAD_L, y, W - PAD_R, y))
    o.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" '
             'stroke="#94a3b8"/>'
             % (PAD_L, PAD_T, W - PAD_L - PAD_R, H - PAD_T - PAD_B))
    for v in X_TICKS:
        o.append('<text x="%.1f" y="%d" font-size="11" text-anchor="middle" '
                 'fill="#475569">%d</text>' % (sx(v), H - PAD_B + 18, v))
    for v, lab in Y_TICKS:
        o.append('<text x="%d" y="%.1f" font-size="11" text-anchor="end" '
                 'fill="#475569">%s</text>' % (PAD_L - 8, sy(v) + 4, lab))
    o.append('<text x="%d" y="%d" font-size="12.5" text-anchor="middle" '
             'fill="#0f172a">arithmetic intensity (FLOP per DDR byte, '
             'tasks/0010 accounting)</text>'
             % ((PAD_L + W - PAD_R) / 2, H - 16))
    cy = (PAD_T + H - PAD_B) / 2
    o.append('<text x="18" y="%d" font-size="12.5" fill="#0f172a" '
             'transform="rotate(-90 18 %d)" text-anchor="middle">'
             'array throughput (GMAC/s)</text>' % (cy, cy))
    o.append('<text x="%d" y="%d" font-size="15" font-weight="600" '
             'fill="#0f172a">XDNA2 roofline &#8212; the 24 production GEMM '
             'dispatches</text>' % (PAD_L, PAD_T - 26))
    o.append('<text x="%d" y="%d" font-size="11" fill="#64748b">three compute '
             'ceilings, one per datapath; bandwidth band 28&#8211;60 GB/s. '
             'Circles are analytic (geometry only), squares measured.</text>'
             % (PAD_L, PAD_T - 10))

    for bw, _note in BANDWIDTH_GBS:
        y_lo = X0 * bw * 1e9 / 2.0 / 1e9
        y_hi = X1 * bw * 1e9 / 2.0 / 1e9
        dash = "" if bw in (40.0, 60.0) else ' stroke-dasharray="3 3"'
        col = "#0f172a" if bw in (40.0, 60.0) else "#94a3b8"
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                 'stroke-width="1.2" opacity="0.75"%s/>'
                 % (sx(X0), sy(max(y_lo, Y0)), sx(X1), sy(min(y_hi, Y1)),
                    col, dash))
        if y_lo > Y0:
            o.append('<text x="%.1f" y="%.1f" font-size="9.5" fill="%s">'
                     '%g GB/s</text>' % (sx(X0) + 4, sy(y_lo) - 4, col, bw))

    for dp, mpc in PEAK_MACS_PER_CYCLE.items():
        gm = mpc * N_CORES * CLOCK_HZ / 1e9
        y = sy(gm)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="2"/>' % (PAD_L, y, W - PAD_R, y, PALETTE[dp]))
        o.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end" '
                 'fill="%s">%s &#8212; %s GMAC/s</text>'
                 % (W - PAD_R - 4, y - 5, PALETTE[dp], DP_LABEL[dp],
                    format(int(round(gm)), ",d").replace(",", " ")))
        mm = MEASURED_MACS_PER_CYCLE[dp]
        if mm:
            y2 = sy(mm * N_CORES * CLOCK_HZ / 1e9)
            o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                     'stroke-width="1.4" stroke-dasharray="6 4" opacity="0.8"/>'
                     % (PAD_L, y2, W - PAD_R, y2, PALETTE[dp]))
            o.append('<text x="%d" y="%.1f" font-size="9.5" text-anchor="end" '
                     'fill="%s" opacity="0.9">microkernel traced in isolation, '
                     '%s MAC/cyc/core</text>'
                     % (W - PAD_R - 4, y2 - 5, PALETTE[dp], mm))

    for r in rows:
        dp = r["datapath"]
        x = sx(r["ai_flop_per_byte"])
        # The roofline PREDICTION: min(compute ceiling, AI x BW / 2), taken at
        # the documented band's LOWER edge (40 GB/s). A stated constant, not a
        # fit -- so a measured square sitting above a circle means the chip beat
        # the conservative bandwidth assumption, and one sitting below means
        # something the roofline does not model is binding.
        y = sy(min(r["peak_gmac_s"], r["bw_bound_gmac_s"]["40"]))
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                 'stroke-width="0.6" opacity="0.16"/>'
                 % (x, y, x, H - PAD_B, PALETTE[dp]))
        o.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="%s" '
                 'fill-opacity="0.30" stroke="%s" stroke-width="1"/>'
                 % (x, y, PALETTE[dp], PALETTE[dp]))

    for m in measured:
        dp = m["datapath"]
        x, y = sx(m["ai_flop_per_byte"]), sy(m["gmac_s"])
        o.append('<rect x="%.1f" y="%.1f" width="8" height="8" fill="%s" '
                 'stroke="#ffffff" stroke-width="1.1"/>'
                 % (x - 4, y - 4, PALETTE[dp]))
        if m.get("annotate"):
            o.append('<text x="%.1f" y="%.1f" font-size="9.5" fill="%s">%s</text>'
                     % (x + 8, y + 3.5, PALETTE[dp], m["annotate"]))

    lx, ly = W - PAD_R + 14, PAD_T + 10
    o.append('<text x="%d" y="%d" font-size="11" font-weight="600" '
             'fill="#0f172a">datapath</text>' % (lx, ly))
    ly += 16
    for dp in ("vector_fp32", "mmac_bf16", "mmac_int8"):
        o.append('<rect x="%d" y="%d" width="10" height="10" fill="%s" '
                 'fill-opacity="0.35" stroke="%s"/>'
                 % (lx, ly - 8, PALETTE[dp], PALETTE[dp]))
        o.append('<text x="%d" y="%d" font-size="9.5" fill="#334155">%s</text>'
                 % (lx + 15, ly, DP_LABEL[dp]))
        ly += 15
    ly += 12
    o.append('<text x="%d" y="%d" font-size="11" font-weight="600" '
             'fill="#0f172a">marks</text>' % (lx, ly))
    ly += 15
    o.append('<circle cx="%d" cy="%d" r="3.4" fill="#64748b" '
             'fill-opacity="0.3" stroke="#64748b"/>' % (lx + 5, ly - 4))
    o.append('<text x="%d" y="%d" font-size="9.5" fill="#334155">analytic '
             '(AI from geometry)</text>' % (lx + 15, ly))
    ly += 15
    o.append('<rect x="%d" y="%d" width="8" height="8" fill="#64748b"/>'
             % (lx + 1, ly - 8))
    o.append('<text x="%d" y="%d" font-size="9.5" fill="#334155">measured '
             '(measured.json)</text>' % (lx + 15, ly))
    ly += 26
    for line in ("Circles are the roofline prediction:",
                 "min(compute ceiling, AI x 40 GB/s / 2).",
                 "40 is the documented band's lower edge,",
                 "a stated constant and not a fit."):
        o.append('<text x="%d" y="%d" font-size="9" fill="#64748b">%s</text>'
                 % (lx, ly, line))
        ly += 12

    o.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True,
                    help="gemm_rtp directories, design.json paths, or globs")
    ap.add_argument("--measured", default=None)
    ap.add_argument("--out", default="roofline")
    args = ap.parse_args()

    paths = []
    for pat in args.designs:
        for p in (sorted(glob.glob(pat)) or [pat]):
            paths.append(p if p.endswith(".json")
                         else os.path.join(p, "design.json"))

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        label = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rows += analyse(d, label.replace("artifacts_", ""))

    measured = []
    if args.measured:
        with open(args.measured, encoding="utf-8") as f:
            spec = json.load(f)
        for src in spec["points"]:
            m = dict(src)
            if m.get("kind") == "aggregate":
                # One point for a whole layer: the four shapes summed. Time is
                # `--bench`'s per-dispatch `wait (hardware)`, which averages
                # over exactly those four, so total time is n_shapes * us. The
                # aggregate AI is total FLOP / total bytes, NOT a mean of the
                # four AIs -- the latter would weight a small shape like a
                # large one.
                with open(m["design"], encoding="utf-8") as g:
                    dj = json.load(g)
                sub = analyse(dj, "agg")
                macs = sum(r["gmac"] for r in sub) * 1e9
                tot = sum(r["traffic_mb"] for r in sub) * 1e6
                m["op"] = "layer (%d shapes)" % len(sub)
                m["datapath"] = sub[0]["datapath"]
                m["us"] = m["us_per_dispatch"] * len(sub)
                tr = dict(total=tot)
            else:
                tr = traffic_bytes(m["M"], m["K"], m["N"],
                                   elem_bytes(m["a_dtype"]),
                                   elem_bytes(m["c_dtype"]),
                                   m.get("tile_m", 64), m["tile_n"], m["cols"])
                macs = float(m["M"]) * m["K"] * m["N"]
            m["traffic_mb"] = tr["total"] / 1e6
            m["ai_flop_per_byte"] = 2.0 * macs / tr["total"]
            m["gmac_s"] = macs / (m["us"] * 1e-6) / 1e9
            m["frac_of_peak"] = m["gmac_s"] / (
                PEAK_MACS_PER_CYCLE[m["datapath"]] * N_CORES * CLOCK_HZ / 1e9)
            m["implied_gbs"] = tr["total"] / (m["us"] * 1e-6) / 1e9
            measured.append(m)

    hdr = ("%-13s%-10s%-13s%6s%6s%6s%8s%8s%8s%10s%11s"
           % ("model", "op", "datapath", "M", "K", "N", "GMAC", "MB", "AI",
              "ridge@40", "bound@40"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-13s%-10s%-13s%6d%6d%6d%8.2f%8.1f%8.1f%10.0f%11s"
              % (r["model"], r["op"], r["datapath"], r["M"], r["K"], r["N"],
                 r["gmac"], r["traffic_mb"], r["ai_flop_per_byte"],
                 r["ridge_ai"]["40"], r["bound_by"]["40"]))

    if measured:
        print()
        hdr2 = ("%-24s%-19s%-13s%8s%9s%8s%8s%8s"
                % ("source", "op", "datapath", "us", "GMAC/s", "AI", "%peak",
                   "GB/s"))
        print(hdr2)
        print("-" * len(hdr2))
        for m in measured:
            op = m["op"]
            if m.get("kind") == "aggregate":
                op = "%s, %s" % (m.get("annotate", "?"), op.split(" ")[0])
            print("%-24s%-19s%-13s%8.0f%9.0f%8.1f%7.1f%%%8.1f"
                  % (m["source"][:23], op[:18], m["datapath"], m["us"],
                     m["gmac_s"], m["ai_flop_per_byte"],
                     100 * m["frac_of_peak"], m["implied_gbs"]))

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(dict(
            constants=dict(clock_hz=CLOCK_HZ, cores=N_CORES,
                           peak_macs_per_cycle=PEAK_MACS_PER_CYCLE,
                           measured_macs_per_cycle=MEASURED_MACS_PER_CYCLE,
                           bandwidth_gbs=[list(b) for b in BANDWIDTH_GBS]),
            designs=paths, dispatches=rows, measured=measured), f, indent=2)
    svg(rows, measured, args.out + ".svg")
    print("\nwrote %s.json and %s.svg (%d dispatches, %d measured)"
          % (args.out, args.out, len(rows), len(measured)))


if __name__ == "__main__":
    main()
