#!/usr/bin/env python3
# NpuEmbeddings -- what a quantised pose container does to the DETECTIONS.
# SPDX-License-Identifier: Apache-2.0
#
# WHY A SEPARATE TOOL AND NOT A FLAG ON verify_pose.py
# -----------------------------------------------------
# verify_pose.py's question is "is the runtime's network the checkpoint's
# network": it runs an independent NumPy ONNX interpreter node by node and finds
# the first divergence. That question has a binary answer and this one does not.
#
# A quantised container is not wrong. It is a different network that is useful
# at some size/accuracy points and useless at others, and the operator picks the
# point -- so this tool has no threshold and never exits non-zero on accuracy.
# What it refuses to do is the thing a threshold invites: pick a number, call it
# a requirement, and then let `--conf` or the image set move it. It prints the
# deltas and the container sizes, which is the whole trade, and says plainly that
# nobody has decided where the line is.
#
# It measures the RUNTIME's detections through the CLI's own JSON, not a tensor.
# A relative Frobenius error on the weights (which the packer prints) is the
# input to this, not a proxy for it: 3% on a C2f bottleneck and 3% on the head's
# 1x1 convolutions are very different numbers of people in a picture, and the
# only one an operator cares about is the second.
#
# MATCHING, AND WHY IT IS NOT NEAREST-NEAREST
# -------------------------------------------
# Detections are matched greedily by box IoU, best pair first, each detection
# used once. Sorted by IoU rather than by index, because the two runs' order is
# the score order and quantisation reorders scores: a fixed index would pair
# person 0 with whichever quantised box took the top slot and report a two-metre
# error for a detector that found exactly the same three people.
#
# An UNMATCHED detection is reported as such and never scored. A quantised run
# that invents a fifth person and loses one of the three is a worse outcome than
# one that shifts every box by a pixel, and a mean over the matched subset hides
# it completely.
#
# Env: python3 stdlib only -- it shells out to the CLI and reads JSON.
#
# Usage:
#   python3 tools/verify/verify_pose_quant.py \
#       --ref models/pose_f32.npue --test models/pose_i8.npue \
#       images/bus.jpg images/person.jpg

import argparse
import json
import math
import os
import subprocess
import sys


def run(cli, container, image, extra):
    cmd = [cli, "pose", container, image, "--json"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(
            f"the runtime failed on {os.path.basename(container)} / "
            f"{os.path.basename(image)}:\n{p.stderr.strip()}\n"
            f"  (command: {' '.join(cmd)})")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"{os.path.basename(container)} / {os.path.basename(image)}: the "
            f"CLI's stdout is not JSON ({e}). The comparison is against the "
            f"JSON emitter on purpose -- the --text summary rounds to one "
            f"decimal, which is coarser than every delta this tool reports.")


def box(d):
    b = d["box"]
    return (b["x1"], b["y1"], b["x2"], b["y2"])


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def match(ref, test, thr):
    """Greedy IoU matching. Returns (pairs, only_ref, only_test)."""
    pairs = []
    cand = []
    for i, r in enumerate(ref):
        for j, t in enumerate(test):
            v = iou(box(r), box(t))
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(key=lambda c: -c[0])
    used_r, used_t = set(), set()
    for v, i, j in cand:
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        pairs.append((i, j, v))
    pairs.sort()
    only_r = [i for i in range(len(ref)) if i not in used_r]
    only_t = [j for j in range(len(test)) if j not in used_t]
    return pairs, only_r, only_t


def corner_delta(a, b):
    return max(abs(x - y) for x, y in zip(box(a), box(b)))


def kpt_delta(a, b):
    ka, kb = a["keypoints"], b["keypoints"]
    if len(ka) != len(kb):
        return float("inf")
    if not ka:
        return 0.0
    return max(math.hypot(p["x"] - q["x"], p["y"] - q["y"])
               for p, q in zip(ka, kb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, metavar="NPUE",
                    help="the fp32 container, the reference")
    ap.add_argument("--test", required=True, metavar="NPUE",
                    help="the quantised container")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--iou-match", type=float, default=0.50,
                    help="box IoU above which two detections are the same "
                         "person (default %(default)s)")
    ap.add_argument("--cli", default=None,
                    help="the npuimage binary (default: the one next to "
                         "this repo's runtime/build)")
    ap.add_argument("--conf", type=float, default=None,
                    help="passed to both runs; the point of a low threshold is "
                         "to see the detections the quantised run loses rather "
                         "than only the ones that clear it")
    ap.add_argument("--pass", dest="extra", action="append", default=[],
                    help="an extra CLI flag, repeatable, applied to BOTH runs")
    args = ap.parse_args()

    cli = args.cli or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "runtime", "build", "npuimage")
    if not os.path.exists(cli):
        raise SystemExit(f"no binary at {cli}. Build it, or pass --cli.")

    extra = list(args.extra)
    if args.conf is not None:
        extra += ["--conf", str(args.conf)]

    def mb(p):
        n = os.path.getsize(p)
        return f"{n / 1e6:.2f} MB"

    print(f"reference  {args.ref}  ({mb(args.ref)})")
    print(f"test       {args.test}  ({mb(args.test)})")
    r0 = run(cli, args.ref, args.images[0], extra)
    t0 = run(cli, args.test, args.images[0], extra)
    print(f"containers state their own precision; the reference says "
          f"{r0.get('backend', {}).get('conv', '?')} convs on "
          f"{r0['backend']['dispatches']} dispatches")
    print()

    tot = {"score": 0.0, "box": 0.0, "kpt": 0.0, "n": 0,
           "missing": 0, "extra": 0, "flip": 0}
    worst = {"score": 0.0, "box": 0.0, "kpt": 0.0, "where": ""}

    for img in args.images:
        r = run(cli, args.ref, img, extra)
        t = run(cli, args.test, img, extra)
        rl, tl = r["landmarks"], t["landmarks"]
        pairs, only_r, only_t = match(rl, tl, args.iou_match)
        name = os.path.basename(img)
        if not rl and not tl:
            print(f"{name}: no detections in either run")
            continue
        print(f"{name}:  {len(rl)} -> {len(tl)} detections, "
              f"{len(pairs)} matched at IoU >= {args.iou_match}")
        for i in only_r:
            print(f"   lost    score {rl[i]['score']:.3f} "
                  f"box {box(rl[i])}")
        for j in only_t:
            print(f"   gained  score {tl[j]['score']:.3f} "
                  f"box {box(tl[j])}")
        tot["missing"] += len(only_r)
        tot["extra"] += len(only_t)
        for i, j, v in pairs:
            ds = abs(rl[i]["score"] - tl[j]["score"])
            db = corner_delta(rl[i], tl[j])
            dk = kpt_delta(rl[i], tl[j])
            flips = sum(1 for p, q in zip(rl[i]["keypoints"], tl[j]["keypoints"])
                        if p["visible"] != q["visible"])
            tot["n"] += 1
            tot["score"] += ds
            tot["box"] += db
            tot["kpt"] += dk
            tot["flip"] += flips
            for key, val in (("score", ds), ("box", db), ("kpt", dk)):
                if val > worst[key]:
                    worst[key] = val
                    worst["where"] = f"{name} detection {i} (iou {v:.3f})"
            print(f"   #{i} score {rl[i]['score']:.3f}->{tl[j]['score']:.3f} "
                  f"({ds:+.4f})  box corner {db:6.1f} px  keypoint {dk:6.1f} px"
                  + (f"  {flips} visibility flip(s)" if flips else ""))
        print()

    n = max(1, tot["n"])
    print(f"over {tot['n']} matched detections on {len(args.images)} images")
    print(f"   mean |score delta|   {tot['score'] / n:.4f}"
          f"   worst {worst['score']:.4f}   ({worst['where']})")
    print(f"   mean box corner      {tot['box'] / n:.1f} px"
          f"   worst {worst['box']:.1f} px   ({worst['where']})")
    print(f"   mean keypoint        {tot['kpt'] / n:.1f} px"
          f"   worst {worst['kpt']:.1f} px   ({worst['where']})")
    print(f"   lost {tot['missing']}, gained {tot['extra']}, "
          f"{tot['flip']} keypoint visibility flips")
    print()
    print("NOT A PASS/FAIL. No accuracy requirement has been decided for a "
          "quantised pose container, so this tool exits 0 on every accuracy it "
          "measures -- including an i4 per-channel container that returns five "
          "people for a photograph of three. What it gives you is the trade: the "
          "container sizes above and these numbers below them.")


if __name__ == "__main__":
    main()
