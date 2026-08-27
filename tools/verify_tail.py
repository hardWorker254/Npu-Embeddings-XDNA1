# NpuEmbeddings -- the tail gate: does any single input come back badly wrong?
# SPDX-License-Identifier: Apache-2.0
#
# Every other accuracy gate in this repo reports a central tendency -- the
# golden 1-cos is a mean over sentences, MTEB averages thousands of pairs, the
# semantic gate is a ranking with no absolute error term by design. bge-large
# passed all three while carrying a measured 100x max/median tail on
# single-word inputs (tasks/0122, research/OPEN-THREADS.md#t51), because a tail
# is exactly what an average is built to remove. This gate reads the tail.
#
# THE GATE IS A p99, PER MODEL, PER DATAPATH -- decided by measurement, not
# taste (tasks/0129): a max-gate is uninformative where the median already
# fails (int8 bge-large's median sits above 2e-03; its adoption verdict rests
# on MTEB, tasks/0085), while bfp16 bge-large's p99/median of 30x is precisely
# the signal the averaging instruments miss. `max` and `max/median` are
# REPORTED on every run and never gated.
#
# The threshold is a RATCHET, not a constant: each model's reference JSON
# carries `baseline.p99_ceiling = max(2e-3, 2 x p99 measured at baseline
# time)` -- tight where the model is tight (a bge-base regression to 4e-04
# fires), honest where it is not (bge-large's documented tail is waived by a
# note pointing at T51, not averaged away). `--write-baseline` is the only
# thing that moves it, and it records the datapath the runtime REPORTED at
# measurement time; a later run on a different datapath refuses rather than
# comparing across baselines (the catalogue is per-model per-datapath since
# tasks/0104).
#
# STDLIB ONLY, like verify_semantics.py and for the same reason: 224 vectors
# of at most 1024 floats is nothing, and paying it in pure Python buys running
# this against a COLD dist zip with nothing installed but CPython and the exe.
# The fp32 references cross the env boundary as FILES from
# tools/make_tail_reference.py (.venv-ref), per CLAUDE.md.
#
# Usage:
#   python tools\verify_tail.py                        # gate, all baselined models
#   python tools\verify_tail.py --write-baseline       # measure + record baselines
#   python tools\verify_tail.py --exe dist\...\npuembeddings.exe --root dist\...

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent

BUILTIN = [
    "all-MiniLM-L6-v2",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "nomic-embed-text-v1.5",
    "embeddinggemma-300m",
]

# The bge-large waiver text, written into its baseline by --write-baseline.
# The wide ceiling is a DOCUMENTED, REGISTER-LINKED waiver, not an endorsement:
# the model ships with a known 100x single-word tail whose mechanism is an
# open question. Do not tighten this by hand; close T51 first.
T51_NOTE = ("p99 ceiling is wide because bge-large has a known, input-"
            "determined accuracy tail on single-word inputs -- see "
            "research/OPEN-THREADS.md#t51 (tasks/0122, 0129). This is a "
            "documented waiver, not an endorsement.")

_HEADER = "<4sIII QQQQ 16s"
_HEADER_SIZE = 64
assert struct.calcsize(_HEADER) == _HEADER_SIZE


def container_config(path: Path) -> dict:
    """The .npue config dict, read with nothing but `struct` and `json`."""
    with open(path, "rb") as f:
        magic, _v, arch, _f, jo, jl, _do, _dl, _r = struct.unpack(
            _HEADER, f.read(_HEADER_SIZE))
        if magic != b"NPUE":
            raise SystemExit(f"{path}: not a .npue file (magic {magic!r})")
        f.seek(jo)
        cfg = dict(json.loads(f.read(jl).decode("utf-8"))["config"])
    cfg["_arch_field"] = arch
    return cfg


def read_f32(path: Path, rows: int, cols: int) -> list[list[float]]:
    """CHECK the byte count against the expected shape; never infer the width
    from the file -- an inferred width reshapes a truncated file into a
    plausible array (verify_semantics.py, trap 8's fail-open shape)."""
    raw = path.read_bytes()
    want = rows * cols * 4
    if len(raw) != want:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {want} "
                         f"({rows} x {cols} fp32)")
    a = array("f")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return [list(a[i * cols:(i + 1) * cols]) for i in range(rows)]


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if not (n > 0.0) or not math.isfinite(n):
        raise SystemExit("a vector has zero or non-finite norm")
    return [x / n for x in v]


def percentile(sorted_e: list[float], p: float) -> float:
    """numpy's default 'linear' interpolation, so these numbers are directly
    comparable with tasks/0122/0129's np.percentile figures."""
    n = len(sorted_e)
    pos = (n - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_e[lo] + (pos - lo) * (sorted_e[hi] - sorted_e[lo])


def stats(e: list[float]) -> dict:
    s = sorted(e)
    med = percentile(s, 50)
    return {"n": len(e), "median": med, "p90": percentile(s, 90),
            "p99": percentile(s, 99), "max": s[-1],
            "max_over_median": (s[-1] / med) if med > 0 else float("inf")}


def embed(exe: Path, root: Path, model: str, texts: list[str], hidden: int,
          prefix: str | None, threads: int,
          artifacts: str | None) -> tuple[list[list[float]], dict]:
    """The shipped CLI form, and the datapath scraped from the runtime's OWN
    status line -- never restated from what this file asked for. Same
    discipline and same regex as verify_semantics.py."""
    with tempfile.TemporaryDirectory(prefix="tailgate_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        cmd = [str(exe), "embed", model, str(d / "in.txt"), str(d / "out.f32"),
               "--root", str(root), "--threads", str(threads)]
        if artifacts:
            cmd += ["--artifacts", artifacts]
        if prefix is not None:
            cmd += ["--prefix", prefix]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.stdout.write(r.stdout or "")
            sys.stderr.write(r.stderr or "")
            raise SystemExit(f"{model}: npuembeddings embed failed "
                             f"({r.returncode})")
        said = {}
        for line in (r.stdout or "").splitlines():
            m = re.match(r"\s{2}(datapath|designs|toolchain|shape)\s{2,}(.+)",
                         line)
            if m and m.group(1) not in said:
                said[m.group(1)] = m.group(2).strip()
        vs = [unit(v) for v in read_f32(d / "out.f32", len(texts), hidden)]
        return vs, said


def check_model(model: str, ref_dir: Path, exe: Path, root: Path,
                args) -> dict:
    meta_path = ref_dir / f"{model}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    texts = meta["texts"]
    hidden = int(meta["hidden"])
    groups = meta["corpus"]["groups"]

    cfg = container_config(root / "models" / f"{model}.npue")
    if int(cfg["hidden"]) != hidden:
        raise SystemExit(f"{model}: container hidden {cfg['hidden']} != "
                         f"reference hidden {hidden} -- the reference was "
                         f"generated for a different checkpoint")
    # The runtime refuses --prefix on a model without a prompts table and
    # refuses to run without it on a model that has one (tasks/0118) -- so the
    # name recorded at reference time is passed through verbatim and the
    # runtime itself arbitrates whether it is still valid.
    prefix = meta["prompt_name"]

    ref = [unit(v) for v in read_f32(ref_dir / f"{model}.f32",
                                     len(texts), hidden)]
    vs, said = embed(exe, root, model, texts, hidden, prefix,
                     args.threads, args.artifacts)

    e = [max(0.0, 1.0 - sum(a * b for a, b in zip(g, r)))
         for g, r in zip(vs, ref)]
    overall = stats(e)
    per_group = {name: stats(e[lo:hi]) for name, (lo, hi) in groups.items()}
    worst = sorted(range(len(e)), key=lambda i: -e[i])[:5]

    out = {
        "model": model, "hidden": hidden,
        "prompt": prefix,
        "datapath_reported": said.get("datapath", "UNREPORTED"),
        "designs_reported": said.get("designs", "UNREPORTED"),
        "overall": overall,
        "groups": per_group,
        "worst": [{"text": texts[i], "one_minus_cos": e[i]} for i in worst],
    }

    baseline = meta.get("baseline")
    if args.write_baseline:
        baseline = {
            "p99_measured": overall["p99"],
            "p99_ceiling": max(2e-3, 2.0 * overall["p99"]),
            "measured_date": datetime.date.today().isoformat(),
            "datapath": out["datapath_reported"],
        }
        if model == "bge-large-en-v1.5":
            baseline["note"] = T51_NOTE
        meta["baseline"] = baseline
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        out["baseline"] = baseline
        out["verdict"] = "BASELINED"
        out["pass"] = True
        return out

    if baseline is None:
        # UNBASELINED is a FAILURE, not a skip. A gate that silently passes
        # models it has no ceiling for is fail-open -- the exact shape this
        # project keeps finding (traps 6b/6c/7c/7d, tasks/0110).
        out["verdict"] = ("UNBASELINED -- run `python tools\\verify_tail.py "
                          "--write-baseline` deliberately, then commit the "
                          "reference JSON")
        out["pass"] = False
        return out

    out["baseline"] = baseline
    if (baseline["datapath"] != out["datapath_reported"]):
        # A ceiling measured on one datapath says nothing about another
        # (tasks/0129: int8 and bfp16 bge-large differ 10x at the median).
        out["verdict"] = (f"DATAPATH MISMATCH -- baseline was measured on "
                          f"{baseline['datapath']!r}, the runtime reports "
                          f"{out['datapath_reported']!r}. Re-baseline "
                          f"deliberately; refusing to compare across "
                          f"datapaths")
        out["pass"] = False
    elif overall["p99"] > baseline["p99_ceiling"]:
        out["verdict"] = (f"FAIL -- p99 {overall['p99']:.3e} > ceiling "
                          f"{baseline['p99_ceiling']:.3e} (measured "
                          f"{baseline['p99_measured']:.3e} on "
                          f"{baseline['measured_date']})")
        out["pass"] = False
    else:
        out["verdict"] = "PASS"
        out["pass"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=BUILTIN)
    ap.add_argument("--reference", default=str(REPO / "reference" / "tail"),
                    help="directory holding <model>.f32 + <model>.json from "
                         "make_tail_reference.py (default: the repo's)")
    ap.add_argument("--exe", default=None,
                    help="npuembeddings.exe -- a dev build or the one in a "
                         "dist zip (default: runtime/build)")
    ap.add_argument("--root", default=None,
                    help="the directory holding models/ and artifacts_* "
                         "(default: the repo)")
    ap.add_argument("--artifacts", default=None,
                    help="override the design set; by default the runtime "
                         "picks, which exercises pick_artifacts() too")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--write-baseline", action="store_true",
                    help="measure p99 per model and record it in the "
                         "reference JSON as the ceiling's basis "
                         "(ceiling = max(2e-3, 2 x p99)). Deliberate act; "
                         "commit the JSONs it rewrites")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else REPO
    exe = (Path(args.exe).resolve() if args.exe
           else REPO / "runtime" / "build" / "npuembeddings.exe")
    if not exe.exists():
        raise SystemExit(f"no runtime at {exe} -- build it, or pass --exe")
    ref_dir = Path(args.reference).resolve()

    print("tail gate -- per-text 1-cos against stored fp32 references, "
          "gated on p99")
    print(f"exe {exe}")
    print("THE GATE IS THE p99 against each model's recorded ceiling. max and")
    print("max/median are REPORTED, never gated -- a max-gate is uninformative")
    print("where the median already fails (tasks/0129). bge-large's wide")
    print("ceiling is a register-linked waiver: research/OPEN-THREADS.md#t51.\n")

    results = []
    for m in args.models:
        if not (ref_dir / f"{m}.json").exists():
            print(f"  {m:<24} SKIP -- no reference (run "
                  f"make_tail_reference.py in .venv-ref)")
            continue
        if not (root / "models" / f"{m}.npue").exists():
            print(f"  {m:<24} SKIP -- no container")
            continue
        r = check_model(m, ref_dir, exe, root, args)
        results.append(r)
        o = r["overall"]
        verdict = r["verdict"].split(" -- ")[0]
        print(f"  {m:<24} {verdict:<10} "
              f"median {o['median']:.3e}  p90 {o['p90']:.3e}  "
              f"p99 {o['p99']:.3e}  max {o['max']:.3e}  "
              f"(max/median {o['max_over_median']:.1f}x)")
        if r.get("baseline"):
            b = r["baseline"]
            print(f"{'':>26}ceiling {b['p99_ceiling']:.3e}  "
                  f"(p99 {b['p99_measured']:.3e} measured "
                  f"{b['measured_date']})"
                  + ("   [T51 waiver]" if b.get("note") else ""))
        print(f"{'':>26}{r['datapath_reported']}")
        for g in ("words", "sentences", "phrases"):
            if g in r["groups"]:
                s = r["groups"][g]
                print(f"{'':>26}{g:<10} n={s['n']:>3}  "
                      f"median {s['median']:.2e}  p99 {s['p99']:.2e}  "
                      f"max {s['max']:.2e}")
        print(f"{'':>26}worst: " + ", ".join(
            f"{w['text'][:18]!r} {w['one_minus_cos']:.2e}"
            for w in r["worst"]))
        if not r["pass"]:
            print(f"{'':>26}{r['verdict']}")

    if not results:
        raise SystemExit("no models checked")

    ok = all(r["pass"] for r in results)
    out = Path(args.out) if args.out else (
        REPO / "tasks" / "0132-t51-tail-gate" / "tail_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "hardware measurement -- tail gate, p99-gated per model",
        "exe": str(exe),
        "reference_dir": str(ref_dir),
        "write_baseline": bool(args.write_baseline),
        "models": results, "pass": ok,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    print("PASS -- every model's p99 is under its recorded ceiling" if ok
          else "FAIL -- see the verdict lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
