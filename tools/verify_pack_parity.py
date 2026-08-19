# NpuEmbeddings -- do the two model packers agree, byte for byte?
# SPDX-License-Identifier: Apache-2.0
#
# There are two implementations of the .npue layout: tools/pack_npue.py (the
# reference, used at build time) and `npuembed --prepare-model` (C++, so a
# downloaded release can build the container without Python).
#
# Two implementations of one binary layout is a real risk. A disagreement would
# put correctly-sized weights in the wrong order -- exactly the failure
# tasks/0022 hit, which produced rel_fro 1.186 and which no size check can
# catch. So the gate is not a tolerance. It is byte equality.
#
# It has already earned its keep. The first C++ version differed in three
# independent ways, and none would have been found by comparing embeddings:
#
#   1. the source checksum was left empty, so the container could not say
#      which checkpoint it came from;
#   2. the layout descriptor was emitted with sorted keys, while the reference
#      stores insertion order -- a byte-different file with an IDENTICAL
#      layout hash, so the existing guard would have passed it;
#   3. the 1/sqrt(head_dim) fold was computed in double and rounded once at
#      the end, where numpy multiplies a float32 array by a Python float in
#      float32. That double rounding moved 86 of 2304 bias values and one
#      weight by 1 ULP.
#
# Env: iron env (numpy only). Usage:
#   python tools\verify_pack_parity.py

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(p: Path) -> str:
    head = p.open("rb").read(64)
    _, ver, arch, flags, jo, jl, do, dl = struct.unpack("<4sIII QQQQ", head[:48])
    return (f"{p.stat().st_size / 1e6:.2f} MB, json {jl} B at {jo}, "
            f"data {dl / 1e6:.2f} MB at {do}, v{ver} arch{arch} flags{flags}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--exe",
                    default=str(REPO / "runtime" / "build" / "npuembed.exe"))
    # Tile size stopped being a constant when bge-large arrived: its N in
    # {1024, 3072, 4096} makes tile_n 48 illegal, so it packs at 32. A gate
    # that only ever checked 48 would not have covered the packer's newest
    # parameter -- which is exactly where two implementations drift.
    ap.add_argument("--tile-n", type=int, default=48)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    for need in ("model.safetensors", "vocab.txt", "config.json"):
        if not (model_dir / need).exists():
            print(f"missing {model_dir / need} -- see BUILD.md")
            return 2

    with tempfile.TemporaryDirectory(prefix="packparity_") as td:
        d = Path(td)
        py_out, cc_out = d / "python.npue", d / "cpp.npue"

        r = subprocess.run([sys.executable, str(REPO / "tools" / "pack_npue.py"),
                            "--model-dir", str(model_dir), "--out", str(py_out),
                            "--tile-n", str(args.tile_n)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"pack_npue.py failed:\n{r.stdout}\n{r.stderr}")
            return 2

        r = subprocess.run([args.exe, "--prepare-model", str(model_dir),
                            str(cc_out), "--tile-n", str(args.tile_n)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"npuembed --prepare-model failed:\n{r.stdout}\n{r.stderr}")
            return 2

        a, b = sha256(py_out), sha256(cc_out)
        print(f"  tile_n {args.tile_n}, model {model_dir.name}")
        print(f"  pack_npue.py    {describe(py_out)}")
        print(f"  --prepare-model {describe(cc_out)}")
        print(f"\n  python : {a}")
        print(f"  c++    : {b}")

        if a == b:
            print("\nPASS -- byte-identical")
            return 0

        # Not equal: say WHERE, because "they differ" is not a diagnosis.
        pa, pb = py_out.read_bytes(), cc_out.read_bytes()
        print(f"\nFAIL -- differ ({len(pa)} vs {len(pb)} bytes)")
        _, _, _, _, jo, jl, do, _ = struct.unpack("<4sIII QQQQ", pa[:48])
        n = min(len(pa), len(pb))
        diffs = [i for i in range(n) if pa[i] != pb[i]]
        print(f"  {len(diffs)} differing bytes")
        if diffs:
            first = diffs[0]
            where = ("header" if first < 64 else
                     "json" if first < jo + jl else f"data +{first - do}")
            print(f"  first at {first} ({where})")
            if where == "json":
                s = max(jo, first - 80)
                print(f"    python: ...{pa[s:first + 60].decode('utf-8', 'replace')}")
                print(f"    c++   : ...{pb[s:first + 60].decode('utf-8', 'replace')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
