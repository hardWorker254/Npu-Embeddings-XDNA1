"""Compare our runtime against an externally-obtained reference vector.

The reference is quoted to 3 decimals, so this can only ever place a CEILING
on our error: rounding alone permits +-0.0005 per component, and a uniform
rounding error has mean |e| = 0.00025. If the measured mean |delta| sits at
that floor, the disagreement is the quoting, not the encoder.
"""
import math
import re
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
REPO = Path.cwd()
exe = REPO / "runtime" / "build" / "npuembeddings.exe"

CASES = [
    ("bge-small-en-v1.5", "ref_bge_small_test.txt", 384, "test"),
    ("bge-large-en-v1.5", "ref_bge_large_test.txt", 1024, "test"),
]


def parse(path, dim):
    raw = (SCRATCH / path).read_text(encoding="utf-8")
    pairs = re.findall(r"(?<![\w.-])(\d{1,4})\s+(-?\d+\.\d+)", raw)
    ref = {}
    for i, v in pairs:
        i = int(i)
        assert i not in ref, f"index {i} twice"
        ref[i] = float(v)
    assert sorted(ref) == list(range(dim)), f"{len(ref)} of {dim} indices"
    return [ref[i] for i in range(dim)]


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v], n


for model, ref_file, dim, text in CASES:
    r = parse(ref_file, dim)
    with tempfile.TemporaryDirectory(prefix="refcmp_") as td:
        d = Path(td)
        (d / "in.txt").write_text(text + "\n", encoding="utf-8")
        p = subprocess.run(
            [str(exe), "embed", model, str(d / "in.txt"), str(d / "out.f32"),
             "--root", str(REPO), "--threads", "8"],
            capture_output=True, text=True, encoding="utf-8")
        if p.returncode != 0:
            print(p.stdout, p.stderr)
            sys.exit(1)
        datapath = next((l.split(None, 1)[1].strip()
                         for l in p.stdout.splitlines()
                         if l.strip().startswith("datapath")), "?")
        a = array("f")
        a.frombytes((d / "out.f32").read_bytes())
        got = list(a[:dim])

    ru, rn = unit(r)
    gu, gn = unit(got)
    cos = sum(x * y for x, y in zip(ru, gu))
    da = [abs(x - y) for x, y in zip(ru, gu)]
    # A -0.000 in the reference parses to -0.0, and -0.0 >= 0 is True in
    # Python -- which manufactured five phantom sign disagreements the first
    # time this was run. Compare signs only where the reference is resolvable.
    cmp_idx = [i for i in range(dim) if abs(r[i]) >= 0.0005]
    same = sum(1 for i in cmp_idx if (ru[i] >= 0) == (gu[i] >= 0))

    print(f"=== {model}  '{text}'  dim {dim}")
    print(f"    {datapath}")
    print(f"    L2: reference {rn:.6f}   ours {gn:.6f}"
          f"        (a rounded unit vector does not land on 1.000000)")
    print(f"    cosine {cos:.6f}    1-cos {1.0-cos:.3e}")
    print(f"    |delta|: mean {sum(da)/dim:.5f}   max {max(da):.5f}"
          f"   (3-decimal rounding floor: mean 0.00025, max 0.00050)")
    print(f"    sign agreement {same}/{len(cmp_idx)} on components the "
          f"reference can resolve ({dim - len(cmp_idx)} are |x| < 0.0005)")
    worst = sorted(range(dim), key=lambda i: -da[i])[:4]
    print("      worst:  " + "   ".join(
        f"[{i}] {gu[i]:+.4f} vs {ru[i]:+.4f}" for i in worst))
    print()
