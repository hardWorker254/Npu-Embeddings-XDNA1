"""Record the sequence length in design sets exported before it was a field.

Sequence length belongs to the DESIGN -- the container's `max_seq_len` is how
many position embeddings were packed (256), not what the array was compiled
for (64). tools/export_gemm_rtp.py writes `"seq"` now; the 200-odd design sets
exported before it did not, and the runtime refuses to guess.

This does not guess either. Within one design set the value is RECOVERABLE
from two independent artifacts that must agree:

    softmax/design.json  "cols" is the softmax row length, which IS seq
    softmax/design.json  "rows" is batch * heads * seq
    qkv/design.json      "M"    is batch * seq

so  heads = softmax.rows / qkv.M  must come out a positive integer, and
qkv.M must divide by seq. Two sources, one answer, or the set is skipped and
named rather than stamped with a plausible number.

Metadata only -- no design is recompiled and no .xclbin is touched.

    python tools/backfill_design_seq.py runtime          # report
    python tools/backfill_design_seq.py runtime --write  # stamp
"""

import json
import sys
from pathlib import Path


def derive(setdir: Path):
    """(seq, why) for one artifacts/ directory, or (None, reason)."""
    sm, qkv = setdir / "softmax" / "design.json", setdir / "qkv" / "design.json"
    if not sm.exists():
        return None, "no softmax design to read the row length from"
    if not qkv.exists():
        return None, "no qkv design to cross-check against"
    s, q = json.loads(sm.read_text()), json.loads(qkv.read_text())
    seq, rows, m = s.get("cols"), s.get("rows"), q.get("M")
    if not (seq and rows and m):
        return None, "softmax cols/rows or qkv M missing"
    if m % seq:
        return None, f"qkv M={m} is not a whole number of seq-{seq} sequences"
    if rows % m:
        return None, f"softmax rows={rows} / qkv M={m} is not a head count"
    heads = rows // m
    if heads < 1:
        return None, f"derived head count {heads} is not positive"
    return seq, f"seq {seq}, batch {m // seq}, {heads} heads (two sources agree)"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runtime")
    write = "--write" in sys.argv
    stamped = skipped = already = 0

    for setdir in sorted(p for p in root.glob("artifacts*") if p.is_dir()):
        targets = [d for d in sorted(setdir.glob("*/design.json"))
                   if "seq" not in json.loads(d.read_text())]
        if not targets:
            already += 1
            continue
        seq, why = derive(setdir)
        if seq is None:
            print(f"  SKIP  {setdir.name:24s} {why}")
            skipped += 1
            continue
        print(f"  {'stamp' if write else 'would'} {setdir.name:24s} {why}"
              f"  [{len(targets)} files]")
        if write:
            for d in targets:
                j = json.loads(d.read_text())
                j["seq"] = seq
                # Keep the two-space form the exporters write, so a diff shows
                # one added line rather than a reformat.
                d.write_text(json.dumps(j, indent=2), encoding="utf-8")
        stamped += 1

    print(f"\n{stamped} design set(s) {'stamped' if write else 'to stamp'}, "
          f"{already} already carried seq, {skipped} skipped.")
    if skipped and not write:
        print("Skipped sets keep failing loudly at load, which is the point.")


if __name__ == "__main__":
    main()
