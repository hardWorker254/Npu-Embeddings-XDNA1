#!/usr/bin/env python3
r"""sha256 of a .npue container's DATA REGION -- the tensor bytes only.

WHY NOT THE WHOLE FILE (tasks/0119, fixed for 0.5.0 in tasks/0132): the
release check used to compare whole-file hashes, and a correct release
failed it. A cold-built container differed from the validated one by
exactly 17 bytes of JSON -- the `a_dtype` key tasks/0104 added to the
directory -- while the tensor data region was byte-identical. The JSON
directory legitimately gains keys as the format evolves; the data region
is what the accuracy claims were validated on. So the check that means
what the release intends is on [data_offset, data_offset + data_length),
read from the container's own header.

Prints `<sha256>  <path>` per file, exactly like sha256sum, so two
containers compare with two invocations and a string equality.

    python tools/npue_data_hash.py <a.npue> [<b.npue> ...]

Exit 1 if two or more files are given and their data hashes are NOT all
equal -- so the release check is one invocation:

    python tools/npue_data_hash.py repo\models\m.npue coldzip\models\m.npue
"""
from __future__ import annotations

import hashlib
import struct
import sys

HEADER_FORMAT = "<4sIII QQQQ 16s"          # mirrors tools/npue.py
HEADER_SIZE = 64
MAGIC = b"NPUE"


def data_hash(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
        if len(head) != HEADER_SIZE:
            raise SystemExit(f"{path}: shorter than a .npue header")
        (magic, _version, _arch, _flags,
         _json_off, _json_len, data_off, data_len, _r) = struct.unpack(
            HEADER_FORMAT, head)
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a .npue file (magic {magic!r})")
        f.seek(data_off)
        h = hashlib.sha256()
        remaining = data_len
        while remaining:
            chunk = f.read(min(1 << 22, remaining))
            if not chunk:
                raise SystemExit(
                    f"{path}: data region truncated -- header promises "
                    f"{data_len} bytes at {data_off}, file ends early")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    hashes = []
    for p in paths:
        d = data_hash(p)
        hashes.append(d)
        print(f"{d}  {p}")
    if len(hashes) > 1 and len(set(hashes)) != 1:
        print("MISMATCH -- the data regions differ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
