"""JIT-cache identity for the elementwise designs.

The matching is by symbol set and element count, not by geometry: two column
counts produce the same symbol and the same buffer size, which is why the
caller purges first and then insists on exactly one candidate.
"""

import re
import shutil
from pathlib import Path

TILE_RE = re.compile(r"aie\.tile\((\d+)\s*,\s*(\d+)\)")

def core_columns(d: Path) -> int:
    """Count distinct core columns from the placed MLIR. Raise if unreadable.

    `aie.mlir` is pre-placement and carries no coordinates, so counting there
    yields 0 for every design; the placed form is `input_with_addresses.mlir`.
    A design whose width cannot be established is never accepted -- that is how
    a 2-column GELU once shipped as a 1-column one.
    """
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        raise RuntimeError(f"{d.name}: no input_with_addresses.mlir")
    text = m.read_text(encoding="utf-8", errors="ignore")
    cols = {int(c) for c, r in TILE_RE.findall(text) if int(r) >= 2}
    if not cols:
        raise RuntimeError(f"{d.name}: placed MLIR has no core tiles")
    return len(cols)

def _markers_match(text: str, symbols: list[str], n_elem: int) -> bool:
    if not all(s in text for s in symbols):
        return False
    return re.search(rf"memref<\s*{n_elem}\s*x\s*bf16\s*>", text) is not None

def purge(cache_dir: Path, symbols: list[str], n_elem: int, what: str) -> int:
    """Remove every cached design for this op before rebuilding.

    Symbol and buffer size cannot distinguish a design built at one column count
    from the same op at another, and a JIT cache hit does not restamp the
    directory, so mtime is not a tie-break. Removing first and requiring exactly
    one match afterwards is the same determinism the GEMM exporter uses.
    """
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for d in list(cache_dir.iterdir()):
        if not d.is_dir():
            continue
        mlir = d / "aie.mlir"
        if not mlir.exists():
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if _markers_match(text, symbols, n_elem):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        print(f"  {what}: purged {removed} cache candidate(s)")
    return removed

def find_cache(cache_dir: Path, symbols: list[str], n_elem: int,
               n_cols: int, what: str) -> Path:
    if not cache_dir.is_dir():
        raise SystemExit(f"{what}: cache directory does not exist: {cache_dir}")
    hits = []
    for d in cache_dir.iterdir():
        if not d.is_dir():
            continue
        if not ((d / "aie.mlir").exists() and (d / "final.xclbin").exists()
                and (d / "insts.bin").exists()):
            continue
        text = (d / "aie.mlir").read_text(encoding="utf-8", errors="ignore")
        if not _markers_match(text, symbols, n_elem):
            continue
        try:
            if core_columns(d) != n_cols:
                continue
        except RuntimeError:
            continue
        hits.append(d)
    if len(hits) != 1:
        raise SystemExit(
            f"{what}: {len(hits)} cache candidates after purge -- expected "
            f"exactly 1 in {cache_dir}")
    return hits[0]
