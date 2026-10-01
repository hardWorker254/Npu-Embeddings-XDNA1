"""JIT-cache identification: which cached entry is this design, and is it stale.

A design set is loaded once and kept, so this is on the critical path of every
rebuild rather than every run. The markers are regexes rather than substrings
because MLIR pretty-printing moves whitespace around; they are still strict
enough to separate shapes, dtypes and core-column counts.
"""

import re
import shutil
from pathlib import Path

TILE_RE = re.compile(r"aie\.tile\((\d+)\s*,\s*(\d+)\)")

def all_markers_match(text: str, markers: list[re.Pattern[str]]) -> bool:
    return all(marker.search(text) for marker in markers)

def core_columns(build_dir: Path, min_row: int = 2) -> int | None:
    """
    Count distinct AIE columns used by cores.

    This parses input_with_addresses.mlir and counts distinct column indices
    for tiles with row >= `min_row`. If the file is absent or no such tiles are
    found, returns None.

    `min_row = 2` is the historical filter and it is a proxy for "not a shim
    row", which holds because every design set in this tree is four AIE rows
    tall (`AIE_ROWS`). A design shorter than that would report None for ITSELF,
    and purge/find_cache would then reject every candidate including the one
    just built -- so if a one-row or two-row design is ever wanted, this filter
    and the AIE shim placement in gemm_pretiled.py have to move together, and
    the column count alone is not enough to tell such a design apart from a
    four-row one with the same M/K/N.
    """
    mlir = build_dir / "input_with_addresses.mlir"
    if not mlir.exists():
        return None

    try:
        text = mlir.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    cols = set()
    for col_s, row_s in TILE_RE.findall(text):
        try:
            col = int(col_s)
            row = int(row_s)
        except ValueError:
            continue
        if row >= min_row:
            cols.add(col)

    return len(cols) if cols else None

def purge(
    markers: list[re.Pattern[str]],
    cols: int,
    what: str,
    cache_dir: Path,
    min_row: int = 2,
) -> int:
    """
    Remove cache entries matching this design before rebuilding.

    This avoids stale cache entries with identical shape markers but different
    datapath or architecture. It is intentionally conservative: it only purges
    entries whose core column count matches the expected cols. Entries with
    unknown core column count are left alone; they will not be selected by
    find_cache() anyway.
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

        try:
            text = mlir.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not all_markers_match(text, markers):
            continue

        if core_columns(d, min_row) != cols:
            continue

        shutil.rmtree(d, ignore_errors=True)
        removed += 1

    if removed:
        print(f"  {what}: purged {removed} cache candidate(s)")

    return removed

def find_cache(
    markers: list[re.Pattern[str]],
    cols: int,
    what: str,
    cache_dir: Path,
    min_row: int = 2,
) -> Path:
    """
    Find exactly one cache entry matching the compiled markers and core cols.
    """
    if not cache_dir.is_dir():
        raise SystemExit(
            f"{what}: cache directory does not exist: {cache_dir}\n"
            f"Hint: if you use --per-arch-cache, make sure the toolchain "
            f"actually honors IRON_CACHE_DIR or remove --per-arch-cache."
        )

    hits: list[Path] = []

    for d in cache_dir.iterdir():
        if not d.is_dir():
            continue

        mlir = d / "aie.mlir"
        if not (
            mlir.exists()
            and (d / "final.xclbin").exists()
            and (d / "insts.bin").exists()
        ):
            continue

        try:
            text = mlir.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not all_markers_match(text, markers):
            continue

        if core_columns(d, min_row) != cols:
            continue

        hits.append(d)

    if len(hits) != 1:
        raise SystemExit(
            f"{what}: {len(hits)} cache candidates after purge -- expected exactly 1\n"
            f"Cache dir: {cache_dir}\n"
            f"Hint: if building multiple architectures in the same cache, use "
            f"--per-arch-cache or --require-arch-marker/--arch-marker."
        )

    return hits[0]

def xclbin_identical_mod_uuid(
    a: bytes,
    b: bytes,
    threshold: int,
) -> tuple[bool, str]:
    """
    Original 0029 check: same size and <= threshold differing bytes.

    The difference budget is intended to cover UUID and small metadata.
    If architecture 2 adds more harmless metadata, use --identity-threshold.
    """
    if len(a) != len(b):
        return False, f"sizes differ: {len(a)} vs {len(b)}"

    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs <= threshold, f"{diffs} differing bytes"
