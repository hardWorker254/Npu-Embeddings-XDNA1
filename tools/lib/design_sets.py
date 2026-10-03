# SPDX-License-Identifier: Apache-2.0
"""Where design sets live, for the Python side.

THE WRITER IS `tools/export/exporters/common/paths.py` (`arch_out_dir`) and the
READER IS `model_set_candidates` in `runtime/include/common/design_selection.hpp`.
This is the third copy, and it exists because two tools need to FIND design sets
rather than write or resolve one:

  * `tools/pipeline.py`, to answer whether a tier's `designs` prerequisite is
    satisfied at all;
  * `tools/verify/verify_design_numerics.py`, whose default is "every design set
    in the tree".

Three copies of a path convention is how they stop agreeing, and the last time
they did disagree the symptom was a gate that reported 0 design sets in a tree
that had twelve. So the patterns live here, once, with the layouts written down
beside them -- and each pattern is checked against the writer's own output by
`verify_layout_agrees()`, which is a gate, not a comment.

    runtime/artifacts/<model>/artifacts_npu<N>/<set>/design.json   <- current
    runtime/<model>/artifacts_npu<N>/<set>/design.json             <- pre-2026
    runtime/<model>-i8/gemm_rtp/design.json                       <- pre-2026 int8

The two older shapes are still searched because a design set is a build artifact:
a tree can legitimately hold sets exported before the layout changed, and
`design_fits` refuses any of them that does not actually fit the container being
run.
"""

from __future__ import annotations

from pathlib import Path

# The directory that groups per-model design sets. Must equal ARTIFACTS_DIR in
# tools/export/exporters/common/paths.py.
ARTIFACTS_DIR = "artifacts"

# (label, glob relative to the repo root). Order is the search order, which is
# also the order the runtime's own candidate list tries.
LAYOUTS: tuple[tuple[str, str], ...] = (
    ("artifacts", f"runtime/{ARTIFACTS_DIR}/*/artifacts_npu*/**/design.json"),
    ("per-model", "runtime/*/artifacts_npu*/**/design.json"),
    ("per-model-i8", "runtime/*-i8/**/design.json"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def design_sets(root: Path | None = None) -> list[Path]:
    """Every design.json in the tree, deduplicated, sorted for reproducibility.

    Sorted rather than left in glob order because a gate that reports "12 design
    sets" must not print a different 12 depending on the filesystem.
    """
    r = Path(root) if root is not None else repo_root()
    seen: dict[str, Path] = {}
    for _label, pattern in LAYOUTS:
        for p in r.glob(pattern):
            seen.setdefault(str(p), p)
    return sorted(seen.values())


def verify_layout_agrees(root: Path | None = None) -> list[str]:
    """Design sets in the tree that NO declared layout can reach.

    Empty is the pass condition. A non-empty result means a design set exists on
    disk that `arch_out_dir` would never write and that the pipeline's
    prerequisite check would not count -- which is the failure this module
    exists to make impossible to have quietly.
    """
    r = Path(root) if root is not None else repo_root()
    found = {str(p) for p in design_sets(r)}
    reachable: set[str] = set()
    for _label, pattern in LAYOUTS:
        reachable.update(str(p) for p in r.glob(pattern))
    return sorted(found - reachable)
