"""Where a design set is written, and which JIT cache it comes from.

arch_out_dir is the ONE output convention in the tree: without --target it is
the historic <base>/artifacts_npu<arch>, with --target the model name is
inserted. Both the in-process and the subprocess --arch all path go through
set_out_dir, so the layout cannot drift between them.
"""

import argparse
from pathlib import Path

def cache_dir_for(args: argparse.Namespace, arch: str) -> Path:
    root = Path(args.cache_root).expanduser()
    if args.per_arch_cache:
        return root / f"arch{arch}"
    return root

def arch_out_dir(base: str | Path, arch: str, target: str | None = None) -> Path:
    """
    The one output convention: one directory per generation, each holding a
    `gemm_rtp` design set. `base` is the artifacts root (default runtime/).

    Without `target` this is the historic `<base>/artifacts_npu<arch>`. With a
    `--target` model the model name is inserted, so
    `<base>/<model>/artifacts_npu<arch>` is the per-model design set.

    The function is idempotent: if `base` already IS the generation root
    (`.../artifacts_npu<arch>`), it is returned unchanged. That keeps the
    subprocess path correct, where the parent hands the resolved set-dir to
    the child as `--out`.
    """
    base = Path(base)
    if base.name == f"artifacts_npu{arch}":
        return base
    if target:
        return base / target / f"artifacts_npu{arch}"
    return base / f"artifacts_npu{arch}"


def set_out_dir(args: argparse.Namespace, arch: str) -> Path:
    """
    Single source of truth for where an architecture's design set is written.

    Both `--arch all` branches (in-process and subprocess) and the single-arch
    path go through here, so the layout cannot drift between them.
    """
    return arch_out_dir(args.out, arch, getattr(args, "target", None))
