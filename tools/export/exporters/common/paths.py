"""Where a design set is written, and which JIT cache it comes from.

arch_out_dir is the ONE output convention in the tree: without --target it is
the shared `<base>/artifacts_npu<arch>`, with --target the per-model set gains an
`artifacts/` level so every per-model set in the tree sits under ONE parent
directory:

    The function is idempotent, twice over, because two callers depend on it:
    if `base` already IS the generation root (`.../artifacts_npu<arch>`) it is
    returned unchanged -- that is the subprocess path, where the parent hands
    the resolved set-dir to the child as `--out`, `-i8` suffix and all; and if
    `base` already IS the `artifacts/` parent it does not gain a second one, so
    `--out runtime/artifacts` and `--out runtime` name the same place.

    runtime/artifacts/<model>/artifacts_npu<arch>/{gemm_rtp,gelu,layernorm,softmax}

Both the in-process and the subprocess --arch all path go through
set_out_dir, so the layout cannot drift between them.

WHY THE `artifacts/` LEVEL. It used to be `runtime/<model>/artifacts_npu<arch>`,
which put a dozen build-artifact directories at the top level of runtime/ --
indistinguishable by name from the hand-maintained source directories beside
them, and it is what forced the int8 sets into a second convention
(`runtime/<model>-i8/gemm_rtp/`), because discovery scans a fixed depth.
Grouping them under one parent makes `git status`, `ls runtime/`, and the
discovery scan agree about what is an artifact and what is source.

WHY THE `-i8` SUFFIX, and why it is here rather than in the runtime. The two
datapaths of one model are two design sets that cannot be told apart by
geometry -- same hidden, same intermediate, same device, same streams, same
everything except `b_layout_hash`. So they need two directories, and the
runtime has to know both names. It proposes `<model>` and `<model>-i8`
(model_set_candidates), which means the WRITER has to produce both spellings;
a suffix decided by the writer and guessed at by the reader is how a container
ends up with no design set to run on and a message that says so.
"""

import argparse
from pathlib import Path

# The directory name that groups per-model design sets. A constant because the
# writer, the runtime's discovery scan, .gitignore and the documents all have to
# agree on it, and four copies of the string is how they stop agreeing.
ARTIFACTS_DIR = "artifacts"

def cache_dir_for(args: argparse.Namespace, arch: str) -> Path:
    root = Path(args.cache_root).expanduser()
    if args.per_arch_cache:
        return root / f"arch{arch}"
    return root

def arch_out_dir(base: str | Path, arch: str, target: str | None = None,
                 int8: bool = False) -> Path:
    """
    The one output convention: one directory per generation, each holding a
    `gemm_rtp` design set. `base` is the artifacts root (default runtime/).

    Without `target` this is the shared `<base>/artifacts_npu<arch>`. With a
    `--target` model it is `<base>/artifacts/<model>/artifacts_npu<arch>`, so
    every per-model set in the tree shares one parent directory.

    AND AN int8 SET IS NAMED `<model>-i8`, because one directory holds one
    design set and a model's two datapaths do not fit each other:
    `b_layout_hash` covers the B operand's dtype, so the int8 set hashes
    177088d6 and the bf16 set 52a4adad, and `design_fits` refuses the wrong
    pairing. The reader proposes `<model>` and `<model>-i8` for every model
    (model_set_candidates in runtime/include/common/design_selection.hpp), so
    writing them under one name and looking for them under another is how an
    int8 container ends up with nothing to run on.

    The suffix follows `--int8` and nothing else. `--int8 --emulate-bfp16`
    still gets it, because that pair sets the int8 operand dtype and is what
    the container hash will be built from.
    """
    base = Path(base)
    if base.name == f"artifacts_npu{arch}":
        return base
    if target:
        name = f"{target}-i8" if int8 else target
        if base.name != ARTIFACTS_DIR:
            base = base / ARTIFACTS_DIR
        return base / name / f"artifacts_npu{arch}"
    return base / f"artifacts_npu{arch}"


def set_out_dir(args: argparse.Namespace, arch: str) -> Path:
    """
    Single source of truth for where an architecture's design set is written.

    Both `--arch all` branches (in-process and subprocess) and the single-arch
    path go through here, so the layout cannot drift between them.
    """
    return arch_out_dir(args.out, arch, getattr(args, "target", None),
                        bool(getattr(args, "int8", False)))
