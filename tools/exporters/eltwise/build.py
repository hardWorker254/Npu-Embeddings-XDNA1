"""Writing one architecture's gelu / layernorm / softmax design directories."""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import npu_ops  # tools/ is on sys.path (the shim puts it there) -- see the
                # module's own header for why this table is written twice.
from ..common.consts import ARCH_DEVICES, DEFAULT_CACHE_ROOT, DEFAULT_SEQ
from .cache import find_cache, purge
from .consts import GELU_TILE, HEADS, MAX_GELU_COLS, MAX_LN_SM_COLS
from .kernels import _arrays

def _write_dir(dst: Path, src: Path, meta: dict) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        shutil.copy2(src / f, dst / f)
    meta = dict(meta)
    meta["insts_bytes"] = (dst / "insts.bin").stat().st_size
    meta["source_cache_dir"] = src.name
    (dst / "design.json").write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    print(f"  {meta['name']:<10} arch {meta['arch']} {meta['device']:<5} "
          f"xclbin {(dst / 'final.xclbin').stat().st_size / 1024:7.1f} KB  "
          f"insts {meta['insts_bytes'] / 1024:6.1f} KB  ({src.name})")
    return meta

def export_arch(out_dir: Path, arch: str, batch: int, hidden: int = 384,
                seq: int = DEFAULT_SEQ, n_cols: int = 1, gelu_tile: int = GELU_TILE,
                ln_variant: str = "il4", sm_variant: str = "poly_il4",
                cache_root: Path | None = DEFAULT_CACHE_ROOT,
                per_arch_cache: bool = False,
                ops: set[str] | None = None) -> list[dict]:
    """Build the requested eltwise designs into `out_dir` for one generation.

    `ops` is the set of CODES from tools/npu_ops.py (gelu, layn, softm); an
    empty or omitted set means all three, which is what this function has always
    done and what a caller with no opinion should get. An op that is not built
    is a directory the runtime will refuse by name if --npu-ops asks for it, so
    building a subset is a real saving (a compile and an xclbin each) and never a
    silent one.
    """
    device = ARCH_DEVICES.get(arch)
    if not device:
        raise SystemExit(f"unknown architecture: {arch}")

    # The documented width walls, refused before compiling rather than deep in
    # the placer. Name the constraint so the fix is obvious.
    if n_cols > MAX_LN_SM_COLS:
        raise SystemExit(
            f"eltwise: {n_cols} columns is above the LayerNorm/softmax wall "
            f"({MAX_LN_SM_COLS}). Both still open three and two fifos per core "
            f"and exhaust the shimNOC DMA capacity "
            f"(`no ShimNOCTile has sufficient DMA capacity`) -- they must be "
            f"joined hierarchically through the mem tiles first, the way GELU "
            f"was. Use a smaller --elt-cols, or rewrite the kernels' dataflow.")
    if n_cols > MAX_GELU_COLS:
        raise SystemExit(
            f"eltwise: {n_cols} columns is above the GELU wall "
            f"({MAX_GELU_COLS}).")
    if batch % 4:
        raise SystemExit(f"--batch {batch} must be a multiple of 4")

    cache_dir = (Path(cache_root).expanduser() / f"arch{arch}"
                 if per_arch_cache else Path(cache_root).expanduser())
    if per_arch_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["IRON_CACHE_DIR"] = str(cache_dir)

    # `ops is not None`, not `ops`: an EMPTY set is a request for nothing and is
    # refused below, while an omitted argument keeps this function's historical
    # "all three" default. `if ops` would have quietly turned the first into the
    # second.
    want = set(npu_ops.OPS) if ops is None else set(ops)
    unknown = want - set(npu_ops.OPS)
    if unknown:
        raise SystemExit(
            f"eltwise: unknown op(s) {sorted(unknown)}; valid codes: "
            f"[{npu_ops.CODES}]"
        )
    if not want:
        raise SystemExit(
            "eltwise: no ops requested, so there is nothing to build. Pass "
            f"{npu_ops.EXPORTER_FLAG} with at least one of [{npu_ops.CODES}]."
        )

    arr = _arrays()
    arr.iron.set_current_device(arr.from_name(device, n_cols=None))
    print(f"[eltwise arch {arch}] device={device} cache_dir={cache_dir} "
          f"ops={','.join(sorted(want))}")

    n_gelu = batch * seq * 4 * hidden
    ln_rows = batch * seq
    sm_rows = batch * HEADS * seq

    metas: list[dict] = []

    # -- GELU ----------------------------------------------------------------
    if "gelu" in want:
      gelu_sym = arr.gelu_symbols[gelu_tile]
      purge(cache_dir, [gelu_sym], n_gelu, f"gelu@b{batch}")
      X = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
      Y = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
      arr.gelu_array(X, Y, n_elem=n_gelu, n_cols=n_cols, tile=gelu_tile)
      src = find_cache(cache_dir, [gelu_sym], n_gelu, n_cols, f"gelu@b{batch}")
      metas.append(_write_dir(out_dir / "gelu", src, {
          "name": "gelu", "kind": "eltwise", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [n_gelu * 2, n_gelu * 2],
          "rows": n_gelu, "hidden": hidden, "seq": seq, "batch": batch,
          "cols": n_cols, "tile": gelu_tile}))

    # -- LayerNorm -----------------------------------------------------------
    if "layn" in want:
      ln_sym, _ = arr.ln_symbols[ln_variant]
      purge(cache_dir, [ln_sym], ln_rows * 384, f"layernorm@b{batch}")
      X = arr.iron.zeros(ln_rows * 384, dtype=bfloat16, device="npu")
      P = arr.iron.zeros(2 * 384, dtype=np.float32, device="npu")
      Y = arr.iron.zeros(ln_rows * 384, dtype=bfloat16, device="npu")
      arr.ln_array(X, P, Y, rows=ln_rows, n_cols=n_cols, variant=ln_variant)
      src = find_cache(cache_dir, [ln_sym], ln_rows * 384, n_cols,
                       f"layernorm@b{batch}")
      metas.append(_write_dir(out_dir / "layernorm", src, {
          "name": "layernorm", "kind": "layernorm", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [ln_rows * 384 * 2, 2 * 384 * 4, ln_rows * 384 * 2],
          "rows": ln_rows, "cols": 384, "hidden": hidden, "seq": seq,
          "batch": batch, "variant": ln_variant, "aie_cols": n_cols}))

    # -- softmax -------------------------------------------------------------
    if "softm" in want:
      sm_sym = arr.sm_symbols[sm_variant][0]
      purge(cache_dir, [sm_sym], sm_rows * 64, f"softmax@b{batch}")
      X = arr.iron.zeros(sm_rows * 64, dtype=bfloat16, device="npu")
      Y = arr.iron.zeros(sm_rows * 64, dtype=bfloat16, device="npu")
      arr.sm_array(X, Y, rows=sm_rows, n_cols=n_cols, variant=sm_variant)
      src = find_cache(cache_dir, [sm_sym], sm_rows * 64, n_cols,
                       f"softmax@b{batch}")
      metas.append(_write_dir(out_dir / "softmax", src, {
          "name": "softmax", "kind": "softmax", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [sm_rows * 64 * 2, sm_rows * 64 * 2],
          "rows": sm_rows, "cols": 64, "hidden": hidden, "seq": seq,
          "batch": batch, "variant": sm_variant, "aie_cols": n_cols}))

    return metas


def arch_out_dir(base: str | Path, arch: str) -> Path:
    """<base>/artifacts_npu<arch> -- the eltwise sibling of the GEMM set's root."""
    return Path(base) / f"artifacts_npu{arch}"


def build_child_argv(args, arch: str, outdir: Path) -> list[str]:
    # The child re-runs the HISTORIC entry point, not this module: --arch all
    # spawns one process per generation, and that process must go through the
    # same argparser the documented command uses.
    script = Path(__file__).resolve().parents[2] / "export_eltwise.py"
    cmd = [
        sys.executable or "python", str(script),
        "--arch", arch, "--out", str(outdir),
        "--batch", str(args.batch), "--seq", str(args.seq),
        "--hidden", str(args.hidden), "--elt-cols", str(args.elt_cols),
        "--gelu-tile", str(args.gelu_tile),
        "--ln-variant", args.ln_variant, "--sm-variant", args.sm_variant,
        "--cache-root", str(args.cache_root),
        # Forwarded verbatim: the child parses it with the same parser, so the
        # set cannot differ between the parent and the process that builds.
        "--extra-ops", getattr(args, "extra_ops", "gelu,layn,softm"),
    ]
    if args.per_arch_cache:
        cmd.append("--per-arch-cache")
    return cmd
