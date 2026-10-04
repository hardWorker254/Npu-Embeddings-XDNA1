"""Writing one architecture's gelu / layernorm / softmax design directories."""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import npu_ops  # tools/lib/ is on sys.path (the shim and consts.py put it
                # there) -- see the module's own header for why this table is
                # written twice.
from ..common.consts import ARCH_DEVICES, DEFAULT_CACHE_ROOT, DEFAULT_SEQ
from .cache import find_cache, purge
from .consts import (GELU_TILE, HEADS, MAX_GELU_COLS, MAX_LN_SM_COLS,
                     SM_ROWS_PER_CALL)
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

def ln_rows_per_call(cols: int) -> int:
    """Rows per core call that fit L1, for a row of `cols` columns.

    L1 is 64 KB and the kernel's ping-pong buffers are 2 * (rows * cols * 2 B)
    for the input plus the same for the output, so a block has to satisfy
    4 * rows * cols <= 65536 with room to spare -- 48 KB is the ceiling this
    uses, which is where MiniLM's own sixteen rows of 384 sit (49152 B). Sixteen
    rows of 1280 columns would need 160 KB, so the design is exported with four
    rows per call instead. Only multiples of four are candidates: the body
    interleaves four rows and cannot do fewer.

    This is a property of the HARDWARE, not a taste call, and it is the reason
    whisper-large-v3 cannot reuse whisper-tiny's LayerNorm design: same kernel,
    different LN_ROWS, and a different symbol with it.
    """
    for r in (16, 8, 4):
        if 4 * r * cols * 2 <= 48 * 1024:
            return r
    raise SystemExit(
        f"eltwise: a LayerNorm row of {cols} columns does not fit L1 even at "
        f"four rows per call (4 * 4 * {cols} * 2 B > 48 KB of ping-pong). This "
        f"kernel cannot take that width; a wider model needs its own kernel, "
        f"not a smaller batch.")


def export_arch(out_dir: Path, arch: str, batch: int, hidden: int = 384,
                seq: int = DEFAULT_SEQ, n_cols: int = 1, gelu_tile: int = GELU_TILE,
                ln_variant: str = "il4", sm_variant: str = "poly_il4",
                cache_root: Path | None = DEFAULT_CACHE_ROOT,
                per_arch_cache: bool = False,
                ln_cols: int | None = None, ln_eps: float = 1e-12,
                sm_cols: int | None = None, sm_rows: int | None = None,
                gelu_variant: str = "poly",
                ops: set[str] | None = None) -> list[dict]:
    """Build the requested eltwise designs into `out_dir` for one generation.

    `ops` is the set of CODES from tools/lib/npu_ops.py (gelu, layn, softm); an
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
            f"--extra-ops with at least one of [{npu_ops.CODES}] -- not "
            f"{npu_ops.EXPORTER_FLAG}, which this tool refuses by name: that one "
            "belongs to the runtime and only selects among designs that were "
            "already built."
        )

    arr = _arrays()
    arr.iron.set_current_device(arr.from_name(device, n_cols=None))
    print(f"[eltwise arch {arch}] device={device} cache_dir={cache_dir} "
          f"ops={','.join(sorted(want))}")

    n_gelu = batch * seq * 4 * hidden
    ln_rows = batch * seq
    # The softmax design's row CAPACITY, and it is the one number that decides
    # how much traffic an attention dispatch costs: every dispatch fills and drains
    # the whole buffer, so a design with 6144 rows syncs 37 MB to move the 512
    # rows a query chunk holds. `sm_rows` is the caller's when the caller knows
    # what one dispatch is (Whisper: the query chunk, batch*seq); the
    # batch*heads*seq formula is the EMBEDDER's attention, where a dispatch really
    # is batch*heads rows wide.
    sm_rows = batch * HEADS * seq if sm_rows is None else sm_rows

    # What has to divide is each op's OWN row count, not `batch`. Whisper's unit
    # of work is one audio file, so its batch tier is 1 and its LayerNorm still
    # has 512 rows to spread over four cores -- a rule stated on `batch` refused
    # the one model that needed the design most. Each refusal below names the
    # divisor it wanted and the two flags that move it.
    n_cores = 4 * n_cols
    if "gelu" in want:
        if n_gelu % gelu_tile:
            raise SystemExit(
                f"eltwise: {n_gelu} GELU elements are not a multiple of the "
                f"{gelu_tile}-element tile. Re-export with a --batch/--seq/"
                f"--hidden whose product is.")
        if (n_gelu // gelu_tile) % n_cores:
            raise SystemExit(
                f"eltwise: {n_gelu // gelu_tile} GELU tiles do not spread over "
                f"{n_cores} cores. Re-export with a larger --batch/--seq, or a "
                f"smaller --elt-cols.")
    if "softm" in want and sm_cols is None and \
            sm_rows % (SM_ROWS_PER_CALL * n_cores):
        raise SystemExit(
            f"eltwise: {sm_rows} softmax rows do not split into "
            f"{SM_ROWS_PER_CALL}-row blocks over {n_cores} cores. Re-export with "
            f"a --batch/--seq whose product is a multiple of "
            f"{SM_ROWS_PER_CALL * n_cores}.")

    metas: list[dict] = []

    # -- GELU ----------------------------------------------------------------
    if "gelu" in want:
      if gelu_variant not in arr.gelu_symbols and gelu_variant not in ("poly", "erf"):
        raise SystemExit(
            f"eltwise: unknown GELU variant {gelu_variant!r}; expected poly or "
            f"erf")
      syms = arr.gelu_erf_symbols if gelu_variant == "erf" else arr.gelu_symbols
      if gelu_tile not in syms:
        raise SystemExit(
            f"eltwise: the {gelu_variant} GELU has no {gelu_tile}-element tile; "
            f"it has {sorted(syms)}")
      gelu_sym = syms[gelu_tile]
      purge(cache_dir, [gelu_sym], n_gelu, f"gelu@b{batch}")
      X = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
      Y = arr.iron.zeros(n_gelu, dtype=bfloat16, device="npu")
      arr.gelu_array(X, Y, n_elem=n_gelu, n_cols=n_cols, tile=gelu_tile,
                     variant=gelu_variant)
      src = find_cache(cache_dir, [gelu_sym], n_gelu, n_cols, f"gelu@b{batch}")
      # One row of n_gelu elements, which is what an elementwise pass over a
      # flat activation buffer is: the same (row_capacity, cols) contract the
      # LayerNorm and softmax designs write, so one runtime class reads all
      # three. `cols` is therefore the WIDTH and `aie_cols` the array columns,
      # the same split the other two use.
      metas.append(_write_dir(out_dir / "gelu", src, {
          "name": "gelu", "kind": "eltwise", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [n_gelu * 2, n_gelu * 2],
          "rows": n_gelu, "row_capacity": 1, "cols": n_gelu,
          "hidden": hidden, "seq": seq, "batch": batch,
          "aie_cols": n_cols, "tile": gelu_tile, "variant": gelu_variant}))

    # -- LayerNorm -----------------------------------------------------------
    if "layn" in want:
      # The width is the MODEL's d_model, not this build's 384, and the epsilon
      # is the checkpoint's `layer_norm_eps` (1e-5 for Whisper, 1e-12 for
      # MiniLM). Both are compiled into the kernel and both are recorded in
      # design.json, so the runtime can refuse a design whose eps is not the one
      # the container it is holding was packed with.
      lcols = hidden if ln_cols is None else ln_cols
      if lcols % 16:
        raise SystemExit(
            f"eltwise: a LayerNorm row of {lcols} columns is not a multiple of "
            f"16, and the kernel reduces a row in 16-wide vectors. Re-export "
            f"with a --hidden/--ln-cols that is.")
      rpc = ln_rows_per_call(lcols)
      if ln_rows % (rpc * 4 * n_cols):
        raise SystemExit(
            f"eltwise: {ln_rows} LayerNorm rows do not split into {rpc}-row "
            f"blocks over {4 * n_cols} cores (a {lcols}-column row needs {rpc} "
            f"rows per call). Re-export with a --batch/--seq whose product is a "
            f"multiple of {rpc * 4 * n_cols}.")
      # The variant names the row count, so the JIT cache can tell a 16-row
      # design from a 4-row one with the same buffer sizes.
      if "il4" in ln_variant and rpc != 16:
        ln_variant = f"il4_{rpc}"
      ln_sym, _ = arr.ln_symbols[ln_variant]
      purge(cache_dir, [ln_sym], ln_rows * lcols, f"layernorm@b{batch}")
      X = arr.iron.zeros(ln_rows * lcols, dtype=bfloat16, device="npu")
      P = arr.iron.zeros(2 * lcols, dtype=np.float32, device="npu")
      Y = arr.iron.zeros(ln_rows * lcols, dtype=bfloat16, device="npu")
      arr.ln_array(X, P, Y, rows=ln_rows, n_cols=n_cols, variant=ln_variant,
                   cols=lcols, eps=ln_eps, rows_per_call=rpc)
      src = find_cache(cache_dir, [ln_sym], ln_rows * lcols, n_cols,
                       f"layernorm@b{batch}")
      metas.append(_write_dir(out_dir / "layernorm", src, {
          "name": "layernorm", "kind": "layernorm", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [ln_rows * lcols * 2, 2 * lcols * 4, ln_rows * lcols * 2],
          "rows": ln_rows, "cols": lcols, "eps": ln_eps,
          "row_capacity": ln_rows, "ln_eps": ln_eps,
          "rows_per_call": rpc,
          "hidden": hidden, "seq": seq,
          "batch": batch, "variant": ln_variant, "aie_cols": n_cols}))

    # -- softmax -------------------------------------------------------------
    if "softm" in want:
      # A Whisper score row is n_kv wide, not a sequence length, so the width
      # and the kernel both come from the caller: 64 columns is the shipped
      # kernel's own constant and holds an embedder's attention, while 1536 needs
      # softmax_w.cc and one row per call. The rows are the model's: the encoder
      # dispatches one softmax per (query chunk, head), so its capacity is the
      # chunk size, and the decoder's smaller tiers fit inside it.
      cols = 64 if sm_cols is None else sm_cols
      variant = sm_variant
      if cols != 64:
        if variant in ("lib", "poly", "poly_il4", "poly_rne", "poly_il4_rne"):
          raise SystemExit(
              f"eltwise: a {cols}-column softmax row is not something "
              f"kernels/softmax.cc can hold (its SM_COLS is 64, compiled in), "
              f"so --sm-variant {variant} would build the wrong kernel. Use "
              f"--sm-variant wide, or --sm-cols 64 for an embedder's attention.")
        rpc = 2 if variant == "wide2" else 1
        if sm_rows % (rpc * n_cores):
          raise SystemExit(
              f"eltwise: {sm_rows} softmax rows do not split into {rpc}-row "
              f"blocks over {n_cores} cores. Re-export with a --batch/--seq "
              f"whose product is a multiple of {rpc * n_cores}.")
      sm_sym = arr.sm_symbols[variant][0]
      n_elem = sm_rows * cols
      purge(cache_dir, [sm_sym], n_elem, f"softmax@b{batch}")
      X = arr.iron.zeros(n_elem, dtype=bfloat16, device="npu")
      Y = arr.iron.zeros(n_elem, dtype=bfloat16, device="npu")
      arr.sm_array(X, Y, rows=sm_rows, n_cols=n_cols, variant=variant,
                   cols=cols)
      src = find_cache(cache_dir, [sm_sym], n_elem, n_cols,
                       f"softmax@b{batch}")
      metas.append(_write_dir(out_dir / "softmax", src, {
          "name": "softmax", "kind": "softmax", "kernel": "MLIR_AIE",
          "arch": int(arch), "device": device,
          "a_dtype": "bf16", "c_dtype": "bf16",
          "buffers": [n_elem * 2, n_elem * 2],
          "rows": sm_rows, "row_capacity": sm_rows, "cols": cols,
          "hidden": hidden, "seq": seq,
          "batch": batch, "variant": variant, "aie_cols": n_cols}))

    return metas


def arch_out_dir(base: str | Path, arch: str) -> Path:
    """<base>/artifacts_npu<arch> -- the eltwise sibling of the GEMM set's root."""
    return Path(base) / f"artifacts_npu{arch}"


def build_child_argv(args, arch: str, outdir: Path) -> list[str]:
    # The child re-runs the HISTORIC entry point, not this module: --arch all
    # spawns one process per generation, and that process must go through the
    # same argparser the documented command uses.
    # tools/export/, one level above the exporters/ package this module is in.
    script = Path(__file__).resolve().parents[2] / "export_eltwise.py"
    cmd = [
        sys.executable or "python", str(script),
        "--arch", arch, "--out", str(outdir),
        "--batch", str(args.batch), "--seq", str(args.seq),
        "--hidden", str(args.hidden), "--elt-cols", str(args.elt_cols),
        "--gelu-tile", str(args.gelu_tile),
        "--gelu-variant", args.gelu_variant,
        "--ln-variant", args.ln_variant, "--sm-variant", args.sm_variant,
        "--ln-eps", repr(args.ln_eps),
        *(["--ln-cols", str(args.ln_cols)] if args.ln_cols is not None else []),
        *(["--sm-cols", str(args.sm_cols)] if args.sm_cols is not None else []),
        "--cache-root", str(args.cache_root),
        # Forwarded verbatim: the child parses it with the same parser, so the
        # set cannot differ between the parent and the process that builds.
        "--extra-ops", getattr(args, "extra_ops", "gelu,layn,softm"),
    ]
    if args.per_arch_cache:
        cmd.append("--per-arch-cache")
    return cmd
