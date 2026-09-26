"""Turning --target plus tools/npu_targets.json into concrete per-arch arguments.

resolve_args is the only place that reads the targets file for a build, so a
geometry change lands in exactly one code path. model.overrides.* beats
defaults.* for that model, and the result is one ResolvedArch per generation.
"""

import argparse
import copy
from dataclasses import dataclass

from ..common.consts import (
    AIE_ROWS,
    ARCH_DEVICES,
    ARCHES,
    DEFAULT_SEQ,
    FALLBACK_BATCH,
    FALLBACK_COLS,
    FALLBACK_COLS_BY_ARCH,
    FALLBACK_HIDDEN,
    FALLBACK_IDENTITY_THRESHOLD,
    FALLBACK_K,
    FALLBACK_M,
    FALLBACK_N,
    KNOWN_DATAPATHS,
)
from ..common.targets import load_targets
from ..common.validate import parse_tiers
from .geometry import (STT_DECODE_ROWS, STT_DECODE_SEQ,
                       STT_DECODE_TIERS, STT_DECODE_TILE_M)

def _pick(*candidates: object) -> object:
    """First candidate that is not None (CLI beats overrides beats arch/default)."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def apply_datapath(
    ns: argparse.Namespace,
    datapath: str | None,
    defaults: dict | None = None,
) -> argparse.Namespace:
    """
    Derive the datapath flags from a target's `datapath` field.

    Mapping: 'bfp16' -> --emulate-bfp16 --c-bf16; 'bf16' -> --c-bf16 only.
    Flags explicitly passed on the command line are never overwritten.
    """
    default_c = None if defaults is None else defaults.get("c_bf16")

    if datapath == "bfp16":
        if ns.emulate_bfp16 is None:
            ns.emulate_bfp16 = True
        if ns.c_bf16 is None:
            ns.c_bf16 = True if default_c is None else bool(default_c)
    elif datapath == "bf16":
        if ns.c_bf16 is None:
            ns.c_bf16 = True if default_c is None else bool(default_c)
    elif datapath is not None:
        raise SystemExit(
            f"unknown datapath {datapath!r}; expected one of "
            f"{sorted(KNOWN_DATAPATHS)}"
        )

    if ns.c_bf16 is None:
        ns.c_bf16 = bool(default_c)

    return ns

@dataclass
class ResolvedArch:
    """One architecture with every compile argument resolved to a value."""

    arch: str
    args: argparse.Namespace


def resolve_args(args: argparse.Namespace) -> list[ResolvedArch]:
    """
    Resolve concrete compile arguments per architecture.

    Priority for every field: explicit CLI value, then model.overrides, then
    arches[arch], then defaults. Without --target the historic fallbacks are
    used, so a fully manual invocation is unchanged.
    """
    arches = list(ARCHES) if args.arch == "all" else [args.arch]

    targets: dict | None = None
    model_spec: dict | None = None
    if args.target:
        targets = load_targets(args.targets_file)
        model_spec = targets["models"].get(args.target)
        if model_spec is None:
            known = ", ".join(sorted(targets["models"]))
            raise SystemExit(
                f"--target {args.target!r} is not defined in "
                f"{args.targets_file}\nKnown targets: {known}"
            )

    resolved: list[ResolvedArch] = []

    for arch in arches:
        if arch not in ARCH_DEVICES:
            raise SystemExit(f"unknown architecture: {arch}")

        ns = copy.deepcopy(args)
        ns.arch = arch

        if targets is None:
            ns.batch = FALLBACK_BATCH if args.batch is None else args.batch
            ns.cols = FALLBACK_COLS_BY_ARCH.get(arch, FALLBACK_COLS) if args.cols is None else args.cols
            ns.hidden = FALLBACK_HIDDEN if args.hidden is None else args.hidden
            ns.seq = DEFAULT_SEQ if args.seq is None else args.seq
            ns.m = FALLBACK_M if args.m is None else args.m
            ns.k = FALLBACK_K if args.k is None else args.k
            ns.n = FALLBACK_N if args.n is None else args.n
            ns.rows = AIE_ROWS if args.rows is None else args.rows
            ns.tb_rows = args.tb_rows or 0
            # The manual path honours --stream-set rather than assuming the
            # encoder: the child process of an STT --arch all reaches this code
            # with no --target, and hardcoding the encoder set there would
            # silently rebuild the decoder's design as an encoder.
            ns.stream_set = getattr(args, "stream_set", None) or "gemm_rtp"
            ns.set_name = ns.stream_set
            ns.stt_tiers = ns.stream_set == "stt"

            ns.identity_threshold = (
                FALLBACK_IDENTITY_THRESHOLD
                if args.identity_threshold is None
                else args.identity_threshold
            )
            ns.gated_ffn = bool(args.gated_ffn)
            ns.int8 = bool(args.int8)
            ns.c_bf16 = bool(args.c_bf16)
            ns.emulate_bfp16 = bool(args.emulate_bfp16)
            resolved.append(ResolvedArch(arch, ns))
            continue

        assert targets is not None and model_spec is not None
        defaults = targets["defaults"]
        arch_spec = targets["arches"].get(arch)
        if arch_spec is None:
            raise SystemExit(
                f"--target {args.target!r}: architecture {arch} is not defined "
                f"in {args.targets_file}"
            )
        overrides = model_spec.get("overrides") or {}

        ns.batch = _pick(
            args.batch, overrides.get("batch"), arch_spec.get("batch"),
            FALLBACK_BATCH,
        )
        batches = _pick(
            args.batches, overrides.get("batches"), arch_spec.get("batches"),
        )
        if isinstance(batches, (list, tuple)):
            batches = ",".join(str(int(b)) for b in batches)
        ns.batches = batches
        ns.cols = _pick(
            args.cols, overrides.get("cols"), arch_spec.get("cols"),
            FALLBACK_COLS,
        )
        ns.hidden = _pick(
            args.hidden, overrides.get("hidden"), model_spec.get("hidden"),
            defaults.get("hidden"), FALLBACK_HIDDEN,
        )
        ns.intermediate = _pick(
            args.intermediate, overrides.get("intermediate"),
            model_spec.get("intermediate"),
        )
        ns.qkv_n = _pick(
            args.qkv_n, overrides.get("qkv_n"), model_spec.get("qkv_n"),
        )
        ns.gated_ffn = (
            args.gated_ffn
            if args.gated_ffn is not None
            else bool(model_spec.get("gated_ffn"))
        )
        ns.seq = _pick(
            args.seq, overrides.get("seq"), defaults.get("seq"), DEFAULT_SEQ,
        )
        ns.m = _pick(
            args.m, overrides.get("tile_m"), defaults.get("tile_m"), FALLBACK_M,
        )
        ns.k = _pick(
            args.k, overrides.get("tile_k"), defaults.get("tile_k"), FALLBACK_K,
        )
        ns.n = _pick(
            args.n, overrides.get("tile_n"), defaults.get("tile_n"), FALLBACK_N,
        )
        ns.rows = _pick(
            args.rows, overrides.get("rows"), defaults.get("rows"), AIE_ROWS,
        )
        ns.tb_rows = _pick(args.tb_rows, overrides.get("tb_rows"), 0)
        ns.identity_threshold = _pick(
            args.identity_threshold, overrides.get("identity_threshold"),
            defaults.get("identity_threshold"), FALLBACK_IDENTITY_THRESHOLD,
        )

        # int8 is never implied by a datapath; only an explicit flag sets it.
        ns.int8 = bool(args.int8)
        ns.c_bf16 = args.c_bf16
        ns.emulate_bfp16 = args.emulate_bfp16
        apply_datapath(ns, model_spec.get("datapath"), defaults)
        ns.stream_set = "gemm_rtp"
        ns.set_name = "gemm_rtp"
        # An STT model's unit of work is one audio file, so its ENCODER tiers are
        # utterance counts too, not text batches. See common/validate.py.
        ns.stt_tiers = model_spec.get("kind") == "stt"

        tiers = parse_tiers(ns)
        max_batch = arch_spec.get("max_batch")
        if max_batch is not None:
            offenders = sorted(
                {t for t in tiers if t > max_batch}
                | ({ns.batch} if ns.batch > max_batch else set())
            )
            if offenders:
                raise SystemExit(
                    f"--target {args.target} --arch {arch}: batch tier(s) "
                    f"{offenders} exceed arches[{arch}].max_batch={max_batch} "
                    f"(device {arch_spec.get('device')}).\n"
                    f"Lower --batch or --batches, or export with --arch 2."
                )

        resolved.append(ResolvedArch(arch, ns))

        if model_spec.get("kind") == "stt":
            resolved.append(_decoder_pass(args, arch, ns, model_spec, targets))

    return resolved


def _decoder_pass(args, arch, enc_ns, model_spec, targets) -> ResolvedArch:
    """The SECOND design set an STT model needs, in one invocation.

    An embedder's four streams cover its whole forward pass, so one design set
    is the whole model. Whisper needs two, and they are not variations of each
    other: the encoder runs M = frames (1536 here), the decoder runs M = tokens
    per step (8). One xclbin cannot serve both, and a design exported for the
    encoder's M would be 192x too large for a single decode step -- the waste is
    silent, because a bigger design still computes a correct answer.

    The decoder pass therefore overrides seq, m, rows and the tiering, and names
    its output directory differently so the two sets never overwrite each other.
    Explicit command-line geometry WINS over these defaults, because a caller
    who typed --seq has said what they want; only the unset fields are filled in.
    """
    ns = copy.deepcopy(enc_ns)
    ns.stream_set = "stt"
    ns.set_name = "gemm_rtp_dec"

    overrides = model_spec.get("overrides") or {}
    dec_seq = _pick(args.dec_seq, overrides.get("dec_seq"), STT_DECODE_SEQ)
    ns.seq = args.seq if args.seq is not None else dec_seq
    ns.m = _pick(args.m, overrides.get("dec_tile_m"), STT_DECODE_TILE_M)
    ns.rows = _pick(args.rows, overrides.get("dec_rows"), STT_DECODE_ROWS)

    dec_batches = _pick(overrides.get("dec_batches"), STT_DECODE_TIERS)
    if args.batches is not None:
        ns.batches = args.batches
    else:
        ns.batches = ",".join(str(int(b)) for b in dec_batches)
    ns.batch = max(int(b) for b in str(ns.batches).split(",") if b.strip())

    # The decoder's tiers are utterance/token counts, not text batches, so the
    # "multiple of 4" rule for the encoder's tiers does not apply to them. See
    # common/validate.py.
    ns.stt_tiers = True
    return ResolvedArch(arch, ns)
