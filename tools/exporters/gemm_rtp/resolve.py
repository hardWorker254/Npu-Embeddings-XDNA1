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
    FALLBACK_LN_EPS,
    FALLBACK_M,
    FALLBACK_N,
    KNOWN_DATAPATHS,
)
import npu_ops
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
            # Manual invocations have no target, so the LayerNorm design gets
            # the historic values: this build's width and MiniLM's epsilon.
            ns.ln_cols = args.hidden if args.ln_cols is None else args.ln_cols
            ns.ln_eps = FALLBACK_LN_EPS if args.ln_eps is None else args.ln_eps
            # No target, so no head count and no window: the attention streams
            # are Whisper's and a manual invocation has neither. An embedder's
            # softmax is the 64-column one it has always been.
            ns.attn_streams = ()
            ns.attn_geometry = None
            ns.mel_geometry = None
            ns.fft_geometry = None
            ns.logit_geometry = None
            ns.sm_cols = None
            ns.sm_rows = args.sm_rows
            ns.sm_variant = args.sm_variant
            ns.gelu_variant = args.gelu_variant
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
        # The LayerNorm design is the ONE eltwise design whose geometry is a
        # property of the model rather than of the exporter: its row is d_model
        # wide and its epsilon is the checkpoint's layer_norm_eps. Both are
        # compiled into the kernel, so both come from the target and both are
        # recorded in design.json for the runtime to check against the
        # container. MiniLM's 1e-12 is the fallback, and it is wrong for
        # Whisper by four orders of magnitude inside a square root.
        # Whisper's attention as two GEMMs, and the softmax row they produce.
        # n_kv is the model's own window, PADDED to the tile: the B panel has to
        # tile evenly and 1500 does not divide by 32. Both the encoder (which
        # attends over all `frames` positions) and the decoder (whose
        # cross-attention cache is the same tensor) read the same padded width,
        # which is why one number comes from the target and not from each pass.
        # A Whisper checkpoint declares `activation: gelu` and the packer
        # refuses anything else, so its GELU design is the exact-erf kernel. The
        # poly one is 2.49e-3 away from it, which is a different activation
        # rather than a faster one.
        ns.gelu_variant = ("erf" if model_spec.get("kind") == "stt"
                           else args.gelu_variant)
        ns.attn_streams = ()
        ns.attn_geometry = None
        ns.mel_geometry = None
        ns.fft_geometry = None
        ns.logit_geometry = None
        ns.sm_cols = None
        ns.sm_rows = args.sm_rows
        ns.sm_variant = args.sm_variant
        if model_spec.get("kind") == "stt":
            # Padded to the LEAST COMMON MULTIPLE of the two tiles, not to
            # tile_n alone: attn_qk's N only has to divide by tile_n, but
            # attn_av's K is the same n_kv and has to divide by tile_k as well,
            # and 1500 rounds to 1504 against 32 -- which attn_av cannot tile.
            # 1536 is the number both of them accept.
            import math as _math
            step = _math.lcm(int(ns.n), int(ns.k))
            n_kv = -(-int(model_spec.get("frames", 1500)) // step) * step
            if "attn" in npu_ops.parse_ops(args.npu_extra_ops or "",
                                           npu_ops.EXPORTER_FLAG):
                ns.attn_streams = npu_ops.GEMM_STREAMS["attn"]
                ns.attn_geometry = (int(model_spec["head_dim"]), n_kv)
            if "mproj" in npu_ops.parse_ops(args.npu_extra_ops or "",
                                             npu_ops.EXPORTER_FLAG):
                # (n_bins, n_mels) of the slaney bank: 201 frequency bins, and
                # the model's own mel count. Both belong to the front end, not
                # to the encoder, which is why neither is an override.
                ns.attn_streams = ns.attn_streams + npu_ops.GEMM_STREAMS["mproj"]
                ns.mel_geometry = (201, int(model_spec["mel_bins"]))
            if "fft" in npu_ops.parse_ops(args.npu_extra_ops or "",
                                           npu_ops.EXPORTER_FLAG):
                # (n_fft, n_bins) of the front end's transform: Whisper's own
                # 400 and its own 201 half-spectrum bins. Neither is an encoder
                # number, which is why they are read here and not from
                # overrides.
                ns.attn_streams = ns.attn_streams + npu_ops.GEMM_STREAMS["fft"]
                ns.fft_geometry = (400, 201)
            if "logit" in npu_ops.parse_ops(args.npu_extra_ops or "",
                                             npu_ops.EXPORTER_FLAG):
                if "vocab" not in model_spec:
                    raise SystemExit(
                        f"--target {args.target}: --npu-extra-ops logit needs the "
                        f"model's `vocab` (the tied embedding's columns), and "
                        f"this target does not carry one. Add it from the "
                        f"checkpoint's config.json.")
                ns.logit_geometry = (int(model_spec["hidden"]),
                                     int(model_spec["vocab"]),
                                     len(npu_ops.GEMM_STREAMS["logit"]))
            if "softm" in npu_ops.parse_ops(args.npu_extra_ops or "",
                                            npu_ops.EXPORTER_FLAG):
                # A Whisper score row is n_kv wide, which kernels/softmax.cc
                # cannot hold at 64 columns, so the width and the kernel travel
                # together: asking for a wide row with the shipped variant is
                # refused rather than quietly building a 64-column design.
                ns.sm_cols = n_kv
                ns.sm_variant = "wide"
                # One attention dispatch is one QUERY CHUNK, so the softmax
                # design needs the chunk and not the embedder's batch*heads*seq:
                ns.sm_rows = int(ns.batch) * int(ns.seq)
        ns.ln_cols = _pick(args.ln_cols, model_spec.get("layer_norm_cols"),
                           model_spec.get("hidden"), defaults.get("hidden"),
                           FALLBACK_HIDDEN)
        ns.ln_eps = _pick(args.ln_eps, model_spec.get("layer_norm_eps"),
                          defaults.get("layer_norm_eps"), FALLBACK_LN_EPS)

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
    # The mel bank and the transform are FRONT-END streams: they run once per
    # window, against the encoder set, and nothing in the decoder dispatches
    # them. Carried into this pass they would cost two streams x three tiers of
    # compiles for operands no instruction ever binds.
    ns.attn_streams = tuple(s for s in ns.attn_streams
                            if s in npu_ops.GEMM_STREAMS["attn"])
    ns.mel_geometry = None
    ns.fft_geometry = None
    # ... and the vocabulary projection belongs to THIS pass, at the step tier's
    # M: a step is one query row and the projection is the only thing that
    # consumes it.
    if ns.logit_geometry is not None:
        ns.attn_streams = ns.attn_streams + npu_ops.GEMM_STREAMS["logit"]

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
