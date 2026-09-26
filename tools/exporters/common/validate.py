"""Argument validation that does not depend on which exporter is running."""

import argparse

from .consts import DEFAULT_SEQ

def validate_positive_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.cols <= 0:
        raise SystemExit("--cols must be positive")
    if args.hidden <= 0:
        raise SystemExit("--hidden must be positive")
    if args.seq <= 0:
        raise SystemExit("--seq must be positive")
    if args.m <= 0:
        raise SystemExit("-m must be positive")
    if args.k <= 0:
        raise SystemExit("-k must be positive")
    if args.n <= 0:
        raise SystemExit("-n must be positive")
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    if args.identity_threshold < 0:
        raise SystemExit("--identity-threshold must be >= 0")

    if args.intermediate is not None and args.intermediate <= 0:
        raise SystemExit("--intermediate must be positive")

    if args.qkv_n is not None and args.qkv_n <= 0:
        raise SystemExit("--qkv-n must be positive")

def parse_tiers(args: argparse.Namespace) -> list[int]:
    if args.batches:
        try:
            tiers = [int(x) for x in args.batches.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"--batches must be comma-separated integers: {exc}") from exc
    else:
        tiers = [args.batch]

    tiers = sorted(set(tiers))

    if not tiers:
        raise SystemExit("no batch tiers specified")

    return tiers

def validate_tiers_and_seq(args: argparse.Namespace) -> list[int]:
    validate_positive_args(args)

    tiers = parse_tiers(args)

    # A text batch of 4 is the floor for an embedder: the design was built around
    # four AIE rows and a tier below that left them idle. An STT model's unit of
    # work is one AUDIO FILE, so its tiers are 1 and the rule does not apply --
    # `stt_tiers` is set by resolve.py for the decoder pass, and the real
    # constraint (M must tile into m*rows) is checked right below either way.
    floor = 1 if getattr(args, "stt_tiers", False) else 4
    for b in tiers:
        if b <= 0:
            raise SystemExit(f"batch tier {b}: must be positive")
        if b % floor:
            raise SystemExit(f"batch tier {b}: must be a multiple of {floor}")

    if max(tiers) != args.batch:
        raise SystemExit(
            f"--batch {args.batch} must be the largest tier; tiers are {tiers}"
        )

    if args.seq % 8:
        raise SystemExit(
            f"--seq {args.seq}: must be positive and a multiple of 8 "
            f"(the runtime refuses anything else)"
        )

    # M % (m * rows) == 0
    for b in tiers:
        M = b * args.seq
        if M % (args.m * args.rows):
            raise SystemExit(
                f"--seq {args.seq} x batch tier {b} gives M = {M}, "
                f"which is not a multiple of m*rows = {args.m * args.rows}. "
                f"Pick a tier, a seq, -m or --rows whose product divides it."
            )

    if args.seq != DEFAULT_SEQ:
        print(
            f"  seq        {args.seq} (default {DEFAULT_SEQ}). "
            f"M = batch*seq, so tiers {tiers} give M {[b * args.seq for b in tiers]}."
        )
        print(
            "             Host attention is O(seq^2) and this repo has no "
            "measurement above seq 64. Treat this design's throughput as "
            "unknown until it is traced."
        )

    return tiers

def _require_int(
    obj: dict,
    key: str,
    ctx: str,
    minimum: int = 1,
    allow_none: bool = False,
) -> int | None:
    if key not in obj:
        raise SystemExit(f"{ctx}: missing '{key}'")
    value = obj[key]
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SystemExit(
            f"{ctx}: '{key}' must be an integer >= {minimum}, got {value!r}"
        )
    return value
