"""Geometry validation specific to the GEMM exporter."""

import argparse

from .geometry import shapes_for

def validate_geometry(args: argparse.Namespace, tiers: list[int]) -> None:
    """
    Fail early if any shape is not tileable with the requested m/k/n/cols.

    The original script relied on asserts inside pretiled_array. That is fine,
    but late: it can die after several expensive builds. This check is cheap
    and gives clearer errors.
    """
    for b in tiers:
        shapes = shapes_for(
            b,
            args.hidden,
            args.intermediate,
            args.gated_ffn,
            args.qkv_n,
            args.seq,
        )

        for name, sh in shapes.items():
            M = sh["M"]
            K = sh["K"]
            N = sh["N"]

            if M % (args.m * args.rows):
                raise SystemExit(
                    f"batch {b}, stream {name}: M={M} is not divisible by "
                    f"m*rows={args.m * args.rows}"
                )

            if K % args.k:
                raise SystemExit(
                    f"batch {b}, stream {name}: K={K} is not divisible by k={args.k}"
                )

            if N % (args.n * args.cols):
                raise SystemExit(
                    f"batch {b}, stream {name}: N={N} is not divisible by "
                    f"n*cols={args.n * args.cols}"
                )
