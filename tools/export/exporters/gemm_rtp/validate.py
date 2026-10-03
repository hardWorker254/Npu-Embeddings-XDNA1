"""Geometry validation specific to the GEMM exporter."""

import argparse

from .geometry import shapes_for, shapes_for_stream_set


def validate_geometry(args: argparse.Namespace, tiers: list[int]) -> None:
    """
    Fail early if any shape is not tileable with the requested m/k/n/cols.

    The original script relied on asserts inside pretiled_array. That is fine,
    but late: it can die after several expensive builds. This check is cheap
    and gives clearer errors.

    The shapes come from shapes_for_stream_set() -- the SAME dispatch point
    build.py uses -- and not from shapes_for() directly. Calling shapes_for()
    here meant this function checked the EMBEDDER's four streams whatever the
    stream set was: a pose export was validated against qkv/attn_out/ffn_up/
    ffn_down at the embedder's M = batch*seq, and the first thing it complained
    about was `stream qkv: M=64`, a stream the pose export does not contain and
    a shape it will never build. Two functions deciding what the streams are is
    one too many, and this one was silently describing a different network.
    """
    stream_set = getattr(args, "stream_set", None) or "gemm_rtp"
    for b in tiers:
        _, shapes = shapes_for_stream_set(
            stream_set,
            b,
            args.hidden,
            args.intermediate,
            gated=args.gated_ffn,
            qkv_n=args.qkv_n,
            seq=args.seq,
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
