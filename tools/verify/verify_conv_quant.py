#!/usr/bin/env python3
# NpuEmbeddings -- gate a quantised pose container against its checkpoint.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS IS, AND WHY IT IS NOT THE PACKER
# -----------------------------------------
# conv_quant.check_conv_w is the inverse of conv_quant.add_conv_w, and the two
# live in the same file on purpose -- the scheme has one definition. The GATE
# does not, and that is the point of it being here instead of at the end of
# pack_pose: this reads the container back OFF DISK with npue.Reader and checks
# the bytes the writer produced, not the arrays the writer was handed. The writer
# may re-interpret, pad, transpose or reorder on the way to the file -- the
# four-bit nibble packing, the [G, N] group-scale transpose and the odd-K pad
# column all happen inside it -- and every one of those is a place where what
# was computed and what was stored can come apart. A packer checking its own
# arguments would pass on all three.
#
# WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
# Per convolution, against the ONNX checkpoint's own weights:
#
#   1. the payload, BIT-EXACTLY against rint(W * factor) clipped -- which catches
#      a transposed tensor, a swapped nibble order, a group boundary off by one
#      and a scale applied on the wrong axis, and which a tolerance would blunt
#      all four of;
#   2. the scales' shapes and signs, because both are multiplied (a negative one
#      is a sign flip, not an error) and a zero wscale is a divide by zero at
#      load;
#   3. for i4, that gscale * wscale IS the source's own per-group max|K|/7 --
#      which is the invariant that a ratio-quantised-everywhere container would
#      satisfy internally and still be wrong;
#   4. the DEQUANTISED weight against the checkpoint's, relative Frobenius, so
#      the number the packer reported is confirmed rather than trusted.
#
# It does NOT re-derive the scales from the checkpoint and compare them to the
# file's. That would measure this script's own copy of the packer's arithmetic.
# It does NOT check that the runtime agrees -- that is verify_pose_quant.py, on
# detections, because the only error an operator can see is a detection.
#
# It DOES check one thing no numpy function can: that the C++ reader's
# dequantisation is the same arithmetic. `rt_reads` below is a line-by-line
# transcription of runtime/src/pose/geometry.cpp's i8 and i4 paths, and it has
# to agree with conv_quant.dequantise BIT for BIT. Two implementations of one
# scheme that agree to 1e-7 would still disagree in the last place the last
# weight lands, and there is no test of the C++ reader here that would notice.
#
# Env: numpy + onnx.
#
# Usage:
#   python3 tools/verify/verify_conv_quant.py models/pose_i8.npue model.onnx
#   python3 tools/verify/verify_conv_quant.py models/pose_i4.npue model.onnx --verbose

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

import conv_quant                                    # noqa: E402
from npue import Reader                              # noqa: E402


def rt_reads(payload, wscale, gscale, N, K, dtype, group):
    """runtime/src/pose/geometry.cpp's dequantisation, transcribed.

    Deliberately naive: one element at a time, the same nibble extraction, the
    same sign extension, the same `k / group` and the same [G, N] stride. It is
    a transcription to be COMPARED, not an implementation to be fast, and the
    obvious vectorised version of it is the thing most likely to differ from the
    C++ in exactly the way this is looking for.
    """
    s = np.asarray(wscale, dtype=np.float32)
    if dtype == "i8":
        q = np.asarray(payload, dtype=np.int8).reshape(N, K)
        return q.astype(np.float32) * s[:, None]
    p = np.asarray(payload, dtype=np.uint8).reshape(N, -1)
    gs = np.asarray(gscale, dtype=np.float32).reshape(-1)
    gsz = group if group else K
    out = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            v = ((int(p[n, k // 2]) >> 4) & 0xF) if (k & 1) \
                else (int(p[n, k // 2]) & 0xF)
            if v >= 8:
                v -= 16
            out[n, k] = v * s[n] * gs[(k // gsz) * N + n]
    return out


def checkpoint_weights(onnx_path):
    """initialiser name -> [Cout, Cin, kh, kw] fp32, for every Conv.

    Keyed by the INITIALISER, not by the node. The container's `conv_sources`
    names initialisers, and the two are not interchangeable: the exporter sees
    `/model.22/cv2.1/cv2.1.2/Conv` and its weight `/model.22/cv2.1/cv2.1.2.conv.weight`,
    and a checker keyed on nodes would resolve neither.

    Read straight from the protobuf rather than through a framework: the shapes
    are what this checks against, and a framework that upcasts or transposes on
    the way in would make the check circular.
    """
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(onnx_path))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    out = {}
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        name = node.input[1]
        w = init.get(name)
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 4:
            out[name] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("onnx")
    ap.add_argument("--verbose", action="store_true",
                    help="one line per convolution, not just the worst")
    args = ap.parse_args()

    src = checkpoint_weights(args.onnx)
    if not src:
        raise SystemExit(f"{args.onnx}: no rank-4 Conv initialisers. That is not "
                         f"a YOLOv8-pose checkpoint.")

    with Reader(args.container) as r:
        dtype = r.config.get("conv_weight_dtype")
        group = int(r.config.get("conv_int4_group", 0) or 0)
        n_convs = int(r.config.get("num_convs", 0) or 0)
        print(f"{args.container}")
        print(f"  conv_weight_dtype {dtype!r}, conv_int4_group {group}, "
              f"{n_convs} convolutions")
        if dtype == "f32":
            print("  an fp32 container has nothing to check: the payload is the "
                  "checkpoint's own bytes. (verify_pose.py checks the runtime "
                  "against the ONNX node by node; this checks the SCHEME.)")
            return 0

        # The container's weights in the packer's order, so a mismatch between
        # the two is reported as a tensor rather than as a count.
        names = json_names(r)
        if len(names) != n_convs:
            raise SystemExit(
                f"{args.container}: the config says {n_convs} convolutions and "
                f"the directory holds {len(names)} conv.*.w entries. One of them "
                f"was not written, and the runtime would read past the end of "
                f"the shorter list rather than notice.")
        # The checkpoint tensor each one came from. Required for a quantised
        # container and refused without it, because the fallback -- matching by
        # shape -- is not a weaker check but a WRONG one on this architecture:
        # see conv_sources' comment in the packer. Six of these 73 convolutions
        # have a shape-identical sibling.
        raw_src = r.config.get("conv_sources")
        if raw_src is None:
            raise SystemExit(
                f"{args.container}: no config conv_sources, so there is no way "
                f"to tell which checkpoint weight each conv.N.w is. Matching by "
                f"shape would gate six convolutions against the wrong sibling's "
                f"weights and report packing bugs that are not there. Repack "
                f"with tools/pack/packers/pose.py, which records it.")
        import json as _json
        sources = _json.loads(raw_src)
        if len(sources) != n_convs:
            raise SystemExit(
                f"{args.container}: conv_sources lists {len(sources)} names for "
                f"{n_convs} convolutions.")

        worst = (0.0, "")
        checked = 0
        for i, base in enumerate(names):
            e = r.entries[f"{base}.w"]
            shp = list(e["logical_shape"])
            if len(shp) != 4:
                raise SystemExit(f"{base}.w: logical_shape {shp}, not rank 4. "
                                 f"The whole format is [Cout, Cin, kh, kw] and "
                                 f"every index in the reader depends on it.")
            N, K = shp[0], shp[1] * shp[2] * shp[3]
            src_name, w = conv_src(src, sources[i], N, K, base, args.container)
            # `.view(int8)`, not `.astype(int8)`: the bytes on disk ARE the
            # two's-complement values, and astype would first map -49 to 207 and
            # then back -- which round-trips, except at the one value the
            # comparison would flag: a stored -49 read as +49.
            pay = r.payload(f"{base}.w")
            pay = pay.view(np.int8) if e["dtype"] == "I8" else pay
            # [G, N], not flat: check_conv_w asks the shape, because a gscale
            # read with any other stride is every weight past the first group
            # wrong and every other check still passing. And the payload is
            # [N, ceil(K/2)] for the same reason -- both are 2-D things that
            # `payload()` hands back flat, and a flat array has no shape to
            # disagree with.
            gs = None
            G = 1
            if dtype == "i4":
                g = group if group else K
                G = (K + g - 1) // g
                gs = r.raw(f"{base}.gscale").reshape(G, N)
                pay = pay.reshape(N, (K + 1) // 2)
            conv_quant.check_conv_w(base, pay, r.raw(f"{base}.wscale"), gs,
                                    w, dtype, group)
            # BOTH SIDES OF THE READER COMPARISON READ THE FILE. The scheme
            # applied to the container's own payload and the container's own
            # scales, against a transcription of geometry.cpp doing the same.
            # It used to compare against `dequantise(w, ...)`, which re-derives
            # the scales from the checkpoint: that is a different float32
            # pipeline, and a container whose stored scale sits one ulp from a
            # recomputation was reported as a reader bug on 88 of the stem's 432
            # weights with both readers agreeing exactly. The quantisation error
            # below is still measured against the source, which is what it is
            # for.
            deq = conv_quant.dequantise_payload(pay, r.raw(f"{base}.wscale"),
                                                gs, N, K, dtype, group)
            rt = rt_reads(pay, r.raw(f"{base}.wscale"), gs,
                          N, K, dtype, group)
            if not np.array_equal(rt, deq):
                bad = np.argwhere(rt != deq)
                n, k = (int(x) for x in bad[0])
                raise SystemExit(
                    f"{base}: the C++ reader and the packer disagree at [{n}, "
                    f"{k}] -- runtime {rt[n, k]!r}, packer {deq[n, k]!r}, "
                    f"{len(bad)} of {N * K} values. The scheme is agreed by this "
                    f"script; the two implementations of it are not, and only "
                    f"one of them is compiled into the binary.")
            src_deq = conv_quant.dequantise(w, dtype, group)
            den = float(np.linalg.norm(w))
            rel = float(np.linalg.norm(src_deq - w) / den) if den else 0.0
            checked += 1
            if rel > worst[0]:
                worst = (rel, f"{base} ({src_name})")
            if args.verbose:
                print(f"    {base:<10s} {N:4d}x{K:<5d} from {src_name}  "
                      f"rel_fro {rel:.4g}")
        print(f"  checked {checked} convolutions against the checkpoint")
        print(f"  worst relative Frobenius {worst[0]:.4g}  {worst[1]}")
        print(f"  C++ reader and packer agree bit-for-bit on all of them")
        print()
        print("This gates the CONTAINER against the checkpoint. It does not say "
              "the detections are the same -- tools/verify/verify_pose_quant.py "
              "measures that, and there is no threshold for either.")
    return 0


def json_names(r):
    """conv.N for every conv.N.w in the directory, N ascending."""
    import json
    ns = []
    for name in r.entries:
        if name.startswith("conv.") and name.endswith(".w") \
                and name.count(".") == 2:
            ns.append(int(name.split(".")[1]))
    if not ns:
        raise SystemExit("no conv.N.w entries at all.")
    return [f"conv.{i}" for i in sorted(ns)]


def conv_src(src, want, N, K, base, container):
    """The checkpoint weight this container entry came from, BY NAME.

    The container records the initializer name per conv.N (`conv_sources`), and
    that is the only reliable link: the container numbers its convolutions by
    graph position and the ONNX names them `/model.22/cv2.1/cv2.1.2/Conv`, so
    there is no positional correspondence to fall back on either.

    Shape is still CHECKED, because a name can be wrong -- a hand-edited config,
    a repacked checkpoint under the old file name -- and a weight whose shape
    disagrees with the entry that claims it is not a rounding difference: it is
    the transposed-tensor case, read with the right stride and the wrong
    meaning.
    """
    w = src.get(want)
    if w is None:
        raise SystemExit(
            f"{base}: the container says its weight came from {want!r}, which "
            f"is not a Conv initialiser in the checkpoint. Either the container "
            f"and the ONNX are different networks or one of them is misnamed, "
            f"and there is no way to check the payload against anything.")
    got = (w.shape[0], int(np.prod(w.shape[1:])))
    if got != (N, K):
        raise SystemExit(
            f"{base}: the container says [{N}, {K}], and {want} is "
            f"{list(w.shape)} = [{got[0]}, {got[1]}]. A weight stored with the "
            f"wrong shape is not a rounding difference; it is a transposed "
            f"tensor that reads back as something else entirely.")
    return want, np.ascontiguousarray(w, dtype=np.float32).reshape(N, K)


if __name__ == "__main__":
    sys.exit(main())
