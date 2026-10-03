#!/usr/bin/env python3
#===----------------------------------------------------------------------===//
# Numeric check of an exported gemm_rtp design set against numpy.
#
# A design set compiles and dispatches; neither proves it computes the GEMM it
# claims to. This feeds random matrices through the exported instruction
# streams -- the same A/B/C buffers and the same instruction slots the runtime
# uses -- and compares C with a float32 numpy reference. It is the verdict on
# P4 that does not need the C++ encoder to exist first.
#
# One process holds one hw_context at a time, so the device budget is the same
# one Design spends: this must not be a second resident design set.
#===----------------------------------------------------------------------===//

import argparse
import json
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from design_sets import design_sets                                # noqa: E402
from npue import gemm_b_layout, layout_hash, tile_b  # noqa: E402

BF16 = ml_dtypes.bfloat16

# bf16 carries 8 significand bits, so one output rounding is at most half an
# ulp = 2^-9 relative. Anything beyond that is not a rounding artefact.
BF16_HALF_ULP = 2.0 ** -9

DEFAULT_RTOL_COS = 2e-3  # the repo's 1-cos bound for a whole hidden state

# What the MMAC sub-tile really is, per generation AND per operand dtype.
# MEASURED, not assumed: aie.iron.kernels.mm(...).mac_dims returns (r, s, t) --
#     npu1 / aie2 : bf16 (4, 8, 4)    i8 (4, 8, 8)
#     npu2 / aie2p: bf16 (4, 8, 8)    i8 (4, 8, 8)
# -- and the B operand's byte order inside a (tile_k, tile_n) panel is (s, t).
# A container packed with one pair and read by a core built for the other has its
# columns permuted. The permutation is inside a tile, so the bytes are the right
# size, the shapes agree, the recorded layout_hash matches on both sides, and
# every number that comes out is plausible. Only feeding the design and
# comparing with numpy sees it.
#
# The dtype key is load-bearing, and this table used to be device-only. That is
# how a bge-small int8 design came to record npu1's bf16 pair: the export, the
# pack, the runtime hash check and this gate all read the same wrong constant,
# so all four agreed and the design was silently wrong on every product. This
# table now mirrors npue.MAC_BY_DEVICE -- the single source of truth -- rather
# than restating it, because a fourth copy is what made the mistake look like a
# settled fact.
MAC_BY_DEVICE = {
    "npu1": {"bf16": (8, 4), "i8": (8, 8)},
    "npu2": {"bf16": (8, 8), "i8": (8, 8)},
}
MAC_S_DEFAULT = (8, 8)  # what an artifact with no b_layout was packed with


def dtype_of(tag):
    if tag in ("bf16", "BF16"):
        return BF16, 2
    if tag in ("i8", "int8", "I8"):
        return np.int8, 1
    if tag in ("f32", "fp32", "F32"):
        return np.float32, 4
    raise SystemExit(f"unsupported operand dtype {tag!r}")


def draw(rng, shape, a_np):
    """One operand in the design's own dtype, NON-DEGENERATE in it.

    The naive `rng.uniform(-1.0, 1.0).astype(np.int8)` is ALL ZEROS:
    truncation toward zero of a value already strictly inside (-1, 1) is 0, for
    every element. An int8 stream drawn that way is `0 @ 0`, and `report()`
    then compares the array's zero against a zero reference through two
    `or 1.0` fallbacks and reports a FAIL of 1-cos 1.0 -- a gate that cannot
    tell a correct design from a broken one, which is the failure that matters
    most. bge-small `--int8` was the first int8 design exported and found this:
    twelve FAILs, every one of them the harness's own operands.

    So integers over the range `quantise_a_int8` actually produces: symmetric
    +-127, which is also the range `pack_i4`/`pack_i8` clamp to, so the stream
    is the one the array meets in service rather than the corners it never
    visits. bf16 and f32 keep the [-1, 1) draw -- `.astype` ROUNDS those, it
    does not truncate them to zero.
    """
    if a_np is np.int8:
        return rng.integers(-127, 128, size=shape).astype(np.int8)
    return rng.uniform(-1.0, 1.0, size=shape).astype(a_np)


class Design:
    """The Python twin of npu::Design: one xclbin, N instruction slots, 3 BOs."""

    def __init__(self, directory, device_index, verbose=True):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "design.json").read_text(encoding="utf-8"))
        self.buffer_bytes = list(self.meta["buffers"])
        self.check_layout()

        try:
            import pyxrt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "pyxrt not importable. Source the XRT setup first:\n"
                "  source /opt/xilinx/xrt/setup.sh\n"
                "  export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH\n"
                f"({exc})"
            )
        self.pyxrt = pyxrt

        self.dev = pyxrt.device(device_index)
        xclbin = pyxrt.xclbin(str(self.dir / "final.xclbin"))
        names = [k.get_name() for k in xclbin.get_kernels()]
        kname = next((n for n in names if n.startswith("MLIR_AIE")), None)
        if kname is None:
            raise SystemExit(f"{self.dir}: no MLIR_AIE kernel in {names}")
        self.ctx = pyxrt.hw_context(self.dev, self.dev.register_xclbin(xclbin))
        self.kernel = pyxrt.kernel(self.ctx, kname)

        to_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.to_dev, self.from_dev = to_dev, from_dev

        self.instr = []
        for entry in self.in_meta():
            raw = (self.dir / entry["file"]).read_bytes()
            words = len(raw) // 4
            bo = pyxrt.bo(self.dev, len(raw), pyxrt.bo.flags.cacheable,
                          self.kernel.group_id(1))
            bo.write(raw, 0)
            bo.sync(to_dev, len(raw), 0)
            self.instr.append((bo, words))

        self.bos = []
        for i, nbytes in enumerate(self.buffer_bytes):
            # host_only, the runtime's default BoMode: a host-visible buffer
            # with a device address, which is what a correctness check needs.
            # The ext/1M-aligned modes are performance variants of the same
            # memory and would only make this harness disagree with the
            # shipping path for no numeric gain.
            bo = pyxrt.bo(self.dev, nbytes, pyxrt.bo.flags.host_only,
                          self.kernel.group_id(3 + i))
            bo.map()[:] = bytes(nbytes)  # zero once: no stale tail bytes
            self.bos.append(bo)

        if verbose:
            print(f"{self.dir}")
            print(f"  {kname}  {len(self.instr)} instr slots  "
                  f"buffers {['%.2f MB' % (b / 1e6) for b in self.buffer_bytes]}")
            self.report_layout()

    def report_layout(self):
        """Name a recorded B layout that this generation cannot consume.

        Not fatal on its own: the numbers below are the verdict, and a
        --mac-t override is how you measure what the hardware actually wants.
        But a container packed for the other generation is silently wrong, and
        this line is the difference between "the design is broken" and "the
        file is in the wrong byte order".
        """
        hw = self.hardware_mac()
        if hw is None:
            return
        _, _, s, t = self.b_tiles()
        if (s, t) != hw:
            dt = self.meta.get("b_layout", {}).get("dtype", "BF16")
            print(f"  LAYOUT MISMATCH: b_layout records mac (s={s}, t={t}) for "
                  f"{dt}; device {self.meta['device']} consumes (s={hw[0]}, "
                  f"t={hw[1]}) for that dtype. A container packed as recorded "
                  f"is misread by this design, and the layout_hash check cannot "
                  f"see it -- both sides derive the same hash from the same wrong "
                  f"pair. Re-export the design so its recorded b_layout matches "
                  f"the hardware, then re-pack against it.")

    def check_layout(self):
        """Refuse a B layout this driver does not implement, rather than
        comparing against a reference the bytes were never in."""
        lay = self.meta.get("b_layout")
        if lay is None:
            return
        want = gemm_b_layout(lay["tile_k"], lay["tile_n"],
                             lay.get("mac_s", 8), lay.get("mac_t", 8),
                             lay.get("dtype", "BF16"))
        if layout_hash(want) != self.meta.get("b_layout_hash"):
            raise SystemExit(
                f"{self.dir}: b_layout_hash does not match its own b_layout -- "
                "the artifact was written by a different layout definition")
        if lay.get("order") not in ("k,n,kt,nt", None):
            raise SystemExit(f"{self.dir}: unsupported B order {lay['order']!r}")

    def b_tiles(self, mac=None):
        """(tile_k, tile_n, mac_s, mac_t) of the B panel.

        `mac` overrides the recorded values. A pre-split artifact may carry no
        b_layout at all; the tiles are then the design's own k/n.
        """
        lay = self.meta.get("b_layout")
        if lay is None:
            tile = self.meta["tile"]
            return (tile["k"], tile["n"]) + tuple(mac or MAC_S_DEFAULT)
        s, t = lay.get("mac_s", 8), lay.get("mac_t", 8)
        if mac:
            s, t = mac
        return (lay["tile_k"], lay["tile_n"], s, t)

    def hardware_mac(self):
        """What this design's own generation consumes for THIS design's B dtype,
        or None if unknown.

        The dtype is the design's own recorded b_layout dtype: the int8 MMAC's
        sub-tile is not bf16's on npu1, so asking "what does npu1 consume?"
        without it is the question whose wrong answer hid a broken design.
        """
        by_dtype = MAC_BY_DEVICE.get(self.meta.get("device"))
        if not by_dtype:
            return None
        tag = self.meta.get("b_layout", {}).get("dtype", "BF16")
        try:
            dt, _ = dtype_of(tag)
        except SystemExit:
            return None
        return by_dtype.get("i8" if dt is np.int8 else "bf16")

    def in_meta(self):
        """Instruction slot 0 first (insts.bin), then the streams in slot order."""
        entries = [self.meta["slot0"]] if "slot0" in self.meta else []
        entries += sorted(self.meta.get("streams", []),
                          key=lambda s: s["slot"])
        return entries

    def clear_c(self, c_index=2):
        """Zero C and PUSH IT TO THE DEVICE before every dispatch.

        Not a precaution. A buffer written through map() and not synced is
        still dirty in the host cache, and a later read-back can then show a
        stale line for a region the array never wrote -- which looks exactly
        like "the design did not compute this tile", and sent this harness
        chasing a drain bug that does not exist. Zeroing costs one pass over
        the buffer and makes "the element is still zero" mean what it says.
        """
        self.bos[c_index].map()[:] = bytes(self.buffer_bytes[c_index])
        self.bos[c_index].sync(self.to_dev, self.buffer_bytes[c_index], 0)

    def stage(self, index, array):
        raw = np.ascontiguousarray(array).tobytes()
        if len(raw) > self.buffer_bytes[index]:
            raise SystemExit(
                f"staged {len(raw)} bytes for argument {index}, design allows "
                f"{self.buffer_bytes[index]}")
        self.bos[index].write(raw, 0)
        self.bos[index].sync(self.to_dev, len(raw), 0)

    def dispatch(self, instr_slot, c_index=2, c_bytes=None):
        bo, words = self.instr[instr_slot]
        if c_index == 2 and len(self.bos) > 2:
            args = [self.bos[0], self.bos[1], self.bos[2]]
        else:
            args = list(self.bos)
        run = self.kernel(3, bo, words, *args)
        state = run.wait()
        if state != self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise SystemExit(f"{self.dir}: kernel state {state}")
        if c_bytes is not None:
            self.bos[c_index].sync(self.from_dev, c_bytes, 0)

    def read_c(self, rows, cols, elem_bytes, c_index=2):
        nbytes = rows * cols * elem_bytes
        raw = bytes(self.bos[c_index].read(nbytes, 0))
        dtype = BF16 if elem_bytes == 2 else np.float32
        return np.frombuffer(raw, dtype=dtype, count=rows * cols).reshape(rows, cols)


def check_stream(design, stream, rng, rtol_cos, mac=None):
    M, K, N = stream["M"], stream["K"], stream["N"]
    tile_k, tile_n, s, t = design.b_tiles(mac)
    a_np, a_bytes = dtype_of(design.meta.get("a_dtype", "bf16"))
    c_bytes = 2 if design.meta.get("c_dtype", "bf16") == "bf16" else 4

    # An operand the array can actually be judged on: `draw` rather than
    # uniform, because for an int8 design the uniform draw IS zero (see draw).
    a = draw(rng, (M, K), a_np)
    b = draw(rng, (K, N), a_np)

    tiled = tile_b(b, tile_k, tile_n, s, t)
    design.clear_c()
    design.stage(0, a)
    design.stage(1, tiled)
    design.dispatch(stream["slot"], c_bytes=M * N * c_bytes)
    got = design.read_c(M, N, c_bytes)

    # Reference in float32 on the SAME rounded operands: the device cannot be
    # blamed for the bf16 the caller handed it, only for its own arithmetic.
    ref = a.astype(np.float32) @ b.astype(np.float32)
    got32 = got.astype(np.float32)

    return report(stream, got32, ref, rtol_cos)


def report(stream, got32, ref, rtol_cos, note=""):
    # A ZERO REFERENCE IS NOT A COMPARISON, and it must not look like one.
    # Both `or 1.0` fallbacks below exist so a zero vector cannot divide by
    # zero -- but together they turn "nothing was measured" into a verdict:
    # the denominator becomes 1.0, cos becomes 0, and the gate prints
    # `1-cos 1.00e+00` for a design that may be perfectly correct. That is
    # fail-open in the direction that matters most, so it is refused by name
    # instead of reported as a fault in the array.
    if not np.any(ref):
        raise SystemExit(
            f"{stream['op']} b{stream['batch']} "
            f"M{stream['M']} K{stream['K']} N{stream['N']}: the float32 "
            f"reference is ALL ZEROS, so nothing was measured and nothing "
            f"can be judged. This is the HARNESS handing the array "
            f"degenerate operands in their own dtype (see draw()), not a "
            f"design that failed.")
    err = np.abs(got32 - ref)
    scale = float(np.abs(ref).max()) or 1.0
    cos = float((got32 * ref).sum() / (np.linalg.norm(got32) *
                                        np.linalg.norm(ref) or 1.0))
    one_minus_cos = 1.0 - cos
    max_rel = float(err.max()) / scale
    unwritten = int((got32 == 0.0).sum() - (ref == 0.0).sum())

    ok = one_minus_cos <= rtol_cos and max_rel <= 4 * BF16_HALF_ULP
    print(f"  {'OK  ' if ok else 'FAIL'} {stream['op']:<16} b{stream['batch']:<3} "
          f"M{stream['M']:<5} K{stream['K']:<5} N{stream['N']:<5} "
          f"1-cos {one_minus_cos:.2e}  max|d|/|ref|max {max_rel:.2e}  "
          f"|ref|max {scale:.3f}"
          + (f"  UNWRITTEN {unwritten}" if unwritten > 0 else "")
          + (f"  {note}" if note else ""))
    return ok


def check_container(design, stream, rng, rtol_cos, reader, tensor):
    """The same GEMM with the operand a real .npue actually stores.

    This is the check that closes the loop. The synthetic path above proves the
    design computes what its own layout descriptor says; it cannot notice that
    the descriptor and the FILE disagree about the generation, because both
    sides then agree on a layout the hardware does not consume. Staging the
    stored bytes verbatim and comparing against the same tensor de-tiled with
    its OWN recorded layout is the only comparison that spans design, file and
    hardware at once.
    """
    entry = reader.entries[tensor]
    shape = list(entry["logical_shape"])
    if shape != [stream["K"], stream["N"]]:
        raise SystemExit(
            f"{tensor} is {shape}, the {stream['op']} stream is "
            f"[{stream['K']}, {stream['N']}] -- pick the matching tensor")
    K, N = entry["padded_shape"]

    # raw() is the bytes as stored: uint16 bit patterns, already tiled. A .view
    # onto bfloat16, NOT an astype: casting the integers would reinterpret the
    # patterns as small numbers and quietly stage zeros.
    stored = reader.raw(tensor)
    b_panel = stored.view(BF16) if stored.dtype == np.uint16 else stored
    logical = reader.tensor(tensor)[:stream["K"], :stream["N"]].astype(np.float32)

    a_np, _ = dtype_of(design.meta.get("a_dtype", "bf16"))
    c_bytes = 2 if design.meta.get("c_dtype", "bf16") == "bf16" else 4
    a = draw(rng, (stream["M"], stream["K"]), a_np)

    design.clear_c()
    design.stage(0, a)
    design.stage(1, b_panel)
    design.dispatch(stream["slot"], c_bytes=stream["M"] * stream["N"] * c_bytes)
    got = design.read_c(stream["M"], stream["N"], c_bytes).astype(np.float32)

    lay = entry.get("layout") or {}
    return report(stream, got, a.astype(np.float32) @ logical, rtol_cos,
                  note=f"[{tensor} as stored, mac "
                       f"s={lay.get('mac_s')} t={lay.get('mac_t')}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Feed random matrices through an exported gemm_rtp design "
                    "set and compare C with a float32 numpy reference.")
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="design directories; default every runtime/*/artifacts_npu*/gemm_rtp*")
    ap.add_argument("--op", action="append", default=[],
                    help="only this op (repeatable)")
    ap.add_argument("--tier", action="append", type=int, default=[],
                    help="only this batch tier (repeatable)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL_COS,
                    help="1-cos bound (default %(default)s)")
    ap.add_argument("--list", action="store_true",
                    help="print the streams and exit without touching the device")
    ap.add_argument("--mac-s", type=int, default=None,
                    help="override the recorded B sub-tile s")
    ap.add_argument("--mac-t", type=int, default=None,
                    help="override the recorded B sub-tile t (npu1 consumes 4)")
    ap.add_argument("--npue", type=Path, default=None,
                    help="also check real operands out of this container")
    ap.add_argument("--tensor", action="append", default=[],
                    metavar="OP=NAME",
                    help="with --npue: the container tensor holding OP's "
                         "weight, e.g. --tensor qkv=layer.0.qkv (repeatable)")
    args = ap.parse_args()
    mac = ((args.mac_s, args.mac_t)
           if args.mac_s and args.mac_t else None)

    reader, tensors = None, {}
    if args.npue:
        from npue import Reader
        reader = Reader(str(args.npue))
        for spec in args.tensor:
            if "=" not in spec:
                raise SystemExit(f"--tensor wants OP=NAME, got {spec!r}")
            op, name = spec.split("=", 1)
            tensors[op] = name
        if not tensors:
            raise SystemExit("--npue needs at least one --tensor OP=NAME")

    dirs = args.dirs
    if not dirs:
        # Via design_sets, which states the layout once. The glob this replaces
        # (`runtime/*/artifacts_npu*/gemm_rtp*/design.json`) matched NOTHING once
        # the per-model sets moved under runtime/artifacts/, so this gate would
        # have silently verified zero designs and reported it as a pass -- the
        # failure mode this whole layout change is trying to make impossible.
        dirs = sorted({p.parent for p in design_sets()
                       if p.parent.name.startswith("gemm_rtp")})

    failed = 0
    for d in dirs:
        meta = json.loads((d / "design.json").read_text(encoding="utf-8"))
        want = [s for s in meta.get("streams", [])
                if (not args.op or s["op"] in args.op)
                and (not args.tier or s["batch"] in args.tier)]
        if args.list:
            print(f"{d}  arch={meta.get('arch')} device={meta.get('device')} "
                  f"tile={meta.get('tile')}")
            for s in want:
                print(f"  slot {s['slot']:<3} {s['op']:<16} b{s['batch']:<3} "
                      f"M{s['M']:<5} K{s['K']:<5} N{s['N']:<5} {s['file']}")
            continue

        rng = np.random.default_rng(args.seed)
        design = Design(d, args.device)
        for s in want:
            if not check_stream(design, s, rng, args.rtol, mac):
                failed += 1
            if reader is not None and s["op"] in tensors:
                if not check_container(design, s, rng, args.rtol, reader,
                                       tensors[s["op"]]):
                    failed += 1
        del design

    if reader is not None:
        reader.close()

    if args.list:
        return 0
    print(f"\n{'FAILED' if failed else 'all streams match numpy'}"
          f" ({failed} bad)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
