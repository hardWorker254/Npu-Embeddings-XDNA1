# NpuEmbeddings -- M4 GATE: verify the .npue container.
#
# Four checks, in increasing order of what they would catch:
#
#   A. SPEC       -- header size, magic, version, and 4096-byte alignment of
#                    every tensor. A misaligned tensor is a DMA descriptor that
#                    silently reads the wrong bytes on hardware.
#   B. ROUND-TRIP -- de-tile every pre-tiled operand and compare BIT-EXACTLY
#                    against the source. Not "close": the tiling is a
#                    permutation, so anything but zero differing elements is a
#                    bug. Two schemes, two comparisons -- see check_roundtrip.
#   C. GUARD      -- a wrong layout must RAISE, not silently produce garbage
#                    embeddings. This is what layout_hash is for.
#   D. GOLDENS    -- run the actual encoder off the packed weights and compare
#                    to the M3 goldens. This is the check the other three exist
#                    to support: it is the only one that would catch a fusion
#                    that round-trips perfectly but is mathematically wrong
#                    (scale folded into K instead of Q, a transpose applied
#                    twice, Q/K/V concatenated in the wrong order).
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue.py

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))
sys.path.insert(0, str(REPO / "reference"))

# End-to-end bf16 1-cos, MEASURED, keyed by source repo. Only models we have
# actually measured belong here; a missing entry means the check reports the
# absolute number rather than a ratio against someone else's model.
BF16_BASELINE = {
    "sentence-transformers/all-MiniLM-L6-v2": 1.271e-05,   # M3, tasks/0005
    "BAAI/bge-small-en-v1.5": 9.631e-06,                   # tasks/0039
    "BAAI/bge-large-en-v1.5": 1.355e-05,                   # tasks/0042
}

# int8's own end-to-end limit, and why the bf16 table above is not the gate for
# it. Two reasons, and the second is the one that matters:
#
#   1. the numbers are three orders apart, so a ratio against a bf16 baseline
#      would be ~90x and would "fail" on arithmetic rather than on quality;
#   2. the shipped int8 datapath quantises the ACTIVATION per row as well
#      (host_kernels.hpp `quantise_a_int8`), and the Python encoder here does
#      not model that. So the number this gate computes for an int8 container
#      is the WEIGHT half of the quantisation cost -- a lower bound, and it is
#      labelled as one rather than reported as the datapath.
#
# 3e-03 is the measured bge-large int8 figure (STATE.md sec 8.1 item 5, which
# also records that the common 2e-03 gate is not reachable for int8 at all)
# with a little room above it. The full int8 error needs the array, or
# tools/verify/verify_design_numerics.py --npue, which runs the design's own
# instruction streams.
INT8_WEIGHT_ONLY_LIMIT = 3e-03

from encoder import MiniLMReference                                # noqa: E402
from precision_study import make_gemm                              # noqa: E402
from gemm_i8 import check_i8_operand                                # noqa: E402
from npue import (ALIGN, ASMOOTH_SUFFIX, HEADER_SIZE, MAGIC, VERSION,  # noqa: E402
                  Reader, WSCALE_SUFFIX, find_goldens, to_bf16_bits, untile_b)
from safetensors_io import load                                    # noqa: E402


def rel_fro(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# -- A. spec conformance ---------------------------------------------------

def check_spec(path, r):
    print("A. spec conformance")
    problems = []
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
    magic, version, arch, flags, jo, jl, do, dl, reserved = struct.unpack(
        "<4sIII QQQQ 16s", head)
    size = Path(path).stat().st_size

    checks = [
        ("header is exactly 64 bytes", struct.calcsize("<4sIII QQQQ 16s") == 64),
        ("magic == b'NPUE'", magic == MAGIC),
        (f"version == {VERSION}", version == VERSION),
        ("flags has bit0 (pre-tiled)", bool(flags & 1)),
        ("reserved is zero", reserved == b"\0" * 16),
        (f"data_offset {do} is {ALIGN}-aligned", do % ALIGN == 0),
        ("json sits between header and data", jo == HEADER_SIZE and jo + jl <= do),
        ("file size == data_offset + data_length", size == do + dl),
    ]
    misaligned = [e["name"] for e in r.entries.values()
                  if (do + e["offset"]) % ALIGN]
    checks.append((f"all {len(r.entries)} tensors 4096-aligned", not misaligned))
    if misaligned:
        problems.append(f"misaligned: {misaligned[:5]}")

    for label, ok in checks:
        print(f"   {'ok  ' if ok else 'FAIL'}  {label}")
        if not ok:
            problems.append(label)
    overhead = size - sum(e["nbytes"] for e in r.entries.values())
    print(f"         alignment + json overhead: {overhead/1024:.1f} KB "
          f"on {size/1e6:.1f} MB ({overhead/size*100:.3f}%)")
    return problems


# -- B. bit-exact round-trip ----------------------------------------------

def check_roundtrip(r, src, cfg, fold_scale):
    L, hidden = cfg["num_layers"], cfg["hidden"]
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    problems, total = [], 0

    def expect_qkv(i):
        sa = f"encoder.layer.{i}.attention.self."
        m = np.ascontiguousarray(np.concatenate(
            [src[sa + n + ".weight"] for n in ("query", "key", "value")], axis=0).T)
        if fold_scale:
            m = m.copy()
            m[:, :hidden] *= scale
        return m

    def expect_t(name):
        return np.ascontiguousarray(src[name].T)

    # WHICH SCHEME, decided per operand and not from the config. A container
    # whose `a_dtype` says one thing while its panels carry another is exactly
    # the disagreement runtime/src/whisper/npu_ops.cpp refuses to run on, and
    # reading the claim instead of the bytes would let it through here.
    #
    # Decided from layer.0's four operands alone: they are emitted by one emitter
    # from one switch, and if they ever disagreed with each other that is itself
    # the finding, reported below. Asking all L layers instead would print the
    # same set L times over.
    ops = ("qkv", "attn_out", "ffn_up", "ffn_down")
    schemes = {r.entries[f"layer.0.{op}"]["dtype"] for op in ops}
    claimed = cfg.get("a_dtype", "bf16")
    if len(schemes) > 1:
        problems.append(f"layer.0's four per-layer operands disagree about their "
                        f"own dtype: {sorted(schemes)} -- one emitter wrote them, "
                        f"so this is a container that cannot be run at all")
    scheme = schemes.pop() if len(schemes) == 1 else None
    if scheme == "I8" and claimed != "i8":
        problems.append(f"config says a_dtype={claimed!r} but the operands are "
                        f"I8 -- the container contradicts itself, and the "
                        f"runtime would refuse it against the design")
    if scheme == "BF16" and claimed == "i8":
        problems.append(f"config claims a_dtype='i8' but the operands are BF16 "
                        f"-- an int8 design would be handed bf16 weights")
    print("\nB. round-trip (de-tile == the source, bit-exact), "
          f"{'int8 W8A8' if scheme == 'I8' else 'bf16'}"
          f"{'' if scheme else ' -- SCHEME UNDETERMINED, see below'}")

    # The per-operand weight-quantisation error, recomputed here from the file
    # rather than trusted from the packer's own printout. Reported for int8 only
    # because for bf16 the round-trip below already proves bit-equality and the
    # number would be 0 by construction.
    qerr = []

    for i in range(L):
        cases = {
            f"layer.{i}.qkv": expect_qkv(i),
            f"layer.{i}.attn_out":
                expect_t(f"encoder.layer.{i}.attention.output.dense.weight"),
            f"layer.{i}.ffn_up":
                expect_t(f"encoder.layer.{i}.intermediate.dense.weight"),
            f"layer.{i}.ffn_down":
                expect_t(f"encoder.layer.{i}.output.dense.weight"),
        }
        for name, want_f32 in cases.items():
            e = r.entries[name]
            lay = e.get("layout")
            if not lay:
                # An untiled operand is legitimate in the format -- that is what
                # `add_gemm_b_host` writes for a host-side weight -- but it is
                # not one of the four NPU GEMM operands this check is about.
                # Say that, rather than dying on a KeyError three lines later.
                problems.append(f"{name} carries no layout, so it is not a "
                                f"pre-tiled NPU operand; check B cannot judge it")
                continue
            K, N = e["padded_shape"]
            got_bits = untile_b(r.raw(name), K, N, lay["tile_k"], lay["tile_n"],
                                lay["mac_s"], lay["mac_t"])
            kl, nl = e["logical_shape"]
            got_bits = got_bits[:kl, :nl]

            if e["dtype"] == "I8":
                # There is no bit pattern to compare against: the panel is a
                # rounding OF the source at the container's own scale, so the
                # comparison is against that rounding -- and the scale rule, the
                # saturation invariant and the dead-column invariant come with
                # it. See tools/lib/gemm_i8.py check_i8_operand for why each exists.
                probs, rel = check_i8_operand(
                    name, got_bits, r.tensor(name + WSCALE_SUFFIX),
                    r.tensor(name + ASMOOTH_SUFFIX), want_f32)
                problems += probs
                qerr.append((name, rel))
                total += got_bits.size
                continue

            want_bits = to_bf16_bits(want_f32)
            ndiff = int((got_bits != want_bits).sum())
            total += want_bits.size
            if ndiff:
                problems.append(f"{name}: {ndiff} of {want_bits.size} differ")

    unit = "int8" if scheme == "I8" else "bf16"
    ndiff, nother = _tally(problems)
    print(f"   {'ok  ' if not problems else 'FAIL'}  "
          f"{4*L} operands, {total:,} {unit} elements")
    if problems:
        print(f"         {ndiff} element(s) differ, {nother} further problem(s)"
              if nother else f"         {ndiff} element(s) differ")
    for p in problems[:5]:
        print(f"   FAIL  {p}")
    if qerr:
        worst = max(qerr, key=lambda t: t[1])
        print(f"         weight quantisation rel_fro: mean "
              f"{sum(e for _, e in qerr) / len(qerr):.3e}, worst "
              f"{worst[1]:.3e} ({worst[0]})")
        print(f"         this is the OPERAND's error, not the model's -- it is "
              f"not comparable to an end-to-end gate")
    return problems


def _tally(problems):
    """(elements differing, problems that are not element counts).

    Both schemes open their per-tensor message with "N of M" -- bf16 as
    "1234 of 5678 differ", int8 as "1234 of 5678 int8 values are not ..." --
    which is what lets one counter read either. The rest (a container that
    contradicts itself, a scale rule broken) counts as itself: a summary line
    that said "0 differing" next to a FAIL would be the kind of line that
    trains a reader to skim past it.
    """
    n, counted = 0, 0
    for p in problems:
        head = p.split(":", 1)[1].strip() if ":" in p else ""
        first = head.split(" ")[0] if head else ""
        if first.isdigit():
            n += int(first)
            counted += 1
    return n, len(problems) - counted


# -- C. the stale-layout guard --------------------------------------------

def check_guard(r):
    print("\nC. layout_hash guard")
    name = "layer.0.qkv"
    good = dict(r.entries[name]["layout"])
    bad = dict(good, tile_n=good["tile_n"] + 16)
    problems = []
    try:
        r.check_layout(name, good)
        print("   ok    matching layout accepted")
    except ValueError as e:
        problems.append(f"correct layout rejected: {e}")
        print(f"   FAIL  matching layout rejected: {e}")
    try:
        r.check_layout(name, bad)
        problems.append("wrong layout accepted -- a stale file would run silently")
        print("   FAIL  wrong layout accepted")
    except ValueError:
        print(f"   ok    tile_n {good['tile_n']} -> {bad['tile_n']} refused")
    return problems


# -- D. the real check: goldens -------------------------------------------

def build_from_npue(r, cfg, gemm=None):
    """Reconstruct an encoder that reads ONLY packed weights.

    Everything comes back out of the container: the fused QKV, the transposed
    projections, the fp32 biases and LayerNorm params. Nothing is re-derived
    from the original checkpoint, so a fusion bug has nowhere to hide.
    """
    L, hidden = cfg["num_layers"], cfg["hidden"]
    w = {
        "embeddings.word_embeddings.weight": r.tensor("embeddings.word"),
        "embeddings.position_embeddings.weight": r.tensor("embeddings.position"),
        "embeddings.token_type_embeddings.weight": r.tensor("embeddings.token_type"),
        "embeddings.LayerNorm.weight": r.tensor("embeddings.ln.weight"),
        "embeddings.LayerNorm.bias": r.tensor("embeddings.ln.bias"),
    }
    qkv_w, qkv_b = {}, {}
    for i in range(L):
        # encoder.py's linear() does weight.T, and .npue stores [K,N], so
        # transposing here hands back exactly the packed values.
        qkv_w[i] = np.ascontiguousarray(r.tensor(f"layer.{i}.qkv").T)
        qkv_b[i] = r.tensor(f"layer.{i}.qkv.bias")
        p = f"encoder.layer.{i}."
        w[p + "attention.output.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.attn_out").T)
        w[p + "attention.output.dense.bias"] = r.tensor(f"layer.{i}.attn_out.bias")
        w[p + "attention.output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln1.weight")
        w[p + "attention.output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln1.bias")
        w[p + "intermediate.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.ffn_up").T)
        w[p + "intermediate.dense.bias"] = r.tensor(f"layer.{i}.ffn_up.bias")
        w[p + "output.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.ffn_down").T)
        w[p + "output.dense.bias"] = r.tensor(f"layer.{i}.ffn_down.bias")
        w[p + "output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln2.weight")
        w[p + "output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln2.bias")

    folded = cfg["fusions"]["qk_scale_folded_into_q"]
    return MiniLMReference(
        w, num_layers=L, num_heads=cfg["num_heads"], eps=cfg["layer_norm_eps"],
        gemm=gemm, qkv_w=qkv_w, qkv_b=qkv_b,
        # The container records how this model pools. Defaulting to mean here
        # made a CLS model look 9.3e-02 wrong -- four orders worse than the
        # same weights measure on the actual NPU -- and the gate below did not
        # catch it, because it only compared against a baseline this model has
        # not got.
        pooling=cfg.get("pooling", "mean"),
        qk_scale=1.0 if folded else math.sqrt(cfg["head_dim"]))


def check_goldens(r, cfg, goldens):
    print("\nD. the encoder, running off packed weights, vs the M3 goldens")
    g, meta = load(goldens)
    if meta["source_sha256"] != cfg["source_sha256"]:
        return [f"checkpoint mismatch: goldens {meta['source_sha256'][:16]} vs "
                f".npue {cfg['source_sha256'][:16]}"]

    i8 = r.entries["layer.0.qkv"]["dtype"] == "I8"

    # Two runs, because they answer different questions and conflating them
    # would flatter the result:
    #
    #  * fp32 activations -- isolates what PACKING costs (the weight format +
    #    the fusions), with nothing else in the way.
    #  * bf16 activations -- the actual NPU datapath, and the only run
    #    comparable to M3's end-to-end bf16 figure of 1-cos = 1.271e-05.
    #
    # FOR AN int8 CONTAINER THE SECOND RUN IS NOT THE DATAPATH. The array also
    # quantises the activation per row (host_kernels.hpp `quantise_a_int8`) and
    # the reference encoder here does not model that, so this measures the
    # WEIGHT half of the quantisation and nothing else. It is printed and gated
    # as what it is; the full int8 number needs the array, or
    # tools/verify/verify_design_numerics.py --npue.
    L = cfg["num_layers"]
    names = ["emb.ln"] + [f"L{i}.ln2" for i in range(L)] + ["last_hidden_state"]
    # M3 measured this end to end for MiniLM-L6. A 12-layer model
    # legitimately differs, so dividing by it would manufacture a regression.
    # A model with no measured baseline reports the absolute number.
    baseline = BF16_BASELINE.get(cfg.get("source_repo"))

    runs = {}
    for label, gemm in (("fp32 activations", None),
                        ("bf16 activations", make_gemm("bf16"))):
        ref = build_from_npue(r, cfg, gemm=gemm)
        taps = {}
        emb = ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"],
                         taps=taps)
        runs[label] = (emb, taps)

    head = "weights only" if not i8 else "int8 weights only"
    print(f"   {'tensor':<20} {head:>19} {'+ bf16 activations':>20}")
    for nm in names:
        print(f"   {nm:<20} "
              f"{rel_fro(runs['fp32 activations'][1][nm], g['hf.' + nm]):>19.3e} "
              f"{rel_fro(runs['bf16 activations'][1][nm], g['hf.' + nm]):>20.3e}")

    problems = []
    if i8:
        print("   NOTE  int8 container: these two runs differ only in the "
              "ACTIVATION\n         format. Neither quantises A per row, which "
              "the array does, so\n         both are the weight half of the "
              "int8 cost -- a lower bound.")
    print()
    for label, (emb, _) in runs.items():
        cos = (emb.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
        shift = float(np.abs(emb @ emb.T
                             - g["hf.out.embedding"] @ g["hf.out.embedding"].T).max())
        print(f"   {label:<18} 1-cos {1 - cos.min():.3e}   "
              f"similarity shift {shift:.3e}")
        if label == "bf16 activations":
            if i8:
                # Its own limit, and NO ratio against the bf16 baseline: the
                # two schemes are three orders apart, so the ratio would be ~90x
                # and would report arithmetic rather than quality. See
                # INT8_WEIGHT_ONLY_LIMIT for why this is the weight half only.
                print(f"   {'':18} int8 weight half, limit "
                      f"{INT8_WEIGHT_ONLY_LIMIT:.0e} "
                      f"({'within' if 1 - cos.min() <= INT8_WEIGHT_ONLY_LIMIT else 'OVER'})"
                      f" -- the A-side row quantisation is NOT in this number")
                if 1 - cos.min() > INT8_WEIGHT_ONLY_LIMIT:
                    problems.append(
                        f"int8 1-cos {1 - cos.min():.3e} exceeds the weight-only "
                        f"limit {INT8_WEIGHT_ONLY_LIMIT:.0e} -- the operand "
                        f"round-trip in check B passed, so this is the "
                        f"quantisation itself, not a packing fault")
                continue
            if baseline is None:
                # No relative baseline is not the same as no check. Well below
                # the runtime's own 2e-03 gate and far above any legitimate
                # bf16 result, so it catches a wrong pooling or a broken fusion
                # without pretending to be a precision measurement.
                ABS_LIMIT = 1e-03
                print(f"   {'':18} no measured bf16 baseline for "
                      f"{cfg.get('source_repo')} -- gating on the absolute "
                      f"limit {ABS_LIMIT:.0e}")
                if 1 - cos.min() > ABS_LIMIT:
                    problems.append(
                        f"1-cos {1 - cos.min():.3e} exceeds the absolute limit "
                        f"{ABS_LIMIT:.0e} for a model with no measured "
                        f"baseline")
                continue
            ratio = (1 - cos.min()) / baseline
            print(f"   {'':18} vs the measured bf16 baseline ({baseline:.2e}): "
                  f"{ratio:.2f}x")
            # Packing must not be lossier than bf16 alone. If it were, a fusion
            # is doing damage that the number format does not explain.
            if ratio > 2.0:
                problems.append(f"packing costs {ratio:.2f}x M3's bf16 baseline "
                                f"-- a fusion is lossier than bf16 alone")
    return problems


# -- E. isolate what the scale fold costs ---------------------------------

def check_fold_cost(cfg, goldens, packed_path, model_dir, r=None):
    """Does folding 1/sqrt(32) into Q cost anything BEYOND bf16 rounding?

    It is not obviously free: 1/sqrt(32) is not a power of two, so folding
    changes which bf16 grid point each Q weight rounds to. Runs the identical
    encoder on a --no-fold-scale pack and compares. Anything else would be
    guessing at a question that takes one extra file to answer.
    """
    print("\nE. what the 1/sqrt(head_dim) fold costs, in isolation")
    # SKIPPED FOR int8, and saying so is the whole point. The control below is
    # packed WITHOUT --int8, so it would be a bf16 container; comparing an int8
    # container against a bf16 one measures the quantisation, not the fold, and
    # reports it as "the fold costs 90x -- NOT free". A gate that invents a
    # finding is worse than a gate that skips, but only if it says which it is
    # doing. Forwarding --int8 instead would need the calibration corpus and
    # the numpy oracle this gate does not otherwise require.
    if r is not None and r.entries["layer.0.qkv"]["dtype"] == "I8":
        print("   SKIP  int8 container: the unfolded control would have to be "
              "packed\n         --int8 too, which needs a calibration corpus "
              "and the oracle. Comparing\n         across schemes would measure "
              "the quantisation and call it the fold.")
        return []
    unfolded = Path(packed_path).with_name("_verify_nofold.npue")
    import subprocess
    # --model-dir was missing here, so this packed the DEFAULT checkpoint
    # and compared it against whichever model was actually under test.
    # --model-dir AND --tile-n. Forwarding only the first was the same bug one
    # layer down: bge-large packs at tile_n 32 because its N in
    # {1024, 3072, 4096} makes 48 illegal, so the unfolded control was being
    # built at a tile size the model cannot use. The container knows its own.
    cmd = [sys.executable, str(REPO / "tools" / "pack" / "pack_npue.py"),
           "--model-dir", str(model_dir),
           "--tile-k", str(cfg["tile_k"]), "--tile-n", str(cfg["tile_n"]),
           "--no-fold-scale", "--out", str(unfolded)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        print(f"   FAIL  could not pack the unfolded variant:\n{res.stderr[-400:]}")
        return [f"unfolded pack failed: {res.returncode}"]

    g, _ = load(goldens)
    out = {}
    for label, path in (("folded into Q", packed_path), ("applied at runtime", unfolded)):
        # bf16 activations: the fold changes which bf16 grid point each Q weight
        # lands on, so it must be judged on the datapath that actually rounds.
        with Reader(path) as rr:
            ref = build_from_npue(rr, rr.config, gemm=make_gemm("bf16"))
            emb = ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"])
        cos = (emb.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
        out[label] = 1 - cos.min()
    unfolded.unlink(missing_ok=True)

    for label, v in out.items():
        print(f"   1 - cos, scale {label:<20}: {v:.3e}")
    d = out["folded into Q"] / out["applied at runtime"]
    print(f"   folding costs {d:.3f}x -- "
          f"{'free' if 0.5 <= d <= 2.0 else 'NOT free'}")
    return [] if 0.5 <= d <= 2.0 else [f"scale fold costs {d:.2f}x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="sets --npue, --model-dir and --goldens together")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--npue")
    ap.add_argument("--model-dir")
    ap.add_argument("--goldens")
    args = ap.parse_args()

    # Three paths that must describe ONE checkpoint. Derive them together by
    # default, and check the sha256 below whether they were derived or given.
    if not args.npue:
        args.npue = str(REPO / "models" / f"{args.model}.npue")
    if not args.model_dir:
        # The CHECKPOINT directory, from the container's source_repo -- the
        # same rule reference/fetch_model.py uses to name it. Deriving it from
        # the container name broke bge-large-n16.npue, which is bge-large's
        # weights at a different tile size and has no directory of its own.
        with Reader(args.npue) as _r0:
            _repo = _r0.config["source_repo"]
        args.model_dir = str(REPO / "models" / _repo.split("/")[-1])
    if not args.goldens:
        # By checkpoint, not by name: bge-large-n16.npue is the same weights at
        # a different tile size and shares bge-large's goldens.
        with Reader(args.npue) as _r:
            _sha = _r.config["source_sha256"]
        args.goldens = str(find_goldens(REPO / "reference" / "goldens",
                                        _sha, args.seq, load)[0])

    r = Reader(args.npue)
    cfg = r.config
    print(f"{Path(args.npue).name}  "
          f"({Path(args.npue).stat().st_size/1e6:.2f} MB, {len(r.entries)} tensors)")
    _dp = "int8 W8A8" if r.entries.get("layer.0.qkv", {}).get("dtype") == "I8" else "bf16"
    print(f"  tile ({cfg['tile_k']}, {cfg['tile_n']}), "
          f"mac (s={cfg['mac_s']}, t={cfg['mac_t']}), "
          f"datapath: {_dp}, "
          f"fusions: {', '.join(k for k, v in cfg['fusions'].items() if v is True)}\n")

    # The .npue, the checkpoint and the goldens must be one checkpoint.
    # Comparing right-shaped tensors against another model's answers is the
    # failure the runtime's fixture guard exists for; it applies here too.
    _g, _gmeta = load(args.goldens)
    if _gmeta.get("source_sha256") != cfg.get("source_sha256"):
        print(f"FAIL -- the .npue and the goldens are different checkpoints:\n"
              f"  .npue    {cfg.get('source_sha256')}\n"
              f"  goldens  {_gmeta.get('source_sha256')}")
        return 1

    src, _ = load(Path(args.model_dir) / "model.safetensors")

    problems = []
    problems += check_spec(args.npue, r)
    problems += check_roundtrip(r, src, cfg, cfg["fusions"]["qk_scale_folded_into_q"])
    problems += check_guard(r)
    problems += check_goldens(r, cfg, args.goldens)
    problems += check_fold_cost(cfg, args.goldens, args.npue,
                                args.model_dir, r=r)

    if problems:
        print(f"\nFAIL -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS -- spec conformant, round-trip bit-exact, layout guarded, "
          "goldens reproduced")
    if _dp != "bf16":
        print("       (int8: the operand round-trip is the SCHEME's; the "
              "golden number above\n        is the weight half only -- the A-side "
              "row quantisation needs the array)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
