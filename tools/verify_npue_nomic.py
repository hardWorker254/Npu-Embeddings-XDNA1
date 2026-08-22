# NpuEmbeddings -- arch=2 GATE: verify the nomic-embed-text-v1.5 .npue container.
#
# Modeled on tools/verify_npue.py, but NOT a rewrite of it: that file is
# arch=0 (BERT) only and indexes cfg["tile_k"]/fusions/"layer.i.qkv" in
# BERT-specific ways that happen to still parse for arch=2 (the tensor names
# are deliberately identical -- tasks/0069 item 3) but would silently miss
# this arch's own risks. This is a separate, arch=2-aware gate.
#
# Six checks:
#
#   A. SPEC        -- header 64 B, magic, version, arch==2, reserved zero,
#                      4096-alignment, sizes.
#   B. ROUND-TRIP  -- de-tile every gemm_b operand and compare BIT-EXACTLY
#                      against the bf16 of the fused source -- including the
#                      fused ffn_up (fc11|fc12), where "bit-exact" also
#                      proves both halves survived in the right order.
#   C. GUARD       -- a wrong layout_hash must RAISE.
#   D. ZERO        -- every *.bias and embeddings.position must be EXACTLY
#                      zero for this arch (nomic has none of either); as a
#                      discriminating control, the SAME check against an
#                      arch=0 container must find them NON-zero -- a test
#                      that cannot fail proves nothing.
#   E. ROPE-FOLD   -- numeric proof that folding 1/sqrt(head_dim) into Q
#                      before RoPE is exact: RoPE is linear in q, so
#                      rope(s*q) == s*rope(q). Compared at fp32, the
#                      precision the packed Q actually uses.
#   F. CONFIG      -- rope_theta and swiglu_halves are present in the
#                      packed config and match the checkpoint.
#
# Goldens comparison (reference/encoder_nomic.py + goldens_nomic/) is owned by
# another task -- structured so a golden check can be ADDED later without
# touching what is here, but nothing here depends on it existing yet.
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue_nomic.py --model nomic-embed-text-v1.5

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

from npue import ALIGN, ARCH_NOMIC_ROPE_SWIGLU, HEADER_SIZE, MAGIC, VERSION, \
    Reader, to_bf16_bits, untile_b                                  # noqa: E402
from safetensors_io import load                                     # noqa: E402


def rel_fro(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


# -- A. spec conformance ----------------------------------------------------

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
        (f"arch == {ARCH_NOMIC_ROPE_SWIGLU} (nomic_bert_rope_swiglu)",
         arch == ARCH_NOMIC_ROPE_SWIGLU),
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


# -- B. bit-exact round-trip -------------------------------------------------

def check_roundtrip(r, src, cfg, fold_scale):
    print("\nB. round-trip (de-tile == bf16 of the fused source, bit-exact)")
    L, hidden, inter = cfg["num_layers"], cfg["hidden"], cfg["intermediate"]
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    problems, total = [], 0

    def expect_qkv(i):
        p = f"encoder.layers.{i}.attn."
        m = np.ascontiguousarray(src[p + "Wqkv.weight"].T)
        if fold_scale:
            m = m.copy()
            m[:, :hidden] *= scale
        return m

    def expect_t(name):
        return np.ascontiguousarray(src[name].T)

    def expect_ffn_up(i):
        p = f"encoder.layers.{i}.mlp."
        up = np.ascontiguousarray(src[p + "fc11.weight"].T)
        gate = np.ascontiguousarray(src[p + "fc12.weight"].T)
        return np.concatenate([up, gate], axis=1)

    for i in range(L):
        cases = {
            f"layer.{i}.qkv": expect_qkv(i),
            f"layer.{i}.attn_out":
                expect_t(f"encoder.layers.{i}.attn.out_proj.weight"),
            f"layer.{i}.ffn_up": expect_ffn_up(i),
            f"layer.{i}.ffn_down":
                expect_t(f"encoder.layers.{i}.mlp.fc2.weight"),
        }
        for name, want_f32 in cases.items():
            e = r.entries[name]
            lay = e["layout"]
            K, N = e["padded_shape"]
            got_bits = untile_b(r.raw(name), K, N, lay["tile_k"], lay["tile_n"],
                                lay["mac_s"], lay["mac_t"])
            want_bits = to_bf16_bits(want_f32)
            ndiff = int((got_bits != want_bits).sum())
            total += want_bits.size
            if ndiff:
                problems.append(f"{name}: {ndiff} of {want_bits.size} differ")

    print(f"   {'ok  ' if not problems else 'FAIL'}  "
          f"{4*L} operands, {total:,} bf16 elements, "
          f"{sum(int(p.split(':')[1].split()[0]) for p in problems) if problems else 0}"
          f" differing")
    print(f"         (of which ffn_up: {L} operands, "
          f"{L * hidden * 2 * inter:,} bf16 elements -- the fused fc11|fc12 "
          f"check, proving both halves survived in the right order)")
    for p in problems[:5]:
        print(f"   FAIL  {p}")
    return problems


# -- C. the stale-layout guard ----------------------------------------------

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


# -- D. zero-fill assertions, with a discriminating control -----------------

def _zero_tensor_names(r, cfg):
    names = ["embeddings.position"]
    for i in range(cfg["num_layers"]):
        names += [f"layer.{i}.qkv.bias", f"layer.{i}.attn_out.bias",
                  f"layer.{i}.ffn_up.bias", f"layer.{i}.ffn_down.bias"]
    return [n for n in names if n in r.entries]


def check_zero_fill(r, cfg, control_npue):
    print("\nD. zero-fill assertions (nomic has no biases and no position table)")
    problems = []

    names = _zero_tensor_names(r, cfg)
    nonzero = []
    for n in names:
        v = r.tensor(n)
        if not np.all(v == 0.0):
            nonzero.append((n, float(np.abs(v).max())))
    print(f"   arch=2 ({Path(r.path).name}): {len(names)} tensors checked "
          f"(embeddings.position + 4 x {cfg['num_layers']} biases)")
    if nonzero:
        print(f"   FAIL  {len(nonzero)} tensor(s) are NOT all-zero:")
        for n, m in nonzero[:5]:
            print(f"         {n}: max abs = {m:.3e}")
        problems.append(f"{len(nonzero)} arch=2 tensor(s) that should be "
                        f"zero-filled are not: {[n for n, _ in nonzero[:5]]}")
    else:
        print("   ok    every checked tensor is exactly zero")

    # Discriminating control: the SAME check against an arch=0 container must
    # find them NON-zero. A test that cannot fail proves nothing.
    if control_npue is None or not Path(control_npue).exists():
        print(f"   SKIP  control: {control_npue!r} not found -- cannot "
              f"discriminate 'always zero' from 'correctly zero'")
        problems.append("no arch=0 control container available -- check D is "
                        "unfalsifiable without it")
        return problems

    with Reader(control_npue) as cr:
        ccfg = cr.config
        cnames = ["embeddings.position"]
        for i in range(ccfg["num_layers"]):
            cnames += [f"layer.{i}.qkv.bias", f"layer.{i}.attn_out.bias",
                      f"layer.{i}.ffn_up.bias", f"layer.{i}.ffn_down.bias"]
        cnames = [n for n in cnames if n in cr.entries]
        all_zero_control = all(np.all(cr.tensor(n) == 0.0) for n in cnames)
    print(f"   control ({Path(control_npue).name}, arch={ccfg.get('arch')}): "
          f"{len(cnames)} equivalent tensors -- "
          f"{'ALL ZERO (control is broken)' if all_zero_control else 'non-zero, as expected'}")
    if all_zero_control:
        problems.append(f"control container {control_npue} has all-zero "
                        f"bias/position tensors too -- check D cannot "
                        f"discriminate 'zero because nomic has none' from "
                        f"'zero because something is broken'")
    return problems


# -- E. the scale-fold-through-RoPE proof ------------------------------------

def build_rope_cos_sin(seq_len, head_dim, theta, dtype=np.float64):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs).astype(dtype), np.sin(freqs).astype(dtype)


def rotate_half_neox(x):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos_half, sin_half):
    """NeoX-style RoPE -- concat(freqs,freqs), rotate-half -- tasks/0068 sec 5."""
    cos = np.concatenate([cos_half, cos_half], axis=-1)
    sin = np.concatenate([sin_half, sin_half], axis=-1)
    return x * cos + rotate_half_neox(x) * sin


def check_rope_fold(cfg, seq_len=64, seed=0):
    """Prove rope(scale*q) == scale*rope(q) numerically, at fp32.

    RoPE is a per-position rotation matrix R(pos) applied to q, and a
    rotation is LINEAR: R(pos) @ (s*q) = s * (R(pos) @ q) for any scalar s.
    So folding 1/sqrt(head_dim) into the Q weight BEFORE the GEMM (which is
    what pack_nomic does) must be exact up to floating-point rounding, since
    RoPE is applied to Q strictly AFTER the GEMM at runtime, and there is no
    Q bias to also fold (qkv_proj_bias is False). This does not depend on
    the packed weights at all -- it is a property of RoPE itself -- so it is
    tested directly against the same NeoX-style RoPE this container's
    config declares (theta from cfg["rope_theta"]).
    """
    print("\nE. scale-fold-through-RoPE proof (rope(s*q) == s*rope(q))")
    head_dim = cfg["head_dim"]
    theta = cfg["rope_theta"]
    scale = np.float32(1.0 / math.sqrt(head_dim))

    rng = np.random.default_rng(seed)
    # A stand-in for "x @ Q_weight" -- one head's worth of pre-RoPE Q
    # activations, fp32 (the precision the packed Q GEMM actually produces).
    q = rng.standard_normal((seq_len, head_dim)).astype(np.float32)

    cos_half, sin_half = build_rope_cos_sin(seq_len, head_dim, theta,
                                            dtype=np.float32)

    fold_then_rope = apply_rope(scale * q, cos_half, sin_half)      # what pack_nomic does
    rope_then_scale = scale * apply_rope(q, cos_half, sin_half)      # the un-folded alternative

    err = rel_fro(fold_then_rope, rope_then_scale)
    print(f"   fold-before-RoPE vs RoPE-then-scale, seq_len={seq_len}, "
          f"head_dim={head_dim}, theta={theta}: rel_fro = {err:.3e}")
    print(f"   {'ok    within fp32 round-off (< 1e-6)' if err < 1e-6 else 'FAIL  NOT exact'}")
    return [] if err < 1e-6 else [f"scale-fold-through-RoPE rel_fro {err:.3e} >= 1e-6"]


# -- F. config facts present and matching ------------------------------------

def check_config(cfg, checkpoint_cfg):
    print("\nF. config: rope_theta and swiglu_halves present and correct")
    problems = []

    checks = [
        ("rope_theta present", "rope_theta" in cfg),
        (f"rope_theta == checkpoint's rotary_emb_base "
         f"({checkpoint_cfg.get('rotary_emb_base')})",
         cfg.get("rope_theta") == checkpoint_cfg.get("rotary_emb_base")),
        ("swiglu_halves present", "swiglu_halves" in cfg),
        ('swiglu_halves == "fc11_up|fc12_gate"',
         cfg.get("swiglu_halves") == "fc11_up|fc12_gate"),
        ("gated_ffn is True", cfg.get("gated_ffn") is True),
        ("position_embedding_type == 'rope'",
         cfg.get("position_embedding_type") == "rope"),
        ("prompts table present", isinstance(cfg.get("prompts"), dict)
         and len(cfg["prompts"]) == 4),
        ("prompts_source labels the table as ours, not the checkpoint's",
         "npuembeddings" in cfg.get("prompts_source", "").lower()
         and "not from the checkpoint" in cfg.get("prompts_source", "").lower()),
    ]
    for label, ok in checks:
        print(f"   {'ok  ' if ok else 'FAIL'}  {label}")
        if not ok:
            problems.append(label)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-embed-text-v1.5",
                    help="sets --npue and --model-dir together")
    ap.add_argument("--npue")
    ap.add_argument("--model-dir")
    ap.add_argument("--control-npue", default=str(REPO / "models" / "bge-base-en-v1.5.npue"),
                    help="an arch=0 container, for check D's discriminating control")
    args = ap.parse_args()

    if not args.npue:
        args.npue = str(REPO / "models" / f"{args.model}.npue")
    if not args.model_dir:
        with Reader(args.npue) as _r0:
            _repo = _r0.config["source_repo"]
        args.model_dir = str(REPO / "models" / _repo.split("/")[-1])

    r = Reader(args.npue)
    cfg = r.config
    print(f"{Path(args.npue).name}  "
          f"({Path(args.npue).stat().st_size/1e6:.2f} MB, {len(r.entries)} tensors)")
    print(f"  arch={cfg.get('arch')}  tile ({cfg['tile_k']}, {cfg['tile_n']}), "
          f"mac (s={cfg['mac_s']}, t={cfg['mac_t']})")
    print(f"  fusions: {', '.join(k for k, v in cfg['fusions'].items() if v is True)}\n")

    checkpoint_cfg_path = Path(args.model_dir) / "config.json"
    import json
    checkpoint_cfg = json.loads(checkpoint_cfg_path.read_text(encoding="utf-8"))
    src, _ = load(Path(args.model_dir) / "model.safetensors")

    problems = []
    problems += check_spec(args.npue, r)
    problems += check_roundtrip(r, src, cfg, cfg["fusions"]["qk_scale_folded_into_q"])
    problems += check_guard(r)
    problems += check_zero_fill(r, cfg, args.control_npue)
    problems += check_rope_fold(cfg)
    problems += check_config(cfg, checkpoint_cfg)

    if problems:
        print(f"\nFAIL -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS -- spec conformant, round-trip bit-exact, layout guarded, "
          "zero-fills verified against a live control, scale fold proved "
          "exact through RoPE, config facts present and correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
