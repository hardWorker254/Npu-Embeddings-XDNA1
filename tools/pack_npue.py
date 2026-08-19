# NpuEmbeddings -- M4: pack a HuggingFace checkpoint into a pre-tiled .npue.
#
# Applies the five offline fusions from docs/04-model. All are pure weight
# rewrites with zero runtime cost, and each removes work from the hot path:
#
#   1. Fuse Q, K, V into one [384, 1152] B matrix + [1152] bias.
#      One GEMM instead of three, and 3x better weight reuse per DMA.
#   2. Transpose everything to [K, N] so the runtime never transposes.
#      nn.Linear stores [out, in]; the transpose is free offline.
#   3. Convert GEMM operands to bf16, but keep LayerNorm gamma/beta and every
#      bias in fp32 -- they are 384 floats each and numerically sensitive.
#   4. Fold 1/sqrt(head_dim) into the Q weight and bias, so the attention
#      kernel has no scale multiply.
#   5. Pre-slice position_embeddings to the real max sequence length (256 for
#      MiniLM, NOT the 512 in config.json -- docs/04-model, sharp edge 1).
#
# And the point of the milestone: PRE-TILE every GEMM operand, which is both
# the main performance lever now that M2 showed we are data-movement bound, and
# the only way to express ffn_down at all (K=1536 > the 1023 DMA BD limit).
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py
#   ... --tile-n 32 --out models\minilm_n32.npue

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

# gemm_b_layout was factored into npue.py so the descriptor has ONE definition
# (its docstring records two copies drifting apart). The import was never added
# here, so pack_npue.py has not run since that refactor -- the shipped .npue
# predates it and nothing repacked. Found while adding the vocabulary, 0036.
from npue import (Writer, gemm_b_layout, layout_hash, tile_b,   # noqa: E402
                  to_bf16_bits)
from safetensors_io import load                              # noqa: E402

# From M2's traced results: mac_dims are (r,s,t) = (4,8,8) for plain bf16 on
# npu2 and (8,8,8) with bfp16 emulation. Only s and t affect the B operand
# layout, and BOTH configurations give s=t=8 -- so the pre-tiled B layout is
# the same either way, and the bf16/bfp16 decision does not force a repack.
MAC_S, MAC_T = 8, 8

# tile_n=48, not M2's winning 32. At 8 columns the design requires
# N % (tile_n * n_cols) == 0, and MiniLM's N dims are 384 / 1152 / 1536:
#   gcd(384/8, 1152/8, 1536/8) = 48, so tile_n must divide 48.
# n=32 fails at 8 columns (1152/256 = 4.5) even though it was M2's best at 4.
# 48 is the largest legal choice, needs zero padding on all three shapes, and
# fits L1: 2*(64*64*2 + 64*48*2 + 64*48*4) = 53,248 < 65,536.
DEFAULT_TILE_K, DEFAULT_TILE_N = 64, 48


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pooling(model_dir):
    """Which pooling the checkpoint actually uses, from its own metadata.

    sentence-transformers records this in 1_Pooling/config.json and
    fetch_model.py already downloads it; until now both packers wrote "mean"
    as a literal, which is right for MiniLM and wrong for every bge model.

    Ambiguity REFUSES rather than picking. A model using max or
    mean_sqrt_len pooling is not something this runtime implements, and
    approximating it with mean would be a silent quality loss.
    """
    p = Path(model_dir) / "1_Pooling" / "config.json"
    if not p.exists():
        raise SystemExit(f"{p} not found -- cannot determine pooling mode. "
                         f"Re-fetch with reference/fetch_model.py.")
    c = json.loads(p.read_text(encoding="utf-8"))
    modes = [k for k, v in c.items()
             if k.startswith("pooling_mode_") and v is True]
    if modes == ["pooling_mode_cls_token"]:
        return "cls"
    if modes == ["pooling_mode_mean_tokens"]:
        return "mean"
    raise SystemExit(f"{p}: this runtime implements cls and mean pooling; "
                     f"the checkpoint asks for {modes or 'nothing'}")


def add_gemm_b(w, name, mat, tile_k, tile_n, fold=None):
    """Stage a [K,N] GEMM operand: optional scale fold, bf16, pre-tile."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * fold
    K, N = mat.shape
    layout = gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T)
    bits = to_bf16_bits(mat)
    flat = tile_b(bits, tile_k, tile_n, MAC_S, MAC_T)
    return w.add(name, flat, "BF16", "gemm_b", [K, N], layout=layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--out", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--tile-k", type=int, default=DEFAULT_TILE_K)
    ap.add_argument("--tile-n", type=int, default=DEFAULT_TILE_N)
    ap.add_argument("--max-seq", type=int, default=256,
                    help="pre-slice position embeddings to this (MiniLM: 256)")
    ap.add_argument("--no-fold-scale", action="store_true",
                    help="do NOT fold 1/sqrt(head_dim) into Q")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")

    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // H
    scale = 1.0 / math.sqrt(head_dim)

    tk, tn = args.tile_k, args.tile_n
    print(f"packing {model_dir.name} -> {Path(args.out).name}")
    print(f"  tile ({tk}, {tn}), mac (s={MAC_S}, t={MAC_T}), "
          f"1/sqrt({head_dim}) = {scale:.17g}"
          f"{' NOT folded' if args.no_fold_scale else ' folded into Q'}")

    config = {
        "arch": "bert_abs_gelu_postln",
        "source_repo": json.loads(
            (model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))["repo_id"],
        "source_sha256": src_sha,
        "num_layers": L, "num_heads": H, "hidden": hidden, "head_dim": head_dim,
        "intermediate": cfg["intermediate_size"],
        "layer_norm_eps": cfg["layer_norm_eps"],
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": args.max_seq,
        "pooling": read_pooling(model_dir), "l2_normalize": True,
        "activation": "gelu_erf_exact",
        "tile_k": tk, "tile_n": tn, "mac_s": MAC_S, "mac_t": MAC_T,
        "fusions": {
            "qkv_fused": True,
            "transposed_to_kn": True,
            "qk_scale_folded_into_q": not args.no_fold_scale,
            "gemm_operands_bf16": True,
            "biases_and_layernorm_fp32": True,
            "position_embeddings_presliced_to": args.max_seq,
        },
        "not_implemented": ["pooler.dense (unused by sentence-transformers)"],
    }

    w = Writer(config)

    # -- embeddings: gathered, never multiplied, so never tiled --------------
    # Kept fp32 deliberately: a memory-bound gather costs nothing extra in
    # compute, and it removes one source of numerical drift at the input.
    w.add("embeddings.word", src["embeddings.word_embeddings.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    w.add("embeddings.position",
          src["embeddings.position_embeddings.weight"][:args.max_seq],
          "F32", "embedding", [args.max_seq, hidden])
    w.add("embeddings.token_type", src["embeddings.token_type_embeddings.weight"],
          "F32", "embedding", [cfg["type_vocab_size"], hidden])
    w.add("embeddings.ln.weight", src["embeddings.LayerNorm.weight"],
          "F32", "layernorm", [hidden])
    # The tokenizer vocabulary, verbatim, as bytes. It is not model data and
    # it is not touched by any kernel -- it rides here so that deploying the
    # model is copying ONE file. The C++ tokenizer reads it straight out of
    # the mapping, exactly as the weights are read.
    vocab_path = model_dir / "vocab.txt"
    if vocab_path.exists():
        vb = np.frombuffer(vocab_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.vocab", vb, "U8", "tokenizer", [int(vb.size)])
        print(f"  tokenizer.vocab   {vb.size / 1024:.1f} KB "
              f"({vocab_path.read_bytes().count(chr(10).encode()[0])} lines)")
    else:
        print(f"  WARNING: {vocab_path} not found -- .npue will have no vocab")

    w.add("embeddings.ln.bias", src["embeddings.LayerNorm.bias"],
          "F32", "layernorm", [hidden])

    n_tiled = 0
    for i in range(L):
        p = f"encoder.layer.{i}."
        sa = p + "attention.self."

        # fusion 1 + 2: concat Q|K|V along `out`, then transpose to [K, N].
        qkv = np.concatenate([src[sa + n + ".weight"]
                              for n in ("query", "key", "value")], axis=0)   # [1152,384]
        qkv_b = np.concatenate([src[sa + n + ".bias"]
                                for n in ("query", "key", "value")], axis=0)  # [1152]
        qkv = np.ascontiguousarray(qkv.T)                                    # [384,1152]

        # fusion 4: fold 1/sqrt(head_dim) into the Q block only -- the first
        # `hidden` columns of the fused N axis, and the matching bias slice.
        if not args.no_fold_scale:
            qkv = qkv.copy()
            qkv[:, :hidden] *= scale
            qkv_b = qkv_b.copy()
            qkv_b[:hidden] *= scale

        add_gemm_b(w, f"layer.{i}.qkv", qkv, tk, tn)
        w.add(f"layer.{i}.qkv.bias", qkv_b, "F32", "bias", [3 * hidden])
        n_tiled += 1

        ao = p + "attention.output."
        add_gemm_b(w, f"layer.{i}.attn_out",
                   np.ascontiguousarray(src[ao + "dense.weight"].T), tk, tn)
        w.add(f"layer.{i}.attn_out.bias", src[ao + "dense.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln1.weight", src[ao + "LayerNorm.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln1.bias", src[ao + "LayerNorm.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 1

        add_gemm_b(w, f"layer.{i}.ffn_up",
                   np.ascontiguousarray(src[p + "intermediate.dense.weight"].T), tk, tn)
        w.add(f"layer.{i}.ffn_up.bias", src[p + "intermediate.dense.bias"],
              "F32", "bias", [cfg["intermediate_size"]])
        add_gemm_b(w, f"layer.{i}.ffn_down",
                   np.ascontiguousarray(src[p + "output.dense.weight"].T), tk, tn)
        w.add(f"layer.{i}.ffn_down.bias", src[p + "output.dense.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln2.weight", src[p + "output.LayerNorm.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln2.bias", src[p + "output.LayerNorm.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 2

    info = w.write(args.out)

    # What the DMA will actually see, per distinct GEMM shape. The 1023 limit is
    # on a BD size field, so what matters is that no access-pattern dimension
    # reaches it -- with whole tiles read linearly, the dims are tile counts.
    print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
          f"{'tiles':>7} {'max BD dim':>11}")
    shapes = {"qkv": (hidden, 3 * hidden), "attn_out": (hidden, hidden),
              "ffn_up": (hidden, cfg["intermediate_size"]),
              "ffn_down": (cfg["intermediate_size"], hidden)}
    for nm, (K, N) in shapes.items():
        kb, nb = K // tk, N // tn
        flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
        print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} {kb*nb:>7} "
              f"{max(kb, nb):>11}{flag}")
    print(f"  (row-major DDR would need K={cfg['intermediate_size']} as a BD "
          f"dimension for ffn_down -- over 1023, which is why it could not be "
          f"expressed before)")

    total = Path(args.out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(tk, tn, MAC_S, MAC_T))[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
