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
from npue import (ARCH_GEMMA3_MQA_ROPE_GEGLU, ARCH_NOMIC_ROPE_SWIGLU, Writer,  # noqa: E402
                  gemm_b_layout, layout_hash, tile_b, to_bf16_bits)
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


def add_gemm_b_host(w, name, mat):
    """Stage a [K,N] GEMM operand PLAIN: F32, row-major, no tiling.

    Gemma has no NPU kernel yet (tasks/0064-m12-embeddinggemma-arch1-integration): every GEMM in this arch runs on
    the HOST, so there is no DMA descriptor to pre-tile for and no
    layout_hash to check -- `layout=None` (Writer.add's default) is exactly
    that: an entry with no "layout" key, which npue.py's Reader.tensor() and
    the C++ loader both already treat as "not tiled, read [K,N] row-major".
    F32, not bf16: this checkpoint's own weights are F32 on disk (verified in
    tasks/0055 by direct safetensors inspection, contradicting an earlier
    assumption that it shipped bf16 -- see this task's TASK.md), and the
    point of this task is a correctness gate, not a size/speed one; a bf16
    weight path is future work the container format does not block.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    K, N = mat.shape
    return w.add(name, mat.reshape(-1), "F32", "gemm_b_host", [K, N])


def pack_gemma(model_dir, out, source_repo_override=None):
    """Pack an EmbeddingGemma-300M-shaped checkpoint (arch=1).

    Deliberately NOT the BERT path above, reused only via helpers (Writer,
    sha256) -- see tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md for
    why: 4 RMSNorms/layer (not 2 LayerNorms), MQA (num_key_value_heads=1,
    NOT hidden/num_heads), q_norm/k_norm, per-layer RoPE base, separate
    gate/up GeGLU matrices, two post-pool Dense heads, no biases anywhere
    (attention_bias=false, both Dense heads bias=false), no
    token_type_embeddings, no absolute position table (RoPE instead).
    Every architectural fact below is read from reference/encoder_gemma.py
    (tasks/0055, validated 1-cos 1.065e-07 against real HuggingFace) rather
    than re-derived.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")

    L = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    swp = cfg.get("_sliding_window_pattern", 6)

    print(f"packing {model_dir.name} -> {Path(out).name}  (arch=gemma3, HOST-only GEMMs)")
    print(f"  hidden={hidden} heads={heads} kv_heads={kv_heads} head_dim={head_dim} "
          f"layers={L} inter={inter}")

    source_repo = source_repo_override
    if not source_repo:
        ckpt = model_dir / "CHECKPOINT.json"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} not found and no source_repo given -- "
                             f"refusing to guess which repository these weights "
                             f"came from")
        source_repo = json.loads(ckpt.read_text(encoding="utf-8"))["repo_id"]

    config = {
        "arch": "gemma3_mqa_rope_geglu",
        "model_type": cfg["model_type"],
        "source_repo": source_repo,
        "source_sha256": src_sha,
        "num_layers": L, "hidden": hidden,
        "num_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate": inter,
        "dense_hidden": int(d2["linear.weight"].shape[0]),
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "rope_local_base_freq": cfg["rope_local_base_freq"],
        "sliding_window": cfg["sliding_window"],
        "sliding_window_pattern": swp,
        "query_pre_attn_scalar": cfg["query_pre_attn_scalar"],
        "vocab_size": cfg["vocab_size"],
        # Informational only: unlike BERT's absolute position TABLE (which
        # really is sliced to this many rows), Gemma has no position
        # embedding to pre-slice -- RoPE tables are computed at runtime for
        # whatever sequence length is asked for. This is the HF config's own
        # `max_position_embeddings`, a ceiling, not a packed array size.
        "max_seq_len": cfg["max_position_embeddings"],
        "pooling": "mean_include_prompt", "l2_normalize": True,
        "activation": "gelu_pytorch_tanh",
        "attention_bias": False, "dense_bias": False,
        "not_implemented": [
            "sliding-window mask (exact for seq_len<=512, see "
            "reference/encoder_gemma.py's file header)",
        ],
    }

    w = Writer(config, arch=ARCH_GEMMA3_MQA_ROPE_GEGLU)

    w.add("embed_tokens.weight", src["embed_tokens.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    w.add("norm.weight", src["norm.weight"], "F32", "layernorm", [hidden])

    tok_path = model_dir / "gemma_tokenizer.bin"
    if tok_path.exists():
        tb = np.frombuffer(tok_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.gemma_table", tb, "U8", "tokenizer", [int(tb.size)])
        print(f"  tokenizer.gemma_table  {tb.size / 1e6:.2f} MB")
    else:
        print(f"  WARNING: {tok_path} not found (run "
              f"tools/gen_gemma_tokenizer_table.py) -- .npue will have no "
              f"tokenizer table")

    for i in range(L):
        p = f"layers.{i}."
        sa = p + "self_attn."

        add_gemm_b_host(w, f"layer.{i}.q_proj",
                         np.ascontiguousarray(src[sa + "q_proj.weight"].T))
        add_gemm_b_host(w, f"layer.{i}.k_proj",
                         np.ascontiguousarray(src[sa + "k_proj.weight"].T))
        add_gemm_b_host(w, f"layer.{i}.v_proj",
                         np.ascontiguousarray(src[sa + "v_proj.weight"].T))
        w.add(f"layer.{i}.q_norm.weight", src[sa + "q_norm.weight"],
              "F32", "layernorm", [head_dim])
        w.add(f"layer.{i}.k_norm.weight", src[sa + "k_norm.weight"],
              "F32", "layernorm", [head_dim])
        add_gemm_b_host(w, f"layer.{i}.o_proj",
                         np.ascontiguousarray(src[sa + "o_proj.weight"].T))

        w.add(f"layer.{i}.input_layernorm.weight",
              src[p + "input_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.post_attention_layernorm.weight",
              src[p + "post_attention_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.pre_feedforward_layernorm.weight",
              src[p + "pre_feedforward_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.post_feedforward_layernorm.weight",
              src[p + "post_feedforward_layernorm.weight"], "F32", "layernorm", [hidden])

        mp = p + "mlp."
        add_gemm_b_host(w, f"layer.{i}.gate_proj",
                         np.ascontiguousarray(src[mp + "gate_proj.weight"].T))
        add_gemm_b_host(w, f"layer.{i}.up_proj",
                         np.ascontiguousarray(src[mp + "up_proj.weight"].T))
        add_gemm_b_host(w, f"layer.{i}.down_proj",
                         np.ascontiguousarray(src[mp + "down_proj.weight"].T))

    add_gemm_b_host(w, "dense2.weight", np.ascontiguousarray(d2["linear.weight"].T))
    add_gemm_b_host(w, "dense3.weight", np.ascontiguousarray(d3["linear.weight"].T))

    info = w.write(out)
    total = Path(out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    return 0


def pack_nomic(model_dir, out, tile_k, tile_n, max_seq, fold_scale):
    """Pack a nomic-embed-text-v1.5-shaped checkpoint (arch=2).

    Emits the SAME tensor names and the SAME emission order as the BERT
    (arch=0) path above -- tasks/0069-m13-nomic-arch2-container/TASK.md item
    3 -- so Encoder::stage_all() and the whole NPU dispatch path work
    UNCHANGED. Every architectural fact asserted below (post-LN, SiLU on
    fc12 not fc11, no mlp.norm, RoPE NeoX-style on Q/K only starting at
    position 0, theta=1000, three-major Wqkv row order, no biases anywhere)
    was settled EMPIRICALLY against the real checkpoint in tasks/0068 (see
    its TASK.md sec 5), not re-derived here -- this function only implements
    that already-settled architecture and asserts the config facts it
    depends on, so a checkpoint that silently changed underneath it would
    refuse to pack rather than pack wrong.

    Departures from BERT:
      * no absolute position table (RoPE instead) -- zero-filled placeholder
      * no biases anywhere (qkv_proj_bias / mlp_fc1_bias / mlp_fc2_bias all
        False) -- zero-filled placeholders, same rationale
      * gated SwiGLU FFN: fc11 (untouched up-path) and fc12 (SiLU gate) are
        fused into ONE [hidden, 2*intermediate] ffn_up along the N axis, so
        the array still sees four GEMMs per layer, not five --
        out = fc11(x) * silu(fc12(x)) = lo * silu(hi)
      * 1/sqrt(head_dim) is folded into the Q block exactly as the BERT path
        does. This is legal here only because RoPE is a rotation and
        therefore LINEAR in q: rope(s*q) = s*rope(q), so folding the scale
        before the GEMM and before RoPE is exact. tools/verify_npue_nomic.py
        check E proves this numerically rather than assuming it.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")

    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    scale = 1.0 / math.sqrt(head_dim)

    if hidden != H * head_dim:
        raise SystemExit(f"hidden={hidden} != num_heads={H} * head_dim={head_dim}")
    if head_dim % 2:
        raise SystemExit(f"head_dim={head_dim} is odd -- RoPE cannot "
                         f"half-split it into rotation pairs")

    # rope_theta: ASSERT, never default. tasks/0068 measured a wrong theta
    # at rel_fro 9.2e-02 on the attention output -- the ONE wrong reading in
    # that whole probe subtle enough to slip past a loose gate (every other
    # wrong reading there was 0.5-5.0).
    theta = cfg["rotary_emb_base"]
    if theta != 1000:
        raise SystemExit(f"rotary_emb_base={theta}, expected 1000 -- "
                         f"refusing to pack against an unverified RoPE base")

    # layer_norm_epsilon and layer_norm_eps are two keys for the same value
    # in this checkpoint's config.json -- read one, assert they agree rather
    # than silently picking one and hoping.
    eps_a, eps_b = cfg["layer_norm_epsilon"], cfg["layer_norm_eps"]
    if eps_a != eps_b:
        raise SystemExit(f"layer_norm_epsilon ({eps_a}) != layer_norm_eps "
                         f"({eps_b}) -- checkpoint is internally inconsistent")
    eps = eps_a

    for flag in ("qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"):
        if cfg[flag] is not False:
            raise SystemExit(f"{flag}={cfg[flag]!r}, expected False -- this "
                             f"packer zero-fills every bias on the assumption "
                             f"nomic has none; a checkpoint with real biases "
                             f"would be packed WRONG")
    if cfg["prenorm"] is not False:
        raise SystemExit(f"prenorm={cfg['prenorm']!r}, expected False "
                         f"(post-LN block order -- tasks/0068 Q1)")
    if cfg["activation_function"] != "swiglu" or cfg["hidden_act"] != "silu":
        raise SystemExit(f"activation_function={cfg['activation_function']!r} "
                         f"hidden_act={cfg['hidden_act']!r}, expected "
                         f"'swiglu'/'silu'")
    if cfg["rotary_emb_interleaved"] is not False:
        raise SystemExit(f"rotary_emb_interleaved={cfg['rotary_emb_interleaved']!r} "
                         f"-- this packer/runtime assumes NeoX-style RoPE "
                         f"(concat(freqs,freqs), rotate-half) -- tasks/0068")
    if cfg["rotary_emb_fraction"] != 1.0:
        raise SystemExit(f"rotary_emb_fraction={cfg['rotary_emb_fraction']}, "
                         f"expected 1.0 (whole head rotated)")

    print(f"packing {model_dir.name} -> {Path(out).name}  (arch=nomic_bert_rope_swiglu)")
    print(f"  hidden={hidden} heads={H} head_dim={head_dim} layers={L} "
          f"inter={inter} rope_theta={theta}")
    print(f"  tile ({tile_k}, {tile_n}), mac (s={MAC_S}, t={MAC_T}), "
          f"1/sqrt({head_dim}) = {scale:.17g}"
          f"{' folded into Q' if fold_scale else ' NOT folded'}")

    # This project's OWN choice, not the checkpoint's -- labelled as such in
    # the container for the same reason tools/gen_gemma_tokenizer_table.py
    # labels its table (see its lines 63-77):
    # config_sentence_transformers.json for this checkpoint carries no
    # "prompts" dict at all (verified tasks/0068 sec 4/10), so presenting
    # this table as read-from-the-checkpoint would be a lie in a file other
    # tools read. Prefix strings and token costs measured in tasks/0068 sec 4.
    # How many embedding rows the tokenizer can actually reach. nomic pads
    # vocab_size up to a multiple of 64 (pad_vocab_size_multiple), and those
    # extra rows are NOT zero -- ordinary trained-looking values that no token
    # id can ever select (tasks/0068 sec 5c). Counted, not assumed.
    n_reachable = (len((model_dir / "vocab.txt").read_bytes()
                       .decode("utf-8").splitlines())
                   if (model_dir / "vocab.txt").exists() else 0)

    prompts = {
        "search_document": "search_document: ",
        "search_query": "search_query: ",
        "clustering": "clustering: ",
        "classification": "classification: ",
    }

    config = {
        "arch": "nomic_bert_rope_swiglu",
        "model_type": cfg["model_type"],
        "source_repo": json.loads(
            (model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))["repo_id"],
        "source_sha256": src_sha,
        "num_layers": L, "num_heads": H, "hidden": hidden, "head_dim": head_dim,
        "intermediate": inter,
        "layer_norm_eps": eps,
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": max_seq,
        "pooling": read_pooling(model_dir), "l2_normalize": True,
        "activation": "silu", "gated_ffn": True,
        "swiglu_halves": "fc11_up|fc12_gate",
        "position_embedding_type": "rope",
        "rope_theta": theta,
        "attention_bias": False, "mlp_bias": False,
        "tile_k": tile_k, "tile_n": tile_n, "mac_s": MAC_S, "mac_t": MAC_T,
        "prompts": prompts,
        "prompt_default": "search_document",
        "prompts_source": "npuembeddings, NOT from the checkpoint -- "
                          "config_sentence_transformers.json carries no "
                          "'prompts' dict for this checkpoint, so presenting "
                          "this table as the model's own would be a lie in a "
                          "file other tools read. Same precedent as "
                          "tools/gen_gemma_tokenizer_table.py:63-77.",
        "l2_normalize_note": "sentence-transformers does NOT L2-normalize "
                             "this model (measured output norm 20.93, "
                             "tasks/0068 sec 5b) -- l2_normalize:true here "
                             "matches THIS RUNTIME's own hardcoded behaviour "
                             "(main.cpp g_l2_normalize) and nomic's own "
                             "documented usage (F.normalize), not "
                             "sentence-transformers' default pipeline for "
                             "this particular model.",
        "fusions": {
            "qkv_fused": True,
            "transposed_to_kn": True,
            "qk_scale_folded_into_q": fold_scale,
            "gemm_operands_bf16": True,
            "biases_and_layernorm_fp32": True,
            "gated_ffn_fused_fc11_fc12": True,
            "position_embeddings_zeroed_rope_instead": True,
        },
        # Copied from the BERT path, this said "pooler.dense" -- which this
        # checkpoint does not have. All 112 of its tensors are consumed
        # (tasks/0068 sec 1), so there is no dead weight to declare, and a
        # claim about a tensor that does not exist is worse than no claim.
        # What IS genuinely not implemented:
        "not_implemented": [
            "Matryoshka truncation (layer_norm(768) -> slice -> normalize is a "
            "different post-processing chain, not just a shorter vector)",
            f"vocab rows {n_reachable}-{cfg['vocab_size'] - 1} are "
            f"pad_vocab_size_multiple padding: non-zero but unreachable from "
            f"the tokenizer (max id {n_reachable - 1}), packed only so "
            f"vocab_size and the tensor agree",
        ],
    }

    w = Writer(config, arch=ARCH_NOMIC_ROPE_SWIGLU)

    # -- embeddings: SAME order as the BERT path, including the odd
    # ln.weight -> tokenizer.vocab -> ln.bias interleaving, which is
    # load-bearing for byte parity with the C++ mirror. --------------------
    w.add("embeddings.word", src["embeddings.word_embeddings.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    # nomic has NO position table -- RoPE is computed inside attention
    # instead. Zero-filled rather than omitted: Encoder::stage_all() and the
    # --embed path both dereference "embeddings.position" UNCONDITIONALLY
    # (main.cpp:2889), so a zero tensor of the right shape is exact (adds
    # nothing) and keeps that read path untouched.
    w.add("embeddings.position", np.zeros((max_seq, hidden), dtype=np.float32),
          "F32", "embedding", [max_seq, hidden])
    w.add("embeddings.token_type", src["embeddings.token_type_embeddings.weight"],
          "F32", "embedding", [cfg["type_vocab_size"], hidden])
    # emb_ln lives at the TOP LEVEL upstream (not embeddings.LayerNorm, as
    # in BERT) -- tasks/0068 sec 1.
    w.add("embeddings.ln.weight", src["emb_ln.weight"],
          "F32", "layernorm", [hidden])
    vocab_path = model_dir / "vocab.txt"
    if vocab_path.exists():
        vb = np.frombuffer(vocab_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.vocab", vb, "U8", "tokenizer", [int(vb.size)])
        print(f"  tokenizer.vocab   {vb.size / 1024:.1f} KB "
              f"({vocab_path.read_bytes().count(chr(10).encode()[0])} lines)")
    else:
        print(f"  WARNING: {vocab_path} not found -- .npue will have no vocab")
    w.add("embeddings.ln.bias", src["emb_ln.bias"],
          "F32", "layernorm", [hidden])

    n_tiled = 0
    for i in range(L):
        p = f"encoder.layers.{i}."   # plural upstream, unlike BERT's "layer."
        attn = p + "attn."

        # fused upstream already: Wqkv is [2304,768] three-major
        # [Q(768)|K(768)|V(768)] -- tasks/0068 sec 5 Wqkv row-order check.
        qkv = np.ascontiguousarray(src[attn + "Wqkv.weight"].T)      # [768,2304]
        if fold_scale:
            # RoPE is linear in q, so folding 1/sqrt(head_dim) into the Q
            # block before the GEMM (and before RoPE) is exact -- see
            # tools/verify_npue_nomic.py check E. No qkv bias exists to fold.
            qkv = qkv.copy()
            qkv[:, :hidden] *= scale
        add_gemm_b(w, f"layer.{i}.qkv", qkv, tile_k, tile_n)
        w.add(f"layer.{i}.qkv.bias", np.zeros(3 * hidden, dtype=np.float32),
              "F32", "bias", [3 * hidden])
        n_tiled += 1

        add_gemm_b(w, f"layer.{i}.attn_out",
                   np.ascontiguousarray(src[attn + "out_proj.weight"].T),
                   tile_k, tile_n)
        w.add(f"layer.{i}.attn_out.bias", np.zeros(hidden, dtype=np.float32),
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln1.weight", src[p + "norm1.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln1.bias", src[p + "norm1.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 1

        # gated ffn_up: [fc11 (up, untouched) | fc12 (gate, gets SiLU)]
        # fused along N. Runtime computes out = lo * silu(hi), where
        # lo = cols [0, inter), hi = cols [inter, 2*inter) -- see
        # config["swiglu_halves"]. ONE GEMM, so the array still sees four
        # GEMMs per layer, not five.
        mp = p + "mlp."
        up = np.ascontiguousarray(src[mp + "fc11.weight"].T)         # [768,3072]
        gate = np.ascontiguousarray(src[mp + "fc12.weight"].T)       # [768,3072]
        ffn_up = np.concatenate([up, gate], axis=1)                  # [768,6144]
        add_gemm_b(w, f"layer.{i}.ffn_up", ffn_up, tile_k, tile_n)
        w.add(f"layer.{i}.ffn_up.bias", np.zeros(2 * inter, dtype=np.float32),
              "F32", "bias", [2 * inter])

        add_gemm_b(w, f"layer.{i}.ffn_down",
                   np.ascontiguousarray(src[mp + "fc2.weight"].T),
                   tile_k, tile_n)
        w.add(f"layer.{i}.ffn_down.bias", np.zeros(hidden, dtype=np.float32),
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln2.weight", src[p + "norm2.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln2.bias", src[p + "norm2.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 2

    info = w.write(out)

    print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
          f"{'tiles':>7} {'max BD dim':>11}")
    shapes = {"qkv": (hidden, 3 * hidden), "attn_out": (hidden, hidden),
              "ffn_up": (hidden, 2 * inter), "ffn_down": (inter, hidden)}
    for nm, (K, N) in shapes.items():
        kb, nb = K // tile_k, N // tile_n
        flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
        print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} {kb*nb:>7} "
              f"{max(kb, nb):>11}{flag}")

    total = Path(out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T))[:16]}...")
    return 0


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

    # arch=1 branch (tasks/0064-m12-embeddinggemma-arch1-integration): a Gemma3-family checkpoint is a completely
    # different tensor shape and a completely different container -- routed
    # to its own function rather than threaded through the BERT logic below,
    # so the BERT path (in production since M4) is unchanged code, not
    # refactored code. Detected from the checkpoint's OWN config.json
    # (model_type), never assumed from --model-dir's directory name.
    if cfg.get("model_type") == "gemma3_text":
        out = args.out
        if out == str(REPO / "models" / "all-MiniLM-L6-v2.npue"):
            out = str(model_dir.parent / (model_dir.name + ".npue"))
        return pack_gemma(model_dir, out)

    # arch=2 branch (tasks/0069-m13-nomic-arch2-container): nomic_bert is
    # RoPE + gated SwiGLU rather than BERT's absolute-position + GELU, so it
    # gets its own function -- same reasoning as the gemma3_text branch
    # above. Unlike Gemma, its GEMM operands DO get pre-tiled (see
    # pack_nomic's docstring), so --tile-k/--tile-n/--max-seq are forwarded.
    if cfg.get("model_type") == "nomic_bert":
        out = args.out
        if out == str(REPO / "models" / "all-MiniLM-L6-v2.npue"):
            out = str(model_dir.parent / (model_dir.name + ".npue"))
        return pack_nomic(model_dir, out, args.tile_k, args.tile_n,
                          args.max_seq, not args.no_fold_scale)

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
