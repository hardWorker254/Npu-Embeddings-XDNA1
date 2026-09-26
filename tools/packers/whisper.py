# NpuEmbeddings -- pack an openai/whisper-* checkpoint into an arch=4 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# THE ONE ARCHITECTURE THAT IS NOT AN EMBEDDER. Every other container in this
# tree answers "vector for text". This one answers "text for audio", so it
# carries two stacks and a frontend:
#
#   audio -> log-mel -> conv1 -> GELU -> conv2 -> GELU -> + positional
#         -> 32 (or 4, or 24, or 6, or 4) encoder layers
#         -> decoder layers, each self-attending over its own output so far and
#            cross-attending over the encoder's, until <|endoftext|>
#         -> logits = h @ embed_tokens.T   (the checkpoint has no proj_out)
#         -> token ids -> text
#
# WHAT IS FUSED, and why each fusion is exact rather than approximate:
#
#   1. Q|K|V into one [d, 3d] operand per attention, as in every other arch.
#      Whisper's k_proj has NO BIAS, so the fused bias is
#      [q_bias*scale | 0 | v_bias*scale]: the K half being zero reproduces
#      k = x*Wk exactly, and the array still sees one GEMM per attention.
#   2. 1/sqrt(head_dim) folded into Q's weight AND Q's bias, for the decoder's
#      self-attention and its cross-attention both -- HF scales both queries.
#      A GEMM is linear, so folding before the GEMM and folding after it agree.
#   3. Cross-attention's K|V into one [d, 2d] operand, with the same
#      zero-K-bias trick. Its Q is a separate [d, d] operand because its A
#      operand is the decoder state, not the encoder output.
#   4. Every GEMM operand pre-tiled to bf16 block_panel at (64, 32).
#   5. LayerNorm gamma/beta, every bias, the two conv stacks, both embedding
#      tables and the tokenizer table stay fp32/raw.
#
# WHY (64, 32) FOR ALL SIX SIZES: it is forced by the geometry, not chosen.
# K is d or 4d and N is d, 2d, 3d or 4d, and for d in {384, 512, 768, 1024,
# 1280} both d % 64 == 0 and d/32 is a multiple of the 4 AIE columns, so every
# operand tiles with no padding. tile_n 48, which every BERT-family container
# uses, does not divide 5120 -- large-v3's FFN -- and would force a per-size
# repack and a per-size design set.
#
# Env: numpy only (plus the repo's own npue.py and safetensors_mmap.py).

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from npue import (ARCH_WHISPER_ENC_DEC_GELU, MAC_BY_DEVICE,  # noqa: E402
                  MAC_DEFAULT_DEVICE, Writer, gemm_b_layout, layout_hash,
                  mac_for_device, tile_b, to_bf16_bits)
from safetensors_mmap import SafeTensors                              # noqa: E402
from whisper_bpe import VocabMerges, build_table                      # noqa: E402

# Only s and t of the MAC geometry affect the B operand's byte order, and both
# production datapaths (plain bf16 and bfp16-emulated) give the same pair on a
# given board -- so the layout does not depend on the datapath. It DOES depend
# on the board: npu1's MMAC sub-tile is (s=8, t=4) and npu2's is (8, 8), so
# `mac` is resolved from the target device and threaded into every operand
# rather than fixed here. The wrong pair is unreadable only by the hardware:
# same byte count, same shapes, same layout_hash on both sides, plausible
# products. `mac_for_device` refuses a device it does not know.
MAC_DEFAULT = MAC_BY_DEVICE["npu2"]

TILE_K, TILE_N = 64, 32

# The control tokens a transcription cannot start without. Refused by name if
# the checkpoint does not have them, because a Whisper that cannot emit
# <|endoftext|> has no stopping condition and a container that cannot find
# <|notimestamps|> silently produces timestamp soup.
REQUIRED_TOKENS = ("<|startoftranscript|>", "<|notimestamps|>",
                   "<|transcribe|>", "<|translate|>", "<|endoftext|>")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what}: {p} not found. A Whisper checkpoint needs "
                         f"config.json, preprocessor_config.json, "
                         f"model.safetensors, vocab.json, merges.txt and "
                         f"added_tokens.json.")
    return json.loads(p.read_text(encoding="utf-8"))


def add_gemm_b(w, name, mat, fold=None, mac=MAC_DEFAULT):
    """Stage a [K, N] GEMM operand: optional scale fold, bf16, pre-tiled.

    The fold is applied in fp32 BEFORE the bf16 rounding, never after: rounding
    first and scaling second would scale the rounding error too, and the error
    is already the dominant term in a bf16 operand.

    `mac` is the target generation's (mac_s, mac_t) -- see MAC_DEFAULT. It is a
    parameter and not a constant because a container packed for the other board
    is not refused by anything, it is merely wrong.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * np.float32(fold)
    K, N = mat.shape
    if K % TILE_K or N % TILE_N:
        raise SystemExit(
            f"{name}: [{K},{N}] does not tile into ({TILE_K},{TILE_N}). The "
            f"shipped sizes all do; an unexpected geometry is refused here "
            f"rather than padded, because a padded operand would be a container "
            f"whose layout_hash lies about its shape.")
    layout = gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1])
    bits = to_bf16_bits(mat)
    flat = tile_b(bits, TILE_K, TILE_N, mac[0], mac[1])
    return w.add(name, flat, "BF16", "gemm_b", [K, N], layout=layout)


def _fuse_qkv(w, name, st, prefix, hidden, scale, fold_scale=True,
              mac=MAC_DEFAULT):
    """Fuse q|k|v for an attention that has a bias on q and v but NOT on k.

    Stages `name` ([d, 3d] pre-tiled) and `name.bias` ([3d] fp32).

    The scale is folded into the Q BLOCK ONLY. Folding it into the whole fused
    operand would scale K and V as well, which is a different model, and folding
    it into the bias alone would leave the weight unscaled. Both halves of Q --
    weight columns and bias -- get it, because the attention scale multiplies
    the whole projection.
    """
    if prefix + "k_proj.bias" in st:
        raise SystemExit(
            f"{prefix}k_proj.bias exists. The fused bias written here has a "
            f"zero K half, which is exact only because k has no bias; a "
            f"checkpoint that has one must have that bias dropped explicitly "
            f"rather than silently ignored.")
    qw = st.array(prefix + "q_proj.weight")
    kw = st.array(prefix + "k_proj.weight")
    vw = st.array(prefix + "v_proj.weight")
    qb = st.array(prefix + "q_proj.bias")
    vb = st.array(prefix + "v_proj.bias")
    q = qw.T
    qb = qb.astype(np.float32, copy=True)
    if fold_scale:
        q = np.ascontiguousarray(q) * np.float32(scale)
        qb *= np.float32(scale)
    fused = np.concatenate([q, kw.T, vw.T], axis=1)            # [d, 3d]
    bias = np.concatenate([qb, np.zeros(hidden, np.float32), vb])
    add_gemm_b(w, name, fused, mac=mac)
    w.add(name + ".bias", bias, "F32", "bias", [3 * hidden])


def _out_proj(w, name, st, prefix, mac=MAC_DEFAULT):
    ob = st.array(prefix + "out_proj.bias")
    add_gemm_b(w, name, st.array(prefix + "out_proj.weight").T, mac=mac)
    w.add(name + ".bias", ob, "F32", "bias", [int(ob.shape[0])])


def _ln(w, name, st, prefix_w, prefix_b, hidden):
    w.add(name + ".weight", st.array(prefix_w), "F32", "layernorm", [hidden])
    w.add(name + ".bias", st.array(prefix_b), "F32", "layernorm", [hidden])


def _generation_policy(w, model_dir, config):
    """The checkpoint's own decoding policy, stored so the runtime applies it.

    `generation_config.json` is not decoration. Two of its fields decide WHICH
    token a greedy step picks:

      suppress_tokens          forbidden at every step
      begin_suppress_tokens    forbidden at the FIRST generated token only

    and the second list contains <|endoftext|>, so without it a model may emit
    an empty transcript on the very first step. A raw argmax therefore does NOT
    reproduce the reference implementation: on a 3 s tone with whisper-tiny,
    unconstrained greedy picks 522 (" (") and the reference picks 509 (" You"),
    because 522 is in the suppress list. A transcription that differs from every
    reference implementation is a wrong answer with no signal, so the lists
    travel in the container rather than being hardcoded here.

    Refused when the file is missing or has neither list: this checkpoint's
    policy is not knowable, and guessing OpenAI's would be a guess.
    """
    path = model_dir / "generation_config.json"
    if not path.exists():
        raise SystemExit(
            f"{model_dir}/generation_config.json not found. Whisper's decoding "
            f"policy (suppress_tokens, begin_suppress_tokens) lives there, and "
            f"without it a greedy step can pick a token the reference "
            f"implementation forbids -- see the long comment in this function.")
    gen = _read_json(path, "whisper generation config")
    suppress = [int(x) for x in gen.get("suppress_tokens", [])]
    begin = [int(x) for x in gen.get("begin_suppress_tokens", [])]
    if not suppress and not begin:
        raise SystemExit(
            f"{path} carries neither suppress_tokens nor begin_suppress_tokens. "
            f"This build applies the checkpoint's policy rather than inventing "
            f"one, so it will not pack a container that would silently "
            f"transcribe differently from transformers.")
    for name, ids in (("suppress_tokens", suppress),
                      ("begin_suppress_tokens", begin)):
        if ids:
            if min(ids) < 0 or max(ids) >= int(_vocab_of(config)):
                raise SystemExit(
                    f"{path}: {name} holds id {min(ids)}..{max(ids)}, outside "
                    f"the {config['vocab_size']}-entry vocabulary")
            w.add("decoder." + name, np.asarray(ids, np.int32), "I32",
                  "generation", [len(ids)])
    config["suppress_tokens"] = len(suppress)
    config["begin_suppress_tokens"] = len(begin)


def _vocab_of(config):
    return int(config["vocab_size"])


def _ffn(w, st, ck_prefix, name_prefix, hidden, inter, mac=MAC_DEFAULT):
    """fc1 -> GELU -> fc2, both pre-tiled, both with their real biases.

    `ck_prefix` is the checkpoint-side directory ("model.encoder.layers.0."),
    `name_prefix` the container-side one ("encoder.layers.0."). They are passed
    separately because they are not the same string, and quietly deriving one
    from the other is how a packer ends up reading tensors that do not exist.
    """
    add_gemm_b(w, name_prefix + "ffn_up", st.array(ck_prefix + "fc1.weight").T,
              mac=mac)
    w.add(name_prefix + "ffn_up.bias", st.array(ck_prefix + "fc1.bias"),
          "F32", "bias", [inter])
    add_gemm_b(w, name_prefix + "ffn_down", st.array(ck_prefix + "fc2.weight").T,
              mac=mac)
    w.add(name_prefix + "ffn_down.bias", st.array(ck_prefix + "fc2.bias"),
          "F32", "bias", [hidden])


def pack_whisper(model_dir, out, max_seq=None, max_target=None,
                 fold_scale=True, dry_run=False, device=None):
    model_dir = Path(model_dir)
    mac = mac_for_device(device)
    cfg = _read_json(model_dir / "config.json", "whisper config")
    if cfg.get("model_type") != "whisper":
        raise SystemExit(
            f"{model_dir}/config.json says model_type="
            f"{cfg.get('model_type')!r}, not 'whisper'")
    pre = _read_json(model_dir / "preprocessor_config.json", "whisper features")

    d = int(cfg["d_model"])
    heads = int(cfg["encoder_attention_heads"])
    if int(cfg["decoder_attention_heads"]) != heads:
        raise SystemExit(
            f"encoder and decoder disagree on head count ({heads} vs "
            f"{cfg['decoder_attention_heads']}); the container stores one")
    head_dim = d // heads
    if heads * head_dim != d:
        raise SystemExit(
            f"d_model {d} is not divisible by {heads} heads. HF uses "
            f"head_dim = d_model // n_heads with integer division, so a "
            f"non-dividing pair is a real truncated head, not a rounding "
            f"detail, and the attention scale would be wrong.")
    enc_layers = int(cfg["encoder_layers"])
    dec_layers = int(cfg["decoder_layers"])
    enc_inter = int(cfg["encoder_ffn_dim"])
    dec_inter = int(cfg["decoder_ffn_dim"])
    if int(cfg["decoder_ffn_dim"]) != enc_inter:
        raise SystemExit("encoder and decoder FFN widths differ; the container "
                         "stores one `intermediate` and would be ambiguous")
    mel = int(pre.get("feature_size", cfg.get("num_mel_bins", 80)))
    n_fft = int(pre["n_fft"])
    hop = int(pre["hop_length"])
    rate = int(pre["sampling_rate"])
    n_samples = int(pre.get("n_samples", 480000))
    nb_max_frames = int(pre.get("nb_max_frames", 3000))
    max_src = int(cfg["max_source_positions"])
    max_tgt = int(cfg["max_target_positions"])
    vocab = int(cfg["vocab_size"])
    if max_seq is None:
        max_seq = max_src
    if max_target is None:
        max_target = max_tgt
    if max_seq < max_src:
        # The encoder ALWAYS sees max_source_positions rows: 3000 mel frames
        # halved by conv2's stride. A container whose position table is shorter
        # cannot transcribe one window of audio, so slicing it below the model's
        # own window is not a smaller model -- it is a container that refuses at
        # the first request. (This is what --max-seq 256, the flag's default for
        # EMBEDDERS, does to a whisper checkpoint.)
        raise SystemExit(
            f"--max-seq {max_seq} is below this checkpoint's "
            f"max_source_positions ({max_src}), and a Whisper encoder always "
            f"sees {max_src} positions -- one 30 s window. A container sliced "
            f"to {max_seq} cannot transcribe anything; it refuses at the first "
            f"request. Re-run with --max-seq {max_src} (or omit the flag: for "
            f"whisper the default IS max_source_positions)."
        )
    if max_seq > max_src or max_target > max_tgt:
        raise SystemExit(
            f"--max-seq {max_seq} / --max-target {max_target} exceed the "
            f"checkpoint's own tables ({max_src} / {max_tgt}); the container "
            f"would claim positions the model has no weights for")
    eps = float(cfg.get("layer_norm_eps", 1e-5))
    scale = 1.0 / math.sqrt(head_dim)

    vm = VocabMerges.load(model_dir)
    missing = [t for t in REQUIRED_TOKENS if t not in vm.index]
    if missing:
        raise SystemExit(
            f"{model_dir}: the tokenizer has no {', '.join(missing)}. Whisper's "
            f"control tokens live in added_tokens.json, not vocab.json; a "
            f"vocab.json-only read produces a container that cannot start or "
            f"stop a transcription.")
    if len(vm.ids) < vocab:
        raise SystemExit(
            f"tokenizer has {len(vm.ids)} entries but the config claims "
            f"vocab_size {vocab}; the logit matrix would be narrower than the "
            f"ids the decoder can emit")

    st = SafeTensors(model_dir / "model.safetensors")
    src_sha = _sha256(model_dir / "model.safetensors")

    config = {
        "arch": "whisper_encdec_gelu",
        "kind": "stt",
        "source_repo": f"openai/{model_dir.name}",
        "source_sha256": src_sha,
        "d_model": d,
        "hidden": d,                       # what the design selector reads
        "num_heads": heads,
        "head_dim": head_dim,
        "num_layers": enc_layers,
        "num_encoder_layers": enc_layers,
        "num_decoder_layers": dec_layers,
        "intermediate": enc_inter,
        "encoder_intermediate": enc_inter,
        "decoder_intermediate": dec_inter,
        "max_seq_len": max_seq,
        "max_target_positions": max_target,
        "num_mel_bins": mel,
        "n_fft": n_fft,
        "hop_length": hop,
        "sample_rate": rate,
        "n_samples": n_samples,
        "nb_max_frames": nb_max_frames,
        "vocab_size": vocab,
        "activation": "gelu",
        "layer_norm_eps": eps,
        "attention_scale": scale,
        "qkv_scale_folded": bool(fold_scale),
        "tied_embeddings": True,
        "gated_ffn": False,
        "fusions": [
            "qkv fused into one [d,3d] operand per attention; the K half of "
            "the fused bias is zero because Whisper's k_proj has no bias",
            "cross-attention K|V fused into [d,2d]; its Q is separate because "
            "its A operand is the decoder state",
            "1/sqrt(head_dim) folded into every Q weight and Q bias",
            "GEMM operands pre-tiled bf16 block_panel (64,32)",
            "convs, LayerNorms, biases, embedding tables and the tokenizer "
            "table kept fp32 / raw",
            "the checkpoint's decoding policy (generation_config.json's "
            "suppress_tokens and begin_suppress_tokens) stored as int32, so a "
            "greedy step picks what the reference implementation picks",
        ],
        "not_implemented": [
            "beam search and sampling: greedy only",
            "language DETECTION: the decoder is primed with a language token "
            "the caller names, and an assumed language is reported rather "
            "than guessed silently",
            "timestamp-aware long-form merging: the overlap IS handled, by the "
            "same longest-common-sequence over token ids transformers uses "
            "without timestamps, so a word spoken inside the 5 s overlap is "
            "transcribed once. What is missing is merging by TIME, which needs "
            "the timestamp tokens this decoder does not emit",
            "timestamps: the decoder is primed with <|notimestamps|> and no "
            "timestamp token is decoded",
        ],
    }

    w = Writer(config, arch=ARCH_WHISPER_ENC_DEC_GELU)

    # The decoding policy, before the weights: if this file cannot be read the
    # container would answer differently from transformers, and that is a
    # refusal rather than a default.
    _generation_policy(w, model_dir, config)

    # -- frontend: conv1 and conv2 (HF's conv1_pos), both fp32 --------------
    conv1 = st.array("model.encoder.conv1.weight")
    conv2 = st.array("model.encoder.conv2.weight")
    if conv1.shape != (d, mel, 3) or conv2.shape != (d, d, 3):
        raise SystemExit(
            f"conv shapes are {conv1.shape} and {conv2.shape}; expected "
            f"{(d, mel, 3)} and {(d, d, 3)} for a {mel}-mel-bin model")
    w.add("frontend.conv1.weight", conv1, "F32", "conv", [d, mel, 3])
    w.add("frontend.conv1.bias", st.array("model.encoder.conv1.bias"),
          "F32", "conv", [d])
    w.add("frontend.conv2.weight", conv2, "F32", "conv", [d, d, 3])
    w.add("frontend.conv2.bias", st.array("model.encoder.conv2.bias"),
          "F32", "conv", [d])

    # -- positional tables and the tied token embedding ---------------------
    w.add("encoder.embed_positions",
          st.array("model.encoder.embed_positions.weight")[:max_seq],
          "F32", "embedding", [max_seq, d])
    w.add("decoder.embed_positions",
          st.array("model.decoder.embed_positions.weight")[:max_target],
          "F32", "embedding", [max_target, d])
    w.add("decoder.embed_tokens", st.array("model.decoder.embed_tokens.weight"),
          "F32", "embedding", [vocab, d])
    _ln(w, "encoder.layer_norm", st,
        "model.encoder.layer_norm.weight", "model.encoder.layer_norm.bias", d)
    _ln(w, "decoder.layer_norm", st,
        "model.decoder.layer_norm.weight", "model.decoder.layer_norm.bias", d)

    # -- encoder stack -------------------------------------------------------
    n_tiled = 0
    for i in range(enc_layers):
        p = f"model.encoder.layers.{i}."
        e = f"encoder.layers.{i}."
        _fuse_qkv(w, e + "qkv", st, p + "self_attn.", d, scale, fold_scale,
                  mac=mac)
        _out_proj(w, e + "attn_out", st, p + "self_attn.", mac=mac)
        _ln(w, e + "ln1", st, p + "self_attn_layer_norm.weight",
            p + "self_attn_layer_norm.bias", d)
        _ffn(w, st, p, e, d, enc_inter, mac=mac)
        _ln(w, e + "ln2", st, p + "final_layer_norm.weight",
            p + "final_layer_norm.bias", d)
        n_tiled += 4

    # -- decoder stack -------------------------------------------------------
    for i in range(dec_layers):
        p = f"model.decoder.layers.{i}."
        dcl = f"decoder.layers.{i}."
        _fuse_qkv(w, dcl + "self_qkv", st, p + "self_attn.", d, scale, fold_scale,
                  mac=mac)
        _out_proj(w, dcl + "self_attn_out", st, p + "self_attn.", mac=mac)
        _ln(w, dcl + "ln1", st, p + "self_attn_layer_norm.weight",
            p + "self_attn_layer_norm.bias", d)
        # cross-attention: Q from the decoder state, K|V from the encoder
        # output, so two operands and the same zero-K-bias trick.
        cq = st.array(p + "encoder_attn.q_proj.weight").T
        cqb = st.array(p + "encoder_attn.q_proj.bias").astype(np.float32, copy=True)
        if fold_scale:
            cq = np.ascontiguousarray(cq) * np.float32(scale)
            cqb *= np.float32(scale)
        add_gemm_b(w, dcl + "cross_q", cq, mac=mac)
        w.add(dcl + "cross_q.bias", cqb, "F32", "bias", [d])
        kv = np.concatenate([st.array(p + "encoder_attn.k_proj.weight").T,
                             st.array(p + "encoder_attn.v_proj.weight").T], axis=1)
        add_gemm_b(w, dcl + "cross_kv", kv, mac=mac)
        w.add(dcl + "cross_kv.bias",
              np.concatenate([np.zeros(d, np.float32),
                              st.array(p + "encoder_attn.v_proj.bias")]),
              "F32", "bias", [2 * d])
        _out_proj(w, dcl + "cross_attn_out", st, p + "encoder_attn.", mac=mac)
        _ln(w, dcl + "ln2", st, p + "encoder_attn_layer_norm.weight",
            p + "encoder_attn_layer_norm.bias", d)
        _ffn(w, st, p, dcl, d, dec_inter, mac=mac)
        _ln(w, dcl + "ln3", st, p + "final_layer_norm.weight",
            p + "final_layer_norm.bias", d)
        n_tiled += 6

    # -- tokenizer table -----------------------------------------------------
    blob = build_table(vm)
    w.add("tokenizer.whisper_table", np.frombuffer(blob, dtype=np.uint8),
          "U8", "tokenizer", [len(blob)])

    st.close()

    if dry_run:
        print(f"  would write {len(w.entries)} tensors "
              f"({n_tiled} pre-tiled GEMM operands) to {out}")
        return 0

    info = w.write(out)
    total = Path(out).stat().st_size
    print(f"\n  arch       : 4 whisper_encdec_gelu  d_model {d}, "
          f"{enc_layers}+{dec_layers} layers, {heads} heads x {head_dim}, "
          f"{mel} mel bins, vocab {vocab}")
    print(f"  source     : {src_sha[:16]}...  (openai/{model_dir.name})")
    print(f"  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  tokenizer  : {len(blob) / 1e6:.2f} MB table, "
          f"{len(vm.ids)} ids, {len(vm.merges)} merges")
    print(f"  decoding   : {config['suppress_tokens']} suppressed ids, "
          f"{config['begin_suppress_tokens']} suppressed at the first step "
          f"(from generation_config.json)")
    print(f"  data       : {info['data_length'] / 1e6:.2f} MB")
    print(f"  file       : {out}  ({total / 1e6:.2f} MB)")
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1]))[:16]}..."
          f"  (mac s={mac[0]} t={mac[1]}, {device or MAC_DEFAULT_DEVICE})")
    return 0
