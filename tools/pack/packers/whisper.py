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
# Env: numpy only (plus the repo's own npue.py and onnx_weights.py).

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

from gemm_i8 import add_gemm_b_int8                              # noqa: E402
from gemm_i4 import add_gemm_b_int4                              # noqa: E402
from npue import (ARCH_WHISPER_ENC_DEC_GELU, MAC_BY_DEVICE,  # noqa: E402
                  MAC_DEFAULT_DEVICE, Writer, gemm_b_layout, layout_hash,
                  mac_for_device, tile_b, to_bf16_bits)
from onnx_weights import (Combined, OnnxWeights, model_digest,    # noqa: E402
                          WHISPER_DECODER_ONNX, WHISPER_ENCODER_ONNX)
from whisper_bpe import VocabMerges, build_table                      # noqa: E402

# Only s and t of the MAC geometry affect the B operand's byte order, and both
# production datapaths (plain bf16 and bfp16-emulated) give the same pair on a
# given board -- so the layout does not depend on the datapath. It DOES depend
# on the board: npu1's MMAC sub-tile is (s=8, t=4) and npu2's is (8, 8), so
# `mac` is resolved from the target device and threaded into every operand
# rather than fixed here. The wrong pair is unreadable only by the hardware:
# same byte count, same shapes, same layout_hash on both sides, plausible
# products. `mac_for_device` refuses a device it does not know.
MAC_DEFAULT = MAC_BY_DEVICE["npu2"]["bf16"]

TILE_K, TILE_N = 64, 32

# The B operand's dtype, and with it the layout hash the runtime compares
# against the design's. Switched by --int8 and set once, here, so that the
# writer and the printed hash cannot disagree: they read the same constant and
# a disagreement would be a container whose layout_hash describes the other
# datatype, which nothing except the hardware would notice.
I8_DTYPE = "BF16"

# The control tokens a transcription cannot start without. Refused by name if
# the checkpoint does not have them, because a Whisper that cannot emit
# <|endoftext|> has no stopping condition and a container that cannot find
# <|notimestamps|> silently produces timestamp soup.
REQUIRED_TOKENS = ("<|startoftranscript|>", "<|notimestamps|>",
                   "<|transcribe|>", "<|translate|>", "<|endoftext|>")


def _read_json(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what}: {p} not found. A Whisper checkpoint needs "
                         f"config.json, preprocessor_config.json, "
                         f"{WHISPER_ENCODER_ONNX}, {WHISPER_DECODER_ONNX}, "
                         f"vocab.json, merges.txt and "
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
    layout = gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1],
                           dtype=I8_DTYPE)
    bits = to_bf16_bits(mat)
    flat = tile_b(bits, TILE_K, TILE_N, mac[0], mac[1])
    return w.add(name, flat, "BF16", "gemm_b", [K, N], layout=layout)


class _Int8:
    """int8 emission for one container, driven by calibrated activation maxima.

    The scheme is `gemm_i8.add_gemm_b_int8` (per-output-channel symmetric scale,
    int8 x int8 -> int32) and the smoothing factor is SmoothQuant's, computed
    here from the calibration's per-input-channel maximum and this operand's own
    rows:

        s_j = max_i |X[i,j]|^alpha / max_n |W[j,n]|^(1-alpha)

    so the runtime computes (X/s) @ (sW), which is X @ W in exact arithmetic.
    `add_gemm_b_int8` applies s to the weight BEFORE quantising and ships it as
    `.asmooth`, and the runtime divides the activation by it while quantising --
    the same contract `calibrate_smoothing` + `quantise_a_int8` have for the
    BERT-family, and the reason whisper needs its own calibration (audio
    activations) but not its own scheme.

    The per-tensor relative error `add_gemm_b_int8` returns is collected and
    printed, because a scale set that is wrong for the model is visible here and
    nowhere else.
    """

    def __init__(self, amax: dict, alpha: float, mac, layout_dtype: str,
                 int4_group: int | None = None):
        self.amax = amax
        self.alpha = alpha
        self.mac = mac
        self.layout_dtype = layout_dtype
        # None for an int8 pack; a row count for an int4 one -- see gemm_i4.py.
        self.int4_group = int4_group
        self.errors: list[tuple[str, float]] = []

    def smoothing(self, key: str, mat: np.ndarray) -> np.ndarray:
        a = self.amax[key]
        if a.shape[0] != mat.shape[0]:
            raise SystemExit(
                f"{key}: the calibration measured {a.shape[0]} input channels "
                f"and the operand has {mat.shape[0]}. The hook and the operand "
                f"must be the same tensor; a mismatch here is a per-channel "
                f"scale broadcast onto the wrong axis, which is silent.")
        w = np.abs(np.asarray(mat, dtype=np.float32)).max(axis=1)
        s = (np.maximum(a, 1e-8) ** self.alpha) / (
            np.maximum(w, 1e-8) ** (1.0 - self.alpha))
        return np.where(np.isfinite(s) & (s > 0), s, 1.0).astype(np.float32)

    def emit(self, w, name, mat, key, fold=None):
        if key is None:
            add_gemm_b(w, name, mat, fold=fold, mac=self.mac)
            return
        s = self.smoothing(key, mat)
        # The SAME smoothing either way: int4 is the int8 datapath with a
        # narrower weight, so the activation half of the SmoothQuant identity
        # and the audio calibration that feeds it are shared with int8.
        if self.int4_group is None:
            err = add_gemm_b_int8(w, name, mat, TILE_K, TILE_N, fold=fold,
                                  asmooth=s, mac=self.mac)
        else:
            err = add_gemm_b_int4(w, name, mat, TILE_K, TILE_N, fold=fold,
                                  asmooth=s, group=self.int4_group,
                                  mac=self.mac)
        self.errors.append((name, err))

    def report(self) -> str:
        if not self.errors:
            return "  (no int8 operands)"
        es = [e for _, e in self.errors]
        worst = max(self.errors, key=lambda kv: kv[1])
        return (f"  {'int4' if self.int4_group is not None else 'int8'}: "
                f"{len(self.errors)} operands, quantisation error "
                f"median {float(np.median(es)):.2e} max {worst[1]:.2e} "
                f"({worst[0]})")


class _Host:
    """The `--dtype f32` emitter: row-major F32 [K, N], no tiling, no scales.

    The SAME two methods as _Int8 -- `emit(w, name, mat, key, fold)` and
    `report()` -- because every call site below branches on `if ctx` and not on
    the dtype family: a third branch at those six sites would be six more
    places for the packer's three schemes to disagree about which one wrote
    which operand.

    `key` is the calibration lookup and is ignored here: an f32 operand is the
    checkpoint's own precision, so there is nothing to calibrate and no error
    to report. `fold` is accepted for the same reason `_Int8` accepts it -- a
    call site passes it whether or not it applies -- and applied in fp32
    before the write, exactly as the bf16 path does.

    The tensor's layout kind is `gemm_b_host`, the string pack_gemma's host
    branch and tools/pack/pack_npue.py's add_gemm_b_host() both use, so a
    reader cannot tell which of the four packers wrote it -- which is the
    point: the format is one thing, not five.
    """

    def emit(self, w, name, mat, key=None, fold=None):
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        if fold is not None:
            mat = mat * np.float32(fold)
        K, N = mat.shape
        w.add(name, mat.reshape(-1), "F32", "gemm_b_host", [K, N])

    def report(self) -> str:
        return ("  f32: row-major operands, no quantisation, GEMM on the host "
                "(the array multiplies bf16 and int8 only)")


def _operand_note(host_only: bool, int8: bool) -> str:
    """The one `fusions` entry that names the operand scheme, for this packer.

    Three schemes and one sentence, chosen in one place so the container's own
    prose cannot describe the array's panel while the bytes are row-major --
    the disagreement tasks/0078 fixed for `a_dtype`, in the strings.
    """
    if host_only:
        return ('GEMM operands row-major F32 (gemm_layout "host"), GEMM on '
                "the CPU -- the array multiplies bf16 and int8 only, so "
                "--npu-ops gemm is refused on this container by name")
    if int8:
        return ("GEMM operands pre-tiled int8 block_panel (64,32) with a "
                "per-output-channel scale and a SmoothQuant factor")
    return "GEMM operands pre-tiled bf16 block_panel (64,32)"


def _fuse_qkv(w, name, st, prefix, hidden, scale, fold_scale=True,
              mac=MAC_DEFAULT, ctx=None, key=None):
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
    if ctx:
        ctx.emit(w, name, fused, key, fold=None)
    else:
        add_gemm_b(w, name, fused, mac=mac)
    w.add(name + ".bias", bias, "F32", "bias", [3 * hidden])


def _out_proj(w, name, st, prefix, mac=MAC_DEFAULT, ctx=None, key=None):
    ob = st.array(prefix + "out_proj.bias")
    if ctx:
        ctx.emit(w, name, st.array(prefix + "out_proj.weight").T, key)
    else:
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


def _ffn(w, st, ck_prefix, name_prefix, hidden, inter, mac=MAC_DEFAULT,
         ctx=None):
    """fc1 -> GELU -> fc2, both pre-tiled, both with their real biases.

    `ck_prefix` is the checkpoint-side directory ("model.encoder.layers.0."),
    `name_prefix` the container-side one ("encoder.layers.0."). They are passed
    separately because they are not the same string, and quietly deriving one
    from the other is how a packer ends up reading tensors that do not exist.
    """
    if ctx:
        ctx.emit(w, name_prefix + "ffn_up", st.array(ck_prefix + "fc1.weight").T,
                 name_prefix + "ffn_up")
    else:
        add_gemm_b(w, name_prefix + "ffn_up",
                   st.array(ck_prefix + "fc1.weight").T, mac=mac)
    w.add(name_prefix + "ffn_up.bias", st.array(ck_prefix + "fc1.bias"),
          "F32", "bias", [inter])
    if ctx:
        ctx.emit(w, name_prefix + "ffn_down",
                 st.array(ck_prefix + "fc2.weight").T,
                 name_prefix + "ffn_down")
    else:
        add_gemm_b(w, name_prefix + "ffn_down",
                   st.array(ck_prefix + "fc2.weight").T, mac=mac)
    w.add(name_prefix + "ffn_down.bias", st.array(ck_prefix + "fc2.bias"),
          "F32", "bias", [hidden])


# -- embedding the design set -----------------------------------------------
#
# The same three lines in five packers, and a shared body rather than a shared
# NAME: these modules are imported by pack_npue.py directly and are also runnable
# as scripts, so they each need their own import line. What they share is the
# rules, and those live in design_embed.py -- which one packer instead of five
# would be five opportunities for four of them to disagree with the fifth about
# which directory a datapath's set is in.
def _embed(w, model_name, datapath, npue_args, where):
    if npue_args is None or getattr(npue_args, "no_embed_artifacts", False):
        return None
    from design_embed import (GEMM_SET, embed_design_sets, find_design_dir)
    import os
    explicit = getattr(npue_args, "embed_artifacts", None)
    # FOUR levels up, not three: this file is <root>/tools/pack/packers/<x>.py,
    # so dirname is .../packers, and three of them reaches <root>/tools -- which
    # produced the path "tools/runtime/artifacts" and a container that reported
    # "none embedded" on a machine with every set built. A wrong root is silent
    # here: the directory simply is not there, and the honest-looking message is
    # about the absence rather than about the arithmetic.
    here = os.path.abspath(__file__)
    for _ in range(4):
        here = os.path.dirname(here)
    root = os.path.join(here, "runtime", "artifacts")
    d = find_design_dir(model_name, datapath, root, explicit)
    if d is None:
        if explicit:
            raise SystemExit(
                f"--embed-artifacts {explicit} holds no {GEMM_SET}/final.xclbin. "
                f"Point it at the directory CONTAINING the sets, not at one of "
                f"them.")
        print(f"  design     none embedded: no set under {root} for "
              f"{model_name} ({datapath}). The container still runs; it needs "
              f"the design set beside it.")
        return None
    r = embed_design_sets(w, d,
                          device=getattr(npue_args, "device", "npu1") or "npu1",
                          datapath=datapath)
    sets = ", ".join(f"{k} {v['bytes'] / 1024:.0f} KB"
                     for k, v in sorted(r["sets"].items()))
    print(f"  design     {r['total_bytes'] / 1024:.0f} KB embedded from {d} "
          f"({r['device']}, {r['datapath']}): {sets}")
    return r


def pack_whisper(model_dir, out, max_seq=None, max_target=None,
                 fold_scale=True, dry_run=False, device=None,
                 int8=False, int4_group=None, int8_alpha=0.5, int8_clips=8,
                 int8_corpus=None,
                npue_args=None, host_only=False):
    # The B operand's dtype for this whole container, and therefore the layout
    # hash: switched once, here, so the writer, `add_gemm_b` and the printed
    # report cannot disagree about which one they wrote. A disagreement would be
    # a container whose layout_hash describes the other datatype, which nothing
    # except the hardware would notice.
    global I8_DTYPE
    if int4_group is not None and not int8:
        # int4 is the int8 datapath with 4-bit weights (gemm_i4.py), so with
        # int8=False there is no quantisation to attach a group size to. See
        # pack_vit's identical guard for why this is refused rather than
        # dropped: an accepted-and-ignored flag exits 0 and tells the caller
        # nothing.
        raise SystemExit(
            f"--int4-group {int4_group} was passed to a pack that does not "
            f"quantise (int8=False); int4 is the int8 datapath with 4-bit "
            f"weights, so it has nothing to attach to. Refusing rather than "
            f"emitting a bf16 container that ignores it.")
    if host_only and int8:
        # Two answers to one question: an f32 container is the checkpoint's own
        # precision on the CPU, an int8 one is a pre-tiled panel for the array.
        # pack_gemma's guard, restated here rather than shared, because these
        # modules are runnable as scripts as well as imported.
        raise SystemExit(
            "--dtype f32 and the int8 datapath are two answers to the same "
            "question: f32 is row-major F32 on the host, i8 is a pre-tiled "
            "panel for the array. Refusing rather than dropping one. int8="
            "False for the f32 control, or drop --dtype f32.")
    model_dir = Path(model_dir)
    # int8's sub-tile is NOT bf16's on npu1 -- see npue.MAC_BY_DEVICE. The
    # dtype is required by mac_for_device for that reason.
    mac = mac_for_device(device, "I8" if int8 else "BF16")
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

    # Two ONNX files behind one reader: this function asks for
    # `model.encoder.conv1.weight` about a hundred lines after
    # `model.decoder.embed_tokens.weight` and cannot sensibly be split, so
    # Combined() presents them as one namespace and refuses a name that
    # resolves in both. The encoder export dropped the `model.encoder.` root
    # its tensors are named for -- prefix= puts it back -- while the decoder
    # export kept `model.decoder.` and needs nothing; OnnxWeights documents
    # both directions.
    st = Combined([
        OnnxWeights(model_dir / WHISPER_ENCODER_ONNX,
                    prefix="model.encoder."),
        OnnxWeights(model_dir / WHISPER_DECODER_ONNX),
    ])
    # The whole model, not just the two graph files: an export large enough
    # to trip protobuf's 2 GB limit keeps its weights in an .onnx_data side
    # file, and hashing only the graphs would pin the tensor names while
    # leaving every weight unguarded.
    src_sha = model_digest(model_dir / WHISPER_ENCODER_ONNX,
                           model_dir / WHISPER_DECODER_ONNX)

    # The int8 calibration, before any operand is written: it needs torch and
    # transformers, which are build-time only, and it is the reason this branch
    # exists at all. `I8_DTYPE` is switched here rather than inside add_gemm_b so
    # that the layout hash, the writer and the printed report all read one value.
    ictx = None
    if host_only:
        # FIRST, and exclusive: there is no calibration to run for an operand
        # that is the checkpoint's own precision, so I8_DTYPE stays BF16 here
        # and the config below overrides a_dtype to "f32" from `host_only`.
        ictx = _Host()
    elif int8:
        I8_DTYPE = "I8"
        import whisper_int8
        clips, corpus_label = whisper_int8.load_or_make_corpus(
            int8_corpus, int8_clips)
        amax = whisper_int8.calibrate(
            model_dir, enc_layers, dec_layers, clips, alpha=int8_alpha,
            verbose=False)
        ictx = _Int8(amax, int8_alpha, mac, I8_DTYPE, int4_group)
        print(f"  {'int4' if int4_group is not None else 'int8'}: "
              f"{corpus_label}, alpha={int8_alpha}, "
              f"{len(amax)} GEMM sites, {len(clips) * 4 * 3000} mel frames "
              f"per encoder pass")

    config = {
        "arch": "whisper_encdec_gelu",
        "kind": "stt",
        # The layout the operands are stored in, stated by the container: "host"
        # is plain row-major F32 for runtime/include/common/host_b.hpp (a
        # self-describing file, so the runtime needs no hint from the packer),
        # "pretiled_bf16" is the array's block_panel. The MMAC multiplies bf16
        # and int8 only, so `--npu-ops gemm` on a host container is refused by
        # name at run time rather than ignored.
        "gemm_layout": "host" if host_only else "pretiled_bf16",
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
        # The operand datatype the array is to be built for. "bf16" is the
        # default and was the only value this packer wrote; "i8" means every
        # GEMM operand is int8 with a per-output-channel scale and a
        # SmoothQuant factor, and the design set must be exported with --int8 or
        # the runtime refuses the container by name when the two disagree.
        "a_dtype": "f32" if host_only else I8_DTYPE,
        **({"int4_group": int4_group} if int4_group is not None else {}),
        "fusions": [
            "qkv fused into one [d,3d] operand per attention; the K half of "
            "the fused bias is zero because Whisper's k_proj has no bias",
            "cross-attention K|V fused into [d,2d]; its Q is separate because "
            "its A operand is the decoder state",
            "1/sqrt(head_dim) folded into every Q weight and Q bias",
            _operand_note(host_only, int8),
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
                  mac=mac, ctx=ictx, key=e + "qkv")
        _out_proj(w, e + "attn_out", st, p + "self_attn.", mac=mac, ctx=ictx,
                   key=e + "attn_out")
        _ln(w, e + "ln1", st, p + "self_attn_layer_norm.weight",
            p + "self_attn_layer_norm.bias", d)
        _ffn(w, st, p, e, d, enc_inter, mac=mac, ctx=ictx)
        _ln(w, e + "ln2", st, p + "final_layer_norm.weight",
            p + "final_layer_norm.bias", d)
        n_tiled += 4

    # -- decoder stack -------------------------------------------------------
    for i in range(dec_layers):
        p = f"model.decoder.layers.{i}."
        dcl = f"decoder.layers.{i}."
        _fuse_qkv(w, dcl + "self_qkv", st, p + "self_attn.", d, scale,
                  fold_scale, mac=mac, ctx=ictx, key=dcl + "self_qkv")
        _out_proj(w, dcl + "self_attn_out", st, p + "self_attn.", mac=mac,
                  ctx=ictx, key=dcl + "self_attn_out")
        _ln(w, dcl + "ln1", st, p + "self_attn_layer_norm.weight",
            p + "self_attn_layer_norm.bias", d)
        # cross-attention: Q from the decoder state, K|V from the encoder
        # output, so two operands and the same zero-K-bias trick.
        cq = st.array(p + "encoder_attn.q_proj.weight").T
        cqb = st.array(p + "encoder_attn.q_proj.bias").astype(np.float32, copy=True)
        if fold_scale:
            cq = np.ascontiguousarray(cq) * np.float32(scale)
            cqb *= np.float32(scale)
        if ictx:
            ictx.emit(w, dcl + "cross_q", cq, dcl + "cross_q")
        else:
            add_gemm_b(w, dcl + "cross_q", cq, mac=mac)
        w.add(dcl + "cross_q.bias", cqb, "F32", "bias", [d])
        kv = np.concatenate([st.array(p + "encoder_attn.k_proj.weight").T,
                             st.array(p + "encoder_attn.v_proj.weight").T], axis=1)
        if ictx:
            ictx.emit(w, dcl + "cross_kv", kv, "cross_kv")
        else:
            add_gemm_b(w, dcl + "cross_kv", kv, mac=mac)
        w.add(dcl + "cross_kv.bias",
              np.concatenate([np.zeros(d, np.float32),
                              st.array(p + "encoder_attn.v_proj.bias")]),
              "F32", "bias", [2 * d])
        _out_proj(w, dcl + "cross_attn_out", st, p + "encoder_attn.", mac=mac,
                  ctx=ictx, key=dcl + "cross_attn_out")
        _ln(w, dcl + "ln2", st, p + "encoder_attn_layer_norm.weight",
            p + "encoder_attn_layer_norm.bias", d)
        _ffn(w, st, p, dcl, d, dec_inter, mac=mac, ctx=ictx)
        _ln(w, dcl + "ln3", st, p + "final_layer_norm.weight",
            p + "final_layer_norm.bias", d)
        n_tiled += 6

    # -- tokenizer table -----------------------------------------------------
    blob = build_table(vm)
    w.add("tokenizer.whisper_table", np.frombuffer(blob, dtype=np.uint8),
          "U8", "tokenizer", [len(blob)])

    st.close()

    if dry_run:
        print(f"  would write {len(w.entries)} tensors"
              + ("" if host_only else f" ({n_tiled} pre-tiled GEMM operands)")
              + f" to {out}")
        return 0

    # THE DATAPATH IS THIS PACK'S, not always bf16. It was hardcoded, so an
    # int8 container embedded the bf16 set beside it -- and the runtime prefers
    # the embedded one (stt_mode reads a_dtype out of it to decide int8_design),
    # which means the file came with the WRONG design glued in: bf16 design,
    # i8 operands, refused at load by layout hash and by a_elem_bytes. The
    # int8 and int4 designs exist under <model>-i8 (export_gemm_rtp --int8) and
    # find_design_dir looks for exactly that spelling when datapath is "i8".
    _embed(w, model_dir.name, "i8" if (int8 or int4_group) else "bf16",
           npue_args, "whisper")
    info = w.write(out)
    total = Path(out).stat().st_size
    print(f"\n  arch       : 4 whisper_encdec_gelu  d_model {d}, "
          f"{enc_layers}+{dec_layers} layers, {heads} heads x {head_dim}, "
          f"{mel} mel bins, vocab {vocab}")
    print(f"  source     : {src_sha[:16]}...  (openai/{model_dir.name})")
    print(f"  tensors    : {len(w.entries)}"
          + ("" if host_only else f"  ({n_tiled} pre-tiled GEMM operands)"))
    print(f"  tokenizer  : {len(blob) / 1e6:.2f} MB table, "
          f"{len(vm.ids)} ids, {len(vm.merges)} merges")
    print(f"  decoding   : {config['suppress_tokens']} suppressed ids, "
          f"{config['begin_suppress_tokens']} suppressed at the first step "
          f"(from generation_config.json)")
    print(f"  data       : {info['data_length'] / 1e6:.2f} MB")
    print(f"  file       : {out}  ({total / 1e6:.2f} MB)")
    if ictx:
        print(ictx.report())
    # A host pack has no layout to hash: the tiles it would hash were never
    # written. Printing one anyway is the fail-open shape this project has
    # already paid for twice (tasks/0042, tasks/0078).
    if host_only:
        print("  layout     : none -- row-major F32, GEMM on the host")
    else:
        print(f"  layout_hash: "
              f"{layout_hash(gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1], dtype=I8_DTYPE))[:16]}..."
              f"  (mac s={mac[0]} t={mac[1]}, {device or MAC_DEFAULT_DEVICE}"
              f"{', i8 operands' if int8 else ''})")
    return 0
