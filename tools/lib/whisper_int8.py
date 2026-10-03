# NpuEmbeddings -- SmoothQuant calibration for a Whisper container's activations.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS EXISTS FOR
# --------------------
# int8 operands need one number per input channel per GEMM: the factor that
# moves range out of the activation and into the weight column that can absorb
# it. `calibrate_smoothing` in pack_npue.py produces those for the BERT-family
# from a TEXT corpus, and its own refusal for a whisper container says why this
# file has to exist: "a whisper container's activations are log-mel frames and a
# decoder state". So the same estimator, over audio.
#
#   s_j = max_i |X[i,j]|^alpha / max_n |W[j,n]|^(1-alpha)
#
# and the array computes (X/s) @ (sW), which is X @ W in exact arithmetic. The
# ratio matters because the two operands have different dynamic ranges and the
# quantiser is per-row on one side and per-column on the other: too much range
# left in X makes every other channel of the row share one outlier's step size.
#
# THE STATISTICS ARE TAKEN FROM THE GEMM'S OWN INPUT, NOT FROM ITS OUTPUT
# -----------------------------------------------------------------------
# A forward PRE-hook on the nn.Linear of each site sees exactly the tensor the
# array will multiply, so there is nothing to reason about: no module-order
# assumption, no "the activation is the attention output" reconstruction. The
# encoder's cross-attention K|V is the interesting one -- its input is the
# encoder output, the same tensor for every decoder layer, and the hook on layer
# 0's k_proj sees it once.
#
# THE CORPUS
# -----------
# Default: a synthetic one, built here, so a pack has no external dependency and
# two packs of the same checkpoint produce the same scales. It is NOT speech: it
# is a harmonic stack with a syllabic envelope, plus noise bursts and silence,
# which is closer to speech's spectrum and level distribution than a single tone
# and is enough to expose the per-channel outliers that smoothing exists for.
# A deployment should pass its own audio: --int8-corpus, one WAV per line of
# text, and then the statistics describe the audio the model will actually see.
# The printed line says which corpus was used, because a scale set is only as
# good as the corpus it was measured on.
#
# Env: numpy, torch, transformers -- BUILD TIME ONLY. The shipped runtime stays
# C++; this runs when a container is packed, never when it is served.

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from onnx_weights import (WHISPER_DECODER_ONNX,                        # noqa: E402
                          WHISPER_ENCODER_ONNX)

REPO = Path(__file__).resolve().parents[2]

# The GEMM sites the packer asks about, and the torch module whose INPUT is that
# operand. Keys are CONTAINER names, so the packer looks them up by the string it
# already writes.
#
#   encoder.layers.i.qkv       <- self_attn.q_proj      (ln1's output)
#   encoder.layers.i.attn_out  <- self_attn.out_proj    (the attention output)
#   encoder.layers.i.ffn_up    <- fc1                   (ln2's output)
#   encoder.layers.i.ffn_down  <- fc2                   (the GELU output)
#   decoder.layers.i.self_qkv  <- self_attn.q_proj      (ln1's output)
#   decoder.layers.i.self_attn_out   <- self_attn.out_proj
#   decoder.layers.i.cross_q        <- encoder_attn.q_proj
#   decoder.layers.i.cross_attn_out <- encoder_attn.out_proj
#   decoder.layers.i.ffn_up        <- fc1
#   decoder.layers.i.ffn_down      <- fc2
#   cross_kv                       <- decoder layer 0's encoder_attn.k_proj
#
# Only q_proj is hooked, not k_proj/v_proj, on the self-attention sites: they
# share one A operand (the fused qkv), and hooking the fused site's first member
# is what makes the statistics describe the operand the array multiplies. The
# decoder's cross-attention is the exception and gets its own entries, because
# its Q and its K|V have different A operands by construction.
OPS: dict[str, str] = {
    "encoder.layers.{i}.qkv": "model.encoder.layers.{i}.self_attn.q_proj",
    "encoder.layers.{i}.attn_out":
        "model.encoder.layers.{i}.self_attn.out_proj",
    "encoder.layers.{i}.ffn_up": "model.encoder.layers.{i}.fc1",
    "encoder.layers.{i}.ffn_down": "model.encoder.layers.{i}.fc2",
    "decoder.layers.{i}.self_qkv": "model.decoder.layers.{i}.self_attn.q_proj",
    "decoder.layers.{i}.self_attn_out":
        "model.decoder.layers.{i}.self_attn.out_proj",
    "decoder.layers.{i}.cross_q": "model.decoder.layers.{i}.encoder_attn.q_proj",
    "decoder.layers.{i}.cross_attn_out":
        "model.decoder.layers.{i}.encoder_attn.out_proj",
    "decoder.layers.{i}.ffn_up": "model.decoder.layers.{i}.fc1",
    "decoder.layers.{i}.ffn_down": "model.decoder.layers.{i}.fc2",
}


def site_keys(enc_layers: int, dec_layers: int) -> list[tuple[str, str]]:
    """(container name, hook path) for every site, in a stable order."""
    out: list[tuple[str, str]] = []
    for i in range(enc_layers):
        for tmpl, path in OPS.items():
            if tmpl.startswith("encoder."):
                out.append((tmpl.format(i=i), path.format(i=i)))
    for i in range(dec_layers):
        for tmpl, path in OPS.items():
            if tmpl.startswith("decoder."):
                out.append((tmpl.format(i=i), path.format(i=i)))
    # The K|V side of the cross-attention reads the ENCODER output, which is the
    # same tensor for every layer, so one hook on layer 0 answers for all of them.
    if dec_layers:
        out.append(("cross_kv",
                   "model.decoder.layers.0.encoder_attn.k_proj"))
    return out


def synth_corpus(n_clips: int = 8, rate: int = 16000, seed: int = 4) -> list[np.ndarray]:
    """Deterministic, speech-shaped audio: harmonic stacks, bursts, silence.

    A single tone would put all of a clip's energy in two or three FFT bins and
    the per-channel maxima would be whatever that tone's phase happened to be --
    which is a calibration of the corpus's narrowest case, not of the model. This
    moves a fundamental, two formants and an envelope, and varies the level, so
    the maxima are taken over a spread of spectra.
    """
    rng = np.random.default_rng(seed)
    clips: list[np.ndarray] = []
    for c in range(n_clips):
        secs = float(rng.choice([1.5, 3.0, 5.0, 11.0]))
        t = np.arange(int(secs * rate), dtype=np.float64) / rate
        f0 = float(rng.uniform(90, 260))            # a speaker's pitch
        f1 = f0 * float(rng.uniform(2.0, 3.0))      # first formant-ish
        f2 = f0 * float(rng.uniform(4.0, 6.0))
        # a syllabic envelope: speech is not a stationary sinusoid
        env = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * math.pi * 3.7 * t + c)) ** 2
        sig = (0.6 * np.sin(2 * math.pi * f0 * t)
               + 0.3 * np.sin(2 * math.pi * f1 * t)
               + 0.15 * np.sin(2 * math.pi * f2 * t))
        sig = env * sig
        # a burst of noise every couple of seconds: fricatives and room tone
        burst = (rng.standard_normal(sig.size) * 0.08
                 * (np.sin(2 * math.pi * 0.4 * t) > 0.6))
        sig = sig + burst
        if c % 4 == 3:                                # a clip of near silence
            sig *= 0.02
        peak = float(np.abs(sig).max()) or 1.0
        clips.append((sig / peak * 0.7).astype(np.float32))
    return clips


def read_corpus(path: str | Path, rate: int = 16000) -> list[np.ndarray]:
    """Every *.wav under `path`, read as mono float32 at `rate`."""
    import wave
    p = Path(path)
    files = sorted(p.rglob("*.wav")) if p.is_dir() else sorted(p.glob("*.wav"))
    if not files:
        raise SystemExit(f"{path}: no *.wav under it")
    out: list[np.ndarray] = []
    for f in files:
        with wave.open(str(f), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != rate:
                raise SystemExit(
                    f"{f}: {w.getnchannels()} ch, {w.getsampwidth() * 8} bit, "
                    f"{w.getframerate()} Hz. The calibration corpus is read "
                    f"here, not by the runtime's WAV reader, so it is strict: "
                    f"mono 16-bit PCM at {rate} Hz.")
            raw = w.readframes(w.getnframes())
        out.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
    return out


def calibrate(model_dir: str | Path, enc_layers: int, dec_layers: int,
              clips: list[np.ndarray], alpha: float = 0.5,
              weights: dict[str, np.ndarray] | None = None,
              verbose: bool = True) -> dict[str, np.ndarray]:
    """Per-(layer, op) smoothing factors, keyed by CONTAINER operand name.

    `weights` maps the same container names to the [K, N] matrix the packer will
    quantise, so the denominator of the SmoothQuant ratio is measured on the
    matrix that is actually written (post-fold, post-transpose). Without it the
    caller gets the activation maxima only, which is what the printout needs.
    """
    import torch
    from transformers import WhisperFeatureExtractor

    import onnx_torch

    mp = Path(model_dir)
    # Weights from the ONNX export, for the reason vit_int8.calibrate gives:
    # this repository has no safetensors and no pytorch_model.bin, so
    # from_pretrained raised an OSError naming five files that do not exist.
    #
    # The TWO PREFIXES ARE NOT SYMMETRIC and that is measured, not tidied.
    # `encoder_model.onnx` names a tensor `conv1.weight` where the module tree
    # has `model.encoder.conv1.weight`, so it needs a prefix. `decoder_model.onnx`
    # already carries `model.decoder.` in its own tensor names, so prefixing it
    # too produces `model.decoder.model.decoder.embed_positions.weight` --
    # which matches nothing, and before onnx_torch's unmapped-tensor refusal
    # would have left every decoder parameter randomly initialised while the
    # encoder calibrated happily.
    #
    # `proj_out` is the tied `embed_tokens` projection. The decoder-only export
    # does not carry it -- it lives in the encoder graph -- so it is named as
    # absent rather than left to look like an oversight.
    model = onnx_torch.build(
        mp, "WhisperConfig", "WhisperForConditionalGeneration",
        [(mp / WHISPER_ENCODER_ONNX, "model.encoder."),
         (mp / WHISPER_DECODER_ONNX, "")],
        allow_missing=("proj_out.weight",),
        what=f"whisper int8 {mp.name}")
    # THE MEL BANK MUST COME FROM THE MODEL, NOT FROM THE DEFAULTS.
    # `WhisperFeatureExtractor()` is constructed empty, and its default
    # feature_size is 80 -- correct for tiny/base/small/medium and wrong for
    # large-v3 and large-v3-turbo, whose conv1 is [1280, 128, 3]. The failure
    # was not subtle: the encoder's first conv rejected an 80-channel input
    # with "weight of size [1280, 128, 3], expected input[1, 80, 3000] to have
    # 128 channels", which names the mismatch and lands on both large-v3
    # models and on neither of the four that work.
    #
    # from_pretrained reads models/<name>/preprocessor_config.json, where
    # feature_size is 80 or 128 as the checkpoint declares. A model directory
    # with no such file still has to calibrate, so the default is kept as the
    # fallback -- and stated when it is used, because "I assumed 80" and "the
    # model says 80" are different claims and only one of them is a fact about
    # the checkpoint.
    import json as _json
    mel_bins = None
    pre = mp / "preprocessor_config.json"
    if pre.is_file():
        mel_bins = _json.loads(pre.read_text(encoding="utf-8")).get("feature_size")
    if mel_bins is None:
        mel_bins = 80
        print(f"  {mp.name}: no preprocessor_config.json; assuming the default "
              f"feature_size=80 mel bank, which is what WhisperFeatureExtractor "
              f"would build. Confirm against the checkpoint if this model is "
              f"not one of the four that use it.")
    else:
        print(f"  {mp.name}: mel bank from preprocessor_config.json, "
              f"feature_size={mel_bins}")
    fe = WhisperFeatureExtractor(feature_size=int(mel_bins))
    # The forced prefix in `_forced_ids` wants REAL token ids, and used to get
    # them from `model.processor.tokenizer`. A model built from config.json has
    # no processor, so that attribute is gone; the tokenizer files themselves
    # are in the model directory (they are what the runtime loads, and what the
    # BERT calibration already asks transformers for), so read them directly.
    # Absent tokenizer -> None -> the synthetic ids `_forced_ids` already has a
    # branch for. That is a coarser corpus, not a wrong one, so it degrades
    # rather than refuses.
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(mp))
    except Exception:
        tok = None

    wanted = dict(site_keys(enc_layers, dec_layers))
    # (module path) -> container key. One path can feed one key here, because the
    # sites were chosen to be the fused-operand boundaries.
    seen: dict[str, str] = {}
    for key, path in wanted.items():
        if path in seen:
            raise SystemExit(f"two container keys share the hook {path}: "
                             f"{seen[path]} and {key}. The site list is the "
                             f"place to fix it, not the hooks.")
        seen[path] = key

    amax: dict[str, np.ndarray] = {}

    def hook(path: str):
        def fn(module, inputs):
            x = inputs[0]
            with torch.no_grad():
                v = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0)
            k = seen[path]
            a = v.to(torch.float32).numpy()
            amax[k] = a if k not in amax else np.maximum(amax[k], a)
        return fn

    handles = []
    for name, module in model.named_modules():
        if name in seen:
            handles.append(module.register_forward_pre_hook(hook(name)))
    if len(handles) != len(seen):
        raise SystemExit(f"whisper int8: hooked {len(handles)} of {len(seen)} "
                         f"sites. The module names in OPS do not match this "
                         f"checkpoint's tree; fix OPS, do not ship fewer "
                         f"statistics than sites.")

    n_frames = 0
    with torch.no_grad():
        for c, audio in enumerate(clips):
            feat = fe(audio, sampling_rate=16000, return_tensors="pt")["input_features"]
            enc = model.model.encoder(input_features=feat)
            ids = _forced_ids(model, tok, c)
            model.model.decoder(
                input_ids=ids,
                encoder_hidden_states=enc.last_hidden_state,
                use_cache=False)
            n_frames += int(feat.shape[-1])
    for h in handles:
        h.remove()

    out: dict[str, np.ndarray] = {}
    for key, a in amax.items():
        if weights is None or key not in weights:
            out[key] = a.astype(np.float32)
            continue
        w = np.abs(np.asarray(weights[key], dtype=np.float32)).max(axis=1)
        s = (np.maximum(a, 1e-8) ** alpha) / (np.maximum(w, 1e-8) ** (1 - alpha))
        out[key] = np.where(np.isfinite(s) & (s > 0), s, 1.0).astype(np.float32)
    if verbose:
        print(f"  whisper int8: calibrated on {len(clips)} clips, "
              f"{n_frames} mel frames, {len(amax)} GEMM sites "
              f"({enc_layers} encoder + {dec_layers} decoder layers), "
              f"alpha={alpha}")
    return out


def _forced_ids(model, tok, clip: int) -> "object":
    """A short forced prefix, so the decoder state's statistics are realistic.

    The prompt is the same one the runtime primes (SOT, language, task,
    no-timestamps) and it is followed by real tokens, because the decoder states
    a greedy step visits depend on what came before: statistics from an
    immediately-terminated step describe one position and nothing else.
    """
    import torch
    text = [" the quick brown fox", " and then she said",
            " that is the way it is", " well, i think so"][clip % 4]
    ids = [50258, 50259, 50359, 50363]
    if tok is not None:
        ids += tok(text, add_special_tokens=False)["input_ids"][:24]
    else:
        ids += [220] + list(range(1000, 1024))
    vocab = int(model.config.vocab_size)
    ids = [i for i in ids if 0 <= i < vocab]
    return torch.tensor([ids], dtype=torch.long)


def load_or_make_corpus(corpus: str | None, n_clips: int) -> tuple[list[np.ndarray], str]:
    if corpus:
        clips = read_corpus(corpus)
        return clips, f"{corpus} ({len(clips)} wav)"
    clips = synth_corpus(n_clips)
    return clips, f"synthetic, {len(clips)} clips (synth_corpus)"


def smooth_error(mat: np.ndarray, s: np.ndarray) -> float:
    """Relative error of a smoothed-then-quantised operand, for the report.

    NOT the same number as the quantisation error `add_gemm_b_int8` returns: that
    one measures Wq*s against W*s, this one measures the product's sensitivity,
    i.e. how much the smoothing moved the operand before quantisation saw it.
    Printed so a bad s is visible at pack time.
    """
    m = np.asarray(mat, dtype=np.float64)
    sm = m * s[:, None]
    den = float(np.linalg.norm(m))
    if not den:
        return 0.0
    return float(np.linalg.norm(sm - m) / den)


if __name__ == "__main__":  # a smoke test, not a gate
    sys.path.insert(0, str(REPO / "tools" / "lib"))
    print("import ok; sites for 4+4:", len(site_keys(4, 4)))
    for k, p in site_keys(4, 4)[:6]:
        print(" ", k, "->", p)
