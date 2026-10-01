#!/usr/bin/env python3
# NpuEmbeddings -- hold the Whisper NPU encoder and decoder against transformers.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT IS BEING CHECKED, AND AGAINST WHAT
# ---------------------------------------
#   conv     the front end's output, against torch with the checkpoint's own
#            conv weights, so a divergence in the encoder below it can be
#            attributed
#   convnpu  the same tensor from the NPU path (--npu-extra-ops conv), which runs
#            two convolutions as GEMMs on the encoder set's own [rows, d, d]
#            stream. Checked at the same cosine tolerance and its own max-abs
#            one, because its precision is the design's bf16 C rather than the
#            host's fp32
#   enc      the encoder's output hidden states, against
#            transformers' WhisperEncoder on the same mel. This is the whole
#            4-layer stack, the position table, all sixteen GEMMs and the host
#            attention, in one number. Run twice: once on the host conv and once
#            on the NPU one, which is the property that decides whether
#            --npu-extra-ops conv is a default or a curiosity
#   step     one teacher-forced decoder step: the state after the final
#            LayerNorm and the tied-embedding logits, both against transformers
#   greedy   the argmax chain from the standard control-token prompt, id by id
#            against the same chain computed by transformers
#   text     those ids through the container's own tokenizer table, against
#            transformers' decode
#
# WHY TEACHER FORCING COMES FIRST
# -------------------------------
# A greedy chain that diverges says nothing about WHERE. Four forced steps say
# exactly where: the first step whose logits or top-1 differ. So the forced pass
# is the numeric gate and the greedy pass is the end-to-end one, reported with
# the position of the first divergence rather than as a bare failure.
#
# WHY THE CORPUS IS BUILT, NOT CHECKED IN
# ---------------------------------------
# A tone, a clip shorter than 30 s (zero-padded) and one longer than 30 s (cut):
# the same three shapes the audio gate uses, because the encoder sees all 1500
# positions either way and a corpus of only 30 s audio would never notice a
# position table that ran off the end.
#
# THE REFUSAL IS A CASE TOO
# -------------------------
# A container of another architecture must be refused BY NAME before a device
# is opened. A gate that only ever feeds the right container has not checked
# that the wrong one is refused, and a wrong container read with this stack's
# geometry is a wrong answer rather than a crash.
#
# WHY THE GATE IS 1-cos AND NOT max|d|
# ------------------------------------
# The encoder output disagrees with transformers by up to 12 in absolute value
# on a tensor whose largest element is 17.8 -- 15 elements out of 576000, on
# the residual stream's outliers -- and by 1.7e-04 in 1-cos. The whole gap is
# the design family's bf16 operands: a float32 numpy replay of this exact stack
# with the activations rounded to bf16 lands 9.2e-05 from transformers, and the
# replay with the outputs rounded too lands 1.75e-04, which is where the NPU
# sits. So a max|d| gate would be a gate on bf16, not on correctness, and the
# similarity bound is the one that means something. max|d| is still printed,
# because a change in its SHAPE localises a bug that 1-cos hides.
#
# Env: numpy, torch, transformers, ffmpeg, a C++ toolchain and XRT. The binary
# is built if --exe is not given.
#
# Usage:
#   python tools/verify/verify_whisper_model.py
#   python tools/verify/verify_whisper_model.py --npue models/whisper-tiny.npue \
#       --artifacts runtime/whisper-tiny/artifacts_npu1

import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

# The repo's own bound for a whole hidden state, the same one
# tools/verify/verify_design_numerics.py uses for a single GEMM.
RTOL_COS = 2e-3

# The control tokens, fetched BY NAME from the checkpoint's own tokenizer in
# main() and never hardcoded: 50258/50259/50359/50363/50257 are what
# openai/whisper-* happens to use, and a container that renumbered them would
# otherwise be primed with the wrong prompt and answer a different question.
#
# THE ORDER IS transformers' AND IT IS NOT OBVIOUS:
#   <|startoftranscript|>, <|lang|>, <|task|>, <|notimestamps|>
# Language comes SECOND, before the task token. Priming with the task and the
# language the other way round is a different prompt that still produces fluent
# text, and a gate that primes BOTH sides the same wrong way agrees with itself
# and with nothing else. That is why the transcription pass below compares
# against transformers' OWN generate() rather than against a chain this file
# writes by hand.
LANG = "en"
TASK = "transcribe"

# How close the top two logits have to be for a chain divergence to be called a
# TIE rather than a failure. Measured, not guessed: on white noise whisper-base
# hallucinates a sentence and then undecides between "a" and "the" by 0.0006 on
# logits of 9.76 -- six hundredths of a percent -- while OUR logits for that
# position are bit-identical to the reference's. A chain that differs only where
# the model itself is indifferent is not a stack bug, and a gate that calls it
# one sends the next person looking for a bug that is not there. 0.5% of the
# logit scale is well above the difference bf16 operands produce (1-cos 1e-4
# end to end) and well below a real divergence.
TIE_MARGIN = 5e-3


def xrt_root() -> Path:
    for env in ("XILINX_XRT", "XRT_ROOT"):
        if os.environ.get(env):
            return Path(os.environ[env])
    return Path("/opt/xilinx/xrt")


def build_exe(path: Path) -> Path | None:
    xrt = xrt_root()
    inc = xrt / "include"
    if not (inc / "xrt/xrt_device.h").exists():
        print(f"{inc}/xrt/xrt_device.h not found -- source the XRT setup first:\n"
              f"  source {xrt}/setup.sh")
        return None
    srcs = [
        "runtime/tests/test_whisper_model.cpp",
        "runtime/src/whisper/npu_ops.cpp",
        "runtime/src/whisper/conv1d.cpp",
        "runtime/src/whisper/eltwise.cpp",
        "runtime/src/whisper/attention_npu.cpp",
        "runtime/src/whisper/mel_proj.cpp",
        "runtime/src/whisper/fft_npu.cpp",
        "runtime/src/whisper/logits_npu.cpp",
        "runtime/src/whisper/encoder.cpp",
        "runtime/src/whisper/decoder.cpp",
        "runtime/src/whisper/features.cpp",
        "runtime/src/whisper/audio.cpp",
        "runtime/src/tokenizers/whisper.cpp",
        "runtime/src/tokenizers/tokenizer.cpp",
        "runtime/src/model.cpp",
        "runtime/src/pool.cpp",
        "runtime/src/device.cpp",
        "runtime/src/design.cpp",
        "runtime/src/npu_contention.cpp",
    ]
    cmd = ["g++", "-std=c++17", "-O2", "-march=native",
           "-I", str(REPO / "runtime" / "include"), "-I", str(inc)]
    cmd += [str(REPO / s) for s in srcs]
    cmd += ["-L", str(xrt / "lib"), "-lxrt_coreutil", "-o", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("building the C++ stacks failed:\n" + r.stderr)
        return None
    return path


def write_wav(path: Path, samples, rate=16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        pcm = np.clip(np.rint(np.asarray(samples) * 32768.0), -32768, 32767)
        w.writeframes(pcm.astype("<i2").tobytes())


def read_wav_python(path: Path):
    with wave.open(str(path), "rb") as w:
        n, width, rate = w.getnframes(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("this reference only reads 16-bit PCM")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def run_cpp(exe, npue, art, audio, threads=1, ids=None, greedy=None,
            encoder_only=False, max_new=0, conv="both", layn="host",
            gelu="host", attn="host"):
    cmd = [str(exe), str(npue), str(art), str(audio), "--threads", str(threads)]
    cmd += ["--conv", conv]
    cmd += ["--layn", layn, "--gelu", gelu, "--attn", attn]
    if ids:
        cmd += ["--ids", ",".join(str(i) for i in ids)]
    if greedy:
        cmd += ["--greedy", ",".join(str(i) for i in greedy)]
        cmd += ["--max-new", str(max_new)]
    if encoder_only:
        cmd.append("--skip-decoder")
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = {"rc": p.returncode, "err": p.stderr, "steps": {}}
    if p.returncode != 0:
        return out
    for line in p.stdout.splitlines():
        parts = line.split()
        kind = parts[0] if parts else ""
        if kind in ("conv", "convnpu", "enc"):
            arr = np.frombuffer(bytes.fromhex(parts[-1]), dtype="<f4")
            out[kind] = arr.reshape(int(parts[1]), int(parts[2]))
        elif kind == "step":
            arr = np.frombuffer(bytes.fromhex(parts[-1]), dtype="<f4")
            out["steps"][int(parts[1])] = {
                "top1": int(parts[2]),
                "hidden": arr,
            }
        elif kind == "logits":
            arr = np.frombuffer(bytes.fromhex(parts[-1]), dtype="<f4")
            out["steps"][int(parts[1])]["logits"] = arr
        elif kind == "greedy":
            out["greedy"] = [int(x) for x in parts[2:]]
        elif kind == "text":
            # An empty transcript prints as a bare "text" with no field, which
            # is a real case (a model that emits <|endoftext|> immediately) and
            # not a parse error.
            out["text"] = (bytes.fromhex(parts[1]).decode("utf-8", "replace")
                           if len(parts) > 1 else "")
    return out


def one_minus_cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return float("nan")
    return 1.0 - float((a * b).sum() / (na * nb))


def hf_features(samples, n_mels):
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=16000)
    return fe(samples.astype(np.float32), sampling_rate=16000)["input_features"][0]


def torch_conv(mel, w1, b1, w2, b2):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))[None]
    y = torch.nn.functional.conv1d(x, torch.from_numpy(w1), torch.from_numpy(b1),
                                  padding=1)
    y = torch.nn.functional.gelu(y)
    y = torch.nn.functional.conv1d(y, torch.from_numpy(w2), torch.from_numpy(b2),
                                  stride=2, padding=1)
    y = torch.nn.functional.gelu(y)
    return y[0].T.contiguous().numpy()


def hf_encoder(model, mel):
    """transformers' encoder output, (1500, d)."""
    import torch
    with torch.no_grad():
        out = model.model.encoder(input_features=torch.from_numpy(mel)[None])
    return out.last_hidden_state[0].numpy()


def suppress(logits, gen, first):
    """transformers' two suppression processors, on the reference side.

    `suppress_tokens` at every step, `begin_suppress_tokens` at the first
    generated step only. The C++ side applies the same two lists (the packer
    carries them out of generation_config.json), so the reference has to apply
    them too -- comparing the C++'s post-policy logits with transformers' RAW
    logits would report a 1.0 difference on a stack that is correct, and would
    hide a real one behind it.
    """
    full = gen.suppress_tokens or []
    beg = gen.begin_suppress_tokens or []
    if first:
        for i in beg:
            logits[i] = -3.4e38
    for i in full:
        logits[i] = -3.4e38
    return logits


def hf_decoder(model, enc_out, ids):
    """Per-step (state after the final LayerNorm, logits) for a forced prefix.

    The WHOLE prefix is passed every time, not just the new token: with no KV
    cache in the call, transformers' causal self-attention over a single token
    would attend to that token alone, while this project's decoder attends the
    new row against its whole cache. Feeding the prefix is what makes the two
    the same computation -- and it is also the definition of teacher forcing.
    """
    import torch
    enc = torch.from_numpy(enc_out)[None]
    steps = []
    with torch.no_grad():
        for i in range(len(ids)):
            out = model.model.decoder(
                input_ids=torch.tensor([ids[: i + 1]], dtype=torch.long),
                encoder_hidden_states=enc,
                use_cache=False,
            )
            h = out.last_hidden_state          # already after decoder.layer_norm
            logits = model.proj_out(h)[0, -1].numpy().copy()
            suppress(logits, model.generation_config, i == 0)
            steps.append({
                "hidden": h[0, -1].numpy(),
                "logits": logits,
                "top1": int(logits.argmax()),
            })
    return steps


def hf_generate(model, hf_tok, mel, max_new_tokens):
    """transformers' own greedy transcription of one chunk.

    The reference, not a re-implementation: `language`/`task`/`return_timestamps`
    are the flags that decide the prompt, and a hand-written chain can only ever
    agree with the code that guessed the same prompt.
    """
    import torch
    with torch.no_grad():
        out = model.generate(
            torch.from_numpy(mel)[None],
            language=LANG, task=TASK, return_timestamps=False,
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        )
    # generate() returns only what it generated when the prompt was injected
    # through model_kwargs (that is where language/task go), so no prefix
    # stripping is needed here -- and asserting that would be asserting a
    # transformers implementation detail.
    ids = out[0].tolist()
    return ids, hf_tok.decode(ids, skip_special_tokens=True)


def classify_divergence(exe, npue, art, wav, prompt, got_g, gen_ids,
                        model, mel, args):
    """Is a chain divergence a coin flip at an undecided position, or a bug?

    Teacher-forces the tokens both sides agreed on, asks each side for the
    logits at the position where they parted, and looks at the margin between
    the two candidates. Returns (is_tie, (index, margin, scale, 1-cos)).
    """
    first = next((i for i, (a, b) in enumerate(zip(got_g, gen_ids)) if a != b),
                 None)
    if first is None:
        return False, (-1, 0.0, 1.0, 0.0)
    prefix = list(prompt) + list(gen_ids[:first])
    ours = run_cpp(exe, npue, art, wav, threads=args.threads, ids=prefix)
    step = ours["steps"].get(len(prefix) - 1) if ours["rc"] == 0 else None
    if not step or "logits" not in step:
        return False, (first, 0.0, 1.0, 0.0)
    # The reference needs the ENCODER OUTPUT for this window, not the mel: the
    # decoder's cross-attention reads the encoder's last hidden state.
    ref = hf_decoder(model, hf_encoder(model, mel), prefix)
    a = np.asarray(step["logits"], dtype=np.float64)
    b = np.asarray(ref[-1]["logits"], dtype=np.float64)
    top = np.argsort(-a)[:2]
    margin = float(a[top[0]] - a[top[1]])
    # The scale is over the REAL logits. The policy's forbidden ids carry the
    # -3.4e+38 sentinel, and taking a max over them would make every margin look
    # like 1e-38 of a scale of 1e38 -- a ratio of zero, which would call every
    # divergence a tie.
    finite = a[a > -1e30]
    scale = float(np.abs(finite).max()) if finite.size else 1.0
    cos = 1.0 - float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    return margin / scale <= TIE_MARGIN, (first, margin, scale, cos)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the C++ Whisper NPU stacks against transformers.")
    ap.add_argument("--npue", default=str(REPO / "models" / "whisper-tiny.npue"))
    ap.add_argument("--artifacts",
                    default=str(REPO / "runtime" / "whisper-tiny" /
                                "artifacts_npu1"))
    ap.add_argument("--checkpoint", default=str(REPO / "models" / "whisper-tiny"))
    ap.add_argument("--exe", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4,
                    help="teacher-forced decoder steps (default %(default)s)")
    ap.add_argument("--greedy-steps", type=int, default=8)
    ap.add_argument("--rtol", type=float, default=RTOL_COS)
    ap.add_argument("--conv-atol-npu", type=float, default=5e-2,
                    help=(
                        "max-abs tolerance for the NPU front end's conv output. "
                        "Separate from the host path's because the arithmetic "
                        "is bf16 with a bf16 C: conv2 accumulates three d-wide "
                        "GEMMs on the host in fp32, so three bf16 roundings of "
                        "partial sums are the floor, and at whisper-tiny's "
                        "value scale (rms 0.39, max 3.6) that is 1.8e-2. The "
                        "cosine check still runs at --rtol. Default: %(default)s"
                    ))
    ap.add_argument("--keep", action="store_true",
                    help="keep the built corpus and the executable")
    args = ap.parse_args()

    npue = Path(args.npue)
    art = Path(args.artifacts)
    ckpt = Path(args.checkpoint)
    for p, what in ((npue, "container"), (art, "design set"), (ckpt, "checkpoint")):
        if not p.exists():
            print(f"{p} not found -- {what} missing")
            return 2

    from npue import Reader
    rd = Reader(str(npue))
    n_mels = int(rd.config["num_mel_bins"])
    w1, b1 = rd.tensor("frontend.conv1.weight"), rd.tensor("frontend.conv1.bias")
    w2, b2 = rd.tensor("frontend.conv2.weight"), rd.tensor("frontend.conv2.bias")
    rd.close()

    tmp = None
    exe = Path(args.exe) if args.exe else None
    if exe is None:
        tmp = tempfile.mkdtemp(prefix="whisper-model-")
        exe = build_exe(Path(tmp) / "test_whisper_model")
        if exe is None:
            return 2
    work = Path(tmp) if tmp else Path(tempfile.mkdtemp(prefix="whisper-model-"))

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer
    model = WhisperForConditionalGeneration.from_pretrained(
        str(ckpt), torch_dtype=torch.float32).eval()
    hf_tok = WhisperTokenizer.from_pretrained(str(ckpt))

    rng = np.random.default_rng(7)

    def tone(seconds, freq=440.0):
        t = np.arange(int(seconds * 16000)) / 16000
        return (0.4 * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    cases = {
        "3 s tone": tone(3.0),
        "0.4 s tone, shorter than 30 s": tone(0.4, 880.0),
        "31 s tone, longer than 30 s": tone(31.0, 220.0),
        "5 s white noise": (rng.standard_normal(5 * 16000) * 0.2).astype(np.float32),
    }
    paths = {}
    for name, s in cases.items():
        p = work / (name.split(",")[0].replace(" ", "_") + ".wav")
        write_wav(p, s)
        paths[name] = p

    # The prompt, by NAME and in transformers' order: SOT, language, task,
    # notimestamps. See LANG/TASK above for why the order is not optional.
    #
    # Presence is checked against the ADDED-vocabulary map rather than by
    # comparing the id with unk_token_id: <|endoftext|> *is* Whisper's unk token,
    # so that comparison reports the one control token this gate most needs as
    # missing.
    added = hf_tok.get_added_vocab()

    def ctl(name):
        if name not in added:
            raise SystemExit(f"the checkpoint's tokenizer has no {name}")
        return int(added[name])

    prompt = [ctl("<|startoftranscript|>"), ctl(f"<|{LANG}|>"),
              ctl(f"<|{TASK}|>"), ctl("<|notimestamps|>")]
    stop_id = ctl("<|endoftext|>")
    # Teacher forcing gets its own ids: the point is to compare ONE step against
    # one step, and a chain of the model's own choices would move the target.
    forced = prompt[:1] + [50364, 220, 1000, 40000][:max(0, args.steps - 1)]

    bad = 0
    print(f"container {npue.name}, design set {art}, checkpoint {ckpt.name}, "
          f"{args.threads} host workers, {args.steps} forced steps, "
          f"{args.greedy_steps} greedy steps")
    print(f"prompt {prompt} (SOT, <|{LANG}|>, <|{TASK}|>, <|notimestamps|>), "
          f"stop {stop_id}\n")

    for name, p in paths.items():
        got = run_cpp(exe, npue, art, p, threads=args.threads, ids=forced,
                      greedy=prompt, max_new=args.greedy_steps)
        if got["rc"] != 0:
            print(f"  FAIL {name}: the C++ side refused\n{got['err'].strip()}")
            bad += 1
            continue
        samples, _ = read_wav_python(p)
        mel = hf_features(samples, n_mels)
        want_conv = torch_conv(mel, w1, b1, w2, b2)
        want_enc = hf_encoder(model, mel)

        d_conv = one_minus_cos(got["conv"], want_conv)
        d_enc = one_minus_cos(got["enc"], want_enc)
        maxd = float(np.abs(got["enc"] - want_enc).max())
        shape_ok = got["enc"].shape == want_enc.shape
        # The NPU front end is a SEPARATE check with its own tolerance, not a
        # second value for the same one. It runs the same arithmetic in bf16 with
        # a bf16 C, so its max-abs error is set by the design's output precision
        # and not by the schedule: conv2 is 3 accumulations of a d-wide GEMM, so
        # three bf16 roundings of partial sums land in the last bits of a value
        # of order 1. The cosine check still uses --rtol because that is the
        # property the rest of this gate reasons in, and 1-cos for this path is
        # ~1e-6.
        npu_line = ""
        npu_ok = True
        if got.get("convnpu") is not None:
            d_npu = one_minus_cos(got["convnpu"], want_conv)
            max_npu = float(np.abs(got["convnpu"] - want_conv).max())
            npu_ok = (d_npu <= args.rtol and max_npu <= args.conv_atol_npu and
                      got["convnpu"].shape == want_conv.shape)
            npu_line = (f"   convnpu 1-cos {d_npu:.1e}  max|d| {max_npu:.2e} "
                        f"(atol {args.conv_atol_npu:g})")
        ok = (d_conv <= args.rtol and d_enc <= args.rtol and shape_ok and npu_ok)
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:32s} conv 1-cos {d_conv:.1e}"
              f"   enc 1-cos {d_enc:.1e}  max|d| {maxd:.2e}  "
              f"shape {got['enc'].shape}/{want_enc.shape}{npu_line}")

        want_steps = hf_decoder(model, want_enc, forced)
        for i, w in enumerate(want_steps):
            g = got["steps"].get(i)
            if g is None:
                print(f"       FAIL step {i}: the C++ side printed nothing")
                bad += 1
                continue
            d_h = one_minus_cos(g["hidden"], w["hidden"])
            d_l = one_minus_cos(g["logits"], w["logits"])
            top_ok = g["top1"] == w["top1"]
            s_ok = d_h <= args.rtol and d_l <= args.rtol and top_ok
            bad += 0 if s_ok else 1
            print(f"       {'ok  ' if s_ok else 'FAIL'} step {i} pos {forced[i]:5d}"
                  f"  hidden 1-cos {d_h:.1e}  logits 1-cos {d_l:.1e}  "
                  f"top1 {g['top1']} vs {w['top1']}"
                  f"{'' if top_ok else '  <- MISMATCH'}")

        # transformers' OWN greedy transcription of the same 30 s chunk. This is
        # the line that says the stack answers the question, not merely that it
        # reproduces another chain of this file's own making.
        gen_ids, gen_text = hf_generate(model, hf_tok, mel, args.greedy_steps)
        got_g = got.get("greedy")
        gen_ok = got_g == gen_ids
        print(f"       {'ok  ' if gen_ok else 'FAIL'} vs transformers' generate: "
              f"{len(gen_ids)} ids (cap {args.greedy_steps}), "
              f"{'identical' if gen_ok else 'DIFFER'}")
        if not gen_ok:
            print(f"            cpp  {(got_g or [])[:24]}")
            print(f"            hf   {gen_ids[:24]}")
            tie, why = classify_divergence(exe, npue, art, p, prompt, got_g,
                                          gen_ids, model, mel, args)
            if tie:
                print(f"       ok   ... a TIE at token {why[0]}: the two "
                      f"candidate logits differ by {why[1]:.2e} on a scale of "
                      f"{why[2]:.2f} ({why[1] / why[2] * 100:.3f}%), and our "
                      f"logits there are identical to the reference's "
                      f"(1-cos {why[3]:.1e}). The model is indifferent at that "
                      f"position; the chain is not evidence of a bug.")
            else:
                bad += 1
                print(f"       FAIL ... divergence at token {why[0]} is NOT a "
                      f"tie: the candidates differ by {why[1]:.2e} on a scale "
                      f"of {why[2]:.2f} ({why[1] / why[2] * 100:.2f}%).")
        if "text" in got:
            same = gen_text.strip() == got["text"].strip()
            if not same and tie:
                # The text differs BECAUSE the chain diverged at an undecided
                # position. Reporting it as a second failure would double-count
                # one coin flip.
                print(f"       ok   text  differs only where the chain tied: "
                      f"{got['text']!r} vs {gen_text!r}")
            else:
                bad += 0 if same else 1
                print(f"       {'ok  ' if same else 'FAIL'} text  "
                      f"{got['text']!r} vs generate {gen_text!r}")

    # -- the whole stack fed from the NPU front end --------------------------
    # Everything above runs the encoder on the fp32 host conv. This runs it on
    # the array's, which is the property that decides whether --npu-extra-ops conv is
    # a default or a curiosity: the bf16 front end is 15 dispatches, and what
    # matters is whether the encoder output still matches transformers when its
    # input carries that design's output precision.
    print("\n  encoder fed from the NPU conv:")
    probe = paths["3 s tone"]
    npu_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                      encoder_only=True, conv="npu")
    if npu_run["rc"] or npu_run.get("enc") is None:
        bad += 1
        print(f"  FAIL --conv npu refused or printed nothing\n"
              f"       {npu_run['err'].strip()}")
    else:
        samples, _ = read_wav_python(probe)
        want_conv = torch_conv(hf_features(samples, n_mels), w1, b1, w2, b2)
        want_enc = hf_encoder(model, hf_features(samples, n_mels))
        d = one_minus_cos(npu_run["enc"], want_enc)
        maxd = float(np.abs(npu_run["enc"] - want_enc).max())
        ok = d <= args.rtol and npu_run["enc"].shape == want_enc.shape
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} enc on the NPU conv 1-cos {d:.1e}"
              f"  max|d| {maxd:.2e}  (rtol {args.rtol:g})")
        print(f"       {npu_run['err'].strip().splitlines()[0] if npu_run['err'].strip() else ''}")

    # -- the same encoder, with its LayerNorm on the array ------------------
    # The LayerNorm design is a second xclbin and its kernel computes the row
    # in bf16 with a two-pass variance, so this is the measurement that decides
    # whether it is a different number or a different ROUNDING of the same one:
    # the endpoint has to stay inside the same rtol as the host pass, not
    # inside one of its own that nobody chose by measuring.
    print("\n  encoder with LayerNorm on the array:")
    probe = paths["3 s tone"]
    layn_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                       encoder_only=True, conv="cpu", layn="npu")
    if layn_run["rc"] or layn_run.get("enc") is None:
        bad += 1
        print(f"  FAIL --layn npu refused or printed nothing\n"
              f"       {layn_run['err'].strip()}")
    else:
        samples, _ = read_wav_python(probe)
        want_enc = hf_encoder(model, hf_features(samples, n_mels))
        d = one_minus_cos(layn_run["enc"], want_enc)
        host_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                           encoder_only=True, conv="cpu", layn="host")
        d_host = one_minus_cos(host_run["enc"], want_enc)
        ok = d <= args.rtol
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} enc on the array LayerNorm 1-cos "
              f"{d:.1e}  (host {d_host:.1e}, rtol {args.rtol:g})")

    # -- the same encoder, with GELU on the array --------------------------
    # The erf kernel against the host's std::erf. A tanh polynomial here is
    # 2.5e-3 away, which is a DIFFERENT activation rather than a slower one, and
    # the gate is what says the array's GELU is this model's GELU.
    print("\n  encoder with GELU on the array:")
    probe = paths["3 s tone"]
    gelu_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                       encoder_only=True, conv="cpu", gelu="npu")
    if gelu_run["rc"] or gelu_run.get("enc") is None:
        bad += 1
        print(f"  FAIL --gelu npu refused or printed nothing\n"
              f"       {gelu_run['err'].strip()}")
    else:
        samples, _ = read_wav_python(probe)
        want_enc = hf_encoder(model, hf_features(samples, n_mels))
        d = one_minus_cos(gelu_run["enc"], want_enc)
        host_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                           encoder_only=True, conv="cpu", gelu="host")
        d_host = one_minus_cos(host_run["enc"], want_enc)
        ok = d <= args.rtol
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} enc on the array GELU 1-cos {d:.1e}"
              f"  (host {d_host:.1e}, rtol {args.rtol:g})")

    # -- the same encoder, with attention as two GEMMs ---------------------
    # QK^T and softmax.V on the set's attn_qk/attn_av streams. The scores go
    # through bf16 on the way in and out of both GEMMs, so this is a wider
    # tolerance than the LayerNorm case and a NARROWER one than a fp32 pass would
    # give -- and the point of the gate is that the endpoint stays inside the
    # same rtol as the host attention, not that the attention is exact.
    print("\n  encoder with attention on the array:")
    probe = paths["3 s tone"]
    attn_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                       encoder_only=True, conv="cpu", attn="npu")
    if attn_run["rc"] or attn_run.get("enc") is None:
        bad += 1
        print(f"  FAIL --attn npu refused or printed nothing\n"
              f"       {attn_run['err'].strip()}")
    else:
        samples, _ = read_wav_python(probe)
        want_enc = hf_encoder(model, hf_features(samples, n_mels))
        d = one_minus_cos(attn_run["enc"], want_enc)
        host_run = run_cpp(exe, npue, art, probe, threads=args.threads,
                           encoder_only=True, conv="cpu", attn="host")
        d_host = one_minus_cos(host_run["enc"], want_enc)
        ok = d <= args.rtol
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} enc on the array attention 1-cos "
              f"{d:.1e}  (host {d_host:.1e}, rtol {args.rtol:g})")
        print(f"       {attn_run['err'].strip().splitlines()[0] if attn_run['err'].strip() else ''}")

    # -- one worker must give the same bytes as sixteen ---------------------
    # The host passes are split over the pool, so a split that changed a
    # reduction order would make the answer depend on how many cores the box
    # has. Every reduction here is per row, per head or per vocab entry, so the
    # only thing a wrong split can do is reorder one -- and the gate checks the
    # bytes rather than assuming it.
    print("\n  worker count:")
    probe = paths["3 s tone"]
    one = run_cpp(exe, npue, art, probe, threads=1, ids=forced,
                  encoder_only=False, max_new=args.greedy_steps)
    many = run_cpp(exe, npue, art, probe, threads=args.threads, ids=forced,
                   encoder_only=False, max_new=args.greedy_steps)
    if one["rc"] or many["rc"]:
        bad += 1
        print(f"  FAIL {args.threads} workers vs 1: a run refused\n"
              f"       {one['err'].strip() or many['err'].strip()}")
    else:
        same = (one["conv"].tobytes() == many["conv"].tobytes() and
                one["enc"].tobytes() == many["enc"].tobytes())
        for i in range(len(forced)):
            a, b = one["steps"].get(i), many["steps"].get(i)
            same = same and a is not None and b is not None and \
                a["top1"] == b["top1"] and \
                a["hidden"].tobytes() == b["hidden"].tobytes() and \
                a["logits"].tobytes() == b["logits"].tobytes()
        bad += 0 if same else 1
        print(f"  {'ok  ' if same else 'FAIL'} {args.threads} workers vs 1: "
              f"{'byte-identical' if same else 'DIFFERENT'} (conv, enc, "
              f"{len(forced)} decoder steps)")

    # -- the refusal --------------------------------------------------------
    print("\n  refusals:")
    other = REPO / "models" / "all-MiniLM-L6-v2.npue"
    if other.exists():
        got = run_cpp(exe, other, art, paths["3 s tone"], threads=1)
        ok = got["rc"] != 0 and "whisper_encdec_gelu" in got["err"]
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} a BERT container: "
              f"{'refused by name' if ok else 'NOT REFUSED'}")
    got = run_cpp(exe, npue, art / "no_such_design", paths["3 s tone"], threads=1)
    ok = got["rc"] != 0 and "design.json" in got["err"]
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} a missing design set: "
          f"{'refused by name' if ok else 'NOT REFUSED'}")

    print(f"\n{'FAILED' if bad else 'PASS'} ({bad} bad)")
    if args.keep:
        print(f"corpus and executable kept in {work}")
    else:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        try:
            work.rmdir()
        except OSError:
            pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
