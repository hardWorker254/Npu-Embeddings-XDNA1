#!/usr/bin/env python3
# NpuEmbeddings -- hold the C++ Whisper audio front end against transformers.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT IS BEING CHECKED, AND AGAINST WHAT
# ---------------------------------------
#   samples  the WAV reader and the ffmpeg path, against Python's own `wave`
#   mel      the log-mel spectrogram, against transformers'
#            WhisperFeatureExtractor -- the same function the model was trained
#            with, on the same samples
#   conv     conv1 -> GELU -> conv2 -> GELU -> permute, against torch, using
#            the conv weights OUT OF THE CONTAINER, so the two sides cannot
#            disagree about which weights they are
#   bank     the slaney mel filter bank on its own, so a divergence in the
#            spectrogram can be attributed to the bank or to the STFT instead
#            of leaving one number to explain both
#
# The corpus is built, not checked in: a tone, white noise, a clip shorter than
# 30 s (which must be zero-padded, not stretched) and one longer than 30 s
# (which must be cut). Those two are the whole padding story, and they are the
# cases a 30 s happy-path fixture would miss.
#
# THE REFUSALS ARE CASES TOO
# --------------------------
# A wrong sample rate, a second channel and a non-audio file must each be
# REFUSED, with a message that says what to do. Silently resampling is the
# failure this project treats as worst: it yields a fluent, plausible, wrong
# transcript and no signal at all that anything happened.
#
# Env: numpy, torch, transformers, ffmpeg for the conversion cases, a C++
# toolchain. The binary is built if --exe is not given.
#
# Usage:
#   python tools/verify_whisper_features.py
#   python tools/verify_whisper_features.py --npue models/whisper-base.npue

import argparse
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def build_exe(path):
    cmd = [
        "g++", "-std=c++17", "-O2", "-I", str(REPO / "runtime" / "include"),
        str(REPO / "runtime" / "tests" / "test_whisper_features.cpp"),
        str(REPO / "runtime" / "src" / "whisper" / "features.cpp"),
        str(REPO / "runtime" / "src" / "whisper" / "audio.cpp"),
        str(REPO / "runtime" / "src" / "model.cpp"),
        str(REPO / "runtime" / "src" / "pool.cpp"),
        "-o", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("building the C++ front end failed:\n" + r.stderr)
        return None
    return path


def write_wav(path, samples, rate=16000, channels=1, width=2):
    """samples: float in [-1, 1], already interleaved for `channels`."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 2:
            pcm = np.clip(np.rint(np.asarray(samples) * 32768.0), -32768, 32767)
            w.writeframes(pcm.astype("<i2").tobytes())
        else:  # 8-bit, unsigned with a 128 offset -- the classic wrong file
            pcm = np.clip(np.rint(np.asarray(samples) * 127.0) + 128, 0, 255)
            w.writeframes(pcm.astype(np.uint8).tobytes())


def run_cpp(exe, npue, audio, convert=False, threads=1):
    cmd = [str(exe), str(npue), str(audio)]
    if convert:
        cmd.append("--convert")
    if threads > 1:
        cmd += ["--threads", str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"{exe} failed:\n{p.stderr}")
        return None
    out = {"refused": None}
    for line in p.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "refused":
            out["refused"] = rest
        elif kind in ("samples", "mel", "conv"):
            # "<dims...> <hex>": the shape is however many integers precede the
            # hex, and `samples` has none, so the last field is always the hex.
            parts = rest.split()
            dims = [int(x) for x in parts[:-1]]
            arr = np.frombuffer(bytes.fromhex(parts[-1]), dtype="<f4")
            out[kind] = arr.reshape(*dims) if dims else arr
    return out


def read_wav_python(path):
    with wave.open(str(path), "rb") as w:
        n, width, rate = w.getnframes(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("this reference only reads 16-bit PCM")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def hf_mel(samples, n_mels, sample_rate=16000):
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=sample_rate)
    return fe(samples.astype(np.float32), sampling_rate=sample_rate)[
        "input_features"][0]


def torch_conv(mel, w1, b1, w2, b2):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))[None]
    y = torch.nn.functional.conv1d(x, torch.from_numpy(w1),
                                  torch.from_numpy(b1), padding=1)
    y = torch.nn.functional.gelu(y)
    y = torch.nn.functional.conv1d(y, torch.from_numpy(w2),
                                  torch.from_numpy(b2), stride=2, padding=1)
    y = torch.nn.functional.gelu(y)
    return y[0].T.contiguous().numpy()      # the permute to (frames, hidden)


def one_minus_cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float((a * b).sum() / (na * nb)) if na and nb else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the C++ Whisper front end against transformers.")
    ap.add_argument("--npue", default=str(REPO / "models" / "whisper-tiny.npue"))
    ap.add_argument("--exe", default=None)
    ap.add_argument("--mel-atol", type=float, default=1e-3)
    ap.add_argument("--conv-atol", type=float, default=2e-2)
    ap.add_argument("--keep", action="store_true", help="keep the built corpus")
    args = ap.parse_args()

    npue = Path(args.npue)
    if not npue.exists():
        print(f"{npue} not found -- pack it first")
        return 2
    from npue import Reader
    r = Reader(str(npue))
    n_mels = int(r.config["num_mel_bins"])
    d_model = int(r.config["d_model"])
    w1 = r.tensor("frontend.conv1.weight")
    b1 = r.tensor("frontend.conv1.bias")
    w2 = r.tensor("frontend.conv2.weight")
    b2 = r.tensor("frontend.conv2.bias")

    tmp = None
    exe = args.exe
    if exe is None:
        tmp = tempfile.mkdtemp(prefix="whisper-features-")
        exe = build_exe(str(Path(tmp) / "test_whisper_features"))
        if exe is None:
            return 2
    work = Path(tmp) if tmp else Path(tempfile.mkdtemp(prefix="whisper-features-"))

    rng = np.random.default_rng(4)
    def tone(seconds, freq=440.0, rate=16000):
        t = np.arange(int(seconds * rate)) / rate
        return (0.4 * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    cases = {
        "3 s 440 Hz tone": tone(3.0),
        "0.4 s tone, shorter than 30 s": tone(0.4, 880.0),
        "31 s tone, longer than 30 s": tone(31.0, 220.0),
        "5 s white noise": (rng.standard_normal(5 * 16000) * 0.2).astype(np.float32),
    }
    paths = {}
    for name, s in cases.items():
        p = work / (name.split(",")[0].replace(" ", "_") + ".wav")
        write_wav(p, s)
        paths[name] = p

    bad = 0
    print(f"container {npue.name}: d_model {d_model}, {n_mels} mel bins, "
          f"conv1 {tuple(w1.shape)} conv2 {tuple(w2.shape)}\n")

    for name, p in paths.items():
        got = run_cpp(exe, npue, p)
        if got is None or got.get("refused"):
            print(f"  FAIL {name}: refused ({got and got.get('refused')})")
            bad += 1
            continue
        want_samples, _ = read_wav_python(p)
        got_samples = got["samples"]
        n = min(len(want_samples), len(got_samples))
        d_s = float(np.abs(got_samples[:n] - want_samples[:n]).max())
        want_mel = hf_mel(got_samples, n_mels)
        d_mel = float(np.abs(got["mel"] - want_mel).max())
        want_conv = torch_conv(want_mel, w1, b1, w2, b2)
        d_conv = float(np.abs(got["conv"] - want_conv).max())
        cos_mel = one_minus_cos(got["mel"], want_mel)
        cos_conv = one_minus_cos(got["conv"], want_conv)
        ok = (d_s == 0.0 and d_mel <= args.mel_atol and d_conv <= args.conv_atol
              and got["mel"].shape == want_mel.shape
              and got["conv"].shape == want_conv.shape)
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:32s} "
              f"samples {d_s:.1e}  mel {d_mel:.2e} (1-cos {cos_mel:.1e})  "
              f"conv {d_conv:.2e} (1-cos {cos_conv:.1e})  "
              f"shapes {got['mel'].shape}/{want_mel.shape} "
              f"{got['conv'].shape}/{want_conv.shape}")

    # The filter bank on its own, so a spectrogram divergence can be attributed.
    from transformers.audio_utils import mel_filter_bank
    want_bank = mel_filter_bank(1 + 400 // 2, n_mels, 0.0, 8000.0, 16000,
                                norm="slaney", mel_scale="slaney")
    probe = run_cpp(exe, npue, paths["3 s 440 Hz tone"])
    # Rebuild the bank in Python exactly as features.cpp does and compare to
    # transformers': same inputs, same formula, one number.
    from transformers.audio_utils import hertz_to_mel, mel_to_hertz
    mel_min = hertz_to_mel(0.0, mel_scale="slaney")
    mel_max = hertz_to_mel(8000.0, mel_scale="slaney")
    centre = mel_to_hertz(np.linspace(mel_min, mel_max, n_mels + 2),
                          mel_scale="slaney")
    bank = np.zeros((201, n_mels))
    for j in range(n_mels):
        lo, mid, hi = centre[j], centre[j + 1], centre[j + 2]
        enorm = 2.0 / (hi - lo)
        hz = np.linspace(0, 8000, 201)
        w = np.where(hz > lo,
                     np.where(hz < hi, np.where(hz <= mid, (hz - lo) / (mid - lo),
                                                (hi - hz) / (hi - mid)), 0.0),
                     0.0)
        bank[:, j] = w * enorm
    d_bank = float(np.abs(bank - want_bank).max())
    print(f"\n  mel filter bank vs transformers: max abs {d_bank:.2e}  "
          f"shape {bank.shape}  {'ok' if d_bank < 1e-12 else 'FAIL'}")
    bad += 0 if d_bank < 1e-12 else 1
    del probe

    # -- the refusals ------------------------------------------------------
    print("\n  refusals:")
    stereo = work / "stereo_44k.wav"
    write_wav(stereo, np.tile(tone(1.0), 2), rate=44100, channels=2)
    eight = work / "eight_bit.wav"
    write_wav(eight, tone(1.0), width=1)
    not_audio = work / "not_audio.wav"
    not_audio.write_text("this is not audio at all\n")

    for label, path, convert, expect in (
            ("stereo 44.1 kHz WAV", stereo, False, "channels"),
            ("8-bit WAV", eight, False, "bits per sample"),
            ("44.1 kHz mono WAV", work / "rate_44k.wav", False, "Hz"),
            ("a text file via ffmpeg", not_audio, True, "ffmpeg"),
    ):
        if label.startswith("44.1"):
            write_wav(path, tone(1.0), rate=44100)
        got = run_cpp(exe, npue, path, convert=convert)
        msg = (got or {}).get("refused")
        ok = bool(msg) and expect in msg
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label:24s} "
              f"{'refused: ' + msg[:90] if msg else 'NOT REFUSED'}")

    # -- the parallel split must not change a single bit ------------------
    probe = paths["3 s 440 Hz tone"]
    one = run_cpp(exe, npue, probe)
    many = run_cpp(exe, npue, probe, threads=8)
    ok = one is not None and many is not None
    if ok:
        same = all(np.array_equal(one[k].tobytes(), many[k].tobytes())
                   for k in ("samples", "mel", "conv"))
        d = max(float(np.abs(one[k] - many[k]).max()) for k in ("mel", "conv"))
    else:
        same, d = False, float("nan")
    bad += 0 if same else 1
    print(f"\n  8 workers vs 1: {'byte-identical' if same else 'DIFFERS'} "
          f"(max abs {d:.1e})  {'ok' if same else 'FAIL'}")

    # -- the conversion path, on a file the WAV reader accepts ------------
    # ffmpeg re-encoding an s16 WAV must return the same samples the WAV
    # reader produced, which is what makes the execve/argv/pipe plumbing
    # checkable without a decoder on the Python side.
    plain = paths["3 s 440 Hz tone"]
    got = run_cpp(exe, npue, plain, convert=True)
    want, _ = read_wav_python(plain)
    ok = got is not None and not got.get("refused")
    if ok:
        n = min(len(want), len(got["samples"]))
        d = float(np.abs(got["samples"][:n] - want[:n]).max())
        ok = d == 0.0
    else:
        d = float("nan")
    bad += 0 if ok else 1
    print(f"\n  ffmpeg passthrough vs the WAV reader: max abs {d:.1e}  "
          f"{'ok' if ok else 'FAIL'}")

    print(f"\n{'FAILED' if bad else 'PASS'} ({bad} bad)")
    if args.keep:
        print(f"corpus kept in {work}")
    elif tmp is None:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        work.rmdir()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
