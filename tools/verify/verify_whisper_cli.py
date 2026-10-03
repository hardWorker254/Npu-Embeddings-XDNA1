#!/usr/bin/env python3
# NpuEmbeddings -- hold the transcription CLI and the /v1/audio/transcriptions
# endpoint against transformers.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT IS BEING CHECKED
# ---------------------
#   windows   the long-form SCHEDULE: 30 s windows, 5 s stride on each side, so
#             consecutive windows start 20 s apart, and the last one covers the
#             tail. The offsets are in the response, so this checks the rule and
#             not just the number of pieces.
#   per window the transcript of each window alone, against transformers'
#             generate() on the same window. (verify_whisper_model.py already
#             holds the stack itself against generate(); this is the same
#             comparison at the level a request sees it.)
#   merged    the whole text, against transformers' OWN merge: its
#             _find_longest_common_sequence, imported from the installed package
#             rather than reimplemented here. A reimplementation would compare
#             this build against this build.
#   endpoint  the same audio over HTTP, plus the refusals a client can hit.
#
# WHY NOT THE PIPELINE AS THE REFERENCE
# -------------------------------------
# transformers' ASR pipeline detects the language when the caller does not name
# one, and it did not take the language/task arguments this gate passes -- the
# ids it produced were not the ones its own generate() produces for the same
# window. A reference that answers a different question is worse than no
# reference, so this uses generate() plus the merge function directly.
#
# Env: numpy, torch, transformers, a C++ toolchain and XRT, and a built runtime
# (--exe, or runtime/build/npueembed).
#
# Usage:
#   python tools/verify/verify_whisper_cli.py
#   python tools/verify/verify_whisper_cli.py --max-new 16 --threads 8

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

LANG = "en"
TASK = "transcribe"
RATE = 16000
CHUNK_S = 30
STRIDE_S = 5


def write_wav(path: Path, samples) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        pcm = np.clip(np.rint(np.asarray(samples) * 32768.0), -32768, 32767)
        w.writeframes(pcm.astype("<i2").tobytes())


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        n, width, rate = w.getnframes(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width != 2 or rate != RATE:
        raise ValueError(f"{path}: the gate's corpus is 16 kHz 16-bit mono")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def windows(n_samples: int):
    """The long-form schedule, as the reference lays it out.

    transformers strides BOTH sides of a window, so the step is
    chunk - 2*stride (20 s for the default 30/5), and the loop stops on the
    window that covers the tail -- a one-second sliver after it would be a whole
    encoder pass over nothing.
    """
    chunk = CHUNK_S * RATE
    step = chunk - 2 * STRIDE_S * RATE
    out = []
    pos = 0
    while True:
        end = min(n_samples, pos + chunk)
        out.append((pos, end))
        if end >= n_samples:
            return out
        pos += step


class Reference:
    """transformers' own per-window generate() and its own merge."""

    def __init__(self, ckpt: Path, n_mels: int):
        import torch
        from transformers import (GenerationConfig,
                                  WhisperFeatureExtractor,
                                  WhisperTokenizer)

        import onnx_torch
        from onnx_weights import (WHISPER_DECODER_ONNX,
                                  WHISPER_ENCODER_ONNX)

        self.torch = torch
        # Weights from the ONNX export rather than from_pretrained, for the
        # reason whisper_int8 documents: this tree ships no pytorch_model.bin and
        # no safetensors file, so from_pretrained raised an OSError listing five
        # filenames that do not exist for a checkpoint whose weights were present
        # one directory up. `proj_out.weight` is absent by construction -- the
        # tied `embed_tokens` projection lives in the encoder graph.
        #
        # The tokenizer reads vocab.json/merges.txt, which this tree does ship,
        # so it stays on from_pretrained: it needs no weights.
        self.model = onnx_torch.build(
            ckpt, "WhisperConfig", "WhisperForConditionalGeneration",
            [(ckpt / WHISPER_ENCODER_ONNX, "model.encoder."),
             (ckpt / WHISPER_DECODER_ONNX, "")],
            allow_missing=("proj_out.weight",),
            what=f"verify_whisper_cli {Path(ckpt).name}").eval()
        # The generation config comes from the checkpoint directory, for the
        # reason verify_whisper_model spells out at length: the ONNX-built tree
        # has the class-default generation config, which carries no
        # `lang_to_id`/`task_to_id`, and generate() then refuses a `language=`
        # as an outdated config. models/whisper-tiny/generation_config.json has
        # both maps, and from_pretrained used to attach it.
        self.model.generation_config = GenerationConfig.from_pretrained(
            str(ckpt))
        self.tok = WhisperTokenizer.from_pretrained(str(ckpt))
        self.fe = WhisperFeatureExtractor(feature_size=n_mels,
                                           sampling_rate=RATE)
        self.max_seq = int(self.model.config.max_source_positions)
        self.max_target = int(self.model.config.max_target_positions)

    def window_ids(self, samples, max_new):
        """The ids for one window, exactly as the reference produces them."""
        import torch
        mel = np.asarray(self.fe(samples, sampling_rate=RATE)["input_features"],
                         dtype=np.float32)
        # The extractor hands back a list for some lengths and an array for
        # others, so a window can arrive as (1, n_mels, 3000) or as
        # (n_mels, 3000). Normalise before the batch dimension goes on, or the
        # model's conv1d gets a 4-D tensor and says so.
        if mel.ndim == 3:
            mel = mel[0]
        with torch.no_grad():
            out = self.model.generate(
                torch.from_numpy(mel)[None],
                language=LANG, task=TASK, return_timestamps=False,
                do_sample=False, num_beams=1, max_new_tokens=max_new)
        return [int(i) for i in out[0].tolist()]

    def merge(self, per_window_ids):
        """transformers' merge, imported -- not this file's idea of one."""
        from transformers.models.whisper.tokenization_whisper import (
            _find_longest_common_sequence)
        merged = _find_longest_common_sequence([list(x) for x in per_window_ids])
        return self.tok.decode([int(i) for i in merged], skip_special_tokens=True)


def run_cli(exe, npue: Path, art, audio, args, extra=()):
    # By NAME, not by path: the subcommand resolves the model through the hub
    # catalogue, exactly as `embed` does, and a name is what a user has.
    cmd = [str(exe), "transcribe", npue.stem, str(audio), "--artifacts", str(art),
           "--language", LANG, "--max-new", str(args.max_new),
           "--json", "--threads", str(args.threads), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["(no message)"])[-1]
    try:
        return json.loads(p.stdout), ""
    except json.JSONDecodeError as e:
        return None, f"stdout is not JSON: {e}: {p.stdout[:200]!r}"


def post_multipart(port, fields, files, timeout=1800):
    """A multipart POST built by hand, so the test does not depend on requests.

    The boundary is mixed-case on purpose: it is opaque data, and a server that
    folds the case of a Content-Type parameter cannot match the body it was sent
    (that was a real bug, and a lowercase-only boundary would never have found
    it).
    """
    boundary = "----npuGate" + uuid.uuid4().hex[:12].upper()
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for name, (filename, ctype, data) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n').encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the Whisper CLI and endpoint against transformers.")
    ap.add_argument("--npue", default=str(REPO / "models" / "whisper-tiny.npue"))
    # runtime/artifacts/<model>/artifacts_npu<N>, where every model's sets live.
    # The default used to be runtime/<model>/artifacts_npu<N>, which is the
    # pre-relocation path: it still parses and still names a directory that is
    # simply not there, so this gate failed on a missing design set and said
    # nothing about the CLI it exists to check.
    ap.add_argument("--artifacts",
                    default=str(REPO / "runtime" / "artifacts" / "whisper-tiny" /
                                "artifacts_npu1"))
    ap.add_argument("--checkpoint", default=str(REPO / "models" / "whisper-tiny"))
    ap.add_argument("--exe",
                    default=str(REPO / "runtime" / "build" / "npuembeddings"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=16,
                    help="cap on generated tokens per window; the same cap goes "
                         "to transformers' generate(), so the two chains are "
                         "the same string (default %(default)s)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    npue, art, ckpt = Path(args.npue), Path(args.artifacts), Path(args.checkpoint)
    exe = Path(args.exe)
    for p, what in ((npue, "container"), (art, "design set"),
                    (ckpt, "checkpoint"), (exe, "runtime binary")):
        if not p.exists():
            print(f"{p} not found -- {what} missing")
            return 2
    if not os.access(exe, os.X_OK):
        print(f"{exe} is not executable")
        return 2

    from npue import Reader
    rd = Reader(str(npue))
    n_mels = int(rd.config["num_mel_bins"])
    rd.close()

    ref = Reference(ckpt, n_mels)
    work = Path(tempfile.mkdtemp(prefix="whisper-cli-"))
    rng = np.random.default_rng(11)

    def tone(seconds, freq):
        t = np.arange(int(seconds * RATE)) / RATE
        return (0.4 * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    cases = {
        "3 s tone": tone(3.0, 440.0),
        "0.4 s tone, shorter than 30 s": tone(0.4, 880.0),
        "5 s white noise": (rng.standard_normal(5 * RATE) * 0.2).astype(np.float32),
        "31 s tone, two windows": tone(31.0, 220.0),
        "51 s tone, three windows": tone(51.0, 330.0),
    }
    paths = {}
    for name, s in cases.items():
        p = work / (name.split(",")[0].replace(" ", "_") + ".wav")
        write_wav(p, s)
        paths[name] = p

    bad = 0
    print(f"exe {exe.name}, container {npue.name}, checkpoint {ckpt.name}, "
          f"{args.threads} threads, max_new {args.max_new}\n")

    for name, p in paths.items():
        got, why = run_cli(exe, npue, art, p, args)
        if got is None:
            print(f"  FAIL {name}: the CLI refused -- {why}")
            bad += 1
            continue
        samples = read_wav(p)
        wins = windows(len(samples))
        ids = [ref.window_ids(samples[a:b], args.max_new) for a, b in wins]
        want_segs = [ref.tok.decode(i, skip_special_tokens=True) for i in ids]
        want_text = ref.merge(ids)

        segs = got.get("segments", [])
        offs_ok = len(segs) == len(wins) and all(
            s["seek"] == a and abs(s["start"] - a / RATE) < 0.01 and
            abs(s["end"] - b / RATE) < 0.01
            for s, (a, b) in zip(segs, wins))
        text_ok = got.get("text", "").strip() == want_text.strip()
        seg_ok = [s["text"] for s in segs] == want_segs
        dur_ok = abs(got.get("duration", 0) - len(samples) / RATE) < 0.01
        meta_ok = got.get("language") == LANG and got.get("task") == TASK
        ok = offs_ok and text_ok and seg_ok and dur_ok and meta_ok
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:32s} {len(wins)} window(s) "
              f"{[f'{a/RATE:.0f}-{b/RATE:.0f}s' for a, b in wins]}")
        print(f"       {'ok  ' if offs_ok else 'FAIL'} window offsets and count")
        print(f"       {'ok  ' if seg_ok else 'FAIL'} per window: "
              f"{[s['text'] for s in segs]}")
        if not seg_ok:
            print(f"            want {[t[:40] for t in want_segs]}")
        print(f"       {'ok  ' if text_ok else 'FAIL'} merged text: "
              f"{got.get('text', '')[:70]!r} vs {want_text[:70]!r}")
        print(f"       {'ok  ' if dur_ok and meta_ok else 'FAIL'} duration and "
              f"task/language fields")

    # -- the endpoint --------------------------------------------------------
    print("\n  endpoint:")
    port = 8000 + (os.getpid() % 1000)
    srv_log = work / "server.log"
    with srv_log.open("w") as lf:
        srv = subprocess.Popen(
            [str(exe), "serve", npue.stem, "--artifacts", str(art),
             "--port", str(port), "--threads", str(args.threads),
             "--root", str(REPO)],
            stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ready = False
        for _ in range(120):
            time.sleep(0.5)
            if srv.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
        if not ready:
            print(f"  FAIL the endpoint never came up:\n"
                  f"       {srv_log.read_text()[-800:]}")
            bad += 1
        else:
            wav = paths["3 s tone"].read_bytes()
            cli_text = run_cli(exe, npue, art, paths["3 s tone"], args)[0]["text"]
            for label, fields, files, want_status in (
                ("a WAV, json", {}, {"file": ("a.wav", "audio/wav", wav)}, 200),
                ("a WAV, verbose_json", {"response_format": "verbose_json"},
                 {"file": ("a.wav", "audio/wav", wav)}, 200),
                ("language en", {"language": "en"},
                 {"file": ("a.wav", "audio/wav", wav)}, 200),
            ):
                st, body = post_multipart(port, fields, files)
                ok = st == want_status
                got_text = ""
                if ok:
                    try:
                        got_text = json.loads(body).get("text", "")
                    except json.JSONDecodeError:
                        ok = False
                if ok and label != "a WAV, verbose_json":
                    ok = got_text.strip() == cli_text.strip()
                bad += 0 if ok else 1
                print(f"  {'ok  ' if ok else 'FAIL'} {label:24s} status {st} "
                      f"text {got_text[:48]!r}"
                      + ("" if ok or got_text.strip() == cli_text.strip()
                         else f"   (cli {cli_text[:48]!r})"))
            for label, fields, files, want_status, want_in in (
                ("temperature 0.7", {"temperature": "0.7"},
                 {"file": ("a.wav", "audio/wav", wav)}, 400, "greedy only"),
                ("prompt given", {"prompt": "hello"},
                 {"file": ("a.wav", "audio/wav", wav)}, 400, "not implemented"),
                ("no file part", {}, {}, 400, "'file' is required"),
                # Derived from the container, not hardcoded: run against
                # whisper-base and a literal "whisper-base" is the RIGHT model,
                # so the case would pass for the wrong reason.
                ("wrong model", {"model": npue.stem + "-other"},
                 {"file": ("a.wav", "audio/wav", wav)}, 400, "serves"),
                ("word timings", {"timestamp_granularities": "word"},
                 {"file": ("a.wav", "audio/wav", wav)}, 400, "timestamps"),
            ):
                st, body = post_multipart(port, fields, files)
                ok = st == want_status and want_in in body
                bad += 0 if ok else 1
                print(f"  {'ok  ' if ok else 'FAIL'} {label:24s} status {st} "
                      f"(wanted {want_status} mentioning {want_in!r})")
            st, body = post_multipart(port, {}, {"file": ("a.wav", "audio/wav",
                                                          wav)})
            bad += 0 if st == 200 else 1
            print(f"  {'ok  ' if st == 200 else 'FAIL'} no language field      "
                  f"status {st} (an assumed language is the server's default "
                  f"and it says so in /health)")
    finally:
        if srv.poll() is None:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.kill()

    # -- the CLI's refusals --------------------------------------------------
    print("\n  refusals:")
    for label, extra, want in (
        # No audio argument at all: the subcommand's own usage refusal.
        ("no audio at all", ["__NOAUDIO__"], "needs an audio file"),
        # The flag form has no subcommand to catch it, so the MODE refuses it --
        # and `transcribe <model>` is caught before Runtime::run ever sees a
        # container. Both paths exist, so both refusals are checked.
        ("flag form, nothing to do", ["__FLAGFORM__"],
         "say what to transcribe"),
        # The refusal comes from the TOKENIZER, by name, before any of this
        # build's own checks: a language with no token is a name this container
        # cannot speak, and saying so is better than falling back to English.
        ("a language with no token", ["--language", "klingon"],
         "tokenizer has no <|klingon|>"),
        ("a window longer than the model", ["--chunk-seconds", "60"],
         "encoder positions"),
        ("a stride over half the window", ["--stride-seconds", "20"],
         "at most half the chunk length"),
        ("a missing file", ["__MISSING__"], "cannot open"),
    ):
        if extra and extra[0] == "__FLAGFORM__":
            cmd = [str(exe), "--model", npue.stem, "--artifacts", str(art),
                   "--root", str(REPO)]
        elif extra and extra[0] == "__NOAUDIO__":
            cmd = [str(exe), "transcribe", npue.stem, "--artifacts", str(art)]
        else:
            cmd = [str(exe), "transcribe", npue.stem, "--artifacts", str(art),
                   *extra]
            audio = (str(work / "nope.wav")
                     if extra and extra[0] == "__MISSING__"
                     else str(paths["3 s tone"]))
            cmd.insert(3, audio)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # The WHOLE message, not its last line: a refusal here is deliberately
        # several lines (what went wrong, then what to type instead), and
        # matching against the tail of it would test the formatter, not the
        # refusal.
        msg = p.stderr.strip()
        ok = p.returncode != 0 and want in msg
        bad += 0 if ok else 1
        # Quote the line that carries the refusal, not the status block above it.
        line = next((l for l in msg.splitlines() if "error:" in l), "")
        if not ok:
            line = line or msg.replace("\n", " ")[:120]
        print(f"  {'ok  ' if ok else 'FAIL'} {label:28s} "
              f"{(line or msg)[:110]}")

    print(f"\n{'FAILED' if bad else 'PASS'} ({bad} bad)")
    if args.keep:
        print(f"corpus kept in {work}")
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
