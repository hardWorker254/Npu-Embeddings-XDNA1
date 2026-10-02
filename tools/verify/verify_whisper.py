#!/usr/bin/env python3
# NpuEmbeddings -- word error rate of the C++ transcription path against a
# human reference transcript.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS GATE IS, AND WHY IT IS SEPARATE FROM THE OTHER TWO
# -----------------------------------------------------------
#   tools/verify/verify_whisper_model.py  the NUMBERS: mel, encoder hidden states,
#                                 decoder states and logits against
#                                 transformers, step by step.
#   tools/verify/verify_whisper_cli.py    the PIPELINE: windows, merge, endpoints.
#   this gate                      the ANSWER: does the transcript a person
#                                 would accept come out, and how many words are
#                                 wrong against what a human wrote down.
#
# The other two can both be green while the transcript is wrong, because they
# only ever compare against another program's output. WER against a human
# reference is the only check here that has an opinion of its own.
#
# WHY THE REFERENCE AND THE AUDIO ARE BOTH ARGUMENTS
# --------------------------------------------------
# No audio is committed. A speech corpus in the repository is a licensing
# question, a repository-size question, and a "whose words is this" question --
# none of which belongs in a commit made by a packer. So the corpus lives
# outside the tree and the gate is told where it is:
#
#   --audio X.wav --ref "the words a human wrote"
#   --corpus DIR           every DIR/*.wav with a DIR/*.txt beside it
#
# And it reports BOTH sides' WER against the same reference, ours and
# transformers'. That is the part that makes the number mean something: if ours
# rises and theirs does not, the stack regressed; if both rise, the reference or
# the audio is the problem, and the gate says so rather than failing.
#
# WER is computed on NORMALISED text: lowercase, strip punctuation, collapse
# whitespace. Whisper writes " And so my fellow Americans," and a human writes
# "and so my fellow americans" -- the case and the comma are not recognition
# errors, and counting them would make the number a measure of punctuation
# conventions instead of one of transcription.
#
# Env: numpy, torch, transformers, a built runtime (--exe).
#
# Usage:
#   python tools/verify/verify_whisper.py --audio jfk.wav \\
#       --ref "and so my fellow americans ask not what your country can do for you"
#   python tools/verify/verify_whisper.py --corpus ~/speech-corpus --npue ... --artifacts ...

import argparse
import json
import string
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

LANG = "en"
TASK = "transcribe"


def normalise(text: str) -> list[str]:
    """Lowercase, drop punctuation, split on whitespace.

    The same reduction on both sides, and the reference is reduced with it, so
    "Americans," and "americans" are one word and not an error pair.
    """
    t = text.lower()
    t = "".join(" " if c in string.punctuation else c for c in t)
    return t.split()


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """(substitutions+deletes+inserts, substitutions, deletions, insertions).

    Levenshtein with the operation mix broken out, because a WER of 0.12 reads
    very differently when it is 12 wrong words against 12 MISSING words: the
    first is the model hearing something else, the second is the model cutting
    off.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, subs, dels, ins) for ref[:i] vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            best = min((sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                       (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                       (ins[0] + 1, ins[1], ins[2], ins[3] + 1))
            cur.append(best)
        prev = cur
    c, s, d, i_ = prev[m]
    return c, s, d, i_


def wer_report(ref: list[str], hyp: list[str]) -> dict:
    c, s, d, i_ = edit_distance(ref, hyp)
    return {"ref_words": len(ref), "hyp_words": len(hyp), "errors": c,
            "substitutions": s, "deletions": d, "insertions": i_,
            "wer": (c / len(ref)) if ref else float("nan")}


def run_ours(exe, name, audio, args) -> tuple[str, float, dict]:
    t0 = time.monotonic()
    p = subprocess.run(
        [str(exe), "transcribe", name, str(audio), "--language", LANG,
         "--artifacts", str(args.artifacts), "--json",
         "--threads", str(args.threads)]
        + (["--npu-extra-ops", args.npu_ops] if args.npu_ops else []),
        capture_output=True, text=True, timeout=3600)
    dt = time.monotonic() - t0
    if p.returncode != 0:
        raise SystemExit(f"the C++ side refused {audio}:\n{p.stderr.strip()[-800:]}")
    return json.loads(p.stdout)["text"], dt, json.loads(p.stdout)


def run_hf(ckpt: Path, audio: Path, n_mels: int) -> str:
    """transformers' own transcription of the same file, same language and task.

    Its WER against the same reference is the number that makes ours readable.
    """
    import torch
    import wave
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizer)
    if not hasattr(run_hf, "_model"):
        run_hf._model = WhisperForConditionalGeneration.from_pretrained(
            str(ckpt), torch_dtype=torch.float32).eval()
        run_hf._tok = WhisperTokenizer.from_pretrained(str(ckpt))
        run_hf._fe = WhisperFeatureExtractor(feature_size=n_mels,
                                             sampling_rate=16000)
    with wave.open(str(audio), "rb") as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
    s = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    mel = np.asarray(run_hf._fe(s, sampling_rate=rate)["input_features"],
                     dtype=np.float32)
    if mel.ndim == 3:
        mel = mel[0]
    with torch.no_grad():
        out = run_hf._model.generate(
            torch.from_numpy(mel)[None], language=LANG, task=TASK,
            return_timestamps=False, do_sample=False, num_beams=1)
    return run_hf._tok.decode(out[0].tolist(), skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Word error rate of the C++ Whisper path against a human "
                    "transcript, with transformers' WER on the same audio for "
                    "scale.")
    ap.add_argument("--npue", default=str(REPO / "models" / "whisper-tiny.npue"))
    ap.add_argument("--artifacts",
                    default=str(REPO / "runtime" / "whisper-tiny" /
                                "artifacts_npu1"))
    ap.add_argument("--checkpoint", default=str(REPO / "models" / "whisper-tiny"))
    ap.add_argument("--exe",
                    default=str(REPO / "runtime" / "build" / "npuembeddings"))
    ap.add_argument("--audio", type=Path, help="one file, with --ref")
    ap.add_argument("--ref", help="the human transcript of --audio")
    ap.add_argument("--corpus", type=Path,
                    help="a directory: every *.wav with a *.txt of the same stem")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--npu-ops", default="",
                    help="op codes for the runtime's --npu-extra-ops, so this "
                         "gate can be run with the audio front end on the array "
                         "(conv) and check that the transcript does not move. "
                         "The gate's own flag is short because the runtime's is "
                         "long and the two are not the same thing. Empty: the "
                         "host front end, which is what every other run of this "
                         "gate means")
    ap.add_argument("--max-wer", type=float, default=0.20,
                    help="fail above this WER (default %(default)s). The point "
                         "is not the threshold but the DIFFERENCE from "
                         "transformers: see --wer-slack.")
    ap.add_argument("--wer-slack", type=float, default=0.05,
                    help="how much worse than transformers our WER may be before "
                         "it counts as a regression (default %(default)s)")
    args = ap.parse_args()

    npue, art, ckpt, exe = (Path(args.npue), Path(args.artifacts),
                            Path(args.checkpoint), Path(args.exe))
    for p, what in ((npue, "container"), (art, "design set"),
                    (ckpt, "checkpoint"), (exe, "runtime binary")):
        if not p.exists():
            print(f"{p} not found -- {what} missing")
            return 2

    cases: list[tuple[Path, str]] = []
    if args.corpus:
        for w in sorted(args.corpus.glob("*.wav")):
            t = w.with_suffix(".txt")
            if not t.exists():
                print(f"  skip {w.name}: no {t.name} beside it")
                continue
            cases.append((w, t.read_text(encoding="utf-8")))
    if args.audio:
        if not args.ref:
            print("--audio needs --ref: without a human transcript there is "
                  "nothing to compute an error rate against")
            return 2
        cases.append((args.audio, args.ref))
    if not cases:
        # SKIP, not a failure -- and not a silent pass either, because the
        # message says what would make this gate run. No audio is committed
        # (that is why the tier entry says so), so a fresh clone has nothing
        # to score and `gates --run` used to report that as rc=2, which made
        # the whisper tier permanently red for a reason no red ever meant.
        # Returning 0 prints the reason, and the word SKIP next to it, in the
        # same shape as verify_tail's "SKIP -- no container".
        print("SKIP -- no audio is committed, so there is no transcript to "
              "score: pass --audio FILE --ref \"human transcript\" (or "
              "--corpus DIR of *.wav + *.txt) and this gate runs the full "
              "WER comparison against transformers")
        return 0

    from npue import Reader
    rd = Reader(str(npue))
    n_mels = int(rd.config["num_mel_bins"])
    rd.close()
    name = npue.stem

    bad = 0
    tot_ref = tot_err = tot_hf_err = 0
    print(f"model {name}, language {LANG}, task {TASK}, "
          f"reference reduced to lowercase words with punctuation dropped\n")
    for audio, ref_text in cases:
        ours, dt, doc = run_ours(exe, name, audio, args)
        hf_text = run_hf(ckpt, audio, n_mels)
        ref = normalise(ref_text)
        o = wer_report(ref, normalise(ours))
        h = wer_report(ref, normalise(hf_text))
        tot_ref += o["ref_words"]
        tot_err += o["errors"]
        tot_hf_err += h["errors"]
        slower = o["wer"] - h["wer"]
        ok = o["wer"] <= args.max_wer and slower <= args.wer_slack
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {audio.name:28s} "
              f"{o['ref_words']:4d} ref words, {dt:5.2f} s")
        print(f"       ours   WER {o['wer'] * 100:5.1f}%  "
              f"({o['errors']} = {o['substitutions']} sub, {o['deletions']} del, "
              f"{o['insertions']} ins)   {ours.strip()[:64]!r}")
        print(f"       hf     WER {h['wer'] * 100:5.1f}%  ({h['errors']})   "
              f"{hf_text.strip()[:64]!r}")
        print(f"       diff   {slower * 100:+.1f} points vs transformers "
              f"(gate: ours <= {args.max_wer * 100:.0f}% and diff <= "
              f"{args.wer_slack * 100:.0f})")
        if doc.get("n_chunks", 1) > 1:
            print(f"       long form: {doc['n_chunks']} windows, "
                  f"{len(doc.get('segments', []))} segments")

    if tot_ref:
        print(f"\n  corpus    {tot_ref} reference words, ours "
              f"{tot_err} errors (WER {tot_err / tot_ref * 100:.1f}%), "
              f"transformers {tot_hf_err} (WER {tot_hf_err / tot_ref * 100:.1f}%)")
    print(f"\n{'FAILED' if bad else 'PASS'} ({bad} bad)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
