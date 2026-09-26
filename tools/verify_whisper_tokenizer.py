#!/usr/bin/env python3
# NpuEmbeddings -- hold the C++ Whisper tokenizer against the reference and
# against HuggingFace.
# SPDX-License-Identifier: Apache-2.0
#
# THREE IMPLEMENTATIONS, ONE TABLE
# -------------------------------
#   C++       runtime/src/tokenizers/whisper.cpp, the one that ships
#   reference tools/whisper_tokenizer_ref.py, the executable specification
#   HF        transformers' own WhisperTokenizer over the same checkpoint
#
# The reference exists so the C++ has something to be written against, and the
# reference is only worth anything if it is itself held against HuggingFace --
# which is why all three run here rather than two.
#
# WHY THE GATE IS EXACT
# ---------------------
# A transcription is text. There is no tolerance to spend: a token id that is
# off by one is a different word, and the failure mode is a fluent, confident,
# wrong transcript rather than an error. So the gate compares id lists
# byte for byte and the decoded string byte for byte, and prints the first
# divergence in full rather than a count.
#
# THE CORPUS IS ADVERSARIAL ON PURPOSE
# ------------------------------------
# Every case below exists because it is a place the GPT-2 pre-tokeniser is
# known to be subtle, and each one is annotated with what it is testing. An
# English sentence proves nothing here: it is the case where every
# implementation agrees.
#
# Env: numpy + regex for the reference, transformers for HuggingFace, a C++
# toolchain for the binary. The binary is built if --exe is not given.
#
# Usage:
#   python tools/verify_whisper_tokenizer.py
#   python tools/verify_whisper_tokenizer.py --exe /tmp/test_whisper_tokenizer

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

# (text, what it is testing)
CORPUS = [
    ("hello world", "the trivial case"),
    ("Hello, world!", "punctuation run, then a space"),
    ("Привет мир", "Cyrillic: one letter-run pretoken, not five"),
    ("Привет, мир! Как дела?", "Cyrillic with punctuation and a space"),
    ("你好世界", "CJK: every char is a letter, no spaces at all"),
    ("こんにちは世界", "Japanese, including kana"),
    ("안녕하세요 세계", "Hangul"),
    ("🎤🎧 emoji test", "astral plane: 4-byte UTF-8 sequences"),
    ("👨‍👩‍👧‍👦 family", "ZWJ sequence: many code points, zero letters"),
    ("1234567890", "a digit run longer than any pretokeniser likes"),
    ("42", "short digit run"),
    ("3.14159", "digits around a full stop"),
    ("a1b2c3", "letters and digits ALTERNATING, no separator"),
    ("line1\nline2", "a bare newline between letters"),
    ("line1\r\nline2", "CRLF: the punctuation branch takes [\\r\\n]*"),
    ("end.\n\n\nnext", "several newlines, and one after a full stop"),
    ("trailing space ", "a space at end of input: the (?!\\S) branch"),
    ("two  spaces", "two spaces before a word: one is given back"),
    ("tab\there", "a tab is \\s but NOT the literal space of ` ?`"),
    ("it's", "contraction: 's"),
    ("they're we've I'm we'll he'd", "every contraction in the pattern"),
    ("'sup 'tis ''", "an apostrophe that starts no contraction"),
    ("a" * 200, "a long run: the BPE loop's worst case"),
    ("ab" * 100, "alternating, so merges keep firing"),
    ("1234567890" * 10, "a long digit run"),
    ("mixed Привет 123 🎤 end.", "all four classes in one line"),
    ("\x85\xa0", "U+0085 and U+00A0: \\s, but not ASCII space"),
    ("　全角", "U+3000 ideographic space: \\s, and not the ` ?` space"),
    ("", "the empty string"),
]

# The control tokens a transcription cannot start or end without. Their ids are
# not invented: the packer carries them and the C++ looks them up by name.
CONTROL = ("<|startoftranscript|>", "<|notimestamps|>", "<|transcribe|>",
           "<|translate|>", "<|endoftext|>")


def build_exe(path):
    """Compile the tokenizer plus the container reader into one binary."""
    cmd = [
        "g++", "-std=c++17", "-O1", "-I", str(REPO / "runtime" / "include"),
        str(REPO / "runtime" / "tests" / "test_whisper_tokenizer.cpp"),
        str(REPO / "runtime" / "src" / "tokenizers" / "whisper.cpp"),
        str(REPO / "runtime" / "src" / "model.cpp"),
        "-o", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("building the C++ tokenizer failed:\n" + r.stderr)
        return None
    return path


def run_cpp(exe, npue, corpus):
    """ids and decoded text per corpus entry.

    The corpus contains newlines and CRLF on purpose, so the framing is hex --
    one entry per line. Line-framed plain text would split those entries and
    silently compare every case after one against the wrong input, which is
    what a line-framed protocol did here before the gate noticed.
    """
    framed = "\n".join(t.encode("utf-8").hex() for t in corpus) + "\n"
    p = subprocess.run([str(exe), str(npue)], input=framed,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"{exe} failed:\n{p.stderr}")
        return None, None
    ids, back, controls = [], [], {}
    for line in p.stdout.splitlines():
        parts = line.split()
        ids.append([int(x) for x in parts[1:]] if parts else [])
    for line in p.stderr.splitlines():
        if line.startswith("back "):
            back.append(bytes.fromhex(line[5:]).decode("utf-8", "replace"))
        elif line.startswith("control "):
            name, _, rest = line.partition("id ")
            controls[name[len("control "):].strip()] = int(rest)
    return ids, back, controls


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the C++ Whisper tokenizer against the reference "
                    "and against HuggingFace.")
    ap.add_argument("--model-dir", default=str(REPO / "models" / "whisper-tiny"))
    ap.add_argument("--npue", default=str(REPO / "models" / "whisper-tiny.npue"))
    ap.add_argument("--exe", default=None, help="prebuilt test binary")
    ap.add_argument("--no-hf", action="store_true",
                    help="skip HuggingFace (the reference check still runs)")
    args = ap.parse_args()

    from whisper_tokenizer_ref import WhisperTokenizerRef

    tmp = None
    exe = args.exe
    if exe is None:
        tmp = tempfile.NamedTemporaryFile(suffix="_whisper_tok", delete=False)
        tmp.close()
        exe = build_exe(tmp.name)
        if exe is None:
            return 2

    npue = Path(args.npue)
    if not npue.exists():
        print(f"{npue} not found -- pack it first "
              f"(python tools/pack_npue.py --model-dir {args.model_dir})")
        return 2

    corpus = [t for t, _ in CORPUS]
    cpp_ids, cpp_back, controls = run_cpp(exe, npue, corpus)
    if cpp_ids is None:
        return 2

    ref = WhisperTokenizerRef.from_npz(npue)
    ref_ids = [ref.encode(t) for t in corpus]

    bad = 0
    print(f"C++   vocab {len(ref.tokens)} merges {ref.t.n_merges}, "
          f"{len(CORPUS)} cases\n")

    for i, (text, why) in enumerate(CORPUS):
        problems = []
        if cpp_ids[i] != ref_ids[i]:
            problems.append(f"ids differ from the reference:\n"
                            f"        C++ {cpp_ids[i]}\n"
                            f"        ref {ref_ids[i]}")
        # The round trip is exact for every case except the empty string, where
        # there is nothing to round-trip, and the astral cases, where the
        # decoded bytes are the input's bytes and so must match too.
        if text and cpp_back[i] != text:
            problems.append(f"round trip differs:\n"
                            f"        in  {text!r}\n"
                            f"        out {cpp_back[i]!r}")
        if problems:
            bad += 1
            print(f"  FAIL [{i:2d}] {why}\n    text {text!r}")
            for p in problems:
                print("      " + p)
        else:
            print(f"  ok   [{i:2d}] {why:52s} {len(cpp_ids[i]):4d} ids")

    if not args.no_hf:
        try:
            from transformers import WhisperTokenizerFast
            hf = WhisperTokenizerFast.from_pretrained(args.model_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  HuggingFace unavailable ({exc}); "
                  f"the C++/reference comparison above still stands")
        else:
            hf_bad = 0
            for i, (text, why) in enumerate(CORPUS):
                want = hf(text, add_special_tokens=False)["input_ids"]
                if want != cpp_ids[i]:
                    hf_bad += 1
                    bad += 1
                    print(f"  FAIL [{i:2d}] ids differ from HuggingFace "
                          f"({why})\n    text {text!r}\n"
                          f"        C++ {cpp_ids[i]}\n        HF  {want}")
            print(f"\n  HuggingFace: {len(CORPUS) - hf_bad}/{len(CORPUS)} exact")

    missing = [t for t in CONTROL if t not in controls]
    if missing:
        bad += 1
        print(f"  FAIL control tokens not found in the container: {missing}")
    else:
        print("  control tokens present: " +
              ", ".join(f"{t}={controls[t]}" for t in CONTROL))

    if tmp is not None:
        Path(tmp.name).unlink(missing_ok=True)

    print(f"\n{'FAILED' if bad else 'PASS'} "
          f"({len(CORPUS)} cases, {bad} bad)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
