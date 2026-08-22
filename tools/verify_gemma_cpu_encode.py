# NpuEmbeddings -- verify the C++ CPU-only Gemma encode (arch=1) against the
# numpy reference oracle, on the SAME real sentences and the SAME prefix.
# SPDX-License-Identifier: Apache-2.0
#
# This is the actual end-to-end correctness gate for
# tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md: it does NOT trust
# anything the C++ side computed as ground truth (no device/host read-back
# used as its own oracle, matching CLAUDE.md trap 6c's spirit even though
# this is host-only) -- it runs reference/encoder_gemma.py (tasks/0055,
# validated 1-cos 1.065e-07 against real HuggingFace) independently on the
# same texts and reports 1-cos against runtime/build_gemma_encode/out.f32.
#
# Env: .venv-ref (needs `transformers` for tokenization -- the SAME real HF
# tokenizer tasks/0061 verified our C++ tokenizer_gemma.cpp byte-identical
# against, 1,925/1,925, so tokenizing here with it and not with our own C++
# tokenizer is not a second thing to trust, it is the ALREADY-TRUSTED thing).
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_gemma_cpu_encode.py \
#       --texts runtime\build_gemma_encode\texts.txt \
#       --cpu-out runtime\build_gemma_encode\out.f32 --max-len 64 --prefix document

import argparse
import sys
from pathlib import Path

# Windows console default codepage (cp1252) cannot print CJK -- same trap
# make_goldens.py and every other corpus-printing script in this repo
# already documents.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

from encoder_gemma import PROMPTS, load_reference  # noqa: E402


def tokenize_batch(model_dir, texts, max_len, prefix_name):
    """Tokenize with the real HF tokenizer -- the same one tasks/0061 proved
    our C++ tokenizer_gemma.cpp matches byte-for-byte, 1,925/1,925. Manual
    prefix concatenation (not the sentence-transformers Pooling wrapper)
    because encode_batch_np's caller wants raw input_ids/attention_mask for
    the numpy encoder, and the tokenizer's own post_processor already adds
    <bos>/<eos>, matching GemmaTokenizer::encode()'s wrapping exactly."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    prefix = PROMPTS[prefix_name] if prefix_name else ""
    prefixed = [prefix + t for t in texts]
    enc = tok(prefixed, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="np")
    return enc["input_ids"], enc["attention_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(REPO / "runtime" / "build_gemma_encode" / "texts.txt"))
    ap.add_argument("--cpu-out", default=str(REPO / "runtime" / "build_gemma_encode" / "out.f32"))
    ap.add_argument("--model-dir", default=str(REPO / "models" / "embeddinggemma-300m"))
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--prefix", default="document")
    args = ap.parse_args()

    texts = Path(args.texts).read_text(encoding="utf-8").splitlines()
    texts = [t for t in texts if t != ""]
    print(f"{len(texts)} texts from {args.texts}")

    ref = load_reference(args.model_dir)
    input_ids, attention_mask = tokenize_batch(args.model_dir, texts, args.max_len, args.prefix)
    ref_out = ref.encode(input_ids, attention_mask)   # [n, hidden]

    n, hidden = ref_out.shape
    cpu_bytes = Path(args.cpu_out).read_bytes()
    cpu_out = np.frombuffer(cpu_bytes, dtype=np.float32).reshape(-1, hidden)
    if cpu_out.shape[0] != n:
        print(f"FAIL: cpu output has {cpu_out.shape[0]} rows, expected {n}")
        return 2

    worst = 0.0
    for i in range(n):
        a, b = ref_out[i].astype(np.float64), cpu_out[i].astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        one_minus_cos = 1.0 - cos
        worst = max(worst, one_minus_cos)
        rel_fro = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        print(f"  [{i}] 1-cos={one_minus_cos:.6e}  rel_fro={rel_fro:.6e}  "
              f"\"{texts[i][:50]}\"")

    print(f"\nworst 1-cos across {n} sentences: {worst:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
