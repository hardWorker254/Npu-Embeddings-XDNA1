# NpuEmbeddings -- is the C++ EmbeddingGemma tokenizer the same function as
# HuggingFace's? SPDX-License-Identifier: Apache-2.0
#
# Same discipline as tools/verify_tokenizer.py (the BERT/WordPiece checker):
# a tokenizer is a pure function from bytes to ids, so the only honest test
# is running both implementations over the same inputs and comparing every
# id. This reuses that script's adversarial corpus (accents, CJK,
# punctuation, subwords, casing, degenerate inputs, emoji -- the corpus is
# prefix-agnostic, it stresses the BPE+byte-fallback path regardless of
# which task prefix wraps it) and adds EmbeddingGemma-specific coverage:
# every task prefix in the checkpoint's own config_sentence_transformers.json
# (not just the default), plus the four reference/corpus_gemma.py sentences
# this project's numpy oracle already validated against.
#
# Env: .venv-ref (transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer_gemma.py
#   ... --cli path\to\gemma_tok_cli.exe --table path\to\gemma_tokenizer.bin

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

from verify_tokenizer import CORPUS as BERT_CORPUS  # noqa: E402 -- adversarial corpus, reused
from corpus_gemma import SENTENCES as GEMMA_SENTENCES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--extra-file", default=None,
                    help="additional corpus, one text per line")
    ap.add_argument("--cli", default=str(
        REPO / "runtime" / "build_gemma_tok" / "gemma_tok_cli.exe"))
    ap.add_argument("--table", default=str(
        REPO / "models" / "embeddinggemma-300m" / "gemma_tokenizer.bin"))
    ap.add_argument("--model-dir", default=str(
        REPO / "models" / "embeddinggemma-300m"))
    ap.add_argument("--out", default=str(REPO / "tasks" / "0061-m12-embeddinggemma-tokenizer"
                                         / "verify_tokenizer_gemma.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.model_dir)

    sbert_cfg = json.loads(
        Path(args.model_dir, "config_sentence_transformers.json").read_text(encoding="utf-8"))
    prompts: dict[str, str] = sbert_cfg["prompts"]

    corpus = list(BERT_CORPUS) + list(GEMMA_SENTENCES)
    if args.extra_file:
        for line in Path(args.extra_file).read_text(encoding="utf-8").splitlines():
            corpus.append(line)
    flat = [c.replace("\n", " ").replace("\r", " ") for c in corpus]
    n_flattened = sum(1 for a, b in zip(corpus, flat) if a != b)

    # Cases: (prefix_name_for_cpp, prefix_text_for_hf_or_None)
    # "-" tells the C++ CLI "no prefix" (see tokenizer_gemma_cli.cpp).
    # None means the raw <bos>+text+<eos> with no prefix concatenated on the
    # HF side either -- the true no-prefix control.
    cases = [("-", None), ("document", prompts["document"]), ("query", prompts["query"]),
              ("STS", prompts["STS"]), ("Clustering", prompts["Clustering"])]

    n_ok = 0
    n_total = 0
    mismatches = []

    with tempfile.TemporaryDirectory(prefix="gemmatokverify_") as td:
        inp = Path(td) / "corpus.txt"
        inp.write_text("\n".join(flat) + "\n", encoding="utf-8")

        for cpp_prefix_name, hf_prefix_text in cases:
            r = subprocess.run(
                [args.cli, args.table, str(inp), str(args.max_len), cpp_prefix_name],
                capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(f"[{cpp_prefix_name}] CLI failed ({r.returncode}):\n{r.stderr}")
                return 2
            cpp_lines = [ln for ln in r.stdout.splitlines() if ln.strip() != "" or True]
            # keep exact line count -- an empty-string input still emits a
            # (padding-only) line, so don't drop blank output lines here.
            cpp_lines = r.stdout.split("\n")
            if cpp_lines and cpp_lines[-1] == "":
                cpp_lines.pop()

            if len(cpp_lines) != len(flat):
                print(f"[{cpp_prefix_name}] FAIL -- C++ produced {len(cpp_lines)} "
                      f"lines for {len(flat)} texts")
                return 1

            for text, line in zip(flat, cpp_lines):
                full = (hf_prefix_text or "") + text
                want = hf(full, padding="max_length", truncation=True,
                          max_length=args.max_len)["input_ids"]
                got = [int(x) for x in line.split()]
                n_total += 1
                if got == list(want):
                    n_ok += 1
                else:
                    i = next((k for k in range(max(len(got), len(want)))
                              if k >= len(got) or k >= len(want) or got[k] != want[k]),
                             0)
                    mismatches.append({
                        "prefix": cpp_prefix_name,
                        "text": text[:90],
                        "first_diff_index": i,
                        "hf": [int(v) for v in want[max(0, i - 2):i + 4]],
                        "cpp": got[max(0, i - 2):i + 4],
                        "hf_tokens": hf.convert_ids_to_tokens(
                            [int(v) for v in want[max(0, i - 2):i + 4]]),
                        "cpp_tokens": hf.convert_ids_to_tokens(got[max(0, i - 2):i + 4]),
                    })

    print(f"tokenizer verification: {n_ok}/{n_total} exact "
          f"({n_ok / n_total * 100:.2f}%)  max_len {args.max_len}  "
          f"{len(flat)} texts x {len(cases)} prefixes")
    if n_flattened:
        print(f"  ({n_flattened} corpus entries had newlines flattened to spaces)")
    for m in mismatches[:12]:
        print(f"\n  MISMATCH prefix={m['prefix']!r} at token {m['first_diff_index']}: "
              f"{m['text']!r}")
        print(f"    HF  {m['hf']}  {m['hf_tokens']}")
        print(f"    C++ {m['cpp']}  {m['cpp_tokens']}")
    if len(mismatches) > 12:
        print(f"\n  ... and {len(mismatches) - 12} more")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "verification", "model": "embeddinggemma-300m",
        "max_len": args.max_len, "n_texts": len(flat), "n_prefixes": len(cases),
        "n_total": n_total, "n_exact": n_ok, "agreement": n_ok / n_total,
        "prefixes_tested": [c[0] for c in cases],
        "mismatches": mismatches,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    ok = n_ok == n_total
    print("PASS -- byte-for-byte identical to HuggingFace" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
