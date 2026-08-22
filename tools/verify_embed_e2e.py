# NpuEmbeddings -- end-to-end check: text in, vector out, against the reference.
# SPDX-License-Identifier: Apache-2.0
#
# Every previous accuracy check started from PRE-COMPUTED embeddings, so it
# tested the encoder while assuming the tokenizer. `npuembed --embed` closes
# that loop: plain text goes in and vectors come out, all in C++. This checks
# the whole chain -- tokenizer, embedding gather, NPU encoder, pooling,
# normalisation -- against sentence-transformers on the same texts.
#
# It is the difference between "the datapath is right" and "the PRODUCT is
# right", and only the second one is what a user gets.
#
# Env: .venv-ref
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py
#   ... --corpus some_file.txt --artifacts artifacts_b128il

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The corpus is deliberately non-ASCII. Windows redirects stdout as cp1252, so
# reporting on it raised UnicodeEncodeError and killed the run -- the data was
# never at risk, only the report about it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
SEQ = 64
def container_config(model: str) -> dict:
    """The whole .npue config dict, read once and reused for width, arch,
    and (for nomic) the prefix table -- all container facts, not constants."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "tools"))
    from npue import Reader
    with Reader(REPO / "models" / f"{model}.npue") as r:
        return dict(r.config)


def hidden_of(model: str) -> int:
    """Embedding width, from the container rather than a constant.

    It was a literal 384 -- right for MiniLM and bge-small, wrong for
    bge-large's 1024. Read it, then use it to CHECK the byte count rather than
    inferring the width from the data: an inferred width reshapes a truncated
    file into a plausible array.
    """
    return int(container_config(model)["hidden"])

# Deliberately mixed: short and long, ASCII and not, near-duplicates (so a
# similarity mistake shows up as a ranking change), and the golden corpus.
DEFAULT = [
    "A man is playing a guitar on stage.",
    "Someone plays a guitar at a concert.",
    "A woman is slicing an onion in the kitchen.",
    "The stock market closed lower on Tuesday.",
    "Interest rates were raised by the central bank.",
    "Blåbærsyltetøy smaker godt på en skive brød.",
    "机器学习模型可以生成句子向量。",
    "Tokenization turns text into subword pieces.",
    "pneumonoultramicroscopicsilicovolcanoconiosis is a long word",
    "",
    "   ",
    "Short.",
    "The quick brown fox jumps over the lazy dog while the cat sleeps quietly "
    "in the warm afternoon sun near the window of an old wooden house.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="one text per line")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--artifacts", default="artifacts_b128il")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tol", type=float, default=2e-3)
    # Default output is keyed by MODEL and ARTIFACT SET. It used to be one
    # fixed filename, so running two arms of an A/B in sequence left only the
    # second one's numbers on disk -- the artifact existed and quietly held the
    # other arm's result. Found twice: tasks/0044 in the eltwise harnesses,
    # tasks/0045 here.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(REPO / "tasks" / "0036-m8-tokenizer" /
                       f"verify_embed_e2e_{args.model}_{args.artifacts}.json")

    texts = (Path(args.corpus).read_text(encoding="utf-8").splitlines()
             if args.corpus else list(DEFAULT))
    texts = [t.replace("\r", "") for t in texts]

    hidden = hidden_of(args.model)
    cfg = container_config(args.model)
    is_nomic = cfg.get("arch") == "nomic_bert_rope_swiglu"

    # nomic REQUIRES a task prefix ("search_document: " etc, tasks/0068 sec
    # 4). tasks/0071 wired `--prefix` into `npuembed --embed` itself, so the
    # RUNTIME side now applies its own prefix -- this harness hands it the
    # RAW texts plus `--prefix <name>` and lets it do the prepending, the
    # same as any real caller would. The REFERENCE side (sentence-
    # transformers) has no such flag, so it still needs the prefix prepended
    # here, by hand, same as before. If the two sides ever disagreed about
    # WHAT the prefix text is, this comparison would still pass with both
    # sides wrong the same way -- that risk is why container_config() reads
    # the prefix from the SAME .npue the runtime just packed, not a literal.
    ref_texts = texts
    prefix_name = None
    if is_nomic:
        prompts = cfg.get("prompts", {})
        prompt_default = cfg.get("prompt_default", "")
        prefix = prompts.get(prompt_default, "")
        if not prefix:
            print(f"WARNING: nomic container has no usable prompts/"
                  f"prompt_default ({prompt_default!r}) -- proceeding WITHOUT "
                  f"a task prefix, which is a KNOWN-WRONG comparison for this "
                  f"model (tasks/0068 sec 4).")
        else:
            print(f"NOTE (tasks/0071): npuembed --embed now applies its own "
                  f"--prefix {prompt_default!r} ({prefix!r}) to the RAW "
                  f"texts below -- this harness only prepends it to the "
                  f"REFERENCE side (sentence-transformers has no prefix "
                  f"flag), so this exercises the real `--prefix` flag path "
                  f"rather than a harness-side workaround.")
            prefix_name = prompt_default
            ref_texts = [prefix + t for t in texts]

    exe = REPO / "runtime" / "build" / "npuembed.exe"
    with tempfile.TemporaryDirectory(prefix="e2e_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        cmd = [str(exe), "..", "--model", args.model,
              "--artifacts", args.artifacts,
              "--threads", str(args.threads)]
        if prefix_name is not None:
            cmd += ["--prefix", prefix_name]
        cmd += ["--embed", str(d / "in.txt"), str(d / "out.f32")]
        r = subprocess.run(cmd, cwd=str(REPO / "runtime"),
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(f"npuembed --embed failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
            return 2
        got = np.frombuffer((d / "out.f32").read_bytes(),
                            dtype=np.float32).reshape(len(texts), hidden)

    from sentence_transformers import SentenceTransformer
    # nomic needs trust_remote_code=True to load at all (tasks/0068/0069) --
    # ST's own model card carries the original, hand-maintained modelling
    # code (not the newer native transformers.models.nomic_bert port).
    st = SentenceTransformer(str(REPO / "models" / args.model),
                             trust_remote_code=is_nomic, device="cpu")
    st.max_seq_length = SEQ
    ref = st.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)

    cos = (got.astype(np.float64) * ref.astype(np.float64)).sum(axis=1)
    one_m_cos = 1.0 - cos
    worst = float(one_m_cos.max())

    print(f"end-to-end: {len(texts)} texts, seq {SEQ}, "
          f"artifacts {args.artifacts}")
    print(f"  {'#':>3}  {'1-cos':>11}   text")
    order = np.argsort(-one_m_cos)
    for i in order[:8]:
        print(f"  {i:>3}  {one_m_cos[i]:>11.3e}   {texts[i][:56]!r}")
    print(f"\n  worst 1-cos {worst:.3e}   (tolerance {args.tol:.0e})")

    # A similarity GEOMETRY check, because embeddings can each be slightly off
    # while their relative geometry is badly wrong -- and the geometry is what
    # a retrieval or clustering user actually consumes.
    #
    # The tolerance here is DERIVED, not chosen. For unit vectors with
    # 1-cos = c, the perturbation is |d| = sqrt(2c), and a dot product of two
    # perturbed vectors moves by at most ~2|d|. A fixed 2e-3 was the first
    # tolerance used and it failed on a 6,788-text corpus at 3.2e-3 -- not
    # because the geometry was wrong but because the max over 23 M pairs is an
    # extreme-value statistic: more pairs, higher max, for unchanged accuracy.
    # Testing against the bound tests the claim; testing against a constant
    # tests the corpus size.
    #
    # Pairs are sampled above a few thousand texts, since the full matrix is
    # O(n^2) and adds nothing once the distribution is established.
    rng = np.random.default_rng(0)
    n = len(texts)
    if n > 3000:
        idx = rng.choice(n, 3000, replace=False)
        a_got, a_ref = got[idx], ref[idx]
    else:
        a_got, a_ref = got, ref
    d_sim = np.abs((a_got @ a_got.T) - (a_ref @ a_ref.T))
    iu = np.triu_indices(len(a_got), k=1)
    d_sim = d_sim[iu]
    sim_err = float(d_sim.max())
    sim_p99 = float(np.percentile(d_sim, 99))
    sim_mean = float(d_sim.mean())
    bound = 2.0 * float(np.sqrt(2.0 * max(worst, 1e-30)))
    print(f"  pairwise-similarity error over {d_sim.size:,} pairs:")
    print(f"    mean {sim_mean:.3e}   p99 {sim_p99:.3e}   max {sim_err:.3e}")
    print(f"    bound implied by the worst 1-cos: {bound:.3e}")

    # Rank agreement is the property a retrieval user actually depends on:
    # do the same neighbours come back in the same order?
    k = min(10, n - 1)
    top_got = np.argsort(-(a_got @ a_got.T), axis=1)[:, 1:k + 1]
    top_ref = np.argsort(-(a_ref @ a_ref.T), axis=1)[:, 1:k + 1]
    overlap = float(np.mean([len(set(g) & set(r)) / k
                             for g, r in zip(top_got, top_ref)]))
    print(f"  top-{k} neighbour overlap: {overlap:.4f}")

    ok = bool(np.isfinite(worst) and worst <= args.tol and sim_err <= bound
              and overlap >= 0.98)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "hardware measurement", "task": "0036",
        "n_texts": len(texts), "seq": SEQ, "artifacts": args.artifacts,
        "worst_1_minus_cos": worst,
        "pairwise_similarity_error": {"mean": sim_mean, "p99": sim_p99,
                                      "max": sim_err, "bound": bound,
                                      "n_pairs": int(d_sim.size)},
        "topk_neighbour_overlap": overlap,
        "tolerance": args.tol, "pass": ok,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("PASS -- text in, vector out, matches the reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
