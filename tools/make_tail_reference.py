# NpuEmbeddings -- fp32 reference vectors for the tail gate (T51 step 3).
# SPDX-License-Identifier: Apache-2.0
#
# Every accuracy instrument this project owned before T51 reported a central
# tendency, and bge-large's 100x max/median tail on single-word inputs sailed
# through all of them (tasks/0122, research/OPEN-THREADS.md#t51). The gate that
# can see a tail (tools/verify_tail.py) needs a reference that CONTAINS one --
# a corpus heavy on the degenerate single-word inputs where the tail lives --
# and that reference has to cross the env boundary as FILES (CLAUDE.md), because
# the gate is stdlib-only and this generator needs torch.
#
# The corpus is tasks/0129's byte for byte (which is tasks/0122's byte for
# byte): 180 single words drawn deterministically from bge-base's vocab, the 36
# semantic-corpus sentences, 8 five-word phrases. One shared corpus for all six
# models, so a cross-model comparison is a comparison of models, not corpora --
# and so the baselines stay comparable with 0122/0129's published numbers.
#
# Per-model reference generation mirrors what already exists rather than
# inventing a fourth reference lineage:
#   * bge-* / MiniLM: transformers AutoModel with the pooling the checkpoint's
#     own 1_Pooling/config.json declares (CLS for bge, mean for MiniLM) --
#     exactly tasks/0129's loop, which tasks/0122 cross-checked against
#     sentence-transformers to four significant figures.
#   * nomic: sentence-transformers with trust_remote_code, the container's
#     prompt_default prefix prepended by hand -- verify_embed_e2e.py's
#     approach, verbatim (ST has no prefix flag; the runtime side gets
#     --prefix and does its own prepending).
#   * embeddinggemma: sentence-transformers with prompt_name="document" --
#     the prompt every existing gemma harness pins (tasks/0118: no default
#     exists any more, so the choice is explicit here and recorded in the
#     JSON the gate reads back).
#
# Output, per model:
#   reference/tail/<model>.f32   float32 unit vectors, row-major
#   reference/tail/<model>.json  the texts, group boundaries, pooling, prompt,
#                                versions, and a `baseline` slot that
#                                verify_tail.py --write-baseline fills in
#
# Env: .venv-ref. Usage:
#   .venv-ref\Scripts\python tools\make_tail_reference.py
#   ... --models bge-large-en-v1.5

from __future__ import annotations

import argparse
import datetime
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
SEQ = 64
OUT = REPO / "reference" / "tail"

BUILTIN = [
    "all-MiniLM-L6-v2",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "nomic-embed-text-v1.5",
    "embeddinggemma-300m",
]

# tools/npue.py's header, re-declared like verify_semantics.py does, so this
# file depends on nothing in tools/ -- the prompt text MUST come from the same
# .npue the runtime will run, not from a literal (verify_embed_e2e.py's
# argument: if the two sides disagreed about WHAT the prefix is, the
# comparison would pass with both wrong the same way).
_HEADER = "<4sIII QQQQ 16s"


def container_config(path: Path) -> dict:
    with open(path, "rb") as f:
        magic, _v, _a, _f, jo, jl, _do, _dl, _r = struct.unpack(
            _HEADER, f.read(struct.calcsize(_HEADER)))
        if magic != b"NPUE":
            raise SystemExit(f"{path}: not a .npue file (magic {magic!r})")
        f.seek(jo)
        return dict(json.loads(f.read(jl).decode("utf-8"))["config"])


def build_corpus() -> tuple[list[str], dict, int]:
    """tasks/0129's corpus, byte for byte. Deterministic: same checkpoint,
    same filter, same sort, same stride -> same 180 words on every machine."""
    from transformers import AutoTokenizer
    tok0 = AutoTokenizer.from_pretrained(str(REPO / "models" /
                                             "bge-base-en-v1.5"))
    vocab = [w for w in tok0.get_vocab()
             if w.isalpha() and len(w) >= 4 and w.islower()]
    vocab.sort()
    words = vocab[::max(1, len(vocab) // 180)][:180]
    corpus = json.loads((REPO / "tools" / "semantic_corpus.json")
                        .read_text(encoding="utf-8"))
    # English sentences ONLY, even after the corpus grew multilingual
    # clusters (tasks/0137): this corpus is 0129's byte for byte, and the
    # cross-model tail comparison (including gte's) is only a comparison of
    # models while every model sees the same texts.
    sents = [s["text"] for s in corpus["sentences"]
             if s.get("lang", "en") == "en"]
    phrases = [" ".join(words[i:i + 5]) for i in range(0, 40, 5)]
    texts = words + sents + phrases
    groups = {"words": [0, len(words)],
              "sentences": [len(words), len(words) + len(sents)],
              "phrases": [len(words) + len(sents), len(texts)]}
    return texts, groups, int(corpus["version"])


def pooling_of(model: str) -> str:
    """Read the checkpoint's own declaration, never guess from the name."""
    cfg = json.loads((REPO / "models" / model / "1_Pooling" / "config.json")
                     .read_text(encoding="utf-8"))
    if cfg.get("pooling_mode_cls_token"):
        return "cls"
    if cfg.get("pooling_mode_mean_tokens"):
        return "mean"
    raise SystemExit(f"{model}: 1_Pooling declares neither CLS nor mean -- "
                     f"refusing to guess")


def ref_automodel(model: str, texts: list[str], pooling: str) -> np.ndarray:
    """tasks/0129's reference loop, with the pooling made explicit."""
    from transformers import AutoModel, AutoTokenizer
    d = REPO / "models" / model
    tk = AutoTokenizer.from_pretrained(str(d))
    md = AutoModel.from_pretrained(str(d)).eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tk(texts[i:i + 64], padding=True, truncation=True,
                 max_length=SEQ, return_tensors="pt")
        with torch.no_grad():
            h = md(**enc).last_hidden_state.double()
        if pooling == "cls":
            c = h[:, 0].numpy()
        else:
            m = enc["attention_mask"].unsqueeze(-1).double()
            c = ((h * m).sum(dim=1) / m.sum(dim=1)).numpy()
        out.append(c / np.linalg.norm(c, axis=1, keepdims=True))
    return np.concatenate(out)


def ref_gte(model: str, texts: list[str], pooling: str) -> np.ndarray:
    """arch=3 (gte-multilingual-base): the 0129 AutoModel loop, but the
    checkpoint's remote code needs trust_remote_code AND the two 0134/0136
    reference repairs, or this fp32 reference is silently wrong:
      L1. transformers v5 materialises the persistent=False rotary buffers
          and embeddings.position_ids as uninitialised meta-device memory --
          repair_rotary() (reference/make_goldens_gte.py) rebuilds both.
      L2. the checkpoint config says torch_dtype: float16 and v5 honours
          it -- load fp32 explicitly."""
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(REPO / "reference"))
    from make_goldens_gte import repair_rotary
    d = REPO / "models" / model
    tk = AutoTokenizer.from_pretrained(str(d))
    md = AutoModel.from_pretrained(str(d), trust_remote_code=True,
                                   torch_dtype=torch.float32).eval()
    repair_rotary(md)
    assert next(md.parameters()).dtype is torch.float32, "L2 is back"
    if pooling != "cls":
        raise SystemExit(f"{model}: expected CLS pooling, checkpoint "
                         f"declares {pooling!r}")
    out = []
    for i in range(0, len(texts), 64):
        enc = tk(texts[i:i + 64], padding=True, truncation=True,
                 max_length=SEQ, return_tensors="pt")
        with torch.no_grad():
            h = md(**enc).last_hidden_state.double()
        c = h[:, 0].numpy()
        out.append(c / np.linalg.norm(c, axis=1, keepdims=True))
    return np.concatenate(out)


def ref_st(model: str, texts: list[str], trust_remote_code: bool,
           prompt_name: str | None) -> np.ndarray:
    """verify_embed_e2e.py's reference path, for the two non-BERT archs."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(REPO / "models" / model),
                             trust_remote_code=trust_remote_code, device="cpu")
    st.max_seq_length = SEQ
    kw = {"prompt_name": prompt_name} if prompt_name else {}
    v = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                  batch_size=32, **kw).astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=BUILTIN)
    args = ap.parse_args()

    import transformers
    versions = {"transformers": transformers.__version__,
                "torch": torch.__version__}
    try:
        import sentence_transformers
        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass

    texts, groups, corpus_ver = build_corpus()
    print(f"corpus: {len(texts)} texts "
          f"({groups['words'][1]} single words, "
          f"{groups['sentences'][1] - groups['sentences'][0]} sentences, "
          f"{groups['phrases'][1] - groups['phrases'][0]} phrases)")
    OUT.mkdir(parents=True, exist_ok=True)

    failed = []
    for model in args.models:
        ckpt = REPO / "models" / model
        if not (ckpt / "config.json").exists():
            print(f"  {model:<24} SKIP -- no local checkpoint at {ckpt}")
            failed.append(model)
            continue
        cfg = container_config(REPO / "models" / f"{model}.npue")
        hidden = int(cfg["hidden"])
        arch = cfg.get("arch", "")
        prompts = cfg.get("prompts") or {}

        # Which prompt, and where its text comes from. BOTH are recorded in
        # the JSON so verify_tail.py passes the same name to the runtime that
        # this reference actually baked in.
        prompt_name = prompt_prefix = None
        if arch == "nomic_bert_rope_swiglu":
            prompt_name = cfg["prompt_default"]        # "search_document"
            prompt_prefix = prompts[prompt_name]
        elif arch == "gemma3_mqa_rope_geglu":
            prompt_name = "document"                   # pinned, tasks/0118
            prompt_prefix = prompts[prompt_name]
        elif prompts:
            raise SystemExit(f"{model}: has a prompts table but this "
                             f"generator names no prompt for its arch "
                             f"{arch!r} -- refusing to pick one")

        pooling = pooling_of(model)
        print(f"  {model:<24} hidden {hidden:>4}  pooling {pooling:<4} "
              f"prompt {prompt_name!r} ...", flush=True)
        try:
            if arch == "nomic_bert_rope_swiglu":
                # ST prepends nothing for nomic (no config_sentence_
                # transformers.json in that checkpoint) -- prefix by hand,
                # exactly as verify_embed_e2e.py does for its reference side.
                ref = ref_st(model, [prompt_prefix + t for t in texts],
                             trust_remote_code=True, prompt_name=None)
            elif arch == "gemma3_mqa_rope_geglu":
                # ST's own prompts table carries "document" -> the same
                # prefix string the container does; prompt_name applies it.
                ref = ref_st(model, texts, trust_remote_code=False,
                             prompt_name=prompt_name)
            elif arch == "gte_new_rope_geglu":
                ref = ref_gte(model, texts, pooling)
            else:
                ref = ref_automodel(model, texts, pooling)
        except Exception as e:  # noqa: BLE001 -- record honestly, keep going
            print(f"  {model:<24} FAILED to build reference: {e}")
            failed.append(model)
            continue

        assert ref.shape == (len(texts), hidden), ref.shape
        f32 = ref.astype(np.float32)
        (OUT / f"{model}.f32").write_bytes(f32.tobytes())
        meta = {
            "kind": "fp32 reference for the tail gate (T51 step 3)",
            "model": model,
            "hidden": hidden,
            "pooling": pooling,
            "seq": SEQ,
            "prompt_name": prompt_name,
            "prompt_prefix": prompt_prefix,
            "reference_impl": ("sentence-transformers"
                               if arch in ("nomic_bert_rope_swiglu",
                                           "gemma3_mqa_rope_geglu")
                               else "transformers AutoModel"),
            "versions": versions,
            "corpus": {
                "n": len(texts),
                "groups": groups,
                "words_drawn_from": ("bge-base-en-v1.5 vocab: isalpha, "
                                     "len>=4, islower, sorted, stride "
                                     "len//180 -- tasks/0129 byte for byte"),
                "semantic_corpus_version": corpus_ver,
            },
            "texts": texts,
            "generated": datetime.date.today().isoformat(),
            "command": (".venv-ref\\Scripts\\python "
                        "tools\\make_tail_reference.py"),
            # Filled in by `verify_tail.py --write-baseline`, never here:
            # the ceiling must come from a MEASUREMENT of the runtime, and
            # this file never runs the runtime.
            "baseline": None,
        }
        (OUT / f"{model}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {model:<24} wrote {model}.f32 "
              f"({f32.nbytes:,} B) + {model}.json")

    if failed:
        print(f"\nNOT generated (recorded, not hidden): {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
