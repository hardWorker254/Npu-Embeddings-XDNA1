# NpuEmbeddings -- M3: generate golden vectors from HuggingFace.
#
# This is the ONLY place torch/transformers is allowed to run. It produces
# .safetensors files that cross into the iron env as plain data -- never as an
# import (CLAUDE.md: "Golden data crosses env boundaries as files").
#
# Two outputs, deliberately split by size:
#
#   goldens/<slug>_s64_boundary.safetensors      ~6 MB, COMMITTED
#   (<slug> is derived from the checkpoint; all-MiniLM-L6-v2 keeps its
#    historical "minilm_l6" because task logs cite the filename)
#       tokenizer output, every layer boundary (emb.ln, L*.ln1, L*.ln2),
#       last_hidden_state, the pooled and normalized sentence embedding, plus
#       the sentence-transformers embedding as an independent second oracle.
#       This is the contract M5 kernels are checked against.
#
#   goldens/<slug>_s64_taps.safetensors          ~55 MB, GITIGNORED
#       every intermediate our reference produces, including attention scores
#       and the FFN interior. Deterministic and CPU-only: regenerate with
#       --taps. Kept out of git for the same reason the parsed Perfetto traces
#       are -- it is a derivative of a sha256-pinned input.
#
# Env: .venv-ref
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens.py
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens.py --taps

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The corpus contains non-ASCII on purpose -- accents and CJK are what the
# tokenizer checks are for. Windows redirects stdout as cp1252 by default, so
# printing the token dump to a file raised UnicodeEncodeError and killed a
# 20-minute golden run at the very first print. The DATA was never at risk;
# the report about it was.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from corpus import SENTENCES, SEQ_LEN          # noqa: E402
from npue import golden_slug                   # noqa: E402
from safetensors_io import save                # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "reference" / "goldens"

# The naming rule lives in tools/npue.py, imported by every reader of these
# files. Three copies of a convention is how the three pooling
# implementations started.


def refuse_to_clobber(path, sha, force):
    """A golden belongs to exactly one checkpoint.

    The derived slug is the first line of defence and a naming convention is
    not a guard, so this is the second: if a golden already exists for a
    DIFFERENT checkpoint, stop. Same checkpoint is a regenerate and proceeds.
    """
    if not path.exists() or force:
        return
    from safetensors_io import load
    try:
        _, meta = load(path)
    except Exception:
        return  # unreadable: let the write replace it
    have = meta.get("source_sha256", "")
    if have and have != sha:
        raise SystemExit(
            f"\nREFUSING to overwrite {path.name}\n"
            f"  it holds goldens for checkpoint {have[:16]}...\n"
            f"  you are generating from        {sha[:16]}...\n"
            f"  These are different models. Pass --force only if you mean to "
            f"discard the existing goldens.")

# The layer boundaries. These are the tensors an M5 kernel has to reproduce;
# everything else is interior detail that lives in the --taps file.
def boundary_names(n_layers):
    names = ["emb.ln"]
    for i in range(n_layers):
        names += [f"L{i}.ln1", f"L{i}.ln2"]
    return names + ["last_hidden_state", "pool.mean", "out.embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--taps", action="store_true",
                    help="also write the full intermediate dump (large, gitignored)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite goldens that belong to a different checkpoint")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir)
    pin = json.loads((model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    n_layers = cfg["num_hidden_layers"]

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    enc = tok(SENTENCES, padding="max_length", max_length=SEQ_LEN,
              truncation=True, return_tensors="pt")

    # Report what the tokenizer actually did -- if a sentence silently truncated
    # or an accent survived, this is where it is visible.
    print(f"corpus: {len(SENTENCES)} sentences, padded to S={SEQ_LEN}")
    for i, s in enumerate(SENTENCES):
        ids = enc["input_ids"][i]
        n = int(enc["attention_mask"][i].sum())
        print(f"  [{i}] {n:3d} tokens  {tok.convert_ids_to_tokens(ids[:n])}")

    model = AutoModel.from_pretrained(str(model_dir)).eval()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    last_hidden = out.last_hidden_state                      # [B,S,hidden]

    # Pooling as the checkpoint specifies it -- MiniLM means, bge takes CLS.
    # Pooling a CLS model by mean produces a perfectly plausible embedding
    # that is simply wrong, so this is read rather than assumed.
    from encoder import read_pooling
    pooling = read_pooling(model_dir)
    if pooling == "cls":
        pooled = last_hidden[:, 0, :]
    else:
        # including the 1e-9 clamp on the denominator (docs/04-model).
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
    print(f"\npooling: {pooling} (from 1_Pooling/config.json)")

    # Independent second oracle: the actual sentence-transformers pipeline,
    # including its own tokenization. If our tokenizer settings are wrong this
    # disagrees even though the BertModel path above is self-consistent.
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(model_dir), device="cpu")
    st_emb = st.encode(SENTENCES, convert_to_numpy=True, normalize_embeddings=True)
    delta = float(np.abs(st_emb - embedding.numpy()).max())
    print(f"sentence-transformers vs manual {pooling}-pool: max abs diff "
          f"{delta:.3e}")
    # ASSERTED, not merely printed. sentence-transformers reads
    # 1_Pooling/config.json itself, so this number is the proof that we read it
    # the same way; leaving it as a print meant a pooling mismatch could not
    # fail the build.
    if not (delta < 2e-6):
        raise SystemExit(
            f"\nFAIL -- our {pooling} pooling disagrees with "
            f"sentence-transformers by {delta:.3e} (limit 2e-6).\n"
            f"  The two pipelines pooled the same last_hidden_state "
            f"differently, so one of them has the wrong mode.")

    meta = {
        "repo_id": pin["repo_id"],
        "source_sha256": pin["sha256"],
        "seq_len": str(SEQ_LEN),
        "batch": str(len(SENTENCES)),
        "num_layers": str(n_layers),
        "pooling": pooling,
        "sentences": json.dumps(SENTENCES, ensure_ascii=False),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "note": "fp32 CPU reference. hf.* are HuggingFace outputs; st.embedding "
                "is the sentence-transformers pipeline as an independent oracle.",
    }

    GOLDENS.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        "token_type_ids": enc["token_type_ids"].numpy().astype(np.int64),
        "hf.last_hidden_state": last_hidden.numpy(),
        "hf.pool.mean": pooled.numpy(),
        "hf.out.embedding": embedding.numpy(),
        "st.embedding": st_emb,
    }
    # hidden_states[0] is the embedding output; [i+1] is the output of layer i,
    # i.e. our L{i}.ln2. There is no HF tap for the mid-layer L{i}.ln1.
    hs = [h.numpy() for h in out.hidden_states]
    tensors["hf.emb.ln"] = hs[0]
    for i in range(n_layers):
        tensors[f"hf.L{i}.ln2"] = hs[i + 1]

    slug = golden_slug(model_dir, n_layers)
    path = GOLDENS / f"{slug}_s{SEQ_LEN}_boundary.safetensors"
    refuse_to_clobber(path, pin["sha256"], args.force)
    save(path, tensors, meta)
    print(f"\nwrote {path.relative_to(REPO)}  "
          f"({path.stat().st_size/1e6:.1f} MB, {len(tensors)} tensors)")

    if args.taps:
        # The full dump comes from OUR reference, not HF -- HF has no hook for
        # qkv/scores/gelu. It is only trustworthy because check_reference.py
        # proves the reference agrees with HF at every point HF does expose.
        from encoder import MiniLMReference
        from safetensors_io import load

        w, _ = load(model_dir / "model.safetensors")
        ref = MiniLMReference(w, num_layers=n_layers,
                              num_heads=cfg["num_attention_heads"],
                              eps=cfg["layer_norm_eps"],
                              pooling=pooling)
        taps = {}
        ref.encode(enc["input_ids"].numpy(), enc["attention_mask"].numpy(),
                   enc["token_type_ids"].numpy(), taps=taps)
        tap_meta = dict(meta)
        tap_meta["note"] = ("Full intermediate dump from reference/encoder.py. "
                            "Derivative of a sha256-pinned checkpoint; gitignored. "
                            "Regenerate: make_goldens.py --taps")
        tpath = GOLDENS / f"{slug}_s{SEQ_LEN}_taps.safetensors"
        refuse_to_clobber(tpath, pin["sha256"], args.force)
        save(tpath, taps, tap_meta)
        print(f"wrote {tpath.relative_to(REPO)}  "
              f"({tpath.stat().st_size/1e6:.1f} MB, {len(taps)} tensors)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
