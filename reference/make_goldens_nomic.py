# NpuEmbeddings -- M13 (tasks/0069, reference half): generate golden vectors
# for nomic-embed-text-v1.5 from HuggingFace, same discipline as
# make_goldens.py (M3) / make_goldens_gemma.py (C1).
#
# TWO INDEPENDENT ORACLES, gated against each other before either is trusted:
#
#   1. The NATIVE transformers.models.nomic_bert port (plain
#      AutoModel.from_pretrained(model_dir), NO trust_remote_code) --
#      code-generated from HuggingFace's own modular_nomic_bert.py, driven by
#      their own weight-conversion table. Exposes per-layer boundaries
#      cleanly via output_hidden_states=True (its `_can_record_outputs`
#      mechanism records NomicBertLayer's output at every layer -- see
#      tasks/0069-.../probe_hidden_states.py, confirmed empirically: 13
#      hidden_states for 12 layers, hs[-1] bit-identical to
#      last_hidden_state).
#
#   2. The ORIGINAL remote code (nomic-ai/nomic-bert-2048's
#      modeling_hf_nomic_bert.py, `trust_remote_code=True`), driven through
#      the real SentenceTransformer pipeline -- its own tokenization, its own
#      prompt handling, its own pooling. A genuinely different codebase from
#      (1): hand-maintained vs. code-generated, `get_extended_attention_mask`
#      vs `create_bidirectional_mask`, `NomciBertGatedMLP` vs `NomicBertMLP`.
#
# tasks/0068's probe_nomic_arch.py section 12 already found these two
# implementations bit-identical (max_abs 0.0) on a single sentence's
# last_hidden_state; this script re-proves it on the real 4-sentence,
# 64-token, prefixed corpus and GATES on it rather than merely printing it.
#
# sentence-transformers does NOT L2-normalize this checkpoint (no Normalize
# module in modules.json) -- both oracles here are compared UNNORMALIZED, and
# the boundary file stores both hf.pool.mean_raw (native) and
# st.pool.mean_raw (remote, via ST) so a reader never has to guess which
# convention either tensor uses.
#
# Env: .venv-ref
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --taps

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_nomic import SENTENCES, SEQ_LEN     # noqa: E402
from encoder_nomic import PROMPTS               # noqa: E402
from safetensors_io import save, load           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "reference" / "goldens_nomic"

# Cross-oracle gate: native (transformers.models.nomic_bert) vs remote
# (trust_remote_code=True) pooled output, on the full padded/prefixed batch.
# tasks/0068 measured 0.0 (bit-identical) on one unpadded sentence; padding +
# a batch of 4 could in principle surface a masking-implementation difference
# between the two codebases' extended-attention-mask utilities, so this is a
# real gate, not a formality.
TOL_ORACLE_AGREE = 2e-5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_to_clobber(path, sha, force):
    """A golden belongs to exactly one checkpoint. Same behaviour as
    make_goldens.py's refuse_to_clobber() (reference/make_goldens.py) --
    reimplemented here rather than imported so this file has no dependency
    on the BERT-model script (arch=0's make_goldens.py is owned by the M3
    lineage and not part of this task's file set).
    """
    if not path.exists() or force:
        return
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "nomic-embed-text-v1.5"))
    ap.add_argument("--prompt", default="search_document",
                    help="which task prefix from encoder_nomic.PROMPTS to prepend")
    ap.add_argument("--taps", action="store_true",
                    help="also write the full intermediate dump (large, gitignored)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir)
    pin = json.loads((model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    n_layers = cfg["num_hidden_layers"]

    digest = sha256(model_dir / "model.safetensors")
    if digest != pin["sha256"]:
        raise SystemExit(f"model.safetensors sha256 {digest} != CHECKPOINT.json pin "
                         f"{pin['sha256']} -- re-run fetch_model.py --model nomic-ai/nomic-embed-text-v1.5")

    prefix = PROMPTS[args.prompt]
    prefixed = [prefix + s for s in SENTENCES]

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    enc = tok(prefixed, padding="max_length", max_length=SEQ_LEN,
              truncation=True, return_tensors="pt")

    print(f"corpus: {len(SENTENCES)} sentences, prompt={args.prompt!r} "
          f"({prefix!r}), padded to S={SEQ_LEN}")
    for i, s in enumerate(prefixed):
        ids = enc["input_ids"][i]
        n = int(enc["attention_mask"][i].sum())
        print(f"  [{i}] {n:3d} tokens  {tok.convert_ids_to_tokens(ids[:n])}")
        if n >= SEQ_LEN:
            raise SystemExit(
                f"\nFAIL -- sentence {i} used all {SEQ_LEN} slots; it may have "
                f"been silently truncated. Raise SEQ_LEN in corpus_nomic.py.")

    # -- Oracle 1: NATIVE transformers.models.nomic_bert (no trust_remote_code) --
    model = AutoModel.from_pretrained(str(model_dir)).eval()
    print(f"\noracle 1 (boundary taps): {type(model).__module__}.{type(model).__name__} "
          f"(native, no trust_remote_code)")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    last_hidden = out.last_hidden_state
    hs = [h.numpy() for h in out.hidden_states]
    assert len(hs) == n_layers + 1, f"expected {n_layers + 1} hidden_states, got {len(hs)}"
    assert np.array_equal(hs[-1], last_hidden.numpy()), \
        "hs[-1] is expected to equal last_hidden_state -- nomic has no final " \
        "top-level norm (unlike Gemma's model.norm), so layer L-1's output " \
        "IS last_hidden_state. Indexing assumption broke -- re-check against " \
        "the installed transformers version."

    # Manual mean pooling (mask-weighted, UNNORMALIZED -- see file header) as
    # an independent check of the pooling formula.
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled_raw = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    # -- Oracle 2: ORIGINAL remote code, via the real SentenceTransformer
    #    pipeline (its own tokenization + prompt handling + pooling). --
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu")
    print(f"oracle 2 (end-to-end)     : SentenceTransformer(trust_remote_code=True) "
          f"-- forces the ORIGINAL remote modeling_hf_nomic_bert.py, independent of oracle 1")
    # normalize_embeddings=False: match oracle 1's unnormalized pooled_raw --
    # sentence-transformers has no Normalize module for this checkpoint
    # anyway (modules.json), so this is what st.encode() returns by default.
    st_emb = st.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)

    delta = float(np.abs(st_emb - pooled_raw.numpy()).max())
    print(f"\noracle 1 (native, manual mean-pool) vs oracle 2 (remote, via "
          f"SentenceTransformer): max abs diff {delta:.3e}")
    if not (delta < TOL_ORACLE_AGREE):
        raise SystemExit(
            f"\nFAIL -- the two independent oracles disagree by {delta:.3e} "
            f"(limit {TOL_ORACLE_AGREE:.0e}).\n"
            f"  Either the native port and the remote code have genuinely "
            f"diverged, or one of pooling/prompt/tokenization is wrong here.\n"
            f"  Do NOT proceed to trust either oracle until this is resolved.")
    print(f"PASS -- oracles agree within {TOL_ORACLE_AGREE:.0e}")

    meta = {
        "repo_id": pin["repo_id"],
        "source_sha256": pin["sha256"],
        "seq_len": str(SEQ_LEN),
        "batch": str(len(SENTENCES)),
        "num_layers": str(n_layers),
        "prompt_name": args.prompt,
        "prompt_text": prefix,
        "sentences": json.dumps(SENTENCES, ensure_ascii=False),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "note": "fp32 CPU reference for nomic-embed-text-v1.5 (tasks/0069). "
                "hf.* are the NATIVE transformers.models.nomic_bert port's "
                "outputs (no trust_remote_code); st.* is the ORIGINAL remote "
                "code (trust_remote_code=True) via the real SentenceTransformer "
                "pipeline -- two independent oracles, gated against each other "
                "before writing (see this script's header). Neither is "
                "L2-normalized (this checkpoint has no Normalize module); "
                "'pool.mean_raw' names are unnormalized on purpose. "
                "hs_layer_semantics: hs[0]=emb_ln output (before layer 0), "
                "hs[i+1] for i in 0..num_layers-1 = layer i's raw post-norm2 "
                "output; hs[-1] == last_hidden_state exactly (no separate "
                "final top-level norm, unlike Gemma's model.norm).",
    }

    GOLDENS.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        "hf.emb.ln": hs[0],
        "hf.last_hidden_state": last_hidden.numpy(),
        "hf.pool.mean_raw": pooled_raw.numpy(),
        "st.pool.mean_raw": st_emb,
    }
    for i in range(n_layers):
        tensors[f"hf.L{i}.norm2"] = hs[i + 1]

    slug = "nomic-embed-text-v1.5_l12"
    path = GOLDENS / f"{slug}_s{SEQ_LEN}_boundary.safetensors"
    refuse_to_clobber(path, digest, args.force)
    save(path, tensors, meta)
    print(f"\nwrote {path.relative_to(REPO)}  "
          f"({path.stat().st_size/1e6:.1f} MB, {len(tensors)} tensors)")

    if args.taps:
        from encoder_nomic import NomicEmbeddingReference

        w, _ = load(model_dir / "model.safetensors")
        ref = NomicEmbeddingReference(
            w, num_layers=n_layers, hidden=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"], head_dim=cfg["head_dim"],
            intermediate=cfg["intermediate_size"], eps=cfg["layer_norm_epsilon"],
            rope_theta=float(cfg.get("rope_parameters", {}).get(
                "rope_theta", cfg.get("rotary_emb_base", 1000.0))))
        taps = {}
        ref.encode(enc["input_ids"].numpy(), enc["attention_mask"].numpy(), taps=taps)
        tap_meta = dict(meta)
        tap_meta["note"] = ("Full intermediate dump from reference/encoder_nomic.py. "
                            "Derivative of a sha256-pinned checkpoint; gitignored. "
                            "Regenerate: make_goldens_nomic.py --taps")
        tpath = GOLDENS / f"{slug}_s{SEQ_LEN}_taps.safetensors"
        refuse_to_clobber(tpath, digest, args.force)
        save(tpath, taps, tap_meta)
        print(f"wrote {tpath.relative_to(REPO)}  "
              f"({tpath.stat().st_size/1e6:.1f} MB, {len(taps)} tensors)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
