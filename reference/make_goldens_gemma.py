# NpuEmbeddings -- C1 spike (tasks/0055): generate golden vectors for
# EmbeddingGemma-300M from HuggingFace, same discipline as make_goldens.py.
#
# Two oracles, same as M3:
#   1. Gemma3TextModel (AutoModel) with output_hidden_states=True -- exposes
#      the per-layer boundary tensors.
#   2. The real SentenceTransformer pipeline (Transformer -> Pooling ->
#      Dense -> Dense -> Normalize) -- the end-to-end ground truth, including
#      its own tokenization and task-prefix handling.
#
# hidden_states indexing (verified empirically, tasks/0055 TASK.md): with 24
# layers, output_hidden_states gives 25 entries. hs[0] is the scaled
# embedding (before layer 0). hs[i+1] for i in 0..22 is the RAW output of
# layer i (our L{i}.resid2 tap, BEFORE the final model.norm). hs[24]
# (the very last entry) is bit-identical to last_hidden_state, i.e. it is
# POST the final norm, not layer 23's raw output -- so layer 23's raw resid2
# has no direct HF tap and is validated only transitively (same situation as
# encoder.py's L{i}.ln1 for BERT). Do not assume hs[i+1] == raw layer i output
# for i == num_layers - 1.
#
# Env: .venv-ref
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_gemma.py
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_gemma.py --taps

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_gemma import SENTENCES, SEQ_LEN     # noqa: E402
from encoder_gemma import PROMPTS               # noqa: E402
from safetensors_io import save                 # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "reference" / "goldens_gemma"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "embeddinggemma-300m"))
    ap.add_argument("--prompt", default="document",
                    help="which task prefix from encoder_gemma.PROMPTS to prepend")
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

    model = AutoModel.from_pretrained(str(model_dir)).eval()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    last_hidden = out.last_hidden_state
    hs = [h.numpy() for h in out.hidden_states]
    assert len(hs) == n_layers + 1, f"expected {n_layers + 1} hidden_states, got {len(hs)}"
    assert np.array_equal(hs[-1], last_hidden.numpy()), \
        "hs[-1] is expected to equal last_hidden_state (post final norm) -- " \
        "indexing assumption broke, re-check against the installed transformers version"

    # Mean pooling (include_prompt=true -> plain masked mean, matches
    # 1_Pooling/config.json) done by hand here as an independent check of the
    # Pooling module, same pattern as make_goldens.py.
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    from safetensors_io import load
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")
    dense2 = pooled @ torch.from_numpy(d2["linear.weight"]).T
    dense3 = dense2 @ torch.from_numpy(d3["linear.weight"]).T
    manual_embedding = torch.nn.functional.normalize(dense3, p=2, dim=1)

    # Independent second oracle: the real sentence-transformers pipeline,
    # including ITS tokenization and prompt handling (it applies the same
    # "document" prompt from config_sentence_transformers.json by name).
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(model_dir), device="cpu")
    # Prefix by hand (same `prefix` string used above) rather than via ST's
    # prompt_name= lookup, so both oracles are proven to agree on the EXACT
    # prefix text, not merely on ST's internal name-to-prefix mapping.
    st_emb = st.encode([prefix + s for s in SENTENCES], convert_to_numpy=True,
                       normalize_embeddings=True)

    delta = float(np.abs(st_emb - manual_embedding.numpy()).max())
    print(f"sentence-transformers vs manual mean-pool+dense+normalize: "
          f"max abs diff {delta:.3e}")
    if not (delta < 2e-5):
        raise SystemExit(
            f"\nFAIL -- our mean-pool+Dense+normalize disagrees with "
            f"sentence-transformers by {delta:.3e} (limit 2e-5).\n"
            f"  Either the Dense heads, the pooling, or the prompt text is wrong.")

    def sha256(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    digest = sha256(model_dir / "model.safetensors")
    if digest != pin["sha256"]:
        raise SystemExit(f"model.safetensors sha256 {digest} != CHECKPOINT.json pin "
                         f"{pin['sha256']} -- re-run fetch_model_gemma.py")

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
        "note": "fp32 CPU reference for EmbeddingGemma-300M (tasks/0055). "
                "hf.* are HuggingFace Gemma3TextModel outputs; st.embedding is "
                "the sentence-transformers pipeline as an independent oracle. "
                "hs_layer_semantics: hs[0]=scaled embed, hs[i+1] for "
                "i in 0..num_layers-2 = RAW layer i output (pre-final-norm), "
                "hs[-1] = last_hidden_state (POST final norm) -- NOT raw layer "
                "num_layers-1 output. See make_goldens_gemma.py header.",
    }

    GOLDENS.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        "hf.emb.scaled": hs[0],
        "hf.last_hidden_state": last_hidden.numpy(),
        "hf.pool.mean": pooled.numpy(),
        "hf.dense2": dense2.numpy(),
        "hf.dense3": dense3.numpy(),
        "hf.out.embedding": manual_embedding.numpy(),
        "st.embedding": st_emb,
    }
    # hs[i+1] is layer i's RAW output for i in 0..n_layers-2 only (see header).
    for i in range(n_layers - 1):
        tensors[f"hf.L{i}.resid2"] = hs[i + 1]

    slug = "embeddinggemma-300m_l24"
    path = GOLDENS / f"{slug}_s{SEQ_LEN}_boundary.safetensors"
    if path.exists() and not args.force:
        _, old_meta = load(path)
        if old_meta.get("source_sha256") and old_meta["source_sha256"] != digest:
            raise SystemExit(f"REFUSING to overwrite {path.name}: belongs to a "
                             f"different checkpoint. Pass --force to discard.")
    save(path, tensors, meta)
    print(f"\nwrote {path.relative_to(REPO)}  "
          f"({path.stat().st_size/1e6:.1f} MB, {len(tensors)} tensors)")

    if args.taps:
        from encoder_gemma import GemmaEmbeddingReference

        w, _ = load(model_dir / "model.safetensors")
        dense_w = {"2": d2["linear.weight"], "3": d3["linear.weight"]}
        ref = GemmaEmbeddingReference(
            w, dense_w, num_layers=n_layers, hidden=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"], num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"], intermediate=cfg["intermediate_size"],
            eps=cfg["rms_norm_eps"], rope_theta_global=cfg["rope_theta"],
            rope_theta_local=cfg["rope_local_base_freq"],
            sliding_window=cfg["sliding_window"],
            sliding_window_pattern=cfg.get("_sliding_window_pattern", 6),
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"])
        taps = {}
        ref.encode(enc["input_ids"].numpy(), enc["attention_mask"].numpy(), taps=taps)
        tap_meta = dict(meta)
        tap_meta["note"] = ("Full intermediate dump from reference/encoder_gemma.py. "
                            "Derivative of a sha256-pinned checkpoint; gitignored. "
                            "Regenerate: make_goldens_gemma.py --taps")
        tpath = GOLDENS / f"{slug}_s{SEQ_LEN}_taps.safetensors"
        save(tpath, taps, tap_meta)
        print(f"wrote {tpath.relative_to(REPO)}  "
              f"({tpath.stat().st_size/1e6:.1f} MB, {len(taps)} tensors)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
