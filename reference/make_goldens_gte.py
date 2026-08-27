# NpuEmbeddings -- 0.5.0 (tasks/0136, reference half): generate golden
# vectors for gte-multilingual-base from HuggingFace, same discipline as
# make_goldens_nomic.py (M13) / make_goldens_gemma.py (C1).
#
# TWO ORACLES, gated against each other before either is trusted -- with an
# honest caveat the nomic file does not need: gte's `model_type: "new"` has
# NO native transformers port, so BOTH oracles here run the author's
# trust_remote_code modeling.py. They are still worth gating against each
# other because they exercise independent PIPELINES (manual tokenize/pool/
# normalize here vs sentence-transformers' own tokenization, CLS pooling and
# Normalize module), but a bug inside the shared NewModel forward pass would
# escape this gate. The genuinely independent implementation is
# reference/encoder_gte.py (numpy, tasks/0134), and check_reference_gte.py
# is where it is held against these goldens -- with negative controls.
#
#   1. AutoModel.from_pretrained(trust_remote_code=True), fp32, hidden
#      states per layer via output_hidden_states=True (13 states for 12
#      layers; hs[-1] IS last_hidden_state -- no final norm, tasks/0134).
#   2. The real SentenceTransformer pipeline over the same checkpoint --
#      its own tokenization, its own CLS pooling, its own Normalize module
#      (modules.json genuinely carries 2_Normalize for this checkpoint).
#
# BOTH carry the two 0134 reference-repair landmines, non-negotiably:
#
#   L1. transformers v5 materialises the remote code's persistent=False
#       rotary buffers as UNINITIALISED meta-device memory -- inv_freq reads
#       run-to-run garbage. Repair: construct the rotary module fresh on CPU
#       (the author's own code, run as intended) and swap it in. Applied to
#       BOTH oracles' underlying model below.
#   L2. the checkpoint config says torch_dtype: float16 and v5 honours it.
#       Repair: torch_dtype=torch.float32, explicitly, on both.
#
# Any fp32 reference for this model on this env WITHOUT both repairs is
# silently wrong at every position (L1) or ~1e-03 off per layer (L2).
#
# gte has NO task prompts: nothing is prepended, and there is no --prompt
# flag here (unlike make_goldens_nomic.py).
#
# Env: .venv-ref
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_gte.py
#   & .\.venv-ref\Scripts\python.exe reference\make_goldens_gte.py --seq 256

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_gte import SENTENCES, SEQ_LEN as DEFAULT_SEQ_LEN  # noqa: E402
from safetensors_io import save, load           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "reference" / "goldens_gte"

# Cross-pipeline gate: oracle 1's manual CLS + L2-normalize vs oracle 2's
# end-to-end sentence-transformers output. Same forward code (see header),
# different tokenization/pooling/normalize code -- so this tolerance is
# about PIPELINE agreement, set at the same order as nomic's cross-oracle
# gate. The forward pass itself is discriminated in check_reference_gte.py.
TOL_ORACLE_AGREE = 2e-5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_to_clobber(path, sha, force):
    """A golden belongs to exactly one checkpoint. Same behaviour and same
    local-copy rationale as make_goldens_nomic.py's refuse_to_clobber()."""
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


def repair_rotary(model):
    """0134 landmine 1: rebuild the NTK rotary module fresh on CPU so its
    persistent=False buffers are computed rather than materialised as
    garbage. The constructor arguments are the checkpoint's own config
    values, asserted by pack_gte and by 0134's probe.

    EXTENDED HERE (tasks/0136): the SAME v5 meta-device mechanism also
    leaves `embeddings.position_ids` (modeling.py:308, persistent=False)
    uninitialised, which 0134 never hit because its probe always passed
    explicit position_ids. Any forward WITHOUT explicit positions -- which
    is exactly what SentenceTransformer does -- indexes the rope cache with
    that garbage: out-of-bounds garbage throws (loud), in-bounds garbage
    silently applies WRONG positions (measured here: oracle 2 disagreed
    with oracle 1 by 9.0e-02 max-abs on normalized CLS before this line).
    Rebuild it exactly as the author's __init__ does."""
    import torch

    _rot = model.embeddings.rotary_emb
    model.embeddings.rotary_emb = type(_rot)(
        dim=64, max_position_embeddings=8192, base=20000,
        scaling_factor=8.0, mixed_b=None)
    model.embeddings.register_buffer(
        "position_ids", torch.arange(model.config.max_position_embeddings),
        persistent=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default=str(REPO / "models" / "gte-multilingual-base"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--taps", action="store_true",
                    help="also write <slug>_s<seq>_taps.safetensors from "
                         "reference/encoder_gte.py (emb.sum / emb.ln / "
                         "L0.qkv), the fixtures tools/export_validation.py "
                         "consumes. Gitignored -- a deterministic derivative "
                         "of the pinned checkpoint.")
    # T41 (tasks/0116): seq is a flag, not an edit -- the golden FILENAME
    # carries it (_s<seq>_), so goldens at different lengths coexist.
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ_LEN,
                    help=f"sequence length to pad the corpus to "
                         f"(default {DEFAULT_SEQ_LEN}). Part of the golden "
                         f"filename; the .npue and the design must carry at "
                         f"least this many positions / this seq.")
    args = ap.parse_args()
    SEQ_LEN = args.seq

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    n_layers = cfg["num_hidden_layers"]

    digest = sha256(model_dir / "model.safetensors")
    # gte was downloaded manually (tasks/0134): CHECKPOINT.json pins the
    # repo_id but carries no sha256 (unlike a hub fetch). The identity gate
    # that IS available is the packed container's own source_sha256 -- the
    # goldens must describe the same checkpoint the runtime serves.
    pin = json.loads((model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))
    npue_path = REPO / "models" / "gte-multilingual-base.npue"
    if npue_path.exists():
        sys.path.insert(0, str(REPO / "tools"))
        import npue as npue_mod
        packed_sha = npue_mod.Reader(str(npue_path)).config["source_sha256"]
        if packed_sha != digest:
            raise SystemExit(
                f"\nFAIL -- model.safetensors sha256 {digest[:16]}... does "
                f"not match the packed container's source_sha256 "
                f"{packed_sha[:16]}...\n  These goldens would describe a "
                f"different checkpoint than the runtime serves. Re-download "
                f"or re-pack first.")
        print(f"checkpoint sha256 {digest[:16]}... matches the packed "
              f"container's source_sha256")
    else:
        print(f"checkpoint sha256 {digest[:16]}... (no packed container to "
              f"cross-check against)")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    enc = tok(SENTENCES, padding="max_length", max_length=SEQ_LEN,
              truncation=True, return_tensors="pt")

    print(f"corpus: {len(SENTENCES)} sentences, NO prompt (gte has no "
          f"prompts table), padded to S={SEQ_LEN}")
    for i, s in enumerate(SENTENCES):
        ids = enc["input_ids"][i]
        n = int(enc["attention_mask"][i].sum())
        print(f"  [{i}] {n:3d} tokens  {tok.convert_ids_to_tokens(ids[:n])}")
        if n >= SEQ_LEN:
            raise SystemExit(
                f"\nFAIL -- sentence {i} used all {SEQ_LEN} slots; it may "
                f"have been silently truncated. Pass a larger --seq.")

    # -- Oracle 1: repaired trust_remote_code NewModel, boundary taps -------
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    repair_rotary(model)
    print(f"\noracle 1 (boundary taps): "
          f"{type(model).__module__}.{type(model).__name__} "
          f"(trust_remote_code, fp32, rotary repaired -- 0134 L1+L2)")
    # Explicit position_ids: the remote code's derived-position path is the
    # half of landmine 1 that is LOUD; explicit positions are also exactly
    # what the runtime and encoder_gte.py compute (0..S-1).
    pos = torch.arange(SEQ_LEN)[None].expand(len(SENTENCES), -1)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    position_ids=pos, output_hidden_states=True)

    last_hidden = out.last_hidden_state
    hs = [h.numpy() for h in out.hidden_states]
    assert len(hs) == n_layers + 1, \
        f"expected {n_layers + 1} hidden_states, got {len(hs)}"
    assert np.array_equal(hs[-1], last_hidden.numpy()), \
        "hs[-1] is expected to equal last_hidden_state -- gte has no final " \
        "norm (tasks/0134: layer 12's mlp_ln output IS last_hidden_state). " \
        "Indexing assumption broke -- re-check the installed transformers."

    # Manual CLS pooling + L2 normalize (both are the checkpoint's own:
    # 1_Pooling/config.json says cls, modules.json carries 2_Normalize).
    cls_raw = last_hidden[:, 0, :].numpy()
    cls_norm = cls_raw / np.linalg.norm(cls_raw, axis=1, keepdims=True)

    # -- Oracle 2: the real SentenceTransformer pipeline --------------------
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(model_dir), trust_remote_code=True,
                             device="cpu",
                             model_kwargs={"torch_dtype": torch.float32})
    repair_rotary(st[0].auto_model)
    print(f"oracle 2 (end-to-end)     : SentenceTransformer"
          f"(trust_remote_code=True, fp32, rotary repaired) -- its own "
          f"tokenization, CLS pooling and Normalize module")
    st_emb = st.encode(SENTENCES, convert_to_numpy=True)

    delta = float(np.abs(st_emb - cls_norm).max())
    print(f"\noracle 1 (manual CLS + L2) vs oracle 2 (SentenceTransformer): "
          f"max abs diff {delta:.3e}")
    if not (delta < TOL_ORACLE_AGREE):
        raise SystemExit(
            f"\nFAIL -- the two pipelines disagree by {delta:.3e} "
            f"(limit {TOL_ORACLE_AGREE:.0e}).\n"
            f"  One of pooling/normalization/tokenization is wrong here "
            f"(the forward code is shared -- see this script's header).\n"
            f"  Do NOT proceed to trust either until this is resolved.")
    print(f"PASS -- pipelines agree within {TOL_ORACLE_AGREE:.0e}")

    meta = {
        "repo_id": pin["repo_id"],
        "source_sha256": digest,
        "seq_len": str(SEQ_LEN),
        "batch": str(len(SENTENCES)),
        "num_layers": str(n_layers),
        "sentences": json.dumps(SENTENCES, ensure_ascii=False),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "note": "fp32 CPU reference for gte-multilingual-base (tasks/0136). "
                "hf.* are the trust_remote_code NewModel's outputs with BOTH "
                "0134 reference repairs applied (rotary rebuilt on CPU, "
                "explicit fp32); st.* is the same checkpoint through the "
                "real SentenceTransformer pipeline (also repaired). The two "
                "share the forward code -- the independent implementation is "
                "reference/encoder_gte.py, held against these goldens by "
                "check_reference_gte.py with negative controls. "
                "hs_layer_semantics: hs[0]=embeddings.LayerNorm output "
                "(before layer 0), hs[i+1] = layer i's mlp_ln output; "
                "hs[-1] == last_hidden_state exactly (no final norm). "
                "pool.cls_norm is L2-normalized (modules.json carries "
                "2_Normalize -- genuinely this checkpoint's own, unlike "
                "nomic).",
    }

    GOLDENS.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        "hf.emb.ln": hs[0],
        "hf.last_hidden_state": last_hidden.numpy(),
        "hf.pool.cls_raw": cls_raw,
        "hf.pool.cls_norm": cls_norm,
        "st.pool.cls_norm": st_emb,
    }
    for i in range(n_layers):
        tensors[f"hf.L{i}.mlp_ln"] = hs[i + 1]

    slug = "gte-multilingual-base_l12"
    path = GOLDENS / f"{slug}_s{SEQ_LEN}_boundary.safetensors"
    refuse_to_clobber(path, digest, args.force)
    save(path, tensors, meta)
    print(f"\nwrote {path.relative_to(REPO)}  "
          f"({path.stat().st_size/1e6:.1f} MB, {len(tensors)} tensors)")

    if args.taps:
        # Fixture taps from the INDEPENDENT implementation
        # (reference/encoder_gte.py, held at 1e-06 against the hf.* boundary
        # by check_reference_gte.py) -- same division of labour as
        # make_goldens_nomic.py: the boundary file is the torch reference,
        # the taps file is the numpy oracle's intermediates, and
        # tools/export_validation.py consumes the taps.
        sys.path.insert(0, str(REPO / "reference"))
        from encoder_gte import GteEncoder
        w, _ = load(model_dir / "model.safetensors")
        w32 = {}
        for k, v in w.items():
            kk = k[4:] if k.startswith("new.") else k
            w32[kk] = v.astype(np.float32) if v.dtype == np.float16 else v
        ref = GteEncoder(w32, num_layers=n_layers,
                         hidden=cfg["hidden_size"],
                         num_heads=cfg["num_attention_heads"],
                         intermediate=cfg["intermediate_size"],
                         eps=cfg["layer_norm_eps"])
        per_key = {}
        for i in range(len(SENTENCES)):
            row_taps = {}
            ref.encode(enc["input_ids"][i].numpy(),
                       enc["attention_mask"][i].numpy(), taps=row_taps)
            for k, v in row_taps.items():
                per_key.setdefault(k, []).append(v)
        taps = {k: np.ascontiguousarray(np.stack(v), dtype=np.float32)
                for k, v in per_key.items()}
        tap_meta = dict(meta)
        tap_meta["note"] = ("Intermediates from reference/encoder_gte.py "
                            "(emb.sum / emb.ln / L0.qkv, UNFOLDED qkv). "
                            "Derivative of a sha256-pinned checkpoint; "
                            "gitignored. Regenerate: make_goldens_gte.py "
                            "--taps")
        tpath = GOLDENS / ("gte-multilingual-base_l12_s%d_taps.safetensors"
                           % SEQ_LEN)
        refuse_to_clobber(tpath, digest, args.force)
        save(tpath, taps, tap_meta)
        print("wrote %s  (%.1f MB, %d tensors)"
              % (tpath.relative_to(REPO), tpath.stat().st_size / 1e6,
                 len(taps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
