# NpuEmbeddings -- C1 spike (tasks/0055): fetch and pin the EmbeddingGemma-300M
# checkpoint used for the numpy reference encoder / goldens.
#
# UNLIKE reference/fetch_model.py, this does NOT assert a BERT-shaped tensor
# inventory -- Gemma3's architecture is completely different (see
# reference/encoder_gemma.py's header). It asserts the structural facts this
# project's geometry analysis and reference encoder actually depend on.
#
# Source: the GATED official `google/embeddinggemma-300m` needs an accepted
# Gemma license + HF_TOKEN. This script defaults to the UNGATED community
# mirror `unsloth/embeddinggemma-300m` (config verified identical) so the
# spike does not depend on gated credentials. If HF_TOKEN is set, it prefers
# the official repo instead -- see tasks/0055 TASK.md for why a future
# production integration (C3) should NOT depend on the mirror long-term.
#
# Env: .venv-ref
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\fetch_model_gemma.py

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

ALLOW = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
]

# Structural facts the reference encoder and geometry note depend on --
# verified against the real HF Gemma3 modeling code (see encoder_gemma.py),
# not assumed. A checkpoint that disagrees on any of these needs the encoder
# rewritten, not silently run.
EXPECT_CONFIG = {
    "model_type": "gemma3_text",
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_hidden_layers": 24,
    "num_attention_heads": 3,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "use_bidirectional_attention": True,
    "attention_bias": False,
    "query_pre_attn_scalar": 256,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "sliding_window": 512,
    "_sliding_window_pattern": 6,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="override: HF repo id. Default picks official repo "
                         "if HF_TOKEN is set, else the unsloth mirror.")
    ap.add_argument("--dest", default=str(MODELS / "embeddinggemma-300m"))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.model:
        repo_id = args.model
    elif token:
        repo_id = "google/embeddinggemma-300m"
    else:
        repo_id = "unsloth/embeddinggemma-300m"
        print("HF_TOKEN not set -- using the ungated community mirror "
              f"{repo_id} (research/spike use only; see tasks/0055 for why "
              "production (C3) should fetch from google/embeddinggemma-300m "
              "with a token instead).")

    from huggingface_hub import snapshot_download

    local = Path(args.dest)
    print(f"fetching {repo_id} -> {local}")
    snapshot_download(repo_id=repo_id, local_dir=str(local),
                      allow_patterns=ALLOW, token=token)

    digest = sha256(local / "model.safetensors")
    print(f"  model.safetensors     : {(local / 'model.safetensors').stat().st_size/1e6:.1f} MB")
    print(f"  sha256                : {digest}")

    cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
    problems = []
    for k, want in EXPECT_CONFIG.items():
        got = cfg.get(k)
        if got != want:
            problems.append(f"config.{k}: expected {want!r}, got {got!r}")

    (local / "CHECKPOINT.json").write_text(
        json.dumps({"repo_id": repo_id, "file": "model.safetensors", "sha256": digest},
                   indent=2),
        encoding="utf-8",
    )

    if problems:
        print("\nFAIL -- checkpoint does not match encoder_gemma.py's assumptions:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK -- checkpoint matches encoder_gemma.py's structural assumptions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
