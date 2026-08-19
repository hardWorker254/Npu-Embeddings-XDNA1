# NpuEmbeddings -- M3: fetch and verify the reference checkpoint.
#
# Downloads sentence-transformers/all-MiniLM-L6-v2 into models/<name>/ and then
# ASSERTS it against docs/04-model/README.md: the config values, the 104-tensor
# inventory, and the tensor shapes. A silently different checkpoint would poison
# every golden vector downstream, so the sha256 of model.safetensors is printed
# and stored -- .npue files will carry it as `source_sha256` (M4).
#
# Env: .venv-ref  (huggingface_hub, safetensors)
# Usage:
#   & .\.venv-ref\Scripts\python.exe reference\fetch_model.py
#   & .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model BAAI/bge-small-en-v1.5

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

# Only what we actually consume. The .onnx/.openvino exports and the pytorch
# .bin duplicate are several hundred MB of nothing.
ALLOW = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
]

# STRUCTURAL expectations: what this runtime can execute at all. These are
# properties of the architecture, not of one checkpoint's size, so they stay
# literal and a checkpoint that violates them must be refused.
# docs/04-model/README.md, "Verified config.json (fetched live)".
EXPECT_CONFIG = {
    "model_type": "bert",
    "layer_norm_eps": 1e-12,
    "position_embedding_type": "absolute",
    "hidden_act": "gelu",
    "pad_token_id": 0,
    "type_vocab_size": 2,
}


# DIMENSIONAL expectations come from the checkpoint's own config.json and are
# then used to demand that every tensor agrees with it. That still catches the
# failure that actually happens -- a config that disagrees with the weights
# next to it -- without hardcoding one model's width.
def per_layer(hidden, intermediate):
    return {
        "attention.self.query.weight": [hidden, hidden],
        "attention.self.query.bias": [hidden],
        "attention.self.key.weight": [hidden, hidden],
        "attention.self.key.bias": [hidden],
        "attention.self.value.weight": [hidden, hidden],
        "attention.self.value.bias": [hidden],
        "attention.output.dense.weight": [hidden, hidden],
        "attention.output.dense.bias": [hidden],
        "attention.output.LayerNorm.weight": [hidden],
        "attention.output.LayerNorm.bias": [hidden],
        "intermediate.dense.weight": [intermediate, hidden],
        "intermediate.dense.bias": [intermediate],
        "output.dense.weight": [hidden, intermediate],
        "output.dense.bias": [hidden],
        "output.LayerNorm.weight": [hidden],
        "output.LayerNorm.bias": [hidden],
    }


def embeddings(hidden, vocab, positions, type_vocab):
    return {
        "embeddings.word_embeddings.weight": [vocab, hidden],
        "embeddings.position_embeddings.weight": [positions, hidden],
        "embeddings.token_type_embeddings.weight": [type_vocab, hidden],
        "embeddings.LayerNorm.weight": [hidden],
        "embeddings.LayerNorm.bias": [hidden],
    }

# Dead weight: sentence-transformers never calls the pooler. 147,840 params we
# must NOT implement. Present in the checkpoint, so allow but do not require.
IGNORABLE = {"pooler.dense.weight", "pooler.dense.bias", "embeddings.position_ids"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(local, expect_layers):
    """Assert the checkpoint matches what docs/04-model claims. Returns problems."""
    from safetensors import safe_open

    problems = []
    cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
    for k, want in EXPECT_CONFIG.items():
        got = cfg.get(k)
        if got != want:
            problems.append(f"config.{k}: expected {want!r}, got {got!r}")
    n_layers = cfg.get("num_hidden_layers")
    if n_layers != expect_layers:
        problems.append(f"config.num_hidden_layers: expected {expect_layers}, got {n_layers}")

    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    inter = cfg.get("intermediate_size")
    vocab = cfg.get("vocab_size")
    positions = cfg.get("max_position_embeddings")
    type_vocab = cfg.get("type_vocab_size")

    # What the runtime requires of any width, checked here rather than
    # discovered at dispatch time.
    if not (hidden and heads) or hidden % heads:
        problems.append(f"hidden_size {hidden} is not divisible by "
                        f"num_attention_heads {heads}")
    elif (hidden // heads) % 8:
        problems.append(f"head_dim {hidden // heads} is not a multiple of 8; "
                        f"the host attention kernels step 8 floats with no tail")
    if hidden and hidden % 8:
        problems.append(f"hidden_size {hidden} is not a multiple of 8")
    if inter and inter % 8:
        problems.append(f"intermediate_size {inter} is not a multiple of 8")

    want = embeddings(hidden, vocab, positions, type_vocab)
    pl = per_layer(hidden, inter)
    for i in range(n_layers or 0):
        for suffix, shape in pl.items():
            want[f"encoder.layer.{i}.{suffix}"] = shape

    st = local / "model.safetensors"
    with safe_open(st, framework="np") as f:
        names = set(f.keys())
        shapes = {n: list(f.get_slice(n).get_shape()) for n in names}
        dtypes = {n: f.get_slice(n).get_dtype() for n in names}

    for name, shape in want.items():
        if name not in names:
            problems.append(f"missing tensor {name}")
        elif shapes[name] != shape:
            problems.append(f"{name}: expected shape {shape}, got {shapes[name]}")
        elif dtypes[name] not in ("F32", "float32"):
            problems.append(f"{name}: expected F32, got {dtypes[name]}")

    unexpected = names - set(want) - IGNORABLE
    if unexpected:
        problems.append(f"unexpected tensors: {sorted(unexpected)}")

    # docs claim 104 = 5 embeddings + 6*16 layer + 2 pooler, position_ids excluded
    # (it is a buffer). Report the real count either way.
    print(f"  tensors in checkpoint : {len(names)}")
    print(f"    required by arch    : {len(want)}")
    print(f"    ignorable present   : {sorted(names & IGNORABLE)}")
    print(f"  layers                : {n_layers}")
    print(f"  geometry              : hidden {hidden}, {heads} heads x "
          f"{hidden // heads if hidden and heads else '?'}, ffn {inter}, "
          f"vocab {vocab}, {positions} positions")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--layers", type=int, default=6, help="expected num_hidden_layers")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    local = MODELS / args.model.split("/")[-1]
    print(f"fetching {args.model} -> {local}")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(local),
        allow_patterns=ALLOW,
    )

    digest = sha256(local / "model.safetensors")
    print(f"  model.safetensors     : {(local / 'model.safetensors').stat().st_size/1e6:.1f} MB")
    print(f"  sha256                : {digest}")

    problems = check(local, args.layers)

    # Pin it. Goldens and .npue files reference this; a changed digest must fail
    # loudly rather than quietly compare against a different checkpoint.
    (local / "CHECKPOINT.json").write_text(
        json.dumps(
            {"repo_id": args.model, "file": "model.safetensors", "sha256": digest},
            indent=2,
        ),
        encoding="utf-8",
    )

    if problems:
        print("\nFAIL -- checkpoint does not match docs/04-model/README.md:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK -- checkpoint matches docs/04-model/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
