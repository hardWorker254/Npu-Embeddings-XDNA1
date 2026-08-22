# NpuEmbeddings -- round-trip verification for the arch=1 (Gemma) .npue.
# SPDX-License-Identifier: Apache-2.0
#
# Confirms every packed tensor is bit-exact against the SOURCE checkpoint's
# actual values (not a re-derivation of them, per CLAUDE.md trap 6c's
# spirit -- this reads model.safetensors directly, not anything the packer
# itself computed). transpose-only tensors must be an exact (permuted) copy;
# norm vectors must be byte-identical.
#
# Env: iron (numpy only). Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_gemma.py

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

from npue import Reader                    # noqa: E402
from safetensors_io import load            # noqa: E402


def main():
    model_dir = REPO / "models" / "embeddinggemma-300m"
    npue_path = REPO / "models" / "embeddinggemma-300m.npue"

    src, _ = load(model_dir / "model.safetensors")
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    r = Reader(npue_path)
    assert r.arch == 1, f"arch {r.arch}, expected 1"
    assert r.config["model_type"] == "gemma3_text"
    print(f"opened {npue_path.name}: arch={r.arch}, {len(r.entries)} tensors, "
          f"config.arch={r.config['arch']!r}")

    checks = 0
    fails = []

    def check(name, packed, want):
        nonlocal checks
        checks += 1
        packed = np.asarray(packed)
        want = np.asarray(want, dtype=np.float32)
        if packed.shape != want.shape:
            fails.append(f"{name}: shape {packed.shape} != {want.shape}")
            return
        if not np.array_equal(packed, want):
            maxdiff = np.abs(packed.astype(np.float64) - want.astype(np.float64)).max()
            fails.append(f"{name}: NOT bit-exact, max abs diff {maxdiff:.3e}")

    check("embed_tokens.weight", r.tensor("embed_tokens.weight"),
          src["embed_tokens.weight"])
    check("norm.weight", r.tensor("norm.weight"), src["norm.weight"])
    check("dense2.weight", r.tensor("dense2.weight"), d2["linear.weight"].T)
    check("dense3.weight", r.tensor("dense3.weight"), d3["linear.weight"].T)

    L = cfg["num_hidden_layers"]
    for i in range(L):
        p = f"layers.{i}."
        sa = p + "self_attn."
        mp = p + "mlp."
        check(f"layer.{i}.q_proj", r.tensor(f"layer.{i}.q_proj"),
              src[sa + "q_proj.weight"].T)
        check(f"layer.{i}.k_proj", r.tensor(f"layer.{i}.k_proj"),
              src[sa + "k_proj.weight"].T)
        check(f"layer.{i}.v_proj", r.tensor(f"layer.{i}.v_proj"),
              src[sa + "v_proj.weight"].T)
        check(f"layer.{i}.o_proj", r.tensor(f"layer.{i}.o_proj"),
              src[sa + "o_proj.weight"].T)
        check(f"layer.{i}.q_norm.weight", r.tensor(f"layer.{i}.q_norm.weight"),
              src[sa + "q_norm.weight"])
        check(f"layer.{i}.k_norm.weight", r.tensor(f"layer.{i}.k_norm.weight"),
              src[sa + "k_norm.weight"])
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            check(f"layer.{i}.{nm}.weight", r.tensor(f"layer.{i}.{nm}.weight"),
                  src[p + nm + ".weight"])
        check(f"layer.{i}.gate_proj", r.tensor(f"layer.{i}.gate_proj"),
              src[mp + "gate_proj.weight"].T)
        check(f"layer.{i}.up_proj", r.tensor(f"layer.{i}.up_proj"),
              src[mp + "up_proj.weight"].T)
        check(f"layer.{i}.down_proj", r.tensor(f"layer.{i}.down_proj"),
              src[mp + "down_proj.weight"].T)

    # Tokenizer table travels as raw bytes -- verified against the file on
    # disk, not re-generated.
    tok_bytes = (model_dir / "gemma_tokenizer.bin").read_bytes()
    packed_tok = r.raw("tokenizer.gemma_table").tobytes()
    checks += 1
    if packed_tok != tok_bytes:
        fails.append(f"tokenizer.gemma_table: {len(packed_tok)} bytes packed "
                     f"!= {len(tok_bytes)} bytes on disk")

    r.close()

    print(f"\n{checks} checks, {len(fails)} failures")
    if fails:
        for f in fails[:20]:
            print(f"  FAIL {f}")
        return 1
    print("PASS -- every packed tensor is bit-exact against the source checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
