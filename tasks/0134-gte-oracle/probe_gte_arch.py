"""Probe: reference/encoder_gte.py vs the real trust_remote_code NewModel,
per layer, WITH negative controls -- the 0068 discipline: every architectural
fact gets a measured discriminator, so a future reader sees not just that the
right reading matches but by how much the plausible wrong ones miss.

Run in .venv-ref from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "reference"))
import encoder_gte  # noqa: E402

M = REPO / "models" / "gte-multilingual-base"

tok = AutoTokenizer.from_pretrained(str(M))
md = AutoModel.from_pretrained(str(M), trust_remote_code=True,
                               torch_dtype=torch.float32).eval()

# TRANSFORMERS 5.15 LANDMINE (found here, recorded in the task log): the
# remote code computes its rotary buffers in __init__, but v5 instantiates
# custom modules on the meta device and materialises persistent=False
# buffers as UNINITIALISED memory -- the loaded model's inv_freq reads
# run-to-run garbage and its cos/sin caches with it. The reference is
# repaired by constructing the rotary module fresh on CPU (the author's own
# code, run as intended). The fresh buffer equals our derived formula at
# max|diff| = 0.0.
_rot = md.embeddings.rotary_emb
md.embeddings.rotary_emb = type(_rot)(
    dim=64, max_position_embeddings=8192, base=20000,
    scaling_factor=8.0, mixed_b=None)

TEXTS = [
    "The AMD Ryzen AI NPU accelerates transformer encoder models.",
    "Retrieval quality depends on the embedding model more than the index.",
]

def relfro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

# --- weights into numpy, 'new.' prefix stripped -------------------------------
sd = md.state_dict()
w = {}
for k, v in sd.items():
    kk = k[4:] if k.startswith("new.") else k
    w[kk] = v.detach().to(torch.float32).numpy()
print("state dict:", len(w), "tensors; sample names:",
      sorted(w)[:3], "...", sorted(w)[-2:])

# --- 0. the inv_freq check: our formula vs the module's registered buffer ----
rot = md.embeddings.rotary_emb
real_if = rot.inv_freq.detach().numpy().astype(np.float32)
ours_if = encoder_gte.gte_inv_freq()
print(f"\ninv_freq: ours vs module buffer  max|diff| = {np.abs(ours_if - real_if).max():.3e}")
plain20k = 1.0 / (20000.0 ** (np.arange(0, 64, 2) / 64.0))
plain160k = 1.0 / (160000.0 ** (np.arange(0, 64, 2) / 64.0))
print(f"  negative controls: plain-20000 relfro {relfro(plain20k, real_if):.3e}, "
      f"plain-160000 relfro {relfro(plain160k, real_if):.3e} (ours must be ~0, these must not)")

enc = encoder_gte.GteEncoder(w)

for text in TEXTS:
    t = tok(text, return_tensors="pt")
    ids = t["input_ids"][0].numpy()
    S = len(ids)
    print(f"\n=== {S} tokens")

    # explicit position_ids: the derived-position path indexes with garbage
    # under this transformers version (same landmine, second half).
    t["position_ids"] = torch.arange(S)[None]
    with torch.no_grad():
        out = md(**t, output_hidden_states=True)
    hs = [h[0].to(torch.float32).numpy() for h in out.hidden_states]  # 13 states

    # embeddings
    h = enc.embed(ids)
    print(f"  embeddings           relfro {relfro(h, hs[0]):.3e}")

    # per layer
    mask = np.zeros(S, dtype=np.float32)
    cos, sin = encoder_gte.rope_cos_sin(S)
    hh = h
    for i in range(12):
        hh = enc.layer(hh, i, cos, sin, mask)
        print(f"  layer {i:2d}             relfro {relfro(hh, hs[i+1]):.3e}")

    ours = enc.encode(ids)
    real_cls = hs[-1][0]
    print(f"  CLS                  relfro {relfro(ours['cls_raw'], real_cls):.3e}")

    # --- negative controls, layer 0 only (cheap, decisive) ------------------
    def layer0_with(cos_c, sin_c, label):
        h0 = enc.layer(h, 0, cos_c, sin_c, mask)
        print(f"  control {label:<22} relfro {relfro(h0, hs[1]):.3e}")
    c20, s20 = None, None
    inv = plain20k.astype(np.float32)
    t_ = np.arange(S, dtype=np.float32)
    f_ = np.einsum("i,j->ij", t_, inv); e_ = np.concatenate([f_, f_], -1)
    layer0_with(np.cos(e_), np.sin(e_), "theta=20000 plain")
    inv = plain160k.astype(np.float32)
    f_ = np.einsum("i,j->ij", t_, inv); e_ = np.concatenate([f_, f_], -1)
    layer0_with(np.cos(e_), np.sin(e_), "theta=160000 no-corr")
    # GELU on the WRONG half
    real_mlp = enc.mlp
    def wrong_mlp(hx):
        ww = enc.w; p = enc.pfx
        ug = enc.linear(hx, ww[p + "mlp.up_gate_proj.weight"])
        up, gate = ug[:, :enc.inter], ug[:, enc.inter:]
        act = enc.gelu(up) * gate            # WRONG: activation on up
        return enc.linear(act, ww[p + "mlp.down_proj.weight"], ww[p + "mlp.down_proj.bias"])
    enc.mlp = wrong_mlp
    h0 = enc.layer(h, 0, cos, sin, mask)
    print(f"  control gelu-on-up-half      relfro {relfro(h0, hs[1]):.3e}")
    enc.mlp = real_mlp
