"""Phase 8: the NTK frequencies at LONG positions -- oracle vs repaired
reference at seq ~500, where the scaling factor exists to matter.

The 0134 probe validated at 18/19 tokens; a theta error grows with position
(the rotation angle is position x inv_freq), so this is the check that would
catch a long-position divergence the short probe could not.
Run in .venv-ref from the repo root.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "reference"))
import encoder_gte

M = REPO / "models" / "gte-multilingual-base"
tok = AutoTokenizer.from_pretrained(str(M))
md = AutoModel.from_pretrained(str(M), trust_remote_code=True,
                               torch_dtype=torch.float32).eval()
_rot = md.embeddings.rotary_emb
md.embeddings.rotary_emb = type(_rot)(dim=64, max_position_embeddings=8192,
                                      base=20000, scaling_factor=8.0,
                                      mixed_b=None)

base_text = ("The analytic roofline places every dispatch by its arithmetic "
             "intensity, and the measured roof at forty-five gigabytes per "
             "second decides which side of the ridge production lands on. ")
text = base_text * 24
t = tok(text, return_tensors="pt", truncation=True, max_length=512)
ids = t["input_ids"][0].numpy()
S = len(ids)
print(f"tokens: {S}")
t["position_ids"] = torch.arange(S)[None]
with torch.no_grad():
    out = md(**t, output_hidden_states=True)
hs = [h[0].to(torch.float32).numpy() for h in out.hidden_states]

enc = encoder_gte.GteEncoder({k[4:] if k.startswith("new.") else k:
                              v.detach().to(torch.float32).numpy()
                              for k, v in md.state_dict().items()})
ours = enc.encode(ids)

def relfro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

print(f"last_hidden relfro {relfro(ours['last_hidden'], hs[-1]):.3e}")
print(f"CLS         relfro {relfro(ours['cls_raw'], hs[-1][0]):.3e}")

mask = np.zeros(S, dtype=np.float32)
cos, sin = encoder_gte.rope_cos_sin(S)
h0 = enc.embed(ids)
right = enc.layer(h0, 0, cos, sin, mask)
inv = (1.0 / (20000.0 ** (np.arange(0, 64, 2) / 64.0))).astype(np.float32)
t_ = np.arange(S, dtype=np.float32)
f_ = np.einsum("i,j->ij", t_, inv)
e_ = np.concatenate([f_, f_], -1)
wrong = enc.layer(h0, 0, np.cos(e_), np.sin(e_), mask)
print(f"layer0 right relfro {relfro(right, hs[1]):.3e}   "
      f"control theta=20000-plain relfro {relfro(wrong, hs[1]):.3e}")
