# Why do oracle 1 (padded, manual CLS+L2) and oracle 2 (ST) disagree by 9e-02?
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
sys.path.insert(0, str(REPO / "reference"))
from corpus_gte import SENTENCES  # noqa: E402

M = REPO / "models" / "gte-multilingual-base"

tok = AutoTokenizer.from_pretrained(str(M))
md = AutoModel.from_pretrained(str(M), trust_remote_code=True,
                               torch_dtype=torch.float32).eval()
_rot = md.embeddings.rotary_emb
md.embeddings.rotary_emb = type(_rot)(
    dim=64, max_position_embeddings=8192, base=20000,
    scaling_factor=8.0, mixed_b=None)


def cls_norm_padded(seq):
    enc = tok(SENTENCES, padding="max_length", max_length=seq,
              truncation=True, return_tensors="pt")
    pos = torch.arange(seq)[None].expand(len(SENTENCES), -1)
    with torch.no_grad():
        out = md(input_ids=enc["input_ids"],
                 attention_mask=enc["attention_mask"], position_ids=pos)
    c = out.last_hidden_state[:, 0, :].numpy()
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def cls_norm_single():
    vs = []
    for s in SENTENCES:
        enc = tok(s, return_tensors="pt")
        S = enc["input_ids"].shape[1]
        enc["position_ids"] = torch.arange(S)[None]
        with torch.no_grad():
            out = md(**enc)
        c = out.last_hidden_state[0, 0].numpy()
        vs.append(c / np.linalg.norm(c))
    return np.stack(vs)


def cls_norm_padded_nopos(seq):
    enc = tok(SENTENCES, padding="max_length", max_length=seq,
              truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = md(input_ids=enc["input_ids"],
                 attention_mask=enc["attention_mask"])
    c = out.last_hidden_state[:, 0, :].numpy()
    return c / np.linalg.norm(c, axis=1, keepdims=True)


a = cls_norm_padded(64)
b = cls_norm_single()
c = cls_norm_padded_nopos(64)
print("padded-explicit-pos vs single :", np.abs(a - b).max())
print("padded-derived-pos  vs single :", np.abs(c - b).max())
print("padded-derived vs padded-expl :", np.abs(c - a).max())

from sentence_transformers import SentenceTransformer  # noqa: E402
st = SentenceTransformer(str(M), trust_remote_code=True, device="cpu",
                         model_kwargs={"torch_dtype": torch.float32})
am = st[0].auto_model
print("st auto_model type:", type(am).__name__)
_r = am.embeddings.rotary_emb
am.embeddings.rotary_emb = type(_r)(
    dim=64, max_position_embeddings=8192, base=20000,
    scaling_factor=8.0, mixed_b=None)
st_emb = st.encode(SENTENCES, convert_to_numpy=True)
print("st vs single:", np.abs(st_emb - b).max())
print("st vs padded:", np.abs(st_emb - a).max())
print("st norms:", np.linalg.norm(st_emb, axis=1))
# what does ST's tokenizer produce?
feats = st.tokenize(SENTENCES)
print("st tokenize keys:", list(feats.keys()),
      "shape:", feats["input_ids"].shape)
print("st max_seq_length:", st.max_seq_length)
