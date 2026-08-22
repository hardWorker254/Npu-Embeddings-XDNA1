"""Quick probe (not part of the deliverable): does AutoModel.from_pretrained on
models/nomic-embed-text-v1.5, with no trust_remote_code argument, resolve to the
NATIVE transformers.models.nomic_bert port? Does output_hidden_states=True work,
and what is hs[i] indexing? Does it agree with the remote (trust_remote_code=True)
code bit-identically, as tasks/0068's probe_nomic_arch.py section 12 found for a
single sentence? Every print below is a fact this task's make_goldens_nomic.py
and check_reference_nomic.py rely on.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = "models/nomic-embed-text-v1.5"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
enc = tok(["search_document: A man is eating food.",
           "search_document: Le café coûte 5 euros."],
          padding="max_length", max_length=32, truncation=True, return_tensors="pt")

print("=== default AutoModel.from_pretrained (no trust_remote_code arg) ===")
native = AutoModel.from_pretrained(MODEL_DIR)
native.eval()
print("class:", type(native).__name__, type(native).__module__)

with torch.no_grad():
    out_native = native(**enc, output_hidden_states=True)
hs = [h.numpy() for h in out_native.hidden_states]
print("num hidden_states:", len(hs), " (n_layer=12 -> expect 13)")
print("hs[0] shape:", hs[0].shape)

# Compare hs[0] against the embeddings module's own output (word+ttype, then emb_ln)
# by hooking it.
captured = {}
def _hook(mod, inp, outp):
    captured["emb_ln_out"] = outp
# find the LayerNorm inside embeddings in the native module tree
print("\nnative module tree (top level):")
for name, _ in native.named_children():
    print(" ", name)

hs_last = hs[-1]
last_hidden_native = out_native.last_hidden_state.numpy()
print("\nhs[-1] == last_hidden_state:", np.array_equal(hs_last, last_hidden_native))

print("\n=== remote code AutoModel.from_pretrained(trust_remote_code=True) ===")
remote = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
remote.eval()
print("class:", type(remote).__name__)
with torch.no_grad():
    out_remote = remote(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
last_hidden_remote = out_remote.last_hidden_state.detach().numpy()
d = float(np.abs(last_hidden_native - last_hidden_remote).max())
print("max abs diff native vs remote last_hidden_state:", d)

print("\n=== SentenceTransformer, no trust_remote_code ===")
try:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_DIR, device="cpu")
    v = st.encode(["search_document: A man is eating food."], convert_to_numpy=True)
    print("OK, norm:", np.linalg.norm(v))
except Exception as e:
    print("FAILED:", repr(e))
