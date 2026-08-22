import sys, math
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "reference"))
from encoder_nomic import NomicEmbeddingReference, PROMPTS, rope_cos_sin, apply_rope  # noqa: E402
from safetensors_io import load  # noqa: E402

texts = ["A man is playing a guitar on stage.",
         "Someone plays a guitar at a concert.",
         "A woman is slicing an onion in the kitchen.",
         "The stock market closed lower on Tuesday."]
prefix = PROMPTS["search_document"]
prefixed = [prefix + t for t in texts]

model_dir = REPO / "models" / "nomic-embed-text-v1.5"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(model_dir))
enc = tok(prefixed, padding="max_length", max_length=64, truncation=True, return_tensors="np")

w, _ = load(model_dir / "model.safetensors")
hidden, H, hd = 768, 12, 64
ref = NomicEmbeddingReference(w, num_layers=12, hidden=hidden, num_heads=H,
                              head_dim=hd, intermediate=3072, eps=1e-12,
                              rope_theta=1000.0)
taps = {}
ref.encode(enc["input_ids"], enc["attention_mask"], taps=taps)

qkv = taps["L0.qkv"].astype(np.float64)  # [4,64,2304] three-major [Q|K|V]
scale = 1.0 / math.sqrt(hd)
qkv_scaled = qkv.copy()
qkv_scaled[..., :hidden] *= scale   # matches the .npue's fold-into-Q

B, S, _ = qkv.shape
got_pre = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/prerope.f32",
                      dtype=np.float32).reshape(B, S, 3 * hidden).astype(np.float64)
got_post = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/postrope.f32",
                       dtype=np.float32).reshape(B, S, 3 * hidden).astype(np.float64)

print("=== pre-RoPE (should match scaled oracle qkv exactly, bf16-ish) ===")
for b in range(B):
    diff = np.abs(got_pre[b] - qkv_scaled[b])
    print(f"  b={b}  max_abs_diff={diff.max():.4e}  rel={diff.max()/np.abs(qkv_scaled[b]).max():.4e}")

# Oracle RoPE on Q,K (three-major slice), same theta.
cos, sin = rope_cos_sin(S, hd, 1000.0)
q = qkv_scaled[..., :hidden].reshape(B, S, H, hd).transpose(0, 2, 1, 3)  # [B,H,S,D]
k = qkv_scaled[..., hidden:2*hidden].reshape(B, S, H, hd).transpose(0, 2, 1, 3)
q2, k2 = apply_rope(q, k, cos, sin)
q2b = q2.transpose(0, 2, 1, 3).reshape(B, S, hidden)
k2b = k2.transpose(0, 2, 1, 3).reshape(B, S, hidden)
oracle_post = qkv_scaled.copy()
oracle_post[..., :hidden] = q2b
oracle_post[..., hidden:2*hidden] = k2b

print("\n=== post-RoPE (Q,K rotated; V unchanged) ===")
for b in range(B):
    dq = np.abs(got_post[b, :, :hidden] - oracle_post[b, :, :hidden])
    dk = np.abs(got_post[b, :, hidden:2*hidden] - oracle_post[b, :, hidden:2*hidden])
    dv = np.abs(got_post[b, :, 2*hidden:] - oracle_post[b, :, 2*hidden:])
    print(f"  b={b}  Q max_abs={dq.max():.4e}  K max_abs={dk.max():.4e}  V max_abs={dv.max():.4e}")
