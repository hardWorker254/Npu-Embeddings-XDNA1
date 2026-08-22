import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "reference"))
from encoder_nomic import NomicEmbeddingReference, PROMPTS  # noqa: E402
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
hidden = 768
ref = NomicEmbeddingReference(w, num_layers=12, hidden=hidden, num_heads=12,
                              head_dim=64, intermediate=3072, eps=1e-12,
                              rope_theta=1000.0)
taps = {}
ref.encode(enc["input_ids"], enc["attention_mask"], taps=taps)

norm1 = taps["L0.norm1"].astype(np.float64)   # [4,64,768]
norm2 = taps["L0.norm2"].astype(np.float64)

got_n1 = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/norm1.f32",
                     dtype=np.float32).reshape(4, 64, hidden).astype(np.float64)
got_n2 = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/norm2.f32",
                     dtype=np.float32).reshape(4, 64, hidden).astype(np.float64)

print("=== L0.norm1 (post-attention) ===")
for b in range(4):
    d = np.abs(got_n1[b] - norm1[b])
    rel = d.max() / np.abs(norm1[b]).max()
    print(f"  b={b}  max_abs={d.max():.4e}  rel={rel:.4e}")

print("\n=== L0.norm2 (post-FFN, layer0 output) ===")
for b in range(4):
    d = np.abs(got_n2[b] - norm2[b])
    rel = d.max() / np.abs(norm2[b]).max()
    print(f"  b={b}  max_abs={d.max():.4e}  rel={rel:.4e}")
