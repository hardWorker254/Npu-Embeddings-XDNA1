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
inter = 3072
ref = NomicEmbeddingReference(w, num_layers=12, hidden=768, num_heads=12,
                              head_dim=64, intermediate=3072, eps=1e-12,
                              rope_theta=1000.0)
taps = {}
ref.encode(enc["input_ids"], enc["attention_mask"], taps=taps)

fc11 = taps["L0.fc11"].astype(np.float64)  # [4,64,3072]
fc12 = taps["L0.fc12"].astype(np.float64)
gated = taps["L0.gated"].astype(np.float64)

got_up = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/up.f32",
                     dtype=np.float32).reshape(4, 64, 2 * inter).astype(np.float64)
got_lo = got_up[..., :inter]
got_hi = got_up[..., inter:]
got_gated = np.fromfile(REPO / "tasks/0070-m13-nomic-runtime/dumps/gated.f32",
                        dtype=np.float32).reshape(4, 64, inter).astype(np.float64)

print("=== pre-SwiGLU (fc11=lo, fc12=hi) ===")
for b in range(4):
    dlo = np.abs(got_lo[b] - fc11[b])
    dhi = np.abs(got_hi[b] - fc12[b])
    print(f"  b={b}  lo(fc11) max_abs={dlo.max():.4e} rel={dlo.max()/np.abs(fc11[b]).max():.4e}"
          f"   hi(fc12) max_abs={dhi.max():.4e} rel={dhi.max()/np.abs(fc12[b]).max():.4e}"
          f"   fc12 range [{fc12[b].min():.2f},{fc12[b].max():.2f}]")

print("\n=== post-SwiGLU (gated = lo*silu(hi)) ===")
for b in range(4):
    d = np.abs(got_gated[b] - gated[b])
    rel = d.max() / np.abs(gated[b]).max()
    # locate worst element
    idx = np.unravel_index(np.argmax(d), d.shape)
    s_idx, j_idx = idx
    print(f"  b={b}  max_abs={d.max():.4e}  rel={rel:.4e}"
          f"   worst at (s={s_idx},j={j_idx}): got={got_gated[b][idx]:.6f} want={gated[b][idx]:.6f}"
          f"   fc12@that={fc12[b][idx]:.6f} fc11@that={fc11[b][idx]:.6f}")
