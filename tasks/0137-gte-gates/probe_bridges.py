# 0137 pre-flight: validate BOTH MTEB harness sides for gte before any MTEB run.
#  (a) NPU side: NpuEncoder's --encode-file bridge (HF tokenizer + emb_sum)
#      must agree with the shipped CLI's own --embed output (C++ tokenizer)
#      on 0136's 8 texts -- near bit-level, since it is the same datapath.
#  (b) CPU side: run_mteb.py's build("cpu") (repaired ST) must sit at the
#      known ~2-4e-04 1-cos against the NPU vectors. An unrepaired baseline
#      measured 9.0e-02 (tasks/0136 problem 1), so this discriminates.
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "experiments" / "m8-npu-vs-cpu"))

texts = (SCR / "gte_in.txt").read_text(encoding="utf-8").splitlines()
texts = [t for t in texts if t.strip()]
cli = np.frombuffer((SCR / "gte_cli_out.f32").read_bytes(),
                    dtype=np.float32).reshape(len(texts), 768)

# (a) the MTEB NPU bridge
from npu_encoder import NpuEncoder
enc = NpuEncoder(model="gte-multilingual-base",
                 artifacts="artifacts_nomic_bfp16", threads=24)
npu = enc.encode(list(texts))
d = np.abs(npu - cli).max()
cos = 1.0 - (npu / np.linalg.norm(npu, axis=1, keepdims=True)
             * cli / np.linalg.norm(cli, axis=1, keepdims=True)).sum(axis=1)
print(f"(a) encode-file bridge vs shipped CLI --embed: max|d| {d:.3e}, "
      f"worst 1-cos {cos.max():.3e}")

# (b) the repaired CPU side, exactly as run_mteb.py builds it
import torch
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(str(REPO / "models" / "gte-multilingual-base"),
                        device="cpu", trust_remote_code=True,
                        model_kwargs={"torch_dtype": torch.float32})
sys.path.insert(0, str(REPO / "reference"))
from make_goldens_gte import repair_rotary
repair_rotary(m[0].auto_model)
assert next(m[0].auto_model.parameters()).dtype is torch.float32
m.max_seq_length = 64
ref = m.encode(list(texts), convert_to_numpy=True).astype(np.float32)
ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
nn = npu / np.linalg.norm(npu, axis=1, keepdims=True)
one_cos = 1.0 - (nn * ref).sum(axis=1)
print("(b) repaired-ST CPU side vs NPU, per text 1-cos:")
for i, c in enumerate(one_cos):
    print(f"    text_{i}: {c:.3e}")
print(f"    worst {one_cos.max():.3e}  (expect 2e-04..4e-04; "
      f"unrepaired would read ~9e-02)")
