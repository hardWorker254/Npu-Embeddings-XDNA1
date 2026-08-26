# T8 reproduction: note 0007 sec 3.2's degree-8 vs degree-7 error numbers,
# on real L0.ffn_up activations, monomial basis u in [0,4], R=4.
# Reuses design_gelu_poly.py's own fit()/eval_poly_f32()/to_bf16() so the
# methodology is identical to what shipped gelu_poly.cc's coefficients.
#
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" repro_deg7.py

import sys
from pathlib import Path

REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
sys.path.insert(0, str(REPO / "experiments" / "m5-eltwise"))
sys.path.insert(0, str(REPO / "reference"))

import numpy as np
from design_gelu_poly import fit, eval_poly_f32, to_bf16
from encoder import gelu as gelu_exact
from safetensors_io import load

taps, _ = load(REPO / "reference" / "goldens" / "minilm_l6_s64_taps.safetensors")
x = taps["L0.ffn_up"].reshape(-1)
want = taps["L0.gelu"].reshape(-1).astype(np.float64)
x16 = to_bf16(x)

rel = lambda g: float(np.linalg.norm(np.asarray(g, np.float64) - want) / np.linalg.norm(want))
floor_bf16 = rel(to_bf16(gelu_exact(x16)))
print(f"bf16 output floor (exact erf, bf16 out): {floor_bf16:.3e}")

R = 4.0
print(f"\n{'degree':>6} {'A: rel_fro incl bf16 round':>28} {'B: rel_fro fp32 (design limit)':>32} {'B / floor':>10}")
for degree in (5, 6, 7, 8, 9, 10):
    coef = fit(degree, R)
    xf = x16.astype(np.float32)
    u = np.minimum(np.abs(xf), np.float32(R))
    acc = eval_poly_f32(coef, u)
    g_fp32 = (np.maximum(xf, np.float32(0.0)) + acc).astype(np.float32)
    rA = rel(to_bf16(g_fp32))              # full pipeline incl. bf16 output rounding
    rB = rel(g_fp32.astype(np.float64))    # "design limit" -- CPU model vs golden, fp32 out
    print(f"{degree:>6} {rA:>28.3e} {rB:>32.3e} {rB / floor_bf16:>10.2f}x")
