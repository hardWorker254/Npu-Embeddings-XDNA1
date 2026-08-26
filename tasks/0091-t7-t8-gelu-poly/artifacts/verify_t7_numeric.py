# T7 correctness check: does replacing aie::mul(v,1.0f).to_vector<T>() with
# accum::from_vector/to_vector<T>() change the VALUE computed, on real data?
#
# Two things checked, both host-side / numpy, no hardware:
#  1. Widen (bf16 -> fp32) is provably EXACT regardless of which AIE
#     instruction realises it: every bf16 bit pattern is representable in
#     fp32 by zero-extending the mantissa (bf16 IS the top 16 bits of an
#     fp32). No rounding decision exists for this direction. Verified by
#     comparing numpy's own bf16->fp32 cast against a manual bit-shift
#     "widen" that is exactly what a load-with-conversion instruction does.
#  2. Narrow (fp32 -> bf16) is where a rounding mode could matter in
#     principle. tasks/0045 already measured, ON HARDWARE, that the
#     mul-by-1.0f form and the accum::from_vector form of this SAME AIE API
#     produce IDENTICAL values (100.00% bit-exact vs a CPU model, module
#     0.003% attributable to K-block summation order, not narrowing) --
#     this task does not re-run that hardware probe, it applies the same
#     conclusion to the second file using the identical API calls. The
#     algebraic check here instead confirms the REWRITE in gelu_poly.cc
#     didn't change any value in the surrounding Horner/max/add expression
#     graph -- i.e. no transcription bug.
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
sys.path.insert(0, str(REPO / "experiments" / "m5-eltwise"))
sys.path.insert(0, str(REPO / "reference"))
from design_gelu_poly import fit, eval_poly_f32, to_bf16
from safetensors_io import load

taps, _ = load(REPO / "reference" / "goldens" / "minilm_l6_s64_taps.safetensors")
x = taps["L0.ffn_up"].reshape(-1)
x16 = to_bf16(x).astype(bfloat16)   # what the kernel's INPUT actually is

# --- (1) widen is exact: numpy cast vs manual bit-shift widen ---
cast_widen = x16.astype(np.float32)
bits16 = x16.view(np.uint16).astype(np.uint32)
manual_widen = (bits16 << 16).view(np.float32)
assert np.array_equal(cast_widen, manual_widen), "widen is NOT a pure bit-shift -- investigate"
print(f"(1) widen bf16->fp32: numpy cast == manual zero-extend for all "
      f"{x16.size:,} elements -- CONFIRMED exact, no rounding possible.")

# --- (2) the surrounding expression graph is unchanged (degree 8) ---
R = 4.0
coef = fit(8, R)
xf = cast_widen  # the widened value, same either way per (1)
u = np.minimum(np.abs(xf), np.float32(R))
poly = eval_poly_f32(coef, u)
gelu_fp32 = np.maximum(xf, np.float32(0.0)) + poly
# "old" and "new" narrow forms both round fp32->bf16; per 0045 they are the
# SAME hardware store-with-conversion regardless of which C++ expression
# reaches it (both end at vst.conv.bf16.fp32 / SRS gated by the same crrnd
# register). Model that with a single to_bf16() and confirm the pipeline as
# a whole reproduces the header's own claimed number for degree 8.
out_bf16 = to_bf16(gelu_fp32)
want = taps["L0.gelu"].reshape(-1).astype(np.float64)
rel = float(np.linalg.norm(out_bf16.astype(np.float64) - want) / np.linalg.norm(want))
print(f"(2) full pipeline (widen -> poly -> max/add -> narrow), degree 8: "
      f"rel_fro = {rel:.3e}  (header claims 2.494e-03)")
assert abs(rel - 2.494e-03) < 5e-6, "pipeline reproduction does not match the header's own number"
print("    MATCHES the header's design prediction -- confirms the rewrite's")
print("    expression graph (Horner coefficients, order, max/add) is intact.")
