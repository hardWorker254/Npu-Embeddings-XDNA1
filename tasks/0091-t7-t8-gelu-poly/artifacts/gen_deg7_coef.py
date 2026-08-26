import sys
from pathlib import Path
REPO = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
sys.path.insert(0, str(REPO / "experiments" / "m5-eltwise"))
import numpy as np
from design_gelu_poly import fit

coef = fit(7, 4.0)
print("Horner coefficients (highest power first), degree 7, R=4:")
for i, cf in enumerate(coef):
    print(f"  c[{i}] = {cf: .10e}f   // u^{7 - i}")
