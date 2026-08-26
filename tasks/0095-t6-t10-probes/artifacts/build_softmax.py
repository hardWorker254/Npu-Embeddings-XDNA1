# T10 static-cost probe: compile softmax.cc (compile-only, no NPU execution)
# to get an object/ELF containing BOTH softmax_bf16 (aie::exp2) and
# softmax_poly_bf16 (exp2_poly) so they can be objdump'd side by side from
# the SAME compilation unit and flags.
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-eltwise")
import importlib
sk = importlib.import_module("softmax_kernel")

import aie.iron as iron
from aie.iron.device import from_name

iron.set_current_device(from_name("npu2", n_cols=None))

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
spec = sk.sm_array.specialize(rows=256, n_cols=1, variant="lib", stack=0xD00)
spec.compile(xclbin_path=str(out_dir / "sm.xclbin"), inst_path=str(out_dir / "sm.insts.bin"))
print("BUILD OK ->", out_dir)
