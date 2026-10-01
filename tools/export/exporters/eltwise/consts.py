"""Elementwise-exporter constants: kernel tile geometry and the column ceilings.

MAX_LN_SM_COLS / MAX_GELU_COLS are hard limits, not preferences: LayerNorm and
softmax refuse above 2 AIE columns, and the argparers refuse to offer more, so
an impossible geometry is rejected before aiecc is invoked.
"""

from ..common.consts import REPO

KERNELS = REPO / "kernels"

# Rows of AIE the eltwise programs lay their tiles out on.
HEADS = 12

# GELU is a poly approximation; the kernel is tiled at 1024 elements unless the
# caller asks for the 4096 variant.
GELU_TILE = 1024

MAX_LN_SM_COLS = 2
MAX_GELU_COLS = 8

# Rows one softmax call handles, and the LayerNorm default. These are the
# divisors the exporter checks a design's row count against, so they live here
# rather than inside the program builder: a refusal that quotes a number and a
# program that uses a different one is a refusal nobody can act on.
SM_ROWS_PER_CALL = 64
LN_ROWS_PER_CALL = 16
