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
