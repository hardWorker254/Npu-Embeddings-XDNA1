#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Build the three elementwise design directories the encoder needs to keep
# LayerNorm, softmax and GELU on the array instead of the host:
#
#   <out>/artifacts_npu<N>/gelu
#   <out>/artifacts_npu<N>/layernorm
#   <out>/artifacts_npu<N>/softmax
#
# The runtime only touches these when `--npu-ops` names the op; with nothing
# named the unified `gemm_rtp` set runs all three on the host, which is the
# measured-faster path. Build only the ops you will ask for:
# `--extra-ops layn,softm`. The exporter and the flag therefore agree on one layout:
# the eltwise directories sit next to `gemm_rtp` under the generation root, the
# same seven-design names the runtime already resolved before the unified path
# existed.
#
# The array programs are the ones the M5 experiments developed and are kept
# byte-for-byte in kernels/: `gelu_poly.cc`, `layernorm.cc`, `softmax.cc`. This
# tool only wraps them in an IRON program, compiles it once per (op, batch), and
# copies the two artifacts XRT needs out of the JIT cache. The C++ runtime
# compiles nothing (ground rule 3).
#
# Usage:
#   python tools/export_eltwise.py --arch all --batch 128 --hidden 384
#   python tools/export_eltwise.py --arch 1   --batch 128 --extra-ops layn
#
# The eltwise designs are normally emitted alongside the GEMM set by
# `tools/export_gemm_rtp.py --npu-extra-ops CODES`; this tool exists so they can
# be rebuilt on their own. The op codes are in tools/npu_ops.py, next to the
# runtime's copy of the same table.
#
# This file is the entry point only. The implementation is the package under
# tools/exporters/eltwise/.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Re-exported on purpose: tools/exporters/gemm_rtp/build.py calls
# export_eltwise.export_arch() when --npu-extra-ops is given, and that call
# site predates the package split.
from exporters.eltwise.main import export_arch, main  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
