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
# named all three run on the host, and that is where every measurement in this
# repository has put them: on vit-base the array's layernorm is 0.377 s against
# 0.248 s of host and its softmax 0.822 s against 0.244 s. Build only the ops
# you will ask for: `--extra-ops layn,softm`. The exporter and the flag
# therefore agree on one layout: the eltwise directories sit next to
# `gemm_rtp` under the generation root, the same seven-design names the runtime
# already resolved before the unified path existed.
#
# `--extra-ops` is THIS tool's flag and only this tool's. The GEMM exporter has
# no op flag at all -- its eltwise plan is the registry's (tools/lib/npu_ops.py,
# buildable codes), so what is compiled is what the registry says the
# architecture honours, with no command line able to disagree with it.
#
# The array programs are the ones the M5 experiments developed and are kept
# byte-for-byte in kernels/: `gelu_poly.cc`, `layernorm.cc`, `softmax.cc`. This
# tool only wraps them in an IRON program, compiles it once per (op, batch), and
# copies the two artifacts XRT needs out of the JIT cache. The C++ runtime
# compiles nothing (ground rule 3).
#
# Usage:
#   python tools/export/export_eltwise.py --arch all --batch 128 --hidden 384
#   python tools/export/export_eltwise.py --arch 1   --batch 128 --extra-ops layn
#
# The eltwise designs are normally emitted alongside the GEMM set by
# tools/export/export_gemm_rtp.py, which derives that plan from the registry
# rather than from a flag; this tool exists so the three can be rebuilt on
# their own. The op codes are in tools/lib/npu_ops.py, next to the runtime's
# copy of the same table.
#
# This file is the entry point only. The implementation is the package under
# tools/export/exporters/eltwise/.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

# Re-exported on purpose: tools/export/exporters/gemm_rtp/build.py calls
# export_eltwise.export_arch() for the registry's buildable eltwise codes, and
# that call site predates the package split.
from exporters.eltwise.main import export_arch, main  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
