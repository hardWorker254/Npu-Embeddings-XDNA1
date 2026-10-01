#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# One invocation writes a complete, self-describing design set per NPU
# generation. The output convention is one name everywhere:
#
#   <out>/artifacts_npu1/gemm_rtp   (arch 1, device npu1)
#   <out>/artifacts_npu2/gemm_rtp   (arch 2, device npu2)
#
# where <out> defaults to runtime/. That is the same path the README and the
# runtime's --artifacts flag name, so the tool and the documents agree.
#
# With --target <model> the model name becomes a subfolder so several models
# can share one artifacts root without overwriting each other:
#
#   <out>/<model>/artifacts_npu1/gemm_rtp
#   <out>/<model>/artifacts_npu2/gemm_rtp
#
# Usage:
#   python tools/export/export_gemm_rtp.py --target gte-multilingual-base --arch 1
#   python tools/export/export_gemm_rtp.py --arch all --batch 128 --cols 8
#   python tools/export/export_gemm_rtp.py --arch 1   --batch 128 --cols 8
#   python tools/export/export_gemm_rtp.py --arch 2   --batch 128 --cols 8
#
# This file is the entry point only. The implementation is the package under
# tools/export/exporters/, split so the machinery shared with export_eltwise.py (the
# targets file, JIT-cache identity, the output layout, validation) has one home
# instead of one copy per script.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from exporters.gemm_rtp.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
