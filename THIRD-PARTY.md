# Third-party code

This repository is Apache-2.0. Some of its NPU designs and kernels
began as examples from **[MLIR-AIE](https://github.com/Xilinx/mlir-aie)**,
which is **Apache-2.0 WITH LLVM-exception** — a compatible licence that
adds a permission rather than a restriction. Those files keep their
original copyright headers and say what was changed.

Upstream measured the longest contiguous run of shared non-comment lines
against an mlir-aie checkout with `tools/audit_third_party.py` (not present in
this fork). Shared *line counts* are misleading here: every AIE kernel
includes the same headers and calls the same API, so unrelated files score
30–40%. Contiguity separates copying from idiom.

Of the files that survive in this fork, one has a shared run of **8 lines or
more**:

| file | longest shared run | of our | upstream origin |
|---|---:|---:|---|
| `tools/gemm_pretiled.py` | 11 | 574 | `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py` |

Everything else in this repository is original work. Files below the
threshold share only API idiom — the same includes, the same
`aie::` calls, the same `event0()`/`event1()` bracketing — which is
what using a library looks like.

### Copyright

> Portions Copyright (C) 2024–2026 Advanced Micro Devices, Inc.  
> SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

See each file's header for the specific origin and the changes made.
