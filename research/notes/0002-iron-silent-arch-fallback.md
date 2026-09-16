# 0002 — IRON silently compiles for the wrong architecture

*Found during [M2](../../tasks/0003-m2-bf16-gemm/TASK.md), 2026-08-16.*

**Severity: high.** It costs ~5× performance, produces no error, no warning at the user
level, and the run completes with correct results — so nothing prompts you to look.

## The bug

`kernels.mm(...)` (and every other `aie.iron.kernels` entry point) resolves the target
architecture through `kernels._common._detect_arch()`:

```python
def _detect_arch() -> str:
    """Return 'aie2p' or 'aie2' based on the active device.
    Falls back to 'aie2' if no device is currently set.
    """
    try:
        device = get_current_device(probe_runtime=False)   # <-- note the False
        return resolve_target_arch(device)
    except (ImportError, RuntimeError, AttributeError, ValueError):
        _log.warning("_detect_arch: no explicit device or unrecognised device; "
                     "falling back to 'aie2'", exc_info=True)
        return "aie2"
```

`probe_runtime=False` means it will **not** ask the hardware. It requires a device to
have been set *explicitly*. If none was, it raises `RuntimeError`, the handler catches
it, and silently returns `'aie2'` — **NPU1 / Phoenix** — on an NPU2 machine.

The fallback is logged at `warning` level through the `aie` logger, which is not
configured to print by default. In practice you see nothing.

Note the asymmetry that makes this so easy to miss: `iron.get_current_device()` — the
call a user would naturally make to check — **does** probe the runtime and cheerfully
reports `NPU2`. So the device looks right while the kernels are compiled for the wrong
architecture.

## What it costs

Measured on this machine (Ryzen AI 9 HX 370, NPU2/aie2p):

| | `mac_dims` plain | `mac_dims` with `emulate_bf16_mmul_with_bfp16=True` |
|---|---|---|
| **without** explicit device | `(4, 8, 4)` | `(4, 8, 4)` |
| **with** explicit device | `(4, 8, 8)` | `(8, 8, 8)` |

Three separate consequences:

1. **Half the MACs per intrinsic** — bf16 resolves to the aie2 geometry `(4,8,4)` =
   128 MACs instead of aie2p's `(4,8,8)` = 256.
2. **`emulate_bf16_mmul_with_bfp16` becomes a silent no-op**, because
   `_MM_EMULATED_BF16_MAC_DIMS_AIE2P` is only consulted when `arch == "aie2p"`. On this
   workload that flag is worth **5.5×** (see [M2](../../tasks/0003-m2-bf16-gemm/TASK.md)),
   so losing it silently is the expensive part.
3. **The wrong kernel source is compiled** — `_kernel_source()` picks
   `aie_kernels/aie2/mm.cc` rather than `aie_kernels/aie2p/mm.cc`.

## The fix

Set the device explicitly, **before** any `kernels.*` call and before the JIT design
runs:

```python
import aie.iron as iron
from aie.iron.device import from_name

iron.set_current_device(from_name("npu1", n_cols=None))
```

`n_cols=None` matters too: `from_name("npu2")` defaults to **1 column**, not the full
8-column array.

## How to verify, always

Cheap assertion worth putting in every design entry point:

```python
from aie.utils.compile.utils import resolve_target_arch
arch = resolve_target_arch(iron.get_current_device())
assert arch == "aie2p", f"expected aie2p, got {arch}"
print(kernels.mm(dim_m=64, dim_k=64, dim_n=64,
                 input_dtype=bfloat16, output_dtype=np.float32).mac_dims)
# aie2p bf16 -> (4, 8, 8); with bfp16 emulation -> (8, 8, 8)
# if you see (4, 8, 4) you are compiling for NPU1
```

Or check the artifact directly — the compiled object name and the kernel source path in
the JIT cache both reveal the arch.

## Scope

`NPU2=1` in the environment does **not** help; it is consumed by the Makefile flow
(`devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)`), not by `_detect_arch`.

This affects **any** IRON script that uses `aie.iron.kernels` without explicitly setting
a device — including the stock `getting_started` examples, whose measurements should be
treated as NPU1-geometry unless they set the device. That is a plausible partial
explanation for pre-existing traces on this machine looking slower than expected.

`whole_array.py` is not affected: it calls `_device_for(...)` → `set_current_device`
before building the design.
