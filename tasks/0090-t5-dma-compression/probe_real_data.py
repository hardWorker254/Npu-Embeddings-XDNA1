"""T5 probe: run the mlir-aie dma_compression example's `cmp_only` config
against REAL weight bytes instead of arange.

Does NOT modify anything under C:\\dev\\mlir-aie (read-only reference tree,
CLAUDE.md). Instead this script adds the example directory to sys.path,
imports its `dma_compression` module unchanged, and monkeypatches the
module-level `RATIOED_N` constant before calling the single `@iron.jit`
entry point -- `_linear_tap(RATIOED_N)` reads that name from the module's
own globals at call time (it is not captured as a function default), so
this changes the destination shim BD's programmed length without touching
the source file.

Why RATIOED_N = N (4096, i.e. "assume no compression"), not a guess at the
real compressed length: per the example's own README, "the consumer shim
BD length must match the compressed byte count, or the DMA stalls" --
oversizing hangs, undersizing silently truncates. N is the one value that
is safe to try without already knowing the answer: compression cannot
(by construction, and per the vendor's own naming of the feature) emit
MORE bytes than the raw input, so requesting exactly N is requesting an
upper bound.
  - If the real compressed length is also N (no reduction), the transfer
    completes normally, matches == N, and that IS the measured ratio: 1.00x.
  - If the real compressed length is < N, the shim waits for bytes that
    never arrive and the dispatch times out (ERT_CMD_STATE_TIMEOUT) --
    a bounded, previously-observed-safe failure mode in this project
    (tasks/0054, tasks/0087: no stuck hw_context afterward, `xrt-smi
    examine -r all` clean). This script does not chase the exact ratio
    below N; see tasks/0090/TASK.md for why that was judged not worth
    the additional hardware risk once the structural blocker (shim DMA
    has no compression hardware on AIE2P at all) was confirmed from the
    aie-rt source.

Usage:
    python probe_real_data.py <tile.bin>
One real hardware dispatch per process (fresh @iron.jit cache each run,
so a change to RATIOED_N cannot be served from a stale trace of a
different value).
"""
import os
import sys
import time

import numpy as np

EXAMPLE_DIR = r"C:\dev\mlir-aie\programming_examples\basic\dma_compression"
sys.path.insert(0, EXAMPLE_DIR)

import aie.iron as iron  # noqa: E402
import dma_compression as dc  # noqa: E402

N = dc.N
assert N == 4096

SENTINEL = np.uint32(0xDEADBEEF)


def main():
    if len(sys.argv) != 2:
        print("usage: probe_real_data.py <tile.bin>", file=sys.stderr)
        return 2
    tile_path = sys.argv[1]
    with open(tile_path, "rb") as f:
        raw = f.read()
    assert len(raw) == N * 4, f"{tile_path}: {len(raw)} bytes, want {N * 4}"
    data = np.frombuffer(raw, dtype=np.uint32).copy()

    # The safe upper bound: assume zero compression, so the destination
    # shim BD is sized to the raw length. See module docstring.
    dc.RATIOED_N = N

    in_tensor = iron.tensor(data.copy(), dtype=np.uint32, device="npu")
    out_tensor = iron.full(N, SENTINEL, dtype=np.uint32, device="npu")

    print(f"[{os.path.basename(tile_path)}] dispatching cmp_only with out_tap=N={N} ...", flush=True)
    t0 = time.perf_counter()
    dc.dma_compression(in_tensor, out_tensor, config="cmp_only")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out = out_tensor.numpy()
    matches = int(np.count_nonzero(out == data))
    untouched = int(np.count_nonzero(out == SENTINEL))
    mismatches = N - matches - untouched
    print(
        f"[{os.path.basename(tile_path)}] COMPLETED in {elapsed_ms:.1f} ms  "
        f"matches={matches} mismatches={mismatches} untouched={untouched}"
    )
    if matches == N:
        print(f"[{os.path.basename(tile_path)}] ratio = 1.000x (no reduction -- "
              f"compressed length == raw length == {N * 4} bytes)")
    else:
        print(f"[{os.path.basename(tile_path)}] some compression occurred but the exact "
              f"ratio is NOT determined by this probe (out_tap=N only proves compressed "
              f"length == N when matches==N; a partial/garbled match here would need the "
              f"low-level BD introspection this task did not attempt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
