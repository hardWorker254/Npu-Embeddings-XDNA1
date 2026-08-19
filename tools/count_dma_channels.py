# NpuEmbeddings -- how many DMA channels does each tile actually use?
# SPDX-License-Identifier: Apache-2.0
#
# npu2 budget (CLAUDE.md trap 3b): core tile 2 in / 2 out, mem tile 6 in/6 out.
#
# WHY THIS EXISTS
# ---------------
# tasks/0046 spent a session discovering that B-reuse cannot be added to the
# production GEMM because there is no spare input channel anywhere -- every one
# of the 32 core tiles is at 2/2 and five of eight mem tiles are at 6/6. That is
# a question answerable in a second, BEFORE designing a dataflow that needs one
# more stream, and the compiler's own answer ("number of input DMA channel
# exceeded") names a tile but not a budget.
#
# Reads input_with_addresses.mlir -- the POST-PLACEMENT module. aie.mlir is
# pre-placement (CLAUDE.md trap 7c) and its counts mean nothing.
#
# Usage:
#   python tools/count_dma_channels.py <NPU_CACHE_HOME>/<hash>
import re, sys, pathlib
d = pathlib.Path(sys.argv[1])
txt = (d / "input_with_addresses.mlir").read_text(encoding="utf-8", errors="ignore")
tiles = {m[0]: (int(m[1]), int(m[2]))
         for m in re.findall(r"%(\S+?)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", txt)}

def region(pat, label, cap_in, cap_out):
    rows = []
    for m in re.finditer(pat, txt):
        start = m.end(); depth = 0; i = start; end = start
        while i < len(txt):
            if txt[i] == "{": depth += 1
            elif txt[i] == "}":
                depth -= 1
                if depth == 0: end = i; break
            i += 1
        body = txt[start:end]
        rows.append((tiles.get(m.group(1), ("?", "?")),
                     len(re.findall(r"aie\.dma_start\(S2MM", body)),
                     len(re.findall(r"aie\.dma_start\(MM2S", body))))
    if not rows: return
    print(f"\n  {label}   budget {cap_in} in / {cap_out} out")
    for cr, s2mm, mm2s in sorted(rows):
        flag = "  <-- FULL" if (s2mm >= cap_in or mm2s >= cap_out) else ""
        print(f"    tile {str(cr):>8}   in {s2mm}/{cap_in}   out {mm2s}/{cap_out}{flag}")

region(r"aie\.memtile_dma\(%(\S+?)\)", "MEM TILES", 6, 6)
region(r"aie\.mem\(%(\S+?)\)",         "CORE TILES", 2, 2)
