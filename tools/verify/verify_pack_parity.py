# NpuEmbeddings -- do the two model packers agree, byte for byte?
# SPDX-License-Identifier: Apache-2.0
#
# There are two implementations of the .npue layout: tools/pack/pack_npue.py (the
# reference, used at build time) and `npuembed --prepare-model` (C++, so a
# downloaded release can build the container without Python).
#
# Two implementations of one binary layout is a real risk. A disagreement would
# put correctly-sized weights in the wrong order -- exactly the failure
# tasks/0022 hit, which produced rel_fro 1.186 and which no size check can
# catch. So the gate is not a tolerance. It is byte equality.
#
# It has already earned its keep. The first C++ version differed in three
# independent ways, and none would have been found by comparing embeddings:
#
#   1. the source checksum was left empty, so the container could not say
#      which checkpoint it came from;
#   2. the layout descriptor was emitted with sorted keys, while the reference
#      stores insertion order -- a byte-different file with an IDENTICAL
#      layout hash, so the existing guard would have passed it;
#   3. the 1/sqrt(head_dim) fold was computed in double and rounded once at
#      the end, where numpy multiplies a float32 array by a Python float in
#      float32. That double rounding moved 86 of 2304 bias values and one
#      weight by 1 ULP.
#
# Env: iron env (numpy only). Usage:
#   python tools\verify_pack_parity.py

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

# The weight layout, named once: onnx_weights.py is the contract between this
# gate, the packers, CHECKPOINT.json and the runtime's manifest.
from onnx_weights import (MODEL_ONNX,                          # noqa: E402
                          WHISPER_DECODER_ONNX, WHISPER_ENCODER_ONNX)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(p: Path) -> str:
    head = p.open("rb").read(64)
    _, ver, arch, flags, jo, jl, do, dl = struct.unpack("<4sIII QQQQ", head[:48])
    return (f"{p.stat().st_size / 1e6:.2f} MB, json {jl} B at {jo}, "
            f"data {dl / 1e6:.2f} MB at {do}, v{ver} arch{arch} flags{flags}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    # The default used to be a bare "npuembed.exe", which does not exist on
    # any platform this project supports building on except Windows -- so the
    # gate died with FileNotFoundError before comparing anything, on Linux and
    # macOS, every time. Picking the suffix from the host makes `pipeline run
    # verify_pack_parity` mean the same thing everywhere.
    ap.add_argument("--exe",
                    default=str(REPO / "runtime" / "build" /
                                ("npuembed.exe" if os.name == "nt"
                                 else "npuembed")))
    # Tile size stopped being a constant when bge-large arrived: its N in
    # {1024, 3072, 4096} makes tile_n 48 illegal, so it packs at 32. A gate
    # that only ever checked 48 would not have covered the packer's newest
    # parameter -- which is exactly where two implementations drift.
    ap.add_argument("--tile-n", type=int, default=48)
    # The B panel's sub-tile is per-generation, so the two packers can only be
    # compared for a STATED one. Run it for both: the default (npu2) is what
    # every container shipped before this was packed with, and npu1 is the pair
    # its designs actually consume. A packer that quietly kept the old constant
    # on one path passes one of these two and fails the other.
    ap.add_argument("--device", default="npu2", choices=("npu1", "npu2"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    # arch=1 (EmbeddingGemma / Gemma3) checkpoints have no vocab.txt -- the
    # tokenizer travels as gemma_tokenizer.bin (tasks/0061), not a WordPiece
    # vocabulary file -- and pack_gemma() needs the two Dense heads BERT
    # checkpoints do not have. Detected the same way both packers detect it:
    # config.json's own model_type, never the directory name.
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        print(f"missing {cfg_path} -- see BUILD.md")
        return 2
    model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get(
        "model_type")
    is_gemma = model_type == "gemma3_text"
    # arch=2 (nomic-embed-text-v1.5, tasks/0069-0071): RoPE + gated SwiGLU
    # rather than BERT's absolute-position + GELU, but it needs the SAME two
    # files as the BERT family (model.safetensors, vocab.txt) -- no special
    # file list, and neither packer takes a different CLI shape for it: both
    # tools/pack/pack_npue.py's main() and `npuembed --prepare-model` dispatch on
    # config.json's OWN model_type internally, so the subprocess calls below
    # are identical regardless of arch. This flag exists only to make the
    # printed diagnosis say which arch was actually compared.
    is_nomic = model_type == "nomic_bert"
    # What a checkpoint directory must CONTAIN changed with the source: the
    # weights are an ONNX model now (onnx_weights.MODEL_ONNX names the
    # layout), and embeddinggemma's two Dense heads, which used to arrive as
    # sibling 2_Dense/ and 3_Dense/ checkpoints, were fused into its export
    # rather than fetched separately. The tokenizer files are not weights and
    # never were -- `--prepare-model` reads them to build tokenizer.vocab.
    is_whisper = model_type == "whisper"
    need_files = (
        (MODEL_ONNX,)
        if is_gemma else
        (WHISPER_ENCODER_ONNX, WHISPER_DECODER_ONNX, "vocab.json")
        if is_whisper else
        (MODEL_ONNX, "vocab.txt")
    )
    for need in need_files:
        if not (model_dir / need).exists():
            print(f"missing {model_dir / need} -- see BUILD.md")
            return 2

    with tempfile.TemporaryDirectory(prefix="packparity_") as td:
        d = Path(td)
        py_out, cc_out = d / "python.npue", d / "cpp.npue"

        r = subprocess.run([sys.executable,
                            str(REPO / "tools" / "pack" / "pack_npue.py"),
                            "--model-dir", str(model_dir), "--out", str(py_out),
                            "--tile-n", str(args.tile_n),
                            "--device", args.device],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"pack_npue.py failed:\n{r.stdout}\n{r.stderr}")
            return 2

        # --dev is what the C++ packer resolves its sub-tile from: it is the
        # same generation name the design set is built for, read before the
        # pack path runs.
        # --model is not optional in practice: `--prepare-model` refuses to
        # guess when models/*.npue holds more than one, which is the normal
        # state of this tree, so without it the gate never reached the packer
        # on any machine with a second container installed.
        r = subprocess.run([args.exe, "--dev", args.device,
                            "--prepare-model", str(model_dir),
                            str(cc_out), "--tile-n", str(args.tile_n),
                            "--model", model_dir.name],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"npuembed --prepare-model failed:\n{r.stdout}\n{r.stderr}")
            return 2

        a, b = sha256(py_out), sha256(cc_out)
        arch_note = " (arch=1 gemma)" if is_gemma else \
                   " (arch=2 nomic_bert)" if is_nomic else " (arch=0 bert)"
        print(f"  tile_n {args.tile_n}, device {args.device}, "
              f"model {model_dir.name}{arch_note}")
        print(f"  pack_npue.py    {describe(py_out)}")
        print(f"  --prepare-model {describe(cc_out)}")
        print(f"\n  python : {a}")
        print(f"  c++    : {b}")

        if a == b:
            print("\nPASS -- byte-identical")
            return 0

        # Not equal: say WHERE, because "they differ" is not a diagnosis.
        pa, pb = py_out.read_bytes(), cc_out.read_bytes()
        print(f"\nFAIL -- differ ({len(pa)} vs {len(pb)} bytes)")
        _, _, _, _, jo, jl, do, _ = struct.unpack("<4sIII QQQQ", pa[:48])
        n = min(len(pa), len(pb))
        diffs = [i for i in range(n) if pa[i] != pb[i]]
        print(f"  {len(diffs)} differing bytes")
        if diffs:
            first = diffs[0]
            where = ("header" if first < 64 else
                     "json" if first < jo + jl else f"data +{first - do}")
            print(f"  first at {first} ({where})")
            if where == "json":
                s = max(jo, first - 80)
                print(f"    python: ...{pa[s:first + 60].decode('utf-8', 'replace')}")
                print(f"    c++   : ...{pb[s:first + 60].decode('utf-8', 'replace')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
