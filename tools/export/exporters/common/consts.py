"""Constants shared by the exporters.

Split out of the two original monoliths (export_gemm_rtp.py, export_eltwise.py)
so both halves of the toolchain name the same device table, the same default
sequence length and the same AIE row count.

REPO is the repository root, NOT tools/: these modules live four levels deep
(tools/export/exporters/common/), so the root is parents[4].
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TOOLS = REPO / "tools"
# tools/export/ holds the two entry-point shims; tools/lib/ holds the shared
# modules, which the exporters import by their historic flat names (npue,
# npu_ops, gemm_pretiled, toolchain_provenance). Both roots go on sys.path: the
# first resolves `exporters.` and the `import export_eltwise` that
# gemm_rtp/build.py makes, the second the flat names.
EXPORT = TOOLS / "export"
LIB = TOOLS / "lib"
for _root in (EXPORT, LIB):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# If architecture 2 has another device name in your toolchain, override with:
#   NPU_ARCH1_DEVICE=...
#   NPU_ARCH2_DEVICE=...
ARCH_DEVICES = {
    "1": os.environ.get("NPU_ARCH1_DEVICE", "npu1"),
    "2": os.environ.get("NPU_ARCH2_DEVICE", "npu2"),
}

ARCHES = ("1", "2")

# The B panel's sub-tile, per generation AND per operand dtype. Recorded in
# design.json's b_layout so the runtime's layout_hash check compares like with
# like -- that check is only worth anything if BOTH sides derive the descriptor
# from the same device and the same dtype, and the value is not the same on
# every combination. See npue.MAC_BY_DEVICE for the measurement; this delegates
# there rather than keeping a fourth copy, because a table that is edited in one
# place and read in three is exactly how a design shipped whose every product
# was wrong while every check agreed. `mac_for_arch` is the only accessor.
def mac_for_arch(arch, dtype):
    """(mac_s, mac_t) for `arch` with B operand `dtype` -- npue's table, not a
    copy of it. `dtype` is required: the int8 pair differs from bf16's on npu1,
    so defaulting it is the bug this indirection exists to prevent."""
    from npue import mac_for_device

    a = str(arch)
    if a not in ARCH_DEVICES:
        raise SystemExit(f"unknown arch {arch!r}: {', '.join(ARCHES)}")
    return mac_for_device(ARCH_DEVICES[a], dtype)

DEFAULT_SEQ = 64

# AIE rows per core group: the DEFAULT for the --rows flag, and the value every
# shipping design set was built with. It became a flag because Whisper's decoder
# has M = 1, which rows=4 cannot express (the smallest M would be m*4 = 256).
# gemm_pretiled.pretiled_array takes it as a parameter defaulting to this, so
# nothing already built changes.
AIE_ROWS = 4

# Fallbacks for a fully manual invocation without --target. These mirror the
# historic argparse defaults; with --target the values come from
# tools/data/npu_targets.json instead.
FALLBACK_BATCH = 128
FALLBACK_COLS = 8
FALLBACK_COLS_BY_ARCH = {
    "1": 4,
    "2": 8,
}
FALLBACK_HIDDEN = 384
FALLBACK_M = 32
FALLBACK_K = 32
FALLBACK_N = 48
# How many differing bytes two xclbins of one design set may have and still be
# the same static configuration. The number is the METADATA FOOTPRINT of this
# toolchain, measured: two builds of the same shape differ in six regions -- a
# 16-byte UUID and an 18-byte header at 296, another at 416, the `aie_image`
# block with a build checksum at 1184, a 16-byte UUID at 66736, an
# `"aie_TimeStamp":"1790505983"` JSON field at 69267 and a 32-byte hex UUID at
# 69458 -- and the count lands between 69 and 82 depending on how much of a
# random UUID happens to differ.
#
# 80 was one byte short of the worst case this reaches, so a set whose shapes
# were all fine refused to export. The same inspection says the check cannot see
# a real static difference in the other direction either: the four pre-existing
# encoder streams have genuinely different shapes and sit 69-79 bytes apart,
# because a shape difference is a BUFFER size and the static program does not
# change. 128 leaves room for the UUID variance and is still three orders below
# a program section, so anything it lets through is metadata and not a design.
FALLBACK_IDENTITY_THRESHOLD = 128
# MiniLM's LayerNorm epsilon, and the only fallback that is right without a
# target. It is compiled into kernels/layernorm.cc and sits inside a square
# root, so a Whisper design MUST be exported with 1e-5 instead -- which is what
# a target's layer_norm_eps is for.
FALLBACK_LN_EPS = 1e-12

# tools/data/npu_targets.json -- the geometry source of truth. Lives in
# tools/data/ rather than beside this module so the path does not depend on the
# package depth.
DEFAULT_TARGETS_FILE = TOOLS / "data" / "npu_targets.json"
TARGETS_SCHEMA = 1
KNOWN_DEFAULT_KEYS = {
    "seq", "tile_m", "tile_k", "tile_n", "identity_threshold", "c_bf16",
    # Per-model batch tiering. The arches table supplies the default; a model
    # overrides it when its unit of work is not a text batch -- Whisper's is one
    # audio file, so its tiers are [1] while every embedder's start at 4.
    "batch", "batches",
    # AIE rows per core group. See AIE_ROWS for why it is a knob now.
    "rows",
    # C-path ping-pong depth in AIE row-blocks. See gemm_pretiled._build_design.
    "tb_rows",
}
KNOWN_DATAPATHS = {"bf16", "bfp16"}

# Per-model keys. A CLOSED set, like defaults and overrides: a typo in a geometry
# field must fail here rather than export a design for a shape nobody asked for.
# The stt-only keys are meaningless for an embedder and are checked only when the
# entry's `kind` is "stt".
KNOWN_MODEL_KEYS = {
    "kind", "hidden", "intermediate", "gated_ffn", "qkv_n", "datapath",
    "overrides", "heads", "head_dim", "enc_layers", "dec_layers", "mel_bins",
    "frames", "max_target",
    # The context this model's CONTAINER was packed for, and the width of one
    # attention head -- the two numbers the attn_qk/attn_av streams are built
    # from for every kind that honours `attn` (`frames` covers kind stt, whose
    # window is audio; this covers an embedder's, whose is token positions).
    #
    # It is deliberately NOT the checkpoint's max_position_embeddings. The
    # exporter cannot see the container, and the two genuinely differ: every
    # shipped embedder is packed to 256 positions whatever its config declares
    # (nomic 2048, gte 8192) because the packer preslices, so a target read from
    # config.json would ask for a panel wider than the tensor it addresses.
    # `sliding_window` is gemma's third number and is not optional next to it:
    # that model's attention is BANDED, so n_kv is the window rather than the
    # packed context, and an entry carrying neither cannot be exported honestly.
    "max_seq_len", "sliding_window",
    # The vocabulary size, which is the number of columns the tied-embedding
    # projection is chunked into. It is the CONTAINER's vocab_size, and a target
    # that pins a different one would build chunks for the wrong vocabulary.
    "vocab",
    # The checkpoint's own LayerNorm epsilon. It is compiled into the kernel
    # (kernels/layernorm.cc, inside a square root) and recorded in the design,
    # so a model that differs from the 1e-12 default has to say so here.
    "layer_norm_eps",
    # kind "cls" only: the image front end's own geometry. image_size and
    # patch_size are what fix the position count (image/patch)^2 + 1, and the
    # position count is what an overrides.seq is checked against -- so they are
    # load-bearing here, not documentation. Checked only when kind == "cls".
    "image_size", "patch_size",
}
STT_MODEL_KEYS = {"kind", "heads", "head_dim", "enc_layers", "dec_layers",
                  "mel_bins", "frames", "max_target"}
# "cls" -- an image classifier (arch 5, vit_patch16_prenorm_gelu). It is a KIND
# and not another arch because the array does not know what a patch is: this
# family's four GEMM streams are the embedder's four, at the embedder's shapes,
# and its patch-embedding conv is rewritten as an exact GEMM that rides the
# attn_out stream. What is new is the FRONT END (decode, im2col) and the HEAD,
# and both of those are host work -- so the geometry here is identical to the
# embedder's and only the batches differ.
#
# Note the stream list below is gemm_rtp's verbatim. That is the point of the
# kind: if a cls entry ever needed a stream an embedder does not have, the kind
# would be lying about the claim this file exists to make.
# "pose" is in here because arch 6 is a real target with its own exporter and
# its own stream set, and because the check that a kind HAS an exporter is the
# one that stops a new architecture from being silently treated as an
# embedder -- which is what would happen if a pose entry arrived as a kind with
# no entry here.
KNOWN_KINDS = {"gemm_rtp", "stt", "cls", "pose"}

DEFAULT_CACHE_ROOT = Path.home() / ".npu" / "cache"

