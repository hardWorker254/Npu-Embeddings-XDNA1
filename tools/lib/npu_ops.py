# NpuEmbeddings -- the vocabulary for "which ops go on the array".
# SPDX-License-Identifier: Apache-2.0
#
# WHY A TABLE AND NOT A LIST OF DIRECTORIES
# -----------------------------------------
# The code has to be named on BOTH sides of the build: the runtime's
# `--npu-ops` says which ops it wants at RUN time, and the EXPORTER uses the
# registry at the bottom of this file to decide which designs to build, with no
# flag of its own. A code that meant one directory on one side and another on
# the other would produce a run that asks for an op and cannot find it -- so the
# code, the design directory and the kernel family are one row.
#
# The codes are short because they are typed on a command line and the long
# names run to nine characters: `layn`, `softm`, `gelu`. The runtime's default
# is the EMPTY set -- every op on the host -- which is the measured-faster path,
# so the flag is an opt-in and nothing changes for anyone who does not pass it.
#
# WHY THIS IS WRITTEN TWICE
# -------------------------
# The exporter is Python and the runtime's parser is C++
# (runtime/include/common/npu_ops_flag.hpp), and neither can include the other.
# Each file points at the other. The failure mode of the duplication is a
# refusal by name on both sides -- an unknown code throws and lists the valid
# ones, and a missing design directory is refused with the command that builds
# it -- not a wrong number.
#
# Env: stdlib only.

import sys

# code -> (design directory, long name for messages)
#
# A row with an EMPTY directory is a code that has no sibling xclbin to build,
# and the two of them are empty for different reasons -- so the reason is a third
# field, not a template:
#
#   `conv`  runs on the encoder set's own [rows, d, d] stream (attn_out's
#           shape), which every export already has.
#   `attn`  ADDS two streams (attn_qk, attn_av) to the same set, so the
#           exporter has work to do -- in the GEMM set, not in a directory of
#           its own. What it does NOT do is compile anything here.
OPS: dict[str, tuple[str, str]] = {
    "gelu": ("gelu", "GELU"),
    "layn": ("layernorm", "LayerNorm"),
    "softm": ("softmax", "softmax"),
    "conv": ("", "conv1d (Whisper's audio front end)"),
    # The name is model-neutral on purpose. It used to read "Whisper's
    # attention, as GEMMs", which was true while only Whisper had an array
    # branch; the same NpuAttention now serves BERT, ViT and gemma too, and a
    # status line under "Image classification" that announces Whisper's
    # attention describes the kernel's provenance rather than the model that
    # is running. Provenance belongs in the cell's reason, where every one of
    # them says "the shared NpuAttention"; the long name has to be something
    # the line above it can honestly print for all five architectures.
    "attn": ("", "attention, as two GEMMs"),
    "mproj": ("", "Whisper's mel filter bank, as a GEMM"),
    "fft": ("", "Whisper's 400-point transform, as a GEMM"),
    "logit": ("", "the vocabulary projection, as a GEMM"),
}

# Why a code with no design directory has none. One line per code, printed when
# the code is skipped: a requested op that quietly did nothing is the failure
# this project treats as worst, and a line that says WHY keeps the fact honest.
NO_DESIGN: dict[str, str] = {
    "conv": "it runs on the stream gemm_rtp already exports, so there is "
            "nothing to compile",
    "attn": "its two streams are added to the gemm_rtp and gemm_rtp_dec sets "
            "themselves, not as a directory of their own",
    "mproj": "its stream is added to the gemm_rtp set itself, not as a "
             "directory of its own",
    "fft": "its stream is added to the gemm_rtp set itself, not as a directory "
           "of its own",
    "logit": "its streams are added to the gemm_rtp_dec set itself, not as a "
             "directory of its own",
}

# Every code, for an error message that has to list them.
CODES: str = ", ".join(OPS)

# ONE SPELLING. The runtime's flag SELECTS and never builds: the exporter
# builds every design the model can use (see THE REGISTRY at the bottom), and
# this flag is the only thing that says which of those to RUN. The name the
# exporter's copy used to carry -- `--npu-extra-ops` -- was the longer one,
# because it had to distinguish itself from the runtime's `--npu-ops`. There is
# only one flag left, so the longer spelling is gone: `--npu-ops` is what the
# runtime takes, and `--npu-extra-ops` is refused BY NAME. RUNTIME_FLAG and
# EXPORTER_FLAG are kept as two names because two different programs print them
# and neither should import the other's string by accident -- but they are the
# same text, and a message that tells the user what to type says what the
# runtime actually takes.
RUNTIME_FLAG = "--npu-ops"
EXPORTER_FLAG = RUNTIME_FLAG

# Flags the GEMM exporter used to take, or that mean "select ops" and so belong
# to the runtime only. Every one is refused BY NAME there, for the reason each
# stopped existing, rather than left to argparse's "unrecognized arguments" --
# which names the flag but neither why it is gone nor what to type now. Kept here
# beside the flag they used to share, because the reason is the same fact.
RETIRED_EXPORTER_FLAGS: tuple[str, ...] = (
    "--npu-extra-ops",  # the exporter's former spelling of --npu-ops
    "--npu-ops",        # the runtime's flag; selects, does not build
    "--npu-eltwise",    # built all three eltwise designs or none
)

# The one thing a spawned per-architecture child cannot work out for itself,
# passed down through the environment instead of argv.
#
# The child is handed the geometry as explicit flags and --target deliberately is
# NOT among them, so that its resolve cannot disagree with the parent's. The
# cost of that is that anything which depends on WHICH MODEL this is has to come
# from somewhere else -- and "which designs does this model honour" is exactly
# that. An environment variable rather than a flag, because a flag would put the
# decision back into the command line, where the user used to be able to get it
# wrong.
#
# An EMPTY value is a real answer (gemma honours none of the eltwise codes), so
# consumers test for presence, never for truthiness.
ENV_OPS = "NPUEMBEDDINGS_OPS"


def parse_ops(listing: str, flag: str) -> set[str]:
    """Parse a comma-separated op list. Empty components are skipped.

    So "gelu, layn" and " gelu ,layn " both mean {gelu, layn}, a trailing comma
    is not an error, and an empty string is the empty set -- which is the
    default, and means every op on the host.
    """
    out: set[str] = set()
    for item in listing.split(","):
        code = "".join(item.split())
        if not code:
            continue
        if code not in OPS:
            raise SystemExit(
                f"{flag}: '{code}' is not an op this build knows. Valid codes: "
                f"[{CODES}] (layn = LayerNorm, softm = softmax, gelu = GELU, "
                f"conv = Whisper's conv1/conv2). This flag SENDS the named ops "
                f"to the array; the exporter builds their designs on its own, "
                f"from the registry, and takes no such flag. Nothing listed "
                f"means every op runs on the host, which is the measured-faster "
                f"path. The full table is NPU_OPS.md."
            )
        out.add(code)
    return out


def design_of(code: str) -> str:
    """The design directory for a code, which is also the kernel family name."""
    try:
        return OPS[code][0]
    except KeyError:
        raise SystemExit(
            f"'{code}' is not an op this build knows. Valid codes: [{CODES}]"
        ) from None


def parse_exportable(listing: str, flag: str) -> set[str]:
    """parse_ops, minus the codes that have no design directory to build.

    `conv` is one: the runtime runs Whisper's two convolutions on a stream
    gemm_rtp already exports (the encoder set's own [rows, d, d] one), so there
    is nothing to compile for it. The codes are the runtime's codes and the flag
    takes the same list, so refusing half of it would make the shared vocabulary
    useless -- a user who typed `conv` at run time must be able to type it here
    and get an explanation rather than "unrecognised code".

    This is `export_eltwise`'s `--extra-ops` only. The GEMM exporter has no op
    flag at all; it derives its set from the registry (`buildable_codes`) and this
    function exists for the elt tool, which genuinely is asked for a list.

    A skipped code is REPORTED on stderr rather than dropped in silence: a
    requested op that quietly did nothing is the failure this project treats as
    worst, and one line on stderr keeps the build output clean. The report says
    WHY there is nothing to build, so it reads as a fact about the op rather
    than as a warning about the command line.
    """
    ops = parse_ops(listing, flag)
    for code in sorted(ops):
        if not OPS[code][0]:
            print(
                f"{flag}: {code} ({OPS[code][1]}) has no design directory of its "
                f"own -- {NO_DESIGN.get(code, 'nothing to compile')}. Skipping it "
                f"here; the same list at run time is what sends it to the array.",
                file=sys.stderr,
            )
            ops.discard(code)
    return ops


# The codes that change the GEMM set rather than a sibling directory. The GEMM
# exporter asks this which EXTRA STREAMS a set carries, and the runtime asks it
# nothing: at run time the streams are simply in design.json, and whether they
# are there is the export's business.
GEMM_STREAMS: dict[str, tuple[str, ...]] = {
    "attn": ("attn_qk", "attn_av"),
    "mproj": ("mel_proj",),
    "fft": ("dft400",),
    # The vocabulary does not fit in one N -- 51865 columns is 1621 tiles of 32,
    # against the 12 the biggest shipping design uses -- so it is CHUNKED and
    # each chunk is a stream of its own. The count is a property of the design
    # library's limits, not a tuning knob, and it is checked against the
    # container's vocab_size at run time.
    "logit": tuple(f"logits_{i}" for i in range(8)),
}


# ===========================================================================
# THE REGISTRY: what each architecture can do with each code
# ===========================================================================
#
# ONE FLAG, AND EVERY CELL IS ABOUT THE MODEL
# -------------------------------------------
# The runtime's flag is one string with eight codes and one parser
# (runtime/include/common/npu_ops_flag.hpp, which is the second copy of the
# table above and points back here). There are no architecture-specific NPU
# flags anywhere in the tree: the GEMMs are on the array in all four
# architectures and unconditionally, because there is no GEMM code.
#
# What differs is which codes an architecture can HONOUR, and that is a
# property of the MODEL, not of the flag. So this file records all 40 cells
# (5 architectures x 8 codes) with a status and a reason, and three programs
# read it:
#
#   * the EXPORTER builds every design whose cell says `honours` -- see
#     buildable_codes() below and exporters/gemm_rtp/build.py. There is no
#     exporter flag for this any more: one command builds what the model can
#     use, and the runtime's flag is what selects among the built designs.
#   * NPU_OPS.md is the human-readable copy of this table, and the README's
#     "Which architecture honours which code" section is the summary of it.
#   * tools/verify/verify_npu_op_matrix.py checks the runtime against it, so
#     the table cannot drift into prose that the binary contradicts.
#
# Five statuses, and the difference between the last three is the whole reason
# this table exists rather than a sentence in a README:
#
#   honours      the code runs on the array today
#   on_array     the operation is ALREADY dispatched and needs no code. The code
#                is REFUSED rather than accepted, because there is nothing for
#                it to select and accepting it would be a requested op that
#                quietly did nothing. The refusal says the work is already done.
#   unimplemented the model HAS the operation and no code reaches the array:
#                a BRANCH WAS NEVER WRITTEN. Not a property of the model, and
#                not a design the exporter can fix on its own.
#   blocked      the operation exists but cannot be moved on this board, for a
#                stated reason (fused into another op, needs a different kernel
#                and a new code, or does not tile)
#   absent       the model has no such operation. Permanent, and correct.
#
# Measured timings are in the reasons where a measurement exists, because "it
# works" and "it is worth asking for" are different claims and only the first
# one is about the code.

HONOURS = "honours"
ON_ARRAY = "on_array"
UNIMPLEMENTED = "unimplemented"
BLOCKED = "blocked"
ABSENT = "absent"

# The order they are listed in, worst-news-last: what works, what is already
# there, what is missing code, what is impossible, what does not exist.
STATUSES = (HONOURS, ON_ARRAY, UNIMPLEMENTED, BLOCKED, ABSENT)

STATUS_MEANING: dict[str, str] = {
    HONOURS: "runs on the array today",
    # Not "accepts the code": there is nothing FOR the code to select, and
    # accepting it would be the failure this project treats as worst -- a
    # requested op that quietly did nothing. The runtime refuses it and says why
    # the work is already dispatched. (cls/conv is the case: a ViT's patch
    # embedding rides attn_out's instruction slot with no flag at all.)
    ON_ARRAY: "the work is already dispatched; the code has nothing to select "
              "and is refused with that reason",
    UNIMPLEMENTED: "the model has the op; the branch is unwritten",
    BLOCKED: "cannot be moved on this board, for the stated reason",
    ABSENT: "the model has no such operation",
}

# The codes that are a GEMM-shaped op inside an existing design set rather than
# a sibling xclbin, so "build it" means "add streams to the set" and not
# "compile a directory". Kept as a named set because the exporter's two halves
# (resolve.py adds the streams, build.py compiles the directories) have to
# agree about which codes they are.
STREAM_ONLY = frozenset({"conv", "attn", "mproj", "fft", "logit"})

_NO_AUDIO = "an audio front end, and this architecture has none"

# kind -> code -> (status, reason). The four kinds are the ones npu_targets.json
# knows (KNOWN_KINDS in exporters/common/consts.py); "gemm_rtp" is both a kind
# and the default for a text embedder, which has no kind of its own.
#
# model -> the same shape, for the models whose ENCODER differs from its kind's
# default. There is exactly one today and the reason it cannot be expressed as
# a kind is the point of the override: GemmaNpuEncoder reads no per-op flag at
# all, which is a property of that one encoder and not of "embedders".
KIND_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "gemm_rtp": {
        "gelu": (HONOURS,
                 "an ungated FFN's activation is a standalone pass, so the "
                 "gelu/ design takes it over. A GATED FFN (nomic, gte, gemma) "
                 "computes the activation inside the gated path instead and has "
                 "no such pass -- see buildable_codes(), which is where that is "
                 "enforced. Same CAVEAT as the cls row: only kind stt gets the "
                 "exact-erf kernel, so this design is the degree-8 `poly` fit, "
                 "2.49e-3 relative from the host's exact erf. Measured end to "
                 "end on bge-base with the code on, relfro 6.1e-03 -- which is "
                 "why the runtime accepts it and does not call it a refusal."),
        "layn": (HONOURS,
                 "pre-LN: two LayerNorms per layer plus the final one. The "
                 "layernorm/ design is built for THIS model's d_model and "
                 "layer_norm_eps, and the runtime checks both against the "
                 "container it is holding, so a design exported for another "
                 "model is refused by name rather than silently normalising "
                 "with the wrong numbers."),
        "softm": (HONOURS,
                  "softmax over the score matrix, its own pass between the two "
                  "attention GEMMs. MEASURED on bge-base, 15 texts, arch 1: "
                  "1.70 s against 0.16 s of host embedding, relfro 1.15e-02 / "
                  "cos 0.999933 against the host vectors (max|d| 1.6e-03). "
                  "The row WIDTH is the whole of the difficulty here, and it "
                  "was wrong until it was measured: the design is compiled for "
                  "sm_cols = padded n_kv (64 -> 384, resolve.py:414) because "
                  "NpuAttention, its other consumer, pads its score rows to the "
                  "kernel's own width, while the host's qk_impl laid rows "
                  "g_seq wide. elt_chunks cannot catch that -- its only guard "
                  "is `n % cols != 0`, and batch*heads*g_seq*g_seq is a "
                  "multiple of 384 anyway -- so the kernel normalised "
                  "384-element windows, six real score rows at a time, and the "
                  "embedding came back at relfro 7.08e-01 / cos 0.749237895. "
                  "BertEncoder now lays the rows out at the design's width, "
                  "with -1e30 past g_seq, whenever the array takes the "
                  "softmax; the default path is byte-identical to what it was "
                  "and `attn,softm` (relfro 1.21e-02) is unchanged. COST, and "
                  "it is not a defect in this cell but in how the two flags "
                  "combine: asking for attn,softm puts the softmax inside "
                  "NpuAttention, which dispatches it once per head per chunk "
                  "rather than once per layer, and every dispatch fills the "
                  "design's whole 12288-row capacity -- 100.90 s for 15 texts "
                  "against 1.38 s for attn alone."),
        "conv": (ABSENT, "names Whisper's conv1/conv2 -- " + _NO_AUDIO + "."),
        "attn": (HONOURS,
                 "qk() and av() in BertEncoder::run dispatch on the set's own "
                 "attn_qk/attn_av streams through the shared NpuAttention. "
                 "BERT's fused qkv is Q|K|V per position, so the K|V block the "
                 "class wants is reached by passing qkv + d_model with a stride "
                 "of 3*d_model -- the same offset Whisper's own self-attention "
                 "uses, and no new code. The exporter builds the streams from "
                 "the registry like any other honoured code; n_kv comes from the "
                 "target's max_seq_len, which is the context the CONTAINER was "
                 "packed for (all-MiniLM: 256, presliced) and not the "
                 "checkpoint's 512 positions -- reading config.json instead "
                 "would build a panel 512 wide over a tensor that cannot address "
                 "past 256. Padding past the live positions is exact: "
                 "NpuAttention writes -1e30 into those columns before the "
                 "softmax, so a wider n_kv costs arithmetic and not accuracy. "
                 "attn_qk's K is head_dim padded up to tile_k, which is what "
                 "makes a 32-wide head (three of these eight models) buildable at "
                 "all, and the runtime zeroes the matching rows of the gathered "
                 "Q block so the extra MACs contribute 0*0. Same CAVEAT as "
                 "cls/stt: attention on the array is SLOWER than the host pass. "
                 "MEASURED on bge-base, 15 texts, arch 1: 1.38 s against "
                 "0.16 s of host embedding (8.6x SLOWER), relfro 1.58e-02 / "
                 "cos 0.999876, max|d| 2.2e-03 -- against Whisper's 4.6x. This "
                 "code makes the model runnable on the array, not faster."),
        "mproj": (ABSENT, "names the slaney mel filter bank -- " + _NO_AUDIO + "."),
        "fft": (ABSENT, "names the 400-point transform -- " + _NO_AUDIO + "."),
        "logit": (ABSENT,
                  "names Whisper's TIED TOKEN EMBEDDING used as the logit "
                  "matrix (decoder.cpp:196: the checkpoint has no proj_out). "
                  "An embedder stops at its pooling head and carries no such "
                  "tensor: bert_encoder.cpp contains zero occurrences of "
                  "`logit` or `vocab`."),
    },
    "stt": {
        "gelu": (HONOURS,
                 "a Whisper checkpoint declares `activation: gelu` and the "
                 "packer refuses anything else, so the design is the exact-erf "
                 "kernel. The poly one is 2.49e-3 away from it, which is a "
                 "different activation rather than a faster one."),
        "layn": (HONOURS,
                 "pre-LN with this checkpoint's own layer_norm_eps (1e-05, "
                 "inside a square root in kernels/layernorm.cc), built per "
                 "target rather than defaulted."),
        "softm": (HONOURS,
                  "softmax over the score matrix is its own pass; the design's "
                  "row width follows the attention geometry it is built beside."),
        "conv": (HONOURS,
                 "conv1/conv2 run on the encoder set's OWN [rows, d, d] stream "
                 "-- attn_out's shape -- so the code costs no hw_context and no "
                 "extra xclbin. It is the one op that wins on BOTH axes: the "
                 "host's cost grows with d^2 and the array's with the dispatch "
                 "count, which is constant."),
        "attn": (HONOURS,
                 "the two GEMMs that bracket the softmax, as attn_qk and attn_av "
                 "streams in the same set. MEASURED 4.32 s against 0.94 s on "
                 "the host over a 3 s window: it works, and it is 4.6x slower."),
        "mproj": (HONOURS,
                  "the mel bank as one more stream in the encoder set, at "
                  "(201 bins, the model's mel count)."),
        "fft": (HONOURS,
                "the front end's 400-point transform as one more stream in the "
                "encoder set, at (400, 201)."),
        "logit": (HONOURS,
                  "the tied embedding, transposed and tiled into eight chunk "
                  "streams in the DECODER set; 39 MB of staged panels. Chunked "
                  "because 51865 columns is 1621 tiles of 32 against the 12 the "
                  "largest shipping design uses."),
    },
    "cls": {
        "gelu": (HONOURS,
                 "pre-LN with an ungated FFN, so the activation IS a standalone "
                 "pass and the gelu/ design takes it over. MEASURED on "
                 "vit-base-patch16-224, bf16, bus.jpg, best of 8: 0.343 s "
                 "against the host's 0.248 s, 1.38x SLOWER. CAVEAT, and it is "
                 "about accuracy rather than the code: resolve.py forces the "
                 "exact-erf kernel only for kind stt, and a ViT keeps the "
                 "exporter's `poly` default, so this design is the degree-8 fit "
                 "-- 2.49e-3 relative from the exact erf the host computes. The "
                 "measured drift on the label set was under 0.011 in "
                 "confidence, which is why it is a caveat and not a refusal. "
                 "A target wanting the exact kernel has to say so in its "
                 "overrides; nothing in the tree does."),
        "layn": (HONOURS,
                 "25 sites (2 per layer plus the final one), the same kernel. "
                 "MEASURED 0.377 s against 0.248 s: 1.52x slower."),
        "softm": (HONOURS,
                  "the softmax is INSIDE attention, so it is named here as the "
                  "array's half of that pass, and it is MEASURED on its own: "
                  "vit-base-patch16-224, bus.jpg, arch 5, `--npu-ops softm` "
                  "puts the encoder at 0.822 s against the host's 0.244 s "
                  "(61 dispatches, 49 of them the GEMMs and 12 of them one "
                  "softmax per block), against 0.349 s for `attn` alone. "
                  "Correctness is checked on the full 1000-way probability "
                  "row, not on the top-5 that would have agreed anyway: "
                  "relfro 1.959e-02, cos 0.999869, max|d| 1.02e-02, with the "
                  "label and the whole top-5 identical to the host and the "
                  "top-10 centered-logit shift under 5.3e-02. The design's "
                  "width is the padded key count (197 -> 384 against "
                  "lcm(tile_k, tile_n)=192), because npue::whisper::attention "
                  "-- which IS this model's host path -- lays score rows out "
                  "at the kernel's own width, so a whole row fits one "
                  "dispatch rather than 64 columns of it. COST of asking for "
                  "both codes at once, stated because it is the one result a "
                  "reader would not guess: `attn,softm` is 6.685 s. "
                  "NpuAttention then calls the softmax once per head per "
                  "chunk -- 144 extra dispatches -- and each dispatch fills "
                  "the design's whole 12288-row capacity whatever the caller "
                  "needed. The cell honours the flag; the combination honours "
                  "it slowly."),
        "conv": (ON_ARRAY,
                 "the patch embedding IS a convolution, Conv2d(3, d, "
                 "kernel=16, stride=16), and it is ALREADY on the array. im2col "
                 "makes it [n_patches, patch_dim] x [patch_dim, d], which is "
                 "attn_out's own shape, so it is dispatched on attn_out's "
                 "instruction slot with no flag at all "
                 "(runtime/src/vit/encoder.cpp:267). Asking for conv would add "
                 "no dispatch. The rewrite is exact on one ground only: stride "
                 "equals kernel with padding 0, so the 196 windows neither "
                 "overlap nor skip and im2col is a permutation, not a sum."),
        "attn": (HONOURS,
                 "12 heads over 197 positions, on the set's own attn_qk/attn_av "
                 "streams -- the exporter builds them from this registry entry "
                 "like any other honoured code, and n_kv is the container's "
                 "max_seq_len (197) padded to 384 against lcm(tile_k, tile_n). A "
                 "ViT has no attention of its own: vit/encoder.cpp calls "
                 "npue::whisper::attention(), so the array branch added here is "
                 "the one NpuAttention already provides, reached through the "
                 "same shared host path. MEASURED on vit-base-patch16-224, "
                 "bus.jpg, arch 5: 0.349 s against the host's 0.244 s for the "
                 "whole encoder (1.43x SLOWER), 337 dispatches against 49. "
                 "Correctness on the full 1000-way probability row: relfro "
                 "4.384e-03, cos 0.999993, max|d| 2.1e-03, label and top-5 "
                 "identical to the host, top-10 centered-logit shift under "
                 "1.9e-02 -- the 197-position row, so nothing here rests on "
                 "the seq 64 the older notes stopped at. The cost above 64 "
                 "positions is this number now, measured rather than left "
                 "open."),
        "mproj": (ABSENT, "a mel filter bank is part of an " + _NO_AUDIO + "."),
        "fft": (ABSENT, "a 400-point transform is part of an " + _NO_AUDIO + "."),
        "logit": (BLOCKED,
                  "the classification head is a real [768, 1000] projection, so "
                  "this is not an absent op -- but 1000 is not a multiple of "
                  "tile_n*cols = 48*4 = 192, so no legal B panel of that width "
                  "exists on this array and one cannot be built. Not missing: it "
                  "does not exist on this board. (It is also 0.8% of the image "
                  "cost as a host matvec, so the point is moot for speed.)"),
    },
    "pose": {
        "conv": (HONOURS,
                 "the 72 dispatched convolutions, on the pose stream set's own "
                 "convNNxMM streams (a shim has 16 DMA buffer descriptors, which "
                 "is why there are 14 shapes rather than one). MEASURED 290 ms "
                 "against the host's 150 ms at 640x640: 1.4x SLOWER here, and "
                 "the runtime's refusal message for a container with no design "
                 "set says exactly that."),
        "gelu": (ABSENT,
                 "the activation is SiLU, not GELU, and it is FUSED into the "
                 "convolution's epilogue: Mul(x, Sigmoid(x)) is a graph op the "
                 "packer matches as the activation pattern. There is no "
                 "standalone pass to move."),
        "layn": (ABSENT,
                 "there is no normalisation op anywhere in this network. "
                 "packers/pose.py's GRAPH_OPS is {Conv, Mul, Sigmoid, Add, "
                 "Concat, Split, MaxPool, Resize} and BatchNormalization is "
                 "absent from it, so a graph carrying BN would be REFUSED BY "
                 "NAME -- ultralytics folds BN into the conv weights at export, "
                 "which is why grep finds no BatchNormalization anywhere in "
                 "tools/."),
        "softm": (ABSENT, "no attention, so no softmax."),
        "attn": (ABSENT,
                 "no attention: net.hpp exposes conv/concat/slice/add/maxpool/"
                 "upsample/head and nothing else weighted."),
        "mproj": (ABSENT, "a mel filter bank is part of an " + _NO_AUDIO + "."),
        "fft": (ABSENT, "a 400-point transform is part of an " + _NO_AUDIO + "."),
        "logit": (ABSENT,
                  "no vocabulary: the head is a 1x1 convolution producing one "
                  "class score and a DFL grid, not a projection into a token "
                  "space."),
    },
    # arch=7. Two graphs in one container -- a 87-node palm detector and a
    # 58-node hand-landmark network -- which is why this row is the only one
    # whose `conv` cell is blocked rather than honoured: see it.
    "hands": {
        "conv": (HONOURS,
                 "It dispatches, and it is 1.8x SLOWER here. Both are "
                 "measured, and both belong in the same cell.\n\n"
                 "What changed, and when: this cell used to be `blocked` with the "
                 "reason that no design set carried arch=7's dense (K, N) pairs. "
                 "That was true when it was written and stopped being true when "
                 "tools/export/exporters/gemm_rtp/geometry.py gained "
                 "HANDS_CONV_SHAPES and "
                 "runtime/artifacts/mediapipe-hands/artifacts_npu1/gemm_rtp/ was "
                 "built from it: twelve streams, one xclbin, checked against the "
                 "packed container in both directions by "
                 "tools/verify/verify_pose_streamset.py. --npu-ops conv sends the "
                 "61 dense convolutions there in 149 dispatches; the other 39 "
                 "stay on the host.\n\n"
                 "MEASURED on hand_plain.png, one hand, five runs each: the array "
                 "takes 155 ms against the host's 85 ms. Of the array's own time, "
                 "47 ms is the device's GEMM, 4.5 ms is repacking A into the "
                 "design's padded stride and 3.6 ms is transposing C back -- so "
                 "here the DEVICE is most of it and the host shuffling is small, "
                 "which makes this the clearest case in the file of an array that "
                 "loses on the device rather than on the data movement around "
                 "it. N pads to tile_n*cols = 128 and this network's channel "
                 "counts are 16, 24, 32, 48, 64, 96, 128, 192 and 256, so most "
                 "pad, and the useful 356M dense MACs become far more dispatched "
                 "work.\n\n"
                 "AND THE TWO PATHS DISAGREE BY ABOUT A PIXEL AND A HALF. The "
                 "panels are bf16, so the detector's score moves 1.1e-02 and its "
                 "box 1.08 px, and the landmark network's 21 points land a mean "
                 "1.63 px and at worst 4.71 px from the host's (median 1.72, "
                 "p90 2.45), with depth 0.41 px mean and 2.17 px worst, presence "
                 "within 3.4e-04 and the world points within 0.0023 m. That is "
                 "the same order as arch=6's array path, which agrees with its "
                 "host to 1.1 px, and it is BETTER than arch=8's, where the "
                 "detector's noise is amplified by a rotated crop into 44 px -- "
                 "arch=7's crop is not rotated by an angle the detector "
                 "predicts, so nothing here multiplies the error.\n\n"
                 "39 of the container's 100 convolutions are DEPTHWISE and carry "
                 "77.9M MAC, and a depthwise filter reduces within one channel so "
                 "there is no [M, N] GEMM in it: they stay on the host whatever a "
                 "design set says."),
        "gelu": (ABSENT,
                 "the activations are ReLU, ReLU6 and PReLU, and each is FUSED "
                 "into its convolution's epilogue: packers/hands.py attaches "
                 "`act` to a Conv, a dwconv or an Add and there is no standalone "
                 "pass for any of the three. There is nothing for a gelu/ "
                 "design to take over."),
        "layn": (ABSENT,
                 "no normalisation op anywhere in either graph. "
                 "packers/hands.py's op inventory is {Conv, Add, MaxPool, Pad, "
                 "Resize} for the palm detector and {Conv, Add, MaxPool} for "
                 "the landmark network, and there is no "
                 "BatchNormalization or InstanceNormalization to refuse by name "
                 "-- MediaPipe's checkpoints carry none, which is why the "
                 "depthwise layers are plain convolutions and not "
                 "depthwise-separable normalisation blocks."),
        "softm": (ABSENT,
                  "no attention and no softmax. The nearest thing is the palm "
                  "head's score, and that is a SIGMOID folded into the decode "
                  "(runtime/src/hands/decode.cpp applies it to the logit after "
                  "the graph), not a softmax over a score matrix between two "
                  "GEMMs."),
        "attn": (ABSENT,
                 "no attention in either graph: the landmark net's argmax over "
                 "its 63 heatmap channels is folded into the graph as a "
                 "reshape, and runtime/src/hands/net.cpp walks conv, add, "
                 "maxpool, pad and resize and nothing else weighted."),
        "mproj": (ABSENT, "a mel filter bank is part of an " + _NO_AUDIO + "."),
        "fft": (ABSENT, "a 400-point transform is part of an " + _NO_AUDIO + "."),
        "logit": (ABSENT,
                  "no vocabulary: the landmark head emits 21 screen points, a "
                  "presence and a handedness, and the palm head emits boxes and "
                  "scores. Nothing anywhere in this architecture projects into "
                  "a token space."),
    },
    # arch=8. Two graphs in one container -- a 93-node person detector and a
    # 120-node pose-landmark network -- and one NEW operation, DepthToSpace, which
    # is what builds the detector's three pyramid levels out of one 7x7 map.
    "mppose": {
        "conv": (HONOURS,
                 "It dispatches, and it is 2.4x SLOWER here, and it does NOT "
                 "agree with the host path to within a pixel. All three are "
                 "measured and all three belong in the same cell.\n\n"
                 "The 99 dense convolutions across the two graphs have 53 "
                 "distinct raw (K, N) pairs, which pad the way "
                 "gemm_rtp/geometry.py pads pose's -- K to a multiple of "
                 "tile_k = 64, N to tile_n*cols = 128 -- down to TWENTY-TWO. "
                 "That is the largest stream set in this file, and "
                 "runtime/artifacts/mediapipe-pose/artifacts_npu1/gemm_rtp/ now "
                 "carries it: one design, 22 streams, M = 1024, and "
                 "tools/verify/verify_pose_streamset.py checks it against "
                 "geometry.py, npu_targets.json and the packed container in both "
                 "directions. --npu-ops conv sends those 99 convolutions to the "
                 "array in 447 dispatches; the other 62 stay on the host.\n\n"
                 "MEASURED on docs/bus.jpg, one person, best of several runs: the "
                 "array takes 327 ms against the host's 137 ms for the same "
                 "frame. Of the array's own 327 ms, 134 ms is the device's GEMM, "
                 "56 ms is repacking A into the design's padded stride and 24 ms "
                 "is transposing C back -- so the device is 41 % of it and the "
                 "host shuffling around the dispatch is 80 ms, which is the "
                 "honest reason the array loses here rather than the device "
                 "being slow. The three structural reasons are arch=6's, and "
                 "they apply more strongly: N must be a multiple of 128 so most "
                 "of these channel counts pad (the useful 550.5M dense MACs "
                 "become far more dispatched work), the detector's stem alone is "
                 "12544 output pixels cut into 13 chunks of 1024, and the "
                 "landmark network's 256x256 output is 64 chunks for every one "
                 "of its 54 convolutions.\n\n"
                 "AND THE TWO PATHS DO NOT AGREE. The array's panels are bf16, so "
                 "the detector differs from the host's by 7.7e-04 in score, "
                 "0.48 px in the box and 1.37 px in the keypoints -- ordinary "
                 "bf16 noise, and arch=6's array path agrees with its host to "
                 "1.1 px. Here it is AMPLIFIED, and by the geometry rather than "
                 "by the network: the crop is a square rotated by an angle the "
                 "DETECTOR's two body keypoints decide, so 1.37 px of keypoint "
                 "noise is a different rotation of a 565 px crop, and the "
                 "landmarks that resample worst come out 44 px from the host's "
                 "with a median of 5.1 px and a mean of 8.4. The pose confidence "
                 "moves with it, 0.9422 -> 0.9832.\n\n"
                 "That amplification is NOT the array network being wrong, and "
                 "the way that was established is worth stating because it is the "
                 "only evidence in this cell that separates the two: the OpenCV "
                 "zoo -- an independent implementation -- was fed the ARRAY "
                 "path's detection row and answered conf 0.9809 with bbox "
                 "[148.2, 333.3, 392.7, 885.2], against the array runtime's 0.9832 "
                 "and [144.1, 334.1, 401.5, 890.2], and answered 0.9449 with the "
                 "HOST's detection row, against the host runtime's 0.9422. So "
                 "the detector's bf16 noise accounts for the whole difference, "
                 "and anyone comparing the two paths on this architecture is "
                 "comparing two crops rather than two networks.\n\n"
                 "TWO THINGS A DESIGN SET WILL NOT MOVE. 62 of the container's "
                 "161 convolutions are DEPTHWISE and carry 77.2M MAC, 12.3 % of "
                 "it, and a depthwise filter reduces within one channel so there "
                 "is no [M, N] GEMM in it -- those stay on the host whatever a "
                 "design set says. And this architecture has an op that is not a "
                 "GEMM at all and will never be one: three DepthToSpace steps, "
                 "pure plane copies, which build the detector's 28/14/7 pyramid "
                 "out of one 7x7 map."),
        "gelu": (ABSENT,
                 "the activations are ReLU and ReLU6, and each is FUSED into its "
                 "convolution's epilogue or into the residual add: "
                 "packers/mppose.py attaches `act` to a Conv, a dwconv or an Add "
                 "and there is no standalone pass for either. There is nothing "
                 "for a gelu/ design to take over."),
        "layn": (ABSENT,
                 "no normalisation op anywhere in either graph. The op "
                 "inventories are {Conv, DwConv, Add, MaxPool, Resize, "
                 "DepthToSpace} for the detector -- whose three spatial Pads "
                 "fold into the six Convs that consume them, so they are not "
                 "separate nodes -- and {Conv, DwConv, Add, MaxPool, Resize} for "
                 "the landmark network. There is no BatchNormalization or "
                 "InstanceNormalization to refuse by name: MediaPipe's "
                 "checkpoints carry none, which is why the depthwise layers are "
                 "plain convolutions and not depthwise-separable "
                 "normalisation blocks."),
        "softm": (ABSENT,
                  "no attention and no softmax. The nearest thing is the "
                  "detector's score, and that is a SIGMOID the GRAPH carries -- "
                  "unlike hands, where the palm head's sigmoid is folded into "
                  "the decode and applied to the logit in "
                  "runtime/src/hands/decode.cpp -- because in this container the "
                  "packer records `sigmoid` on the landmark head's confidence "
                  "output and the runtime applies it there. Neither is a softmax "
                  "over a score matrix between two GEMMs."),
        "attn": (ABSENT,
                 "no attention in either graph: the landmark network's five "
                 "outputs are five single convolutions, one of them transposed "
                 "on the way out and one squashed, and "
                 "runtime/src/mppose/net.cpp walks conv, dwconv, add, maxpool, "
                 "resize and d2s and nothing else weighted."),
        "mproj": (ABSENT, "a mel filter bank is part of an " + _NO_AUDIO + "."),
        "fft": (ABSENT, "a 400-point transform is part of an " + _NO_AUDIO + "."),
        "logit": (ABSENT,
                  "no vocabulary: the landmark head emits 39 rows of "
                  "x/y/z/visibility/presence, a confidence, a 256x256 "
                  "segmentation mask and a 64x64x39 heatmap, and the detector "
                  "head emits boxes, four keypoints and scores. Nothing "
                  "anywhere in this architecture projects into a token space."),
    },
}

# Models whose ENCODER differs from its kind's row. gemma is here and not as a
# fifth kind because the difference is one encoder's code, not a family of them:
# everything below would be identical for another RMSNorm+GQA model that DID
# read a per-op flag, and making it a kind would claim a family that does not
# exist.
MODEL_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "embeddinggemma-300m": {
        "gelu": (BLOCKED,
                 "GeGLU computes the activation INSIDE the gated path, between "
                 "ffn_up and ffn_down, so there is no separate pass for a "
                 "hw_context to take over and honouring the code would print "
                 "'on the ARRAY' while moving nothing. Measured rather than "
                 "argued: the runtime once took host_gelu straight off the flag, "
                 "printed the ARRAY line for a design it never opened, and "
                 "returned vectors at relfro 0.000e+00 against its own host "
                 "path (6.1e-03 for the ungated bge-base, same flag)."),
        "layn": (BLOCKED,
                 "gemma normalises with RMSNorm, not LayerNorm: no mean pass, no "
                 "beta, its own rms_norm_eps. kernels/layernorm.cc is "
                 "parameterised only by -DLN_COLS/-DLN_EPS/-DLN_ROWS, so this is "
                 "a different kernel body, a new design kind and a NINTH code. "
                 "It is deliberately not done: that would make the code set "
                 "larger and less uniform -- three of the four transformer "
                 "architectures would carry a norm code and one would not. "
                 "Refused rather than quietly meaning two different operations."),
        "softm": (HONOURS,
                  "GemmaNpuEncoder::attention reads the same per-op flag the "
                  "gemm_rtp row does -- the enc dispatch set is built from this "
                  "registry entry, so asking for softm reaches it and asking "
                  "for attn does too. Softmax is INSIDE attention here, so it is "
                  "the array's half of that one pass: the wide softmax design "
                  "runs over the score chunk NpuAttention has already staged, at "
                  "n_kv = 512 (the sliding window, padded to 576 against "
                  "lcm(tile_k, tile_n)=192) rather than the container's 2048. "
                  "MEASURED on embeddinggemma-300m, arch 1: the attention "
                  "column of the breakdown goes from 21 ms on the host path to "
                  "1497 ms with `softm` -- the host figure is 4 texts (42 ms "
                  "at 8), while softmax itself is flat in the request count "
                  "(1517 ms at 4 texts, 1559 at 8) because the design is paid "
                  "in its padded row width rather than in rows touched -- with "
                  "96 dispatches becoming 96 + 24 softmax ones, at relfro "
                  "6.227e-03 against the host vectors. Asking for "
                  "`attn,softm` at once is the outlier and is stated as "
                  "measured: 18711 ms and 672 + 288 dispatches at 4 texts, "
                  "relfro 6.664e-03, and 37.7 s of wall at 8. The cause is "
                  "one line long -- "
                  "NpuAttention calls the softmax per head per chunk while "
                  "every call fills the design's whole 12288-row capacity -- "
                  "and it is the reason this flag should be wanted for "
                  "correctness on the array rather than for speed."),
        "attn": (HONOURS,
                 "attention() dispatches on the enc set's own attn_qk/attn_av "
                 "streams, but it CANNOT reach Whisper's NpuAttention by "
                 "changing a stride, and the reason is worth recording because "
                 "it is the one place this model's geometry is genuinely "
                 "different rather than a different number. That class takes ONE "
                 "`d_model` and uses it three times over: the offset from a key "
                 "row to its value (d_model), the stride of the output row "
                 "(d_model), and the head count (d_model / head_dim). For BERT "
                 "and Whisper those three are the same number -- K and V are "
                 "each one d_model wide. Gemma is MULTI-QUERY: 3 query heads of "
                 "256 over ONE key/value head of 256, so the K|V half-width is "
                 "256 while the output row is 768 and the head count is 3. One "
                 "parameter cannot carry all three, so the class gains a "
                 "separate kv_width and this model is what needs it. Second "
                 "thing that is this model's own: its attention is BANDED -- "
                 "sliding_window 512, every 6th layer -- while the container is "
                 "packed to 2048. NpuAttention masks a SUFFIX of a score row, "
                 "the padding past the sequence, and a band is not a suffix, so "
                 "n_kv is the window and the runtime REFUSES a sequence longer "
                 "than it rather than computing full attention where the model "
                 "computes local attention. RoPE is applied to the qkv buffer "
                 "before the pass runs, so the array's A operand is the "
                 "post-RoPE activations and nothing has to move it. MEASURED "
                 "here rather than borrowed from Whisper: on "
                 "embeddinggemma-300m, 4 texts, arch 1, the attention column "
                 "of the breakdown goes 21 ms (host) -> 358 ms (`attn`), 96 "
                 "dispatches becoming 672, at relfro 5.735e-03 against the "
                 "host vectors -- 17x slower on this model, where Whisper's "
                 "own attn is 4.6x. The text count is part of the measurement "
                 "because this column scales with the query rows: the same "
                 "run over 8 texts is 708 ms and 1248 dispatches, so match "
                 "the dispatch count before comparing milliseconds. Slower on "
                 "both, so this code makes the model runnable on the array, "
                 "not faster."),
        "conv": (ABSENT, "names Whisper's conv1/conv2 -- " + _NO_AUDIO + "."),
        "mproj": (ABSENT, "a mel filter bank is part of an " + _NO_AUDIO + "."),
        "fft": (ABSENT, "a 400-point transform is part of an " + _NO_AUDIO + "."),
        "logit": (ABSENT,
                  "names the tied token embedding used as the logit matrix; "
                  "gemma_npu_encoder.cpp contains zero occurrences of `logit` "
                  "or `vocab`."),
    },
}


def registry_for(kind: str | None, target: str | None = None
                 ) -> dict[str, tuple[str, str]]:
    """The (status, reason) for every code, for one model.

    `kind` is npu_targets.json's kind, and `target` is its model name: a model
    in MODEL_REGISTRY replaces its kind's row entirely rather than patching it,
    so a cell can never keep a kind's status while carrying a model's reason.
    """
    if target and target in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[target])
    return dict(KIND_REGISTRY.get(kind or "gemm_rtp", KIND_REGISTRY["gemm_rtp"]))


def buildable_codes(kind: str | None, target: str | None = None,
                    gated_ffn: bool = False) -> set[str]:
    """The sibling design directories the exporter must compile for this model.

    This is what replaced the exporter's `--npu-ops`: one command builds
    every design the model can actually use, and the runtime's `--npu-ops`
    selects among the built ones. So the set is derived, never asked for.

    Three filters, each for a different reason, and none of them silent:

      * a cell must say `honours` -- an `unimplemented` cell has no branch to
        open a design from, and an `absent` one has no operation to hold it;
      * the code must name a design DIRECTORY, since `conv`/`attn`/`mproj`/
        `fft`/`logit` are streams inside a set resolve.py already builds;
      * a GATED FFN drops `gelu`, because there is no activation pass to move.
        That one is a property of the architecture rather than of the kind, so
        it is a parameter and not a fifth kind.
    """
    reg = registry_for(kind, target)
    if gated_ffn and reg.get("gelu", ("", ""))[0] == HONOURS:
        return {c for c, (status, _) in reg.items()
                if status == HONOURS and OPS[c][0] and c != "gelu"}
    return {c for c, (status, _) in reg.items()
            if status == HONOURS and OPS[c][0]}


def skipped_codes(kind: str | None, target: str | None = None,
                  gated_ffn: bool = False) -> list[tuple[str, str, str]]:
    """(code, status, why-not-built) for every cell the exporter will not build.

    Returned rather than printed from inside buildable_codes() so the caller
    decides where it goes -- and so the registry gate can assert the same list
    the exporter prints. A design that was not built and not mentioned is the
    failure this project treats as worst, so this list is never optional at a
    call site.
    """
    reg = registry_for(kind, target)
    out: list[tuple[str, str, str]] = []
    for code in sorted(reg):
        status, reason = reg[code]
        if status == HONOURS and not OPS[code][0]:
            why = (NO_DESIGN.get(code, "nothing to compile")
                   + " -- it is a stream inside a design set resolve.py builds")
        elif status == HONOURS and gated_ffn and code == "gelu":
            why = ("this target has a GATED FFN, whose activation is part of the "
                   "gated path between ffn_up and ffn_down rather than a "
                   "separate pass, so there is no host-or-array choice to make")
        elif status == HONOURS:
            why = "built"
        else:
            why = reason
        if why != "built":
            out.append((code, status, why))
    return out


def count_statuses(kind: str | None = None, target: str | None = None
                   ) -> dict[str, int]:
    """How many cells of one row are in each status. For the registry gate."""
    counts = dict.fromkeys(STATUSES, 0)
    for status, _ in registry_for(kind, target).values():
        counts[status] += 1
    return counts
