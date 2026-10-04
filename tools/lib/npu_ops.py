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
    "attn": ("", "Whisper's attention, as GEMMs"),
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
                  "softmax over the score matrix is its own pass between the "
                  "two attention GEMMs, so the softmax/ design takes it "
                  "as-is."),
        "conv": (ABSENT, "names Whisper's conv1/conv2 -- " + _NO_AUDIO + "."),
        "attn": (UNIMPLEMENTED,
                 "qk() and av() in BertEncoder::run compute QK^T and softmax.V "
                 "on the host with no array branch. The softmax between them is "
                 "already a honours, which is why only these two are missing. "
                 "What it would take: the exporter's attn_qk/attn_av streams, "
                 "which only `kind: stt` builds today, and an n_kv read from "
                 "this model's 256 positions instead of from `frames` -- an "
                 "embedder carries no `frames` key, so the geometry would "
                 "silently fall back to Whisper's 1500."),
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
        "softm": (UNIMPLEMENTED,
                  "softmax is INSIDE attention here. Moving it alone ships the "
                  "whole 197x197 score matrix to the array and back for one "
                  "elementwise pass, and the two GEMMs that bracket it are "
                  "`attn` -- the next cell."),
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
        "attn": (UNIMPLEMENTED,
                 "12 heads over 197 positions. Two things are missing and "
                 "neither is the model: kinds.cls's stream list carries "
                 "qkv/attn_out/ffn_up/ffn_down and not attn_qk/attn_av, and "
                 "npu_targets.json records no attention measurement above seq "
                 "64 while a ViT has 197 -- so the catalogue already marks its "
                 "own entry's throughput UNMEASURED. ViT has no attention of "
                 "its own: vit/encoder.cpp:333 calls "
                 "npue::whisper::attention(), so the host path is already "
                 "shared with Whisper and only the branch is missing."),
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
        "softm": (UNIMPLEMENTED,
                  "GemmaNpuEncoder reads no per-op flag at all, so no code "
                  "reaches it; and softmax is inside attention, which is the "
                  "next cell."),
        "attn": (UNIMPLEMENTED,
                 "attention() at gemma_npu_encoder.cpp:166 is a real host pass "
                 "over the sequence with no array branch -- unimplemented, not "
                 "unsupported. RoPE is applied to the qkv buffer before it "
                 "runs, so the array's A operand would be those post-RoPE "
                 "activations. What it would take is the gemm_rtp row's list "
                 "plus that, and it would not be faster: Whisper's own attn is "
                 "measured 4.6x slower than its host path."),
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
