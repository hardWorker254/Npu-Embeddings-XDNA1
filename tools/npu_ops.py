# NpuEmbeddings -- the vocabulary for "which elementwise ops go on the array".
# SPDX-License-Identifier: Apache-2.0
#
# WHY A TABLE AND NOT A LIST OF DIRECTORIES
# -----------------------------------------
# The op has to be named on BOTH sides of the build: the runtime's
# `--npu-ops` says which ops it wants, and the exporter's `--npu-extra-ops`
# says which design directories to build. A code that meant one directory on one
# side and another on the other would produce a run that asks for an op and
# cannot find it -- so the code, the design directory and the kernel family are
# one row.
#
# The codes are short because they are typed on a command line and the long
# names run to nine characters: `layn`, `softm`, `gelu`. The default is the
# EMPTY set -- every op on the host -- which is the measured-faster path, so a
# flag is an opt-in and nothing changes for anyone who does not pass it.
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
             "directory of their own",
    "fft": "its stream is added to the gemm_rtp set itself, not as a directory "
           "of its own",
    "logit": "its streams are added to the gemm_rtp_dec set itself, not as a "
             "directory of its own",
}

# Every code, for an error message that has to list them.
CODES: str = ", ".join(OPS)

# ONE SPELLING, BOTH SIDES. The exporter's flag BUILDS the design that the
# runtime's flag of the same name SELECTS, and it used to be the other way round
# (runtime `--npu-ops`, exporter `--npu-extra-ops`) -- two names one suffix apart,
# taking the same codes, so the wrong one was silently dropped at run time. Named
# here so a message that tells the user what to type says the same thing the
# exporter and the runtime do.
RUNTIME_FLAG = "--npu-extra-ops"
EXPORTER_FLAG = "--npu-extra-ops"


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
                f"conv = Whisper's conv1/conv2). On the runtime this flag "
                f"SENDS the named ops to the array; on an exporter it BUILDS "
                f"their designs. Nothing listed means every op runs on the "
                f"host, which is the measured-faster path."
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

    `conv` is the one: the runtime runs Whisper's two convolutions on a stream
    gemm_rtp already exports (the encoder set's own [rows, d, d] one), so there
    is nothing to compile for it. The runtime's flag and the exporter's are the
    SAME STRING, and that is the point of sharing it -- one list sends the ops
    to the array and builds their designs -- so the list a user types at run
    time is the list they type here, and refusing half of it would make the
    shared name useless.

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
