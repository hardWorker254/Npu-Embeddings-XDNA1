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

# code -> (design directory, long name for messages)
OPS: dict[str, tuple[str, str]] = {
    "gelu": ("gelu", "GELU"),
    "layn": ("layernorm", "LayerNorm"),
    "softm": ("softmax", "softmax"),
}

# Every code, for an error message that has to list them.
CODES: str = ", ".join(OPS)

# The runtime's flag, named here so a message that tells the user what to type
# says the same thing the exporter and the runtime do.
RUNTIME_FLAG = "--npu-ops"
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
                f"[{CODES}] (layn = LayerNorm, softm = softmax, gelu = GELU). "
                f"Nothing listed means all three run on the host, which is the "
                f"measured-faster path."
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
