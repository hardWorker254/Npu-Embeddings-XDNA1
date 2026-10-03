#!/usr/bin/env python3
"""Does the runtime's flag table still describe the flags the runtime reads?

The one bug this gate exists to prevent was a strict "unrecognised option"
refusal in cli.cpp written against the PARSER's own list of flags. The parser
handles about half this binary's options; the rest are read straight off argv by
Runtime::run, run_setup.hpp, run_execute.hpp, run_probes.hpp, vit_mode.hpp and
stt_mode.hpp. A refusal that asks the wrong question rejects 21 flags that had
worked for years -- `--dev npu1`, `serve --json`, `--top-k`, `--classify <png>`,
every `--probe-*`. It failed silently everywhere except verify_pack_parity, which
is the one caller that passes `--dev`, so it reached a committed gate before
anyone noticed it was the code that was wrong.

So the flag set lives in one table (runtime/include/cli/flags.hpp) and this gate
is what keeps it true. Three claims, in order of how badly each has bitten:

  1. NO STALE ENTRY. Every flag in the table is one the code actually reads, one
     the parser handles, or one of the two REFUSED-BY-NAME families. A row left
     behind by a renamed flag would make a dead name look accepted, which is the
     opposite of what a reader of this table takes from it.
  2. NO MISSING FLAG. Every flag read off argv appears in the table. This is the
     direction that broke: a new `--probe-*` added to run_probes.hpp and not to
     flags.hpp works fine until someone adds the refusal check, and then it
     rejects itself.
  3. The two tables AGREE on arity where they overlap. cli.cpp knows what each
     flag swallows; the runtime reads `argv[i + k]`. Disagreement means one of
     them is reading the wrong token, which is silent and produces plausible
     numbers.

Deliberately a text-level gate rather than a linked test: the flags live in
scattered `argv[i] == "--..."` comparisons across headers that are included, not
called, so there is no single translation unit to assert against.

Run:  python tools/verify/verify_cli_flags.py
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNTIME = REPO / "runtime"
TABLE = RUNTIME / "include" / "cli" / "flags.hpp"
PARSER = RUNTIME / "src" / "cli" / "cli.cpp"

# Sources scanned for flag reads. build/ is excluded by construction (it is not
# under src/ or include/), and the third-party headers that ship inside the tree
# are excluded by the explicit suffix set.
SUFFIXES = {".cpp", ".hpp", ".h", ".cc"}
SKIP = {"build", ".venv", "third_party", "external"}

# A flag read off argv looks like `argv[i] == "--foo"` or `a == "--foo"`. The
# bare-string form is the common one and is what the table has to cover; the
# multi-token comparisons in run_execute.hpp (`argv[i] == "--transcribe",
# argv[i] == "--transcribe" ? ...`) are caught by the same pattern.
LONG_FLAG = re.compile(r'"(--[a-z0-9][a-z0-9-]*)"')
# The table's rows.
TABLE_ROW = re.compile(r'\{"(--[a-z0-9][a-z0-9-]*)",\s*(\d+)\}')

# The two families that are refused BY NAME rather than accepted. They must be in
# the table or CLI::parse()'s refusal fires first and their far better message
# (which names the replacement) is never reached -- main.cpp runs the parse
# before the subcommand dispatcher, and run_setup.hpp/stt_mode.hpp/vit_mode.hpp
# call both refusals downstream of it.
REFUSED_BY_NAME = {"--npu-eltwise", "--host-gelu", "--host-ln", "--host-sm",
                   "--npu-ops", "--extra-ops"}

# cli/subcommand.cpp has a SECOND list, `flag_takes_value()`'s kWithValue, and
# the two have to agree. This is claim 3 above, and it is not decoration:
# forward_common() (subcommand.cpp line 143) forwards a flag's value only if
# kWithValue says the flag takes one, so a value flag that is in flags.hpp and
# missing from kWithValue is DROPPED ENTIRELY -- not mis-parsed, dropped. Adding
# `--pose-dump` to flags.hpp alone was enough to make
# `pose <model> <img> --pose-dump /tmp/x` try to open /tmp/x as a PNG.
#
# These are exempt, each because the value is swallowed somewhere else. A new
# value flag that is not here and not in kWithValue fails the gate, which is the
# point: the exemption is a place to write the reason, not a place to be absent.
EXEMPT_ARITY = {
    # cli.cpp's own if/else chain swallows these before forward_common() sees
    # them: `else if (a == "--tile-k" && i + 1 < argc)`.
    "--bench", "--model", "--embed", "--add", "--tokenize",
    "--prepare-model", "--source-repo", "--tile-k", "--tile-n",
    # Each is read by its own handler with explicit index arithmetic
    # (`ctx.argv[i + 1]` in run_probes.hpp / run_execute.hpp), or built straight
    # into the store (--transcribe, subcommand.cpp line 242), so no shared list
    # has to know it.
    "--probe-design", "--probe-insts", "--probe-rtp", "--probe-bo",
    "--soak-npu", "--soak-cpu", "--encode-file", "--transcribe",
}

# The second list itself.
SWALLOW = RUNTIME / "src" / "cli" / "subcommand.cpp"


def swallows_value() -> set[str]:
    """The flags in subcommand.cpp's kWithValue."""
    src = SWALLOW.read_text()
    i = src.index("kWithValue")
    body = src[i:src.index("};", i)]
    return set(LONG_FLAG.findall(body))

# Flags the parser itself handles, read off cli.cpp's if/else chain. These are
# legitimately absent from any argv scan and so must not be reported missing.
def parser_flags() -> set[str]:
    src = PARSER.read_text()
    return set(LONG_FLAG.findall(src))


def sources() -> list[pathlib.Path]:
    out = []
    for sub in ("src", "include"):
        for p in sorted((RUNTIME / sub).rglob("*")):
            if p.suffix in SUFFIXES and not (SKIP & set(p.parts)):
                out.append(p)
    return out


# How a line can show that it ACTS on a flag rather than merely mentioning one.
# `--dev` is read as `std::string(argv[i]) == "--dev"`, so the argv pattern
# allows the std::string wrapper; `--bo-mode` is only ever compared with `!=`;
# and the STT/ViT/gemma modes read their flags through helpers (`has`,
# `has_flag`, `flag`, `flag_val`, `value_after`) rather than by comparison, which
# is why a comparison-only scan called all 23 of them stale on the first run.
# What all three share is that the flag is an ARGUMENT to something, which is
# also what separates them from a flag quoted inside an error message.
def reads_flag(line: str, flag: str) -> bool:
    f = re.escape(flag)
    return bool(
        re.search(r'argv[^;]*[!<>=]=\s*"' + f + r'"', line)          # argv scan
        or re.search(r'\ba\s*==\s*"' + f + r'"', line)              # parse loops
        or re.search(r'\b(has|has_flag|flag|flag_val|value_after|'
                     r'flag_arg|swallow|arg_after)\s*\([^;()]*"' + f + r'"',
                     line)                                        # lookup helper
    )


def flags_read_from_argv() -> dict[str, list[str]]:
    """Flags some line acts on, mapped to where."""
    read: dict[str, list[str]] = {}
    for p in sources():
        if p == TABLE:
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue  # prose, not code
            for f in LONG_FLAG.findall(line):
                if reads_flag(line, f):
                    read.setdefault(f, []).append(
                        f"{p.relative_to(REPO)}:{i}")
    return read


def main() -> int:
    if not TABLE.exists():
        print(f"missing {TABLE} -- see BUILD.md")
        return 2

    table_src = TABLE.read_text()
    rows = TABLE_ROW.findall(table_src)
    if not rows:
        print(f"{TABLE.name}: no {{\"--flag\", arity}} rows found -- the table's "
              f"shape changed and this gate no longer knows how to read it")
        return 1
    arity = {name: int(n) for name, n in rows}
    dupes = sorted({n for n, _ in rows if sum(1 for m, _ in rows if m == n) > 1})
    read = flags_read_from_argv()
    parsed = parser_flags()
    known = set(arity)

    print(f"  table        {len(arity)} flags in {TABLE.name}")
    print(f"  argv reads   {len(read)} flags read by a lookup across "
          f"{len(sources())} sources")
    print(f"  parser       {len(parsed)} flags handled by cli.cpp")

    bad = 0

    # 0. A flag with two rows. Harmless to flag_arity() -- the first match wins
    #    -- but it means two people added the same flag and one of them did not
    #    look, which is the question this table exists to answer.
    for f in dupes:
        print(f"  dup    {f}: {sum(1 for m, _ in rows if m == f)} rows in "
              f"{TABLE.name}; flag_arity() returns the first, so the second is "
              f"invisible")
        bad += 1

    # 1. A row the code does not know: either a flag nothing reads, or one of the
    #    two refused-by-name families, which read nothing on purpose.
    unknown = sorted(
        f for f in known
        if f not in read and f not in parsed and f not in REFUSED_BY_NAME)
    for f in unknown:
        print(f"  stale   {f}: in {TABLE.name} but no source reads it")
        bad += 1

    # 2. A flag read off argv with no row: the regression that shipped.
    missing = sorted(f for f in read if f not in known)
    for f in missing:
        sites = ", ".join(read[f][:3])
        print(f"  missing {f}: read at {sites} but not in {TABLE.name} -- "
              f"CLI::parse() would refuse it as unrecognised")
        bad += 1

    # 3. The refused-by-name families, spelled out, because their absence is
    #    silent: the command still fails, just with a worse message.
    for f in sorted(REFUSED_BY_NAME):
        if f not in known:
            print(f"  missing {f}: refused by name (npu_ops_flag.hpp), so it "
                  f"needs a table row to reach its own message -- without one "
                  f"CLI::parse() rejects it first as merely unrecognised")
            bad += 1

    # 4. The two value-tables agree. A flag the table says takes a value, which
    #    kWithValue does not, is dropped by forward_common() -- both the flag and
    #    its value -- so the mode silently falls back to a default. That is the
    #    failure mode claim 3 describes and the one this gate now checks.
    swallow = swallows_value()
    for f in sorted(k for k, n in arity.items() if n >= 1):
        if f in swallow or f in EXEMPT_ARITY:
            continue
        print(f"  dropped {f}: {TABLE.name} says arity {arity[f]} but "
              f"subcommand.cpp's kWithValue does not list it, so "
              f"forward_common() forwards neither the flag nor its value. Add it "
              f"to kWithValue, or to EXEMPT_ARITY here with the reason the value "
              f"is swallowed elsewhere.")
        bad += 1
    for f in sorted(swallow - known):
        print(f"  dropped {f}: in subcommand.cpp's kWithValue but no row in "
              f"{TABLE.name}, so CLI::parse()'s refusal fires before "
              f"forward_common() can use the list")
        bad += 1

    if bad:
        print(f"FAIL -- {bad} flag-table disagreement(s). Add the flag to "
              f"{TABLE.name} with its arity, or delete the row if the flag is "
              f"gone; a removed flag must move to refuse_removed_op_flags() so "
              f"it keeps saying what replaced it. A `dropped` line means the two "
              f"value tables disagree: fix kWithValue in "
              f"runtime/src/cli/subcommand.cpp, and put the reason in "
              f"EXEMPT_ARITY here if the value is swallowed elsewhere.")
        return 1

    print(f"  ok    every flag read off argv is in the table ({len(read)}), and "
          f"every table row is read ({len(known)})")
    print(f"  ok    every value flag reaches forward_common()'s whitelist "
          f"({len([k for k, n in arity.items() if n >= 1]) - len(EXEMPT_ARITY & set(arity))} "
          f"swallowed, {len(EXEMPT_ARITY & set(arity))} exempt with a reason)")
    print("PASS -- the CLI's flag table matches the flags the runtime actually "
          "reads, so the unrecognised-option refusal rejects typos and nothing "
          "else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
