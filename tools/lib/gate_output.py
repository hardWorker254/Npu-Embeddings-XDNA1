# SPDX-License-Identifier: Apache-2.0
"""Where a gate writes its result, when nobody passed --out.

THREE GATES, THREE DIRECTORIES, is how this started. Each defaulted its output
into a directory named after the task that had asked for it --
`tasks/0121-semantic-gate/`, `tasks/0132-t51-tail-gate/`,
`tasks/0037-m9-tiers-endpoint/` -- and so:

  * `tasks/` looked load-bearing. It was not: three of its four files were gate
    output and one was a real calibration corpus (`corpus_520.txt`, read by
    pack_npue.py as gemma's default), which is the worst place for a build
    input, because a directory named after a work log reads as disposable and
    deleting it breaks the packer;
  * each of the three needed its own `.gitignore` line, by full path, and the
    list had to be edited whenever a gate was added or renamed;
  * a reader could not tell, from the tree, which JSONs were inputs to something
    and which were last Tuesday's measurements.

So: ONE directory, named for what it holds rather than for who asked, holding
the three gates' JSON, entirely gitignored. Each gate still takes --out for the
cases where a caller wants the file somewhere specific, which is the only reason
this exists at all -- the default has to be one place, or "where did that gate
write" is a question with three answers.

Not in runtime/ because that is source; not in models/ because that is what the
runtime serves; not in tools/ because a JSON that appears there looks like part
of the tool rather than something it produced.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The one default. Created on write, not here -- a directory that exists only
# because a module was imported is its own kind of litter.
GATE_OUT = REPO / "gate_output"


def gate_path(name: str) -> Path:
    """Where a gate's result goes, by file name.

    `name` is a file name and not a path: these are outputs, and a caller that
    can write them anywhere can write them over the repository. Callers pass a
    constant, so the check is here to make the mistake loud if one is made.
    """
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError(
            f"gate_path takes a file name, not a path: {name!r}. A gate's "
            f"default output goes in {GATE_OUT}; pass --out for anywhere else.")
    return GATE_OUT / name
