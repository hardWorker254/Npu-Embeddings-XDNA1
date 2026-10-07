"""Store a compiled design set inside the container it belongs to.

WHY
---
A design set is 288-536 KB of instruction streams plus a ~71 KB xclbin, and
building one needs MLIR-AIE -- a Python toolchain the size of a compiler, in a
.venv, on the build machine. Without embedding, "download the model and run it"
is false: the reader has to install a compiler before the binary it already has
can dispatch anything. That is a compiler as a RUNTIME dependency, which is
backwards.

With embedding, one .npue is self-sufficient and the toolchain is a build-time
dependency, which is the only place a compiler belongs.

WHAT IT COSTS, MEASURED
-----------------------
    all-MiniLM-L6-v2    4 sets   332 KB      against a 69 MB container
    bge-*               4 sets   332 KB      against 90-732 MB
    whisper-*           1-2 sets 532 KB      against 120 MB - 3.1 GB
    nomic / gte         3 sets   288 KB

So between 0.05% and 0.4% of the container. The weight payload is what costs
bytes; the thing that costs an afternoon is a compiler.

WHY THE DATAPATH DECIDES, AND WHY THAT IS NOT OPTIONAL
------------------------------------------------------
An int8 container's operands are int8 panels and a bf16 design cannot read them.
The runtime already refuses a mismatched design set -- design_fits() compares
want_datapath against the set's own record -- and the sets live in sibling
directories named `<model>` and `<model>-i8`. Embedding therefore EMBEDS THE SET
THAT MATCHES THE CONTAINER, and embeds only that one: a bf16 container carrying a
dead int8 set would be a second false claim inside a file whose whole purpose is
to be one thing, right.

The generation (npu1 / npu2) is recorded rather than resolved here. Nothing has
ever built an npu2 set in this tree, and a container cannot carry two
generations' xclbins under one set name, so the embedded set is npu1 and a
reader on npu2 gets the same refusal it gets today from a directory -- which is
the point: embedding adds a source, it does not add a new way to be wrong.
"""
import json
import os

from npue import DESIGN_ROLE, Writer

# The set names the runtime asks for by name. A container that lacks one of these
# is not incomplete -- it is a container whose reader has that set on disk, and
# the runtime falls back. Listed so the summary can say what was embedded without
# the packer knowing which of them a given architecture uses.
GEMM_SET = "gemm_rtp"


def candidate_dirs(model_name, datapath, artifacts_root):
    """Where this container's design set lives, best first.

    `datapath` is the container's OWN, and it picks the directory: "i8" is the
    int8 set, anything else the bf16 one. Mirrors the runtime's own
    `model_set_candidates`, which looks at `{model_name, model_name + "-i8"}` and
    then filters by datapath -- this looks in the same places with the same
    spelling, because the two lists drifting apart would mean a container carries
    a set the runtime never looks for.
    """
    roots = [artifacts_root] if artifacts_root else []
    names = [model_name] if datapath != "i8" else [model_name + "-i8"]
    # And the other spelling as a fallback, so an int8 container packed from a
    # tree whose sets are all under the unsuffixed name still finds one.
    names += [model_name + "-i8"] if datapath == "i8" else [model_name]
    out = []
    for r in roots:
        for n in names:
            for gen in ("artifacts_npu1", "artifacts_npu2"):
                p = os.path.join(r, n, gen)
                if os.path.isdir(p) and p not in out:
                    out.append(p)
    return out


def find_design_dir(model_name, datapath, artifacts_root, explicit=None):
    """The first candidate directory that actually holds a gemm_rtp set, or None.

    `gemm_rtp` is the test and not "any subdirectory": a half-built set, or a
    directory holding only the elementwise designs, would otherwise be accepted
    and produce a container that cannot run a single GEMM.
    """
    cands = [explicit] if explicit else candidate_dirs(model_name, datapath,
                                                       artifacts_root)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, GEMM_SET, "final.xclbin")):
            return c
    return None


def embed_design_sets(writer: Writer, design_dir, device="npu1", datapath="bf16"):
    """Store every set under `design_dir` in `writer`, and record what went in.

    Returns a summary dict, or None if there was nothing to embed. The config
    keys it writes are what the runtime reads to decide whether to look INSIDE
    the container before it looks on disk, and they are also what a gate asserts
    against -- so the record is written here, once, rather than assembled by the
    packer from its own arguments.

    FILES ARE STORED VERBATIM. An xclbin is a loaded binary and an instruction
    stream is a word array; neither is normalised, tidied or re-serialized,
    because a design set that is byte-identical to the one on disk is the only
    version of this that can be checked against one.
    """
    if not design_dir or not os.path.isdir(design_dir):
        return None
    sets = sorted(d for d in os.listdir(design_dir)
                  if os.path.isdir(os.path.join(design_dir, d)))
    if not sets:
        return None
    stored = {}
    total = 0
    for set_name in sets:
        sdir = os.path.join(design_dir, set_name)
        files = sorted(f for f in os.listdir(sdir)
                       if os.path.isfile(os.path.join(sdir, f)))
        if not files:
            continue
        n = 0
        b = 0
        for f in files:
            with open(os.path.join(sdir, f), "rb") as fh:
                blob = fh.read()
            writer.add_design(set_name, f, blob)
            n += 1
            b += len(blob)
        stored[set_name] = {"files": n, "bytes": b}
        total += b
    if not stored:
        return None
    writer.config["artifacts_embedded"] = "true"
    writer.config["artifacts_device"] = device
    writer.config["artifacts_datapath"] = datapath
    writer.config["artifacts_sets"] = json.dumps(
        {k: {"files": v["files"], "bytes": v["bytes"]} for k, v in stored.items()},
        separators=(",", ":"))
    return {"dir": design_dir, "device": device, "datapath": datapath,
            "sets": stored, "total_bytes": total}


def design_sets_in(reader):
    """What a container carries, straight from its manifest. The gate's own view.

    Deliberately NOT the packer's summary: a gate that read back what the packer
    said it wrote would agree with a packer that wrote nothing.
    """
    sets = reader.design_sets()
    out = {}
    for name, files in sets.items():
        out[name] = {"files": len(files),
                     "bytes": sum(f["nbytes"] for f in files.values())}
    return out


def carries_design(reader):
    """True when this container holds a gemm_rtp set, i.e. it can dispatch a GEMM.

    Checks the SET rather than the config flag, for the reason above: the flag is
    what the writer claimed, and this is what the file actually contains.
    """
    return GEMM_SET in reader.design_sets()


__all__ = ["DESIGN_ROLE", "GEMM_SET", "candidate_dirs", "find_design_dir",
           "embed_design_sets", "design_sets_in", "carries_design"]
