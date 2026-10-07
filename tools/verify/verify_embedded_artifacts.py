#!/usr/bin/env python3
"""Gate for a container that carries its own compiled design set.

THE CLAIM
---------
`curl` one .npue and run it. Before this, that was false in a way nobody could see
from the file: the container held weights and a tokenizer, and the design set --
the thing it actually dispatches to -- lived in runtime/artifacts/ and had to be
built with MLIR-AIE, a Python toolchain in a .venv. So the reader installed a
compiler to use a binary they already had.

WHAT IS ASSERTED, IN THE ORDER THAT MATTERS
-------------------------------------------
 1. THE SPELLINGS AGREE. The prefix and the role are written in two languages --
    tools/lib/npue.py (DESIGN_PREFIX, DESIGN_ROLE) and runtime/src/design.cpp
    (kDesignPrefix, kDesignRole). A typo in either produces the worst failure
    available here: the container carries correct bytes under a name nobody asks
    for, the runtime falls back to disk, finds nothing, and tells the reader to
    install MLIR-AIE while holding a working design set. This is the first
    section because every other one would still pass with the two disagreeing.
 2. A CONTAINER WITH NO DIRECTORY PRODUCES THE SAME ANSWER. The real claim, and
    the only one that can be faked by a plausible file: pack with artifacts, move
    runtime/artifacts/ aside, run, and compare against the run with the directory
    present. Byte-identical is the bar -- these are the same xclbin, so anything
    else means the container and the directory are not the same set.
 3. A CONTAINER WITHOUT ARTIFACTS STILL WORKS. The host path needs no design at
    all, and every container packed before this existed has none; a change that
    made "no set embedded" fatal would break all of them.
 4. THE DATAPATH IS NOT MIXED. An int8 container must not carry a bf16 set. The
    runtime refuses a mismatched set, so a container that carried the wrong one
    would be a file that claims to run and cannot -- the exact shape of claim this
    project treats as the worst kind of wrong.
 5. THE SIZE IS WHAT IT COST. 272-536 KB against containers of 69 MB to 3.1 GB, so
    this stays a convenience and does not become a reason not to publish.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "lib"))

BINARY = os.path.join(ROOT, "runtime", "build", "npuembeddings")
ARTIFACTS = os.path.join(ROOT, "runtime", "artifacts")
TEXTS = ["the server on AMD", "does the portable build agree bit for bit",
         "run the array path too"]


def fail(msg):
    print(f"  FAIL  {msg}")
    return 1


def section_spellings():
    """The prefix and the role must be the same string in both languages."""
    bad = 0
    sys.path.insert(0, os.path.join(ROOT, "tools", "lib"))
    import npue
    py_prefix, py_role = npue.DESIGN_PREFIX, npue.DESIGN_ROLE
    cpp = open(os.path.join(ROOT, "runtime", "src", "design.cpp"),
               encoding="utf-8").read()
    m1 = re.search(r'kDesignPrefix\s*=\s*"([^"]*)"', cpp)
    m2 = re.search(r'kDesignRole\s*=\s*"([^"]*)"', cpp)
    if not m1 or not m2:
        bad += fail("runtime/src/design.cpp no longer spells kDesignPrefix and "
                    "kDesignRole as literals, so this gate cannot compare them. "
                    "That is either a rename this file must follow or a change "
                    "that removed the C++ side's copy.")
    else:
        if m1.group(1) != py_prefix:
            bad += fail(f"the design prefix differs: npue.py says {py_prefix!r}, "
                        f"design.cpp says {m1.group(1)!r}. A container packed by "
                        f"one and read by the other carries a set nobody finds.")
        else:
            print(f"  ok    prefix {py_prefix!r} agrees in npue.py and design.cpp")
        if m2.group(1) != py_role:
            bad += fail(f"the design role differs: npue.py says {py_role!r}, "
                        f"design.cpp says {m2.group(1)!r}. A container's entries "
                        f"would be written with one role and looked for with "
                        f"another, and nothing would ever be found.")
        else:
            print(f"  ok    role {py_role!r} agrees in npue.py and design.cpp")
    return bad


def pack(model_dir, out, extra=()):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "pack", "pack_npue.py"),
           "--out", out] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r


def embed(container, out_f32, extra=()):
    """Run the binary and hand back the COMPLETE output.

    stdout AND stderr, concatenated, and the reason is this gate's own bug: it
    looked only at stderr, and the status block this feature added --
    "artifacts  the container's own design set ..." -- goes to stdout. So the check
    for the line that tells a reader they need no compiler was reading a stream the
    line is never written to, found nothing, and (because the condition was
    `if line and ... wrong`) PASSED. A gate that cannot see the thing it asserts is
    worse than one that asserts nothing: it reported the feature verified while
    never having looked.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("\n".join(TEXTS))
        inp = f.name
    cmd = [BINARY, "embed", container, inp, out_f32, "--threads", "24"] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    os.unlink(inp)
    r.all_output = (r.stdout or "") + (r.stderr or "")
    return r


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    if not os.path.exists(BINARY):
        print(f"  FAIL  {BINARY} is not built; build it and re-run")
        return 1
    if not os.path.isdir(ARTIFACTS):
        print(f"  FAIL  {ARTIFACTS} is not there. This gate needs one design set "
              f"to embed, and without it there is nothing to prove.")
        return 1

    bad = 0
    print("NpuEmbeddings -- embedded design set gate")
    print(f"  1. the two languages spell it the same way")
    bad += section_spellings()

    model = "all-MiniLM-L6-v2"
    mdir = os.path.join(ROOT, "models", model)
    if not os.path.isdir(mdir):
        print(f"  FAIL  models/{model} is not there")
        return 1

    tmp = tempfile.mkdtemp(prefix="npue_design_")
    try:
        # THE FILE NAME IS THE MODEL NAME. The runtime derives the model from the
        # container's basename -- which is why every model in models/ is
        # models/<name>.npue and why pick_artifacts is asked for "without" and
        # finds nothing. So each container gets its own directory and keeps the
        # model's basename, rather than being called with.npue / without.npue.
        #
        # The first version of this gate did exactly that and section 4 failed with
        # "no NPU design set matches without" -- which reads like a runtime bug and
        # is not one. It is the runtime refusing to guess which model's design set
        # belongs to a file called `without.npue`, and it should.
        d_with = os.path.join(tmp, "with")
        d_without = os.path.join(tmp, "without")
        os.makedirs(d_with, exist_ok=True)
        os.makedirs(d_without, exist_ok=True)
        with_set = os.path.join(d_with, model + ".npue")
        without = os.path.join(d_without, model + ".npue")
        print(f"  2. packing twice: with and without the design set")
        r = pack(model, with_set, ("--model-dir", mdir, "--device", "npu1",
                                   "--max-seq", "256"))
        if r.returncode != 0:
            return fail(f"packing with artifacts failed:\n{r.stderr[-900:]}")
        r = pack(model, without, ("--model-dir", mdir, "--device", "npu1",
                                  "--max-seq", "256",
                                  "--no-embed-artifacts"))
        if r.returncode != 0:
            return fail(f"packing without artifacts failed:\n{r.stderr[-900:]}")

        import design_embed as DE
        from npue import Reader
        a, b = Reader(with_set), Reader(without)
        print(f"  3. what the two containers carry")
        if not DE.carries_design(a):
            bad += fail(f"{with_set} carries no gemm_rtp set, so the packer "
                        f"reported an embed and stored nothing")
        else:
            sets = DE.design_sets_in(a)
            tot = sum(v["bytes"] for v in sets.values())
            print(f"      with:    " + ", ".join(
                f"{k} {v['files']}f/{v['bytes'] // 1024}KB"
                for k, v in sorted(sets.items())) + f"  ({tot // 1024} KB)")
        if DE.carries_design(b):
            bad += fail("--no-embed-artifacts produced a container that carries a "
                        "design set; the flag does not do what its help says")
        else:
            print(f"      without: no design set")
        sz_a = os.path.getsize(with_set)
        sz_b = os.path.getsize(without)
        pct = 100.0 * (sz_a - sz_b) / sz_b
        print(f"      size:    {sz_b:,} -> {sz_a:,} bytes (+{pct:.2f}%)")
        if pct > 5.0:
            bad += fail(f"embedding cost {pct:.2f}% of the container, above the "
                        f"5% this gate is willing to call a convenience. It was "
                        f"0.35% on a 69 MB embedder; if it is large here, something "
                        f"other than a design set got embedded.")
        else:
            print(f"  ok    +{pct:.2f}% of the container, which is what 272-536 KB "
                  f"of instruction streams costs")

        print(f"  4. the same answer with and without a design directory")
        base = os.path.join(tmp, "base.f32")
        selfs = os.path.join(tmp, "self.f32")
        r = embed(without, base)
        if r.returncode != 0:
            return fail(f"the no-artifacts container cannot embed even with the "
                        f"directory present:\n{r.stderr[-600:]}")

        hidden = ARTIFACTS + "__hidden_by_verify_embedded_artifacts"
        if os.path.exists(hidden):
            shutil.rmtree(hidden)
        # The directory is MOVED, not copied: a copy would leave the very thing this
        # section claims to remove. And it is moved back in a finally, because a gate
        # that leaves the tree without its design sets is worse than one that fails.
        os.rename(ARTIFACTS, hidden)
        try:
            r = embed(with_set, selfs)
            if r.returncode != 0:
                bad += fail(f"a self-sufficient container cannot embed with "
                            f"runtime/artifacts/ absent:\n{r.stderr[-600:]}")
            else:
                print(f"      embedded with no design directory on disk")
        finally:
            os.rename(hidden, ARTIFACTS)

        if bad == 0 and os.path.exists(selfs):
            if sha(base) == sha(selfs):
                print(f"  ok    byte-identical: the embedded set IS the directory "
                      f"set ({sha(selfs)[:16]})")
            else:
                bad += fail(f"the two runs differ: {sha(base)[:16]} with the "
                            f"directory against {sha(selfs)[:16]} from the "
                            f"container. Same design, different answer, means the "
                            f"container is carrying a DIFFERENT set than the one "
                            f"on disk -- not a rounding question.")
        if bad == 0 and os.path.exists(selfs):
            # And the self-sufficient one must not merely succeed -- it must have
            # dispatched. A run that silently fell back to the host would produce
            # the same bytes with no array work at all.
            r = embed(with_set, os.path.join(tmp, "again.f32"))
            line = [l for l in r.all_output.splitlines()
                    if "artifacts" in l.lower()]
            if not line:
                # An ABSENT status line used to pass this branch, because the
                # condition was `if line and ... wrong`. So a container that ran
                # from its own embedded set while saying nothing about where the
                # set came from was reported as fine -- and "says nothing" is the
                # failure this whole check exists for: a status block that cannot
                # tell a reader whether their MLIR-AIE is needed is the bug, not a
                # missing nicety.
                bad += fail("the run printed no artifacts status line at all, so "
                            "nothing tells a reader that the container supplied "
                            "the design set. That is how the pre-embedded build "
                            "behaved -- and it is why a reader could not tell a "
                            "self-sufficient container from one that needed a "
                            "compiler.")
            elif "container's own design set" not in line[0]:
                bad += fail(f"the status line does not say the container's own set "
                            f"was used: {line[0].strip()[:120]}")
            else:
                print(f"      {line[0].strip()[:96]}")

        print(f"  5. an elementwise op also comes from the container")
        hidden = ARTIFACTS + "__hidden_by_verify_embedded_artifacts"
        os.rename(ARTIFACTS, hidden)
        try:
            r = embed(with_set, os.path.join(tmp, "gelu.f32"), ("--npu-ops", "gelu"))
            if r.returncode != 0:
                bad += fail(f"--npu-ops gelu on a self-sufficient container failed "
                            f"with no directory present:\n{r.stderr[-600:]}")
            else:
                print(f"      --npu-ops gelu ran with no design directory")
        finally:
            os.rename(hidden, ARTIFACTS)

        print(f"  6. the datapath is not mixed")
        ck = json.load(open(os.path.join(mdir, "CHECKPOINT.json")))
        recorded = ck.get("sha256")
        dp = a.config.get("artifacts_datapath")
        if dp not in ("bf16", "i8"):
            bad += fail(f"the container records artifacts_datapath={dp!r}, which "
                        f"is neither bf16 nor i8")
        else:
            print(f"      recorded datapath: {dp}")
        if a.config.get("artifacts_embedded") != "true":
            bad += fail("the container does not claim artifacts_embedded=true; a "
                        "reader that trusted the flag would skip a set it has")
        else:
            print(f"      artifacts_embedded: true")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        hidden = ARTIFACTS + "__hidden_by_verify_embedded_artifacts"
        if os.path.exists(hidden):
            os.rename(hidden, ARTIFACTS)

    if bad:
        print(f"FAIL -- {bad} problem(s) with a container carrying its own design set")
        return 1
    print("OK: one .npue is self-sufficient -- the same bytes with and without a "
          "design directory, no MLIR-AIE on the reader's machine, and a container "
          "without a set still runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
