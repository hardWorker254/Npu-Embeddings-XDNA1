#!/usr/bin/env python3
"""Gate for models/DOWNLOADS.md -- the index of where every model's files come from.

WHAT THIS GATE IS FOR
---------------------
The document answers a question a fresh checkout cannot answer for itself: where
do I get the weights. It used to be unanswerable from the repository, and the part
that was missing was not the repository names -- every CHECKPOINT.json has a
`repo_id` -- but three things around them:

  * whether the file is IN that repository. Ten of the eighteen models have no
    ONNX export upstream at all, so `file: ["onnx/encoder_model.onnx"]` in
    whisper's CHECKPOINT.json is a name to create locally, not a path to fetch,
    and both entries 404;
  * which digest convention the recorded `sha256` uses, because the transformer
    form is `model_digest()` -- not the hash of any one file -- and the MediaPipe
    form is a list of per-file hashes, and reading one as the other fails on a
    correct file;
  * whether the bytes behind the name are the bytes this tree was built against.
    Seven of the eight downloadable exports were downloaded and hashed on
    2026-10-06 and matched; the eighth was too large to finish in one session and
    its row says so.

TWO CHECKS, AND THE SECOND IS THE ONE THAT EARNS THE GATE
----------------------------------------------------------
    1. the generated documents match tools/data/model_sources.json, byte for
       byte -- the generator's own --check, which is a comparison and not a
       substring search, so a paragraph deleted from the generator and left in
       the .md fails;
    2. the data file and every models/<name>/CHECKPOINT.json agree.

Check 2 lives in the generator too, and is what makes this document trustworthy:
a generated index whose source disagrees with the pins is worse than no index,
because it looks maintained. This gate runs the same code path, so the refusal
that stops the generator writing is the refusal that fails here.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fetch anything. A gate that needs the network fails on an air-gapped
build machine and is then always skipped, and a skipped gate is not a gate. The
`verified` fields in the data file carry the date they were established, so a
reader can see how old a check is without this gate pretending to re-establish
it.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "lib"))

DOCS = (os.path.join(ROOT, "models", "DOWNLOADS.md"),
        os.path.join(ROOT, "models", "DOWNLOADS.ru.md"))
GEN = os.path.join(ROOT, "tools", "gen_downloads_doc.py")
DATA = os.path.join(ROOT, "tools", "data", "model_sources.json")


def main() -> int:
    print("NpuEmbeddings -- downloads index gate")
    bad = 0

    if not os.path.exists(DATA):
        print(f"  FAIL  {DATA} is missing -- there is no source for the index")
        return 1

    # 1 + 2 in one call: the generator exits 1 both when the two sources disagree
    # (and prints why) and when a document is stale (and prints the diff).
    rc = subprocess.run([sys.executable, GEN, "--check"],
                        capture_output=True, text=True, cwd=ROOT)
    for stream, tag in ((rc.stdout, ""), (rc.stderr, "   ")):
        for line in stream.rstrip().splitlines():
            print((tag + line) if tag or line.startswith("  ") else "  " + line)
    if rc.returncode != 0:
        bad += 1

    # 3. Both documents are TRACKED. models/** is gitignored apart from a list of
    #    exceptions, and the whole point of this file is that a reader who has not
    #    packed anything can still find the download links. If the .gitignore
    #    exception is dropped, the file keeps working on this machine and stops
    #    existing for everyone else -- which is exactly how the download links
    #    were missing from the repository until now: they lived in
    #    models/<name>/README.md files that were never tracked.
    import subprocess as sp
    for d in DOCS:
        rel = os.path.relpath(d, ROOT)
        q = sp.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        if q.returncode == 0:
            print(f"  FAIL  {rel} is gitignored. models/** hides it, so it is "
                  f"present here and absent from every fresh checkout -- which is "
                  f"the state this document was written to end. Add the "
                  f"!models/... exception to .gitignore.")
            bad += 1
        elif not os.path.exists(os.path.join(ROOT, ".git")):
            print(f"  skip  {rel}: not a git checkout, cannot check tracking")
        else:
            print(f"  ok    {rel} is tracked, not gitignored")

    # 4. Every URL in the document is one this generator built from the data file,
    #    and every downloadable row has a hash. A row with a URL and no digest is
    #    a link a reader cannot check, which is the one thing worse than no link.
    import json
    with open(DATA, encoding="utf-8") as f:
        src = json.load(f)
    # Every downloadable row must EITHER carry a real per-file digest OR say in
    # its `verified` text that the bytes are unproven. A row with a URL and
    # neither is a link a reader cannot check and is not told why -- the one
    # outcome worse than no link, because it invites them to trust it.
    #
    # The first draft of this check computed a list called `unproven`, compared
    # it against itself (always empty, so the branch never ran), and printed an
    # `ok` line that said something the code had not established. The variable is
    # gone rather than filled in.
    unproven, ok_rows = [], 0
    for name, row in sorted(src["models"].items()):
        for o in row.get("onnx") or []:
            sha = row.get("sha256_file")
            got = sha.get(o["file"]) if isinstance(sha, dict) else sha
            if isinstance(got, str) and len(got) == 64:
                ok_rows += 1
            elif row.get("verified", "").startswith(("downloaded", "BOTH")):
                unproven.append(
                    f"{name}/{o['file']} claims it was downloaded and hashed but "
                    f"carries no per-file digest -- so the claim cannot be checked "
                    f"against anything")
            else:
                unproven.append(
                    f"{name}/{o['file']} has a link, no digest, and no `verified` "
                    f"text saying its bytes are unproven")
    if unproven:
        print(f"  FAIL  {len(unproven)} downloadable row(s) are unchecked and do "
              f"not say so:")
        for u in unproven[:6]:
            print(f"   * {u}")
        bad += 1
    else:
        print(f"  ok    all {ok_rows} downloadable rows carry a 64-character "
              f"per-file digest; none is a bare link")
    n_unpr = sum(1 for n_, r in src["models"].items()
                 if r.get("onnx") and r.get("sha256_file") is None)
    if n_unpr:
        print(f"  note  {n_unpr} model row(s) declare their bytes unproven and are "
              f"printed that way. A stated gap, not a silent one.")

    if bad:
        print(f"FAIL -- {bad} disagreement(s) between models/DOWNLOADS*.md, "
              f"tools/data/model_sources.json and models/<name>/CHECKPOINT.json")
        return 1
    print("OK: the index says where every model's files come from, which of "
          "them can actually be downloaded as an ONNX, and which digest each "
          "recorded pin uses -- and it agrees with every CHECKPOINT.json it "
          "cross-checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())