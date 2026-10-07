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
    # THREE STATES, DECIDED BY AN EXPLICIT FIELD.
    #
    # `byte_evidence` is `hashed` (this row's own file was downloaded and hashed,
    # so it must carry a 64-character digest) or `listed` (only a HEAD request, so
    # it must carry NO digest and must say in `verified` that its bytes are
    # unproven).
    #
    # IT IS A FIELD AND NOT A PHRASE FOR TWO REASONS, both learned the hard way in
    # this file: this gate originally had no state for "listed" at all and so
    # rejected every honest un-downloaded row, and the prose it would have had to
    # read instead is English whose wording had already shipped one stale
    # paragraph in models/DOWNLOADS.md (a sentence naming bge-large as unproven,
    # which survived the download that proved it). A field cannot go stale when a
    # sentence is reworded; a prefix match can.
    missing, overclaimed = [], []
    ok_rows = 0
    for name, row in sorted(src["models"].items()):
        ev = row.get("byte_evidence")
        files = row.get("onnx") or []
        if not files:
            if ev is not None:
                overclaimed.append(
                    f"{name} declares byte_evidence={ev!r} but has no ONNX row to "
                    f"apply it to -- a stale field is a claim about nothing")
            continue
        if ev not in ("hashed", "listed"):
            missing.append(
                f"{name} has a download and no byte_evidence; it must say 'hashed' "
                f"or 'listed', because the difference between 'the bytes are proven' "
                f"and 'the bytes are a HEAD request' is the one a reader cannot "
                f"infer from a URL")
            continue
        sha = row.get("sha256_file")
        for o in files:
            got = sha.get(o["file"]) if isinstance(sha, dict) else sha
            hashed = isinstance(got, str) and len(got) == 64
            if ev == "hashed" and not hashed:
                missing.append(
                    f"{name}/{o['file']} declares byte_evidence=hashed but carries no "
                    f"64-character digest, so the row claims bytes it does not have")
            if ev == "listed" and hashed:
                overclaimed.append(
                    f"{name}/{o['file']} declares byte_evidence=listed yet carries a "
                    f"digest -- either the file really was hashed and the field is "
                    f"wrong, or the digest was copied from somewhere else")
            if ev == "hashed" and hashed:
                ok_rows += 1
    if missing:
        print(f"  FAIL  {len(missing)} row(s) make a claim they cannot support:")
        for m in missing[:6]:
            print(f"   * {m}")
        bad += 1
    else:
        print(f"  ok    every download row declares byte_evidence, and all "
              f"{ok_rows} hashed rows carry a 64-character digest")
    if overclaimed:
        print(f"  FAIL  {len(overclaimed)} row(s) contradict their own evidence:")
        for o in overclaimed[:6]:
            print(f"   * {o}")
        bad += 1
    else:
        print(f"  ok    no row claims bytes it does not carry")
    n_listed = sum(1 for r in src["models"].values()
                   if r.get("byte_evidence") == "listed")
    if n_listed:
        print(f"  note  {n_listed} row(s) are 'listed': the file is there and is the "
              f"right shape, and the document prints it as bytes-unproven. A stated "
              f"gap, not a silent one.")

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