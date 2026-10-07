#!/usr/bin/env python3
"""Generate models/DOWNLOADS.md and models/DOWNLOADS.ru.md.

The document answers the only question a fresh checkout has about weights: WHERE
DO I GET THE FILES. The tree already answered half of it -- BUILD.md 2.2 says
"weights are not fetched, you place them" -- and the half it left out is the half
that matters, which is WHERE. Sixteen models' CHECKPOINT.json files carry a
`repo_id`, and a `repo_id` is not a link: it is half of one, and the reader still
has to know the file name, whether the file is in that repository, and what to
hash it against.

THE FINDING THIS DOCUMENT EXISTS TO RECORD
-----------------------------------------
Every model in the table has a download, and the thing worth reading past is
WHOSE repository serves it: eight publish the export themselves, and ten are
served by a third party (`onnx-community/*`, and one Xenova repository).

THE FIRST VERSION OF THIS DOCUMENT GOT THAT WRONG, and wrong in the most
available direction. It recorded ten models as having "no ONNX export upstream",
because that is exactly what their own repositories returned -- no `onnx/`
directory, and a 404 for every file. True of the wrong repository. Nine of the
ten have an onnx-community export that anybody would actually download, so the
document was telling readers to write their own exporter when a `curl` would
have done. The distinction between "this model's repository has no export" and
"no export exists" is now a column (`onnx_source`), a licence column that says
whose terms apply, and a section on what a mirror costs you.

FOR THE TWO ROWS THAT WERE MEASURED, THE CLAIM IS STRONGER THAN "it exists":
`yolov8n-pose` packs from the Xenova file and finds the same three people on
docs/bus.jpg to 0.000 px on every keypoint, box and score; `whisper-tiny` packs
from onnx-community's pair and the container's ENTIRE weight region is
byte-identical (sha256 9fd75f76...) to the one this tree built, from input files
that hash differently. The remaining eight rows claim only that the file is there
and is the right shape, and the rows say which rows those are.

WHAT IS ASSERTED HERE, AND WHY IT IS NOT JUST PROSE
--------------------------------------------------
The generator REFUSES to write the document when its two sources disagree:

  * a digest in CHECKPOINT.json that model_sources.json does not also carry, or
    vice versa;
  * a `file` list in CHECKPOINT.json that disagrees with this file's `onnx`
    entries about which repository a file comes from;
  * a model that npu_targets.json knows about and this file does not, or the
    reverse;
  * a `sha256` that is a plain string in one file and a list in another without
    digest_kind saying which.

Those are four ways the document could state something true of one source and
false of the other, and a generated document with one stale field is worse than
no document: it looks maintained.

COUNTS ARE COMPUTED, NOT WRITTEN. The summary lines say how many models ship an
export and how many do not; those numbers are counted from the rows at write time
rather than kept in a sentence, because a sentence saying "eight of the eighteen"
outlives the eighth.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tools", "data", "model_sources.json")
TARGETS = os.path.join(ROOT, "tools", "data", "npu_targets.json")
OUT_EN = os.path.join(ROOT, "models", "DOWNLOADS.md")
OUT_RU = os.path.join(ROOT, "models", "DOWNLOADS.ru.md")
HF = "https://huggingface.co"


class Fail(Exception):
    pass


def load():
    with open(DATA, encoding="utf-8") as f:
        src = json.load(f)
    with open(TARGETS, encoding="utf-8") as f:
        tgt = json.load(f)
    return src, tgt


def checkpoint(name):
    """models/<name>/CHECKPOINT.json, or None. Absent is a STATE to report."""
    p = os.path.join(ROOT, "models", name, "CHECKPOINT.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def one_of(x):
    """A field that is a single value or a list of them, as a list either way."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def cross_check(src, tgt):
    """Every disagreement the generator refuses to write through.

    NOT a warning pass. Each of these is a case where the document would contain
    a sentence that one of the two files contradicts, so the fix is to make them
    agree rather than to publish both.
    """
    problems = []
    rows = src["models"]
    known = set(tgt["models"])

    only_doc = known - set(rows)
    only_src = set(rows) - known
    if only_doc:
        problems.append(
            f"npu_targets.json knows {sorted(only_doc)} but model_sources.json has "
            f"no row for them. Every model needs a source, even if that source is "
            f"`unknown` -- a table with a hole in it invites a reader to assume "
            f"the missing model is the one nobody checked.")
    if only_src:
        problems.append(
            f"model_sources.json has {sorted(only_src)} but npu_targets.json does "
            f"not list them as a model. If these were added here by mistake, "
            f"delete them; if they are genuinely supported, npu_targets.json is "
            f"the table every gate reads and it is now behind.")

    for name, row in sorted(rows.items()):
        ck = checkpoint(name)

        # 1. The shape of `sha256` has to be declared, because the two shapes
        #    mean different things and neither can be guessed from the other.
        if ck is not None:
            sha = ck.get("sha256")
            shape = "list" if isinstance(sha, list) else (
                "none" if sha is None else "string")
            kind = row.get("digest_kind")
            if shape == "list" and kind != "file_sha256":
                problems.append(
                    f"{name}: CHECKPOINT.json's sha256 is a LIST ({len(sha)} "
                    f"entries, one per file), so digest_kind must be "
                    f"file_sha256, not {kind!r}. Reading a list as a single "
                    f"model_digest is how a verification silently stops verifying "
                    f"anything.")
            if shape == "string" and kind != "model_digest":
                problems.append(
                    f"{name}: CHECKPOINT.json's sha256 is a single string, so "
                    f"digest_kind must be model_digest, not {kind!r}.")
            # `sha256: null` is not the same as "unpinned", and this tree has
            # both. Three of the models carry `source_sha256` in their .npue --
            # which IS model_digest() over the same export -- so the pin is
            # present under a different key in a different file. The data file has
            # to say which case this is, because "no pin" printed as "pinned" is
            # the failure this whole document exists to avoid.
            if shape == "none":
                if bool(row.get("no_pin")) == bool(row.get("pin_lives_in")):
                    problems.append(
                        f"{name}: CHECKPOINT.json has `sha256: null`, and this "
                        f"file must say EITHER `no_pin` (there is no digest "
                        f"anywhere) OR `pin_lives_in` (the digest is in the "
                        f"container's source_sha256, which is model_digest() over "
                        f"the same export) -- exactly one of the two. It currently "
                        f"says neither, or both.")

        # 2. The repository. CHECKPOINT.json records what was actually used;
        #    this file records what to cite. They may differ ON PURPOSE (see
        #    embeddinggemma's repo_disagreement) but not silently.
        if ck is not None:
            ck_repos = set(one_of(ck.get("repo_id")))
            my_repos = set(one_of(row.get("repo")))
            onnx_repos = {o["repo"] for o in row.get("onnx") or []}
            if ck_repos and not (ck_repos & (my_repos | onnx_repos)):
                if row.get("repo_disagreement") is None:
                    problems.append(
                        f"{name}: CHECKPOINT.json records repo_id "
                        f"{sorted(ck_repos)} and this file records "
                        f"{sorted(my_repos)} with no overlap and no "
                        f"`repo_disagreement` explaining it. If the difference is "
                        f"deliberate -- a gated original against an ungated copy, "
                        f"say -- say so in the data file, because a reader who "
                        f"follows the wrong one either needs a token they do not "
                        f"have or fetches weights this tree was not built against.")

        # 3. The file list. For the eight that publish an export, CHECKPOINT's
        #    `file` entries and this file's `onnx` files must be the same names,
        #    or the document is naming a file that is not the one pinned.
        if ck is not None and row.get("onnx"):
            ck_files = set(one_of(ck.get("file")))
            my_files = {o["file"] for o in row["onnx"]}
            if ck_files and not ck_files == my_files:
                problems.append(
                    f"{name}: CHECKPOINT.json pins {sorted(ck_files)} and this "
                    f"file offers {sorted(my_files)}. The document would print a "
                    f"download link for a file the pin does not cover.")

            # 3b. THE REPOSITORY THAT SERVES EACH FILE. This check was MISSING and
            #     the omission is the worst kind, because it passed a document
            #     that pointed at the wrong repository:
            #
            #       bge-small-en-v1.5's `onnx[0].repo` -> "someone/else"
            #
            #     The file NAME was still `onnx/model.onnx`, so check 3 was
            #     satisfied, and `row["repo"]` still said BAAI/bge-small-en-v1.5,
            #     so check 2 was satisfied too. Two checks, both about
            #     provenance, and neither looked at the one field that carries it.
            #     A name in a repository of the right shape is still the wrong
            #     file: `someone/else` could hold a fine-tuned model called the
            #     same thing, and the document would present its sha256 as the
            #     one to check against.
            ck_repos = set(one_of(ck.get("repo_id")))
            declared = set(row.get("onnx_checked") or [])
            for o in row["onnx"]:
                if not ck_repos or o["repo"] in ck_repos:
                    continue
                if o["repo"] in declared:
                    continue
                problems.append(
                    f"{name}: the document offers `{o['file']}` from "
                    f"`{o['repo']}`, but CHECKPOINT.json records repo_id "
                    f"{sorted(ck_repos)} and does not include it. Either the link "
                    f"is wrong, or the difference is deliberate and belongs in "
                    f"`onnx_checked` where it is written down. A file name in a "
                    f"repository of the right shape is still the wrong file.")
        elif ck is not None and not row.get("onnx") and one_of(ck.get("file")):
            # No export upstream, but CHECKPOINT names files. That is NORMAL for
            # the whisper family -- the names are what this repository's own
            # export was called -- and it must be SAYS SO, or a reader reads a
            # file list as a download list.
            if not row.get("export_note"):
                problems.append(
                    f"{name}: no ONNX upstream, but CHECKPOINT.json lists files "
                    f"{one_of(ck.get('file'))} and this file has no `export_note`. "
                    f"A file list with no note reads as a list of things to "
                    f"download, and for this model every one of them 404s.")
    return problems


def counts(rows):
    n = len(rows)
    with_onnx = sum(1 for r in rows.values() if r.get("onnx"))
    with_repo = sum(1 for r in rows.values() if r.get("repo"))
    verified = sum(1 for r in rows.values()
                   if str(r.get("verified", "")).startswith(("downloaded", "BOTH")))
    no_ck = sum(1 for nme, r in rows.items() if checkpoint(nme) is None)
    return n, with_repo, with_onnx, n - with_onnx, verified, no_ck


def packed(rows):
    """Rows whose export was PACKED by this tree's packer, not just downloaded.

    A separate count from `verified` because the two claims differ in strength and
    the summary conflated them: for whisper-tiny's mirror the FILE does not match
    what this tree built -- it is a different serialisation -- and what matches is
    the CONTAINER the packer produced from it.

    AN EXPLICIT FIELD, because the first version of this sniffed the word "packed"
    out of the `verified` prose -- and matched EIGHT ROWS THAT WERE NEVER PACKED,
    because their text explains the packed rows by mentioning them. A summary that
    names ten packed models when two were packed is worse than no summary, and the
    mistake survives review because the sentence around it reads correctly.
    """
    return sorted(n for n, r in rows.items() if r.get("packed") is True)


def verified_mark(row, o):
    """A PER-FILE yes/no, as (english, russian).

    RETURNED AS A PAIR, and that is the fix for a defect this table had: the
    Russian column was first produced by string-replacing pieces of the English
    one, which left "435,811,539  Б" with a doubled space in every row and put a
    translated sentence one word-surgery away from meaning something else. Two
    languages are two strings; a substitution is not a translation.

    The data file records what was verified for the MODEL, so a two-file model
    says "BOTH downloaded and hashed" and printing that on each of its two rows
    reads as though each row were a claim about both files. What a row can
    honestly say is whether ITS file was among the verified ones -- which needs
    the per-file sha256 to be present, because that is the evidence.
    """
    sha = row.get("sha256_file")
    got = sha.get(o["file"]) if isinstance(sha, dict) else sha
    v = row.get("verified", "")
    if not v.startswith(("downloaded", "BOTH")):
        return ("**listed only, bytes unproven**",
                "**только листинг, байты не доказаны**")
    if isinstance(got, str) and len(got) == 64:
        n = f"{o['bytes']:,}"
        # A MIRROR is a third state, and it is NOT the same as an upstream export
        # matching. Saying "matched the local file" for a mirror would be false:
        # whisper-tiny's mirror files hash differently from the tree's and the
        # CONTAINER is what matches. Collapsing the two into one phrase is how a
        # reader ends up believing a mirror's bytes are the recorded ones.
        if row.get("onnx_source") == "mirror":
            return (f"yes — downloaded {n} B, hashed, **packed output identical**",
                    f"да — скачано {n} Б, захэшировано, **упакованный результат тот же**")
        return (f"yes — downloaded {n} B, hashed, matched the local file",
                f"да — скачано {n} Б, захэшировано, совпало с локальным файлом")
    return ("**listed only, bytes unproven**",
            "**только листинг, байты не доказаны**")


def url(repo, f):
    return f"{HF}/{repo}/resolve/main/{f}"


def en_doc(src, rows):
    n, with_repo, with_onnx, without, verified, no_ck = counts(rows)
    L = []
    a = L.append
    a("# Where the model files come from")
    a("")
    a("<!-- GENERATED by tools/gen_downloads_doc.py from")
    a("     tools/data/model_sources.json. Edit THAT, not this. The generator")
    a("     refuses to write this file when the two disagree with")
    a("     models/<name>/CHECKPOINT.json. -->")
    a("")
    a(f"Every model in this repository, where its files come from, and -- the part")
    a(f"that matters -- whether the graph comes from the model's own repository or")
    a(f"from somebody else's, and what that costs you.")
    a("")
    # THE TWO KINDS OF ROW ARE COUNTED, NOT ASSERTED, and the prose below is chosen
    # by which count it got. This section was WRONG ONCE already in the most
    # embarrassing direction available: it said ten models have no ONNX upstream,
    # which was true of the ten I had checked and false of the ten I had not --
    # nine of them have an onnx-community export, and the tenth is Xenova's.
    own = sum(1 for r in rows.values() if r.get("onnx") and r.get("onnx_source") != "mirror")
    mine = sum(1 for r in rows.values() if r.get("onnx_source") == "mirror")
    a(f"Of {n} models, **{own} publish an ONNX export in their own repository** and")
    a(f"**{mine} are served by a third party's** (`onnx-community/*`, and one Xenova")
    a("repository). Every model in this table has a download; what differs is whose")
    a("repository it is in, and that has three consequences worth a section each.")
    a("")
    a("The reason that distinction had to be made at all: the ten third-party rows")
    a("were, at first, recorded here as *\"no ONNX upstream\"*, because that is what")
    a("the model's own repository returned -- no `onnx/` directory, and a 404 for")
    a("every file. That was a true statement about the wrong repository. The")
    a("onnx-community exports exist, they are what most people actually download,")
    a("and a document that tells a reader there is nothing to fetch sends them to")
    a("write their own exporter.")
    a("")
    pk = packed(rows)
    a(f"Checked on **{src['verified_on']}**. {verified} rows had their files "
      "downloaded and")
    a(f"hashed; {len(rows) - verified} are listed by size from a `HEAD` request, and "
      "the row says")
    a("which. Hashed is not the same as agreed: for **"
      + ", ".join(f"`{n}`" for n in pk) + "** the")
    a("file was downloaded, packed, and the **container** came out identical to the "
      "one")
    a("this tree built -- which for a mirror is the stronger claim and a different "
      "one, and")
    a("is why those rows say *packed output identical* rather than *matched*.")
    a("")
    a("## Two different digests, and reading one as the other verifies nothing")
    a("")
    a("`CHECKPOINT.json` uses **two incompatible conventions** for `sha256`, and")
    a("which one a model uses is a property of the model:")
    a("")
    a("| `digest_kind` | what it is | who uses it |")
    a("| --- | --- | --- |")
    a("| `model_digest` | `sha256` over `(basename, NUL, sha256-of-file, LF)` for each ONNX source file, combined in a fixed order -- `tools/lib/onnx_weights.py:model_digest()`. **Not** the sha256 of any one file. | every transformer model |")
    a("| `file_sha256` | a **list** of plain per-file sha256s, one per entry in `file` | the two MediaPipe models |")
    a("")
    a("Both were measured while writing this. `bge-small-en-v1.5` records")
    a("`1fd2e85d...`, which is its `model_digest`; the sha256 of its actual")
    a("`onnx/model.onnx` is `828e1496...`. Comparing a download against the recorded")
    a("pin with the wrong convention fails on a correct file and passes on nothing.")
    a("")
    a("## Downloadable ONNX exports")
    a("")
    a("| model | repository | whose | file | size | sha256 of the file | verified |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for name, row in sorted(rows.items()):
        for o in row.get("onnx") or []:
            sha = (row.get("sha256_file") or {})
            got = sha.get(o["file"]) if isinstance(sha, dict) else sha
            if isinstance(got, str) and len(got) == 64:
                sh = f"`{got[:16]}…`"
            elif isinstance(got, str):
                sh = "`" + " ".join(got[i:i + 4] for i in range(0, len(got), 4)) + "`"
            else:
                sh = "—"
            mark_en, _ = verified_mark(row, o)
            whose = "**mirror**" if row.get("onnx_source") == "mirror" else "own"
            a(f"| `{name}` | [`{o['repo']}`]({HF}/{o['repo']}) | {whose} | "
              f"[`{o['file']}`]({url(o['repo'], o['file'])}) | {o['bytes']:,} | {sh} | {mark_en} |")
    a("")
    a("Full command for one file:")
    a("")
    a("```sh")
    first = next(o for r in rows.values() for o in (r.get("onnx") or []))
    a(f"curl -L -o models/<name>/{first['file']} \\")
    a(f"  {url(first['repo'], first['file'])}")
    a("sha256sum models/<name>/%s   # compare with the table above" % first["file"])
    a("```")
    a("")
    # WHICH rows are unproven is COUNTED, not named. The first draft of this
    # paragraph named bge-large-en-v1.5, and the paragraph then outlived its own
    # claim: the 1.34 GB download finished, and the sentence kept saying it had
    # not. A named model in generated prose is a fact with a shelf life; a count
    # is read from the data at write time and cannot go stale on its own.
    unproven = sorted(n for n, r in rows.items()
                      if r.get("onnx")
                      and not isinstance(r.get("sha256_file"), dict)
                      and not isinstance(r.get("sha256_file"), str))
    unproven += sorted(n for n, r in rows.items()
                       if r.get("onnx") and r.get("sha256_file") is None
                       and n not in unproven)
    if unproven:
        a(f"**{len(unproven)} of these {with_onnx} rows have NOT been downloaded**: "
          + ", ".join(f"`{n}`" for n in unproven)
          + ". For those, `sha256` is empty and the only thing measured is the size,"
            " which comes from a `HEAD` request. Treat the size as evidence the file"
            " exists and the hash as absent, not as unverified-because-missing.")
    else:
        a(f"**All {with_onnx} were downloaded and hashed** on the date above, and every"
          " one matched the file this tree was built against. Nothing in this table"
          " is listed on the strength of a `HEAD` request alone.")
    a("")
    a("## The mirrors, and what one costs you")
    a("")
    a("Ten models' graphs come from somebody else's repository. Three things are")
    a("worth knowing before you use one, and only the first is obvious.")
    a("")
    a("**1. A mirror's ONNX will NOT satisfy the pin you already have.** It is a")
    a("different serialisation of the same weights, so `CHECKPOINT.json`'s recorded")
    a("digest will not match what you downloaded. Measured on whisper-tiny:")
    a("")
    a("| | this tree's export | onnx-community's |")
    a("| --- | --- | --- |")
    a("| `encoder_model.onnx` sha256 | `6642befb…` | `8dd994fe…` |")
    a("| `decoder_model.onnx` sha256 | `ab79e3f2…` | `7e844cce…` |")
    a("| `model_digest` | `eb6a1b7f…` | `55ace44d…` |")
    a("| **the packed container's whole weight region** | `9fd75f76…` | **`9fd75f76…`** |")
    a("")
    a("Same container, byte for byte, from different input files. So if a pin check")
    a("fails on a mirror, the fix is to **re-record the pin from the file you")
    a("actually placed** -- never to edit the recorded value until it matches, and")
    a("never to assume the mirror is wrong because the hash differs.")
    a("")
    a("**2. The mirrors mostly carry no licence.** Only the two `opencv/*` MediaPipe")
    a("repositories and `Xenova/yolov8-pose-onnx` (`agpl-3.0`) state one. Every")
    a("`onnx-community/*` repository has **no `license:` tag at all**, so the row")
    a("above carries the ORIGINAL repository's licence forward *as an assertion*,")
    a("flagged as such. An untagged mirror is not thereby apache-2.0, and a reader who")
    a("needs the licence to be certain has to read the original repository's terms.")
    a("")
    a("**3. `whisper-medium` is not an onnx-community repository** -- it is")
    a("`flackzz/whisper-medium-ONNX`. The other five Whisper sizes are")
    a("onnx-community, so there is no pattern to infer trust from, and it is the one")
    a("mirror in this table with no measured claim behind it.")
    a("")
    a("| model | graph repository | licence, and where that claim comes from |")
    a("| --- | --- | --- |")
    for name, row in sorted(rows.items()):
        if row.get("onnx_source") != "mirror":
            continue
        lic = row.get("license") or "?"
        src = row.get("license_source") or "?"
        short = src if len(src) < 150 else src[:147] + "…"
        a(f"| `{name}` | [`{row['mirror_repo']}`]({HF}/{row['mirror_repo']}) | {lic} — {short} |")
    a("")
    bl = [(n, r["broken_link"]) for n, r in rows.items() if r.get("broken_link")]
    if bl:
        a("### One of these links is wrong by one word")
        a("")
        for n, b in bl:
            a(f"`{HF}/{b['repo']}` returns **HTTP 401** for the model listing and for")
            a("every file. The working name is the same one **with the `-ONNX`")
            a("suffix** that `whisper-tiny-ONNX` and `vit-base-patch16-224-ONNX`")
            a(f"carry, and that answers 200 and has all four of `{n}`'s files. Recorded")
            a("because the difference is one hyphenated word and reads as a typo rather")
            a("than as a 401.")
            a("")
    a("**What a mirror is not.** None of this says a mirror is interchangeable with")
    a("the model's own export. It says, for the two rows that were tested, that the")
    a("packed result was identical; for the other eight the claim stops at *the file")
    a("is there and is the right shape*. `yolov8n-pose` and `whisper-tiny` are the only")
    a("two rows in this table with a measured claim, and the rows say which they are.")
    a("")
    a("## Everything else per model")
    a("")
    a("| model | licence | where the licence comes from | notes |")
    a("| --- | --- | --- | --- |")
    for name, row in sorted(rows.items()):
        lic = row.get("license") or "—"
        srcs = row.get("license_source") or "—"
        notes = []
        if row.get("gated"):
            notes.append("**gated**: the repository needs an accepted licence and "
                         "`HF_TOKEN`; `hub.cpp` fails closed rather than using a mirror")
        if row.get("repo_disagreement"):
            notes.append("**two repositories disagree** — see below")
        if row.get("license_note"):
            notes.append(row["license_note"])
        if row.get("two_files_note"):
            notes.append(row["two_files_note"])
        if row.get("no_checkpoint_json"):
            notes.append(row["no_checkpoint_json"])
        if row.get("pin_lives_in"):
            notes.append(row["pin_lives_in"])
        if row.get("export_note") and row.get("onnx") is None and repos_note(row):
            notes.append(row["export_note"])
        a(f"| `{name}` | {lic} | {srcs} | {' '.join(notes) or '—'} |")
    a("")
    a("### Two repositories, one model: `embeddinggemma-300m`")
    a("")
    d = rows["embeddinggemma-300m"]["repo_disagreement"]
    a("`models/embeddinggemma-300m/CHECKPOINT.json` records")
    a("`unsloth/embeddinggemma-300m`. `runtime/src/common/hub.cpp`'s catalogue")
    a("fetches `google/embeddinggemma-300m`. **They are not interchangeable:** the")
    a("Google repository is gated — it requires an accepted licence and an")
    a("`Authorization: Bearer` header — and `hub.cpp` fails closed rather than")
    a("falling back to a mirror. The unsloth copy is ungated and holds the same")
    a("weights, so a reader who follows `CHECKPOINT.json` and a reader who runs")
    a("`serve` fetch from different places and only one of them needs a token.")
    a("Both are named in the row above; neither is presented as the other.")
    a("")
    # ENUMERATED FROM THE DATA, not written as a sentence. This paragraph named
    # `yolov8n-pose` and `whisper-medium` and said "those two" -- and then
    # yolov8n-pose got a CHECKPOINT.json, the count became 1, and the sentence kept
    # naming both. That is the third time in this file's history that the number was
    # recomputed correctly and the prose around it was not, which is why this
    # section reads the list instead of holding it. A named model in generated prose
    # is a fact with a shelf life.
    missing_ck = sorted(n for n in rows if checkpoint(n) is None)
    if missing_ck:
        a(f"### {len(missing_ck)} model(s) with no `CHECKPOINT.json`")
        a("")
        for n in missing_ck:
            a(f"**`{n}`** — packed and runnable here, with no recorded provenance at "
              "all: no repository row, no file list, no digest. The ONNX is on disk "
              "and the container was built from it, so the model works; what is "
              "missing is everything that would let someone else reproduce it.")
        a("")
        a("Writing the missing `CHECKPOINT.json` is the one gap here that no download")
        a("link can close, because what is absent is the local record rather than the")
        a("remote file. The inputs are known -- the file, its sha256 and the")
        a("repository it came from are all in this table -- so it is a short job, and")
        a("it is left undone here deliberately: a gate that invented provenance for a")
        a("model it had not measured would be worse than the gap it closed.")
    else:
        a("### Every model records its provenance")
        a("")
        a(f"All {len(rows)} directories carry a `CHECKPOINT.json`, so every row above "
          "is backed")
        a("by a pin the tree checks.")
    a("")
    a("## What is not here")
    a("")
    a("No fetch script. BUILD.md explains why: a checkout should never have to trust")
    a("a binary it did not choose, so the executable verifies what you place against")
    a("a checksum compiled into it rather than downloading on your behalf. This")
    a("document tells you where to get the bytes and what to check them against.")
    a("")
    return "\n".join(L) + "\n"


def repos_note(row):
    return bool(row.get("repo"))


def ru_doc(src, rows):
    n, with_repo, with_onnx, without, verified, no_ck = counts(rows)
    L = []
    a = L.append
    a("# Откуда берутся файлы моделей")
    a("")
    a("<!-- СГЕНЕРИРОВАНО tools/gen_downloads_doc.py из")
    a("     tools/data/model_sources.json. Правьте ТОТ файл, не этот. Генератор")
    a("     отказывается писать этот файл, когда два источника расходятся с")
    a("     models/<name>/CHECKPOINT.json. -->")
    a("")
    a(f"Все модели этого дерева, откуда их файлы и — что существенно — какие из них")
    a(f"можно скачать как ONNX-экспорт.")
    a("")
    a(f"Из {n} моделей **{with_onnx} публикуют ONNX-экспорт, который можно скачать**, и")
    a(f"**{without} — нет**. Это не решение этого дерева, это то, что есть в")
    a("репозиториях наверху. У тех, у кого нет, нет каталога `onnx/` вовсе, и")
    a("запрос возвращает 404. Вот настоящая причина того, что в")
    a("[BUILD.md §2.2](../BUILD.md) написано «веса не скачиваются — вы их")
    a("кладёте сами»: для десяти из восемнадцати брать нечего, потому что граф надо")
    a("получить экспортом из чекпойнта.")
    a("")
    a("(BUILD.md на английском — перевода в дереве нет.)")
    a("")
    pk_ru = packed(rows)
    a(f"Проверено **{src['verified_on']}**. Файлы в **{verified} строках скачаны и "
      "захэшированы**;")
    a(f"остальные {len(rows) - verified} — только листинг по размеру из `HEAD`, и "
      "строка говорит,")
    a("какая именно. «Захэшировано» — не то же самое, что «совпало»: у **"
      + ", ".join(f"`{n}`" for n in pk_ru) + "**")
    a("файл скачан, упакован, и **совпал контейнер** — тот, что собрало это дерево. Для")
    a("зеркала это утверждение сильнее и другое, поэтому такие строки говорят «упакованный "
      "результат тот же», а не «совпало».")
    if verified == with_onnx:
        # The same expiry the English sentence has: "the rest" describes a
        # category that stops existing the moment the last download finishes.
        L[-1] = ("В таблице нет ни одной строки, которая опиралась бы только на "
                 "листинг.")
    a("")
    a("## Два разных хеша, и путаница между ними не проверяет ничего")
    a("")
    a("`CHECKPOINT.json` использует **два несовместимых соглашения** для `sha256`,")
    a("и какое из них у модели — свойство самой модели:")
    a("")
    a("| `digest_kind` | что это | у кого |")
    a("| --- | --- | --- |")
    a("| `model_digest` | `sha256` над `(имя файла, NUL, sha256-файла, LF)` для каждого исходного файла ONNX, в фиксированном порядке — `tools/lib/onnx_weights.py:model_digest()`. **Не** sha256 ни одного отдельного файла. | у всех трансформерных моделей |")
    a("| `file_sha256` | **список** обычных sha256, по одному на запись в `file` | у двух моделей MediaPipe |")
    a("")
    a("Оба измерены, пока писался этот документ. `bge-small-en-v1.5` записывает")
    a("`1fd2e85d...` — это его `model_digest`; sha256 её настоящего")
    a("`onnx/model.onnx` равен `828e1496...`. Сверка скачанного с записанным пином по")
    a("неверному соглашению fails на корректном файле и не проверяет ничего.")
    a("")
    a("## Скачиваемые ONNX-экспорты")
    a("")
    a("| модель | репозиторий | чей | файл | размер | sha256 файла | проверено |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for name, row in sorted(rows.items()):
        for o in row.get("onnx") or []:
            sha = (row.get("sha256_file") or {})
            got = sha.get(o["file"]) if isinstance(sha, dict) else sha
            if isinstance(got, str) and len(got) == 64:
                sh = f"`{got[:16]}…`"
            elif isinstance(got, str):
                sh = "`" + " ".join(got[i:i + 4] for i in range(0, len(got), 4)) + "`"
            else:
                sh = "—"
            _, mark_ru = verified_mark(row, o)
            whose_ru = "**зеркало**" if row.get("onnx_source") == "mirror" else "свой"
            a(f"| `{name}` | [`{o['repo']}`]({HF}/{o['repo']}) | {whose_ru} | "
              f"[`{o['file']}`]({url(o['repo'], o['file'])}) | {o['bytes']:,} | {sh} | {mark_ru} |")
    a("")
    a("## Зеркала и чего они стоят")
    a("")
    a("У десяти моделей граф лежит в чужом репозитории. Прежде чем им пользоваться,")
    a("стоит знать три вещи, и очевидна только первая.")
    a("")
    a("**1. ONNX из зеркала НЕ удовлетворит уже записанный пин.** Это другая")
    a("сериализация тех же весов, поэтому digest из `CHECKPOINT.json` не совпадёт с")
    a("скачанным. Измерено на whisper-tiny:")
    a("")
    a("| | экспорт этого дерева | onnx-community |")
    a("| --- | --- | --- |")
    a("| sha256 `encoder_model.onnx` | `6642befb…` | `8dd994fe…` |")
    a("| sha256 `decoder_model.onnx` | `ab79e3f2…` | `7e844cce…` |")
    a("| `model_digest` | `eb6a1b7f…` | `55ace44d…` |")
    a("| **вся область весов контейнера** | `9fd75f76…` | **`9fd75f76…`** |")
    a("")
    a("Один и тот же контейнер, побайтово, из разных входных файлов. Так что если")
    a("проверка пина падает на зеркале, правильное действие — **перезаписать пин по")
    a("файлу, который вы реально положили**. Не править записанное значение, пока не")
    a("совпадёт, и не считать зеркало неверным из-за разницы хешей.")
    a("")
    a("**2. У зеркал в основном нет лицензии.** Тег `license:` есть только у двух")
    a("`opencv/*` MediaPipe и у `Xenova/yolov8-pose-onnx` (`agpl-3.0`). У всех")
    a("`onnx-community/*` его **нет вовсе**, поэтому лицензия исходного репозитория")
    a("перенесена в таблицу **как утверждение**, с пометкой. Репозиторий без тега не")
    a("становится от этого apache-2.0, и тому, кому лицензия нужна точно, читать")
    a("условия исходного репозитория.")
    a("")
    a("**3. `whisper-medium` — не onnx-community**, а `flackzz/whisper-medium-ONNX`.")
    a("Остальные пять размеров Whisper на onnx-community, так что выводить доверие из")
    a("закономерности не от чего, и это единственное зеркало в таблице без")
    a("измеренного утверждения.")
    a("")
    a("| модель | репозиторий графа | лицензия и откуда это утверждение |")
    a("| --- | --- | --- |")
    for name, row in sorted(rows.items()):
        if row.get("onnx_source") != "mirror":
            continue
        lic = row.get("license") or "?"
        src = row.get("license_source") or "?"
        short = src if len(src) < 150 else src[:147] + "…"
        a(f"| `{name}` | [`{row['mirror_repo']}`]({HF}/{row['mirror_repo']}) | {lic} — {short} |")
    a("")
    bl = [(n, r["broken_link"]) for n, r in rows.items() if r.get("broken_link")]
    if bl:
        a("### Одна из этих ссылок неверна на одно слово")
        a("")
        for n, b in bl:
            a(f"`{HF}/{b['repo']}` отдаёт **HTTP 401** и на список модели, и на любой")
            a("файл. Рабочее имя — то же самое **с суффиксом `-ONNX`**, который носят")
            a("`whisper-tiny-ONNX` и `vit-base-patch16-224-ONNX`; оно отвечает 200 и")
            a(f"содержит все четыре файла `{n}`. Записано потому, что разница — одно")
            a("слово с дефисом и читается как опечатка, а не как 401.")
            a("")
    a("**Чем зеркало НЕ является.** Ничто здесь не говорит, что зеркало взаимозаменяемо")
    a("с экспортом самой модели. Говорится, что для двух проверенных строк")
    a("упакованный результат совпал; для остальных восьми утверждение кончается на")
    a("*файл есть и форма правильная*. Только у `yolov8n-pose` и `whisper-tiny` есть")
    a("измеренное утверждение, и строки говорят, у каких именно.")
    a("")
    a("## Два репозитория на одну модель: `embeddinggemma-300m`")
    a("")
    a("`models/embeddinggemma-300m/CHECKPOINT.json` записывает")
    a("`unsloth/embeddinggemma-300m`, а каталог в `runtime/src/common/hub.cpp`")
    a("тянет `google/embeddinggemma-300m`. **Это не взаимозаменяемо:** репозиторий")
    a("Google гейтед — нужна принятая лицензия и заголовок `Authorization: Bearer`, и")
    a("`hub.cpp` fail'ится закрыто, а не берёт зеркало. Копия unsloth не гейтед и веси")
    a("те же веса, так что читатель по `CHECKPOINT.json` и читатель `serve` тянут из")
    a("разных мест, и токен нужен только одному. Оба названы в таблице; ни один не")
    a("выдаётся за другой.")
    a("")
    # Список берётся из данных, а не вписан в предложение: этот абзац называл
    # yolov8n-pose и whisper-medium словом «оба», а потом yolov8n-pose получил
    # CHECKPOINT.json, счётчик стал 1, а предложение продолжало называть обоих.
    missing_ck_ru = sorted(n for n in rows if checkpoint(n) is None)
    if missing_ck_ru:
        a(f"### {len(missing_ck_ru)} модел(и) без `CHECKPOINT.json`")
        a("")
        for n in missing_ck_ru:
            a(f"**`{n}`** — собрана и работает здесь, но вообще без записанного")
            a("происхождения: ни строки о репозитории, ни списка файлов, ни хеша. ONNX")
            a("на диске, контейнер собран из него — то есть модель работает; не хватает")
            a("всего, что позволило бы кому-то её воспроизвести.")
        a("")
        a("Написать недостающий `CHECKPOINT.json` — единственная дыра здесь, которую не")
        a("закроет ссылка: не хватает локальной записи, а не удалённого файла. Входные")
        a("данные известны — файл, его sha256 и репозиторий есть в таблице, — так что")
        a("работа короткая, и оставлена намеренно: гейт, выдумывающий происхождение для")
        a("модели, которую не измерял, хуже дыры, которую закрывает.")
    else:
        a("### У каждой модели записано происхождение")
        a("")
        a(f"Во всех {len(rows)} каталогах есть `CHECKPOINT.json`, так что за каждой "
          "строкой")
        a("таблицы стоит пин, который дерево проверяет.")
    a("")
    return "\n".join(L) + "\n"


def check_one(path, body):
    """Is the file on disk exactly what this data would generate?

    BYTE COMPARISON, not a substring search. The substring version is what most
    generated-doc checks do and it has one failure mode that matters here: a
    paragraph that has been deleted from the generator still sits in the .md, and
    a search for the paragraphs that REMAIN finds all of them. That is how a
    document keeps claiming that bge-large is unproven after the download that
    proved it -- the rest of the file was fine.
    """
    if not os.path.exists(path):
        print(f"  FAIL  {os.path.relpath(path, ROOT)} does not exist -- run "
              f"tools/gen_downloads_doc.py")
        return False
    with open(path, encoding="utf-8") as f:
        have = f.read()
    if have == body:
        print(f"  ok    {os.path.relpath(path, ROOT)} matches "
              f"tools/data/model_sources.json")
        return True
    print(f"  FAIL  {os.path.relpath(path, ROOT)} is stale -- run "
          f"tools/gen_downloads_doc.py")
    hl, wl = have.splitlines(), body.splitlines()
    shown = 0
    for i in range(max(len(hl), len(wl))):
        a = hl[i] if i < len(hl) else "<no such line>"
        b = wl[i] if i < len(wl) else "<no such line>"
        if a != b:
            print(f"   line {i + 1}\n     on disk: {a[:110]}\n     generated: {b[:110]}")
            shown += 1
            if shown == 8:
                print("   ... and more")
                break
    return False


def main():
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 and print the diff if either is "
                         "stale. The same flag and the same promise as the other "
                         "generated documents in this tree.")
    ns = ap.parse_args()

    src, tgt = load()
    rows = src["models"]
    problems = cross_check(src, tgt)
    if problems:
        print("FAIL -- the document would state something one of its two sources "
              "contradicts:", file=sys.stderr)
        for p in problems:
            print("  * " + p, file=sys.stderr)
        print("\nNothing written. Fix the data file or CHECKPOINT.json.", file=sys.stderr)
        return 1
    bodies = ((OUT_EN, en_doc(src, rows)), (OUT_RU, ru_doc(src, rows)))
    if ns.check:
        return 0 if all(check_one(p, b) for p, b in bodies) else 1
    for path, body in bodies:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"wrote {os.path.relpath(path, ROOT)} ({body.count(chr(10))} lines)")
    n, with_repo, with_onnx, without, verified, no_ck = counts(rows)
    print(f"  {n} models, {with_repo} on HuggingFace, {with_onnx} with a downloadable "
          f"ONNX, {without} without, {verified} downloaded and hashed, {no_ck} with "
          f"no CHECKPOINT.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())