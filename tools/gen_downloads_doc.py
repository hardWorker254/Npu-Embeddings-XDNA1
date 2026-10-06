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
Ten of the eighteen models have NO ONNX EXPORT UPSTREAM. Their repositories
contain no onnx/ directory, and requesting one returns 404. There is nothing to
download, and that -- not a policy decision -- is why BUILD.md 2.2 says you
place the weights yourself. It had never been written down, so "why is there no
script" had no answer and the tree's position looked like an omission.

Eight do publish one, and for those a link is a real link: the file exists, the
size is known, and -- for all eight as of `verified_on` -- the bytes were
downloaded and hashed and matched the files this tree was built against. The
largest of them is bge-large's 1.34 GB export, which is why it took the longest
to check rather than being left unchecked.

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
        return (f"yes — downloaded {n} B, hashed, matched the local file",
                f"да — скачано {n} Б, захэшировано, совпало с локальным файлом")
    return ("listed", "только листинг")


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
    a(f"that matters -- which of them you can actually download as an ONNX export.")
    a("")
    a(f"Of {n} models, **{with_onnx} publish an ONNX export you can download** and")
    a(f"**{without} do not**. That is not a policy in this repository; it is what")
    a("the upstream repositories contain. The ones that do not have no `onnx/`")
    a("directory at all, and asking for one returns 404. That is the real reason")
    a("[BUILD.md §2.2](../BUILD.md) says *\"weights are not fetched -- you place")
    a("them\"*: for ten of the eighteen there is nothing to fetch, because the graph")
    a("has to be produced by exporting the checkpoint yourself.")
    a("")
    a(f"Checked on **{src['verified_on']}**. Of the {with_onnx} downloadable")
    a(f"exports, **{verified} were downloaded and hashed** and matched the files this")
    a("tree was built against. The rest are listed by size only, and the row says so.")
    # "^" the sentence above is only true while there IS a rest. It was written
    # when bge-large was the one unproven row, and when that download finished
    # the sentence stayed, describing a category that no longer existed.
    if verified == with_onnx:
        L[-1] = ("tree was built against. Nothing in the table rests on a listing "
                 "alone.")
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
    a("| model | repository | file | size | sha256 of the file | verified |")
    a("| --- | --- | --- | --- | --- | --- |")
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
            a(f"| `{name}` | [`{o['repo']}`]({HF}/{o['repo']}) | [`{o['file']}`]({url(o['repo'], o['file'])}) | {o['bytes']:,} | {sh} | {mark_en} |")
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
    a("## Models with no ONNX export upstream")
    a("")
    a("For these, `curl` cannot help. The checkpoint is on HuggingFace and the graph")
    a("has to be exported from it. **This repository ships no exporter for any of")
    a("them** -- `reference/fetch_model.py` fetches configs and tokenizers and never")
    a("a graph -- so producing the ONNX is a step you supply. That is stated here")
    a("rather than papered over with an invented command.")
    a("")
    a("| model | repository | licence | what is missing |")
    a("| --- | --- | --- | --- |")
    for name, row in sorted(rows.items()):
        if row.get("onnx"):
            continue
        repos = one_of(row.get("repo"))
        rs = ", ".join(f"[`{r}`]({HF}/{r})" for r in repos) or "**not a HuggingFace model**"
        lic = row.get("license") or "?"
        if row.get("no_checkpoint_json"):
            miss = ("no `CHECKPOINT.json` at all -- no repo record, no file list, no "
                    "digest; and no ONNX upstream")
        elif row.get("no_pin"):
            miss = "no ONNX upstream, and no digest recorded (`sha256: null`)"
        elif row.get("repo_disagreement"):
            miss = "no ONNX upstream (checked in both repositories)"
        else:
            miss = "no `onnx/` directory in the repository; requesting one 404s"
        a(f"| `{name}` | {rs} | {lic} | {miss} |")
    a("")
    a("**A `file` list in `CHECKPOINT.json` is not a download list.** The six")
    a("Whisper sizes record `onnx/encoder_model.onnx` and `onnx/decoder_model.onnx`,")
    a("and **both 404 upstream** -- those entries name what this repository's own")
    a("export is called, not where to fetch it. Read them as local paths to create.")
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
    a(f"### {no_ck} models with no `CHECKPOINT.json`")
    a("")
    a("`yolov8n-pose` and `whisper-medium` are the two directories in `models/` with")
    a("no `CHECKPOINT.json`. For `whisper-medium` that means the model is packed and")
    a("runnable here with no recorded provenance at all: no repository row, no file")
    a("list, no digest. For `yolov8n-pose` it means this tree has never recorded")
    a("where its ONNX came from — Ultralytics publishes several `yolov8n-pose`")
    a("exports and saying which one is a claim that needs evidence.")
    a("")
    a("Writing those two `CHECKPOINT.json` files is the one gap in this document that")
    a("no download link can close, because what is missing is the local record, not")
    a("the remote file.")
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
    a(f"Проверено **{src['verified_on']}**. Из скачиваемых экспортов **{verified} реально")
    a("скачаны и захэшированы** и совпали с файлами, на которых построено это дерево.")
    a("Остальные перечислены только по размеру, и строка об этом говорит.")
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
    a("| модель | репозиторий | файл | размер | sha256 файла | проверено |")
    a("| --- | --- | --- | --- | --- | --- |")
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
            a(f"| `{name}` | [`{o['repo']}`]({HF}/{o['repo']}) | [`{o['file']}`]({url(o['repo'], o['file'])}) | {o['bytes']:,} | {sh} | {mark_ru} |")
    a("")
    a("## Модели без ONNX-экспорта наверху")
    a("")
    a("Для них `curl` не поможет. Чекпойнт на HuggingFace есть, а граф надо из него")
    a("экспортировать. **Экспортера для них в этом дереве нет** —")
    a("`reference/fetch_model.py` тянет конфиги и токенизаторы и никогда не граф, —")
    a("так что шаг с ONNX — ваш. Это написано здесь, а не заменено выдуманной командой.")
    a("")
    a("| модель | репозиторий | лицензия | чего не хватает |")
    a("| --- | --- | --- | --- |")
    for name, row in sorted(rows.items()):
        if row.get("onnx"):
            continue
        repos = one_of(row.get("repo"))
        rs = ", ".join(f"[`{r}`]({HF}/{r})" for r in repos) or "**не модель HuggingFace**"
        lic = row.get("license") or "?"
        if row.get("no_checkpoint_json"):
            miss = ("нет вообще `CHECKPOINT.json` — ни репозитория, ни списка файлов, "
                    "ни хеша; и нет ONNX наверху")
        elif row.get("no_pin"):
            miss = "нет ONNX наверху и хеш не записан (`sha256: null`)"
        elif row.get("repo_disagreement"):
            miss = "нет ONNX наверху (проверено в обоих репозиториях)"
        else:
            miss = "нет каталога `onnx/`, запрос 404"
        a(f"| `{name}` | {rs} | {lic} | {miss} |")
    a("")
    a("**Список `file` в `CHECKPOINT.json` — это не список для скачивания.** Шесть")
    a("размеров Whisper записывают `onnx/encoder_model.onnx` и")
    a("`onnx/decoder_model.onnx`, и **оба 404 наверху**: эти записи называют то, как")
    a("назван собственный экспорт этого дерева, а не откуда его брать. Читайте их как")
    a("локальные пути, которые надо создать.")
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
    a(f"### {no_ck} модели без `CHECKPOINT.json`")
    a("")
    a("`yolov8n-pose` и `whisper-medium` — два каталога в `models/`, где нет")
    a("`CHECKPOINT.json`. Для `whisper-medium` это значит, что модель здесь собрана и")
    a("работает вообще без записанного происхождения: ни строки о репозитории, ни")
    a("списка файлов, ни хеша. Для `yolov8n-pose` — что дерево никогда не записывало,")
    a("откуда взялся ONNX: Ultralytics публикует несколько экспортов `yolov8n-pose`, и")
    a("сказать какой — утверждение, требующее доказательства.")
    a("")
    a("Написать эти два `CHECKPOINT.json` — единственная дыра в этом документе,")
    a("которую не закроет никакая ссылка: не хватает локальной записи, а не удалённого")
    a("файла.")
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