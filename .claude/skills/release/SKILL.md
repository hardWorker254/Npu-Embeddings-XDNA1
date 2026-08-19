---
name: release
description: Cut a NpuEmbeddings release — regenerate and push the public repository, bring README/CURRENT_STATUS/CLAUDE.md in line with what actually ships, write the release note, and build and cold-test the dist zip. Use when the user says "release", "cut 0.x.y", "update the repo", "ship it", or asks for release notes.
---

# Cutting a release

Four things, in this order, because each one can invalidate the next:

1. **Prove what ships** — validate on hardware before writing anything down.
2. **Update the documents** that describe it.
3. **Regenerate and push the public repository.**
4. **Build the zip, and cold-test it as a stranger would.**

The order matters. Documents written before validation describe intentions;
a release note written before the cold test describes a bundle nobody has run.

---

## 0. Before anything: what actually changed

Read `research/OPEN-THREADS.md` and the `tasks/` entries since the last
release tag. The release note is written from **task logs**, never from
`CLAUDE.md`'s summaries — a second-hand claim is exactly what this project
does not keep (`tasks/README.md` records the same rule for its own index).

Check the previous release's manifest for the true delta:

```powershell
(Get-Content dist\npuembeddings-<prev>\manifest.json -Raw | ConvertFrom-Json) |
    Select-Object version, models, designs, verified
```

**Version naming: pick one form and keep it.** `dist/` currently holds
`0.1.0`, `v0.1.0` and a stray `0.20`, which is what happens otherwise. This
project uses bare `0.2.0` in `-Version`; the git tag, if any, is the user's
call.

## 1. Prove what ships

Never release a build whose numbers came from an earlier one. For every model
the release claims:

```powershell
# correctness on hardware -- do this first, always
.\runtime\build\npuembed.exe .. --model <name> --artifacts <set> --threads 16

# text in, vector out, against the reference
& .\.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py `
      --model <name> --artifacts <set> --threads 24
```

Gate: `1 - cos <= 2e-03`. Record the actual figure, not "passes".

Throughput, if the note quotes any, obeys `docs/05-measurement/`: `--bench`
refuses to run when another process holds the NPU, and **a ratio against the
CPU is only defensible if both sides were measured interleaved in one
session** (`tasks/0040`). If the CPU side was not re-measured, quote the
NPU figure alone and say so. Do not extrapolate a cost model across a shape
it was not fitted at — that produced a 27% miss in `tasks/0051`.

## 2. Update the documents

Four files drift, and they drift differently:

| file | what must be true |
|---|---|
| `README.md` | user-facing: model table, CLI form, ports, sizes, quickstart |
| `docs/CURRENT_STATUS.md` | the snapshot: model table, "last updated", what runs where |
| `CLAUDE.md` | ground rules and current state, for the next session |
| `tasks/NNNN-*/TASK.md` | the diary entry for this work, with exact commands |

**README is the one users read.** Check it for claims that were true of an
older release: a single model named in prose, a default port, an `.exe` name,
a bundle listing, an embedding dimension in a code sample.

Then add the task log. Failures are the valuable part and are never deleted
or rewritten (`CLAUDE.md` rule 3b), and every open question goes into
`research/OPEN-THREADS.md` — a `TASK.md` is written once and never revisited,
which is the wrong property for a question.

> **Trap, hit in `tasks/0051`.** Do not paste a broken link verbatim into a
> log that documents fixing it. The sync's link checker cannot tell a
> demonstration from a defect, and it will refuse the very file explaining
> the repair. Describe the shape instead.

## 3. Regenerate and push the public repository

`repo/` is the publishable subset; this repository stays private because it
holds indexed summaries of other people's papers.

```powershell
python tools\sync_public_repo.py
```

**This is a gate, not a report. Check the exit code and stop on FAIL.**

> Hit in `tasks/0051`: the commands were chained with `;`, the sync failed,
> and a commit with a dead link went out anyway. The failure mode is silent
> because the push succeeds — git has no opinion about the sync.

```powershell
python tools\sync_public_repo.py
if ($LASTEXITCODE -ne 0) { throw "sync failed -- fix the source, do not push" }
cd repo; git add -A; git commit; git push origin main
```

The sync rewrites links into excluded material (`research/papers/<id>.md`
becomes an arXiv citation) and then **re-scans and refuses** if any survive.
Two failure shapes it catches:

- a link whose target is the excluded **directory** rather than a file
  inside it — the rewriter has nothing to rewrite it to
- a plain typo in a relative path (`tasks/0031-m7-eltwise-il4` when the
  directory is `-ilp`)

Fix them **in this repository**, then re-run. Never patch `repo/` by hand:
it is generated, and the next sync discards the edit.

## 4. Build the zip, and cold-test it

```powershell
cmake --build runtime\build --config Release
.\tools\make_release.ps1 -Version 0.2.0
# several widths -- one design set per hidden size the release supports
.\tools\make_release.ps1 -Version 0.2.0 -Artifacts artifacts_b128il,artifacts_base
```

`make_release.ps1` reads each design's hidden size **out of its own
`design.json`** rather than trusting the directory name, and refuses if a
named set has no design.

### The cold test is not optional

Unzip **outside the repository** and run it as somebody who has never seen
the source:

```powershell
$S = "$env:TEMP\shiptest"; Remove-Item $S -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive dist\npuembeddings-<ver>-win-x64.zip -DestinationPath $S
cd $S
.\npuembeddings.exe                      # help + catalogue
.\npuembeddings.exe list                 # check the printed ROOT
"hello" | Out-File -Encoding ascii in.txt
.\npuembeddings.exe embed <model> in.txt out.f32     # fetch, verify, pack, run
```

Then the check that catches a mispacked release:

```powershell
# the container the release built must equal the one you validated
(Get-FileHash "$S\models\<model>.npue").Hash -eq
    (Get-FileHash "models\<model>.npue").Hash
```

**Also test a release staged *inside* the repository**, which is where
`make_release.ps1` puts it. Root resolution is layout-sensitive and the two
cases fail differently — a release inside the tree once climbed to the repo
root and served the repository's models while claiming to be self-contained
(`tasks/0051`). Everything worked and everything was wrong. Verify the root
`list` prints in **all** of: source tree, staged-inside-repo, unzipped
elsewhere, and `--root` given explicitly.

Because empty directories do not survive a zip, `models/` is absent from a
fresh bundle and gets created by the first fetch. Anything that recognises a
release by looking for `models/` will therefore be wrong.

## 5. The release note

Write it to `dist/RELEASE-NOTES-<ver>.md`. Structure that has worked:

1. **One-line headline** naming the user-visible change, not the internals.
2. **The thing that most affects a new user**, first and in full.
3. **What is new**, with a table of measured figures.
4. **Upgrading from `<prev>`** — every rename, moved path and changed
   default, as a list. This is the section people actually need.
5. **What is in the zip**, and what deliberately is not.
6. **Also in this release** — internal findings, kept short.
7. **Known limits**, stated plainly, including anything measured for some
   models and not others.

Voice: state the figure, state the tolerance, name what was not measured.
Do not quote a CPU ratio that was not measured interleaved. If a prediction
missed, the note can stay silent but the task log may not.

## 6. Outward-facing steps — confirm first

Producing the zip locally is part of the job. **Publishing is not**: creating
a git tag and running `gh release create` pushes artifacts to the internet
under the user's name. Report that the bundle is built and verified, and let
the user say when to publish and under which tag.

## Checklist

- [ ] hardware validation re-run for every model the note names
- [ ] `1 - cos` figures recorded, not "passes"
- [ ] no CPU ratio unless both sides were interleaved in one session
- [ ] README, CURRENT_STATUS, CLAUDE.md, task log updated
- [ ] open questions filed in `research/OPEN-THREADS.md`
- [ ] `sync_public_repo.py` **PASSED** before any push
- [ ] public repo committed and pushed
- [ ] zip built; cold-tested from an unzip outside the repo
- [ ] container from the release byte-identical to the validated one
- [ ] root correct in all four layouts
- [ ] release note written to `dist/RELEASE-NOTES-<ver>.md`
- [ ] tag and GitHub release left for the user to confirm
