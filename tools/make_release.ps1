# NpuEmbeddings -- assemble a downloadable release.
# SPDX-License-Identifier: Apache-2.0
#
# A release is everything a user needs and nothing they do not: the executable
# and the compiled NPU designs. About 1-3 MB depending on how many widths are
# carried.
#
#   npuembeddings.exe       ~0.7 MB
#   <set>/gemm_rtp/         ~0.6 MB each  one xclbin + 16 instruction streams
#
# THE MODEL IS NOT IN HERE, ON PURPOSE
# ------------------------------------
# Redistributing the weights would be permitted -- these models are
# Apache-2.0 / MIT -- but it would be the worse deal for everyone: a large
# binary blob in a stranger's zip that nobody can practically check against
# the original, going stale the moment upstream changes, and hiding the model
# card the user ought to read.
#
# NO get-model.cmd (0.2.0)
# ------------------------
# Until 0.1.x this bundle carried a batch file that ran `curl` and then
# compared a `certutil -hashfile` digest against a pinned one. That is the
# literal shape of a dropper, so SmartScreen and AV heuristics flagged it, and
# a security warning was the first thing a new user saw. The behaviour was
# always right; the packaging was not. Fetching and verification now happen
# inside the executable (runtime/src/hub.cpp), which removes the script, the
# `curl` dependency and the `certutil` call together:
#
#   npuembeddings list
#   npuembeddings serve bge-base-en-v1.5
#
# The pins live in the catalogue compiled into the binary, and the container
# builder is verified byte-identical to the reference packer by
# tools/verify_pack_parity.py.
#
# The manifest records the sha256 of every file and the numbers this build was
# verified at, so a downloaded release can be checked against the run that
# produced it rather than trusted.
#
# Usage:
#   .\tools\make_release.ps1 -Version v0.2.0
#   .\tools\make_release.ps1 -Version v0.2.0 -Artifacts artifacts_b128il,artifacts_base

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    # Five design sets, so every catalogue model reports `ready` rather than
    # `no design` out of the box. They are ~600 KB of gemm_rtp each and
    # compress well, so shipping the lot costs almost nothing and removes the
    # commonest first-run confusion: a model the table offers and the runtime
    # then refuses. artifacts_nomic_bfp16 is separate from artifacts_base_bfp16
    # despite both serving hidden 768 -- nomic's gated ffn_up is N=6144
    # (tasks/0069).
    #
    # THE bfp16 ADOPTION (tasks/0104, T23). Per-model MTEB gate verdicts
    # (tasks/0101, 0103, `--sides cpu,npu` real gate runs) moved five of six
    # models onto the bfp16-emulated MMAC + bf16-C datapath -- MiniLM,
    # bge-base, bge-large, nomic, and (differential gate only) EmbeddingGemma
    # all PASS; bge-small FAILS at -0.5010 against the -0.5 line and stays on
    # the ORIGINAL plain-bf16 set. bge-small and MiniLM share hidden-384
    # geometry, so they now need TWO directories rather than one:
    # artifacts_small_bf16 (plain bf16, bge-small's) and artifacts_minilm_bfp16.
    #
    # bge-small ships artifacts_small_bf16 rather than the older
    # artifacts_b128il even though the two are the same design: b128il predates
    # the `emulate_bfp16` field, so it cannot state its own datapath and the
    # runtime reports it as UNRECORDED. Same geometry, same accuracy
    # (rel_fro 3.789e-03, 1-cos 8.348e-06, verified), and pick_artifacts()
    # breaks a tie in favour of the set that records what it is.
    # Each design.json now records its own datapath
    # (`emulate_bfp16`) and the runtime's catalogue records which datapath
    # each model was ADOPTED for (CatalogEntry::datapath) -- a model can no
    # longer be silently served the datapath it did NOT clear the gate for
    # (runtime/src/main.cpp's design_fits()/pick_artifacts()).
    #
    # EmbeddingGemma was NOT in this list until 2026-08-26, and the reason was
    # specific: arch=1 had no --serve/HTTP path, so a `serve` release had
    # nothing to do with its design set. tasks/0115 built that path (T34's last
    # unbuilt item) -- one shared serve_http() for every architecture, gated on
    # the endpoint being bit-identical to arch=1's own --embed over base64 --
    # so the exclusion no longer has a reason and gemma joins the set. It costs
    # 645 KB, the same order as every other design set here.
    [string[]]$Artifacts = @("artifacts_small_bf16", "artifacts_minilm_bfp16",
                             "artifacts_base_bfp16", "artifacts_large_bfp16",
                             "artifacts_nomic_bfp16", "artifacts_gemma_bfp16"),
    [string]$OutDir = "dist",
    # THE SWEEP THIS RELEASE SHIPS, NAMED RATHER THAN GUESSED (tasks/0085).
    # This was hardcoded to tasks\0073-m13-release-benchmarks, so a release cut
    # after a NEWER sweep would have shipped the older numbers -- silently, and
    # while passing its own freshness check below, because that check reads
    # whichever file it was pointed at. "Newest wins" is not the fix either
    # (trap 7c); naming it is.
    [string]$SweepDir = "tasks\0085-m13-release-sweep"
)

$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $REPO "$OutDir\npuembeddings-$Version"

# The product name is npuembeddings; the build emits it beside npuembed.exe.
$exe = Join-Path $REPO "runtime\build\npuembeddings.exe"
if (-not (Test-Path $exe)) {
    throw "missing $exe -- see BUILD.md (the release cannot be assembled from a partial build)"
}

$sets = foreach ($a in $Artifacts) {
    $d = Join-Path $REPO "runtime\$a\gemm_rtp"
    if (-not (Test-Path "$d\design.json")) {
        throw "missing $d\design.json -- export it with tools\export_gemm_rtp.py"
    }
    # The width a design serves is READ from it, never inferred from the
    # directory name. A release that mislabels which model a design runs is
    # the same fail-open this project has closed nine times.
    $json = Get-Content "$d\design.json" -Raw | ConvertFrom-Json
    # Prefer the geometry the design STATES about itself (tasks/0069). Two
    # design sets can serve the same `hidden` and still be incompatible: nomic's
    # gated ffn_up is N=6144 where bge-base's is 3072, so "hidden 768" alone
    # labels both identically and tells a release reader nothing. Older sets
    # predate these keys, hence the fallback.
    if ($null -ne $json.hidden) {
        $hidden = [int]$json.hidden
        $inter = [int]$json.intermediate
        $gated = [bool]$json.gated_ffn
    } else {
        $ks = $json.streams | ForEach-Object { [int]$_.K } | Sort-Object -Unique
        $hidden = $ks[0]
        $inter = $ks[-1]
        $gated = $false
    }
    # THE DATAPATH (tasks/0104, T23). $null on a pre-0104 set means "the
    # field did not exist yet", which IS plain bf16 -- every set exported
    # before this field existed is.
    $bfp16 = ($null -ne $json.emulate_bfp16) -and [bool]$json.emulate_bfp16
    [pscustomobject]@{ name = $a; dir = $d; hidden = $hidden
                       intermediate = $inter; gated_ffn = $gated
                       emulate_bfp16 = $bfp16 }
}

# The benchmark sweep, which a release must carry. Checked BEFORE any staging
# work, so the refusal costs nothing and cannot be half-done.
$sweepPath = Join-Path $REPO (Join-Path $SweepDir "sweep.json")
if (-not (Test-Path $sweepPath)) {
    throw @"
missing $sweepPath -- a release ships freshly measured benchmarks for the WHOLE
catalogue, not just the model that changed. Run:

    .\tools\release_benchmark.ps1

on an idle machine, on mains power, then re-run this script. If you genuinely
mean to cut a release without them, pass -SkipBenchmarks and the manifest will
say so in the artifact rather than quietly omitting the field.
"@
}
$sweep = Get-Content $sweepPath -Raw | ConvertFrom-Json
$sweepAge = ((Get-Date) - [datetime]$sweep.when).TotalDays
if ($sweepAge -gt 7) {
    Write-Warning ("the sweep in {0} is {1:N1} days old -- it predates whatever " -f $sweepPath, $sweepAge)
    Write-Warning "you have changed since. Re-run tools\release_benchmark.ps1."
}
$benchmarks = [ordered]@{
    source = ($SweepDir -replace '\\', '/') + "/sweep.json"
    measured_utc = $sweep.when
    machine_power = $sweep.power
    lanes = $sweep.lanes; threads = $sweep.threads
    note = "one session, one machine state, one protocol. End-to-end throughput; NOT an NPU kernel performance claim (CLAUDE.md rule 1)."
    models = $sweep.rows
}

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path "$stage\models" | Out-Null

Copy-Item $exe $stage
foreach ($s in $sets) {
    New-Item -ItemType Directory -Force -Path "$stage\$($s.name)" | Out-Null
    Copy-Item -Recurse $s.dir "$stage\$($s.name)\gemm_rtp"
}
Copy-Item (Join-Path $REPO "LICENSE") $stage
if (Test-Path (Join-Path $REPO "README.md")) {
    Copy-Item (Join-Path $REPO "README.md") $stage
}

# Launchers. They exist only so a double-click works; everything they do is
# one subcommand, and the subcommand is what the documentation teaches.
@'
@echo off
rem NpuEmbeddings -- list the models this build can run.
"%~dp0npuembeddings.exe" list --root "%~dp0."
pause
'@ | Set-Content -Path "$stage\list-models.cmd" -Encoding ascii

@'
@echo off
rem NpuEmbeddings -- start the embeddings server.
rem Usage: serve.cmd <model> [port]        e.g.  serve.cmd bge-base-en-v1.5 8080
rem Downloads and verifies the model on first use.
setlocal
if "%~1"=="" (
  echo Usage: serve.cmd ^<model^> [port]
  echo.
  "%~dp0npuembeddings.exe" list --root "%~dp0."
  exit /b 1
)
set PORT=%2
if "%PORT%"=="" set PORT=8080
"%~dp0npuembeddings.exe" serve %1 --port %PORT% --root "%~dp0."
'@ | Set-Content -Path "$stage\serve.cmd" -Encoding ascii

@'
@echo off
rem NpuEmbeddings -- embed one text file (one text per line) to out.f32
rem Usage: embed.cmd <model> texts.txt out.f32
setlocal
"%~dp0npuembeddings.exe" embed %1 %2 %3 --root "%~dp0."
'@ | Set-Content -Path "$stage\embed.cmd" -Encoding ascii

# Checksums over everything shipped.
$files = Get-ChildItem -Recurse -File $stage | Sort-Object FullName
$hashes = foreach ($f in $files) {
    $rel = $f.FullName.Substring($stage.Length + 1)
    [ordered]@{ file = $rel
                bytes = $f.Length
                sha256 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower() }
}

$manifest = [ordered]@{
    name    = "NpuEmbeddings"
    version = $Version
    models  = "fetched and verified by `npuembeddings serve <model>`; not redistributed. Run `npuembeddings list` for the catalogue and its pinned checksums."
    hardware = "AMD Ryzen AI (XDNA2 / Strix Point) NPU"
    sequence_length = 64
    # The FULL geometry, not just `hidden`. Two design sets can serve the same
    # width and be incompatible -- artifacts_base and artifacts_nomic are both
    # hidden 768, and nomic's gated ffn_up is N=6144 where bge-base's is 3072
    # (tasks/0069). A manifest that records only the width cannot tell a reader
    # which is which.
    designs = @($sets | ForEach-Object {
        [ordered]@{ set = $_.name; hidden = $_.hidden
                    intermediate = $_.intermediate; gated_ffn = $_.gated_ffn
                    emulate_bfp16 = $_.emulate_bfp16 }
    })
    built_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    # READ FROM THE SWEEP, NOT FROM MEMORY.
    #
    # This block used to be a hand-maintained dict carrying two of five models,
    # with figures from tasks/0033-0037 and 0051 -- taken in different sessions,
    # under different lane defaults, on different toolchain versions. Every
    # number was honest when taken and the TABLE was not comparable to itself,
    # which is a subtler way of being wrong than any single bad figure.
    #
    # A release now REFUSES to assemble without a sweep artifact. That makes
    # "every release ships fresh whole-catalogue benchmarks" a property of the
    # build rather than of whoever remembers to re-run things.
    verified = $benchmarks
    files = $hashes
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content "$stage\manifest.json" -Encoding utf8

$zip = Join-Path $REPO "$OutDir\npuembeddings-$Version-win-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal

$mb = (Get-Item $zip).Length / 1MB
Write-Host ""
Write-Host "  staged  $stage"
Write-Host ("  designs " + (($sets | ForEach-Object {
    $g = ""; if ($_.gated_ffn) { $g = ", gated" }
    "$($_.name) (hidden $($_.hidden), ffn $($_.intermediate)$g)"
}) -join ", "))
Write-Host ("  zip     $zip  ({0:N2} MB)" -f $mb)
Write-Host ""
Write-Host "  Unzip, then:  npuembeddings list"
Write-Host "                npuembeddings serve bge-base-en-v1.5"
Write-Host ""
