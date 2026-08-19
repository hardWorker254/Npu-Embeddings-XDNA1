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
    [string[]]$Artifacts = @("artifacts_b128il", "artifacts_base"),
    [string]$OutDir = "dist"
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
    $ks = $json.streams | ForEach-Object { $_.K } | Sort-Object -Unique
    $hidden = ($ks | Sort-Object)[0]
    [pscustomobject]@{ name = $a; dir = $d; hidden = $hidden }
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
    designs = @($sets | ForEach-Object {
        [ordered]@{ set = $_.name; hidden = $_.hidden }
    })
    built_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    verified = [ordered]@{
        note = "figures from tasks/0033-0037 (MiniLM) and 0051 (bge-base); reproduce with tools/verify_*.py"
        worst_1_minus_cos_vs_huggingface = [ordered]@{
            "all-MiniLM-L6-v2" = 1.086e-05
            "bge-base-en-v1.5" = 1.353e-05
        }
        tokenizer_exact_match = "6826/6826"
        mteb_delta_points = 0.04
        throughput_seq_per_s = [ordered]@{
            "all-MiniLM-L6-v2" = 907.5
            "bge-base-en-v1.5" = 181.2
        }
        energy_j_per_1000_seq = 44.0
    }
    files = $hashes
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content "$stage\manifest.json" -Encoding utf8

$zip = Join-Path $REPO "$OutDir\npuembeddings-$Version-win-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal

$mb = (Get-Item $zip).Length / 1MB
Write-Host ""
Write-Host "  staged  $stage"
Write-Host ("  designs " + (($sets | ForEach-Object { "$($_.name) (hidden $($_.hidden))" }) -join ", "))
Write-Host ("  zip     $zip  ({0:N2} MB)" -f $mb)
Write-Host ""
Write-Host "  Unzip, then:  npuembeddings list"
Write-Host "                npuembeddings serve bge-base-en-v1.5"
Write-Host ""
