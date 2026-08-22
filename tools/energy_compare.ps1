# NpuEmbeddings -- joules per 1000 sequences: NPU vs CPU (tasks/0034).
# SPDX-License-Identifier: Apache-2.0
#
# THE DIFFERENTIAL METHOD, and why it is the whole design
# ------------------------------------------------------
# Measuring one run conflates the work with process startup, model load, xclbin
# registration and weight staging. So every configuration is measured TWICE, at
# a low and a high encode count, and the answer is the difference:
#
#     J_per_encode = (E_high - E_low) / (n_high - n_low)
#
# Everything that happens once -- startup, load, staging, the harness itself --
# is identical in both runs and cancels exactly. What remains is the marginal
# cost of an encode, which is the quantity the claim is about.
#
# The package RAPL meter covers CPU cores AND the NPU block (proven in 0034's
# control experiment: a pure-dispatch soak with zero host work raises package
# power 9.1 W over idle, against 3.9 W for the one thread that drives it). So
# both sides are measured by the same instrument, and the systematic errors of
# that instrument cancel in the comparison -- which is what makes this a
# defensible claim without external instrumentation.
#
# Usage:
#   .\tools\energy_compare.ps1                       # full matrix
#   .\tools\energy_compare.ps1 -Low 10 -High 30      # quicker

[CmdletBinding()]
param(
    [int]$Low = 20,              # encodes in the low run
    [int]$High = 60,             # encodes in the high run
    [int]$Batch = 128,
    # WHICH MODEL. There was no such parameter (tasks/0071): every npuembed
    # invocation below omitted --model, which has been REQUIRED since
    # tasks/0038 made selection explicit once a second model existed. So this
    # script has been unrunnable since then, and nobody noticed -- because the
    # one energy figure on record was taken in tasks/0034, back when there was
    # only one model to measure. That is also why energy exists for MiniLM and
    # for nothing else.
    [string]$Model = "all-MiniLM-L6-v2",
    [string]$Artifacts = "artifacts_b128il",
    [int]$Threads = 24,
    # Lanes. The production default moved 2 -> 4 in tasks/0052; this script
    # still measured 2, so its figure described a configuration we no longer
    # ship.
    [int]$Lanes = 4,
    [int]$Idle = 15,
    [int]$Repeats = 1,
    # PER MODEL. A constant output path is an A/B waiting to overwrite its own
    # baseline (CLAUDE.md, tasks/0045) -- a whole-catalogue sweep through
    # tasks\0034-m8-energy would have left one model's numbers.
    [string]$OutDir = ""
)

if (-not $OutDir) { $OutDir = "tasks\0073-m13-release-benchmarks\energy-$Model" }

$REPO = Split-Path -Parent $PSScriptRoot
$RUNTIME = Join-Path $REPO "runtime"
$OUT = Join-Path $REPO $OutDir
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$measure = Join-Path $PSScriptRoot "measure_energy.ps1"

function Run-Pair {
    param([string]$Name, [string]$WorkDir, [scriptblock]$CmdFor, [int]$SeqPerEncode)

    $lo = & $measure -Label "$Name-$Low" -WorkDir $WorkDir -Command (& $CmdFor $Low) `
        -Idle $Idle -Repeats $Repeats -Out (Join-Path $OUT "$Name-lo.json")
    $hi = & $measure -Label "$Name-$High" -WorkDir $WorkDir -Command (& $CmdFor $High) `
        -Idle $Idle -Repeats $Repeats -Out (Join-Path $OUT "$Name-hi.json")

    # Differential on the MEANS. Note this subtraction never touches the idle
    # baseline -- an unstable idle window degrades the "marginal" figure in the
    # per-run logs but cannot move this number, which is the whole point of
    # measuring at two encode counts instead of subtracting an idle estimate.
    $dJ = $hi.mean_joules - $lo.mean_joules
    # ordered hashtables: Measure-Object -Property cannot see into them on
    # Windows PowerShell 5.1, so project the field first.
    $loS = ($lo.runs | ForEach-Object { $_.seconds } | Measure-Object -Average).Average
    $hiS = ($hi.runs | ForEach-Object { $_.seconds } | Measure-Object -Average).Average
    $dS = $hiS - $loS
    $dN = $High - $Low
    $jPerEncode = $dJ / $dN
    $jPer1k = $jPerEncode / $SeqPerEncode * 1000.0
    $seqPerS = if ($dS -gt 0) { $dN * $SeqPerEncode / $dS } else { [double]::NaN }
    $wDuring = if ($dS -gt 0) { $dJ / $dS } else { [double]::NaN }

    Write-Host ""
    Write-Host ("  >> {0}: {1:N1} J / 1000 seq   ({2:N1} W during, {3:N1} seq/s)" -f `
        $Name, $jPer1k, $wDuring, $seqPerS) -ForegroundColor Magenta
    Write-Host ""

    [ordered]@{
        name = $Name; low_encodes = $Low; high_encodes = $High
        seq_per_encode = $SeqPerEncode
        delta_joules = $dJ; delta_seconds = $dS
        j_per_encode = $jPerEncode; j_per_1000_seq = $jPer1k
        watts_during = $wDuring; seq_per_s = $seqPerS
        idle_w = ($lo.idle_w + $hi.idle_w) / 2.0
    }
}

$results = @()

Write-Host "=== CPU: sentence-transformers, batch $Batch ===" -ForegroundColor Cyan
$py = Join-Path $REPO ".venv-ref\Scripts\python.exe"
$results += Run-Pair -Name "cpu-st" -WorkDir $REPO -SeqPerEncode $Batch -CmdFor {
    param($n)
    "`"$py`" experiments\m8-npu-vs-cpu\energy_cpu_load.py --encodes $n --batch $Batch --model $Model"
}

Write-Host "=== NPU: single lane ===" -ForegroundColor Cyan
$results += Run-Pair -Name "npu-single" -WorkDir $RUNTIME -SeqPerEncode $Batch -CmdFor {
    param($n)
    ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --bench $n"
}

Write-Host "=== NPU: pipelined, $Lanes lanes ===" -ForegroundColor Cyan
# --bench N with --pipeline L runs N GROUPS of L encodes, so divide the counts
# by L to keep the encode totals identical to the other two configurations.
$script:Low = [int]($Low / $Lanes); $script:High = [int]($High / $Lanes)
$results += Run-Pair -Name "npu-pipe$Lanes" -WorkDir $RUNTIME -SeqPerEncode ($Lanes * $Batch) -CmdFor {
    param($n)
    ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --pipeline $Lanes --bench $n"
}
$script:Low = $Low * $Lanes; $script:High = $High * $Lanes

Write-Host ""
Write-Host "================ RESULT ================" -ForegroundColor Green
Write-Host ("  {0,-12} {1,14} {2,10} {3,12}" -f "config", "J / 1000 seq", "W", "seq/s")
foreach ($r in $results) {
    Write-Host ("  {0,-12} {1,14:N1} {2,10:N1} {3,12:N1}" -f `
        $r.name, $r.j_per_1000_seq, $r.watts_during, $r.seq_per_s)
}
$cpu = $results | Where-Object { $_.name -eq "cpu-st" }
foreach ($r in $results | Where-Object { $_.name -ne "cpu-st" }) {
    Write-Host ("  {0}: {1:N2}x better energy per sequence than CPU" -f `
        $r.name, ($cpu.j_per_1000_seq / $r.j_per_1000_seq)) -ForegroundColor Yellow
}

$payload = [ordered]@{
    kind = "hardware measurement"; task = "0034"
    method = "differential: E(high) - E(low) over encode counts, package RAPL"
    meter = "RAPL_Package0_PKG"; energy_unit_j = 3.6e-9
    batch = $Batch; threads = $Threads; artifacts = $Artifacts
    low_encodes = $Low; high_encodes = $High
    results = $results
}
$payload | ConvertTo-Json -Depth 6 |
    Set-Content -Path (Join-Path $OUT "energy_compare.json") -Encoding utf8
Write-Host ("  wrote " + (Join-Path $OutDir "energy_compare.json"))
