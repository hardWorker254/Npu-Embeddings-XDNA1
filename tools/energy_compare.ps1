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
    # WHICH MODEL THE CPU ARM LOADS. `$Model` names a .npue container; the CPU
    # arm loads a HUGGINGFACE checkpoint directory, and for an int8 container
    # no such directory exists -- `models\all-MiniLM-L6-v2.int8\` is not a
    # thing. Left to default, every int8 row's CPU run died instantly and the
    # differential came out at -0.8 to 1.2 J/1000 seq: a number that is not
    # merely wrong but the wrong SIGN, reported without complaint (tasks/0085).
    #
    # Defaulting to $Model keeps every existing caller identical.
    [string]$CpuModel = "",
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

# WHICH ARCHITECTURE, read from the container's own header rather than from a
# flag or the model's name (0075). arch=1 has no `--bench` mode -- it measures
# by encoding a real corpus -- so the two NPU commands below differ, and
# guessing wrong would silently measure nothing (`--bench` is simply ignored by
# that path, so the process would exit after one demo encode and the
# differential would be noise).
#
# The .npue header is `magic[4] version[4] arch[4] flags[4] ...`, so arch is a
# little-endian uint32 at byte 8.
$container = Join-Path $REPO "models\$Model.npue"
if (-not (Test-Path $container)) { throw "no container at $container" }

# Resolve the CPU arm's checkpoint, and REFUSE rather than measure nothing.
# The differential method cannot tell "the load ran and used little energy"
# from "the load died in 0.2 s", so a missing checkpoint has to be caught here
# or it is reported as a small negative number (tasks/0085).
if (-not $CpuModel) { $CpuModel = $Model }
$cpuDir = Join-Path $REPO "models\$CpuModel"
if (-not (Test-Path $cpuDir)) {
    throw @"
no HuggingFace checkpoint at $cpuDir, so the CPU arm cannot run.

`$Model names a .npue CONTAINER; the CPU arm loads a checkpoint DIRECTORY, and
for a quantised container there is none -- an int8 model's CPU baseline is the
bf16 model it was quantised from. Pass -CpuModel with that name.
"@
}
$hdr = [byte[]]::new(16)
$fs = [IO.File]::OpenRead($container)
try { $null = $fs.Read($hdr, 0, 16) } finally { $fs.Dispose() }
$arch = [BitConverter]::ToUInt32($hdr, 8)
$isGemma = ($arch -eq 1)
Write-Host ("container arch = {0}{1}" -f $arch,
            $(if ($isGemma) { " (arch=1: --embed corpus, no --bench)" } else { "" })) -ForegroundColor Yellow

# arch=1's NPU side encodes a FILE, so the corpus has to exist before the
# command string is built. One file per encode-count, each holding
# encodes*Batch lines, so "an encode" means the same amount of work as it does
# on the CPU side (energy_cpu_load.py --encodes n --batch B).
$corpusFor = @{}
if ($isGemma) {
    $src = Get-Content (Join-Path $REPO "tasks\0074-m13-gemma-on-npu\corpus.txt")
    foreach ($n in @($Low, $High, [int]($Low / $Lanes), [int]($High / $Lanes))) {
        if ($corpusFor.ContainsKey($n)) { continue }
        $need = $n * $Batch
        $lines = New-Object System.Collections.Generic.List[string]
        while ($lines.Count -lt $need) { foreach ($l in $src) { if ($lines.Count -lt $need) { $lines.Add($l) } } }
        $f = Join-Path $OUT "corpus-$n.txt"
        # NO -Encoding utf8 on Windows PowerShell 5.1: it writes a BOM, which
        # becomes part of line 0 and silently changes the first sequence
        # (tasks/0074 spent a debugging cycle on exactly this).
        [IO.File]::WriteAllLines($f, $lines, (New-Object Text.UTF8Encoding $false))
        $corpusFor[$n] = $f
    }
}

$results = @()

Write-Host "=== CPU: sentence-transformers, batch $Batch ===" -ForegroundColor Cyan
$py = Join-Path $REPO ".venv-ref\Scripts\python.exe"
$results += Run-Pair -Name "cpu-st" -WorkDir $REPO -SeqPerEncode $Batch -CmdFor {
    param($n)
    "`"$py`" experiments\m8-npu-vs-cpu\energy_cpu_load.py --encodes $n --batch $Batch --model $CpuModel"
}

Write-Host "=== NPU: single lane ===" -ForegroundColor Cyan
$results += Run-Pair -Name "npu-single" -WorkDir $RUNTIME -SeqPerEncode $Batch -CmdFor {
    param($n)
    if ($isGemma) {
        ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --pipeline 1 --embed `"$($corpusFor[$n])`""
    } else {
        ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --bench $n"
    }
}

Write-Host "=== NPU: pipelined, $Lanes lanes ===" -ForegroundColor Cyan
# --bench N with --pipeline L runs N GROUPS of L encodes, so divide the counts
# by L to keep the encode totals identical to the other two configurations.
#
# arch=1 is NOT like that: `--embed` encodes one corpus of a fixed size, and
# lanes only decide how that fixed work is split across concurrent encoders.
# So its counts stay put -- dividing them would have measured a quarter of the
# work while labelling it the same, i.e. a 4x energy "win" that is entirely an
# accounting error.
if (-not $isGemma) {
    $script:Low = [int]($Low / $Lanes); $script:High = [int]($High / $Lanes)
}
$results += Run-Pair -Name "npu-pipe$Lanes" -WorkDir $RUNTIME `
    -SeqPerEncode $(if ($isGemma) { $Batch } else { $Lanes * $Batch }) -CmdFor {
    param($n)
    if ($isGemma) {
        ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --pipeline $Lanes --embed `"$($corpusFor[$n])`""
    } else {
        ".\build\npuembed.exe .. --model $Model --artifacts $Artifacts --threads $Threads --pipeline $Lanes --bench $n"
    }
}
if (-not $isGemma) { $script:Low = $Low * $Lanes; $script:High = $High * $Lanes }

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
