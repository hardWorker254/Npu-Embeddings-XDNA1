# NpuEmbeddings -- the whole-catalogue benchmark sweep a release ships with.
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS EXISTS
# ---------------
# Before this script, each release's numbers were gathered piecemeal across
# sessions: throughput from one night, the interleaved CPU ratio from another,
# energy from tasks/0034 (MiniLM only, and with a script that could not even
# name a second model), MTEB for two of five models. Every individual number was
# honest when taken. The TABLE was not comparable to itself -- different lane
# defaults (2 vs 4), different mlir-aie versions, different machine states.
#
# A row-by-row patchwork misleads even when no row lies. So the sweep runs the
# whole catalogue in ONE session, on one machine state, with one protocol.
#
# WHAT IT REFUSES TO DO
# ---------------------
#  * It does not pass --allow-contention, ever. tasks/0044's ninth fail-open
#    read 221.4 seq/s against a true 691.0 (3.1x) because a stale npuembed held
#    an Active hw_context. Rule 1's usual mitigation does not cover it either:
#    interleaving corrects drift that hits BOTH sides, while a resident NPU
#    context hits only ours, so it makes the RATIO confidently wrong.
#  * It does not report wall clock as an NPU performance claim. Throughput here
#    is end-to-end throughput and is labelled as such.
#  * It does not write two models' results to one path (tasks/0045's bug class).
#
# Usage:
#   .\tools\release_benchmark.ps1                     # everything
#   .\tools\release_benchmark.ps1 -Skip mteb,energy   # the quick pass
#   .\tools\release_benchmark.ps1 -Models nomic-embed-text-v1.5

[CmdletBinding()]
param(
    [string[]]$Models = @(),
    [string[]]$Skip = @(),          # accuracy | throughput | interleaved | energy | mteb
    [int]$Bench = 5,
    [int]$Threads = 24,
    [int]$Lanes = 4,
    [int]$Rounds = 8,
    # Passed through to energy_compare.ps1's differential method: encodes in the
    # low and high runs. Everything that happens once -- startup, model load,
    # weight staging -- is identical in both and cancels in the subtraction, so
    # the gap between them is what sets the signal-to-noise.
    [int]$EnergyLow = 20,
    [int]$EnergyHigh = 60,
    # Measure anyway on a busy machine. Named to match --allow-contention's
    # spirit on the NPU side: available, loud, and it taints every ratio.
    [switch]$AllowCpuContention,
    [string]$OutDir = "tasks\0073-m13-release-benchmarks"
)

$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot

# Run a native command, tee its combined output to a log, and return its REAL
# exit code.
#
# The obvious `& $exe args 2>&1 | Tee-Object $log` is a trap on Windows
# PowerShell 5.1 (CLAUDE.md's own warning): redirecting a native command's
# stderr inside PowerShell wraps every line in an ErrorRecord, which under
# `$ErrorActionPreference = "Stop"` aborts the whole sweep on a program that
# merely printed to stderr and returned 0. This sweep's most important output
# -- the applied task prefix -- goes to stderr, so it cannot simply be dropped.
# Letting cmd.exe do the redirection means the OS joins the streams and
# PowerShell only ever sees a file.
# NOTE the parameter name. It was `$Args` first, which is an AUTOMATIC
# PowerShell variable holding a function's unbound arguments -- so the named
# parameter never bound, every command was launched with NO arguments, and
# `python.exe` with no argument is an interactive REPL. With stdin at EOF under
# cmd redirection it error-looped and wrote a 1 GB log before it was caught.
# Hence also the `--- command:` line below: a malformed invocation is now
# visible in the first line of its own log rather than inferred from the wreckage.
function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments, [string]$Log,
          [string]$WorkDir = $REPO)
    if (-not $Arguments -or $Arguments.Count -eq 0) {
        throw "Invoke-Logged called with no arguments for $Exe -- refusing (an argument-less interpreter is an interactive REPL)"
    }
    # -u on the interpreter. Python buffers stdout when it is redirected, so a
    # long stage looked completely stalled -- an empty log for twenty minutes
    # while it was in fact several rounds in. Progress you cannot see is
    # indistinguishable from a hang, and the reflex on a hang is to kill it.
    if ($Exe -like "*python.exe" -and $Arguments[0] -notlike "-*") {
        $Arguments = @("-u") + $Arguments
    }
    $quoted = @("`"$Exe`"") + ($Arguments | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ } })
    $line = ($quoted -join ' ')
    # ROTATE, never truncate. Re-measuring a model -- which is exactly what you
    # do when a result looks wrong -- overwrote the result you were comparing
    # against. That is tasks/0045's "any script writing a result to a constant
    # path is an A/B waiting to overwrite its baseline", and this file has now
    # hit it twice: once in sweep.json, once here, while fixing other people's
    # instances of it. A re-run keeps the previous log as <name>.runN.txt.
    if ((Test-Path $Log) -and (Get-Item $Log).Length -gt 0) {
        $n = 1
        $base = [IO.Path]::ChangeExtension($Log, $null).TrimEnd('.')
        while (Test-Path "$base.run$n.txt") { $n++ }
        Move-Item $Log "$base.run$n.txt"
        Write-Host "    (previous log kept as $(Split-Path -Leaf "$base.run$n.txt"))" -ForegroundColor DarkGray
    }
    "--- command: $line" | Set-Content $Log
    Push-Location $WorkDir
    try {
        cmd /c "$line >> `"$Log`" 2>&1"
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if (Test-Path $Log) { Get-Content $Log -Tail 400 | Write-Host }
    return $code
}
$RUNTIME = Join-Path $REPO "runtime"
$EXE = Join-Path $RUNTIME "build\npuembed.exe"
$PY = Join-Path $REPO ".venv-ref\Scripts\python.exe"
$OUT = Join-Path $REPO $OutDir
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# The design set per model. `pick_artifacts` resolves this by itself since
# tasks/0069, but the measurement is pinned explicitly so a rerun months from
# now measures the same pairing rather than whatever happens to be on disk.
# `npu = $false` means there is no NPU kernel for the architecture at all --
# arch=1 runs entirely on the host (tasks/0064), so throughput, the interleaved
# ratio and energy are not comparable quantities for it and are skipped rather
# than quietly reported next to the others.
$CATALOG = @(
    @{ name = "all-MiniLM-L6-v2";      artifacts = "artifacts_b128il"; npu = $true  }
    @{ name = "bge-small-en-v1.5";     artifacts = "artifacts_b128il"; npu = $true  }
    @{ name = "bge-base-en-v1.5";      artifacts = "artifacts_base";   npu = $true  }
    @{ name = "bge-large-en-v1.5";     artifacts = "artifacts_large";  npu = $true  }
    @{ name = "nomic-embed-text-v1.5"; artifacts = "artifacts_nomic";  npu = $true  }
    @{ name = "embeddinggemma-300m";   artifacts = "";                 npu = $false }
)

# Split on commas ourselves. `powershell -File script.ps1 -Skip a,b,c` hands the
# whole "a,b,c" through as ONE string rather than an array -- unlike dot-sourcing
# or `-Command`, where PowerShell parses it. So -Skip silently matched nothing
# and the first whole-catalogue run went on to do the very stage it was told to
# skip (tasks/0073). Accept both shapes rather than depending on how the script
# happens to be invoked.
$Skip = @($Skip | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
          Where-Object { $_ })
$Models = @($Models | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
            Where-Object { $_ })
if ($Skip) { Write-Host "skipping stages: $($Skip -join ', ')" -ForegroundColor DarkYellow }
if ($Models.Count) {
    $CATALOG = $CATALOG | Where-Object { $Models -contains $_.name }
    if (-not $CATALOG) { throw "no catalogue entry matched -Models $($Models -join ',')" }
}
function Want([string]$stage) { return -not ($Skip -contains $stage) }

# --- machine state, recorded BY TOOL ------------------------------------------
# tasks/0040: a hand-rolled check reported "ON BATTERY" for a machine with no
# battery, because Win32_Battery returns nothing there and the else branch
# fired. An absent data source is not a negative reading, so report what the
# query actually returned.
$bat = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)
$power = if ($bat.Count -eq 0) { "no battery device reported (desktop, or the class is empty)" }
         elseif ($bat[0].BatteryStatus -eq 2) { "on AC" }
         else { "ON BATTERY -- throughput and energy are not comparable" }
Write-Host "power: $power" -ForegroundColor Yellow

# --- CPU CONTENTION GUARD -----------------------------------------------------
# The counterpart to the NPU guard, and it exists because of a real incident
# (tasks/0073): a stray `find` left over from an unrelated command burned ONE
# FULL CORE continuously for two and a half hours, straight through an entire
# interleaved measurement series. The effect was exactly what you would predict
# and exactly what confused us: the CPU sides came out ~30% below their recorded
# figures with wild variance, while the NPU side -- which offloads the work to
# the array -- barely moved. So the RATIO looked like a large improvement.
#
# tasks/0044 built a guard against a foreign process holding an NPU context, on
# the reasoning that contention hitting only ONE side makes the ratio
# confidently wrong. That reasoning applies just as well to the CPU side, and
# nothing checked it. It does now.
#
# Cumulative CPU time is useless here -- a process that finished an hour ago
# still shows a large total. Only the DELTA over a sampling window says whether
# something is burning CPU right now.
function Test-CpuQuiet {
    param([int]$Samples = 6, [int]$IntervalMs = 500, [double]$MaxPercent = 12.0)
    $before = @{}
    foreach ($p in Get-Process) { $before[$p.Id] = $p.CPU }
    $loads = @()
    for ($i = 0; $i -lt $Samples; $i++) {
        $loads += (Get-CimInstance Win32_Processor |
                   Measure-Object -Property LoadPercentage -Average).Average
        Start-Sleep -Milliseconds $IntervalMs
    }
    $window = ($Samples * $IntervalMs) / 1000.0
    $busy = @()
    foreach ($p in Get-Process) {
        if (-not $before.ContainsKey($p.Id)) { continue }
        $d = $p.CPU - $before[$p.Id]
        # >20% of one core, sustained across the window, and not this sweep.
        if ($d -gt ($window * 0.2) -and $p.Name -notin @("npuembed", "python", "powershell", "cmd", "claude", "node")) {
            $busy += [pscustomobject]@{ Name = $p.Name; Id = $p.Id
                                        Cores = [math]::Round($d / $window, 2) }
        }
    }
    $mean = ($loads | Measure-Object -Average).Average
    return [pscustomobject]@{ MeanPercent = $mean; Busy = $busy
                              Quiet = ($mean -le $MaxPercent -and $busy.Count -eq 0) }
}

# Only TIMING stages care. MTEB measures embedding QUALITY -- scores, not
# seconds -- so a busy CPU cannot move its numbers, and refusing to run it on a
# machine that happens to be playing music would be a guard protecting nothing
# at the cost of the longest stage in the sweep. Scope the check to what it
# actually defends.
$timingStages = @("throughput", "interleaved", "energy") | Where-Object { Want $_ }
$cpu = Test-CpuQuiet
Write-Host ("cpu: {0:N1}% mean over the sampling window" -f $cpu.MeanPercent) -ForegroundColor Yellow
if (-not $timingStages) {
    Write-Host "  no timing stage requested -- CPU quiet check is advisory only" -ForegroundColor DarkGray
}
if ($cpu.Busy.Count) {
    Write-Host "  processes burning CPU right now:" -ForegroundColor Red
    $cpu.Busy | ForEach-Object { Write-Host ("    {0} (pid {1}) ~{2} core(s)" -f $_.Name, $_.Id, $_.Cores) -ForegroundColor Red }
}
if ($timingStages -and -not $cpu.Quiet -and -not $AllowCpuContention) {
    throw @"
the machine is not idle, and a CPU-side contender makes the NPU/CPU ratio
confidently WRONG rather than merely noisy -- it slows the CPU sides while the
NPU path, which offloads to the array, barely notices.

Close what is running and re-run, or pass -AllowCpuContention to measure anyway
(in which case no ratio from this sweep is a defensible comparison, and the
artifact will say so).
"@
}
if ($timingStages -and -not $cpu.Quiet) {
    Write-Host "  -AllowCpuContention given: ratios from this sweep are NOT defensible." -ForegroundColor Red
}

$stamp = (Get-Date).ToString("s")
$summary = @()

foreach ($m in $CATALOG) {
    $name = $m.name
    $art = $m.artifacts
    Write-Host ""
    Write-Host "############ $name ############" -ForegroundColor Green
    $row = [ordered]@{ model = $name; artifacts = $art; npu = $m.npu }

    if (-not (Test-Path (Join-Path $REPO "models\$name.npue"))) {
        Write-Host "  not installed -- skipping" -ForegroundColor DarkYellow
        $row.skipped = "container absent"
        $summary += $row; continue
    }

    # --- accuracy: reproduce the recorded 1-cos against the goldens -----------
    if (Want "accuracy") {
        Write-Host "-- accuracy (golden check)" -ForegroundColor Cyan
        $log = Join-Path $OUT "accuracy-$name.txt"
        if ($m.npu) {
            $null = Invoke-Logged -Exe $EXE -Log $log -Arguments @($REPO, "--model", $name, "--artifacts", $art, "--threads", "16")
        } else {
            # arch=1 has no golden check mode; its correctness gate is
            # tools/verify_gemma_cpu_encode.py (tasks/0064), not this one.
            "arch=1 host path -- no golden check mode; see tools/verify_gemma_cpu_encode.py" |
                Tee-Object $log
        }
        $row.accuracy_log = $log
    }

    # --- throughput: guarded, three runs, spread reported --------------------
    if ((Want "throughput") -and $m.npu) {
        Write-Host "-- throughput ($Lanes lanes, contention guard ON)" -ForegroundColor Cyan
        $log = Join-Path $OUT "throughput-$name.txt"
        "" | Set-Content $log
        for ($i = 1; $i -le 3; $i++) {
            $one = Join-Path $OUT "throughput-$name.run$i.txt"
            $code = Invoke-Logged -Exe $EXE -Log $one -Arguments @($REPO, "--model", $name, "--artifacts", $art, "--threads", "$Threads", "--pipeline", "$Lanes", "--bench", "$Bench")
            Get-Content $one | Add-Content $log
            if ($code -ne 0) {
                Write-Host "  REFUSED (exit $code) -- a foreign hw_context is Active, or xrt-smi could not be parsed. NOT retrying with --allow-contention." -ForegroundColor Red
                $row.throughput_error = "contention guard refused (exit $code)"
                break
            }
        }
        $row.throughput_log = $log
    }

    # --- interleaved NPU vs torch vs ORT, one session, same statistic --------
    if ((Want "interleaved") -and $m.npu) {
        Write-Host "-- interleaved CPU ratio ($Rounds rounds)" -ForegroundColor Cyan
        $log = Join-Path $OUT "interleaved-$name.txt"
        $null = Invoke-Logged -Exe $PY -Log $log -Arguments @(
            (Join-Path $REPO "experiments\m8-npu-vs-cpu\compare_three.py"),
            "--model", $name, "--artifacts", $art, "--rounds", "$Rounds",
            "--threads", "$Threads", "--pipeline", "$Lanes",
            # Land the JSON with the sweep rather than in the harness's own
            # artifacts dir. A re-measurement replaces it; the rotated .runN
            # text logs beside it are what keep the earlier reading.
            "--out", (Join-Path $OUT "interleaved-$name.json"))
        $row.interleaved_log = $log
    }

    # --- energy, differential method ----------------------------------------
    if ((Want "energy") -and $m.npu) {
        Write-Host "-- energy (RAPL, differential)" -ForegroundColor Cyan
        $log = Join-Path $OUT "energy-$name.txt"
        & (Join-Path $PSScriptRoot "energy_compare.ps1") -Model $name `
            -Artifacts $art -Threads $Threads -Lanes $Lanes `
            -Low $EnergyLow -High $EnergyHigh `
            -OutDir "$OutDir\energy-$name" 2>&1 | Tee-Object $log
        $row.energy_log = $log
    }

    # --- MTEB, the accuracy gate that is about QUALITY not fidelity ----------
    if (Want "mteb") {
        Write-Host "-- MTEB (cpu + npu, one session)" -ForegroundColor Cyan
        $log = Join-Path $OUT "mteb-$name.txt"
        if ($m.npu) {
            $code = Invoke-Logged -Exe $PY -Log $log -Arguments @(
                (Join-Path $REPO "experiments\m8-npu-vs-cpu\run_mteb.py"),
                "--model", $name, "--artifacts", $art, "--threads", "$Threads",
                "--pipeline", "$Lanes",
                "--out", (Join-Path $OUT "mteb-$name.json"))
            $row.mteb_pass = ($code -eq 0)
        } else {
            # Be explicit rather than silently absent: at ~7.9 s/sentence
            # (tasks/0064) a full MTEB run on the host path is impractical, and
            # "unmeasured" is a result that has to be stated.
            "arch=1 host path at ~7.9 s/sentence -- a full MTEB run is impractical. UNMEASURED, deliberately." |
                Tee-Object $log
            $row.mteb_pass = $null
        }
        $row.mteb_log = $log
    }

    $summary += $row
}

$sum = Join-Path $OUT "sweep.json"

# MERGE, DO NOT CLOBBER. The sweep is meant to be runnable in stages -- a full
# pass with MTEB takes hours, and staging it means a failure costs one stage
# rather than all of them. But writing this index fresh each time made each
# stage delete the previous one's rows: the very "constant path overwrites its
# own baseline" bug (CLAUDE.md, tasks/0045) that this script's own header warns
# about, reintroduced one level up. Per-model rows merge key-by-key, so an
# interleaved-only run keeps the throughput fields an earlier run wrote.
$merged = @{}
if (Test-Path $sum) {
    $old = Get-Content $sum -Raw | ConvertFrom-Json
    foreach ($r in $old.rows) {
        $h = [ordered]@{}
        foreach ($p in $r.PSObject.Properties) { $h[$p.Name] = $p.Value }
        $merged[[string]$r.model] = $h
    }
}
foreach ($r in $summary) {
    $name = [string]$r.model
    if ($merged.ContainsKey($name)) {
        foreach ($k in $r.Keys) { $merged[$name][$k] = $r[$k] }
    } else { $merged[$name] = $r }
}
# Catalogue order, not hashtable order, so the file reads like the table it is.
$rowsOut = @()
foreach ($m in $CATALOG) { if ($merged.ContainsKey($m.name)) { $rowsOut += $merged[$m.name] } }
foreach ($k in $merged.Keys) {
    if (-not ($CATALOG | Where-Object { $_.name -eq $k })) { $rowsOut += $merged[$k] }
}

$json = @{
    kind = "hardware measurement"
    what = "whole-catalogue release benchmark sweep, one session"
    when = $stamp
    power = $power
    lanes = $Lanes; threads = $Threads; bench = $Bench; rounds = $Rounds
    # What the machine was doing while this was measured. A ratio taken on a
    # busy machine is not merely noisy, it is biased -- so the artifact records
    # the condition rather than leaving a reader to assume it was idle.
    cpu_mean_percent = [math]::Round($cpu.MeanPercent, 1)
    cpu_quiet = $cpu.Quiet
    cpu_contenders = @($cpu.Busy | ForEach-Object { "$($_.Name) ~$($_.Cores) core(s)" })
    stages_this_run = @("accuracy", "throughput", "interleaved", "energy", "mteb" |
                        Where-Object { -not ($Skip -contains $_) })
    rows = $rowsOut
} | ConvertTo-Json -Depth 6
# WITHOUT a BOM. `Set-Content -Encoding utf8` on Windows PowerShell 5.1 writes
# one, and Python's json.load() rejects it outright -- so the artifact this
# sweep exists to produce could not be read by half the tooling in the repo.
[System.IO.File]::WriteAllText($sum, $json,
    (New-Object System.Text.UTF8Encoding($false)))

Write-Host ""
Write-Host "================ SWEEP DONE ================" -ForegroundColor Green
Write-Host "  logs and per-model artifacts: $OutDir"
Write-Host "  index: $sum"
Write-Host ""
Write-Host "  Every figure above is end-to-end throughput or a host-side cost." -ForegroundColor Yellow
Write-Host "  NONE of it is an NPU kernel performance claim -- those come from" -ForegroundColor Yellow
Write-Host "  hardware traces or static instruction counts (CLAUDE.md rule 1)." -ForegroundColor Yellow
