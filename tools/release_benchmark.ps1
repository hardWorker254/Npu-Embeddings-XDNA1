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
    [string[]]$Skip = @(),          # accuracy | throughput | interleaved | energy | mteb | tail
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
# `npu = $false` would mean there is no NPU kernel for the architecture at all.
# No row is in that state any more: tasks/0074 put arch=1 on the array.
#
# `harness` is a SECOND axis and must not be folded into `npu`. It says which
# COMMAND SHAPE drives this model: the five BERT-family rows have `--bench` and
# a golden fixture; arch=1 has neither, so it measures by encoding a real
# corpus and gates differentially against its own host-only path. Everything
# else -- the interleaved CPU ratio, energy, MTEB -- now runs for every row
# (0075 taught npu_encoder.py, compare_three.py and energy_compare.ps1 about
# arch=1), so there are no UNMEASURED arms left here. If one comes back, it
# says so in the log and writes $null in sweep.json rather than omitting the
# row, and it never borrows another model's number.
#
# THE int8 ROWS ARE A SECOND DATAPATH, NOT A SECOND MODEL (tasks/0077-0082).
# They are listed explicitly rather than derived from the bf16 rows, because
# the pairing is what the measurement is about: an int8 container needs an int8
# design set, and the two carry different `b_layout_hash` values so a wrong
# pairing refuses rather than reading garbage. bge-large is the only model
# whose int8 design uses tile_n = 64 -- legal only at int8's 1-byte operands
# (tasks/0081) -- so its artifacts directory differs from the pattern.
#
# `gate` records what this row's accuracy verdict IS, so a table generated from
# sweep.json cannot quietly print a number for a row that fails: bge-large int8
# misses the 2e-03 1-cos gate at 2.968e-03 and its verdict rests on MTEB.
#
# THE bfp16 ADOPTION (tasks/0104, T23, 2026-08-24). Five of six bf16-family
# models moved to `--emulate-bfp16 --c-bf16` -- own artifact directories,
# `*_bfp16` -- after clearing the per-model MTEB gate; `bge-small-en-v1.5`
# FAILED it (-0.5010 against the -0.5 line) and stays on plain bf16, now in
# its own directory too (`artifacts_small_bf16`, not the old `artifacts_b128il`
# -- same design, same numbers, but it STATES its datapath where b128il
# predates the field and reads UNRECORDED). `datapath` here is each row's
# INTENDED assignment per 0104's adoption table; it is a comment, not the
# truth -- the truth is `datapath_reported`, scraped below from the runtime's
# own status line (`datapath   ...`), which is what 0104's whole guard exists
# to make authoritative. int8 is a THIRD, separate datapath (native int8
# MMAC, unaffected by the bfp16 decision) -- labelled `int8-native` rather
# than probed, since the bf16-vs-bfp16-emulated status line does not describe
# it either way.
$CATALOG = @(
    @{ name = "all-MiniLM-L6-v2";      artifacts = "artifacts_minilm_bfp16"; npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bfp16"; gate = "pass" }
    @{ name = "bge-small-en-v1.5";     artifacts = "artifacts_small_bf16";   npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bf16";  gate = "pass" }
    @{ name = "bge-base-en-v1.5";      artifacts = "artifacts_base_bfp16";   npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bfp16"; gate = "pass" }
    @{ name = "bge-large-en-v1.5";     artifacts = "artifacts_large_bfp16";  npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bfp16"; gate = "pass" }
    @{ name = "nomic-embed-text-v1.5"; artifacts = "artifacts_nomic_bfp16";  npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bfp16"; gate = "pass" }
    @{ name = "embeddinggemma-300m";   artifacts = "artifacts_gemma_bfp16";  npu = $true; harness = "gemma"; dtype = "bf16"; datapath = "bfp16"; gate = "pass" }
    # gte (arch=3, 0.5.0): interleaved/energy are PER-ROW opt-outs, not
    # oversights. Both stages need a CPU reference arm, and gte's
    # trust_remote_code model is unusable without the 0134/0136 buffer
    # repairs (compare_three.py and the energy CPU arm would either crash on
    # the derived-position path or measure a silently position-scrambled
    # model). Per docs/05-measurement the NPU figure is quoted alone and
    # labelled; the quality delta vs fp32 is 0137's symmetric MTEB run.
    @{ name = "gte-multilingual-base"; artifacts = "artifacts_nomic_bfp16"; npu = $true; harness = "bert"; dtype = "bf16"; datapath = "bfp16"; gate = "pass"; interleaved = $false; energy = $false }

    @{ name = "all-MiniLM-L6-v2.int8";      artifacts = "artifacts_int8c_mini"; cpu = "all-MiniLM-L6-v2";     npu = $true; harness = "bert";  dtype = "int8"; datapath = "int8-native"; gate = "pass" }
    @{ name = "bge-small-en-v1.5.int8";     artifacts = "artifacts_int8c_mini"; cpu = "bge-small-en-v1.5";     npu = $true; harness = "bert";  dtype = "int8"; datapath = "int8-native"; gate = "pass" }
    @{ name = "bge-base-en-v1.5.int8";      artifacts = "artifacts_int8c_base"; cpu = "bge-base-en-v1.5";     npu = $true; harness = "bert";  dtype = "int8"; datapath = "int8-native"; gate = "pass" }
    @{ name = "bge-large-en-v1.5.int8n64";  artifacts = "artifacts_int8c_large_n64"; cpu = "bge-large-en-v1.5"; npu = $true; harness = "bert";  dtype = "int8"; datapath = "int8-native"; gate = "1-cos FAIL 2.968e-03; verdict rests on MTEB" }
    @{ name = "nomic-embed-text-v1.5.int8"; artifacts = "artifacts_int8c_nomic"; cpu = "nomic-embed-text-v1.5";    npu = $true; harness = "bert";  dtype = "int8"; datapath = "int8-native"; gate = "pass" }
    @{ name = "embeddinggemma-300m.int8";   artifacts = "artifacts_int8c_gemma"; cpu = "embeddinggemma-300m";    npu = $true; harness = "gemma"; dtype = "int8"; datapath = "int8-native"; gate = "pass" }
)

# The corpus the arch=1 throughput measurement encodes. Distinct sentences,
# repeated to fill the batch tiers -- see tasks/0074 for why a tiled fixture is
# not acceptable as an ACCURACY corpus (thread T32) even though it is fine for
# throughput.
$GEMMA_CORPUS = Join-Path $REPO "tasks\0074-m13-gemma-on-npu\corpus_520.txt"

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

# THE TRUTH, NOT THE INTENTION (tasks/0105, following 0104's own discipline).
# `$m.datapath` above is what the catalogue INTENDS to run; the runtime prints
# what it actually loaded on every startup, as `  datapath   <text>`, read off
# the loaded Design's own `emulate_bfp16`/`datapath_recorded`, never off a flag.
# Scrape that line out of the first log a row produces so a mismatch (wrong
# artifacts dir, a stale build, pick_artifacts() choosing something else) shows
# up in sweep.json rather than only in a log nobody re-reads.
function Get-DatapathLine([string]$logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $line = Get-Content $logPath | Where-Object { $_ -match '^\s*datapath\s+\S' } | Select-Object -First 1
    if ($line) { return ($line -replace '^\s*datapath\s+', '').Trim() }
    return $null
}

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
    # `dtype` and `gate` ride into sweep.json so a generated table cannot print
    # a throughput number without the verdict that qualifies it (tasks/0081:
    # bge-large int8 is the fastest bge-large we have and it fails 1-cos).
    # datapath_reported is DELIBERATELY OMITTED here rather than pre-set to
    # $null. The merge below is per-KEY ("foreach $k in $r.Keys"), so a row
    # that includes the key -- even as $null -- overwrites an earlier stage's
    # real reading with nothing. This bit a real run (tasks/0105): the
    # interleaved stage doesn't scrape the status line, but its row still
    # carried `datapath_reported = $null` and wiped out what the accuracy
    # stage had just found for every model. Only set the key when a stage
    # actually extracts a value (below); an unset key cannot clobber.
    $row = [ordered]@{ model = $name; artifacts = $art; npu = $m.npu
                       dtype = $m.dtype; datapath = $m.datapath
                       gate = $m.gate }

    if (-not (Test-Path (Join-Path $REPO "models\$name.npue"))) {
        Write-Host "  not installed -- skipping" -ForegroundColor DarkYellow
        $row.skipped = "container absent"
        $summary += $row; continue
    }

    # --- accuracy: reproduce the recorded 1-cos against the goldens -----------
    if (Want "accuracy") {
        Write-Host "-- accuracy (golden check)" -ForegroundColor Cyan
        $log = Join-Path $OUT "accuracy-$name.txt"
        if ($m.harness -eq "gemma") {
            # arch=1 has no golden fixture. Its gate is DIFFERENTIAL: the NPU
            # encode against the host-only path, which tasks/0064-0065 tied to
            # reference/encoder_gemma.py at 1-cos 5.496e-13. The control
            # vectors are stored (tasks/0074) rather than re-derived, because
            # regenerating them costs 65 s for 13 texts.
            # THE PREFIX IS PART OF THE CONTROL, so it is pinned rather than
            # left to a default (tasks/0118). out_cpu.f32 was produced under
            # the old silent `--prefix document`, and comparing an NPU encode
            # under any OTHER prompt against it would measure the prompt and
            # report it as a datapath difference. Since 0118 removed that
            # default the flag is also REQUIRED -- without it this stage
            # refuses, which is how this coupling was found.
            $GEMMA_PREFIX = "document"
            $ctl = Join-Path $REPO "tasks\0074-m13-gemma-on-npu\out_cpu.f32"
            $corp = Join-Path $REPO "tasks\0074-m13-gemma-on-npu\corpus.txt"
            $tmp = Join-Path $OUT "accuracy-$name.f32"
            $null = Invoke-Logged -Exe $EXE -Log $log -Arguments @($REPO, "--model", $name, "--artifacts", $art, "--embed", $corp, $tmp, "--threads", "$Threads", "--pipeline", "$Lanes", "--prefix", $GEMMA_PREFIX)
            $code = Invoke-Logged -Exe $PY -Log (Join-Path $OUT "accuracy-$name.gate.txt") -Arguments @(
                (Join-Path $REPO "tools\verify_gemma_npu_encode.py"),
                "--npu", $tmp, "--cpu", $ctl)
            $row.accuracy_pass = ($code -eq 0)
        } else {
            $null = Invoke-Logged -Exe $EXE -Log $log -Arguments @($REPO, "--model", $name, "--artifacts", $art, "--threads", "16")
        }
        $row.accuracy_log = $log
        if (-not $row.datapath_reported) {
            $dp = Get-DatapathLine $log
            if ($dp) { $row.datapath_reported = $dp }
        }
    }

    # --- throughput: guarded, three runs, spread reported --------------------
    if ((Want "throughput") -and $m.npu) {
        Write-Host "-- throughput ($Lanes lanes, contention guard ON)" -ForegroundColor Cyan
        $log = Join-Path $OUT "throughput-$name.txt"
        "" | Set-Content $log
        for ($i = 1; $i -le 3; $i++) {
            $one = Join-Path $OUT "throughput-$name.run$i.txt"
            # arch=1 has no --bench mode; it measures by encoding a real
            # corpus. --guard-contention gives it the SAME refusal --bench
            # applies (tasks/0044): a foreign Active hw_context makes the
            # number confidently wrong, and an absent xrt-smi is not a
            # negative reading.
            $tArgs = if ($m.harness -eq "gemma") {
                @($REPO, "--model", $name, "--artifacts", $art, "--embed", $GEMMA_CORPUS,
                  "--threads", "$Threads", "--pipeline", "$Lanes", "--guard-contention",
                  "--prefix", "document")
            } else {
                @($REPO, "--model", $name, "--artifacts", $art, "--threads", "$Threads", "--pipeline", "$Lanes", "--bench", "$Bench")
            }
            $code = Invoke-Logged -Exe $EXE -Log $one -Arguments $tArgs
            Get-Content $one | Add-Content $log
            if ($code -ne 0) {
                # DIAGNOSE, do not assume. This branch used to report every
                # non-zero exit as a contention refusal, and the first time it
                # fired it was a missing design set -- so the log said "a
                # foreign hw_context is Active" about a machine whose array was
                # idle. An error message that names the wrong cause is worse
                # than one that says it does not know.
                $txt = if (Test-Path $one) { (Get-Content $one -Raw) } else { "" }
                if ($txt -match "contended|foreign|xrt-smi") {
                    Write-Host "  REFUSED (exit $code) -- contention guard. NOT retrying with --allow-contention." -ForegroundColor Red
                    $row.throughput_error = "contention guard refused (exit $code)"
                } else {
                    $first = ($txt -split "`n" | Where-Object { $_ -match "^error:" } | Select-Object -First 1)
                    Write-Host "  FAILED (exit $code) -- $first" -ForegroundColor Red
                    $row.throughput_error = "exit $code -- $first"
                }
                break
            }
        }
        $row.throughput_log = $log
        if (-not $row.datapath_reported) {
            $dp = Get-DatapathLine $log
            if ($dp) { $row.datapath_reported = $dp }
        }
    }

    # --- interleaved NPU vs torch vs ORT, one session, same statistic --------
    if ((Want "interleaved") -and $m.npu -and ($m.interleaved -ne $false)) {
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
    if ((Want "energy") -and $m.npu -and ($m.energy -ne $false)) {
        Write-Host "-- energy (RAPL, differential)" -ForegroundColor Cyan
        $log = Join-Path $OUT "energy-$name.txt"
        # -CpuModel for the same reason MTEB needs --cpu-model: `$name` is a
        # CONTAINER name, and the CPU arm loads a HuggingFace checkpoint
        # DIRECTORY, which an int8 container does not have. Without it every
        # int8 row's CPU run died instantly and the differential came out
        # between -0.8 and 1.2 J/1000 seq -- the wrong sign, reported without
        # complaint, because the method cannot tell "used little energy" from
        # "died in 0.2 s" (tasks/0085). energy_compare.ps1 now refuses instead.
        # SPLAT A HASHTABLE, NOT AN ARRAY. `& $script @arr` splats an array
        # POSITIONALLY, so an array of "-Model", $name, ... binds "-Model" to
        # the first positional parameter -- here `-Low`, an [int] -- and fails
        # with `Cannot convert value "-Model" to type "System.Int32"`. Named
        # splatting needs a hashtable. That is the second splatting mistake in
        # this file today; the first was `$(if ...)` inside `@()` not splatting
        # at all (see the MTEB block below).
        $energyArgs = @{ Model = $name; Artifacts = $art
                         Threads = $Threads; Lanes = $Lanes
                         Low = $EnergyLow; High = $EnergyHigh
                         OutDir = "$OutDir\energy-$name" }
        if ($m.cpu) { $energyArgs.CpuModel = $m.cpu }
        & (Join-Path $PSScriptRoot "energy_compare.ps1") @energyArgs 2>&1 |
            Tee-Object $log
        $row.energy_log = $log
    }

    # --- MTEB, the accuracy gate that is about QUALITY not fidelity ----------
    if (Want "mteb") {
        Write-Host "-- MTEB (cpu + npu, one session)" -ForegroundColor Cyan
        $log = Join-Path $OUT "mteb-$name.txt"
        if ($m.npu) {
            # An int8 container has no HuggingFace checkpoint directory of its
            # own -- its CPU baseline is the bf16 model it was quantised FROM,
            # which is the comparison the gate is about. Without it the CPU
            # side looks for models/<name>.int8/ and fails (tasks/0082).
            #
            # BUILD THIS WITH +=, NOT A NESTED $(...) INSIDE @() (tasks/0085).
            # `@("a", $(if ($c) { "--x"; $c }), "b")` does NOT splat the two
            # values into two elements: the subexpression stays ONE element
            # holding an array, Invoke-Logged stringifies it to
            # "--cpu-model all-MiniLM-L6-v2", sees the space, and quotes the
            # whole thing into a single argv token. run_mteb.py then reports
            #     unrecognized arguments: --cpu-model all-MiniLM-L6-v2
            # which reads like a missing flag and is really a quoting bug. It
            # cost this sweep all six int8 MTEB rows, two hours in.
            $mtebArgs = @((Join-Path $REPO "experiments\m8-npu-vs-cpu\run_mteb.py"),
                          "--model", $name, "--artifacts", $art,
                          "--threads", "$Threads", "--pipeline", "$Lanes")
            if ($m.cpu) { $mtebArgs += "--cpu-model"; $mtebArgs += $m.cpu }
            $mtebArgs += "--out"; $mtebArgs += (Join-Path $OUT "mteb-$name.json")
            $code = Invoke-Logged -Exe $PY -Log $log -Arguments $mtebArgs
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

# --- tail: p99 accuracy tail against stored fp32 references (T51) -----------
# ONE invocation for the whole catalogue, not a per-model block: verify_tail.py
# sweeps every model that has a reference under reference\tail\ and gates each
# on its own recorded ceiling (baseline.p99_ceiling in the reference JSON).
# This is the instrument every other stage lacks: accuracy/MTEB/semantic all
# report central tendencies, and bge-large passed all three while carrying a
# measured 100x max/median tail on single-word inputs (tasks/0122, 0129,
# research/OPEN-THREADS.md#t51). The gate is stdlib-only, so $PY here is
# convenience, not a dependency.
$tailPass = $null
$tailLog = $null
if (Want "tail") {
    Write-Host ""
    Write-Host "-- tail gate (p99 per model per datapath, T51)" -ForegroundColor Cyan
    $tailLog = Join-Path $OUT "tail.txt"
    $code = Invoke-Logged -Exe $PY -Log $tailLog -Arguments @(
        (Join-Path $REPO "tools\verify_tail.py"),
        "--threads", "$Threads",
        "--out", (Join-Path $OUT "tail_gate.json"))
    $tailPass = ($code -eq 0)
    if (-not $tailPass) {
        Write-Host "  TAIL GATE FAILED -- a model's p99 exceeds its recorded ceiling (see $tailLog). NOT re-baselining; that is a deliberate act (verify_tail.py --write-baseline)." -ForegroundColor Red
    }
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
    stages_this_run = @("accuracy", "throughput", "interleaved", "energy", "mteb", "tail" |
                        Where-Object { -not ($Skip -contains $_) })
    rows = $rowsOut
}
# Only set the tail keys when the stage actually ran -- an unset key cannot
# misreport a skipped stage as a verdict (the same rule the per-model rows
# follow for datapath_reported).
if ($null -ne $tailPass) {
    $json.tail_pass = $tailPass
    $json.tail_log = $tailLog
}
$json = $json | ConvertTo-Json -Depth 6
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
