# NpuEmbeddings -- energy measurement on native Windows, without external
# instrumentation (tasks/0034).
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS EXISTS
# ---------------
# docs/CURRENT_STATUS said "Windows Power Meter counters have NO INSTANCES on
# this machine; needs external instrumentation", and that closed the door on
# the project's own "energy is the point" claim. It was the wrong counter set.
#
#   \Power Meter(*)   -- 2 paths, no real instances     <- what was checked
#   \Energy Meter(*)  -- 14 instances INCLUDING         <- what actually works
#                        RAPL_Package0_PKG and 12 per-core meters
#
# UNITS, calibrated rather than assumed (see 0034):
#   Time    milliseconds  (verified against system uptime, 257,251,558 ms)
#   Power   milliwatts
#   Energy  PICOWATT-HOURS = 3.6e-9 J  (the Windows EMI unit)
# Cross-check: dEnergy/dTime / Power = 277.8 +/- 0.1 over every sample, which
# is exactly 1/(3.6e-9 * 1000 * 1000). Two independent counter paths agreeing
# to 0.1% is the instrument-level version of this project's two-signal rule.
#
# WHY CUMULATIVE ENERGY IS THE RIGHT SIGNAL
# -----------------------------------------
# The Energy counter is monotonic and cumulative, so total energy over a window
# is exactly (E_after - E_before) -- no sampling rate, no missed transients, no
# integration error. Sampling is done anyway, but only to record the power
# trace and to prove the idle baseline was stable.
#
# WHAT IT CANNOT TELL YOU
# -----------------------
# Whether RAPL_Package0_PKG includes the NPU block. That is a QUESTION, not an
# assumption -- answer it with the control experiment in 0034 (a pure-NPU soak
# with no host work: if package power does not move, the meter does not see the
# NPU and every NPU number here is a lower bound).
#
# Usage:
#   .\tools\measure_energy.ps1 -Label cpu-baseline -Command '...' -Idle 15
#   .\tools\measure_energy.ps1 -Label npu-soak -Command '...' -Out out.json

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$WorkDir = (Get-Location).Path,
    [int]$Idle = 15,             # seconds of idle baseline, before AND after
    [int]$Repeats = 1,
    [string]$Out = $null,
    [switch]$NoIdle
)

$METER = "RAPL_Package0_PKG"
$PJ_PER_UNIT = 3.6e-9            # picowatt-hour -> joule

function Read-Meter {
    $s = Get-Counter -Counter "\Energy Meter($METER)\Energy",
                              "\Energy Meter($METER)\Time" -MaxSamples 1
    $h = @{}
    foreach ($c in $s.CounterSamples) { $h[($c.Path -split '\\')[-1]] = $c.CookedValue }
    [pscustomobject]@{
        Energy = $h['energy']
        Time   = $h['time']
        Wall   = [double](Get-Date -UFormat %s)
    }
}

function Measure-Window {
    param([scriptblock]$Body)
    $a = Read-Meter
    $r = & $Body
    $b = Read-Meter
    $dE = ($b.Energy - $a.Energy) * $PJ_PER_UNIT       # joules
    $dT = ($b.Time - $a.Time) / 1000.0                 # seconds (meter clock)
    [pscustomobject]@{
        Joules      = $dE
        Seconds     = $dT
        WallSeconds = $b.Wall - $a.Wall
        Watts       = if ($dT -gt 0) { $dE / $dT } else { [double]::NaN }
        Output      = $r
    }
}

function Measure-Idle {
    param([int]$Seconds)
    if ($NoIdle -or $Seconds -le 0) { return $null }
    Measure-Window { Start-Sleep -Seconds $Seconds } | Select-Object Joules, Seconds, Watts
}

Push-Location $WorkDir
try {
    Write-Host "== $Label ==" -ForegroundColor Cyan
    Write-Host "   meter $METER, energy in picowatt-hours (3.6e-9 J)"

    $idleBefore = Measure-Idle -Seconds $Idle
    if ($idleBefore) {
        Write-Host ("   idle before : {0,7:N2} W  ({1:N1} J over {2:N1} s)" -f `
            $idleBefore.Watts, $idleBefore.Joules, $idleBefore.Seconds)
    }

    $runs = @()
    for ($i = 1; $i -le $Repeats; $i++) {
        # `cmd /c $Command 2>&1` lets POWERSHELL join the streams, and on
        # Windows PowerShell 5.1 that wraps every stderr line of a native
        # command in an ErrorRecord -- fatal under $ErrorActionPreference =
        # "Stop", for a program that merely wrote a progress bar to stderr and
        # exited 0. That is exactly what killed the first whole-catalogue
        # energy run (tasks/0073): sentence-transformers prints its weight
        # loading bar to stderr. Move the redirection INSIDE the cmd string so
        # the OS joins them and PowerShell only ever sees stdout.
        $m = Measure-Window { cmd /c "$Command 2>&1" | Out-String }
        $runs += $m
        Write-Host ("   run {0}       : {1,7:N2} W  ({2,8:N1} J over {3,6:N2} s)" -f `
            $i, $m.Watts, $m.Joules, $m.Seconds)
    }

    $idleAfter = Measure-Idle -Seconds $Idle
    if ($idleAfter) {
        Write-Host ("   idle after  : {0,7:N2} W  ({1:N1} J over {2:N1} s)" -f `
            $idleAfter.Watts, $idleAfter.Joules, $idleAfter.Seconds)
    }

    # The idle baseline must be stable, or the marginal figure is meaningless.
    # A drifting baseline is a discarded measurement, not a footnote.
    $idleW = $null
    $idleDrift = $null
    if ($idleBefore -and $idleAfter) {
        $idleW = ($idleBefore.Watts + $idleAfter.Watts) / 2.0
        $idleDrift = [math]::Abs($idleAfter.Watts - $idleBefore.Watts) / $idleW
        $verdict = if ($idleDrift -le 0.15) { "OK" } else { "UNSTABLE -- DISCARD" }
        Write-Host ("   idle drift  : {0,6:P1}  {1}" -f $idleDrift, $verdict) `
            -ForegroundColor $(if ($idleDrift -le 0.15) { "Green" } else { "Red" })
    }

    $best = $runs | Sort-Object Joules | Select-Object -First 1
    $meanW = ($runs | Measure-Object -Property Watts -Average).Average
    $meanJ = ($runs | Measure-Object -Property Joules -Average).Average
    $marginalJ = if ($idleW) { $meanJ - $idleW * ($runs | Measure-Object -Property Seconds -Average).Average } else { $null }

    Write-Host ("   MEAN        : {0,7:N2} W  {1,8:N1} J" -f $meanW, $meanJ) -ForegroundColor Yellow
    if ($null -ne $marginalJ) {
        Write-Host ("   MARGINAL    : {0,8:N1} J above idle" -f $marginalJ) -ForegroundColor Yellow
    }

    $result = [ordered]@{
        kind          = "hardware measurement"
        label         = $Label
        command       = $Command
        meter         = $METER
        energy_unit_j = $PJ_PER_UNIT
        repeats       = $Repeats
        idle_before_w = if ($idleBefore) { $idleBefore.Watts } else { $null }
        idle_after_w  = if ($idleAfter) { $idleAfter.Watts } else { $null }
        idle_w        = $idleW
        idle_drift    = $idleDrift
        runs          = @($runs | ForEach-Object {
                [ordered]@{ joules = $_.Joules; seconds = $_.Seconds; watts = $_.Watts }
            })
        mean_watts    = $meanW
        mean_joules   = $meanJ
        best_joules   = $best.Joules
        marginal_j    = $marginalJ
    }
    if ($Out) {
        $result | ConvertTo-Json -Depth 6 | Set-Content -Path $Out -Encoding utf8
        Write-Host "   wrote $Out"
    }
    $result
}
finally { Pop-Location }
