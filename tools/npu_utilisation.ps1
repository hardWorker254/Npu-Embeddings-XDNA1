# NpuEmbeddings -- sample NPU busy-percentage for one process, from Windows.
#
# Windows exposes NO NPU-specific counter set and no NPU memory counter, but
# the NPU is an MCDM device and MCDM shares the GPU counter infrastructure.
# So NPU busy time arrives as \GPU Engine(...engtype_compute)\Utilization
# Percentage, scoped to a pid. On this machine that instance carries luid
# 0x000170c6, which matches the NPU's BDF [00c6:00:01.1] from xrt-smi; the
# only Win32_VideoController present is the Radeon 890M and our runtime issues
# no graphics work, so nothing else can be producing it.
#
# Validated against our own instrumentation: 51.1% here against 50.7% from the
# runtime's own "NPU dispatch+wait (serialized)" line, same run.
#
# THIS IS NOT AN NPU PERFORMANCE CLAIM (CLAUDE.md rule 1). It is a busy
# fraction of wall time, useful for "is the array actually being used" and for
# cross-checking our own accounting -- not for kernel quality.
#
# Get-Counter samples at roughly 1 Hz and several calls return nothing, so a
# short run can yield ZERO samples and read as 0%. That is a fail-open answer.
# The script reports the sample count and refuses to print a percentage when
# it has too few.
#
#   .\tools\npu_utilisation.ps1 -Exe runtime\build\npuembed.exe `
#       -Arguments ".. --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 40" `
#       -WorkingDirectory runtime

param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [Parameter(Mandatory = $true)][string]$Arguments,
  [string]$WorkingDirectory = ".",
  [int]$MinSamples = 3
)

$ErrorActionPreference = "Stop"
$log = Join-Path $env:TEMP "npu_utilisation_$PID.txt"

Push-Location $WorkingDirectory
try {
  $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -PassThru -NoNewWindow `
                     -RedirectStandardOutput $log
  $util = @(); $mem = 0.0; $ws = 0

  # Filter on the instance name IN THE COUNTER PATH, not afterwards. Asking
  # for \GPU Engine(*) enumerates every engine instance on the machine and
  # costs seconds per call -- a 13 s run yielded ONE sample. Scoped to a pid it
  # is fast enough to sample properly.
  $engPath = "\GPU Engine(pid_$($p.Id)*)\Utilization Percentage"
  $memPath = "\GPU Process Memory(pid_$($p.Id)*)\Local Usage"
  $i = 0
  while (-not $p.HasExited) {
    try {
      (Get-Counter $engPath -EA Stop).CounterSamples |
        ForEach-Object { $util += $_.CookedValue }
      if (($i++ % 4) -eq 0) {
        (Get-Counter $memPath -EA SilentlyContinue).CounterSamples |
          ForEach-Object { if ($_.CookedValue -gt $mem) { $mem = $_.CookedValue } }
        $q = Get-CimInstance Win32_PerfRawData_PerfProc_Process -Filter "IDProcess=$($p.Id)" -EA SilentlyContinue
        if ($q -and $q.WorkingSetPeak -gt $ws) { $ws = $q.WorkingSetPeak }
      }
    } catch {}
  }

  Write-Output ""
  Write-Output "NPU engine utilisation (MCDM compute engine, this pid only)"
  Write-Output ("  samples          : {0}" -f $util.Count)
  if ($util.Count -lt $MinSamples) {
    Write-Output "  NOT ENOUGH SAMPLES -- refusing to report a percentage."
    Write-Output "  Get-Counter runs at about 1 Hz; use a longer --bench."
  } else {
    $nz = $util | Where-Object { $_ -gt 0 }
    Write-Output ("  peak             : {0,6:N1}%" -f ($util | Measure-Object -Maximum).Maximum)
    Write-Output ("  mean of non-zero : {0,6:N1}%   ({1} of {2} samples)" -f `
                  (($nz | Measure-Object -Average).Average), $nz.Count, $util.Count)
  }
  Write-Output ""
  Write-Output "Memory -- all of it host DRAM; XDNA2 has no device-local pool"
  Write-Output ("  XRT buffers (GPU Process Memory local) : {0,7:N1} MB" -f ($mem / 1MB))
  Write-Output ("  process peak working set               : {0,7:N1} MB" -f ($ws / 1MB))
  Write-Output ""
  Get-Content $log | Select-String "NPU dispatch\+wait|seq/s" | Select-Object -Last 2
}
finally {
  Pop-Location
  Remove-Item $log -EA SilentlyContinue
}
