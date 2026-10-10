param(
    [string]$BundlePath = 'results/revision_round2/r2_r05c02_dqi_sensitivity_v1',
    [ValidateRange(2,300)][int]$IntervalSeconds = 10,
    [switch]$Once,
    [switch]$IncludeGpu
)

$ErrorActionPreference = 'Stop'
$R5C2Project = Split-Path -Parent $PSScriptRoot
$R5C2Bundle = if ([IO.Path]::IsPathRooted($BundlePath)) { $BundlePath } else { Join-Path $R5C2Project $BundlePath }

function Read-R5C2Json {
    param([string]$Name)
    $R5C2Path = Join-Path $R5C2Bundle $Name
    if (Test-Path -LiteralPath $R5C2Path) {
        return Get-Content -LiteralPath $R5C2Path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    return $null
}

do {
    if (-not $Once) { Clear-Host }
    Write-Host ('R5-C2 monitor: {0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
    Write-Host $R5C2Bundle
    $R5C2Data = Read-R5C2Json 'data/prepared_manifest.json'
    $R5C2Protocol = Read-R5C2Json 'protocol_frozen.json'
    $R5C2Models = Read-R5C2Json 'model_index.json'
    $R5C2Runs = Read-R5C2Json 'run_index.json'
    $R5C2Freeze = Read-R5C2Json 'runs_frozen.json'
    $R5C2PilotModels = Read-R5C2Json 'pilot/model_index.json'
    $R5C2PilotRuns = Read-R5C2Json 'pilot/run_index.json'
    $R5C2PilotModelRows = if ($R5C2PilotModels) { @($R5C2PilotModels.PSObject.Properties | ForEach-Object { $_.Value }) } else { @() }
    $R5C2PilotRunRows = if ($R5C2PilotRuns) { @($R5C2PilotRuns.PSObject.Properties | ForEach-Object { $_.Value }) } else { @() }
    $R5C2ModelRows = if ($R5C2Models) { @($R5C2Models.PSObject.Properties | ForEach-Object { $_.Value }) } else { @() }
    $R5C2RunRows = if ($R5C2Runs) { @($R5C2Runs.PSObject.Properties | ForEach-Object { $_.Value }) } else { @() }
    $R5C2Total = if ($R5C2Freeze) { @($R5C2Freeze.tasks).Count } else { 75 }
    $R5C2Completed = @($R5C2RunRows | Where-Object status -eq 'completed').Count
    $R5C2Failed = @($R5C2RunRows | Where-Object status -eq 'failed')
    $R5C2Running = @($R5C2RunRows | Where-Object status -eq 'running')
    Write-Host ('Data ready: {0}; protocol frozen: {1}; DQI models: {2}/11' -f [bool]$R5C2Data,[bool]$R5C2Protocol,$R5C2ModelRows.Count)
    Write-Host ('RL completed: {0}/{1}; running: {2}; failed: {3}' -f $R5C2Completed,$R5C2Total,$R5C2Running.Count,$R5C2Failed.Count)
    Write-Host ('Pilot models: {0}; pilot RL completed: {1}; pilot RL running: {2}' -f $R5C2PilotModelRows.Count,@($R5C2PilotRunRows | Where-Object status -eq 'completed').Count,@($R5C2PilotRunRows | Where-Object status -eq 'running').Count)
    if ($R5C2Protocol) { Write-Host ('DQI update: {0} slots' -f $R5C2Protocol.selected_update_slots) }
    if ($R5C2Running.Count) {
        $R5C2Running | Select-Object condition_id,seed,pid | Format-Table -AutoSize
    }
    if (@($R5C2PilotRunRows | Where-Object status -eq 'running').Count) {
        $R5C2PilotRunRows | Where-Object status -eq 'running' | Select-Object condition_id,seed,pid | Format-Table -AutoSize
    }
    if ($R5C2Failed.Count) {
        $R5C2Failed | Select-Object condition_id,seed,error | Format-Table -Wrap
    }
    $R5C2LogCandidates = @()
    foreach ($R5C2Subdir in @('models','logs','pilot/models','pilot/logs','terminal_logs')) {
        $R5C2Dir = Join-Path $R5C2Bundle $R5C2Subdir
        if (Test-Path -LiteralPath $R5C2Dir) {
            $R5C2LogCandidates += @(Get-ChildItem -LiteralPath $R5C2Dir -Filter '*.log' -File -Recurse)
        }
    }
    $R5C2Latest = $R5C2LogCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($R5C2Latest) {
        Write-Host ('Latest log: {0}' -f $R5C2Latest.FullName)
        Get-Content -LiteralPath $R5C2Latest.FullName -Tail 8 -Encoding UTF8
    }
    if ($IncludeGpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        & nvidia-smi '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu' '--format=csv,noheader'
    }
    if (-not $Once) { Start-Sleep -Seconds $IntervalSeconds }
} while (-not $Once)
