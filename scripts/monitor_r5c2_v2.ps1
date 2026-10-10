param([string]$Root='results/revision_round2/r2_r05c02_dqi_sensitivity_v2',[switch]$Once)
$TaskRoot=Split-Path -Parent $PSScriptRoot
$TaskBundle=if ([IO.Path]::IsPathRooted($Root)) { $Root } else { Join-Path $TaskRoot $Root }
do {
    Write-Host (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    $TaskModelFile=Join-Path $TaskBundle 'model_index.json'
    $TaskRunFile=Join-Path $TaskBundle 'run_index.json'
    $TaskFrozenFile=Join-Path $TaskBundle 'runs_frozen.json'
    if (Test-Path -LiteralPath $TaskModelFile) {
        $TaskModels=Get-Content -LiteralPath $TaskModelFile -Raw | ConvertFrom-Json
        Write-Host "Predictors: $(@($TaskModels.PSObject.Properties).Count)/11"
    }
    if (Test-Path -LiteralPath $TaskRunFile) {
        $TaskRuns=Get-Content -LiteralPath $TaskRunFile -Raw | ConvertFrom-Json
        $TaskRuns.PSObject.Properties.Value | Group-Object status | Select-Object Name,Count | Format-Table -AutoSize
        $TaskRuns.PSObject.Properties.Value | Where-Object status -eq 'running' | Select-Object pid,log | Format-Table -AutoSize
    }
    if (Test-Path -LiteralPath $TaskFrozenFile) {
        $TaskFrozen=Get-Content -LiteralPath $TaskFrozenFile -Raw | ConvertFrom-Json
        Write-Host "Expected RL tasks: $($TaskFrozen.tasks.Count)"
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader }
    if (-not $Once) { Start-Sleep -Seconds 5 }
} while (-not $Once)
