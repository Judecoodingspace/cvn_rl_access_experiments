param([string]$Config='configs/r2_r05c02_dqi_sensitivity_v2.json')
$ErrorActionPreference='Stop'
$TaskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $TaskRoot
$TaskPython=Join-Path $TaskRoot '.venv\Scripts\python.exe'
$TaskCli=Join-Path $PSScriptRoot 'r5c2_dqi_sensitivity_v2.py'
function Invoke-V2([string[]]$Arguments) {
    & $TaskPython $TaskCli @Arguments --config $Config
    if ($LASTEXITCODE -ne 0) { throw "v2 stage failed: $Arguments" }
}
Invoke-V2 -Arguments @('train-predictors','--stage','formal','--resume')
Invoke-V2 -Arguments @('freeze','--phase','runs')
Invoke-V2 -Arguments @('run-rl','--stage','formal','--resume')
Invoke-V2 -Arguments @('summarize')
Invoke-V2 -Arguments @('plot')
Invoke-V2 -Arguments @('verify')
