param(
    [string]$ConfigPath = 'configs/r2_r05c02_dqi_sensitivity.json',
    [ValidateSet('predictors','runs','analysis')][string]$StartAt = 'predictors'
)

$ErrorActionPreference = 'Stop'
$R5C2Project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $R5C2Project
$R5C2Python = Join-Path $R5C2Project '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $R5C2Python)) { throw 'Project .venv Python was not found.' }
$R5C2Entry = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($R5C2Entry.execution_scope -eq 'smoke') { throw 'This launcher is for the formal entry config.' }
if (-not $R5C2Entry.output_root) { throw 'Config must explicitly specify output_root.' }
$R5C2Bundle = if ([IO.Path]::IsPathRooted($R5C2Entry.output_root)) { $R5C2Entry.output_root } else { Join-Path $R5C2Project $R5C2Entry.output_root }
if (-not (Test-Path -LiteralPath (Join-Path $R5C2Bundle 'protocol_frozen.json'))) {
    throw 'Complete data preparation and G5 pilot, then freeze --phase protocol before launching.'
}
$R5C2LogDirectory = Join-Path $R5C2Bundle 'terminal_logs'
[IO.Directory]::CreateDirectory($R5C2LogDirectory) | Out-Null
$R5C2Transcript = Join-Path $R5C2LogDirectory ('formal_{0}_{1}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)

function Invoke-R5C2Phase {
    param([string]$Phase, [string[]]$Extra = @())
    Write-Host "R5-C2 phase: $Phase" -ForegroundColor Cyan
    & $R5C2Python 'scripts/r5c2_dqi_sensitivity.py' $Phase '--config' $ConfigPath @Extra
    if ($LASTEXITCODE -ne 0) { throw "R5-C2 phase failed: $Phase (exit $LASTEXITCODE)" }
}

Start-Transcript -LiteralPath $R5C2Transcript | Out-Null
try {
    if ($StartAt -eq 'predictors') {
        Invoke-R5C2Phase -Phase 'train-predictors' -Extra @('--stage','formal','--resume')
    }
    if ($StartAt -in @('predictors','runs')) {
        Invoke-R5C2Phase -Phase 'freeze' -Extra @('--phase','runs')
        Invoke-R5C2Phase -Phase 'run-rl' -Extra @('--stage','formal','--resume')
    }
    Invoke-R5C2Phase -Phase 'summarize'
    & $R5C2Python 'scripts/render_r5c2_figure.py' '--config' $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw 'R5-C2 figure rendering failed.' }
    Invoke-R5C2Phase -Phase 'verify'
    Write-Host "Completed. Bundle: $R5C2Bundle" -ForegroundColor Green
}
finally { Stop-Transcript | Out-Null }
