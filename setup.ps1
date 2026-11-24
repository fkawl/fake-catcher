param([string]$Python = 'python', [ValidateSet('cpu','cu124')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/$Device"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& '.\.venv\Scripts\python.exe' -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Host 'Ready. Run run.cmd and open http://127.0.0.1:8765'
