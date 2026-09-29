# optimizer setup (Windows PowerShell): the from-clone front door.
#
#   git clone https://github.com/SilvioBaratto/optimizer; cd optimizer
#   powershell -File setup.ps1
#
# Razor-thin funnel — no logic lives here. It ensures uv, installs the `portopt`
# CLI from THIS checkout (the `ingestion` member provides the `portopt` dist; not
# PyPI), sets OPTIMIZER_REPO so the out-of-repo tool venv can still locate
# scripts\optimizer, then hands off to the tested core `portopt setup` with args.
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:OPTIMIZER_REPO = $here

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Astral)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

Set-Location $here
Write-Host "Installing the portopt CLI from this checkout..."
uv tool install --from ./ingestion portopt

portopt setup @args
