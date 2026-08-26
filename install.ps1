# portopt bootstrap (Windows): install uv if missing, install the portopt CLI,
# then launch the interactive setup wizard.
#
#   powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.ps1 | iex"
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Astral)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

Write-Host "Installing the portopt CLI..."
uv tool install portopt

Write-Host "Launching the setup wizard..."
portopt setup
