# portopt bootstrap (Windows): install uv if missing, install the portopt CLI,
# then launch the interactive setup wizard.
#
#   powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.ps1 | iex"
#   powershell -File install.ps1 --non-interactive --llm-provider openrouter …
#
# No /dev/tty dance is needed here (unlike install.sh): PowerShell's pipeline passes
# objects, not process stdin handles, so `portopt setup` inherits the console stdin
# directly and interactive prompts reach the terminal. Parity is arg-forwarding.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Astral)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# uv drops tool executables in ~/.local/bin; surface it to this session so the
# portopt shim is callable right after install.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# Nothing in this project ships to PyPI, so install the CLI member straight from
# git. The install tracks a branch instead of a published release.
$PortoptRepo = if ($env:PORTOPT_REPO) { $env:PORTOPT_REPO } else { "https://github.com/SilvioBaratto/optimizer.git" }
$PortoptRef  = if ($env:PORTOPT_REF)  { $env:PORTOPT_REF }  else { "development" }
Write-Host "Installing the portopt CLI from $PortoptRepo@$PortoptRef..."
uv tool install --force "git+$PortoptRepo@$PortoptRef#subdirectory=ingestion"

Write-Host "Launching the setup wizard..."
portopt setup @args
