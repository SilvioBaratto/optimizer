# portopt bootstrap (Windows): clone the repo (needs git; PORTOPT_HOME, default
# %USERPROFILE%\optimizer), install uv if missing, install the portopt CLI, then
# launch the interactive setup wizard.
#
#   powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.ps1 | iex"
#   powershell -File install.ps1 --non-interactive --llm-provider openrouter …
#
# No /dev/tty dance is needed here (unlike install.sh): PowerShell's pipeline passes
# objects, not process stdin handles, so `portopt setup` inherits the console stdin
# directly and interactive prompts reach the terminal. Parity is arg-forwarding.
$ErrorActionPreference = "Stop"

# The CLI drives `docker compose` against a checkout of this repo (compose file,
# Dockerfiles, scripts\), and installing it into a uv tool venv does not provide
# one. Leave a clone behind, reuse an existing one, and point the CLI at it.
$PortoptRepo = if ($env:PORTOPT_REPO) { $env:PORTOPT_REPO } else { "https://github.com/SilvioBaratto/optimizer.git" }
$PortoptRef  = if ($env:PORTOPT_REF)  { $env:PORTOPT_REF }  else { "development" }
$PortoptHome = if ($env:PORTOPT_HOME) { $env:PORTOPT_HOME } else { Join-Path $env:USERPROFILE "optimizer" }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to fetch the optimizer checkout; install it and re-run."
}

# Reduce a clone URL to host/owner/repo so https, ssh and scp-style spellings of the
# same repo compare equal.
function Get-RepoId([string]$Url) {
    $id = $Url -replace '^[A-Za-z0-9+.-]+://', '' -replace '^[^@/]+@', '' -replace ':', '/' -replace '/?\.git/?$', '' -replace '/$', ''
    return $id.ToLowerInvariant()
}

if (Test-Path (Join-Path $PortoptHome ".git")) {
    # fetch + checkout + merge rewrite the working copy, so only do it to a checkout of
    # this repo. A developer's own clone, or an unrelated repo, would otherwise be
    # switched to PORTOPT_REF without a word.
    $origin = git -C $PortoptHome remote get-url origin
    if ($LASTEXITCODE -ne 0 -or -not $origin -or (Get-RepoId $origin) -ne (Get-RepoId $PortoptRepo)) {
        throw "$PortoptHome is a git checkout of '$origin', not $PortoptRepo. Point PORTOPT_HOME at another path, or at a clone of $PortoptRepo, and re-run."
    }
    Write-Host "Updating the checkout in $PortoptHome..."
    # The CLI below is installed from PORTOPT_REF while the compose file, Dockerfiles and
    # migrations come from this checkout, so one left behind would pair a new CLI with
    # old stack files. Each native command's failure is invisible to
    # $ErrorActionPreference, so test it.
    git -C $PortoptHome fetch --quiet origin $PortoptRef
    if ($LASTEXITCODE -ne 0) { throw "git fetch $PortoptRef failed (exit $LASTEXITCODE); could not update $PortoptHome." }
    git -C $PortoptHome checkout --quiet $PortoptRef
    if ($LASTEXITCODE -ne 0) { throw "git checkout $PortoptRef failed (exit $LASTEXITCODE); could not update $PortoptHome." }
    git -C $PortoptHome merge --quiet --ff-only FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "could not fast-forward $PortoptHome to $PortoptRef (offline, or local changes or commits stop it). Resolve that, or point PORTOPT_HOME at another path, and re-run." }
} elseif (Test-Path (Join-Path $PortoptHome "docker-compose.yml")) {
    # A tree that carries the compose file but no git metadata (an unpacked archive)
    # cannot be updated, and `git clone` would refuse to write into it.
    Write-Host "Using the existing checkout at $PortoptHome"
} else {
    Write-Host "Cloning $PortoptRepo into $PortoptHome..."
    git clone $PortoptRepo $PortoptHome
    # A native command's failure is invisible to $ErrorActionPreference, so test it.
    if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE)." }
    # A separate checkout (not `clone --branch`) so PORTOPT_REF may be a branch, tag
    # or commit.
    git -C $PortoptHome checkout --quiet $PortoptRef
    if ($LASTEXITCODE -ne 0) { throw "git checkout $PortoptRef failed (exit $LASTEXITCODE)." }
}
$env:OPTIMIZER_REPO = $PortoptHome

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Astral)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# uv drops tool executables in ~/.local/bin; surface it to this session so the
# portopt shim is callable right after install.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# release.yml only builds the portopt-core library, never the CLI member, so
# install the CLI straight from git. The install tracks a branch instead of a release.
Write-Host "Installing the portopt CLI from $PortoptRepo@$PortoptRef..."
uv tool install --force "git+$PortoptRepo@$PortoptRef#subdirectory=ingestion"
# A native command's non-zero exit is not an exception even under "Stop"; without this
# a failed install would fall through and launch whatever portopt is already on PATH.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Launching the setup wizard..."
portopt setup @args
