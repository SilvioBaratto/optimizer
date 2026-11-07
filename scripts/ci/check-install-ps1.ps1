# Runs install.ps1 end to end against stub git, uv and portopt commands and checks what
# it did, so the Windows installer is exercised on a real Windows runner under both
# Windows PowerShell 5.1 and PowerShell 7 (it also runs under pwsh on macOS and Linux).
#
# Nothing is installed or cloned: the stubs are .ps1 files placed first on PATH that log
# their arguments and exit with a code the scenario chooses. Syntax stays within
# Windows PowerShell 5.1 (no ternary, no &&, no ??).
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$install = Join-Path $repo "install.ps1"
# Lets a test point the harness at a modified copy to prove it catches a regression.
if ($env:PORTOPT_INSTALL_PS1) { $install = $env:PORTOPT_INSTALL_PS1 }
$engine = (Get-Process -Id $PID).Path
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("portopt-install-check-" + [guid]::NewGuid().ToString("N"))
$stubs = Join-Path $work "stubs"
New-Item -ItemType Directory -Force -Path $stubs | Out-Null

# git: logs the call, answers `remote get-url`, fails fetch/merge on request, fakes clone.
$gitStub = @'
$log = $env:STUB_LOG
Add-Content -Path $log -Value ("git " + ($args -join " "))
$joined = " " + ($args -join " ") + " "
if ($joined -like "* remote get-url *") {
    if ($env:STUB_ORIGIN) { Write-Output $env:STUB_ORIGIN; exit 0 }
    exit 2
}
if ($joined -like "* fetch *") { $rc = 0; if ($env:STUB_FETCH_RC) { $rc = [int]$env:STUB_FETCH_RC }; exit $rc }
if ($joined -like "* merge *") { $rc = 0; if ($env:STUB_MERGE_RC) { $rc = [int]$env:STUB_MERGE_RC }; exit $rc }
if ($joined -like "* clone *") {
    $dest = $args[$args.Count - 1]
    New-Item -ItemType Directory -Force -Path (Join-Path $dest ".git") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $dest "docker-compose.yml") | Out-Null
    exit 0
}
exit 0
'@
$uvStub = @'
Add-Content -Path $env:STUB_LOG -Value ("uv " + ($args -join " "))
exit 0
'@
$portoptStub = @'
Add-Content -Path $env:STUB_LOG -Value ("portopt " + ($args -join " "))
exit 0
'@
Set-Content -Path (Join-Path $stubs "git.ps1") -Value $gitStub
Set-Content -Path (Join-Path $stubs "uv.ps1") -Value $uvStub
Set-Content -Path (Join-Path $stubs "portopt.ps1") -Value $portoptStub

$failures = New-Object System.Collections.Generic.List[string]

function Invoke-Scenario {
    param(
        [string]$Name,
        [string]$HomeKind,
        [hashtable]$Environment,
        [bool]$ExpectSuccess,
        [string[]]$MustLog,
        [string[]]$MustNotLog
    )
    $dir = Join-Path $work $Name
    $homeDir = Join-Path $dir "home"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $log = Join-Path $dir "calls.log"
    New-Item -ItemType File -Force -Path $log | Out-Null
    if ($HomeKind -eq "git") { New-Item -ItemType Directory -Force -Path (Join-Path $homeDir ".git") | Out-Null }
    if ($HomeKind -eq "compose") {
        New-Item -ItemType Directory -Force -Path $homeDir | Out-Null
        New-Item -ItemType File -Force -Path (Join-Path $homeDir "docker-compose.yml") | Out-Null
    }

    $saved = @{}
    $settings = @{
        PORTOPT_HOME = $homeDir
        STUB_LOG     = $log
        STUB_ORIGIN  = ""
        STUB_FETCH_RC = ""
        STUB_MERGE_RC = ""
    }
    foreach ($key in $Environment.Keys) { $settings[$key] = $Environment[$key] }
    foreach ($key in $settings.Keys) {
        $saved[$key] = [System.Environment]::GetEnvironmentVariable($key)
        [System.Environment]::SetEnvironmentVariable($key, $settings[$key])
    }
    $savedPath = $env:PATH
    $env:PATH = $stubs + [System.IO.Path]::PathSeparator + $env:PATH

    $arguments = @("-NoProfile")
    if (($PSVersionTable.PSVersion.Major -lt 6) -or ($IsWindows)) { $arguments += @("-ExecutionPolicy", "Bypass") }
    $arguments += @("-File", $install, "--non-interactive")
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $engine @arguments *> (Join-Path $dir "out.txt")
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
        $env:PATH = $savedPath
        foreach ($key in $saved.Keys) { [System.Environment]::SetEnvironmentVariable($key, $saved[$key]) }
    }

    $calls = @(Get-Content -Path $log)
    $text = $calls -join "`n"
    $problems = New-Object System.Collections.Generic.List[string]
    if ($ExpectSuccess -and ($code -ne 0)) { $problems.Add("exit code $code, expected 0") }
    if ((-not $ExpectSuccess) -and ($code -eq 0)) { $problems.Add("exit code 0, expected a failure") }
    foreach ($needle in $MustLog) { if ($text -notlike "*$needle*") { $problems.Add("log lacks '$needle'") } }
    foreach ($needle in $MustNotLog) { if ($text -like "*$needle*") { $problems.Add("log has '$needle'") } }

    if ($problems.Count -eq 0) {
        Write-Host ("ok   {0} (exit {1}, {2} calls)" -f $Name, $code, $calls.Count)
    } else {
        Write-Host ("FAIL {0}: {1}" -f $Name, ($problems -join "; "))
        Write-Host ("     calls: {0}" -f ($calls -join " | "))
        $failures.Add($Name)
    }
}

$repoUrl = "https://github.com/SilvioBaratto/optimizer.git"

Invoke-Scenario -Name "same_repo_ssh_spelling" -HomeKind "git" -ExpectSuccess $true `
    -Environment @{ STUB_ORIGIN = "git@github.com:SilvioBaratto/optimizer.git" } `
    -MustLog @("fetch --quiet origin", "checkout --quiet", "merge --quiet --ff-only FETCH_HEAD", "uv tool install --force git+$repoUrl", "portopt setup") `
    -MustNotLog @()

Invoke-Scenario -Name "different_repo" -HomeKind "git" -ExpectSuccess $false `
    -Environment @{ STUB_ORIGIN = "https://github.com/someone/else.git" } `
    -MustLog @("remote get-url origin") `
    -MustNotLog @("fetch", "uv tool install", "portopt setup")

Invoke-Scenario -Name "merge_fails" -HomeKind "git" -ExpectSuccess $false `
    -Environment @{ STUB_ORIGIN = $repoUrl; STUB_MERGE_RC = "1" } `
    -MustLog @("merge --quiet --ff-only FETCH_HEAD") `
    -MustNotLog @("uv tool install", "portopt setup")

Invoke-Scenario -Name "fetch_fails" -HomeKind "git" -ExpectSuccess $false `
    -Environment @{ STUB_ORIGIN = $repoUrl; STUB_FETCH_RC = "1" } `
    -MustLog @("fetch --quiet origin") `
    -MustNotLog @("checkout", "uv tool install", "portopt setup")

Invoke-Scenario -Name "archive_without_git" -HomeKind "compose" -ExpectSuccess $true `
    -Environment @{} `
    -MustLog @("uv tool install --force git+$repoUrl", "portopt setup") `
    -MustNotLog @("git ")

Invoke-Scenario -Name "fresh_clone" -HomeKind "none" -ExpectSuccess $true `
    -Environment @{} `
    -MustLog @("git clone $repoUrl", "checkout --quiet", "uv tool install --force git+$repoUrl", "portopt setup") `
    -MustNotLog @()

Remove-Item -Recurse -Force -Path $work -ErrorAction SilentlyContinue

if ($failures.Count -gt 0) {
    Write-Host ("{0} scenario(s) failed under PowerShell {1}: {2}" -f $failures.Count, $PSVersionTable.PSVersion, ($failures -join ", "))
    exit 1
}
Write-Host ("install.ps1: all scenarios passed under PowerShell {0}" -f $PSVersionTable.PSVersion)
