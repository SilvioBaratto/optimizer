@echo off
rem optimizer.cmd - Windows launcher: locate bash.exe and delegate to the POSIX
rem `optimizer` shim. OPTIMIZER_REPO is the repo root; path_install embeds the
rem absolute path when copying this file onto the User PATH. Run in place from
rem scripts/, %~dp0.. already points at the repo root.
setlocal

if not defined OPTIMIZER_REPO set "OPTIMIZER_REPO=%~dp0.."

rem Prefer a real Git Bash. A bare PATH lookup finds C:\Windows\System32\bash.exe
rem first, the WSL launcher, which has no MSYS environment and dies with a relay
rem error. Probe Git's own locations before any non-WSL bash left on PATH.
set "BASH_EXE="
if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%ProgramFiles%\Git\usr\bin\bash.exe" set "BASH_EXE=%ProgramFiles%\Git\usr\bin\bash.exe"
if not defined BASH_EXE if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH_EXE=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"
if not defined BASH_EXE for %%B in (bash.exe) do if not "%%~$PATH:B"=="" if /I not "%%~$PATH:B"=="%SystemRoot%\System32\bash.exe" set "BASH_EXE=%%~$PATH:B"
if not defined BASH_EXE (echo optimizer: cannot find Git Bash. Install Git for Windows. 1>&2 & exit /b 1)

"%BASH_EXE%" "%OPTIMIZER_REPO%\scripts\optimizer" %*
