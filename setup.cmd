@echo off
rem optimizer setup (Windows cmd.exe): the from-clone front door.
rem
rem   git clone https://github.com/SilvioBaratto/optimizer && cd optimizer
rem   setup.cmd
rem
rem Razor-thin funnel — no logic lives here. It ensures uv, installs the `portopt`
rem CLI from THIS checkout (the workspace, not PyPI), sets OPTIMIZER_REPO so the
rem out-of-repo tool venv can still locate scripts\optimizer, then hands off to the
rem tested core `portopt setup` with the caller's args.
setlocal
set "OPTIMIZER_REPO=%~dp0"

where uv >nul 2>nul || powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"

cd /d "%~dp0"
echo Installing the portopt CLI from this checkout...
uv tool install --from . portopt || exit /b 1

portopt setup %*
