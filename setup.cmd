@echo off
rem optimizer setup (Windows cmd.exe): the from-clone front door.
rem
rem   git clone https://github.com/SilvioBaratto/optimizer && cd optimizer
rem   setup.cmd
rem
rem Razor-thin funnel — no logic lives here. It ensures uv, installs the `portopt`
rem CLI from THIS checkout (the `ingestion` member provides the `portopt` dist; not
rem PyPI), sets OPTIMIZER_REPO so the out-of-repo tool venv can still locate
rem scripts\optimizer, then hands off to the tested core `portopt setup` with args.
setlocal
set "OPTIMIZER_REPO=%~dp0"

where uv >nul 2>nul || powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"

cd /d "%~dp0"
echo Installing the portopt CLI from this checkout...
uv tool install --from ./ingestion portopt || exit /b 1

portopt setup %*
