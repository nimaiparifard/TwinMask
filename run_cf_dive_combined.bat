@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
if not defined CF_DIVE_PYTHON set "CF_DIVE_PYTHON=python"
"%CF_DIVE_PYTHON%" -u -m cf_dive.combined %*
set "CF_DIVE_STATUS=%ERRORLEVEL%"
popd
if not "%CF_DIVE_STATUS%"=="0" echo CF-DIVE failed with exit code %CF_DIVE_STATUS%. 1>&2
exit /b %CF_DIVE_STATUS%
