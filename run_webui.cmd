@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul 2>&1

set "PYEXE="
set "SCRIPT=%~dp0webui.py"

if defined PYBOT_PYTHON (
    if exist "%PYBOT_PYTHON%" call :trypython "%PYBOT_PYTHON%"
)

if not defined PYEXE call :trypython "D:\Anaconda\python.exe"
if not defined PYEXE call :trypython "D:\anaconda3\python.exe"
if not defined PYEXE call :trypython "D:\ProgramData\Anaconda3\python.exe"
if not defined PYEXE call :trypython "%USERPROFILE%\Anaconda3\python.exe"
if not defined PYEXE call :trypython "%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE call :trypython "%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PYEXE call :trypython "C:\ProgramData\Anaconda3\python.exe"

if not defined PYEXE call :trypython "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYEXE call :trypython "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE call :trypython "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE call :trypython "%ProgramFiles%\Python313\python.exe"
if not defined PYEXE call :trypython "%ProgramFiles%\Python312\python.exe"

if not defined PYEXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYEXE call :trypython "%%P"
    )
)

if not defined PYEXE goto :nopython

echo Using interpreter: %PYEXE%
pushd "%~dp0"
"%PYEXE%" -B "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" pause
exit /b %RC%

:nopython
echo.
echo [ERROR] No usable Python 3.10+ interpreter with numpy and scipy was found.
echo.
echo   Fix: set PYBOT_PYTHON to an Anaconda/Python interpreter with numpy and scipy,
echo   for example:
echo       set PYBOT_PYTHON=D:\Anaconda\python.exe
echo   then run this script again.
echo.
pause
exit /b 1

:trypython
if defined PYEXE exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys, numpy, scipy; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 set "PYEXE=%~1"
exit /b 0
