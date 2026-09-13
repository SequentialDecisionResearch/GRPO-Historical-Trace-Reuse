@echo off
setlocal EnableExtensions

set "ROOT=C:\lsg\grpo_ope_reuse"
set "PYTHON=C:\Anaconda\envs\grpo_ope\python.exe"
set "WORKER=%ROOT%\run_program04_chunk_worker.py"
set "FINAL=%ROOT%\run_program04_reduced_grid.py"
set "CHUNK=20"

cd /d "%ROOT%"

if not exist "%PYTHON%" (
  echo ERROR: Python not found: %PYTHON%
  goto fail
)
if not exist "%WORKER%" (
  echo ERROR: Worker not found: %WORKER%
  goto fail
)
if not exist "%FINAL%" (
  echo ERROR: Final runner not found: %FINAL%
  goto fail
)

echo ============================================================
echo PROGRAM 04 AUTOMATIC FRESH-PROCESS SUPERVISOR
echo Chunk size: %CHUNK% new shard units per Python process
echo Do NOT run Program 04 in Spyder at the same time.
echo ============================================================

:loop
echo.
echo [%date% %time%] Starting fresh worker...
"%PYTHON%" "%WORKER%" --root "%ROOT%" --device cuda --max-new-shards %CHUNK%
set "RC=%ERRORLEVEL%"

if "%RC%"=="75" (
  echo [%date% %time%] Fresh worker ended normally.
  echo Waiting 5 seconds before next Python process...
  timeout /t 5 /nobreak >nul
  goto loop
)

if not "%RC%"=="0" (
  echo Worker returned error code %RC%.
  goto fail
)

echo.
echo ============================================================
echo All selected shard counts complete.
echo Running full reduced-grid finalization...
echo ============================================================
"%PYTHON%" "%FINAL%" --root "%ROOT%" --device cuda
if errorlevel 1 goto fail

echo.
echo ============================================================
echo Running final verify-only...
echo ============================================================
"%PYTHON%" "%FINAL%" --root "%ROOT%" --device cuda --verify-only
if errorlevel 1 goto fail

echo.
echo ============================================================
echo PROGRAM 04 REDUCED GRID COMPLETED AND VERIFIED
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo SUPERVISOR STOPPED ON ERROR
echo Existing published shards were NOT deleted.
echo Send the last screen output to ChatGPT.
echo ============================================================
pause
exit /b 2
