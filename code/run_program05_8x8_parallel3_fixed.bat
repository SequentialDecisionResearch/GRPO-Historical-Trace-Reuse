@echo off
setlocal
cd /d C:\lsg\grpo_ope_reuse
C:\Anaconda\envs\grpo_ope\python.exe run_program05_8x8_parallel3_fixed.py --root "C:\lsg\grpo_ope_reuse" --chunk 10
echo.
echo Program 05 supervisor exited with code %ERRORLEVEL%.
pause
