@echo off
setlocal
cd /d C:\lsg\grpo_ope_reuse
C:\Anaconda\envs\grpo_ope\python.exe run_program04_direct_repair_supervisor.py --root "C:\lsg\grpo_ope_reuse" --chunk 20
echo.
echo Direct Program 04 repair exited with code %ERRORLEVEL%.
pause
