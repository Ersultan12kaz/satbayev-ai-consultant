@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Re-running only the evaluation (step 9) with the current models...
.venv\Scripts\python scripts\run_all.py --from 9
echo EVAL DONE > logs\eval_status.txt
timeout /t 10
