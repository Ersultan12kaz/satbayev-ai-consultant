@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ================================================================
echo  Full pipeline: download documents - extract - chunk -
echo  train 4 models - build index - evaluate (300 questions)
echo  Logs: logs\*.log   Reports: reports\
echo ================================================================
.venv\Scripts\python scripts\run_all.py %*
echo.
pause
