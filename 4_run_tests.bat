@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist logs mkdir logs
.venv\Scripts\python -m pytest -v > logs\tests.log 2>&1
type logs\tests.log
pause
