@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo PRETRAINING > logs\lm_status.txt
echo ================================================================
echo  OUR OWN LANGUAGE MODEL (LLM) - step 2: pre-training from scratch
echo  Runs up to 10 hours, then saves. Run this file again to continue
echo  from the same place (nothing is lost if the window is closed:
echo  a checkpoint is saved every 500 steps).
echo  Keep the laptop plugged in; do not put it to sleep.
echo ================================================================
.venv\Scripts\python scripts\15_lm_pretrain.py --size small --hours 10
echo.
type logs\lm_status.txt
pause
