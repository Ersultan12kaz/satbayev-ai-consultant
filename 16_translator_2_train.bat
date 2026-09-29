@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ================================================================
echo  OUR OWN TRANSLATION MODEL (kk / ru / en) - step 2: training on the GPU
echo  Up to 2 hours; run again to continue. The best model is saved to
echo  models\nmt\model.pt and used by the assistant after a restart.
echo ================================================================
.venv\Scripts\python scripts\12_train_translator.py --hours 2
start "Satbayev AI assistant" cmd /c 3_start_assistant.bat
pause
