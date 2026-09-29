@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title Satbayev - improve the assistant (website + retrain + LLM)
echo ================================================================
echo  IMPROVE THE ASSISTANT (about 4-5 hours; one job at a time so the laptop does not overheat):
echo   1. download the rest of the university website pages
echo   2. read them and retrain the search models (index, embeddings, reranker)
echo   3. new LLM training examples (2x more kk-ru translation) + fine-tuning
echo   4. big test: 1100+ questions -^> reports\big_test.md
echo   5. restart the assistant with the new models
echo  Do not close this window. Keep the laptop plugged in.
echo ================================================================
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*satbayev-rag-assistant*scripts*' -and $_.CommandLine -notlike '*serve.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo 1 WEBSITE %date% %time% > logs\improve_status.txt
.venv\Scripts\python scripts\01b_download_website.py --max-pages 6000
.venv\Scripts\python scripts\02_extract_text.py --incremental
echo 2 RETRAIN SEARCH %date% %time% > logs\improve_status.txt
.venv\Scripts\python scripts\run_all.py --from 3
if errorlevel 1 (
  echo FAILED at step 2 %date% %time% > logs\improve_status.txt
  goto end
)
echo 3 LLM %date% %time% > logs\improve_status.txt
.venv\Scripts\python scripts\16_lm_sft_data.py
if errorlevel 1 (
  echo FAILED at step 3 sft data %date% %time% > logs\improve_status.txt
  goto end
)
.venv\Scripts\python scripts\17_lm_finetune.py --hours 3
if errorlevel 1 (
  echo FAILED at step 3 finetune %date% %time% > logs\improve_status.txt
  goto end
)
echo 4 BIG TEST %date% %time% > logs\improve_status.txt
.venv\Scripts\python scripts\21_big_test.py
echo IMPROVE DONE %date% %time% > logs\improve_status.txt
start "Satbayev AI assistant" cmd /c 3_start_assistant.bat
:end
echo Finished - this window can be closed.
pause
