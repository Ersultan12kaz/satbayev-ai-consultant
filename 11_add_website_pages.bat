@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo RUNNING > logs\website_status.txt
rem --- stop older crawls/trainings of this project still running in the background (the server keeps running)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*satbayev-rag-assistant*scripts*' -and $_.CommandLine -notlike '*serve.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo ================================================================
echo  Add general university information from the websites
echo  (history, management, institutes, admission, student life ...)
echo  1. second pass of the document download (files that failed)
echo  2. crawl satbayev.university + official.satbayev.university pages (continues where it stopped)
echo  3. read new files, retrain all our models, evaluate, restart the assistant
echo ================================================================
.venv\Scripts\python scripts\01_download_documents.py
.venv\Scripts\python scripts\01b_download_website.py
.venv\Scripts\python scripts\02_extract_text.py --incremental
.venv\Scripts\python scripts\run_all.py --from 3
if errorlevel 1 (
  echo WEBSITE TRAINING FAILED > logs\website_status.txt
) else (
  echo WEBSITE DONE > logs\website_status.txt
  rem the assistant restarts with the new models in its own window
  start "Satbayev AI assistant" cmd /c 3_start_assistant.bat
)
echo.
echo Finished. This window stays open - you can close it.
pause
