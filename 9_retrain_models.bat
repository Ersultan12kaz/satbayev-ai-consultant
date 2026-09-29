@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo RUNNING > logs\retrain_status.txt
rem --- stop older runs of this project that may still be working in the background (they hold RAM and log files)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*satbayev-rag-assistant*' -or $_.CommandLine -like '*backend.app.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo ================================================================
echo  Retrain all OUR models (no pre-trained models, no new download)
echo  step 2 (incremental): read the files skipped earlier (.doc/.xls forms)
echo          and OCR the Kazakh PDFs with a broken font encoding
echo  steps 3-9: chunking, language detector, scope + question-type
echo  classifiers, embeddings, index, reranker + sentence ranker, evaluation
echo ================================================================
rem --- remove the pre-trained LLM experiment (Ollama) - the project uses only its own models
if exist ml\llm.py del /q ml\llm.py
if exist backend\app\services\llm.py del /q backend\app\services\llm.py
if exist 10_install_local_ai.bat del /q 10_install_local_ai.bat
taskkill /f /im "ollama app.exe" >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1
if exist "%LOCALAPPDATA%\Programs\Ollama" winget uninstall -e --id Ollama.Ollama --silent --accept-source-agreements >nul 2>&1
if exist "%USERPROFILE%\.ollama\models" rmdir /s /q "%USERPROFILE%\.ollama\models"
rem --- the answer generator is our composer
powershell -NoProfile -Command "if (Test-Path .env) { $l = (Get-Content .env) -replace '^GENERATOR=.*','GENERATOR=composer'; [IO.File]::WriteAllLines((Resolve-Path .env).Path, $l) }"
.venv\Scripts\python scripts\02_extract_text.py --incremental
.venv\Scripts\python scripts\run_all.py --from 3
if errorlevel 1 (echo RETRAIN FAILED > logs\retrain_status.txt) else (echo RETRAIN DONE > logs\retrain_status.txt)
echo.
echo Finished. This window stays open - you can close it.
pause
