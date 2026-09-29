@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo RUNNING > logs\update_status.txt
echo ================================================================
echo  Update the documents from official.satbayev.university
echo  1. download new / missing / broken (truncated) files - already
echo     downloaded complete files are not downloaded again
echo  2. read only the new and changed files (incremental)
echo  3. retrain all our models and evaluate
echo ================================================================
.venv\Scripts\python scripts\01_download_documents.py
.venv\Scripts\python scripts\02_extract_text.py --incremental
.venv\Scripts\python scripts\run_all.py --from 3
echo UPDATE DONE > logs\update_status.txt
timeout /t 15
