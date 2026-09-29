@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI
title Satbayev - upload models and documents to GitHub
echo ================================================================
echo  1. new/changed code  -^> GitHub (commit on main and develop)
echo  2. trained models + documents -^> GitHub Release v1.0 (zip files, about 3 GB)
echo  Uploading takes 10-40 minutes depending on the internet speed.
echo ================================================================
echo STARTED %date% %time% > logs\github_release_status.txt
gh auth status >nul 2>nul
if errorlevel 1 gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
echo CODE %date% %time% >> logs\github_release_status.txt
%PY% scripts\22_push_to_github.py > logs\github_push.log 2>&1
type logs\github_push.log
echo RELEASE %date% %time% >> logs\github_release_status.txt
%PY% scripts\23_upload_release.py > logs\github_release.log 2>&1
set RC=%errorlevel%
type logs\github_release.log
if "%RC%"=="0" (echo DONE %date% %time% >> logs\github_release_status.txt) else (echo ERROR %RC% %date% %time% >> logs\github_release_status.txt)
