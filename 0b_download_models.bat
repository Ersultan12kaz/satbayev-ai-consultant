@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Downloading the trained models and documents from GitHub (about 3 GB) ...
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
%PY% scripts\download_release.py %*
pause
