@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI
echo ================================================================
echo  GitHub: publishing the project stage by stage (plan sections 27 and 29)
echo  branches: main, develop, feature/rag, feature/backend, feature/frontend, feature/evaluation
echo ================================================================
echo STARTED %date% %time% > logs\github_status.txt
where gh >nul 2>nul
if errorlevel 1 winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
gh auth status >nul 2>nul
if errorlevel 1 (
  echo.
  echo  1. Press Enter - the browser opens github.com
  echo  2. Log in to GitHub and type the 8-letter code shown below
  echo.
  gh auth login --hostname github.com --git-protocol https --web
)
gh auth status >nul 2>nul
if errorlevel 1 (
  echo LOGIN FAILED %date% %time% >> logs\github_status.txt
  echo GitHub login did not finish. Run this file again.
  pause
  exit /b 1
)
echo LOGGED IN %date% %time% >> logs\github_status.txt
gh auth setup-git
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
echo.
echo  Building commits and pushing, please wait ...
%PY% scripts\22_push_to_github.py %* > logs\github_push.log 2>&1
set RC=%errorlevel%
type logs\github_push.log
if "%RC%"=="0" (echo DONE %date% %time% >> logs\github_status.txt) else (echo ERROR %RC% %date% %time% >> logs\github_status.txt)
echo.
if exist logs\github_url.txt (echo Repository: & type logs\github_url.txt)
pause
