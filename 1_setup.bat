@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist logs mkdir logs
echo ================================================================
echo  Creating virtual environment .venv and installing libraries
echo ================================================================
set PY=
py -3.11 --version >nul 2>nul && set PY=py -3.11
if "%PY%"=="" py --version >nul 2>nul && set PY=py
if "%PY%"=="" set PY=python
%PY% --version
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
  echo SETUP FAILED > logs\setup_status.txt
  echo pip install failed, see the messages above
  pause
  exit /b 1
)
if not exist .env .venv\Scripts\python -c "import secrets;t=open('.env.example',encoding='utf-8').read().replace('put-a-long-random-string-here',secrets.token_hex(32));open('.env','w',encoding='utf-8').write(t)"
.venv\Scripts\python -m pip freeze > logs\pip_freeze.txt
echo SETUP OK > logs\setup_status.txt
echo.
echo SETUP OK. Next: 2_train_all.bat
pause
