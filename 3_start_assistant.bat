@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ================================================================
echo  Satbayev AI Consultant:  http://127.0.0.1:8000
echo  Admin panel:             http://127.0.0.1:8000/admin.html
echo  API docs (Swagger):      http://127.0.0.1:8000/docs
echo  The browser opens automatically when the server is ready.
echo  Close this window to stop the server.
echo ================================================================
rem --- stop an older copy of the server if it is still running (then this window starts a fresh one)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*serve.py*' -or $_.CommandLine -like '*backend.app.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
rem --- facts (rector, pro-rectors) from the signature blocks of the documents: a few seconds
.venv\Scripts\python scripts\07b_extract_facts.py
.venv\Scripts\python scripts\serve.py
pause
