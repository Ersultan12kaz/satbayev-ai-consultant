@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Windows "Application Control" blocked a compiled SQLAlchemy module.
echo Installing SQLAlchemy 2.0 (it falls back to pure Python automatically)...
.venv\Scripts\python -m pip install "sqlalchemy>=2.0.30,<2.1" > logs\fix_sqlalchemy.log 2>&1
.venv\Scripts\python -c "import sqlalchemy, fastapi, pydantic_core; print('imports OK', sqlalchemy.__version__)" >> logs\fix_sqlalchemy.log 2>&1
type logs\fix_sqlalchemy.log
echo FIX DONE >> logs\fix_sqlalchemy.log
call 3_start_assistant.bat
