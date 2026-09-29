@echo off
rem one-time chain: upload everything to GitHub, then start the improvement pipeline (20)
cd /d "%~dp0"
call 23_upload_models_to_github.bat
start "Satbayev - improve" cmd /c 20_improve_assistant.bat
