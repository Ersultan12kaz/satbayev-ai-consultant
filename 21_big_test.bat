@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title Satbayev - big test (1100+ questions)
echo Big automatic test: 1100+ questions (kk/ru/en/mixed/typos/off-topic). Result: reports\big_test.md
.venv\Scripts\python scripts\21_big_test.py
echo BIGTEST DONE %date% %time% > logs\bigtest_status.txt
pause
