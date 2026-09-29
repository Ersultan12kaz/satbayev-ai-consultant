@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo FINETUNING > logs\lm_status.txt
echo ================================================================
echo  OUR OWN LANGUAGE MODEL (LLM) - step 3: learning to answer
echo  1. question - context - answer examples from our documents
echo  2. fine-tuning of the pre-trained model (answers in own words,
echo     in the language of the question, only from the documents)
echo  3. restart of the assistant with the language model
echo ================================================================
.venv\Scripts\python scripts\16_lm_sft_data.py
if errorlevel 1 goto failed
.venv\Scripts\python scripts\17_lm_finetune.py
if errorlevel 1 goto failed
start "Satbayev AI assistant" cmd /c 3_start_assistant.bat
goto end
:failed
echo FINETUNE FAILED > logs\lm_status.txt
echo Something failed - see the logs folder.
:end
pause
