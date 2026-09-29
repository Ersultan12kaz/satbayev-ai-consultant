@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ================================================================
echo  OUR OWN TRANSLATION MODEL (kk / ru / en) - step 1: data
echo  1. parallel sentences of the university documents (our aligner)
echo  2. open parallel corpora from OPUS (free, no login)
echo  3. our BPE tokenizer, encoded training data
echo  Uses only the processor and the internet (can run while the LLM trains).
echo ================================================================
.venv\Scripts\python scripts\11_prepare_translation_data.py
echo.
echo Finished. Next: 16_translator_2_train.bat (uses the GPU)
pause
