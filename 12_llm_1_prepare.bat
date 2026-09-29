@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo PREPARING > logs\lm_status.txt
echo ================================================================
echo  OUR OWN LANGUAGE MODEL (LLM) - step 1: preparation
echo  1. PyTorch with GPU support (about 2.5 GB download, once)
echo  2. text corpus: Kazakh Wikipedia + part of Russian Wikipedia
echo     + all university documents and website pages
echo  3. our tokenizer + corpus turned into tokens
echo  4. self-test of the model on the GPU (30 steps)
echo  Takes 1-3 hours (mostly downloading). You can use the computer meanwhile.
echo ================================================================
nvidia-smi > logs\gpu.txt 2>&1
.venv\Scripts\python -c "import torch" 2>nul
if errorlevel 1 (
  echo Installing PyTorch with CUDA ...
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 > logs\torch_install.log 2>&1
)
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > logs\torch_check.txt 2>&1
type logs\torch_check.txt
.venv\Scripts\python scripts\13_lm_corpus.py
if errorlevel 1 goto failed
.venv\Scripts\python scripts\14_lm_tokenize.py
if errorlevel 1 goto failed
.venv\Scripts\python scripts\15_lm_pretrain.py --selftest
if errorlevel 1 goto failed
echo PREPARED > logs\lm_status.txt
echo.
echo Step 1 finished. Next: 13_llm_2_pretrain.bat
goto end
:failed
echo PREPARE FAILED > logs\lm_status.txt
echo Something failed - see the logs folder.
:end
pause
