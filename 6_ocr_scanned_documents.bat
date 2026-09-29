@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ================================================================
echo  ALL FILES: pdf, docx, doc, rtf, xls, xlsx + OCR for scanned PDFs
echo  1) Tesseract OCR + LibreOffice (for .doc/.xls)  2) OCR language models
echo  3) download every file of every section (up to 80 MB)
echo  4) extract (with OCR) + retrain all models + evaluate
echo ================================================================
if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
if not exist "C:\Program Files\LibreOffice\program\soffice.exe" winget install -e --id TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\setup_ocr.py
.venv\Scripts\python scripts\01_download_documents.py --max-mb 80
.venv\Scripts\python scripts\run_all.py --from 2
echo ALL FILES DONE > logs\all_files_status.txt
echo.
pause
