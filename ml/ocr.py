"""OCR for scanned PDFs (pages that are only pictures, no text layer).

Pipeline: PDF page -> image (pypdfium2, 250 dpi) -> Tesseract OCR (rus / kaz / eng models)
-> text.  Tesseract is an open-source OCR engine (no API); the language models
(rus.traineddata, kaz.traineddata, eng.traineddata) are stored in <project>/tessdata.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
TESSDATA = ROOT / "tessdata"
OCR_LANGS = {"ru": "rus+eng", "kk": "kaz+rus+eng", "en": "eng"}
WINDOWS_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                 os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")]


def tesseract_cmd() -> Optional[str]:
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    for p in WINDOWS_PATHS:
        if Path(p).exists():
            return p
    return None


def available_langs() -> set:
    cmd = tesseract_cmd()
    if not cmd:
        return set()
    args = [cmd, "--list-langs"]
    if TESSDATA.exists():
        args += ["--tessdata-dir", str(TESSDATA)]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    return {l.strip() for l in out.splitlines()[1:] if l.strip()}


def ocr_available() -> bool:
    return bool(tesseract_cmd()) and "eng" in available_langs()


def _langs_for(lang: str) -> str:
    have = available_langs()
    wanted = [l for l in OCR_LANGS.get(lang, "rus+kaz+eng").split("+") if l in have]
    return "+".join(wanted) or "eng"


def ocr_pdf(path: Path, lang: str = "ru", dpi: int = 250, max_pages: int = 300) -> List[str]:
    import pypdfium2 as pdfium
    cmd = tesseract_cmd()
    if not cmd:
        raise RuntimeError("Tesseract is not installed (run 6_ocr_scanned_documents.bat)")
    langs = _langs_for(lang)
    pdf = pdfium.PdfDocument(str(path))
    pages: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(min(len(pdf), max_pages)):
            img = pdf[i].render(scale=dpi / 72).to_pil().convert("L")
            png = Path(tmp) / f"p{i}.png"
            img.save(png)
            args = [cmd, str(png), "stdout", "-l", langs, "--psm", "3"]
            if TESSDATA.exists():
                args += ["--tessdata-dir", str(TESSDATA)]
            r = subprocess.run(args, capture_output=True, timeout=300)
            pages.append(r.stdout.decode("utf-8", errors="ignore"))
    pdf.close()
    return pages
