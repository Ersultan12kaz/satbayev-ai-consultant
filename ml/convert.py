"""Text extraction for the other office formats published on the site:
.doc / .rtf (old Word), .xls / .xlsx (Excel).

.doc and .xls are converted with LibreOffice in headless mode (free, open source):
    soffice --headless --convert-to docx|xlsx
.rtf is additionally handled by a small built-in RTF-to-text converter, .xlsx is read with openpyxl.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

WINDOWS_SOFFICE = [r"C:\Program Files\LibreOffice\program\soffice.exe",
                   r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]


def soffice_cmd() -> Optional[str]:
    cmd = os.getenv("SOFFICE_CMD") or shutil.which("soffice") or shutil.which("libreoffice")
    if cmd:
        return cmd
    for p in WINDOWS_SOFFICE:
        if Path(p).exists():
            return p
    return None


def convert_with_libreoffice(path: Path, target_ext: str, out_dir: Path) -> Optional[Path]:
    cmd = soffice_cmd()
    if not cmd:
        return None
    profile = Path(out_dir) / "lo_profile"
    args = [cmd, f"-env:UserInstallation={profile.resolve().as_uri()}", "--headless", "--convert-to", target_ext,
            "--outdir", str(out_dir), str(path)]
    try:
        subprocess.run(args, capture_output=True, timeout=240)
    except Exception:
        return None
    out = Path(out_dir) / (path.stem + "." + target_ext)
    return out if out.exists() else None


_SKIP_DEST = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "listtable", "listoverridetable", "rsidtbl",
              "themedata", "colorschememapping", "datastore", "latentstyles", "generator", "xmlnstbl", "mmathPr",
              "pgdsctbl", "header", "footer", "headerl", "headerr", "footerl", "footerr", "object", "fldinst"}
_RTF_TOKEN = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}]+)", re.S)


def rtf_to_text(rtf: str, codepage: str = "cp1251") -> str:
    """Small stack-based RTF parser: skips non-text destinations, decodes \\uN and \\'hh."""
    out, stack, skip, uc, pending_skip = [], [], False, 1, 0
    first_in_group = False
    for word, num, hexc, sym, brace, text in _RTF_TOKEN.findall(rtf):
        if brace == "{":
            stack.append((skip, uc)); first_in_group = True
            continue
        if brace == "}":
            skip, uc = stack.pop() if stack else (False, 1)
            first_in_group = False
            continue
        if sym == "*":
            skip = True; first_in_group = False
            continue
        if word:
            if first_in_group and word in _SKIP_DEST:
                skip = True
            first_in_group = False
            if skip:
                continue
            if word == "uc" and num:
                uc = int(num)
            elif word == "u" and num:
                n = int(num)
                out.append(chr(n + 65536 if n < 0 else n)); pending_skip = uc
            elif word in ("par", "line", "row", "sect", "page"):
                out.append("\n")
            elif word == "cell":
                out.append(" | ")
            elif word == "tab":
                out.append(" ")
            continue
        first_in_group = False
        if skip:
            continue
        if hexc:
            if pending_skip:
                pending_skip -= 1
                continue
            out.append(bytes([int(hexc, 16)]).decode(codepage, "ignore"))
        elif sym:
            if sym in "\\{}":
                out.append(sym)
            elif sym == "~":
                out.append(" ")
        elif text:
            t = text.replace("\r", "").replace("\n", "")
            if pending_skip and t:
                drop = min(pending_skip, len(t)); t = t[drop:]; pending_skip -= drop
            out.append(t)
    return re.sub(r"\n\s*\n+", "\n", "".join(out)).strip()


def extract_xlsx(path: Path, rows_per_page: int = 40) -> List[str]:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    pages: List[str] = []
    for ws in wb.worksheets:
        lines = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
        for i in range(0, len(lines), rows_per_page):
            pages.append(f"{ws.title}\n" + "\n".join(lines[i:i + rows_per_page]))
    wb.close()
    return pages


def extract_office(path: Path, extract_docx) -> List[str]:
    """.doc / .rtf / .xls / .xlsx -> pages.

    The real format is sniffed from the first bytes (the site has .doc files that are really RTF or DOCX).
    Old binary Word / Excel files are read by our own parser (ml/msdoc.py); LibreOffice is used only
    as a fallback when it is installed.
    """
    from .msdoc import OLE_MAGIC, extract_binary_office
    head = path.read_bytes()[:8]
    suf = path.suffix.lower()
    if head.startswith(b"PK"):                                   # really a .docx / .xlsx
        return extract_xlsx(path) if suf in (".xls", ".xlsx") else extract_docx(path)
    if head.startswith(b"{\\rtf") or suf == ".rtf":
        text = rtf_to_text(path.read_text(encoding="latin-1", errors="ignore"))
        lines = text.split("\n")
        pages = ["\n".join(lines[i:i + 40]) for i in range(0, len(lines), 40)]
        if any(p.strip() for p in pages):
            return pages
    if head == OLE_MAGIC:
        pages = extract_binary_office(path)
        if pages:
            return pages
    with tempfile.TemporaryDirectory() as tmp:
        if suf in (".doc", ".rtf"):
            out = convert_with_libreoffice(path, "docx", Path(tmp))
            return extract_docx(out) if out else []
        if suf == ".xls":
            out = convert_with_libreoffice(path, "xlsx", Path(tmp))
            return extract_xlsx(out) if out else []
    return []
