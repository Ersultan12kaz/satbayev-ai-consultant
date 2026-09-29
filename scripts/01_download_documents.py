"""STEP 1 - Collect the official documents of Satbayev University.

Source: https://official.satbayev.university/<lang>/docs  ("Внутренние нормативные документы")
The script visits the sections that are relevant for students in three languages
(ru / kk / en), finds every /download/document/<id>/<name>.pdf|docx link and saves
the file to data/raw/<lang>/.  A manifest (data/raw/manifest.jsonl) keeps the
metadata: title, language, section, url, local path, size.

Usage:
    python scripts/01_download_documents.py                # default sections
    python scripts/01_download_documents.py --student-only   # only the student sections, pdf/docx
    python scripts/01_download_documents.py --max-mb 30 --langs ru kk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import urllib.parse as up

import requests
import urllib3
from bs4 import BeautifulSoup

from _common import RAW, Tee, banner

BASE = "https://official.satbayev.university"

# Sections of the "Внутренние нормативные документы" page
STUDENT_SECTIONS = {
    "regulatory/3-educational": "Управление образовательным процессом / Education",
    "regulatory/3-student-affairs": "Воспитательная работа, общежития / Student affairs",
    "regulatory/3-finance": "Стипендии и финансы / Finance & scholarships",
    "2-dp": "Документированные процедуры (учебный процесс, практика, мобильность)",
    "docs/drugie-dokumenty": "Инструкции для студентов / Other documents",
    "regulatory/3-korp": "Корпоративное управление / Corporate",
    "regulatory/3-science": "Научная деятельность / Research",
    "regulatory/3-qms": "Система качества / QMS",
    "regulatory/3-info": "Информационное обеспечение",
    "docs/iso-9001-1": "Политики (1 уровень)",
    "regulatory/2-standards": "Стандарты КазНИТУ",
}
OTHER_SECTIONS = {
    "regulatory/2-forms": "Формы", "2-labs": "Лаборатории", "2-nilas": "НИЛАС", "regulatory/2-ltstm": "ЛЦТМ",
    "regulatory/2-antikor": "Антикоррупционный менеджмент",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (SatbayevRAG diploma project; educational use)"}


def get(session: requests.Session, url: str, **kw) -> requests.Response:
    """GET with retries. If the site certificate chain is incomplete on this computer,
    retry without verification (public documents, read-only)."""
    for attempt in range(4):
        try:
            try:
                r = session.get(url, timeout=60, **kw)
            except requests.exceptions.SSLError:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                r = session.get(url, timeout=60, verify=False, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def find_links(session, lang: str, section: str):
    r = get(session, f"{BASE}/{lang}/{section}")
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select('a[href*="/download/document/"]'):
        href = a.get("href", "")
        full = up.urljoin(BASE, href)
        name = up.unquote(full.rsplit("/", 1)[-1]).strip()
        m = re.search(r"/download/document/(\d+)/", full)
        if not m:
            continue
        out.append({"doc_num": m.group(1), "file_name": name, "url": full})
    return out


def safe_name(name: str, limit: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name).strip()
    stem, dot, ext = name.rpartition(".")
    stem = re.sub(r"\s+", " ", stem)[:limit].strip()
    return f"{stem}.{ext}" if dot else name[:limit]


def save_manifest(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for d in records.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def file_complete(path: Path) -> bool:
    """A cached file is used only if it is complete: a PDF must end with %%EOF, DOCX/XLSX must be a valid ZIP.
    (An interrupted download leaves a truncated file of a "round" size that cannot be read.)"""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    suf = path.suffix.lower()
    try:
        if suf == ".pdf":
            with open(path, "rb") as f:
                f.seek(max(0, path.stat().st_size - 4096))
                return b"%%EOF" in f.read()
        if suf in (".docx", ".xlsx"):
            import zipfile
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None
    except Exception:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ru", "kk", "en"])
    ap.add_argument("--student-only", action="store_true", help="skip forms / laboratory procedures")
    ap.add_argument("--max-mb", type=float, default=80.0, help="skip files bigger than this")
    ap.add_argument("--types", nargs="+", default=["pdf", "docx", "doc", "rtf", "xlsx", "xls"])
    ap.add_argument("--limit", type=int, default=0, help="max files per language (0 = no limit)")
    args = ap.parse_args()
    Tee("01_download")
    banner("STEP 1. Downloading official documents from official.satbayev.university")

    sections = dict(STUDENT_SECTIONS, **({} if args.student_only else OTHER_SECTIONS))
    session = requests.Session()
    session.headers.update(HEADERS)
    manifest_path = RAW / "manifest.jsonl"
    existing = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                existing[d["doc_id"]] = d

    records = dict(existing)
    stats = {"found": 0, "downloaded": 0, "skipped_big": 0, "skipped_type": 0, "cached": 0, "errors": 0}
    for lang in args.langs:
        n_lang = 0
        for section, label in sections.items():
            try:
                links = find_links(session, lang, section)
            except Exception as e:
                print(f"  [{lang}] {section}: page error {e}")
                continue
            print(f"  [{lang}] {section:32s} {len(links):4d} files  ({label})")
            for l in links:
                stats["found"] += 1
                ext = l["file_name"].rsplit(".", 1)[-1].lower()
                if ext not in args.types:
                    stats["skipped_type"] += 1
                    continue
                doc_id = f"{lang}_{l['doc_num']}"
                if doc_id in records and file_complete(RAW / records[doc_id]["path"]):
                    stats["cached"] += 1
                    continue
                if args.limit and n_lang >= args.limit:
                    continue
                target = RAW / lang / f"{l['doc_num']}_{safe_name(l['file_name'])}"
                target.parent.mkdir(parents=True, exist_ok=True)
                title = l["file_name"].rsplit(".", 1)[0].strip()
                rec = {"doc_id": doc_id, "doc_num": l["doc_num"], "lang": lang, "category": section,
                       "title": title, "file_name": l["file_name"], "url": l["url"],
                       "path": str(target.relative_to(RAW)).replace("\\", "/")}
                if file_complete(target):                                # downloaded in a previous run
                    records[doc_id] = dict(rec, size=target.stat().st_size)
                    stats["cached"] += 1
                    continue
                try:
                    url = requests.utils.requote_uri(l["url"])
                    r = get(session, url, stream=True)
                    size = int(r.headers.get("content-length") or 0)
                    if size and size > args.max_mb * 1e6:
                        stats["skipped_big"] += 1
                        print(f"      skip (>{args.max_mb} MB): {l['file_name'][:80]}")
                        continue
                    tmp = target.with_suffix(target.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for block in r.iter_content(1 << 16):
                            f.write(block)
                    if size and tmp.stat().st_size < size:
                        raise IOError(f"incomplete download ({tmp.stat().st_size} of {size} bytes)")
                    tmp.replace(target)
                    records[doc_id] = dict(rec, size=target.stat().st_size)
                    save_manifest(manifest_path, records)   # progress is kept if the run is stopped
                    stats["downloaded"] += 1
                    n_lang += 1
                    time.sleep(0.2)
                except Exception as e:
                    stats["errors"] += 1
                    print(f"      error: {l['file_name'][:80]} -> {e}")
        save_manifest(manifest_path, records)

    # documents added by hand: data/raw/extra/<lang>/<file>  (e.g. the freshman guide)
    extra_dir = RAW / "extra"
    n_extra = 0
    for f in sorted(extra_dir.glob("*/*")) if extra_dir.exists() else []:
        if f.suffix.lower().lstrip(".") not in ("pdf", "docx", "doc", "rtf", "xlsx", "xls", "txt"):
            continue
        lang = f.parent.name if f.parent.name in ("ru", "kk", "en") else "ru"
        doc_id = "extra_" + hashlib.md5(str(f.relative_to(RAW)).encode("utf-8")).hexdigest()[:10]
        records[doc_id] = {"doc_id": doc_id, "doc_num": doc_id, "lang": lang, "category": "extra",
                           "title": f.stem, "file_name": f.name, "url": "",
                           "path": str(f.relative_to(RAW)).replace("\\", "/"), "size": f.stat().st_size}
        n_extra += 1
    save_manifest(manifest_path, records)
    if n_extra:
        print(f"  extra documents added from data/raw/extra: {n_extra}")

    banner("Summary")
    for k, v in stats.items():
        print(f"  {k:14s}: {v}")
    by_lang = {}
    for d in records.values():
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1
    print(f"  documents in manifest: {len(records)}  by language: {by_lang}")
    print(f"  saved to: {RAW}")


if __name__ == "__main__":
    main()
