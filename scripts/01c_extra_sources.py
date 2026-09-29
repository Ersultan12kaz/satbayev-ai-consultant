"""STEP 1c - Other public sources about the university and student life (added on top of the documents).

The official regulations and the university websites do not contain everything a student asks:
who Kanysh Satbayev was, facts about the university from encyclopedias, the current amount of the
state scholarship (set by the Government, not by the university), etc.  This step downloads a
curated list of public pages (Wikipedia in kk/ru/en, gov.kz news, education news), keeps their main
text and adds them to data/raw/manifest.jsonl as documents of category "extra-web".  New pages are
then read by step 2 (incremental) - nothing already read is deleted.

    python scripts/01c_extra_sources.py
Add more pages to data/extra_sources.txt (one URL per line, optional "lang|url").
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import requests

from _common import RAW, ROOT, Tee, banner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

web = import_module("01b_download_website")          # reuse fetch() and main_text()

SOURCES = [
    # the university
    ("en", "https://en.wikipedia.org/wiki/Satbayev_University"),
    ("ru", "https://ru.wikipedia.org/w/index.php?search=Satbayev+University&go=Go"),
    ("kk", "https://kk.wikipedia.org/w/index.php?search=Сәтбаев+университеті&go=Go"),
    # Kanysh Satbayev
    ("en", "https://en.wikipedia.org/wiki/Kanysh_Satbayev"),
    ("ru", "https://ru.wikipedia.org/w/index.php?search=Сатпаев+Каныш+Имантаевич&go=Go"),
    ("kk", "https://kk.wikipedia.org/w/index.php?search=Қаныш+Сәтбаев&go=Go"),
    # university pages that the crawler may reach late
    ("en", "https://satbayev.university/en/history"),
    ("en", "https://satbayev.university/en/about"),
    ("ru", "https://satbayev.university/ru/history"),
    ("kk", "https://satbayev.university/kk/history"),
    # state scholarship amounts (set by the Government)
    ("ru", "https://www.gov.kz/memleket/entities/edu/press/news/details/802339?lang=ru"),
    ("ru", "https://tengrinews.kz/newseducation/kakimi-budut-stipendii-v-kazahstane-v-2026-godu-583350/"),
]


def load_sources():
    items = list(SOURCES)
    extra = ROOT / "data" / "extra_sources.txt"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lang, _, url = line.partition("|") if "|" in line else ("ru", "", line)
            items.append((lang.strip(), url.strip()))
    return items


def main():
    Tee("01c_extra_sources")
    banner("STEP 1c. Other public sources (encyclopedia, gov.kz, news)")
    manifest_path = RAW / "manifest.jsonl"
    records = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); records[d["doc_id"]] = d
    session = requests.Session(); session.headers.update(web.HEADERS)
    added = 0
    for lang, url in load_sources():
        r = web.fetch(session, url)
        if r is None or r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            print(f"  skipped ({'no answer' if r is None else r.status_code}): {url}")
            continue
        title, text = web.main_text(r.text)
        if len(text) < 300:
            print(f"  too short: {url}")
            continue
        doc_id = "xweb_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        path = RAW / "extra_web" / lang / f"{doc_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = text if text.startswith(title) else f"{title}\n{text}"
        path.write_text(body, encoding="utf-8")
        records[doc_id] = {"doc_id": doc_id, "doc_num": doc_id, "lang": lang, "category": "extra-web",
                           "title": title or url, "file_name": path.name, "url": r.url or url,
                           "path": str(path.relative_to(RAW)).replace("\\", "/"), "size": path.stat().st_size}
        added += 1
        print(f"  saved [{lang}] {title[:70]}  ({len(text)} chars)")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for d in records.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n  pages saved: {added}")
    # read the new pages (only new / changed files are processed; nothing already read is removed)
    rc = subprocess.call([sys.executable, str(Path(__file__).resolve().parent / "02_extract_text.py"),
                          "--incremental"], cwd=str(ROOT))
    sys.exit(rc)


if __name__ == "__main__":
    main()
