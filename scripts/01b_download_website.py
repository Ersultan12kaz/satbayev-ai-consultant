"""STEP 1b - Pages of the university websites (general information that is not in the regulations).

The regulations do not describe the university itself: history, rector and management,
institutes and departments, admission, student life, contacts, campus, museum, alumni...
This step crawls the public pages of

    https://satbayev.university/{kk,ru,en}/...            (about, history, admission, student life, news)
    https://official.satbayev.university/{kk,ru,en}/...   (institutes, departments, structure)

keeps the main text of every page (menus, headers and footers are removed) and adds each page
to data/raw/manifest.jsonl as a document of category "website" (saved as data/raw/web/<lang>/*.txt).
Only same-site HTML pages are visited; robots.txt is respected; the crawl is resumable.

    python scripts/01b_download_website.py [--max-pages 2500] [--max-news 150]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse as up
import urllib.robotparser
from collections import deque
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

from _common import RAW, Tee, banner

SITES = ["https://satbayev.university", "https://official.satbayev.university"]
LANGS = ("kk", "ru", "en")
HEADERS = {"User-Agent": "Mozilla/5.0 (SatbayevRAG diploma project; educational use)"}
SKIP_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|rtf|zip|rar|jpe?g|png|gif|svg|webp|mp4|mp3|avi|ico|css|js)$", re.I)
SKIP_PATH = re.compile(r"/(download|storage|uploads|files|login|logout|register|search|print|rss|feed|cabinet|"
                       r"docs|regulatory|2-forms|2-dp|2-labs|2-nilas)(/|$)", re.I)   # document lists: step 1
PRIORITY = re.compile(r"(about|history|istori|tarih|rector|leadership|rukovod|management|structure|institut|faculty|"
                      r"kafedr|department|admission|abiturient|priem|qabyldau|bachelor|master|doctor|phd|program|"
                      r"student|dorm|obshch|zhataq|campus|library|museum|sport|career|alumni|international|"
                      r"exchange|mobility|grant|scholar|tuition|oplat|contact|kontakt|faq|science|military|"
                      r"medic|polytech|club|volunteer|stoimost|price|cost|fee|summer|letn|zhazgy|zhataqhana|payment)", re.I)
NEWS = re.compile(r"/(news|novosti|zhanalyq|events|sobytiya|announcement)", re.I)


def fetch(session, url):
    for attempt in range(3):
        try:
            try:
                r = session.get(url, timeout=40)
            except requests.exceptions.SSLError:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                r = session.get(url, timeout=40, verify=False)
            return r
        except Exception:
            time.sleep(2 + 3 * attempt)                  # unstable internet / DNS: wait and retry
    return None


def main_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").split("—")[0].split("|")[0].strip()
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe",
                     "button", "select"]):
        tag.decompose()
    for tag in soup.select('[class*="menu"], [class*="nav"], [class*="footer"], [class*="header"], '
                           '[class*="breadcrumb"], [class*="cookie"], [class*="social"], [id*="menu"]'):
        tag.decompose()
    root = soup.find("main") or soup.find("article")
    if root is None:                                      # the block with the most paragraph text
        blocks = soup.find_all(["div", "section"])
        root = max(blocks, key=lambda b: sum(len(p.get_text()) for p in b.find_all(["p", "li"], recursive=True)),
                   default=soup.body or soup)
    h1 = root.find("h1") or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    lines = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote"]):
        if el.find(["p", "li", "td"]):                   # avoid repeating nested text
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) >= 3:
            lines.append(t)
    text = "\n".join(dict.fromkeys(lines))               # drop repeated lines
    return title, text


def lang_of(url):
    parts = up.urlparse(url).path.strip("/").split("/")
    return parts[0] if parts and parts[0] in LANGS else None


def normalize(url):
    u = up.urlparse(url)
    return up.urlunparse((u.scheme, u.netloc.lower(), re.sub(r"/+$", "", u.path) or "/", "", "", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=2500)
    ap.add_argument("--max-news", type=int, default=150, help="news pages per language")
    ap.add_argument("--restart", action="store_true", help="ignore the saved crawl state and start from the beginning")
    args = ap.parse_args()
    Tee("01b_website")
    banner("STEP 1b. Pages of the university websites")
    out_dir = RAW / "web"
    manifest_path = RAW / "manifest.jsonl"
    records = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); records[d["doc_id"]] = d
    session = requests.Session(); session.headers.update(HEADERS)
    robots = {}
    for site in SITES:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = fetch(session, site + "/robots.txt")
            rp.parse(r.text.splitlines() if r is not None and r.status_code == 200 else [])
        except Exception:
            rp.parse([])
        robots[up.urlparse(site).netloc] = rp
    hosts = set(robots)
    queue = deque(normalize(f"{s}/{l}") for s in SITES for l in LANGS)
    seen = set(queue)
    news_count = {l: 0 for l in LANGS}
    texts_seen = set()
    stats = {"visited": 0, "saved": 0, "cached": 0, "skipped": 0, "errors": 0}
    # resume: an interrupted crawl continues from its saved queue instead of starting from the first page again
    state_path = out_dir / "crawl_state.json"
    if state_path.exists() and not args.restart:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        queue, seen = deque(st["queue"]), set(st["seen"])
        news_count.update(st.get("news_count", {}))
        texts_seen = set(st.get("texts_seen", []))
        stats["visited"] = st.get("visited", 0)
        print(f"  resuming: {stats['visited']} pages visited earlier, {len(queue)} in the queue", flush=True)

    def save_state():
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"queue": list(queue), "seen": sorted(seen), "news_count": news_count,
                                          "texts_seen": sorted(texts_seen), "visited": stats["visited"]},
                                         ensure_ascii=False), encoding="utf-8")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for d in records.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    while queue and stats["visited"] < args.max_pages:
        url = queue.popleft()
        lang = lang_of(url)
        if lang is None:
            continue
        host = up.urlparse(url).netloc
        if not robots[host].can_fetch(HEADERS["User-Agent"], url):
            stats["skipped"] += 1; continue
        is_news = bool(NEWS.search(url)) and up.urlparse(url).path.count("/") >= 3
        if is_news:
            if news_count[lang] >= args.max_news:
                continue
            news_count[lang] += 1
        stats["visited"] += 1
        r = fetch(session, url)
        if r is None or r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            stats["errors"] += 1; continue
        # links first (priority pages go to the front of the queue)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = normalize(up.urljoin(url, a["href"].split("#")[0]))
            p = up.urlparse(link)
            if p.netloc not in hosts or link in seen or SKIP_EXT.search(p.path) or SKIP_PATH.search(p.path):
                continue
            if lang_of(link) is None:
                continue
            seen.add(link)
            (queue.appendleft if PRIORITY.search(p.path) else queue.append)(link)
        title, text = main_text(r.text)
        if len(text) < 300:
            continue
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in texts_seen:
            continue
        texts_seen.add(h)
        doc_id = "web_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        path = out_dir / lang / f"{doc_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = text if text.startswith(title) else f"{title}\n{text}"      # the h1 is often the title itself
        path.write_text(body, encoding="utf-8")
        records[doc_id] = {"doc_id": doc_id, "doc_num": doc_id, "lang": lang, "category": "website",
                           "title": title or url, "file_name": path.name, "url": url,
                           "path": str(path.relative_to(RAW)).replace("\\", "/"), "size": path.stat().st_size}
        stats["saved"] += 1
        if stats["saved"] % 50 == 0:
            print(f"  pages saved: {stats['saved']} (visited {stats['visited']}, queue {len(queue)})", flush=True)
        if stats["visited"] % 20 == 0:
            save_state()
        time.sleep(0.1)
    save_state()
    banner("Summary")
    for k, v in stats.items():
        print(f"  {k:10s}: {v}")
    print(f"  website pages in manifest: {sum(1 for d in records.values() if d.get('category') == 'website')}")


if __name__ == "__main__":
    main()
