"""STEP LM-1 - Text corpus for our own language model (Kazakh + Russian + the university data).

A language model learns a language from a lot of text.  Sources (all public and free):
  * Kazakh Wikipedia - the full dump        https://dumps.wikimedia.org/kkwiki/latest/
  * Russian Wikipedia - the first part(s) of the dump (a few hundred MB, the full dump is too big)
  * our university data: every page of the documents and website pages (data/processed/pages.jsonl)
    - repeated several times so that the model knows the university vocabulary well.

Wikipedia articles are cleaned from wiki markup by our own parser (templates, tables, links, references).
Result: data/lm/corpus.txt (one paragraph per line, an empty line between documents) + data/lm/stats.json.
Downloads are resumable; already processed dumps are not processed again.

    python scripts/13_lm_corpus.py [--ru-parts 2] [--uni-repeat 3]
"""
from __future__ import annotations

import argparse
import bz2
import html
import json
import re
import shutil
import time
from pathlib import Path

import requests

from _common import DATA, PROCESSED, Tee, banner, save_json

LM = DATA / "lm"
RAW_LM = LM / "raw"
UA = {"User-Agent": "SatbayevRAG/1.0 (diploma project, Satbayev University; educational use)"}
KK_DUMP = "https://dumps.wikimedia.org/kkwiki/latest/kkwiki-latest-pages-articles.xml.bz2"
RU_INDEX = "https://dumps.wikimedia.org/ruwiki/latest/"


# ------------------------------------------------------------------ download
def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.with_suffix(dest.suffix + ".done")
    if done.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    ok = False
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:                       # already complete
                    ok = True
                    break
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + (have if r.status_code == 206 else 0)
                mode = "ab" if r.status_code == 206 else "wb"
                got = have if r.status_code == 206 else 0
                t0, last = time.time(), 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk); got += len(chunk)
                        if time.time() - last > 30:
                            last = time.time()
                            print(f"    {dest.name}: {got / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
            ok = True
            break
        except Exception as e:
            print(f"    download interrupted ({e.__class__.__name__}), retrying ...", flush=True)
            time.sleep(10 + 5 * attempt)
    if not ok:
        raise SystemExit(f"  could not download {url} - check the internet and run again (it continues)")
    shutil.move(str(part), str(dest))
    done.write_text("ok")
    print(f"  downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def ru_part_urls(n: int):
    """URLs of the first n parts of the Russian Wikipedia dump (pages-articlesK.xml-p...bz2)."""
    try:
        page = requests.get(RU_INDEX, headers=UA, timeout=60).text
    except Exception:
        return []
    names = sorted(set(re.findall(r'href="(ruwiki-latest-pages-articles(\d+)\.xml-p\d+p\d+\.bz2)"', page)),
                   key=lambda x: int(x[1]))
    return [RU_INDEX + name for name, _ in names[:n]]


# ------------------------------------------------------------------ wiki markup -> plain text
def strip_nested(text: str, open_s: str, close_s: str) -> str:
    """Remove {{...}} / {|...|} blocks with nesting."""
    out, depth, i, n = [], 0, 0, len(text)
    lo, lc = len(open_s), len(close_s)
    while i < n:
        if text.startswith(open_s, i):
            depth += 1; i += lo; continue
        if depth and text.startswith(close_s, i):
            depth -= 1; i += lc; continue
        if not depth:
            out.append(text[i])
        i += 1
    return "".join(out)


_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAGS = re.compile(r"<(gallery|math|timeline|score|syntaxhighlight|source|pre|code|imagemap)[^>]*>.*?</\1>", re.S | re.I)
_HTML = re.compile(r"<[^>]+>")
_FILE = re.compile(r"\[\[(?:File|Image|Файл|Изображение|Сурет|Category|Категория|Санат)[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]",
                   re.I)
_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_EXT = re.compile(r"\[https?://[^\s\]]+\s?([^\]]*)\]")
_HEAD = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_QUOTES = re.compile(r"'{2,}")
_SKIP_SECTIONS = re.compile(r"^(Сілтемелер|Дереккөздер|Әдебиет|Тағы қараңыз|Ескертпелер|Ссылки|Литература|Примечания|"
                            r"См\. также|Источники|References|External links|See also)$", re.I)


def clean_wiki(text: str) -> str:
    text = html.unescape(text)
    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _TAGS.sub("", text)
    text = strip_nested(text, "{{", "}}")
    text = strip_nested(text, "{|", "|}")
    text = _FILE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _EXT.sub(r"\1", text)
    text = _HTML.sub("", text)
    text = _QUOTES.sub("", text)
    paras, skip = [], False
    for line in text.split("\n"):
        line = line.strip()
        m = _HEAD.match(line)
        if m:
            skip = bool(_SKIP_SECTIONS.match(m.group(1)))
            continue
        if skip or not line or line.startswith(("*", "#", ":", ";", "|", "!", "__")):
            if line.startswith(("*", "#")) and not skip:            # list items are real text too
                item = line.lstrip("*#:; ").strip()
                if len(item) >= 40:
                    paras.append(item)
            continue
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\(\s*[,;]?\s*\)", "", line)
        if len(line) >= 40 and sum(ch.isalpha() for ch in line) > 0.6 * len(line):
            paras.append(line)
    return "\n".join(paras)


def wiki_articles(path: Path):
    """(title, cleaned text) of every main-namespace article of a .xml.bz2 dump (streamed)."""
    title, ns, buf, in_text, redirect = "", "", [], False, False
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_text:
                if "<title>" in line:
                    title = line.split("<title>", 1)[1].split("</title>", 1)[0]
                    ns, redirect = "", False
                elif "<ns>" in line:
                    ns = line.split("<ns>", 1)[1].split("</ns>", 1)[0]
                elif "<redirect" in line:
                    redirect = True
                elif "<text" in line:
                    after = line.split(">", 1)[1] if ">" in line else ""
                    if "</text>" in after:
                        buf = [after.split("</text>", 1)[0]]
                        if ns == "0" and not redirect:
                            yield title, clean_wiki("".join(buf))
                    elif line.rstrip().endswith("/>"):
                        continue
                    else:
                        in_text, buf = True, [after]
            else:
                if "</text>" in line:
                    buf.append(line.split("</text>", 1)[0])
                    in_text = False
                    if ns == "0" and not redirect:
                        yield title, clean_wiki("".join(buf))
                else:
                    buf.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru-parts", type=int, default=2, help="parts of the Russian Wikipedia dump")
    ap.add_argument("--uni-repeat", type=int, default=3, help="how many times the university text is repeated")
    args = ap.parse_args()
    Tee("13_lm_corpus")
    banner("STEP LM-1. Text corpus for our language model")
    free = shutil.disk_usage(str(DATA)).free / 1e9
    print(f"  free disk space: {free:.1f} GB")
    if free < 8:
        raise SystemExit("  not enough free disk space (need about 8 GB)")
    LM.mkdir(parents=True, exist_ok=True)
    stats = {}
    parts = []                                                    # (name, lang, path)
    parts.append(("kkwiki", "kk", download(KK_DUMP, RAW_LM / "kkwiki.xml.bz2")))
    for i, url in enumerate(ru_part_urls(args.ru_parts), 1):
        parts.append((f"ruwiki{i}", "ru", download(url, RAW_LM / f"ruwiki{i}.xml.bz2")))
    for name, lang, path in parts:
        out = LM / f"{name}.txt"
        if out.exists() and out.with_suffix(".done").exists():
            print(f"  already cleaned: {out.name}")
        else:
            n_art, n_chars, t0 = 0, 0, time.time()
            with open(out, "w", encoding="utf-8") as f:
                for title, text in wiki_articles(path):
                    if len(text) < 200:
                        continue
                    f.write(f"{title}\n{text}\n\n")
                    n_art += 1; n_chars += len(text)
                    if n_art % 20000 == 0:
                        print(f"    {name}: {n_art} articles, {n_chars / 1e6:.0f}M characters, "
                              f"{time.time() - t0:.0f}s", flush=True)
            out.with_suffix(".done").write_text("ok")
        stats[name] = {"lang": lang, "mb": round(out.stat().st_size / 1e6, 1)}
    # the university data (documents + website pages), several times
    uni = LM / "university.txt"
    n_pages = 0
    with open(uni, "w", encoding="utf-8") as f:
        for line in (PROCESSED / "pages.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            text = re.sub(r"[ \t]+", " ", p.get("text", "")).strip()
            if len(text) >= 200:
                f.write(f"{p.get('title', '')}\n{text}\n\n")
                n_pages += 1
    stats["university"] = {"pages": n_pages, "mb": round(uni.stat().st_size / 1e6, 1), "repeat": args.uni_repeat}
    # one corpus file
    corpus = LM / "corpus.txt"
    with open(corpus, "w", encoding="utf-8") as out:
        for name, _, _ in parts:
            with open(LM / f"{name}.txt", encoding="utf-8") as f:
                shutil.copyfileobj(f, out, 1 << 24)
        for _ in range(args.uni_repeat):
            with open(uni, encoding="utf-8") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    stats["corpus_mb"] = round(corpus.stat().st_size / 1e6, 1)
    save_json(LM / "stats.json", stats)
    banner("Summary")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
