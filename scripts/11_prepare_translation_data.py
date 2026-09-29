"""STEP 11 - Data for our own translation model (Kazakh <-> Russian <-> English).

1. Parallel sentences from the university's own documents (they are published in 3 languages):
   document pairing + sentence alignment (ml/nmt/align.py).
2. Open parallel corpus KazParC (ISSAI, kk/en/ru/tr, ~372k sentence groups) - DATA only, no model.
   Downloaded once into data/raw/kazparc/.  If it cannot be downloaded, only university data is used.
3. Our BPE tokenizer is trained on all sentences (joint vocabulary for kk/ru/en).
4. Everything is encoded into token ids: data/translation/*.npz

    python scripts/11_prepare_translation_data.py [--vocab 16000] [--no-kazparc]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict

import numpy as np

from _common import DATA, MODELS, PROCESSED, RAW, Tee, Timer, banner, save_json

from ml.langdetect import heuristic_language
from ml.nmt.align import parallel_from_documents
from ml.nmt.bpe import BPE, EOS, LANG_TOKEN

OUT = DATA / "translation"
NMT = MODELS / "nmt"
KAZPARC_DIR = RAW / "kazparc"
HF_API = "https://huggingface.co/api/datasets/issai/kazparc/tree/main?recursive=true"
HF_FILE = "https://huggingface.co/datasets/issai/kazparc/resolve/main/{path}"
LANGS = ("kk", "ru", "en")


def load_documents():
    meta = {d["doc_id"]: d for d in json.loads((PROCESSED / "documents.json").read_text(encoding="utf-8"))}
    pages = defaultdict(list)
    with open(PROCESSED / "pages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pages[p["doc_id"]].append(p["text"])
    docs = {}
    for d, ps in pages.items():
        text = "\n".join(ps)
        m = meta.get(d, {})
        docs[d] = {"lang": heuristic_language(text[:5000]), "category": m.get("category", ""),
                   "title": m.get("title", ""), "text": text}
    return docs


def download_kazparc() -> list:
    """-> list of paths of downloaded CSV files (raw file with kk/en/ru columns preferred)."""
    import requests
    KAZPARC_DIR.mkdir(parents=True, exist_ok=True)
    have = sorted(KAZPARC_DIR.glob("*.csv"))
    if have:
        print(f"  KazParC already downloaded: {[p.name for p in have]}")
        return have
    try:
        r = requests.get(HF_API, timeout=60)
        if r.status_code in (401, 403):
            print("  KazParC is gated on HuggingFace (login / accepting terms is needed) - skipped")
            return []
        r.raise_for_status()
        files = [x["path"] for x in r.json() if x.get("type") == "file" and x["path"].endswith(".csv")]
    except Exception as e:
        print(f"  KazParC file list not available ({e}) - only university data will be used")
        return []
    raw = [p for p in files if "01_" in p.split("/")[-1] or "raw" in p.lower()]
    if not raw:
        raw = [p for p in files if "train" in p.lower() and any(k in p for k in ("kk_ru", "kk_en", "en_ru"))]
    out = []
    for path in raw[:4]:
        dst = KAZPARC_DIR / path.split("/")[-1]
        print(f"  downloading {path} ...", flush=True)
        try:
            with requests.get(HF_FILE.format(path=path), stream=True, timeout=120) as resp:
                if resp.status_code in (401, 403):
                    print("  gated file - skipped"); continue
                resp.raise_for_status()
                tmp = dst.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dst)
            out.append(dst)
        except Exception as e:
            print(f"  failed: {e}")
    return out


OPUS_API = "https://opus.nlpl.eu/opusapi/?source={a}&target={b}&preprocessing=moses&version=latest"
OPUS_DIR = RAW / "opus"
OPUS_PREFER = ["WikiMatrix", "TED2020", "QED", "Tatoeba", "GlobalVoices", "NeuLab-TedTalks", "wikimedia",
               "CCMatrix", "NLLB", "OpenSubtitles"]


def download_opus(max_rows_per_pair: int = 300000, max_zip_mb: int = 700) -> list:
    """Open parallel corpora from OPUS (https://opus.nlpl.eu, free, no login): kk-ru, en-kk, en-ru.
    -> list of dicts {lang: sentence}.  Downloads are kept in data/raw/opus/ (done once)."""
    import io
    import zipfile
    import requests
    OPUS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in (("kk", "ru"), ("en", "kk"), ("en", "ru")):
        try:
            info = requests.get(OPUS_API.format(a=a, b=b), timeout=60).json().get("corpora", [])
        except Exception as e:
            print(f"  OPUS list {a}-{b} not available ({e})"); continue
        by = {c["corpus"]: c for c in info if c.get("url", "").endswith(".zip")}
        got = 0
        for name in OPUS_PREFER:
            if got >= max_rows_per_pair or name not in by:
                continue
            c = by[name]
            if c.get("size", 0) / 1024 > max_zip_mb:         # size is given in kB
                print(f"  {name} {a}-{b}: {c.get('size', 0) / 1024:.0f} MB - too big, skipped"); continue
            dst = OPUS_DIR / f"{name}.{a}-{b}.zip"
            if not dst.exists():
                print(f"  downloading OPUS {name} {a}-{b} ({c.get('size', 0) / 1024:.0f} MB) ...", flush=True)
                try:
                    with requests.get(c["url"], stream=True, timeout=120) as r:
                        r.raise_for_status()
                        tmp = dst.with_suffix(".part")
                        with open(tmp, "wb") as f:
                            for ch in r.iter_content(1 << 20):
                                f.write(ch)
                        tmp.replace(dst)
                except Exception as e:
                    print(f"  failed: {e}"); continue
            try:
                z = zipfile.ZipFile(dst)
                fa = next(n for n in z.namelist() if n.endswith(f".{a}"))
                fb = next(n for n in z.namelist() if n.endswith(f".{b}"))
                with z.open(fa) as xa, z.open(fb) as xb:
                    for la, lb in zip(io.TextIOWrapper(xa, encoding="utf-8", errors="replace"),
                                      io.TextIOWrapper(xb, encoding="utf-8", errors="replace")):
                        la, lb = la.strip(), lb.strip()
                        na, nb = len(la.split()), len(lb.split())
                        if 3 <= na <= 60 and 3 <= nb <= 60 and 0.5 <= (na + 1) / (nb + 1) <= 2.0:
                            rows.append({a: la, b: lb}); got += 1
                            if got >= max_rows_per_pair:
                                break
            except Exception as e:
                print(f"  {name} {a}-{b}: cannot read ({e})")
        print(f"  OPUS {a}-{b}: {got} sentence pairs")
    return rows


def read_kazparc(paths) -> list:
    """-> list of dicts {lang: sentence} (a row with the same sentence in several languages)."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    rows = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            for r in rd:
                if all(c in cols for c in LANGS):                              # raw file: kk, en, ru, tr
                    rows.append({l: (r[l] or "").strip() for l in LANGS if (r.get(l) or "").strip()})
                elif "source_lang" in cols:                                     # pair file
                    s, t = r.get("source_lang"), r.get("target_lang")
                    if s in LANGS and t in LANGS:
                        src_col = next((c for c in cols if c.startswith("source") and c != "source_lang"), None)
                        tgt_col = next((c for c in cols if c.startswith("target") and c != "target_lang"), None)
                        if src_col and tgt_col:
                            rows.append({s: r[src_col].strip(), t: r[tgt_col].strip()})
    return [r for r in rows if len(r) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--no-kazparc", action="store_true")
    ap.add_argument("--max-kazparc-rows", type=int, default=400000)
    ap.add_argument("--no-opus", action="store_true")
    ap.add_argument("--max-opus-rows", type=int, default=300000, help="per language pair")
    args = ap.parse_args()
    Tee("11_translation_data")
    rnd = random.Random(7)
    OUT.mkdir(parents=True, exist_ok=True); NMT.mkdir(parents=True, exist_ok=True)

    banner("STEP 11a. Parallel sentences from the university documents (our aligner)")
    with Timer() as t:
        uni = parallel_from_documents(load_documents())
    print(f"  aligned sentence pairs: {len(uni)} ({t.sec:.0f}s) "
          f"{dict(Counter((p['src_lang'], p['tgt_lang']) for p in uni))}")
    with open(OUT / "university_parallel.jsonl", "w", encoding="utf-8") as f:
        for p in uni:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    banner("STEP 11b. Open parallel corpus (KazParC, data only)")
    kp = [] if args.no_kazparc else read_kazparc(download_kazparc())
    rnd.shuffle(kp)
    kp = kp[: args.max_kazparc_rows]
    print(f"  KazParC rows used: {len(kp)}")
    if not args.no_opus:
        banner("STEP 11b'. Open parallel corpora from OPUS (data only, no login)")
        op = download_opus(args.max_opus_rows)
        rnd.shuffle(op)
        kp = kp + op
        print(f"  OPUS rows used: {len(op)}")

    # ---------------------------------------------------------------- examples (src_lang, tgt_lang, src, tgt)
    doc_pairs = sorted({(p["doc_a"], p["doc_b"]) for p in uni})
    rnd.shuffle(doc_pairs)
    val_docs = set(doc_pairs[: max(3, len(doc_pairs) // 15)])            # whole documents held out
    train, valid = [], []
    for p in uni:
        pairs = [(p["src_lang"], p["tgt_lang"], p["src"], p["tgt"]), (p["tgt_lang"], p["src_lang"], p["tgt"], p["src"])]
        if (p["doc_a"], p["doc_b"]) in val_docs:
            valid += [dict(s=a, t=b, src=x, tgt=y, dom="university") for a, b, x, y in pairs]
        else:
            for _ in range(3):                                            # in-domain data is up-weighted
                train += [(a, b, x, y) for a, b, x, y in pairs]
    n_val_kp = min(1500, len(kp) // 50)
    for i, row in enumerate(kp):
        langs = [l for l in LANGS if l in row]
        dirs = [(a, b) for a in langs for b in langs if a != b]
        if i < n_val_kp:
            a, b = rnd.choice(dirs)
            valid.append(dict(s=a, t=b, src=row[a], tgt=row[b], dom="kazparc"))
        else:
            for a, b in rnd.sample(dirs, min(2, len(dirs))):
                train.append((a, b, row[a], row[b]))
    rnd.shuffle(train)
    print(f"  training examples: {len(train)}   validation: {len(valid)}")
    print(f"  directions: {dict(Counter((a, b) for a, b, _, _ in train))}")

    banner("STEP 11c. Our BPE tokenizer (joint kk/ru/en vocabulary)")
    sample = [x for _, _, x, _ in train[:250000]] + [y for _, _, _, y in train[:250000]]
    with Timer() as t:
        bpe = BPE().train(sample, vocab_size=args.vocab)
    bpe.save(NMT / "bpe.json")
    print(f"  vocabulary: {len(bpe)} sub-words ({t.sec:.0f}s)")
    for s in ["Академиялық демалыс ректордың бұйрығымен беріледі.", "Студент подает заявление в деканат."]:
        print(f"  {s!r} -> {[bpe.inv[i] for i in bpe.encode(s)]}")

    banner("STEP 11d. Encoding")

    def encode(rows):
        src, tgt, lens = [], [], []
        for a, b, x, y in rows:
            s = [LANG_TOKEN[b]] + bpe.encode(x)[:200] + [EOS]
            tt = bpe.encode(y)[:200] + [EOS]
            if len(s) < 4 or len(tt) < 3:
                continue
            src.append(np.array(s, dtype=np.int32)); tgt.append(np.array(tt, dtype=np.int32))
        so = np.cumsum([0] + [len(x) for x in src]); to = np.cumsum([0] + [len(x) for x in tgt])
        return dict(src=np.concatenate(src), src_off=so, tgt=np.concatenate(tgt), tgt_off=to)

    with Timer() as t:
        np.savez(OUT / "train.npz", **encode(train))
    print(f"  train encoded ({t.sec:.0f}s)")
    with open(OUT / "valid.jsonl", "w", encoding="utf-8") as f:
        for v in valid:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    save_json(NMT / "data_stats.json", {"university_pairs": len(uni), "kazparc_rows": len(kp),
                                         "train_examples": len(train), "valid_examples": len(valid),
                                         "vocab": len(bpe)})
    print("  saved: data/translation/train.npz, valid.jsonl, models/nmt/bpe.json")


if __name__ == "__main__":
    main()
