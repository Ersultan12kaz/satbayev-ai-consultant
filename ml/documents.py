"""Document processing pipeline: text extraction -> cleaning -> chunking -> metadata."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .text import clean_page, remove_repeated_lines, split_sentences, tokenize


@dataclass
class Page:
    doc_id: str
    title: str
    lang: str
    category: str
    url: str
    page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    lang: str
    category: str
    url: str
    page: int
    section: str
    text: str
    n_words: int = 0
    extra: Dict = field(default_factory=dict)


# ------------------------------------------------------------------ extraction
def extract_pdf(path: Path) -> List[str]:
    """Return the text of every page. pdfplumber first (keeps words intact),
    pypdf as a fallback for files pdfplumber cannot open."""
    pages: List[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                try:
                    pages.append(p.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
                except Exception:
                    pages.append("")
    except Exception:
        pages = []
    if not pages or sum(len(p) for p in pages) < 50:
        try:
            from pypdf import PdfReader
            alt = [(pg.extract_text() or "") for pg in PdfReader(str(path)).pages]
            if sum(len(p) for p in alt) > sum(len(p) for p in pages):
                pages = alt
        except Exception:
            pass
    return pages


def extract_docx(path: Path, paragraphs_per_page: int = 25) -> List[str]:
    """DOCX has no fixed pages: we emulate pages by groups of paragraphs."""
    import docx  # python-docx
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                paras.append(" | ".join(dict.fromkeys(cells)))
    return ["\n".join(paras[i:i + paragraphs_per_page]) for i in range(0, len(paras), paragraphs_per_page)] or [""]


def extract_document(path: Path, lang: str = "ru", ocr: bool = True) -> List[str]:
    """Text of every page. Scanned PDFs are OCR-ed when Tesseract is installed."""
    suf = path.suffix.lower()
    if suf == ".pdf":
        pages = extract_pdf(path)
        cleaned = [clean_page(p) for p in pages]
        if is_scanned(cleaned) or is_garbage(cleaned, lang):
            pages = []                      # no usable text layer
            from .ocr import ocr_available, ocr_pdf
            if ocr and ocr_available():
                pages = ocr_pdf(path, lang=lang)
    elif suf == ".docx":
        pages = extract_docx(path)
    elif suf in (".txt", ".md"):
        pages = [path.read_text(encoding="utf-8", errors="ignore")]
    elif suf in (".doc", ".rtf", ".xls", ".xlsx"):
        from .convert import extract_office
        pages = extract_office(path, extract_docx)
    else:
        return []
    pages = [clean_page(p) for p in pages]
    return remove_repeated_lines(pages)


_NOISE_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі][0-9{}|\\/\[\]<>@#$%^&*_=+~]|"
                       r"[0-9{}|\\/\[\]<>@#$%^&*_=+~][A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")


_KK_FUNCTION_WORDS = {"туралы", "бойынша", "жэне", "және", "мен", "сэйкес", "сәйкес", "болып", "тиіс", "тиис", "жылгы",
                      "жылғы", "ушин", "үшін", "бекітілген", "бекітілген", "щазацстан", "казацстан"}


def broken_kazakh(text: str) -> bool:
    """Kazakh text written with a broken font encoding: the special letters were replaced by Russian ones
    ("Щазацстан Республикасыныц... жэне"), so the text contains Kazakh words but no Kazakh letters.
    Such pages are re-read with OCR."""
    from .text import kazakh_ratio
    words = re.findall(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+", text.lower())
    if len(words) < 40:
        return False
    kk = sum(w in _KK_FUNCTION_WORDS for w in words)
    return kk / len(words) >= 0.01 and kazakh_ratio(text) < 0.003


def is_garbage(pages: List[str], lang: str = "ru") -> bool:
    """Broken text layer: some scanned PDFs carry a bad hidden OCR layer where Cyrillic letters
    became Latin look-alikes ("HEKOMMEPqECKOE KA3AXCKI{fr").  Detected by (a) almost no Cyrillic
    in a Russian/Kazakh document or (b) many tokens that mix letters with digits/symbols."""
    from .text import cyrillic_ratio
    text = " ".join(pages)[:20000]
    if len(text) < 200:
        return False
    if lang in ("ru", "kk") and cyrillic_ratio(text) < 0.3:
        return True
    if broken_kazakh(text):
        return True
    toks = [w.strip('.,;:()"«»-') for w in text.split()]
    toks = [w for w in toks if len(w) >= 3]
    return bool(toks) and sum(1 for w in toks if _NOISE_RE.search(w)) / len(toks) > 0.15


def is_scanned(pages: List[str], min_chars_per_page: int = 80) -> bool:
    if not pages:
        return True
    good = sum(len(p) >= min_chars_per_page for p in pages)
    return good / len(pages) < 0.3


# ------------------------------------------------------------------ chunking
_HEADING_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][^\n]{3,120}|"      # 1.2 Общие положения
    r"(РАЗДЕЛ|ГЛАВА|БӨЛІМ|SECTION|CHAPTER)\s+[\dIVX]+[^\n]{0,120}|"
    r"[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][A-ZА-ЯЁӘҒҚҢӨҰҮҺІ\s,\-]{6,100})$"
)


def detect_heading(line: str) -> Optional[str]:
    s = line.strip()
    if 4 <= len(s) <= 130 and _HEADING_RE.match(s) and not s.endswith((",", ";")):
        return s.rstrip(".:")
    return None


def chunk_pages(pages: Iterable[Page], max_words: int = 160, overlap_words: int = 35,
                min_words: int = 25) -> Iterator[Chunk]:
    """Sentence-aware sliding window inside each page; the current section heading is kept as metadata."""
    section_by_doc: Dict[str, str] = {}
    counter: Dict[str, int] = {}
    for pg in pages:
        section = section_by_doc.get(pg.doc_id, "")
        units: List[tuple] = []  # (sentence, section)
        buf: List[str] = []
        lines = pg.text.split("\n")
        for li, line in enumerate(lines):
            h = detect_heading(line)
            nxt = lines[li + 1].strip() if li + 1 < len(lines) else ""
            if h and nxt[:1].islower():
                h = None        # "8.5 Для оформления ... обучающийся" continues on the next line: not a heading
            if h:
                if buf:
                    units.extend((s, section) for s in split_sentences(" ".join(buf)))
                    buf = []
                section = h
                units.append((h if h.endswith((".", "!", "?", ":")) else h + ".", section))
            else:
                buf.append(line)
        if buf:
            units.extend((s, section) for s in split_sentences(" ".join(buf)))
        section_by_doc[pg.doc_id] = section

        cur: List[tuple] = []
        cur_words = 0

        def flush(items: List[tuple]):
            text = " ".join(s for s, _ in items).strip()
            nw = len(tokenize(text))
            if nw < min_words:
                return None
            counter[pg.doc_id] = counter.get(pg.doc_id, 0) + 1
            sec = next((sc for _, sc in items if sc), "")
            return Chunk(chunk_id=f"{pg.doc_id}_{counter[pg.doc_id]:04d}", doc_id=pg.doc_id, title=pg.title,
                         lang=pg.lang, category=pg.category, url=pg.url, page=pg.page, section=sec[:150],
                         text=text, n_words=nw)

        for sent, sec in units:
            w = len(tokenize(sent))
            if cur and cur_words + w > max_words:
                ch = flush(cur)
                if ch:
                    yield ch
                # overlap: keep the tail sentences
                tail, tw = [], 0
                for s in reversed(cur):
                    tw += len(tokenize(s[0]))
                    if tw > overlap_words:
                        break
                    tail.insert(0, s)
                cur, cur_words = tail, sum(len(tokenize(s[0])) for s in tail)
            cur.append((sent, sec))
            cur_words += w
        if cur:
            ch = flush(cur)
            if ch:
                yield ch


# ------------------------------------------------------------------ io helpers
def write_jsonl(path: Path, rows: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
