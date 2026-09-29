"""Fallback source: Wikipedia (kk / ru / en), searched only when our own data has no answer.

Order of sources in the assistant:
  1. the official documents and the university websites (our index, our models);
  2. Wikipedia - free, no key, public MediaWiki search (https://<lang>.wikipedia.org/w/api.php);
  3. only then "not found".

The Wikipedia text is not answered by any external model: we download the article text and OUR models pick
the sentences (the same sub-word embedding model + lexical support as for the documents).
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

import numpy as np

from .text import content_tokens, fold, split_sentences, stems

UA = {"User-Agent": "SatbayevRAG/1.0 (diploma project, Satbayev University; educational use)"}
LANG_ORDER = {"kk": ["kk", "ru", "en"], "ru": ["ru", "kk", "en"], "en": ["en", "ru", "kk"]}
NOTE = {"kk": "Университет құжаттарында табылмады, Википедия бойынша:",
        "ru": "В документах университета не найдено, по данным Википедии:",
        "en": "Not found in the university documents; according to Wikipedia:"}


DISAMBIG_RE = re.compile(r"(мағынаны білдіреді|мағыналары|может означать|может относиться|may refer to|"
                         r"значения:|Елді мекендер:)", re.I)


PERSON_RE = re.compile(r"(\(.{0,80}?\d{4}.{0,60}?[–—-].{0,60}?\d{4}|\bтуған\b|дүниеге келген|"
                       r"\bродил(ся|ась)\b|\bborn\b)", re.I)


UNI_RE = re.compile(r"(с[әа]тбаев|сатпаев|satbayev|satpayev|казнту|казниту|қазұтзу|kazntu|kaznrtu|"
                    r"политехнически|политехникалық|горно-металлургическ|тау-кен металлургия)", re.I)


class WikiFallback:
    def __init__(self, encoder=None, timeout: float = 6.0, enabled: bool = True):
        self.encoder = encoder
        self.timeout = timeout
        self.enabled = enabled
        self._cache: Dict[str, Optional[Dict]] = {}
        self._session = None

    # ------------------------------------------------------------ http
    def _get(self, lang: str, params: Dict) -> Optional[Dict]:
        try:
            import requests
            if self._session is None:
                self._session = requests.Session(); self._session.headers.update(UA)
            r = self._session.get(f"https://{lang}.wikipedia.org/w/api.php",
                                  params={**params, "format": "json", "formatversion": 2}, timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def query_words(question: str) -> str:
        """The question without question words ("кім", "қалай", "кто", "what is" ...), original spelling kept."""
        keep = set(content_tokens(question))
        words = [w for w in re.findall(r"[\w-]+", question) if fold(w.lower()) in keep or w[:1].isupper()]
        return " ".join(words) or question

    def _articles(self, question: str, lang: str, n: int = 4) -> List[Dict]:
        q = self.query_words(question)
        res = self._get(lang, {"action": "query", "list": "search", "srsearch": q, "srlimit": n})
        hits = (res or {}).get("query", {}).get("search", [])
        out = []
        for h in hits:
            page = self._get(lang, {"action": "query", "prop": "extracts|pageprops", "explaintext": 1,
                                    "exsectionformat": "plain", "titles": h["title"], "redirects": 1})
            pages = (page or {}).get("query", {}).get("pages", [])
            if pages and "disambiguation" in (pages[0].get("pageprops") or {}):
                continue                                             # "Сәтбаев (айрық)" - a list of meanings
            if pages and pages[0].get("extract") and not DISAMBIG_RE.search(pages[0]["extract"][:400]):
                out.append({"title": pages[0]["title"], "text": pages[0]["extract"], "lang": lang,
                            "url": f"https://{lang}.wikipedia.org/wiki/{pages[0]['title'].replace(' ', '_')}"})
        return out

    # ------------------------------------------------------------ answer
    def answer(self, question: str, lang: str, min_support: float = 0.5) -> Optional[Dict]:
        """{"text", "sources", "lang"} built from the best Wikipedia sentences, or None."""
        if not self.enabled:
            return None
        key = f"{lang}|{question.strip().lower()}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.time()
        qstems = set(stems(question))
        person = bool(re.search(r"(ким болган|кім болған|кто так|кто был|who (was|is)|биограф|омирбаян)", fold(question)))
        best = None
        for wl in LANG_ORDER.get(lang, ["ru", "kk", "en"]):
            if time.time() - t0 > 3 * self.timeout:
                break
            for art in self._articles(question, wl):
                sents = [s.strip() for s in split_sentences(art["text"][:20000]) if 25 <= len(s.strip()) <= 400]
                if not sents:
                    continue
                art_stems = set(stems(art["title"] + " " + art["text"][:20000]))
                sup = len(qstems & art_stems) / max(1, len(qstems))
                if sup < min_support:
                    continue
                if not UNI_RE.search(art["title"] + " " + art["text"][:3000]):
                    continue                                         # only articles about our university / Satbayev
                if person and not PERSON_RE.search(art["text"][:300]):
                    continue                                         # "кім болған" -> a biography, not a town
                scores = self._rank(question, sents)
                order = np.argsort(-scores)[:3]
                chosen = sorted(order.tolist())                      # keep the article's order
                if person and 0 not in chosen:                       # a biography starts with who the person is
                    chosen = [0] + chosen[:2]
                cand = {"score": float(scores[order[0]]) + sup, "art": art, "sents": [sents[i] for i in chosen]}
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best is not None and best["art"]["lang"] == lang:
                break                                                # an article in the student's language
        result = None
        if best is not None:
            art = best["art"]
            text = f"{NOTE.get(lang, NOTE['ru'])}\n\n" + " ".join(best["sents"])
            result = {"text": text, "lang": lang,
                      "sources": [{"n": 1, "doc_id": "wikipedia", "title": f"Wikipedia: {art['title']}",
                                   "page": 1, "url": art["url"], "lang": art["lang"]}]}
        self._cache[key] = result
        return result

    def _rank(self, question: str, sents: List[str]) -> np.ndarray:
        qs = set(stems(question))
        lex = np.array([len(qs & set(stems(s))) / max(1, len(qs)) for s in sents], dtype=float)
        if self.encoder is None:
            return lex
        try:
            v = self.encoder.encode([question] + sents)
            dense = v[1:] @ v[0]
            return 0.6 * dense + 0.4 * lex - 0.002 * np.arange(len(sents))   # slight preference for the intro
        except Exception:
            return lex
