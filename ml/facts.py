"""University facts extracted from the documents (our own rule-based information extraction).

Some questions ("Ректор кім?", "Кто проректор по академическим вопросам?") are answered in the
documents only in signature blocks such as

    Председатель Правления - Ректор ____________ М. Бегентаев

which are not normal sentences, so retrieval + sentence selection cannot answer them.  Here we
extract (role -> person) pairs from all pages with patterns in Russian, Kazakh and English, vote over
all occurrences (the most frequent name wins) and keep the documents where the fact was found.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .text import fold, stems

_UP = "A-ZА-ЯЁӘҒҚҢӨҰҮҺІ"
_LO = "a-zа-яёәғқңөұүһі"
NAME = (rf"(?:[{_UP}]\.\s?){{1,2}}\s?[{_UP}][{_LO}]{{2,}}(?:-[{_UP}][{_LO}]+)?"          # М. Бегентаев / М.М. Бегентаев
        rf"|[{_UP}][{_LO}]{{2,}}\s(?:[{_UP}]\.\s?){{1,2}}")                              # Бегентаев М.М.
GAP = r"[\s_,:\-–—\.\(\)подписьқолы]{0,60}?"

ROLE_PATTERNS = [
    ("rector", re.compile(rf"(?:Председатель\s+Правления\s*[-–—]?\s*[Рр]ектор|Басқарма\s+[Тт]өрағасы\s*[-–—]?\s*"
                          rf"[Рр]ектор\w*|Chairman\s+of\s+the\s+(?:Management\s+)?Board\s*[-–—]\s*Rector)"
                          rf"{GAP}({NAME})")),
    ("prorector", re.compile(rf"((?:Первый\s+)?[Пп]роректор\s+по\s+[а-яё ,]{{5,70}}?|"
                             rf"[А-ЯӘҒҚҢӨҰҮҺІа-яәғқңөұүһі][а-яәғқңөұүһі ,]{{4,69}}?жөніндегі\s+(?:бірінші\s+)?[Пп]роректор\w*|"
                             rf"(?:First\s+)?Vice[- ]Rector\s+for\s+[a-z ,]{{5,60}}?)"
                             rf"{GAP}({NAME})")),
]

ROLE_WORDS = {
    "rector": re.compile(r"(?<!про)(?<!pro)\b(ректор(?!ат)\w*|rector)\b", re.I),     # not "ректорат"
    "prorector": re.compile(r"(проректор\w*|vice[- ]rector|pro-?rector)", re.I),
}

TEMPLATES = {
    "rector": {"kk": "Satbayev University-дің Басқарма Төрағасы – Ректоры: {name}.",
               "ru": "Председатель Правления – Ректор Satbayev University: {name}.",
               "en": "Chairman of the Management Board – Rector of Satbayev University: {name}."},
    "prorector": {"kk": "{role}: {name}.", "ru": "{role}: {name}.", "en": "{role}: {name}."},
}
NOTE = {"kk": "(Құжаттардағы қолтаңба блоктары бойынша анықталды.)",
        "ru": "(Определено по блокам подписей в официальных документах.)",
        "en": "(Found in the signature blocks of the official documents.)"}


def _clean_name(n: str) -> str:
    n = re.sub(r"\s+", " ", n.replace(". ", ".").strip())
    n = re.sub(rf"^((?:[{_UP}]\.){{1,2}})(?=[{_UP}])", r"\1 ", n)          # "М.Бегентаев" -> "М. Бегентаев"
    return n


def _clean_role(r: str) -> str:
    r = re.sub(r"\s+", " ", r).strip(" ,-–—")
    return r[:1].upper() + r[1:]


def extract_facts(pages: Iterable[Dict]) -> Dict:
    votes: Dict[str, Counter] = defaultdict(Counter)
    where: Dict[tuple, List[Dict]] = defaultdict(list)
    roles_text: Dict[str, Counter] = defaultdict(Counter)
    for p in pages:
        text = p["text"]
        for key, rx in ROLE_PATTERNS:
            for m in rx.finditer(text):
                if key == "rector":
                    role, name = "rector", _clean_name(m.group(1))
                else:
                    role_text = _clean_role(m.group(1))
                    role = "prorector:" + " ".join(sorted(set(stems(role_text)) - {"проректор", "прорек"}))[:80]
                    name = _clean_name(m.group(2))
                    roles_text[role][role_text] += 1
                if len(name) < 5 or re.search(r"(?i)правлен|ректор|фамилия|подпись|аты-жөні|surname|name", name):
                    continue                                        # empty form fields "Фамилия И.О."
                votes[role][name] += 1
                if len(where[(role, name)]) < 5:
                    where[(role, name)].append({"doc_id": p["doc_id"], "title": p.get("title", ""), "page": p["page"],
                                                "url": p.get("url", ""), "lang": p.get("lang", "")})
    facts = {}
    for role, cnt in votes.items():
        name, n = cnt.most_common(1)[0]
        if n < 2 and role != "rector":
            continue
        facts[role] = {"name": name, "count": n, "alternatives": cnt.most_common(4)[1:],
                       "role_text": roles_text[role].most_common(1)[0][0] if roles_text.get(role) else role,
                       "sources": where[(role, name)]}
    return facts


# ------------------------------------------------------------------ lists from the university websites
LIST_TEXT = {
    "institutes": {"kk": "Satbayev University құрамындағы институттар мен академия ({n}):",
                   "ru": "Институты и академия Satbayev University ({n}):",
                   "en": "Institutes and the academy of Satbayev University ({n}):"},
    "departments": {"kk": "Университет сайтында көрсетілген кафедралар ({n}):",
                    "ru": "Кафедры, указанные на сайте университета ({n}):",
                    "en": "Departments listed on the university website ({n}):"},
}
_DEP_TITLE = {"kk": re.compile(r"кафедрасы\s*$", re.I), "ru": re.compile(r"^кафедра\s", re.I),
              "en": re.compile(r"^department of|department\s*$", re.I)}
_DEP_SKIP = re.compile(r"(ғылыми жоба|түлек|практика|тәжірибе|научн|выпуск|практик|project|graduat|practice|"
                       r"istoriya|history|obrazovatel|educational program|/subdivisions/|otdel)", re.I)


def extract_lists(records) -> Dict:
    """Institutes and departments of the university from the titles / addresses of the crawled website pages."""
    inst = {"kk": {}, "ru": {}, "en": {}}
    deps = {"kk": {}, "ru": {}, "en": {}}
    for r in records:
        if r.get("category") != "website":
            continue
        lang, title, url = r.get("lang"), (r.get("title") or "").strip(), r.get("url", "")
        if lang not in inst or not title:
            continue
        if re.search(r"satbayev\.university/(kk|ru|en)/institutes/[^/]+$", url):
            inst[lang].setdefault(title, url)
        elif _DEP_TITLE[lang].search(title) and not _DEP_SKIP.search(title + " " + url) and url.count("/") >= 5:
            deps[lang].setdefault(re.sub(r"\s+", " ", title), url)
    return {"institutes": {l: sorted(v.items()) for l, v in inst.items()},
            "departments": {l: sorted(v.items()) for l, v in deps.items()}}


class FactBase:
    def __init__(self, facts: Dict):
        self.facts = facts

    @classmethod
    def load(cls, path) -> Optional["FactBase"]:
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.facts, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_answer(self, question: str, lang: str) -> Optional[Dict]:
        """"институттар тізімі", "қанша кафедра бар", "какие институты есть" -> the list from the website."""
        lists = self.facts.get("_lists")
        if not lists:
            return None
        f = fold(question)
        asks = re.search(r"(тизим|список|перечень|\blist\b|канша|неше|сколько|how many|which|кандай|какие|бар ма|есть ли)", f)
        kind = "departments" if re.search(r"(кафедр|department|chair)", f) else \
            "institutes" if re.search(r"(институт|institute|факультет|faculty|академия)", f) else None
        plural = re.search(r"(кафедралар\w*|институттар\w*|кафедры|кафедрах|институты|институтах|departments|institutes)", f)
        info = re.search(r"(акпарат|туралы|жайлы|айт|керек|информац|расскаж|about|info|tell)", f)
        short = len(re.findall(r"\w+", f)) <= 2                     # "кафедра", "институттар?"
        if not kind or re.search(r"(военн|аскери|military)", f) or not (asks or (plural and info) or short):
            return None
        items = lists[kind].get(lang) or lists[kind].get("ru") or []
        if not items:
            return None
        lines = [LIST_TEXT[kind][lang].format(n=len(items))] + [f"{i}. {t}" for i, (t, _) in enumerate(items, 1)]
        return {"text": "\n".join(lines), "list": True, "fact": {"sources": [{"title": t, "url": u, "page": 1, "doc_id": "web",
                                                                "lang": lang} for t, u in items[:5]]}}

    def answer(self, question: str, lang: str) -> Optional[Dict]:
        """A fact answer for "who is the rector / pro-rector for ..." questions, else None."""
        lst = self.list_answer(question, lang)
        if lst:
            return lst
        f = fold(question)
        asks_who = re.search(r"\b(ким|кто|who|кимдер|есими|фио|аты|кім|кімдер|is)\b", f) or len(f.split()) <= 2
        if not asks_who:
            return None
        if ROLE_WORDS["rector"].search(question) and "rector" in self.facts:
            fact = self.facts["rector"]
            return {"text": TEMPLATES["rector"][lang].format(name=fact["name"].rstrip(".")), "fact": fact}
        if ROLE_WORDS["prorector"].search(question):
            qs = set(stems(question)) - {"проректор", "прорек", "vice", "rector"}
            pro = {k: v for k, v in self.facts.items() if k.startswith("prorector:")}
            if not pro:
                return None
            scored = sorted(pro.items(), key=lambda kv: -len(qs & set(kv[0].split(":", 1)[1].split())))
            best = scored[0]
            if qs & set(best[0].split(":", 1)[1].split()):
                items = [best]
            else:
                items = scored[:6]                          # "who are the pro-rectors?" -> all of them
            lines = [TEMPLATES["prorector"][lang].format(role=v["role_text"], name=v["name"].rstrip("."))
                     for _, v in items]
            return {"text": "\n".join(lines), "fact": items[0][1], "facts": [v for _, v in items]}
        return None
