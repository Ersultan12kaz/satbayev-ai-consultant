"""End-to-end RAG pipeline (all models are our own, trained in scripts/04-08).

Question -> small talk / follow-up handling -> language detection -> scope classifier ->
query expansion (lexicon) -> hybrid search (embedding + BM25) -> reranker ->
hallucination guard -> question type -> sentence ranker -> composed answer with sources.
"""
from __future__ import annotations

import math
import re

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .composer import Composer, SentenceRanker
from .embeddings import SubwordEncoder
from .facts import NOTE as FACT_NOTE, FactBase
from .followup import SMALLTALK_ANSWERS, resolve as resolve_followup, smalltalk
from .generator import MESSAGES, Answer, ExtractiveGenerator
from .langdetect import LanguageDetector, heuristic_language, kazakh_without_letters
from .lexicon import detect_concepts, query_stem_groups, support
from .qtype import QuestionTypeClassifier
from .reranker import Reranker
from .retriever import Retriever
from .scope import ScopeClassifier
from .text import fold, stems
from .wiki import UNI_RE, WikiFallback


PRICE_RE = re.compile(r"(канша\s+(тенге|тг|акша|турады|толе|акы)|(акы|акысы|толем|оплат\w*)\s+(канша|сколько)|сколько\s+(платить|оплат|денег)|тенге|\bтг\b|багас|баасы|стоимост|сколько\s+сто|"
                      r"\bцен[аыу]|расценк|how\s+much|price|\bcost|\bfee)")
MONEY_RE = re.compile(r"\d[\d\s.,]*(\([^)]{0,40}\))?\s*(тенге|теңге|тг\b|kzt|₸|мрп|аек|mci|%)", re.I)
PRICE_NOTE = {
    "kk": "Университеттің ресми құжаттарында нақты сомасы (теңгемен) табылмады - сомасын Студенттерге қызмет "
          "көрсету орталығынан немесе бухгалтериядан нақтылаңыз. Осы тақырып бойынша құжаттарда мынадай мәлімет бар:",
    "ru": "В официальных документах университета точная сумма (в тенге) не найдена - уточните её в Центре "
          "обслуживания студентов или в бухгалтерии. По этой теме в документах есть следующее:",
    "en": "The exact amount (in tenge) is not stated in the official university documents - please check it with "
          "the Student Service Center or the accounting office. On this topic the documents say:"}


# words that only ask to "tell / explain / give information about" - they are removed before the search
FILLER = set(fold("""айт айтшы айтып айтыңыз айтыныз айтасыз айтасын айтсаң айтсан айтсаңыз берші берши беріңіз
    бериниз беріңізші берiнiзшi бер туралы жайлы жайында ақпарат акпарат мәлімет малимет мағлұмат түсіндір тусиндир
    түсіндірші тусиндирши түсіндіріп толығырақ расскажи расскажите рассказать про информация информацию инфо
    объясни объясните поясни tell about info information explain me please""").split())


def strip_filler(question: str) -> str:
    words = question.split()
    kept = [w for w in words if fold(re.sub(r"[^\w-]", "", w)) not in FILLER]
    return " ".join(kept) if kept and len(kept) < len(words) else question


PHONE_Q_RE = re.compile(r"(телефон|\bтел\b|phone|номер(и|ы)? телефон|нөмір|номир|whatsapp|call)")
PHONE_RE = re.compile(r"(\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}|\b\d{3}[\s-]\d{2}[\s-]\d{2}\b")
CONTACT_NOTE = {
    "kk": "Бұл бөлімнің телефон нөмірі университет құжаттарында табылмады. Байланыс деректерін университет сайтының "
          "«Байланыс» бөлімінен немесе Студенттерге қызмет көрсету орталығынан алуға болады.",
    "ru": "Номер телефона этого подразделения в документах университета не найден. Контакты можно уточнить в разделе "
          "«Контакты» сайта университета или в Центре обслуживания студентов.",
    "en": "The phone number of this office is not stated in the university documents. Please see the Contacts page "
          "of the university website or ask the Student Service Center."}


FORM_TITLE_RE = re.compile(r"^\s*(Ф|F)\.?\s*(КазНИТУ|KazNRTU|КазНТУ)|^\s*(ҚазҰТЗУ|ҚазҰЗТУ|КазҰТЗУ)\s*\d{3}-\d{2}"
                           r"|бланк|журнал|келісім-шарт|договор|\bcontract\b|заявка|карточка", re.I)
FORM_Q_RE = re.compile(r"(форма|формасы|бланк|улги|образец|шаблон|template|\bform\b|журнал|договор|келисим)")


def is_form(chunk: Dict) -> bool:
    return chunk.get("category") == "regulatory/2-forms" or bool(FORM_TITLE_RE.search(chunk.get("title", "")))


# everyday topics that have nothing to do with the university (checked only when no university concept is found)
OFFTOPIC_RE = re.compile(
    r"(ауа райы|погод|weather|рецепт|recipe|пиццa|борщ|футбол|football|чемпионат мира|world cup|фильм|кино\b|movie|"
    r"сериал|песн|\bсонг|song|музык|music|курс доллар|курс валют|exchange rate|биткоин|bitcoin|анекдот|joke|"
    r"столица|capital of|астана қай|гороскоп|horoscope|\b\d+\s*[-+*/x×]\s*\d+\b|президент (сша|америк|росси)|"
    r"messi|месси|роналду|ronaldo|тик ?ток|tiktok|инстаграм|instagram)")


COUNT_Q_RE = re.compile(r"(\bканша\b|\bнеше\b|сколько|how many|how much)")
TRANSLATED_NOTE = {"kk": "(Құжат басқа тілде жазылған, жауап біздің аударма моделімен қазақшаға аударылды.)",
                   "ru": "(Документ написан на другом языке, ответ переведён нашей моделью перевода.)",
                   "en": "(The document is in another language; the answer was translated by our translation model.)"}
WHO_WAS_RE = re.compile(r"(ким болган|кім болған|кто так|кто был|who (was|is) [a-z]|биограф|омирбаян)")


def asks_price(question: str) -> bool:
    return bool(PRICE_RE.search(fold(question)))


class RAGPipeline:
    def __init__(self, retriever: Retriever, lang_model: Optional[LanguageDetector] = None,
                 scope_model: Optional[ScopeClassifier] = None, reranker: Optional[Reranker] = None,
                 not_found_threshold: float = 0.30, scope_threshold: float = 0.3, min_overlap: float = 0.5,
                 pool_size: int = 20, qtype_model: Optional[QuestionTypeClassifier] = None,
                 sentence_ranker: Optional[SentenceRanker] = None):
        self.retriever = retriever
        self.lang_model = lang_model
        self.scope_model = scope_model
        self.reranker = reranker or Reranker()
        self.not_found_threshold = not_found_threshold
        self.scope_threshold = scope_threshold
        self.min_overlap = min_overlap
        self.pool_size = pool_size
        # Wikipedia is searched only when our own data has no answer (free public search, no key); switched off
        # for the evaluation, which measures the models on the university documents only
        self.wiki = WikiFallback(encoder=getattr(retriever, "encoder", None), enabled=True)
        self.llm = None                          # our language model (set by load() when it is trained)
        self.translator = None                   # our translation model (set by load() when it is trained)
        self.qtype_model = qtype_model
        self.extractive = ExtractiveGenerator(encoder=retriever.encoder)
        self.composer = Composer(encoder=retriever.encoder, ranker=sentence_ranker, qtype_model=qtype_model)
        self.facts: Optional[FactBase] = None

    def translate_answer(self, text: str, lang: str) -> str:
        """Lines of the answer written in another language (the rule exists only in a Russian document) are
        translated by our translation model; numbers are checked (Translator.safe_translate)."""
        from .composer import OTHER_LANG_NOTE
        out, translated = [], False
        for line in text.split("\n"):
            m = re.match(r"^(\s*(?:\d+\.\s+|[^:\n]{1,40}:\s+)?)(.*?)(\s*\[\d+\])?\s*$", line)
            head, body, tail = (m.group(1) or ""), (m.group(2) or ""), (m.group(3) or "")
            if len(body) >= 15 and heuristic_language(body) not in (lang, None):
                tr = self.translator.safe_translate(body, lang)
                if tr:
                    line, translated = f"{head}{tr}{tail}", True
            out.append(line)
        text = "\n".join(out)
        if translated:
            for v in OTHER_LANG_NOTE.values():
                text = text.replace(v, "")
            text = text.rstrip() + "\n\n" + TRANSLATED_NOTE[lang]
        return text.strip()

    MIXED_WORD_RE = re.compile(r"[a-z]{2,}[а-яёәғқңөұүһі]{2,}|[а-яёәғқңөұүһі]{2,}[a-z]{2,}", re.I)

    def llm_ok(self, gen: str, question: str, lang: str, followup: bool = False) -> bool:
        """Extra checks of the language model's answer (added after the 28 Sep test): the answer must be in the
        language of the question, long enough, free of table debris / mixed-script words, and about the
        question (at least one content word of the question, or its kk<->ru dictionary equivalent)."""
        if heuristic_language(gen) != lang:
            return False                                    # e.g. a Russian rewrite for a Kazakh question
        words = gen.split()
        if len(words) < 8 or gen.count("|") >= 2 or gen.count("*") >= 2 or self.MIXED_WORD_RE.search(gen):
            return False
        if lang in ("kk", "ru") and re.search(r"\b[a-z]{10,}\b", gen):
            return False                                    # a broken English word inside a Kazakh/Russian answer
        low = [w.lower().strip(".,;:()\"«»") for w in words]
        grams = [" ".join(low[i:i + 4]) for i in range(len(low) - 3)]
        if len(grams) != len(set(grams)):
            return False                                    # the model repeats itself
        if sum(bool(re.fullmatch(r"[\d.]+", w)) for w in low) > max(3, len(low) // 8):
            return False                                    # a table of contents / list of numbers
        if followup:
            return True                                     # «толығырақ айтшы» has no topic words of its own
        q = set(stems(question))
        if not q:
            return True
        bilex = getattr(self.llm, "bilex", {}) or {}
        allowed = set(q)
        for st in q:
            allowed.update(bilex.get(st, []))
        g = set(stems(gen))
        pref = {a[:4] for a in allowed if len(a) >= 4}
        return bool(g & allowed) or any(x[:4] in pref for x in g if len(x) >= 4)

    @classmethod
    def load(cls, models_dir: str | Path = "models", **kw) -> "RAGPipeline":
        d = Path(models_dir)
        encoder = SubwordEncoder.load(d / "embedder.pkl")
        retriever = Retriever.load(d / "index", encoder=encoder)
        lang = LanguageDetector.load(d / "langdetect.json") if (d / "langdetect.json").exists() else None
        scope = ScopeClassifier.load(d / "scope.pkl") if (d / "scope.pkl").exists() else None
        rer = Reranker.load(d / "reranker.json") if (d / "reranker.json").exists() else None
        qt = QuestionTypeClassifier.load(d / "qtype.pkl") if (d / "qtype.pkl").exists() else None
        facts = FactBase.load(d / "facts.json")
        sr = SentenceRanker.load(d / "sentence_ranker.json") if (d / "sentence_ranker.json").exists() else None
        if (d / "config.json").exists():
            import json
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            for k in ("not_found_threshold", "scope_threshold", "min_overlap"):
                if k in cfg and k not in kw:
                    kw[k] = cfg[k]
        p = cls(retriever, lang, scope, rer, qtype_model=qt, sentence_ranker=sr, **kw)
        p.facts = facts
        # our own translation model (models/nmt, steps 11-12): answers in the language of the question
        try:
            from .nmt.translate import Translator
            p.translator = Translator.load(d / "nmt")
        except Exception:
            p.translator = None
        # our own language model writes the final answer (models/lm/lm_rag.pt, steps LM-1..LM-5); optional
        try:
            from .lm.answerer import LMAnswerer
            p.llm = LMAnswerer.load(d)
        except Exception:
            p.llm = None
        return p

    def detect_language(self, text: str) -> str:
        lang = self.lang_model.predict(text) if self.lang_model else heuristic_language(text)
        if lang == "ru" and kazakh_without_letters(text):
            lang = "kk"                  # "калай", "керек", "откизиледи" typed without Kazakh letters
        return lang

    def answer(self, question: str, mode: str = "hybrid", top_k: int = 5, generator: str = "composer",
               lang: Optional[str] = None, use_reranker: bool = True, use_scope: bool = True,
               history: Optional[List[str]] = None, dialog: Optional[List] = None) -> Dict:
        """generator: 'composer' (question type + sentence ranker, default) | 'extractive' (baseline).
        history: previous questions of the same conversation (for follow-ups like "қазақша нұсқада бер")."""
        t0 = time.perf_counter()
        timings = {}
        message = (question or "").strip()
        intent = smalltalk(message)
        if intent:                   # "сәлем", "рахмет", "неге қате жауап бердің", "сен кімсің"
            f = fold(message)
            lang = lang or ("kk" if re.search(r"(рахмет|ракмет|салем|кайырлы|армысыз|сау бол|кош бол|кимсин|кимсиз)", f)
                            else "en" if re.search(r"[a-z]", f) and not re.search(r"[а-я]", f)
                            else "ru" if re.search(r"(спасибо|привет|здравств|добр|кто ты|кто вы)", f)
                            else self.detect_language(message))
            return {"text": SMALLTALK_ANSWERS[intent][lang], "lang": lang, "sources": [], "confidence": "high",
                    "confidence_score": 1.0, "status": "smalltalk", "generator": "rules", "used_sentences": [],
                    "question": message, "resolved_question": message, "followup": intent, "mode": mode,
                    "top_k": top_k, "concepts": [], "question_type": None, "scope_probability": 1.0,
                    "evidence_overlap": 0.0,
                    "timings": {"total_ms": round((time.perf_counter() - t0) * 1000, 2)}, "retrieved": []}
        fu = resolve_followup(message, list(history or []))
        question = fu["question"]
        lang = lang or fu["lang"] or self.detect_language(message)
        question = strip_filler(question)            # "стипендия туралы айтшы" -> "стипендия"
        lang_boost = 0.15 if fu["prefer_lang"] else 0.08     # "қазақша" -> prefer documents in Kazakh
        timings["language_ms"] = (time.perf_counter() - t0) * 1000

        # "Ректор кім?" - facts extracted from the signature blocks of the documents (ml/facts.py)
        # "ректор Гарварда кім" is about another organisation: our facts are only about Satbayev University
        other_org = [m.group(0) for m in re.finditer(r"\b[A-ZА-ЯӘҒҚҢӨҰҮҺІЁ][\w-]+", question.strip())
                     if m.start() > 0 and not re.match(r"(?i)(с[әа]т[бп]аев|satbayev|каз|қаз|kaz|su\b|university|"
                                                       r"университет|ректор|проректор|кто|кім|who)", m.group(0))]
        fact = self.facts.answer(question, lang) if self.facts and not other_org else None
        if fact:
            srcs = []
            for n, s in enumerate(fact["fact"]["sources"][:2], 1):
                srcs.append({"n": n, "doc_id": s["doc_id"], "title": s["title"], "page": s["page"], "url": s.get("url", ""),
                             "lang": s.get("lang", ""), "section": "", "chunk_id": None, "score": 1.0})
            text = fact["text"] if fact.get("list") else fact["text"] + " [1]\n\n" + FACT_NOTE[lang]
            return {"text": text, "lang": lang, "sources": srcs, "confidence": "high", "confidence_score": 1.0,
                    "status": "answered", "generator": "facts", "used_sentences": [], "question": message,
                    "resolved_question": question, "followup": fu["followup"], "mode": mode, "top_k": top_k,
                    "concepts": [], "question_type": None, "scope_probability": 1.0, "evidence_overlap": 1.0,
                    "timings": {"total_ms": round((time.perf_counter() - t0) * 1000, 2)}, "retrieved": []}

        t1 = time.perf_counter()
        scope_p = 1.0
        if use_scope and self.scope_model is not None:
            _, scope_p = self.scope_model.is_university_question(question)
        concepts = detect_concepts(question)
        if concepts:          # the question uses university vocabulary ("общага", "академка", "стипуха", ...)
            scope_p = max(scope_p, self.scope_threshold + 0.05)
        timings["scope_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        pool = self.retriever.search(question, mode=mode, k=max(self.pool_size, top_k), lang=lang,
                                     lang_boost=lang_boost)
        timings["retrieval_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        if use_reranker:
            pool = self.reranker.rerank(question, pool, lang=lang)
        else:
            for c in pool:
                c.rerank = c.hybrid
        top = pool[0].rerank if pool else 0.0
        # blank forms, journals and contracts ("Ф КазНИТУ 705-14. Журнал ...") rarely answer a student's question:
        # when a regulation is almost as relevant, it goes first (unless the student asks for a form)
        if pool and not FORM_Q_RE.search(fold(question)):
            regs = [c for c in pool if not is_form(c.chunk) and c.rerank >= 0.5 * top]
            if regs and is_form(pool[0].chunk):
                pool = regs + [c for c in pool if c not in regs]
        # answer in the student's language: when the same rule exists in a document written in the language of
        # the question and it is almost as relevant (>= 75 %), it goes first; otherwise the better document in
        # the other language is used and our translation model translates the answer
        if pool and pool[0].chunk.get("lang") != lang:
            extra = self.retriever.search(question, mode=mode, k=10, lang=lang, lang_boost=0.35)
            known = {c.idx for c in pool}
            extra = [c for c in extra if c.chunk.get("lang") == lang and c.idx not in known]
            if extra and use_reranker:
                extra = self.reranker.rerank(question, extra, lang=lang)
            same = sorted([c for c in pool + extra if c.chunk.get("lang") == lang and c.rerank >= 0.75 * top],
                          key=lambda c: -c.rerank)
            if same:
                pool = same[:2] + [c for c in pool if c not in same[:2]]
        cands = pool[:top_k]
        if fu["more"] and cands:
            # "толығырақ айтшы": the next fragments of the same documents continue the text
            from .retriever import Candidate
            chunks = self.retriever.chunks
            extra, have = [], {c.idx for c in cands}
            for c in cands[:2]:
                for j in (c.idx + 1, c.idx + 2, c.idx - 1):
                    if 0 <= j < len(chunks) and j not in have and chunks[j].get("doc_id") == c.chunk.get("doc_id"):
                        have.add(j)
                        extra.append(Candidate(idx=j, rerank=c.rerank * 0.9, hybrid=c.hybrid, chunk=chunks[j]))
            cands = cands[:2] + extra + cands[2:]
        timings["rerank_ms"] = (time.perf_counter() - t3) * 1000
        # lexical support: share of the question's content words found in the best chunks
        # (synonyms from ml/lexicon.py and small typos count as support)
        groups = query_stem_groups(question)
        overlap = max((support(groups, set(stems(f"{c.chunk['title']} {c.chunk.get('section', '')} {c.chunk['text']}")),
                               fuzzy=True) for c in cands[:3]), default=0.0)

        t4 = time.perf_counter()
        # off-topic unless the documents themselves strongly support the question
        strong_evidence = top >= 0.9 and overlap >= 0.6
        if use_scope and not concepts and OFFTOPIC_RE.search(fold(message)):
            scope_p = 0.0                               # weather, recipes, football, "2+2", capitals ...
        if use_scope and scope_p < self.scope_threshold and (not strong_evidence or scope_p == 0.0):
            ans = Answer(text=MESSAGES["out_of_scope"][lang], lang=lang, status="out_of_scope")
        elif not cands or overlap < self.min_overlap or \
                (use_reranker and self.reranker.trained and top < self.not_found_threshold *
                 (0.5 if concepts and overlap >= 0.66 else 1.0)):     # a known topic, fully supported: lower bar
            ans = Answer(text=MESSAGES["not_found"][lang], lang=lang, status="not_found",
                         confidence_score=round(top, 4))
            if cands and concepts and asks_price(question) and not (use_scope and scope_p < self.scope_threshold):
                # "стипендия канша теңге": the amount is not in the documents, but the rules on the topic are
                ans = self.composer.generate(question, lang, cands, more=fu["more"])
                ans.confidence = "low"
        elif generator == "extractive":
            ans = ExtractiveGenerator(encoder=self.retriever.encoder, max_sentences=8 if fu["more"] else 4,
                                      max_sources=4 if fu["more"] else 3).generate(question, lang, cands)
        else:
            ans = self.composer.generate(question, lang, cands, more=fu["more"])
        def relevant(a) -> bool:
            # the answer must mention what was asked ("қанша кафедра бар" -> not a sentence about alumni)
            if a.status != "answered" or not groups:
                return a.status == "answered"
            pool_st = set(stems(a.text))
            key = [g for g in groups if len(g) > 1]                  # words of a known topic (lexicon concepts)
            if key and support(key, pool_st, fuzzy=True) < 0.99:
                return False
            return support(groups, pool_st, fuzzy=True) >= (0.34 if key else 0.5)

        if generator != "extractive" and ans.generator != "facts" and cands and not relevant(ans) and \
                not (use_scope and scope_p < self.scope_threshold):
            # try again only with the fragments that contain every asked word (any language)
            full = [c for c in cands + pool[top_k:] if support(groups, set(stems(
                f"{c.chunk['title']} {c.chunk['text']}")), fuzzy=True) >= 0.99][:top_k]
            ans2 = self.composer.generate(question, lang, full, more=fu["more"]) if full else None
            ans = ans2 if ans2 is not None and relevant(ans2) else \
                Answer(text=MESSAGES["not_found"][lang], lang=lang, status="not_found",
                       confidence_score=ans.confidence_score)
        if ans.status == "answered" and PHONE_Q_RE.search(fold(question)) and not PHONE_RE.search(ans.text):
            # "ректорат телефон" - a found blank form with a "Телефон" field is not an answer
            ans = Answer(text=CONTACT_NOTE[lang], lang=lang, status="not_found", confidence_score=ans.confidence_score)
        if ans.status == "answered" and ans.generator != "facts" and COUNT_Q_RE.search(fold(question)) and \
                not asks_price(question) and not re.search(r"\d", re.sub(r"\[\d+\]", "", ans.text)):
            # "қанша кафедра бар?" - an answer without any number does not answer "how many"
            ans = Answer(text=MESSAGES["not_found"][lang], lang=lang, status="not_found",
                         confidence_score=ans.confidence_score)
        if ans.status == "answered" and asks_price(question) and not MONEY_RE.search(ans.text):
            # "how much does one credit cost?" - the found rules do not contain the amount itself: say so openly
            ans.text = f"{PRICE_NOTE[lang]}\n\n{ans.text}"
            ans.confidence = "low"
        if self.wiki.enabled and ans.status == "answered" and WHO_WAS_RE.search(fold(question)):
            # "Сәтбаев кім болған?" - a person's biography is better in the encyclopedia than in the regulations
            w = self.wiki.answer(question, lang)
            if w:
                ans = Answer(text=w["text"].split("\n\n", 1)[-1], lang=lang, sources=w["sources"],
                             confidence="medium", confidence_score=0.0, status="answered", generator="wikipedia")
        # the encyclopedia knows general facts, not the university's own rules: for a question about a
        # university procedure (dormitory, military department, scholarship ...) it is not asked
        # ... and only about the university itself or Kanysh Satbayev (not footballers, not other people)
        # (28 Sep, user's request) any in-scope question that the documents do not answer is looked up in
        # Wikipedia; WikiFallback itself keeps only articles about Satbayev University / K. Satbayev (UNI_RE)
        wiki_ok = True
        if ans.status == "not_found" and self.wiki.enabled and wiki_ok and \
                not (use_scope and scope_p < self.scope_threshold):
            w = self.wiki.answer(question, lang)
            if w:
                ans = Answer(text=w["text"], lang=lang, sources=w["sources"], confidence="low",
                             confidence_score=0.0, status="answered", generator="wikipedia")
        if self.translator is not None and ans.status == "answered" and ans.generator in ("composer", "facts"):
            ans.text = self.translate_answer(ans.text, lang)
        if self.llm is not None and ans.status == "answered" and ans.generator in ("composer", "extractive") \
                and generator != "extractive":
            # our language model rewrites the answer in its own words, in the language of the question;
            # it is accepted only if every fact of it is in the found sentences (LMAnswerer.faithful)
            context = [u["text"] for u in ans.used_sentences][:6] or \
                [c.chunk["text"][:600] for c in cands[:3]]
            gen = self.llm.answer(message if fu["followup"] else question, lang, context, dialog=dialog)
            if gen and self.translator is not None and heuristic_language(gen) != lang:
                # the model copied the document's language: our translation model puts it into the question's
                gen = self.translate_answer(gen, lang)
            if gen and self.llm_ok(gen, question, lang, followup=bool(fu["followup"])):
                ans.text, ans.generator = gen, "llm"
        timings["generation_ms"] = (time.perf_counter() - t4) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        out = asdict(ans)
        out.update(question=message, resolved_question=question, followup=fu["followup"], mode=mode, top_k=top_k,
                   question_type=self.composer.question_type(question) if self.qtype_model else None,
                   concepts=sorted(concepts), scope_probability=round(float(scope_p), 4),
                   evidence_overlap=round(float(overlap), 4),
                   timings={k: round(v, 2) for k, v in timings.items()},
                   retrieved=[{"chunk_id": c.chunk.get("chunk_id"), "doc_id": c.chunk["doc_id"],
                               "title": c.chunk["title"], "page": c.chunk["page"], "lang": c.chunk.get("lang"),
                               "dense": round(c.dense, 4), "bm25": round(c.bm25, 4), "hybrid": round(c.hybrid, 4),
                               "rerank": round(c.rerank, 4)} for c in cands])
        return out
