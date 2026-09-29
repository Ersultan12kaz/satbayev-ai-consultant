"""Multilingual university lexicon: colloquial words and synonyms -> official terms.

Students rarely use the wording of the regulations: they write "общага",
"академка", "стипуха", "жатакхана" (without Kazakh letters) or "dorm", while the
documents say "общежитие", "академический отпуск", "шәкіртақы", "dormitory".
Each concept below lists *triggers* (word prefixes after letter folding, see
ml.text.fold; a trigger with several words needs all of them) and the official
terms in Russian, Kazakh and English that are added to the query
("query expansion") before retrieval.  The lexicon is also used by the
hallucination guard: a question word counts as supported by a document when the
document contains any official term of the same concept.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .text import content_tokens, fold, stems

# concept: (triggers, expansion terms)
CONCEPTS: Dict[str, Tuple[List[str], str]] = {
    "dormitory": (["общаг", "общеж", "обшеж", "общяг", "обшаг", "жатакхан", "жатахан", "жатакана", "жатаккан", "жатакан", "жатак хан", "dorm",
                   "hostel", "хостел", "заселен", "заселит", "поселен", "студгород", "студенческий дом"],
                  "общежитие заселение место в общежитии жатақхана жатақханаға орналастыру dormitory accommodation"),
    "academic_leave": (["академк", "академотп", "академ отпуск", "академическ отпуск", "академиялык демалыс",
                        "академ демалыс", "academic leave", "leave of absence"],
                       "академический отпуск академиялық демалыс academic leave"),
    "scholarship": (["стипух", "степух", "стипа", "стипен", "степен", "шакиртак", "шекиртак", "scholarship",
                     "stipend", "грант", "grant"],
                    "стипендия государственный грант шәкіртақы scholarship grant"),
    "retake": (["пересда", "пересдач", "пересдат", "кайта тапсыр", "кайта окы", "ретейк", "retake", "re-take",
                "повторн", "летн семестр", "жазгы семестр", "summer semester", "fx"],
               "повторная сдача пересдача экзамена повторное изучение дисциплины летний семестр "
               "қайта тапсыру жазғы семестр retake summer semester"),
    "summer": (["лето", "летом", "летн", "жазда", "жазгы", "жазгы сем", "summer"],
               "летний семестр летний период обучения дополнительный семестр оплата за летний семестр "
               "жазғы семестр жазғы оқу summer semester"),
    "lesson": (["сабак", "занят", "lesson", "classes"],
               "учебные занятия лекции практические занятия сабақ lessons classes"),
    "student_id": (["студенттик билет", "студенческ билет", "студбилет", "студак", "student id", "student card",
                    "id карт"],
                   "студенческий билет ID-карта студента дубликат студенческого билета студенттік билет student ID card"),
    "contacts": (["мекенжай", "адрес", "address", "телефон", "байланыс", "контакт", "contact", "phone", "email",
                  "почта", "кайда орналаск", "где находится", "where is"],
                 "адрес университета контакты телефон электронная почта мекенжайы байланыс телефоны "
                 "address contacts phone"),
    "medical": (["медпункт", "медицин", "дариг", "поликлин", "емхан", "ауру", "болеть", "заболел", "health",
                 "doctor", "clinic", "medical"],
                "медицинский пункт медицинское обслуживание студентов поликлиника медициналық пункт дәрігер "
                "medical center health"),
    "temp_registration": (["вакытша тирке", "уакытша тирке", "временн регистр", "прописк", "registration of stay",
                           "migration", "миграц"],
                          "временная регистрация иностранных студентов регистрация по месту пребывания "
                          "уақытша тіркеу миграциялық temporary registration"),
    "institutes": (["институт", "факультет", "кафедр", "institute", "faculty", "department", "school of", "мектеп"],
                   "институты университета список институтов кафедры институттар кафедралар institutes departments"),
    "grant_loss": (["грант сакта", "грант жогалт", "гранттан айыр", "гранттан айрыл", "грант айыр", "грантты сакта",
                    "грантты жогалт", "грантымды", "лишен грант", "лиша грант", "лишат грант", "потер грант",
                    "сохран грант", "сохранить грант", "lose grant", "lost grant", "keep grant", "keep my grant"],
                   "лишение образовательного гранта сохранение гранта вакантный грант GPA для сохранения гранта "
                   "білім беру грантынан айыру грантты сақтау бос грант loss of the educational grant"),
    "internship": (["практик", "практич", "internship", "intern", "стажир", "тажириб", "кәсіптік"],
                   "профессиональная практика база практики кәсіптік практика practice internship"),
    "transfer": (["перевод", "перевест", "перевож", "ауысу", "ауыс", "transfer"],
                 "перевод студентов перевод с курса на курс ауысу transfer"),
    "expulsion": (["отчисл", "исключ", "окудан шыгар", "окудан шык", "шыгарыл", "шыгарып жибер", "expel", "expuls",
                   "dismiss"],
                  "отчисление обучающихся оқудан шығару expulsion"),
    "reinstatement": (["восстанов", "кайта кабылд", "кайта кел", "кайтып кел", "кайта орал", "кайта тус", "reinstat",
                       "readmi", "re-admi", "come back"],
                      "восстановление в число обучающихся оқуға қайта қабылдау reinstatement"),
    "thesis": (["диплом", "дипломк", "thesis", "выпускн работ", "бiтiру жумыс", "битиру жумыс", "защит",
                "коргау", "defen"],
               "дипломная работа дипломный проект защита итоговая аттестация дипломдық жұмыс қорғау thesis defense"),
    "exam": (["экзам", "сессия", "сессии", "сесси", "емтихан", "exam", "аттестац", "rubezh", "рубеж", "midterm",
              "endterm", "финал"],
             "экзамен экзаменационная сессия промежуточная аттестация рубежный контроль емтихан "
             "аралық аттестаттау exam examination session"),
    "appeal": (["апелл", "аппел", "appeal", "шагым", "жалоб"],
               "апелляция апелляциялық комиссия appeal"),
    "gpa": (["gpa", "джипиа", "средн балл", "орташа балл", "рейтинг", "бага", "оценк", "grade"],
            "GPA средний балл успеваемость оценка балл бағалау grade point average"),
    "tuition": (["оплат", "плата", "платн", "стоимост", "акысы", "акы", "толем", "tuition", "fee",
                 "payment", "скидк", "жениллик", "рассроч", "болип толе", "болшектеп", "installment"],
                "оплата за обучение стоимость обучения скидка оқу ақысы төлем tuition fee payment"),
    "admission": (["поступ", "абитур", "прием", "приём", "талапкер", "кабылдау", "admission", "apply", "enrol",
                   "enroll", "ент", "unt"],
                  "прием поступление абитуриент ЕНТ талапкер қабылдау admission enrollment"),
    "attendance": (["пропуск", "пропуст", "прогул", "посещ", "келмеу", "келмесе", "келмей", "бармасам", "бармаса", "бармау",
                    "бармай", "катыспа", "босат", "absence", "absent", "attendance", "miss class", "skip class"],
                   "пропуски занятий посещаемость сабаққа қатысу attendance absence"),
    "schedule": (["расписан", "кесте", "schedule", "timetable", "сабак уакыт"],
                 "расписание занятий сабақ кестесі schedule timetable"),
    "military": (["военк", "военн кафедр", "аскери", "military"],
                 "военная кафедра әскери кафедра military department"),
    "library": (["библиот", "китапхан", "library"], "библиотека кітапхана library"),
    "certificate": (["справк", "аныктам", "certificate", "reference letter"],
                    "справка анықтама certificate"),
    "plagiarism": (["плагиат", "антиплаг", "plagiar", "коширм"], "плагиат антиплагиат plagiarism"),
    "mobility": (["мобильн", "обмен", "exchange", "mobility", "академиялык уткырл", "shetel", "шетел"],
                 "академическая мобильность академиялық ұтқырлық academic mobility exchange"),
    "advisor": (["эдвайз", "адвайз", "advis", "куратор", "curator", "тьютор", "tutor"],
                "эдвайзер академический консультант куратор advisor"),
    "registration": (["регистрац", "запис на", "элективн", "иуп", "жоспар", "курсқа жазыл", "курска жазыл",
                      "registration", "elective", "enrolment in course"],
                     "регистрация на дисциплины индивидуальный учебный план ИУП элективные дисциплины "
                     "жеке оқу жоспары registration elective"),
    "credits": (["кредит", "credit", "ects"], "кредит академический кредит ECTS credits"),
    "history": (["университет тарих", "университеттин тарих", "тарихы университет", "история университет",
                 "история сатбаев", "university history", "history of the university", "history of satbayev",
                 "университет кашан", "кашан курыл", "когда основан", "когда был основан", "год основания",
                 "when was the university founded", "founded", "курылган жыл", "кай жылы ашыл", "ашылган жыл",
                 "сатбаев ким", "сатпаев ким", "сатбаев кто", "сатпаев кто", "who is satbayev", "who was satbayev",
                 "каныш", "сатбаев тарих", "сатпаев тарих", "тарихы кандай",
                 "университет турал", "университети турал", "университет жайл", "университети жайл",
                 "об университете", "про университет", "about the university", "about satbayev",
                 "сатбаев университет", "сатпаев университет", "satbayev university"],
                "университеттің тарихы 1934 жылдан басталады құрылған жылы история университета основан в 1934 году "
                "Казахский горно-металлургический институт history of the university founded in 1934"),
    "documents": (["документ", "кужат", "document", "papers", "бумаг"],
                  "документы справка заявление копия құжаттар анықтама өтініш көшірме documents application"),
}

# human-readable topic names (used in the heading of a composed answer)
CONCEPT_LABELS: Dict[str, Dict[str, str]] = {
    "dormitory": {"kk": "Жатақхана", "ru": "Общежитие", "en": "Dormitory"},
    "academic_leave": {"kk": "Академиялық демалыс", "ru": "Академический отпуск", "en": "Academic leave"},
    "scholarship": {"kk": "Шәкіртақы", "ru": "Стипендия", "en": "Scholarship"},
    "retake": {"kk": "Қайта тапсыру", "ru": "Пересдача", "en": "Retake"},
    "internship": {"kk": "Практика", "ru": "Практика", "en": "Internship"},
    "transfer": {"kk": "Ауысу", "ru": "Перевод", "en": "Transfer"},
    "expulsion": {"kk": "Оқудан шығару", "ru": "Отчисление", "en": "Expulsion"},
    "reinstatement": {"kk": "Оқуға қайта қабылдау", "ru": "Восстановление", "en": "Reinstatement"},
    "thesis": {"kk": "Дипломдық жұмыс", "ru": "Дипломная работа", "en": "Thesis"},
    "exam": {"kk": "Емтихан", "ru": "Экзамен", "en": "Exam"},
    "appeal": {"kk": "Апелляция", "ru": "Апелляция", "en": "Appeal"},
    "gpa": {"kk": "GPA және бағалау", "ru": "GPA и оценки", "en": "GPA and grading"},
    "tuition": {"kk": "Оқу ақысы", "ru": "Оплата обучения", "en": "Tuition"},
    "admission": {"kk": "Қабылдау", "ru": "Поступление", "en": "Admission"},
    "attendance": {"kk": "Сабаққа қатысу", "ru": "Посещаемость", "en": "Attendance"},
    "schedule": {"kk": "Сабақ кестесі", "ru": "Расписание", "en": "Schedule"},
    "military": {"kk": "Әскери кафедра", "ru": "Военная кафедра", "en": "Military department"},
    "library": {"kk": "Кітапхана", "ru": "Библиотека", "en": "Library"},
    "certificate": {"kk": "Анықтама", "ru": "Справка", "en": "Certificate"},
    "plagiarism": {"kk": "Плагиат", "ru": "Плагиат", "en": "Plagiarism"},
    "mobility": {"kk": "Академиялық ұтқырлық", "ru": "Академическая мобильность", "en": "Academic mobility"},
    "advisor": {"kk": "Эдвайзер", "ru": "Эдвайзер", "en": "Advisor"},
    "registration": {"kk": "Пәндерге тіркелу", "ru": "Регистрация на дисциплины", "en": "Course registration"},
    "credits": {"kk": "Кредиттер", "ru": "Кредиты", "en": "Credits"},
    "summer": {"kk": "Жазғы семестр", "ru": "Летний семестр", "en": "Summer semester"},
    "lesson": {"kk": "Сабақтар", "ru": "Занятия", "en": "Classes"},
    "student_id": {"kk": "Студенттік билет", "ru": "Студенческий билет", "en": "Student ID"},
    "contacts": {"kk": "Байланыс", "ru": "Контакты", "en": "Contacts"},
    "medical": {"kk": "Медициналық қызмет", "ru": "Медицинское обслуживание", "en": "Medical service"},
    "temp_registration": {"kk": "Уақытша тіркеу", "ru": "Временная регистрация", "en": "Temporary registration"},
    "institutes": {"kk": "Институттар", "ru": "Институты", "en": "Institutes"},
    "grant_loss": {"kk": "Грантты сақтау / жоғалту", "ru": "Сохранение и лишение гранта", "en": "Keeping or losing a grant"},
    "history": {"kk": "Университет тарихы", "ru": "История университета", "en": "University history"},
}

# prepared (folded) triggers
_TRIGGERS: List[Tuple[str, List[str]]] = []
for _name, (_trigs, _exp) in CONCEPTS.items():
    for _t in _trigs:
        _TRIGGERS.append((_name, fold(_t).split()))

# key stems of each concept (after folding): a document "talks about" the concept when it contains one of them
CONCEPT_STEMS: Dict[str, Set[str]] = {k: set(v.split()) for k, v in {
    "dormitory": "обшежи жатакх dormit заселе hostel",
    "academic_leave": "отпуск демалы leave",
    "scholarship": "стипен шакирт schola грант grant",
    "retake": "пересд retake повтор летний жазгы summer",
    "summer": "летний летнем летнег летня летне лето летом жазгы summer",
    "grant_loss": "грант grant лишен айыру айыр сохран сактау вакант",
    "lesson": "сабак заняти занято lesson classe лекци",
    "student_id": "билет студен id карта дублик",
    "contacts": "адрес мекенж телефо байлан контак contac addres phone почта",
    "medical": "медици медпун дариге поликл емхан medica health",
    "temp_registration": "регист тирке миграц пребыв registr",
    "institutes": "инстит кафедр факуль instit depart",
    "internship": "практи касипт intern practi",
    "transfer": "перево ауысу transf",
    "expulsion": "отчисл окудан expuls",
    "reinstatement": "восста reinst",
    "thesis": "диплом thesis",
    "exam": "екзаме емтиха exam examin сессия",
    "appeal": "апелля appeal",
    "gpa": "gpa балл grade",
    "tuition": "оплата стоимо акысы толем tuitio",
    "admission": "поступ абитур талапк admiss прием",
    "attendance": "пропус посеша attend absenc",
    "schedule": "распис кестес schedu timeta",
    "military": "военна аскери milita",
    "library": "библио китапх librar",
    "certificate": "справк аныкта certif",
    "plagiarism": "плагиа антипл plagia",
    "mobility": "мобиль уткырл mobili exchan",
    "advisor": "едвайз adviso курато",
    "registration": "регист иуп електи electi",
    "credits": "кредит credit ects",
    "documents": "докуме кужат справк аныкта отиниш заявле копия кошир docume",
    "history": "тарихы истори histor 1934",
}.items()}


def detect_concepts(text: str) -> Set[str]:
    """Concepts mentioned in the text (by prefix match of folded words)."""
    toks = [t for t in fold(text).replace("-", " ").split()]
    toks = ["".join(ch for ch in t if ch.isalnum()) for t in toks]
    toks = [t for t in toks if t]
    found = set()
    for name, words in _TRIGGERS:
        if all(any(_match(tok, w) for tok in toks) for w in words):
            found.add(name)
    return found


def _match(tok: str, w: str) -> bool:
    """Prefix match; long triggers also tolerate one wrong letter ("жатаққана" -> "жатакхан")."""
    if tok.startswith(w):
        return len(w) >= 4 or tok == w or len(tok) <= len(w) + 3
    if len(w) >= 8 and len(tok) >= len(w):
        return sum(a != b for a, b in zip(tok[:len(w)], w)) <= 1
    return False


_BILEX = None


def _bilex() -> dict:
    """Our kk<->ru stem dictionary (models/lm/bilex.json, learned from aligned sentences of the documents)."""
    global _BILEX
    if _BILEX is None:
        try:
            import json
            from pathlib import Path
            _BILEX = json.loads((Path(__file__).resolve().parents[1] / "models" / "lm" / "bilex.json")
                                .read_text(encoding="utf-8"))
        except Exception:
            _BILEX = {}
    return _BILEX


def bilingual_terms(text: str, per_word: int = 1) -> List[str]:
    """Mixed Kazakh-Russian questions ("академка алсам грант сохраняется ма"): every content word also gets
    its equivalent in the other language, so documents in both languages are found."""
    lex, out = _bilex(), []
    if not lex:
        return out
    have = set(stems(text))
    for st in list(have)[:10]:
        if len(st) < 4:
            continue
        for tr in lex.get(st, [])[:per_word]:
            if tr not in have and tr not in out:
                out.append(tr)
    return out


def expand_query(text: str) -> str:
    """Question + official terms of the concepts it mentions + kk<->ru equivalents of its words."""
    extra = [CONCEPTS[c][1] for c in sorted(detect_concepts(text))] + bilingual_terms(text)
    return text if not extra else f"{text} {' '.join(extra)}"


def concept_of_stem(stem: str) -> Set[str]:
    return {name for name, st in CONCEPT_STEMS.items() if stem in st}


def query_stem_groups(question: str) -> List[Set[str]]:
    """For every content word of the question: the set of stems that count as its support.

    A word that triggers a lexicon concept is supported by any official term of that concept.
    """
    groups = []
    whole = detect_concepts(question)
    # words of multi-word triggers found in the whole question ("кайта кел" -> "кайта" and "келу" both count)
    multi = [(name, words) for name, words in _TRIGGERS if name in whole and len(words) > 1]
    for tok in dict.fromkeys(content_tokens(question)):
        st = tok[:6] if len(tok) > 6 else tok
        alts = {st}
        for c in detect_concepts(tok):
            alts |= CONCEPT_STEMS[c]
        for name, words in multi:
            if any(tok.startswith(w) for w in words) and detect_concepts(" ".join(
                    t for t in fold(question).split() if any(t.startswith(w) for w in words))) >= {name}:
                alts |= CONCEPT_STEMS[name]
        groups.append(alts)
    return groups


def _trigrams(s: str) -> Set[str]:
    s = f"#{s}#"
    return {s[i:i + 3] for i in range(len(s) - 2)}


def similar_stem(a: str, pool: Set[str], thr: float = 0.6) -> bool:
    """Typo-tolerant match: character-trigram Jaccard between stems (only for words of 5+ letters)."""
    if len(a) < 5:
        return False
    ta = _trigrams(a)
    for b in pool:
        if len(b) >= 5 and abs(len(a) - len(b)) <= 2:
            tb = _trigrams(b)
            if len(ta & tb) / len(ta | tb) >= thr:
                return True
    return False


def support(groups: List[Set[str]], pool: Set[str], fuzzy: bool = False) -> float:
    """Share of question words (groups from query_stem_groups) supported by the stems in `pool`."""
    if not groups:
        return 0.0
    pre = {p[:4] for p in pool if len(p) >= 5} if fuzzy else set()
    ok = 0
    for g in groups:
        if g & pool:
            ok += 1
        elif fuzzy and len(g) == 1:
            a = next(iter(g))
            # another word form ("гостей" / "гости", "длится" / "длительность") or a small typo
            if (len(a) >= 5 and a[:4] in pre) or similar_stem(a, pool):
                ok += 1
    return ok / len(groups)
