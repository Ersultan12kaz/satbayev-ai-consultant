"""Dialogue follow-ups: "қазақша нұсқада бер", "на русском", "in English", "толығырақ", "а для магистрантов?".

A follow-up message has (almost) no content words of its own, so it is answered
using the previous question of the same conversation:
  * a language request re-answers the previous question in the requested language
    and prefers documents written in that language;
  * "more details" re-answers it with more sentences;
  * a short clarification ("а для магистрантов?") is appended to the previous question.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .lexicon import detect_concepts
from .text import content_tokens, fold

LANG_REQUEST = {
    "kk": [r"казакш", r"казак тил", r"на казахском", r"по-?казахски", r"казахск", r"\bkazakh", r"\bkaz\b", r"\bkz\b"],
    "ru": [r"орысш", r"орыс тил", r"на русском", r"по-?русски", r"русск", r"\brussian", r"\brus\b", r"\bru\b"],
    "en": [r"агылшынш", r"агылшын тил", r"на английском", r"по-?английски", r"английск", r"\benglish", r"\ben\b",
           r"\beng\b"],
}
MORE_DETAILS = [r"толыгырак", r"толык", r"подробн", r"детальн", r"больше", r"еще", r"more", r"detail", r"тагы",
                r"кенирек", r"тусиндир", r"объясни", r"explain"]
# words that only describe the request itself
META_WORDS = set(fold("""нуска нускада нускасы нускасын версия версию вариант вариантта аудар аударып аударма
    переведи перевод перевести translate translation version бер беру берши берин бериниз дай дайте напиши
    жаз жазып жазшы write give answer жауап жауабын ответ ответь тили тилинде языке language please
    осы осыны сол соны это этот this that it мына мынаны""").split())


def _has(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def analyze(message: str) -> Dict:
    """-> {"lang": requested answer language or None, "more": bool, "content": content words that remain}."""
    f = fold(message)
    lang = next((l for l, pats in LANG_REQUEST.items() if _has(pats, f)), None)
    more = _has(MORE_DETAILS, f)
    rest = []
    for t in content_tokens(message):
        if t in META_WORDS or (lang and _has(LANG_REQUEST[lang], t)) or _has(MORE_DETAILS, t):
            continue
        rest.append(t)
    return {"lang": lang, "more": more, "content": rest}


def resolve(message: str, history: List[str]) -> Dict:
    """Decide which question to answer.

    history: previous questions of the conversation (oldest first).
    Returns {"question", "lang", "prefer_lang", "more", "followup"}.
    """
    info = analyze(message)
    # the last previous question that has its own content (skip earlier follow-ups)
    prev = next((h for h in reversed(history) if analyze(h)["content"]), None)
    out = {"question": message, "lang": None, "prefer_lang": None, "more": False, "followup": None}
    if not prev:
        if info["lang"] and not info["content"]:
            out["followup"] = "no_history"
        return out
    if not info["content"] and (info["lang"] or info["more"]):
        out.update(question=prev, lang=info["lang"], prefer_lang=info["lang"], more=info["more"],
                   followup="language" if info["lang"] else "more")
        return out
    if info["lang"] and len(info["content"]) <= 1 and not detect_concepts(message):
        out.update(question=prev, lang=info["lang"], prefer_lang=info["lang"], followup="language")
        return out
    # short clarification without its own topic: "а для магистрантов?", "ал шетелдиктерге ше?"
    if len(info["content"]) <= 2 and not detect_concepts(message) and detect_concepts(prev) \
            and re.match(r"^\s*(а|ал|and|а если|а для|а как|ал егер)\b", fold(message)):
        out.update(question=f"{prev} {message}", followup="clarification")
    return out


# ------------------------------------------------------------------ small talk
SMALLTALK = {
    "greeting": [r"^\s*(салем|сәлем|сaлем|привет|здравств|добрый (день|вечер|утро)|hello|hi|hey|кайырлы (кун|кеш|тан)|"
                 r"армысыз|ассалаумагалейкум|салеметсиз)\b"],
    "thanks": [r"\b(рахмет|рақмет|спасибо|благодар\w*|thank\w*|thx|кош бол)\b"],
    "complaint": [r"(кате жауап|жауабын кате|дурыс емес|дурыс жауап берме|неправильн(о|ый) ответ|ответ неправильн|"
                  r"ты ошиб|вы ошиб|ошибаешься|wrong answer|incorrect|not correct|answer is wrong|тусинбедин|"
                  r"не понял(а)? вопрос|бекер жауап|жауап бермей|неге булай|неге олай)"],
    "about": [r"\b(сен кимсин|сиз кимсиз|кто ты|кто вы|who are you|what can you do|what questions|не истей ала|"
              r"не биле|что (ты )?умеешь|что ты знаешь|на какие вопросы|кандай сурак\w*|кандай сурактарга|"
              r"какие вопросы|жауап бере ала\w*|help|комек|помощь)(?!\w)"],
}
SMALLTALK_ANSWERS = {
    "greeting": {
        "kk": "Сәлеметсіз бе! Мен Satbayev University-дің виртуалды кеңесшісімін. Оқу, емтихан, шәкіртақы, "
              "жатақхана, практика, академиялық демалыс, ауысу т.б. туралы сұрағыңызды жазыңыз.",
        "ru": "Здравствуйте! Я виртуальный консультант Satbayev University. Задайте вопрос об учёбе, экзаменах, "
              "стипендии, общежитии, практике, академическом отпуске, переводе и т.д.",
        "en": "Hello! I am the Satbayev University virtual consultant. Ask me about studies, exams, scholarships, "
              "dormitories, internships, academic leave, transfers, etc."},
    "thanks": {
        "kk": "Оқасы жоқ! Тағы сұрағыңыз болса, жазыңыз.",
        "ru": "Пожалуйста! Если будут ещё вопросы — пишите.",
        "en": "You're welcome! Feel free to ask another question."},
    "complaint": {
        "kk": "Кешіріңіз, жауап дәл болмаған сияқты. Мен жауапты тек университеттің ресми құжаттарынан іздеймін. "
              "Сұрақты толығырақ жазып көріңіз, мысалы: «Жатақханаға орналасу үшін қандай құжаттар керек?». "
              "Жауаптың астындағы 👎 батырмасын бассаңыз, әкімші бұл сұрақты көреді.",
        "ru": "Извините, похоже, ответ оказался неточным. Я ищу ответы только в официальных документах университета. "
              "Попробуйте сформулировать вопрос подробнее, например: «Какие документы нужны для заселения в "
              "общежитие?». Нажмите 👎 под ответом — администратор увидит этот вопрос.",
        "en": "Sorry, the answer seems to be inaccurate. I search only the official university documents. "
              "Please ask in more detail, e.g. \"What documents are needed to get a dormitory place?\". "
              "Press 👎 under the answer so that the administrator can review it."},
    "about": {
        "kk": "Мен Satbayev University-дің ресми құжаттары (750-ден астам ереже, нұсқаулық, форма) мен бірінші курсқа "
              "арналған анықтама-кітапша бойынша үйретілген AI-кеңесшімін. Мына тақырыптар бойынша сұрақ қоя аласыз:\n"
              "• оқу процесі: кредиттер, GPA, сабаққа қатысу, пәндерге тіркелу, жеке оқу жоспары, эдвайзер\n"
              "• емтихандар: сессия, қайта тапсыру, апелляция, жазғы семестр\n"
              "• шәкіртақы, гранттар, оқу ақысы мен жеңілдіктер\n"
              "• жатақхана: орналасу, құжаттар, тұру ережелері\n"
              "• академиялық демалыс, ауысу, оқудан шығару және қайта қабылдау\n"
              "• практика, дипломдық жұмыс, қорытынды аттестаттау, плагиат\n"
              "• институттар, байланыс телефондары, кітапхана, университет туралы мәліметтер\n"
              "Қазақша, орысша немесе ағылшынша жаза беріңіз, әр жауапқа құжат пен бетін көрсетемін.",
        "ru": "Я AI-консультант, обученный на официальных документах Satbayev University (более 750 положений, "
              "инструкций, форм) и справочнике первокурсника. Можно спрашивать о:\n"
              "• учебном процессе: кредиты, GPA, посещаемость, регистрация на дисциплины, ИУП, эдвайзер\n"
              "• экзаменах: сессия, пересдача, апелляция, летний семестр\n"
              "• стипендиях, грантах, оплате обучения и скидках\n"
              "• общежитии: заселение, документы, правила проживания\n"
              "• академическом отпуске, переводе, отчислении и восстановлении\n"
              "• практике, дипломной работе, итоговой аттестации, антиплагиате\n"
              "• институтах, контактах, библиотеке, сведениях об университете\n"
              "Пишите на казахском, русском или английском — к каждому ответу я указываю документ и страницу.",
        "en": "I am an AI consultant trained on the official documents of Satbayev University (750+ regulations, "
              "instructions and forms) and the freshman guide. You can ask about:\n"
              "• studies: credits, GPA, attendance, course registration, individual study plan, advisor\n"
              "• exams: session, retakes, appeals, summer semester\n"
              "• scholarships, grants, tuition fees and discounts\n"
              "• dormitories: moving in, documents, rules\n"
              "• academic leave, transfer, expulsion and reinstatement\n"
              "• internships, thesis, final attestation, plagiarism\n"
              "• institutes, contacts, library, general information about the university\n"
              "Ask in Kazakh, Russian or English — every answer shows the document and page."},
}


def smalltalk(message: str) -> Optional[str]:
    """Intent of a short non-question message (greeting / thanks / complaint / about) or None."""
    f = fold(message)
    n = len(content_tokens(message))
    if detect_concepts(message) or n > 5:
        return None
    for intent, pats in SMALLTALK.items():
        if intent == "greeting" and n > 3:
            continue
        if intent == "about" and n > 3 and not re.search(r"\b(сен|сиз|ты|вы|you)\b", f):
            continue                     # "How does the university help graduates ..." is a real question
        if _has([fold(p) for p in pats], f):
            return intent
    return None
