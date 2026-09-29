"""STEP LM-4 - Training examples that teach our language model to ANSWER from the found documents.

Format of one example:   question  +  context (sentences found by our retrieval)  ->  answer
The model sees the question and the context and learns to write the answer in the language of the question.
All examples are made automatically from our own data (no external model):

  A. real pipeline answers: every evaluation / student question and many generated questions are answered by
     our RAG pipeline (retrieval + composer); context = the sentences it used, answer = the composed answer
     in plain words;
  B. sentence answers: a question is generated about one sentence of a chunk (topic words + a question
     template); context = that sentence mixed with other sentences; answer = the cleaned sentence;
  C. cross-lingual answers: the same rule written in Kazakh and in Russian (aligned sentence pairs of the
     university documents) - the context is in one language, the question and the answer in the other one;
     this is how the model learns to answer in Kazakh from a Russian document;
  D. "no information": the context does not contain the answer -> the model must say so.

    python scripts/16_lm_sft_data.py [--n-sentence 40000] [--n-pipeline 6000]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from _common import DATA, EVAL, MODELS, PROCESSED, Tee, banner, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.composer import clean_sentence, natural_text  # noqa: E402
from ml.documents import read_jsonl  # noqa: E402
from ml.lexicon import CONCEPT_LABELS  # noqa: E402
from ml.text import content_tokens, split_sentences, tokenize  # noqa: E402

LM = DATA / "lm"
NO_INFO = {"kk": "Берілген құжаттарда бұл сұрақ бойынша мәлімет жоқ.",
           "ru": "В найденных документах нет информации по этому вопросу.",
           "en": "The documents found do not contain information on this question."}
TEMPLATES = {
    "kk": ["{t} туралы айтып берші", "{t} деген не?", "{t} қалай жүзеге асырылады?", "{t} туралы не білесің?",
           "{t} бойынша не істеу керек?", "{t} қандай?", "{t} туралы ақпарат керек", "{t} жайлы түсіндірші"],
    "ru": ["Расскажи про {t}", "Что такое {t}?", "Как происходит {t}?", "Что известно про {t}?",
           "Что нужно знать про {t}?", "Какие правила про {t}?", "Объясни {t}"],
    "en": ["Tell me about {t}", "What is {t}?", "How does {t} work?", "What are the rules for {t}?",
           "Explain {t}"],
}
FOLLOW = {"kk": ["толығырақ айтшы", "тағы не бар?", "толығырақ түсіндірші", "тағы айтшы", "одан кейін ше?"],
          "ru": ["подробнее", "а что ещё?", "расскажи подробнее", "а дальше?"],
          "en": ["tell me more", "more details please", "what else?"]}
SWITCH = {"kk": "қазақша айтшы", "ru": "а на русском?", "en": "in English please"}
_BAD = re.compile(r"\.{4,}|_{4,}|…{2,}|\bФ\s*КазНИТУ|\bF\s*KazNRTU")


def good_sentence(s: str) -> bool:
    n = len(tokenize(s))
    if n < 6 or n > 60 or _BAD.search(s):
        return False
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) < 0.3 and s[:1].isupper()


def topic_words(sentence: str, rnd: random.Random) -> str:
    """2-3 consecutive informative words of the sentence, original spelling (the question topic)."""
    words = re.findall(r"[\w-]+", sentence)
    keep = set(content_tokens(sentence))
    idx = [i for i, w in enumerate(words) if w.lower() in keep or len(w) > 6]
    if not idx:
        return " ".join(words[:3])
    i = rnd.choice(idx[: max(1, len(idx) // 2 + 1)])
    return " ".join(words[i:i + rnd.choice([1, 2, 2, 3])]).lower()


def make_question(topic: str, lang: str, rnd: random.Random) -> str:
    return rnd.choice(TEMPLATES.get(lang, TEMPLATES["ru"])).format(t=topic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sentence", type=int, default=20000)
    ap.add_argument("--n-pipeline", type=int, default=6000)
    ap.add_argument("--n-cross", type=int, default=100000)
    args = ap.parse_args()
    Tee("16_lm_sft_data")
    banner("STEP LM-4. Question -> context -> answer examples")
    rnd = random.Random(7)
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    sent_pool = []                                               # (lang, sentence, chunk index)
    for ci, ch in enumerate(chunks):
        for s in split_sentences(ch["text"]):
            s = s.strip()
            if good_sentence(s):
                sent_pool.append((ch.get("lang", "ru"), s, ci))
    by_chunk = {}
    for lang, s, ci in sent_pool:
        by_chunk.setdefault(ci, []).append(s)
    print(f"  clean sentences: {len(sent_pool)} in {len(by_chunk)} chunks")
    out = []

    def distractors(k, lang=None):
        res = []
        while len(res) < k:
            l2, s, _ = rnd.choice(sent_pool)
            if lang is None or l2 == lang:
                res.append(s)
        return res

    # ---- B. one sentence answers
    for _ in range(args.n_sentence):
        lang, s, ci = rnd.choice(sent_pool)
        same = [x for x in by_chunk[ci] if x != s]
        ctx = [s] + rnd.sample(same, min(len(same), rnd.randint(1, 3))) + distractors(rnd.randint(0, 2), lang)
        rnd.shuffle(ctx)
        out.append({"kind": "sentence", "lang": lang, "q": make_question(topic_words(s, rnd), lang, rnd),
                    "ctx": ctx, "answer": clean_sentence(s)})
    print(f"  B. sentence answers: {args.n_sentence}")

    # ---- E. dialogue: the model must remember the previous turn ("толығырақ айтшы", "қазақша айтшы")
    n_dialog = 0
    rich = [(l, s, ci) for l, s, ci in sent_pool if len(by_chunk[ci]) >= 4]
    for _ in range(args.n_sentence // 4 if rich else 0):
        lang, s, ci = rnd.choice(rich)
        others = [x for x in by_chunk[ci] if x != s]
        more = rnd.sample(others, min(len(others), rnd.randint(2, 3)))
        q1 = make_question(topic_words(s, rnd), lang, rnd)
        ctx = [s] + more + distractors(rnd.randint(0, 2), lang)
        rnd.shuffle(ctx)
        out.append({"kind": "dialog", "lang": lang, "history": [[q1, clean_sentence(s)]],
                    "q": rnd.choice(FOLLOW[lang]), "ctx": ctx, "answer": " ".join(clean_sentence(x) for x in more)})
        n_dialog += 1
    print(f"  E. dialogue follow-ups: {n_dialog}")

    # ---- C. cross-lingual answers (aligned kk <-> ru sentences of the same documents)
    par_path = LM / "uni_parallel.json"
    n_cross = 0
    if par_path.exists():
        pairs = [p for p in json.loads(par_path.read_text(encoding="utf-8"))
                 if good_sentence(p["src"]) and good_sentence(p["tgt"])]
        rnd.shuffle(pairs)
        # every aligned pair is used in BOTH directions (Russian document -> Kazakh answer and back): the model
        # must learn to answer in the language of the question, not to copy the document's language
        both = [(p["src_lang"], p["src"], p["tgt_lang"], p["tgt"]) for p in pairs] + \
               [(p["tgt_lang"], p["tgt"], p["src_lang"], p["src"]) for p in pairs]
        rnd.shuffle(both)
        for a_lang, a, b_lang, b in both[: args.n_cross]:
            ctx = [b] + distractors(rnd.randint(1, 3), b_lang)       # the document is in language b
            rnd.shuffle(ctx)
            if rnd.random() < 0.3:          # "... қазақша айтшы": the previous answer again, in the other language
                qb = make_question(topic_words(b, rnd), b_lang, rnd)
                out.append({"kind": "cross-dialog", "lang": a_lang, "history": [[qb, clean_sentence(b)]],
                            "q": SWITCH[a_lang], "ctx": ctx, "answer": clean_sentence(a)})
            else:
                out.append({"kind": "cross", "lang": a_lang, "q": make_question(topic_words(a, rnd), a_lang, rnd),
                            "ctx": ctx, "answer": clean_sentence(a)})
            n_cross += 1
    print(f"  C. cross-lingual answers: {n_cross}" + ("" if par_path.exists() else " (no aligned pairs file)"))

    # ---- A. answers of our RAG pipeline
    from ml.rag import RAGPipeline
    pipe = RAGPipeline.load(MODELS)
    pipe.wiki.enabled = False
    pipe.llm = None               # targets must come from the documents (+ our translator), not from the LLM itself
    questions = [json.loads(l)["question"] for l in (EVAL / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    sq = EVAL / "student_questions.json"
    if sq.exists():
        questions += json.loads(sq.read_text(encoding="utf-8"))["student_questions"]
    for c, labels in CONCEPT_LABELS.items():
        for lang, label in labels.items():
            for tpl in TEMPLATES[lang][:4]:
                questions.append(tpl.format(t=label.lower()))
    while len(questions) < args.n_pipeline:
        lang, s, _ = rnd.choice(sent_pool)
        questions.append(make_question(topic_words(s, rnd), lang, rnd))
    n_pipe = 0
    for i, q in enumerate(questions[: args.n_pipeline]):
        try:
            r = pipe.answer(q)
        except Exception:
            continue
        if r["status"] != "answered" or r.get("generator") not in ("composer", "extractive"):
            continue
        ctx = [u["text"] for u in r.get("used_sentences", [])][:6]
        if len(ctx) < 1:
            continue
        ctx += distractors(rnd.randint(0, 2))
        rnd.shuffle(ctx)
        ans = natural_text(r["text"], r["lang"])
        ans = re.sub(r"\n*\((Бұл туралы ереже|Это правило|This rule|Құжат басқа тілде|Документ написан|The document is in)[^)]*\)", "", ans).strip()
        out.append({"kind": "pipeline", "lang": r["lang"], "q": q, "ctx": ctx, "answer": ans})
        n_pipe += 1
        if i % 500 == 0:
            print(f"    pipeline questions: {i}/{args.n_pipeline}, examples {n_pipe}", flush=True)
    print(f"  A. pipeline answers: {n_pipe}")

    # ---- D. no information
    n_no = len(out) // 20
    for _ in range(n_no):
        lang, s, _ = rnd.choice(sent_pool)
        ctx = distractors(rnd.randint(2, 4))
        out.append({"kind": "noinfo", "lang": lang, "q": make_question(topic_words(s, rnd), lang, rnd),
                    "ctx": [c for c in ctx if c != s], "answer": NO_INFO[lang]})
    print(f"  D. 'no information' examples: {n_no}")

    rnd.shuffle(out)
    with open(LM / "sft.jsonl", "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    save_json(LM / "sft_stats.json", {"total": len(out), "sentence": args.n_sentence, "cross": n_cross,
                                      "pipeline": n_pipe, "noinfo": n_no})
    banner("Summary")
    print(f"  examples: {len(out)} -> data/lm/sft.jsonl")
    for ex in out[:3]:
        print(f"   Q: {ex['q']}\n   A: {ex['answer'][:160]}\n")


if __name__ == "__main__":
    main()
