"""Colloquial questions, letter folding, synonyms and dialogue follow-ups."""
from ml.followup import resolve
from ml.lexicon import detect_concepts, expand_query, query_stem_groups, support
from ml.text import fold, stems


def test_fold_kazakh_letters_and_misspellings():
    assert fold("Жатақханаға") == "жатакханага"
    assert set(stems("общежития")) == set(stems("ОБШЕЖИТЯ"))
    assert set(stems("жатақхана")) == set(stems("жатакхана"))


def test_colloquial_words_map_to_official_terms():
    assert "dormitory" in detect_concepts("ОБШАГАҒА ҚАНДАЙ ДОКУМЕНТТЕР КЕРЕК")
    assert "dormitory" in detect_concepts("как получить общагу")
    assert "academic_leave" in detect_concepts("академка калай алам")
    assert "scholarship" in detect_concepts("стипуха канша")
    assert not detect_concepts("What is the weather today?")
    assert "общежитие" in expand_query("общага")


def test_synonym_counts_as_evidence():
    groups = query_stem_groups("общагага калай орналасамын")
    assert support(groups, set(stems("Порядок заселения в общежитие студентов"))) >= 0.5


def test_question_words_are_stopwords():
    assert stems("калай алуга болады") == []


def test_language_followup_uses_previous_question():
    r = resolve("ҚАЗАҚША НҰСҚАДА БЕР", ["ОБШАГАҒА ҚАНДАЙ ДОКУМЕНТТЕР КЕРЕК"])
    assert r["question"] == "ОБШАГАҒА ҚАНДАЙ ДОКУМЕНТТЕР КЕРЕК" and r["lang"] == "kk"
    assert resolve("in English please", ["Как получить стипендию?"])["lang"] == "en"
    assert resolve("подробнее", ["Как получить стипендию?"])["more"]
    # a new full question is not a follow-up
    assert resolve("Как подать апелляцию?", ["Как получить стипендию?"])["followup"] is None
