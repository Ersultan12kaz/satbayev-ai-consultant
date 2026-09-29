from ml.documents import Page, chunk_pages, detect_heading
from ml.text import clean_page, fix_hyphenation, remove_repeated_lines, split_sentences, stems, tokenize


def test_tokenize_kazakh_and_russian():
    assert tokenize("Жатақханаға орын, общежитие!") == ["жатақханаға", "орын", "общежитие"]


def test_stems_match_word_forms():
    assert set(stems("стипендия")) == set(stems("стипендии"))


def test_hyphenation_and_headers():
    assert fix_hyphenation("обуче-\nния") == "обучения"
    cleaned = clean_page("Редакция № 5 от 26.09.2022\nСтраница 3 из 41\nТекст правил.\n12")
    assert cleaned == "Текст правил."


def test_repeated_lines_removed():
    pages = ["КазНИТУ им. Сатпаева\nтекст 1", "КазНИТУ им. Сатпаева\nтекст 2", "КазНИТУ им. Сатпаева\nтекст 3"]
    out = remove_repeated_lines(pages)
    assert all("КазНИТУ" not in p for p in out)


def test_sentence_split():
    s = split_sentences("Первое предложение. Второе предложение! Third one?")
    assert len(s) == 3


def test_heading_detection_and_chunking():
    assert detect_heading("2. Академический отпуск")
    text = "1. Общие положения\n" + " ".join(["Студент обязан соблюдать правила университета."] * 60)
    chunks = list(chunk_pages([Page("d1", "Doc", "ru", "c", "u", 1, text)], max_words=80, overlap_words=10))
    assert len(chunks) >= 3
    assert all(c.section.startswith("1. Общие") for c in chunks)
    assert all(c.n_words <= 90 for c in chunks)
