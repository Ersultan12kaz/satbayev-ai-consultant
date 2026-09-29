def test_answer_ru_with_sources(pipeline):
    r = pipeline.answer("Какие документы нужны для академического отпуска?")
    assert r["lang"] == "ru" and r["status"] == "answered"
    assert r["sources"] and "Академическая политика" in r["sources"][0]["title"]
    assert "[1]" in r["text"]


def test_answer_kz(pipeline):
    r = pipeline.answer("Жатақханаға қоныстану үшін қандай құжаттар қажет?")
    assert r["lang"] == "kk" and r["status"] == "answered"
    assert "жатақхана" in r["sources"][0]["title"].lower()


def test_answer_en(pipeline):
    r = pipeline.answer("How can I appeal an exam grade?")
    assert r["lang"] == "en" and r["status"] == "answered"
    assert "examination" in r["sources"][0]["title"].lower()


def test_out_of_scope(pipeline):
    r = pipeline.answer("Кто выиграл чемпионат мира по футболу?")
    assert r["status"] == "out_of_scope"


def test_modes(pipeline):
    for mode in ("dense", "bm25", "hybrid"):
        r = pipeline.answer("Как получить место в общежитии?", mode=mode, top_k=3)
        assert len(r["retrieved"]) <= 3


def test_extractive_answer_is_faithful(pipeline):
    from ml.generator import supported_ratio
    r = pipeline.answer("Как подать апелляцию на результат экзамена?")
    ctx = [c["text"] for c in pipeline.retriever.chunks if c["chunk_id"] in {s["chunk_id"] for s in r["sources"]}]
    assert supported_ratio(r["text"], ctx) == 1.0



def test_composer_answer_is_readable(pipeline):
    r = pipeline.answer("Как подать апелляцию на результат экзамена?")
    assert r["status"] == "answered" and r["generator"] == "composer"
    assert r["text"].startswith("Коротко:") and "[1]" in r["text"]
    assert r["question_type"]


def test_followup_in_kazakh(pipeline):
    r = pipeline.answer("қазақша нұсқада бер", history=["Как подать апелляцию на результат экзамена?"])
    assert r["lang"] == "kk" and r["resolved_question"].startswith("Как подать")
