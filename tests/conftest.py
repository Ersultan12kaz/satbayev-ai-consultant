"""Test fixtures: a tiny model set trained on the synthetic sample corpus (seconds)."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="su_rag_test_"))
os.environ["MODELS_DIR"] = str(_TMP / "models")
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["SECRET_KEY"] = "test-secret-key-0123456789abcdef0123456789abcdef"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

from ml.embeddings import SubwordEncoder  # noqa: E402
from ml.langdetect import LanguageDetector  # noqa: E402
from ml.rag import RAGPipeline  # noqa: E402
from ml.reranker import Reranker, features  # noqa: E402
from ml.retriever import Retriever  # noqa: E402
from ml.scope import ScopeClassifier, generate_training_data  # noqa: E402
from ml.training_data import make_pairs, split_pairs  # noqa: E402
from tests.fixtures.make_sample_corpus import as_chunks  # noqa: E402


def _train(models_dir: Path):
    chunks = as_chunks()
    enc = SubwordEncoder(dim=32).fit_lsa([c["section"] + ". " + c["text"] for c in chunks], verbose=False)
    tr, va = split_pairs(make_pairs(chunks), val_ratio=0.2)
    enc.fine_tune([(q, p) for q, p, _ in tr], [(q, p) for q, p, _ in va], epochs=2, batch_size=8, verbose=False)
    enc.save(models_dir / "embedder.pkl")
    r = Retriever.build(chunks, enc)
    r.save(models_dir / "index")
    LanguageDetector().fit([c["text"] for c in chunks], [c["lang"] for c in chunks]).save(models_dir / "langdetect.json")
    X, y = generate_training_data()
    ScopeClassifier(epochs=25).fit(X, y, verbose=False).save(models_dir / "scope.pkl")
    # reranker on ICT pairs
    import numpy as np
    id2 = {c["chunk_id"]: i for i, c in enumerate(r.chunks)}
    feats, labels = [], []
    for q, _, ci in make_pairs(chunks, seed=1):
        cands = r.search(q, k=10)
        F = features(q, cands)
        for c, f in zip(cands, F):
            feats.append(f); labels.append(1.0 if c.idx == id2[chunks[ci]["chunk_id"]] else 0.0)
    Reranker().fit(np.array(feats), np.array(labels), epochs=100, verbose=False).save(models_dir / "reranker.json")
    # question type + sentence ranker (answer composer)
    from ml.composer import SentenceRanker, make_sentence_training
    from ml.qtype import QuestionTypeClassifier, generate_training_data as qtype_data
    qt = QuestionTypeClassifier(epochs=20).fit(*qtype_data(), verbose=False)
    qt.save(models_dir / "qtype.pkl")
    Xs, ys, _ = make_sentence_training(chunks, r, qt, enc, n=60)
    SentenceRanker().fit(Xs, ys, epochs=100, verbose=False).save(models_dir / "sentence_ranker.json")
    return models_dir


@pytest.fixture(scope="session")
def models_dir():
    d = Path(os.environ["MODELS_DIR"])
    if not (d / "reranker.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        _train(d)
    return d


@pytest.fixture(scope="session")
def pipeline(models_dir):
    p = RAGPipeline.load(models_dir, not_found_threshold=0.2)
    p.wiki.enabled = False
    p.llm = None
    return p


@pytest.fixture(scope="session")
def client(models_dir):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        yield c
