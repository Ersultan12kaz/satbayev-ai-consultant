import numpy as np

from ml.bm25 import BM25
from ml.embeddings import SubwordEncoder
from ml.langdetect import LanguageDetector
from ml.metrics import make_title_matcher, reciprocal_rank, recall_at_k, token_f1
from ml.scope import ScopeClassifier
from tests.fixtures.make_sample_corpus import as_chunks


def test_bm25_ranks_relevant_first():
    docs = ["правила проживания в общежитии", "стипендия назначается по итогам сессии", "академический отпуск"]
    bm = BM25().fit(docs)
    assert bm.search("общежитие правила", k=1)[0][0] == 0


def test_language_detector(models_dir):
    m = LanguageDetector.load(models_dir / "langdetect.json")
    assert m.predict("Какие документы нужны для общежития?") == "ru"
    assert m.predict("Жатақханаға қандай құжаттар қажет?") == "kk"
    assert m.predict("What documents are required for a dormitory?") == "en"


def test_scope_classifier(models_dir):
    m = ScopeClassifier.load(models_dir / "scope.pkl")
    assert m.is_university_question("Как оформить академический отпуск?")[0]
    assert not m.is_university_question("Кто выиграл чемпионат мира по футболу?")[0]


def test_embedding_gradient_matches_numeric():
    """The hand-written InfoNCE gradient must match a finite-difference estimate."""
    chunks = as_chunks()
    enc = SubwordEncoder(dim=8, seed=0).fit_lsa([c["text"] for c in chunks], verbose=False)
    rng = np.random.default_rng(0)
    A, P = rng.normal(size=(6, enc.dim)), rng.normal(size=(6, enc.dim))
    enc.W = np.eye(enc.dim) + 0.1 * rng.normal(size=(enc.dim, enc.dim))
    loss, g = enc._step(A, P, reg=1e-3)
    i, j, eps = 2, 3, 1e-6
    enc.W[i, j] += eps; lp, _ = enc._step(A, P, reg=1e-3)
    enc.W[i, j] -= 2 * eps; lm, _ = enc._step(A, P, reg=1e-3)
    enc.W[i, j] += eps
    assert abs((lp - lm) / (2 * eps) - g[i, j]) < 1e-4


def test_embeddings_are_normalised(models_dir):
    enc = SubwordEncoder.load(models_dir / "embedder.pkl")
    v = enc.encode(["академический отпуск", "dormitory rules"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_metrics():
    assert recall_at_k([0, 1, 0], 1) == 0 and recall_at_k([0, 1, 0], 2) == 1
    assert reciprocal_rank([0, 0, 1]) == 1 / 3
    assert token_f1("академический отпуск", "академический отпуск студента") > 0.5
    assert make_title_matcher([r"общежит"])("Правила проживания в общежитиях")
