"""Our own multilingual text-embedding model.

Stage 1 (unsupervised pre-training)
    Sub-word TF-IDF (character 3-5 grams inside words + word stems) followed by
    truncated SVD = Latent Semantic Analysis.  Character n-grams make the model
    robust to Kazakh agglutination and Russian inflection.

Stage 2 (contrastive fine-tuning)
    A projection matrix W is trained with the InfoNCE loss (in-batch negatives,
    symmetric, temperature tau) on (question, passage) pairs, so that a question
    and the passage that answers it end up close in the vector space.
    Training pairs are produced automatically with the Inverse Cloze Task
    (a sentence of a chunk acts as a pseudo-question for the rest of the chunk)
    plus section-heading -> chunk pairs.

Gradients are derived and implemented by hand with numpy (see ``_step``).
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import fold, normalize, stem, tokenize


def _stem_analyzer(text: str) -> List[str]:
    return [stem(t) for t in tokenize(fold(text)) if len(t) > 1]


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class SubwordEncoder:
    """Embedding model = TF-IDF(subwords) -> SVD -> learned projection W."""

    def __init__(self, dim: int = 256, max_char_features: int = 150_000, max_word_features: int = 60_000,
                 temperature: float = 0.05, seed: int = 42):
        self.dim = dim
        self.temperature = temperature
        self.seed = seed
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                        max_features=max_char_features, dtype=np.float32)
        self.word_vec = TfidfVectorizer(analyzer=_stem_analyzer, min_df=2, sublinear_tf=True,
                                        max_features=max_word_features, dtype=np.float32)
        self.svd: Optional[TruncatedSVD] = None
        self.W: Optional[np.ndarray] = None
        self.history: List[Dict] = []
        self.fitted_ = False

    # ------------------------------------------------------------ stage 1
    def _tfidf(self, texts: Sequence[str], fit: bool = False) -> sparse.csr_matrix:
        texts = [fold(normalize(t)) for t in texts]      # "жатақхана" == "жатакхана", "общежитие" == "обшежитие"
        if fit:
            A = self.char_vec.fit_transform(texts)
            B = self.word_vec.fit_transform(texts)
        else:
            A = self.char_vec.transform(texts)
            B = self.word_vec.transform(texts)
        return sparse.hstack([A, B * 1.5]).tocsr()

    def fit_lsa(self, corpus: Sequence[str], verbose: bool = True) -> "SubwordEncoder":
        t0 = time.time()
        X = self._tfidf(corpus, fit=True)
        dim = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=7, random_state=self.seed)
        self.svd.fit(X)
        self.dim = dim
        self.W = np.eye(dim, dtype=np.float64)
        self.fitted_ = True
        if verbose:
            ev = self.svd.explained_variance_ratio_.sum()
            print(f"  LSA fitted: {X.shape[0]} texts, {X.shape[1]} sub-word features -> {dim} dims "
                  f"(explained variance {ev:.3f}) in {time.time() - t0:.1f}s")
        return self

    def base(self, texts: Sequence[str]) -> np.ndarray:
        # X @ V (same as svd.transform) with a cached C-contiguous projection matrix: sklearn transposes the
        # 200k x 256 components matrix on every call, which made one query ~150 ms instead of a few ms
        proj = getattr(self, "_proj", None)
        if proj is None:
            proj = self._proj = np.ascontiguousarray(self.svd.components_.T)
        return _l2n(np.asarray(self._tfidf(texts) @ proj, dtype=np.float64))

    def encode(self, texts: Sequence[str], batch_size: int = 2048) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(_l2n(self.base(texts[i:i + batch_size]) @ self.W))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    # ------------------------------------------------------------ stage 2
    def _step(self, A: np.ndarray, P: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
        """One InfoNCE forward/backward pass. Returns (loss, dL/dW)."""
        tau = self.temperature
        B = A.shape[0]
        U, V = A @ self.W, P @ self.W
        nu = np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
        nv = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        q, p = U / nu, V / nv
        S = q @ p.T / tau
        # row-wise softmax (question -> passage) and column-wise (passage -> question)
        Sr = S - S.max(1, keepdims=True); Pr = np.exp(Sr); Pr /= Pr.sum(1, keepdims=True)
        Sc = S - S.max(0, keepdims=True); Pc = np.exp(Sc); Pc /= Pc.sum(0, keepdims=True)
        idx = np.arange(B)
        loss = -0.5 * (np.log(Pr[idx, idx] + 1e-12).mean() + np.log(Pc[idx, idx] + 1e-12).mean())
        I = np.eye(B)
        dS = 0.5 * ((Pr - I) + (Pc - I)) / B
        dq = dS @ p / tau
        dp = dS.T @ q / tau
        dU = (dq - q * (dq * q).sum(1, keepdims=True)) / nu
        dV = (dp - p * (dp * p).sum(1, keepdims=True)) / nv
        dW = A.T @ dU + P.T @ dV
        D = self.W - np.eye(self.dim)
        loss += reg * float((D * D).sum())
        dW += 2 * reg * D
        return float(loss), dW

    def recall_at_k(self, queries: Sequence[str], passages: Sequence[str], ks=(1, 5, 10)) -> Dict[str, float]:
        q = self.encode(queries); p = self.encode(passages)
        S = np.nan_to_num(q @ p.T, nan=-1.0)
        ranks = (S >= S[np.arange(len(q)), np.arange(len(q))][:, None]).sum(1) - 1
        out = {f"R@{k}": float((ranks < k).mean()) for k in ks}
        out["MRR"] = float((1.0 / (ranks + 1)).mean())
        return out

    def fine_tune(self, pairs: Sequence[Tuple[str, str]], val_pairs: Sequence[Tuple[str, str]] = (),
                  epochs: int = 8, batch_size: int = 128, lr: float = 2e-3, reg: float = 1e-3,
                  verbose: bool = True) -> "SubwordEncoder":
        rng = np.random.default_rng(self.seed)
        A_all = self.base([q for q, _ in pairs])
        P_all = self.base([p for _, p in pairs])
        ok = (np.linalg.norm(A_all, axis=1) > 0.5) & (np.linalg.norm(P_all, axis=1) > 0.5)
        if not ok.all():                       # empty texts (no known sub-words) would give 0/0 = NaN
            if verbose:
                print(f"  skipped {int((~ok).sum())} pairs with empty text")
            A_all, P_all = A_all[ok], P_all[ok]
            pairs = [pr for pr, k in zip(pairs, ok) if k]
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        best_W, best_score = self.W.copy(), -float("inf")
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        if val_pairs:
            r = self.recall_at_k([q for q, _ in val_pairs], [p for _, p in val_pairs])
            self.history.append({"epoch": 0, "train_loss": None, **{"val_" + k: v for k, v in r.items()}})
            best_score = r["MRR"]
            if verbose:
                print("  epoch  0 (LSA only)  | " + " | ".join(f"val_{k}={v:.3f}" for k, v in r.items()))
        for ep in range(1, epochs + 1):
            t0 = time.time()
            order = rng.permutation(len(pairs))
            losses = []
            for s in range(0, len(order) - 1, batch_size):
                idx = order[s:s + batch_size]
                if len(idx) < 4:
                    continue
                loss, g = self._step(A_all[idx], P_all[idx], reg)
                losses.append(loss)
                t += 1
                mW = b1 * mW + (1 - b1) * g; vW = b2 * vW + (1 - b2) * g * g
                self.W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            if not losses or not np.isfinite(losses).all() or not np.isfinite(self.W).all():
                if verbose:
                    print(f"  epoch {ep}: training became unstable - keeping the best weights and stopping")
                self.W = best_W.copy()
                break
            rec = {"epoch": ep, "train_loss": float(np.mean(losses)) if losses else None}
            if val_pairs:
                r = self.recall_at_k([q for q, _ in val_pairs], [p for _, p in val_pairs])
                rec.update({"val_" + k: v for k, v in r.items()})
            self.history.append(rec)
            score = rec.get("val_MRR", -float(rec["train_loss"]))
            if score > best_score:
                best_score, best_W = score, self.W.copy()
            if verbose:
                extra = " | ".join(f"{k}={v:.3f}" for k, v in rec.items() if k.startswith("val_"))
                print(f"  epoch {ep:2d} | train_loss={rec['train_loss']:.4f} | {extra} | {time.time() - t0:.1f}s")
        if np.isfinite(best_score):
            self.W = best_W                      # the epoch with the best validation MRR
        return self

    # ------------------------------------------------------------ io
    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_proj", None)          # cache, rebuilt on first use
        return state

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "SubwordEncoder":
        with open(path, "rb") as f:
            return pickle.load(f)
