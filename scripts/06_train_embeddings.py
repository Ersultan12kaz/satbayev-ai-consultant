"""STEP 6 - Train our own embedding model.

Stage 1: sub-word LSA (unsupervised) on all chunks.
Stage 2: contrastive fine-tuning (InfoNCE) of a projection matrix on automatically
         generated (question, passage) pairs (Inverse Cloze Task + headings).
Validation: Recall@1/5/10 and MRR of finding the right passage among all
validation passages, before (epoch 0) and after every epoch.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import FIGURES, MODELS, PROCESSED, REPORTS, Tee, Timer, banner, save_json

from ml.documents import read_jsonl
from ml.embeddings import SubwordEncoder
from ml.training_data import make_pairs, split_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    Tee("06_embeddings")
    banner("STEP 6. Training the embedding model")
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    texts = [f"{c.get('section', '')}. {c['text']}" for c in chunks]
    print(f"  corpus: {len(texts)} chunks")

    print("\n  Stage 1 - unsupervised sub-word LSA")
    enc = SubwordEncoder(dim=args.dim)
    with Timer() as t:
        enc.fit_lsa(texts)

    print("\n  Stage 2 - contrastive fine-tuning (InfoNCE, in-batch negatives)")
    pairs = make_pairs(chunks)
    train, val = split_pairs(pairs)
    print(f"  training pairs: {len(train)} | validation pairs: {len(val)}")
    print("  example pair:")
    if train:
        print(f"    query  : {train[0][0][:150]}")
        print(f"    passage: {train[0][1][:150]}...")
    with Timer() as t2:
        enc.fine_tune([(q, p) for q, p, _ in train], [(q, p) for q, p, _ in val], epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr)
    print(f"  fine-tuning time: {t2.sec:.1f}s")
    enc.save(MODELS / "embedder.pkl")
    save_json(REPORTS / "embedding_training.json", {"history": enc.history, "dim": enc.dim,
                                                    "n_train_pairs": len(train), "n_val_pairs": len(val),
                                                    "lsa_seconds": t.sec, "finetune_seconds": t2.sec})
    h = enc.history
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ep = [r["epoch"] for r in h if r["train_loss"] is not None]
    ax[0].plot(ep, [r["train_loss"] for r in h if r["train_loss"] is not None], marker="o")
    ax[0].set_title("Embedding model: InfoNCE training loss"); ax[0].set_xlabel("epoch")
    for k in ("val_R@1", "val_R@5", "val_MRR"):
        ax[1].plot([r["epoch"] for r in h], [r.get(k) for r in h], marker="o", label=k.replace("val_", ""))
    ax[1].set_title("Validation retrieval (epoch 0 = LSA only)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES / "embedding_training.png", dpi=150)
    if h:
        print(f"\n  Validation Recall@5: before fine-tuning {h[0].get('val_R@5', 0):.3f} -> "
              f"after {h[-1].get('val_R@5', 0):.3f}")
    print("  saved: models/embedder.pkl, reports/figures/embedding_training.png")


if __name__ == "__main__":
    main()
