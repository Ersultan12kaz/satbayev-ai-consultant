"""STEP LM-2 - Our sub-word tokenizer + the corpus turned into token ids.

1. trains the BPE tokenizer (ml/lm/tokenizer.py) on a sample of data/lm/corpus.txt -> models/lm/tokenizer.json
2. encodes the whole corpus with several processes -> data/lm/train.bin, data/lm/val.bin (uint16 token ids,
   </s> between documents); 0.5% of the documents are kept for validation.

    python scripts/14_lm_tokenize.py [--vocab 32000] [--sample-mb 120]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from _common import DATA, MODELS, Tee, banner, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.lm.tokenizer import EOS, LMTokenizer  # noqa: E402

LM = DATA / "lm"
TOK_PATH = MODELS / "lm" / "tokenizer.json"
_tok = None


def documents(path: Path):
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line.rstrip("\n"))
            elif buf:
                yield "\n".join(buf); buf = []
    if buf:
        yield "\n".join(buf)


def _init():
    global _tok
    _tok = LMTokenizer.load(TOK_PATH)


def _encode(doc: str):
    ids = _tok.encode(doc)
    ids.append(EOS)
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--sample-mb", type=int, default=120)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    Tee("14_lm_tokenize")
    banner("STEP LM-2. Tokenizer + token ids")
    corpus = LM / "corpus.txt"
    size = corpus.stat().st_size
    print(f"  corpus: {size / 1e6:.0f} MB")
    TOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TOK_PATH.exists():
        keep = min(1.0, args.sample_mb * 1e6 / size)
        rnd = random.Random(0)
        sample = [d for d in documents(corpus) if rnd.random() < keep]
        print(f"  tokenizer sample: {len(sample)} documents ({sum(map(len, sample)) / 1e6:.0f}M characters)")
        t0 = time.time()
        tok = LMTokenizer().train(sample, vocab_size=args.vocab)
        tok.save(TOK_PATH)
        print(f"  tokenizer: {len(tok)} tokens, trained in {time.time() - t0:.0f}s")
        del sample
    tok = LMTokenizer.load(TOK_PATH)
    demo = "Жатақханаға орналасу үшін қандай құжаттар керек?"
    print(f"  demo: {demo} -> {[tok.inv[i] for i in tok.encode(demo)]}")
    rnd = random.Random(1)
    n_tok = {"train": 0, "val": 0}
    t0 = time.time()
    with open(LM / "train.bin", "wb") as ftr, open(LM / "val.bin", "wb") as fva, \
            Pool(args.workers, initializer=_init) as pool:
        for i, arr in enumerate(pool.imap(_encode, documents(corpus), chunksize=64)):
            split = "val" if rnd.random() < 0.005 else "train"
            (fva if split == "val" else ftr).write(arr.tobytes())
            n_tok[split] += len(arr)
            if i % 50000 == 0 and i:
                print(f"    {i} documents, {sum(n_tok.values()) / 1e6:.0f}M tokens, {time.time() - t0:.0f}s", flush=True)
    chars_per_tok = size / max(1, sum(n_tok.values()))
    save_json(LM / "tokens.json", {"train_tokens": n_tok["train"], "val_tokens": n_tok["val"],
                                   "vocab": len(tok), "chars_per_token": round(chars_per_tok, 2)})
    banner("Summary")
    print(f"  train tokens: {n_tok['train'] / 1e6:.1f}M | val tokens: {n_tok['val'] / 1e6:.2f}M | "
          f"{chars_per_tok:.2f} characters per token")


if __name__ == "__main__":
    main()
