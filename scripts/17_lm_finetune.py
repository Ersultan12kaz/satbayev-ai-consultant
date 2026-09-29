"""STEP LM-5 - Fine-tuning our pre-trained language model to answer from the documents (RAG answers).

Input  : models/lm/lm_base.pt (step LM-3) + data/lm/sft.jsonl (step LM-4)
Output : models/lm/lm_rag.pt - the model used by the assistant, models/lm/bilex.json - our kk<->ru word
         dictionary (learned from the aligned sentences) for the faithfulness check of cross-lingual answers.

Every example is  <s> <|lang|> <|question|> q <|context|> sentences <|answer|> answer </s>; the loss is computed
only on the answer tokens.  20% of the batches are plain text (so that the language is not forgotten).

    python scripts/17_lm_finetune.py [--epochs 3] [--hours 6]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from _common import DATA, FIGURES, MODELS, REPORTS, Tee, banner, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.lm.answerer import LMAnswerer  # noqa: E402
from ml.lm.model import GPT, GPTConfig  # noqa: E402
from ml.lm.tokenizer import ANS_TOK, BOS, CTX_TOK, EOS, LANG_TOK, NL, PAD, Q_TOK, SEP_TOK, LMTokenizer  # noqa: E402
from ml.text import stems  # noqa: E402

LM = DATA / "lm"
OUT = MODELS / "lm"


def build_bilex(pairs, min_count=3, min_dice=0.2, top=4):
    """Our bilingual stem dictionary: Dice co-occurrence of stems in aligned kk/ru sentences."""
    cnt_a, cnt_b, co = Counter(), Counter(), Counter()
    for p in pairs:
        a, b = set(stems(p["src"])), set(stems(p["tgt"]))
        if len(a) > 40 or len(b) > 40:
            continue
        cnt_a.update(a); cnt_b.update(b)
        for x in a:
            for y in b:
                co[(x, y)] += 1
    lex = defaultdict(list)
    for (x, y), c in co.items():
        if c < min_count:
            continue
        d = 2 * c / (cnt_a[x] + cnt_b[y])
        if d >= min_dice:
            lex[x].append((d, y)); lex[y].append((d, x))
    return {k: [w for _, w in sorted(v, reverse=True)[:top]] for k, v in lex.items()}


def encode_example(tok, ex, ctx_len):
    q = tok.encode(ex["q"])[:60]
    ctx = []
    for s in ex["ctx"]:
        ids = tok.encode(s) + [NL]
        if len(ctx) + len(ids) > 380 - 8 * len(ex.get("history", [])):
            break
        ctx += ids
    ans = tok.encode(ex["answer"])[:max(16, ctx_len - 8 - len(q) - len(ctx))] + [EOS]
    hist = []
    for hq, ha in ex.get("history", [])[-2:]:                # the previous turns of the conversation
        hist += [Q_TOK] + tok.encode(hq)[:40] + [ANS_TOK] + tok.encode(ha)[:70] + [SEP_TOK]
    hist = hist[-180:]
    prompt = [BOS, LANG_TOK.get(ex["lang"], LANG_TOK["ru"])] + hist + [Q_TOK] + q + [CTX_TOK] + ctx + [ANS_TOK]
    ids = (prompt + ans)[:ctx_len + 1]
    mask = ([0] * len(prompt) + [1] * len(ans))[:ctx_len + 1]
    return ids, mask


def batches(examples, B, ctx_len, rnd):
    order = list(range(len(examples)))
    rnd.shuffle(order)
    for i in range(0, len(order) - B + 1, B):
        items = [examples[j] for j in order[i:i + B]]
        T = max(len(x[0]) for x in items)
        X = np.full((B, T - 1), PAD, dtype=np.int64)
        Y = np.full((B, T - 1), PAD, dtype=np.int64)
        M = np.zeros((B, T - 1), dtype=np.float32)
        for k, (ids, mask) in enumerate(items):
            n = len(ids) - 1
            X[k, :n] = ids[:-1]; Y[k, :n] = ids[1:]; M[k, :n] = mask[1:]
        yield torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--hours", type=float, default=6.0)
    args = ap.parse_args()
    Tee("17_lm_finetune")
    banner("STEP LM-5. Fine-tuning our language model for answers from the documents")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx_amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16 else torch.float16) \
        if device == "cuda" else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))
    tok = LMTokenizer.load(OUT / "tokenizer.json")

    # our kk <-> ru dictionary for the faithfulness check
    par = LM / "uni_parallel.json"
    if par.exists():
        bilex = build_bilex(json.loads(par.read_text(encoding="utf-8")))
        (OUT / "bilex.json").write_text(json.dumps(bilex, ensure_ascii=False), encoding="utf-8")
        print(f"  bilingual dictionary: {len(bilex)} stems")

    ck = torch.load(OUT / "lm_base.pt", map_location=device)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"  base model: {model.n_params() / 1e6:.1f}M parameters, pre-trained for {ck.get('step')} steps")

    rnd = random.Random(0)
    raw = [json.loads(l) for l in (LM / "sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    enc = [encode_example(tok, ex, cfg.ctx) for ex in raw]
    enc = [e for e in enc if sum(e[1]) > 1]
    n_val = max(50, len(enc) // 30)
    val, train = enc[:n_val], enc[n_val:]
    val_raw = raw[:n_val]
    # plain text keeps the language; after the clean-up the corpus may live in _training_files
    tb = next((x for x in (LM / "train.bin", LM.parents[1] / "_training_files" / "data" / "lm" / "train.bin")
               if x.exists()), None)
    if tb is None:
        print("  train.bin not found - fine-tuning without plain-text batches")
        lm_text = None
    else:
        lm_text = np.memmap(tb, dtype=np.uint16, mode="r")
    print(f"  examples: {len(train)} train, {len(val)} validation")
    steps_total = args.epochs * (len(train) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def lr_at(s):
        w = 200
        return args.lr * min(1.0, (s + 1) / w) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total)))))

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = []
        for X, Y, M in batches(val, args.batch, cfg.ctx, random.Random(1)):
            with ctx_amp:
                _, l = model(X.to(device), Y.to(device), M.to(device))
            ls.append(l.item())
        model.train()
        return float(np.mean(ls)) if ls else float("nan")

    history, step, t0 = [], 0, time.time()
    best = float("inf")
    model.train()
    for ep in range(1, args.epochs + 1):
        for X, Y, M in batches(train, args.batch, cfg.ctx, rnd):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            if lm_text is not None and rnd.random() < 0.2:      # plain text: keep the language
                T = 256
                ix = np.random.randint(0, len(lm_text) - T - 1, size=args.batch)
                X = torch.from_numpy(np.stack([lm_text[i:i + T] for i in ix]).astype(np.int64))
                Y = torch.from_numpy(np.stack([lm_text[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
                M = torch.ones_like(Y, dtype=torch.float32)
            with ctx_amp:
                _, loss = model(X.to(device), Y.to(device), M.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  epoch {ep} step {step}/{steps_total} | loss {loss.item():.3f}", flush=True)
            if (time.time() - t0) / 3600 > args.hours:
                break
        vl = val_loss()
        history.append({"epoch": ep, "step": step, "val_loss": round(vl, 4)})
        print(f"  == epoch {ep}: validation loss (answer tokens) {vl:.3f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": ep}, OUT / "lm_rag.pt")
            print("     saved models/lm/lm_rag.pt")
        if (time.time() - t0) / 3600 > args.hours:
            print("  time budget used"); break

    # examples of generated answers (validation questions, the model never saw them)
    ans = LMAnswerer.load(MODELS)
    shown, accepted = [], 0
    if ans is not None:
        for ex in val_raw[:40]:
            gen = ans.generate(ex["q"], ex["lang"], ex["ctx"], dialog=ex.get("history"))
            ok = ans.faithful(gen, ex["ctx"], ex["q"])
            accepted += ok
            if len(shown) < 12:
                shown.append({"q": ex["q"], "reference": ex["answer"][:300], "generated": gen[:300], "accepted": bool(ok)})
        for s in shown:
            print(f"   Q: {s['q']}\n   model: {s['generated']}\n   reference: {s['reference']}\n   accepted: {s['accepted']}\n")
        print(f"  accepted by the faithfulness check: {accepted}/{min(40, len(val_raw))}")
    save_json(REPORTS / "lm_finetuning.json", {"history": history, "examples": shown,
                                               "accepted": accepted, "n_train": len(train), "n_val": len(val)})
    if history:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], marker="o")
        ax.set_xlabel("epoch"); ax.set_ylabel("validation loss"); ax.set_title("Our LM: answer fine-tuning")
        fig.tight_layout(); fig.savefig(FIGURES / "lm_finetuning.png", dpi=150)
    (Path(__file__).resolve().parents[1] / "logs" / "lm_status.txt").write_text("FINETUNE DONE", encoding="utf-8")


if __name__ == "__main__":
    main()
