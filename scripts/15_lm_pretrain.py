"""STEP LM-3 - Pre-training our own language model from scratch (random weights -> Kazakh / Russian text).

The model learns to predict the next token of data/lm/train.bin.  Training is resumable: a checkpoint
(models/lm/pretrain.pt) is written every --save-every steps and when the time budget ends, and the next run
continues from it.  Validation loss / perplexity, sample texts and a learning curve are saved to reports/.

    python scripts/15_lm_pretrain.py [--size small] [--hours 10] [--tokens 1.2e9]
    python scripts/15_lm_pretrain.py --selftest      # 30 steps of a tiny model: checks GPU + code
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from _common import DATA, FIGURES, MODELS, REPORTS, Tee, banner, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.lm.model import GPT, GPTConfig  # noqa: E402
from ml.lm.tokenizer import BOS, LMTokenizer  # noqa: E402

LM = DATA / "lm"
OUT = MODELS / "lm"
SIZES = {   # d_model, layers, heads  (about 17M / 45M / 85M parameters with a 32k vocabulary)
    "tiny": (256, 4, 4),
    "small": (512, 8, 8),
    "base": (640, 12, 10),
}
PROMPTS = ["Сәтбаев университеті", "Жатақханаға орналасу үшін", "Студенты университета", "Қазақстан Республикасы"]


def get_batch(data, B, T, device):
    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, B, T, device, ctx, iters=40):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, B, T, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def samples(model, tok, device, n_tokens=60):
    out = []
    for p in PROMPTS:
        ids = torch.tensor([[BOS] + tok.encode(p)], device=device)
        gen = model.generate(ids, max_new=n_tokens, temperature=0.8, top_k=40)
        out.append(p + tok.decode(gen) if gen else p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--tokens", type=float, default=1.2e9, help="total training tokens (all runs together)")
    ap.add_argument("--hours", type=float, default=10.0, help="time budget of this run")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    Tee("15_lm_pretrain" + ("_selftest" if args.selftest else ""))
    banner("STEP LM-3. Pre-training our language model (from scratch)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__} | device: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
             if device == "cuda" else "  (!) no GPU - training would be very slow"))
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = LMTokenizer.load(OUT / "tokenizer.json")
    train = np.memmap(LM / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(LM / "val.bin", dtype=np.uint16, mode="r")
    print(f"  data: {len(train) / 1e6:.0f}M train tokens, {len(val) / 1e6:.2f}M validation tokens, vocab {len(tok)}")
    if args.selftest:
        args.size, args.batch, args.accum, args.ctx = "tiny", 4, 1, 128
    d, L, H = SIZES[args.size]
    cfg = GPTConfig(vocab_size=len(tok), d_model=d, n_layers=L, n_heads=H, ctx=args.ctx)
    model = GPT(cfg).to(device)
    tokens_per_step = args.batch * args.accum * args.ctx
    total_steps = 30 if args.selftest else int(args.tokens / tokens_per_step)
    warmup = min(1000, max(10, total_steps // 50))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    ckpt_path = OUT / ("selftest.pt" if args.selftest else "pretrain.pt")
    step, history = 0, []
    if ckpt_path.exists() and not args.selftest:
        ck = torch.load(ckpt_path, map_location=device)
        if ck["config"] == cfg.to_dict():
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            step, history = ck["step"], ck.get("history", [])
            print(f"  resumed from step {step}")
        else:
            print("  (!) the saved checkpoint has another size - starting a new model")
    print(f"  model: {model.n_params() / 1e6:.1f}M parameters | {L} layers, d={d}, heads={H}, context {args.ctx}")
    print(f"  {tokens_per_step} tokens per step | steps: {step} -> {total_steps} | precision: {'bf16' if bf16 else 'fp16'}")

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        p = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def save():
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg.to_dict(),
                    "history": history}, tmp)
        tmp.replace(ckpt_path)
        if not args.selftest:          # the weights alone, for the assistant
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, OUT / "lm_base.pt")

    t_start, t_log, tok_seen = time.time(), time.time(), 0
    model.train()
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, args.ctx, device)
            with ctx:
                _, loss = model(x, y)
            scaler.scale(loss / args.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_seen += tokens_per_step
        if not math.isfinite(loss.item()):
            print("  (!) loss is not finite - stopping; the last checkpoint is kept"); break
        if step % 50 == 0 or args.selftest:
            el = time.time() - t_log
            print(f"  step {step}/{total_steps} | loss {loss.item():.3f} | lr {lr_at(step):.2e} | "
                  f"{tok_seen / max(el, 1e-9) / 1e3:.1f}k tokens/s", flush=True)
            t_log, tok_seen = time.time(), 0
        if step % args.eval_every == 0 or step == total_steps:
            vl = evaluate(model, val, args.batch, args.ctx, device, ctx, iters=10 if args.selftest else 40)
            history.append({"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4),
                            "val_ppl": round(math.exp(min(vl, 20)), 2), "hours": round((time.time() - t_start) / 3600, 3)})
            print(f"  == validation: loss {vl:.3f}, perplexity {math.exp(min(vl, 20)):.1f}")
            for s in samples(model, tok, device):
                print(f"     > {s[:200]}")
        if step % args.save_every == 0:
            save()
        if (time.time() - t_start) / 3600 > args.hours:
            print(f"  time budget of {args.hours} h used - saving; the next run continues from step {step}")
            break
    save()
    save_json(REPORTS / ("lm_selftest.json" if args.selftest else "lm_pretraining.json"),
              {"config": cfg.to_dict(), "params_m": round(model.n_params() / 1e6, 1), "step": step,
               "total_steps": total_steps, "tokens_per_step": tokens_per_step, "history": history})
    if history and not args.selftest:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot([h["step"] for h in history], [h["val_loss"] for h in history], marker="o", label="validation")
        ax.plot([h["step"] for h in history], [h["train_loss"] for h in history], alpha=0.6, label="train")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Our language model: pre-training"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / "lm_pretraining.png", dpi=150)
    done = step >= total_steps
    (Path(__file__).resolve().parents[1] / "logs" / "lm_status.txt").write_text(
        ("PRETRAIN DONE" if done else f"PRETRAIN PAUSED at step {step}/{total_steps}") if not args.selftest
        else "SELFTEST OK", encoding="utf-8")
    print("  " + ("SELFTEST OK" if args.selftest else ("pre-training finished" if done else "paused - run again to continue")))


if __name__ == "__main__":
    main()
