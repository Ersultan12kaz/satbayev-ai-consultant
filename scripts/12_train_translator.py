"""STEP 12 - Training our Transformer translation model from scratch (GPU).

    python scripts/12_train_translator.py --hours 6          # continues automatically from the last checkpoint

Every 2000 steps: validation loss, BLEU / chrF (our metrics) per direction on held-out
university documents and KazParC sentences, checkpoint (models/nmt/checkpoint.pt).
The best model by chrF is saved to models/nmt/model.pt and used by the assistant.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict

import numpy as np

from _common import DATA, FIGURES, LOGS, MODELS, REPORTS, Tee, banner, save_json

NMT = MODELS / "nmt"
TR = DATA / "translation"


def batches(src, so, tgt, to, max_tokens, rnd):
    """Length-bucketed batches of about max_tokens tokens (src+tgt, padded)."""
    n = len(so) - 1
    lens = np.maximum(so[1:] - so[:-1], to[1:] - to[:-1])
    rs = np.random.RandomState(rnd.randrange(1 << 30))
    order = np.lexsort((rs.rand(n), lens)).tolist()          # by length, random order among equal lengths
    chunks = [order[i:i + 20000] for i in range(0, n, 20000)]
    out = []
    for ch in chunks:
        cur, cur_max = [], 0
        for i in ch:
            L = int(lens[i]) + 1
            if cur and (max(cur_max, L) * (len(cur) + 1) * 2 > max_tokens):
                out.append(cur); cur, cur_max = [], 0
            cur.append(i); cur_max = max(cur_max, L)
        if cur:
            out.append(cur)
    rnd.shuffle(out)
    return out


def to_tensor(ids_list, torch, device, pad=0):
    L = max(len(x) for x in ids_list)
    a = np.full((len(ids_list), L), pad, dtype=np.int64)
    for i, x in enumerate(ids_list):
        a[i, :len(x)] = x
    return torch.from_numpy(a).to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-tokens", type=int, default=7000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=2000)
    args = ap.parse_args()
    Tee("12_translator")
    import torch
    from ml.nmt.bpe import BPE, BOS, EOS, LANG_TOKEN, PAD
    from ml.nmt.metrics import chrf, corpus_bleu
    from ml.nmt.model import NMTConfig, Transformer, label_smoothed_loss

    banner("STEP 12. Training our Transformer translation model (kk <-> ru <-> en)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                   "  (no GPU found - training on CPU is very slow)"))
    bpe = BPE.load(NMT / "bpe.json")
    d = np.load(TR / "train.npz")
    src, so, tgt, to = d["src"], d["src_off"], d["tgt"], d["tgt_off"]
    valid = [json.loads(l) for l in open(TR / "valid.jsonl", encoding="utf-8")]
    rnd_v = random.Random(1)
    rnd_v.shuffle(valid)
    by_dir = defaultdict(list)
    for v in valid:
        by_dir[(v["s"], v["t"])].append(v)
    eval_set = [x for k in sorted(by_dir) for x in by_dir[k][:60]]
    print(f"  training examples: {len(so) - 1}, validation sentences for BLEU/chrF: {len(eval_set)}")

    cfg = NMTConfig(vocab_size=len(bpe), d_model=args.d_model, n_heads=8, enc_layers=args.layers,
                    dec_layers=args.layers, d_ff=4 * args.d_model)
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg.to_dict()}  parameters: {n_params / 1e6:.1f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    sched = lambda step: 2.0 * cfg.d_model ** -0.5 * min((step + 1) ** -0.5, (step + 1) * args.warmup ** -1.5)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))
    step, epoch, best, history = 0, 0, -1.0, []
    ck_path = NMT / "checkpoint.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device)
        if ck.get("config") == cfg.to_dict():
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); lr_sched.load_state_dict(ck["sched"])
            step, epoch, best, history = ck["step"], ck["epoch"], ck["best"], ck["history"]
            print(f"  resumed from step {step} (epoch {epoch}, best chrF {best:.2f})")

    def evaluate():
        model.eval()
        hyps, refs, dirs = [], [], []
        losses = []
        with torch.no_grad():
            for i in range(0, len(eval_set), 32):
                part = eval_set[i:i + 32]
                s = [[LANG_TOKEN[v["t"]]] + bpe.encode(v["src"])[:200] + [EOS] for v in part]
                t = [bpe.encode(v["tgt"])[:200] + [EOS] for v in part]
                S = to_tensor(s, torch, device); T = to_tensor(t, torch, device)
                T_in = torch.cat([torch.full((T.shape[0], 1), BOS, device=device), T[:, :-1]], 1)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
                    loss, _ = label_smoothed_loss(model(S, T_in), T, eps=0.0)
                    out = model.greedy(S, max_len=int(S.shape[1] * 1.6) + 10)
                losses.append(float(loss))
                hyps += [bpe.decode(o) for o in out]; refs += [v["tgt"] for v in part]
                dirs += [(v["s"], v["t"]) for v in part]
        model.train()
        res = {"valid_loss": float(np.mean(losses)), "bleu": corpus_bleu(hyps, refs), "chrf": chrf(hyps, refs)}
        for k in sorted(set(dirs)):
            idx = [i for i, x in enumerate(dirs) if x == k]
            res[f"chrf_{k[0]}-{k[1]}"] = chrf([hyps[i] for i in idx], [refs[i] for i in idx])
        res["examples"] = [{"dir": f"{dirs[i][0]}-{dirs[i][1]}", "src": eval_set[i]["src"], "hyp": hyps[i],
                            "ref": refs[i]} for i in range(0, len(hyps), max(1, len(hyps) // 6))][:6]
        return res

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    model.train()
    stop = False
    while epoch < args.epochs and not stop:
        rnd = random.Random(epoch)
        bl = batches(src, so, tgt, to, args.batch_tokens, rnd)
        run_loss, run_tok, t_log = 0.0, 0, time.time()
        for bi, b in enumerate(bl):
            S = to_tensor([src[so[i]:so[i + 1]] for i in b], torch, device)
            T = to_tensor([tgt[to[i]:to[i + 1]] for i in b], torch, device)
            T_in = torch.cat([torch.full((T.shape[0], 1), BOS, device=device), T[:, :-1]], 1)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(S, T_in)
            loss, ntok = label_smoothed_loss(logits, T)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); lr_sched.step()
            step += 1
            run_loss += float(loss) * int(ntok); run_tok += int(ntok)
            if step % 200 == 0:
                el = time.time() - t_log
                print(f"  epoch {epoch + 1} step {step} ({bi + 1}/{len(bl)})  loss={run_loss / max(1, run_tok):.3f}  "
                      f"lr={lr_sched.get_last_lr()[0]:.2e}  {run_tok / max(1e-9, el):.0f} tok/s", flush=True)
                (LOGS / "nmt_status.txt").write_text(f"epoch {epoch + 1} step {step} loss {run_loss / max(1, run_tok):.3f}",
                                                     encoding="utf-8")
                history.append({"step": step, "train_loss": run_loss / max(1, run_tok)})
                run_loss, run_tok, t_log = 0.0, 0, time.time()
            if step % args.eval_every == 0 or time.time() > deadline:
                r = evaluate()
                history.append({"step": step, **{k: v for k, v in r.items() if k != "examples"}})
                print(f"\n  == validation @ step {step}: loss={r['valid_loss']:.3f} BLEU={r['bleu']:.2f} "
                      f"chrF={r['chrf']:.2f}  " + "  ".join(f"{k[5:]}={v:.1f}" for k, v in r.items()
                                                            if k.startswith("chrf_")))
                for ex in r["examples"][:3]:
                    print(f"     [{ex['dir']}] {ex['src'][:90]}\n        -> {ex['hyp'][:90]}")
                if r["chrf"] > best:
                    best = r["chrf"]
                    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, NMT / "model.pt")
                    save_json(NMT / "info.json", {"step": step, "epoch": epoch + 1, **{k: v for k, v in r.items()}})
                    print(f"     new best model saved (chrF {best:.2f})")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": lr_sched.state_dict(),
                            "step": step, "epoch": epoch, "best": best, "history": history,
                            "config": cfg.to_dict()}, ck_path)
                print()
                if time.time() > deadline:
                    print(f"  time limit of {args.hours} h reached - run again to continue training")
                    stop = True
                    break
        else:
            epoch += 1
    save_json(REPORTS / "translator_metrics.json", {"best_chrf": best, "steps": step, "history": history})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = [h for h in history if "train_loss" in h]; va = [h for h in history if "chrf" in h]
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
        ax[0].plot([h["step"] for h in tr], [h["train_loss"] for h in tr], label="train")
        ax[0].plot([h["step"] for h in va], [h["valid_loss"] for h in va], label="validation")
        ax[0].set_title("Translator loss"); ax[0].set_xlabel("step"); ax[0].legend()
        ax[1].plot([h["step"] for h in va], [h["chrf"] for h in va], label="chrF")
        ax[1].plot([h["step"] for h in va], [h["bleu"] for h in va], label="BLEU")
        ax[1].set_title("Translation quality (validation)"); ax[1].set_xlabel("step"); ax[1].legend()
        fig.tight_layout(); fig.savefig(FIGURES / "translator_training.png", dpi=150)
    except Exception as e:
        print(f"  plot failed: {e}")
    (LOGS / "nmt_status.txt").write_text(f"DONE step {step} best chrF {best:.2f}", encoding="utf-8")
    print(f"  finished: {step} steps, best chrF {best:.2f}. Model: models/nmt/model.pt")


if __name__ == "__main__":
    main()
