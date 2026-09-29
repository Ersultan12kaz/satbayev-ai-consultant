"""Our own generative language model (decoder-only Transformer, GPT architecture), trained from scratch.

No pre-trained weights are used: the model starts from random numbers and learns Kazakh / Russian from the
corpus prepared in scripts/13_lm_corpus.py, then learns to answer from the documents in
scripts/16_lm_finetune.py.

  token embedding (tied with the output layer) + rotary position encoding (RoPE)
  N x [ RMSNorm -> causal multi-head self-attention -> RMSNorm -> SwiGLU feed-forward ]  (pre-norm, residual)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ctx: int = 512
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.w


def rope_cache(ctx: int, hd: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(ctx).float()
    f = torch.outer(t, inv)                       # (ctx, hd/2)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):                      # x: (B, H, T, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h, self.hd = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        offset = 0 if past is None else past[0].shape[2]
        q = apply_rope(q, cos[offset:], sin[offset:])
        k = apply_rope(k, cos[offset:], sin[offset:])
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h = int(8 * d / 3 / 64 + 0.999) * 64
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n1, self.att = RMSNorm(c.d_model), Attention(c)
        self.n2, self.ff = RMSNorm(c.d_model), SwiGLU(c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, past=None):
        a, kv = self.att(self.n1(x), cos, sin, past)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, kv


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.emb.weight                     # tied input / output embeddings
        cos, sin = rope_cache(c.ctx * 2, c.d_model // c.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():                   # GPT-2 style scaled init of residual outputs
            if n.endswith("out.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        x = self.emb(idx)
        for b in self.blocks:
            x, _ = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        if loss_mask is not None:                               # fine-tuning: learn only the answer tokens
            m = loss_mask.view(-1).float()
            return logits, (loss * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss.mean()

    @torch.no_grad()
    def generate(self, idx, max_new: int = 200, temperature: float = 0.7, top_k: int = 40,
                 repetition_penalty: float = 1.15, stop_ids=(2,)):
        """Sampling with a key/value cache. idx: (1, T) prompt."""
        self.eval()
        idx = idx[:, -self.c.ctx:]
        past = [None] * len(self.blocks)
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x, past[i] = b(x, self.cos, self.sin)
        out = []
        for _ in range(max_new):
            logits = self.head(self.norm(x[:, -1:]))[:, -1].float()
            if repetition_penalty != 1.0 and out:
                seen = torch.tensor(sorted(set(out[-100:])), device=logits.device)
                sc = logits[0, seen]
                logits[0, seen] = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
            if temperature <= 0:
                nxt = int(logits.argmax(-1))
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, -1:]] = -float("inf")
                nxt = int(torch.multinomial(F.softmax(logits, -1), 1))
            if nxt in stop_ids:
                break
            out.append(nxt)
            if idx.shape[1] + len(out) >= self.c.ctx * 2:
                break
            x = self.emb(torch.tensor([[nxt]], device=idx.device))
            for i, b in enumerate(self.blocks):
                x, past[i] = b(x, self.cos, self.sin, past[i])
        return out
