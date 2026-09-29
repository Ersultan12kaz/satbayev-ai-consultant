"""Transformer translation model (Vaswani et al., 2017) written by us with PyTorch primitives.

One multilingual model translates between Kazakh, Russian and English: the source
sentence starts with a target-language token (<2kk>, <2ru>, <2en>), the vocabulary is the
joint BPE of ml/nmt/bpe.py, input/output embeddings are tied.  Pre-LayerNorm layers,
sinusoidal positions, label-smoothed cross-entropy, greedy and beam-search decoding.
All weights are trained from scratch (scripts/12_train_translator.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bpe import BOS, EOS, PAD


@dataclass
class NMTConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 256

    def to_dict(self):
        return asdict(self)


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, mem, mask=None, causal: bool = False):
        B, T, D = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(mem).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(B, S, self.h, self.dk).transpose(1, 2)
        # softmax(QK^T / sqrt(dk) + mask) V  (fused kernel when available)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, d, ff, dropout):
        super().__init__()
        self.l1 = nn.Linear(d, ff); self.l2 = nn.Linear(ff, d); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.drop(F.gelu(self.l1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, c: NMTConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d_model); self.att = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.n2 = nn.LayerNorm(c.d_model); self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, src_mask):
        h = self.n1(x)
        x = x + self.drop(self.att(h, h, src_mask))
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: NMTConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d_model); self.self_att = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.n2 = nn.LayerNorm(c.d_model); self.cross = MultiHeadAttention(c.d_model, c.n_heads, c.dropout)
        self.n3 = nn.LayerNorm(c.d_model); self.ff = FeedForward(c.d_model, c.d_ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, y, mem, src_mask, causal=True):
        h = self.n1(y)
        y = y + self.drop(self.self_att(h, h, None, causal=causal))
        y = y + self.drop(self.cross(self.n2(y), mem, src_mask))
        return y + self.drop(self.ff(self.n3(y)))


def sinusoid(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Transformer(nn.Module):
    def __init__(self, c: NMTConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, 0, c.d_model ** -0.5)
        self.register_buffer("pe", sinusoid(c.max_len + 2, c.d_model), persistent=False)
        self.enc = nn.ModuleList([EncoderLayer(c) for _ in range(c.enc_layers)])
        self.dec = nn.ModuleList([DecoderLayer(c) for _ in range(c.dec_layers)])
        self.enc_norm = nn.LayerNorm(c.d_model); self.dec_norm = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def embed(self, ids, offset: int = 0):
        return self.drop(self.emb(ids) * math.sqrt(self.c.d_model) + self.pe[offset:offset + ids.shape[1]])

    @staticmethod
    def src_mask(src):                       # (B,1,1,S) boolean mask: True = real token, False = padding
        return (src != PAD)[:, None, None, :]

    def encode(self, src):
        mask = self.src_mask(src)
        x = self.embed(src)
        for layer in self.enc:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt_in, mem, mask):
        y = self.embed(tgt_in)
        for layer in self.dec:
            y = layer(y, mem, mask)
        return self.dec_norm(y) @ self.emb.weight.T          # tied output projection

    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def greedy(self, src, max_len: int = 160) -> List[List[int]]:
        max_len = min(max_len, self.c.max_len)              # positions are limited to max_len + 2
        src = src[:, : self.c.max_len]
        mem, mask = self.encode(src)
        B = src.shape[0]
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, mem, mask)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return [[t for t in row[1:].tolist() if t not in (PAD,)] for row in ys]

    @torch.no_grad()
    def beam(self, src, beam: int = 4, max_len: int = 160, alpha: float = 0.6) -> List[int]:
        """Beam search for one sentence (src: 1 x S)."""
        max_len = min(max_len, self.c.max_len)
        src = src[:, : self.c.max_len]
        mem, mask = self.encode(src)
        hyps = [([BOS], 0.0)]
        finished = []
        for _ in range(max_len):
            seqs = torch.tensor([h[0] for h in hyps], device=src.device)
            logp = F.log_softmax(self.decode(seqs, mem.expand(len(hyps), -1, -1),
                                             mask.expand(len(hyps), -1, -1, -1))[:, -1].float(), -1)
            cand = []
            for hi, (seq, score) in enumerate(hyps):
                top = torch.topk(logp[hi], beam)
                for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((seq + [tok], score + lp))
            cand.sort(key=lambda x: -x[1])
            hyps = []
            for seq, score in cand:
                if seq[-1] == EOS:
                    finished.append((seq, score / (((5 + len(seq)) / 6) ** alpha)))
                else:
                    hyps.append((seq, score))
                if len(hyps) == beam:
                    break
            if len(finished) >= beam or not hyps:
                break
        if not finished:
            finished = [(s, sc / (((5 + len(s)) / 6) ** alpha)) for s, sc in hyps]
        best = max(finished, key=lambda x: x[1])[0]
        return [t for t in best[1:] if t not in (EOS, PAD)]


def label_smoothed_loss(logits, target, eps: float = 0.1):
    logp = F.log_softmax(logits.float(), -1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(-1)
    loss = (1 - eps) * nll + eps * smooth
    mask = target != PAD
    return (loss * mask).sum() / mask.sum().clamp(min=1), mask.sum()
