"""Translator: our trained Transformer + BPE, with a guard against changed numbers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

from .bpe import BPE, BOS, EOS, LANG_TOKEN

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def same_numbers(a: str, b: str) -> bool:
    norm = lambda s: sorted(n.replace(",", ".").rstrip(".") for n in _NUM_RE.findall(s))
    return norm(a) == norm(b)


class Translator:
    def __init__(self, model, bpe: BPE, device: str = "cpu", info: Optional[dict] = None):
        self.model, self.bpe, self.device = model, bpe, device
        self.info = info or {}
        self.cache = {}

    @classmethod
    def load(cls, d="models/nmt", device: Optional[str] = None) -> Optional["Translator"]:
        """None when the model is not trained or PyTorch is not installed (the assistant then works without it)."""
        d = Path(d)
        if not (d / "model.pt").exists() or not (d / "bpe.json").exists():
            return None
        try:
            import torch
            from .model import NMTConfig, Transformer
        except Exception:
            return None
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(d / "model.pt", map_location=device)
        model = Transformer(NMTConfig(**ck["config"])).to(device).eval()
        model.load_state_dict(ck["model"])
        info = json.loads((d / "info.json").read_text(encoding="utf-8")) if (d / "info.json").exists() else {}
        return cls(model, BPE.load(d / "bpe.json"), device, info)

    def translate(self, texts: Sequence[str], tgt_lang: str, beam: int = 4, max_src: int = 200) -> List[str]:
        import torch
        out = []
        for t in texts:
            key = (t, tgt_lang)
            if key not in self.cache:
                ids = [LANG_TOKEN[tgt_lang]] + self.bpe.encode(t)[:max_src] + [EOS]
                src = torch.tensor([ids], device=self.device)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    hyp = self.model.beam(src, beam=beam, max_len=int(len(ids) * 1.6) + 10)
                self.cache[key] = self.bpe.decode(hyp)
            out.append(self.cache[key])
        return out

    def safe_translate(self, text: str, tgt_lang: str) -> Optional[str]:
        """Translation, or None when it is not trustworthy (numbers changed, empty or degenerate output)."""
        try:
            tr = self.translate([text], tgt_lang)[0]
        except Exception:
            return None
        if not tr or not same_numbers(text, tr):
            return None
        words = tr.split()
        if len(words) >= 6 and len(set(words)) / len(words) < 0.5:       # "the the the ..." loops
            return None
        if not (0.4 <= len(tr) / max(1, len(text)) <= 2.5):
            return None
        return tr
