"""Our language model inside the assistant: writes the answer from the sentences our retrieval found.

    question + found sentences  ->  our fine-tuned GPT (models/lm/lm_rag.pt)  ->  answer in the question's language

Safety net (no invented facts): the generated text is accepted only if
  * almost all of its content words are found in the context (or are common question words),
  * every number it contains is present in the context;
otherwise the assistant keeps the composer's answer built from the original sentences.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ..text import fold, stems
from .tokenizer import ANS_TOK, BOS, CTX_TOK, EOS, LANG_TOK, NL, Q_TOK, SEP_TOK, LMTokenizer

NO_INFO_RE = re.compile(r"(мәлімет жоқ|нет информации|do not contain information)", re.I)


class LMAnswerer:
    def __init__(self, model, tok: LMTokenizer, device: str, bilex: Optional[dict] = None):
        self.model, self.tok, self.device = model, tok, device
        self.bilex = bilex or {}                                 # our kk <-> ru stem dictionary

    @classmethod
    def load(cls, models_dir) -> Optional["LMAnswerer"]:
        d = Path(models_dir) / "lm"
        if not (d / "lm_rag.pt").exists() or not (d / "tokenizer.json").exists():
            return None
        try:
            import torch
            from .model import GPT, GPTConfig
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ck = torch.load(d / "lm_rag.pt", map_location=device)
            model = GPT(GPTConfig(**ck["config"])).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            if device == "cuda":
                model.half()
            import json
            bilex = json.loads((d / "bilex.json").read_text(encoding="utf-8")) if (d / "bilex.json").exists() else {}
            return cls(model, LMTokenizer.load(d / "tokenizer.json"), device, bilex)
        except Exception as e:                                   # no torch / broken file: the assistant still works
            print(f"[lm] language model not loaded: {e}")
            return None

    def dialog_ids(self, dialog) -> List[int]:
        """Previous turns of the conversation (the model remembers what was said): <|question|> q <|answer|> a <|sep|>."""
        ids: List[int] = []
        for q, a in (dialog or [])[-2:]:
            ids += [Q_TOK] + self.tok.encode(q)[:40] + [ANS_TOK] + self.tok.encode(a)[:70] + [SEP_TOK]
        return ids[-180:]

    def prompt_ids(self, question: str, lang: str, context: List[str], max_ctx_tokens: int = 380,
                   dialog=None) -> List[int]:
        ctx_ids: List[int] = []
        for s in context:
            ids = self.tok.encode(s) + [NL]
            if len(ctx_ids) + len(ids) > max_ctx_tokens:
                break
            ctx_ids += ids
        q = self.tok.encode(question)[:60]
        hist = self.dialog_ids(dialog)
        return [BOS, LANG_TOK.get(lang, LANG_TOK["ru"])] + hist + [Q_TOK] + q + [CTX_TOK] + ctx_ids + [ANS_TOK]

    def generate(self, question: str, lang: str, context: List[str], max_new: int = 160, dialog=None) -> str:
        import torch
        ids = self.prompt_ids(question, lang, context, max_ctx_tokens=380 - min(180, 8 * len(dialog or [])),
                              dialog=dialog)
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model.generate(x, max_new=max_new, temperature=0.0, repetition_penalty=1.2, stop_ids=(EOS,))
        return self.tok.decode(out).strip()

    def faithful(self, answer: str, context: List[str], question: str, min_support: float = 0.85) -> bool:
        if not answer or len(answer) < 15 or NO_INFO_RE.search(answer):
            return False
        ctx = " ".join(context)
        nums_a = set(re.findall(r"\d+(?:[.,]\d+)?", answer))
        if nums_a - set(re.findall(r"\d+(?:[.,]\d+)?", ctx)):
            return False                                          # a number that is not in the documents
        st_a = [s for s in stems(answer) if len(s) >= 4]
        if not st_a:
            return False
        allowed = set(stems(ctx)) | set(stems(question))
        for st in list(allowed):                                 # a Kazakh answer from a Russian document
            allowed.update(self.bilex.get(st, []))
        pref = {s[:4] for s in allowed}
        ok = sum(1 for s in st_a if s in allowed or s[:4] in pref)
        return ok / len(st_a) >= min_support

    def answer(self, question: str, lang: str, context: List[str], dialog=None) -> Optional[str]:
        try:
            text = self.generate(question, lang, context, dialog=dialog)
        except Exception as e:
            print(f"[lm] generation failed: {e}")
            return None
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text if self.faithful(text, context, question) else None
