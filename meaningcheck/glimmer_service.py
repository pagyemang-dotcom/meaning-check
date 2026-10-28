"""Muse Glimmer as reader and writer: the model-side logic, independent of where it is served.

Reading. A reader's answer is read from the probability of each option letter at the answer position. The answer
position is found by calibration when the service starts: several answer prefixes are tried on a probe question and the
one that puts the most probability on the option letters is kept. If a reading still puts less than half its
probability on the letters (for example because the model wants to reason first), the service lets the model write its
answer, finds the letter it chose, and reads the letter probabilities at that point instead. Every result reports which
route was used and how much probability fell on the letters, so nothing is silently guessed.

Writing. The same model can write the readings, statements, rewrite and question when the Muse Spark API is not
available; `generate` returns the text and the client extracts the JSON.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

LETTERS = "ABCDEFGH"
PREFIXES = ["Answer:", "Answer: ", "\nAnswer:", "The answer is", "Answer: (", ""]
MIN_MASS = 0.5

PROBE = {"system": "You live in the United Kingdom. You work in an office.",
         "user": "Message you received:\n\"\"\"\nThe shop opens at nine in the morning.\n\"\"\"\n\nWhat does the sender "
                 "most likely want you to understand?\n\nA) The shop opens at 9 am.\nB) The shop opens at 9 pm.\n\n"
                 "Answer with the letter of one option.",
         "letters": ["A", "B"], "reasoning": "low"}


class GlimmerService:
    def __init__(self, proc, model, device: str = "cuda", batch: int = 8, gen_tokens: int = 384):
        import torch
        self.torch, self.proc, self.model, self.device = torch, proc, model, device
        self.tok = getattr(proc, "tokenizer", proc)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.batch, self.gen_tokens = batch, gen_tokens
        self.letter_ids: Dict[str, List[int]] = {}
        for L in LETTERS:
            ids = set()
            for v in (L, " " + L):
                enc = self.tok.encode(v, add_special_tokens=False)
                if enc:
                    ids.add(enc[0] if v == L else enc[-1])
            self.letter_ids[L] = sorted(ids)
        self.prefix = "Answer:"
        self.calibration: dict = {}

    # ── prompts ──
    def render(self, messages: List[dict], reasoning: Optional[str] = "low") -> str:
        """Chat template with a generation prompt. Tries list-of-parts content (multimodal processors) and plain
        strings, with and without the reasoning setting."""
        as_parts = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        last = None
        for msgs in (as_parts, messages):
            for kw in (({"reasoning_strength": reasoning},) if reasoning else ()) + ({},):
                try:
                    return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                except Exception as e:                      # template does not accept this form; try the next
                    last = e
        raise RuntimeError(f"chat template failed: {last}")

    def _reader_text(self, it: dict) -> str:
        return self.render([{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}],
                           it.get("reasoning", "low"))

    # ── forward passes ──
    def _letter_logprobs(self, texts: List[str], letters_list: List[List[str]]) -> List[dict]:
        torch = self.torch
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            for k in range(len(chunk)):
                lp, mass = {}, 0.0
                for L in letters_list[i + k]:
                    v = logp[k, self.letter_ids[L]]
                    lp[L] = float(torch.logsumexp(v, dim=0))
                    mass += float(v.exp().sum())
                out.append({"logprobs": lp, "mass": mass})
        return out

    def _generate(self, texts: List[str], max_new_tokens: int) -> List[str]:
        torch = self.torch
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for k in range(len(chunk)):
                out.append(self.tok.decode(gen[k, enc["input_ids"].shape[1]:], skip_special_tokens=False))
        return out

    # ── calibration ──
    def calibrate(self) -> dict:
        base = self._reader_text(PROBE)
        res = self._letter_logprobs([base + p for p in PREFIXES], [PROBE["letters"]] * len(PREFIXES))
        masses = {p: r["mass"] for p, r in zip(PREFIXES, res)}
        self.prefix = max(masses, key=masses.get)
        self.calibration = {"prefix": self.prefix, "masses": masses, "template_tail": base[-300:]}
        return self.calibration

    # ── reading ──
    @staticmethod
    def find_letter(text: str, letters: List[str]) -> Optional[int]:
        """Character position of the chosen option letter in a written answer (last explicit answer wins)."""
        allowed = "".join(letters)
        pats = [rf"(?:answer|choose|choice)(?:\s+(?:is|would be|will be))?[^A-Za-z0-9]{{0,6}}\(?([{allowed}])\b",
                rf"\(([{allowed}])\)", rf"(?:^|\n)\s*([{allowed}])\)", rf"\b([{allowed}])\b[.\s]*$"]
        for pat in pats:
            ms = list(re.finditer(pat, text, flags=re.IGNORECASE))
            ms = [m for m in ms if m.group(1).upper() in letters and m.group(1).isupper()]
            if ms:
                return ms[-1].start(1)
        return None

    def score(self, items: List[dict]) -> List[dict]:
        texts = [self._reader_text(it) for it in items]
        res = self._letter_logprobs([t + self.prefix for t in texts], [it["letters"] for it in items])
        for r in res:
            r["route"] = "direct"
        weak = [i for i, r in enumerate(res) if r["mass"] < MIN_MASS]
        if weak:
            gens = self._generate([texts[i] for i in weak], self.gen_tokens)
            redo_t, redo_i = [], []
            for i, g in zip(weak, gens):
                pos = self.find_letter(g, items[i]["letters"])
                if pos is not None:
                    redo_t.append(texts[i] + g[:pos]); redo_i.append(i)
                else:
                    res[i]["route"] = "no_answer_found"
            for i, r in zip(redo_i, self._letter_logprobs(redo_t, [items[i]["letters"] for i in redo_i])):
                if r["mass"] > res[i]["mass"]:
                    res[i] = dict(r, route="after_written_answer")
        return res

    # ── writing ──
    def generate(self, items: List[dict]) -> List[str]:
        """items: {"messages": [...], "max_new_tokens": int, "reasoning": str}"""
        texts = [self.render(it["messages"], it.get("reasoning", "low")) for it in items]
        mx = max(int(it.get("max_new_tokens", 1500)) for it in items) if items else 0
        return self._generate(texts, mx)
