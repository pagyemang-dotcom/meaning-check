"""The Muse Glimmer service logic, run on CPU with a tiny tokenizer and tiny models (no download, no GPU)."""
import json
import threading
import time
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from meaningcheck.glimmer_service import PROBE, GlimmerService

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{% if m['content'] is string %}{{ m['content'] }}"
            "{% else %}{{ m['content'][0]['text'] }}{% endif %}\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [PROBE["system"], PROBE["user"], "Answer: A B C D E F G H", "The answer is (A)", "Answer: B",
              "Message you received What does the sender most likely want you to understand"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<eos>", "<pad>",
                                                                                         "<|system|>", "<|user|>",
                                                                                         "<|assistant|>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<pad>",
                                             unk_token="<unk>")
    t.chat_template = TEMPLATE
    return t


class LetterModel(torch.nn.Module):
    """Puts most next-token probability on one letter, except right after `blind_after` text (to force the written-
    answer route), and writes "Answer: <letter>" when asked to generate."""
    def __init__(self, tok, letter="B", blind=False):
        super().__init__()
        self.tok, self.letter, self.blind = tok, letter, blind
        self.V = len(tok)
        self.target = tok.encode(" " + letter, add_special_tokens=False)[-1]
        self.target2 = tok.encode(letter, add_special_tokens=False)[0]

    def forward(self, input_ids=None, attention_mask=None, **kw):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.V)
        for b in range(B):
            text = self.tok.decode(input_ids[b], skip_special_tokens=False)
            written = text.rstrip().endswith("Answer:") and "Let me think" in text
            if self.blind and not written:
                logits[b, -1, self.tok.eos_token_id] = 20.0           # wants to reason, not answer
            else:
                logits[b, -1, self.target] = 10.0
                logits[b, -1, self.target2] = 9.0
        return types.SimpleNamespace(logits=logits)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
        add = self.tok.encode(f" Let me think. Answer: {self.letter}", add_special_tokens=False)
        return torch.cat([input_ids, torch.tensor([add] * input_ids.shape[0])], dim=1)


class KeywordModel(LetterModel):
    """Answers with the letter of the option that contains a keyword: an order-independent reader."""
    def __init__(self, tok, keyword):
        super().__init__(tok, "A"); self.keyword = keyword

    def forward(self, input_ids=None, attention_mask=None, **kw):
        import re
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.V)
        for b in range(B):
            text = self.tok.decode(input_ids[b], skip_special_tokens=False)
            m = [x for x in re.findall(r"\n([A-H])\) ([^\n]*)", text) if self.keyword in x[1]]
            L = m[-1][0] if m else "A"
            logits[b, -1, self.tok.encode(" " + L, add_special_tokens=False)[-1]] = 10.0
        return types.SimpleNamespace(logits=logits)


def test_render_and_calibrate_direct_route():
    tok = tiny_tokenizer()
    svc = GlimmerService(tok, LetterModel(tok, "A"), device="cpu", batch=3)
    assert svc.render([{"role": "user", "content": "hi"}]).endswith("<|assistant|>")
    cal = svc.calibrate()
    assert cal["masses"][cal["prefix"]] > 0.9
    res = svc.score([PROBE, dict(PROBE, letters=["A", "B", "C"]), PROBE, PROBE])     # crosses a batch boundary
    assert all(r["route"] == "direct" and r["mass"] > 0.9 for r in res)
    assert max(res[0]["logprobs"], key=res[0]["logprobs"].get) == "A"


def test_written_answer_route_when_model_wants_to_reason():
    tok = tiny_tokenizer()
    svc = GlimmerService(tok, LetterModel(tok, "B", blind=True), device="cpu")
    svc.calibrate()
    r = svc.score([PROBE])[0]
    assert r["route"] == "after_written_answer" and r["mass"] > 0.9
    assert max(r["logprobs"], key=r["logprobs"].get) == "B"


def test_find_letter():
    f = GlimmerService.find_letter
    assert f("I think the answer is (B).", ["A", "B"]) is not None
    t = "Option A seems odd. Final answer: B"
    assert t[f(t, ["A", "B"])] == "B"
    assert f("no letter here", ["A", "B"]) is None


def test_real_tiny_transformer_plumbing():
    """Left padding, batching and generation with an actual transformers model (random weights)."""
    tok = tiny_tokenizer()
    cfg = transformers.GPT2Config(vocab_size=len(tok), n_layer=1, n_embd=32, n_head=2, n_positions=1024,
                                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    svc = GlimmerService(tok, model, device="cpu", batch=2, gen_tokens=4)
    svc.calibrate()
    res = svc.score([PROBE, dict(PROBE, user=PROBE["user"] + " Please.")])
    assert len(res) == 2 and all(set(r["logprobs"]) == {"A", "B"} for r in res)
    out = svc.generate([{"messages": [{"role": "user", "content": "hello"}], "max_new_tokens": 3}])
    assert isinstance(out[0], str)


def test_end_to_end_over_http(monkeypatch, tmp_path):
    """The live client path: engine -> HTTP -> service, with readers that follow the task."""
    import uvicorn
    from fastapi import FastAPI
    tok = tiny_tokenizer()
    svc = GlimmerService(tok, KeywordModel(tok, "agenda"), device="cpu")
    svc.calibrate()
    api = FastAPI()

    @api.post("/score")
    def score(body: dict):
        return {"results": svc.score(body.get("items", []))}

    @api.post("/generate")
    def generate(body: dict):
        return {"texts": ['Thinking {not json}. {"rewrite": "Please add the budget item to Friday\'s agenda."}']}
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=8799, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True); th.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    monkeypatch.setenv("GLIMMER_URL", "http://127.0.0.1:8799/score")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    from meaningcheck.backends import make_backend
    from meaningcheck.engine import Engine
    from meaningcheck.examples import Library
    from meaningcheck.writer import Writer
    readers, writer = make_backend("glimmer", cache_dir=str(tmp_path))
    eng = Engine(readers, Writer(writer, Library()))
    case = Library().get("table")
    rep = eng.check(case.message, case=case, attribution=False)
    assert len(rep.before) == 16 and all(r.understood for r in rep.before)   # it always picks the agenda reading
    assert all(abs(sum(r.probs.values()) - 1) < 1e-9 for r in rep.before)
    assert rep.holdout["after"] == 1.0 and (tmp_path / "glimmer.jsonl").exists()   # readings cached
    assert rep.certificate["writer"] == "Muse Glimmer"
    server.should_exit = True
