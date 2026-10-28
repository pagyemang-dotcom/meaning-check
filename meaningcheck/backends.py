"""Model access.

Readers need the probability of each option letter, read in one forward pass. The Muse Spark API refuses token
probabilities (it is a reasoning model), so readers run on the open-weight Muse Glimmer served from Modal
(deploy/modal_glimmer.py). Muse Spark writes: readings, statements, the rewrite and the clarifying question.

Backends:
  sim      deterministic simulated readers and writer, for tests and offline demos (built-in examples only)
  glimmer  Muse Glimmer reader service (GLIMMER_URL) + Muse Spark writer (MODEL_API_KEY)
  spark    readers by sampling Muse Spark several times per question (fallback when Modal is unavailable)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

META_BASE = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
SPARK_MODEL = os.environ.get("SPARK_MODEL", "muse-spark-1.3")
VOICE_MODEL = os.environ.get("VOICE_MODEL", "muse-voice-transcribe")


@dataclass
class Query:
    system: str
    user: str
    letters: List[str]
    reasoning: str = "low"
    meta: dict = field(default_factory=dict)      # used only by the simulated backend


def _post(url: str, payload: dict, headers: dict, timeout: float = 600.0, retries: int = 3) -> dict:
    data = json.dumps(payload).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
            last = f"HTTP {e.code} {detail}".strip()
            if 400 <= e.code < 500 and e.code not in (408, 429):     # a refusal: retrying will not help
                break
            time.sleep(2 * (k + 1))
        except Exception as e:                          # network hiccup or cold start: back off and retry
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"request to {url} failed: {last}")


def _softmax(xs: Dict[str, float]) -> Dict[str, float]:
    m = max(xs.values())
    e = {k: math.exp(v - m) for k, v in xs.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def extract_json(text: str, want: tuple = ()) -> dict:
    """The last complete JSON object in a model reply that has the wanted keys (tolerates reasoning text, code fences
    and prose around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    found = []
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        depth, in_str, esc = 0, False, False
        nxt = start + 1
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:j + 1])
                        if isinstance(obj, dict):
                            found.append(obj)
                            nxt = j + 1                    # skip the objects nested inside this one
                    except Exception:
                        pass
                    break
        i = nxt
    good = [o for o in found if all(k in o for k in want)]
    if not good:
        raise ValueError("no JSON object with " + (", ".join(want) or "content") + " in reply")
    return good[-1]


# ───────────────────────────── writer: Muse Spark ─────────────────────────────
class SparkWriter:
    def __init__(self, api_key: Optional[str] = None, model: str = SPARK_MODEL):
        self.key = api_key or os.environ.get("MODEL_API_KEY", "")
        if not self.key:
            raise RuntimeError("MODEL_API_KEY is not set")
        self.model = model

    name = "Muse Spark"

    def chat(self, messages: List[dict], max_tokens: int = 4000, json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if json_mode and getattr(self, "_json_ok", True):
            try:
                out = _post(f"{META_BASE}/chat/completions", dict(body, response_format={"type": "json_object"}),
                            {"Authorization": f"Bearer {self.key}"}, retries=1)
                return out["choices"][0]["message"]["content"] or ""
            except Exception:
                self._json_ok = False                     # JSON mode not accepted: ask in words instead
        out = _post(f"{META_BASE}/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
        return out["choices"][0]["message"]["content"] or ""

    def models(self) -> List[str]:
        req = urllib.request.Request(f"{META_BASE}/models", headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return [m.get("id") for m in json.loads(r.read().decode()).get("data", [])]

    def json(self, system: str, user: str, retries: int = 2, want: tuple = ()) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err = None
        for _ in range(retries + 1):
            try:
                return extract_json(self.chat(msgs, json_mode=True), want)
            except RuntimeError as e:                      # the API refused: asking again will not help
                raise
            except Exception as e:
                err = e
                msgs = msgs + [{"role": "user", "content": "Reply again with only the JSON object, nothing else."}]
        raise RuntimeError(f"writer did not return JSON: {err}")

    def describe_image(self, image_b64: str, mime: str = "image/png") -> str:
        """Muse Spark vision: the text of a photographed notice, slide or sign, as written."""
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe the message in this image exactly as written. Reply with the text only."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}]
        return self.chat(msgs).strip()

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Muse Voice Transcribe through the OpenAI-style transcription endpoint (multipart upload)."""
        boundary = "----meaningcheck" + hashlib.md5(audio[:64]).hexdigest()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{VOICE_MODEL}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{META_BASE}/audio/transcriptions", data=body, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode()).get("text", "").strip()


class GlimmerWriter:
    """Muse Glimmer as the writer (POST /generate on the reader service), used when the Muse Spark API is unavailable."""
    name = "Muse Glimmer"

    def __init__(self, url: Optional[str] = None):
        g = GlimmerReaders(url)
        self.url, self.headers = g.url, g.headers

    def chat(self, messages: List[dict], max_tokens: int = 1500, json_mode: bool = False) -> str:
        return _post(self.url + "/generate", {"items": [{"messages": messages, "max_new_tokens": max_tokens,
                                                         "reasoning": "low"}]}, self.headers)["texts"][0]

    def json(self, system: str, user: str, retries: int = 2, want: tuple = ()) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err = None
        for _ in range(retries + 1):
            try:
                return extract_json(self.chat(msgs), want)
            except Exception as e:
                err = e
                msgs = msgs + [{"role": "user", "content": "Reply again with only the JSON object, nothing else."}]
        raise RuntimeError(f"writer did not return JSON: {err}")

    def describe_image(self, image_b64: str, mime: str = "image/png") -> str:
        raise RuntimeError("Image input uses Muse Spark vision; set MODEL_API_KEY with an active Meta account.")

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        raise RuntimeError("Voice input uses Muse Voice Transcribe; set MODEL_API_KEY with an active Meta account.")


class FallbackWriter:
    """Muse Spark first; if its API refuses (no key, account restricted, out of credit), Muse Glimmer writes instead.
    The switch is logged and reported, never silent."""
    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup
        self.active = primary if primary is not None else backup
        self.events: List[str] = []

    @property
    def name(self):
        return self.active.name

    def _call(self, fn, *a, **k):
        """Muse Spark is tried on every call; a refused call is answered by Muse Glimmer and logged."""
        if self.primary is not None:
            try:
                out = getattr(self.primary, fn)(*a, **k)
                self.active = self.primary
                return out
            except Exception as e:
                if fn in ("describe_image", "transcribe"):
                    raise
                self.events.append(f"Muse Spark unavailable for one call ({str(e)[:120]}); Muse Glimmer wrote it instead")
                self.active = self.backup
        return getattr(self.backup, fn)(*a, **k)

    def json(self, *a, **k):
        return self._call("json", *a, **k)

    def chat(self, *a, **k):
        return self._call("chat", *a, **k)

    def describe_image(self, *a, **k):
        return self._call("describe_image", *a, **k)

    def transcribe(self, *a, **k):
        return self._call("transcribe", *a, **k)


# ───────────────────────────── readers ─────────────────────────────
class GlimmerReaders:
    """Muse Glimmer reader service on Modal: returns log-probabilities of the option letters."""
    def __init__(self, url: Optional[str] = None, batch: int = 32):
        self.url = (url or os.environ.get("GLIMMER_URL", "")).rstrip("/")
        for tail in ("/score", "/generate", "/health"):
            if self.url.endswith(tail):
                self.url = self.url[: -len(tail)]
        if not self.url:
            raise RuntimeError("GLIMMER_URL is not set (deploy deploy/modal_glimmer.py first)")
        self.batch = batch
        self.headers = ({"Modal-Key": os.environ["MODAL_PROXY_KEY"], "Modal-Secret": os.environ["MODAL_PROXY_SECRET"]}
                        if os.environ.get("MODAL_PROXY_KEY") else {})

    def score(self, queries: List[Query]) -> List[dict]:
        out = []
        for i in range(0, len(queries), self.batch):
            chunk = queries[i:i + self.batch]
            res = _post(self.url + "/score", {"items": [{"system": q.system, "user": q.user, "letters": q.letters,
                                              "reasoning": q.reasoning} for q in chunk]},
                        self.headers)
            for r in res["results"]:
                out.append({"probs": _softmax(r["logprobs"]), "mass": float(r.get("mass", 1.0)),
                            "route": r.get("route", "direct")})
        return out


class SparkSampledReaders:
    """Readers on Muse Spark, which gives no token probabilities: each reading is asked n times and the chosen letters
    are counted. Calls run in parallel. `mass` is the share of answers in which a letter could be found."""
    def __init__(self, writer: SparkWriter, n: int = 3, workers: int = 6):
        self.w, self.n, self.workers = writer, n, workers

    def _one(self, q: Query) -> dict:
        from .glimmer_service import GlimmerService
        counts = {L: 0.0 for L in q.letters}; found = 0
        for _ in range(self.n):
            for attempt in range(3):                      # an empty or letterless reply is asked again (up to twice)
                txt = self.w.chat([{"role": "system", "content": q.system},
                                   {"role": "user", "content": q.user + "\nReply with the letter only."}],
                                  max_tokens=4000) or ""
                pos = GlimmerService.find_letter(txt, q.letters)
                if pos is None:
                    m = re.search(r"\b([A-H])\b", txt)
                    pos = m.start(1) if m and m.group(1) in counts else None
                if pos is not None:
                    counts[txt[pos]] += 1; found += 1
                    break
        if not found:                                     # no usable answer: equal probabilities, never "understood"
            return {"probs": {k: 1 / len(counts) for k in counts}, "mass": 0.0}
        tot = sum(counts.values())
        probs = {k: (v + 0.1) / (tot + 0.1 * len(counts)) for k, v in counts.items()}
        return {"probs": probs, "mass": found / self.n}

    def score(self, queries: List[Query]) -> List[dict]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, queries))


class SimReaders:
    """Deterministic simulated readers for the built-in examples: each reader has a plausibility for every reading
    (shared base + country and background leanings + personal noise), reads statements literally (except those its
    world marks as read differently), and favours the first option slightly (position bias, removed by averaging the
    two option orders)."""
    def __init__(self, model: str = "glimmer"):
        self.model = model

    def score(self, queries: List[Query]) -> List[dict]:
        out = []
        for q in queries:
            case, persona, order, stated = q.meta["case"], q.meta["persona"], q.meta["order"], q.meta["stated"]
            world = dict(case.sim or {})
            if self.model != "glimmer" and "second" in world:       # how the other model's readers differ
                world.update(world["second"])
            plaus = {}
            for r in case.readings:
                v = world.get("base", {}).get(r.id, 0.0)
                v += world.get("country", {}).get(persona.country, {}).get(r.id, 0.0)
                v += world.get("background", {}).get(persona.background[:20], {}).get(r.id, 0.0)
                v += world.get("reasoning", {}).get(persona.reasoning, {}).get(r.id, 0.0)
                h = int(hashlib.sha256(f"{persona.id}|{r.id}|{case.message[:40]}".encode()).hexdigest()[:8], 16)
                v += world.get("noise", 0.3) * ((h / 0xFFFFFFFF) - 0.5) * 2
                rem = (q.meta.get("removed") or "").lower()
                for trig, deltas in world.get("triggers", {}).items():
                    shared = set(rem.split()) & set(trig.lower().split())
                    if rem and (rem in trig.lower() or trig.lower() in rem or len(shared) >= 2):
                        v += deltas.get(r.id, 0.0)
                plaus[r.id] = v
            ignored = set(world.get("ignore", {}).get(persona.country, []))
            from .cover import open_set
            live = set(open_set(case, [a for a in stated if a not in ignored]))
            logits = {}
            for k, rid in enumerate(order):
                L = q.letters[k]
                logits[L] = 3.0 * plaus[rid] - (0 if rid in live else 12.0) + (0.4 if k == 0 else 0.0)
            out.append({"probs": _softmax(logits), "mass": 0.98})
        return out


def make_backend(name: str, cache_dir: str = ""):
    """(readers, writer). writer is None for the simulated backend (the examples library writes instead)."""
    cache_dir = cache_dir or os.environ.get("MC_CACHE_DIR", ".mc_cache")
    if name == "sim":
        return SimReaders(), None
    if name == "glimmer":
        readers = CachedReaders(GlimmerReaders(), os.path.join(cache_dir, "glimmer.jsonl"), "glimmer")
        spark = SparkWriter() if os.environ.get("MODEL_API_KEY") and os.environ.get("MC_WRITER") != "glimmer" else None
        return readers, FallbackWriter(spark, GlimmerWriter())
    if name == "spark":
        w = SparkWriter()
        return CachedReaders(SparkSampledReaders(w), os.path.join(cache_dir, "spark.jsonl"), "spark"), w
    raise ValueError(f"unknown backend {name!r}")


def make_second(name: str, cache_dir: str = ""):
    """Second-opinion readers from another model: Muse Spark when a Meta key is set (MC_SECOND=off disables)."""
    if os.environ.get("MC_SECOND", "").lower() == "off":
        return None
    if name == "sim":
        return SimReaders(model="spark")
    if not os.environ.get("MODEL_API_KEY"):
        return None
    cache_dir = cache_dir or os.environ.get("MC_CACHE_DIR", ".mc_cache")
    n = int(os.environ.get("MC_SECOND_SAMPLES", "1"))          # 1 answer x 2 option orders per reader keeps it fast
    return CachedReaders(SparkSampledReaders(SparkWriter(), n=n, workers=16),
                         os.path.join(cache_dir, f"spark_second_n{n}.jsonl"), f"spark-second-n{n}")


def default_backend() -> str:
    if os.environ.get("MC_BACKEND"):
        return os.environ["MC_BACKEND"]
    return "glimmer" if os.environ.get("GLIMMER_URL") else "sim"


class CachedReaders:
    """Stores every reading on disk keyed by (model, system, prompt, reasoning), so a rerun is instant and anyone can
    reproduce a result from the cache file without calling the model again."""
    def __init__(self, inner, path: str, tag: str):
        self.inner, self.path, self.tag = inner, path, tag
        self.mem: Dict[str, dict] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line); self.mem[d["k"]] = d["v"]
                    except Exception:
                        pass

    def _key(self, q: Query) -> str:
        return hashlib.sha256(json.dumps([self.tag, q.system, q.user, q.reasoning, q.letters]).encode()).hexdigest()

    def score(self, queries: List[Query]) -> List[dict]:
        keys = [self._key(q) for q in queries]
        todo = [i for i, k in enumerate(keys) if k not in self.mem]
        if todo:
            res = self.inner.score([queries[i] for i in todo])
            with open(self.path, "a") as f:
                for i, r in zip(todo, res):
                    self.mem[keys[i]] = r
                    f.write(json.dumps({"k": keys[i], "v": r}) + "\n")
        return [self.mem[k] for k in keys]
