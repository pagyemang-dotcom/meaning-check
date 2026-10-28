"""Web app: the reader grid, the fix, the certificate and the one question."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .backends import default_backend, make_backend, make_second
from .engine import Engine
from .examples import EXAMPLES, Library
from .panels import panels
from .writer import Writer

HERE = Path(__file__).parent
app = FastAPI(title="Meaning Check")
BACKEND = default_backend()
_readers, _spark = make_backend(BACKEND)
LIB = Library()
SECOND = make_second(BACKEND)
ENGINE = Engine(_readers, Writer(_spark, LIB), second=SECOND)

# Spending guard for a public link: at most MC_SECOND_PER_HOUR checks an hour use the paid Muse Spark second opinion;
# beyond that a check still runs on Muse Glimmer alone and says so.
import threading, time as _time
_GUARD = {"window": 0.0, "n": 0}
_LOCK = threading.Lock()
SECOND_PER_HOUR = int(os.environ.get("MC_SECOND_PER_HOUR", "40"))


def _second_allowed() -> bool:
    if not SECOND or BACKEND == "sim":
        return bool(SECOND)
    with _LOCK:
        now = _time.time()
        if now - _GUARD["window"] > 3600:
            _GUARD.update(window=now, n=0)
        if _GUARD["n"] >= SECOND_PER_HOUR:
            return False
        _GUARD["n"] += 1
        return True


class CheckIn(BaseModel):
    message: str = ""
    meaning: str = ""
    mode: str = "people"
    example: str = ""
    live: bool = False          # for a scenario: run it live even when a recorded live run exists


class AttributionIn(BaseModel):
    case: dict


# Recorded live runs of the one-click scenarios. The first live run of a scenario is saved (on the app's volume); later
# clicks replay it in about two seconds instead of waking the GPU, and say so on screen, with a button to run it live.
import json as _json
from . import __version__
SNAP_DIR = Path(os.environ.get("MC_CACHE_DIR", ".mc_cache")) / "scenarios"


def _snap_path(key: str) -> Path:
    return SNAP_DIR / f"{key}.json"


def load_snapshot(key: str):
    if BACKEND == "sim" or not key:
        return None
    if key in EXAMPLES and not _snap_path(key).exists():
        _volume("reload")
    for f in (_snap_path(key), HERE / "web" / "data" / "scenarios" / f"{key}.json"):
        try:
            d = _json.loads(f.read_text())
            if (d.get("backend") == BACKEND and d.get("version") == __version__
                    and d.get("second", False) == bool(SECOND)):       # recorded with the same models as now
                return d
        except Exception:
            continue
    return None


def save_snapshot(key: str, events: list):
    if BACKEND == "sim":
        return
    try:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _snap_path(key).with_suffix(".tmp")
        tmp.write_text(_json.dumps({"backend": BACKEND, "version": __version__, "second": bool(SECOND),
                                    "recorded_at": _time.strftime("%Y-%m-%d %H:%M UTC", _time.gmtime()),
                                    "events": events}, default=str))
        tmp.replace(_snap_path(key))
        _volume("commit")
    except Exception as e:
        print("could not save the recorded run:", e)


_LAST_RELOAD = [0.0]


def _volume(action: str):
    """On Modal, recordings live on a shared volume: commit after writing so other containers see them, and reload
    (at most every 30 s) when one is missing. Elsewhere this does nothing."""
    name = os.environ.get("MC_VOLUME")
    if not name:
        return
    try:
        import modal
        vol = modal.Volume.from_name(name)
        if action == "reload":
            if _time.time() - _LAST_RELOAD[0] < 30:
                return
            _LAST_RELOAD[0] = _time.time()
        getattr(vol, action)()
    except Exception as e:
        print(f"volume {action} failed:", e)


class TestIn(BaseModel):
    case: dict
    text: str
    statements: list = []


class ImageIn(BaseModel):
    data: str            # base64
    mime: str = "image/png"


def persona_info(mode: str):
    from .panels import second_panel
    build, held = panels(mode)
    f = lambda p: {"id": p.id, "country": p.country, "flag": p.flag, "reasoning": p.reasoning,
                   "background": p.background, "prompt": p.system_prompt(mode)}
    return {"build": [f(p) for p in build], "heldout": [f(p) for p in held],
            "second": [f(p) for p in second_panel(mode)]}


def _index_path() -> Path:
    for d in (os.environ.get("MC_WEB_DIR", ""), str(HERE / "web"), "/web"):
        if d and (Path(d) / "index.html").exists():
            return Path(d) / "index.html"
    raise FileNotFoundError("index.html not found (set MC_WEB_DIR)")


@app.get("/")
def index():
    return FileResponse(_index_path())


@app.get("/api/info")
def info():
    return {"backend": BACKEND, "writer": ENGINE.writer.name, "second": "Muse Spark" if SECOND else None,
            "version": __version__, "recorded": [k for k in EXAMPLES if load_snapshot(k)],
            "second_status": ("on" if SECOND else "off: MC_SECOND=off" if os.environ.get("MC_SECOND", "").lower() == "off"
                              else "off: this app has no MODEL_API_KEY" if BACKEND != "sim" else "simulated"),
            "examples": [{"key": k, "message": v["case"].message, "meaning": v["case"].meaning,
                          "mode": v["case"].mode} for k, v in EXAMPLES.items() if k != "plate" or BACKEND == "sim"]}


@app.get("/api/warm")
def warm():
    """Wakes the Muse Glimmer service (loading the model takes a few minutes on a cold start)."""
    if BACKEND != "glimmer":
        return {"ready": True}
    import json, urllib.request
    from .backends import GlimmerReaders
    g = GlimmerReaders()
    try:
        with urllib.request.urlopen(urllib.request.Request(g.url + "/health", headers=g.headers), timeout=600) as r:
            return {"ready": True, "glimmer": json.loads(r.read().decode())}
    except Exception as e:
        return {"ready": False, "detail": str(e)[:200]}


@app.post("/api/check")
def check(body: CheckIn):
    try:
        case = LIB.get(body.example) if body.example else None
        if case is None and not body.message.strip():
            raise ValueError("Type a message first.")
        use2 = _second_allowed()
        rep = ENGINE.check(body.message, body.meaning, body.mode, case=case, second_opinion=use2)
        d = rep.to_dict()
        if SECOND and not use2:
            d["log"].append("The Muse Spark second opinion is paused for this hour (spending guard); Muse Glimmer only.")
        d["personas"] = persona_info(rep.case["mode"])
        d["backend"] = BACKEND
        return JSONResponse(d)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/check_stream")
def check_stream(body: CheckIn):
    """The same check, streamed as server-sent events: each stage and each panel's readings arrive as soon as they are
    ready, then the full report. Keeps long checks responsive and avoids request time limits."""
    import json, queue, threading
    from fastapi.responses import StreamingResponse
    q: "queue.Queue" = queue.Queue()

    snap = None if body.live else load_snapshot(body.example)

    def replay():
        # a recorded live run, replayed with short pauses so the stages are still visible
        for ev, data in snap["events"]:
            if ev == "done":
                data = dict(data, recorded_at=snap["recorded_at"])
            q.put((ev, data))
            _time.sleep(0.12 if ev == "stage" else 0.3)
        q.put(None)

    def work():
        events = []

        def put(ev, data):
            events.append((ev, data)); q.put((ev, data))
        try:
            case = LIB.get(body.example) if body.example else None
            if case is None and not body.message.strip():
                raise ValueError("Type a message first.")
            use2 = _second_allowed()
            mode = case.mode if case else body.mode
            put("personas", persona_info(mode))
            # which words cause the misreading is worked out after the result is shown (/api/attribution)
            rep = ENGINE.check(body.message, body.meaning, body.mode, case=case, second_opinion=use2,
                               attribution=False, progress=put)
            d = rep.to_dict()
            if SECOND and not use2:
                d["log"].append("The Muse Spark second opinion is paused for this hour (spending guard); Muse Glimmer only.")
            d["personas"] = persona_info(rep.case["mode"]); d["backend"] = BACKEND
            put("done", d)
            q.put(None)
            if case is not None and (use2 or not SECOND):          # record complete scenario runs only
                d = dict(d); events[-1] = ("done", d)                # the streamed copy is left untouched
                try:
                    d["attribution"] = _attribution(case)
                except Exception as e:
                    print("attribution failed:", e)
                save_snapshot(body.example, events)
            return
        except Exception as e:
            q.put(("error", {"detail": str(e)}))
        q.put(None)

    threading.Thread(target=replay if snap else work, daemon=True).start()

    def events():
        while True:
            item = q.get()
            if item is None:
                break
            ev, data = item
            yield f"event: {ev}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _attribution(case):
    build, _ = panels(case.mode)
    from .reader import read
    base = sum(not r.understood for r in read(ENGINE.readers, case, case.message, [], build))   # cached: no new call
    return ENGINE.attribute_phrases(case, build, base)


@app.post("/api/attribution")
def attribution(body: AttributionIn):
    """Which words cause the misreading: the panel reads the message with each phrase removed (one request)."""
    from .core import Attribute, Case, Reading
    c = body.case
    try:
        case = Case(c["message"], c["intended"], [Reading(**r) for r in c["readings"]],
                    [Attribute(**a) for a in c["attributes"]], c.get("mode", "people"), c.get("meaning", ""))
        return {"attribution": _attribution(case)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/results")
def results():
    """The recorded evaluation results shipped with the app (the same files as in the repository's results/)."""
    import json
    out = {}
    for name in ("eval_v1.4.json", "crossmodel.json"):
        for d in (os.environ.get("MC_WEB_DIR", ""), str(HERE / "web"), "/web"):
            f = Path(d) / "data" / name if d else None
            if f and f.exists():
                out[name.split(".json")[0]] = json.loads(f.read_text()); break
    return out


@app.post("/api/test")
def test_wording(body: TestIn):
    """Reads any wording (your own, or another tool's rewrite) with the same panel and held-out readers."""
    from .core import Attribute, Case, Reading
    from .reader import rate, read
    c = body.case
    try:
        case = Case(c["message"], c["intended"], [Reading(**r) for r in c["readings"]],
                    [Attribute(**a) for a in c["attributes"]], c.get("mode", "people"), c.get("meaning", ""))
        build, held = panels(case.mode)
        stated = [a for a in body.statements if a in {x.id for x in case.attributes}] if BACKEND == "sim" else []
        rb = read(ENGINE.readers, case, body.text, stated, build)
        rh = read(ENGINE.readers, case, body.text, stated, held)
        out = {"panel": [r.__dict__ for r in rb], "panel_rate": rate(rb), "heldout_rate": rate(rh),
               "words": len(body.text.split())}
        if SECOND and _second_allowed():
            from .panels import second_panel
            r2 = read(SECOND, case, body.text, stated, second_panel(case.mode))
            out.update(second=[r.__dict__ for r in r2], second_rate=rate(r2))
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/image")
def image(body: ImageIn):
    if _spark is None:
        raise HTTPException(400, "Image input needs MODEL_API_KEY (Muse Spark vision).")
    return {"text": _spark.describe_image(body.data, body.mime)}


@app.post("/api/voice")
async def voice(file: UploadFile = File(...)):
    if _spark is None:
        raise HTTPException(400, "Voice input needs MODEL_API_KEY (Muse Voice Transcribe).")
    return {"text": _spark.transcribe(await file.read(), file.filename or "audio.webm")}


def run(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
