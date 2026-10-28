"""The Muse Spark client against a local stand-in for the Meta Model API: JSON mode refused, then plain requests;
model listing; the switch to Muse Glimmer when the account is refused."""
import json
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from meaningcheck import backends as B


@pytest.fixture(scope="module")
def fake_meta():
    api = FastAPI(); seen = []

    @api.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json(); seen.append(body)
        if req.headers.get("authorization") != "Bearer good":
            raise HTTPException(403, "account restricted")
        if "response_format" in body:
            raise HTTPException(400, "response_format not supported")
        return {"choices": [{"message": {"content": 'Here you go: {"ok": true, "n": 2}'}}]}

    @api.get("/v1/models")
    def models():
        return {"data": [{"id": "muse-spark-1.3"}, {"id": "muse-voice-transcribe"}]}

    srv = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=8798, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    old = B.META_BASE
    B.META_BASE = "http://127.0.0.1:8798/v1"
    yield seen
    B.META_BASE = old
    srv.should_exit = True


def test_json_mode_refused_then_plain(fake_meta):
    w = B.SparkWriter(api_key="good")
    assert w.json("s", "u", want=("ok",)) == {"ok": True, "n": 2}
    assert "response_format" in fake_meta[0] and "response_format" not in fake_meta[-1]
    assert w.models() == ["muse-spark-1.3", "muse-voice-transcribe"]


def test_restricted_account_falls_back_to_glimmer_writer(fake_meta):
    class Backup:
        name = "Muse Glimmer"
        def json(self, *a, **k):
            return {"readings": "from glimmer"}
    w = B.FallbackWriter(B.SparkWriter(api_key="restricted"), Backup())
    assert w.json("s", "u") == {"readings": "from glimmer"}
    assert w.name == "Muse Glimmer" and "unavailable" in w.events[0]
