"""The independent-model test, end to end against a local stand-in for the Meta Model API."""
import re
import threading
import time

import uvicorn
from fastapi import FastAPI, Request

from meaningcheck import backends as B
from meaningcheck.crossmodel import plan_calls, run


def test_crossmodel_against_fake_spark(tmp_path):
    api = FastAPI()

    @api.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        user = body["messages"][-1]["content"]
        msg = re.search(r'"""\n(.*?)\n"""', user, re.S).group(1)
        opts = dict(re.findall(r"\n([A-H])\) ([^\n]*)", user))
        # a reader that understands only fixed wordings, and replies in a reasoning style
        want = "right" if msg.startswith("fixed") or msg.startswith("unchanged") else "wrong"
        letter = next(L for L, t in opts.items() if t.startswith(want))
        return {"choices": [{"message": {"content": f"Let me think about option A and B. Final answer: {letter}"}}]}

    srv = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=8797, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    old = B.META_BASE; B.META_BASE = "http://127.0.0.1:8797/v1"
    try:
        rows = []
        for k in range(6):
            ch = k < 4
            rows.append({"message": f"orig {k}" if ch else f"unchanged {k}", "mode": "people", "rewrite": f"fixed {k}",
                         "baseline_text": f"orig base {k}", "statements": 1 if ch else 0, "panel_before": 0.0 if ch else 1.0,
                         "panel_flagged": ch, "intended": "r1",
                         "readings": [{"id": "r1", "text": f"right {k}"}, {"id": "r2", "text": f"wrong {k}"}]})
        assert plan_calls(rows, 1) == 4 * 8 * 2 * 3 + 2 * 8 * 2
        readers = B.CachedReaders(B.SparkSampledReaders(B.SparkWriter(api_key="k"), n=1, workers=4),
                                  str(tmp_path / "c.jsonl"), "t")
        res = run(rows, readers, log=lambda *_: None)
        assert res["HM1_fixed_minus_original"] == 1.0 and res["HM1_supported"]
        assert res["HM2_supported"] and res["exploratory_fixed_minus_baseline"] == 1.0
        assert res["lowest_letter_share"] == 1.0
    finally:
        B.META_BASE = old; srv.should_exit = True
