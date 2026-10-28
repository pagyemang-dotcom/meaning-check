import json

import pytest

from meaningcheck import cover as C
from meaningcheck.backends import CachedReaders, SimReaders, extract_json, make_backend
from meaningcheck.core import Attribute, Case, Reading
from meaningcheck.engine import Engine, theorem14_bound
from meaningcheck.examples import Library
from meaningcheck.panels import agent_panel, people_heldout, people_panel
from meaningcheck.reader import read
from meaningcheck.writer import Writer, _validate, phrases


@pytest.fixture
def eng():
    readers, _ = make_backend("sim")
    return Engine(readers, Writer(None, Library()))


def toy_case():
    rs = [Reading(f"r{i}", f"reading {i}") for i in range(1, 5)]
    attrs = [Attribute("a1", "q1", {"r1": "x", "r2": "y", "r3": "x", "r4": "x"}, "s1"),
             Attribute("a2", "q2", {"r1": "x", "r2": "x", "r3": "y", "r4": "x"}, "s2"),
             Attribute("a3", "q3", {"r1": "x", "r2": "y", "r3": "y", "r4": "x"}, "s3"),
             Attribute("a4", "q4", {"r1": "x", "r2": "x", "r3": "x", "r4": "y"}, "s4")]
    return Case("m", "r1", rs, attrs)


def test_exclusion_and_open_set():
    c = toy_case()
    assert c.attribute("a3").excludes("r1") == ["r2", "r3"]
    assert C.open_set(c, ["a3"]) == ["r1", "r4"]
    assert C.open_set(c, ["a3", "a4"]) == ["r1"]


def test_cover_is_minimal_and_matches_brute_force():
    import itertools
    c = toy_case()
    for U in [{"r2"}, {"r2", "r3"}, {"r2", "r3", "r4"}, set()]:
        sol = C.solve_cover(c, U)
        best = min((k for k in range(5) for S in itertools.combinations([a.id for a in c.attributes], k)
                    if U <= set().union(*[set(c.attribute(a).excludes("r1")) for a in S])), default=None)
        assert sol["length"] == best and sol["optimal"]
        assert U <= set().union(*[set(c.attribute(a).excludes("r1")) for a in sol["attributes"]])


def test_uncoverable_is_reported():
    c = toy_case(); c.attributes = c.attributes[:2]
    sol = C.solve_cover(c, {"r4"})
    assert not sol["feasible"] and sol["uncoverable"] == ["r4"]


def test_dominators():
    assert C.dominators({"r1": 0.3, "r2": 0.5, "r3": 0.2}, "r1") == {"r2"}
    assert C.dominators({"r1": 0.6, "r2": 0.4}, "r1") == set()


def test_panels_sizes_and_disjoint():
    b, h = people_panel(), people_heldout()
    assert len(b) == 16 and len(h) == 8 and not {p.id for p in b} & {p.id for p in h}
    assert len(agent_panel()) == 16


def test_order_averaging_removes_position_bias():
    """A reader with no preference but a first-position bias ends up even after averaging the two orders."""
    class Biased:
        def score(self, qs):
            return [{"probs": {L: (0.7 if L == "A" else 0.3) for L in q.letters}, "mass": 1.0} for q in qs]
    c = Case("m", "r1", [Reading("r1", "a"), Reading("r2", "b")], [Attribute("a1", "q", {"r1": "x", "r2": "y"}, "s")])
    res = read(Biased(), c, "m", [], people_panel()[:1])
    assert abs(res[0].probs["r1"] - 0.5) < 1e-12


@pytest.mark.parametrize("key", ["table", "ticket", "date"])
def test_examples_fixed_for_whole_panel_and_heldout(eng, key):
    case = Library().get(key)
    rep = eng.check(case.message, case=case)
    assert sum(r.understood for r in rep.before) < len(rep.before)          # someone misreads as written
    assert all(r.understood for r in rep.after)                              # everyone after the fix
    assert rep.certificate["proved_optimal"] and rep.certificate["minimum_length"] >= 1
    assert rep.certificate["greedy_length"] >= rep.certificate["minimum_length"]
    assert rep.holdout["after"] >= rep.holdout["before"]
    assert rep.context is None


def test_no_single_wording_gives_question(eng):
    case = Library().get("plate")
    rep = eng.check(case.message, case=case)
    assert rep.context is not None and rep.context["question"]
    assert rep.context["unreachable"]                    # those readers misread every wording on their own


def test_theorem14_bound():
    assert abs(theorem14_bound(3, 16, 0.05) - (3 * 0.6931471805599453 + 2.995732273553991) / 16) < 1e-12
    assert theorem14_bound(50, 2) == 1.0


def test_extract_json_and_validate():
    d = extract_json('Sure:\n```json\n{"readings":[{"id":"r1","text":"a"},{"id":"r2","text":"b"}],'
                     '"attributes":[{"id":"a1","question":"q","values":{"r1":"X","r2":"y"},"statement":"s"},'
                     '{"id":"a2","question":"same","values":{"r1":"z","r2":"z"}}]}\n```')
    case = _validate(d, "m", "", "people")
    assert case.intended == "r1" and [a.id for a in case.attributes] == ["a1"]          # a2 tells nothing apart
    assert case.attribute("a1").values["r1"] == "x"


def test_phrases():
    assert phrases("Please table the budget item at Friday's meeting.")[0].startswith("Please")


def test_cache_roundtrip(tmp_path):
    calls = []
    class Inner:
        def score(self, qs):
            calls.append(len(qs)); return [{"probs": {"A": 1.0}, "mass": 1.0} for _ in qs]
    from meaningcheck.backends import Query
    cr = CachedReaders(Inner(), str(tmp_path / "c.jsonl"), "t")
    q = [Query("s", "u", ["A"])]
    cr.score(q); cr.score(q)
    assert calls == [1]
    assert CachedReaders(Inner(), str(tmp_path / "c.jsonl"), "t").score(q)[0]["probs"]["A"] == 1.0


def test_web_api():
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    cl = TestClient(app)
    assert cl.get("/").status_code == 200
    info = cl.get("/api/info").json()
    assert info["backend"] == "sim" and len(info["examples"]) >= 3
    r = cl.post("/api/check", json={"example": "table"}).json()
    assert len(r["personas"]["build"]) == len(r["before"]) == 16 and r["rewrite"]
    assert cl.post("/api/check", json={"message": "unknown message"}).status_code == 400


def test_web_test_wording():
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    cl = TestClient(app)
    r = cl.post("/api/check", json={"example": "date"}).json()
    t = cl.post("/api/test", json={"case": r["case"], "text": "Please send it by 3 April.", "statements": ["a2"]}).json()
    assert t["panel_rate"] == 1.0 and t["words"] == 6
    t0 = cl.post("/api/test", json={"case": r["case"], "text": r["case"]["message"]}).json()
    assert t0["panel_rate"] < 1.0


def test_kept_words():
    from meaningcheck.cli import _kept
    assert _kept("a b c d", "a b x c d e") == 1.0 and _kept("a b c d", "d c") == 0.25


def test_shortest_panel_approved_wording_is_kept(eng):
    case = Library().get("ticket")
    rep = eng.check(case.message, case=case, attribution=False)
    best = max(c["rate"] for c in rep.candidates)
    chosen = next(c for c in rep.candidates if c["text"] == rep.rewrite)
    assert chosen["rate"] == best == 1.0
    assert chosen["words"] == min(c["words"] for c in rep.candidates if c["rate"] == best)
    assert rep.rewrite != case.message


def test_tie_is_not_understood():
    from meaningcheck.core import Attribute, Case, Reading
    from meaningcheck.reader import read
    class Blank:
        def score(self, qs):
            return [{"probs": {L: 1 / len(q.letters) for L in q.letters}, "mass": 0.0} for q in qs]
    c = Case("m", "r1", [Reading("r1", "a"), Reading("r2", "b")], [Attribute("a1", "q", {"r1": "x", "r2": "y"}, "s")])
    r = read(Blank(), c, "m", [], people_panel()[:2])
    assert not any(x.understood for x in r) and all(x.choice == "r2" for x in r)


def test_fallback_writer_retries_spark_each_call():
    from meaningcheck.backends import FallbackWriter
    class Flaky:
        name = "Muse Spark"; n = 0
        def json(self, *a, **k):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("rate limited")
            return {"by": "spark"}
    class Backup:
        name = "Muse Glimmer"
        def json(self, *a, **k):
            return {"by": "glimmer"}
    w = FallbackWriter(Flaky(), Backup())
    assert w.json() == {"by": "glimmer"} and w.json() == {"by": "spark"} and w.name == "Muse Spark"


def test_second_opinion_catches_what_one_model_misses():
    from meaningcheck.backends import SimReaders
    readers, _ = make_backend("sim")
    eng = Engine(readers, Writer(None, Library()), second=SimReaders(model="spark"))
    case = Library().get("biweekly")
    rep = eng.check(case.message, case=case, attribution=False)
    assert all(r.understood for r in rep.before)                    # Muse Glimmer readers all understand it
    assert rep.second["before_rate"] < 0.5                           # Muse Spark readers mostly do not
    assert rep.statements and rep.rewrite != case.message            # so it is fixed anyway
    assert rep.second["after_rate"] == 1.0 and all(r.understood for r in rep.after)
    single = Engine(readers, Writer(None, Library())).check(case.message, case=case, attribution=False)
    assert single.second is None and single.rewrite == case.message  # one model alone would have passed it


def test_web_reports_second_opinion():
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    cl = TestClient(app)
    assert cl.get("/api/info").json()["second"] == "Muse Spark"
    r = cl.post("/api/check", json={"example": "biweekly"}).json()
    assert len(r["personas"]["second"]) == 8 and r["second"]["after_rate"] == 1.0
    t = cl.post("/api/test", json={"case": r["case"], "text": "Our sync starts next week."}).json()
    assert "second_rate" in t


def test_check_stream_events_and_report():
    import json
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    cl = TestClient(app)
    r = cl.post("/api/check_stream", json={"example": "table"})
    assert r.status_code == 200
    events = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        events.append((lines["event"], json.loads(lines["data"])))
    names = [e for e, _ in events]
    assert names[0] == "personas" and names[-1] == "done"
    assert "case" in names and "before" in names and "stage" in names
    personas = events[0][1]
    assert all(p["prompt"] for p in personas["build"])
    done = events[-1][1]
    assert done["after"] and done["rewrite"] and len(done["before"]) == len(personas["build"])


def test_check_stream_reports_errors():
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    r = TestClient(app).post("/api/check_stream", json={"message": "  "})
    assert "event: error" in r.text


def test_results_endpoint_matches_repository_files():
    import json
    from pathlib import Path
    from fastapi.testclient import TestClient
    from meaningcheck.server import app
    d = TestClient(app).get("/api/results").json()
    root = Path(__file__).resolve().parents[1]
    assert d["eval_v1.4"] == json.loads((root / "results" / "eval_v1.4.json").read_text())
    assert d["crossmodel"] == json.loads((root / "results" / "crossmodel.json").read_text())


def test_reproduce_recomputes_every_reported_statistic():
    from pathlib import Path
    from meaningcheck.reproduce import run
    assert run(Path(__file__).resolve().parents[1], log=lambda *a: None)


def _events(text):
    import json
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_scenario_runs_are_recorded_and_replayed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import meaningcheck.server as S
    monkeypatch.setattr(S, "BACKEND", "live-test")
    monkeypatch.setattr(S, "SNAP_DIR", tmp_path)
    cl = TestClient(S.app)
    first = _events(cl.post("/api/check_stream", json={"example": "deadline"}).text)
    assert first[-1][0] == "done" and "recorded_at" not in first[-1][1]
    import time
    for _ in range(50):                                   # the recording is written after the result is sent
        if (tmp_path / "deadline.json").exists():
            break
        time.sleep(0.1)
    again = _events(cl.post("/api/check_stream", json={"example": "deadline"}).text)
    done = again[-1][1]
    assert done["recorded_at"] and done["rewrite"] == first[-1][1]["rewrite"]
    assert "attribution" in done                          # recorded with the word attribution included
    live = _events(cl.post("/api/check_stream", json={"example": "deadline", "live": True}).text)
    assert "recorded_at" not in live[-1][1]
    assert "deadline" in cl.get("/api/info").json()["recorded"]


def test_attribution_endpoint_and_batched_reads_agree():
    from fastapi.testclient import TestClient
    from meaningcheck.server import app, ENGINE
    from meaningcheck.examples import Library
    from meaningcheck.panels import panels
    from meaningcheck.reader import read, read_many
    case = Library().get("table")
    d = TestClient(app).post("/api/attribution", json={"case": case.to_dict()}).json()
    assert d["attribution"] and all("phrase" in a for a in d["attribution"])
    build, held = panels(case.mode)
    one = [read(ENGINE.readers, case, case.message, [], build), read(ENGINE.readers, case, case.message, [], held)]
    many = read_many(ENGINE.readers, case, [(case.message, [], build), (case.message, [], held)])
    assert [[r.__dict__ for r in x] for x in one] == [[r.__dict__ for r in x] for x in many]
