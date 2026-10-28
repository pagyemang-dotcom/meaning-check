"""Command line: demo, smoke, serve, eval, lint."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .backends import Query, default_backend, make_backend
from .engine import Engine
from .examples import Library
from .writer import Writer


def _engine(backend: str, second: bool = True) -> Engine:
    from .backends import make_second
    readers, spark = make_backend(backend)
    return Engine(readers, Writer(spark, Library()), log=lambda m: print("  ·", m),
                  second=make_second(backend) if second else None)


def _summary(rep) -> str:
    b = sum(r.understood for r in rep.before); a = sum(r.understood for r in rep.after); n = len(rep.before)
    c = rep.certificate
    lines = [f"  as written: {b}/{n} understand",
             *[f"    misread by {r.persona}: {next(x['text'] for x in rep.case['readings'] if x['id'] == r.choice)}"
               for r in rep.before if not r.understood][:6],
             f"  fix: {len(c['statements'])} statement(s); the shortest set ruling out every preferred rival has "
             f"{c['minimum_length']} (proved: {c['proved_optimal']})"
             + (f", {c['added_after_rereading']} added after re-reading" if c.get('added_after_rereading') else "") + ":",
             *[f"    + {s['statement']}" for s in c["statements"]],
             f"  rewrite: {rep.rewrite}",
             f"  after: {a}/{n} understand"]
    if rep.holdout:
        lines.append(f"  unseen readers: {rep.holdout['before']:.0%} -> {rep.holdout['after']:.0%} "
                     f"(bound for new readers: <= {c['unseen_reader_bound']:.0%})")
    if rep.second:
        s2 = rep.second
        lines.append(f"  second opinion ({s2['model']}): {s2['before_rate']:.0%} -> {s2['after_rate']:.0%} of {len(s2['readers'])} readers")
    if rep.context:
        lines.append(f"  ask first: {rep.context['question']}")
    return "\n".join(lines)


def cmd_demo(a):
    eng = _engine(a.backend); lib = Library()
    for k in ([a.example] if a.example else [x for x in lib.keys()]):
        case = lib.get(k)
        print(f"\n{case.message}  [{case.mode}]")
        print(_summary(eng.check(case.message, case=case, attribution=False)))


def cmd_check(a):
    eng = _engine(a.backend)
    rep = eng.check(a.message, a.meaning, a.mode)
    print(_summary(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep.to_dict(), indent=1, default=str))


def cmd_smoke(a):
    """Checks every live part and says exactly what works. Exit code 0 when the readers work."""
    from .backends import GlimmerReaders, GlimmerWriter, SparkWriter, _post
    import urllib.request
    readers_ok = False
    print("GLIMMER_URL set:  ", bool(os.environ.get("GLIMMER_URL")))
    print("MODEL_API_KEY set:", bool(os.environ.get("MODEL_API_KEY")))
    try:
        g = GlimmerReaders()
        t0 = time.time()
        req = urllib.request.Request(g.url + "/health", headers=g.headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            health = json.loads(r.read().decode())
        cal = health["calibration"]
        print(f"Glimmer service up in {time.time() - t0:.0f}s (first call includes loading the model)")
        print("  answer prefix chosen:", repr(cal.get("prefix")), " letter mass by prefix:",
              {k: round(v, 3) for k, v in cal.get("masses", {}).items()})
        res = _post(g.url + "/score", {"debug": True, "items": [
            {"system": "You live in the United Kingdom. You work in an office.",
             "user": "Message you received:\n\"\"\"\nPlease table the budget item at Friday's meeting.\n\"\"\"\n\n"
                     "What does the sender most likely want you to understand?\n\nA) Discuss it on Friday.\n"
                     "B) Postpone it.\n\nAnswer with the letter of one option.", "letters": ["A", "B"],
             "reasoning": "low"}]}, g.headers)
        r0 = res["results"][0]
        print("  one reading:", {k: round(v, 3) for k, v in r0["logprobs"].items()}, "mass", round(r0["mass"], 3),
              "route", r0.get("route"))
        readers_ok = r0["mass"] >= 0.5
        print("  READERS OK" if readers_ok else "  READERS WEAK: under half the probability on the letters")
    except Exception as e:
        print("Glimmer readers FAILED:", e)
    try:
        w = GlimmerWriter()
        d = w.json("Return only JSON.", 'Return the JSON object {"ok": true} and nothing else.', want=("ok",))
        print("Glimmer writer OK:", d)
    except Exception as e:
        print("Glimmer writer FAILED:", e)
    try:
        s = SparkWriter()
        print("Meta models visible:", s.models())
        print("Muse Spark writer OK:", s.json("Return only JSON.", 'Return {"ok": true}', want=("ok",)))
    except Exception as e:
        print("Muse Spark FAILED (Glimmer will write instead):", str(e)[:300])
    sys.exit(0 if readers_ok else 1)


def cmd_models(a):
    from .backends import SparkWriter
    for m in SparkWriter().models():
        print(m)


def cmd_serve(a):
    from .server import run
    run(a.host, a.port)


def _bootstrap(xs, B=4000, seed=0):
    rng = random.Random(seed); n = len(xs)
    if n == 0:
        return (float("nan"),) * 3
    m = sum(xs) / n
    bs = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


from .engine import kept_words as _kept


def cmd_eval(a):
    """Held-out evaluation fixed in docs/PREREGISTRATION.md: every message in the set, readers never used to build."""
    eng = _engine(a.backend, second=a.second_opinion)
    eng.writer.prefer_library = False          # every message is analysed by the writer, as registered
    items = json.loads(Path(a.set).read_text())
    rows = []
    for it in items:
        print(f"\n{it['message']}")
        try:
            rep = eng.check(it["message"], it.get("meaning", ""), it.get("mode", "people"), attribution=False,
                            baseline=a.baseline)
        except Exception as e:
            print("  skipped:", e); rows.append({"message": it["message"], "error": str(e)}); continue
        print(_summary(rep))
        rows.append({"message": it["message"], "mode": it.get("mode", "people"), "meaning": it.get("meaning", ""),
                     "readings": rep.case["readings"], "intended": rep.case["intended"],
                     "panel_flagged": any(not r.understood for r in rep.before),
                     "panel_before": sum(r.understood for r in rep.before) / len(rep.before),
                     "panel_after": rep.certificate["panel_understood"],
                     "heldout_before": rep.holdout["before"], "heldout_after": rep.holdout["after"],
                     "statements": len(rep.statements), "rewrite": rep.rewrite, "candidates": rep.candidates,
                     "needs_question": rep.context is not None,
                     "letter_mass_min": rep.certificate.get("letter_mass_min"), "writer": rep.certificate.get("writer"),
                     "words_original": len(it["message"].split()), "words_final": len(rep.rewrite.split()),
                     "words_kept": _kept(it["message"], rep.rewrite),
                     **({"heldout_baseline": rep.holdout.get("baseline"), "baseline_text": rep.holdout.get("baseline_text")}
                        if a.baseline else {}),
                     **({"words_baseline": len(rep.holdout["baseline_text"].split()),
                         "words_kept_baseline": _kept(it["message"], rep.holdout["baseline_text"])} if a.baseline else {})})
    ok = [r for r in rows if "error" not in r]
    gain = [r["heldout_after"] - r["heldout_before"] for r in ok]
    m, lo, hi = _bootstrap(gain)
    summary = {"messages": len(ok), "skipped": len(rows) - len(ok),
               "heldout_before": sum(r["heldout_before"] for r in ok) / max(len(ok), 1),
               "heldout_after": sum(r["heldout_after"] for r in ok) / max(len(ok), 1),
               "gain": m, "gain_95ci": [lo, hi], "H1_supported": lo > 0.10 if ok else None,
               "H2_no_harm": sum(g < 0 for g in gain), "backend": a.backend,
               "H3_median_statements": sorted(r["statements"] for r in ok)[len(ok) // 2] if ok else None}
    if a.baseline and ok:
        d = [r["heldout_after"] - r["heldout_baseline"] for r in ok]
        mb, lb, hb = _bootstrap(d)
        summary.update(exploratory_baseline=sum(r["heldout_baseline"] for r in ok) / len(ok),
                       exploratory_gain_over_baseline=mb, exploratory_gain_over_baseline_95ci=[lb, hb])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n" + json.dumps(summary, indent=1))


def cmd_lint(a):
    """CI check: every entry {path, meaning, mode} must be understood by every reader on the panel."""
    eng = _engine(a.backend)
    cfg = json.loads(Path(a.config).read_text())
    failed = 0
    for it in cfg:
        text = Path(it["path"]).read_text().strip() if "path" in it else it["message"]
        rep = eng.check(text, it.get("meaning", ""), it.get("mode", "people"), attribution=False, holdout=False)
        miss = [r for r in rep.before if not r.understood]
        name = it.get("path", text[:50])
        if miss:
            failed += 1
            print(f"FAIL {name}: misread by {len(miss)}/{len(rep.before)} readers")
            print(f"     suggested fix: {rep.rewrite}")
        else:
            print(f"ok   {name}")
    sys.exit(1 if failed else 0)


def cmd_human_form(a):
    from .human import apps_script, build_items
    items = build_items(json.loads(Path(a.eval).read_text()))
    Path(a.items).parent.mkdir(parents=True, exist_ok=True)
    Path(a.items).write_text(json.dumps(items, indent=1, ensure_ascii=False))
    Path(a.out).write_text(apps_script(items))
    print(f"{len(items)} messages ({sum(i['changed'] for i in items)} with a fix); items -> {a.items}; script -> {a.out}")


def cmd_human_analyse(a):
    from .human import analyse
    items = json.loads(Path(a.items).read_text())
    res = analyse(Path(a.responses).read_text(encoding="utf-8-sig"), items)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "per_item"}, indent=1))


def cmd_crossmodel(a):
    from .backends import CachedReaders, SparkSampledReaders, SparkWriter
    from .crossmodel import plan_calls, run
    rows = json.loads(Path(a.eval).read_text())["rows"]
    n_calls = plan_calls(rows, a.samples)
    print(f"This test makes about {n_calls} Muse Spark calls ({a.samples} per reading).")
    if a.dry_run:
        return
    readers = CachedReaders(SparkSampledReaders(SparkWriter(), n=a.samples, workers=a.workers),
                            os.path.join(os.environ.get("MC_CACHE_DIR", ".mc_cache"), f"spark_n{a.samples}_v2.jsonl"),
                            f"spark-n{a.samples}-v2")
    res = run(rows, readers)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def cmd_reproduce(a):
    from .reproduce import run
    sys.exit(0 if run(Path(a.root)) else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="meaningcheck", description="Make sure every reader understands a message as meant.")
    p.add_argument("--backend", default=default_backend(), choices=["sim", "glimmer", "spark"])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--example", default=""); d.set_defaults(f=cmd_demo)
    c = sub.add_parser("check"); c.add_argument("message"); c.add_argument("--meaning", default="")
    c.add_argument("--mode", default="people", choices=["people", "agents"]); c.add_argument("--json", default="")
    c.set_defaults(f=cmd_check)
    s = sub.add_parser("smoke"); s.set_defaults(f=cmd_smoke)
    mo = sub.add_parser("models"); mo.set_defaults(f=cmd_models)
    v = sub.add_parser("serve"); v.add_argument("--host", default="127.0.0.1"); v.add_argument("--port", type=int, default=8000)
    v.set_defaults(f=cmd_serve)
    e = sub.add_parser("eval"); e.add_argument("--set", default="examples/eval_messages.json")
    e.add_argument("--out", default="results/eval.json")
    e.add_argument("--baseline", action="store_true", help="exploratory: also test a plain 'make it clear' rewrite")
    e.add_argument("--second-opinion", action="store_true", help="exploratory: add Muse Spark readers to the fix")
    e.set_defaults(f=cmd_eval)
    l = sub.add_parser("lint"); l.add_argument("config"); l.set_defaults(f=cmd_lint)
    cm = sub.add_parser("crossmodel"); cm.add_argument("--eval", default="results/eval.json")
    cm.add_argument("--samples", type=int, default=3); cm.add_argument("--workers", type=int, default=6)
    cm.add_argument("--out", default="results/crossmodel.json"); cm.add_argument("--dry-run", action="store_true")
    cm.set_defaults(f=cmd_crossmodel)
    hf = sub.add_parser("human-form"); hf.add_argument("--eval", default="results/eval.json")
    hf.add_argument("--items", default="results/human_items.json"); hf.add_argument("--out", default="results/human_form.gs")
    hf.set_defaults(f=cmd_human_form)
    rp = sub.add_parser("reproduce", help="recompute every reported statistic from results/ (offline)")
    rp.add_argument("--root", default="."); rp.set_defaults(f=cmd_reproduce)
    ha = sub.add_parser("human-analyse"); ha.add_argument("responses")
    ha.add_argument("--items", default="results/human_items.json"); ha.add_argument("--out", default="results/human_results.json")
    ha.set_defaults(f=cmd_human_analyse)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
