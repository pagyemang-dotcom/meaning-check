"""Recompute every reported statistic from the recorded result files, offline, and check them against RESULTS.md.

No model is called: the per-message rows in results/*.json are the raw data; the summaries, intervals and the
numbers quoted in RESULTS.md are derived from them again here.
"""
from __future__ import annotations

import json
from pathlib import Path


def _close(a, b, tol=1e-9):
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def eval_summary(rows):
    from .cli import _bootstrap
    ok = [r for r in rows if "error" not in r]
    gain = [r["heldout_after"] - r["heldout_before"] for r in ok]
    m, lo, hi = _bootstrap(gain)
    out = {"messages": len(ok), "heldout_before": sum(r["heldout_before"] for r in ok) / len(ok),
           "heldout_after": sum(r["heldout_after"] for r in ok) / len(ok), "gain": m, "gain_95ci": [lo, hi],
           "H1_supported": lo > 0.10, "H2_no_harm": sum(g < 0 for g in gain),
           "H3_median_statements": sorted(r["statements"] for r in ok)[len(ok) // 2]}
    if all("heldout_baseline" in r for r in ok):
        mb, lb, hb = _bootstrap([r["heldout_after"] - r["heldout_baseline"] for r in ok])
        out.update(exploratory_baseline=sum(r["heldout_baseline"] for r in ok) / len(ok),
                   exploratory_gain_over_baseline=mb, exploratory_gain_over_baseline_95ci=[lb, hb])
    return out


def run(root: Path, log=print) -> bool:
    from .crossmodel import summarise
    res, ok = root / "results", True
    text = (root / "RESULTS.md").read_text()

    def check(label, got, want):
        nonlocal ok
        good = _close(got, want)
        ok &= good
        log(f"  {'ok  ' if good else 'FAIL'} {label}: {got}" + ("" if good else f" (recorded {want})"))

    def quoted(s):
        nonlocal ok
        good = s in text
        ok &= good
        log(f"  {'ok  ' if good else 'FAIL'} RESULTS.md states \"{s}\"")

    pct = lambda x: f"{100 * x:.1f}"
    for name in ("eval.json", "eval_v1.4.json", "eval_run2_baseline.json"):
        f = res / name
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        log(f"{name} ({len(d['rows'])} messages)")
        s = eval_summary(d["rows"])
        for k, v in s.items():
            if k in d["summary"]:
                check(k, v, d["summary"][k])
    d = json.loads((res / "eval_run2_baseline.json").read_text())["summary"]        # the pre-registered result
    quoted(f"{pct(d['heldout_before'])}% to {pct(d['heldout_after'])}%, gain {pct(d['gain'])} points "
           f"[{pct(d['gain_95ci'][0])}, {pct(d['gain_95ci'][1])}]")
    d = json.loads((res / "eval.json").read_text())["summary"]                      # the later mixed-writer run
    quoted(f"{pct(d['heldout_before'])}% before and {d['heldout_after']:.0%} after (gain {pct(d['gain'])} points "
           f"[{pct(d['gain_95ci'][0])}, {pct(d['gain_95ci'][1])}])")
    f = res / "crossmodel.json"
    if f.exists():
        d = json.loads(f.read_text())
        log(f"crossmodel.json ({len(d['rows'])} messages, Muse Spark readers)")
        s = summarise(d["rows"])
        for k in ("HM1_fixed_minus_original", "HM1_95ci", "HM1_supported", "HM2_passed_minus_flagged", "HM2_95ci",
                  "HM2_supported", "exploratory_fixed_minus_baseline", "exploratory_95ci"):
            check(k, s[k], d[k])
        quoted(f"+{pct(d['HM1_fixed_minus_original'])} points [{pct(d['HM1_95ci'][0])}, {pct(d['HM1_95ci'][1])}]")
    web = root / "meaningcheck" / "web" / "data"
    for name in ("eval_v1.4.json", "crossmodel.json"):
        if (web / name).exists():
            same = (web / name).read_bytes() == (res / name).read_bytes()
            ok &= same
            log(f"  {'ok  ' if same else 'FAIL'} the web app shows the same {name} as results/")
    log("All reported statistics reproduce." if ok else "Some statistics do not reproduce.")
    return ok
