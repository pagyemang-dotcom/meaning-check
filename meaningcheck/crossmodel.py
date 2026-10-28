"""Independent-model test: Muse Spark, which took no part in building any fix, reads the original and fixed messages
(and the plain-rewrite baseline) as the eight held-out readers. Registered in docs/CROSSMODEL.md."""
from __future__ import annotations

import json
import random
from typing import List

from .core import Case, Reading
from .panels import panels
from .reader import rate, read


def case_from_row(r: dict) -> Case:
    return Case(r["message"], r["intended"], [Reading(**x) for x in r["readings"]], [], r["mode"], r.get("meaning", ""))


def plan_calls(rows: List[dict], n: int) -> int:
    return sum(len(panels(r["mode"])[1]) * 2 * n * (3 if r.get("statements", 0) > 0 else 1) for r in rows)


def boot(xs, B=4000, seed=0):
    rng = random.Random(seed); n = len(xs)
    if not n:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    bs = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def run(rows: List[dict], readers, log=print) -> dict:
    out = []
    for r in rows:
        if "readings" not in r:
            raise ValueError("eval.json has no readings: rerun `meaningcheck eval --baseline` with version 1.5 or later")
        case = case_from_row(r); held = panels(r["mode"])[1]
        row = {"message": r["message"], "mode": r["mode"], "changed": r.get("statements", 0) > 0,
               "panel_flagged": r.get("panel_flagged", r["panel_before"] < 1)}
        res = read(readers, case, r["message"], [], held)
        row["original"] = rate(res); row["mass"] = min(x.letter_mass for x in res)
        row["no_answer_readers"] = sum(x.letter_mass == 0 for x in res)
        if row["changed"]:
            row["fixed"] = rate(read(readers, case, r["rewrite"], [], held))
            row["baseline"] = rate(read(readers, case, r.get("baseline_text") or r["rewrite"], [], held))
        log(json.dumps(row))
        out.append(row)
    return summarise(out)


def summarise(out: List[dict]) -> dict:
    """The pre-registered statistics from the per-message rows (also used by `meaningcheck reproduce`)."""
    ch = [x for x in out if x["changed"]]
    m1 = boot([x["fixed"] - x["original"] for x in ch])
    fl = [x["original"] for x in out if x["panel_flagged"]]; ps = [x["original"] for x in out if not x["panel_flagged"]]
    diffs = []
    rng = random.Random(1)
    for _ in range(4000):                      # difference of means, resampling messages within each group
        a = [rng.choice(ps) for _ in ps]; b = [rng.choice(fl) for _ in fl]
        diffs.append(sum(a) / len(a) - sum(b) / len(b))
    diffs.sort()
    m2 = (sum(ps) / len(ps) - sum(fl) / len(fl) if ps and fl else float("nan"), diffs[100], diffs[3899])
    m3 = boot([x["fixed"] - x["baseline"] for x in ch], seed=2)
    return {"readers": "Muse Spark (sampled), the 8 held-out personas", "messages": len(out),
            "HM1_fixed_minus_original": m1[0], "HM1_95ci": [m1[1], m1[2]], "HM1_supported": m1[1] > 0,
            "HM2_passed_minus_flagged": m2[0], "HM2_95ci": [m2[1], m2[2]], "HM2_supported": m2[1] > 0,
            "exploratory_fixed_minus_baseline": m3[0], "exploratory_95ci": [m3[1], m3[2]],
            "lowest_letter_share": min(x["mass"] for x in out),
            "readers_without_any_answer_on_originals": sum(x["no_answer_readers"] for x in out), "rows": out}
