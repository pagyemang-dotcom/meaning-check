"""Human study: do people misread what the panel flags, and does the fix help them?

Design (fixed in docs/HUMAN_STUDY.md before any response):
- Every participant reads all 14 evaluation messages and picks what the sender means from the same options the readers
  saw. Options are listed in a shuffled order fixed per message.
- For each message the tool changed, a participant sees either the original or the fix, never both. Two versions of
  the form (A and B) swap which half of the changed messages appear fixed; participants are routed by birth month (odd:
  A, even: B), so each person contributes to both conditions.
- Messages the tool left unchanged appear as written in both versions.

`build_items` turns results/eval.json into the item list; `apps_script` writes a Google Apps Script that creates the
form; `analyse` reads the responses exported from Google Forms (CSV) and tests the registered hypotheses.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
from typing import Dict, List

COUNTRIES = ["Ghana", "Nigeria", "Kenya", "South Africa", "India", "United Kingdom", "United States", "Other"]


def build_items(eval_json: dict, seed: int = 20261027) -> List[dict]:
    rng = random.Random(seed)
    items = []
    changed = [r for r in eval_json["rows"] if r.get("statements", 0) > 0]
    order = {r["message"]: k for k, r in enumerate(changed)}
    for k, r in enumerate(eval_json["rows"]):
        if "readings" not in r:
            raise ValueError("results/eval.json has no readings: rerun `meaningcheck eval --baseline` with version 1.5")
        opts = list(r["readings"]); rng.shuffle(opts)
        it = {"id": f"M{k + 1:02d}", "mode": r["mode"], "original": r["message"], "fixed": r["rewrite"],
              "changed": r.get("statements", 0) > 0, "panel_flagged": r.get("panel_flagged", r["panel_before"] < 1),
              "intended": r["intended"], "options": opts}
        # form A shows the fix for even-numbered changed messages, form B for odd-numbered ones
        it["fixed_in"] = ("A" if order[r["message"]] % 2 == 0 else "B") if it["changed"] else None
        items.append(it)
    return items


def _question(it: dict, version: str) -> str:
    who = ("You are an AI assistant and your operator sends you this instruction. What do you do?"
           if it["mode"] == "agents" else "You receive this message. What does the sender most likely mean?")
    return who


def text_for(it: dict, form: str) -> str:
    return it["fixed"] if it["changed"] and it["fixed_in"] == form else it["original"]


def tag(it: dict, form: str) -> str:
    v = "F" if it["changed"] and it["fixed_in"] == form else "O"
    return f"[{it['id']}-{v}]"


def apps_script(items: List[dict], title: str = "How would you read these messages?") -> str:
    """Google Apps Script source that builds the form (paste into script.google.com and run createForm)."""
    data = json.dumps([{"tagA": tag(it, "A"), "tagB": tag(it, "B"), "textA": text_for(it, "A"),
                        "textB": text_for(it, "B"), "question": _question(it, "A"),
                        "options": [o["text"] for o in it["options"]]} for it in items], ensure_ascii=False)
    countries = json.dumps(COUNTRIES)
    return f"""// Creates the Meaning Check human-study form. Run createForm once; the form link appears in the log.
function createForm() {{
  var ITEMS = {data};
  var COUNTRIES = {countries};
  var form = FormApp.create({json.dumps(title)});
  form.setDescription("A 10-minute study on how people read short workplace messages. It is anonymous: no names or " +
    "emails are collected. There are no right or wrong answers; pick what you would actually understand. " +
    "By continuing you agree that your anonymous answers may be used in a public research summary.");
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);
  form.setProgressBar(true);
  form.addListItem().setTitle("Which country did you grow up in?").setChoiceValues(COUNTRIES).setRequired(true);
  form.addTextItem().setTitle("What is your first language?").setRequired(true);
  var month = form.addMultipleChoiceItem().setTitle("Is your birthday in an odd-numbered month (Jan, Mar, May, Jul, Sep, Nov)?")
    .setRequired(true);
  var pageA = form.addPageBreakItem().setTitle("Messages");
  ITEMS.forEach(function (it) {{
    form.addMultipleChoiceItem().setTitle(it.tagA + " " + it.question).setHelpText(it.textA)
      .setChoiceValues(it.options).setRequired(true);
  }});
  var pageB = form.addPageBreakItem().setTitle("Messages");
  ITEMS.forEach(function (it) {{
    form.addMultipleChoiceItem().setTitle(it.tagB + " " + it.question).setHelpText(it.textB)
      .setChoiceValues(it.options).setRequired(true);
  }});
  pageB.setGoToPage(FormApp.PageNavigationType.SUBMIT);   // after the A section, submit (skip B)
  month.setChoices([month.createChoice("Yes", pageA), month.createChoice("No", pageB)]);
  Logger.log("Share this link: " + form.getPublishedUrl());
  Logger.log("Edit link (keep private): " + form.getEditUrl());
}}
"""


def _boot(xs: List[float], B: int = 4000, seed: int = 0):
    rng = random.Random(seed); n = len(xs)
    if not n:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    bs = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def parse_responses(csv_text: str, items: List[dict]) -> List[dict]:
    """One record per participant: country, language, and {item id: (version, understood)}."""
    by_id = {it["id"]: it for it in items}
    rows = list(csv.reader(io.StringIO(csv_text)))
    head, body = rows[0], rows[1:]
    out = []
    for row in body:
        rec = {"country": "", "language": "", "answers": {}}
        for h, v in zip(head, row):
            m = re.match(r"\[(M\d\d)-([OF])\]", h.strip())
            if m and v.strip():
                it = by_id[m.group(1)]
                choice = next((o["id"] for o in it["options"] if o["text"].strip() == v.strip()), None)
                rec["answers"][m.group(1)] = (m.group(2), choice == it["intended"])
            elif h.startswith("Which country"):
                rec["country"] = v
            elif h.startswith("What is your first language"):
                rec["language"] = v
        if rec["answers"]:
            out.append(rec)
    return out


def analyse(csv_text: str, items: List[dict]) -> dict:
    people = parse_responses(csv_text, items)
    by_id = {it["id"]: it for it in items}
    # HH1: within each participant, understanding of fixed minus original, over the changed messages they saw
    d1 = []
    for p in people:
        f = [u for k, (v, u) in p["answers"].items() if by_id[k]["changed"] and v == "F"]
        o = [u for k, (v, u) in p["answers"].items() if by_id[k]["changed"] and v == "O"]
        if f and o:
            d1.append(sum(f) / len(f) - sum(o) / len(o))
    # HH2: on original wordings, panel-passed minus panel-flagged understanding
    d2 = []
    for p in people:
        fl = [u for k, (v, u) in p["answers"].items() if v == "O" and by_id[k]["panel_flagged"]]
        ps = [u for k, (v, u) in p["answers"].items() if v == "O" and not by_id[k]["panel_flagged"]]
        if fl and ps:
            d2.append(sum(ps) / len(ps) - sum(fl) / len(fl))
    per_item = {}
    for it in items:
        for v in ("O", "F"):
            us = [p["answers"][it["id"]][1] for p in people if it["id"] in p["answers"] and p["answers"][it["id"]][0] == v]
            if us:
                per_item[f"{it['id']}-{v}"] = {"message": it["fixed"] if v == "F" else it["original"], "n": len(us),
                                               "understood": sum(us) / len(us)}
    countries: Dict[str, int] = {}
    for p in people:
        countries[p["country"] or "not given"] = countries.get(p["country"] or "not given", 0) + 1
    m1, l1, h1 = _boot(d1); m2, l2, h2 = _boot(d2, seed=1)
    return {"participants": len(people), "countries": countries,
            "HH1_fix_minus_original": m1, "HH1_95ci": [l1, h1], "HH1_supported": l1 > 0 if d1 else None,
            "HH2_passed_minus_flagged": m2, "HH2_95ci": [l2, h2], "HH2_supported": l2 > 0 if d2 else None,
            "per_item": per_item}
