import csv
import io
import json

from meaningcheck.human import analyse, apps_script, build_items


def fake_eval():
    rows = []
    for k in range(6):
        changed = k < 4
        rows.append({"message": f"orig {k}", "mode": "agents" if k == 5 else "people", "rewrite": f"fixed {k}" if changed else f"orig {k}",
                     "statements": 1 if changed else 0, "panel_before": 0.2 if changed else 1.0, "panel_flagged": changed,
                     "intended": "r1", "readings": [{"id": "r1", "text": f"right {k}"}, {"id": "r2", "text": f"wrong {k}"}]})
    return {"rows": rows}


def test_items_and_script():
    items = build_items(fake_eval())
    assert len(items) == 6 and sum(i["changed"] for i in items) == 4
    assert {i["fixed_in"] for i in items if i["changed"]} == {"A", "B"}
    js = apps_script(items)
    assert "createForm" in js and "[M01-F]" in js and "[M01-O]" in js and "pageB.setGoToPage" in js
    json.loads(js[js.index("var ITEMS = ") + 12: js.index(";\n  var COUNTRIES")])          # valid embedded data


def test_analysis_detects_effects():
    items = build_items(fake_eval())
    heads = ["Timestamp", "Which country did you grow up in?", "What is your first language?", "odd month"]
    for form in "AB":
        for it in items:
            v = "F" if it["changed"] and it["fixed_in"] == form else "O"
            heads.append(f"[{it['id']}-{v}] question")
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(heads)
    for n in range(20):
        form = "AB"[n % 2]; row = ["t", "Ghana", "English", "Yes" if form == "A" else "No"]
        for f2 in "AB":
            for it in items:
                if f2 != form:
                    row.append(""); continue
                fixed = it["changed"] and it["fixed_in"] == form
                ok = fixed or not it["panel_flagged"] or n % 4 == 0
                row.append(next(o["text"] for o in it["options"] if (o["id"] == "r1") == ok))
        w.writerow(row)
    res = analyse(buf.getvalue(), items)
    assert res["participants"] == 20 and res["HH1_supported"] and res["HH2_supported"]
    assert res["countries"] == {"Ghana": 20}
