"""The meaning-code pipeline.

1. Readings: the writer lists what the message could be taken to mean and what tells the readings apart.
2. Panel: every build reader gives a probability for each reading of the bare message.
3. Who misreads: each reader acts on its most probable reading.
4. Say enough: the fewest statements covering every rival some reader prefers (exact integer programme).
5. Verify and repair: the panel re-reads the message with those statements. AI readers sometimes misread even a full
   description, so a reader that still misreads adds its new rivals to the cover and the programme is solved again.
6. Rewrite: the writer turns the statements into one natural message; it is kept only if the panel understands it at
   least as well as the plain composition.
7. One question: if no single message reaches the panel, readers are grouped so that each group has a message that
   works (price of context), and the writer gives the one question that tells the groups apart.
8. Unseen readers: the final message is read by a held-out panel never used while building.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from . import cover as C
from .core import Case, Persona, ReaderResult, Report
from .panels import panels, second_panel
from concurrent.futures import ThreadPoolExecutor

from .reader import rate, read, read_many
from .writer import Writer, phrases


def kept_words(original: str, new: str) -> float:
    """Share of the original's words that survive, in order (longest common subsequence of words)."""
    a, b = original.lower().split(), new.lower().split()
    if not a:
        return 1.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1] / len(a)


def theorem14_bound(n_attrs: int, n_readers: int, delta: float = 0.05) -> float:
    """With probability 1 − δ over the draw of readers, any statement message understood by all T sampled readers is
    misread by a fresh reader from the same population with probability at most (ln|M| + ln(1/δ))/T, |M| ≤ 2^A."""
    return min(1.0, (n_attrs * math.log(2) + math.log(1 / delta)) / max(n_readers, 1))


class Engine:
    def __init__(self, readers, writer: Writer, max_repairs: int = 4, log=None, second=None,
                 second_name: str = "Muse Spark"):
        """second: readers from a different model (a second opinion). A message counts as misread when either model's
        readers misread it, and the fix is chosen to reach both."""
        self.readers, self.writer, self.max_repairs = readers, writer, max_repairs
        self.second, self.second_name = second, second_name
        self.log = log or (lambda *a: None)

    # ── steps 4–5 ──
    def cover_and_verify(self, case: Case, panel: List[Persona], before: List[ReaderResult], extra=()):
        U = set()
        for r in list(before) + list(extra):
            U |= C.dominators(r.probs, case.intended)
        history = []
        sol = C.solve_cover(case, U)
        stated, after = sol["attributes"], before
        for rnd in range(self.max_repairs + 1):
            text = self.writer.compose(case, stated)
            after = read(self.readers, case, text, stated, panel)
            history.append({"round": rnd, "statements": list(stated), "understood": rate(after)})
            missed = [r for r in after if not r.understood]
            if not missed:
                break
            new = set()
            for r in missed:
                new |= C.dominators(r.probs, case.intended)
            if new <= U:
                # every rival is already ruled out in words, yet some readers still misread: state one more attribute
                # that also rules out what they chose (redundancy helps AI readers; it never hurts literal ones)
                choices = {r.choice for r in missed}
                extra = [a for a in case.attributes if a.id not in stated and set(a.excludes(case.intended)) & choices]
                if not extra:
                    break
                best = max(extra, key=lambda a: len(set(a.excludes(case.intended)) & choices))
                stated = stated + [best.id]
                continue
            U |= new
            sol = C.solve_cover(case, U, required=stated)
            if not sol["attributes"] or set(sol["attributes"]) == set(stated):
                break
            stated = sol["attributes"]
        return stated, after, U, sol, history

    # ── step 7 ──
    def context_cells(self, case: Case, panel: List[Persona], before: Dict[str, ReaderResult]):
        """Greedy partition (price-of-context theorem, part iii): add each reader to the first group that still has a
        message every member understands."""
        cells: List[dict] = []
        unreachable = []
        for p in panel:
            placed = False
            for cell in cells + [None]:
                members = (cell["personas"] if cell else []) + [p]
                U = set()
                for m in members:
                    U |= C.dominators(before[m.id].probs, case.intended)
                sol = C.solve_cover(case, U)
                if not sol["feasible"]:
                    continue
                res = read(self.readers, case, self.writer.compose(case, sol["attributes"]), sol["attributes"], members)
                if all(r.understood for r in res):
                    if cell:
                        cell.update(personas=members, statements=sol["attributes"])
                    else:
                        cells.append({"personas": members, "statements": sol["attributes"]})
                    placed = True
                    break
            if not placed:
                unreachable.append(p.id)
        out = [{"readers": [m.id for m in c["personas"]], "statements": c["statements"],
                "message": self.writer.compose(case, c["statements"])} for c in cells]
        return out, unreachable

    # ── attribution ──
    def attribute_phrases(self, case: Case, panel: List[Persona], base_missed: int) -> list:
        phs = phrases(case.message)
        allres = read_many(self.readers, case, [(case.message.replace(ph, "").strip(), [], panel, ph) for ph in phs])
        return [{"phrase": ph, "misreaders_without_it": sum(not r.understood for r in res),
                 "misreaders_with_it": base_missed} for ph, res in zip(phs, allres)]

    def check(self, message: str, meaning: str = "", mode: str = "people", case: Optional[Case] = None,
              attribution: bool = True, holdout: bool = True, baseline: bool = False,
              second_opinion: bool = True, progress=None) -> Report:
        emit = progress or (lambda *a: None)
        emit("stage", {"name": "Listing the readings"})
        case = case or self.writer.analyse(message, meaning, mode)
        emit("case", case.to_dict())
        panel, held = panels(case.mode)
        self.log(f"{len(case.readings)} readings, {len(case.attributes)} attributes; panel of {len(panel)}")
        use2 = bool(self.second) and second_opinion
        p2 = second_panel(case.mode) if use2 else []
        pool = ThreadPoolExecutor(4)        # the two models are separate services: their readers work at the same time
        f2 = pool.submit(read, self.second, case, case.message, [], p2) if use2 else None
        emit("stage", {"name": "Muse Glimmer readers"})
        # one request for the panel and the unseen readers' reading of the original
        jobs = [(case.message, [], panel)] + ([(case.message, [], held)] if holdout and held else [])
        got = read_many(self.readers, case, jobs)
        before, hb = got[0], (got[1] if len(got) > 1 else None)
        emit("before", [r.__dict__ for r in before])
        rep = Report(case.to_dict(), case.message, before)
        rep.log.append(f"bare message understood by {sum(r.understood for r in before)}/{len(before)} readers")
        if use2:
            if not f2.done():
                emit("stage", {"name": "Muse Spark readers"})
            s_before = f2.result()
            emit("second_before", [r.__dict__ for r in s_before])
        else:
            s_before = []
        emit("stage", {"name": "Shortest fix"})
        if use2:
            rep.log.append(f"{self.second_name}: {sum(r.understood for r in s_before)}/{len(s_before)} understand as written")
        stated, after, U, sol, history = self.cover_and_verify(case, panel, before, extra=s_before)
        rep.statements = stated
        emit("stage", {"name": "Checking wordings"})
        composed = self.writer.compose(case, stated)
        # Candidate wordings: the plain composition, the natural rewrite of the statements, a shortened rewrite, and a
        # direct clear rewrite. The panel reads each; among those understood by the most panel readers, the shortest
        # is kept (ties: the one that keeps more of the sender's words). Held-out readers never take part in this.
        cands = [("composition", composed, after)]
        if stated:
            fd = pool.submit(self.writer.plain_rewrite, case)          # independent of the other two: written alongside
            rw = self.writer.rewrite(case, stated)
            cands.append(("rewrite", rw, None))
            cands.append(("shortened", self.writer.compress(case, stated, rw), None))
            cands.append(("direct", fd.result(), None))
        seen, keep = set(), []
        for name, text, res in cands:
            if not text or text in seen:
                continue
            seen.add(text)
            keep.append((name, text, res))
        # every new wording read in one request; the direct rewrite was not built from the statements, so (for
        # simulated readers) it states none of them
        todo = [(name, text) for name, text, res in keep if res is None]
        got = dict(zip([t for _, t in todo], read_many(self.readers, case, [(t, stated if n != "direct" else [], panel)
                                                                           for n, t in todo])))
        scored = []
        for name, text, res in keep:
            res = res if res is not None else got[text]
            scored.append({"name": name, "text": text, "rate": rate(res), "words": len(text.split()),
                           "kept": kept_words(case.message, text), "res": res})
        best_rate = max(c["rate"] for c in scored)
        if use2:
            # second opinion: the other model reads the best candidates (shortest first, at most 2); the kept wording is
            # the one understood by the most readers across both models, then the shortest
            top = sorted([c for c in scored if c["rate"] == best_rate], key=lambda c: (c["words"], -c["kept"]))[:2]
            for c, r2 in zip(top, read_many(self.second, case, [(c["text"], stated if c["name"] != "direct" else [], p2)
                                                                for c in top])):
                c["res2"], c["rate2"] = r2, rate(r2)
            n1, n2 = len(panel), len(p2)
            pick = min(top, key=lambda c: (-(c["rate"] * n1 + c["rate2"] * n2), c["words"], -c["kept"]))
        else:
            pick = min((c for c in scored if c["rate"] == best_rate), key=lambda c: (c["words"], -c["kept"]))
        rep.rewrite, rep.after = pick["text"], pick["res"]
        if use2:
            rep.second = {"model": self.second_name, "readers": [p.id for p in p2],
                          "before": [r.__dict__ for r in s_before], "after": [r.__dict__ for r in pick["res2"]],
                          "before_rate": rate(s_before), "after_rate": pick["rate2"],
                          "disagree_before": rate(s_before) != rate(before)}
        rep.log.append(f"kept the {pick['name']} wording ({pick['words']} words, panel {pick['rate']:.0%}); "
                       + "; ".join(f"{c['name']}: {c['words']} words, panel {c['rate']:.0%}" for c in scored))
        rep.candidates = [{k: c[k] for k in ("name", "text", "rate", "words", "kept", "rate2") if k in c} for c in scored]
        greedy = C.greedy_cover(case, U)
        rep.certificate = {
            "rivals_to_rule_out": sorted(U),
            "statements": [{"id": a, "statement": case.attribute(a).statement,
                            "rules_out": sorted(set(case.attribute(a).excludes(case.intended)) & U)} for a in stated],
            "minimum_length": sol.get("length") if sol.get("feasible") else None,
            "proved_optimal": bool(sol.get("optimal")) and sol.get("feasible", False),
            "greedy_length": len(greedy),
            "added_after_rereading": max(0, len(stated) - (sol.get("length") or 0)) if sol.get("feasible") else len(stated),
            "repair_rounds": history,
            "panel_understood": rate(rep.after),
            "unseen_reader_bound": theorem14_bound(len(case.attributes), len(panel)),
            "letter_mass_min": min(r.letter_mass for r in before + rep.after),
        }
        if rate(rep.after) < 1.0:                     # the question is built from the main panel's groups
            cells, unreachable = self.context_cells(case, panel, {r.persona: r for r in before})
            q = self.writer.clarify(case, cells) if cells else {"question": "", "messages": {}}
            rep.context = {"cells": cells, "unreachable": unreachable, "bits": math.ceil(math.log2(max(len(cells), 1))),
                           "question": q.get("question", ""), "messages": q.get("messages", {})}
        if holdout and held:
            emit("stage", {"name": "Unseen readers"})
            ha = read(self.readers, case, rep.rewrite, stated, held)
            rep.holdout = {"readers": [p.id for p in held], "before": rate(hb), "after": rate(ha),
                           "before_results": [r.__dict__ for r in hb], "after_results": [r.__dict__ for r in ha]}
            if baseline:
                # comparison: the writer asked to make the message clear, with no reader panel and no cover
                text = self.writer.plain_rewrite(case)
                hp = read(self.readers, case, text, [], held)
                rep.holdout.update(baseline_text=text, baseline=rate(hp))
        if attribution:
            rep.attribution = self.attribute_phrases(case, panel, sum(not r.understood for r in before))
        pool.shutdown(wait=False)
        rep.certificate["writer"] = self.writer.name
        rep.log += list(getattr(self.writer.spark, "events", []) or [])
        return rep
