"""Reading a message with a panel: each reader's probability for every reading, from the option letters in one forward
pass, averaged over two option orders (original and reversed) to remove position bias."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .backends import Query
from .core import Case, Persona, ReaderResult

LETTERS = "ABCDEFGH"


def build_prompt(case: Case, text: str, order: List[str]) -> str:
    opts = "\n".join(f"{LETTERS[k]}) {case.reading(rid).text}" for k, rid in enumerate(order))
    if case.mode == "agents":
        head = f"Instruction from your operator:\n\"\"\"\n{text}\n\"\"\"\n\nWhich of these will you do?"
    else:
        head = f"Message you received:\n\"\"\"\n{text}\n\"\"\"\n\nWhat does the sender most likely want you to understand?"
    return f"{head}\n\n{opts}\n\nAnswer with the letter of one option."


def _queries(case: Case, text: str, stated: List[str], panel: List[Persona], removed: Optional[str]):
    ids = [r.id for r in case.readings]
    qs = []
    for p in panel:
        for order in (ids, ids[::-1]):
            qs.append(Query(system=p.system_prompt(case.mode), user=build_prompt(case, text, order),
                            letters=list(LETTERS[:len(order)]), reasoning=p.reasoning,
                            meta={"case": case, "persona": p, "order": order, "stated": stated, "removed": removed}))
    return qs


def _results(case: Case, panel: List[Persona], scored: List[dict]) -> List[ReaderResult]:
    ids = [r.id for r in case.readings]
    orders = [ids, ids[::-1]]
    out = []
    for i, p in enumerate(panel):
        acc: Dict[str, float] = {r: 0.0 for r in ids}
        mass = 0.0
        for j, order in enumerate(orders):
            res = scored[2 * i + j]
            for k, rid in enumerate(order):
                acc[rid] += res["probs"].get(LETTERS[k], 0.0) / len(orders)
            mass += res["mass"] / len(orders)
        # Understood only when the intended reading is strictly the most probable. A tie (for example a reader that gave
        # no usable answer, whose probabilities are all equal) is not understanding; its choice is the strongest rival.
        rivals = {k: v for k, v in acc.items() if k != case.intended}
        best_rival = max(rivals, key=rivals.get) if rivals else None
        understood = best_rival is None or acc.get(case.intended, 0.0) > acc[best_rival] + 1e-12
        choice = case.intended if understood else best_rival
        out.append(ReaderResult(p.id, acc, choice, understood, mass))
    return out


def read(readers, case: Case, text: str, stated: Iterable[str], panel: List[Persona],
         removed: Optional[str] = None) -> List[ReaderResult]:
    return read_many(readers, case, [(text, stated, panel, removed)])[0]


def read_many(readers, case: Case, jobs) -> List[List[ReaderResult]]:
    """Several readings in one call to the model (one request, fuller GPU batches): jobs are
    (text, stated, panel[, removed]) tuples; returns one list of reader results per job."""
    jobs = [tuple(j) + (None,) * (4 - len(j)) for j in jobs]
    qs, spans = [], []
    for text, stated, panel, removed in jobs:
        q = _queries(case, text, list(stated), panel, removed)
        spans.append((len(qs), len(qs) + len(q), panel))
        qs += q
    scored = readers.score(qs) if qs else []
    return [_results(case, panel, scored[a:b]) for a, b, panel in spans]


def rate(results: List[ReaderResult]) -> float:
    return sum(r.understood for r in results) / max(len(results), 1)
