"""The writer (Muse Spark): lists the readings of a message, the attributes that tell them apart, the statements, the
rewrite and the clarifying question. Every output is checked before use; the reader panel decides what works."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .core import Attribute, Case, Reading

ANALYSE_SYSTEM = """You help a sender make sure every reader understands a message as intended.
Readers come from different countries and backgrounds and read through their own expectations.
Return only a JSON object."""

ANALYSE_USER = """Message: \"\"\"{message}\"\"\"
What the sender means: \"\"\"{meaning}\"\"\"
Readers: {who}

1. List 3 to 6 distinct readings: plausible things a reader could take this message to mean. The first reading must be
   what the sender means. Each reading is one short plain sentence, written as what the reader would understand.
2. List 3 to 8 attributes: questions on which the readings differ (when, who, what exactly, how much, what counts as done,
   and so on). For each attribute give every reading's answer, as a few words. Answers must be consistent: two readings
   share an answer only if they truly agree on it.
3. For each attribute, write the statement: one short sentence the sender could add that states the intended
   reading's answer plainly, without idioms.

JSON shape:
{{"readings": [{{"id": "r1", "text": "..."}}, ...],
  "attributes": [{{"id": "a1", "question": "...", "values": {{"r1": "...", "r2": "...", ...}}, "statement": "..."}}, ...]}}"""

REWRITE_USER = """Original message: \"\"\"{message}\"\"\"
Facts that must be stated: {facts}

Rewrite the message so it states every fact plainly, keeps the sender's tone, adds nothing else, and is as short as
possible. Use plain words, no idioms. Return JSON: {{"rewrite": "..."}}"""

COMPRESS_USER = """Message: \"\"\"{text}\"\"\"
Facts it must still state: {facts}

Make the message as short as possible: one or two plain sentences, no repetition, every fact kept, nothing added.
Return JSON: {{"rewrite": "..."}}"""

PLAIN_USER = """Message: \"\"\"{message}\"\"\"
What the sender means: \"\"\"{meaning}\"\"\"

Rewrite the message so that readers in any country understand it as the sender means. Keep it short.
Return JSON: {{"rewrite": "..."}}"""

CLARIFY_USER = """A message cannot be worded so that every reader understands it the same way.
Message: \"\"\"{message}\"\"\"
Readers fall into these groups, each needing different wording:
{groups}

Write the one short question the sender should ask, or the one fact the sender should establish first, that tells the
groups apart. Then give the message for each group. Return JSON:
{{"question": "...", "messages": {{"group 1": "...", "group 2": "..."}}}}"""


def _who(mode: str) -> str:
    if mode == "agents":
        return "AI agents deployed in the UK, US, Ghana and India, quick and careful reasoning settings"
    return "people in the UK, US, Ghana and India, office workers and small-business owners"


def _validate(d: dict, message: str, meaning: str, mode: str) -> Case:
    readings = [Reading(str(r["id"]), str(r["text"]).strip()) for r in d["readings"]][:8]
    ids = [r.id for r in readings]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("readings must have at least two distinct ids")
    attrs = []
    for a in d["attributes"]:
        vals = {rid: str(a["values"].get(rid, "")).strip().lower() for rid in ids}
        if len(set(vals.values())) < 2:                     # tells nothing apart
            continue
        attrs.append(Attribute(str(a["id"]), str(a["question"]), vals, str(a.get("statement", "")).strip()))
    if not attrs:
        raise ValueError("no attribute tells the readings apart")
    return Case(message, ids[0], readings, attrs, mode, meaning)


class Writer:
    def __init__(self, spark=None, library=None, prefer_library: bool = True):
        self.spark, self.library, self.prefer_library = spark, library, prefer_library

    @property
    def name(self) -> str:
        return getattr(self.spark, "name", "built-in examples") if self.spark is not None else "built-in examples"

    def analyse(self, message: str, meaning: str = "", mode: str = "people") -> Case:
        if self.spark is None:
            return self.library.case_for(message)
        k = (self.library.by_message.get(message.strip().lower())
             if self.library is not None and self.prefer_library else None)
        if k is not None:                              # a built-in example: use its checked readings
            return self.library.get(k)
        d = self.spark.json(ANALYSE_SYSTEM, ANALYSE_USER.format(message=message, meaning=meaning or "(not given)",
                                                                  who=_who(mode)), want=("readings", "attributes"))
        return _validate(d, message, meaning, mode)

    def compose(self, case: Case, stated: List[str]) -> str:
        """Safe composition: the original message followed by the chosen statements (no rewording)."""
        extra = " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in
                         (case.attribute(a).statement for a in stated) if s)
        return (case.message.rstrip() + (" " + extra if extra else "")).strip()

    def compress(self, case: Case, stated: List[str], text: str) -> str:
        """The shortest wording that still states every chosen fact."""
        if self.spark is None:
            return text
        facts = "; ".join(case.attribute(a).statement for a in stated)
        d = self.spark.json(ANALYSE_SYSTEM, COMPRESS_USER.format(text=text, facts=facts), want=("rewrite",))
        return str(d.get("rewrite", "")).strip() or text

    def plain_rewrite(self, case: Case) -> str:
        """Baseline: the writer alone is asked to make the message clear, without readers or statements."""
        if self.spark is None:
            return case.message
        d = self.spark.json(ANALYSE_SYSTEM, PLAIN_USER.format(message=case.message, meaning=case.meaning or "(not given)"),
                            want=("rewrite",))
        return str(d.get("rewrite", "")).strip() or case.message

    def rewrite(self, case: Case, stated: List[str]) -> str:
        if not stated:
            return case.message
        if self.spark is None:
            return self.library.rewrite_for(case, stated) or self.compose(case, stated)
        facts = "; ".join(case.attribute(a).statement for a in stated)
        d = self.spark.json(ANALYSE_SYSTEM, REWRITE_USER.format(message=case.message, facts=facts), want=("rewrite",))
        return str(d.get("rewrite", "")).strip() or self.compose(case, stated)

    def clarify(self, case: Case, cells: List[dict]) -> dict:
        if self.spark is None:
            return self.library.clarify_for(case, cells)
        groups = "\n".join(f"group {k + 1}: readers {', '.join(c['readers'])}; their message would be: {c['message']}"
                           for k, c in enumerate(cells))
        return self.spark.json(ANALYSE_SYSTEM, CLARIFY_USER.format(message=case.message, groups=groups), want=("question",))


def phrases(message: str, max_phrases: int = 8) -> List[str]:
    """Split a message into phrases for attribution (by punctuation, then into groups of words)."""
    parts = [p.strip() for p in re.split(r"[,.;:!?\n]+", message) if p.strip()]
    out = []
    for p in parts:
        w = p.split()
        if len(w) <= 4:
            out.append(p)
        else:
            step = max(2, len(w) // 2)
            out += [" ".join(w[i:i + step]) for i in range(0, len(w), step)]
    return out[:max_phrases]
