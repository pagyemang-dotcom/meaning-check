"""Core objects: readings (intents), attributes (what tells readings apart) and statements.

Following the statement theory of being understood: a message about the intended reading w may add statements, each
naming w's value on one attribute. A statement rules out every reading whose value on that attribute differs (its
exclusion set). A reader acts on the most plausible reading that the message leaves open.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Reading:
    id: str            # short key, e.g. "r1"
    text: str          # what the reader would take the message to mean, in plain words


@dataclass
class Attribute:
    id: str                    # short key, e.g. "a1"
    question: str              # e.g. "When should the item be discussed?"
    values: Dict[str, str]     # reading id -> that reading's answer
    statement: str = ""        # the sentence that states the intended reading's value (filled by the writer)

    def excludes(self, intended: str) -> List[str]:
        """Readings ruled out by stating the intended value of this attribute."""
        v = self.values.get(intended)
        return [r for r, x in self.values.items() if r != intended and x != v]


@dataclass
class Case:
    """Everything the engine needs about one message."""
    message: str
    intended: str                       # id of the reading the sender means
    readings: List[Reading]
    attributes: List[Attribute]
    mode: str = "people"                # "people" or "agents"
    meaning: str = ""                   # the sender's own one-line description of what they mean
    sim: Optional[dict] = None          # simulated-reader world (only for the built-in examples and tests)

    def reading(self, rid: str) -> Reading:
        return next(r for r in self.readings if r.id == rid)

    def attribute(self, aid: str) -> Attribute:
        return next(a for a in self.attributes if a.id == aid)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("sim", None)
        return d


@dataclass
class Persona:
    id: str
    country: str
    background: str
    reasoning: str = "low"              # Muse Glimmer reasoning setting used for this reader
    flag: str = ""

    @property
    def place(self) -> str:
        return ("the " + self.country) if self.country in ("United Kingdom", "United States") else self.country

    def system_prompt(self, mode: str) -> str:
        if mode == "agents":
            return (f"You are an AI assistant deployed by a team in {self.place}. {self.background} "
                    "You receive instructions from your operator and act on what you believe they want.")
        return (f"You live in {self.place}. {self.background} "
                "You read messages the way people around you usually would.")


@dataclass
class ReaderResult:
    persona: str
    probs: Dict[str, float]          # reading id -> probability
    choice: str                      # most probable reading
    understood: bool
    letter_mass: float = 1.0         # share of next-token probability on the option letters (quality check)


@dataclass
class Report:
    case: dict
    message_checked: str
    before: List[ReaderResult]
    statements: List[str] = field(default_factory=list)       # attribute ids chosen
    rewrite: str = ""
    after: List[ReaderResult] = field(default_factory=list)
    certificate: dict = field(default_factory=dict)
    context: Optional[dict] = None
    holdout: Optional[dict] = None
    attribution: Optional[list] = None
    candidates: List[dict] = field(default_factory=list)
    second: Optional[dict] = None                              # second-opinion readers (another model)
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
