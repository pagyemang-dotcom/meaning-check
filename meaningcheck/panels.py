"""Reader panels. The build panel shapes the message; the held-out panel is never used while building, only to test.

Every reader is the same open-weight model (Muse Glimmer) given a different background and reasoning setting, the
approach the earlier pre-registered study used with four smaller models."""
from __future__ import annotations

from typing import List

from .core import Persona

COUNTRIES = [("United Kingdom", "gb"), ("United States", "us"), ("Ghana", "gh"), ("India", "in")]
BACKGROUNDS = [
    ("office", "You work in an office and read a lot of workplace email."),
    ("trade", "You run a small business and deal with customers and suppliers every day."),
]
REASONING = ["low", "high"]

HELDOUT = [("Nigeria", "ng", "trade"), ("Kenya", "ke", "office"), ("Australia", "au", "office"),
           ("Canada", "ca", "trade"), ("South Africa", "za", "office"), ("Jamaica", "jm", "trade"),
           ("Ireland", "ie", "trade"), ("Pakistan", "pk", "office")]

AGENT_STYLES = [
    ("support", "You handle customer-support tickets."),
    ("ops", "You run operations tasks on company systems."),
]


def people_panel() -> List[Persona]:
    out = []
    for country, flag in COUNTRIES:
        for bkey, btext in BACKGROUNDS:
            for r in REASONING:
                out.append(Persona(f"{flag}-{bkey}-{r}", country, btext, r, flag))
    return out


def people_heldout() -> List[Persona]:
    bg = dict(BACKGROUNDS)
    return [Persona(f"{flag}-{b}-low", c, bg[b], "low", flag) for c, flag, b in HELDOUT]


def agent_panel() -> List[Persona]:
    """Agents: four deployment countries x two job types x two reasoning settings (quick and careful)."""
    out = []
    for country, flag in COUNTRIES:
        for skey, stext in AGENT_STYLES:
            for r in REASONING:
                out.append(Persona(f"agent-{flag}-{skey}-{r}", country, stext, r, flag))
    return out


def agent_heldout() -> List[Persona]:
    st = dict(AGENT_STYLES)
    return [Persona(f"agent-{flag}-{b if b in st else 'ops'}-medium", c, st.get(b, st["ops"]), "medium", flag)
            for c, flag, b in HELDOUT]


def panels(mode: str):
    return (agent_panel(), agent_heldout()) if mode == "agents" else (people_panel(), people_heldout())


def second_panel(mode: str) -> List[Persona]:
    """The second-opinion readers (Muse Spark): the same four countries and two backgrounds or jobs, one reader each."""
    if mode == "agents":
        st = AGENT_STYLES
        return [Persona(f"spark-{flag}-{k}", c, t, "default", flag) for c, flag in COUNTRIES for k, t in st]
    return [Persona(f"spark-{flag}-{k}", c, t, "default", flag) for c, flag in COUNTRIES for k, t in BACKGROUNDS]
