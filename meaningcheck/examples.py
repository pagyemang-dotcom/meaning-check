"""Built-in examples. Each has the readings and attributes the writer would produce, and a simulated-reader world so the
whole pipeline runs offline (tests, first demo). With the real backends these messages are analysed by Muse Spark and
read by Muse Glimmer instead; the simulated worlds are illustrations, not measurements."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

from .core import Attribute, Case, Reading


def _case(message, meaning, mode, readings, attrs, sim, rewrite) -> dict:
    return {"case": Case(message, readings[0].id, readings, attrs, mode, meaning, sim), "rewrite": rewrite}


EXAMPLES: Dict[str, dict] = {}

# 1. A workplace idiom read in opposite ways in British and American English.
EXAMPLES["table"] = _case(
    "Please table the budget item at Friday's meeting.",
    "Put the budget item on Friday's agenda and discuss it then.",
    "people",
    [Reading("r1", "Put the budget item on Friday's agenda and discuss it."),
     Reading("r2", "Postpone the budget item; do not discuss it on Friday."),
     Reading("r3", "Bring printed copies of the budget to the meeting table.")],
    [Attribute("a1", "Is the item discussed on Friday?", {"r1": "yes", "r2": "no", "r3": "not said"},
               "We will discuss the budget on Friday"),
     Attribute("a2", "Is the item being put off?", {"r1": "no", "r2": "yes", "r3": "no"},
               "Please do not postpone it"),
     Attribute("a3", "What must be done before the meeting?", {"r1": "add it to the agenda", "r2": "nothing",
                                                                "r3": "print copies"},
               "Add it to the agenda")],
    {"base": {"r1": 0.0, "r2": 0.0, "r3": -1.2}, "noise": 0.5,
     "country": {"United Kingdom": {"r1": 1.3}, "United States": {"r2": 1.5}, "Ghana": {"r1": 0.7},
                 "India": {"r1": 0.25, "r2": 0.45}, "Nigeria": {"r1": 0.6}, "Kenya": {"r1": 0.6},
                 "Australia": {"r1": 1.0}, "Canada": {"r2": 0.8}, "South Africa": {"r1": 0.9},
                 "Jamaica": {"r1": 0.5}, "Ireland": {"r1": 1.0}, "Pakistan": {"r1": 0.3, "r2": 0.3}},
     "triggers": {"Please table the budget item": {"r2": -1.5, "r1": -1.0}}},
    "Please add the budget item to Friday's agenda; we will discuss it at the meeting.")

# 2. An instruction to an AI agent with an unstated decision rule.
EXAMPLES["ticket"] = _case(
    "Summarise the ticket and close it if it's resolved.",
    "Write a summary; close the ticket only after the customer confirms the fix.",
    "agents",
    [Reading("r1", "Write a summary, and close the ticket only if the customer has confirmed the fix."),
     Reading("r2", "Write a summary, and close the ticket if I judge the problem solved."),
     Reading("r3", "Write a summary and close the ticket now."),
     Reading("r4", "Write a summary only; a person decides whether to close it.")],
    [Attribute("a1", "Who decides that it is resolved?", {"r1": "the customer", "r2": "the agent",
                                                         "r3": "nobody", "r4": "a person"},
               "Treat it as resolved only when the customer has confirmed the fix"),
     Attribute("a2", "May the agent close it without customer confirmation?",
               {"r1": "no", "r2": "yes", "r3": "yes", "r4": "no"},
               "Never close it without the customer's confirmation"),
     Attribute("a3", "Does the agent close tickets itself?", {"r1": "yes", "r2": "yes", "r3": "yes", "r4": "no"},
               "You close the ticket yourself")],
    {"base": {"r1": 0.0, "r2": 0.7, "r3": -0.6, "r4": -0.3}, "noise": 0.45,
     "reasoning": {"low": {"r3": 1.0}, "high": {"r1": 0.45}, "medium": {"r2": 0.2}},
     "triggers": {"close it if it's resolved": {"r2": -1.2, "r3": -1.0}}},
    "Summarise the ticket. Close it yourself, but only after the customer confirms the fix; never close it without that confirmation.")

# 3. A date in numeric form.
EXAMPLES["date"] = _case(
    "Please send the signed form by 03/04.",
    "The deadline is 3 April.",
    "people",
    [Reading("r1", "The deadline is 3 April."),
     Reading("r2", "The deadline is 4 March."),
     Reading("r3", "Send the form within three to four days.")],
    [Attribute("a1", "Which month is the deadline in?", {"r1": "april", "r2": "march", "r3": "not a date"},
               "The deadline is in April"),
     Attribute("a2", "What is the deadline, written out?", {"r1": "3 april", "r2": "4 march", "r3": "in 3-4 days"},
               "The deadline is 3 April")],
    {"base": {"r1": 0.0, "r2": 0.0, "r3": -1.4}, "noise": 0.35,
     "country": {"United Kingdom": {"r1": 1.4}, "United States": {"r2": 1.6}, "Ghana": {"r1": 1.0},
                 "India": {"r1": 1.0}, "Canada": {"r2": 0.5, "r1": 0.3}, "Jamaica": {"r1": 0.6},
                 "Nigeria": {"r1": 1.0}, "Kenya": {"r1": 1.0}, "Australia": {"r1": 1.3},
                 "South Africa": {"r1": 0.9}, "Ireland": {"r1": 1.3}, "Pakistan": {"r1": 1.0}},
     "triggers": {"by 03/04": {"r2": -1.0, "r1": -1.0}}},
    "Please send the signed form by 3 April.")

# 4. Read correctly by one model's readers and misread by the other's: why a second opinion matters.
EXAMPLES["biweekly"] = _case(
    "Our biweekly sync starts next week.",
    "The sync happens once every two weeks.",
    "people",
    [Reading("r1", "The sync meets once every two weeks."),
     Reading("r2", "The sync meets twice a week."),
     Reading("r3", "Only the first sync is next week; how often is not said.")],
    [Attribute("a1", "How often does the sync meet?", {"r1": "every two weeks", "r2": "twice a week",
                                                      "r3": "not said"}, "We meet once every two weeks"),
     Attribute("a2", "Is it more than once a week?", {"r1": "no", "r2": "yes", "r3": "not said"},
               "It is not twice a week")],
    {"base": {"r1": 0.9, "r2": 0.0, "r3": -1.0}, "noise": 0.3,
     "second": {"base": {"r1": 0.0, "r2": 0.45, "r3": -1.0}}},
    "Our sync, held once every two weeks, starts next week.")

# Workplace scenarios: a deadline at risk, feedback to a colleague, a status update in a shared channel.
EXAMPLES["deadline"] = _case(
    "We can probably hit the 15th if nothing else slips.",
    "The 15th is at risk; I need your decision on cutting scope by Friday.",
    "people",
    [Reading("r1", "The 15th is at risk, and the sender needs a decision on cutting scope by Friday."),
     Reading("r2", "The team expects to deliver on the 15th; nothing is needed from me."),
     Reading("r3", "The team is asking to move the deadline to a later date.")],
    [Attribute("a1", "Is the 15th at risk?", {"r1": "yes", "r2": "no", "r3": "missed"}, "The 15th is at risk"),
     Attribute("a2", "Is anything needed from the reader?", {"r1": "a scope decision by friday", "r2": "nothing",
                                                            "r3": "approve a new date"},
               "I need your decision on cutting scope by Friday")],
    {"base": {"r1": 0.0, "r2": 0.8, "r3": -0.4}, "noise": 0.45,
     "reasoning": {"high": {"r1": 0.5}}, "country": {"United Kingdom": {"r1": 0.3}},
     "second": {"base": {"r1": -0.2, "r2": 0.9, "r3": -0.2}}},
    "The 15th is at risk. I need your decision on cutting scope by Friday.")

EXAMPLES["feedback"] = _case(
    "Interesting approach in your draft, maybe worth another look at section 3.",
    "Section 3 has a factual error that must be fixed before Thursday's review.",
    "people",
    [Reading("r1", "Section 3 has an error I must fix before Thursday's review."),
     Reading("r2", "An optional suggestion: I could revisit section 3 if I have time."),
     Reading("r3", "A compliment on my approach; nothing needs changing.")],
    [Attribute("a1", "Is a change required?", {"r1": "yes", "r2": "optional", "r3": "no"},
               "Section 3 must be changed"),
     Attribute("a2", "What is wrong?", {"r1": "a factual error", "r2": "could be better", "r3": "nothing"},
               "It contains a factual error"),
     Attribute("a3", "By when?", {"r1": "before thursday's review", "r2": "no deadline", "r3": "no deadline"},
               "Please fix it before Thursday's review")],
    {"base": {"r1": -0.3, "r2": 0.7, "r3": 0.1}, "noise": 0.4,
     "country": {"United Kingdom": {"r1": 0.6}, "India": {"r3": 0.3}},
     "second": {"base": {"r1": -0.6, "r2": 0.6, "r3": 0.4}}},
    "Section 3 has a factual error; please fix it before Thursday's review.")

EXAMPLES["status"] = _case(
    "API migration: mostly done, a couple of loose ends, should be fine for launch.",
    "Two endpoints still fail tests; launch is safe only if they are fixed by Wednesday.",
    "people",
    [Reading("r1", "Two endpoints still fail; launch is safe only if they are fixed by Wednesday."),
     Reading("r2", "The migration is effectively finished and the launch is safe."),
     Reading("r3", "The launch will probably be delayed.")],
    [Attribute("a1", "Is the launch safe as things stand?", {"r1": "only if fixed by wednesday", "r2": "yes",
                                                             "r3": "no"},
               "Launch is safe only if they are fixed by Wednesday"),
     Attribute("a2", "What is still open?", {"r1": "two failing endpoints", "r2": "minor tidying",
                                             "r3": "major work"}, "Two endpoints still fail tests")],
    {"base": {"r1": -0.1, "r2": 0.9, "r3": -0.5}, "noise": 0.4,
     "reasoning": {"high": {"r1": 0.6}},
     "second": {"base": {"r1": 0.2, "r2": 0.7, "r3": -0.3}}},
    "API migration: two endpoints still fail tests. Launch is safe only if they are fixed by Wednesday.")

# 5. (tests) A case where one group does not share the literal meaning of any available statement, so no single message
# reaches everyone and one clarifying question is needed.
EXAMPLES["plate"] = _case(
    "Bring a plate to the team lunch on Friday.",
    "Bring a dish of food to share.",
    "people",
    [Reading("r1", "Bring a dish of food to share."),
     Reading("r2", "Bring an empty plate to eat from."),
     Reading("r3", "Lunch is provided; bring nothing.")],
    [Attribute("a1", "Is food to be brought?", {"r1": "yes", "r2": "no", "r3": "no"}, "Bring a plate of food"),
     Attribute("a2", "Is lunch provided?", {"r1": "partly, shared", "r2": "yes", "r3": "yes"},
               "Everyone's plates make up the lunch")],
    {"base": {"r1": 0.0, "r2": 0.3, "r3": -1.0}, "noise": 0.3,
     "country": {"United Kingdom": {"r1": 0.8}, "Ghana": {"r2": 0.9}, "India": {"r2": 0.9}},
     "ignore": {"Ghana": ["a1", "a2"], "India": ["a1", "a2"]}},
    "Bring a plate of food to share at Friday's team lunch.")


class Library:
    def __init__(self):
        self.by_message = {v["case"].message.strip().lower(): k for k, v in EXAMPLES.items()}

    def keys(self) -> List[str]:
        return list(EXAMPLES)

    def get(self, key: str) -> Case:
        return copy.deepcopy(EXAMPLES[key]["case"])

    def case_for(self, message: str) -> Case:
        k = self.by_message.get(message.strip().lower())
        if k is None:
            raise ValueError("The offline mode only knows the built-in examples: " +
                             "; ".join(EXAMPLES[x]["case"].message for x in EXAMPLES) +
                             ". Set MODEL_API_KEY and GLIMMER_URL to check any message.")
        return self.get(k)

    def rewrite_for(self, case: Case, stated: List[str]) -> Optional[str]:
        k = self.by_message.get(case.message.strip().lower())
        return EXAMPLES[k]["rewrite"] if k else None

    def clarify_for(self, case: Case, cells: List[dict]) -> dict:
        return {"question": "Before sending: will your readers expect to bring food, or only a plate to eat from?",
                "messages": {f"group {k + 1}": c["message"] for k, c in enumerate(cells)}}
