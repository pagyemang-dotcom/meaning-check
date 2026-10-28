"""Say enough: the fewest statements that rule out every rival some reader would wrongly prefer.

Audience theorem: a message is understood by every reader in the panel exactly when its statements cover the union of
the readers' dominator sets (rivals each reader finds at least as plausible as the intended reading). Finding the
smallest cover is a 0-1 integer programme; it is solved exactly with HiGHS, which also proves the optimum.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .core import Case


def dominators(probs: Dict[str, float], intended: str, tol: float = 1e-9) -> Set[str]:
    """Rivals the reader finds at least as plausible as the intended reading."""
    p0 = probs.get(intended, 0.0)
    return {r for r, p in probs.items() if r != intended and p >= p0 - tol}


def solve_cover(case: Case, must_exclude: Iterable[str], weights: Optional[Dict[str, float]] = None,
                required: Iterable[str] = ()) -> dict:
    """Minimum-weight set of attributes whose exclusion sets cover `must_exclude`.

    Returns {"feasible", "attributes", "length", "optimal", "lower_bound", "uncoverable"}.
    `required` attributes are forced in (used by the repair loop)."""
    U = sorted(set(must_exclude))
    attrs = case.attributes
    A = len(attrs)
    ex = [set(a.excludes(case.intended)) for a in attrs]
    uncoverable = [j for j in U if not any(j in e for e in ex)]
    target = [j for j in U if j not in uncoverable]
    req = set(required)
    if not target and not req:
        return {"feasible": not uncoverable, "attributes": [], "length": 0, "optimal": True, "lower_bound": 0.0,
                "uncoverable": uncoverable}
    c = np.array([(weights or {}).get(a.id, 1.0) for a in attrs], float)
    rows = [[1.0 if j in ex[k] else 0.0 for k in range(A)] for j in target]
    cons = []
    if rows:
        cons.append(LinearConstraint(np.array(rows), lb=np.ones(len(rows)), ub=np.full(len(rows), np.inf)))
    lb = np.array([1.0 if a.id in req else 0.0 for a in attrs])
    res = milp(c, constraints=cons, integrality=np.ones(A), bounds=Bounds(lb, np.ones(A)),
               options={"mip_rel_gap": 0.0})
    if res.status != 0 or res.x is None:
        return {"feasible": False, "attributes": [], "length": math.inf, "optimal": False, "lower_bound": None,
                "uncoverable": uncoverable}
    chosen = [attrs[k].id for k in range(A) if res.x[k] > 0.5]
    lb_val = float(getattr(res, "mip_dual_bound", res.fun))
    return {"feasible": not uncoverable, "attributes": chosen, "length": len(chosen), "optimal": True,
            "lower_bound": lb_val, "objective": float(res.fun), "uncoverable": uncoverable}


def greedy_cover(case: Case, must_exclude: Iterable[str]) -> List[str]:
    """The greedy rule (within a factor 1 + ln|U| of the optimum); used as a cross-check."""
    left = set(must_exclude); chosen = []
    ex = {a.id: set(a.excludes(case.intended)) for a in case.attributes}
    while left:
        best = max(ex, key=lambda k: len(ex[k] & left))
        if not ex[best] & left:
            break
        chosen.append(best); left -= ex[best]
    return chosen


def open_set(case: Case, stated: Iterable[str]) -> List[str]:
    """Readings still compatible with what was stated."""
    s = [case.attribute(a) for a in stated]
    w = case.intended
    return [r.id for r in case.readings if all(a.values.get(r.id) == a.values.get(w) for a in s)]
