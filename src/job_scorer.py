"""
Preference scoring — converts job description metadata into a
0–100 preference score using rules from config/preferences.yaml.

Combines with the semantic fit score (from rag_pipeline) into an
overall composite score.
"""
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "preferences.yaml"


def _load_prefs() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def score_work_arrangement(arrangement: str | None, prefs: dict) -> float:
    """
    Map a work arrangement string to a 0–100 score.
    arrangement should be one of: 'remote', 'hybrid', 'onsite', None/unknown.
    Caller can pass a more specific string; we fuzzy-match it.
    """
    wa = prefs["work_arrangement"]
    if not arrangement:
        return wa["not_specified"]

    arr = arrangement.lower()
    if "remote" in arr and "hybrid" not in arr:
        return wa["fully_remote"]
    if "hybrid" in arr:
        # Without location data we can't compute commute, use hybrid_unknown
        return wa["hybrid_unknown"]
    if "onsite" in arr or "on-site" in arr or "on site" in arr or "office" in arr:
        return wa["onsite_nearby"]   # optimistic default; refine with location
    return wa["not_specified"]


def score_compensation(salary: int | None, prefs: dict) -> float:
    """
    Map a stated salary (annual base, USD) to a 0–100 score.
    Pass None if comp is not listed in the JD.
    """
    comp = prefs["compensation"]
    if salary is None:
        return comp["not_specified"]

    floor = comp["floor"]
    target = comp["target"]
    ceiling = comp["ceiling"]

    if comp.get("floor_is_hard") and salary < floor:
        return 0.0
    if salary >= target:
        # Linear from 100 at target to 100 at ceiling (no bonus beyond ceiling)
        return 100.0
    if salary >= floor:
        # Linear 50–100 between floor and target
        ratio = (salary - floor) / (target - floor)
        return 50.0 + ratio * 50.0
    # Below floor but floor_is_hard is False — still penalise
    ratio = max(0.0, salary / floor)
    return ratio * 50.0


def score_role_priorities(jd_text: str, prefs: dict) -> float:
    """
    Score 0–100 based on how many priority keywords appear in the JD text.
    """
    keywords = prefs.get("role_priorities", [])
    if not keywords:
        return 50.0
    jd_lower = jd_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in jd_lower)
    return min(100.0, (hits / len(keywords)) * 150.0)  # cap at 100, easy to hit 100


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_preference_score(
    jd_text: str,
    work_arrangement: str | None = None,
    salary: int | None = None,
) -> dict:
    """
    Compute a preference score dict for a single job description.

    Returns:
        {
            "preference_score": float,   # 0–100 weighted composite
            "work_arrangement_score": float,
            "compensation_score": float,
            "role_priority_score": float,
        }
    """
    prefs = _load_prefs()
    weights = prefs["weights"]

    wa_score = score_work_arrangement(work_arrangement, prefs)
    comp_score = score_compensation(salary, prefs)
    role_score = score_role_priorities(jd_text, prefs)

    # Preference composite (excludes semantic_fit weight — that's added later)
    pref_weight_total = weights["work_arrangement"] + weights["compensation"] + weights["role_priority"]
    preference_score = (
        wa_score * weights["work_arrangement"]
        + comp_score * weights["compensation"]
        + role_score * weights["role_priority"]
    ) / pref_weight_total * 100.0 / 100.0  # normalise back to 0–100

    return {
        "preference_score": round(preference_score, 1),
        "work_arrangement_score": round(wa_score, 1),
        "compensation_score": round(comp_score, 1),
        "role_priority_score": round(role_score, 1),
    }


def compute_composite_score(semantic_fit: float, preference_score: float) -> float:
    """
    Combine semantic fit (0–100) and preference score (0–100) into a
    single overall score using the weights in preferences.yaml.
    """
    prefs = _load_prefs()
    w = prefs["weights"]
    sf_weight = w["semantic_fit"]
    pref_weight = w["compensation"] + w["work_arrangement"] + w["role_priority"]
    total = sf_weight + pref_weight
    composite = (semantic_fit * sf_weight + preference_score * pref_weight) / total
    return round(composite, 1)
