import math
from recidivism_abm.config.risk_config import get_peer_influence_config, get_flat_risk_weights


def safe_weight(weights, key, agent_id=None):
    value = weights.get(key)
    if isinstance(value, (int, float)):
        return value
    return 0.0


def _get(agent, key, default=None):
    """
    Read a live agent attribute first; fall back to agent_data dict.
    Priority order:
        1. Live agent attribute (reflects dynamic simulation updates)
        2. agent_data dict (static initialization snapshot, used as fallback)
        3. Caller-supplied default
    """
    # 1. Live attribute
    live = getattr(agent, key, _SENTINEL)
    if live is not _SENTINEL:
        return live

    # 2. Static snapshot fallback
    data = getattr(agent, "agent_data", None)
    if isinstance(data, dict) and key in data:
        return data[key]

    # 3. Default
    return default


_SENTINEL = object()  # unique sentinel for getattr miss detection


def compute_risk_score(agent, weights=None):

    score = 0.0

    try:
        # ── Weight resolution (unchanged logic, kept for compatibility) ──────
        if weights is None:
            weights = getattr(getattr(agent, "model", None), "weights", None)
        if weights is None:
            weights = get_flat_risk_weights()

        # ── Demographics ──────────────────────────────────────────────────────
        age = _get(agent, "Age_at_Release", 0) or 0
        score += weights["Age_at_Release"] * math.log(1 + age)

        pct_employed = (_get(agent, "Percent_Days_Employed", 0) or 0) / 100.0
        score += weights["Percent_Days_Employed"] * pct_employed

        dependents = _get(agent, "Dependents", 0) or 0
        score += weights["Dependents"] * dependents

        # ── Education ─────────────────────────────────────────────────────────
        # FIX: normalize spaces so "Less than HighSchool Diploma" maps cleanly
        edu_raw = _get(agent, "Education_Level", "None") or "None"
        edu_normalized = edu_raw.replace(" ", "")          # strip all spaces
        edu_key = f"Education_{edu_normalized}"
        score += weights.get(edu_key, 0.0)

        # ── Supervision Profile ───────────────────────────────────────────────
        # FIX: use "Supervision_Level_First_{level}" to match weights dict keys
        # Original used "Supervision_Level_{level}" which never matched.
        supervision_level = _get(agent, "Supervision_Level_First", "Standard") or "Standard"
        supervision_key = f"Supervision_Level_First_{supervision_level}"
        score += safe_weight(weights, supervision_key, agent.unique_id)

        supervision_risk = _get(agent, "Supervision_Risk_Score", 0) or 0
        score += weights["Supervision_Risk_Score"] * supervision_risk

        # ── Social Risk Factors ───────────────────────────────────────────────
        gang = int(_get(agent, "Gang_Affiliated", False) or False)
        score += weights["Gang_Affiliated"] * gang

        residence_changes = _get(agent, "Residence_Changes", 0) or 0
        score += weights["Residence_Changes"] * residence_changes

        # ── Criminal Offense ──────────────────────────────────────────────────
        offense = _get(agent, "offense", "Other(PublicOrder)") or "Other(PublicOrder)"
        offense_key = f"offense_{offense}"
        score += weights.get(offense_key, weights.get("offense_Other(PublicOrder)", 0.0))

        # ── Conviction History (log-compressed to reduce outlier influence) ───
        for field in (
            "Prior_Conviction_Episodes_Violent",
            "Prior_Conviction_Episodes_Property",
            "Prior_Conviction_Episodes_Drug",
            "Prior_Conviction_Episodes_Misd",
            "Prior_Conviction_Episodes_Felony",
        ):
            val = _get(agent, field, 0) or 0
            score += weights[field] * math.log(1 + val)

        # ── Prior Revocations ─────────────────────────────────────────────────
        # FIX: explicit int() cast — agent attribute is boolean in some states
        revocations = int(_get(agent, "Prior_Revocations_Supervision", False) or False)
        score += weights["Prior_Revocations_Supervision"] * math.log(1 + revocations)

        # ── Behavioral Conditions ─────────────────────────────────────────────
        score += weights["Condition_MH_SA"]  * int(_get(agent, "Condition_MH_SA",  False) or False)
        score += weights["Condition_Cog_Ed"] * int(_get(agent, "Condition_Cog_Ed", False) or False)
        score += weights["Condition_Other"]  * int(_get(agent, "Condition_Other",  False) or False)

        # ── Supervision Violations ────────────────────────────────────────────
        # FIX: explicit int() cast — stored as boolean in agent_data
        violations = int(_get(agent, "Violations_Technical", False) or False)
        score += weights["Violations_Technical"] * violations

        # ── Program Participation ─────────────────────────────────────────────
        attendances = _get(agent, "Program_Attendances", 0) or 0
        score += weights["Program_Attendances"] * attendances

        unexcused = _get(agent, "Program_UnexcusedAbsences", 0) or 0
        score += weights["Program_UnexcusedAbsences"] * unexcused

    except Exception as e:
        print(f"Agent {agent.unique_id} scoring error: {e}")

    return score


def normalize_score(score, score_min, score_max):
    """Min-max normalize a raw score to [0, 1]."""
    if score_max == score_min:
        return 0.5  # avoid divide-by-zero; treat as median risk
    return max(0.0, min(1.0, (score - score_min) / (score_max - score_min)))