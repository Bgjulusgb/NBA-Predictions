"""Injury-impact modelling.

Sentiment scraping picks up injury keywords ("injured", "questionable", "out");
this module turns that signal into a per-team probability adjustment.

Each player carries a baseline "value share" — a rough fraction of team output
they account for. When an injury narrative crosses a threshold for a player we
subtract their share from the team and propagate through to a win-probability
shift via simulation.what_if_player_out().
"""

from __future__ import annotations

import re
from collections import Counter

from . import config


# Approximate share-of-team-output, calibrated to a typical NBA stars-and-role
# player distribution. The dashboard uses these directly; in production these
# would come from RAPM or advanced plus-minus.
PLAYER_VALUE_SHARE = {
    # Cavaliers
    "Donovan Mitchell": 0.28,
    "Darius Garland": 0.16,
    "Evan Mobley": 0.18,
    "Jarrett Allen": 0.13,
    "Max Strus": 0.08,
    "De'Andre Hunter": 0.07,
    "Ty Jerome": 0.04,
    # Knicks
    "Jalen Brunson": 0.27,
    "Karl-Anthony Towns": 0.20,
    "OG Anunoby": 0.14,
    "Josh Hart": 0.10,
    "Mikal Bridges": 0.13,
    "Mitchell Robinson": 0.05,
    "Miles McBride": 0.05,
}

INJURY_KEYWORDS = {
    "injured", "injury", "out", "questionable", "doubtful", "ruled out",
    "season ending", "sprained", "torn", "strain", "fractured",
}

POSITIVE_KEYWORDS = {"return", "returns", "cleared", "active", "available"}


def _alias_set():
    aliases = {}
    for side in ("home", "away"):
        for name, alist in config.ROSTERS.get(side, {}).items():
            for a in alist:
                aliases[a.lower()] = (name, side)
    return aliases


def detect_injury_signals(records) -> dict:
    """Score injury chatter per player from a list of enriched records.

    Returns: { player_name -> {team, injury_hits, return_hits, net_signal} }
    """
    aliases = _alias_set()
    hits = {}
    for rec in records:
        text = f"{rec.get('title','')} {rec.get('text','')}".lower()
        if not text.strip():
            continue
        injury_hit = any(re.search(rf"\b{k}\b", text) for k in INJURY_KEYWORDS)
        pos_hit = any(re.search(rf"\b{k}\b", text) for k in POSITIVE_KEYWORDS)
        if not (injury_hit or pos_hit):
            continue
        for alias, (name, side) in aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                h = hits.setdefault(name, {"team": side, "injury_hits": 0,
                                            "return_hits": 0})
                if injury_hit:
                    h["injury_hits"] += 1
                if pos_hit:
                    h["return_hits"] += 1
    for name, h in hits.items():
        h["net_signal"] = h["injury_hits"] - h["return_hits"]
    return hits


def estimated_team_impact(records, threshold: int = 5,
                          max_team_share: float = 0.35) -> dict:
    """Subtract value-share for any player whose net injury signal >= threshold.

    A team's total deducted share is capped at `max_team_share` so a wave of
    routine mentions cannot blow the model past plausible bounds (in practice
    you'd never lose 70%+ of a roster's value to injuries in one playoff game).
    """
    hits = detect_injury_signals(records)
    impact = {"home": 0.0, "away": 0.0, "players_flagged": []}
    for name, h in hits.items():
        if h["net_signal"] >= threshold:
            share = PLAYER_VALUE_SHARE.get(name, 0.05)
            impact[h["team"]] += share
            impact["players_flagged"].append({
                "name": name, "team": h["team"], "share": share,
                "net_signal": h["net_signal"],
            })
    impact["home"] = round(min(max_team_share, impact["home"]), 4)
    impact["away"] = round(min(max_team_share, impact["away"]), 4)
    return impact


def adjust_win_probability(home_win_prob: float, impact: dict,
                            sensitivity: float = 22.0) -> dict:
    """Apply the impact dict to a home win prob.

    Shift in margin per share-point = sensitivity (~22 points for a full team).
    """
    import math
    from . import model
    sigma = 12.0
    base_margin = sigma * model._inv_norm_cdf(
        max(0.01, min(0.99, home_win_prob)))
    margin_shift = (impact.get("away", 0.0) - impact.get("home", 0.0)) * sensitivity
    new_margin = base_margin + margin_shift
    new_p = 0.5 * (1.0 + math.erf((new_margin / sigma) / math.sqrt(2.0)))
    return {
        "base_prob": round(home_win_prob, 4),
        "adjusted_prob": round(new_p, 4),
        "margin_shift": round(margin_shift, 2),
        "delta": round(new_p - home_win_prob, 4),
    }
