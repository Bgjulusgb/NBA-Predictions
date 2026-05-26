"""Clutch / late-game analysis derived from a play-by-play feed.

NBA convention: "clutch time" = last 5 minutes of regulation (or OT) when the
score margin is within 5 points. Functions here aggregate signals during that
window so the dashboard can report e.g. "Knicks +9 clutch points across
this series" or "Cavs are 1-of-7 in last 5 minutes when within 5".
"""

from __future__ import annotations

from typing import Sequence

from . import model


def _clutch_active(period: int, clock: str, margin: int) -> bool:
    """True if play occurred in the official clutch window."""
    if abs(margin) > 5:
        return False
    if period and period >= 4:
        secs_left = model.seconds_remaining(period, clock)
        # In OT the formula still puts us under 300s well.
        return secs_left <= 300
    return False


def clutch_split(events: Sequence[dict]) -> dict:
    """Split an event stream into clutch vs non-clutch and aggregate scoring.

    Each event needs: team, points, period, clock, score_home, score_away.
    Returns: per-team clutch points + non-clutch points + plus/minus.
    """
    out = {"clutch": {"home_pts": 0, "away_pts": 0,
                      "home_attempts": 0, "away_attempts": 0},
           "non_clutch": {"home_pts": 0, "away_pts": 0,
                           "home_attempts": 0, "away_attempts": 0}}
    for ev in events:
        pts = ev.get("points", 0)
        if pts < 0:
            continue
        margin = (ev.get("score_home") or 0) - (ev.get("score_away") or 0)
        bucket = "clutch" if _clutch_active(ev.get("period", 0),
                                             ev.get("clock", ""),
                                             margin) else "non_clutch"
        team = (ev.get("team") or "").upper()
        attempts_key = None
        pts_key = None
        if team and pts > 0:
            # Lazy match: any event with non-zero points is treated as a make.
            if team == "HOME" or _is_home_event(ev):
                bucket_d = out[bucket]
                bucket_d["home_pts"] += pts
                bucket_d["home_attempts"] += 1
            else:
                bucket_d = out[bucket]
                bucket_d["away_pts"] += pts
                bucket_d["away_attempts"] += 1
    out["clutch_plus_minus"] = (
        out["clutch"]["home_pts"] - out["clutch"]["away_pts"]
    )
    out["non_clutch_plus_minus"] = (
        out["non_clutch"]["home_pts"] - out["non_clutch"]["away_pts"]
    )
    return out


def _is_home_event(ev: dict) -> bool:
    """Best-effort: detect home-side from cached tricodes when 'home'/'away' tag missing."""
    from . import config
    home_tri = (config.GAME["home"]["abbr"] or "").upper()
    team = (ev.get("team") or "").upper()
    return team == home_tri


def clutch_efficiency(events: Sequence[dict]) -> dict:
    """Points per clutch attempt for each team (proxy for execution late)."""
    split = clutch_split(events)
    c = split["clutch"]
    h_eff = c["home_pts"] / c["home_attempts"] if c["home_attempts"] else 0.0
    a_eff = c["away_pts"] / c["away_attempts"] if c["away_attempts"] else 0.0
    return {
        "home_clutch_pp_attempt": round(h_eff, 3),
        "away_clutch_pp_attempt": round(a_eff, 3),
        "edge": round(h_eff - a_eff, 3),
    }


def late_game_swing(timeline: Sequence[dict]) -> dict:
    """Net change in margin between the start and end of the 4th quarter."""
    fourth = [t for t in timeline if t.get("period") == 4]
    if not fourth:
        return {"start_margin": 0, "end_margin": 0, "swing": 0}
    start_m = (fourth[0].get("home") or 0) - (fourth[0].get("away") or 0)
    end_m = (fourth[-1].get("home") or 0) - (fourth[-1].get("away") or 0)
    return {
        "start_margin": start_m,
        "end_margin": end_m,
        "swing": end_m - start_m,
    }
