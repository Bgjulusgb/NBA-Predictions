"""Lineup analysis: turn raw Sofascore lineups into actionable views.

Functions:

  * total_starting_value(lineup)   - sum of `value_share` for the 5 starters
  * bench_value(lineup)            - same for the bench (depth indicator)
  * missing_players_impact(lineup) - share lost to injuries/suspensions
  * starting_advantage(home, away) - net starter-value gap
  * position_breakdown(lineup)     - {position: [player, ...]}
  * top_minutes_players(lineup, n) - leaderboard by current minutes
  * box_score_summary(lineup)      - aggregate per-team box-line totals
"""

from __future__ import annotations

from . import injury_impact


def _share(name: str) -> float:
    return injury_impact.PLAYER_VALUE_SHARE.get(name, 0.05)


def total_starting_value(side: dict | None) -> float:
    if not side:
        return 0.0
    return round(sum(_share(p.get("name", "")) for p in (side.get("starters") or [])), 4)


def bench_value(side: dict | None) -> float:
    if not side:
        return 0.0
    return round(sum(_share(p.get("name", "")) for p in (side.get("bench") or [])), 4)


def missing_players_impact(side: dict | None) -> dict:
    if not side:
        return {"shares_lost": 0.0, "players": []}
    flagged = []
    total = 0.0
    for m in (side.get("missing") or []):
        share = _share(m.get("name", ""))
        flagged.append({**m, "share": share})
        total += share
    return {"shares_lost": round(total, 4), "players": flagged}


def starting_advantage(home: dict | None, away: dict | None) -> dict:
    h = total_starting_value(home)
    a = total_starting_value(away)
    return {
        "home_starting_value": h,
        "away_starting_value": a,
        "net_advantage_home": round(h - a, 4),
    }


def position_breakdown(side: dict | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not side:
        return out
    for p in (side.get("starters") or []) + (side.get("bench") or []):
        pos = p.get("position") or "?"
        out.setdefault(pos, []).append(p.get("name") or "")
    return out


def top_minutes_players(side: dict | None, n: int = 5) -> list[dict]:
    if not side:
        return []
    pool = (side.get("starters") or []) + (side.get("bench") or [])
    rated = [(p, _minutes(p)) for p in pool]
    rated.sort(key=lambda t: t[1] or 0, reverse=True)
    return [{"name": p.get("name"), "minutes": m, "pts": p.get("pts"),
              "plus_minus": p.get("plus_minus")} for p, m in rated[:n]]


def _minutes(p: dict) -> float | None:
    mp = p.get("minutes_played") or p.get("mp")
    if mp is None:
        return None
    s = str(mp)
    if ":" in s:
        try:
            m, sec = s.split(":")
            return int(m) + int(sec) / 60.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def box_score_summary(side: dict | None) -> dict:
    """Sum the per-starter box columns when Sofascore reports them live."""
    if not side:
        return {}
    pool = (side.get("starters") or []) + (side.get("bench") or [])
    totals = {"pts": 0, "ast": 0, "reb": 0, "stl": 0, "blk": 0,
               "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0,
               "ftm": 0, "fta": 0, "tov": 0}
    for p in pool:
        for k in totals:
            v = p.get(k)
            if isinstance(v, (int, float)):
                totals[k] += v
    return totals
