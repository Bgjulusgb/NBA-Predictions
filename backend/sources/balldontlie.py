"""BallDontLie NBA season stats + recent game log.

Free public API (no auth required for basic lookups). We fetch team IDs,
then the last 25 games for each team and compute avg PTS, avg OPP PTS, and
last-5 record.
"""
from __future__ import annotations

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

BDL_BASE = "https://api.balldontlie.io/v1"

BDL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}


def _find_team_id(name_fragment: str) -> int | None:
    data, res = fetch_json(f"{BDL_BASE}/teams", headers=BDL_HEADERS)
    if not res.ok or not data:
        return None
    for team in (data.get("data") or []):
        full = team.get("full_name", "").lower()
        nick = team.get("name", "").lower()
        if name_fragment.lower() in full or name_fragment.lower() in nick:
            return team.get("id")
    return None


def _team_stats(team_name: str, team_id: int) -> dict | None:
    url = f"{BDL_BASE}/games?seasons[]=2024&per_page=25&team_ids[]={team_id}"
    data, res = fetch_json(url, headers=BDL_HEADERS)
    if not res.ok or not data:
        return None
    games = data.get("data") or []
    if not games:
        return None

    pts_list: list[int] = []
    opp_list: list[int] = []
    for g in games:
        home_id = (g.get("home_team") or {}).get("id")
        hs = int(g.get("home_team_score") or 0)
        vs = int(g.get("visitor_team_score") or 0)
        if home_id == team_id:
            pts_list.append(hs)
            opp_list.append(vs)
        else:
            pts_list.append(vs)
            opp_list.append(hs)

    wins_last5 = 0
    for g in games[:5]:
        home_id = (g.get("home_team") or {}).get("id")
        hs = int(g.get("home_team_score") or 0)
        vs = int(g.get("visitor_team_score") or 0)
        team_score = hs if home_id == team_id else vs
        opp_score = vs if home_id == team_id else hs
        if team_score > opp_score:
            wins_last5 += 1

    return {
        "team": team_name,
        "avg_pts": round(sum(pts_list) / len(pts_list), 1) if pts_list else None,
        "avg_opp_pts": round(sum(opp_list) / len(opp_list), 1) if opp_list else None,
        "last5_wins": wins_last5,
        "games_fetched": len(games),
        "source": "balldontlie",
    }


def fetch_team_stats() -> SourceResult:
    home = config.GAME["home"]
    away = config.GAME["away"]
    out = []
    errors = []
    for team_cfg in (home, away):
        try:
            # Use last word of team name (e.g. "Cavaliers", "Knicks") for lookup.
            fragment = team_cfg["name"].split()[-1]
            team_id = _find_team_id(fragment)
            if not team_id:
                errors.append(f"{team_cfg['name']}: team ID not found")
                continue
            stats = _team_stats(team_cfg["name"], team_id)
            if stats:
                out.append(stats)
            else:
                errors.append(f"{team_cfg['name']}: no games found")
        except Exception as exc:
            errors.append(f"{team_cfg['name']}: {exc}")

    if not out:
        return SourceResult("balldontlie", STATUS_PARTIAL,
                             error="; ".join(errors) or "no stats fetched")
    status = STATUS_OK if len(out) == 2 else STATUS_PARTIAL
    return SourceResult("balldontlie", status, records=[],
                         meta={"team_stats": out},
                         error="; ".join(errors) or None)
