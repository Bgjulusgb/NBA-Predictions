"""NBA Stats API v2 — playoff standings + SRS via stats.nba.com.

Uses the same public column-store endpoint as nba_stats.py but targets the
Playoffs SeasonType so we get W/L records *in the current playoff run*.
Requires browser-like Referer/Origin headers or stats.nba.com returns 403.
"""
from __future__ import annotations

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

STATS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/standings",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

STANDINGS_URL = (
    "https://stats.nba.com/stats/leaguestandingsv3"
    "?LeagueID=00&Season={season}&SeasonType=Playoffs"
)


def fetch_standings_v2(season: str = "2025-26") -> SourceResult:
    url = STANDINGS_URL.format(season=season)
    data, res = fetch_json(url, headers=STATS_HEADERS)
    if not res.ok or not data:
        return SourceResult("nba_stats_v2", STATUS_ERROR, error=res.error)

    rs = next(
        (r for r in (data.get("resultSets") or []) if r.get("name") == "Standings"),
        None,
    )
    if not rs:
        # Try index 0 if name doesn't match (API occasionally uses a different key).
        result_sets = data.get("resultSets") or []
        rs = result_sets[0] if result_sets else None
    if not rs:
        return SourceResult("nba_stats_v2", STATUS_PARTIAL,
                             error="Standings result set not found")

    headers = rs.get("headers") or []
    rows = [dict(zip(headers, row)) for row in (rs.get("rowSet") or [])]
    if not rows:
        return SourceResult("nba_stats_v2", STATUS_PARTIAL, error="empty rowSet")

    home_abbr = config.GAME["home"]["abbr"]
    away_abbr = config.GAME["away"]["abbr"]

    out = []
    for r in rows:
        abbr = r.get("TeamAbbreviation", "")
        if abbr not in (home_abbr, away_abbr):
            continue
        team_city = r.get("TeamCity") or r.get("TeamSlug") or ""
        team_name = r.get("TeamName") or ""
        out.append({
            "team": f"{team_city} {team_name}".strip() or abbr,
            "abbr": abbr,
            "wins": _i(r.get("WINS") or r.get("W")),
            "losses": _i(r.get("LOSSES") or r.get("L")),
            "win_pct": _f(r.get("WinPCT") or r.get("PCT")),
            "l10": r.get("L10") or r.get("LastTenGames"),
            "streak": r.get("strCurrentStreak") or r.get("CurrentStreak"),
            "pts_pg": _f(r.get("PTS_PG") or r.get("PointsPG") or r.get("PTS")),
            "opp_pts_pg": _f(
                r.get("OPP_PTS_PG") or r.get("OppPointsPG") or r.get("OPP_PTS")
            ),
            "home_record": r.get("HOME"),
            "away_record": r.get("ROAD"),
            "source": "nba_stats",
        })

    if not out:
        # Return all rows so callers can still inspect the data.
        all_out = [{"team": (r.get("TeamCity", "") + " " + r.get("TeamName", "")).strip(),
                    "abbr": r.get("TeamAbbreviation"), "source": "nba_stats"}
                   for r in rows[:30]]
        return SourceResult("nba_stats_v2", STATUS_PARTIAL,
                             error=f"teams {home_abbr}/{away_abbr} not found",
                             meta={"standings_v2": all_out})

    return SourceResult("nba_stats_v2", STATUS_OK, records=[],
                         meta={"standings_v2": out, "season": season})


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
