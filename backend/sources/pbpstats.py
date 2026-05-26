"""PBP Stats — play-by-play advanced stats from pbpstats.com + stats.nba.com fallback.

Fetches per-team OffRating, DefRating, NetRating, Pace, eFG%, TS% for the
current playoff season. Uses pbpstats.com first; falls back to the NBA stats
dashboard endpoint if blocked.
"""
from __future__ import annotations

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

# NBA team IDs (static; won't change mid-season).
TEAM_IDS = {
    "CLE": 1610612739,   # Cleveland Cavaliers
    "NYK": 1610612752,   # New York Knicks
}

PBP_URL = (
    "https://api.pbpstats.com/get-team-stats"
    "?Season={season}&SeasonType=Playoffs&TeamId={team_id}"
)

# NBA Stats fallback (requires browser headers).
NBA_DASH_URL = (
    "https://stats.nba.com/stats/teamdashboardbygeneralsplits"
    "?TeamID={team_id}&Season={season}&SeasonType=Playoffs"
    "&MeasureType=Advanced&PerMode=PerGame&PlusMinus=N&PaceAdjust=N"
    "&Rank=N&Outcome=&Location=&Month=0&SeasonSegment=&DateFrom=&DateTo="
    "&OpponentTeamID=0&VsConference=&VsDivision=&GameSegment=&Period=0"
    "&LastNGames=0&Conference=&Division=&GameScope=&PlayerExperience="
    "&PlayerPosition=&StarterBench=&DraftYear=&DraftPick=&College="
    "&Country=&Height=&Weight=&LeagueID=00&GroupQuantity=5"
)

NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/stats/teams/advanced",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}


def _pbp_stats_for(team_abbr: str, team_id: int, season: str) -> dict | None:
    url = PBP_URL.format(season=season, team_id=team_id)
    data, res = fetch_json(url)
    if not res.ok or not data:
        return None
    # pbpstats response: {"multi_row_response": [...] | {teamId: {stats}}}
    resp = data.get("multi_row_response") or data.get("response") or data
    if isinstance(resp, list) and resp:
        row = resp[0]
    elif isinstance(resp, dict):
        row = resp.get(str(team_id)) or resp.get(team_abbr) or {}
    else:
        return None
    if not row:
        return None
    return {
        "team": team_abbr,
        "off_rtg": _f(row.get("OffRating") or row.get("off_rating")),
        "def_rtg": _f(row.get("DefRating") or row.get("def_rating")),
        "net_rtg": _f(row.get("NetRating") or row.get("net_rating")),
        "pace": _f(row.get("Pace") or row.get("pace")),
        "efg_pct": _f(row.get("eFG%") or row.get("efg_pct")),
        "ts_pct": _f(row.get("TS%") or row.get("ts_pct")),
        "source": "pbpstats",
    }


def _nba_stats_for(team_abbr: str, team_id: int, season: str) -> dict | None:
    url = NBA_DASH_URL.format(team_id=team_id, season=season)
    data, res = fetch_json(url, headers=NBA_HEADERS)
    if not res.ok or not data:
        return None
    for rs in (data.get("resultSets") or []):
        if rs.get("name") in ("TeamDashboardByGeneralSplits", "OverallTeamDashboard"):
            hdrs = rs.get("headers") or []
            rows = rs.get("rowSet") or []
            if not rows:
                continue
            row = dict(zip(hdrs, rows[0]))
            return {
                "team": team_abbr,
                "off_rtg": _f(row.get("OFF_RATING")),
                "def_rtg": _f(row.get("DEF_RATING")),
                "net_rtg": _f(row.get("NET_RATING")),
                "pace": _f(row.get("PACE")),
                "efg_pct": _f(row.get("EFG_PCT")),
                "ts_pct": _f(row.get("TS_PCT")),
                "source": "nba_stats_advanced",
            }
    return None


def fetch_team_stats(season: str = "2025-26") -> SourceResult:
    home = config.GAME["home"]
    away = config.GAME["away"]
    out: list[dict] = []
    errors: list[str] = []

    for team_cfg in (home, away):
        abbr = team_cfg["abbr"]
        team_id = TEAM_IDS.get(abbr)
        if not team_id:
            errors.append(f"{abbr}: no team_id configured")
            continue
        try:
            stats = _pbp_stats_for(abbr, team_id, season)
            if not stats:
                stats = _nba_stats_for(abbr, team_id, season)
            if stats:
                out.append(stats)
            else:
                errors.append(f"{abbr}: no stats from either source")
        except Exception as exc:
            errors.append(f"{abbr}: {exc}")

    if not out:
        return SourceResult("pbpstats", STATUS_PARTIAL,
                             error="; ".join(errors) or "no stats fetched")
    status = STATUS_OK if len(out) == 2 else STATUS_PARTIAL
    return SourceResult("pbpstats", status, records=[],
                         meta={"team_stats": out},
                         error="; ".join(errors) or None)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
