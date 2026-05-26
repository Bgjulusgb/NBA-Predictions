"""NBA.com hidden Stats API scraper.

NBA.com fronts a stats endpoint at stats.nba.com that requires browser-ish
headers (no API key). Heavily rate-limited and often geo-blocked, so callers
must treat results as best-effort and fall back to ESPN / Sofascore.

Endpoints we use:

  * leaguestandings  - current standings (used for power-ranking input)
  * boxscoresummary  - completed game summary
  * playbyplay       - PBP (cross-check with NBA.com CDN)
  * commonteamroster - active roster (cross-check with config.ROSTERS)
"""

from __future__ import annotations

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

STATS_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

STATS_BASE = "https://stats.nba.com/stats"

LEAGUE_STANDINGS = (
    "/leaguestandingsv3"
    "?LeagueID=00&Season={season}&SeasonType=Regular+Season"
)
BOX_SUMMARY = "/boxscoresummaryv2?GameID={game_id}"
COMMON_ROSTER = "/commonteamroster?LeagueID=00&Season={season}&TeamID={team_id}"


def _result_set(data: dict, name: str) -> list[dict]:
    """NBA.com returns column-store payloads; flip to dict rows."""
    for rs in (data.get("resultSets") or data.get("resultSet") or []):
        if rs.get("name") == name:
            headers = rs.get("headers") or []
            return [dict(zip(headers, row)) for row in rs.get("rowSet") or []]
    return []


def fetch_standings(season: str = "2025-26") -> SourceResult:
    """League standings: wins, losses, win%, PF, PA."""
    url = STATS_BASE + LEAGUE_STANDINGS.format(season=season)
    data, res = fetch_json(url, headers=STATS_HEADERS)
    if not res.ok or not data:
        return SourceResult("nba_stats_standings", STATUS_ERROR,
                             error=res.error)
    rows = _result_set(data, "Standings")
    if not rows:
        return SourceResult("nba_stats_standings", STATUS_PARTIAL,
                             error="empty Standings result set")
    out = []
    for r in rows:
        out.append({
            "team": r.get("TeamCity", "") + " " + r.get("TeamName", ""),
            "abbr": r.get("TeamAbbreviation"),
            "wins": _i(r.get("WINS")), "losses": _i(r.get("LOSSES")),
            "win_pct": _f(r.get("WinPCT")),
            "conference": r.get("Conference"),
            "division": r.get("Division"),
            "points_for_pg": _f(r.get("PointsPG")),
            "points_against_pg": _f(r.get("OppPointsPG")),
            "diff_pg": _f(r.get("DiffPointsPG")),
            "home_record": r.get("HOME"),
            "away_record": r.get("ROAD"),
            "last_10": r.get("L10"),
            "streak": r.get("strCurrentStreak"),
        })
    return SourceResult("nba_stats_standings", STATUS_OK, records=[],
                         meta={"standings": out, "season": season})


def fetch_box_summary(game_id: str) -> SourceResult:
    """Per-team summary stats for a completed game."""
    url = STATS_BASE + BOX_SUMMARY.format(game_id=game_id)
    data, res = fetch_json(url, headers=STATS_HEADERS)
    if not res.ok or not data:
        return SourceResult("nba_stats_box", STATUS_PARTIAL, error=res.error)
    teams = _result_set(data, "TeamStats")
    line = _result_set(data, "LineScore")
    return SourceResult("nba_stats_box",
                         STATUS_OK if teams else STATUS_PARTIAL,
                         records=[], meta={"team_stats": teams, "line": line})


def _i(v):
    try: return int(v)
    except (TypeError, ValueError): return None


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def standings_for_teams(home_abbr: str, away_abbr: str,
                       standings: list[dict]) -> dict:
    """Pick the two team rows we care about from a standings list."""
    out = {"home": None, "away": None}
    for row in standings:
        if row.get("abbr") == home_abbr:
            out["home"] = row
        elif row.get("abbr") == away_abbr:
            out["away"] = row
    return out
