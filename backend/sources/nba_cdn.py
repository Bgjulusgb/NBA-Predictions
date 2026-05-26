"""NBA.com live-data CDN: detailed play-by-play (optional enrichment).

Bot-protected and flaky, so it is strictly best-effort and only used on top of
ESPN during live mode for run/momentum detection. Pre-game it returns no
actions, which is expected.
"""

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_OK, STATUS_PARTIAL, STATUS_ERROR


def _find_game_id(home, away):
    data, res = fetch_json(config.NBA_CDN_SCOREBOARD)
    if not res.ok or data is None:
        return None, res.error
    games = data.get("scoreboard", {}).get("games", [])
    want = {home["abbr"].upper(), away["abbr"].upper()}
    for g in games:
        tri = {g.get("homeTeam", {}).get("teamTricode", "").upper(),
               g.get("awayTeam", {}).get("teamTricode", "").upper()}
        if want & tri == want:
            return g.get("gameId"), None
    return None, f"no matching game in {len(games)} NBA.com games"


def _parse_actions(data):
    actions = data.get("game", {}).get("actions", [])
    events = []
    for a in actions:
        pts = a.get("pointsTotal")
        is_score = a.get("actionType") in ("2pt", "3pt", "freethrow") and \
            a.get("shotResult") == "Made"
        events.append({
            "num": a.get("actionNumber"),
            "period": a.get("period"),
            "clock": a.get("clock"),
            "team": a.get("teamTricode"),
            "type": a.get("actionType"),
            "scoring": bool(is_score),
            "points": _points_for(a),
            "score_home": a.get("scoreHome"),
            "score_away": a.get("scoreAway"),
            "desc": a.get("description"),
            "time_actual": a.get("timeActual"),
        })
    return events


def _points_for(a):
    if a.get("shotResult") != "Made" and a.get("actionType") != "freethrow":
        return 0
    return {"2pt": 2, "3pt": 3, "freethrow": 1}.get(a.get("actionType"), 0)


def fetch_playbyplay(home=None, away=None):
    home = home or config.GAME["home"]
    away = away or config.GAME["away"]
    game_id, err = _find_game_id(home, away)
    if not game_id:
        return SourceResult("nba_cdn", STATUS_PARTIAL, error=err)

    url = config.NBA_CDN_PLAYBYPLAY.format(game_id=game_id)
    data, res = fetch_json(url)
    if not res.ok or data is None:
        return SourceResult("nba_cdn", STATUS_PARTIAL, error=res.error,
                            meta={"game_id": game_id})

    events = _parse_actions(data)
    return SourceResult("nba_cdn", STATUS_OK, records=events,
                        meta={"game_id": game_id, "event_count": len(events)})
