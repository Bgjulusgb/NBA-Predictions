"""Reddit social sentiment (best-effort, keyless .json endpoints).

Reads hot listings from r/nba, the two team subs, and r/sportsbook. Requires a
descriptive User-Agent (set globally in http_util). If Reddit is unreachable
(some sandboxes block it), this returns STATUS_ERROR and the pipeline simply
falls back to news-only sentiment — it never crashes the run.
"""

from datetime import datetime, timezone

from .. import config
from ..http_util import fetch_json
from .base import SourceResult, STATUS_OK, STATUS_PARTIAL, STATUS_ERROR

# Map subreddit -> which team its content is "about" (for attribution).
_SUB_TEAM = {
    "NYKnicks": "away",
    "clevelandcavs": "home",
}


def _iso(created_utc):
    try:
        return datetime.fromtimestamp(float(created_utc), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _parse_listing(data, sub):
    records = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("stickied"):
            continue
        title = d.get("title", "")
        body = d.get("selftext", "") or ""
        records.append({
            "id": f"reddit:{d.get('id')}",
            "source": f"reddit:{sub}",
            "kind": "social",
            "title": title,
            "text": body[:2000],
            "url": "https://www.reddit.com" + d.get("permalink", ""),
            "author": d.get("author"),
            "published": _iso(d.get("created_utc")),
            "engagement": int(d.get("score", 0)) + int(d.get("num_comments", 0)),
            "ups": int(d.get("score", 0)),
            "num_comments": int(d.get("num_comments", 0)),
            "team_hint": _SUB_TEAM.get(sub),
        })
    return records


def fetch_social(subreddits=None, limit=None):
    subreddits = subreddits or config.REDDIT_SUBREDDITS
    limit = limit or config.REDDIT_LIMIT
    records = []
    errors = []
    ok_any = False

    for sub in subreddits:
        url = config.REDDIT_LISTING.format(sub=sub, limit=limit)
        data, res = fetch_json(url)
        if not res.ok or data is None:
            errors.append(f"r/{sub}: {res.error}")
            continue
        try:
            records.extend(_parse_listing(data, sub))
            ok_any = True
        except (KeyError, TypeError) as e:
            errors.append(f"r/{sub}: {e}")

    if not ok_any:
        return SourceResult("reddit", STATUS_ERROR,
                            error="; ".join(errors) or "no subreddits returned")
    status = STATUS_OK if not errors else STATUS_PARTIAL
    return SourceResult("reddit", status, records=records,
                        error="; ".join(errors) or None)
