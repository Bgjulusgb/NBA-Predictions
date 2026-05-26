"""Enrichment + import stage.

Takes raw records (articles, social posts), adds sentiment + team attribution,
and assigns a status. Then imports them with the rule the user asked for:

    Records with status "ok" are imported DIRECTLY and are NOT passed through a
    second filtering stage. Only "partial" records get a light repair pass;
    "error" records are dropped.
"""

import re

from . import config
from . import sentiment
from .sources.base import STATUS_OK, STATUS_PARTIAL, STATUS_ERROR


def _build_team_regex(side):
    aliases = sorted(config.GAME[side]["aliases"], key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
                      re.IGNORECASE)


_HOME_RE = _build_team_regex("home")
_AWAY_RE = _build_team_regex("away")


def _attribute_team(text, team_hint):
    """Return 'home' | 'away' | 'both' | None based on alias mentions.

    Uses word-boundary matching so short aliases like "ny"/"cle" don't
    false-match substrings (e.g. "ny" inside "company"/"many").
    """
    if team_hint in ("home", "away"):
        return team_hint
    home_hit = bool(_HOME_RE.search(text))
    away_hit = bool(_AWAY_RE.search(text))
    if home_hit and away_hit:
        return "both"
    if home_hit:
        return "home"
    if away_hit:
        return "away"
    return None


def enrich_record(rec):
    """Add sentiment + team attribution + status to a single record."""
    title = rec.get("title", "") or ""
    body = rec.get("text", "") or ""
    combined = f"{title}. {body}".strip()

    scores = sentiment.score_text(combined)
    rec["sentiment"] = scores
    rec["sentiment_label"] = sentiment.label(scores["compound"])
    rec["team"] = _attribute_team(combined, rec.get("team_hint"))

    rec["status"] = _status_for(rec)
    return rec


def _status_for(rec):
    has_title = bool(rec.get("title", "").strip())
    has_url = bool(rec.get("url"))
    has_team = rec.get("team") is not None
    has_time = bool(rec.get("published"))

    if not has_title or not has_url:
        return STATUS_ERROR
    if has_team and has_time:
        return STATUS_OK
    # Missing attribution or timestamp -> repairable.
    return STATUS_PARTIAL


def _repair(rec):
    """Light repair for 'partial' records. Returns rec or None to drop it."""
    if rec.get("team") is None:
        # Mentions neither team -> general NBA chatter, kept but flagged so it
        # doesn't pollute this matchup's mood meters.
        rec["team"] = "general"
    if not rec.get("published"):
        rec["published"] = None  # keep, timeline simply lacks a timestamp
    # Anything with a title + url is worth keeping after repair.
    return rec if rec.get("title") and rec.get("url") else None


def enrich_and_import(raw_records):
    """Enrich every record, then import per the ok-bypasses-filter rule.

    Returns (imported_records, stats).
    """
    imported = []
    stats = {"ok": 0, "partial": 0, "error": 0, "repaired": 0, "dropped": 0}

    for rec in raw_records:
        enrich_record(rec)
        status = rec["status"]
        stats[status] += 1

        if status == STATUS_OK:
            imported.append(rec)                 # direct import, no re-filter
        elif status == STATUS_PARTIAL:
            repaired = _repair(rec)
            if repaired is not None:
                repaired["imported_via"] = "repair"
                imported.append(repaired)
                stats["repaired"] += 1
            else:
                stats["dropped"] += 1
        else:  # STATUS_ERROR
            stats["dropped"] += 1

    return imported, stats
