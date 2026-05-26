"""CBS Sports NBA injury report + press headlines (free, no key).

Two sub-fetchers:
  fetch_injuries() — scrapes the injury table; returns kind='injury' records
                     filtered to our two target teams.
  fetch_news()     — parses the CBS Sports NBA RSS feed; returns kind='article'
                     records that mention either team.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

from .. import config
from ..http_util import fetch, fetch_json
from .base import SourceResult, STATUS_ERROR, STATUS_OK, STATUS_PARTIAL

INJURIES_URL = "https://www.cbssports.com/nba/injuries/"
NEWS_RSS_URL = "https://www.cbssports.com/rss/headlines/nba/"

CBS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cbssports.com/",
}
RSS_HEADERS = {
    "User-Agent": "MoodMirror/1.0 RSS reader",
    "Accept": "application/rss+xml,application/xml,text/xml,*/*",
}

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_TABLE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", _TAG.sub("", s))).strip()


def fetch_injuries() -> SourceResult:
    try:
        res = fetch(INJURIES_URL, headers=CBS_HEADERS)
        if not res.ok or not res.body:
            return SourceResult("cbssports_injuries", STATUS_ERROR, error=res.error)
        body = res.body

        home_aliases = config.GAME["home"]["aliases"]
        away_aliases = config.GAME["away"]["aliases"]
        home_name = config.GAME["home"]["name"].lower()
        away_name = config.GAME["away"]["name"].lower()

        records: list[dict] = []
        tables = _TABLE.findall(body)
        search_html = "\n".join(tables) if tables else body

        for row_html in _TR.findall(search_html):
            cells = [_strip(c) for c in _TD.findall(row_html)]
            if len(cells) < 4:
                continue
            player = cells[0]
            if not player or player.lower() in ("player", "name", ""):
                continue
            position = cells[1] if len(cells) > 1 else ""
            updated = cells[2] if len(cells) > 2 else ""
            injury_type = cells[3] if len(cells) > 3 else ""
            status = cells[4] if len(cells) > 4 else ""

            # CBS groups injuries by team heading; look for team name in
            # surrounding context (within ±500 chars of this row in body).
            row_pos = body.find(cells[0]) if cells[0] else -1
            context = body[max(0, row_pos - 500): row_pos + 200].lower()
            is_home = any(al in context for al in home_aliases) or home_name in context
            is_away = any(al in context for al in away_aliases) or away_name in context
            if not (is_home or is_away):
                continue

            records.append({
                "id": f"cbs_injury:{player}",
                "source": "cbssports_injuries",
                "kind": "injury",
                "title": f"{player}: {status or injury_type}",
                "text": (
                    f"{player} ({position}) — {injury_type} — "
                    f"status: {status} — updated: {updated}"
                ),
                "url": INJURIES_URL,
                "published": None,
                "engagement": 0,
                "team_hint": "home" if is_home else "away",
                "player": player,
                "status_text": status,
                "injury_desc": injury_type,
            })

        return SourceResult("cbssports_injuries",
                             STATUS_OK if records else STATUS_PARTIAL,
                             records=records)
    except Exception as exc:
        return SourceResult("cbssports_injuries", STATUS_ERROR, error=str(exc))


def fetch_news() -> SourceResult:
    """Parse CBS Sports NBA RSS feed; filter articles mentioning our teams."""
    try:
        res = fetch(NEWS_RSS_URL, headers=RSS_HEADERS)
        if not res.ok or not res.body:
            return SourceResult("cbssports_news", STATUS_PARTIAL, error=res.error)

        home_aliases = config.GAME["home"]["aliases"]
        away_aliases = config.GAME["away"]["aliases"]
        records: list[dict] = []

        try:
            root = ET.fromstring(res.body)
        except ET.ParseError:
            return SourceResult("cbssports_news", STATUS_PARTIAL,
                                 error="RSS XML parse error")

        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pubdate = (item.findtext("pubDate") or "").strip()
            desc = (item.findtext("description") or "").strip()
            full_text = f"{title} {desc}".lower()

            is_home = any(al in full_text for al in home_aliases)
            is_away = any(al in full_text for al in away_aliases)
            if not (is_home or is_away):
                continue

            records.append({
                "id": f"cbs_news:{link}",
                "source": "cbssports_news",
                "kind": "article",
                "title": title,
                "text": desc,
                "url": link,
                "author": item.findtext("dc:creator", namespaces=ns),
                "published": pubdate,
                "engagement": 0,
                "team_hint": (
                    "both" if (is_home and is_away)
                    else "home" if is_home
                    else "away"
                ),
            })

        return SourceResult("cbssports_news",
                             STATUS_OK if records else STATUS_PARTIAL,
                             records=records)
    except Exception as exc:
        return SourceResult("cbssports_news", STATUS_ERROR, error=str(exc))
