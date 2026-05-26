"""Google News + ESPN RSS: the "press review" (Presse Spiegel) layer.

Free, keyless, reliable. Returns article records (title + source + timestamp).
Sentiment is added later in the enrichment step.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from .. import config
from ..http_util import fetch, run_parallel
from .base import SourceResult, STATUS_OK, STATUS_PARTIAL, STATUS_ERROR

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text):
    if not text:
        return ""
    return _TAG_RE.sub("", text).replace("&nbsp;", " ").strip()


def _hash_id(prefix, *parts):
    h = hashlib.sha1("|".join(p or "" for p in parts).encode("utf-8"))
    return f"{prefix}:{h.hexdigest()[:12]}"


def _norm_title(title):
    """Normalise a headline for cross-outlet duplicate detection."""
    return re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip()


def _parse_rss(xml_text, source_name):
    """Parse an RSS document into article records."""
    records = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = _clean(_text(item, "title"))
        link = _text(item, "link")
        pub = _text(item, "pubDate")
        desc = _clean(_text(item, "description"))
        src_el = item.find("source")
        outlet = (src_el.text.strip() if src_el is not None and src_el.text
                  else _outlet_from_title(title))
        records.append({
            "id": _hash_id("news", link, title),
            "source": source_name,
            "kind": "article",
            "title": _strip_outlet(title, outlet),
            "text": desc,
            "url": link,
            "author": None,
            "published": _to_iso(pub),
            "engagement": 0,
            "outlet": outlet,
            "team_hint": None,
        })
    return records


def _text(item, tag):
    el = item.find(tag)
    return el.text if el is not None and el.text else ""


def _to_iso(rfc822):
    """Convert an RSS RFC-822 date to ISO 8601; return original on failure."""
    if not rfc822:
        return None
    try:
        return parsedate_to_datetime(rfc822).isoformat()
    except (TypeError, ValueError, IndexError):
        return rfc822


def _outlet_from_title(title):
    # Google News titles end with " - Outlet".
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "Unknown"


def _strip_outlet(title, outlet):
    if outlet and title.endswith(f" - {outlet}"):
        return title[: -(len(outlet) + 3)].strip()
    return title


def fetch_press_review(queries=None):
    """Aggregate Google News searches + extra NBA RSS feeds, de-duplicated.

    All feeds are fetched concurrently; a broken feed is skipped, not fatal.
    """
    queries = queries or config.GOOGLE_NEWS_QUERIES
    feeds = [(config.GOOGLE_NEWS_RSS.format(query=quote(q)), "google_news")
             for q in queries]
    feeds += list(config.EXTRA_PRESS_FEEDS)

    tasks = {f"{name}#{i}": (lambda u=url: fetch(u))
             for i, (url, name) in enumerate(feeds)}
    fetched = run_parallel(tasks)

    all_records = {}
    seen_titles = set()
    errors = []
    ok_any = False
    for i, (url, name) in enumerate(feeds):
        res = fetched.get(f"{name}#{i}")
        if isinstance(res, Exception) or res is None or not getattr(res, "ok", False):
            errors.append(f"{name}: {getattr(res, 'error', res)}")
            continue
        try:
            for rec in _parse_rss(res.body, name):
                if rec["id"] in all_records:
                    continue
                # Also drop the same story syndicated across outlets.
                norm = _norm_title(rec["title"])
                if norm and norm in seen_titles:
                    continue
                seen_titles.add(norm)
                all_records[rec["id"]] = rec
            ok_any = True
        except ET.ParseError as e:
            errors.append(f"{name}: parse error {e}")

    if not ok_any:
        return SourceResult("press_review", STATUS_ERROR,
                            error="; ".join(errors) or "no feeds returned")
    status = STATUS_OK if not errors else STATUS_PARTIAL
    return SourceResult("press_review", status,
                        records=list(all_records.values()),
                        error="; ".join(errors) or None)
