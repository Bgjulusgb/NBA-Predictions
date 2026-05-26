"""Basketball Reference: historical record scrape to refine Elo seeds.

Uses a cookie jar + full Chrome fingerprint to work around Cloudflare.
Probes the BRef homepage first to pick up session cookies, then scrapes the
team page with exponential-backoff retry. Falls back gracefully so the model
can still run on its built-in Elo seeds.
"""
from __future__ import annotations

import gzip
import http.cookiejar
import io
import re
import time
import urllib.request

from .. import config
from .base import SourceResult, STATUS_OK, STATUS_PARTIAL, STATUS_ERROR

_BREF_ABBR = {"CLE": "CLE", "NYK": "NYK", "NY": "NYK"}

_RECORD_RE = re.compile(r"Record:\s*</strong>\s*([0-9]+)-([0-9]+)", re.I)
_SRS_RE = re.compile(r"SRS</a>:\s*</strong>\s*(-?[0-9.]+)", re.I)

BREF_HOME = "https://www.basketball-reference.com/"
BREF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://www.basketball-reference.com/",
}

_RETRIES = 3
_BACKOFF = 2.0


def _build_opener():
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPRedirectHandler(),
    )


def _decode_resp(resp):
    raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    ct = resp.headers.get("Content-Type", "")
    charset = "utf-8"
    if "charset=" in ct:
        charset = ct.split("charset=")[-1].split(";")[0].strip()
    return raw.decode(charset, errors="replace")


def _scrape_team(abbr):
    bref = _BREF_ABBR.get(abbr.upper(), abbr.upper())
    url = config.BREF_TEAM_PAGE.format(abbr=bref)
    opener = _build_opener()

    # Probe homepage to collect cookies before hitting the team page.
    try:
        probe = urllib.request.Request(BREF_HOME, headers=BREF_HEADERS)
        opener.open(probe, timeout=config.HTTP_TIMEOUT)
        time.sleep(0.5)
    except Exception:
        pass

    body = None
    last_err = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers=BREF_HEADERS)
            with opener.open(req, timeout=config.HTTP_TIMEOUT) as resp:
                body = _decode_resp(resp)
            break
        except Exception as e:
            last_err = str(e)
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF * (2 ** attempt))

    if body is None:
        return None, last_err or "unknown error"

    wins = losses = srs = None
    m = _RECORD_RE.search(body)
    if m:
        wins, losses = int(m.group(1)), int(m.group(2))
    m = _SRS_RE.search(body)
    if m:
        srs = float(m.group(1))
    if wins is None and srs is None:
        return None, "record/SRS not found in page"
    return {"abbr": abbr, "wins": wins, "losses": losses, "srs": srs}, None


def fetch_history(home=None, away=None):
    home = home or config.GAME["home"]
    away = away or config.GAME["away"]
    out = {}
    errors = []
    for side, team in (("home", home), ("away", away)):
        try:
            data, err = _scrape_team(team["abbr"])
        except Exception as exc:
            data, err = None, str(exc)
        if data:
            out[side] = data
        else:
            errors.append(f"{team['abbr']}: {err}")

    if not out:
        return SourceResult("basketball_reference", STATUS_ERROR,
                            error="; ".join(errors))
    status = STATUS_OK if len(out) == 2 else STATUS_PARTIAL
    return SourceResult("basketball_reference", status,
                        meta={"teams": out}, error="; ".join(errors) or None)
