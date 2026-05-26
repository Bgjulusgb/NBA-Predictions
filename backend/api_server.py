"""Tiny stdlib HTTP API server.

Serves the static dashboard (index.html + data/*.json) AND a JSON REST API at
/api/* so frontend tabs can query specific slices without re-downloading the
whole snapshot. Pure http.server / urllib — no Flask/FastAPI/uvicorn needed.

Endpoints:

    GET /                                     -> index.html
    GET /index.html                           -> index.html
    GET /data/snapshot.json                   -> raw snapshot
    GET /data/live.json                       -> raw live (if present)
    GET /api/health                           -> {ok, version, sources}
    GET /api/snapshot                         -> full snapshot
    GET /api/snapshot/<section>               -> a single top-level section
    GET /api/live                             -> live snapshot
    GET /api/categories                       -> grouped record categories
    GET /api/category/<name>?limit=N          -> records in one category
    GET /api/players?team=home&sort=buzz      -> players (filterable)
    GET /api/odds                             -> aggregated odds
    GET /api/lineups                          -> aggregated lineups
    GET /api/sources                          -> source health grid
    GET /api/advanced                         -> snapshot.advanced
    GET /api/simulate?p=0.6&trials=2000       -> on-demand Monte Carlo
    GET /api/momentum                         -> live composite momentum
    GET /api/refresh                          -> rebuild snapshot now (async)

Run:
    python3 -m backend.run api
    python3 -m backend.run api --port 8080 --bind 0.0.0.0
"""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import (advanced_math, aggregator, categorizer, config,
               lineup_analyzer, momentum_composite, simulation)

# In-process cache so /api/snapshot doesn't re-read the JSON file every hit.
_CACHE: dict[str, Any] = {"snapshot": None, "snapshot_mtime": 0,
                          "live": None, "live_mtime": 0}
_CACHE_LOCK = threading.Lock()


def _load_json(path: str, key: str) -> dict | None:
    """Cache-aware loader. Re-reads when file mtime advances."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    with _CACHE_LOCK:
        if _CACHE.get(f"{key}_mtime") == mtime and _CACHE.get(key) is not None:
            return _CACHE[key]
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        _CACHE[key] = data
        _CACHE[f"{key}_mtime"] = mtime
        return data


def _snapshot():
    return _load_json(config.SNAPSHOT_PATH, "snapshot")


def _live():
    return _load_json(config.LIVE_PATH, "live")


def _records_from_snapshot(snap: dict) -> list[dict]:
    """All categorized records in one flat list — articles + social + live."""
    if not snap:
        return []
    items = list(snap.get("press_review") or []) + list(snap.get("social") or [])
    categorizer.categorize_records(items)
    return items


# ===========================================================================
# Router
# ===========================================================================
class _Handler(BaseHTTPRequestHandler):

    server_version = "FootballPredicter/1.0"

    # Quiet the default access log; uncomment to debug.
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:                # noqa: N802
        try:
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            if path.startswith("/api/"):
                self._serve_api(path[len("/api/"):], query)
                return
            self._serve_static(path)
        except Exception as e:                       # noqa: BLE001
            self._reply_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    # -------------------------------------------------------------------
    # Static + raw JSON files
    # -------------------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        if path.startswith("/data/"):
            full = os.path.join(_root(), path.lstrip("/"))
        else:
            full = os.path.join(_root(), path.lstrip("/"))
        # Path-traversal guard.
        if not os.path.abspath(full).startswith(_root()):
            self._reply_text("Forbidden", status=403)
            return
        if not os.path.exists(full) or os.path.isdir(full):
            self._reply_text("Not found", status=404)
            return
        ctype = _content_type(full)
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # -------------------------------------------------------------------
    # API router
    # -------------------------------------------------------------------
    def _serve_api(self, subpath: str, query: dict[str, list[str]]) -> None:
        subpath = subpath.rstrip("/")
        snap = _snapshot()
        live = _live()

        if subpath == "health":
            health = aggregator.source_health((snap or {}).get("sources") or [])
            return self._reply_json({
                "ok": True, "version": "1.0",
                "snapshot_loaded": bool(snap),
                "live_loaded": bool(live),
                "source_health": health,
            })

        if subpath == "snapshot":
            return self._reply_json(snap or {"error": "no snapshot"})

        if subpath.startswith("snapshot/"):
            section = subpath.split("/", 1)[1]
            if not snap:
                return self._reply_json({"error": "no snapshot"}, status=404)
            if section not in snap:
                return self._reply_json({"error": f"section '{section}' not in snapshot",
                                          "available": list(snap.keys())}, status=404)
            return self._reply_json(snap[section])

        if subpath == "live":
            return self._reply_json(live or {"mode": "unavailable"})

        if subpath == "categories":
            records = _records_from_snapshot(snap or {})
            return self._reply_json(categorizer.category_breakdown(records))

        if subpath.startswith("category/"):
            cat = subpath.split("/", 1)[1] or "general"
            limit = int((query.get("limit") or [50])[0])
            records = _records_from_snapshot(snap or {})
            filtered = categorizer.filter_by_category(records, cat)
            return self._reply_json({"category": cat,
                                      "count": len(filtered),
                                      "items": filtered[:limit]})

        if subpath == "players":
            if not snap:
                return self._reply_json({"error": "no snapshot"}, status=404)
            players = list(snap.get("players") or [])
            team = (query.get("team") or [None])[0]
            sort = (query.get("sort") or ["buzz"])[0]
            if team in ("home", "away"):
                players = [p for p in players if p.get("team") == team]
            players.sort(key=lambda p: p.get(sort, 0), reverse=True)
            return self._reply_json({"count": len(players), "items": players})

        if subpath == "odds":
            return self._reply_json((snap or {}).get("odds_unified")
                                    or _legacy_odds(snap))

        if subpath == "lineups":
            return self._reply_json((snap or {}).get("lineups_unified")
                                    or {"home": None, "away": None})

        if subpath == "sources":
            if not snap:
                return self._reply_json({"error": "no snapshot"}, status=404)
            return self._reply_json(aggregator.source_health(snap.get("sources") or []))

        if subpath == "advanced":
            return self._reply_json((snap or {}).get("advanced") or {})

        if subpath == "narratives":
            return self._reply_json({
                "list": (snap or {}).get("narratives") or [],
                "meta": (snap or {}).get("narrative_meta") or {},
            })

        if subpath == "simulate":
            p = float((query.get("p") or ["0.5"])[0])
            trials = int((query.get("trials") or ["3000"])[0])
            sim = simulation.simulate_game(p, trials=trials)
            sim["alt_lines"] = simulation.simulate_alt_lines(p, trials=trials // 2)
            return self._reply_json(sim)

        if subpath == "momentum":
            if not snap or not live:
                return self._reply_json({"value": 0.0,
                                          "error": "need both snapshot+live"})
            home_abbr = config.GAME["home"]["abbr"]
            L = (live.get("live") or {})
            comp = momentum_composite.composite(
                scoring_momentum=L.get("momentum"),
                current_run=L.get("current_run"),
                home_abbr=home_abbr,
                sentiment_zscore=L.get("sentiment_spike"),
                starting_plus_minus_diff=None,
                pace_ratio=None,
            )
            return self._reply_json(comp)

        if subpath == "refresh":
            self._kick_snapshot()
            return self._reply_json({"started": True})

        return self._reply_json({"error": f"unknown api path: {subpath}",
                                  "available": [
                                      "/api/health", "/api/snapshot",
                                      "/api/snapshot/<section>", "/api/live",
                                      "/api/categories", "/api/category/<name>",
                                      "/api/players", "/api/odds",
                                      "/api/lineups", "/api/sources",
                                      "/api/advanced", "/api/narratives",
                                      "/api/simulate", "/api/momentum",
                                      "/api/refresh"]}, status=404)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _kick_snapshot(self) -> None:
        def _bg():
            from . import pipeline
            try:
                pipeline.write_snapshot()
            except Exception:                       # noqa: BLE001
                pass
        threading.Thread(target=_bg, daemon=True).start()

    def _reply_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _reply_text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _jsonable(o):
    """Fallback JSON encoder for any odd object the snapshot may carry."""
    try:
        return list(o)
    except TypeError:
        return str(o)


def _legacy_odds(snap):
    if not snap:
        return None
    return {"book": (snap.get("game") or {}).get("odds")}


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _content_type(path: str) -> str:
    if path.endswith(".html"):
        return "text/html; charset=utf-8"
    if path.endswith(".json"):
        return "application/json; charset=utf-8"
    if path.endswith(".js"):
        return "text/javascript; charset=utf-8"
    if path.endswith(".css"):
        return "text/css; charset=utf-8"
    if path.endswith(".svg"):
        return "image/svg+xml"
    return "application/octet-stream"


def serve(port: int = 8000, bind: str = "127.0.0.1") -> None:
    httpd = ThreadingHTTPServer((bind, port), _Handler)
    print(f"NBA Mood Mirror API + UI: http://{bind}:{port}/")
    print(f"  Web dashboard: http://{bind}:{port}/")
    print(f"  Health check:  http://{bind}:{port}/api/health")
    print(f"  Snapshot:      http://{bind}:{port}/api/snapshot")
    print(f"  Categories:    http://{bind}:{port}/api/categories")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        httpd.server_close()
