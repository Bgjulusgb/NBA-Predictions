"""Append-only history of snapshots, so trends are real over time.

Each run appends one compact line to data/history.jsonl. The dashboard reads
the recent tail to chart how win probability, odds and mood move across runs
(today's per-article timeline only shows coverage age, not movement).
"""

import datetime as dt
import json
import os

from . import config

HISTORY_PATH = os.path.join(config.DATA_DIR, "history.jsonl")
MAX_LINES = 2000


def compact(snapshot):
    """Reduce a full snapshot to the few metrics worth tracking over time."""
    pred = snapshot.get("prediction", {})
    ens = pred.get("ensemble", {})
    market = pred.get("market") or {}
    elo = pred.get("elo", {})
    mood = snapshot.get("mood", {}).get("overall", {})
    ts = snapshot.get("mood", {}).get("team_sentiment", {})
    series = pred.get("series", {})
    return {
        "ts": snapshot.get("generated_at"),
        "mode": snapshot.get("mode"),
        "ens_home": ens.get("home"),
        "ens_away": ens.get("away"),
        "market_home": market.get("home"),
        "elo_home": elo.get("home"),
        "confidence": pred.get("confidence"),
        "heat": mood.get("heat"),
        "hype": mood.get("hype"),
        "toxicity": mood.get("toxicity"),
        "sent_home": ts.get("home"),
        "sent_away": ts.get("away"),
        "clinch": series.get("leader_clinch_probability"),
    }


def append(snapshot, path=HISTORY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(compact(snapshot), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    _trim(path)


def _trim(path, max_lines=MAX_LINES):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    if len(lines) > max_lines:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_lines:])


def load_recent(n=60, path=HISTORY_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
