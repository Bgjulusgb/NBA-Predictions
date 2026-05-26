"""Higher-level data processing over enriched records.

Turns the per-record sentiment into the analytical views the dashboard shows:

  * player_sentiment  - per-player buzz + sentiment (roster-aware matching)
  * narratives        - trending fan/analyst narrative terms + their sentiment
  * emotion_profile   - aggregate joy/anger/fear/sadness/anticipation mix
  * value_bet         - model edge vs. the market + expected value

Pure Python; operates on the records produced by enrich.py.
"""

import re

from . import config, model, sentiment

# Notable narrative terms worth tracking (drawn from the NBA lexicon).
NARRATIVE_TERMS = [
    "clutch", "mvp", "rigged", "refball", "robbed", "choke", "washed",
    "injury", "injured", "dominant", "collapse", "sweep", "comeback",
    "dagger", "bum", "fraud", "elite", "soft", "exposed", "blowout",
]


def _alias_pattern(alias):
    return re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)


def player_sentiment(records):
    """Per-player mention count, mean sentiment and engagement-weighted buzz.

    Multi-word aliases (full names) are matched first and masked so a shared
    surname (e.g. "Mitchell" Robinson) is not double-counted.
    """
    players = []
    for side in ("home", "away"):
        for name, aliases in config.ROSTERS.get(side, {}).items():
            players.append({"name": name, "team": side,
                            "aliases": sorted(aliases, key=len, reverse=True),
                            "mentions": 0, "_comps": [], "buzz": 0.0})

    multi = [(p, a) for p in players for a in p["aliases"] if " " in a]
    single = [(p, a) for p in players for a in p["aliases"] if " " not in a]

    for rec in records:
        text = f"{rec.get('title','')} {rec.get('text','')}"
        if not text.strip():
            continue
        comp = rec.get("sentiment", {}).get("compound", 0.0)
        weight = 1.0 + max(0, rec.get("engagement", 0)) / 100.0
        hit_players = set()

        masked = text
        for p, alias in multi:
            pat = _alias_pattern(alias)
            if pat.search(masked):
                hit_players.add(id(p))
                masked = pat.sub(" ", masked)
        for p, alias in single:
            if _alias_pattern(alias).search(masked):
                hit_players.add(id(p))

        for p in players:
            if id(p) in hit_players:
                p["mentions"] += 1
                p["_comps"].append(comp)
                p["buzz"] += weight

    out = []
    for p in players:
        if p["mentions"] == 0:
            continue
        out.append({
            "name": p["name"],
            "team": p["team"],
            "mentions": p["mentions"],
            "mean_sentiment": round(sum(p["_comps"]) / len(p["_comps"]), 4),
            "buzz": round(p["buzz"], 2),
        })
    out.sort(key=lambda x: x["buzz"], reverse=True)
    return out


def narratives(records, top=10):
    """Trending narrative terms with frequency and the sentiment around them."""
    stats = {t: {"count": 0, "comps": []} for t in NARRATIVE_TERMS}
    patterns = {t: _alias_pattern(t) for t in NARRATIVE_TERMS}
    for rec in records:
        text = f"{rec.get('title','')} {rec.get('text','')}"
        comp = rec.get("sentiment", {}).get("compound", 0.0)
        for term, pat in patterns.items():
            if pat.search(text):
                stats[term]["count"] += 1
                stats[term]["comps"].append(comp)
    out = []
    for term, s in stats.items():
        if s["count"] == 0:
            continue
        out.append({
            "term": term,
            "count": s["count"],
            "mean_sentiment": round(sum(s["comps"]) / len(s["comps"]), 4),
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:top]


def emotion_profile(records):
    """Mean emotion distribution across records (joy/anger/fear/...)."""
    keys = ["joy", "anger", "fear", "sadness", "anticipation"]
    totals = {k: 0.0 for k in keys}
    n = 0
    for rec in records:
        text = f"{rec.get('title','')} {rec.get('text','')}"
        emo = sentiment.emotions(text)
        if any(emo.values()):
            for k in keys:
                totals[k] += emo[k]
            n += 1
    if n == 0:
        return {k: 0.0 for k in keys}
    return {k: round(v / n, 4) for k, v in totals.items()}


def value_bet(market, ensemble, odds):
    """Find the side where the model sees value vs. the market, with EV.

    Returns None if odds/market are unavailable.
    """
    if not market or not ensemble:
        return None
    edges = {
        "home": ensemble["home"] - market["home"],
        "away": ensemble["away"] - market["away"],
    }
    side = max(edges, key=edges.get)
    edge = edges[side]
    ml = (odds or {}).get(f"{side}_moneyline")
    dec = model.american_to_decimal(ml) if ml is not None else None
    p = ensemble[side]
    ev = round(p * (dec - 1) - (1 - p), 4) if dec else None
    return {
        "side": side,
        "edge_pct": round(edge * 100, 2),
        "model_prob": round(p, 4),
        "market_prob": round(market[side], 4),
        "moneyline": ml,
        "expected_value": ev,
        "has_value": edge > 0.01,
    }


# ---------------------------------------------------------------------------
# Narrative dispersion: how spread out is the conversation? A single dominant
# storyline differs from a fragmented one — useful as a "noise" indicator.
# ---------------------------------------------------------------------------
def narrative_concentration(narratives_list):
    """Herfindahl index over narrative counts: 1 = single story, 0 = uniform."""
    if not narratives_list:
        return 0.0
    total = sum(n["count"] for n in narratives_list)
    if total == 0:
        return 0.0
    return round(sum((n["count"] / total) ** 2 for n in narratives_list), 4)


def sentiment_polarity(records):
    """Share of records that are clearly positive / negative / neutral."""
    if not records:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    pos = sum(1 for r in records
              if r.get("sentiment", {}).get("compound", 0) >= 0.35)
    neg = sum(1 for r in records
              if r.get("sentiment", {}).get("compound", 0) <= -0.35)
    n = len(records)
    return {
        "positive": round(pos / n, 4),
        "negative": round(neg / n, 4),
        "neutral": round(1 - (pos + neg) / n, 4),
    }


def top_outlets(records, top=8):
    """Press outlet leaderboard by article count + mean sentiment."""
    by_outlet = {}
    for r in records:
        src = r.get("source", "unknown")
        d = by_outlet.setdefault(src, {"count": 0, "comps": []})
        d["count"] += 1
        d["comps"].append(r.get("sentiment", {}).get("compound", 0))
    out = []
    for src, d in by_outlet.items():
        out.append({
            "source": src,
            "count": d["count"],
            "mean_sentiment": round(sum(d["comps"]) / len(d["comps"]), 4),
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:top]
