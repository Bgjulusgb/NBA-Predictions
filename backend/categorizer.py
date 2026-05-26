"""Rule-based classifier: route every record into a topic category.

The dashboard groups records (articles, social posts, scraped events) by topic
so the user can drill into "show me only injuries" or "only betting talk".
This is pure-stdlib pattern matching — fast, deterministic and good enough
for ~95% of NBA chatter; anything ambiguous lands in "general".

Categories:

    injury        - injury / status / IR / surgery / rehab
    trade         - trade / sign / waive / contract
    lineup        - starter / inactive / DNP / rotation / minutes restriction
    coaching      - coach / fired / hired / scheme / play call
    betting       - odds / line / parlay / prop / sharps / public / EV
    preview       - vs / preview / matchup / tonight / tipoff / how to watch
    recap         - final / beat / defeat / win / loss + past-tense verbs
    drama         - feud / beef / called out / criticism / quote
    history       - record / stat-line / first since / since 19xx
    award         - mvp / dpoy / roty / sixth man / all-NBA / hof
    business      - sponsor / arena / sale / valuation
    general       - everything else
"""

from __future__ import annotations

import re

CATEGORIES = [
    "injury", "trade", "lineup", "coaching", "betting", "preview",
    "recap", "drama", "history", "award", "business", "general",
]

# Compiled keyword lists; each tuple is (category, regex).
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("injury", re.compile(
        r"\b(injur(?:y|ed|ies)|out|questionable|doubtful|probable|"
        r"sprain(?:ed)?|strain(?:ed)?|torn|tore|surgery|rehab|"
        r"ir\b|injury report|game-time decision|sidelined|hamstring|"
        r"ankle|knee|achilles|calf|wrist|shoulder|hip|back|concussion)\b",
        re.IGNORECASE)),
    ("trade", re.compile(
        r"\b(trade(?:d|s)?|deal|swap(?:ped)?|sign(?:ed|ing)?|waive(?:d)?|"
        r"buyout|extension|extension|contract|two-?way|exhibit 10|"
        r"front office|tampering)\b", re.IGNORECASE)),
    ("lineup", re.compile(
        r"\b(starting lineup|starter|starters|bench|inactive(?:s)?|"
        r"rotation|minutes restriction|dnp(?:-cd)?|did not play|"
        r"load management|second unit)\b", re.IGNORECASE)),
    ("coaching", re.compile(
        r"\b(coach(?:ing)?|head coach|fired|hired|assistant coach|"
        r"play call|timeout|in-game adjustment|scheme|"
        r"defensive scheme|offensive scheme)\b", re.IGNORECASE)),
    ("betting", re.compile(
        r"\b(odds|moneyline|spread|over\W?under|o/u|total|"
        r"parlay|prop(?:s)?|teaser|hedge|sharps?|sharp money|"
        r"public bet(?:s|ting)?|line move|consensus|kelly|"
        r"value bet|ev(?: bet)?|book|sportsbook|vig|juice|hold)\b",
        re.IGNORECASE)),
    ("preview", re.compile(
        r"\b(preview|matchup|vs\.?|how to watch|where to watch|"
        r"tip-?off|tonight'?s game|game thread|pre-game|"
        r"projected starters|key matchup)\b", re.IGNORECASE)),
    ("recap", re.compile(
        r"\b(final|beat|beats|defeat(?:ed|s)?|hold off|hung on|"
        r"recap|takeaways|highlights|stat line|notch(?:ed)?|"
        r"win(?:s)?|won|loss(?:es)?|fell to|comeback win)\b",
        re.IGNORECASE)),
    ("drama", re.compile(
        r"\b(feud|beef|called out|criticism|criticize(?:d)?|"
        r"clap back|slam(?:med)?|response|disrespect|war of words|"
        r"verbal|exchange)\b", re.IGNORECASE)),
    ("history", re.compile(
        r"\b(first since|record(?:s)? for|all-time|franchise record|"
        r"career high|career low|since (?:19|20)\d{2}|historic(?:al)?|"
        r"first in|joins (?:[a-z]+) only|history|milestone)\b",
        re.IGNORECASE)),
    ("award", re.compile(
        r"\b(mvp|dpoy|roty|sixth man|coach of the year|all-nba|"
        r"all-defensive|hall of fame|hof|finals mvp)\b", re.IGNORECASE)),
    ("business", re.compile(
        r"\b(sponsor|jersey patch|arena (?:naming|deal)|valuation|"
        r"team sale|ownership|cba|collective bargaining|nbpa|"
        r"luxury tax|salary cap)\b", re.IGNORECASE)),
]


def categorize_text(text: str) -> str:
    """Best-fit category for one text. Falls through to 'general'."""
    if not text:
        return "general"
    for cat, pat in _RULES:
        if pat.search(text):
            return cat
    return "general"


def categorize_records(records: list[dict]) -> list[dict]:
    """Mutate in place: assign rec['category']; return the same list."""
    for rec in records:
        text = f"{rec.get('title','')} {rec.get('text','')}"
        rec["category"] = categorize_text(text)
    return records


def category_breakdown(records: list[dict]) -> dict:
    """Counts + mean sentiment per category for the dashboard."""
    out: dict[str, dict] = {c: {"count": 0, "comps": []} for c in CATEGORIES}
    for r in records:
        c = r.get("category") or categorize_text(
            f"{r.get('title','')} {r.get('text','')}")
        bucket = out.setdefault(c, {"count": 0, "comps": []})
        bucket["count"] += 1
        if "sentiment" in r:
            bucket["comps"].append(r["sentiment"].get("compound", 0.0))
    summary = []
    for cat in CATEGORIES:
        b = out.get(cat, {"count": 0, "comps": []})
        if b["count"] == 0:
            continue
        mean = sum(b["comps"]) / len(b["comps"]) if b["comps"] else 0.0
        summary.append({
            "category": cat,
            "count": b["count"],
            "mean_sentiment": round(mean, 4),
        })
    summary.sort(key=lambda x: x["count"], reverse=True)
    return {"categories": summary, "total": sum(b["count"] for b in out.values())}


def filter_by_category(records: list[dict], category: str) -> list[dict]:
    """Return records whose category equals (or whose text matches) `category`."""
    if category == "all":
        return records
    out = []
    for r in records:
        c = r.get("category") or categorize_text(
            f"{r.get('title','')} {r.get('text','')}")
        if c == category:
            out.append(r)
    return out
