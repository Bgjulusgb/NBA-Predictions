# 🏀 NBA Mood Mirror & Press Review

> Real-time **sentiment analysis**, **press review** ("Presse Spiegel") and a
> multi-model **win-probability prediction** for an NBA game — built entirely on
> **free, keyless** data sources. No API keys, no paid services, no build step.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/backend-stdlib%20only-informational)
![API keys](https://img.shields.io/badge/API%20keys-none-success)
![Cost](https://img.shields.io/badge/cost-%240-success)
![Tests](https://img.shields.io/badge/tests-130%20passing-brightgreen)
![Scrapers](https://img.shields.io/badge/scrapers-14%20sources-blueviolet)
![Language](https://img.shields.io/badge/language-English-lightgrey)

> **v2 highlights:** Sofascore + Flashscore + TheScore + NBA-stats + Rotowire
> + TeamRankings + OddsPortal + ActionNetwork scrapers · 6 Math modules
> (Monte Carlo, Glicko-2, Bayesian, Kelly, Four Factors, …) · ANSI CLI +
> curses TUI + React web dashboard · stdlib HTTP **REST API** at `/api/*` ·
> rule-based **category** classifier · cross-source **score consensus** +
> **multi-book odds** + **lineup analysis** + **composite momentum**.

It blends **analyst & press coverage**, **fan social sentiment**, **betting
odds** and **historical team strength** into one live mood picture and a
prediction, then switches into a **live in-game mode** that detects scoring
runs, momentum swings and fan-sentiment spikes ("AI detects fan panic after a
12-0 run").

Centred on **New York Knicks @ Cleveland Cavaliers — Eastern Conference Finals,
Game 4** (Mon May 25, 2026, 8 PM ET, Rocket Arena; Knicks lead 3-0). The target
game is fully configurable in [`backend/config.py`](backend/config.py).

---

## Table of contents

- [Features](#features)
- [How it looks](#how-it-looks)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Data sources](#data-sources-all-free--keyless)
- [Analytics & mathematics](#analytics--mathematics)
- [The "ok imports directly" rule](#the-ok-imports-directly-rule)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Keywords](#keywords)

---

## Features

**Data collection**
- Per-website tailored scrapers (one module each), fetched **concurrently** so a
  full refresh takes a couple of seconds and one broken source never aborts the run.
- 7 Google-News search angles + Yahoo + CBS + ESPN RSS for a broad press review.
- Live scores, game status **and odds** from one free ESPN endpoint.
- Best-effort Reddit and NBA.com live play-by-play with graceful fallback.

**Analysis**
- VADER-style **sentiment engine** with a custom NBA + toxicity lexicon
  (`clutch`, `washed`, `rigged`, `choke`, `MVP`, `refball`, …), plus
  **multi-word phrases**, **emoji** and **"but"-contrast** handling.
- **Emotion breakdown** (joy / anger / fear / sadness / anticipation).
- **Player-level sentiment & buzz** (roster-aware, avoids shared-surname collisions).
- **Narrative tracker** — trending storylines and the sentiment around each.
- **Heat / Hype / Toxicity** meters.

**Prediction (four independent models + ensemble)**
- Odds → implied probability with **de-vig**, spread→prob fallback,
  **Elo / log5** (seeded from SRS and **adjusted by recent form**),
  the **ESPN Matchup Predictor**, and a bounded **sentiment adjustment**.
- **Value-bet** edge + expected value, cross-model **confidence**,
  **series-clinch** probability, and a **backtest/calibration** harness
  (Brier, log-loss, accuracy, skill score).

**Live mode**
- **Live win-probability** model (margin + time remaining, anchored to the
  pre-game prior), scoring-**run detection**, exponential-decay **momentum**,
  rolling-z-score **sentiment spikes**, and auto-generated alert headlines.

**History & UI**
- Every run is appended to a rolling history → real **movement charts** over time.
- Single-file React dashboard (Mood Mirror · Press Review · Prediction · Live)
  with auto-refresh, dark mode, search and filters.

---

## How it looks

Four tabs, all reading the JSON the backend writes:

| Tab | Shows |
|-----|-------|
| **Mood Mirror** | Heat/Hype/Toxicity meters, team & player sentiment, emotion breakdown, trending narratives, sentiment timeline |
| **Press Review** | Searchable, sentiment-scored article feed (outlet + timestamp + team) |
| **Prediction** | De-vigged market, Elo, ensemble win %, value bet, series-clinch, movement-across-runs chart |
| **Live** | Live score, current run, momentum, sentiment-spike, alerts, score progression |

> Tip: open over **HTTP** (`http://localhost:8000/`), not `file://` — browsers
> block the JSON `fetch` on the `file://` protocol.

---

## Architecture

```
 free web sources ─►  Python backend (stdlib only)  ─►  data/*.json  ─►  index.html (React + Recharts)
 ESPN · GoogleNews     fetch (parallel) → enrich →        snapshot.json     reads JSON, auto-refresh,
 Yahoo · CBS · Reddit   analyse → math → persist          live.json         no build step
 NBA.com · Bball-Ref                                       history.jsonl
```

- **Backend — Python 3, standard library only.** No `pip install`, no keys.
  Does all scraping, sentiment, analytics and prediction math, then writes JSON.
- **Frontend — one static `index.html`** (React 17 + Recharts via CDN).
- **Serving — your choice, both free:** `python3 -m http.server` **or** the
  bundled zero-dependency `node server.js`.

Why Python-primary instead of a heavy Node + headless-browser stack: every
source we use is JSON or RSS, which the Python stdlib parses directly — a
headless browser would add fragility and downloads without helping. `server.js`
is included so a Node-only setup remains a one-command option.

---

## Quick start

```bash
# 1) Build the pre-game snapshot (fetches every source in parallel, runs the math)
python3 -m backend.run snapshot

# 2) Serve the dashboard + JSON REST API (Python OR Node — pick one)
python3 -m backend.run api               # http://localhost:8000/ + /api/*
node server.js                           # same, but Node-only
python3 -m http.server 8000              # legacy: static-only

# 3) Live mode (during the game)
python3 -m backend.run live              # polls every 25s
python3 -m backend.run live --once
python3 -m backend.run live --fixture data/fixture_pbp.json
python3 -m backend.run auto              # snapshot + auto-live if game is on

# 4) Native UIs
python3 -m backend.run dashboard         # ANSI CLI (--watch for auto-refresh)
python3 -m backend.run dashboard --tab categories
python3 -m backend.run tui               # interactive curses TUI

# 5) Math + sources tools
python3 -m backend.run simulate --home-win-prob 0.55 --decimal-odds 2.1
python3 -m backend.run advanced          # pretty-print snapshot.advanced
python3 -m backend.run sources           # dry-run every scraper

# 6) Calibration + tests
python3 -m backend.run evaluate --results data/results_sample.json
python3 -m unittest backend.tests backend.tests_advanced backend.tests_v2
```

### REST API (served by `backend.run api` or `node server.js`)

| Endpoint                       | Returns                                   |
|--------------------------------|--------------------------------------------|
| `GET /api/health`              | overall status + source-uptime counters    |
| `GET /api/snapshot`            | full snapshot JSON                         |
| `GET /api/snapshot/<section>`  | one top-level section (e.g. `mood`)        |
| `GET /api/live`                | latest live snapshot                       |
| `GET /api/categories`          | category counts + mean sentiment           |
| `GET /api/category/<name>`     | records filtered to one category           |
| `GET /api/players?team=home`   | per-player buzz/sentiment leaderboard      |
| `GET /api/odds`                | aggregated multi-book odds                 |
| `GET /api/lineups`             | unified lineup payload                     |
| `GET /api/sources`             | per-source health grid                     |
| `GET /api/advanced`            | Monte Carlo, Glicko-2, Kelly, injuries     |
| `GET /api/narratives`          | trending narrative terms + meta            |
| `GET /api/momentum`            | composite live momentum (needs live data)  |
| `GET /api/simulate?p=0.55`     | on-demand Monte Carlo (no snapshot needed) |
| `GET /api/refresh`             | kicks off a snapshot rebuild in background |

---

## Data sources (all free / keyless)

| Layer | Source | What we use | Reliability |
|-------|--------|-------------|-------------|
| Live + Odds | **ESPN** hidden scoreboard | score, status, leaders, moneyline/spread/total | ✅ solid |
| Press review | **Google News**, **Yahoo**, **CBS**, ESPN RSS | headlines + outlet + time | ✅ solid |
| Historical / Elo | **Basketball Reference** | season record + SRS → Elo seed | ✅ works |
| Live + stats + odds + lineups | **Sofascore** (`api.sofascore.com`) | score, PBP incidents, team stats, lineups, multi-book odds, form, h2h, scoring graph, featured players | ✅ rich |
| Live scores | **Flashscore** live feed | cross-check of score + status | ⚠️ best-effort* |
| Live scores | **TheScore** API | cross-check of score + status | ⚠️ best-effort* |
| Standings | **NBA.com Stats** (`stats.nba.com`) | league standings, win%, PF/PA, streak | ⚠️ best-effort* |
| Lineups + injuries | **Rotowire** HTML | projected starters, injury report rows | ⚠️ best-effort* |
| Power rating + ATS / O/U | **TeamRankings** HTML | power ratings, ATS trends, over/under trends | ✅ works |
| Multi-book odds | **OddsPortal** HTML | listing-page best prices | ⚠️ best-effort* |
| Public betting | **Action Network** | bet/money percentages (sharp signal) | ⚠️ best-effort* |
| Fan sentiment | **Reddit** `.json` (r/nba, team subs, r/sportsbook) | posts, upvotes, comments | ⚠️ best-effort* |
| Live play-by-play | **NBA.com** live CDN | scoring events for run detection | ⚠️ best-effort* |

\* Reddit and NBA.com may be blocked by a restrictive network/egress policy; the
system detects this and **degrades gracefully** (news-only sentiment) — every
source's `ok / partial / error` status is shown live in the dashboard. They work
from a normal machine or with a more permissive network policy.

Twitter/X and YouTube live chat are intentionally **excluded** — they no longer
offer a free, keyless API, which would break the no-cost rule.

---

## Analytics & mathematics

All implemented from scratch in pure Python (`backend/model.py`,
`backend/analysis.py`) and covered by tests:

1. **Odds → de-vigged probability** — `1/decimal` per outcome, normalised to
   strip the bookmaker margin (American↔decimal conversion included).
2. **Spread → probability fallback** — `Φ(spread / σ)`, NBA margin `σ ≈ 12`.
3. **Elo / log5** — `1 / (1 + 10^(-(R_home + HCA − R_away)/400))`, Elo seeds
   refined by Basketball-Reference SRS **and nudged by recent (last-5) form**.
4. **ESPN Matchup Predictor** — pulled from ESPN's summary endpoint as an
   independent model input.
5. **Sentiment-adjusted probability** — bounded `SENT_MAX · tanh(k · Δsentiment)`
   (±6 pts cap, so sentiment never overrides the market).
6. **Heat / Hype / Toxicity meters** — saturating combinations of volume,
   engagement, sentiment magnitude/variance and toxicity density.
7. **Emotion classification** — lexicon-based joy/anger/fear/sadness/anticipation.
8. **Player sentiment & narratives** — roster-aware mention/sentiment aggregation.
9. **Value bet** — model edge vs. market + expected value at the offered price.
10. **Live win probability** — normal model on projected final margin from the
    current score + time left, blended toward the pre-game prior.
11. **Live momentum + run detection** + **sentiment-spike z-score**.
12. **Ensemble + confidence** — weighted blend (`market 0.45 / ESPN 0.30 /
    Elo 0.25`, auto-renormalised) + bounded sentiment nudge; confidence from
    cross-model agreement and sample size.
13. **Series-clinch** — from 3-0, `1 − Π(1 − p_leader,game)` over remaining
    venue-adjusted games.
14. **Backtest / calibration** — Brier score, log-loss, accuracy and a skill
    score vs. a coin-flip baseline (`backend/backtest.py`).

---

## The "ok imports directly" rule

`backend/enrich.py` scores every item (sentiment + team attribution) and assigns
a **status**:

- **`ok`** → complete record → **imported directly, with no second filtering
  pass** (the explicit project requirement).
- **`partial`** → one light **repair** pass, then imported (general NBA chatter
  is tagged `general` so it never pollutes this matchup's mood).
- **`error`** → dropped.

Every snapshot reports `import_stats` (`ok` / `partial` / `repaired` / `dropped`).

---

## Project layout

```
backend/
  config.py                 # target game, rosters, source URLs, tuning
  http_util.py              # urllib fetch (gzip, retry/backoff) + parallel runner
  sentiment.py              # VADER-style scorer + NBA/toxicity/emotion lexicons
  enrich.py                 # sentiment + attribution + status + import rule
  analysis.py               # players, narratives, emotions, value bet
  model.py                  # all prediction math (odds, Elo, live WP, ensemble)
  backtest.py               # calibration metrics (Brier / log-loss / skill)
  history.py                # rolling history (data/history.jsonl) for trends
  pipeline.py               # pre-game: fetch(parallel) → enrich → analyse → snapshot.json
  live.py                   # live mode: win prob/runs/momentum/spikes → live.json
  run.py                    # CLI (snapshot | live | auto | evaluate)
  tests.py                  # 39 unit tests
  sources/                  # one tailored scraper per website
    espn.py  google_news.py  reddit.py  nba_cdn.py  basketball_reference.py
data/
  snapshot.json             # generated pre-game data (committed as a live demo)
  live.json                 # generated live data (git-ignored, transient)
  history.jsonl             # rolling metric history (git-ignored)
  fixture_pbp.json          # synthetic play-by-play to demo live mode
index.html                  # the dashboard (Mood · Press · Prediction · Live)
server.js                   # optional zero-dependency Node static server
```

---

## Testing

```bash
python3 -m unittest backend.tests       # 39 tests, pure stdlib, no network
```

Covers the sentiment engine (incl. phrases/emoji/contrast), emotion classifier,
odds de-vig, Elo + form adjustment, live win probability, ensemble bounding,
run/momentum/spike detection, series-clinch, player attribution, narratives,
value bet, calibration metrics, the import rule and history round-trip.

---

## Configuration

Edit [`backend/config.py`](backend/config.py) to point at a different game:
`GAME` (teams, date, venue, series), `ROSTERS`, the source URLs/queries, and
tuning constants (Elo seeds, ensemble weights, sentiment cap, poll interval).

---

## Roadmap

- Reddit comment-thread descent for finer fan signal (when reachable).
- Feed completed-game results into `backtest.py` automatically to tune the
  ensemble weights and sentiment cap from the live calibration log.
- Compare multiple sportsbooks and flag the sharpest line (when ESPN exposes
  more than one provider).
- Auto-discover the live playoff game from the scoreboard.
- Optional WebSocket push instead of polling for sub-second live updates.

---

## Keywords

`NBA` · `sentiment analysis` · `sports analytics` · `press review` ·
`win probability` · `betting odds` · `de-vig` · `Elo` · `VADER` ·
`Reddit` · `ESPN API` · `Google News RSS` · `Basketball Reference` ·
`Python` · `React` · `Recharts` · `no API key` · `free` · `live scraper` ·
`Knicks` · `Cavaliers` · `Eastern Conference Finals`

> Suggested GitHub repo **topics** (set in the repo's *About* panel for
> discoverability): `nba`, `sentiment-analysis`, `sports-analytics`,
> `web-scraping`, `python`, `react`, `data-visualization`, `prediction`.
