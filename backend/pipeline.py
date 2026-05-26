"""Pre-game pipeline: fetch all sources -> enrich/import -> compute -> write.

Produces data/snapshot.json, the file the dashboard reads. Each source runs in
isolation; a failure degrades gracefully (e.g. Reddit blocked -> news-only
sentiment) and is reported in the snapshot's "sources" section.
"""

import datetime as dt
import json
import os

from . import (advanced_math, aggregator, analysis, categorizer, config,
               enrich, history, injury_impact, lineup_analyzer, model,
               simulation, streaks)
from .http_util import run_parallel
from .sources import (action_network, balldontlie, basketball_reference,
                      cbssports_nba, espn, flashscore, google_news, nba_stats,
                      nba_stats_v2, odds_api_free, pbpstats, reddit, rotowire,
                      sofascore, teamrankings, thescore)
from .sources.base import SourceResult, STATUS_ERROR, STATUS_PARTIAL


def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _mode_from_state(state):
    return {"pre": "pre", "in": "live", "post": "post"}.get(state, "pre")


def _sentiment_timeline(records):
    """Group imported records by calendar day -> mean sentiment + volume."""
    buckets = {}
    for r in records:
        pub = r.get("published")
        day = "undated"
        if pub:
            day = pub[:10]
        b = buckets.setdefault(day, {"comps": [], "count": 0})
        b["comps"].append(r["sentiment"]["compound"])
        b["count"] += 1
    timeline = []
    for day in sorted(d for d in buckets if d != "undated"):
        comps = buckets[day]["comps"]
        timeline.append({
            "date": day,
            "mean_sentiment": round(sum(comps) / len(comps), 4),
            "volume": buckets[day]["count"],
        })
    return timeline


def _per_game_series_probs(ratings, ensemble_away_g4):
    """Knicks (leader, away team) win prob for each remaining game.

    Game 4 uses the market-blended ensemble; games 5-7 use Elo by venue.
    Schedule from Game 4: CLE, NYK, CLE, NYK.
    """
    r_cle, r_nyk = ratings["home"], ratings["away"]
    # NYK at NYK (home): NYK favored by HCA. NYK at CLE (away): subtract HCA.
    nyk_home = model.elo_expected(r_nyk, r_cle)               # NYK home win %
    nyk_away = 1 - model.elo_expected(r_cle, r_nyk)           # NYK away win %
    return [
        {"game": 4, "venue": "CLE", "leader_win": round(ensemble_away_g4, 4)},
        {"game": 5, "venue": "NYK", "leader_win": round(nyk_home, 4)},
        {"game": 6, "venue": "CLE", "leader_win": round(nyk_away, 4)},
        {"game": 7, "venue": "NYK", "leader_win": round(nyk_home, 4)},
    ]


def _resolve_market(game):
    """Market home/away probabilities from moneyline, else from the spread.

    Returns the devig dict (with a 'method' tag) or None.
    """
    if not game or not game.get("odds"):
        return None
    odds = game["odds"]
    market = model.devig_two_way(odds.get("home_moneyline"),
                                 odds.get("away_moneyline"))
    if market:
        market["method"] = "moneyline_devig"
        return market
    # Fallback: derive from the point spread (normal-CDF heuristic).
    spread = odds.get("spread")
    if spread is None:
        return None
    p_fav = model.prob_from_spread(spread)
    home_fav = odds.get("home_favorite")
    p_home = p_fav if home_fav else (1 - p_fav)
    return {"home": round(p_home, 4), "away": round(1 - p_home, 4),
            "overround": None, "margin_pct": None, "method": "spread_normal_cdf"}


def build_snapshot():
    # --- 1. Fetch every source CONCURRENTLY, each fully isolated ----------
    # 4 "core" sources we always pull, plus the new "depth" sources from
    # Sofascore / Flashscore / TheScore / NBA stats / Rotowire / TeamRankings.
    fetched = run_parallel({
        # Core (already in v1).
        "espn": espn.fetch_game,
        "press": google_news.fetch_press_review,
        "reddit": reddit.fetch_social,
        "bref": basketball_reference.fetch_history,
        # New scrapers — each is best-effort, failure isolated.
        "sofascore_bundle": sofascore.fetch_all,
        "flashscore": flashscore.fetch_game,
        "thescore": thescore.fetch_game,
        "nba_standings": nba_stats.fetch_standings,
        "rotowire_lineups": rotowire.fetch_lineups,
        "rotowire_injuries": rotowire.fetch_injuries,
        "teamrankings_power": teamrankings.fetch_power_ratings,
        "teamrankings_ats": teamrankings.fetch_ats_trends,
        "teamrankings_ou": teamrankings.fetch_ou_trends,
        # New sources added in v2.
        "action_network": action_network.fetch_public_betting,
        "nba_stats_v2": nba_stats_v2.fetch_standings_v2,
        "balldontlie": balldontlie.fetch_team_stats,
        "odds_scraper": odds_api_free.fetch_odds,
        "cbssports_injuries": cbssports_nba.fetch_injuries,
        "cbssports_news": cbssports_nba.fetch_news,
        "pbpstats": pbpstats.fetch_team_stats,
    }, max_workers=12)

    def _safe(name, source_name):
        r = fetched.get(name)
        if isinstance(r, Exception):
            return SourceResult(source_name, STATUS_ERROR, error=str(r))
        return r

    espn_res = _safe("espn", "espn")
    news_res = _safe("press", "press_review")
    reddit_res = _safe("reddit", "reddit")
    bref_res = _safe("bref", "basketball_reference")
    flash_res = _safe("flashscore", "flashscore")
    score_res = _safe("thescore", "thescore")
    nba_stand = _safe("nba_standings", "nba_stats_standings")
    rw_line = _safe("rotowire_lineups", "rotowire_lineups")
    rw_inj = _safe("rotowire_injuries", "rotowire_injuries")
    tr_power = _safe("teamrankings_power", "teamrankings_power")
    tr_ats = _safe("teamrankings_ats", "teamrankings_ats")
    tr_ou = _safe("teamrankings_ou", "teamrankings_ou")
    an_res = _safe("action_network", "action_network")
    nba_v2_res = _safe("nba_stats_v2", "nba_stats_v2")
    bdl_res = _safe("balldontlie", "balldontlie")
    odds_res = _safe("odds_scraper", "odds_scraper")
    cbs_inj_res = _safe("cbssports_injuries", "cbssports_injuries")
    cbs_news_res = _safe("cbssports_news", "cbssports_news")
    pbp_res = _safe("pbpstats", "pbpstats")

    # Sofascore bundle is a dict of SourceResult; flatten it for the source list.
    sofa_bundle = fetched.get("sofascore_bundle")
    if isinstance(sofa_bundle, Exception):
        sofa_bundle = {"discover": SourceResult("sofascore",
                                                  STATUS_ERROR,
                                                  error=str(sofa_bundle))}
    sofa_bundle = sofa_bundle or {}

    sources = [espn_res, news_res, reddit_res, bref_res, flash_res, score_res,
               nba_stand, rw_line, rw_inj, tr_power, tr_ats, tr_ou,
               an_res, nba_v2_res, bdl_res, odds_res,
               cbs_inj_res, cbs_news_res, pbp_res]
    sources.extend(sofa_bundle.values())

    # --- 2. Enrich + import (ok bypasses re-filter) -----------------------
    # Now also include Rotowire injuries + lineups so they get sentiment-scored
    # and team-attributed alongside news.
    raw = (list(news_res.records) + list(reddit_res.records)
           + list(rw_line.records) + list(rw_inj.records)
           + list(cbs_inj_res.records) + list(cbs_news_res.records))
    imported, import_stats = enrich.enrich_and_import(raw)
    categorizer.categorize_records(imported)

    press_review = [r for r in imported if r["kind"] == "article"]
    social = [r for r in imported if r["kind"] == "social"]
    lineups_raw = [r for r in imported if r["kind"] == "lineup"]
    injuries_raw = [r for r in imported if r["kind"] == "injury"]
    category_summary = categorizer.category_breakdown(imported)

    # --- 3. Mood meters + per-team sentiment ------------------------------
    # Only matchup-relevant records feed the mood; "general" NBA news is kept
    # in the dataset but excluded here so it doesn't distort this game's read.
    relevant = [r for r in imported if r.get("team") in ("home", "away", "both")]
    home_recs = [r for r in imported if r.get("team") in ("home", "both")]
    away_recs = [r for r in imported if r.get("team") in ("away", "both")]
    mood = {
        "overall": model.mood_meters(relevant),
        "home": model.mood_meters(home_recs),
        "away": model.mood_meters(away_recs),
        "team_sentiment": model.team_sentiment(relevant),
        "timeline": _sentiment_timeline(relevant),
        "emotions": analysis.emotion_profile(relevant),
    }

    # --- 3b. Deeper analysis: players, narratives -------------------------
    players = analysis.player_sentiment(imported)
    narratives_list = analysis.narratives(imported)
    narrative_meta = {
        "concentration": analysis.narrative_concentration(narratives_list),
        "polarity": analysis.sentiment_polarity(relevant),
        "outlets": analysis.top_outlets(press_review),
    }

    # --- 3c. Cross-source unification (scores, odds, lineups, source health)
    sofa_game = (sofa_bundle.get("game").meta.get("game")
                  if (sofa_bundle.get("game")
                      and sofa_bundle["game"].meta) else None)
    flash_game = flash_res.meta.get("game") if flash_res.meta else None
    score_game = score_res.meta.get("game") if score_res.meta else None
    espn_game_for_merge = (espn_res.meta.get("game") if espn_res.meta else None)

    scores_unified = aggregator.merge_scores({
        "espn": espn_game_for_merge,
        "sofascore": sofa_game,
        "flashscore": flash_game,
        "thescore": score_game,
    })

    sofa_lineups = (sofa_bundle.get("lineups").meta.get("lineups")
                     if (sofa_bundle.get("lineups")
                         and sofa_bundle["lineups"].meta) else None)
    lineups_unified = aggregator.pick_lineup(sofa_lineups)
    lineup_meta = None
    if lineups_unified:
        lineup_meta = {
            "starting_advantage": lineup_analyzer.starting_advantage(
                lineups_unified.get("home"), lineups_unified.get("away")),
            "home_missing": lineup_analyzer.missing_players_impact(
                lineups_unified.get("home")),
            "away_missing": lineup_analyzer.missing_players_impact(
                lineups_unified.get("away")),
            "home_box_total": lineup_analyzer.box_score_summary(
                lineups_unified.get("home")),
            "away_box_total": lineup_analyzer.box_score_summary(
                lineups_unified.get("away")),
            "home_top_minutes": lineup_analyzer.top_minutes_players(
                lineups_unified.get("home"), n=5),
            "away_top_minutes": lineup_analyzer.top_minutes_players(
                lineups_unified.get("away"), n=5),
        }

    sofa_odds_block = (sofa_bundle.get("odds").meta.get("odds")
                        if (sofa_bundle.get("odds")
                            and sofa_bundle["odds"].meta) else None)
    espn_odds_block = (espn_game_for_merge or {}).get("odds")
    odds_unified = aggregator.merge_odds_books(espn_odds_block, sofa_odds_block)

    # Sofascore stats / form / h2h / featured / graph passthroughs.
    sofa_stats = (sofa_bundle.get("stats").meta.get("stats")
                   if (sofa_bundle.get("stats")
                       and sofa_bundle["stats"].meta) else None)
    sofa_form = (sofa_bundle.get("form").meta.get("form")
                  if (sofa_bundle.get("form")
                      and sofa_bundle["form"].meta) else None)
    sofa_h2h = (sofa_bundle.get("h2h").meta.get("h2h")
                 if (sofa_bundle.get("h2h")
                     and sofa_bundle["h2h"].meta) else None)
    sofa_graph = (sofa_bundle.get("graph").meta.get("points")
                   if (sofa_bundle.get("graph")
                       and sofa_bundle["graph"].meta) else None)
    sofa_featured = (sofa_bundle.get("featured").meta.get("featured")
                      if (sofa_bundle.get("featured")
                          and sofa_bundle["featured"].meta) else None)
    sofa_incidents = (sofa_bundle.get("incidents").records
                       if sofa_bundle.get("incidents") else None)

    # Team rankings + standings passthroughs.
    teamrank = {
        "power": (tr_power.meta or {}).get("rankings"),
        "ats": (tr_ats.meta or {}).get("trends"),
        "ou": (tr_ou.meta or {}).get("trends"),
    }
    standings = (nba_stand.meta or {}).get("standings")
    if standings:
        standings_target = nba_stats.standings_for_teams(
            config.GAME["home"]["abbr"], config.GAME["away"]["abbr"], standings)
    else:
        standings_target = None

    # --- 4. Prediction math -----------------------------------------------
    game = espn_res.meta.get("game") if espn_res.meta else None
    market = _resolve_market(game)

    # ESPN matchup summary (predictor + form + season series), best-effort.
    summary_meta = {}
    if game and game.get("id"):
        summ = espn.fetch_summary(game["id"])
        sources.append(summ)
        summary_meta = summ.meta or {}
    espn_pred = summary_meta.get("predictor")

    ratings = model.elo_from_history(bref_res.meta if bref_res.meta else None)
    # Form first from ESPN, then refined by Sofascore (more recent).
    ratings = model.elo_adjust_for_form(ratings, summary_meta.get("form"))
    if sofa_form:
        ratings = model.elo_adjust_for_form(ratings, sofa_form)
    p_elo_home = model.elo_expected(ratings["home"], ratings["away"])
    ts = mood["team_sentiment"]
    delta = model.sentiment_delta(ts)

    # Power-rating signal from TeamRankings, if available.
    tr_prob = _power_rating_prob(teamrank.get("power"),
                                  config.GAME["home"]["name"],
                                  config.GAME["away"]["name"])

    model_probs = {
        "market": market["home"] if market else None,
        "elo": p_elo_home,
        "espn": espn_pred["home"] if espn_pred else None,
        "power": tr_prob,
    }
    # NetRating nudge from pbpstats (applied before final ensemble blend).
    pbp_team_stats = (pbp_res.meta or {}).get("team_stats") or []
    if pbp_team_stats:
        ratings = model.elo_from_net_rating(
            ratings, pbp_team_stats,
            config.GAME["home"]["abbr"], config.GAME["away"]["abbr"],
        )
        p_elo_home = model.elo_expected(ratings["home"], ratings["away"])
        model_probs["elo"] = p_elo_home

    ens = model.ensemble(model_probs, delta)
    conf = model.confidence(model_probs, ts["count_home"] + ts["count_away"])

    # Public-bet % nudge from Action Network.
    an_betting = (an_res.meta or {}).get("betting") or {}
    conf = model.apply_public_bet_nudge(conf, an_betting)

    per_game = _per_game_series_probs(ratings, ens["away"])
    clinch = model.series_clinch([g["leader_win"] for g in per_game])

    value = analysis.value_bet(market, ens, game.get("odds") if game else None)
    advanced = _compute_advanced(
        ens=ens, ratings=ratings, per_game=per_game,
        odds=(game.get("odds") if game else None), value=value,
        history_rows=history.load_recent(), imported=imported,
    )

    prediction = {
        "market": market,
        "elo": {
            "home": round(p_elo_home, 4),
            "away": round(1 - p_elo_home, 4),
            "ratings": {k: round(v, 1) for k, v in ratings.items()},
        },
        "espn_predictor": espn_pred,
        "sentiment_delta": round(delta, 4),
        "ensemble": ens,
        "confidence": conf,
        "value_bet": value,
        "series": {
            "leader": config.GAME["series"]["leader"],
            "lead": config.GAME["series"]["lead"],
            "leader_clinch_probability": clinch,
            "per_game": per_game,
        },
    }
    if game is not None:
        game["form"] = summary_meta.get("form")
        game["season_series"] = summary_meta.get("season_series")

    state = game.get("state") if game else "pre"
    snapshot = {
        "generated_at": _now_iso(),
        "mode": _mode_from_state(state),
        "game": game,
        "series": config.GAME["series"],
        "venue": config.GAME["venue"],
        "label": config.GAME["label"],
        "teams": {"home": config.GAME["home"], "away": config.GAME["away"]},
        "sources": [s.to_dict() for s in sources],
        "source_health": aggregator.source_health([s.to_dict() for s in sources]),
        "press_review": sorted(press_review,
                               key=lambda r: r.get("published") or "",
                               reverse=True),
        "social": sorted(social, key=lambda r: r.get("engagement", 0),
                         reverse=True),
        "mood": mood,
        "players": players,
        "narratives": narratives_list,
        "narrative_meta": narrative_meta,
        "categories": category_summary,
        "prediction": prediction,
        "advanced": advanced,
        # New top-level blocks fed by the additional scrapers.
        "scores_unified": scores_unified,
        "lineups_unified": lineups_unified,
        "lineup_meta": lineup_meta,
        "odds_unified": odds_unified,
        "sofascore": {
            "stats": sofa_stats,
            "form": sofa_form,
            "h2h": sofa_h2h,
            "graph": sofa_graph,
            "featured": sofa_featured,
            "incidents": sofa_incidents,
        },
        "teamrankings": teamrank,
        "standings": standings_target,
        "standings_v2": (nba_v2_res.meta or {}).get("standings_v2"),
        "balldontlie": (bdl_res.meta or {}).get("team_stats"),
        "pbpstats": (pbp_res.meta or {}).get("team_stats"),
        "action_network": an_betting or None,
        "odds_scraper": (odds_res.meta or {}).get("books"),
        "lineups_records": lineups_raw,
        "injuries_records": injuries_raw,
        "import_stats": import_stats,
    }
    # Append to the rolling history, then attach the recent tail for trends.
    history.append(snapshot)
    snapshot["history"] = history.load_recent()
    return snapshot


def _power_rating_prob(rankings, home_name: str, away_name: str) -> float | None:
    """Translate TeamRankings power ratings into a home win probability.

    Looks up both teams by name, takes the rating difference and turns it
    into a probability via the same Elo-style logistic mapping we already use.
    """
    if not rankings:
        return None
    h = a = None
    for row in rankings:
        team = (row.get("team") or "").lower()
        if not team:
            continue
        if home_name.lower().split()[-1] in team:
            h = row.get("rating")
        if away_name.lower().split()[-1] in team:
            a = row.get("rating")
    if h is None or a is None:
        return None
    diff = (h + 3.0) - a    # +3 power points ~= home court
    # 8 power points ~= one win expectancy unit.
    return round(1.0 / (1.0 + 10 ** (-diff / 8.0)), 4)


def write_snapshot(path=None):
    path = path or config.SNAPSHOT_PATH
    snapshot = build_snapshot()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot, path


# ---------------------------------------------------------------------------
# Advanced math add-on: Monte Carlo, Glicko-2, Kelly, injuries, time-series.
# Pulls a few seconds of stdlib compute on top of the existing snapshot so
# the dashboard has a richer prediction panel.
# ---------------------------------------------------------------------------
def _compute_advanced(*, ens, ratings, per_game, odds, value, history_rows,
                      imported):
    # Deterministic across runs so the dashboard doesn't jiggle on refresh.
    simulation.seed(42)
    home_wp = ens.get("home", 0.5)

    mc_game = simulation.simulate_game(home_wp, trials=4000)
    leader_probs = [g["leader_win"] for g in per_game]
    mc_series = simulation.simulate_series(
        leader_probs, leader_wins=3, trailer_wins=0, trials=8000)

    # Glicko-2 expressed from current Elo seeds (high-precision win prob).
    glicko = {
        "home_rating": round(ratings["home"], 1),
        "home_rd": 60.0,
        "away_rating": round(ratings["away"], 1),
        "away_rd": 60.0,
        "home_win_prob": round(advanced_math.glicko2_win_prob(
            ratings["home"], 60.0, ratings["away"], 60.0, home_court=50.0), 4),
    }

    # Kelly sizing on the value-bet side, using the model probability + ML.
    kelly_block = None
    if value and odds:
        ml = value.get("moneyline")
        dec = model.american_to_decimal(ml) if ml is not None else None
        if dec is not None:
            p = value.get("model_prob") or 0.0
            kelly_block = {
                "side": value.get("side"),
                "decimal": round(dec, 3),
                "full": round(advanced_math.kelly_fraction(p, dec), 4),
                "quarter": round(advanced_math.fractional_kelly(p, dec, 0.25), 4),
                "ev": round(advanced_math.expected_value(p, dec), 4),
            }

    # Injury detection from the press/social corpus, propagated to a WP delta.
    inj = injury_impact.estimated_team_impact(imported)
    inj_adj = injury_impact.adjust_win_probability(home_wp, inj)
    inj["adjustment"] = inj_adj

    # Time-series moves from history (last 30 runs).
    ts_moves = _history_movement(history_rows)

    return {
        "monte_carlo_game": mc_game,
        "monte_carlo_series": mc_series,
        "glicko": glicko,
        "kelly": kelly_block,
        "injuries": inj,
        "history_movement": ts_moves,
        "what_if_star_out_home": simulation.what_if_player_out(home_wp, 0.20),
        "what_if_star_out_away": simulation.what_if_player_out(home_wp, -0.20)
            if False else simulation.what_if_player_out(1 - home_wp, 0.20),
    }


def _history_movement(rows):
    """Quick stats over the rolling history for the dashboard.

    Computes:
      * EWMA-smoothed ensemble home %
      * Autocorrelation at lag 1 (so a noisy stream vs a trending one is obvious)
      * Linear regression slope vs index (rough drift speed)
      * Pearson(ens_home, market_home) and Pearson(ens_home, sent_home)
    """
    if not rows:
        return {}
    ens_series = [r.get("ens_home") for r in rows if r.get("ens_home") is not None]
    market_series = [r.get("market_home") for r in rows if r.get("market_home") is not None]
    sent_series = [r.get("sent_home") for r in rows if r.get("sent_home") is not None]
    out = {}
    if ens_series:
        out["ewma_ens_home"] = [round(v, 4) for v in advanced_math.ewma(ens_series, 0.3)]
        lr = advanced_math.linear_regression(list(range(len(ens_series))), ens_series)
        out["ens_drift_per_run"] = round(lr["slope"], 5)
        out["ens_r2"] = round(lr["r2"], 4)
        out["ens_autocorr_lag1"] = round(advanced_math.autocorrelation(ens_series, 1), 4)
    if ens_series and market_series:
        n = min(len(ens_series), len(market_series))
        out["pearson_ens_vs_market"] = round(
            advanced_math.pearson_correlation(ens_series[-n:], market_series[-n:]), 4)
    if ens_series and sent_series:
        n = min(len(ens_series), len(sent_series))
        out["pearson_ens_vs_sentiment"] = round(
            advanced_math.pearson_correlation(ens_series[-n:], sent_series[-n:]), 4)
    return out
