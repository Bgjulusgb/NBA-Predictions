"""Live mode: during the game, poll fast-moving sources and recompute the
live indices (momentum, scoring runs, sentiment spikes), writing data/live.json.

The dashboard auto-refreshes this file and switches into its live view. Because
real play-by-play only exists at tip-off, build_live() accepts injected events
so the whole live path is testable now via a synthetic fixture.
"""

import datetime as dt
import json
import os
import time

from . import aggregator, config, enrich, momentum_composite, model
from .sources import espn, flashscore, nba_cdn, reddit, sofascore, thescore


def _now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _score_timeline(events):
    out = []
    for ev in events:
        if ev.get("score_home") is None and ev.get("score_away") is None:
            continue
        out.append({
            "period": ev.get("period"),
            "clock": ev.get("clock"),
            "home": _to_int(ev.get("score_home")),
            "away": _to_int(ev.get("score_away")),
        })
    return out


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _social_sentiment_buckets(records, minutes=5):
    """Bucket social records into time windows -> mean sentiment per bucket."""
    if not records:
        return []
    timed = [(r.get("published"), r["sentiment"]["compound"])
             for r in records if r.get("published") and "sentiment" in r]
    if not timed:
        return []
    timed.sort()
    buckets = {}
    for pub, comp in timed:
        try:
            t = dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except ValueError:
            continue
        key = int(t.timestamp() // (minutes * 60))
        buckets.setdefault(key, []).append(comp)
    return [sum(v) / len(v) for _, v in sorted(buckets.items())]


def _build_alerts(run, momentum_val, spike, game):
    """Generate viral-style headlines from the live signals."""
    alerts = []
    home_abbr = config.GAME["home"]["abbr"]
    away_abbr = config.GAME["away"]["abbr"]
    if run and run.get("points", 0) >= 6 and run.get("team"):
        team = run["team"]
        alerts.append({
            "type": "run",
            "text": f"{team} on a {run['points']}-0 run",
            "severity": "high" if run["points"] >= 10 else "medium",
        })
    if abs(momentum_val) >= 0.5:
        side = home_abbr if momentum_val > 0 else away_abbr
        alerts.append({
            "type": "momentum",
            "text": f"Momentum swinging hard toward {side}",
            "severity": "medium",
        })
    if spike <= -1.5:
        alerts.append({
            "type": "sentiment_drop",
            "text": "Fan sentiment crashing — panic detected in live chatter",
            "severity": "high",
        })
    elif spike >= 1.5:
        alerts.append({
            "type": "sentiment_surge",
            "text": "Fan hype spiking in live chatter",
            "severity": "medium",
        })
    return alerts


def build_live(pbp_events=None, social_records=None, game=None):
    """Compute the live snapshot. Fetches sources unless data is injected.

    Now multi-source: ESPN + Sofascore + Flashscore + TheScore for the game
    state (median-consensus score), Sofascore + NBA.com CDN for play-by-play.
    """
    multi_game = None
    if game is None:
        # Pull every game source concurrently.
        from .http_util import run_parallel
        bundle = run_parallel({
            "espn": espn.fetch_game,
            "sofa_disc": sofascore.discover_event_id,
            "flash": flashscore.fetch_game,
            "score": thescore.fetch_game,
        }, max_workers=4)

        espn_game = (_meta_game(bundle.get("espn"))
                      if not isinstance(bundle.get("espn"), Exception) else None)
        flash_game = (_meta_game(bundle.get("flash"))
                       if not isinstance(bundle.get("flash"), Exception) else None)
        score_game = (_meta_game(bundle.get("score"))
                       if not isinstance(bundle.get("score"), Exception) else None)

        sofa_disc = bundle.get("sofa_disc")
        sofa_game = None
        if (not isinstance(sofa_disc, Exception) and isinstance(sofa_disc, tuple)
                and sofa_disc[0]):
            sg_res = sofascore.fetch_game(sofa_disc[0])
            sofa_game = _meta_game(sg_res)

        multi_game = aggregator.merge_scores({
            "espn": espn_game, "sofascore": sofa_game,
            "flashscore": flash_game, "thescore": score_game,
        })
        # Use ESPN as the primary game object, but fall back through the list.
        game = espn_game or sofa_game or flash_game or score_game

    # Play-by-play from NBA.com CDN AND Sofascore incidents (when game id known).
    pbp_source_status = "injected"
    sofa_inc_status = "skipped"
    if pbp_events is None:
        pbp_res = nba_cdn.fetch_playbyplay()
        pbp_events = list(pbp_res.records)
        pbp_source_status = pbp_res.status
        # Also pull Sofascore incidents and merge them in.
        sofa_disc, _ = sofascore.discover_event_id()
        if sofa_disc:
            inc_res = sofascore.fetch_incidents(sofa_disc)
            sofa_inc_status = inc_res.status
            pbp_events = aggregator.merge_pbp(pbp_events, inc_res.records)

    scoring = [e for e in pbp_events if e.get("points", 0) > 0]
    run = model.detect_current_run(scoring)
    mom = model.momentum(scoring)

    # Live social sentiment unless injected.
    social_status = "injected"
    if social_records is None:
        soc_res = reddit.fetch_social()
        social_records, _ = enrich.enrich_and_import(list(soc_res.records))
        social_status = soc_res.status

    buckets = _social_sentiment_buckets(social_records)
    spike = model.sentiment_spike(buckets)
    mood = model.mood_meters(social_records)
    ts = model.team_sentiment(social_records)

    # Live win probability from current margin + time remaining, anchored to
    # the pre-game ensemble prior.
    win_prob = None
    if game and game.get("home") and game.get("away"):
        hs = game["home"].get("score")
        as_ = game["away"].get("score")
        if hs is not None and as_ is not None:
            pregame = _pregame_home_prob()
            p_home = model.live_win_probability(
                hs - as_, game.get("period"), game.get("clock"), pregame)
            win_prob = {"home": p_home, "away": round(1 - p_home, 4),
                        "pregame_home": pregame}

    alerts = _build_alerts(run, mom, spike, game)

    # Composite momentum across signals (when we have any).
    composite_mom = momentum_composite.composite(
        scoring_momentum=mom,
        current_run=run,
        home_abbr=config.GAME["home"]["abbr"],
        sentiment_zscore=spike,
        starting_plus_minus_diff=None,
        pace_ratio=None,
    )

    state = game.get("state") if game else "pre"
    return {
        "generated_at": _now_iso(),
        "mode": {"pre": "pre", "in": "live", "post": "post"}.get(state, "pre"),
        "game": game,
        "scores_unified": multi_game,
        "live": {
            "current_run": run,
            "momentum": mom,
            "composite_momentum": composite_mom,
            "win_probability": win_prob,
            "sentiment_spike": spike,
            "mood": mood,
            "team_sentiment": ts,
            "score_timeline": _score_timeline(pbp_events),
            "recent_events": [e for e in pbp_events if e.get("desc") or e.get("text")][-15:],
            "pbp_status": pbp_source_status,
            "sofa_inc_status": sofa_inc_status,
            "social_status": social_status,
        },
        "alerts": alerts,
    }


def _meta_game(res):
    """Pull the .meta['game'] from a SourceResult-like (None-safe)."""
    if res is None or isinstance(res, Exception):
        return None
    meta = getattr(res, "meta", None) or {}
    return meta.get("game")


def _pregame_home_prob(default=0.5):
    """Read the pre-game ensemble home probability from snapshot.json."""
    try:
        with open(config.SNAPSHOT_PATH, encoding="utf-8") as f:
            snap = json.load(f)
        return snap["prediction"]["ensemble"]["home"]
    except (OSError, KeyError, ValueError):
        return default


def write_live(path=None, **kwargs):
    path = path or config.LIVE_PATH
    snapshot = build_live(**kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot, path


def run_live_loop(poll_seconds=None, max_iterations=None):
    """Poll until the game reaches 'post' (or max_iterations for testing)."""
    poll_seconds = poll_seconds or config.LIVE_POLL_SECONDS
    i = 0
    while True:
        snapshot, path = write_live()
        i += 1
        mode = snapshot.get("mode")
        run = snapshot["live"]["current_run"]
        print(f"[live] iter={i} mode={mode} momentum={snapshot['live']['momentum']} "
              f"run={run.get('team')} {run.get('points')} -> {path}")
        if mode == "post":
            print("[live] game finished; stopping live loop.")
            break
        if max_iterations and i >= max_iterations:
            break
        time.sleep(poll_seconds)
