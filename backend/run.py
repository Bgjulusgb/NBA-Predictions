"""Command-line entrypoint for the NBA Mood Mirror backend.

Usage:
    python -m backend.run snapshot          # build pre-game data/snapshot.json
    python -m backend.run live              # poll live during the game
    python -m backend.run live --once       # one live iteration
    python -m backend.run live --fixture data/fixture_pbp.json   # test live
    python -m backend.run auto              # snapshot, then live if game is on
    python -m backend.run dashboard         # colourised CLI dashboard
    python -m backend.run dashboard --watch # auto-refresh dashboard
    python -m backend.run tui               # interactive curses UI
    python -m backend.run simulate          # Monte Carlo, Glicko, Kelly
    python -m backend.run evaluate ...      # calibration metrics
"""

import argparse
import json
import sys

from . import cli_dashboard, config, live, pipeline
from .sources import espn


def _cmd_snapshot(_args):
    snap, path = pipeline.write_snapshot()
    pred = snap["prediction"]["ensemble"]
    print(f"Wrote {path}")
    print(f"  mode={snap['mode']}  articles={len(snap['press_review'])}  "
          f"social={len(snap['social'])}")
    print(f"  import_stats={snap['import_stats']}")
    print(f"  ensemble home={pred['home']} away={pred['away']} "
          f"confidence={snap['prediction']['confidence']}")
    print("  sources: " + ", ".join(
        f"{s['name']}={s['status']}" for s in snap["sources"]))
    return 0


def _cmd_live(args):
    if args.fixture:
        with open(args.fixture, encoding="utf-8") as f:
            fixture = json.load(f)
        snap, path = live.write_live(
            pbp_events=fixture.get("events", []),
            social_records=fixture.get("social", []),
            game=fixture.get("game"),
        )
        print(f"Wrote {path} (fixture)")
        print(f"  run={snap['live']['current_run']} "
              f"momentum={snap['live']['momentum']} "
              f"spike={snap['live']['sentiment_spike']}")
        print(f"  alerts={[a['text'] for a in snap['alerts']]}")
        return 0
    if args.once:
        snap, path = live.write_live()
        print(f"Wrote {path}  mode={snap['mode']}")
        return 0
    live.run_live_loop(max_iterations=args.max_iterations)
    return 0


def _cmd_evaluate(args):
    from . import backtest
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    metrics = backtest.evaluate(results)
    print("Calibration metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_auto(args):
    res = espn.fetch_game()
    game = res.meta.get("game") if res.meta else None
    state = game.get("state") if game else "pre"
    pipeline.write_snapshot()
    print(f"Snapshot written. ESPN game state = {state}")
    if state == "in":
        print("Game is live -> starting live loop.")
        live.run_live_loop(max_iterations=args.max_iterations)
    return 0


def _cmd_dashboard(args):
    return cli_dashboard.show(tab=args.tab, watch=args.watch,
                              interval=args.interval)


def _cmd_tui(_args):
    from . import tui
    return tui.main()


def _cmd_simulate(args):
    from . import advanced_math, simulation
    simulation.seed(args.seed)
    home_wp = args.home_win_prob
    print(f"Simulating with home_win_prob={home_wp}, trials={args.trials}")
    game = simulation.simulate_game(home_wp, trials=args.trials)
    print("\n[Monte Carlo - game]")
    for k, v in game.items():
        if k == "histogram":
            continue
        print(f"  {k:24} {v}")

    series = simulation.simulate_series(
        [home_wp] * 4, leader_wins=3, trailer_wins=0, trials=args.trials)
    print("\n[Monte Carlo - series remainder]")
    for k, v in series.items():
        print(f"  {k:24} {v}")

    alt = simulation.simulate_alt_lines(home_wp, trials=args.trials)
    print("\n[Alternate lines]")
    for k, v in alt.items():
        print(f"  {k:24} {v}")

    poss = simulation.simulate_possession(possessions=100, trials=args.trials // 4)
    print("\n[Possession-by-possession]")
    for k, v in poss.items():
        print(f"  {k:24} {v}")

    if args.decimal_odds:
        kf = advanced_math.kelly_fraction(home_wp, args.decimal_odds)
        ev = advanced_math.expected_value(home_wp, args.decimal_odds)
        print("\n[Kelly]")
        print(f"  full Kelly    {kf*100:.3f}%")
        print(f"  1/4 Kelly     {kf*25:.3f}%")
        print(f"  EV per unit   {ev:+.4f}")
    return 0


def _cmd_advanced(_args):
    """Pretty-print the 'advanced' block from snapshot.json (no recomputation)."""
    try:
        with open(config.SNAPSHOT_PATH, encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Couldn't read snapshot: {e}", file=sys.stderr)
        return 1
    print(json.dumps(snap.get("advanced", {}), indent=2))
    return 0


def _cmd_api(args):
    """Run the stdlib HTTP API + static dashboard server."""
    from . import api_server
    api_server.serve(port=args.port, bind=args.bind)
    return 0


def _cmd_sources(_args):
    """List every configured scraper + reachability dry-run."""
    from .sources import (action_network, balldontlie, basketball_reference,
                          cbssports_nba, espn, flashscore, google_news,
                          nba_cdn, nba_stats, nba_stats_v2, odds_api_free,
                          pbpstats, reddit, rotowire, sofascore,
                          teamrankings, thescore)
    print("Configured scrapers:")
    for name, fn in [
        ("espn",                 espn.fetch_game),
        ("google_news",          google_news.fetch_press_review),
        ("reddit",               reddit.fetch_social),
        ("basketball_reference", basketball_reference.fetch_history),
        ("sofascore",            lambda: sofascore.discover_event_id()[1]),
        ("flashscore",           flashscore.fetch_game),
        ("thescore",             thescore.fetch_game),
        ("nba_cdn",              nba_cdn.fetch_playbyplay),
        ("nba_stats_standings",  nba_stats.fetch_standings),
        ("rotowire_lineups",     rotowire.fetch_lineups),
        ("rotowire_injuries",    rotowire.fetch_injuries),
        ("teamrankings_power",   teamrankings.fetch_power_ratings),
        ("teamrankings_ats",     teamrankings.fetch_ats_trends),
        ("teamrankings_ou",      teamrankings.fetch_ou_trends),
        ("nba_stats_v2",         nba_stats_v2.fetch_standings_v2),
        ("action_network",       action_network.fetch_public_betting),
        ("balldontlie",          balldontlie.fetch_team_stats),
        ("odds_scraper",         odds_api_free.fetch_odds),
        ("cbssports_injuries",   cbssports_nba.fetch_injuries),
        ("cbssports_news",       cbssports_nba.fetch_news),
        ("pbpstats",             pbpstats.fetch_team_stats),
    ]:
        try:
            res = fn()
            count = len(getattr(res, "records", []) or [])
            meta_keys = list((getattr(res, "meta", None) or {}).keys())
            print(f"  {name:24} {res.status:8} records={count:<4} "
                  f"meta={meta_keys}  err={res.error or ''}")
        except Exception as e:                       # noqa: BLE001
            print(f"  {name:24} EXC {type(e).__name__}: {e}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="NBA Mood Mirror backend")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("snapshot", help="build pre-game snapshot.json")

    p_live = sub.add_parser("live", help="live mode")
    p_live.add_argument("--once", action="store_true", help="single iteration")
    p_live.add_argument("--fixture", help="path to a synthetic PBP fixture")
    p_live.add_argument("--max-iterations", type=int, default=None,
                        dest="max_iterations")

    p_auto = sub.add_parser("auto", help="snapshot then live if game is on")
    p_auto.add_argument("--max-iterations", type=int, default=None,
                        dest="max_iterations")

    p_eval = sub.add_parser("evaluate", help="score predictions vs outcomes")
    p_eval.add_argument("--results", required=True,
                        help="JSON list of {prob_home, home_won}")

    p_dash = sub.add_parser("dashboard", help="ANSI-coloured CLI dashboard")
    p_dash.add_argument("--tab", default="all",
                        choices=list(cli_dashboard.TABS.keys()),
                        help="which view to show (default: all)")
    p_dash.add_argument("--watch", action="store_true",
                        help="auto-refresh every --interval seconds")
    p_dash.add_argument("--interval", type=int, default=5,
                        help="watch refresh interval in seconds")

    sub.add_parser("tui", help="interactive curses dashboard (TTY required)")

    p_sim = sub.add_parser("simulate", help="run Monte Carlo simulations")
    p_sim.add_argument("--home-win-prob", type=float, default=0.6,
                        dest="home_win_prob",
                        help="seed home win probability (default 0.60)")
    p_sim.add_argument("--trials", type=int, default=8000)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--decimal-odds", type=float, default=None,
                        dest="decimal_odds",
                        help="decimal odds for Kelly + EV calculation")

    sub.add_parser("advanced", help="pretty-print snapshot.advanced")

    p_api = sub.add_parser("api", help="stdlib HTTP API + static dashboard")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--bind", default="127.0.0.1",
                        help="listen address (default 127.0.0.1; use 0.0.0.0 to expose)")

    sub.add_parser("sources", help="dry-run every scraper, show reachability")

    args = parser.parse_args(argv)
    return {"snapshot": _cmd_snapshot, "live": _cmd_live, "auto": _cmd_auto,
            "evaluate": _cmd_evaluate, "dashboard": _cmd_dashboard,
            "tui": _cmd_tui, "simulate": _cmd_simulate,
            "advanced": _cmd_advanced, "api": _cmd_api,
            "sources": _cmd_sources}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
