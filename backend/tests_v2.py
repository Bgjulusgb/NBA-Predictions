"""Tests for the v2 modules: categorizer, aggregator, lineup_analyzer,
momentum_composite, api_server routing, and scraper parsers.

Run:  python3 -m unittest backend.tests_v2
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request

from . import (aggregator, api_server, categorizer, config, lineup_analyzer,
               momentum_composite)
from .sources import flashscore, sofascore, thescore


# ===========================================================================
class TestCategorizer(unittest.TestCase):
    def test_injury_text(self):
        self.assertEqual(categorizer.categorize_text(
            "Donovan Mitchell injured ankle, questionable for Game 4"), "injury")

    def test_trade_text(self):
        self.assertEqual(categorizer.categorize_text(
            "Cavaliers extension talks with Mitchell"), "trade")

    def test_betting_text(self):
        self.assertEqual(categorizer.categorize_text(
            "Knicks moneyline shifted to -150, sharps coming in"), "betting")

    def test_recap_text(self):
        self.assertEqual(categorizer.categorize_text(
            "Knicks beat Cavaliers 110-104 in Game 3 recap"), "recap")

    def test_preview_text(self):
        self.assertEqual(categorizer.categorize_text(
            "Knicks vs Cavaliers Game 4 preview: how to watch"), "preview")

    def test_general_fallback(self):
        self.assertEqual(categorizer.categorize_text("hello world"), "general")

    def test_categorize_records_assigns_field(self):
        recs = [{"title": "Brunson injured", "text": ""}]
        out = categorizer.categorize_records(recs)
        self.assertEqual(out[0]["category"], "injury")

    def test_breakdown_summarises(self):
        recs = [
            {"title": "Mitchell injured", "text": "", "sentiment": {"compound": -0.5}},
            {"title": "Mitchell injured ankle", "text": "", "sentiment": {"compound": -0.6}},
            {"title": "Knicks moneyline", "text": "", "sentiment": {"compound": 0.4}},
            {"title": "Random NBA chatter", "text": "", "sentiment": {"compound": 0.0}},
        ]
        categorizer.categorize_records(recs)
        b = categorizer.category_breakdown(recs)
        cats = {c["category"]: c for c in b["categories"]}
        self.assertGreaterEqual(cats.get("injury", {}).get("count", 0), 1)
        self.assertGreaterEqual(cats.get("betting", {}).get("count", 0), 1)
        self.assertEqual(b["total"], 4)

    def test_filter(self):
        recs = [{"title": "Brunson injured", "text": "", "category": "injury"},
                 {"title": "Knicks fans hyped", "text": "", "category": "general"}]
        only = categorizer.filter_by_category(recs, "injury")
        self.assertEqual(len(only), 1)


# ===========================================================================
class TestAggregator(unittest.TestCase):
    def test_merge_scores_consensus(self):
        per = {
            "espn": {"home": {"score": 100}, "away": {"score": 98}},
            "sofa": {"home": {"score": 100}, "away": {"score": 98}},
            "flash": {"home": {"score": 102}, "away": {"score": 96}},
        }
        out = aggregator.merge_scores(per)
        self.assertEqual(out["consensus_home"], 100)
        self.assertEqual(out["consensus_away"], 98)
        self.assertEqual(out["max_disagreement"], 2)
        self.assertEqual(out["n_sources"], 3)

    def test_merge_scores_handles_missing(self):
        out = aggregator.merge_scores({"espn": None})
        self.assertIsNone(out["consensus_home"])

    def test_merge_odds_books(self):
        espn = {"home_moneyline": -130, "away_moneyline": 110, "provider": "ESPN"}
        sofa = {"moneyline": [{"provider": "DraftKings",
                                 "home_decimal": 1.78, "away_decimal": 2.10}]}
        out = aggregator.merge_odds_books(espn, sofa)
        self.assertEqual(out["count"], 2)
        names = {b["provider"] for b in out["books"]}
        self.assertIn("ESPN", names)
        self.assertIn("DraftKings", names)

    def test_merge_pbp_dedupes(self):
        a = [{"period": 1, "clock": "11:30", "team": "NYK", "points": 2, "desc": "Brunson short"}]
        b = [{"period": 1, "clock": "11:30", "team": "NYK", "points": 2, "desc": "Brunson 2pt makes long form"}]
        out = aggregator.merge_pbp(a, b)
        self.assertEqual(len(out), 1)
        # Longer description wins.
        self.assertIn("long form", out[0]["desc"])

    def test_source_health(self):
        sources = [{"name": "a", "status": "ok"},
                   {"name": "b", "status": "partial"},
                   {"name": "c", "status": "error"}]
        h = aggregator.source_health(sources)
        self.assertEqual(h["counts"], {"ok": 1, "partial": 1, "error": 1})
        self.assertAlmostEqual(h["uptime_pct"], 33.3, delta=0.1)


# ===========================================================================
class TestLineupAnalyzer(unittest.TestCase):
    def setUp(self):
        self.side = {
            "starters": [
                {"name": "Jalen Brunson", "position": "PG"},
                {"name": "Mikal Bridges", "position": "SG"},
                {"name": "OG Anunoby", "position": "SF"},
                {"name": "Karl-Anthony Towns", "position": "PF"},
                {"name": "Mitchell Robinson", "position": "C"},
            ],
            "bench": [{"name": "Josh Hart", "position": "G"},
                       {"name": "Miles McBride", "position": "G"}],
            "missing": [{"name": "OG Anunoby", "type": "doubtful",
                         "reason": "wrist injury"}],
            "formation": "PG-SG-SF-PF-C",
        }

    def test_total_starting_value(self):
        v = lineup_analyzer.total_starting_value(self.side)
        self.assertGreater(v, 0.5)

    def test_missing_impact(self):
        out = lineup_analyzer.missing_players_impact(self.side)
        self.assertGreater(out["shares_lost"], 0)
        self.assertEqual(len(out["players"]), 1)

    def test_position_breakdown(self):
        out = lineup_analyzer.position_breakdown(self.side)
        self.assertIn("PG", out)
        self.assertIn("C", out)

    def test_starting_advantage(self):
        out = lineup_analyzer.starting_advantage(self.side, self.side)
        self.assertEqual(out["net_advantage_home"], 0)


# ===========================================================================
class TestMomentumComposite(unittest.TestCase):
    def test_composite_with_all_signals(self):
        out = momentum_composite.composite(
            scoring_momentum=0.5,
            current_run={"team": "NYK", "points": 8},
            home_abbr="CLE",                  # NYK run = negative for home
            sentiment_zscore=1.0,
            starting_plus_minus_diff=5.0,
            pace_ratio=1.05,
        )
        self.assertIn("value", out)
        self.assertGreater(out["weight_total"], 0.9)
        self.assertEqual(set(out["components"].keys()),
                         {"scoring", "run", "sentiment", "lineup", "pace"})

    def test_composite_partial_signals(self):
        # Only scoring + sentiment -> weights re-normalise to those two.
        out = momentum_composite.composite(scoring_momentum=0.5,
                                            sentiment_zscore=1.0)
        self.assertAlmostEqual(out["weight_total"], 0.55, places=5)
        self.assertGreater(out["value"], 0)

    def test_composite_no_signals(self):
        out = momentum_composite.composite()
        self.assertEqual(out["value"], 0.0)
        self.assertEqual(out["weight_total"], 0.0)


# ===========================================================================
class TestSofascoreParsing(unittest.TestCase):
    def test_period_scores_extracts(self):
        scores = sofascore._period_scores({"period1": 28, "period2": 30,
                                            "period3": 27, "period4": 25})
        self.assertEqual(scores, [28, 30, 27, 25])

    def test_period_scores_handles_partial(self):
        scores = sofascore._period_scores({"period1": 30, "period2": 28})
        self.assertEqual(scores, [30, 28])

    def test_clock_from_status(self):
        self.assertEqual(sofascore._clock_from_status({"description": "Q3 7:24"}),
                         "7:24")
        self.assertIsNone(sofascore._clock_from_status({"description": "Halftime"}))

    def test_infer_points_three(self):
        self.assertEqual(sofascore._infer_points(
            {"incidentClass": "threepointer"}), 3)

    def test_infer_points_text(self):
        self.assertEqual(sofascore._infer_points(
            {"text": "Made 3pt jumper"}), 3)

    def test_to_decimal_fractional(self):
        self.assertAlmostEqual(sofascore._to_decimal("11/10"), 2.1, places=3)

    def test_to_decimal_decimal(self):
        self.assertAlmostEqual(sofascore._to_decimal("1.91"), 1.91, places=3)

    def test_matches_target(self):
        ev = {"homeTeam": {"name": "Cleveland Cavaliers"},
              "awayTeam": {"name": "New York Knicks"}}
        self.assertTrue(sofascore._matches_target(
            ev, config.GAME["home"], config.GAME["away"]))


# ===========================================================================
class TestFlashscoreParsing(unittest.TestCase):
    def test_split_blocks(self):
        # Two minimal blocks in Flashscore's pipe-delimited format.
        payload = (
            "¬~AA÷abc¬AE÷Cavs¬AF÷Knicks¬AG÷98¬AH÷101¬AS÷3¬AT÷Final"
            "¬~AA÷def¬AE÷Lakers¬AF÷Suns¬AG÷108¬AH÷102¬AS÷2¬AT÷Q4"
        )
        blocks = flashscore._split_blocks(payload)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["home_score"], "98")
        self.assertEqual(blocks[0]["status_id"], "3")


# ===========================================================================
class TestApiServerRouting(unittest.TestCase):
    """Spin up the api_server in a background thread and hit a few endpoints.

    Uses the snapshot.json currently on disk; if it's missing we still expect
    /api/health to come back ok=true.
    """

    @classmethod
    def setUpClass(cls):
        # Find a free port without binding twice.
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        cls.port = s.getsockname()[1]
        s.close()
        cls.server_thread = threading.Thread(
            target=api_server.serve, kwargs={"port": cls.port, "bind": "127.0.0.1"},
            daemon=True)
        cls.server_thread.start()
        # Wait briefly for the listen() call to complete.
        time.sleep(0.4)

    def _get(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=4) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_health(self):
        status, body = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_simulate(self):
        status, body = self._get("/api/simulate?p=0.55&trials=200")
        self.assertEqual(status, 200)
        self.assertIn("home_win_prob_sim", body)
        self.assertIn("alt_lines", body)

    def test_unknown(self):
        # urllib raises on non-2xx; verify the 404 lands either way.
        import urllib.error
        try:
            status, _ = self._get("/api/no_such_endpoint")
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
