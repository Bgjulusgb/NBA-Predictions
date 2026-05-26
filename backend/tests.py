"""Unit tests for the sentiment engine, prediction math and import rules.

Run:  python3 -m unittest backend.tests   (or)   python3 -m backend.tests
Pure stdlib, no network — safe to run anywhere.
"""

import os
import tempfile
import unittest

from . import analysis, history, model, sentiment, enrich
from .sources.base import STATUS_OK


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        self.assertGreater(sentiment.score_text("Brunson was absolutely clutch, MVP!")["compound"], 0.3)

    def test_negative(self):
        self.assertLess(sentiment.score_text("The refs robbed us, this is rigged garbage")["compound"], -0.3)

    def test_negation_flips(self):
        pos = sentiment.score_text("this is great")["compound"]
        neg = sentiment.score_text("this is not great")["compound"]
        self.assertGreater(pos, neg)

    def test_toxicity(self):
        s = sentiment.score_text("rigged scam, refball, what a clown fraud")
        self.assertGreater(s["toxicity"], 0.2)

    def test_empty(self):
        self.assertEqual(sentiment.score_text("")["compound"], 0.0)


class TestOdds(unittest.TestCase):
    def test_american_to_decimal(self):
        self.assertAlmostEqual(model.american_to_decimal(100), 2.0)
        self.assertAlmostEqual(model.american_to_decimal(-200), 1.5)
        self.assertAlmostEqual(model.american_to_decimal(110), 2.10, places=2)

    def test_devig_sums_to_one(self):
        m = model.devig_two_way(110, -130)
        self.assertAlmostEqual(m["home"] + m["away"], 1.0, places=6)
        self.assertGreater(m["overround"], 1.0)          # vig present
        self.assertGreater(m["away"], m["home"])          # -130 favorite

    def test_spread_prob(self):
        p = model.prob_from_spread(2.5)
        self.assertTrue(0.5 < p < 0.65)
        self.assertAlmostEqual(model.prob_from_spread(0), 0.5)


class TestElo(unittest.TestCase):
    def test_home_advantage(self):
        # Equal ratings -> home favored by the home-court bump.
        self.assertGreater(model.elo_expected(1500, 1500), 0.5)

    def test_monotonic(self):
        self.assertGreater(model.elo_expected(1700, 1500),
                           model.elo_expected(1550, 1500))

    def test_log5(self):
        self.assertAlmostEqual(model.log5(0.6, 0.6), 0.5, places=6)
        self.assertGreater(model.log5(0.7, 0.5), 0.5)

    def test_form_adjustment(self):
        base = {"home": 1600.0, "away": 1600.0}
        form = {"home": {"wins": 1, "losses": 4}, "away": {"wins": 5, "losses": 0}}
        adj = model.elo_adjust_for_form(base, form)
        self.assertLess(adj["home"], base["home"])     # cold team drops
        self.assertGreater(adj["away"], base["away"])   # hot team rises


class TestLive(unittest.TestCase):
    def test_run_detection(self):
        events = [{"team": "CLE", "points": 2}, {"team": "NYK", "points": 3},
                  {"team": "NYK", "points": 2}, {"team": "NYK", "points": 2}]
        run = model.detect_current_run(events)
        self.assertEqual(run["team"], "NYK")
        self.assertEqual(run["points"], 7)

    def test_momentum_sign(self):
        # All recent scoring by the away team -> negative (home) momentum.
        events = [{"team": "NYK", "points": 2} for _ in range(5)]
        self.assertLess(model.momentum(events), 0)

    def test_sentiment_spike(self):
        buckets = [0.2, 0.1, 0.15, -0.9]
        self.assertLess(model.sentiment_spike(buckets), -1.0)


class TestSeries(unittest.TestCase):
    def test_clinch_from_3_0(self):
        # Leader strongly favored each remaining game -> very high clinch prob.
        p = model.series_clinch([0.6, 0.65, 0.6, 0.65])
        self.assertGreater(p, 0.95)

    def test_clinch_bounds(self):
        self.assertLessEqual(model.series_clinch([0.5, 0.5, 0.5, 0.5]), 1.0)


class TestEnsemble(unittest.TestCase):
    def test_market_weighted(self):
        ens = model.ensemble({"market": 0.7, "elo": 0.5}, 0.0)
        self.assertTrue(0.6 < ens["home"] < 0.7)   # weighted toward market

    def test_three_models(self):
        ens = model.ensemble({"market": 0.6, "elo": 0.5, "espn": 0.55}, 0.0)
        self.assertTrue(0.5 < ens["home"] < 0.6)
        self.assertIn("espn", ens["components"])

    def test_sentiment_bounded(self):
        base = model.ensemble({"market": 0.5, "elo": 0.5}, 0.0)["home"]
        nudged = model.ensemble({"market": 0.5, "elo": 0.5}, model.sentiment_delta(
            {"home": 1.0, "away": -1.0}))["home"]
        self.assertLessEqual(nudged - base, model.SENT_MAX_DELTA + 1e-9)

    def test_confidence_agreement(self):
        agree = model.confidence({"market": 0.6, "elo": 0.6}, 40)
        disagree = model.confidence({"market": 0.6, "elo": 0.2}, 40)
        self.assertGreater(agree, disagree)


class TestLiveWinProb(unittest.TestCase):
    def test_anchors_to_pregame_early(self):
        # Tied at tip-off -> close to the pre-game prior.
        wp = model.live_win_probability(0, 1, "12:00", pregame_home=0.6)
        self.assertTrue(0.5 < wp < 0.7)

    def test_big_late_lead(self):
        wp = model.live_win_probability(18, 4, "1:00", pregame_home=0.5)
        self.assertGreater(wp, 0.95)

    def test_monotonic_in_margin(self):
        a = model.live_win_probability(5, 3, "6:00", 0.5)
        b = model.live_win_probability(-5, 3, "6:00", 0.5)
        self.assertGreater(a, b)

    def test_seconds_remaining(self):
        self.assertAlmostEqual(model.seconds_remaining(1, "12:00"), 2880, delta=1)
        self.assertAlmostEqual(model.seconds_remaining(4, "0:00"), 0, delta=1)


class TestBacktest(unittest.TestCase):
    def test_perfect_predictions(self):
        from . import backtest
        preds = [{"prob_home": 0.99, "home_won": 1},
                 {"prob_home": 0.01, "home_won": 0}]
        m = backtest.evaluate(preds)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertLess(m["brier"], 0.01)
        self.assertGreater(m["skill_score"], 0.9)

    def test_empty(self):
        from . import backtest
        self.assertEqual(backtest.evaluate([])["n"], 0)


class TestImportRule(unittest.TestCase):
    def test_ok_imported_without_refilter(self):
        records = [
            {"title": "Knicks roll past Cavaliers", "text": "great win",
             "url": "http://x/1", "published": "2026-05-25T00:00:00+00:00",
             "kind": "article", "engagement": 0},          # -> ok
            {"title": "", "text": "", "url": "", "kind": "article",
             "engagement": 0},                               # -> error (dropped)
        ]
        imported, stats = enrich.enrich_and_import(records)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["status"], STATUS_OK)
        # The ok record was imported directly, not via repair.
        self.assertNotIn("imported_via", imported[0])

    def test_partial_repaired(self):
        records = [{"title": "NBA playoff roundup", "text": "neutral notes",
                    "url": "http://x/2", "kind": "article", "engagement": 0}]
        imported, stats = enrich.enrich_and_import(records)
        self.assertEqual(stats["repaired"], 1)
        self.assertEqual(imported[0]["imported_via"], "repair")


class TestEmotions(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        e = sentiment.emotions("clutch win, but the injury is devastating tonight")
        self.assertAlmostEqual(sum(e.values()), 1.0, places=4)
        self.assertGreater(e["joy"] + e["sadness"] + e["anticipation"], 0)

    def test_empty_is_zero(self):
        self.assertEqual(sum(sentiment.emotions("the a of to").values()), 0.0)


class TestSentimentAdvanced(unittest.TestCase):
    def test_phrase(self):
        self.assertGreater(sentiment.score_text("what a buzzer beater")["compound"], 0.3)

    def test_emoji(self):
        self.assertGreater(sentiment.score_text("Brunson is on fire 🔥🐐")["compound"], 0.3)
        self.assertLess(sentiment.score_text("the refs 🤡🤡 robbed us")["compound"], 0)

    def test_contrast_but(self):
        s = sentiment.score_text("they played well but it was a terrible collapse")
        self.assertLess(s["compound"], 0)


class TestAnalysis(unittest.TestCase):
    def _recs(self):
        return [
            {"title": "Jalen Brunson clutch as Knicks win", "text": "MVP chants",
             "engagement": 50, "sentiment": sentiment.score_text(
                 "Jalen Brunson clutch as Knicks win MVP chants")},
            {"title": "Donovan Mitchell struggles, injury concern", "text": "",
             "engagement": 10, "sentiment": sentiment.score_text(
                 "Donovan Mitchell struggles, injury concern")},
            {"title": "Mitchell Robinson returns for the Knicks", "text": "",
             "engagement": 5, "sentiment": sentiment.score_text(
                 "Mitchell Robinson returns for the Knicks")},
        ]

    def test_player_sentiment_attribution(self):
        players = {p["name"]: p for p in analysis.player_sentiment(self._recs())}
        self.assertIn("Jalen Brunson", players)
        self.assertGreater(players["Jalen Brunson"]["mean_sentiment"], 0)
        # "Mitchell Robinson" must not be double-counted as Donovan Mitchell.
        self.assertIn("Mitchell Robinson", players)

    def test_narratives(self):
        terms = {n["term"]: n for n in analysis.narratives(self._recs())}
        self.assertIn("injury", terms)
        self.assertIn("clutch", terms)

    def test_value_bet_edge(self):
        market = {"home": 0.45, "away": 0.55}
        ens = {"home": 0.52, "away": 0.48}
        vb = analysis.value_bet(market, ens, {"home_moneyline": 120,
                                              "away_moneyline": -140})
        self.assertEqual(vb["side"], "home")
        self.assertTrue(vb["has_value"])
        self.assertAlmostEqual(vb["edge_pct"], 7.0, places=1)

    def test_value_bet_none_without_market(self):
        self.assertIsNone(analysis.value_bet(None, {"home": 0.5, "away": 0.5}, {}))


class TestAttribution(unittest.TestCase):
    def test_word_boundary_no_false_match(self):
        # "ny" must not match inside "company"; this is general, not the Knicks.
        self.assertEqual(enrich._attribute_team("the company announced", None), None)

    def test_team_match(self):
        self.assertEqual(enrich._attribute_team("Knicks roll on", None), "away")
        self.assertEqual(enrich._attribute_team("Cavaliers respond", None), "home")
        self.assertEqual(enrich._attribute_team("Cavaliers host the Knicks", None), "both")

    def test_recency_weight(self):
        import datetime as dt
        now = dt.datetime(2026, 5, 25, tzinfo=dt.timezone.utc)
        fresh = model.recency_weight("2026-05-25T00:00:00+00:00", now=now)
        old = model.recency_weight("2026-05-19T00:00:00+00:00", now=now)
        self.assertGreater(fresh, old)
        self.assertAlmostEqual(model.recency_weight(None), 0.5)


class TestHistory(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        snap = {"generated_at": "2026-05-25T00:00:00+00:00", "mode": "pre",
                "prediction": {"ensemble": {"home": 0.5, "away": 0.5},
                               "market": {"home": 0.48}, "elo": {"home": 0.6},
                               "confidence": 50,
                               "series": {"leader_clinch_probability": 0.96}},
                "mood": {"overall": {"heat": 90, "hype": 80, "toxicity": 5},
                         "team_sentiment": {"home": 0.1, "away": 0.2}}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.jsonl")
            history.append(snap, path=path)
            history.append(snap, path=path)
            rows = history.load_recent(path=path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ens_home"], 0.5)
            self.assertEqual(rows[0]["clinch"], 0.96)


if __name__ == "__main__":
    unittest.main()
