"""Tests for the new math, simulation, streak, odds, player and injury modules.

Run:  python3 -m unittest backend.tests_advanced
or:   python3 -m unittest discover backend     (picks up both test files)
"""

import math
import unittest

from . import (advanced_math, clutch_analysis, injury_impact, odds_compare,
               player_stats, simulation, streaks)


# ===========================================================================
class TestAdvancedMath(unittest.TestCase):
    def setUp(self):
        advanced_math.seed(123)

    def test_beta_update_and_mean(self):
        a, b = advanced_math.beta_update(2, 2, 3, 1)
        self.assertEqual((a, b), (5, 3))
        mean, var = advanced_math.beta_mean_var(a, b)
        self.assertAlmostEqual(mean, 5 / 8)
        self.assertGreater(var, 0)

    def test_beta_credible_interval_brackets_mean(self):
        lo, hi = advanced_math.beta_credible_interval(5, 3, level=0.95, n=2000)
        self.assertLess(lo, 5 / 8)
        self.assertGreater(hi, 5 / 8)

    def test_poisson_pmf_sums_to_1(self):
        total = sum(advanced_math.poisson_pmf(k, 2.0) for k in range(30))
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_skellam_symmetry(self):
        # Equal lambdas -> P(X-Y>0) == P(X-Y<0); win prob ~= 0.5.
        self.assertAlmostEqual(
            advanced_math.skellam_win_prob(1.0, 1.0), 0.5, places=2)

    def test_linear_regression_perfect_line(self):
        out = advanced_math.linear_regression([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(out["slope"], 2.0)
        self.assertAlmostEqual(out["intercept"], 1.0)
        self.assertAlmostEqual(out["r2"], 1.0)

    def test_logistic_regression_separable(self):
        rows = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        ys = [0, 0, 0, 0, 1, 1, 1]
        out = advanced_math.logistic_regression(rows, ys, epochs=400, lr=0.1)
        p_low = advanced_math.logistic_predict(out["weights"], out["bias"], [0, 0])
        p_high = advanced_math.logistic_predict(out["weights"], out["bias"], [4, 4])
        self.assertLess(p_low, 0.5)
        self.assertGreater(p_high, 0.5)

    def test_sma_and_ewma(self):
        s = advanced_math.sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(s, [2.0, 3.0, 4.0])
        e = advanced_math.ewma([1, 1, 1], 0.5)
        self.assertEqual(e, [1, 1, 1])

    def test_autocorrelation_constant_series(self):
        # All-equal series -> denominator zero -> 0.0 by convention.
        self.assertEqual(advanced_math.autocorrelation([2, 2, 2, 2], 1), 0.0)

    def test_autocorrelation_alternating(self):
        ac = advanced_math.autocorrelation([1, -1, 1, -1, 1, -1], 1)
        self.assertLess(ac, 0)         # alternating -> negative AR(1)

    def test_percentile_iqr_skew_kurtosis(self):
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertAlmostEqual(advanced_math.percentile(v, 50), 5.5)
        self.assertGreater(advanced_math.iqr(v), 0)
        # Symmetric -> ~0 skew and negative excess kurtosis for uniform.
        self.assertAlmostEqual(advanced_math.skewness(v), 0.0, delta=0.1)
        self.assertLess(advanced_math.kurtosis(v), 0)

    def test_z_score(self):
        self.assertGreater(advanced_math.z_score(10, [1, 2, 3, 4, 5]), 2)

    def test_pearson_perfect(self):
        self.assertAlmostEqual(
            advanced_math.pearson_correlation([1, 2, 3], [10, 20, 30]), 1.0)

    def test_spearman_handles_ties(self):
        # Ties -> average rank, no error.
        self.assertEqual(
            advanced_math.spearman_correlation([1, 1, 2, 2], [1, 1, 2, 2]), 1.0)

    def test_glicko2_higher_rated_more_likely(self):
        p = advanced_math.glicko2_win_prob(1700, 50, 1500, 50)
        self.assertGreater(p, 0.7)

    def test_glicko2_inactivity_grows_rd(self):
        r, rd, vol = advanced_math.glicko2_update(1500, 200, 0.06, [])
        self.assertGreaterEqual(rd, 200)

    def test_kelly_no_bet_when_no_edge(self):
        self.assertEqual(advanced_math.kelly_fraction(0.5, 1.9), 0)

    def test_kelly_positive_with_edge(self):
        self.assertGreater(advanced_math.kelly_fraction(0.6, 2.0), 0)

    def test_fractional_kelly_smaller(self):
        f = advanced_math.kelly_fraction(0.6, 2.0)
        self.assertLess(advanced_math.fractional_kelly(0.6, 2.0, 0.25), f)

    def test_sharpe_zero_for_constant(self):
        self.assertEqual(advanced_math.sharpe_ratio([0.01] * 5), 0)

    def test_max_drawdown(self):
        # peak=110 at idx 1, trough=80 at idx 2 -> ~27%.
        dd = advanced_math.max_drawdown([100, 110, 80, 95, 105])
        self.assertAlmostEqual(dd, 30 / 110, places=4)

    def test_pythagorean_expectation_bounds(self):
        self.assertAlmostEqual(advanced_math.pythagorean_expectation(100, 100), 0.5)
        self.assertGreater(advanced_math.pythagorean_expectation(120, 100), 0.5)

    def test_four_factors_shape(self):
        ff = advanced_math.four_factors({
            "fgm": 40, "fga": 90, "fg3m": 12, "fta": 20, "tov": 14,
            "orb": 10, "opp_drb": 30, "ft": 17,
        })
        self.assertIn("efg", ff)
        self.assertGreater(ff["efg"], 0)
        self.assertLessEqual(ff["tov_rate"], 1)

    def test_bootstrap_ci_brackets_mean(self):
        ci = advanced_math.bootstrap_mean_ci([1.0, 1.1, 0.9, 1.2, 1.05],
                                              trials=400)
        self.assertLessEqual(ci["low"], ci["mean"])
        self.assertGreaterEqual(ci["high"], ci["mean"])

    def test_shannon_entropy_uniform_is_log2_n(self):
        self.assertAlmostEqual(advanced_math.shannon_entropy([0.25] * 4), 2.0)

    def test_kl_zero_when_equal(self):
        self.assertAlmostEqual(
            advanced_math.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)

    def test_jensen_shannon_symmetric(self):
        p = [0.7, 0.3]
        q = [0.4, 0.6]
        self.assertAlmostEqual(
            advanced_math.jensen_shannon(p, q),
            advanced_math.jensen_shannon(q, p),
            places=8,
        )

    def test_stationary_distribution(self):
        # Symmetric chain -> uniform stationary distribution.
        T = [[0.5, 0.5], [0.5, 0.5]]
        pi = advanced_math.stationary_distribution(T)
        self.assertAlmostEqual(pi[0], 0.5, places=3)

    def test_monte_carlo_game_win_prob(self):
        out = advanced_math.monte_carlo_game(4.0, 12.0, trials=4000)
        self.assertTrue(0.55 < out["home_win_prob"] < 0.7)

    def test_monte_carlo_series_clinches_high(self):
        out = advanced_math.monte_carlo_series([0.8, 0.8, 0.8, 0.8],
                                                3, 0, trials=2000)
        self.assertGreater(out["leader_clinch_prob"], 0.99)


# ===========================================================================
class TestSimulation(unittest.TestCase):
    def setUp(self):
        simulation.seed(7)

    def test_simulate_game_consistent_with_input_wp(self):
        out = simulation.simulate_game(0.6, trials=4000)
        self.assertTrue(0.55 < out["home_win_prob_sim"] < 0.65)
        self.assertTrue(out["p10_margin"] < out["median_margin"] < out["p90_margin"])

    def test_simulate_alt_lines_monotonic(self):
        out = simulation.simulate_alt_lines(0.6, trials=2000)
        sp = out["spread_cover_prob"]
        keys = sorted(sp.keys())
        for a, b in zip(keys[:-1], keys[1:]):
            self.assertGreaterEqual(sp[a], sp[b])     # tighter spread -> harder

    def test_simulate_possession_centred(self):
        out = simulation.simulate_possession(possessions=100, trials=300)
        # Equal sides -> win prob near 0.5.
        self.assertTrue(0.35 < out["home_win_prob"] < 0.65)

    def test_what_if_player_out_negative_delta(self):
        out = simulation.what_if_player_out(0.6, lost_share=0.2)
        self.assertLess(out["delta_win_prob"], 0)


# ===========================================================================
class TestStreaks(unittest.TestCase):
    def test_win_streak(self):
        out = streaks.win_loss_streak(["W", "W", "L", "W", "W", "W"])
        self.assertEqual(out["current_streak"], 3)
        self.assertEqual(out["longest_win"], 3)
        self.assertEqual(out["longest_loss"], 1)

    def test_scoring_streak(self):
        events = [{"team": "A", "points": 2},
                  {"team": "A", "points": 3},
                  {"team": "B", "points": 2},
                  {"team": "B", "points": 2},
                  {"team": "B", "points": 3}]
        out = streaks.scoring_streak(events)
        self.assertEqual(out["team"], "B")
        self.assertEqual(out["points"], 7)

    def test_lead_changes(self):
        # home, away, home, away => 3 lead changes (first lead doesn't count).
        tl = [{"home": 2, "away": 0}, {"home": 2, "away": 3},
              {"home": 5, "away": 3}, {"home": 5, "away": 7}]
        out = streaks.lead_change_count(tl)
        self.assertEqual(out["lead_changes"], 3)

    def test_sentiment_streak(self):
        out = streaks.sentiment_streak([0.4, 0.5, 0.6, 0.0, -0.5, -0.6])
        self.assertEqual(out["longest_positive"], 3)
        self.assertEqual(out["longest_negative"], 2)

    def test_comeback_distance(self):
        tl = [{"home": 0, "away": 12}, {"home": 10, "away": 12},
              {"home": 20, "away": 14}, {"home": 25, "away": 18}]
        out = streaks.comeback_distance(tl)
        self.assertEqual(out["home_max_deficit"], 12)
        self.assertTrue(out["home_came_back"])

    def test_biggest_lead(self):
        tl = [{"home": 0, "away": 0}, {"home": 15, "away": 0},
              {"home": 15, "away": 8}, {"home": 18, "away": 25}]
        out = streaks.biggest_lead(tl)
        self.assertEqual(out["home_biggest_lead"], 15)
        self.assertEqual(out["away_biggest_lead"], 7)


# ===========================================================================
class TestOddsCompare(unittest.TestCase):
    def test_best_price_picks_highest_decimal(self):
        books = [{"provider": "A", "home_moneyline": 120, "away_moneyline": -140},
                 {"provider": "B", "home_moneyline": 130, "away_moneyline": -150}]
        best = odds_compare.best_price(books)
        self.assertEqual(best["home"]["provider"], "B")
        self.assertEqual(best["away"]["provider"], "A")

    def test_arbitrage_detection_when_exists(self):
        # Make books where 1/dec_h + 1/dec_a < 1.
        # +200 -> 3.0 decimal; +110 -> 2.1 decimal -> 1/3 + 1/2.1 ≈ 0.81 < 1.
        best = {"home": {"decimal": 3.0}, "away": {"decimal": 2.1}}
        arb = odds_compare.arbitrage(best, stake=100)
        self.assertIsNotNone(arb)
        self.assertGreater(arb["profit"], 0)

    def test_arbitrage_none_when_no_edge(self):
        # Both at -110 -> ~0.524 each -> sum > 1, no arb.
        best = {"home": {"decimal": 1.91}, "away": {"decimal": 1.91}}
        self.assertIsNone(odds_compare.arbitrage(best))

    def test_no_vig_consensus(self):
        books = [{"home_moneyline": 110, "away_moneyline": -130},
                 {"home_moneyline": 105, "away_moneyline": -125}]
        out = odds_compare.no_vig_consensus(books)
        self.assertEqual(out["n_books"], 2)
        self.assertAlmostEqual(out["home"] + out["away"], 1.0, places=4)

    def test_sharp_movement_directions(self):
        out = odds_compare.sharp_movement(
            {"home_moneyline": 100, "away_moneyline": -120},
            {"home_moneyline": 120, "away_moneyline": -140})
        # Home odds got longer, implied prob fell.
        self.assertLess(out["home_drift_pct"], 0)


# ===========================================================================
class TestPlayerStats(unittest.TestCase):
    def setUp(self):
        self.player = {
            "pts": 30, "fgm": 11, "fga": 22, "fg3m": 4, "fta": 5, "ftm": 4,
            "ast": 6, "stl": 2, "blk": 1, "tov": 3, "orb": 1, "drb": 5,
            "pf": 2, "mp": 36,
        }
        self.team = {
            "fgm": 40, "fga": 90, "fta": 20, "tov": 14, "mp": 240,
        }

    def test_efg_value_in_range(self):
        v = player_stats.effective_fg_pct(self.player)
        self.assertTrue(0.4 < v < 0.7)

    def test_true_shooting_above_efg(self):
        ts = player_stats.true_shooting_pct(self.player)
        efg = player_stats.effective_fg_pct(self.player)
        self.assertGreater(ts, efg)

    def test_usage_rate_in_range(self):
        u = player_stats.usage_rate(self.player, self.team)
        self.assertTrue(15 < u < 40)

    def test_game_score_positive_for_strong_line(self):
        self.assertGreater(player_stats.game_score(self.player), 15)

    def test_player_efficiency_returns_number(self):
        v = player_stats.player_efficiency(self.player)
        self.assertIsInstance(v, float)

    def test_project_player_line(self):
        history = [self.player, self.player, self.player]
        out = player_stats.project_player_line(history)
        self.assertIn("pts", out)
        self.assertAlmostEqual(out["pts"], 30, delta=1)


# ===========================================================================
class TestInjuryImpact(unittest.TestCase):
    def test_detection_flags_target_player(self):
        recs = [
            {"title": "Jalen Brunson injured ankle", "text": ""},
            {"title": "Brunson out for game 4", "text": ""},
        ]
        hits = injury_impact.detect_injury_signals(recs)
        self.assertIn("Jalen Brunson", hits)
        self.assertGreaterEqual(hits["Jalen Brunson"]["injury_hits"], 2)

    def test_team_impact_capped(self):
        recs = [{"title": "Brunson injured", "text": ""}] * 100
        impact = injury_impact.estimated_team_impact(recs, threshold=1)
        self.assertLessEqual(impact["away"], 0.35 + 1e-9)

    def test_adjust_lowers_when_home_more_hurt(self):
        out = injury_impact.adjust_win_probability(
            0.6, {"home": 0.2, "away": 0.0})
        self.assertLess(out["adjusted_prob"], 0.6)


# ===========================================================================
class TestClutchAnalysis(unittest.TestCase):
    def test_clutch_split_basic(self):
        events = [
            {"team": "CLE", "points": 2, "period": 4, "clock": "4:30",
             "score_home": 95, "score_away": 92},
            {"team": "NY", "points": 3, "period": 4, "clock": "3:30",
             "score_home": 95, "score_away": 95},
            {"team": "CLE", "points": 2, "period": 1, "clock": "10:00",
             "score_home": 5, "score_away": 3},
        ]
        # Q1 won't be clutch; Q4 within 5 will.
        out = clutch_analysis.clutch_split(events)
        self.assertGreater(
            out["clutch"]["home_pts"] + out["clutch"]["away_pts"], 0)
        self.assertGreater(
            out["non_clutch"]["home_pts"], 0)


# ===========================================================================
class TestPipelineAdvancedBlock(unittest.TestCase):
    def test_compute_advanced_smoke(self):
        # Smoke-test the private builder used in pipeline.py.
        from . import pipeline
        ens = {"home": 0.55, "away": 0.45}
        ratings = {"home": 1600, "away": 1612}
        per_game = [{"game": g, "venue": "CLE", "leader_win": 0.6}
                    for g in range(4, 8)]
        block = pipeline._compute_advanced(
            ens=ens, ratings=ratings, per_game=per_game,
            odds={"home_moneyline": 110, "away_moneyline": -130},
            value={"side": "home", "model_prob": 0.55, "moneyline": 110},
            history_rows=[],
            imported=[],
        )
        for key in ("monte_carlo_game", "monte_carlo_series", "glicko",
                    "kelly", "injuries", "history_movement"):
            self.assertIn(key, block)


if __name__ == "__main__":
    unittest.main()
