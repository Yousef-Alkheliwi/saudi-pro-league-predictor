"""Tests. Run with: .venv/bin/python -m unittest discover -s tests -v"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

from spl import aux_models as AUX
from spl import features as F
from spl import model as M
from spl.data import Dataset, Injury, Match, Player, resolve_team_name
from spl.predict import Predictor, render, render_compact
from tests.synthetic import make_dataset

DS, TRUTH = make_dataset()
RATINGS = M.fit(DS.matches)


class TestNameResolution(unittest.TestCase):
    #: the real 2025-26 Saudi Pro League field
    teams = {1: "Al-Hilal Saudi FC", 2: "Al-Nassr", 3: "Al-Ittihad FC",
             4: "Al-Ahli Jeddah", 5: "Al-Ittifaq", 6: "Al-Taawoun",
             7: "Al-Shabab", 8: "Al-Khaleej", 9: "Al-Fateh", 10: "Al-Riyadh",
             11: "Al-Qadsiah", 12: "Al-Okhdood", 13: "Damac", 14: "Al-Wehda",
             15: "Al-Fayha", 16: "Al-Hazem", 17: "Al-Najma", 18: "Neom SC"}

    def test_every_club_resolves_from_its_short_name(self):
        short = {"hilal": 1, "nassr": 2, "ittihad": 3, "ahli": 4, "ittifaq": 5,
                 "taawoun": 6, "shabab": 7, "khaleej": 8, "fateh": 9,
                 "riyadh": 10, "qadsiah": 11, "okhdood": 12, "damac": 13,
                 "wehda": 14, "fayha": 15, "hazem": 16, "najma": 17, "neom": 18}
        for query, expected in short.items():
            self.assertEqual(resolve_team_name(query, self.teams)[0], expected, query)

    def test_every_club_resolves_from_its_full_name(self):
        for tid, name in self.teams.items():
            self.assertEqual(resolve_team_name(name, self.teams)[0], tid, name)

    def test_a_city_name_is_not_treated_as_noise(self):
        """Al-Riyadh once normalised to the empty string, and "" is a substring of
        every name, so it silently hijacked every lookup."""
        from spl.data import normalise_name
        self.assertTrue(normalise_name("Al-Riyadh"))
        self.assertEqual(resolve_team_name("Al-Riyadh", self.teams)[0], 10)
        self.assertEqual(resolve_team_name("nassr", self.teams)[0], 2)
        self.assertEqual(resolve_team_name("hilal", self.teams)[0], 1)

    def test_spacing_and_prefix_variants(self):
        for query in ("ALNASSR", "al  nassr", "Al Nassr", "al-nassr", "AlNassr",
                      "  Al-Nassr  "):
            self.assertEqual(resolve_team_name(query, self.teams)[0], 2, query)

    def test_exact_and_partial(self):
        self.assertEqual(resolve_team_name("Al-Nassr", self.teams)[0], 2)
        self.assertEqual(resolve_team_name("nassr", self.teams)[0], 2)
        self.assertEqual(resolve_team_name("hilal", self.teams)[0], 1)
        self.assertEqual(resolve_team_name("AL HILAL", self.teams)[0], 1)

    def test_similar_names_are_distinguished(self):
        self.assertEqual(resolve_team_name("ittihad", self.teams)[0], 3)
        self.assertEqual(resolve_team_name("ittifaq", self.teams)[0], 5)
        self.assertEqual(resolve_team_name("ahli", self.teams)[0], 4)

    def test_unknown_team_raises(self):
        with self.assertRaises(LookupError):
            resolve_team_name("Manchester United", self.teams)


class TestSchedule(unittest.TestCase):
    def setUp(self):
        base = datetime(2025, 3, 1, 18, 0, tzinfo=timezone.utc)
        self.matches = [
            Match(1, 2025, base.isoformat(), 10, 20, "A", "B", "FT", 1, 0),
            Match(2, 2025, (base + timedelta(days=4)).isoformat(), 30, 10, "C", "A",
                  "FT", 2, 2),
            Match(3, 2025, (base + timedelta(days=11)).isoformat(), 10, 40, "A", "D",
                  "FT", 0, 1),
        ]
        self.kickoff = base + timedelta(days=14)

    def test_rest_days(self):
        self.assertAlmostEqual(F.rest_days(self.matches, 10, self.kickoff), 3.0)
        self.assertAlmostEqual(F.rest_days(self.matches, 20, self.kickoff), 10.0)  # capped

    def test_rest_days_unknown_team(self):
        self.assertIsNone(F.rest_days(self.matches, 999, self.kickoff))

    def test_congestion_counts_only_the_window(self):
        self.assertEqual(F.congestion(self.matches, 10, self.kickoff, 14), 3)
        self.assertEqual(F.congestion(self.matches, 10, self.kickoff, 5), 1)

    def test_future_matches_are_never_used(self):
        early = datetime(2025, 3, 2, tzinfo=timezone.utc)
        self.assertEqual(F.congestion(self.matches, 10, early, 14), 1)


class TestAvailability(unittest.TestCase):
    def _players(self, team=1):
        return [Player(player_id=team * 100 + i, team_id=team, name="P%d" % i,
                       position="Attacker" if i > 16 else "Defender",
                       minutes=2400.0 - 95.0 * i) for i in range(25)]

    def test_full_squad_is_full_strength(self):
        av = F.availability(1, [], self._players())
        self.assertEqual(av.overall_available, 1.0)

    def test_starter_costs_more_than_a_fringe_player(self):
        players = self._players()
        starter = Injury(1, 101, "P1", "Missing Fixture", "Injury")
        fringe = Injury(1, 124, "P24", "Missing Fixture", "Injury")
        av_starter = F.availability(1, [starter], players)
        av_fringe = F.availability(1, [fringe], players)
        self.assertLess(av_starter.overall_available, av_fringe.overall_available)

    def test_doubtful_counts_less_than_out(self):
        players = self._players()
        out = F.availability(1, [Injury(1, 101, "P1", "Missing Fixture", "x")], players)
        doubt = F.availability(1, [Injury(1, 101, "P1", "Questionable", "x")], players)
        self.assertLess(out.overall_available, doubt.overall_available)
        self.assertEqual(len(doubt.doubtful), 1)
        self.assertEqual(len(out.out), 1)

    def test_attacker_hits_attack_more_than_defence(self):
        players = self._players()
        av = F.availability(1, [Injury(1, 118, "P18", "Missing Fixture", "x")], players)
        self.assertLess(av.attack_available, av.defence_available)

    def test_one_ever_present_starter_costs_about_a_ninth(self):
        """A minutes share is already the fraction of the eleven, so losing one
        ever-present starter must cost roughly 1/11 - not the whole team."""
        team_minutes = 38 * 11 * 90
        players = [Player(player_id=i, team_id=1, name="P%d" % i,
                          position="Midfielder",
                          minutes=(38 * 90 if i == 0 else (team_minutes - 38 * 90) / 24.0))
                   for i in range(25)]
        av = F.availability(1, [Injury(1, 0, "P0", "Missing Fixture", "x")], players)
        self.assertAlmostEqual(av.overall_available, 1.0 - 1.0 / 11.0, delta=0.02)

    def test_three_injuries_do_not_gut_the_squad(self):
        players = self._players()
        three = [Injury(1, 100 + i, "P%d" % i, "Missing Fixture", "x")
                 for i in (1, 2, 18)]
        av = F.availability(1, three, players)
        self.assertGreater(av.overall_available, 0.70)
        self.assertLess(av.overall_available, 0.95)

    def test_availability_is_floored(self):
        players = self._players()
        everyone = [Injury(1, 100 + i, "P%d" % i, "Missing Fixture", "x")
                    for i in range(25)]
        av = F.availability(1, everyone, players)
        self.assertGreaterEqual(av.overall_available, 0.4)


class TestOtherCompetitions(unittest.TestCase):
    """A midweek AFC/cup game must shorten the rest days the model sees."""

    def _dataset(self, with_cup: bool):
        base = datetime(2025, 3, 1, 18, 0, tzinfo=timezone.utc)
        league = [Match(1, 2025, base.isoformat(), 10, 20, "A", "B", "FT", 1, 0,
                        league_id=307, league_name="Pro League")]
        ds = Dataset(league_id=307, fetched_at="now", teams={10: "A", 20: "B"},
                     matches=league)
        if with_cup:
            ds.other_matches = [
                Match(99, 2025, (base + timedelta(days=4)).isoformat(), 10, 77,
                      "A", "Foreign", "FT", 2, 1, league_id=17,
                      league_name="AFC Champions League"),
            ]
        return ds, base + timedelta(days=7)

    def test_rest_days_account_for_the_cup_game(self):
        no_cup, kickoff = self._dataset(with_cup=False)
        with_cup, _ = self._dataset(with_cup=True)
        self.assertAlmostEqual(
            F.rest_days(no_cup.schedule_matches, 10, kickoff), 7.0)
        self.assertAlmostEqual(
            F.rest_days(with_cup.schedule_matches, 10, kickoff), 3.0)

    def test_congestion_counts_the_cup_game(self):
        no_cup, kickoff = self._dataset(with_cup=False)
        with_cup, _ = self._dataset(with_cup=True)
        self.assertEqual(F.congestion(no_cup.schedule_matches, 10, kickoff, 14), 1)
        self.assertEqual(F.congestion(with_cup.schedule_matches, 10, kickoff, 14), 2)

    def test_other_competitions_never_enter_the_ratings(self):
        with_cup, _ = self._dataset(with_cup=True)
        self.assertEqual(len(with_cup.played_matches), 1)
        self.assertEqual(len(with_cup.schedule_matches), 2)
        self.assertNotIn(77, with_cup.teams)

    def test_schedule_falls_back_to_league_only(self):
        no_cup, _ = self._dataset(with_cup=False)
        self.assertEqual([m.fixture_id for m in no_cup.schedule_matches],
                         [m.fixture_id for m in no_cup.played_matches])

    def test_other_matches_survive_a_save_load_round_trip(self):
        import tempfile
        from pathlib import Path
        with_cup, _ = self._dataset(with_cup=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            with_cup.save(path)
            back = Dataset.load(path)
            self.assertEqual(len(back.other_matches), 1)
            self.assertEqual(back.other_matches[0].league_name,
                             "AFC Champions League")
            self.assertEqual(len(back.schedule_matches), 2)


class TestHeadToHead(unittest.TestCase):
    def test_orientation_is_by_fixture_home_side(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        matches = [
            Match(1, 2024, base.isoformat(), 10, 20, "A", "B", "FT", 3, 0),
            Match(2, 2024, (base + timedelta(days=100)).isoformat(), 20, 10, "B", "A",
                  "FT", 1, 2),
        ]
        h = F.head_to_head(matches, 10, 20, base + timedelta(days=200))
        self.assertEqual(h.meetings, 2)
        self.assertEqual((h.home_wins, h.draws, h.away_wins), (2, 0, 0))
        self.assertGreater(h.weighted_gd, 0)

        h_rev = F.head_to_head(matches, 20, 10, base + timedelta(days=200))
        self.assertEqual((h_rev.home_wins, h_rev.draws, h_rev.away_wins), (0, 0, 2))
        self.assertLess(h_rev.weighted_gd, 0)


class TestGoalModelRecovery(unittest.TestCase):
    """The fit must recover the parameters that generated the synthetic league."""

    def test_home_advantage_is_unbiased_across_seeds(self):
        """A single simulated season has a standard error around 0.07 on the home
        advantage, so recovery is asserted on the mean over several seeds."""
        fitted = []
        for seed in range(6):
            ds, truth = make_dataset(seed=seed)
            fitted.append(M.fit(ds.matches).home_adv)
        self.assertAlmostEqual(float(np.mean(fitted)), TRUTH["home_adv"], delta=0.06)
        self.assertGreater(min(fitted), 0.0)

    def test_home_advantage_is_positive(self):
        self.assertGreater(RATINGS.home_adv, 0.0)

    def test_base_rate(self):
        self.assertAlmostEqual(RATINGS.base, TRUTH["base"], delta=0.15)

    def test_attack_ratings_correlate_with_truth(self):
        ids = sorted(RATINGS.teams)
        fitted = np.array([RATINGS.attack[t] for t in ids])
        true = np.array([TRUTH["attack"][t] for t in ids])
        self.assertGreater(np.corrcoef(fitted, true)[0, 1], 0.80)

    def test_defence_ratings_correlate_with_truth(self):
        ids = sorted(RATINGS.teams)
        fitted = np.array([RATINGS.defence[t] for t in ids])
        true = np.array([TRUTH["defence"][t] for t in ids])
        self.assertGreater(np.corrcoef(fitted, true)[0, 1], 0.70)

    def test_ratings_are_centred(self):
        self.assertAlmostEqual(np.mean(list(RATINGS.attack.values())), 0.0, places=6)
        self.assertAlmostEqual(np.mean(list(RATINGS.defence.values())), 0.0, places=6)

    def test_schedule_prior_shrinks_a_noisy_rest_coefficient(self):
        """One season barely identifies the rest effect, so the prior must pull a
        noisy estimate toward zero rather than let it assert a wrong sign."""
        from spl.config import ModelConfig
        loose = [M.fit(make_dataset(seed=s)[0].matches,
                       cfg=ModelConfig(schedule_prior_sd=0.0)).b_rest
                 for s in range(5)]
        tight = [M.fit(make_dataset(seed=s)[0].matches,
                       cfg=ModelConfig(schedule_prior_sd=0.05)).b_rest
                 for s in range(5)]
        self.assertLess(float(np.std(tight)), float(np.std(loose)))
        self.assertLess(max(abs(b) for b in tight), max(abs(b) for b in loose))

    def test_rho_stays_in_bounds(self):
        self.assertTrue(-0.18 < RATINGS.rho < 0.18)

    def test_fit_refuses_empty_input(self):
        with self.assertRaises(ValueError):
            M.fit([])

    def test_as_of_excludes_later_matches(self):
        cutoff = sorted(m.dt for m in DS.played_matches)[200]
        r = M.fit(DS.matches, as_of=cutoff)
        self.assertLess(r.n_matches, RATINGS.n_matches)


class TestScoreMatrix(unittest.TestCase):
    def test_matrix_is_a_distribution(self):
        for lam, mu, rho in ((1.6, 1.1, 0.05), (0.4, 3.2, -0.08), (2.5, 2.5, 0.0)):
            mat = M.score_matrix(lam, mu, rho)
            self.assertAlmostEqual(mat.sum(), 1.0, places=9)
            self.assertTrue((mat >= 0).all())

    def test_outcomes_sum_to_one(self):
        mat = M.score_matrix(1.7, 1.2, 0.06)
        p = M.outcome_probabilities(mat)
        self.assertAlmostEqual(p["home"] + p["draw"] + p["away"], 1.0, places=9)

    def test_expected_goals_track_the_rates(self):
        mat = M.score_matrix(1.9, 0.9, 0.0)
        s = M.market_summary(mat)
        self.assertAlmostEqual(s["exp_goals_home"], 1.9, delta=0.02)
        self.assertAlmostEqual(s["exp_goals_away"], 0.9, delta=0.02)

    def test_over_under_are_complementary(self):
        s = M.market_summary(M.score_matrix(1.5, 1.4, 0.05))
        self.assertAlmostEqual(s["over_2.5"] + s["under_2.5"], 1.0, places=6)

    def test_negative_rho_lifts_draws_versus_independent_poisson(self):
        """The empirically fitted Dixon-Coles rho is negative, which is what
        boosts 0-0 and 1-1 relative to independent Poisson."""
        plain = M.outcome_probabilities(M.score_matrix(1.4, 1.2, 0.0))
        corrected = M.outcome_probabilities(M.score_matrix(1.4, 1.2, -0.08))
        self.assertGreater(corrected["draw"], plain["draw"])
        flat = M.score_matrix(1.4, 1.2, 0.0)
        neg = M.score_matrix(1.4, 1.2, -0.08)
        self.assertGreater(neg[0, 0], flat[0, 0])
        self.assertGreater(neg[1, 1], flat[1, 1])
        self.assertLess(neg[1, 0], flat[1, 0])

    def test_stronger_side_is_favoured(self):
        p = M.outcome_probabilities(M.score_matrix(2.2, 0.8, 0.05))
        self.assertGreater(p["home"], p["away"])

    def test_top_scorelines_are_sorted(self):
        rows = M.top_scorelines(M.score_matrix(1.6, 1.1, 0.05), 6)
        self.assertEqual(rows, sorted(rows, key=lambda r: r[2], reverse=True))


class TestAuxModels(unittest.TestCase):
    aux = AUX.fit_aux(DS)

    def test_all_metrics_fit(self):
        for metric in ("shots", "shots_on_target", "possession", "corners"):
            self.assertIn(metric, self.aux.models)

    def test_possession_predictions_sum_to_100(self):
        ids = sorted(DS.teams)
        t = AUX.predict_tempo(self.aux, ids[0], ids[1], 1.5, 1.2)
        self.assertAlmostEqual(t.possession_home + t.possession_away, 100.0, places=6)

    def test_possession_tracks_the_generating_parameter(self):
        model = self.aux.models["possession"]
        ids = sorted(DS.teams)
        fitted = np.array([model.offence[t] for t in ids])
        true = np.array([TRUTH["attack"][t] for t in ids])
        self.assertGreater(np.corrcoef(fitted, true)[0, 1], 0.7)

    def test_shots_are_plausible(self):
        ids = sorted(DS.teams)
        t = AUX.predict_tempo(self.aux, ids[0], ids[1], 1.5, 1.2)
        self.assertTrue(4.0 < t.shots_home < 28.0)
        self.assertTrue(4.0 < t.shots_away < 28.0)
        self.assertLess(t.sot_home, t.shots_home)

    def test_home_effect_is_positive(self):
        self.assertGreater(self.aux.models["possession"].home, 0.0)
        self.assertGreater(self.aux.models["shots"].home, 0.0)

    def test_injuries_reduce_shots_and_possession(self):
        ids = sorted(DS.teams)
        full = AUX.predict_tempo(self.aux, ids[0], ids[1], 1.5, 1.2)
        hurt = AUX.predict_tempo(self.aux, ids[0], ids[1], 1.5, 1.2,
                                 avail_home=0.75)
        self.assertLess(hurt.shots_home, full.shots_home)
        self.assertLess(hurt.possession_home, full.possession_home)

    def test_fallback_derives_possession_from_the_goal_rates(self):
        """The fallback must not silently return a flat 50-50 split."""
        bare = Dataset(league_id=1, fetched_at="now", teams=dict(DS.teams),
                       matches=list(DS.matches))
        aux = AUX.fit_aux(bare)
        strong = AUX.predict_tempo(aux, 1000, 1001, 2.1, 0.9)
        weak = AUX.predict_tempo(aux, 1000, 1001, 0.9, 2.1)
        self.assertGreater(strong.possession_home, 55.0)
        self.assertLess(weak.possession_home, 45.0)
        self.assertGreater(strong.shots_home, weak.shots_home)

    def test_fallback_when_no_box_scores(self):
        bare = Dataset(league_id=1, fetched_at="now", teams=dict(DS.teams),
                       matches=list(DS.matches))
        aux = AUX.fit_aux(bare)
        self.assertEqual(aux.models, {})
        self.assertTrue(aux.fallback_note)
        t = AUX.predict_tempo(aux, sorted(DS.teams)[0], sorted(DS.teams)[1], 1.8, 1.0)
        self.assertTrue(4.0 < t.shots_home < 28.0)
        self.assertAlmostEqual(t.possession_home + t.possession_away, 100.0, places=6)
        self.assertGreater(t.possession_home, 50.0)   # stronger side holds the ball


class TestPredictorEndToEnd(unittest.TestCase):
    predictor = Predictor(DS)

    def test_predicts_a_scheduled_fixture(self):
        fixture = DS.upcoming_matches[0]
        pred = self.predictor.predict_fixture(fixture)
        self.assertEqual(pred.fixture_id, fixture.fixture_id)
        total = pred.markets["home"] + pred.markets["draw"] + pred.markets["away"]
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_a_stale_postponed_fixture_is_not_picked(self):
        """An unplayed fixture whose date has passed must not be treated as next."""
        stale = DS.upcoming_matches[0]
        as_of = stale.dt + timedelta(days=30)
        p = Predictor(DS, as_of=as_of)
        self.assertIsNone(p._find_scheduled(stale.home_id, stale.away_id))
        pred = p.predict_by_name(DS.teams[stale.home_id], DS.teams[stale.away_id])
        self.assertGreater(pred.kickoff, stale.dt)

    def test_scheduled_fixture_is_picked_when_it_is_in_the_future(self):
        upcoming = DS.upcoming_matches[0]
        p = Predictor(DS, as_of=upcoming.dt - timedelta(days=1))
        found = p._find_scheduled(upcoming.home_id, upcoming.away_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.fixture_id, upcoming.fixture_id)

    def test_predict_by_name(self):
        pred = self.predictor.predict_by_name("Team 00", "Team 01")
        self.assertEqual(pred.home_name, "Team 00")

    def test_home_advantage_moves_the_line(self):
        ids = sorted(DS.teams)
        home = self.predictor.predict(ids[0], ids[1])
        neutral = self.predictor.predict(ids[0], ids[1], neutral_venue=True)
        self.assertGreater(home.markets["home"], neutral.markets["home"])
        self.assertGreater(home.lam_home, neutral.lam_home)

    def test_swapping_sides_changes_the_favourite(self):
        ids = sorted(DS.teams)
        a = self.predictor.predict(ids[0], ids[1])
        b = self.predictor.predict(ids[1], ids[0])
        self.assertNotAlmostEqual(a.markets["home"], b.markets["home"], places=3)

    def test_injuries_lower_the_expected_goals_of_the_hurt_team(self):
        hurt = sorted(DS.teams)[0]        # the synthetic injury list targets this club
        other = sorted(DS.teams)[5]
        with_inj = self.predictor.predict(hurt, other, apply_injuries=True)
        without = self.predictor.predict(hurt, other, apply_injuries=False)
        self.assertLess(with_inj.lam_home, without.lam_home)
        self.assertLess(with_inj.markets["home"], without.markets["home"])

    def test_report_renders_every_requested_quantity(self):
        pred = self.predictor.predict_fixture(DS.upcoming_matches[0])
        text = render(pred, verbose=True)
        for needle in ("RESULT", "GOALS", "SHOTS & POSSESSION", "WHY",
                       "Expected goals", "Possession %", "Total shots",
                       "Shots on target", "Goals conceded", "Rest days",
                       "Availability", "Head to head", "Home advantage"):
            self.assertIn(needle, text)
        self.assertIn("vs", render_compact(pred))

    def test_json_shape(self):
        from spl.cli import _as_dict
        blob = _as_dict(self.predictor.predict_fixture(DS.upcoming_matches[0]))
        for key in ("probabilities", "expected_goals", "expected_conceded", "shots",
                    "possession", "inputs", "top_scorelines", "confidence"):
            self.assertIn(key, blob)
        self.assertIn("rest_days", blob["inputs"])
        self.assertIn("availability", blob["inputs"])

    def test_upcoming_window(self):
        as_of = min(m.dt for m in DS.upcoming_matches) - timedelta(days=1)
        p = Predictor(DS, as_of=as_of)
        self.assertTrue(p.upcoming(days=30))


class TestBacktest(unittest.TestCase):
    def test_beats_the_base_rate_baseline(self):
        from spl.backtest import backtest
        res = backtest(DS, min_train_matches=240, refit_every=25,
                       include_tempo=True)
        self.assertGreater(res.n, 100)
        self.assertLess(res.log_loss, res.baseline_log_loss)
        self.assertLess(res.rps, res.baseline_rps)
        self.assertGreater(res.accuracy, 0.35)
        self.assertIsNotNone(res.mae_shots)
        self.assertIsNotNone(res.mae_possession)
        self.assertIn("log loss", res.render())

    def test_refuses_a_tiny_dataset(self):
        from spl.backtest import backtest
        small = Dataset(league_id=1, fetched_at="now",
                        teams=dict(list(DS.teams.items())[:4]),
                        matches=DS.played_matches[:20])
        with self.assertRaises(ValueError):
            backtest(small, min_train_matches=120)


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            DS.save(path)
            back = Dataset.load(path)
            self.assertEqual(len(back.matches), len(DS.matches))
            self.assertEqual(len(back.stats), len(DS.stats))
            self.assertEqual(back.teams, DS.teams)
            self.assertEqual(back.played_matches[0].fixture_id,
                             DS.played_matches[0].fixture_id)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Dataset.load("/nonexistent/path/ds.json")


if __name__ == "__main__":
    unittest.main()


class TestDataVintageHonesty(unittest.TestCase):
    """Stale ratings must be labelled as such, loudly."""

    def _pred(self, age_days, plan_limited=False, injuries=True):
        ds = Dataset(league_id=307, fetched_at="now", teams=dict(DS.teams),
                     matches=list(DS.matches),
                     stats=list(DS.stats), players=list(DS.players),
                     injuries=list(DS.injuries) if injuries else [])
        ds.plan_limited = plan_limited
        p = Predictor(ds)
        newest = ds.played_matches[-1].dt
        return p.predict(sorted(ds.teams)[0], sorted(ds.teams)[1],
                         kickoff=newest + timedelta(days=age_days))

    def test_fresh_data_has_no_warning(self):
        from spl.predict import _data_warnings
        pred = self._pred(7)
        self.assertEqual(_data_warnings(pred), [])
        self.assertNotIn("READ THIS FIRST", render(pred))

    def test_stale_data_warns_and_downgrades_confidence(self):
        from spl.predict import _data_warnings
        pred = self._pred(400)
        self.assertTrue(_data_warnings(pred))
        self.assertIn("READ THIS FIRST", render(pred))
        self.assertIn("NOT current-form", render(pred))
        self.assertIn("stale", pred.confidence())

    def test_data_age_is_measured_to_kickoff(self):
        pred = self._pred(90)
        self.assertAlmostEqual(pred.data_age_days, 90.0, delta=1.0)
        self.assertEqual(pred.confidence(), "low")

    def test_plan_limitation_is_surfaced(self):
        pred = self._pred(7, plan_limited=True)
        text = render(pred)
        self.assertIn("READ THIS FIRST", text)
        self.assertIn("does not cover recent seasons", text)

    def test_absent_injury_feed_is_surfaced(self):
        pred = self._pred(7, injuries=False)
        self.assertFalse(pred.injuries_available)
        text = render(pred)
        self.assertIn("no injury feed", text)
        self.assertIn("assumed fully fit", text)

    def test_present_injury_feed_is_not_flagged(self):
        pred = self._pred(7, injuries=True)
        self.assertTrue(pred.injuries_available)
        self.assertNotIn("no injury feed", render(pred))


class TestHeadToHeadUnits(unittest.TestCase):
    """The h2h term compares an observed goal difference against the one the
    ratings imply. Both must be in goals: `lin_h`/`lin_a` are LOG rates, so
    comparing them directly to a goal difference is incoherent."""

    def _feats(self, weighted_gd, meetings=6):
        f = Predictor(DS).predict(sorted(DS.teams)[0], sorted(DS.teams)[1]).features
        f.h2h.meetings = meetings
        f.h2h.weighted_gd = weighted_gd
        return f

    def test_no_adjustment_when_history_matches_the_model(self):
        p = Predictor(DS)
        home, away = sorted(DS.teams)[0], sorted(DS.teams)[1]
        base = p.predict(home, away, apply_injuries=False)
        implied_goals = base.lam_home - base.lam_away
        feats = self._feats(implied_goals)
        rates = M.goal_rates(p.ratings, home, away, feats, apply_injuries=False)
        self.assertAlmostEqual(rates.components.get("h2h", 0.0), 0.0, delta=0.02)

    def test_history_kinder_than_the_model_favours_the_home_side(self):
        p = Predictor(DS)
        home, away = sorted(DS.teams)[0], sorted(DS.teams)[1]
        base = p.predict(home, away, apply_injuries=False)
        implied = base.lam_home - base.lam_away
        better = M.goal_rates(p.ratings, home, away, self._feats(implied + 2.0),
                              apply_injuries=False)
        worse = M.goal_rates(p.ratings, home, away, self._feats(implied - 2.0),
                             apply_injuries=False)
        self.assertGreater(better.components["h2h"], 0.0)
        self.assertLess(worse.components["h2h"], 0.0)
        self.assertGreater(better.lam_home, worse.lam_home)

    def test_the_comparison_is_made_in_goals_not_log_rates(self):
        """Regression: using `lin_h - lin_a` understates what the ratings imply,
        and most for the biggest favourites."""
        src = (Path(__file__).resolve().parent.parent / "spl" / "model.py").read_text()
        block = src[src.index("if feats.h2h.meetings >= 3"):]
        block = block[:block.index("lam_h = float")]
        self.assertIn("np.exp(lin_h) - np.exp(lin_a)", block)
        self.assertNotIn("implied = (lin_h - lin_a)", block)

    def test_few_meetings_means_no_adjustment(self):
        p = Predictor(DS)
        home, away = sorted(DS.teams)[0], sorted(DS.teams)[1]
        rates = M.goal_rates(p.ratings, home, away, self._feats(3.0, meetings=2),
                             apply_injuries=False)
        self.assertNotIn("h2h", rates.components)

    def test_adjustment_is_bounded_by_its_weight(self):
        p = Predictor(DS)
        home, away = sorted(DS.teams)[0], sorted(DS.teams)[1]
        for gd in (-9.0, -3.0, 0.0, 3.0, 9.0):
            rates = M.goal_rates(p.ratings, home, away, self._feats(gd),
                                 apply_injuries=False)
            self.assertLessEqual(abs(rates.components["h2h"]), 0.12 + 1e-9)


class TestSelfFixtureRejected(unittest.TestCase):
    """A club cannot play itself. The web UI guarded this; the model did not,
    so the CLI happily printed "Al Hilal win 41%" against "Al Hilal win 34%"."""

    def test_predict_rejects_it(self):
        p = Predictor(DS)
        tid = sorted(DS.teams)[0]
        with self.assertRaises(ValueError) as ctx:
            p.predict(tid, tid)
        self.assertIn("cannot play itself", str(ctx.exception))

    def test_the_message_names_the_club(self):
        p = Predictor(DS)
        tid = sorted(DS.teams)[0]
        with self.assertRaises(ValueError) as ctx:
            p.predict(tid, tid)
        self.assertIn(DS.teams[tid], str(ctx.exception))

    def test_predict_by_name_rejects_two_spellings_of_one_club(self):
        p = Predictor(DS)
        name = DS.teams[sorted(DS.teams)[0]]
        with self.assertRaises(ValueError):
            p.predict_by_name(name, name)

    def test_normal_fixtures_are_unaffected(self):
        p = Predictor(DS)
        ids = sorted(DS.teams)
        self.assertIsNotNone(p.predict(ids[0], ids[1]))

    def test_cli_exits_non_zero(self):
        from spl import cli
        args = cli.build_parser().parse_args(
            ["predict", "--home", DS.teams[sorted(DS.teams)[0]],
             "--away", DS.teams[sorted(DS.teams)[0]]])
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with contextlib.redirect_stdout(io.StringIO()):
                original = cli.Dataset.load
                cli.Dataset.load = staticmethod(lambda *a, **k: DS)
                try:
                    code = cli.cmd_predict(args)
                finally:
                    cli.Dataset.load = original
        self.assertEqual(code, 2)
        self.assertIn("cannot play itself", err.getvalue())


class TestBacktestBaselines(unittest.TestCase):
    """MAE figures mean nothing without something to beat."""

    result = None

    @classmethod
    def setUpClass(cls):
        from spl.backtest import backtest
        cls.result = backtest(DS, min_train_matches=300, refit_every=40,
                              include_tempo=True)

    def test_tempo_baselines_are_reported(self):
        self.assertIsNotNone(self.result.baseline_mae_shots)
        self.assertIsNotNone(self.result.baseline_mae_possession)

    def test_the_models_beat_their_baselines(self):
        self.assertLess(self.result.mae_shots, self.result.baseline_mae_shots)
        self.assertLess(self.result.mae_possession,
                        self.result.baseline_mae_possession)

    def test_baselines_appear_in_the_report(self):
        text = self.result.render()
        self.assertIn("MAE shots", text)
        self.assertIn("baseline", text)


class TestScoreMatrixTruncation(unittest.TestCase):
    """The matrix is truncated and renormalised, so the cap has to be high
    enough that the lost tail does not move the numbers read off it."""

    def test_expected_goals_match_the_rate_that_generated_them(self):
        for lam in (0.8, 1.5, 2.5, 3.5, 4.5, 5.5):
            mat = M.score_matrix(lam, 1.3, -0.1)
            idx = np.arange(mat.shape[0])
            got = float((mat.sum(axis=1) * idx).sum())
            self.assertAlmostEqual(got, lam, delta=0.002, msg="lambda=%.1f" % lam)

    def test_a_low_cap_visibly_biases_high_scoring_fixtures(self):
        idx10 = np.arange(11)
        mat10 = M.score_matrix(5.0, 1.3, -0.1, max_goals=10)
        err10 = abs(float((mat10.sum(axis=1) * idx10).sum()) - 5.0)
        self.assertGreater(err10, 0.05)          # the bug, pinned
        mat = M.score_matrix(5.0, 1.3, -0.1)     # the configured cap
        idx = np.arange(mat.shape[0])
        self.assertLess(abs(float((mat.sum(axis=1) * idx).sum()) - 5.0), 0.005)

    def test_the_matrix_is_still_a_distribution(self):
        for lam, mu in ((5.5, 0.4), (0.3, 4.9), (3.0, 3.0)):
            mat = M.score_matrix(lam, mu, -0.1)
            self.assertAlmostEqual(mat.sum(), 1.0, places=9)


class TestFitHonesty(unittest.TestCase):
    """A fit that fails, or a club the fit never saw, must not be presented
    with the same confidence as a good one."""

    def test_convergence_is_recorded(self):
        r = M.fit(DS.matches)
        self.assertTrue(r.converged)
        self.assertTrue(r.fit_message)

    def test_a_failed_fit_is_flagged_in_the_report(self):
        from spl.predict import _data_warnings
        p = Predictor(DS)
        pred = p.predict(sorted(DS.teams)[0], sorted(DS.teams)[1])
        pred.ratings.converged = False
        pred.ratings.fit_message = "ABNORMAL_TERMINATION"
        warnings = _data_warnings(pred)
        self.assertTrue(any("did not converge" in w for w in warnings))
        self.assertIn("ABNORMAL_TERMINATION", render(pred))

    def test_an_unrated_club_is_flagged(self):
        """It is silently scored as exactly league-average, which looks like a
        real prediction unless the report says otherwise."""
        from spl.predict import _data_warnings
        ds = Dataset(league_id=1, fetched_at="now", teams=dict(DS.teams),
                     matches=list(DS.matches), stats=list(DS.stats))
        ghost = 987654
        ds.teams[ghost] = "Unknown FC"
        p = Predictor(ds)
        pred = p.predict(ghost, sorted(DS.teams)[0])
        self.assertNotIn(ghost, p.ratings.attack)
        self.assertTrue(any("no matches in the fitted ratings" in w
                            for w in _data_warnings(pred)))
        self.assertIn("Unknown FC", render(pred))

    def test_a_rated_club_is_not_flagged(self):
        from spl.predict import _data_warnings
        p = Predictor(DS)
        pred = p.predict(sorted(DS.teams)[0], sorted(DS.teams)[1])
        self.assertFalse(any("fitted ratings" in w for w in _data_warnings(pred)))

    def test_an_unrated_club_still_produces_a_valid_distribution(self):
        ds = Dataset(league_id=1, fetched_at="now", teams=dict(DS.teams),
                     matches=list(DS.matches), stats=list(DS.stats))
        ds.teams[987654] = "Unknown FC"
        pred = Predictor(ds).predict(987654, sorted(DS.teams)[0])
        total = pred.markets["home"] + pred.markets["draw"] + pred.markets["away"]
        self.assertAlmostEqual(total, 1.0, places=6)


class TestBoundedParameters(unittest.TestCase):
    """L-BFGS-B reports success even when it has pressed a parameter flat
    against its bound. Such a value is clamped, not estimated."""

    @staticmethod
    def _league(score_fn, n_matches=60, seed=3):
        import random
        rng = random.Random(seed)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        out = []
        for i in range(n_matches):
            h, a = rng.sample(range(6), 2)
            hg, ag = score_fn(rng, i)
            out.append(Match(i, 2025, (start + timedelta(days=i)).isoformat(),
                             h, a, "T%d" % h, "T%d" % a, "FT", hg, ag))
        return out

    def test_a_healthy_fit_reports_nothing_pinned(self):
        self.assertEqual(M.fit(DS.matches).at_bounds, [])

    def test_every_match_won_five_nil_at_home_pins_home_advantage(self):
        r = M.fit(self._league(lambda rng, i: (5, 0)))
        self.assertIn("home advantage", r.at_bounds)

    def test_a_goalless_league_pins_the_base_rate(self):
        r = M.fit(self._league(lambda rng, i: (0, 0)))
        self.assertIn("base", r.at_bounds)

    def test_pinning_is_surfaced_in_the_report(self):
        from spl.predict import _data_warnings
        p = Predictor(DS)
        pred = p.predict(sorted(DS.teams)[0], sorted(DS.teams)[1])
        pred.ratings.at_bounds = ["home advantage"]
        self.assertTrue(any("clamped rather than" in w
                            for w in _data_warnings(pred)))
        self.assertIn("home advantage", render(pred))

    def test_pathological_leagues_still_produce_valid_distributions(self):
        import numpy as np
        for fn in (lambda rng, i: (0, 0),
                   lambda rng, i: (5, 0),
                   lambda rng, i: (rng.randint(8, 15), rng.randint(8, 15)),
                   lambda rng, i: (rng.randint(0, 3),) * 2):
            r = M.fit(self._league(fn))
            mat = M.score_matrix(1.5, 1.2, r.rho)
            self.assertAlmostEqual(mat.sum(), 1.0, places=9)
            self.assertTrue(np.isfinite(list(r.attack.values())).all())


class TestScoreMatrixFuzz(unittest.TestCase):
    """Random and degenerate rates must not break the distribution."""

    def test_random_rates_keep_every_invariant(self):
        import random
        import numpy as np
        rng = random.Random(11)
        for _ in range(400):
            lam, mu = rng.uniform(1e-6, 8.0), rng.uniform(1e-6, 8.0)
            rho = rng.uniform(-0.18, 0.18)
            mat = M.score_matrix(lam, mu, rho)
            self.assertTrue(np.isfinite(mat).all())
            self.assertTrue((mat >= 0).all())
            self.assertAlmostEqual(mat.sum(), 1.0, places=6)
            o = M.outcome_probabilities(mat)
            self.assertAlmostEqual(sum(o.values()), 1.0, places=6)

    def test_degenerate_rates(self):
        for lam, mu in ((0.0, 0.0), (1e-12, 5.0), (20.1, 1e-12)):
            mat = M.score_matrix(lam, mu, -0.1)
            self.assertAlmostEqual(mat.sum(), 1.0, places=6)

    def test_a_matrix_too_small_for_the_correction_does_not_crash(self):
        """The correction touches cells [0,1], [1,0] and [1,1], which a 1x1
        matrix does not have."""
        for cap in (0, 1, 2):
            mat = M.score_matrix(2.0, 1.5, -0.1, max_goals=cap)
            self.assertEqual(mat.shape, (cap + 1, cap + 1))
            self.assertAlmostEqual(mat.sum(), 1.0, places=9)
