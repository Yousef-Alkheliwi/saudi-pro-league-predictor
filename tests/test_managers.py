"""Tests for manager timelines (Wikipedia) and the learned start model."""

from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta, timezone

from spl import squad as S
from spl.data import Appearance, Dataset, ManagerSpell, Match, Player, TeamSheet
from spl.providers import wikipedia as W

PAGE = """
<h2>Personnel and kits</h2>
<table><tr><th>Team</th><th>Manager</th><th>Captain</th></tr>
<tr><td>Al-Hilal</td><td>Simone Inzaghi</td><td>X</td></tr>
<tr><td>Al-Nassr</td><td>Ange Postecoglou</td><td>Y</td></tr>
<tr><td>Damac</td><td>Some Coach<sup>[1]</sup></td><td>Z</td></tr></table>
<h2>Managerial changes</h2>
<table><tr><th>Team</th><th>Outgoing manager</th><th>Manner of departure</th>
<th>Date of vacancy</th><th>Position in table</th><th>Incoming manager</th>
<th>Date of appointment</th></tr>
<tr><td>Al-Nassr</td><td>Jorge Jesus</td><td rowspan="2">End of contract</td>
<td rowspan="2">30 June 2026</td><td rowspan="2">Pre-season</td>
<td>Ange Postecoglou</td><td>3 July 2026</td></tr>
<tr><td>Al-Hilal</td><td>Old Boss (caretaker)</td><td>Simone Inzaghi</td><td>4 June 2025</td></tr>
<tr><td>Al-Nassr</td><td>Ange Postecoglou</td><td>Sacked</td><td>1 November 2026</td><td>6th</td>
<td>Interim Man (caretaker)</td><td>2 November 2026</td></tr>
</table>
"""

TEAMS = {1: "Al Hilal", 2: "Al Nassr", 3: "Damac"}


def resolve(name):
    from spl.data import resolve_team_name
    return resolve_team_name(name, TEAMS)[0]


class TestWikipediaParsing(unittest.TestCase):
    def test_season_titles_use_an_en_dash(self):
        self.assertEqual(W.season_title(2025), "2025\u201326_Saudi_Pro_League")
        self.assertEqual(W.season_title(2099), "2099\u201300_Saudi_Pro_League")

    def test_personnel_table(self):
        managers, _ = W.parse_season(PAGE)
        self.assertEqual(managers["Al-Hilal"], "Simone Inzaghi")
        self.assertEqual(managers["Damac"], "Some Coach")     # footnote dropped

    def test_rowspans_are_expanded_into_the_right_columns(self):
        """Merged cells used to shift every later cell one column left."""
        _, changes = W.parse_season(PAGE)
        hilal = next(c for c in changes if c.team == "Al-Hilal")
        self.assertEqual(hilal.outgoing, "Old Boss")          # caretaker tag stripped
        self.assertEqual(hilal.incoming, "Simone Inzaghi")
        self.assertEqual(hilal.appointed.date().isoformat(), "2025-06-04")
        self.assertEqual(hilal.vacated.date().isoformat(), "2026-06-30")  # carried down

    def test_timelines(self):
        managers, changes = W.parse_season(PAGE)
        spells = W.build_spells(managers, changes, resolve)
        nassr = [name for _, name in spells[2]]
        self.assertEqual(nassr, ["Jorge Jesus", "Ange Postecoglou", "Interim Man"])
        self.assertEqual([name for _, name in spells[3]], ["Some Coach"])

    def test_junk_html_does_not_raise(self):
        for junk in ("", "<table>", "<h2>Managerial changes</h2><table><tr><td>x",
                     "<table><tr><th>Team</th></tr><tr><td></td></tr></table>"):
            W.parse_season(junk)

    def test_manager_at(self):
        managers, changes = W.parse_season(PAGE)
        spells = W.build_spells(managers, changes, resolve)
        ds = Dataset(league_id=0, fetched_at="now", teams=dict(TEAMS))
        ds.managers = [ManagerSpell(t, n, st.isoformat())
                       for t, tl in spells.items() for st, n in tl]
        at = lambda y, m, d: ds.manager_at(2, datetime(y, m, d, tzinfo=timezone.utc)).name
        self.assertEqual(at(2026, 5, 1), "Jorge Jesus")
        self.assertEqual(at(2026, 8, 1), "Ange Postecoglou")
        self.assertEqual(at(2026, 12, 1), "Interim Man")
        self.assertIsNone(ds.manager_at(9, datetime(2026, 1, 1, tzinfo=timezone.utc)))


def synthetic_league(seed=3, teams=8, rounds=30, change_at=15):
    """Clubs with stable XIs, light rotation, and a manager change halfway
    through for every other club - the new man prefers different players."""
    rng = random.Random(seed)
    base = datetime(2025, 8, 1, 18, tzinfo=timezone.utc)
    ds = Dataset(league_id=0, fetched_at="now",
                 teams={t: "Club %d" % t for t in range(1, teams + 1)})
    positions = ["G", "LB", "CD-L", "CD-R", "RB", "CM-L", "CM", "CM-R", "LF", "F", "RF"]
    fid = 1
    for t in ds.teams:
        ds.managers.append(ManagerSpell(t, "Boss %d" % t, base.isoformat()))
        if t % 2 == 0:
            ds.managers.append(ManagerSpell(
                t, "New boss %d" % t, (base + timedelta(days=7 * change_at - 2)).isoformat()))
        for i in range(18):
            ds.players.append(Player(t * 100 + i, t, "P%d-%d" % (t, i),
                                     "Goalkeeper" if i in (0, 17) else "Midfielder"))
    for r in range(rounds):
        ids = list(ds.teams)
        rng.shuffle(ids)
        for h, a in zip(ids[::2], ids[1::2]):
            when = base + timedelta(days=7 * r)
            hg, ag = rng.randint(0, 3), rng.randint(0, 3)
            ds.matches.append(Match(fid, 2025, when.isoformat(), h, a, "", "", "FT", hg, ag))
            for t in (h, a):
                new_era = t % 2 == 0 and r >= change_at
                core = list(range(0, 11)) if not new_era else [0, 1, 2, 3, 4, 11, 12, 13, 8, 14, 15]
                xi = core[:]
                if rng.random() < 0.3:                      # light rotation
                    out = rng.randrange(1, 11)
                    xi[out] = rng.choice([i for i in range(1, 17) if i not in xi])
                ds.team_sheets.append(TeamSheet(fid, t, "4-3-3"))
                for i in range(18):
                    pid = t * 100 + i
                    started = i in xi
                    ds.appearances.append(Appearance(
                        fid, t, pid, "P%d-%d" % (t, i), short_name="P%d" % i,
                        jersey=str(i + 1),
                        position=positions[xi.index(i)] if started else "SUB",
                        starter=started, minutes=90.0 if started else 0.0,
                        goals=1 if started and rng.random() < 0.1 else 0))
            fid += 1
    return ds


class TestStartModel(unittest.TestCase):
    ds = synthetic_league()

    def test_it_fits_on_enough_line_ups(self):
        model = S.fit_selection(self.ds)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.weights), len(S.FEATURES))
        self.assertGreater(model.n_rows, S.MIN_TRAINING_ROWS)

    def test_too_few_line_ups_falls_back_to_the_rule(self):
        tiny = synthetic_league(teams=2, rounds=4)
        self.assertIsNone(S.fit_selection(tiny))
        lu = S.predict_lineup(tiny, 1, datetime(2025, 9, 15, tzinfo=timezone.utc))
        self.assertEqual(lu.method, "rule")

    def test_starting_last_week_is_the_strongest_signal(self):
        odds = dict(S.fit_selection(self.ds).odds_ratios())
        self.assertGreater(odds["started_last"], 1.0)

    def test_probabilities_are_probabilities(self):
        lu = S.predict_lineup(self.ds, 1, datetime(2026, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(lu.method, "model")
        for p in lu.xi + lu.bench:
            self.assertTrue(0.0 <= p.p_start <= 1.0)
        self.assertGreaterEqual(min(p.p_start for p in lu.xi),
                                max((p.p_start for p in lu.bench), default=0.0) - 1e-9)

    def test_the_manager_is_reported(self):
        lu = S.predict_lineup(self.ds, 2, datetime(2026, 3, 1, tzinfo=timezone.utc))
        self.assertEqual(lu.manager, "New boss 2")

    def test_the_new_managers_picks_win_out(self):
        """Club 2 changed manager halfway; his core (11-15) should be in the XI."""
        lu = S.predict_lineup(self.ds, 2, datetime(2026, 3, 1, tzinfo=timezone.utc))
        ids = {p.player_id - 200 for p in lu.xi}
        self.assertGreaterEqual(len(ids & {11, 12, 13, 14, 15}), 4)

    def test_manager_share_only_counts_the_current_manager(self):
        cache = S._cache_for(self.ds)
        kick = datetime(2026, 3, 1, tzinfo=timezone.utc)
        long_history = S._team_history(self.ds, 2, kick, S.MANAGER_WINDOW, *cache)
        feats, meta = S._player_features(self.ds, 2, kick, long_history[:S.WINDOW],
                                         long_history, cache)
        self.assertEqual(meta["manager"].name, "New boss 2")
        old_only = feats.get(205)         # a regular only under the old manager
        if old_only is not None:
            self.assertLess(old_only["manager_share"], 0.5)

    def test_features_never_look_ahead(self):
        """Rewriting every later line-up must not change today's features."""
        import copy
        kick = datetime(2025, 12, 1, tzinfo=timezone.utc)
        changed = copy.deepcopy(self.ds)
        later = {m.fixture_id for m in changed.matches if m.dt >= kick}
        for a in changed.appearances:
            if a.fixture_id in later:
                a.starter = not a.starter
                a.goals = 5
        def feats(ds):
            cache = S._cache_for(ds)
            lh = S._team_history(ds, 3, kick, S.MANAGER_WINDOW, *cache)
            return S._player_features(ds, 3, kick, lh[:S.WINDOW], lh, cache)[0]
        self.assertEqual(feats(self.ds), feats(changed))

    def test_cross_validated_evaluation_reports_all_three(self):
        ev = S.evaluate(self.ds)
        self.assertEqual(ev.method, "model")
        self.assertEqual(ev.folds, 4)
        self.assertGreater(ev.n, 100)
        self.assertGreater(ev.mean_correct, 8.0)
        self.assertIn("fitted start model", ev.render())
        self.assertTrue(ev.calibration)


class TestKeepShape(unittest.TestCase):
    """Picking the ten most likely outfielders can name five defenders for a
    back four; the XI keeps last match's balance of the lines instead."""

    def test_a_fifth_defender_gives_way_to_a_midfielder(self):
        from spl.data import Appearance as A
        positions = ["LB", "CD-L", "CD-R", "RB", "CD",            # five defenders
                     "CM-L", "CM", "CM-R", "LF", "F", "RF", "CM"]
        ranked = []
        for i, pos in enumerate(positions):
            p = S.LineupPlayer(i, "P%d" % i, "P%d" % i, str(i), pos, 0.0, 0, 8)
            p._depth = S.depth(pos)
            ranked.append(p)
        # ranked by likelihood: the fifth defender (index 4) outranks a midfielder
        last = [A(1, 1, i, "P%d" % i, position=pos, starter=True)
                for i, pos in enumerate(["LB", "CD-L", "CD-R", "RB", "CM-L", "CM",
                                         "CM-R", "LF", "F", "RF"])]
        chosen = S._keep_shape(ranked, last, {}, {})
        bands = [S._band(p._depth) for p in chosen]
        self.assertEqual((bands.count(0), bands.count(1), bands.count(2)), (4, 3, 3))
        self.assertNotIn(4, {p.player_id for p in chosen})

    def test_an_unreadable_last_shape_falls_back_to_probability(self):
        ranked = []
        for i in range(12):
            p = S.LineupPlayer(i, "P%d" % i, "P%d" % i, "", "CM", 0.0, 0, 8)
            p._depth = 2.5
            ranked.append(p)
        self.assertEqual([p.player_id for p in S._keep_shape(ranked, [], {}, {})],
                         list(range(10)))


if __name__ == "__main__":
    unittest.main()
