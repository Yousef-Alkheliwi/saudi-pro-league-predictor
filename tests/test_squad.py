"""Tests for the squad predictor and the line-up data behind it."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spl import features as F
from spl import squad as S
from spl.data import Appearance, Dataset, Match, Player, TeamSheet
from spl.providers import espn as E

BASE = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)
HOME, AWAY = 1, 2
# a settled XI: keeper, back four, three in midfield, front three
REGULARS = [(10, "G"), (11, "LB"), (12, "CD-L"), (13, "CD-R"), (14, "RB"),
            (15, "CM-L"), (16, "CM"), (17, "CM-R"), (18, "LF"), (19, "F"), (20, "RF")]
BENCH = [(21, "G"), (22, "CD"), (23, "CM"), (24, "F")]


def league(n_matches=8, xi_for=None, red_in=None, scheduled_after=()):
    """Club 1 plays club 2 weekly; `xi_for(k)` overrides club 1's XI in match k."""
    ds = Dataset(league_id=0, fetched_at="now", teams={HOME: "Home FC", AWAY: "Away FC"})
    for k in range(n_matches):
        kick = BASE + timedelta(days=7 * k)
        ds.matches.append(Match(k + 1, 2026, kick.isoformat(), HOME, AWAY,
                                "Home FC", "Away FC", "FT", 1, 1))
        xi = xi_for(k) if xi_for else REGULARS
        ds.team_sheets.append(TeamSheet(k + 1, HOME, "4-3-3"))
        starters = {pid for pid, _ in xi}
        for pid, pos in xi + [b for b in BENCH if b[0] not in starters]:
            ds.appearances.append(Appearance(
                k + 1, HOME, pid, "P%d" % pid, short_name="P%d" % pid,
                jersey=str(pid), position=pos if pid in starters else "SUB",
                starter=pid in starters, minutes=90.0 if pid in starters else 0.0,
                red_card=(red_in is not None and k == red_in[0] and pid == red_in[1])))
    last = BASE + timedelta(days=7 * (n_matches - 1))
    for j, days in enumerate(scheduled_after):
        ds.matches.append(Match(500 + j, 2026, (last + timedelta(days=days)).isoformat(),
                                HOME, AWAY, "Home FC", "Away FC", "NS"))
    return ds, last


class TestPrediction(unittest.TestCase):
    def test_a_settled_side_is_predicted_exactly(self):
        ds, last = league()
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertEqual({p.player_id for p in lu.xi}, {pid for pid, _ in REGULARS})
        self.assertEqual(lu.formation, "4-3-3")

    def test_last_weeks_xi_beats_older_habits(self):
        """The replay showed managers mostly repeat last week's XI."""
        rotated = REGULARS[:10] + [(24, "F")]
        ds, last = league(xi_for=lambda k: rotated if k == 7 else REGULARS)
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        ids = {p.player_id for p in lu.xi}
        self.assertIn(24, ids)
        self.assertNotIn(20, ids)

    def test_exactly_one_goalkeeper(self):
        ds, last = league()
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertEqual(sum(1 for p in lu.xi if p.position == "G"), 1)
        self.assertEqual(len(lu.xi), 11)

    def test_rows_follow_the_formation(self):
        ds, last = league()
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertEqual([len(r) for r in lu.rows], [1, 4, 3, 3])
        backs = [p.position for p in lu.rows[1]]
        self.assertEqual(backs[0], "LB")        # team's left first
        self.assertEqual(backs[-1], "RB")

    def test_no_lineups_means_no_prediction(self):
        ds, last = league()
        self.assertIsNone(S.predict_lineup(ds, AWAY, last + timedelta(days=7)))

    def test_only_line_ups_before_kickoff_are_used(self):
        ds, last = league()
        lu = S.predict_lineup(ds, HOME, BASE + timedelta(days=14, hours=-1))
        self.assertEqual(lu.matches_used, 2)

    def test_players_no_longer_registered_are_left_out(self):
        ds, last = league()
        ds.players = [Player(pid, HOME, "P%d" % pid) for pid, _ in REGULARS + BENCH
                      if pid != 19]
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertNotIn(19, {p.player_id for p in lu.xi})
        self.assertEqual(len(lu.xi), 11)

    def test_a_regular_missing_from_the_last_squad_is_flagged_not_dropped(self):
        without = [r for r in REGULARS if r[0] != 16] + [(23, "CM")]
        ds, last = league(xi_for=lambda k: without if k == 7 else REGULARS)
        # remove 16 from the last matchday squad entirely
        ds.appearances = [a for a in ds.appearances if not (a.fixture_id == 8 and a.player_id == 16)]
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        kinds = {a.player_id: a.kind for a in lu.absences}
        self.assertEqual(kinds.get(16), "missed last squad")


class TestSuspensions(unittest.TestCase):
    def test_sent_off_last_match_is_suspended_for_the_next(self):
        ds, last = league(red_in=(7, 13), scheduled_after=(7,))
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertNotIn(13, {p.player_id for p in lu.xi})
        self.assertEqual([a.kind for a in lu.absences if a.player_id == 13], ["suspended"])
        self.assertEqual(len(lu.xi), 11)

    def test_the_ban_is_served_once(self):
        """A second fixture after the red card is not the one the ban falls on."""
        ds, last = league(red_in=(7, 13), scheduled_after=(7, 14))
        self.assertEqual(S.suspensions(ds, HOME, last + timedelta(days=14)), [])
        self.assertEqual(len(S.suspensions(ds, HOME, last + timedelta(days=7))), 1)

    def test_an_older_red_card_has_been_served(self):
        ds, last = league(red_in=(5, 13))
        self.assertEqual(S.suspensions(ds, HOME, last + timedelta(days=7)), [])

    def test_a_ban_reaches_the_goal_model(self):
        ds, last = league(red_in=(7, 13), scheduled_after=(7,))
        ds.players = [Player(pid, HOME, "P%d" % pid, "Defender", minutes=630.0)
                      for pid, _ in REGULARS]
        feats = F.build_features(ds, HOME, AWAY, last + timedelta(days=7))
        self.assertLess(feats.availability_home.overall_available, 1.0)
        self.assertTrue(any("P13" in x for x in feats.availability_home.out))

    def test_bans_are_marked_as_derived_not_as_an_injury_feed(self):
        ds, last = league(red_in=(7, 13), scheduled_after=(7,))
        bans = S.suspensions(ds, HOME, last + timedelta(days=7))
        self.assertEqual({b.origin for b in bans}, {"suspension"})


class TestEvaluation(unittest.TestCase):
    def test_a_settled_side_is_called_perfectly(self):
        ds, _ = league()
        ev = S.evaluate(ds)
        self.assertGreater(ev.n, 0)
        self.assertAlmostEqual(ev.mean_correct, 11.0)
        self.assertAlmostEqual(ev.all_eleven, 1.0)

    def test_handling_bans_beats_repeating_last_week(self):
        def xi(k):
            # 13 is sent off in match 3 and serves the ban in match 4
            return [r for r in REGULARS if r[0] != 13] + [(22, "CD")] if k == 4 else REGULARS
        ds, _ = league(xi_for=xi, red_in=(3, 13))
        ds.players = [Player(pid, HOME, "P%d" % pid,
                             "Goalkeeper" if pos == "G" else "Defender")
                      for pid, pos in REGULARS + BENCH]
        ev = S.evaluate(ds)
        self.assertGreater(ev.mean_correct, ev.baseline_correct)

    def test_the_registered_squad_is_not_used_when_replaying_history(self):
        """Today's squad would leak who later left or arrived."""
        ds, _ = league()
        ds.players = [Player(99, HOME, "Someone else")]
        self.assertAlmostEqual(S.evaluate(ds).mean_correct, 11.0)


class TestParsing(unittest.TestCase):
    SUMMARY = {
        "rosters": [{"team": {"id": "1"}, "formation": "4-3-3", "roster": [
            {"athlete": {"id": str(pid), "displayName": "P%d" % pid, "shortName": "P."},
             "jersey": str(pid), "starter": pid <= 21, "formationPlace": pid - 10 if pid <= 21 else 0,
             "position": {"abbreviation": "G" if pid == 11 else ("SUB" if pid > 21 else "CM")},
             "stats": [{"name": "redCards", "value": 1.0 if pid == 15 else 0.0},
                       {"name": "foulsCommitted", "value": 2.0}]}
            for pid in range(11, 25)]}],
        "keyEvents": [
            {"type": {"text": "Substitution"}, "clock": {"value": 3600.0},
             "team": {"id": "1"}, "participants": [{"athlete": {"id": "22"}}, {"athlete": {"id": "20"}}]},
            {"type": {"text": "Red Card"}, "clock": {"value": 1800.0},
             "team": {"id": "1"}, "participants": [{"athlete": {"id": "15"}}]},
            {"type": {"text": "Goal"}, "clock": {"value": 100.0}, "participants": []},
        ],
        "commentary": ["x" * 5000], "news": {"articles": ["y" * 5000]},
    }

    def _parse(self, payload=None):
        return E.parse_summary(77, E.trim_summary(payload or self.SUMMARY))

    def test_trimming_drops_everything_unused(self):
        trimmed = E.trim_summary(self.SUMMARY)
        self.assertEqual(set(trimmed), {"rosters", "events"})
        self.assertLess(len(json.dumps(trimmed)), len(json.dumps(self.SUMMARY)) / 4)
        self.assertEqual([e["kind"] for e in trimmed["events"]], ["sub", "red"])

    def test_minutes_from_substitutions_and_red_cards(self):
        sheets, apps = self._parse()
        by = {a.player_id: a for a in apps}
        self.assertEqual(by[11].minutes, 90.0)          # played throughout
        self.assertEqual(by[20].minutes, 60.0)          # replaced at 60'
        self.assertEqual(by[22].minutes, 30.0)          # came on at 60'
        self.assertEqual(by[15].minutes, 30.0)          # sent off at 30'
        self.assertEqual(by[23].minutes, 0.0)           # unused substitute
        self.assertTrue(by[15].red_card)
        self.assertEqual(sheets[0].formation, "4-3-3")

    def test_a_full_side_totals_990_less_the_red_card(self):
        _, apps = self._parse()
        self.assertEqual(sum(a.minutes for a in apps), 990.0 - 60.0)

    def test_malformed_summaries_never_raise(self):
        deleted = object()

        def paths(obj, prefix=()):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield prefix + (k,)
                    yield from paths(v, prefix + (k,))
            elif isinstance(obj, list):
                for i, v in enumerate(obj[:3]):
                    yield prefix + (i,)
                    yield from paths(v, prefix + (i,))
        checked = 0
        for path in list(paths(self.SUMMARY)):
            for val in (deleted, None, "", 0, [], {}, "xxx", -1, 1e300, True):
                payload = copy.deepcopy(self.SUMMARY)
                cur = payload
                try:
                    for k in path[:-1]:
                        cur = cur[k]
                    if val is deleted:
                        cur.pop(path[-1], None) if isinstance(cur, dict) else cur.pop(path[-1])
                    else:
                        cur[path[-1]] = val
                except Exception:
                    continue
                checked += 1
                try:
                    self._parse(payload)
                except Exception as exc:
                    self.fail("%s=%r raised %s" % (".".join(map(str, path)), val, exc))
        self.assertGreater(checked, 200)
        for junk in (None, [], "x", 5, {"rosters": 5}, {"keyEvents": [None]}):
            E.parse_summary(1, E.trim_summary(junk))

    def test_squad_lists_flat_or_grouped(self):
        flat = {"athletes": [{"id": "5", "displayName": "A", "position": {"abbreviation": "D"}}]}
        grouped = {"athletes": [{"position": "D", "items": [
            {"id": "6", "displayName": "B", "position": {"abbreviation": "F"}}]}]}
        self.assertEqual(E.parse_squad(1, flat)[0].position, "Defender")
        self.assertEqual(E.parse_squad(1, grouped)[0].position, "Attacker")
        self.assertEqual(E.parse_squad(1, {"athletes": [None, {"id": "x"}]}), [])


class TestFetchAndPersistence(unittest.TestCase):
    def test_fetch_attaches_line_ups_squads_and_minutes(self):
        summary = TestParsing.SUMMARY
        event = {"id": "77", "date": "2026-09-13T16:20Z",
                 "status": {"type": {"completed": True, "description": "Full Time"}},
                 "competitions": [{"competitors": [
                     {"homeAway": "home", "score": "1", "team": {"id": "1", "displayName": "Home FC"}},
                     {"homeAway": "away", "score": "0", "team": {"id": "2", "displayName": "Away FC"}}]}]}

        class Stub:
            calls_made = cache_hits = 0

            def month(self, y, m):
                return {"events": [event] if (y, m) == (2026, 9) else []}

            def teams(self):
                return {}

            def injuries(self):
                return {"injuries": []}

            def summary(self, event_id, kickoff):
                return E.trim_summary(summary)

            def squad(self, team_id):
                return {"athletes": [{"id": "20", "displayName": "P20",
                                      "position": {"abbreviation": "M"}}]} if team_id == 1 else {}

        ds = E.fetch_dataset(Stub(), seasons=[2026], with_logos=False, log=lambda *a: None)
        self.assertEqual(len([a for a in ds.appearances if a.starter]), 11)
        p20 = next(p for p in ds.players if p.player_id == 20)
        self.assertEqual(p20.minutes, 60.0)
        self.assertEqual(p20.position, "Midfielder")

    def test_line_ups_survive_a_round_trip(self):
        ds, _ = league()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            back = Dataset.load(path)
        self.assertEqual(len(back.appearances), len(ds.appearances))
        self.assertEqual(back.team_sheets[0].formation, "4-3-3")
        self.assertEqual(back.appearances[0], ds.appearances[0])

    def test_a_snapshot_from_before_line_ups_still_loads(self):
        ds, _ = league()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            blob = json.loads(path.read_text())
            for key in ("appearances", "team_sheets"):
                blob.pop(key)
            for inj in blob["injuries"]:
                inj.pop("origin", None)
            path.write_text(json.dumps(blob))
            back = Dataset.load(path)
        self.assertEqual(back.appearances, [])


class TestPageAndExport(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_export_carries_line_ups_and_measured_accuracy(self):
        from spl.export import build_payload
        from spl.predict import Predictor
        from tests.synthetic import make_dataset
        ds, _ = make_dataset(seed=4)
        ids = sorted(ds.teams)
        played = ds.played_matches
        for m in played[-40:]:
            for tid in (m.home_id, m.away_id):
                ds.team_sheets.append(TeamSheet(m.fixture_id, tid, "4-4-2"))
                for i in range(11):
                    ds.appearances.append(Appearance(
                        m.fixture_id, tid, tid * 100 + i, "P%d" % i, short_name="P%d" % i,
                        jersey=str(i + 1), position="G" if i == 0 else "CM",
                        starter=True, minutes=90.0))
        payload = build_payload(ds, predictor=Predictor(ds), log=lambda *a: None)
        acc = payload["meta"]["lineup_accuracy"]
        self.assertIsNotNone(acc)
        self.assertTrue({"n", "right", "rule", "baseline", "exact", "method"} <= set(acc))
        self.assertIn(acc["method"], ("model", "rule"))
        with_lineup = [c for c in payload["clubs"] if c.get("lineup")]
        self.assertTrue(with_lineup)
        self.assertEqual(len(with_lineup[0]["lineup"]["xi"]), 11)

    def test_page_renders_line_ups_and_escapes_names(self):
        app = (self.ROOT / "ui" / "src" / "app.js").read_text()
        page = (self.ROOT / "ui" / "src" / "page.html").read_text()
        self.assertIn('id="xisec"', page)
        self.assertIn("function pitchHTML", app)
        self.assertIn("function esc(", app)
        block = app[app.index("function pitchHTML"):app.index("function buildLineups")]
        self.assertIn("esc(p.name)", block)
        self.assertNotIn("+ p.name +", block)       # never raw into innerHTML

    def test_name_boxes_are_sized_to_their_row(self):
        """A fixed width let four names collide at phone width."""
        app = (self.ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn("100 / (line.length + 1)", app)

    def test_a_later_fixture_has_the_banned_player_back(self):
        """The page showed a ban on every scheduled fixture, weeks ahead too,
        while the goal model rightly had the player back after one match."""
        from spl.export import club_lineups
        ds, last = league(red_in=(7, 13), scheduled_after=(7, 14))
        nxt, later = club_lineups(ds, HOME, last + timedelta(days=7))
        self.assertNotIn(13, {p["id"] for p in nxt["xi"]})
        self.assertIn("suspended", {a["kind"] for a in nxt["absences"]})
        self.assertIn(13, {p["id"] for p in later["xi"]})
        self.assertNotIn("suspended", {a["kind"] for a in later["absences"]})

    def test_no_second_line_up_without_a_ban(self):
        from spl.export import club_lineups
        ds, last = league(scheduled_after=(7, 14))
        nxt, later = club_lineups(ds, HOME, last + timedelta(days=7))
        self.assertEqual(len(nxt["xi"]), 11)
        self.assertIsNone(later)

    def test_page_picks_the_line_up_for_the_fixture_shown(self):
        app = (self.ROOT / "ui" / "src" / "app.js").read_text()
        block = app[app.index("function lineupFor"):app.index("function buildLineups")]
        self.assertIn("NEXT_OF[club.id] !== fixture", block)
        self.assertIn("club.lineup_later", block)
        self.assertNotIn('"Next fixture</span>', app)   # only for a club's first match


class TestReserveKeeper(unittest.TestCase):
    """ESPN lists unused substitutes only as "SUB", and every player carries
    the same stat fields, so the line-ups alone cannot tell a reserve keeper
    from an outfield sub. He was picked at centre-back to fill a suspension
    gap. The squad list names keepers; failing that, his latest start anywhere
    in the data does."""

    def _banned_case(self):
        return league(red_in=(7, 13), scheduled_after=(7,))

    def test_the_squad_list_keeps_a_reserve_keeper_out_of_defence(self):
        ds, last = self._banned_case()
        ds.players = [Player(pid, HOME, "P%d" % pid,
                             "Goalkeeper" if pos == "G" else "Defender")
                      for pid, pos in REGULARS + BENCH]
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        ids = {p.player_id for p in lu.xi}
        self.assertNotIn(21, ids)
        self.assertNotIn(13, ids)
        self.assertEqual(len(ids), 11)
        self.assertEqual(sum(1 for p in lu.xi if p.position == "G"), 1)

    def test_an_old_start_identifies_him_without_a_squad_list(self):
        ds, last = self._banned_case()
        # a cup-tie-style outing in goal months before the window
        early = BASE - timedelta(days=200)
        ds.matches.append(Match(900, 2025, early.isoformat(), HOME, AWAY,
                                "Home FC", "Away FC", "FT", 0, 0))
        ds.appearances.append(Appearance(900, HOME, 21, "P21", short_name="P21",
                                         position="G", starter=True, minutes=90))
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7), use_squad=False)
        self.assertNotIn(21, {p.player_id for p in lu.xi})

    def test_known_outfielders_fill_a_gap_before_unknowns(self):
        ds, last = self._banned_case()
        early = BASE - timedelta(days=200)
        ds.matches.append(Match(901, 2025, early.isoformat(), HOME, AWAY,
                                "Home FC", "Away FC", "FT", 0, 0))
        ds.appearances.append(Appearance(901, HOME, 24, "P24", short_name="P24",
                                         position="CD", starter=True, minutes=90))
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7), use_squad=False)
        self.assertIn(24, {p.player_id for p in lu.xi})

    def test_a_later_start_is_not_used_to_place_him(self):
        """The replay must not learn a position from after kick-off."""
        ds, last = self._banned_case()
        later = last + timedelta(days=60)
        ds.matches.append(Match(902, 2026, later.isoformat(), HOME, AWAY,
                                "Home FC", "Away FC", "FT", 0, 0))
        ds.appearances.append(Appearance(902, HOME, 21, "P21", position="G",
                                         starter=True, minutes=90))
        roles = S._start_positions_before(ds, last + timedelta(days=7))
        self.assertNotIn(21, roles)

    def test_bench_minutes_decide_between_unused_options(self):
        ds, last = self._banned_case()
        ds.players = [Player(pid, HOME, "P%d" % pid,
                             "Goalkeeper" if pos == "G" else "Defender")
                      for pid, pos in REGULARS + BENCH]
        for a in ds.appearances:                # 23 has been coming on lately
            if a.player_id == 23 and a.fixture_id >= 6:
                a.minutes = 25.0
        lu = S.predict_lineup(ds, HOME, last + timedelta(days=7))
        self.assertIn(23, {p.player_id for p in lu.xi})


if __name__ == "__main__":
    unittest.main()
