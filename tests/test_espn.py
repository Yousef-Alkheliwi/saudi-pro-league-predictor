"""Tests for the free ESPN provider, against realistic payload shapes."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from spl.data import Dataset
from spl.providers import espn as E

EVENT = {
    "id": "401900915",
    "date": "2026-09-13T16:20Z",
    "status": {"type": {"completed": True, "description": "Full Time",
                        "shortDetail": "FT"}},
    "competitions": [{
        "id": "401900915",
        "date": "2026-09-13T16:20Z",
        "venue": {"fullName": "Prince Abdullah bin Jalawi Stadium"},
        "competitors": [
            {"homeAway": "home", "score": "4", "winner": True,
             "team": {"id": "17755", "displayName": "Neom SC"},
             "statistics": [
                 {"name": "foulsCommitted", "displayValue": "13"},
                 {"name": "wonCorners", "displayValue": "11"},
                 {"name": "possessionPct", "displayValue": "62.5"},
                 {"name": "shotsOnTarget", "displayValue": "6"},
                 {"name": "totalShots", "displayValue": "21"},
                 {"name": "accuratePasses", "displayValue": "432"},
             ]},
            {"homeAway": "away", "score": "2", "winner": False,
             "team": {"id": "7712", "displayName": "Al Fateh"},
             "statistics": [
                 {"name": "possessionPct", "displayValue": "37.5"},
                 {"name": "shotsOnTarget", "displayValue": "3"},
                 {"name": "totalShots", "displayValue": "9"},
                 {"name": "wonCorners", "displayValue": "2"},
             ]},
        ],
    }],
}

SCHEDULED = {
    "id": "401900999",
    "date": "2026-10-10T18:00Z",
    "status": {"type": {"completed": False, "description": "Scheduled"}},
    "competitions": [{
        "competitors": [
            {"homeAway": "home", "score": "0",
             "team": {"id": "929", "displayName": "Al Hilal"}},
            {"homeAway": "away", "score": "0",
             "team": {"id": "932", "displayName": "Al Ittihad"}},
        ],
    }],
}


class TestSeasonLabelling(unittest.TestCase):
    def test_august_onward_opens_the_season(self):
        self.assertEqual(E.season_of(datetime(2026, 8, 13, tzinfo=timezone.utc)), 2026)
        self.assertEqual(E.season_of(datetime(2026, 12, 31, tzinfo=timezone.utc)), 2026)

    def test_january_to_july_belongs_to_the_previous_season(self):
        self.assertEqual(E.season_of(datetime(2026, 1, 5, tzinfo=timezone.utc)), 2025)
        self.assertEqual(E.season_of(datetime(2026, 5, 26, tzinfo=timezone.utc)), 2025)
        self.assertEqual(E.season_of(datetime(2026, 7, 31, tzinfo=timezone.utc)), 2025)

    def test_season_months_span_august_to_july(self):
        months = E.season_months(2025)
        self.assertEqual(months[0], (2025, 8))
        self.assertEqual(months[-1], (2026, 7))
        self.assertEqual(len(months), 12)

    def test_months_ahead_rolls_the_year(self):
        now = datetime(2026, 11, 20, tzinfo=timezone.utc)
        self.assertEqual(E._months_ahead(now, 2), (2027, 1))
        self.assertEqual(E._months_ahead(now, 0), (2026, 11))


class TestEventParsing(unittest.TestCase):
    def test_finished_match(self):
        match, stats = E.parse_event(EVENT)
        self.assertEqual(match.fixture_id, 401900915)
        self.assertEqual((match.home_id, match.away_id), (17755, 7712))
        self.assertEqual((match.home_goals, match.away_goals), (4, 2))
        self.assertTrue(match.played)
        self.assertEqual(match.season, 2026)
        self.assertEqual(match.venue, "Prince Abdullah bin Jalawi Stadium")

    def test_home_and_away_are_read_from_the_flag_not_the_order(self):
        flipped = json.loads(json.dumps(EVENT))
        flipped["competitions"][0]["competitors"].reverse()
        match, _ = E.parse_event(flipped)
        self.assertEqual(match.home_id, 17755)
        self.assertEqual(match.home_goals, 4)

    def test_box_score_maps_to_our_fields(self):
        _, stats = E.parse_event(EVENT)
        self.assertEqual(len(stats), 2)
        home = next(s for s in stats if s.team_id == 17755)
        self.assertEqual(home.shots, 21.0)
        self.assertEqual(home.shots_on_target, 6.0)
        self.assertEqual(home.possession, 62.5)
        self.assertEqual(home.corners, 11.0)

    def test_possession_of_both_sides_sums_to_100(self):
        _, stats = E.parse_event(EVENT)
        self.assertAlmostEqual(sum(s.possession for s in stats), 100.0, places=3)

    def test_scheduled_match_carries_no_score(self):
        match, stats = E.parse_event(SCHEDULED)
        self.assertFalse(match.played)
        self.assertIsNone(match.home_goals)
        self.assertIsNone(match.away_goals)
        self.assertEqual(match.status, "NS")
        self.assertEqual(stats, [])

    def test_a_scheduled_match_with_a_0_0_placeholder_is_not_a_draw(self):
        """ESPN sends score "0" before kick-off; treating that as a result would
        feed phantom 0-0 draws into the model."""
        match, _ = E.parse_event(SCHEDULED)
        self.assertIsNone(match.home_goals)

    def test_malformed_events_are_dropped(self):
        self.assertIsNone(E.parse_event({}))
        self.assertIsNone(E.parse_event({"id": "1", "competitions": []}))
        one_side = {"id": "1", "date": "2026-09-13T16:20Z",
                    "competitions": [{"competitors": [{"team": {"id": "1"}}]}]}
        self.assertIsNone(E.parse_event(one_side))

    def test_missing_statistics_are_tolerated(self):
        bare = json.loads(json.dumps(EVENT))
        for c in bare["competitions"][0]["competitors"]:
            c.pop("statistics")
        match, stats = E.parse_event(bare)
        self.assertTrue(match.played)
        self.assertEqual(stats, [])

    def test_unknown_stat_names_are_ignored(self):
        odd = json.loads(json.dumps(EVENT))
        odd["competitions"][0]["competitors"][0]["statistics"].append(
            {"name": "brandNewMetric", "displayValue": "99"})
        _, stats = E.parse_event(odd)
        home = next(s for s in stats if s.team_id == 17755)
        self.assertEqual(home.shots, 21.0)

    def test_kickoff_is_utc(self):
        match, _ = E.parse_event(EVENT)
        self.assertEqual(match.dt.tzinfo.utcoffset(match.dt).total_seconds(), 0)
        self.assertEqual(match.dt.hour, 16)


class TestInjuryParsing(unittest.TestCase):
    def test_empty_feed(self):
        self.assertEqual(E.parse_injuries({"injuries": []}), [])
        self.assertEqual(E.parse_injuries({}), [])

    def test_populated_feed(self):
        payload = {"injuries": [{
            "team": {"id": "929", "displayName": "Al Hilal"},
            "injuries": [{"status": "Out",
                          "athlete": {"id": "1234", "displayName": "A Player"},
                          "type": {"description": "Hamstring"}}],
        }]}
        rows = E.parse_injuries(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].team_id, 929)
        self.assertEqual(rows[0].player_name, "A Player")
        self.assertEqual(rows[0].reason, "Hamstring")


class TestFetchPipeline(unittest.TestCase):
    class Stub:
        calls_made = 0
        cache_hits = 0

        def __init__(self):
            self.requested = []

        def month(self, year, month):
            self.requested.append((year, month))
            if (year, month) == (2026, 9):
                return {"events": [EVENT]}
            if (year, month) == (2026, 10):
                return {"events": [SCHEDULED]}
            return {"events": []}

        def teams(self):
            return {"sports": [{"leagues": [{"teams": [
                {"team": {"id": "929", "displayName": "Al Hilal"}},
                {"team": {"id": "932", "displayName": "Al Ittihad"}},
            ]}]}]}

        def injuries(self):
            return {"injuries": []}

    def test_builds_a_dataset(self):
        client = self.Stub()
        ds = E.fetch_dataset(client, seasons=[2026], log=lambda *a: None)
        self.assertEqual(len(ds.played_matches), 1)
        self.assertEqual(len(ds.upcoming_matches), 1)
        self.assertEqual(len(ds.stats), 2)
        self.assertIn("ESPN", ds.source)

    def test_looks_beyond_the_current_month(self):
        """Otherwise the snapshot has nothing upcoming to predict."""
        client = self.Stub()
        E.fetch_dataset(client, seasons=[2026], log=lambda *a: None)
        now = datetime.now(timezone.utc)
        self.assertTrue(any(m > (now.year, now.month) for m in client.requested),
                        client.requested)

    def test_duplicate_events_across_months_are_collapsed(self):
        class Dupe(self.Stub):
            def month(self, year, month):
                return {"events": [EVENT]}
        ds = E.fetch_dataset(Dupe(), seasons=[2026], log=lambda *a: None)
        self.assertEqual(len(ds.matches), 1)

    def test_team_list_is_merged_in(self):
        ds = E.fetch_dataset(self.Stub(), seasons=[2026], log=lambda *a: None)
        self.assertEqual(ds.teams[929], "Al Hilal")

    def test_snapshot_round_trips_with_its_source(self):
        ds = E.fetch_dataset(self.Stub(), seasons=[2026], log=lambda *a: None)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            back = Dataset.load(path)
            self.assertIn("ESPN", back.source)
            self.assertEqual(len(back.stats), 2)

    def test_a_failing_month_does_not_abort_the_fetch(self):
        class Flaky(self.Stub):
            def month(self, year, month):
                if (year, month) == (2026, 9):
                    raise E.EspnError("boom")
                return self.__class__.__bases__[0].month(self, year, month)
        ds = E.fetch_dataset(Flaky(), seasons=[2026], log=lambda *a: None)
        self.assertEqual(len(ds.upcoming_matches), 1)


class TestClientCaching(unittest.TestCase):
    def test_offline_without_cache_raises(self):
        with tempfile.TemporaryDirectory() as d:
            c = E.Espn(cache_dir=Path(d), offline=True)
            with self.assertRaises(E.EspnError):
                c.get("scoreboard", {"dates": "202609"})

    def test_cache_hit_avoids_a_call(self):
        import time as _t
        with tempfile.TemporaryDirectory() as d:
            c = E.Espn(cache_dir=Path(d))
            fp = c._cache_path("scoreboard", {"dates": "202609"})
            fp.write_text(json.dumps({"fetched_at": _t.time(),
                                      "payload": {"events": [EVENT]}}))
            got = c.get("scoreboard", {"dates": "202609"})
            self.assertEqual(len(got["events"]), 1)
            self.assertEqual(c.calls_made, 0)
            self.assertEqual(c.cache_hits, 1)

    def test_past_months_get_a_long_ttl_and_the_current_month_a_short_one(self):
        with tempfile.TemporaryDirectory() as d:
            c = E.Espn(cache_dir=Path(d), offline=True)
            seen = {}
            c.get = lambda path, params, ttl: seen.__setitem__(params["dates"], ttl)
            now = datetime.now(timezone.utc)
            c.month(2024, 3)
            c.month(now.year, now.month)
            self.assertGreater(seen["202403"], 86400)
            self.assertLessEqual(seen["%04d%02d" % (now.year, now.month)], 3600)
