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

    def test_an_all_zero_box_score_is_treated_as_missing(self):
        """ESPN's placeholder for a match it did not cover: 0% possession for
        both sides. Kept, it made a promoted club look like one that neither
        took nor conceded shots."""
        blank = json.loads(json.dumps(EVENT))
        for c in blank["competitions"][0]["competitors"]:
            for item in c["statistics"]:
                item["displayValue"] = "0"
        match, stats = E.parse_event(blank)
        self.assertTrue(match.played)
        self.assertEqual(stats, [])

    def test_a_real_zero_is_kept(self):
        """A side can fail to manage a shot; that is data, not a gap."""
        shut_out = json.loads(json.dumps(EVENT))
        for item in shut_out["competitions"][0]["competitors"][1]["statistics"]:
            if item["name"] in ("totalShots", "shotsOnTarget"):
                item["displayValue"] = "0"
        _, stats = E.parse_event(shut_out)
        away = next(s for s in stats if s.team_id == 7712)
        self.assertEqual((away.shots, away.possession), (0.0, 37.5))

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

    def test_a_settled_month_needs_a_copy_fetched_after_it_settled(self):
        """A past month gets a long TTL only for a copy fetched after the month
        ended; an open month gets a short TTL and no such condition."""
        from datetime import timedelta
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 11, 20, tzinfo=timezone.utc)
            c = E.Espn(cache_dir=Path(d), offline=True, clock=lambda: now)
            seen = {}
            c.get = lambda path, params, ttl, not_before=None: seen.__setitem__(
                params["dates"], (ttl, not_before))
            c.month(2024, 3)
            c.month(2026, 11)
            ttl_past, gate_past = seen["202403"]
            self.assertGreater(ttl_past, 86400)
            self.assertEqual(gate_past, (E._month_end(2024, 3)
                                         + timedelta(days=E.FINAL_GRACE_DAYS)).timestamp())
            ttl_open, gate_open = seen["202611"]
            self.assertLessEqual(ttl_open, 3600)
            self.assertIsNone(gate_open)


class TestBadges(unittest.TestCase):
    """Club crests are cosmetic and must never break a data fetch."""

    def test_client_without_a_cache_returns_no_badges(self):
        self.assertEqual(E.fetch_logos(TestFetchPipeline.Stub(), {1, 2},
                                       log=lambda *a: None), {})

    def test_fetch_survives_a_badge_failure(self):
        class Boom(TestFetchPipeline.Stub):
            cache_dir = None

            def teams(self):
                raise RuntimeError("badge service down")
        ds = E.fetch_dataset(Boom(), seasons=[2026], log=lambda *a: None)
        self.assertTrue(ds.matches)
        self.assertEqual(ds.logos, {})

    def test_badges_are_read_off_the_match_listings(self):
        listed = json.loads(json.dumps(EVENT))
        listed["competitions"][0]["competitors"][0]["team"]["logo"] = (
            "https://a.espncdn.com/i/teamlogos/soccer/500/17755.png")
        self.assertEqual(E.event_badges(listed),
                         {17755: "https://a.espncdn.com/i/teamlogos/soccer/500/17755.png"})
        self.assertEqual(E.event_badges({}), {})
        self.assertEqual(E.event_badges({"competitions": "junk"}), {})

    def test_a_badge_the_team_list_lacks_comes_from_the_listings(self):
        """ESPN's team list has a null badge for Al Faisaly, so the page drew
        initials for a club whose every fixture carries a badge URL."""
        with tempfile.TemporaryDirectory() as d:
            class Client:
                cache_dir, timeout = Path(d), 5

                def teams(self):
                    return {"sports": [{"leagues": [{"teams": [
                        {"team": {"id": "21446", "logo": None, "logos": None}}]}]}]}
            (Path(d) / "logos").mkdir()
            (Path(d) / "logos" / "21446.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            got = E.fetch_logos(Client(), {21446}, log=lambda *a: None)
            self.assertEqual(got, {})
            got = E.fetch_logos(Client(), {21446}, log=lambda *a: None,
                                fallback={21446: "https://example.invalid/21446.png"})
            self.assertTrue(got[21446].startswith("data:image/png;base64,"))

    def test_badges_can_be_switched_off(self):
        ds = E.fetch_dataset(TestFetchPipeline.Stub(), seasons=[2026],
                             with_logos=False, log=lambda *a: None)
        self.assertEqual(ds.logos, {})

    def test_logos_survive_a_save_load_round_trip(self):
        ds = E.fetch_dataset(TestFetchPipeline.Stub(), seasons=[2026],
                             with_logos=False, log=lambda *a: None)
        ds.logos = {929: "data:image/png;base64,AAAA"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            back = Dataset.load(path)
            self.assertEqual(back.logos[929], "data:image/png;base64,AAAA")

    def test_shrink_returns_input_when_it_cannot_resize(self):
        raw = b"not really a png"
        self.assertIsInstance(E._shrink_png(raw), bytes)


class TestMalformedFeed(unittest.TestCase):
    """Every field here comes from a third-party feed. A malformed row must be
    skipped, never raise - `fetch_dataset` parses events in a loop, so one bad
    row raising would have cost the whole snapshot."""

    MUTATIONS = (object(), None, "", 0, [], {}, "xxx", -1, 1e300, "\u0000",
                 True, [[]], {"a": 1}, "2026-13-45T99:99Z")

    @staticmethod
    def _paths(obj, prefix=()):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield prefix + (k,)
                for p in TestMalformedFeed._paths(v, prefix + (k,)):
                    yield p
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield prefix + (i,)
                for p in TestMalformedFeed._paths(v, prefix + (i,)):
                    yield p

    def test_no_single_field_mutation_can_raise(self):
        import copy
        deleted = self.MUTATIONS[0]
        checked = 0
        for path in list(self._paths(EVENT)):
            for val in self.MUTATIONS:
                payload = copy.deepcopy(EVENT)
                cur = payload
                try:
                    for k in path[:-1]:
                        cur = cur[k]
                    if val is deleted:
                        cur.pop(path[-1], None) if isinstance(cur, dict) \
                            else cur.pop(path[-1])
                    else:
                        cur[path[-1]] = val
                except Exception:
                    continue
                checked += 1
                try:
                    E.parse_event(payload)
                except Exception as exc:
                    self.fail("%s=%r raised %s: %s"
                              % (".".join(map(str, path)), val,
                                 type(exc).__name__, exc))
        self.assertGreater(checked, 200)

    def test_a_valid_event_is_unaffected_by_the_hardening(self):
        match, stats = E.parse_event(EVENT)
        self.assertEqual((match.home_goals, match.away_goals), (4, 2))
        self.assertEqual(len(stats), 2)

    def test_an_unusable_id_is_dropped_not_guessed(self):
        import copy
        for value in ("not-a-number", None, [], {}):
            payload = copy.deepcopy(EVENT)
            payload["id"] = value
            self.assertIsNone(E.parse_event(payload), "id=%r" % (value,))

    def test_an_unusable_date_is_dropped_only_when_both_are_bad(self):
        """An empty event date falls back to the competition's, by design."""
        import copy
        payload = copy.deepcopy(EVENT)
        payload["date"] = ""
        self.assertIsNotNone(E.parse_event(payload))     # fallback still works
        payload["competitions"][0]["date"] = ""
        self.assertIsNone(E.parse_event(payload))        # nothing left to use
        payload["date"] = "nonsense"
        self.assertIsNone(E.parse_event(payload))

    def test_scalar_where_a_list_belongs(self):
        import copy
        for path, val in ((("competitions",), 5),
                          (("competitions", 0, "competitors"), "x"),
                          (("competitions", 0, "competitors", 0, "statistics"), 7)):
            payload = copy.deepcopy(EVENT)
            cur = payload
            for k in path[:-1]:
                cur = cur[k]
            cur[path[-1]] = val
            E.parse_event(payload)      # must not raise

    def test_injuries_tolerate_junk(self):
        for junk in (None, {}, {"injuries": None}, {"injuries": 5},
                     {"injuries": [None]}, {"injuries": [{"team": "x"}]},
                     {"injuries": [{"team": {"id": "1"}, "injuries": 3}]}):
            self.assertIsInstance(E.parse_injuries(junk), list)


class TestMonthFinality(unittest.TestCase):
    """A month's scoreboard is only final once fetched after the month ended.
    Deciding by whether the month is past *now* froze any copy fetched mid-month
    for a year, so late-month results were never picked up."""

    @staticmethod
    def _event(completed, fid="555", date="2026-10-25T18:00Z"):
        return {"id": fid, "date": date,
                "status": {"type": {"completed": completed, "description": "x"}},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "score": "3" if completed else "0",
                     "team": {"id": "929", "displayName": "Al Hilal"}},
                    {"homeAway": "away", "score": "1" if completed else "0",
                     "team": {"id": "817", "displayName": "Al Nassr"}}]}]}

    def _read(self, fetched, now, offline=False):
        """Cache an in-progress October, then read it at `now`."""
        with tempfile.TemporaryDirectory() as d:
            c = E.Espn(cache_dir=Path(d), min_interval=0, offline=offline,
                       clock=lambda: now)
            c._cache_path("scoreboard", {"dates": "202610"}).write_text(json.dumps(
                {"fetched_at": fetched.timestamp(),
                 "payload": {"events": [self._event(False)]}}))
            calls = []
            event = self._event

            class Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"events": [event(True)]}
            c._session.get = lambda *a, **k: (calls.append(1), Resp())[1]
            match, _ = E.parse_event(c.month(2026, 10)["events"][0])
            return len(calls), match.played

    def test_a_copy_fetched_mid_month_is_refetched_after_the_month(self):
        calls, played = self._read(datetime(2026, 10, 20, tzinfo=timezone.utc),
                                   datetime(2026, 11, 5, tzinfo=timezone.utc))
        self.assertEqual(calls, 1)
        self.assertTrue(played)

    def test_inside_the_grace_window_the_month_is_still_open(self):
        calls, _ = self._read(datetime(2026, 10, 20, tzinfo=timezone.utc),
                              datetime(2026, 11, 2, tzinfo=timezone.utc))
        self.assertEqual(calls, 1)

    def test_a_copy_fetched_after_settling_is_trusted(self):
        calls, _ = self._read(datetime(2026, 11, 6, tzinfo=timezone.utc),
                              datetime(2026, 12, 1, tzinfo=timezone.utc))
        self.assertEqual(calls, 0)

    def test_offline_mode_still_uses_whatever_is_cached(self):
        calls, played = self._read(datetime(2026, 10, 20, tzinfo=timezone.utc),
                                   datetime(2026, 11, 5, tzinfo=timezone.utc),
                                   offline=True)
        self.assertEqual(calls, 0)
        self.assertFalse(played)

    def test_month_end_rolls_the_year(self):
        self.assertEqual(E._month_end(2026, 12),
                         datetime(2027, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(E._month_end(2026, 2),
                         datetime(2026, 3, 1, tzinfo=timezone.utc))


class TestDuplicateListings(unittest.TestCase):
    """The same fixture id can appear in more than one month - postponed at its
    original date, played at its new one. The first listing used to win, so the
    real result and its box score were discarded."""

    @staticmethod
    def _ev(fid, date, completed):
        return {"id": fid, "date": date,
                "status": {"type": {"completed": completed, "description": "x"}},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "score": "2" if completed else "0",
                     "team": {"id": "929", "displayName": "Al Hilal"},
                     "statistics": ([{"name": "totalShots", "displayValue": "15"}]
                                    if completed else [])},
                    {"homeAway": "away", "score": "0",
                     "team": {"id": "817", "displayName": "Al Nassr"},
                     "statistics": ([{"name": "totalShots", "displayValue": "7"}]
                                    if completed else [])}]}]}

    def _fetch(self, by_month):
        class Stub:
            calls_made = cache_hits = 0

            def month(self, y, m):
                return {"events": by_month.get((y, m), [])}

            def teams(self):
                return {}

            def injuries(self):
                return {"injuries": []}
        return E.fetch_dataset(Stub(), seasons=[2025], with_logos=False,
                               log=lambda *a: None)

    def test_a_result_in_a_later_month_beats_an_earlier_postponement(self):
        ds = self._fetch({(2025, 9): [self._ev("777", "2025-09-20T18:00Z", False)],
                          (2025, 10): [self._ev("777", "2025-10-15T18:00Z", True)]})
        kept = [m for m in ds.matches if m.fixture_id == 777]
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].played)
        self.assertEqual(kept[0].dt.month, 10)
        self.assertEqual(len([s for s in ds.stats if s.fixture_id == 777]), 2)

    def test_an_unplayed_listing_never_overwrites_a_result(self):
        ds = self._fetch({(2025, 9): [self._ev("777", "2025-09-20T18:00Z", True)],
                          (2025, 10): [self._ev("777", "2025-10-15T18:00Z", False)]})
        kept = [m for m in ds.matches if m.fixture_id == 777]
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].played)

    def test_a_rescheduled_fixture_carries_its_new_date(self):
        ds = self._fetch({(2025, 9): [self._ev("777", "2025-09-20T18:00Z", False)],
                          (2025, 10): [self._ev("777", "2025-10-15T18:00Z", False)]})
        kept = [m for m in ds.matches if m.fixture_id == 777]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].dt.month, 10)

    def test_box_scores_are_never_duplicated(self):
        ds = self._fetch({(2025, 9): [self._ev("777", "2025-09-20T18:00Z", True)],
                          (2025, 10): [self._ev("777", "2025-09-20T18:00Z", True)]})
        self.assertEqual(len([s for s in ds.stats if s.fixture_id == 777]), 2)


class TestSnapshotTime(unittest.TestCase):
    """An offline re-parse used to stamp the snapshot with the current time,
    so week-old data was labelled as fetched today."""

    def _stub_client(self, d, fetched, now, offline):
        c = E.Espn(cache_dir=Path(d), min_interval=0, offline=offline,
                   clock=lambda: now)
        for (y, m) in E.season_months(2026):
            c._cache_path("scoreboard", {"dates": "%04d%02d" % (y, m)}).write_text(
                json.dumps({"fetched_at": fetched.timestamp(),
                            "payload": {"events": [EVENT] if (y, m) == (2026, 9) else []}}))
        for path in ("teams", "injuries"):
            c._cache_path(path, {}).write_text(json.dumps(
                {"fetched_at": fetched.timestamp(), "payload": {}}))
        return c

    def test_an_offline_reparse_keeps_the_original_fetch_time(self):
        fetched = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
        now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            c = self._stub_client(d, fetched, now, offline=True)
            ds = E.fetch_dataset(c, seasons=[2026], with_logos=False,
                                 log=lambda *a: None)
        self.assertEqual(ds.fetched_at, fetched.isoformat(timespec="seconds"))

    def test_a_real_fetch_is_stamped_with_its_own_time(self):
        c = E.Espn(cache_dir=Path(tempfile.mkdtemp()), min_interval=0,
                   clock=lambda: datetime(2026, 10, 1, tzinfo=timezone.utc))
        c._note(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
        c._note(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())
        self.assertEqual(c.newest_fetch,
                         datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())


if __name__ == "__main__":
    unittest.main()
