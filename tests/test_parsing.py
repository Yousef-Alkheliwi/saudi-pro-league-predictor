"""Parser tests against realistic API-Football v3 payload shapes.

These pin the upstream contract: the documented `type` strings for box-score
statistics, the nesting of /fixtures, /injuries and /players, and graceful
degradation when fields are missing or renamed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spl.data import (Dataset, parse_fixture, parse_injuries, parse_players,
                      parse_statistics)
from spl.providers import ApiFootball, ApiFootballError, MissingCredentials
from spl.config import ProviderConfig

FIXTURE = {
    "fixture": {
        "id": 1208745,
        "referee": "Someone",
        "timezone": "UTC",
        "date": "2026-09-19T17:00:00+03:00",
        "timestamp": 1789740000,
        "venue": {"id": 1, "name": "Kingdom Arena", "city": "Riyadh"},
        "status": {"long": "Match Finished", "short": "FT", "elapsed": 90},
    },
    "league": {"id": 307, "name": "Pro League", "country": "Saudi-Arabia",
               "season": 2026, "round": "Regular Season - 4"},
    "teams": {"home": {"id": 2932, "name": "Al-Hilal Saudi FC", "winner": True},
              "away": {"id": 2939, "name": "Al-Nassr", "winner": False}},
    "goals": {"home": 2, "away": 1},
    "score": {"halftime": {"home": 1, "away": 0}},
}

STATISTICS = [
    {"team": {"id": 2932, "name": "Al-Hilal Saudi FC"},
     "statistics": [
         {"type": "Shots on Goal", "value": 7},
         {"type": "Shots off Goal", "value": 5},
         {"type": "Total Shots", "value": 16},
         {"type": "Blocked Shots", "value": 4},
         {"type": "Fouls", "value": 11},
         {"type": "Corner Kicks", "value": 6},
         {"type": "Ball Possession", "value": "58%"},
         {"type": "Yellow Cards", "value": 2},
         {"type": "Goalkeeper Saves", "value": 3},
         {"type": "expected_goals", "value": "2.14"},
     ]},
    {"team": {"id": 2939, "name": "Al-Nassr"},
     "statistics": [
         {"type": "Shots on Goal", "value": 4},
         {"type": "Total Shots", "value": 9},
         {"type": "Corner Kicks", "value": 3},
         {"type": "Ball Possession", "value": "42%"},
         {"type": "expected_goals", "value": None},
     ]},
]

INJURIES = [
    {"player": {"id": 1001, "name": "A Player", "type": "Missing Fixture",
                "reason": "Injury"},
     "team": {"id": 2932, "name": "Al-Hilal Saudi FC"},
     "fixture": {"id": 1208745, "date": "2026-09-19T17:00:00+03:00"},
     "league": {"id": 307, "season": 2026}},
    {"player": {"id": 1002, "name": "B Player", "type": "Questionable",
                "reason": "Knock"},
     "team": {"id": 2932, "name": "Al-Hilal Saudi FC"},
     "fixture": {"id": 1208745, "date": "2026-09-19T17:00:00+03:00"},
     "league": {"id": 307, "season": 2026}},
    # duplicate of the first, as the API repeats a player across fixtures
    {"player": {"id": 1001, "name": "A Player", "type": "Missing Fixture",
                "reason": "Injury"},
     "team": {"id": 2932, "name": "Al-Hilal Saudi FC"},
     "fixture": {"id": 1208799, "date": "2026-09-26T17:00:00+03:00"},
     "league": {"id": 307, "season": 2026}},
]

PLAYERS = [
    {"player": {"id": 1001, "name": "A Player", "age": 27},
     "statistics": [{"team": {"id": 2932}, "league": {"id": 307, "season": 2026},
                     "games": {"appearences": 5, "minutes": 431, "position": "Attacker",
                               "rating": "7.240000"},
                     "goals": {"total": 4, "assists": 2}}]},
    {"player": {"id": 1002, "name": "B Player", "age": 24},
     "statistics": [
         {"team": {"id": 2932}, "league": {"id": 307, "season": 2026},
          "games": {"appearences": 3, "minutes": 210, "position": "Defender",
                    "rating": None},
          "goals": {"total": None, "assists": None}},
         {"team": {"id": 2932}, "league": {"id": 17, "season": 2026},
          "games": {"appearences": 2, "minutes": 180, "position": "Defender",
                    "rating": "6.8"},
          "goals": {"total": 0, "assists": 1}},
     ]},
]


class TestFixtureParsing(unittest.TestCase):
    def test_core_fields(self):
        m = parse_fixture(FIXTURE)
        self.assertIsNotNone(m)
        self.assertEqual(m.fixture_id, 1208745)
        self.assertEqual((m.home_id, m.away_id), (2932, 2939))
        self.assertEqual((m.home_goals, m.away_goals), (2, 1))
        self.assertEqual(m.status, "FT")
        self.assertTrue(m.played)
        self.assertEqual(m.venue, "Kingdom Arena")
        self.assertEqual(m.league_id, 307)
        self.assertEqual(m.league_name, "Pro League")
        self.assertEqual(m.season, 2026)

    def test_kickoff_is_converted_to_utc(self):
        m = parse_fixture(FIXTURE)
        self.assertEqual(m.dt.hour, 14)          # 17:00 +03:00 -> 14:00 UTC
        self.assertEqual(m.dt.tzinfo.utcoffset(m.dt).total_seconds(), 0)

    def test_scheduled_fixture_is_not_played(self):
        row = json.loads(json.dumps(FIXTURE))
        row["fixture"]["status"] = {"short": "NS"}
        row["goals"] = {"home": None, "away": None}
        m = parse_fixture(row)
        self.assertFalse(m.played)

    def test_postponed_and_abandoned_are_not_played(self):
        for short in ("PST", "CANC", "ABD", "TBD", "SUSP"):
            row = json.loads(json.dumps(FIXTURE))
            row["fixture"]["status"] = {"short": short}
            self.assertFalse(parse_fixture(row).played, short)

    def test_extra_time_and_penalties_count_as_played(self):
        for short in ("AET", "PEN"):
            row = json.loads(json.dumps(FIXTURE))
            row["fixture"]["status"] = {"short": short}
            self.assertTrue(parse_fixture(row).played, short)

    def test_malformed_rows_are_dropped(self):
        self.assertIsNone(parse_fixture({}))
        self.assertIsNone(parse_fixture({"fixture": {"id": 1}, "teams": {}}))
        self.assertIsNone(parse_fixture({"fixture": {}, "teams":
                                        {"home": {"id": 1}, "away": {"id": 2}}}))


class TestStatisticsParsing(unittest.TestCase):
    def test_documented_type_strings_map_across(self):
        rows = parse_statistics(1208745, STATISTICS)
        self.assertEqual(len(rows), 2)
        home = next(r for r in rows if r.team_id == 2932)
        self.assertEqual(home.shots, 16.0)
        self.assertEqual(home.shots_on_target, 7.0)
        self.assertEqual(home.possession, 58.0)      # "58%" -> 58.0
        self.assertEqual(home.corners, 6.0)
        self.assertEqual(home.expected_goals, 2.14)

    def test_missing_and_null_values_become_none(self):
        away = next(r for r in parse_statistics(1, STATISTICS) if r.team_id == 2939)
        self.assertIsNone(away.expected_goals)

    def test_unknown_types_are_ignored(self):
        rows = parse_statistics(1, [{"team": {"id": 5}, "statistics": [
            {"type": "Some New Metric", "value": 3},
            {"type": "Total Shots", "value": 11}]}])
        self.assertEqual(rows[0].shots, 11.0)
        self.assertIsNone(rows[0].possession)

    def test_empty_payload(self):
        self.assertEqual(parse_statistics(1, []), [])
        self.assertEqual(parse_statistics(1, None), [])


class TestInjuryParsing(unittest.TestCase):
    def test_types_and_dedup(self):
        rows = parse_injuries(INJURIES)
        self.assertEqual(len(rows), 2)               # the repeat is collapsed
        self.assertEqual({r.type for r in rows}, {"Missing Fixture", "Questionable"})
        self.assertTrue(all(r.team_id == 2932 for r in rows))

    def test_missing_team_is_skipped(self):
        self.assertEqual(parse_injuries([{"player": {"name": "x"}}]), [])

    def test_unnamed_player_still_counts(self):
        rows = parse_injuries([{"team": {"id": 1}, "player": {}}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].player_name, "unknown")
        self.assertEqual(rows[0].type, "Missing Fixture")


class TestPlayerParsing(unittest.TestCase):
    def test_minutes_and_position(self):
        rows = parse_players(2932, PLAYERS)
        a = next(r for r in rows if r.player_id == 1001)
        self.assertEqual(a.minutes, 431.0)
        self.assertEqual(a.position, "Attacker")
        self.assertEqual(a.goals, 4.0)
        self.assertAlmostEqual(a.rating, 7.24, places=2)

    def test_minutes_sum_across_competitions(self):
        b = next(r for r in parse_players(2932, PLAYERS) if r.player_id == 1002)
        self.assertEqual(b.minutes, 390.0)           # 210 league + 180 AFC
        self.assertEqual(b.appearances, 5.0)

    def test_nulls_do_not_crash(self):
        rows = parse_players(1, [{"player": {"id": 9}, "statistics": [
            {"games": {"appearences": None, "minutes": None, "position": None,
                       "rating": None}, "goals": {}}]}])
        self.assertEqual(rows[0].minutes, 0.0)

    def test_player_without_id_is_dropped(self):
        self.assertEqual(parse_players(1, [{"player": {"name": "x"}}]), [])


class TestClientBehaviour(unittest.TestCase):
    def test_missing_key_raises_a_helpful_error(self):
        with tempfile.TemporaryDirectory() as d:
            client = ApiFootball(cfg=ProviderConfig(api_key=None), cache_dir=Path(d))
            with self.assertRaises(MissingCredentials) as ctx:
                client.get("fixtures", {"league": 307, "season": 2026})
            self.assertIn("API_FOOTBALL_KEY", str(ctx.exception))

    def test_offline_without_cache_raises(self):
        with tempfile.TemporaryDirectory() as d:
            client = ApiFootball(cfg=ProviderConfig(api_key="x"), cache_dir=Path(d),
                                 offline=True)
            with self.assertRaises(ApiFootballError):
                client.get("fixtures", {"league": 307})

    def test_offline_serves_a_stale_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            client = ApiFootball(cfg=ProviderConfig(api_key="x"), cache_dir=cache)
            fp = client._cache_path("fixtures", {"league": 307})
            fp.write_text(json.dumps({"fetched_at": 0.0, "response": [FIXTURE]}))
            offline = ApiFootball(cfg=ProviderConfig(api_key=None), cache_dir=cache,
                                  offline=True)
            rows = offline.get("fixtures", {"league": 307})
            self.assertEqual(len(rows), 1)
            self.assertEqual(offline.calls_made, 0)

    def test_fresh_cache_is_reused_without_a_call(self):
        import time
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            client = ApiFootball(cfg=ProviderConfig(api_key="x"), cache_dir=cache)
            fp = client._cache_path("fixtures", {"league": 307})
            fp.write_text(json.dumps({"fetched_at": time.time(),
                                      "response": [FIXTURE]}))
            rows = client.get("fixtures", {"league": 307})
            self.assertEqual(len(rows), 1)
            self.assertEqual(client.calls_made, 0)
            self.assertEqual(client.cache_hits, 1)

    def test_request_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = ProviderConfig(api_key="x", request_budget=0)
            client = ApiFootball(cfg=cfg, cache_dir=Path(d))
            with self.assertRaises(ApiFootballError) as ctx:
                client.get("fixtures", {"league": 307})
            self.assertIn("budget", str(ctx.exception))

    def test_header_shapes(self):
        direct = ProviderConfig(api_key="k", mode="direct")
        self.assertEqual(direct.headers, {"x-apisports-key": "k"})
        self.assertIn("api-sports.io", direct.base_url)
        rapid = ProviderConfig(api_key="k", mode="rapidapi")
        self.assertEqual(rapid.headers["x-rapidapi-key"], "k")
        self.assertIn("rapidapi.com", rapid.base_url)

    def test_cache_key_separates_different_params(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApiFootball(cfg=ProviderConfig(api_key="x"), cache_dir=Path(d))
            self.assertNotEqual(c._cache_path("fixtures", {"season": 2025}),
                                c._cache_path("fixtures", {"season": 2026}))


class TestFetchPipeline(unittest.TestCase):
    """End-to-end fetch against a stub client, no network."""

    class StubClient:
        def __init__(self):
            self.calls = []

        def league_id(self):
            return 307

        def available_seasons(self, league):
            return [2025, 2026]

        def fixtures(self, league, season, force=False):
            self.calls.append(("fixtures", season))
            row = json.loads(json.dumps(FIXTURE))
            row["league"]["season"] = season
            row["fixture"]["id"] = 1000 + season
            return [row]

        def teams(self, league, season):
            return [{"team": {"id": 2932, "name": "Al-Hilal Saudi FC"}},
                    {"team": {"id": 2939, "name": "Al-Nassr"}}]

        def team_fixtures(self, team_id, season, force=False):
            afc = json.loads(json.dumps(FIXTURE))
            afc["fixture"]["id"] = 5000 + team_id
            afc["league"] = {"id": 17, "name": "AFC Champions League",
                             "season": season, "round": "Group"}
            afc["teams"]["home"] = {"id": team_id, "name": "T"}
            afc["teams"]["away"] = {"id": 9999, "name": "Foreign"}
            return [afc]

        def fixture_statistics(self, fixture_id):
            return STATISTICS

        def injuries(self, league, season, force=False):
            return INJURIES

        #: 26 players per club, so page 1 (20) is not the whole squad
        def player_season_stats(self, team_id, league, season, page=1):
            self.calls.append(("players", team_id, page))
            squad = []
            for i in range(26):
                squad.append({
                    "player": {"id": team_id * 100 + i, "name": "P%d" % i},
                    "statistics": [{"team": {"id": team_id},
                                    "games": {"appearences": 5, "minutes": 400,
                                              "position": "Midfielder",
                                              "rating": "7.0"},
                                    "goals": {"total": 1, "assists": 0}}]})
            start = (page - 1) * 20
            return squad[start:start + 20]

    def test_pipeline_populates_every_section(self):
        from spl.data import fetch_dataset
        client = self.StubClient()
        ds = fetch_dataset(client, seasons=[2025, 2026], stats_limit=5,
                           log=lambda *a: None)
        self.assertEqual(ds.league_id, 307)
        self.assertEqual(len(ds.matches), 2)
        self.assertTrue(ds.other_matches)
        self.assertTrue(all(m.league_id == 17 for m in ds.other_matches))
        self.assertTrue(ds.stats)
        self.assertEqual(len(ds.injuries), 2)
        self.assertTrue(ds.players)
        self.assertIn(2932, ds.teams)
        # the foreign AFC opponent must not become a league team
        self.assertNotIn(9999, ds.teams)

    def test_pipeline_survives_a_failing_endpoint(self):
        from spl.data import fetch_dataset
        client = self.StubClient()
        client.injuries = lambda *a, **k: (_ for _ in ()).throw(
            ApiFootballError("quota exhausted"))
        ds = fetch_dataset(client, seasons=[2026], stats_limit=2,
                           log=lambda *a: None)
        self.assertTrue(ds.matches)          # fixtures still made it through
        self.assertEqual(ds.injuries, [])

    def test_player_pagination_is_walked_to_the_end_of_the_squad(self):
        """A single page would drop 6 of 26 players and inflate the remaining
        players' share of squad minutes, skewing the injury weighting."""
        from spl.data import fetch_dataset
        client = self.StubClient()
        ds = fetch_dataset(client, seasons=[2026], with_stats=False,
                           with_schedule=False, log=lambda *a: None)
        by_team = ds.players_by_team()
        self.assertTrue(by_team)
        for team_id, players in by_team.items():
            self.assertEqual(len(players), 26, team_id)
        pages = [c for c in client.calls if c[0] == "players"]
        self.assertTrue(any(c[2] == 2 for c in pages))     # walked to page 2
        self.assertFalse(any(c[2] == 3 for c in pages))    # stopped at a short page

    def test_missing_credentials_are_not_swallowed(self):
        """A keyless fetch must fail loudly, not quietly produce an empty dataset
        that then overwrites a good snapshot."""
        from spl.data import fetch_dataset
        client = self.StubClient()
        client.fixtures = lambda *a, **k: (_ for _ in ()).throw(MissingCredentials())
        with self.assertRaises(MissingCredentials):
            fetch_dataset(client, seasons=[2026], log=lambda *a: None)

    def test_missing_credentials_from_league_lookup_propagate(self):
        from spl.data import fetch_dataset
        client = self.StubClient()
        client.available_seasons = lambda *a, **k: (_ for _ in ()).throw(
            MissingCredentials())
        with self.assertRaises(MissingCredentials):
            fetch_dataset(client, seasons=[2026], log=lambda *a: None)

    def test_cli_refuses_to_save_an_empty_dataset(self):
        """Regression: `fetch` without a key wrote 0 matches over the snapshot
        and still exited 0."""
        from spl import cli
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            good = Dataset(league_id=307, fetched_at="now", teams={1: "A"},
                           matches=[parse_fixture(FIXTURE)])
            good.save(path)
            args = cli.build_parser().parse_args(
                ["--dataset", str(path), "fetch", "--source", "api-football",
                 "--no-stats", "--no-players", "--no-schedule"])
            import spl.data as data_mod
            original = data_mod.fetch_dataset
            data_mod.fetch_dataset = lambda *a, **k: Dataset(
                league_id=307, fetched_at="now")
            cli_fetch = cli.cmd_fetch
            try:
                # cli imported fetch_dataset by name, so patch it there too
                cli.fetch_dataset = data_mod.fetch_dataset
                self.assertEqual(cli_fetch(args), 1)
            finally:
                data_mod.fetch_dataset = original
                cli.fetch_dataset = original
            self.assertEqual(len(Dataset.load(path).matches), 1)   # untouched

    def test_saved_snapshot_is_loadable(self):
        from spl.data import fetch_dataset
        ds = fetch_dataset(self.StubClient(), seasons=[2026], stats_limit=2,
                           log=lambda *a: None)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            back = Dataset.load(path)
            self.assertEqual(len(back.matches), len(ds.matches))
            self.assertEqual(len(back.other_matches), len(ds.other_matches))


if __name__ == "__main__":
    unittest.main()


class TestRateLimiting(unittest.TestCase):
    """The free tier allows 10 calls/minute; exceeding it 429-storms."""

    def _client(self, cache, interval=0.0, **kw):
        cfg = ProviderConfig(api_key="x", min_interval=interval, **kw)
        return ApiFootball(cfg=cfg, cache_dir=Path(cache))

    def test_throttle_spaces_calls_out(self):
        import time
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d, interval=0.25)
            start = time.monotonic()
            c._throttle()
            c._throttle()
            c._throttle()
            # first call is immediate, the next two each wait one interval
            self.assertGreaterEqual(time.monotonic() - start, 0.5)

    def test_throttle_does_not_delay_a_cache_hit(self):
        import json as _json
        import time
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d, interval=5.0)
            fp = c._cache_path("fixtures", {"league": 307})
            fp.write_text(_json.dumps({"fetched_at": time.time(),
                                       "response": [FIXTURE]}))
            start = time.monotonic()
            c.get("fixtures", {"league": 307})
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertEqual(c.calls_made, 0)

    def test_limits_are_read_from_headers(self):
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d)
            c._note_limits({"x-ratelimit-requests-remaining": "83",
                            "x-ratelimit-limit": "10"})
            self.assertEqual(c.daily_remaining, 83)
            self.assertEqual(c.per_minute_limit, 10)

    def test_garbage_headers_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d)
            c._note_limits({"x-ratelimit-requests-remaining": "unknown"})
            self.assertIsNone(c.daily_remaining)

    def test_exhausted_daily_quota_stops_before_calling(self):
        from spl.providers import QuotaExhausted
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d)
            c.daily_remaining = 0
            with self.assertRaises(QuotaExhausted):
                c.get("fixtures", {"league": 307})
            self.assertEqual(c.calls_made, 0)     # no doomed request was sent

    def test_quota_exhausted_is_an_api_error(self):
        """So existing `except ApiFootballError` handlers keep working."""
        from spl.providers import QuotaExhausted
        self.assertTrue(issubclass(QuotaExhausted, ApiFootballError))

    def test_remaining_quota_above_zero_still_calls(self):
        with tempfile.TemporaryDirectory() as d:
            c = self._client(d)
            c.daily_remaining = 1
            c.cfg.request_budget = 0          # stop it before the network
            with self.assertRaises(ApiFootballError) as ctx:
                c.get("fixtures", {"league": 307})
            self.assertIn("budget", str(ctx.exception))


class TestPlanRestriction(unittest.TestCase):
    """A subscription that blocks recent seasons must degrade to the newest
    season it does serve, not lose the team list, squads and injuries entirely.
    """

    class RestrictedClient(TestFetchPipeline.StubClient):
        #: mirrors the free tier: seasons after 2024 are refused
        ALLOWED = {2022, 2023, 2024}

        def _guard(self, season):
            if season not in self.ALLOWED:
                from spl.providers import PlanRestricted
                raise PlanRestricted(
                    "API-Football error on /fixtures: plan: Free plans do not "
                    "have access to this season, try from 2022 to 2024.")

        def available_seasons(self, league):
            return [2022, 2023, 2024, 2025, 2026]

        def fixtures(self, league, season, force=False):
            self._guard(season)
            self.calls.append(("fixtures", season))
            row = json.loads(json.dumps(FIXTURE))
            row["league"]["season"] = season
            row["fixture"]["id"] = 1000 + season
            return [row]

        def teams(self, league, season):
            self._guard(season)
            return [{"team": {"id": 2932, "name": "Al-Hilal Saudi FC"}}]

        def injuries(self, league, season, force=False):
            self._guard(season)
            return INJURIES

        def player_season_stats(self, team_id, league, season, page=1):
            self._guard(season)
            return TestFetchPipeline.StubClient.player_season_stats(
                self, team_id, league, season, page)

    def _fetch(self, **kw):
        from spl.data import fetch_dataset
        client = self.RestrictedClient()
        opts = dict(seasons=[2022, 2023, 2024, 2025, 2026], with_stats=False,
                    with_schedule=False, log=lambda *a: None)
        opts.update(kw)
        return fetch_dataset(client, **opts), client

    def test_allowed_seasons_still_load(self):
        ds, _ = self._fetch()
        self.assertEqual(sorted({m.season for m in ds.matches}), [2022, 2023, 2024])

    def test_blocked_seasons_do_not_abort_the_fetch(self):
        ds, _ = self._fetch()
        self.assertEqual(len(ds.matches), 3)

    def test_downstream_stages_use_the_newest_served_season(self):
        """Regression: these asked for the calendar-current season and so all
        failed together on the free tier."""
        ds, _ = self._fetch(with_players=True)
        self.assertIn(2932, ds.teams)          # team list survived
        self.assertTrue(ds.injuries)           # injuries survived
        self.assertTrue(ds.players)            # squads survived

    def test_limitation_is_recorded_on_the_dataset(self):
        ds, _ = self._fetch()
        self.assertTrue(ds.plan_limited)
        self.assertEqual(ds.newest_available_season, 2024)

    def test_limitation_survives_a_round_trip(self):
        ds, _ = self._fetch()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.json"
            ds.save(path)
            back = Dataset.load(path)
            self.assertTrue(back.plan_limited)
            self.assertEqual(back.newest_available_season, 2024)

    def test_an_unrestricted_plan_is_not_flagged(self):
        from spl.data import fetch_dataset
        client = TestFetchPipeline.StubClient()
        ds = fetch_dataset(client, seasons=[2026], with_stats=False,
                           with_schedule=False, log=lambda *a: None)
        self.assertFalse(ds.plan_limited)

    def test_plan_restricted_is_an_api_error(self):
        from spl.providers import PlanRestricted
        self.assertTrue(issubclass(PlanRestricted, ApiFootballError))


class TestRoundTripCompleteness(unittest.TestCase):
    """Introspects the dataclasses rather than listing fields, so adding a
    field without serialising it fails here instead of silently losing data."""

    @classmethod
    def setUpClass(cls):
        import dataclasses
        from spl.data import Dataset as DS, Injury, Player, TeamStats
        from tests.synthetic import make_dataset
        cls.dataclasses = dataclasses
        ds, _ = make_dataset(seed=2)
        ds.logos = {sorted(ds.teams)[0]: "data:image/png;base64,AAAA"}
        ds.source = "test source"
        ds.plan_limited = True
        ds.newest_available_season = 2024
        ds.injuries.append(Injury(sorted(ds.teams)[0], 1, "P", "Missing Fixture", "x"))
        cls.original = ds
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "ds.json"
        ds.save(path)
        cls.blob = json.loads(path.read_text())
        cls.restored = DS.load(path)
        cls.row_types = {"matches": ds.matches, "other_matches": ds.other_matches,
                         "stats": ds.stats, "injuries": ds.injuries,
                         "players": ds.players}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_dataset_field_is_written(self):
        names = [f.name for f in self.dataclasses.fields(self.original)]
        self.assertEqual([n for n in names if n not in self.blob], [])

    def test_no_dataset_field_changes_on_the_way_back(self):
        changed = []
        for f in self.dataclasses.fields(self.original):
            a = getattr(self.original, f.name)
            b = getattr(self.restored, f.name)
            if isinstance(a, (list, dict)):
                if len(a) != len(b):
                    changed.append(f.name)
            elif a != b:
                changed.append(f.name)
        self.assertEqual(changed, [])

    def test_every_record_field_is_written(self):
        for key, rows in self.row_types.items():
            if not rows:
                continue
            names = [f.name for f in self.dataclasses.fields(rows[0])]
            written = (self.blob.get(key) or [{}])[0]
            self.assertEqual([n for n in names if n not in written], [],
                             "%s loses fields" % key)

    def test_scalar_values_survive_exactly(self):
        self.assertEqual(self.restored.source, "test source")
        self.assertTrue(self.restored.plan_limited)
        self.assertEqual(self.restored.newest_available_season, 2024)
        self.assertEqual(self.restored.logos, self.original.logos)

    def test_an_older_snapshot_without_the_newer_fields_still_loads(self):
        from spl.data import Dataset as DS
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.json"
            legacy = {k: self.blob[k] for k in
                      ("league_id", "fetched_at", "teams", "matches", "stats",
                       "injuries", "players")}
            path.write_text(json.dumps(legacy))
            old = DS.load(path)
            self.assertEqual(old.logos, {})
            self.assertFalse(old.plan_limited)
            self.assertEqual(old.other_matches, [])
