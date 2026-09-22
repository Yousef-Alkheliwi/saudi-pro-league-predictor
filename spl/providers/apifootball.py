"""Live-data client for API-Football (api-sports.io / RapidAPI).

Everything the predictor needs comes from this one upstream:

    /leagues              league + season discovery
    /teams                team ids and names
    /fixtures             results, kickoff times, status  -> goals, rest days
    /fixtures/headtohead  previous meetings between two clubs
    /fixtures/statistics  shots, shots on target, possession, corners
    /injuries             current injury/unavailability list
    /players/squads       squad lists
    /players              per-player minutes, used to weight injury importance

The client is deliberately defensive about the free tier's 100 requests/day:
every response is cached on disk with a per-endpoint TTL, and a per-run request
budget stops a runaway loop from burning the daily quota.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..config import CACHE_DIR, PROVIDER, SAUDI_COUNTRY, SAUDI_LEAGUE_ID_HINT, ProviderConfig

#: cache lifetime in seconds, by endpoint. Finished results never really change,
#: so they are kept for a long time; injuries and live fixtures are volatile.
TTL = {
    "leagues": 30 * 86400,
    "teams": 7 * 86400,
    "players/squads": 2 * 86400,
    "players": 2 * 86400,
    "fixtures": 6 * 3600,
    "fixtures/headtohead": 12 * 3600,
    "fixtures/statistics": 365 * 86400,
    "injuries": 3 * 3600,
    "standings": 6 * 3600,
    "status": 3600,
}
DEFAULT_TTL = 6 * 3600


class ApiFootballError(RuntimeError):
    pass


class QuotaExhausted(ApiFootballError):
    """The plan's daily request allowance is spent."""


class PlanRestricted(ApiFootballError):
    """The subscription tier does not cover the season or endpoint requested.

    API-Football's free tier serves only a window of past seasons, so the
    *current* season - and with it injuries, squads and upcoming fixtures - is
    unavailable. This is a plan limit, not a bug, and the fetch pipeline works
    around it by falling back to the newest season the plan does serve.
    """


class MissingCredentials(ApiFootballError):
    def __init__(self) -> None:
        super().__init__(
            "No API key found. Get a free key at https://dashboard.api-football.com "
            "(or via RapidAPI) and either export API_FOOTBALL_KEY=... or put it in "
            "a .env file next to this project:\n\n    API_FOOTBALL_KEY=your_key_here\n"
        )


class ApiFootball:
    def __init__(self, cfg: Optional[ProviderConfig] = None, cache_dir: Optional[Path] = None,
                 offline: bool = False) -> None:
        self.cfg = cfg or PROVIDER
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.calls_made = 0
        self.cache_hits = 0
        #: from x-ratelimit-requests-remaining; None until the first live call
        self.daily_remaining: Optional[int] = None
        self.per_minute_limit: Optional[int] = None
        self._last_call_at = 0.0
        self._session = requests.Session()

    def _throttle(self) -> None:
        """Space calls out so the per-minute limiter is never tripped."""
        wait = self.cfg.min_interval - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def _note_limits(self, headers) -> None:
        for key, attr in (("x-ratelimit-requests-remaining", "daily_remaining"),
                          ("x-ratelimit-limit", "per_minute_limit")):
            raw = headers.get(key)
            if raw is not None:
                try:
                    setattr(self, attr, int(raw))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------ plumbing
    def _cache_path(self, path: str, params: Dict[str, Any]) -> Path:
        key = json.dumps({"p": path, "q": params}, sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        safe = path.replace("/", "_")
        return self.cache_dir / f"{safe}__{digest}.json"

    def _read_cache(self, fp: Path, ttl: float) -> Optional[List[dict]]:
        if not fp.exists():
            return None
        try:
            blob = json.loads(fp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        if self.offline:
            return blob.get("response")
        if time.time() - blob.get("fetched_at", 0) > ttl:
            return None
        return blob.get("response")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            ttl: Optional[float] = None, force: bool = False) -> List[dict]:
        """GET an endpoint, returning the `response` array. Cached on disk."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        fp = self._cache_path(path, params)
        ttl = TTL.get(path, DEFAULT_TTL) if ttl is None else ttl

        if not force:
            cached = self._read_cache(fp, ttl)
            if cached is not None:
                self.cache_hits += 1
                return cached

        if self.offline:
            raise ApiFootballError(
                "offline mode: %s %s is not in the cache. Run `fetch` with a key first."
                % (path, params)
            )
        if not self.cfg.api_key:
            raise MissingCredentials()
        if self.calls_made >= self.cfg.request_budget:
            raise ApiFootballError(
                "per-run request budget of %d upstream calls is exhausted. Raise "
                "API_FOOTBALL_BUDGET if your plan allows it." % self.cfg.request_budget
            )
        if self.daily_remaining is not None and self.daily_remaining <= 0:
            raise QuotaExhausted(
                "the plan's daily request allowance is spent; it resets at "
                "midnight UTC. Cached data still works with --offline."
            )

        url = "%s/%s" % (self.cfg.base_url, path)
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                self._throttle()
                resp = self._session.get(url, headers=self.cfg.headers, params=params,
                                         timeout=self.cfg.timeout)
                self.calls_made += 1
                self._note_limits(resp.headers)
                if resp.status_code == 429:
                    # the per-minute window is 60s, so short backoffs just burn
                    # retries; wait the window out
                    time.sleep(min(65.0, 20.0 * (attempt + 1)))
                    continue
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:  # network / bad json
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            errors = payload.get("errors")
            # the API returns [] for "no errors" and a dict when something is wrong
            if isinstance(errors, dict) and errors:
                msg = "; ".join("%s: %s" % kv for kv in errors.items())
                if "plan" in errors:
                    raise PlanRestricted("API-Football error on /%s: %s" % (path, msg))
                if "token" in errors or "Bug" not in msg:
                    raise ApiFootballError("API-Football error on /%s: %s" % (path, msg))
            response = payload.get("response") or []
            fp.write_text(json.dumps({"fetched_at": time.time(), "path": path,
                                      "params": params, "response": response}),
                          encoding="utf-8")
            return response

        raise ApiFootballError("giving up on /%s after %d attempts: %s"
                               % (path, self.cfg.max_retries, last_err))

    # ------------------------------------------------------------------ endpoints
    def status(self) -> dict:
        res = self.get("status", ttl=600)
        return res if isinstance(res, dict) else (res[0] if res else {})

    def league_id(self) -> int:
        """Resolve the Saudi Pro League id, falling back to the known value."""
        try:
            rows = self.get("leagues", {"country": SAUDI_COUNTRY, "type": "League"})
        except MissingCredentials:
            raise
        except ApiFootballError:
            return SAUDI_LEAGUE_ID_HINT
        best = None
        for row in rows:
            league = row.get("league") or {}
            name = (league.get("name") or "").lower()
            if "pro league" in name or "professional" in name:
                best = league.get("id")
                break
            if best is None and league.get("id") == SAUDI_LEAGUE_ID_HINT:
                best = league.get("id")
        return int(best or SAUDI_LEAGUE_ID_HINT)

    def available_seasons(self, league: int) -> List[int]:
        rows = self.get("leagues", {"id": league})
        out: List[int] = []
        for row in rows:
            for season in row.get("seasons") or []:
                year = season.get("year")
                if isinstance(year, int):
                    out.append(year)
        return sorted(set(out))

    def teams(self, league: int, season: int) -> List[dict]:
        return self.get("teams", {"league": league, "season": season})

    def fixtures(self, league: int, season: int, force: bool = False) -> List[dict]:
        return self.get("fixtures", {"league": league, "season": season}, force=force)

    def fixtures_next(self, league: int, count: int = 20, force: bool = False) -> List[dict]:
        return self.get("fixtures", {"league": league, "next": count},
                        ttl=1800, force=force)

    def team_fixtures(self, team_id: int, season: int,
                      force: bool = False) -> List[dict]:
        """Every competition a club plays in that season (league, cups, AFC)."""
        return self.get("fixtures", {"team": team_id, "season": season},
                        ttl=6 * 3600, force=force)

    def fixture_statistics(self, fixture_id: int) -> List[dict]:
        return self.get("fixtures/statistics", {"fixture": fixture_id})

    def head_to_head(self, home_id: int, away_id: int, last: int = 20) -> List[dict]:
        return self.get("fixtures/headtohead",
                        {"h2h": "%d-%d" % (home_id, away_id), "last": last})

    def injuries(self, league: int, season: int, force: bool = False) -> List[dict]:
        return self.get("injuries", {"league": league, "season": season}, force=force)

    def injuries_for_fixture(self, fixture_id: int) -> List[dict]:
        return self.get("injuries", {"fixture": fixture_id})

    def squad(self, team_id: int) -> List[dict]:
        return self.get("players/squads", {"team": team_id})

    def player_season_stats(self, team_id: int, league: int, season: int,
                            page: int = 1) -> List[dict]:
        return self.get("players", {"team": team_id, "league": league,
                                    "season": season, "page": page})

    def standings(self, league: int, season: int) -> List[dict]:
        return self.get("standings", {"league": league, "season": season})
