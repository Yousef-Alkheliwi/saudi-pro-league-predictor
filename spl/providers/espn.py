"""Free live-data provider: ESPN's public JSON endpoints.

No API key, no daily quota, and — unlike the paid feeds' free tiers — the
*current* season is available. One request per calendar month returns every
fixture in that month together with its box score, so possession, shots,
shots on target and corners cost nothing extra.

    /soccer/ksa.1/scoreboard?dates=YYYYMM   fixtures + results + team stats
    /soccer/ksa.1/teams                     the clubs
    /soccer/ksa.1/injuries                  injury list (empty for this league)

Coverage runs from the 2022-23 season to the present. Date *ranges*
(`YYYYMMDD-YYYYMMDD`) are rejected by the endpoint; `YYYYMM` is the granularity
that works, which is why the fetch walks month by month.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from ..config import CACHE_DIR
from ..data import Dataset, Injury, Match, TeamStats

BASE = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
LEAGUE = "ksa.1"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: earliest season ESPN carries for this league
EARLIEST_SEASON = 2022

#: how many months past the current one to request, so scheduled fixtures -
#: the ones worth predicting - are in the snapshot
LOOKAHEAD_MONTHS = 2

#: ESPN stat name -> our TeamStats field
STAT_MAP = {
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "possessionPct": "possession",
    "wonCorners": "corners",
}

#: a finished match, by ESPN's status description
FINISHED_STATES = {"Full Time", "FT", "Final", "After Extra Time", "Penalties"}


class EspnError(RuntimeError):
    pass


class Espn:
    """Cached client. Months that are fully in the past never change, so they
    are cached for a year; the current month is refreshed hourly."""

    def __init__(self, cache_dir: Optional[Path] = None, offline: bool = False,
                 min_interval: float = 0.4, timeout: float = 25.0,
                 max_retries: int = 3) -> None:
        self.cache_dir = Path(cache_dir or (CACHE_DIR / "espn"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls_made = 0
        self.cache_hits = 0
        self._last_call = 0.0
        self._session = requests.Session()

    # ---------------------------------------------------------------- plumbing
    def _cache_path(self, path: str, params: Dict[str, str]) -> Path:
        key = json.dumps({"p": path, "q": params}, sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / ("%s__%s.json" % (path.replace("/", "_"), digest))

    def get(self, path: str, params: Optional[Dict[str, str]] = None,
            ttl: float = 3600.0) -> dict:
        params = params or {}
        fp = self._cache_path(path, params)
        if fp.exists():
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
                if self.offline or time.time() - blob.get("fetched_at", 0) < ttl:
                    self.cache_hits += 1
                    return blob["payload"]
            except (ValueError, OSError, KeyError):
                pass
        if self.offline:
            raise EspnError("offline: %s %s not cached" % (path, params))

        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        url = "%s/%s/%s" % (BASE, LEAGUE, path)
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._last_call = time.monotonic()
                resp = self._session.get(url, params=params, timeout=self.timeout,
                                         headers={"User-Agent": USER_AGENT})
                self.calls_made += 1
                if resp.status_code == 429:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                time.sleep(1.0 * (attempt + 1))
                continue
            if isinstance(payload, dict) and payload.get("code") and payload.get("message"):
                raise EspnError("ESPN rejected /%s %s: %s"
                                % (path, params, payload["message"]))
            fp.write_text(json.dumps({"fetched_at": time.time(), "payload": payload}),
                          encoding="utf-8")
            return payload
        raise EspnError("giving up on /%s after %d attempts: %s"
                        % (path, self.max_retries, last))

    # ---------------------------------------------------------------- endpoints
    def month(self, year: int, month: int, ttl: Optional[float] = None) -> dict:
        now = datetime.now(timezone.utc)
        current = (year, month) >= (now.year, now.month)
        if ttl is None:
            ttl = 3600.0 if current else 365 * 86400.0
        return self.get("scoreboard", {"dates": "%04d%02d" % (year, month)}, ttl=ttl)

    def teams(self) -> dict:
        return self.get("teams", ttl=7 * 86400.0)

    def injuries(self) -> dict:
        return self.get("injuries", ttl=3 * 3600.0)


# -------------------------------------------------------------------- parsing
def season_of(dt: datetime) -> int:
    """Seasons are labelled by the year they open in; the Saudi season starts
    in August, so January-July belongs to the previous year's season."""
    return dt.year if dt.month >= 8 else dt.year - 1


def _parse_dt(text: str) -> datetime:
    raw = (text or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        dt = datetime.strptime(raw[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _num(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def parse_event(event: dict) -> Optional[Tuple[Match, List[TeamStats]]]:
    """One ESPN event -> a Match plus whatever box score came with it."""
    comps = event.get("competitions") or []
    if not comps or not event.get("id"):
        return None
    comp = comps[0]
    sides = comp.get("competitors") or []
    if len(sides) != 2:
        return None

    home = next((s for s in sides if s.get("homeAway") == "home"), None)
    away = next((s for s in sides if s.get("homeAway") == "away"), None)
    if home is None or away is None:
        home, away = sides[0], sides[1]

    def team_id(side) -> Optional[int]:
        raw = (side.get("team") or {}).get("id") or side.get("id")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    hid, aid = team_id(home), team_id(away)
    if hid is None or aid is None or hid == aid:
        return None

    status = ((event.get("status") or comp.get("status") or {}).get("type") or {})
    completed = bool(status.get("completed"))
    short = status.get("shortDetail") or status.get("description") or ""
    hg = _num(home.get("score"))
    ag = _num(away.get("score"))
    if not completed or hg is None or ag is None:
        hg = ag = None

    dt = _parse_dt(event.get("date") or comp.get("date") or "")
    fixture_id = int(event["id"])
    match = Match(
        fixture_id=fixture_id,
        season=season_of(dt),
        kickoff=dt.isoformat(),
        home_id=hid, away_id=aid,
        home_name=(home.get("team") or {}).get("displayName") or str(hid),
        away_name=(away.get("team") or {}).get("displayName") or str(aid),
        # the rest of the codebase keys "played" off these codes
        status="FT" if completed else "NS",
        home_goals=int(hg) if hg is not None else None,
        away_goals=int(ag) if ag is not None else None,
        venue=((comp.get("venue") or {}).get("fullName")),
        round=(event.get("week") or {}).get("text") if isinstance(event.get("week"), dict) else None,
        league_id=None, league_name="Saudi Pro League",
    )

    stats: List[TeamStats] = []
    for side, tid in ((home, hid), (away, aid)):
        raw = side.get("statistics") or []
        if not raw:
            continue
        rec = TeamStats(fixture_id=fixture_id, team_id=tid)
        found = False
        for item in raw:
            field = STAT_MAP.get(item.get("name"))
            if field:
                val = _num(item.get("displayValue", item.get("value")))
                if val is not None:
                    setattr(rec, field, val)
                    found = True
        if found:
            stats.append(rec)
    return match, stats


def parse_injuries(payload: dict) -> List[Injury]:
    out: List[Injury] = []
    for block in payload.get("injuries") or []:
        team = block.get("team") or {}
        try:
            tid = int(team.get("id"))
        except (TypeError, ValueError):
            continue
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            out.append(Injury(
                team_id=tid,
                player_id=int(athlete["id"]) if str(athlete.get("id", "")).isdigit() else None,
                player_name=athlete.get("displayName") or "unknown",
                type=item.get("status") or "Missing Fixture",
                reason=(item.get("type") or {}).get("description") or item.get("status")
                       or "unknown",
            ))
    return out


# -------------------------------------------------------------------- fetching
def _months_ahead(now: datetime, n: int) -> Tuple[int, int]:
    total = now.year * 12 + (now.month - 1) + n
    return total // 12, total % 12 + 1


def season_months(season: int) -> List[Tuple[int, int]]:
    """August of the opening year through July of the next."""
    return ([(season, m) for m in range(8, 13)]
            + [(season + 1, m) for m in range(1, 8)])


def fetch_dataset(client: Espn, seasons: Optional[Iterable[int]] = None,
                  log=print) -> Dataset:
    now = datetime.now(timezone.utc)
    current = season_of(now)
    if seasons is None:
        seasons = range(max(EARLIEST_SEASON, current - 3), current + 1)
    seasons = sorted(set(seasons))

    ds = Dataset(league_id=0,
                 fetched_at=now.isoformat(timespec="seconds"))
    ds.source = "ESPN (site.web.api.espn.com), soccer/%s" % LEAGUE

    seen: set = set()
    for season in seasons:
        added = stats_added = 0
        for year, month in season_months(season):
            # look a couple of months ahead so upcoming fixtures are picked up;
            # stopping at the current month leaves nothing to predict
            if (year, month) > _months_ahead(now, LOOKAHEAD_MONTHS):
                break
            try:
                payload = client.month(year, month)
            except EspnError as exc:
                log("  %04d-%02d unavailable: %s" % (year, month, exc))
                continue
            for event in payload.get("events") or []:
                parsed = parse_event(event)
                if parsed is None:
                    continue
                match, stats = parsed
                if match.fixture_id in seen:
                    continue
                seen.add(match.fixture_id)
                ds.matches.append(match)
                ds.stats.extend(stats)
                ds.teams.setdefault(match.home_id, match.home_name)
                ds.teams.setdefault(match.away_id, match.away_name)
                added += 1
                stats_added += len(stats)
        log("season %s: %d fixtures, %d box-score rows" % (season, added, stats_added))

    try:
        payload = client.teams()
        blocks = (payload.get("sports") or [{}])[0].get("leagues") or [{}]
        for row in (blocks[0].get("teams") or []):
            team = row.get("team") or {}
            if str(team.get("id", "")).isdigit():
                ds.teams[int(team["id"])] = team.get("displayName") or str(team["id"])
    except EspnError as exc:
        log("team list unavailable: %s" % exc)

    try:
        ds.injuries = parse_injuries(client.injuries())
        log("injury records: %d%s" % (len(ds.injuries),
            "  (ESPN publishes none for this league)" if not ds.injuries else ""))
    except EspnError as exc:
        log("injuries unavailable: %s" % exc)

    log("upstream calls: %d (cache hits %d)" % (client.calls_made, client.cache_hits))
    return ds
