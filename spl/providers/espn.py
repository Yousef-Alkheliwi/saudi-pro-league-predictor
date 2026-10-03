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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from ..config import CACHE_DIR
from ..data import Appearance, Dataset, Injury, Match, Player, TeamSheet, TeamStats

BASE = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
LEAGUE = "ksa.1"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: earliest season ESPN carries for this league
EARLIEST_SEASON = 2022

#: how many months past the current one to request, so scheduled fixtures -
#: the ones worth predicting - are in the snapshot
LOOKAHEAD_MONTHS = 2

#: A month's scoreboard is only final once it has been fetched this long after
#: the month ended. Late kick-offs (ESPN buckets by US time), score corrections
#: and box scores all settle inside this window.
FINAL_GRACE_DAYS = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _month_end(year: int, month: int) -> datetime:
    """The first instant after the month, in UTC."""
    if month == 12:
        return datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(year, month + 1, 1, tzinfo=timezone.utc)

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
                 max_retries: int = 3, clock=None) -> None:
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
        #: wall-clock source; injectable so cache ageing can be tested
        self.clock = clock or _utcnow
        #: when the newest data served was actually fetched from the network -
        #: a cache hit counts at its original fetch time, not at "now"
        self.newest_fetch: Optional[float] = None

    def _note(self, fetched_at: float) -> None:
        if self.newest_fetch is None or fetched_at > self.newest_fetch:
            self.newest_fetch = fetched_at

    # ---------------------------------------------------------------- plumbing
    def _cache_path(self, path: str, params: Dict[str, str]) -> Path:
        key = json.dumps({"p": path, "q": params}, sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / ("%s__%s.json" % (path.replace("/", "_"), digest))

    def get(self, path: str, params: Optional[Dict[str, str]] = None,
            ttl: float = 3600.0, not_before: Optional[float] = None,
            transform=None) -> dict:
        """Cached GET. A cached copy is used while younger than `ttl`, and -
        when `not_before` is given - only if it was fetched at or after that
        moment. Offline mode uses whatever is cached."""
        params = params or {}
        fp = self._cache_path(path, params)
        if fp.exists():
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
                fetched = blob.get("fetched_at", 0)
                fresh = self.clock().timestamp() - fetched < ttl
                settled = not_before is None or fetched >= not_before
                if self.offline or (fresh and settled):
                    self.cache_hits += 1
                    self._note(fetched)
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
                resp = self._session.get(url, params=self._url_params(path, params),
                                         timeout=self.timeout,
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
            if transform is not None:
                payload = transform(payload)
            stamp = self.clock().timestamp()
            fp.write_text(json.dumps({"fetched_at": stamp, "payload": payload}),
                          encoding="utf-8")
            self._note(stamp)
            return payload
        raise EspnError("giving up on /%s after %d attempts: %s"
                        % (path, self.max_retries, last))

    # ---------------------------------------------------------------- endpoints
    def month(self, year: int, month: int, ttl: Optional[float] = None) -> dict:
        """One month of fixtures and box scores.

        A month is only final once it has been fetched after it ended. Choosing
        the TTL by whether the month is past *now* is not enough: a copy fetched
        on the 20th is still missing the last ten days of results, and treating
        it as final froze those matches as "not started" for a year - they then
        fell out of both the ratings and the fixture list.
        """
        params = {"dates": "%04d%02d" % (year, month)}
        if ttl is not None:
            return self.get("scoreboard", params, ttl=ttl)
        settles = _month_end(year, month) + timedelta(days=FINAL_GRACE_DAYS)
        if self.clock() < settles:
            return self.get("scoreboard", params, ttl=3600.0)
        return self.get("scoreboard", params, ttl=365 * 86400.0,
                        not_before=settles.timestamp())

    def summary(self, event_id: int, kickoff: datetime) -> dict:
        """One match's line-ups, trimmed to what the squad predictor reads.

        The full payload is ~300 KB, almost all commentary, news and video
        links; the trimmed one is a few KB. A finished match's line-up is final
        a day after kick-off, so only a copy fetched after that is kept for good.
        """
        settles = kickoff + timedelta(hours=26)
        params = {"event": str(event_id), "format": SUMMARY_FORMAT}
        if self.clock() < settles:
            return self.get("summary", params, ttl=3600.0, transform=trim_summary)
        return self.get("summary", params, ttl=365 * 86400.0,
                        not_before=settles.timestamp(), transform=trim_summary)

    def _url_params(self, path: str, params: Dict[str, str]) -> Dict[str, str]:
        """`format` only versions the local cache; ESPN never sees it."""
        return {k: v for k, v in params.items() if k != "format"}

    def squad(self, team_id: int) -> dict:
        """A club's registered squad - who is actually at the club now."""
        return self.get("teams/%d/roster" % int(team_id), ttl=86400.0)

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
    if not isinstance(text, str):
        raise ValueError("kickoff is not a string: %r" % (text,))
    raw = text.strip()
    if not raw:
        raise ValueError("empty kickoff")
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


def _display_name(side: dict, fallback: int) -> str:
    team = side.get("team")
    name = team.get("displayName") if isinstance(team, dict) else None
    return name if isinstance(name, str) and name else str(fallback)


def parse_event(event: dict) -> Optional[Tuple[Match, List[TeamStats]]]:
    """One ESPN event -> a Match plus whatever box score came with it.

    Every field here comes from a third-party feed, so nothing about its shape
    is guaranteed. Anything unrecognisable yields None rather than raising.
    """
    if not isinstance(event, dict) or not event.get("id"):
        return None
    comps = event.get("competitions") or []
    if not isinstance(comps, list):
        return None
    if not comps or not event.get("id"):
        return None
    comp = comps[0] if isinstance(comps[0], dict) else {}
    sides = comp.get("competitors") or []
    if not isinstance(sides, list) or len(sides) != 2:
        return None
    if not all(isinstance(x, dict) for x in sides):
        return None

    home = next((s for s in sides if s.get("homeAway") == "home"), None)
    away = next((s for s in sides if s.get("homeAway") == "away"), None)
    if home is None or away is None:
        home, away = sides[0], sides[1]

    def team_id(side) -> Optional[int]:
        team = side.get("team")
        team = team if isinstance(team, dict) else {}
        raw = team.get("id") or side.get("id")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    hid, aid = team_id(home), team_id(away)
    if hid is None or aid is None or hid == aid:
        return None

    raw_status = event.get("status") or comp.get("status") or {}
    raw_status = raw_status if isinstance(raw_status, dict) else {}
    status = raw_status.get("type")
    status = status if isinstance(status, dict) else {}
    completed = bool(status.get("completed"))
    short = status.get("shortDetail") or status.get("description") or ""
    hg = _num(home.get("score"))
    ag = _num(away.get("score"))
    if not completed or hg is None or ag is None:
        hg = ag = None

    try:
        # a fixture we cannot place in time is no use: it drives rest days,
        # season labelling and the whole upcoming/played split
        dt = _parse_dt(event.get("date") or comp.get("date") or "")
    except (ValueError, TypeError):
        return None
    try:
        fixture_id = int(event["id"])
    except (TypeError, ValueError):
        return None                     # an id we cannot key on is unusable
    match = Match(
        fixture_id=fixture_id,
        season=season_of(dt),
        kickoff=dt.isoformat(),
        home_id=hid, away_id=aid,
        home_name=_display_name(home, hid),
        away_name=_display_name(away, aid),
        # the rest of the codebase keys "played" off these codes
        status="FT" if completed else "NS",
        home_goals=int(hg) if hg is not None else None,
        away_goals=int(ag) if ag is not None else None,
        venue=(comp.get("venue") or {}).get("fullName")
              if isinstance(comp.get("venue"), dict) else None,
        round=(event.get("week") or {}).get("text") if isinstance(event.get("week"), dict) else None,
        league_id=None, league_name="Saudi Pro League",
    )

    stats: List[TeamStats] = []
    for side, tid in ((home, hid), (away, aid)):
        raw = side.get("statistics") or []
        if not raw or not isinstance(raw, list):
            continue
        rec = TeamStats(fixture_id=fixture_id, team_id=tid)
        found = False
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            field = STAT_MAP.get(name)
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
    blocks = payload.get("injuries") if isinstance(payload, dict) else None
    for block in (blocks if isinstance(blocks, (list, tuple)) else []):
        if not isinstance(block, dict):
            continue
        team = block.get("team")
        team = team if isinstance(team, dict) else {}
        try:
            tid = int(team.get("id"))
        except (TypeError, ValueError):
            continue
        entries = block.get("injuries")
        for item in (entries if isinstance(entries, (list, tuple)) else []):
            if not isinstance(item, dict):
                continue
            athlete = item.get("athlete")
            athlete = athlete if isinstance(athlete, dict) else {}
            out.append(Injury(
                team_id=tid,
                player_id=int(athlete["id"]) if str(athlete.get("id", "")).isdigit() else None,
                player_name=athlete.get("displayName") or "unknown",
                type=item.get("status") or "Missing Fixture",
                reason=(item.get("type") or {}).get("description") or item.get("status")
                       or "unknown",
            ))
    return out


# -------------------------------------------------------------------- line-ups
#: assumed length of a match when turning substitution times into minutes
MATCH_MINUTES = 90.0

_KEEP_STATS = ("redCards", "yellowCards", "totalGoals", "goalAssists")
#: bump when trim_summary keeps something new, so cached copies are refetched
SUMMARY_FORMAT = "2"


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    return value if isinstance(value, (list, tuple)) else []


def trim_summary(payload: dict) -> dict:
    """Keep the line-ups, substitutions and red cards; drop the other ~97%."""
    payload = _as_dict(payload)
    rosters = []
    for block in _as_list(payload.get("rosters")):
        block = _as_dict(block)
        team = _as_dict(block.get("team"))
        players = []
        for entry in _as_list(block.get("roster")):
            entry = _as_dict(entry)
            athlete = _as_dict(entry.get("athlete"))
            stats = {}
            for item in _as_list(entry.get("stats")):
                item = _as_dict(item)
                if item.get("name") in _KEEP_STATS:
                    stats[item["name"]] = item.get("value")
            players.append({
                "id": athlete.get("id"), "name": athlete.get("displayName"),
                "short": athlete.get("shortName") or athlete.get("lastName"),
                "jersey": entry.get("jersey"), "starter": entry.get("starter"),
                "place": entry.get("formationPlace"),
                "position": _as_dict(entry.get("position")).get("abbreviation"),
                "stats": stats,
            })
        rosters.append({"team": team.get("id"), "formation": block.get("formation"),
                        "players": players})
    events = []
    for e in _as_list(payload.get("keyEvents")):
        e = _as_dict(e)
        kind = str(_as_dict(e.get("type")).get("text") or "")
        if "Substitution" not in kind and "Red" not in kind:
            continue
        events.append({
            "kind": "sub" if "Substitution" in kind else "red",
            "seconds": _as_dict(e.get("clock")).get("value"),
            "team": _as_dict(e.get("team")).get("id"),
            "players": [_as_dict(_as_dict(x).get("athlete")).get("id")
                        for x in _as_list(e.get("participants"))],
        })
    return {"rosters": rosters, "events": events}


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _minute(seconds) -> Optional[float]:
    value = _num(seconds) if not isinstance(seconds, (int, float)) else float(seconds)
    if value is None or value != value or value < 0:
        return None
    return min(value / 60.0, MATCH_MINUTES)


def parse_summary(fixture_id: int, trimmed: dict
                  ) -> Tuple[List[TeamSheet], List[Appearance]]:
    """Line-ups for one match, with minutes worked out from substitution times.

    A starter plays from 0 to the minute they are replaced or sent off; a
    substitute from the minute they come on. Stoppage time is ignored, so a
    full match counts as 90.
    """
    trimmed = _as_dict(trimmed)
    on: Dict[int, float] = {}
    off: Dict[int, float] = {}
    for e in _as_list(trimmed.get("events")):
        e = _as_dict(e)
        minute = _minute(e.get("seconds"))
        ids = [_int(x) for x in _as_list(e.get("players"))]
        if minute is None:
            continue
        if e.get("kind") == "sub" and len(ids) >= 2:
            if ids[0] is not None:
                on.setdefault(ids[0], minute)
            if ids[1] is not None:
                off.setdefault(ids[1], minute)
        elif e.get("kind") == "red" and ids and ids[0] is not None:
            off[ids[0]] = min(off.get(ids[0], MATCH_MINUTES), minute)

    sheets: List[TeamSheet] = []
    apps: List[Appearance] = []
    for block in _as_list(trimmed.get("rosters")):
        block = _as_dict(block)
        tid = _int(block.get("team"))
        if tid is None:
            continue
        formation = block.get("formation")
        sheets.append(TeamSheet(fixture_id, tid,
                                formation if isinstance(formation, str) else ""))
        for pl in _as_list(block.get("players")):
            pl = _as_dict(pl)
            pid = _int(pl.get("id"))
            if pid is None:
                continue
            stats = _as_dict(pl.get("stats"))
            starter = pl.get("starter") is True
            if starter:
                start = 0.0
            elif pid in on:
                start = on[pid]
            else:
                start = None                     # unused substitute
            end = off.get(pid, MATCH_MINUTES)
            minutes = max(0.0, end - start) if start is not None else 0.0
            name = pl.get("name") if isinstance(pl.get("name"), str) else str(pid)
            short = pl.get("short") if isinstance(pl.get("short"), str) else name
            apps.append(Appearance(
                fixture_id=fixture_id, team_id=tid, player_id=pid, name=name,
                short_name=short,
                jersey=str(pl.get("jersey")) if pl.get("jersey") is not None else "",
                position=pl.get("position") if isinstance(pl.get("position"), str) else "",
                starter=starter,
                formation_place=_int(pl.get("place")) or 0,
                minutes=round(minutes, 1),
                red_card=(_num(stats.get("redCards")) or 0) > 0,
                yellow_cards=int(_num(stats.get("yellowCards")) or 0),
                goals=int(_num(stats.get("totalGoals")) or 0),
                assists=int(_num(stats.get("goalAssists")) or 0),
            ))
    return sheets, apps


#: squad-list position letters -> the names the availability model weights by
_SQUAD_POSITION = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder",
                   "F": "Attacker"}


def parse_squad(team_id: int, payload: dict) -> List[Player]:
    """The registered squad. Some ESPN rosters group athletes by position."""
    flat = []
    for item in _as_list(_as_dict(payload).get("athletes")):
        item = _as_dict(item)
        if "items" in item:
            flat.extend(_as_list(item.get("items")))
        else:
            flat.append(item)
    out: List[Player] = []
    for a in flat:
        a = _as_dict(a)
        pid = _int(a.get("id"))
        if pid is None:
            continue
        letter = _as_dict(a.get("position")).get("abbreviation")
        name = a.get("displayName") if isinstance(a.get("displayName"), str) else str(pid)
        out.append(Player(player_id=pid, team_id=team_id, name=name,
                          position=_SQUAD_POSITION.get(letter)))
    return out


# -------------------------------------------------------------------- fetching
def _months_ahead(now: datetime, n: int) -> Tuple[int, int]:
    total = now.year * 12 + (now.month - 1) + n
    return total // 12, total % 12 + 1


def season_months(season: int) -> List[Tuple[int, int]]:
    """August of the opening year through July of the next."""
    return ([(season, m) for m in range(8, 13)]
            + [(season + 1, m) for m in range(1, 8)])


#: how many of a club's most recent line-ups its players' minutes are counted
#: over - the importance weights the availability model uses
MINUTES_WINDOW = 10


def fetch_lineups(client, ds: Dataset, seasons: Iterable[int], log=print) -> None:
    """Line-ups for every finished match in `seasons`, plus each current club's
    registered squad with its players' recent minutes."""
    wanted = set(seasons)
    played = [m for m in ds.played_matches if m.season in wanted]
    fetched = 0
    for m in sorted(played, key=lambda x: x.dt, reverse=True):
        try:
            sheets, apps = parse_summary(m.fixture_id,
                                         client.summary(m.fixture_id, m.dt))
        except EspnError as exc:
            log("  line-up for %s unavailable: %s" % (m.fixture_id, exc))
            continue
        # a line-up is only usable if it names the two clubs in the fixture
        sheets = [x for x in sheets if x.team_id in (m.home_id, m.away_id)]
        apps = [a for a in apps if a.team_id in (m.home_id, m.away_id)]
        if not apps:
            continue
        ds.team_sheets.extend(sheets)
        ds.appearances.extend(apps)
        fetched += 1
    log("line-ups: %d matches, %d player appearances" % (fetched, len(ds.appearances)))

    current = max(wanted) if wanted else None
    clubs = sorted({t for m in ds.matches if m.season == current
                    for t in (m.home_id, m.away_id)})
    by_fixture = ds.appearances_by_fixture()
    dated = sorted(ds.played_matches, key=lambda x: x.dt, reverse=True)
    for tid in clubs:
        try:
            members = parse_squad(tid, client.squad(tid))
        except EspnError as exc:
            log("  squad for %s unavailable: %s" % (tid, exc))
            continue
        recent = [m for m in dated if tid in (m.home_id, m.away_id)
                  and any(a.team_id == tid for a in by_fixture.get(m.fixture_id, []))]
        recent = recent[:MINUTES_WINDOW]
        minutes: Dict[int, float] = {}
        apps_count: Dict[int, int] = {}
        goals: Dict[int, int] = {}
        for m in recent:
            for a in by_fixture.get(m.fixture_id, []):
                if a.team_id != tid:
                    continue
                minutes[a.player_id] = minutes.get(a.player_id, 0.0) + a.minutes
                if a.minutes > 0:
                    apps_count[a.player_id] = apps_count.get(a.player_id, 0) + 1
                goals[a.player_id] = goals.get(a.player_id, 0) + a.goals
        for pl in members:
            pl.minutes = minutes.get(pl.player_id, 0.0)
            pl.appearances = float(apps_count.get(pl.player_id, 0))
            pl.goals = float(goals.get(pl.player_id, 0))
        ds.players.extend(members)
    log("squads: %d players across %d clubs" % (len(ds.players), len(clubs)))


def fetch_dataset(client: Espn, seasons: Optional[Iterable[int]] = None,
                  with_logos: bool = True, with_lineups: bool = True,
                  lineup_seasons: int = 2, managers=None, log=print) -> Dataset:
    now = getattr(client, "clock", _utcnow)()
    current = season_of(now)
    if seasons is None:
        seasons = range(max(EARLIEST_SEASON, current - 3), current + 1)
    seasons = sorted(set(seasons))

    ds = Dataset(league_id=0,
                 fetched_at=now.isoformat(timespec="seconds"))
    ds.source = "ESPN (site.web.api.espn.com), soccer/%s" % LEAGUE

    # One entry per fixture id. The same fixture can be listed more than once -
    # postponed at its original date in one month, played in another - and the
    # first listing is not the best one. A result always beats an unplayed
    # listing; otherwise the later listing wins, since it carries the current
    # date of a rescheduled match.
    kept: Dict[int, Tuple[Match, List[TeamStats]]] = {}
    for season in seasons:
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
                try:
                    parsed = parse_event(event)
                except Exception as exc:          # pragma: no cover - defensive
                    log("  skipped an unparseable event: %s" % exc)
                    continue
                if parsed is None:
                    continue
                match, stats = parsed
                previous = kept.get(match.fixture_id)
                if previous is not None and previous[0].played and not match.played:
                    continue
                kept[match.fixture_id] = (match, stats)

    for match, stats in sorted(kept.values(), key=lambda ms: ms[0].dt):
        ds.matches.append(match)
        ds.stats.extend(stats)
        ds.teams.setdefault(match.home_id, match.home_name)
        ds.teams.setdefault(match.away_id, match.away_name)
    for season in seasons:
        in_season = [m for m in ds.matches if m.season == season]
        ids = {m.fixture_id for m in in_season}
        log("season %s: %d fixtures, %d box-score rows"
            % (season, len(in_season), sum(1 for x in ds.stats if x.fixture_id in ids)))

    try:
        payload = client.teams()
        blocks = (payload.get("sports") or [{}])[0].get("leagues") or [{}]
        for row in (blocks[0].get("teams") or []):
            team = row.get("team") or {}
            if str(team.get("id", "")).isdigit():
                ds.teams[int(team["id"])] = team.get("displayName") or str(team["id"])
    except Exception as exc:
        # the fixtures already name every club, so this step is a refinement:
        # an unexpected failure here must not cost us the whole snapshot
        log("team list unavailable: %s" % exc)

    try:
        ds.injuries = parse_injuries(client.injuries())
        log("injury records: %d%s" % (len(ds.injuries),
            "  (ESPN publishes none for this league)" if not ds.injuries else ""))
    except EspnError as exc:
        log("injuries unavailable: %s" % exc)

    if with_lineups and hasattr(client, "summary") and hasattr(client, "squad"):
        recent_seasons = sorted({m.season for m in ds.played_matches})[-lineup_seasons:]
        try:
            fetch_lineups(client, ds, recent_seasons, log=log)
        except Exception as exc:                      # pragma: no cover
            log("line-ups skipped: %s" % exc)
        if managers is not None:
            from .wikipedia import fetch_managers
            try:
                # one season earlier, to know who was in charge as the window opens
                fetch_managers(managers, ds, [min(recent_seasons) - 1] + recent_seasons,
                               log=log)
            except Exception as exc:                  # pragma: no cover
                log("managers skipped: %s" % exc)

    if with_logos:
        # badges are cosmetic; never let them take the data fetch down with them
        try:
            ds.logos = fetch_logos(client, set(ds.teams), log=log)
        except Exception as exc:                      # pragma: no cover
            log("badges skipped: %s" % exc)

    # The snapshot time is when the newest data was fetched, not when it was
    # parsed: an offline re-parse of week-old caches is still week-old data.
    newest = getattr(client, "newest_fetch", None)
    if newest is not None:
        ds.fetched_at = datetime.fromtimestamp(newest, timezone.utc).isoformat(
            timespec="seconds")

    log("upstream calls: %d (cache hits %d)" % (client.calls_made, client.cache_hits))
    return ds


# -------------------------------------------------------------------- badges
#: rendered at 54px, so 112 covers a 2x display exactly
LOGO_PX = 112


def _shrink_png(raw: bytes, px: int = LOGO_PX) -> bytes:
    """Downscale with macOS `sips`. Returns the original if it is unavailable."""
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("sips"):
        return raw
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.png"
        dst = Path(d) / "out.png"
        src.write_bytes(raw)
        try:
            subprocess.run(["sips", "-Z", str(px), str(src), "--out", str(dst)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=25, check=True)
            return dst.read_bytes() if dst.exists() else raw
        except (subprocess.SubprocessError, OSError):
            return raw


def fetch_logos(client: Espn, team_ids, log=print) -> Dict[int, str]:
    """Club badges as data: URIs, so the built page stays a single file.

    Cached on disk alongside the JSON, because these never change and each one
    is a separate download.
    """
    import base64

    cache_root = getattr(client, "cache_dir", None)
    if cache_root is None:
        return {}
    cache = Path(cache_root) / "logos"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        payload = client.teams()
    except EspnError as exc:
        log("logos unavailable: %s" % exc)
        return {}

    blocks = (payload.get("sports") or [{}])[0].get("leagues") or [{}]
    urls: Dict[int, str] = {}
    for row in (blocks[0].get("teams") or []):
        team = row.get("team") or {}
        if not str(team.get("id", "")).isdigit():
            continue
        logos = team.get("logos") or []
        url = team.get("logo") or (logos[0].get("href") if logos else None)
        if url:
            urls[int(team["id"])] = url

    wanted = set(team_ids)
    out: Dict[int, str] = {}
    fetched = 0
    for tid, url in sorted(urls.items()):
        if tid not in wanted:
            continue
        fp = cache / ("%d.png" % tid)
        if not fp.exists():
            try:
                resp = requests.get(url, timeout=client.timeout,
                                    headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                fp.write_bytes(_shrink_png(resp.content))
                fetched += 1
                time.sleep(0.15)
            except requests.RequestException as exc:
                log("  badge for %d failed: %s" % (tid, exc))
                continue
        try:
            blob = fp.read_bytes()
        except OSError:
            continue
        out[tid] = "data:image/png;base64," + base64.b64encode(blob).decode("ascii")

    missing = sorted(wanted - set(out))
    log("badges: %d embedded (%d newly downloaded)%s"
        % (len(out), fetched,
           ", %d club(s) have none and fall back to initials" % len(missing)
           if missing else ""))
    return out
