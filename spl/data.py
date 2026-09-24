"""Normalise API-Football payloads into a compact, model-ready dataset."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import DATASET_PATH, MODEL
from .providers import ApiFootball, MissingCredentials, PlanRestricted

FINISHED = {"FT", "AET", "PEN"}

#: /players page size, and how many pages to walk before giving up on a squad
PLAYERS_PER_PAGE = 20
MAX_PLAYER_PAGES = 3


# --------------------------------------------------------------------------- records
@dataclass
class Match:
    fixture_id: int
    season: int
    kickoff: str                 # ISO-8601, UTC
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    status: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    venue: Optional[str] = None
    round: Optional[str] = None
    league_id: Optional[int] = None
    league_name: Optional[str] = None

    @property
    def played(self) -> bool:
        return self.status in FINISHED and self.home_goals is not None

    @property
    def dt(self) -> datetime:
        return _parse_dt(self.kickoff)


@dataclass
class TeamStats:
    """Per-team, per-match box score."""
    fixture_id: int
    team_id: int
    shots: Optional[float] = None
    shots_on_target: Optional[float] = None
    possession: Optional[float] = None
    corners: Optional[float] = None
    expected_goals: Optional[float] = None


@dataclass
class Injury:
    team_id: int
    player_id: Optional[int]
    player_name: str
    type: str          # "Missing Fixture" / "Questionable"
    reason: str
    fixture_date: Optional[str] = None


@dataclass
class Player:
    player_id: int
    team_id: int
    name: str
    position: Optional[str] = None
    minutes: float = 0.0
    appearances: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    rating: Optional[float] = None


@dataclass
class Dataset:
    league_id: int
    fetched_at: str
    teams: Dict[int, str] = field(default_factory=dict)
    #: True when the subscription blocked one or more requested seasons, so the
    #: snapshot is historical rather than live
    plan_limited: bool = False
    #: newest season the plan actually served
    newest_available_season: Optional[int] = None
    #: which upstream produced this snapshot
    source: str = "API-Football (api-sports.io)"
    #: team_id -> club badge as a data: URI, embedded so the page stays one file
    logos: Dict[int, str] = field(default_factory=dict)
    matches: List[Match] = field(default_factory=list)
    #: fixtures from other competitions (AFC Champions League, King's Cup, ...).
    #: These are never used to fit ratings - only to compute rest and congestion,
    #: which would otherwise be badly wrong for clubs in continental competition.
    other_matches: List[Match] = field(default_factory=list)
    stats: List[TeamStats] = field(default_factory=list)
    injuries: List[Injury] = field(default_factory=list)
    players: List[Player] = field(default_factory=list)

    # ---------------------------------------------------------------- persistence
    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or DATASET_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "league_id": self.league_id,
            "fetched_at": self.fetched_at,
            "plan_limited": self.plan_limited,
            "newest_available_season": self.newest_available_season,
            "source": self.source,
            "logos": {str(k): v for k, v in self.logos.items()},
            "teams": {str(k): v for k, v in self.teams.items()},
            "matches": [asdict(m) for m in self.matches],
            "other_matches": [asdict(m) for m in self.other_matches],
            "stats": [asdict(s) for s in self.stats],
            "injuries": [asdict(i) for i in self.injuries],
            "players": [asdict(p) for p in self.players],
        }
        path.write_text(json.dumps(blob), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Dataset":
        path = Path(path or DATASET_PATH)
        if not path.exists():
            raise FileNotFoundError(
                "no dataset at %s - run `python -m spl.cli fetch` first" % path
            )
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            league_id=blob["league_id"],
            fetched_at=blob["fetched_at"],
            plan_limited=blob.get("plan_limited", False),
            newest_available_season=blob.get("newest_available_season"),
            source=blob.get("source", "API-Football (api-sports.io)"),
            logos={int(k): v for k, v in (blob.get("logos") or {}).items()},
            teams={int(k): v for k, v in blob["teams"].items()},
            matches=[Match(**m) for m in blob["matches"]],
            other_matches=[Match(**m) for m in blob.get("other_matches", [])],
            stats=[TeamStats(**s) for s in blob["stats"]],
            injuries=[Injury(**i) for i in blob["injuries"]],
            players=[Player(**p) for p in blob["players"]],
        )

    # ---------------------------------------------------------------- convenience
    @property
    def played_matches(self) -> List[Match]:
        return sorted([m for m in self.matches if m.played], key=lambda m: m.dt)

    @property
    def schedule_matches(self) -> List[Match]:
        """Every finished match across all competitions, for rest/congestion.

        Falls back to the league fixtures alone when no other competitions were
        fetched, so callers can always use this property.
        """
        seen = set()
        out: List[Match] = []
        for m in list(self.matches) + list(self.other_matches):
            if m.played and m.fixture_id not in seen:
                seen.add(m.fixture_id)
                out.append(m)
        return sorted(out, key=lambda m: m.dt)

    @property
    def upcoming_matches(self) -> List[Match]:
        return sorted([m for m in self.matches if not m.played], key=lambda m: m.dt)

    def stats_by_fixture(self) -> Dict[int, Dict[int, TeamStats]]:
        out: Dict[int, Dict[int, TeamStats]] = {}
        for s in self.stats:
            out.setdefault(s.fixture_id, {})[s.team_id] = s
        return out

    def injuries_by_team(self) -> Dict[int, List[Injury]]:
        out: Dict[int, List[Injury]] = {}
        for inj in self.injuries:
            out.setdefault(inj.team_id, []).append(inj)
        return out

    def players_by_team(self) -> Dict[int, List[Player]]:
        out: Dict[int, List[Player]] = {}
        for p in self.players:
            out.setdefault(p.team_id, []).append(p)
        return out

    def resolve_team(self, query: str) -> Tuple[int, str]:
        """Fuzzy-resolve a club name ('hilal', 'Al-Nassr', 'ittihad') to an id."""
        return resolve_team_name(query, self.teams)


# --------------------------------------------------------------------------- helpers
def _parse_dt(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# Only genuinely uninformative tokens. City names are NOT noise: "Al-Riyadh" is a
# club in its own right and "Jeddah" distinguishes Al-Ahli Jeddah, so stripping
# them collapses real clubs to the empty string.
_NOISE = re.compile(r"\b(fc|sc|cf|club|saudi)\b")


def normalise_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    text = text.lower().replace("-", " ").replace(".", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    stripped = _NOISE.sub(" ", text)
    # never let stripping empty a name out
    if stripped.strip():
        text = stripped
    return re.sub(r"\s+", " ", text).strip()


def _squash(name: str) -> str:
    """Normalised name with the 'al' prefix and all spaces removed.

    Lets "alnassr", "al nassr" and "Al-Nassr" all land on the same key.
    """
    text = normalise_name(name).replace(" ", "")
    return text[2:] if len(text) > 4 and text.startswith("al") else text


def resolve_team_name(query: str, teams: Dict[int, str]) -> Tuple[int, str]:
    """Resolve a user-typed club name against the league's team list."""
    if not teams:
        raise LookupError("no teams in the dataset")
    q = normalise_name(query)
    if not q:
        raise LookupError("empty team name")

    exact = [(tid, name) for tid, name in teams.items() if normalise_name(name) == q]
    if len(exact) == 1:
        return exact[0]

    # spacing- and prefix-insensitive exact match: "alnassr" == "Al-Nassr"
    qs = _squash(query)
    squashed = [(tid, name) for tid, name in teams.items() if _squash(name) == qs]
    if len(squashed) == 1:
        return squashed[0]

    contains = [(tid, name) for tid, name in teams.items()
                if normalise_name(name)
                and (q in normalise_name(name) or normalise_name(name) in q)]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        # prefer the shortest normalised name, which is the least-qualified match
        contains.sort(key=lambda kv: len(normalise_name(kv[1])))
        if len(normalise_name(contains[0][1])) < len(normalise_name(contains[1][1])):
            return contains[0]
        raise LookupError("'%s' is ambiguous: %s" % (query, ", ".join(n for _, n in contains)))

    # substring on the squashed forms, so "ahli" still finds "Al-Ahli Jeddah"
    sub = [(tid, name) for tid, name in teams.items()
           if _squash(name) and (qs in _squash(name) or _squash(name) in qs)]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        sub.sort(key=lambda kv: len(_squash(kv[1])))
        if len(_squash(sub[0][1])) < len(_squash(sub[1][1])):
            return sub[0]
        raise LookupError("'%s' is ambiguous: %s"
                          % (query, ", ".join(n for _, n in sub)))

    # token overlap fallback
    qt = set(q.split())
    scored = []
    for tid, name in teams.items():
        nt = set(normalise_name(name).split())
        if not nt:
            continue
        overlap = len(qt & nt) / max(1, len(qt | nt))
        if overlap > 0:
            scored.append((overlap, tid, name))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1], scored[0][2]
    raise LookupError("no team matching '%s'. Known teams: %s"
                      % (query, ", ".join(sorted(teams.values()))))


def _stat_value(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    text = text.rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


_STAT_KEYS = {
    "total shots": "shots",
    "shots on goal": "shots_on_target",
    "ball possession": "possession",
    "corner kicks": "corners",
    "expected_goals": "expected_goals",
    "expected goals": "expected_goals",
}


# --------------------------------------------------------------------------- fetching
def parse_fixture(row: dict) -> Optional[Match]:
    # the payload is third-party; treat every level as untrusted
    if not isinstance(row, dict):
        return None
    def _d(value):
        return value if isinstance(value, dict) else {}
    fixture = _d(row.get("fixture"))
    teams = _d(row.get("teams"))
    goals = _d(row.get("goals"))
    league = _d(row.get("league"))
    home, away = _d(teams.get("home")), _d(teams.get("away"))
    if not fixture.get("id") or not home.get("id") or not away.get("id"):
        return None
    status = _d(fixture.get("status")).get("short") or "NS"
    venue = _d(fixture.get("venue")).get("name")
    return Match(
        fixture_id=int(fixture["id"]),
        season=int(league.get("season") or 0),
        kickoff=fixture.get("date") or "",
        home_id=int(home["id"]),
        away_id=int(away["id"]),
        home_name=home.get("name") or str(home["id"]),
        away_name=away.get("name") or str(away["id"]),
        status=status,
        home_goals=goals.get("home"),
        away_goals=goals.get("away"),
        venue=venue,
        round=league.get("round"),
        league_id=int(league["id"]) if league.get("id") else None,
        league_name=league.get("name"),
    )


def parse_statistics(fixture_id: int, rows: Iterable[dict]) -> List[TeamStats]:
    out: List[TeamStats] = []
    for row in (rows if isinstance(rows, (list, tuple)) else []):
        if not isinstance(row, dict):
            continue
        team = (row.get("team") or {}).get("id") if isinstance(row.get("team"), dict) else None
        if not team:
            continue
        rec = TeamStats(fixture_id=fixture_id, team_id=int(team))
        items = row.get("statistics")
        for item in (items if isinstance(items, (list, tuple)) else []):
            if not isinstance(item, dict):
                continue
            key = _STAT_KEYS.get(str(item.get("type") or "").strip().lower())
            if key:
                setattr(rec, key, _stat_value(item.get("value")))
        out.append(rec)
    return out


def parse_injuries(rows: Iterable[dict]) -> List[Injury]:
    out: List[Injury] = []
    seen = set()
    for row in (rows if isinstance(rows, (list, tuple)) else []):
        if not isinstance(row, dict):
            continue
        team = (row.get("team") or {}).get("id") if isinstance(row.get("team"), dict) else None
        player = row.get("player") if isinstance(row.get("player"), dict) else {}
        if not team:
            continue
        pid = player.get("id")
        fixture_date = (row.get("fixture") or {}).get("date")
        key = (int(team), pid, player.get("name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(Injury(
            team_id=int(team),
            player_id=int(pid) if pid else None,
            player_name=player.get("name") or "unknown",
            type=player.get("type") or "Missing Fixture",
            reason=player.get("reason") or player.get("type") or "unknown",
            fixture_date=fixture_date,
        ))
    return out


def parse_players(team_id: int, rows: Iterable[dict]) -> List[Player]:
    out: List[Player] = []
    for row in (rows if isinstance(rows, (list, tuple)) else []):
        if not isinstance(row, dict):
            continue
        player = row.get("player") if isinstance(row.get("player"), dict) else {}
        pid = player.get("id")
        if not pid:
            continue
        minutes = appearances = goals = assists = 0.0
        rating: Optional[float] = None
        position = None
        blocks = row.get("statistics")
        for block in (blocks if isinstance(blocks, (list, tuple)) else []):
            if not isinstance(block, dict):
                continue
            games = block.get("games") if isinstance(block.get("games"), dict) else {}
            minutes += float(games.get("minutes") or 0)
            appearances += float(games.get("appearences") or games.get("appearances") or 0)
            position = position or games.get("position")
            g = block.get("goals") if isinstance(block.get("goals"), dict) else {}
            goals += float(g.get("total") or 0)
            assists += float(g.get("assists") or 0)
            try:
                rating = float(games.get("rating")) if games.get("rating") else rating
            except (TypeError, ValueError):
                pass
        out.append(Player(player_id=int(pid), team_id=team_id,
                          name=player.get("name") or str(pid), position=position,
                          minutes=minutes, appearances=appearances,
                          goals=goals, assists=assists, rating=rating))
    return out


#: Rough upstream cost of a full fetch, for a league of ~18 clubs:
#:   leagues 2 + fixtures 1/season + teams 1 + injuries 1
#:   + other competitions 1/club + players ~2/club + box scores 1/match
#: The free tier allows 100/day, so the order below matters: the cheap,
#: high-value calls run first and box scores - the only part that degrades
#: gracefully to league priors - run last, so a mid-fetch quota exhaustion
#: costs the least.
def fetch_dataset(client: ApiFootball, seasons: Optional[Iterable[int]] = None,
                  with_stats: bool = True, stats_limit: int = 30,
                  with_players: bool = True, with_schedule: bool = True,
                  refresh_live: bool = True, log=print) -> Dataset:
    """Pull everything the models need, reusing the on-disk cache where valid."""
    league = client.league_id()
    log("league id: %s" % league)

    wanted = list(seasons or MODEL.seasons)
    try:
        available = set(client.available_seasons(league))
        if available:
            wanted = [s for s in wanted if s in available] or sorted(available)[-3:]
    except MissingCredentials:
        raise
    except Exception as exc:  # pragma: no cover - availability lookup is advisory
        log("could not list seasons (%s); trying requested ones" % exc)

    ds = Dataset(league_id=league,
                 fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    newest_requested = max(wanted) if wanted else None
    loaded_seasons: List[int] = []
    restricted: List[int] = []
    for season in wanted:
        try:
            rows = client.fixtures(league, season,
                                  force=refresh_live and season == newest_requested)
        except MissingCredentials:
            raise
        except PlanRestricted:
            restricted.append(season)
            log("season %s: not covered by this subscription" % season)
            continue
        except Exception as exc:
            log("season %s fixtures unavailable: %s" % (season, exc))
            continue
        added = 0
        for row in rows:
            match = parse_fixture(row)
            if match is None:
                continue
            ds.matches.append(match)
            ds.teams.setdefault(match.home_id, match.home_name)
            ds.teams.setdefault(match.away_id, match.away_name)
            added += 1
        if added:
            loaded_seasons.append(season)
        log("season %s: %d fixtures" % (season, added))

    # Everything below keys off a *season the plan actually serves*. Asking for
    # the calendar-current season on a tier that does not cover it loses the team
    # list, injuries and squads all at once, which is exactly what happens on the
    # free tier. Using the newest season that loaded keeps those working.
    current_season = max(loaded_seasons) if loaded_seasons else None
    if restricted:
        ds.plan_limited = True
        ds.newest_available_season = current_season
        log("NOTE: this subscription does not cover season(s) %s. Falling back to "
            "%s for squads, injuries and the team list - so 'current' form, the "
            "live injury list and upcoming fixtures are NOT available on this plan."
            % (", ".join(str(x) for x in restricted), current_season))

    # Teams currently in the league (so we don't offer relegated clubs by default)
    if current_season:
        try:
            for row in client.teams(league, current_season):
                team = row.get("team") or {}
                if team.get("id"):
                    ds.teams[int(team["id"])] = team.get("name") or str(team["id"])
        except Exception as exc:
            log("team list unavailable: %s" % exc)

    # Current injuries
    if current_season:
        try:
            ds.injuries = parse_injuries(client.injuries(league, current_season,
                                                         force=refresh_live))
            log("injury records: %d" % len(ds.injuries))
        except Exception as exc:
            log("injuries unavailable: %s" % exc)

    # Other competitions, so rest days reflect midweek continental and cup games
    if with_schedule and current_season:
        league_fixture_ids = {m.fixture_id for m in ds.matches}
        added = 0
        for team_id in sorted(ds.teams):
            try:
                rows = client.team_fixtures(team_id, current_season,
                                            force=refresh_live)
            except Exception as exc:
                log("other-competition fixtures stopped at team %s: %s"
                    % (team_id, exc))
                break
            for row in rows:
                match = parse_fixture(row)
                if match is None or match.fixture_id in league_fixture_ids:
                    continue
                if match.league_id == league:
                    continue
                league_fixture_ids.add(match.fixture_id)
                ds.other_matches.append(match)
                added += 1
        comps = sorted({m.league_name for m in ds.other_matches if m.league_name})
        log("other-competition fixtures: %d (%s)"
            % (added, ", ".join(comps[:6]) if comps else "none"))

    # Player minutes, used to weight how much an absence matters.
    # /players returns 20 per page and a squad is ~25-30, so a single page would
    # drop real players and inflate everyone else's share of squad minutes.
    if with_players and current_season:
        stopped = False
        for team_id in sorted(ds.teams):
            if stopped:
                break
            for page in range(1, MAX_PLAYER_PAGES + 1):
                try:
                    rows = client.player_season_stats(team_id, league,
                                                      current_season, page=page)
                except MissingCredentials:
                    raise
                except PlanRestricted as exc:
                    log("squads: not covered by this subscription (%s)" % exc)
                    stopped = True
                    break
                except Exception as exc:
                    log("player stats stopped at team %s page %d: %s"
                        % (team_id, page, exc))
                    stopped = True
                    break
                ds.players.extend(parse_players(team_id, rows))
                if len(rows) < PLAYERS_PER_PAGE:
                    break                      # last page for this club
        log("player records: %d across %d clubs"
            % (len(ds.players), len(ds.players_by_team())))

    # Box scores: newest finished matches first, bounded by `stats_limit`
    if with_stats:
        played = list(reversed(ds.played_matches))[:max(0, stats_limit)]
        got = 0
        for match in played:
            try:
                rows = client.fixture_statistics(match.fixture_id)
            except Exception as exc:
                log("stats stopped at fixture %s: %s" % (match.fixture_id, exc))
                break
            parsed = parse_statistics(match.fixture_id, rows)
            if parsed:
                ds.stats.extend(parsed)
                got += 1
        log("box scores for %d matches" % got)

    if hasattr(client, "calls_made"):
        log("upstream calls used: %d (cache hits %d)"
            % (client.calls_made, client.cache_hits))

    return ds
