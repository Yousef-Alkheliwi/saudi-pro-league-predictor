"""Match-level features: rest, congestion, head-to-head, squad availability, form."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from .config import MODEL
from .data import Dataset, Injury, Match, Player

# Positional weights used when a player's minutes are unknown, and to convert an
# absence into an attack-side vs defence-side penalty.
POSITION_SIDE = {
    "Attacker": (0.75, 0.25),
    "Midfielder": (0.5, 0.5),
    "Defender": (0.2, 0.8),
    "Goalkeeper": (0.05, 0.95),
}
DEFAULT_SIDE = (0.5, 0.5)

#: most importance a single absence can carry, a little above an ever-present
#: starter's 1/11 share
PER_PLAYER_CAP = 0.13

#: absences flagged "Questionable" are counted at partial weight
TYPE_WEIGHT = {
    "missing fixture": 1.0,
    "questionable": 0.45,
}


# ----------------------------------------------------------------------- schedule
def team_match_history(matches: Sequence[Match], team_id: int,
                       before: datetime) -> List[Match]:
    return [m for m in matches
            if m.played and m.dt < before and team_id in (m.home_id, m.away_id)]


def rest_days(matches: Sequence[Match], team_id: int, kickoff: datetime,
              cap: Optional[int] = None) -> Optional[float]:
    """Days since the team's previous competitive match in this dataset."""
    cap = MODEL.rest_days_cap if cap is None else cap
    prior = team_match_history(matches, team_id, kickoff)
    if not prior:
        return None
    last = max(prior, key=lambda m: m.dt)
    days = (kickoff - last.dt).total_seconds() / 86400.0
    return float(min(days, cap))


def congestion(matches: Sequence[Match], team_id: int, kickoff: datetime,
               window_days: Optional[int] = None) -> int:
    """Matches played in the trailing window before kickoff."""
    window_days = MODEL.congestion_window_days if window_days is None else window_days
    start = kickoff - timedelta(days=window_days)
    return sum(1 for m in team_match_history(matches, team_id, kickoff) if m.dt >= start)


# ----------------------------------------------------------------------- squad
@dataclass
class Availability:
    """How much of a squad's importance-weighted minutes are unavailable."""
    team_id: int
    attack_available: float = 1.0     # 1.0 = full strength
    defence_available: float = 1.0
    overall_available: float = 1.0
    out: List[str] = field(default_factory=list)
    doubtful: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def missing_share(self) -> float:
        return max(0.0, 1.0 - self.overall_available)


def _player_weight(player: Player, squad_minutes: float) -> float:
    """Share of the squad's minutes this player accounts for."""
    if squad_minutes > 0 and player.minutes > 0:
        return player.minutes / squad_minutes
    return 0.0


def availability(team_id: int, injuries: Sequence[Injury],
                 players: Sequence[Player]) -> Availability:
    """Importance-weighted availability from the current injury list.

    A player's importance is their share of the team's minutes this season, so
    losing a regular starter hurts far more than losing a fringe squad member.
    The share is split across attack and defence by position.
    """
    av = Availability(team_id=team_id)
    if not injuries:
        av.note = "no reported absences"
        return av

    by_id = {p.player_id: p for p in players}
    by_name = {p.name.lower(): p for p in players}
    squad_minutes = sum(p.minutes for p in players)
    # A player's share of the squad's total minutes is already their share of the
    # eleven: an ever-present starter plays 38*90 of the team's 38*11*90 minutes,
    # i.e. 1/11 = 0.091. So the share is used directly as the cost of the absence.
    # PER_PLAYER_CAP guards against a bad minutes figure blowing up the index.
    fallback_weight = 1.0 / 25.0        # an unknown player ~ an average squad member

    att_missing = 0.0
    def_missing = 0.0
    overall_missing = 0.0

    for inj in injuries:
        tw = TYPE_WEIGHT.get((inj.type or "").strip().lower(), 0.8)
        player = None
        if inj.player_id is not None:
            player = by_id.get(inj.player_id)
        if player is None:
            player = by_name.get((inj.player_name or "").lower())

        if player is not None:
            weight = _player_weight(player, squad_minutes) or fallback_weight
            side = POSITION_SIDE.get(player.position or "", DEFAULT_SIDE)
        else:
            weight = fallback_weight
            side = DEFAULT_SIDE

        scaled = min(PER_PLAYER_CAP, weight) * tw
        att_missing += scaled * side[0]
        def_missing += scaled * side[1]
        overall_missing += scaled

        label = inj.player_name
        if player is not None and player.minutes:
            label = "%s (%s, %d min)" % (inj.player_name, player.position or "?",
                                         int(player.minutes))
        if tw >= 0.9:
            av.out.append(label)
        else:
            av.doubtful.append(label)

    av.attack_available = max(0.4, 1.0 - att_missing)
    av.defence_available = max(0.4, 1.0 - def_missing)
    av.overall_available = max(0.4, 1.0 - overall_missing)
    av.note = "%d out, %d doubtful" % (len(av.out), len(av.doubtful))
    return av


# ----------------------------------------------------------------------- h2h / form
@dataclass
class HeadToHead:
    meetings: int = 0
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    home_goals_avg: float = 0.0
    away_goals_avg: float = 0.0
    #: goal-difference bias of the home side in past meetings, time-weighted,
    #: net of what the ratings already imply (computed by the model layer)
    weighted_gd: float = 0.0
    last_results: List[str] = field(default_factory=list)


def head_to_head(matches: Sequence[Match], home_id: int, away_id: int,
                 before: datetime, half_life_days: float = 900.0,
                 max_meetings: int = 12) -> HeadToHead:
    """Time-weighted summary of previous meetings, from the home side's view."""
    meetings = [m for m in matches
                if m.played and m.dt < before
                and {m.home_id, m.away_id} == {home_id, away_id}]
    meetings.sort(key=lambda m: m.dt, reverse=True)
    meetings = meetings[:max_meetings]

    h2h = HeadToHead(meetings=len(meetings))
    if not meetings:
        return h2h

    decay = 0.6931471805599453 / half_life_days
    wsum = 0.0
    gd_sum = 0.0
    hg = ag = 0.0
    for m in meetings:
        age = (before - m.dt).total_seconds() / 86400.0
        w = pow(2.718281828459045, -decay * age)
        # orient goals to the fixture's home side
        if m.home_id == home_id:
            g_for, g_against = m.home_goals, m.away_goals
        else:
            g_for, g_against = m.away_goals, m.home_goals
        gd_sum += w * (g_for - g_against)
        hg += w * g_for
        ag += w * g_against
        wsum += w

        if g_for > g_against:
            h2h.home_wins += 1
            h2h.last_results.append("W")
        elif g_for == g_against:
            h2h.draws += 1
            h2h.last_results.append("D")
        else:
            h2h.away_wins += 1
            h2h.last_results.append("L")

    h2h.weighted_gd = gd_sum / wsum
    h2h.home_goals_avg = hg / wsum
    h2h.away_goals_avg = ag / wsum
    return h2h


@dataclass
class Form:
    matches: int = 0
    points_per_game: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0
    results: List[str] = field(default_factory=list)


def form(matches: Sequence[Match], team_id: int, before: datetime,
         last: int = 6) -> Form:
    prior = team_match_history(matches, team_id, before)
    prior.sort(key=lambda m: m.dt, reverse=True)
    prior = prior[:last]
    f = Form(matches=len(prior))
    if not prior:
        return f
    pts = 0
    for m in prior:
        if m.home_id == team_id:
            g_for, g_against = m.home_goals, m.away_goals
        else:
            g_for, g_against = m.away_goals, m.home_goals
        f.goals_for += g_for
        f.goals_against += g_against
        if g_for > g_against:
            pts += 3
            f.results.append("W")
        elif g_for == g_against:
            pts += 1
            f.results.append("D")
        else:
            f.results.append("L")
    f.points_per_game = pts / len(prior)
    f.goals_for /= len(prior)
    f.goals_against /= len(prior)
    return f


# ----------------------------------------------------------------------- assembly
@dataclass
class MatchFeatures:
    home_id: int
    away_id: int
    kickoff: datetime
    rest_home: Optional[float]
    rest_away: Optional[float]
    congestion_home: int
    congestion_away: int
    availability_home: Availability
    availability_away: Availability
    h2h: HeadToHead
    form_home: Form
    form_away: Form
    neutral_venue: bool = False


def build_features(ds: Dataset, home_id: int, away_id: int, kickoff: datetime,
                   neutral_venue: bool = False) -> MatchFeatures:
    league = ds.played_matches          # ratings scale: league form and h2h
    sched = ds.schedule_matches         # fatigue: all competitions
    injuries = ds.injuries_by_team()
    players = ds.players_by_team()
    return MatchFeatures(
        home_id=home_id,
        away_id=away_id,
        kickoff=kickoff,
        rest_home=rest_days(sched, home_id, kickoff),
        rest_away=rest_days(sched, away_id, kickoff),
        congestion_home=congestion(sched, home_id, kickoff),
        congestion_away=congestion(sched, away_id, kickoff),
        availability_home=availability(home_id, injuries.get(home_id, []),
                                       players.get(home_id, [])),
        availability_away=availability(away_id, injuries.get(away_id, []),
                                       players.get(away_id, [])),
        h2h=head_to_head(league, home_id, away_id, kickoff),
        form_home=form(league, home_id, kickoff),
        form_away=form(league, away_id, kickoff),
        neutral_venue=neutral_venue,
    )
