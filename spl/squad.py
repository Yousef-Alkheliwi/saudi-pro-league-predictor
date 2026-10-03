"""Squad predictor: the likely starting XI for a club's next match.

Built from published line-ups and each club's manager timeline. Every player
in the recent matchday squads gets a chance of starting from a logistic model
fitted on all recorded line-ups, over two kinds of signal:

  * the manager's preference - his share of starts under the *current*
    manager (a new man's habits are not his predecessor's), whether the
    manager is new, and whether he started last week;
  * recent performance - goals and assists per 90, whether he was taken off
    before the hour, whether he came on as a substitute, and whether he
    started a defeat.

The best goalkeeper and ten outfielders make the XI, kept to the balance of
defence, midfield and attack the side used last time. With too few line-ups
to fit the model, it falls back to the rule the evidence supports on its own:
last week's XI, less known absences.

Two absences come from the line-ups themselves:

  * suspended - sent off in the club's last match, and this is the first
    fixture since, so the ban falls here. Never picked, and passed to the goal
    model as a confirmed absence.
  * missed last squad - a regular who was not even among the substitutes last
    time. Possibly injured, possibly rested; nobody publishes which, so this
    is shown as a flag and does not change the pick.

There is no injury feed for this league. `evaluate` replays every past
line-up - the model cross-validated in chronological blocks - and scores it
against the rule and against simply repeating the previous XI.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Appearance, Dataset, Injury, Match

# Chosen by replaying all 675 recorded line-ups (`spl lineup --evaluate`):
#   decay 0.75 over 6 matches   8.59 of 11 named right
#   repeat the previous XI      8.70 of 11   (baseline)
#   decay 0.5 over 8 matches    8.76 of 11
# At 0.5 or below the latest XI outweighs all older line-ups together, so the
# model is "last week's XI, less known absences, gaps filled by recent
# history" - which is what the managers in this league mostly do. It departs
# from last week's XI in 42 of the 675 line-ups, all for a suspension, and is
# right in all 42. Tried and dropped, for not measurably helping: replacing a
# missing player like-for-like by position (8.74); docking a regular who
# missed the last squad (no change); counting a returning suspended player as
# if he had kept his place (+0.004, but 23 new wrong calls).
#: line-ups considered, most recent first
WINDOW = 8
#: weight of each older line-up relative to the one after it
DECAY = 0.5
#: started at least this share of the earlier line-ups to count as a regular
REGULAR = 0.5

#: depth up the pitch, by ESPN position abbreviation (goal = 0, attack = 4)
DEPTH = {
    "G": 0.0, "GK": 0.0,
    "SW": 0.8, "CD": 1.0, "CD-L": 1.0, "CD-R": 1.0, "CB": 1.0,
    "LB": 1.1, "RB": 1.1, "LWB": 1.6, "RWB": 1.6,
    "DM": 2.0, "DM-L": 2.0, "DM-R": 2.0, "CDM": 2.0,
    "CM": 2.5, "CM-L": 2.5, "CM-R": 2.5, "LM": 2.6, "RM": 2.6,
    "AM": 3.2, "AM-L": 3.2, "AM-R": 3.2, "CAM": 3.2,
    "LW": 3.6, "RW": 3.6, "LF": 3.8, "RF": 3.8,
    "CF": 4.0, "CF-L": 4.0, "CF-R": 4.0, "F": 4.0, "ST": 4.0, "SS": 3.8,
}
#: fallback from the squad list's position, for a player yet to start
SQUAD_DEPTH = {"Goalkeeper": 0.0, "Defender": 1.0, "Midfielder": 2.5,
               "Attacker": 4.0}


def depth(position: str, squad_position: Optional[str] = None) -> float:
    if position in DEPTH:
        return DEPTH[position]
    if squad_position in SQUAD_DEPTH:
        return SQUAD_DEPTH[squad_position]
    return 2.5


def lateral(position: str) -> float:
    """-1 is the team's left touchline, +1 its right."""
    p = position or ""
    if p.endswith("-L"):
        return -0.35
    if p.endswith("-R"):
        return 0.35
    if p.startswith("L") and p not in ("",):
        return -1.0
    if p.startswith("R"):
        return 1.0
    return 0.0


def is_goalkeeper(position: str, squad_position: Optional[str] = None) -> bool:
    return position in ("G", "GK") or (not position and squad_position == "Goalkeeper")


# --------------------------------------------------------------------- records
@dataclass
class LineupPlayer:
    player_id: int
    name: str
    short_name: str
    jersey: str
    position: str
    share: float              # recency-weighted share of recent starts
    starts: int               # starts in the window
    of: int                   # line-ups in the window
    row: int = 0              # 0 = goalkeeper, then lines up the pitch
    lateral: float = 0.0
    flag: str = ""            # e.g. "missed last squad"
    p_start: Optional[float] = None   # from the fitted model, when there is one


@dataclass
class Absence:
    player_id: int
    name: str
    kind: str                 # "suspended" | "missed last squad"
    reason: str


@dataclass
class PredictedLineup:
    team_id: int
    formation: str
    xi: List[LineupPlayer]
    bench: List[LineupPlayer]
    absences: List[Absence]
    matches_used: int
    last_match: Optional[datetime]
    manager: str = ""
    manager_since: Optional[datetime] = None
    method: str = "rule"      # "model" when the fitted start model chose the XI

    @property
    def rows(self) -> List[List[LineupPlayer]]:
        out: Dict[int, List[LineupPlayer]] = {}
        for p in self.xi:
            out.setdefault(p.row, []).append(p)
        return [sorted(out[r], key=lambda p: p.lateral) for r in sorted(out)]


# --------------------------------------------------------------------- history
def _team_history(ds: Dataset, team_id: int, before: datetime, window: int,
                  by_fixture: Optional[Dict[int, List[Appearance]]] = None,
                  sheets: Optional[Dict[Tuple[int, int], str]] = None
                  ) -> List[Tuple[Match, List[Appearance], str]]:
    """The club's most recent line-ups before `before`, newest first."""
    by_fixture = by_fixture if by_fixture is not None else ds.appearances_by_fixture()
    if sheets is None:
        sheets = {(t.fixture_id, t.team_id): t.formation for t in ds.team_sheets}
    out = []
    for m in sorted(ds.played_matches, key=lambda x: x.dt, reverse=True):
        if m.dt >= before or team_id not in (m.home_id, m.away_id):
            continue
        apps = [a for a in by_fixture.get(m.fixture_id, []) if a.team_id == team_id]
        if not apps:
            continue
        out.append((m, apps, sheets.get((m.fixture_id, team_id), "")))
        if len(out) >= window:
            break
    return out


def _first_fixture_since(ds: Dataset, team_id: int, last: Match,
                         kickoff: datetime) -> bool:
    """True if nothing on the club's league schedule falls between `last` and
    `kickoff` - so a ban earned in `last` is served at `kickoff`."""
    for m in ds.matches:
        if m.fixture_id == last.fixture_id or team_id not in (m.home_id, m.away_id):
            continue
        if last.dt < m.dt < kickoff - timedelta(hours=1):
            return False
    return True


def suspensions(ds: Dataset, team_id: int, kickoff: datetime,
                history=None) -> List[Injury]:
    """Players sent off in the club's last match, when `kickoff` is the next
    fixture - the ban falls on it. Returned as absences for the goal model."""
    history = history if history is not None else _team_history(ds, team_id,
                                                                 kickoff, 1)
    if not history:
        return []
    last, apps, _ = history[0]
    if not _first_fixture_since(ds, team_id, last, kickoff):
        return []
    return [Injury(team_id=team_id, player_id=a.player_id, player_name=a.name,
                   type="Missing Fixture",
                   reason="Suspended - sent off on %s" % last.dt.strftime("%d %b"),
                   origin="suspension")
            for a in apps if a.red_card]


# --------------------------------------------------------------------- predict
def predict_lineup(ds: Dataset, team_id: int, kickoff: Optional[datetime] = None,
                   window: int = WINDOW, decay: float = DECAY,
                   use_squad: bool = True, model="auto", keep_shape: bool = True,
                   _cache=None) -> Optional[PredictedLineup]:
    """The likely XI for `team_id` at `kickoff`, or None with no line-ups.

    `model` is a fitted SelectionModel, "auto" for the one fitted on this
    dataset, or None for the plain rule (last week's XI less absences).
    """
    kickoff = kickoff or datetime.now(timezone.utc)
    by_fixture, sheets = _cache if _cache else (None, None)
    long_history = _team_history(ds, team_id, kickoff, max(window, MANAGER_WINDOW),
                                 by_fixture, sheets)
    history = long_history[:window]
    if not history:
        return None
    if model == "auto":
        model = selection_model(ds)
    feats, meta = ({}, {}) if model is None else _player_features(
        ds, team_id, kickoff, history, long_history,
        _cache if _cache else _cache_for(ds))

    squad = {p.player_id: p for p in ds.players if p.team_id == team_id}
    registered = set(squad) if (use_squad and squad) else None

    weights = [decay ** k for k in range(len(history))]
    total = sum(weights)

    share: Dict[int, float] = {}
    starts: Dict[int, int] = {}
    latest: Dict[int, Appearance] = {}       # most recent appearance
    played: Dict[int, float] = {}            # minutes in the window
    start_pos: Dict[int, str] = {}           # position at most recent start
    last_start_age: Dict[int, int] = {}
    for k, ((m, apps, _), w) in enumerate(zip(history, weights)):
        for a in apps:
            latest.setdefault(a.player_id, a)
            played[a.player_id] = played.get(a.player_id, 0.0) + a.minutes
            if a.starter:
                share[a.player_id] = share.get(a.player_id, 0.0) + w
                starts[a.player_id] = starts.get(a.player_id, 0) + 1
                start_pos.setdefault(a.player_id, a.position)
                last_start_age.setdefault(a.player_id, k)
    for pid in share:
        share[pid] /= total

    last_match, last_apps, _ = history[0]
    in_last_squad = {a.player_id for a in last_apps}
    earlier_role = _start_positions_before(ds, kickoff)
    banned = {i.player_id: i for i in suspensions(ds, team_id, kickoff, history[:1])}

    absences: List[Absence] = []
    candidates: List[LineupPlayer] = []
    for pid, a in latest.items():
        if registered is not None and pid not in registered:
            continue                          # no longer at the club
        sq = squad.get(pid)
        # Position: his latest start in the window, else his latest start
        # anywhere before kick-off. Unused substitutes are only ever listed
        # as "SUB", so without either a reserve keeper looks like anyone else.
        pos = (start_pos.get(pid) or (a.position if a.position != "SUB" else "")
               or earlier_role.get(pid, ""))
        player = LineupPlayer(
            player_id=pid, name=a.name, short_name=a.short_name or a.name,
            jersey=a.jersey, position=pos, share=round(share.get(pid, 0.0), 4),
            starts=starts.get(pid, 0), of=len(history),
            lateral=lateral(pos))
        if model is not None and pid in feats:
            player.p_start = round(model.prob(feats[pid]), 3)
        if pid in banned:
            absences.append(Absence(pid, a.name, "suspended", banned[pid].reason))
            continue
        # A flag only: whether he was injured or rested is not published.
        # Counted on the earlier line-ups, because missing the last one is
        # exactly what pulls his weighted share down.
        earlier = sum(1 for _, apps, _ in history[1:]
                      if any(x.player_id == pid and x.starter for x in apps))
        if (len(history) > 1 and earlier >= REGULAR * (len(history) - 1)
                and pid not in in_last_squad):
            player.flag = "missed last squad"
            absences.append(Absence(pid, a.name, "missed last squad",
                                    "not in the matchday squad on %s"
                                    % last_match.dt.strftime("%d %b")))
        player.position = player.position or ""
        squad_pos = sq.position if sq else None
        player._depth = depth(player.position, squad_pos)
        player._gk = is_goalkeeper(player.position, squad_pos)
        # ESPN lists unused substitutes as "SUB", so a reserve keeper who has
        # never started has no position at all. Unknown players only fill a
        # gap after every known outfielder, or he ends up picked at full-back.
        player._known = bool(player.position) or squad_pos is not None
        candidates.append(player)

    def rank(p):
        # among players with no recent starts, minutes off the bench say more
        # than alphabetical order about who is next in line
        score = p.p_start if p.p_start is not None else p.share
        return (-score, last_start_age.get(p.player_id, 99), -p.starts,
                -played.get(p.player_id, 0.0), p.name)

    keepers = sorted([p for p in candidates if p._gk], key=rank)
    outfield = (sorted([p for p in candidates if not p._gk and p._known], key=rank)
                + sorted([p for p in candidates if not p._gk and not p._known], key=rank))
    chosen = (_keep_shape(outfield, last_apps, squad, earlier_role)
              if keep_shape and model is not None else outfield[:10])
    xi = keepers[:1] + chosen
    picked = {p.player_id for p in xi}
    bench = sorted([p for p in keepers[1:2] + outfield if p.player_id not in picked],
                   key=rank)[:7]

    # the current manager's usual shape, if he has a record here yet
    under = meta.get("under") or []
    formation = (_formation(under[:WINDOW], [decay ** k for k in range(len(under[:WINDOW]))])
                 if under else "") or _formation(history, weights)
    _assign_rows(xi, formation)
    for p in xi + bench:
        for attr in ("_depth", "_gk", "_known"):
            if hasattr(p, attr):
                delattr(p, attr)
    spell = ds.manager_at(team_id, kickoff) if ds.managers else None
    return PredictedLineup(team_id=team_id, formation=formation, xi=xi,
                           bench=bench, absences=absences,
                           matches_used=len(history), last_match=last_match.dt,
                           manager=spell.name if spell else "",
                           manager_since=(_parse_iso(spell.start) if spell else None),
                           method="model" if model is not None else "rule")


def _parse_iso(text: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _band(d: float) -> int:
    """Defence, midfield or attack, by depth up the pitch."""
    return 0 if d < 1.8 else (1 if d < 3.4 else 2)


def _keep_shape(ranked: List[LineupPlayer], last_apps: Sequence[Appearance],
                squad, earlier_role) -> List[LineupPlayer]:
    """The ten most likely outfielders, kept to last match's balance of
    defence, midfield and attack - a team sheet has a shape, and picking on
    probability alone can name five defenders for a back four. A band short
    of candidates is topped up from the rest in order."""
    shape: Dict[int, int] = {}
    for a in last_apps:
        sq = squad.get(a.player_id)
        pos = a.position if a.position != "SUB" else earlier_role.get(a.player_id, "")
        if a.starter and not is_goalkeeper(pos, sq.position if sq else None):
            b = _band(depth(pos, sq.position if sq else None))
            shape[b] = shape.get(b, 0) + 1
    if sum(shape.values()) != 10:
        return ranked[:10]
    chosen: List[LineupPlayer] = []
    for band, n in sorted(shape.items()):
        chosen += [p for p in ranked if _band(p._depth) == band][:n]
    taken = {p.player_id for p in chosen}
    for p in ranked:
        if len(chosen) >= 10:
            break
        if p.player_id not in taken:
            chosen.append(p)
            taken.add(p.player_id)
    return sorted(chosen[:10], key=lambda p: ranked.index(p))


def _start_positions_before(ds: Dataset, kickoff: datetime) -> Dict[int, str]:
    """Each player's position at his most recent start before `kickoff`."""
    when = {m.fixture_id: m.dt for m in ds.matches}
    out: Dict[int, Tuple[datetime, str]] = {}
    for a in ds.appearances:
        if not a.starter or not a.position or a.position == "SUB":
            continue
        dt = when.get(a.fixture_id)
        if dt is None or dt >= kickoff:
            continue
        if a.player_id not in out or dt > out[a.player_id][0]:
            out[a.player_id] = (dt, a.position)
    return {pid: pos for pid, (_, pos) in out.items()}


def _formation(history, weights) -> str:
    votes: Counter = Counter()
    for (_, _, shape), w in zip(history, weights):
        if shape and _lines(shape):
            votes[shape] += w
    return votes.most_common(1)[0][0] if votes else ""


def _lines(formation: str) -> Optional[List[int]]:
    try:
        parts = [int(x) for x in formation.split("-")]
    except (ValueError, AttributeError):
        return None
    if not parts or sum(parts) != 10 or any(x <= 0 for x in parts):
        return None
    return parts


def _assign_rows(xi: List[LineupPlayer], formation: str) -> None:
    """Goalkeeper in row 0; outfield players sorted by depth and dealt into the
    formation's lines. Without a usable formation, players are grouped into
    defence, midfield and attack by depth alone."""
    keepers = [p for p in xi if p._gk]
    outfield = sorted([p for p in xi if not p._gk],
                      key=lambda p: (p._depth, p.lateral))
    for p in keepers:
        p.row = 0
    lines = _lines(formation)
    if lines and sum(lines) == len(outfield):
        i = 0
        for r, n in enumerate(lines, start=1):
            for p in outfield[i:i + n]:
                p.row = r
            i += n
    else:
        for p in outfield:
            p.row = 1 if p._depth < 1.8 else (2 if p._depth < 3.4 else 3)


# --------------------------------------------------------------------- learned
# A logistic model of "does he start?", fitted on every recorded line-up. Two
# groups of signals, as asked for: the manager's preference for the player and
# the player's recent performance. Every feature uses only matches before the
# one being predicted, and the same code builds features for fitting and for
# prediction, so the two cannot drift apart.
FEATURES = (
    "started_last",       # started the club's previous match
    "recent_share",       # share of starts, latest line-ups weighted most
    "steady_share",       # plain share of starts over the window
    "in_last_squad",      # named in the last matchday squad at all
    "off_early_last",     # started last time but was replaced before 60'
    "on_last",            # came on as a substitute last time
    "manager_share",      # share of starts under the current manager
    "new_manager",        # the current manager has had fewer than 3 matches
    "new_manager_x_last", # started last week, but for a different manager
    "form",               # goals + assists per 90 over his last 5 outings
    "lost_x_started",     # started the previous match, and it was lost
)
#: line-ups read for the manager-specific share
MANAGER_WINDOW = 15
#: pseudo-matches pulling a new manager's share toward the club's history
MANAGER_PRIOR = 2.0
#: fewer training rows than this and the rule is used instead
MIN_TRAINING_ROWS = 400


def _player_features(ds: Dataset, team_id: int, kickoff: datetime, history,
                     long_history, cache) -> Tuple[Dict[int, Dict[str, float]], dict]:
    """Feature vectors for every candidate, plus what selection needs."""
    window = history[:WINDOW]
    weights = [DECAY ** k for k in range(len(window))]
    total = sum(weights) or 1.0
    last_match, last_apps, _ = window[0]
    starters_last = {a.player_id for a in last_apps if a.starter}
    squad_last = {a.player_id for a in last_apps}
    minutes_last = {a.player_id: a.minutes for a in last_apps}
    lost_last = _lost(last_match, team_id)

    manager = ds.manager_at(team_id, kickoff) if ds.managers else None
    under = [h for h in long_history
             if manager is not None
             and (ds.manager_at(team_id, h[0].dt) or manager).name == manager.name]
    last_manager = ds.manager_at(team_id, last_match.dt) if ds.managers else None
    changed = (manager is not None and last_manager is not None
               and last_manager.name != manager.name)

    share: Dict[int, float] = {}
    steady: Dict[int, int] = {}
    for (m, apps, _), w in zip(window, weights):
        for a in apps:
            if a.starter:
                share[a.player_id] = share.get(a.player_id, 0.0) + w / total
                steady[a.player_id] = steady.get(a.player_id, 0) + 1
    mgr_starts: Dict[int, int] = {}
    for m, apps, _ in under:
        for a in apps:
            if a.starter:
                mgr_starts[a.player_id] = mgr_starts.get(a.player_id, 0) + 1

    # recent output: goals and assists over each player's last 5 outings
    outings: Dict[int, List[Appearance]] = {}
    for m, apps, _ in long_history:
        for a in apps:
            if a.minutes > 0 and len(outings.setdefault(a.player_id, [])) < 5:
                outings[a.player_id].append(a)

    feats: Dict[int, Dict[str, float]] = {}
    seen = {a.player_id for _, apps, _ in window for a in apps}
    for pid in seen:
        steady_share = steady.get(pid, 0) / len(window)
        n_mgr = len(under)
        mgr_share = ((mgr_starts.get(pid, 0) + MANAGER_PRIOR * steady_share)
                     / (n_mgr + MANAGER_PRIOR))
        recent = outings.get(pid, [])
        mins = sum(a.minutes for a in recent)
        involvement = sum(a.goals + a.assists for a in recent)
        started = 1.0 if pid in starters_last else 0.0
        feats[pid] = {
            "started_last": started,
            "recent_share": share.get(pid, 0.0),
            "steady_share": steady_share,
            "in_last_squad": 1.0 if pid in squad_last else 0.0,
            "off_early_last": 1.0 if started and minutes_last.get(pid, 90) < 60 else 0.0,
            "on_last": 1.0 if not started and minutes_last.get(pid, 0) > 0 else 0.0,
            "manager_share": mgr_share,
            "new_manager": 1.0 if (manager is not None and n_mgr < 3) else 0.0,
            "new_manager_x_last": started if changed else 0.0,
            # shrunk toward zero with a pseudo-90 minutes, so one lucky goal
            # in a cameo does not read as world-class form
            "form": min(2.0, involvement / (mins / 90.0 + 1.0)),
            "lost_x_started": started if lost_last else 0.0,
        }
    meta = {"manager": manager, "under": under, "changed": changed}
    return feats, meta


def _lost(match: Match, team_id: int) -> bool:
    if match.home_goals is None:
        return False
    mine, theirs = ((match.home_goals, match.away_goals) if match.home_id == team_id
                    else (match.away_goals, match.home_goals))
    return mine < theirs


@dataclass
class SelectionModel:
    """Standardised logistic regression over FEATURES."""
    weights: List[float]
    bias: float
    means: List[float]
    scales: List[float]
    n_rows: int

    def prob(self, f: Dict[str, float]) -> float:
        import math
        z = self.bias + sum(w * (f[name] - mu) / sd for w, name, mu, sd
                            in zip(self.weights, FEATURES, self.means, self.scales))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def odds_ratios(self) -> List[Tuple[str, float]]:
        """Multiplier on the odds of starting for a one-unit change, so a
        yes/no signal reads as "how much more likely if true"."""
        import math
        return [(name, math.exp(w / sd)) for w, name, sd
                in zip(self.weights, FEATURES, self.scales)]


def _training_rows(ds: Dataset, matches: Iterable[Match], cache, min_history=3):
    by_fixture, sheets = cache
    rows, labels = [], []
    for m in matches:
        for tid in (m.home_id, m.away_id):
            actual = {a.player_id for a in by_fixture.get(m.fixture_id, [])
                      if a.team_id == tid and a.starter}
            if len(actual) != 11:
                continue
            long_history = _team_history(ds, tid, m.dt, MANAGER_WINDOW, by_fixture, sheets)
            if len(long_history) < min_history:
                continue
            banned = {i.player_id for i in suspensions(ds, tid, m.dt, long_history[:1])}
            feats, _ = _player_features(ds, tid, m.dt, long_history[:WINDOW],
                                        long_history, cache)
            for pid, f in feats.items():
                if pid in banned:
                    continue                # cannot start; not a selection call
                rows.append([f[name] for name in FEATURES])
                labels.append(1.0 if pid in actual else 0.0)
    return rows, labels


def fit_selection(ds: Dataset, matches: Optional[Iterable[Match]] = None,
                  ridge: float = 1.0, cache=None) -> Optional[SelectionModel]:
    """Fit the start model on line-ups; None if there is too little data."""
    import numpy as np
    from scipy.optimize import minimize
    cache = cache or _cache_for(ds)
    rows, labels = _training_rows(ds, matches if matches is not None
                                  else ds.played_matches, cache)
    if len(rows) < MIN_TRAINING_ROWS or len(set(labels)) < 2:
        return None
    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=float)
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales < 1e-9] = 1.0
    Z = (X - means) / scales

    def loss(theta):
        b, w = theta[0], theta[1:]
        z = b + Z @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        grad_z = p - y
        grad = np.concatenate([[grad_z.sum()], Z.T @ grad_z + 2 * ridge * w])
        return nll + ridge * np.sum(w ** 2), grad

    res = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return SelectionModel(weights=[float(v) for v in res.x[1:]], bias=float(res.x[0]),
                          means=[float(v) for v in means],
                          scales=[float(v) for v in scales], n_rows=len(rows))


def _cache_for(ds: Dataset):
    return (ds.appearances_by_fixture(),
            {(t.fixture_id, t.team_id): t.formation for t in ds.team_sheets})


def selection_model(ds: Dataset) -> Optional[SelectionModel]:
    """The model fitted on all line-ups, memoised on the dataset."""
    key = (id(ds.appearances), len(ds.appearances), len(ds.managers))
    cached = getattr(ds, "_selection", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    model = fit_selection(ds)
    ds._selection = (key, model)
    return model


# --------------------------------------------------------------------- evaluate
@dataclass
class LineupEvaluation:
    n: int = 0
    mean_correct: float = 0.0          # of 11, by the method under test
    rule_correct: float = 0.0          # last week's XI less known absences
    baseline_correct: float = 0.0      # repeat the previous XI
    all_eleven: float = 0.0            # share predicted exactly
    method: str = "rule"
    folds: int = 0
    calibration: List[Tuple[str, int, float, float]] = field(default_factory=list)
    odds: List[Tuple[str, float]] = field(default_factory=list)

    def render(self) -> str:
        label = ("fitted start model" if self.method == "model" else "rule")
        lines = ["=" * 66, "  SQUAD PREDICTOR - REPLAYED AGAINST PAST LINE-UPS", "=" * 66,
                 "  line-ups predicted         %d" % self.n]
        if self.method == "model":
            lines.append("  %-26s %.2f of 11   (%d-fold, chronological)"
                         % (label, self.mean_correct, self.folds))
        lines += ["  %-26s %.2f of 11" % ("last XI less suspensions", self.rule_correct),
                  "  %-26s %.2f of 11   (baseline)" % ("repeat last XI", self.baseline_correct),
                  "  %-26s %.1f%%" % ("whole XI exactly right", 100 * self.all_eleven)]
        if self.odds:
            lines += ["", "  what the model learned - how a signal moves the odds of starting",
                      "  (per unit; a yes/no signal is 1 when true)"]
            for name, ratio in self.odds:
                lines.append("    %-20s x%.2f" % (name, ratio))
        if self.calibration:
            lines += ["", "  calibration: predicted chance of starting vs how often he did",
                      "  %-12s %7s %10s %9s" % ("bucket", "players", "predicted", "started")]
            for label, n, pred, obs in self.calibration:
                lines.append("  %-12s %7d %9.0f%% %8.0f%%" % (label, n, 100 * pred, 100 * obs))
        lines.append("=" * 66)
        return "\n".join(lines)


def evaluate(ds: Dataset, window: int = WINDOW, decay: float = DECAY,
             min_history: int = 3, seasons: Optional[Iterable[int]] = None,
             method: str = "model", folds: int = 4) -> LineupEvaluation:
    """Predict every recorded line-up from the ones before it.

    With method="model" the start model is cross-validated: matches are cut
    into `folds` chronological blocks and each block is predicted by a model
    fitted on the others, so no line-up is predicted by a model that saw it.
    The rule and the repeat-last-XI baseline are scored on the same line-ups.
    The registered squad is not used: it is today's squad, and filtering past
    line-ups by it would leak who later left or arrived.
    """
    cache = _cache_for(ds)
    by_fixture, sheets = cache
    wanted = set(seasons) if seasons is not None else None
    cases = []
    for m in ds.played_matches:
        if wanted is not None and m.season not in wanted:
            continue
        for tid in (m.home_id, m.away_id):
            actual = {a.player_id for a in by_fixture.get(m.fixture_id, [])
                      if a.team_id == tid and a.starter}
            if len(actual) != 11:
                continue
            history = _team_history(ds, tid, m.dt, window, by_fixture, sheets)
            if len(history) >= min_history:
                cases.append((m, tid, actual, history))

    res = LineupEvaluation(method=method)
    if not cases:
        return res
    models: Dict[int, Optional[SelectionModel]] = {}
    fixtures = sorted({c[0].fixture_id: c[0] for c in cases}.values(), key=lambda m: m.dt)
    block_of = {m.fixture_id: min(folds - 1, i * folds // len(fixtures))
                for i, m in enumerate(fixtures)}
    if method == "model":
        played = ds.played_matches
        for b in range(folds):
            train = [m for m in played if block_of.get(m.fixture_id) != b]
            models[b] = fit_selection(ds, train, cache=cache)
        res.folds = folds
        full = selection_model(ds)
        if full is not None:
            res.odds = full.odds_ratios()

    correct = rule = base = exact = 0
    buckets: Dict[int, List[Tuple[float, int]]] = {}
    for m, tid, actual, history in cases:
        model = models.get(block_of[m.fixture_id]) if method == "model" else None
        pred = predict_lineup(ds, tid, m.dt, window, decay, use_squad=False,
                              model=model, _cache=cache)
        plain = (pred if model is None else
                 predict_lineup(ds, tid, m.dt, window, decay, use_squad=False,
                                model=None, _cache=cache))
        hit = len({p.player_id for p in pred.xi} & actual)
        correct += hit
        exact += hit == 11
        rule += len({p.player_id for p in plain.xi} & actual)
        base += len({a.player_id for a in history[0][1] if a.starter} & actual)
        res.n += 1
        if model is not None:
            for p in pred.xi + pred.bench:
                if p.p_start is not None:
                    b = min(int(p.p_start * 10), 9)
                    buckets.setdefault(b, []).append((p.p_start, 1 if p.player_id in actual else 0))
    res.mean_correct = correct / res.n
    res.rule_correct = rule / res.n
    res.baseline_correct = base / res.n
    res.all_eleven = exact / res.n
    for b in sorted(buckets):
        pts = buckets[b]
        if len(pts) >= 20:
            res.calibration.append(("%d-%d%%" % (10 * b, 10 * (b + 1)), len(pts),
                                    sum(p for p, _ in pts) / len(pts),
                                    sum(o for _, o in pts) / len(pts)))
    return res
