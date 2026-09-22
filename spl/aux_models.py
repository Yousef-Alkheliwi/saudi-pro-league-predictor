"""Shots, shots-on-target, possession and corners models.

Same additive structure as the goal model, fitted by weighted ridge regression on
the per-team box scores:

    possession_ij = c + p_i - p_j + h          (identity link; sums to 100 by design)
    log(shots_ij)  = c + s_i - q_j + h         (log link, so effects are multiplicative)

`p_i` is a team's possession pull, `q_j` the opponent's shot suppression. The
design is antisymmetric in (team, opponent), which is what makes the two sides of
a predicted match consistent with each other. When there are too few box scores
to fit, the module falls back to league averages scaled by the goal model's rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MODEL, ModelConfig
from .data import Dataset, Match, TeamStats

MIN_ROWS_PER_TEAM = 2
MIN_ROWS = 24

LEAGUE_FALLBACK = {
    # league-wide priors, only used when box scores are unavailable
    "shots": 12.0,
    "shots_on_target": 4.3,
    "possession": 50.0,
    "corners": 4.6,
}


@dataclass
class TeamEffectModel:
    metric: str
    link: str                   # "log" or "identity"
    intercept: float
    home: float
    offence: Dict[int, float]
    suppression: Dict[int, float]
    n_rows: int
    rmse: float
    league_mean: float

    def predict(self, team_id: int, opponent_id: int, at_home: bool) -> float:
        lin = (self.intercept
               + self.offence.get(team_id, 0.0)
               - self.suppression.get(opponent_id, 0.0)
               + (self.home if at_home else 0.0))
        if self.link == "log":
            return float(math.exp(min(max(lin, -3.0), 5.0)))
        return float(lin)


@dataclass
class AuxModels:
    models: Dict[str, TeamEffectModel] = field(default_factory=dict)
    fallback_note: str = ""

    def has(self, metric: str) -> bool:
        return metric in self.models


# --------------------------------------------------------------------------- rows
@dataclass
class _Row:
    team: int
    opponent: int
    at_home: bool
    weight: float
    values: Dict[str, float]


def _collect_rows(ds: Dataset, as_of: Optional[datetime] = None,
                  cfg: Optional[ModelConfig] = None) -> List[_Row]:
    cfg = cfg or MODEL
    by_fixture = ds.stats_by_fixture()
    matches = {m.fixture_id: m for m in ds.matches}
    played = [m for m in ds.played_matches if (as_of is None or m.dt < as_of)]
    if not played:
        return []
    ref = as_of or max(m.dt for m in played)

    rows: List[_Row] = []
    for match in played:
        per_team = by_fixture.get(match.fixture_id)
        if not per_team:
            continue
        age = (ref - match.dt).total_seconds() / 86400.0
        weight = math.exp(-cfg.time_decay * max(age, 0.0))
        for team_id, st in per_team.items():
            opponent = match.away_id if team_id == match.home_id else match.home_id
            if opponent == team_id:
                continue
            values = {}
            for metric in ("shots", "shots_on_target", "possession", "corners"):
                val = getattr(st, metric, None)
                if val is not None and val >= 0:
                    values[metric] = float(val)
            if values:
                rows.append(_Row(team=team_id, opponent=opponent,
                                 at_home=team_id == match.home_id,
                                 weight=weight, values=values))
    return rows


def _fit_metric(rows: Sequence[_Row], metric: str, link: str,
                ridge: float) -> Optional[TeamEffectModel]:
    usable = [r for r in rows if metric in r.values]
    if len(usable) < MIN_ROWS:
        return None

    teams = sorted({r.team for r in usable} | {r.opponent for r in usable})
    counts: Dict[int, int] = {}
    for r in usable:
        counts[r.team] = counts.get(r.team, 0) + 1
    teams = [t for t in teams if counts.get(t, 0) >= MIN_ROWS_PER_TEAM]
    if len(teams) < 4:
        return None
    idx = {t: i for i, t in enumerate(teams)}
    usable = [r for r in usable if r.team in idx and r.opponent in idx]
    if len(usable) < MIN_ROWS:
        return None

    n = len(teams)
    X = np.zeros((len(usable), 2 + 2 * n))
    y = np.zeros(len(usable))
    w = np.zeros(len(usable))
    for i, r in enumerate(usable):
        X[i, 0] = 1.0
        X[i, 1] = 1.0 if r.at_home else 0.0
        X[i, 2 + idx[r.team]] = 1.0
        X[i, 2 + n + idx[r.opponent]] = -1.0
        raw = r.values[metric]
        y[i] = math.log(max(raw, 0.5)) if link == "log" else raw
        w[i] = r.weight

    # weighted ridge; the intercept and home term are left unpenalised
    pen = np.eye(2 + 2 * n) * ridge
    pen[0, 0] = pen[1, 1] = 0.0
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    try:
        beta = np.linalg.solve(Xw.T @ Xw + pen, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(Xw.T @ Xw + pen, Xw.T @ yw, rcond=None)

    resid = y - X @ beta
    rmse = float(np.sqrt(np.average(resid ** 2, weights=w)))
    league_mean = float(np.average([r.values[metric] for r in usable], weights=w))

    off = {t: float(beta[2 + i]) for t, i in idx.items()}
    sup = {t: float(beta[2 + n + i]) for t, i in idx.items()}
    # centre the team effects so the intercept carries the league level
    off_mean = float(np.mean(list(off.values())))
    sup_mean = float(np.mean(list(sup.values())))
    off = {t: v - off_mean for t, v in off.items()}
    sup = {t: v - sup_mean for t, v in sup.items()}

    return TeamEffectModel(
        metric=metric, link=link, intercept=float(beta[0]) + off_mean - sup_mean,
        home=float(beta[1]), offence=off, suppression=sup,
        n_rows=len(usable), rmse=rmse, league_mean=league_mean,
    )


def fit_aux(ds: Dataset, as_of: Optional[datetime] = None,
            cfg: Optional[ModelConfig] = None) -> AuxModels:
    cfg = cfg or MODEL
    rows = _collect_rows(ds, as_of=as_of, cfg=cfg)
    out = AuxModels()
    if not rows:
        out.fallback_note = ("no box scores available - shots/possession fall back to "
                             "league priors scaled by the goal model")
        return out
    specs = (("shots", "log"), ("shots_on_target", "log"),
             ("possession", "identity"), ("corners", "log"))
    for metric, link in specs:
        model = _fit_metric(rows, metric, link, cfg.aux_ridge)
        if model is not None:
            out.models[metric] = model
    if not out.models:
        out.fallback_note = ("only %d box-score rows - too few to fit; using league "
                             "priors" % len(rows))
    return out


# --------------------------------------------------------------------- prediction
@dataclass
class TempoPrediction:
    possession_home: float
    possession_away: float
    shots_home: float
    shots_away: float
    sot_home: float
    sot_away: float
    corners_home: Optional[float]
    corners_away: Optional[float]
    source: str
    sample_rows: int = 0


def _fallback_tempo(lam_home: float, lam_away: float,
                    supremacy_hint: Optional[float] = None) -> TempoPrediction:
    if supremacy_hint is None:
        supremacy_hint = (lam_home - lam_away) / 1.5
    league_goals = 1.35
    shots_h = LEAGUE_FALLBACK["shots"] * (0.55 + 0.45 * lam_home / league_goals)
    shots_a = LEAGUE_FALLBACK["shots"] * (0.55 + 0.45 * lam_away / league_goals)
    ratio = LEAGUE_FALLBACK["shots_on_target"] / LEAGUE_FALLBACK["shots"]
    poss_h = 50.0 + 11.0 * math.tanh(supremacy_hint)
    return TempoPrediction(
        possession_home=poss_h, possession_away=100.0 - poss_h,
        shots_home=shots_h, shots_away=shots_a,
        sot_home=shots_h * ratio, sot_away=shots_a * ratio,
        corners_home=None, corners_away=None,
        source="league priors scaled by expected goals",
    )


def predict_tempo(aux: AuxModels, home_id: int, away_id: int,
                  lam_home: float, lam_away: float,
                  avail_home: float = 1.0, avail_away: float = 1.0,
                  cfg: Optional[ModelConfig] = None,
                  supremacy_hint: Optional[float] = None) -> TempoPrediction:
    """Predicted shots/possession, adjusted for squad availability."""
    cfg = cfg or MODEL
    if supremacy_hint is None:
        supremacy_hint = (lam_home - lam_away) / 1.5
    if not aux.models:
        return _fallback_tempo(lam_home, lam_away, supremacy_hint)

    def _get(metric: str, team: int, opp: int, at_home: bool) -> Optional[float]:
        model = aux.models.get(metric)
        if model is None:
            return None
        return model.predict(team, opp, at_home)

    shots_h = _get("shots", home_id, away_id, True)
    shots_a = _get("shots", away_id, home_id, False)
    sot_h = _get("shots_on_target", home_id, away_id, True)
    sot_a = _get("shots_on_target", away_id, home_id, False)
    poss_h = _get("possession", home_id, away_id, True)
    poss_a = _get("possession", away_id, home_id, False)
    corners_h = _get("corners", home_id, away_id, True)
    corners_a = _get("corners", away_id, home_id, False)

    if shots_h is None or shots_a is None:
        base = _fallback_tempo(lam_home, lam_away, supremacy_hint)
        shots_h = shots_h or base.shots_home
        shots_a = shots_a or base.shots_away
    ratio = LEAGUE_FALLBACK["shots_on_target"] / LEAGUE_FALLBACK["shots"]
    sot_h = sot_h if sot_h is not None else shots_h * ratio
    sot_a = sot_a if sot_a is not None else shots_a * ratio

    # possession: average the two one-sided predictions, then renormalise to 100
    if poss_h is None and poss_a is None:
        poss_h = 50.0 + 11.0 * math.tanh(supremacy_hint)
        poss_a = 100.0 - poss_h
    elif poss_h is None:
        poss_h = 100.0 - poss_a
    elif poss_a is None:
        poss_a = 100.0 - poss_h
    mid = (poss_h + (100.0 - poss_a)) / 2.0

    # availability: missing players cost shot volume and possession control
    k_shots = cfg.injury_shots_elasticity
    shots_h *= math.exp(k_shots * (avail_home - 1.0))
    shots_a *= math.exp(k_shots * (avail_away - 1.0))
    sot_h *= math.exp(k_shots * (avail_home - 1.0))
    sot_a *= math.exp(k_shots * (avail_away - 1.0))
    mid += cfg.injury_possession_points * ((avail_home - 1.0) - (avail_away - 1.0))
    mid = float(min(max(mid, 25.0), 75.0))

    if corners_h is not None:
        corners_h *= math.exp(0.5 * k_shots * (avail_home - 1.0))
    if corners_a is not None:
        corners_a *= math.exp(0.5 * k_shots * (avail_away - 1.0))

    rows = max((m.n_rows for m in aux.models.values()), default=0)
    return TempoPrediction(
        possession_home=mid, possession_away=100.0 - mid,
        shots_home=float(shots_h), shots_away=float(shots_a),
        sot_home=float(sot_h), sot_away=float(sot_a),
        corners_home=corners_h, corners_away=corners_a,
        source="fitted on %d team-match box scores" % rows,
        sample_rows=rows,
    )
