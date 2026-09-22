"""Dixon-Coles goal model for the Saudi Pro League.

Each match's goal rates are

    log lambda_home = base + attack[home] - defence[away] + home_adv
                           + b_rest * rest_z(home) + b_cong * cong_c(home)
    log lambda_away = base + attack[away] - defence[home]
                           + b_rest * rest_z(away) + b_cong * cong_c(away)

fitted by weighted maximum likelihood with

  * Dixon-Coles low-score dependence (rho), which corrects the independent-Poisson
    under-prediction of 0-0/1-1 and over-prediction of 1-0/0-1,
  * exponential time decay, so recent form dominates without discarding history,
  * ridge shrinkage of attack/defence toward the league mean, which is what keeps
    a promoted side with six matches played from getting an extreme rating,
  * rest days and 14-day congestion as fitted covariates - these are computable
    from the fixture list for every historical match, so they are estimated from
    data rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import MODEL, ModelConfig
from .data import Dataset, Match
from . import features as F

REST_REF = 5.0      # days; the league's typical turnaround
REST_SCALE = 3.0
CONG_REF = 2.0      # matches in a 14-day window


def _rest_z(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return (value - REST_REF) / REST_SCALE


def _cong_c(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value) - CONG_REF


@dataclass
class Ratings:
    teams: List[int]
    attack: Dict[int, float]
    defence: Dict[int, float]
    base: float
    home_adv: float
    rho: float
    b_rest: float
    b_cong: float
    n_matches: int
    log_likelihood: float
    config: ModelConfig = field(default_factory=lambda: MODEL)

    def strength(self, team_id: int) -> Tuple[float, float]:
        return self.attack.get(team_id, 0.0), self.defence.get(team_id, 0.0)

    def table(self) -> List[Tuple[str, float, float, float]]:
        """(team_id, attack, defence, net) sorted by net strength."""
        rows = [(t, self.attack.get(t, 0.0), self.defence.get(t, 0.0),
                 self.attack.get(t, 0.0) + self.defence.get(t, 0.0))
                for t in self.teams]
        return sorted(rows, key=lambda r: r[3], reverse=True)


def _dc_tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
            rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction factor."""
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return np.clip(tau, 1e-10, None)


@dataclass
class _TrainingData:
    teams: List[int]
    index: Dict[int, int]
    home: np.ndarray
    away: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    weight: np.ndarray
    rest_h: np.ndarray
    rest_a: np.ndarray
    cong_h: np.ndarray
    cong_a: np.ndarray


def build_training_data(matches: Sequence[Match], as_of: Optional[datetime] = None,
                        cfg: Optional[ModelConfig] = None,
                        min_matches: int = 1,
                        schedule: Optional[Sequence[Match]] = None) -> _TrainingData:
    """Assemble the design arrays.

    `matches` are the league fixtures the ratings are fitted on. `schedule` is the
    all-competition fixture list used only to compute rest days and congestion; it
    defaults to `matches`, which understates fatigue for clubs playing midweek
    continental games.
    """
    cfg = cfg or MODEL
    played = sorted([m for m in matches if m.played], key=lambda m: m.dt)
    if as_of is not None:
        played = [m for m in played if m.dt < as_of]
    if not played:
        raise ValueError("no finished matches to fit on")

    ref = as_of or max(m.dt for m in played)

    counts: Dict[int, int] = {}
    for m in played:
        counts[m.home_id] = counts.get(m.home_id, 0) + 1
        counts[m.away_id] = counts.get(m.away_id, 0) + 1
    teams = sorted(t for t, c in counts.items() if c >= min_matches)
    index = {t: i for i, t in enumerate(teams)}
    keep = [m for m in played if m.home_id in index and m.away_id in index]
    if not keep:
        raise ValueError("no usable matches after filtering")

    sched = sorted([m for m in (schedule if schedule is not None else played)
                    if m.played and (as_of is None or m.dt < as_of)],
                   key=lambda m: m.dt)
    rest_h, rest_a, cong_h, cong_a = [], [], [], []
    for m in keep:
        rest_h.append(_rest_z(F.rest_days(sched, m.home_id, m.dt, cfg.rest_days_cap)))
        rest_a.append(_rest_z(F.rest_days(sched, m.away_id, m.dt, cfg.rest_days_cap)))
        cong_h.append(_cong_c(F.congestion(sched, m.home_id, m.dt,
                                           cfg.congestion_window_days)))
        cong_a.append(_cong_c(F.congestion(sched, m.away_id, m.dt,
                                           cfg.congestion_window_days)))

    ages = np.array([(ref - m.dt).total_seconds() / 86400.0 for m in keep])
    return _TrainingData(
        teams=teams,
        index=index,
        home=np.array([index[m.home_id] for m in keep], dtype=int),
        away=np.array([index[m.away_id] for m in keep], dtype=int),
        hg=np.array([m.home_goals for m in keep], dtype=float),
        ag=np.array([m.away_goals for m in keep], dtype=float),
        weight=np.exp(-cfg.time_decay * np.maximum(ages, 0.0)),
        rest_h=np.array(rest_h), rest_a=np.array(rest_a),
        cong_h=np.array(cong_h), cong_a=np.array(cong_a),
    )


def fit(matches: Sequence[Match], as_of: Optional[datetime] = None,
        cfg: Optional[ModelConfig] = None,
        schedule: Optional[Sequence[Match]] = None) -> Ratings:
    cfg = cfg or MODEL
    td = build_training_data(matches, as_of=as_of, cfg=cfg, schedule=schedule)
    n = len(td.teams)

    def unpack(theta: np.ndarray):
        base, home_adv, rho_raw, b_rest, b_cong = theta[:5]
        attack = theta[5:5 + n]
        defence = theta[5 + n:5 + 2 * n]
        rho = 0.18 * math.tanh(rho_raw)      # keeps rho in (-0.18, 0.18)
        return base, home_adv, rho, b_rest, b_cong, attack, defence

    const = (gammaln(td.hg + 1.0) + gammaln(td.ag + 1.0)) * td.weight

    def nll(theta: np.ndarray) -> float:
        base, home_adv, rho, b_rest, b_cong, attack, defence = unpack(theta)
        lin_h = base + home_adv + attack[td.home] - defence[td.away] \
            + b_rest * td.rest_h + b_cong * td.cong_h
        lin_a = base + attack[td.away] - defence[td.home] \
            + b_rest * td.rest_a + b_cong * td.cong_a
        lin_h = np.clip(lin_h, -6.0, 3.0)
        lin_a = np.clip(lin_a, -6.0, 3.0)
        lam, mu = np.exp(lin_h), np.exp(lin_a)
        tau = _dc_tau(td.hg, td.ag, lam, mu, rho)
        ll = td.weight * (td.hg * lin_h - lam + td.ag * lin_a - mu + np.log(tau))
        penalty = cfg.ridge * (np.sum(attack ** 2) + np.sum(defence ** 2))
        # Gaussian prior on the schedule effects (see ModelConfig.schedule_prior_sd)
        if cfg.schedule_prior_sd > 0:
            penalty += (b_rest ** 2 + b_cong ** 2) / (2.0 * cfg.schedule_prior_sd ** 2)
        # soft identification: attack and defence each average zero
        penalty += 50.0 * n * (attack.mean() ** 2 + defence.mean() ** 2)
        return float(-np.sum(ll - 0.0) + np.sum(const) + penalty)

    theta0 = np.zeros(5 + 2 * n)
    theta0[0] = math.log(max(0.6, float(np.average(td.hg, weights=td.weight))))
    theta0[1] = 0.2
    bounds = [(-2.0, 2.0), (-1.0, 1.5), (-3.0, 3.0), (-0.6, 0.6), (-0.6, 0.6)]
    bounds += [(-2.5, 2.5)] * (2 * n)

    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 3000, "maxfun": 200000, "ftol": 1e-10})

    base, home_adv, rho, b_rest, b_cong, attack, defence = unpack(res.x)
    attack = attack - attack.mean()
    defence = defence - defence.mean()

    return Ratings(
        teams=list(td.teams),
        attack={t: float(attack[i]) for t, i in td.index.items()},
        defence={t: float(defence[i]) for t, i in td.index.items()},
        base=float(base), home_adv=float(home_adv), rho=float(rho),
        b_rest=float(b_rest), b_cong=float(b_cong),
        n_matches=int(len(td.hg)),
        log_likelihood=float(-res.fun),
        config=cfg,
    )


# --------------------------------------------------------------------- prediction
@dataclass
class GoalRates:
    lam_home: float
    lam_away: float
    components: Dict[str, float] = field(default_factory=dict)


def goal_rates(ratings: Ratings, home_id: int, away_id: int,
               feats: Optional[F.MatchFeatures] = None,
               apply_injuries: bool = True,
               h2h_weight: float = 0.12) -> GoalRates:
    """Expected goals for a fixture, with schedule/injury/h2h adjustments."""
    cfg = ratings.config
    att_h, def_h = ratings.strength(home_id)
    att_a, def_a = ratings.strength(away_id)

    home_adv = 0.0 if (feats and feats.neutral_venue) else ratings.home_adv
    comp = {"base": ratings.base, "home_adv": home_adv,
            "attack_home": att_h, "defence_away": def_a,
            "attack_away": att_a, "defence_home": def_h}

    lin_h = ratings.base + home_adv + att_h - def_a
    lin_a = ratings.base + att_a - def_h

    if feats is not None:
        rest_h = ratings.b_rest * _rest_z(feats.rest_home)
        rest_a = ratings.b_rest * _rest_z(feats.rest_away)
        cong_h = ratings.b_cong * _cong_c(feats.congestion_home)
        cong_a = ratings.b_cong * _cong_c(feats.congestion_away)
        lin_h += rest_h + cong_h
        lin_a += rest_a + cong_a
        comp.update({"rest_home": rest_h, "rest_away": rest_a,
                     "congestion_home": cong_h, "congestion_away": cong_a})

        if apply_injuries:
            ah, aa = feats.availability_home, feats.availability_away
            # own attack suffers when attackers are missing; the opponent's rate
            # rises when the defence is depleted
            inj_h = (cfg.injury_attack_elasticity * (ah.attack_available - 1.0)
                     - cfg.injury_defence_elasticity * (aa.defence_available - 1.0))
            inj_a = (cfg.injury_attack_elasticity * (aa.attack_available - 1.0)
                     - cfg.injury_defence_elasticity * (ah.defence_available - 1.0))
            lin_h += inj_h
            lin_a += inj_a
            comp.update({"injury_home": inj_h, "injury_away": inj_a})

        if feats.h2h.meetings >= 3 and h2h_weight > 0:
            # residual head-to-head bias, net of what the ratings already imply
            implied = (lin_h - lin_a)
            observed = feats.h2h.weighted_gd
            resid = np.tanh((observed - implied) / 3.0)
            adj = h2h_weight * float(resid)
            lin_h += adj / 2.0
            lin_a -= adj / 2.0
            comp["h2h"] = adj

    lam_h = float(np.exp(np.clip(lin_h, -6.0, 3.0)))
    lam_a = float(np.exp(np.clip(lin_a, -6.0, 3.0)))
    return GoalRates(lam_home=lam_h, lam_away=lam_a, components=comp)


def score_matrix(lam_home: float, lam_away: float, rho: float,
                 max_goals: Optional[int] = None) -> np.ndarray:
    """Joint distribution over scorelines with the Dixon-Coles correction."""
    k = (MODEL.max_goals if max_goals is None else max_goals) + 1
    goals = np.arange(k)
    ph = np.exp(-lam_home + goals * np.log(max(lam_home, 1e-9)) - gammaln(goals + 1.0))
    pa = np.exp(-lam_away + goals * np.log(max(lam_away, 1e-9)) - gammaln(goals + 1.0))
    mat = np.outer(ph, pa)

    mat[0, 0] *= 1.0 - lam_home * lam_away * rho
    mat[0, 1] *= 1.0 + lam_home * rho
    mat[1, 0] *= 1.0 + lam_away * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0.0, None)
    total = mat.sum()
    return mat / total if total > 0 else mat


# --------------------------------------------------------------------- markets
def outcome_probabilities(mat: np.ndarray) -> Dict[str, float]:
    home = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    away = float(np.triu(mat, 1).sum())
    return {"home": home, "draw": draw, "away": away}


def market_summary(mat: np.ndarray) -> Dict[str, float]:
    k = mat.shape[0]
    idx = np.arange(k)
    total = idx[:, None] + idx[None, :]
    out = outcome_probabilities(mat)
    out["btts"] = float(mat[1:, 1:].sum())
    for line in (1.5, 2.5, 3.5, 4.5):
        out["over_%s" % line] = float(mat[total > line].sum())
        out["under_%s" % line] = float(mat[total < line].sum())
    out["home_cs"] = float(mat[:, 0].sum())
    out["away_cs"] = float(mat[0, :].sum())
    out["exp_goals_home"] = float((mat.sum(axis=1) * idx).sum())
    out["exp_goals_away"] = float((mat.sum(axis=0) * idx).sum())
    out["exp_total"] = out["exp_goals_home"] + out["exp_goals_away"]
    out["exp_supremacy"] = out["exp_goals_home"] - out["exp_goals_away"]
    return out


def top_scorelines(mat: np.ndarray, n: int = 6) -> List[Tuple[int, int, float]]:
    flat = [(i, j, float(mat[i, j])) for i in range(mat.shape[0])
            for j in range(mat.shape[1])]
    flat.sort(key=lambda r: r[2], reverse=True)
    return flat[:n]
