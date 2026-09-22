"""Walk-forward evaluation of the predictor.

For each match in the test window the model is refitted on matches strictly
before kickoff, so no information from the match (or later ones) leaks in. That
is the only honest way to judge a football model: in-sample fit is meaningless
because attack/defence ratings can memorise results.

Reported metrics
  log loss  : mean negative log probability of the actual 1X2 outcome (lower better)
  Brier     : mean squared error over the three outcome probabilities
  RPS       : ranked probability score, the standard football-forecast metric
  accuracy  : share of matches where the most likely outcome occurred
  MAE goals : mean absolute error of expected vs actual goals per side
  calibration: observed frequency inside predicted-probability buckets

Baselines: a fixed league-average prior and a bookmaker-free "home/draw/away
base rate" model, so the numbers have something to be better than.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import model as M
from . import aux_models as AUX
from . import features as F
from .config import MODEL, ModelConfig
from .data import Dataset, Match

EPS = 1e-12


def _rps(probs: Sequence[float], outcome_idx: int) -> float:
    """Ranked probability score over ordered outcomes (home, draw, away)."""
    cum_p = 0.0
    cum_o = 0.0
    total = 0.0
    for i in range(len(probs) - 1):
        cum_p += probs[i]
        cum_o += 1.0 if i == outcome_idx else 0.0
        total += (cum_p - cum_o) ** 2
    return total / (len(probs) - 1)


@dataclass
class BacktestResult:
    n: int = 0
    log_loss: float = 0.0
    brier: float = 0.0
    rps: float = 0.0
    accuracy: float = 0.0
    mae_goals: float = 0.0
    mae_shots: Optional[float] = None
    mae_possession: Optional[float] = None
    baseline_log_loss: float = 0.0
    baseline_rps: float = 0.0
    calibration: List[Tuple[str, int, float, float]] = field(default_factory=list)
    refits: int = 0

    def render(self) -> str:
        L = ["=" * 66, "  WALK-FORWARD BACKTEST", "=" * 66,
             "  matches evaluated       %d  (%d model refits)" % (self.n, self.refits),
             "  log loss (1X2)         %.4f   baseline %.4f   %+.1f%%"
             % (self.log_loss, self.baseline_log_loss,
                -100.0 * (self.log_loss - self.baseline_log_loss)
                / max(self.baseline_log_loss, EPS)),
             "  ranked prob. score     %.4f   baseline %.4f   %+.1f%%"
             % (self.rps, self.baseline_rps,
                -100.0 * (self.rps - self.baseline_rps) / max(self.baseline_rps, EPS)),
             "  Brier (3-way)          %.4f" % self.brier,
             "  outcome accuracy       %.1f%%" % (100.0 * self.accuracy),
             "  MAE expected goals     %.3f goals/side" % self.mae_goals]
        if self.mae_shots is not None:
            L.append("  MAE shots              %.2f shots/side" % self.mae_shots)
        if self.mae_possession is not None:
            L.append("  MAE possession         %.2f pp" % self.mae_possession)
        if self.calibration:
            L.append("")
            L.append("  calibration (predicted -> observed)")
            L.append("  %-14s %6s %10s %10s" % ("bucket", "n", "predicted", "observed"))
            for label, n, pred, obs in self.calibration:
                L.append("  %-14s %6d %9.1f%% %9.1f%%" % (label, n, 100 * pred, 100 * obs))
        L.append("=" * 66)
        return "\n".join(L)


def _calibration(pairs: List[Tuple[float, int]],
                 edges: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 1.01)
                 ) -> List[Tuple[str, int, float, float]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [(p, o) for p, o in pairs if lo <= p < hi]
        if len(bucket) < 5:
            continue
        preds = [p for p, _ in bucket]
        obs = [o for _, o in bucket]
        out.append(("%.0f-%.0f%%" % (100 * lo, 100 * min(hi, 1.0)), len(bucket),
                    float(np.mean(preds)), float(np.mean(obs))))
    return out


def backtest(ds: Dataset, start: Optional[datetime] = None,
             min_train_matches: int = 80, refit_every: int = 5,
             cfg: Optional[ModelConfig] = None,
             include_tempo: bool = True, log=None) -> BacktestResult:
    cfg = cfg or MODEL
    played = ds.played_matches
    if len(played) <= min_train_matches + 5:
        raise ValueError("need more than %d finished matches to backtest; have %d"
                         % (min_train_matches + 5, len(played)))

    test = [m for m in played[min_train_matches:] if start is None or m.dt >= start]
    if not test:
        raise ValueError("no matches in the test window")

    # league base rates from the training portion only, used as the baseline
    train_head = played[:min_train_matches]
    base_counts = [0, 0, 0]
    for m in train_head:
        if m.home_goals > m.away_goals:
            base_counts[0] += 1
        elif m.home_goals == m.away_goals:
            base_counts[1] += 1
        else:
            base_counts[2] += 1
    tot = max(1, sum(base_counts))
    baseline = [c / tot for c in base_counts]

    res = BacktestResult()
    ll = brier = rps = acc = mae_g = 0.0
    b_ll = b_rps = 0.0
    shot_err: List[float] = []
    poss_err: List[float] = []
    cal_pairs: List[Tuple[float, int]] = []

    ratings: Optional[M.Ratings] = None
    aux: Optional[AUX.AuxModels] = None
    stats_by_fixture = ds.stats_by_fixture()

    for i, match in enumerate(test):
        if ratings is None or i % max(1, refit_every) == 0:
            try:
                ratings = M.fit(ds.matches, as_of=match.dt, cfg=cfg,
                                schedule=ds.schedule_matches)
                aux = AUX.fit_aux(ds, as_of=match.dt, cfg=cfg) if include_tempo else None
            except ValueError:
                continue
            res.refits += 1
            if log:
                log("refit %d at %s (%d train matches)"
                    % (res.refits, match.dt.date(), ratings.n_matches))
        if ratings is None:
            continue
        if match.home_id not in ratings.attack or match.away_id not in ratings.attack:
            continue

        # features must also be computed as of kickoff; injuries are excluded here
        # because only the current injury list is retrievable, not a historical one
        hist = [m for m in ds.matches if m.played and m.dt < match.dt]
        sched = [m for m in ds.schedule_matches if m.dt < match.dt]
        feats = F.MatchFeatures(
            home_id=match.home_id, away_id=match.away_id, kickoff=match.dt,
            rest_home=F.rest_days(sched, match.home_id, match.dt, cfg.rest_days_cap),
            rest_away=F.rest_days(sched, match.away_id, match.dt, cfg.rest_days_cap),
            congestion_home=F.congestion(sched, match.home_id, match.dt,
                                         cfg.congestion_window_days),
            congestion_away=F.congestion(sched, match.away_id, match.dt,
                                         cfg.congestion_window_days),
            availability_home=F.Availability(match.home_id),
            availability_away=F.Availability(match.away_id),
            h2h=F.head_to_head(hist, match.home_id, match.away_id, match.dt),
            form_home=F.form(hist, match.home_id, match.dt),
            form_away=F.form(hist, match.away_id, match.dt),
        )
        rates = M.goal_rates(ratings, match.home_id, match.away_id, feats,
                             apply_injuries=False)
        mat = M.score_matrix(rates.lam_home, rates.lam_away, ratings.rho, cfg.max_goals)
        probs = M.outcome_probabilities(mat)
        p = [probs["home"], probs["draw"], probs["away"]]

        if match.home_goals > match.away_goals:
            idx = 0
        elif match.home_goals == match.away_goals:
            idx = 1
        else:
            idx = 2

        ll += -math.log(max(p[idx], EPS))
        b_ll += -math.log(max(baseline[idx], EPS))
        brier += sum((p[k] - (1.0 if k == idx else 0.0)) ** 2 for k in range(3)) / 3.0
        rps += _rps(p, idx)
        b_rps += _rps(baseline, idx)
        acc += 1.0 if max(range(3), key=lambda k: p[k]) == idx else 0.0
        mae_g += (abs(rates.lam_home - match.home_goals)
                  + abs(rates.lam_away - match.away_goals)) / 2.0
        for k in range(3):
            cal_pairs.append((p[k], 1 if k == idx else 0))

        if include_tempo and aux is not None and aux.models:
            actual = stats_by_fixture.get(match.fixture_id) or {}
            if actual:
                tempo = AUX.predict_tempo(aux, match.home_id, match.away_id,
                                          rates.lam_home, rates.lam_away, cfg=cfg)
                ah = actual.get(match.home_id)
                aa = actual.get(match.away_id)
                if ah and ah.shots is not None:
                    shot_err.append(abs(tempo.shots_home - ah.shots))
                if aa and aa.shots is not None:
                    shot_err.append(abs(tempo.shots_away - aa.shots))
                if ah and ah.possession is not None:
                    poss_err.append(abs(tempo.possession_home - ah.possession))
        res.n += 1

    if res.n == 0:
        raise ValueError("no matches could be evaluated")

    res.log_loss = ll / res.n
    res.brier = brier / res.n
    res.rps = rps / res.n
    res.accuracy = acc / res.n
    res.mae_goals = mae_g / res.n
    res.baseline_log_loss = b_ll / res.n
    res.baseline_rps = b_rps / res.n
    res.mae_shots = float(np.mean(shot_err)) if shot_err else None
    res.mae_possession = float(np.mean(poss_err)) if poss_err else None
    res.calibration = _calibration(cal_pairs)
    return res
