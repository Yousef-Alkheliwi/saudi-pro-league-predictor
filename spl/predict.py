"""Orchestration: dataset -> fitted models -> a single match prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import aux_models as AUX
from . import features as F
from . import model as M
from .config import MODEL, ModelConfig
from .data import Dataset, Match


@dataclass
class Prediction:
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    kickoff: datetime
    fixture_id: Optional[int]
    venue: Optional[str]

    lam_home: float
    lam_away: float
    markets: Dict[str, float]
    scorelines: List[Tuple[int, int, float]]
    tempo: AUX.TempoPrediction
    features: F.MatchFeatures
    components: Dict[str, float]
    ratings: M.Ratings
    #: newest finished match the ratings are based on
    data_as_of: Optional[datetime] = None
    #: True when the subscription blocked recent seasons
    plan_limited: bool = False
    #: True when no live injury list was retrievable
    injuries_available: bool = True

    @property
    def data_age_days(self) -> Optional[float]:
        if self.data_as_of is None:
            return None
        return (self.kickoff - self.data_as_of).total_seconds() / 86400.0

    @property
    def most_likely_score(self) -> Tuple[int, int, float]:
        return self.scorelines[0]

    def confidence(self) -> str:
        """Qualitative read on how much data backs this prediction.

        Data vintage dominates: ratings built from a season that ended a year ago
        describe squads that no longer exist, whatever the sample size.
        """
        age = self.data_age_days
        if age is not None and age > 180:
            return "low - data is %.0f months stale" % (age / 30.0)
        n_home = self.features.form_home.matches
        n_away = self.features.form_away.matches
        rows = self.tempo.sample_rows
        if age is not None and age > 45:
            return "low"
        if min(n_home, n_away) >= 5 and self.ratings.n_matches >= 150 and rows >= 60:
            return "high"
        if min(n_home, n_away) >= 3 and self.ratings.n_matches >= 60:
            return "medium"
        return "low"


class Predictor:
    def __init__(self, ds: Dataset, cfg: Optional[ModelConfig] = None,
                 as_of: Optional[datetime] = None):
        self.ds = ds
        self.cfg = cfg or MODEL
        self.as_of = as_of
        self.ratings = M.fit(ds.matches, as_of=as_of, cfg=self.cfg,
                             schedule=ds.schedule_matches)
        self.aux = AUX.fit_aux(ds, as_of=as_of, cfg=self.cfg)

    # ------------------------------------------------------------------ core
    def predict(self, home_id: int, away_id: int,
                kickoff: Optional[datetime] = None,
                fixture_id: Optional[int] = None,
                venue: Optional[str] = None,
                neutral_venue: bool = False,
                apply_injuries: bool = True) -> Prediction:
        if home_id == away_id:
            name = self.ds.teams.get(home_id, home_id)
            raise ValueError("a club cannot play itself (%s)" % name)
        kickoff = kickoff or datetime.now(timezone.utc) + timedelta(days=3)
        feats = F.build_features(self.ds, home_id, away_id, kickoff,
                                 neutral_venue=neutral_venue)
        rates = M.goal_rates(self.ratings, home_id, away_id, feats,
                             apply_injuries=apply_injuries)
        mat = M.score_matrix(rates.lam_home, rates.lam_away, self.ratings.rho,
                             self.cfg.max_goals)
        markets = M.market_summary(mat)
        tempo = AUX.predict_tempo(
            self.aux, home_id, away_id, rates.lam_home, rates.lam_away,
            avail_home=feats.availability_home.overall_available if apply_injuries else 1.0,
            avail_away=feats.availability_away.overall_available if apply_injuries else 1.0,
            cfg=self.cfg,
            supremacy_hint=(rates.lam_home - rates.lam_away) / 1.5,
        )
        played = self.ds.played_matches
        return Prediction(
            data_as_of=played[-1].dt if played else None,
            plan_limited=self.ds.plan_limited,
            injuries_available=bool(self.ds.injuries),
            home_id=home_id, away_id=away_id,
            home_name=self.ds.teams.get(home_id, str(home_id)),
            away_name=self.ds.teams.get(away_id, str(away_id)),
            kickoff=kickoff, fixture_id=fixture_id, venue=venue,
            lam_home=rates.lam_home, lam_away=rates.lam_away,
            markets=markets, scorelines=M.top_scorelines(mat, 8),
            tempo=tempo, features=feats, components=rates.components,
            ratings=self.ratings,
        )

    def predict_fixture(self, match: Match, **kwargs) -> Prediction:
        return self.predict(match.home_id, match.away_id, kickoff=match.dt,
                            fixture_id=match.fixture_id, venue=match.venue, **kwargs)

    def predict_by_name(self, home: str, away: str,
                        kickoff: Optional[datetime] = None, **kwargs) -> Prediction:
        home_id, _ = self.ds.resolve_team(home)
        away_id, _ = self.ds.resolve_team(away)
        scheduled = self._find_scheduled(home_id, away_id)
        if kickoff is None and scheduled is not None:
            return self.predict_fixture(scheduled, **kwargs)
        return self.predict(home_id, away_id, kickoff=kickoff, **kwargs)

    def _find_scheduled(self, home_id: int, away_id: int) -> Optional[Match]:
        """The next scheduled meeting, preferring a future kickoff.

        A fixture left in "not started" after its date has passed is postponed or
        abandoned, so it must not be picked ahead of the real next meeting.
        """
        now = self.as_of or datetime.now(timezone.utc)
        candidates = [m for m in self.ds.upcoming_matches
                      if m.home_id == home_id and m.away_id == away_id]
        future = [m for m in candidates if m.dt >= now]
        if future:
            return future[0]
        return None

    def upcoming(self, days: int = 10, limit: int = 20) -> List[Match]:
        now = self.as_of or datetime.now(timezone.utc)
        horizon = now + timedelta(days=days)
        return [m for m in self.ds.upcoming_matches if now <= m.dt <= horizon][:limit]


# --------------------------------------------------------------------- rendering
def _data_warnings(pred: Prediction) -> List[str]:
    """Anything that makes the numbers below less trustworthy than they look."""
    out: List[str] = []
    age = pred.data_age_days
    if age is not None and age > 45:
        out.append("Ratings are built from matches up to %s - %.0f days before this"
                   % (pred.data_as_of.strftime("%d %b %Y"), age))
        out.append("kickoff. Squads, managers and form have moved on since. These are")
        out.append("NOT current-form predictions.")
    if pred.plan_limited:
        out.append("The API subscription does not cover recent seasons, so no")
        out.append("current-season data could be fetched.")
    if not pred.injuries_available:
        out.append("No injury data available - every squad is treated as fully fit.")
    unrated = [name for tid, name in ((pred.home_id, pred.home_name),
                                      (pred.away_id, pred.away_name))
               if tid not in pred.ratings.attack]
    if unrated:
        out.append("%s ha%s no matches in the fitted ratings, so %s treated as"
                   % (" and ".join(unrated), "ve" if len(unrated) > 1 else "s",
                      "they are" if len(unrated) > 1 else "it is"))
        out.append("exactly league-average. These numbers carry no information "
                   "about that club.")
    if pred.ratings.at_bounds:
        out.append("The fit pushed %s onto %s limit, so %s clamped rather than "
                   % (", ".join(pred.ratings.at_bounds),
                      "their" if len(pred.ratings.at_bounds) > 1 else "its",
                      "they are" if len(pred.ratings.at_bounds) > 1 else "it is"))
        out.append("estimated - the data wanted to go further. Usually a sign of "
                   "too few matches.")
    if not pred.ratings.converged:
        out.append("The rating fit did not converge (%s). Treat every number "
                   "below as unreliable." % pred.ratings.fit_message)
    return out


def _bar(pct: float, width: int = 22) -> str:
    filled = int(round(max(0.0, min(1.0, pct)) * width))
    return "#" * filled + "." * (width - filled)


def _fmt_pct(p: float) -> str:
    return "%5.1f%%" % (100.0 * p)


def _odds(p: float) -> str:
    return "%.2f" % (1.0 / p) if p > 1e-6 else "  -  "


def render(pred: Prediction, verbose: bool = False) -> str:
    m = pred.markets
    f = pred.features
    t = pred.tempo
    L: List[str] = []
    W = 78

    L.append("=" * W)
    L.append("  %s  vs  %s" % (pred.home_name.upper(), pred.away_name.upper()))
    when = pred.kickoff.strftime("%a %d %b %Y  %H:%M UTC")
    site = pred.venue or ("neutral venue" if f.neutral_venue else "home venue")
    L.append("  %s   |   %s" % (when, site))
    L.append("=" * W)

    warnings = _data_warnings(pred)
    if warnings:
        L.append("")
        L.append("!! READ THIS FIRST")
        for line in warnings:
            L.append("   " + line)

    L.append("")
    L.append("RESULT")
    for label, key in (("%s win" % pred.home_name, "home"), ("Draw", "draw"),
                       ("%s win" % pred.away_name, "away")):
        L.append("  %-26s %s  %s   fair odds %s"
                 % (label[:26], _fmt_pct(m[key]), _bar(m[key]), _odds(m[key])))

    L.append("")
    L.append("GOALS")
    L.append("  Expected goals            %-22s %s"
             % ("%s %.2f" % (pred.home_name[:14], m["exp_goals_home"]),
                "%s %.2f" % (pred.away_name[:14], m["exp_goals_away"])))
    L.append("  Expected total / margin   %.2f goals   |   %+.2f to %s"
             % (m["exp_total"], m["exp_supremacy"],
                pred.home_name if m["exp_supremacy"] >= 0 else pred.away_name))
    hs, as_, hp = pred.most_likely_score
    L.append("  Most likely score         %d-%d  (%s)" % (hs, as_, _fmt_pct(hp)))
    L.append("  Other likely scores       " + ",  ".join(
        "%d-%d %s" % (i, j, _fmt_pct(p)) for i, j, p in pred.scorelines[1:5]))
    L.append("  Over 2.5 / Under 2.5      %s / %s"
             % (_fmt_pct(m["over_2.5"]), _fmt_pct(m["under_2.5"])))
    L.append("  Over 1.5 / Over 3.5       %s / %s"
             % (_fmt_pct(m["over_1.5"]), _fmt_pct(m["over_3.5"])))
    L.append("  Both teams to score       %s" % _fmt_pct(m["btts"]))
    L.append("  Clean sheet               %s %s   |   %s %s"
             % (pred.home_name[:14], _fmt_pct(m["home_cs"]),
                pred.away_name[:14], _fmt_pct(m["away_cs"])))

    L.append("")
    L.append("SHOTS & POSSESSION")
    L.append("  %-24s %-14s %-14s" % ("", pred.home_name[:14], pred.away_name[:14]))
    L.append("  %-24s %-14.1f %-14.1f" % ("Possession %", t.possession_home,
                                          t.possession_away))
    L.append("  %-24s %-14.1f %-14.1f" % ("Total shots", t.shots_home, t.shots_away))
    L.append("  %-24s %-14.1f %-14.1f" % ("Shots on target", t.sot_home, t.sot_away))
    if t.corners_home is not None and t.corners_away is not None:
        L.append("  %-24s %-14.1f %-14.1f" % ("Corners", t.corners_home, t.corners_away))
    L.append("  %-24s %-14.2f %-14.2f" % ("Goals conceded (exp)", m["exp_goals_away"],
                                          m["exp_goals_home"]))
    L.append("  source: %s" % t.source)

    L.append("")
    L.append("WHY")
    att_h, def_h = pred.ratings.strength(pred.home_id)
    att_a, def_a = pred.ratings.strength(pred.away_id)
    L.append("  Ratings        %s att %+.2f / def %+.2f   |   %s att %+.2f / def %+.2f"
             % (pred.home_name[:12], att_h, def_h, pred.away_name[:12], att_a, def_a))
    L.append("  Home advantage %+.2f on log goal rate (x%.2f)"
             % (pred.ratings.home_adv, np.exp(pred.ratings.home_adv)))
    rest_h = "n/a" if f.rest_home is None else "%.1fd" % f.rest_home
    rest_a = "n/a" if f.rest_away is None else "%.1fd" % f.rest_away
    L.append("  Rest days      %s %s (%d in last 14d)   |   %s %s (%d in last 14d)"
             % (pred.home_name[:12], rest_h, f.congestion_home,
                pred.away_name[:12], rest_a, f.congestion_away))
    if "rest_home" in pred.components:
        L.append("                 effect on goal rate: %+.3f home / %+.3f away "
                 "(fitted b_rest=%+.3f, b_congestion=%+.3f)"
                 % (pred.components.get("rest_home", 0) + pred.components.get("congestion_home", 0),
                    pred.components.get("rest_away", 0) + pred.components.get("congestion_away", 0),
                    pred.ratings.b_rest, pred.ratings.b_cong))

    ah, aa = f.availability_home, f.availability_away
    if not pred.injuries_available:
        L.append("  Availability   no injury feed on this plan - both squads assumed "
                 "fully fit")
    L.append("  Availability   %s %.0f%% of weighted minutes (%s)"
             % (pred.home_name[:12], 100 * ah.overall_available, ah.note))
    if ah.out:
        L.append("                 out: %s" % "; ".join(ah.out[:6]))
    if ah.doubtful:
        L.append("                 doubtful: %s" % "; ".join(ah.doubtful[:5]))
    L.append("  Availability   %s %.0f%% of weighted minutes (%s)"
             % (pred.away_name[:12], 100 * aa.overall_available, aa.note))
    if aa.out:
        L.append("                 out: %s" % "; ".join(aa.out[:6]))
    if aa.doubtful:
        L.append("                 doubtful: %s" % "; ".join(aa.doubtful[:5]))

    h = f.h2h
    if h.meetings:
        L.append("  Head to head   %d meetings: %dW-%dD-%dL for %s, avg %.1f-%.1f, "
                 "recent %s"
                 % (h.meetings, h.home_wins, h.draws, h.away_wins,
                    pred.home_name[:12], h.home_goals_avg, h.away_goals_avg,
                    "".join(h.last_results[:6])))
    else:
        L.append("  Head to head   no previous meetings in the dataset")
    L.append("  Form (last %d)  %s %s %.2f ppg, %.1f-%.1f  |  %s %s %.2f ppg, %.1f-%.1f"
             % (f.form_home.matches, pred.home_name[:10],
                "".join(f.form_home.results), f.form_home.points_per_game,
                f.form_home.goals_for, f.form_home.goals_against,
                pred.away_name[:10], "".join(f.form_away.results),
                f.form_away.points_per_game, f.form_away.goals_for,
                f.form_away.goals_against))
    L.append("  Confidence     %s  (model fitted on %d matches, rho=%+.3f)"
             % (pred.confidence(), pred.ratings.n_matches, pred.ratings.rho))

    if verbose:
        L.append("")
        L.append("LOG-RATE DECOMPOSITION")
        for key in sorted(pred.components):
            L.append("  %-22s %+.4f" % (key, pred.components[key]))

    L.append("")
    L.append("Model: Dixon-Coles + ridge, time decay %.4f/day, max %d goals/side"
             % (pred.ratings.config.time_decay, pred.ratings.config.max_goals))
    return "\n".join(L)


def render_compact(pred: Prediction) -> str:
    m = pred.markets
    t = pred.tempo
    return ("%-18s vs %-18s  %s  %s/%s/%s  xG %.2f-%.2f  most likely %d-%d  "
            "poss %.0f-%.0f  shots %.1f-%.1f"
            % (pred.home_name[:18], pred.away_name[:18],
               pred.kickoff.strftime("%d %b %H:%M"),
               _fmt_pct(m["home"]).strip(), _fmt_pct(m["draw"]).strip(),
               _fmt_pct(m["away"]).strip(),
               m["exp_goals_home"], m["exp_goals_away"],
               pred.most_likely_score[0], pred.most_likely_score[1],
               t.possession_home, t.possession_away,
               t.shots_home, t.shots_away))
