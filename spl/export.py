"""Export every fixture pairing as JSON for the web UI.

The UI does no modelling of its own. Every number it shows is computed here, by
the same code paths the CLI and the tests exercise, so the page can never drift
from the model. For 18 clubs that is 18x17 = 306 ordered pairings, which is small
enough to precompute in full and ship with the page.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import model as M
from .config import MODEL
from .data import Dataset
from .predict import Predictor, Prediction, _data_warnings

#: goals per side kept in the exported scoreline grid
GRID = 7


def _grid(pred: Prediction) -> List[List[float]]:
    """The scoreline heat-map, recomputed from the prediction's own rates."""
    mat = M.score_matrix(pred.lam_home, pred.lam_away, pred.ratings.rho,
                         pred.ratings.config.max_goals)
    return [[round(float(mat[i, j]), 5) for j in range(GRID)] for i in range(GRID)]


def _pairing(pred: Prediction) -> dict:
    m = pred.markets
    t = pred.tempo
    f = pred.features
    return {
        "home": pred.home_id,
        "away": pred.away_id,
        "p": {"home": round(m["home"], 4), "draw": round(m["draw"], 4),
              "away": round(m["away"], 4)},
        "xg": {"home": round(m["exp_goals_home"], 3),
               "away": round(m["exp_goals_away"], 3),
               "total": round(m["exp_total"], 3),
               "supremacy": round(m["exp_supremacy"], 3)},
        "lam": [round(pred.lam_home, 4), round(pred.lam_away, 4)],
        "markets": {k: round(m[k], 4) for k in
                    ("btts", "over_1.5", "over_2.5", "over_3.5", "under_2.5",
                     "home_cs", "away_cs")},
        "scores": [{"s": "%d-%d" % (i, j), "p": round(p, 4)}
                   for i, j, p in pred.scorelines[:6]],
        "grid": _grid(pred),
        "tempo": {"poss": [round(t.possession_home, 1), round(t.possession_away, 1)],
                  "shots": [round(t.shots_home, 1), round(t.shots_away, 1)],
                  "sot": [round(t.sot_home, 1), round(t.sot_away, 1)],
                  "corners": ([round(t.corners_home, 1), round(t.corners_away, 1)]
                              if t.corners_home is not None else None)},
        "ctx": {
            "rest": [f.rest_home, f.rest_away],
            "cong": [f.congestion_home, f.congestion_away],
            "avail": [round(f.availability_home.overall_available, 3),
                      round(f.availability_away.overall_available, 3)],
            "out": [f.availability_home.out[:6], f.availability_away.out[:6]],
            "h2h": {"n": f.h2h.meetings, "w": f.h2h.home_wins, "d": f.h2h.draws,
                    "l": f.h2h.away_wins,
                    "gf": round(f.h2h.home_goals_avg, 2),
                    "ga": round(f.h2h.away_goals_avg, 2),
                    "recent": f.h2h.last_results[:6]},
            "form": [{"ppg": round(f.form_home.points_per_game, 2),
                      "r": f.form_home.results,
                      "gf": round(f.form_home.goals_for, 2),
                      "ga": round(f.form_home.goals_against, 2)},
                     {"ppg": round(f.form_away.points_per_game, 2),
                      "r": f.form_away.results,
                      "gf": round(f.form_away.goals_for, 2),
                      "ga": round(f.form_away.goals_against, 2)}],
        },
        "conf": pred.confidence(),
    }


def build_payload(ds: Dataset, predictor: Optional[Predictor] = None,
                  log=print) -> dict:
    predictor = predictor or Predictor(ds)
    r = predictor.ratings

    # clubs that actually appear in the fitted ratings, newest season first
    seasons = sorted({m.season for m in ds.played_matches})
    last_season = seasons[-1] if seasons else None
    recent = {m.home_id for m in ds.played_matches if m.season == last_season}
    recent |= {m.away_id for m in ds.played_matches if m.season == last_season}
    club_ids = sorted([t for t in r.teams if t in recent],
                      key=lambda t: ds.teams.get(t, str(t)))
    if not club_ids:
        club_ids = sorted(r.teams, key=lambda t: ds.teams.get(t, str(t)))

    played = ds.played_matches
    record: Dict[int, dict] = {}
    for tid in club_ids:
        w = d = l = gf = ga = 0
        for m in played:
            if m.season != last_season or tid not in (m.home_id, m.away_id):
                continue
            a, b = ((m.home_goals, m.away_goals) if m.home_id == tid
                    else (m.away_goals, m.home_goals))
            gf += a; ga += b
            w += a > b; d += a == b; l += a < b
        record[tid] = {"w": w, "d": d, "l": l, "gf": gf, "ga": ga,
                       "pts": 3 * w + d, "played": w + d + l}

    aux = predictor.aux
    clubs = []
    for tid in club_ids:
        att, dfn = r.strength(tid)
        clubs.append({
            "id": tid, "name": ds.teams.get(tid, str(tid)),
            "logo": ds.logos.get(tid),
            "attack": round(att, 4), "defence": round(dfn, 4),
            "net": round(att + dfn, 4),
            "record": record.get(tid, {}),
            "poss_pull": (round(aux.models["possession"].offence.get(tid, 0.0), 3)
                          if aux.has("possession") else None),
            "shot_gen": (round(aux.models["shots"].offence.get(tid, 0.0), 3)
                         if aux.has("shots") else None),
        })

    log("computing %d pairings..." % (len(club_ids) * (len(club_ids) - 1)))
    pairings = []
    for home in club_ids:
        for away in club_ids:
            if home == away:
                continue
            pairings.append(_pairing(predictor.predict(home, away)))

    # Real scheduled fixtures, so the page can open a club's actual next match
    # rather than making the reader guess an opponent.
    club_set = set(club_ids)
    fixtures = []
    for m in ds.upcoming_matches:
        if m.home_id in club_set and m.away_id in club_set:
            fixtures.append({"home": m.home_id, "away": m.away_id,
                             "kickoff": m.dt.isoformat(),
                             "label": m.dt.strftime("%a %d %b, %H:%M UTC"),
                             "day": m.dt.strftime("%d %b"),
                             "venue": m.venue})
    fixtures.sort(key=lambda f: f["kickoff"])
    log("%d scheduled fixtures exported" % len(fixtures))

    sample = predictor.predict(club_ids[0], club_ids[1])
    newest = played[-1] if played else None
    return {
        "meta": {
            "league": "Saudi Pro League",
            "league_id": ds.league_id,
            "snapshot": ds.fetched_at,
            "seasons": seasons,
            "newest_match": newest.kickoff if newest else None,
            "newest_match_label": (newest.dt.strftime("%d %b %Y") if newest else None),
            "n_matches": len(played),
            "n_clubs": len(club_ids),
            "box_score_rows": len(ds.stats),
            "injuries_available": bool(ds.injuries),
            "plan_limited": ds.plan_limited,
            "newest_available_season": ds.newest_available_season,
            "warnings": _data_warnings(sample),
            "source": ds.source,
        },
        "model": {
            "base": round(r.base, 4),
            "home_adv": round(r.home_adv, 4),
            "home_adv_mult": round(float(np.exp(r.home_adv)), 3),
            "rho": round(r.rho, 4),
            "b_rest": round(r.b_rest, 4),
            "b_cong": round(r.b_cong, 4),
            "n_matches": r.n_matches,
            "time_decay": r.config.time_decay,
            "half_life_days": round(0.6931471805599453 / r.config.time_decay, 0),
            "ridge": r.config.ridge,
            "aux": ({k: {"n": v.n_rows, "rmse": round(v.rmse, 3),
                         "home": round(v.home, 4), "league_mean": round(v.league_mean, 2)}
                     for k, v in sorted(aux.models.items())}
                    if aux.models else {}),
            "aux_note": aux.fallback_note,
            "grid": GRID,
        },
        "clubs": clubs,
        "fixtures": fixtures,
        "pairings": pairings,
    }


def export(ds: Dataset, out: Path, predictor: Optional[Predictor] = None,
           log=print) -> Path:
    payload = build_payload(ds, predictor=predictor, log=log)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log("wrote %s (%.0f KB, %d pairings, %d clubs)"
        % (out, out.stat().st_size / 1024.0, len(payload["pairings"]),
           len(payload["clubs"])))
    return out
