"""Synthetic league generator used by the tests.

Matches are simulated from known attack/defence/home-advantage/rest parameters,
so the tests can assert that the fitting code recovers what generated the data.
This is clearly-labelled simulated data - it is never mixed into ./data, and the
predictor never uses it for real forecasts.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from spl.data import Dataset, Injury, Match, Player, TeamStats

TRUE = {
    "base": math.log(1.30),
    "home_adv": 0.26,
    "b_rest": 0.055,
    "poss_scale": 26.0,     # possession points per unit of attack rating
    "poss_home": 2.0,
    "shots_base": math.log(12.0),
    "shots_home": 0.08,
}


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's method; lam here is always small (< 6)
    l = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1
        if k > 15:
            return k


def make_dataset(n_teams: int = 16, seasons: Tuple[int, ...] = (2024, 2025),
                 seed: int = 7, with_stats: bool = True,
                 with_injuries: bool = True) -> Tuple[Dataset, Dict[str, object]]:
    rng = random.Random(seed)
    teams = {1000 + i: "Team %02d" % i for i in range(n_teams)}
    attack = {t: rng.gauss(0.0, 0.30) for t in teams}
    defence = {t: rng.gauss(0.0, 0.24) for t in teams}
    # centre them, as the model does
    am = sum(attack.values()) / n_teams
    dm = sum(defence.values()) / n_teams
    attack = {t: v - am for t, v in attack.items()}
    defence = {t: v - dm for t, v in defence.items()}

    ds = Dataset(league_id=999, fetched_at=datetime.now(timezone.utc).isoformat(),
                 teams=dict(teams))
    fixture_id = 1
    last_played: Dict[int, datetime] = {}

    # Build the whole calendar first, then simulate in date order, so that the
    # rest days the generator uses are exactly the ones recomputed from the
    # finished dataset. Getting this wrong introduces measurement error in the
    # covariate and attenuates the fitted coefficient.
    calendar: List[Tuple[datetime, int, int]] = []
    for season in seasons:
        start = datetime(season, 8, 10, 18, 0, tzinfo=timezone.utc)
        ids = list(teams)
        arr = ids[:]
        half: List[List[Tuple[int, int]]] = []
        for _ in range(n_teams - 1):                       # circle method
            pairs = [(arr[i], arr[n_teams - 1 - i]) for i in range(n_teams // 2)]
            half.append(pairs)
            arr = [arr[0], arr[-1]] + arr[1:-1]
        rounds = []
        for r, pairs in enumerate(half):                   # alternate venues
            rounds.append([(h, a) if r % 2 == 0 else (a, h) for h, a in pairs])
        rounds += [[(a, h) for h, a in pairs] for pairs in rounds]

        for round_no, pairs in enumerate(rounds):
            for home, away in pairs:
                when = start + timedelta(days=7 * round_no + rng.choice([0, 1, 2, 3]),
                                         hours=rng.choice([0, 2, 3]))
                calendar.append((when, home, away))

    for when, home, away in sorted(calendar, key=lambda row: row[0]):
        rest_h = (when - last_played[home]).days if home in last_played else 7
        rest_a = (when - last_played[away]).days if away in last_played else 7
        rz_h = (min(rest_h, 10) - 5.0) / 3.0
        rz_a = (min(rest_a, 10) - 5.0) / 3.0

        lam = math.exp(TRUE["base"] + TRUE["home_adv"] + attack[home]
                       - defence[away] + TRUE["b_rest"] * rz_h)
        mu = math.exp(TRUE["base"] + attack[away] - defence[home]
                      + TRUE["b_rest"] * rz_a)
        hg, ag = _poisson(rng, lam), _poisson(rng, mu)

        ds.matches.append(Match(
            fixture_id=fixture_id, season=when.year, kickoff=when.isoformat(),
            home_id=home, away_id=away, home_name=teams[home],
            away_name=teams[away], status="FT", home_goals=hg, away_goals=ag,
            venue="Stadium %d" % home, round="Round",
        ))

        if with_stats:
            poss_h = (50.0 + TRUE["poss_home"]
                      + TRUE["poss_scale"] * (attack[home] - attack[away]) / 2.0)
            poss_h = min(max(poss_h + rng.gauss(0, 3.0), 25.0), 75.0)
            sh_h = math.exp(TRUE["shots_base"] + TRUE["shots_home"]
                            + 0.9 * attack[home] + 0.7 * defence[away]
                            + rng.gauss(0, 0.18))
            sh_a = math.exp(TRUE["shots_base"] + 0.9 * attack[away]
                            + 0.7 * defence[home] + rng.gauss(0, 0.18))
            ds.stats.append(TeamStats(fixture_id, home, shots=round(sh_h),
                                      shots_on_target=round(sh_h * 0.36),
                                      possession=round(poss_h, 1),
                                      corners=round(sh_h * 0.38)))
            ds.stats.append(TeamStats(fixture_id, away, shots=round(sh_a),
                                      shots_on_target=round(sh_a * 0.36),
                                      possession=round(100.0 - poss_h, 1),
                                      corners=round(sh_a * 0.38)))

        last_played[home] = when
        last_played[away] = when
        fixture_id += 1

    # a few scheduled fixtures after the last played one
    final = max(m.dt for m in ds.matches)
    ids = list(teams)
    for i in range(0, min(8, n_teams), 2):
        when = final + timedelta(days=4 + i)
        ds.matches.append(Match(
            fixture_id=fixture_id, season=seasons[-1], kickoff=when.isoformat(),
            home_id=ids[i], away_id=ids[i + 1], home_name=teams[ids[i]],
            away_name=teams[ids[i + 1]], status="NS", venue="Stadium %d" % ids[i],
            round="Round 31",
        ))
        fixture_id += 1

    # squads: 25 players, minutes concentrated in the first 11
    for t in teams:
        for p in range(25):
            minutes = max(0.0, 2400.0 - 95.0 * p + rng.gauss(0, 120))
            position = ("Goalkeeper" if p == 0 else "Defender" if p < 8
                        else "Midfielder" if p < 17 else "Attacker")
            ds.players.append(Player(player_id=t * 100 + p, team_id=t,
                                     name="P%d-%d" % (t, p), position=position,
                                     minutes=minutes, appearances=minutes / 90.0,
                                     goals=max(0.0, rng.gauss(3, 3)) if p > 16 else 0.0,
                                     assists=0.0, rating=6.5))

    if with_injuries:
        hurt_team = ids[0]
        for p in (1, 2, 18):     # a defender, a defender and a first-choice attacker
            ds.injuries.append(Injury(team_id=hurt_team, player_id=hurt_team * 100 + p,
                                      player_name="P%d-%d" % (hurt_team, p),
                                      type="Missing Fixture", reason="Injury"))
        ds.injuries.append(Injury(team_id=ids[1], player_id=ids[1] * 100 + 20,
                                  player_name="P%d-20" % ids[1],
                                  type="Questionable", reason="Knock"))

    truth = {"attack": attack, "defence": defence, "teams": teams, **TRUE}
    return ds, truth
