"""Command line interface.

    python -m spl.cli fetch                      pull live data into ./data
    python -m spl.cli teams                      list clubs in the dataset
    python -m spl.cli predict --home X --away Y  predict one match
    python -m spl.cli next --days 10             predict the upcoming fixtures
    python -m spl.cli ratings                    show fitted attack/defence ratings
    python -m spl.cli backtest                   walk-forward evaluation
    python -m spl.cli status                     dataset + API quota summary
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from .config import DATASET_PATH, MODEL, PROVIDER
from .data import Dataset, fetch_dataset
from .predict import Predictor, render, render_compact
from .providers import ApiFootball, ApiFootballError, MissingCredentials


def _load(args) -> Dataset:
    ds = Dataset.load(args.dataset)
    if not ds.matches:
        raise SystemExit("dataset has no matches - run `fetch` again")
    return ds


def _parse_when(text):
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise SystemExit("could not parse --date %r (use YYYY-MM-DD or 'YYYY-MM-DD HH:MM')"
                     % text)


# --------------------------------------------------------------------- commands
def cmd_fetch(args) -> int:
    if args.source == "espn":
        return _fetch_espn(args)
    client = ApiFootball(offline=args.offline)
    seasons = args.seasons or list(MODEL.seasons)
    try:
        ds = fetch_dataset(client, seasons=seasons, with_stats=not args.no_stats,
                           stats_limit=args.stats_limit,
                           with_players=not args.no_players,
                           with_schedule=not args.no_schedule,
                           refresh_live=not args.offline)
    except MissingCredentials as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ApiFootballError as exc:
        print("upstream error: %s" % exc, file=sys.stderr)
        return 1
    if not ds.matches:
        print("\nfetched no fixtures at all - refusing to overwrite %s with an empty "
              "dataset.\nCheck the API key, the plan's league coverage and the quota "
              "with `status`." % (args.dataset or DATASET_PATH), file=sys.stderr)
        return 1
    path = ds.save(args.dataset)
    print("\nsaved %d matches (%d finished), %d teams, %d box-score rows, "
          "%d injuries, %d players -> %s"
          % (len(ds.matches), len(ds.played_matches), len(ds.teams), len(ds.stats),
             len(ds.injuries), len(ds.players), path))
    print("upstream calls: %d   cache hits: %d   daily quota left: %s"
          % (client.calls_made, client.cache_hits,
             client.daily_remaining if client.daily_remaining is not None else "?"))
    return 0


def _fetch_espn(args) -> int:
    """The free provider: no key, no quota, and the current season included."""
    from .providers.espn import Espn, EspnError, fetch_dataset as fetch_espn
    client = Espn(offline=args.offline)
    try:
        ds = fetch_espn(client, seasons=args.seasons or None)
    except EspnError as exc:
        print("ESPN error: %s" % exc, file=sys.stderr)
        return 1
    if not ds.matches:
        print("\nfetched no fixtures - refusing to overwrite %s with an empty dataset"
              % (args.dataset or DATASET_PATH), file=sys.stderr)
        return 1
    path = ds.save(args.dataset)
    print("\nsaved %d matches (%d finished), %d teams, %d box-score rows, "
          "%d injuries -> %s"
          % (len(ds.matches), len(ds.played_matches), len(ds.teams), len(ds.stats),
             len(ds.injuries), path))
    return 0


def cmd_teams(args) -> int:
    ds = _load(args)
    played = {}
    for m in ds.played_matches:
        played[m.home_id] = played.get(m.home_id, 0) + 1
        played[m.away_id] = played.get(m.away_id, 0) + 1
    print("%-6s %-32s %s" % ("id", "team", "matches in dataset"))
    for tid, name in sorted(ds.teams.items(), key=lambda kv: kv[1]):
        print("%-6d %-32s %d" % (tid, name, played.get(tid, 0)))
    return 0


def cmd_predict(args) -> int:
    ds = _load(args)
    predictor = Predictor(ds, cfg=MODEL)
    try:
        pred = predictor.predict_by_name(
            args.home, args.away, kickoff=_parse_when(args.date),
            neutral_venue=args.neutral, apply_injuries=not args.ignore_injuries)
    except LookupError as exc:
        print("team lookup failed: %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(_as_dict(pred), indent=2))
    else:
        print(render(pred, verbose=args.verbose))
    return 0


def cmd_next(args) -> int:
    ds = _load(args)
    predictor = Predictor(ds, cfg=MODEL)
    fixtures = predictor.upcoming(days=args.days, limit=args.limit)
    if not fixtures:
        print("no scheduled fixtures in the next %d days in this dataset" % args.days)
        return 0
    preds = [predictor.predict_fixture(m,
                                       apply_injuries=not args.ignore_injuries)
             for m in fixtures]
    if args.json:
        print(json.dumps([_as_dict(p) for p in preds], indent=2))
        return 0
    if args.full:
        for p in preds:
            print(render(p, verbose=args.verbose))
            print()
        return 0
    print("Next %d fixtures  (home%%/draw%%/away%%)" % len(preds))
    print("-" * 118)
    for p in preds:
        print(render_compact(p))
    return 0


def cmd_ratings(args) -> int:
    ds = _load(args)
    predictor = Predictor(ds, cfg=MODEL)
    r = predictor.ratings
    print("fitted on %d matches  |  base %.3f  home advantage %+.3f (x%.2f)  "
          "rho %+.3f" % (r.n_matches, r.base, r.home_adv, 2.718281828 ** r.home_adv,
                         r.rho))
    print("rest-days coefficient %+.4f per 3 days  |  congestion %+.4f per extra "
          "match in 14 days" % (r.b_rest, r.b_cong))
    print()
    print("higher is better on both columns: attack raises the goals a club scores,")
    print("defence lowers the goals it concedes; net is their sum.")
    print()
    print("%-28s %8s %9s %8s" % ("team", "attack", "defence", "net"))
    print("-" * 56)
    for tid, att, dfn, net in r.table():
        print("%-28s %+8.3f %+9.3f %+8.3f" % (ds.teams.get(tid, tid)[:28], att, dfn, net))
    if predictor.aux.models:
        print()
        print("auxiliary models: " + ", ".join(
            "%s (n=%d, rmse %.2f)" % (k, m.n_rows, m.rmse)
            for k, m in sorted(predictor.aux.models.items())))
    elif predictor.aux.fallback_note:
        print("\nauxiliary models: %s" % predictor.aux.fallback_note)
    return 0


def cmd_backtest(args) -> int:
    from .backtest import backtest
    ds = _load(args)
    try:
        res = backtest(ds, min_train_matches=args.min_train,
                       refit_every=args.refit_every,
                       include_tempo=not args.no_tempo,
                       log=print if args.verbose else None)
    except ValueError as exc:
        print("cannot backtest: %s" % exc, file=sys.stderr)
        return 1
    print(res.render())
    return 0


def cmd_export(args) -> int:
    from .export import export
    ds = _load(args)
    export(ds, args.out)
    return 0


def cmd_status(args) -> int:
    try:
        ds = Dataset.load(args.dataset)
    except FileNotFoundError as exc:
        print(exc)
        ds = None
    if ds:
        print("dataset      %s" % args.dataset or DATASET_PATH)
        print("snapshot     %s" % ds.fetched_at)
        print("source       %s" % ds.source)
        print("league id    %s" % ds.league_id)
        print("matches      %d total, %d finished, %d scheduled"
              % (len(ds.matches), len(ds.played_matches), len(ds.upcoming_matches)))
        seasons = sorted({m.season for m in ds.matches})
        print("seasons      %s" % ", ".join(str(s) for s in seasons))
        print("teams        %d" % len(ds.teams))
        print("other comps  %d fixtures (%s)"
              % (len(ds.other_matches),
                 ", ".join(sorted({m.league_name for m in ds.other_matches
                                   if m.league_name})[:5]) or "none"))
        print("box scores   %d team-match rows" % len(ds.stats))
        print("injuries     %d records over %d clubs"
              % (len(ds.injuries), len(ds.injuries_by_team())))
        print("players      %d records" % len(ds.players))
    print()
    print("api key      %s" % ("configured" if PROVIDER.api_key else "MISSING"))
    print("api mode     %s (%s)" % (PROVIDER.mode, PROVIDER.base_url))
    if PROVIDER.api_key and not args.offline:
        try:
            st = ApiFootball().status()
            acct = st.get("subscription") or {}
            reqs = st.get("requests") or {}
            print("plan         %s, expires %s" % (acct.get("plan"), acct.get("end")))
            print("quota        %s / %s requests used today"
                  % (reqs.get("current"), reqs.get("limit_day")))
        except ApiFootballError as exc:
            print("status check failed: %s" % exc)
    return 0


def _as_dict(pred) -> dict:
    f = pred.features
    t = pred.tempo
    return {
        "fixture_id": pred.fixture_id,
        "kickoff_utc": pred.kickoff.isoformat(),
        "home": pred.home_name,
        "away": pred.away_name,
        "venue": pred.venue,
        "probabilities": {k: round(pred.markets[k], 4)
                          for k in ("home", "draw", "away", "btts",
                                    "over_2.5", "under_2.5", "over_1.5",
                                    "over_3.5", "home_cs", "away_cs")},
        "expected_goals": {"home": round(pred.markets["exp_goals_home"], 3),
                           "away": round(pred.markets["exp_goals_away"], 3),
                           "total": round(pred.markets["exp_total"], 3),
                           "supremacy": round(pred.markets["exp_supremacy"], 3)},
        "expected_conceded": {"home": round(pred.markets["exp_goals_away"], 3),
                              "away": round(pred.markets["exp_goals_home"], 3)},
        "top_scorelines": [{"score": "%d-%d" % (i, j), "p": round(p, 4)}
                           for i, j, p in pred.scorelines],
        "shots": {"home": round(t.shots_home, 2), "away": round(t.shots_away, 2),
                  "on_target_home": round(t.sot_home, 2),
                  "on_target_away": round(t.sot_away, 2),
                  "corners_home": round(t.corners_home, 2) if t.corners_home else None,
                  "corners_away": round(t.corners_away, 2) if t.corners_away else None,
                  "source": t.source},
        "possession": {"home": round(t.possession_home, 1),
                       "away": round(t.possession_away, 1)},
        "inputs": {
            "rest_days": {"home": f.rest_home, "away": f.rest_away},
            "matches_last_14d": {"home": f.congestion_home, "away": f.congestion_away},
            "availability": {
                "home": {"overall": round(f.availability_home.overall_available, 3),
                         "out": f.availability_home.out,
                         "doubtful": f.availability_home.doubtful},
                "away": {"overall": round(f.availability_away.overall_available, 3),
                         "out": f.availability_away.out,
                         "doubtful": f.availability_away.doubtful}},
            "head_to_head": {"meetings": f.h2h.meetings,
                             "home_wins": f.h2h.home_wins, "draws": f.h2h.draws,
                             "away_wins": f.h2h.away_wins,
                             "avg_goals_home": round(f.h2h.home_goals_avg, 2),
                             "avg_goals_away": round(f.h2h.away_goals_avg, 2)},
            "form": {"home": {"ppg": round(f.form_home.points_per_game, 2),
                              "results": f.form_home.results},
                     "away": {"ppg": round(f.form_away.points_per_game, 2),
                              "results": f.form_away.results}},
            "home_advantage_log": round(pred.ratings.home_adv, 4),
        },
        "log_rate_components": {k: round(v, 4) for k, v in pred.components.items()},
        "confidence": pred.confidence(),
    }


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="spl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="path to the dataset json")
    ap.add_argument("--offline", action="store_true",
                    help="never call the network; use the on-disk cache only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="pull live match data")
    p.add_argument("--source", choices=("espn", "api-football"), default="espn",
                   help="espn: free, no key, includes the current season (default). "
                        "api-football: needs a key; adds injuries and squads on a "
                        "paid plan")
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--no-stats", action="store_true", help="skip per-match box scores")
    p.add_argument("--stats-limit", type=int, default=30,
                   help="how many recent matches to pull box scores for (one "
                        "upstream call each; these are fetched last, so a quota "
                        "run-out costs only shot/possession precision)")
    p.add_argument("--no-players", action="store_true",
                   help="skip player minutes (injury weighting falls back to flat)")
    p.add_argument("--no-schedule", action="store_true",
                   help="skip other competitions (rest days then ignore AFC/cup games)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("teams", help="list clubs")
    p.set_defaults(func=cmd_teams)

    p = sub.add_parser("predict", help="predict a single match")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--date", default=None, help="kickoff, YYYY-MM-DD[ HH:MM] UTC")
    p.add_argument("--neutral", action="store_true", help="drop the home advantage")
    p.add_argument("--ignore-injuries", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("next", help="predict upcoming fixtures")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--full", action="store_true", help="full report per fixture")
    p.add_argument("--ignore-injuries", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("ratings", help="show fitted team ratings")
    p.set_defaults(func=cmd_ratings)

    p = sub.add_parser("backtest", help="walk-forward evaluation")
    p.add_argument("--min-train", type=int, default=120)
    p.add_argument("--refit-every", type=int, default=10)
    p.add_argument("--no-tempo", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("export", help="dump every pairing to JSON for the web UI")
    p.add_argument("--out", default="ui/data.json")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("status", help="dataset and API quota summary")
    p.set_defaults(func=cmd_status)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
