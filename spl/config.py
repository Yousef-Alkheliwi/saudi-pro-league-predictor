"""Configuration: credentials, paths, and model hyper-parameters."""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent

#: A football season is labelled by the calendar year it opens in, and the Saudi
#: season starts in August, so before August the current season is last year's.
_NOW = _dt.datetime.now(_dt.timezone.utc)
_THIS_SEASON = _NOW.year if _NOW.month >= 8 else _NOW.year - 1
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DATASET_PATH = DATA_DIR / "spl_dataset.json"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


@dataclass
class ProviderConfig:
    """Where live data comes from.

    Two shapes of the same upstream API are supported:
      * direct  -> https://v3.football.api-sports.io  with header x-apisports-key
      * rapidapi-> https://api-football-v1.p.rapidapi.com/v3 with x-rapidapi-key
    """

    api_key: Optional[str] = field(default_factory=lambda: (
        os.environ.get("API_FOOTBALL_KEY")
        or os.environ.get("APISPORTS_KEY")
        or os.environ.get("RAPIDAPI_KEY")
    ))
    mode: str = field(default_factory=lambda: os.environ.get("API_FOOTBALL_MODE", "direct"))
    #: hard ceiling on upstream calls per process run; the free tier allows 100/day
    #: default sits just under the free tier's 100 calls/day; raise it via
    #: API_FOOTBALL_BUDGET on a paid plan
    request_budget: int = field(default_factory=lambda: int(os.environ.get("API_FOOTBALL_BUDGET", "95")))
    #: Minimum seconds between upstream calls. The free tier allows 10 per
    #: minute (x-ratelimit-limit: 10), so 6.5s keeps us at ~9/min and never
    #: trips the limiter. Paid plans can lower this via API_FOOTBALL_INTERVAL.
    min_interval: float = field(default_factory=lambda: float(
        os.environ.get("API_FOOTBALL_INTERVAL", "6.5")))
    timeout: float = 25.0
    max_retries: int = 4

    @property
    def base_url(self) -> str:
        if self.mode == "rapidapi":
            return "https://api-football-v1.p.rapidapi.com/v3"
        return "https://v3.football.api-sports.io"

    @property
    def headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        if self.mode == "rapidapi":
            return {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            }
        return {"x-apisports-key": self.api_key}


@dataclass
class ModelConfig:
    """Hyper-parameters for the Dixon-Coles goal model and the auxiliary models."""

    #: exponential time-decay rate per day. 0.0022/day ~= half-life of 315 days,
    #: i.e. a match from last season counts roughly half as much as a recent one.
    time_decay: float = 0.0022
    #: ridge strength pulling attack/defence ratings toward the league average.
    #: Keeps newly promoted teams with few matches from being wildly rated.
    ridge: float = 0.35
    #: Max goals per side in the scoreline matrix. The distribution is truncated
    #: here and renormalised, which shifts probability mass off the tail and
    #: biases expected goals downward - worst for the highest-scoring fixtures,
    #: exactly where the model is most confident. At 10 that cost 0.024 goals at
    #: a rate of 4.1 and 0.092 at 5.0; at 15 it is under 0.001 for any rate this
    #: league produces. 18 keeps that guarantee well past any rate this model
    #: emits in practice, for a matrix that is still trivially small.
    max_goals: int = 18
    #: cap on rest days before the effect is assumed to plateau
    rest_days_cap: int = 10
    #: matches in the trailing window used for the congestion covariate
    congestion_window_days: int = 14

    # --- injury / availability prior -------------------------------------------------
    # Historical per-fixture injury lists are not retrievable cheaply from the API,
    # so availability is applied as an explicit, documented prior rather than a
    # coefficient fitted on history. `k` is the elasticity of goal rate with respect
    # to the share of importance-weighted minutes that is unavailable.
    injury_attack_elasticity: float = 0.55
    injury_defence_elasticity: float = 0.45
    injury_possession_points: float = 6.0
    injury_shots_elasticity: float = 0.40

    #: Prior standard deviation on the rest-days and congestion coefficients.
    #: One league season gives a standard error near 0.07 on these, so an
    #: unregularised fit happily reports a wrong-signed rest effect. The prior
    #: shrinks a weakly-identified estimate toward zero instead.
    schedule_prior_sd: float = 0.05

    #: ridge strength for the shots / possession models
    aux_ridge: float = 2.0
    #: Seasons to try, newest last. API-Football labels a season by its opening
    #: year, so 2025 is the 2025-26 campaign. A five-year window is requested and
    #: the subscription filters it: free tiers serve only older seasons, so asking
    #: wide is what lets the same code work on any plan.
    seasons: tuple = field(default_factory=lambda: tuple(
        range(_THIS_SEASON - 4, _THIS_SEASON + 1)))


PROVIDER = ProviderConfig()
MODEL = ModelConfig()

SAUDI_PRO_LEAGUE_NAME = "Pro League"
SAUDI_COUNTRY = "Saudi-Arabia"
#: API-Football's league id for the Saudi Pro League. Used as a hint only; the
#: client resolves the id from the API and falls back to this value.
SAUDI_LEAGUE_ID_HINT = 307
