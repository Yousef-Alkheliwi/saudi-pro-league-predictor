"""Managers, from Wikipedia's Saudi Pro League season pages.

No football API reachable for free publishes coaches for this league - ESPN's
coaches endpoint is empty for every season - but each season article carries
a "Personnel and kits" table (each club's manager) and a "Managerial changes"
table (outgoing, incoming, date of appointment). Together they give every
club's manager on any date, which is what "manager preference" needs: a new
manager's habits are not his predecessor's.

Read through the Wikipedia REST API (content CC BY-SA 4.0), cached for a day.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from ..config import CACHE_DIR

API = "https://en.wikipedia.org/api/rest_v1/page/html/%s"
USER_AGENT = "saudi-pro-league-predictor/0.1 (https://github.com/Yousef-Alkheliwi/saudi-pro-league-predictor)"


def season_title(season: int) -> str:
    """2025 -> '2025–26_Saudi_Pro_League' (en dash, as Wikipedia titles it)."""
    return "%d–%02d_Saudi_Pro_League" % (season, (season + 1) % 100)


class _Tables(HTMLParser):
    """Every table on the page as a grid of cell text, rowspans expanded, each
    tagged with the section heading it sits under."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[Tuple[str, List[List[str]]]] = []
        self._heading = ""
        self._in_heading = False
        self._heading_text = ""
        self._stack: List[dict] = []
        self._cell: Optional[str] = None
        self._muted = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("h2", "h3", "h4"):
            self._in_heading, self._heading_text = True, ""
        elif tag in ("sup", "style", "script"):
            self._muted += 1
        elif tag == "table":
            self._stack.append({"heading": self._heading, "rows": [], "pending": {}})
        elif tag == "tr" and self._stack:
            self._stack[-1]["rows"].append([])
        elif tag in ("td", "th") and self._stack:
            self._cell = ""
            try:
                self._span = max(1, int(a.get("rowspan") or 1))
            except ValueError:
                self._span = 1
        elif tag == "br" and self._cell is not None:
            self._cell += " "

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "h4") and self._in_heading:
            self._in_heading = False
            self._heading = self._heading_text.strip()
        elif tag in ("sup", "style", "script") and self._muted:
            self._muted -= 1
        elif tag in ("td", "th") and self._stack and self._cell is not None:
            t = self._stack[-1]
            if t["rows"]:
                row = t["rows"][-1]
                self._fill_pending(t, row)
                text = re.sub(r"\s+", " ", self._cell).strip()
                col = len(row)
                row.append(text)
                if self._span > 1:
                    t["pending"][col] = (text, self._span - 1)
                self._fill_pending(t, row)
            self._cell = None
        elif tag == "tr" and self._stack and self._stack[-1]["rows"]:
            t = self._stack[-1]
            self._fill_pending(t, t["rows"][-1], final=True)
        elif tag == "table" and self._stack:
            t = self._stack.pop()
            self.tables.append((t["heading"], t["rows"]))

    @staticmethod
    def _fill_pending(t, row, final=False):
        """Insert cells carried down from a rowspan above, at their column."""
        while True:
            col = len(row)
            carried = t["pending"].get(col)
            if carried is None:
                if final:
                    later = [c for c in t["pending"] if c > col]
                    if later:
                        row.append("")
                        continue
                return
            text, left = carried
            row.append(text)
            if left > 1:
                t["pending"][col] = (text, left - 1)
            else:
                del t["pending"][col]

    def handle_data(self, data):
        if self._muted:
            return
        if self._in_heading:
            self._heading_text += data
        if self._cell is not None:
            self._cell += data


def _date(text: str) -> Optional[datetime]:
    m = re.search(r"(\d{1,2}) ([A-Z][a-z]+) (\d{4})", text or "")
    if not m:
        return None
    try:
        return datetime.strptime(" ".join(m.groups()), "%d %B %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _person(text: str) -> str:
    return re.sub(r"\s*\((caretaker|interim)\)\s*$", "", text or "", flags=re.I).strip()


@dataclass
class Change:
    team: str
    outgoing: str
    incoming: str
    appointed: Optional[datetime]
    vacated: Optional[datetime]


def parse_season(html: str) -> Tuple[Dict[str, str], List[Change]]:
    """(club -> manager listed under Personnel, managerial changes)."""
    parser = _Tables()
    parser.feed(html)
    managers: Dict[str, str] = {}
    changes: List[Change] = []
    for heading, rows in parser.tables:
        if not rows:
            continue
        header = [c.lower() for c in rows[0]]
        if "personnel" in heading.lower() and "team" in header:
            ti = header.index("team")
            mi = next((i for i, c in enumerate(header)
                       if "manager" in c or "coach" in c), None)
            if mi is None:
                continue
            for r in rows[1:]:
                if len(r) > max(ti, mi) and r[ti] and r[mi]:
                    managers[r[ti]] = _person(r[mi])
        elif "managerial" in heading.lower() and "team" in header:
            def col(*names):
                return next((i for i, c in enumerate(header)
                             if any(n in c for n in names)), None)
            ti, oi, ii = col("team"), col("outgoing"), col("incoming")
            ai, vi = col("appointment"), col("vacancy")
            if None in (ti, oi, ii):
                continue
            for r in rows[1:]:
                if len(r) <= max(ti, oi, ii) or not r[ti]:
                    continue
                changes.append(Change(
                    team=r[ti], outgoing=_person(r[oi]), incoming=_person(r[ii]),
                    appointed=_date(r[ai]) if ai is not None and ai < len(r) else None,
                    vacated=_date(r[vi]) if vi is not None and vi < len(r) else None))
    return managers, changes


class Wikipedia:
    def __init__(self, cache_dir: Optional[Path] = None, offline: bool = False,
                 ttl: float = 86400.0, timeout: float = 25.0) -> None:
        self.cache_dir = Path(cache_dir or (CACHE_DIR / "wikipedia"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline, self.ttl, self.timeout = offline, ttl, timeout
        self.calls_made = 0

    def page(self, title: str) -> Optional[str]:
        fp = self.cache_dir / ("%s.json" % hashlib.sha1(title.encode()).hexdigest()[:16])
        if fp.exists():
            try:
                blob = json.loads(fp.read_text(encoding="utf-8"))
                if self.offline or time.time() - blob["fetched_at"] < self.ttl:
                    return blob["html"]
            except (ValueError, KeyError, OSError):
                pass
        if self.offline:
            return None
        try:
            resp = requests.get(API % title, timeout=self.timeout,
                                headers={"User-Agent": USER_AGENT})
            self.calls_made += 1
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException:
            return None
        fp.write_text(json.dumps({"fetched_at": time.time(), "html": resp.text}),
                      encoding="utf-8")
        return resp.text


#: the start of a club's first known spell, when nothing earlier is recorded
_BEFORE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def build_spells(personnel: Dict[str, str], changes: List[Change],
                 resolve) -> Dict[int, List[Tuple[datetime, str]]]:
    """Each club's managers as (start, name), oldest first.

    `resolve` maps a Wikipedia club name to a team id (or raises LookupError).
    The manager before a club's first recorded change is that change's
    outgoing manager; a club with no change at all keeps its listed manager.
    """
    by_team: Dict[int, List[Change]] = {}
    for c in changes:
        try:
            tid = resolve(c.team)
        except LookupError:
            continue
        by_team.setdefault(tid, []).append(c)
    spells: Dict[int, List[Tuple[datetime, str]]] = {}
    for tid, cs in by_team.items():
        cs = [c for c in cs if (c.appointed or c.vacated) and c.incoming]
        cs.sort(key=lambda c: c.appointed or c.vacated)
        if not cs:
            continue
        timeline = [(_BEFORE, cs[0].outgoing)] if cs[0].outgoing else []
        for c in cs:
            timeline.append((c.appointed or c.vacated, c.incoming))
        spells[tid] = timeline
    for name, manager in personnel.items():
        try:
            tid = resolve(name)
        except LookupError:
            continue
        if tid not in spells and manager:
            spells[tid] = [(_BEFORE, manager)]
    return spells


def fetch_managers(client: "Wikipedia", ds, seasons, log=print) -> None:
    from ..data import ManagerSpell
    personnel: Dict[str, str] = {}
    changes: List[Change] = []
    read = 0
    for season in sorted(set(seasons)):
        html = client.page(season_title(season))
        if not html:
            continue
        managers, cs = parse_season(html)
        personnel.update(managers)          # later seasons overwrite earlier
        changes.extend(cs)
        read += 1
    spells = build_spells(personnel, changes, lambda n: ds.resolve_team(n)[0])
    ds.managers = [ManagerSpell(tid, name, start.isoformat())
                   for tid, timeline in spells.items() for start, name in timeline]
    log("managers: %d spells across %d clubs, from %d season pages"
        % (len(ds.managers), len(spells), read))
