"""Tests for the JSON export and the two page builds.

The page does no modelling of its own - it renders precomputed numbers - so these
tests check that the export carries everything the page reads, that the numbers in
it agree with the model, and that both builds are structurally sound.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spl.export import build_payload
from spl.predict import Predictor
from tests.synthetic import make_dataset

ROOT = Path(__file__).resolve().parent.parent
DS, _ = make_dataset(seed=5)
PREDICTOR = Predictor(DS)
PAYLOAD = build_payload(DS, predictor=PREDICTOR, log=lambda *a: None)


class TestExportPayload(unittest.TestCase):
    def test_top_level_shape(self):
        for key in ("meta", "model", "clubs", "pairings"):
            self.assertIn(key, PAYLOAD)

    def test_every_ordered_pairing_is_present(self):
        n = len(PAYLOAD["clubs"])
        self.assertEqual(len(PAYLOAD["pairings"]), n * (n - 1))
        seen = {(p["home"], p["away"]) for p in PAYLOAD["pairings"]}
        self.assertEqual(len(seen), n * (n - 1))
        for p in PAYLOAD["pairings"]:
            self.assertNotEqual(p["home"], p["away"])

    def test_probabilities_sum_to_one(self):
        for p in PAYLOAD["pairings"]:
            total = p["p"]["home"] + p["p"]["draw"] + p["p"]["away"]
            self.assertAlmostEqual(total, 1.0, delta=0.002)

    def test_possession_sums_to_100(self):
        for p in PAYLOAD["pairings"]:
            self.assertAlmostEqual(sum(p["tempo"]["poss"]), 100.0, delta=0.2)

    def test_exported_numbers_match_the_model(self):
        """The page must not be able to show anything the model would not."""
        for p in PAYLOAD["pairings"][:12]:
            live = PREDICTOR.predict(p["home"], p["away"])
            self.assertAlmostEqual(p["p"]["home"], live.markets["home"], places=3)
            self.assertAlmostEqual(p["xg"]["home"], live.markets["exp_goals_home"],
                                   places=2)
            self.assertAlmostEqual(p["tempo"]["shots"][0], live.tempo.shots_home,
                                   places=1)

    def test_scoreline_grid_is_square_and_normalised(self):
        grid_n = PAYLOAD["model"]["grid"]
        for p in PAYLOAD["pairings"][:12]:
            g = p["grid"]
            self.assertEqual(len(g), grid_n)
            for row in g:
                self.assertEqual(len(row), grid_n)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in row))
            # a 7x7 window of a 11x11 distribution holds nearly all the mass
            self.assertGreater(sum(sum(r) for r in g), 0.95)

    def test_grid_peak_matches_the_top_scoreline(self):
        for p in PAYLOAD["pairings"][:12]:
            g = p["grid"]
            best = max((g[i][j], i, j) for i in range(len(g)) for j in range(len(g)))
            self.assertEqual("%d-%d" % (best[1], best[2]), p["scores"][0]["s"])

    def test_clubs_carry_ratings_and_records(self):
        for c in PAYLOAD["clubs"]:
            for key in ("id", "name", "attack", "defence", "net", "record"):
                self.assertIn(key, c)
            self.assertAlmostEqual(c["net"], c["attack"] + c["defence"], places=3)
            r = c["record"]
            self.assertEqual(r["pts"], 3 * r["w"] + r["d"])
            self.assertEqual(r["played"], r["w"] + r["d"] + r["l"])

    def test_meta_states_data_provenance(self):
        m = PAYLOAD["meta"]
        for key in ("snapshot", "seasons", "n_matches", "newest_match_label",
                    "injuries_available", "plan_limited", "source", "warnings"):
            self.assertIn(key, m)

    def test_payload_is_json_serialisable(self):
        blob = json.dumps(PAYLOAD)
        self.assertEqual(json.loads(blob)["meta"]["n_matches"],
                         PAYLOAD["meta"]["n_matches"])

    def test_no_nan_or_infinity_leaks_into_the_page(self):
        """json.dumps emits bare NaN/Infinity, which JSON.parse rejects."""
        blob = json.dumps(PAYLOAD)
        self.assertNotIn("NaN", blob)
        self.assertNotIn("Infinity", blob)


class TestPageBuild(unittest.TestCase):
    """Runs the real build script against the real sources."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        ui = Path(cls.tmp.name) / "ui"
        (ui / "src").mkdir(parents=True)
        for name in ("styles.css", "page.html", "app.js", "meta.json"):
            (ui / "src" / name).write_text((ROOT / "ui" / "src" / name).read_text())
        (ui / "build.py").write_text((ROOT / "ui" / "build.py").read_text())
        (ui / "data.json").write_text(json.dumps(PAYLOAD))
        res = subprocess.run([sys.executable, str(ui / "build.py")],
                             capture_output=True, text=True)
        cls.result = res
        cls.standalone = (ui / "index.html").read_text()
        cls.fragment = (ui / "artifact.html").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_succeeds(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_standalone_is_a_whole_document(self):
        self.assertTrue(self.standalone.startswith("<!doctype html>"))
        self.assertTrue(self.standalone.rstrip().endswith("</html>"))
        for tag in ("<html", "<head>", "<body>"):
            self.assertEqual(self.standalone.count(tag), 1, tag)

    def test_fragment_ships_no_skeleton(self):
        """The Artifact platform supplies it; a second copy breaks the page."""
        for tag in ("<!doctype", "<html", "<head>", "<body>"):
            self.assertNotIn(tag, self.fragment.lower(), tag)

    def test_both_carry_exactly_one_title(self):
        self.assertEqual(self.standalone.count("<title>"), 1)
        self.assertEqual(self.fragment.count("<title>"), 1)

    def test_embedded_data_parses_back(self):
        for name, html in (("standalone", self.standalone), ("fragment", self.fragment)):
            island = re.search(
                r'<script type="application/json" id="spl-data">(.*?)</script>',
                html, re.S).group(1)
            self.assertEqual(len(json.loads(island)["pairings"]),
                             len(PAYLOAD["pairings"]), name)

    def test_angle_brackets_in_data_cannot_close_the_script(self):
        island = re.search(
            r'<script type="application/json" id="spl-data">(.*?)</script>',
            self.standalone, re.S).group(1)
        self.assertNotIn("<", island)

    def test_no_external_dependency_beyond_google_fonts(self):
        urls = set(re.findall(r'https?://([a-z0-9.-]+)', self.standalone))
        self.assertTrue(urls <= {"fonts.googleapis.com", "fonts.gstatic.com"}, urls)

    def test_page_does_not_phone_home(self):
        """The page ships its data inline; it must never call out at runtime."""
        for needle in ("fetch(", "XMLHttpRequest", "WebSocket", "navigator.send"):
            self.assertNotIn(needle, self.standalone, needle)

    def test_every_element_id_the_script_reads_exists(self):
        page = (ROOT / "ui" / "src" / "page.html").read_text()
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        wanted = set(re.findall(r'\$\("([\w-]+)"\)', app))
        have = set(re.findall(r'id="([\w-]+)"', page)) | {"spl-data"}
        self.assertEqual(sorted(wanted - have), [])

    def test_theme_tokens_are_complete(self):
        """A token defined only inside a dark block renders one theme's text on
        the other theme's ground - the classic unreadable-page bug."""
        css = (ROOT / "ui" / "src" / "styles.css").read_text()
        bare = re.findall(r"(?<![\w\)\]]):root\s*\{([^}]*)\}", css)
        defined = set()
        for block in bare:
            defined |= set(re.findall(r"(--[a-z0-9-]+)\s*:", block))
        used = set(re.findall(r"var\((--[a-z0-9-]+)[,)]", css))
        self.assertEqual(sorted(used - defined), [])

    def test_both_dark_variants_cover_the_same_tokens(self):
        css = (ROOT / "ui" / "src" / "styles.css").read_text()
        media = re.search(r':root:not\(\[data-theme="light"\]\)\{([^}]*)\}', css)
        attr = re.search(r':root\[data-theme="dark"\]\{([^}]*)\}', css)
        self.assertIsNotNone(media)
        self.assertIsNotNone(attr)
        a = set(re.findall(r"(--[a-z0-9-]+)\s*:", media.group(1)))
        b = set(re.findall(r"(--[a-z0-9-]+)\s*:", attr.group(1)))
        self.assertEqual(a, b)

    def test_result_bar_segments_are_proportional(self):
        """flex-grow alone is not proportional - each segment's base width would
        come from its own label text, so the bar would not read to scale."""
        css = (ROOT / "ui" / "src" / "styles.css").read_text()
        bar = re.search(r"\.bar i\{([^}]*)\}", css).group(1)
        compact = re.sub(r"\s+", "", bar)
        self.assertIn("flex:00", compact)   # flex-basis 0, so width tracks the value

    def test_no_viewport_height_hero(self):
        self.assertNotIn("100vh", self.standalone)


if __name__ == "__main__":
    unittest.main()


class TestScheduledFixtures(unittest.TestCase):
    """Choosing a club must land on their real next match."""

    def test_fixtures_are_exported(self):
        self.assertIn("fixtures", PAYLOAD)
        self.assertIsInstance(PAYLOAD["fixtures"], list)

    def test_fixtures_are_sorted_by_kickoff(self):
        ks = [f["kickoff"] for f in PAYLOAD["fixtures"]]
        self.assertEqual(ks, sorted(ks))

    def test_fixtures_carry_what_the_page_shows(self):
        for f in PAYLOAD["fixtures"]:
            for key in ("home", "away", "kickoff", "label", "day"):
                self.assertIn(key, f)
            self.assertNotEqual(f["home"], f["away"])

    def test_fixtures_only_reference_exported_clubs(self):
        """A fixture naming a club not in the dropdown would break the lookup."""
        ids = {c["id"] for c in PAYLOAD["clubs"]}
        for f in PAYLOAD["fixtures"]:
            self.assertIn(f["home"], ids)
            self.assertIn(f["away"], ids)

    def test_every_fixture_has_a_precomputed_pairing(self):
        """Selecting a fixture must never hit a missing pairing."""
        pairs = {(p["home"], p["away"]) for p in PAYLOAD["pairings"]}
        for f in PAYLOAD["fixtures"]:
            self.assertIn((f["home"], f["away"]), pairs)

    def test_only_unplayed_matches_are_exported_as_fixtures(self):
        played = {(m.home_id, m.away_id, m.kickoff) for m in DS.played_matches}
        for f in PAYLOAD["fixtures"]:
            self.assertNotIn((f["home"], f["away"], f["kickoff"]), played)

    def test_page_wires_the_club_picker(self):
        page = (ROOT / "ui" / "src" / "page.html").read_text()
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn('id="clubsel"', page)
        self.assertIn('id="rail-strip"', page)
        self.assertIn('id="herometa"', page)
        self.assertIn("NEXT_OF", app)
        self.assertIn("FIXTURE_AT", app)

    def test_club_choice_keeps_true_home_and_away(self):
        """The chosen club may be the away side; the page must not flip it."""
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        block = app[app.index('clubsel.addEventListener'):]
        block = block[:block.index("function showFixtureStrip")]
        self.assertIn("homesel.value = f.home", block)
        self.assertIn("awaysel.value = f.away", block)

    def test_real_and_hypothetical_pairings_are_labelled_differently(self):
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn("Next fixture", app)
        self.assertIn("Hypothetical", app)

    def test_clubs_without_a_fixture_are_marked_in_the_picker(self):
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn("no fixture scheduled", app)


class TestClubBadges(unittest.TestCase):
    def test_export_carries_a_logo_field(self):
        for c in PAYLOAD["clubs"]:
            self.assertIn("logo", c)

    def test_page_renders_a_crest_with_an_initials_fallback(self):
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn("paintBadge", app)
        self.assertIn('img.addEventListener("error"', app)   # decode failure
        self.assertIn("initials(club.name)", app)            # and the fallback

    def test_crest_images_carry_alt_text(self):
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn('img.alt = club.name + " badge"', app)

    def test_badge_plate_is_dropped_behind_a_real_crest(self):
        css = (ROOT / "ui" / "src" / "styles.css").read_text()
        self.assertIn(".badge.hasimg", css)

    def test_crest_rule_outranks_the_tinted_plate(self):
        """`.team.h .badge` has three classes, so a bare `.badge.hasimg` loses
        the cascade and the coloured square shows through the crest."""
        css = (ROOT / "ui" / "src" / "styles.css").read_text()
        self.assertIn(".team.h .badge.hasimg", css)
        self.assertIn(".team.a .badge.hasimg", css)

    def test_club_labels_use_the_distinguishing_word(self):
        """Nearly every club is "Al something", so the first word names none of
        them - both form rows once read "Form, Al"."""
        app = (ROOT / "ui" / "src" / "app.js").read_text()
        self.assertIn("function shortName", app)
        self.assertIn("shortName(home.name)", app)
        self.assertIn("shortName(away.name)", app)
        self.assertNotIn('name.split(" ")[0]', app)
