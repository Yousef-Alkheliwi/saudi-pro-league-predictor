#!/usr/bin/env python3
"""Build the Match Lab page from ui/src/ + ui/data.json.

Two outputs from one set of sources:

  ui/index.html      standalone - full HTML document, opens by double-clicking
  ui/artifact.html   fragment   - no <html>/<head>/<body>, for publishing as an
                                  Artifact, whose platform supplies that skeleton
                                  and a small CSS reset

Regenerate after every `python -m spl.cli export`:

    python ui/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

UI = Path(__file__).resolve().parent
SRC = UI / "src"

# The Artifact platform injects this reset; a standalone file has to carry its own
# equivalent so the two render the same.
STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
{fonts}
<style>
:root{{color-scheme:light dark;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px)}}
html,body{{margin:0}}
body{{font:14px system-ui,-apple-system,"Segoe UI",sans-serif; background:#fcfcfb}}
img{{max-width:100%}}
[hidden]{{display:none!important}}
</style>
<style>
{css}</style>
</head>
<body>
"""


def build() -> int:
    data_path = UI / "data.json"
    if not data_path.exists():
        print("missing %s - run `python -m spl.cli export --out ui/data.json` first"
              % data_path, file=sys.stderr)
        return 1

    meta = json.loads((SRC / "meta.json").read_text())
    css = (SRC / "styles.css").read_text()
    page = (SRC / "page.html").read_text()
    app = (SRC / "app.js").read_text()

    data = json.loads(data_path.read_text())
    # escaping "<" means no value in the data can ever close the script element
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    island = '<script type="application/json" id="spl-data">%s</script>\n' % blob
    script = "<script>\n%s</script>\n" % app

    standalone = (STANDALONE_HEAD.format(title=meta["title"], fonts=meta["fonts"],
                                         css=css)
                  + island + page + "\n" + script + "</body>\n</html>\n")
    (UI / "index.html").write_text(standalone, encoding="utf-8")

    fragment = ("<title>%s</title>\n%s\n<style>\n%s</style>\n"
                % (meta["title"], meta["fonts"], css)) + island + page + "\n" + script
    (UI / "artifact.html").write_text(fragment, encoding="utf-8")

    n = len(data.get("pairings", []))
    for name in ("index.html", "artifact.html"):
        p = UI / name
        print("  ui/%-16s %6.0f KB" % (name, p.stat().st_size / 1024))
    print("  %d pairings, %d clubs, self-contained (fonts from Google, with fallbacks)"
          % (n, len(data.get("clubs", []))))
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
