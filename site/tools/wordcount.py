"""Count the words of visible text on the rendered page (excluding code blocks and the collapsed prompt).

    source site/tools/env.sh && $TOOLS_PY site/tools/wordcount.py [--print]
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

SITE = pathlib.Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
    pg.goto((SITE / "index.html").as_uri())
    pg.wait_for_timeout(1200)
    text = pg.evaluate("""() => { const c = document.body.cloneNode(true);
      c.querySelectorAll('pre, script, style, svg, .sr-only, details, .toc').forEach(e => e.remove());
      return c.innerText; }""")
    b.close()
print(len(text.split()), "words of visible prose (figures, code, prompt and navigation excluded)")
if "--print" in sys.argv:
    print(text)
