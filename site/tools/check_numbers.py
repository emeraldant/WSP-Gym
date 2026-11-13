"""Check that every percentage, dollar amount and decimal shown on the rendered page can be derived from
site/data.json. Renders the page in headless Chromium (animations at their end state), collects the
visible text including SVG labels, and lists any number it cannot trace to the data.

    source site/tools/env.sh && $TOOLS_PY site/tools/check_numbers.py
"""
import json
import pathlib
import re
from decimal import ROUND_HALF_UP, Decimal

from playwright.sync_api import sync_playwright

SITE = pathlib.Path(__file__).resolve().parents[1]
D = json.loads((SITE / "data.json").read_text())


def floats(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from floats(v)
    elif isinstance(o, list):
        for v in o:
            yield from floats(v)


def q(x, d):
    return str(Decimal(str(round(x, 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


allowed = set()
for x in set(floats(D)):
    for s in (100 * x, x):
        for d in (0, 1, 2):
            v = q(s, d)
            allowed |= {v, v.lstrip("-"), ("+" + v) if not v.startswith("-") else v}
            if abs(s) >= 1000:
                allowed.add(f"{Decimal(v):,}")
    allowed.add(f"{round(x / 100) * 100:,.0f}")  # "about 10,600"
for v in D["fields"].values():  # display strings computed by build_data.py
    for m in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(v)):
        allowed.add(m)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
    pg.goto((SITE / "index.html").as_uri())
    pg.wait_for_timeout(1500)
    text = pg.evaluate("""() => {
      const parts = [document.body.innerText];
      document.querySelectorAll('svg text').forEach(t => parts.push(t.textContent));
      document.querySelectorAll('[data-tip]').forEach(t => parts.push(t.getAttribute('data-tip')));
      document.querySelectorAll('svg[aria-label],[role=img][aria-label]').forEach(t => parts.push(t.getAttribute('aria-label')));
      return parts.join('\\n');
    }""")
    b.close()

# drop code blocks and the commands/prompt, whose numbers are source text, not results
for pre in re.findall(r"(?s)\n\d+\s+tc = await.*?\n", text):
    text = text.replace(pre, "\n")
tokens = re.findall(r"(?<![\w.\-])[-+~$]?\d[\d,]*(?:\.\d+)?%?", text)
checked, bad = 0, []
for t in tokens:
    if not (t.endswith("%") or "." in t or t.startswith("$")):
        continue
    core = t.lstrip("~$").rstrip("%")
    checked += 1
    if core not in allowed and core.lstrip("+") not in allowed:
        bad.append(t)
print(f"checked {checked} percentages / decimals / dollar amounts on the rendered page")
print("not traceable to data.json:", sorted(set(bad)) or "none")
