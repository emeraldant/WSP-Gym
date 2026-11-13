"""Screenshot the built page at desktop and phone widths, light and dark, with animations at their end state.

    source site/tools/env.sh && $TOOLS_PY site/tools/shoot.py OUT_DIR
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

SITE = pathlib.Path(__file__).resolve().parents[1]
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "shots")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, w, h in (("desktop", 1280, 900), ("phone", 390, 844)):
        for theme in ("light", "dark"):
            ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2 if name == "phone" else 1,
                                reduced_motion="reduce", color_scheme=theme)
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto((SITE / "index.html").as_uri())
            pg.wait_for_timeout(1500)
            pg.screenshot(path=str(out / f"{name}-{theme}.png"), full_page=True)
            sw = pg.evaluate("document.documentElement.scrollWidth")
            print(name, theme, "scrollWidth", sw, "errors", errs)
            ctx.close()
    b.close()
