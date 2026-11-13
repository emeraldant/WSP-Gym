"""Record the page's own animations, frame by frame, into share-ready clips in site/media/.

Each figure's animation is a pure function of time, so a clip is made by loading the page with
?clip=<name> (the figure alone in a 1280x720 frame), calling window.__clip.seek(name, t) for every
frame and taking a screenshot. Frames are encoded to MP4 (H.264) and GIF with the ffmpeg binary
shipped by the imageio-ffmpeg wheel in the private tool environment.

    source site/tools/env.sh && $TOOLS_PY site/tools/export_clips.py [name ...]
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

SITE = pathlib.Path(__file__).resolve().parents[1]
MEDIA = SITE / "media"
FPS = 30
CLIPS = {  # name: (seconds of video, theme). Animations hold their last frame until the clip ends.
    "race": (17.0, "light"),
    "splits": (7.0, "light"),
    "repair": (10.5, "light"),
    "loop": (13.5, "light"),
    "disguise": (14.0, "light"),
}
STILLS = {  # name: (clip, width, height, zoom). The summary card at its end state.
    "summary-1600x900.png": ("card", 1600, 900, 1.0),
    "og-1200x630.png": ("card", 1600, 840, 0.75),
}


def main(names):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    MEDIA.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name in [n for n in (names or list(CLIPS)) if n in CLIPS]:
            secs, theme = CLIPS[name]
            pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
            pg.goto((SITE / "index.html").as_uri() + f"?clip={name}&theme={theme}")
            pg.evaluate("document.fonts.ready")
            pg.wait_for_timeout(800)
            pg.evaluate(f"window.__clip.seek('{name}', 0)")  # rebuild after fonts settled
            dur = pg.evaluate(f"window.__clip.dur('{name}')")
            tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"clip-{name}-"))
            n = int(secs * FPS)
            for i in range(n):
                t = min(dur, i / FPS)
                pg.evaluate(f"window.__clip.seek('{name}', {t})")
                pg.screenshot(path=str(tmp / f"f{i:05d}.png"))
            pg.close()
            mp4, gif = MEDIA / f"{name}.mp4", MEDIA / f"{name}.gif"
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"), "-vf",
                            "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                            str(gif)], check=True)
            shutil.rmtree(tmp)
            print(f"{name}: {n} frames -> {mp4.name} ({mp4.stat().st_size // 1024} KB), {gif.name} ({gif.stat().st_size // 1024} KB)")
        if not names or "card" in names:
            for out, (name, w, h, z) in STILLS.items():
                pg = b.new_page(viewport={"width": int(w * z), "height": int(h * z)})
                pg.goto((SITE / "index.html").as_uri() + f"?clip={name}&theme=light&w={w}&h={h}&zoom={z}")
                pg.evaluate("document.fonts.ready")
                pg.wait_for_timeout(800)
                pg.evaluate(f"window.__clip.seek('{name}', window.__clip.dur('{name}'))")
                pg.screenshot(path=str(MEDIA / out))
                pg.close()
                print(f"{out}: still of '{name}'")
        b.close()


if __name__ == "__main__":
    main(sys.argv[1:])
