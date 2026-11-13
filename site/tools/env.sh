# Source this to use the private headless-browser toolchain installed under site/.tools (not the repo's .venv).
SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PLAYWRIGHT_BROWSERS_PATH="$SITE_DIR/.tools/browsers"
export LD_LIBRARY_PATH="$SITE_DIR/.tools/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOOLS_PY="$SITE_DIR/.tools/venv/bin/python"
