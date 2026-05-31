"""Shared pytest setup for the basketball pipeline tests.

The simulation modules call pygame.init() and load the court image at import
time, so pygame must run headless. They also resolve their output directories
from env vars (falling back to ./<name> off the cwd) -- point the noisy ones at
a temp dir so test runs don't pollute the repo. ASSETS_DIR is intentionally left
unset so it defaults to ./assets, where the court image lives.
"""

import os
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_tmp = tempfile.mkdtemp(prefix="bball_test_")
os.environ.setdefault("LOG_DIR", _tmp)
os.environ.setdefault("VIDEO_DIR", _tmp)
os.environ.setdefault("TRACKING_DIR", _tmp)
