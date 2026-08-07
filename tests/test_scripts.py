from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=30, env=env
    )


def test_play_help_smoke():
    result = _run(["scripts/play.py", "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--preset" in result.stdout


def test_collect_demos_help_smoke():
    result = _run(["scripts/collect_demos.py", "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output" in result.stdout


def test_collect_demos_requires_output():
    result = _run(["scripts/collect_demos.py"])
    assert result.returncode != 0
    assert "--output" in result.stderr
