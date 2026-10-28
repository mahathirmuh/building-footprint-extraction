"""Provenance stamps for generated files, so every reported number can be traced to the code that produced it."""

from __future__ import annotations

import datetime as dt
import subprocess

from densepoly.paths import REPO_ROOT


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_revision() -> str:
    """Current commit, suffixed with '-dirty' when tracked files have uncommitted changes; 'unknown' outside git."""
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def timestamp() -> str:
    """Local time with UTC offset, ISO 8601, to the second."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")
