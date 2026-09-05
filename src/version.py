"""Resolve a shared version for builds, navigation, and exported analysis scripts.

Packaged apps read VERSION.txt, written by the build with this same resolver.
Other runs use APP_VERSION or git metadata, with a development fallback.
Keep this module stdlib-only so builds and standalone callers need no Streamlit.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path

# Shared by the build spec and runtime reader.
STAMP_NAME = "VERSION.txt"

# Last resort when there is neither a stamp nor a reachable repo.
_FALLBACK = "0.0.0-dev"

_GIT_TIMEOUT_S = 2.0

# Resolve git metadata from this project, independent of the launch directory.
_ROOT = Path(__file__).resolve().parent.parent


def _build_stamp() -> str | None:
    """Read the bundled version, or return None when no usable stamp is available."""
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    try:
        return (Path(meipass) / STAMP_NAME).read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _git(*args: str) -> str | None:
    """Run git in the project root with a timeout, returning None on failure.

    Require a local .git entry to prevent git from describing a parent repository.
    Accept both directories and worktree/submodule files. Suppress diagnostics and
    close stdin so metadata reads cannot prompt the user.
    """
    if not (_ROOT / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


@functools.lru_cache(maxsize=1)
def get_app_version() -> str:
    """Return the unprefixed version, cached until cache_clear() or module reload.

    Prefer the bundle stamp, then APP_VERSION, a git tag description, a labelled
    development commit, and finally the fallback. Include lightweight tags and
    dirty state in git descriptions.
    """
    return (
        _build_stamp()
        or os.environ.get("APP_VERSION", "").strip()
        or _git("describe", "--tags", "--dirty")
        or _dev_commit()
        or _FALLBACK
    )


def _dev_commit() -> str | None:
    """Return ``dev+<sha>`` when a commit is available without a tag description."""
    sha = _git("rev-parse", "--short", "HEAD")
    return f"dev+{sha}" if sha else None


def get_version_label() -> str:
    """Prefix numeric versions with ``v`` for display; leave development labels bare."""
    version = get_app_version()
    return f"v{version}" if version[:1].isdigit() else version
