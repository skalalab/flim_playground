"""The one place the app's version string is resolved.

FLIM Playground is not an installed distribution -- pyproject.toml has no
[build-system], so nothing named "flim-playground" is ever on the path for
importlib.metadata to query, and pyproject.toml is not in the spec's `datas`
either. The version therefore travels two different ways, and this module is the
only code that knows either of them:

- **Packaged app**: Flim-Playground.spec calls this same resolver at *build*
  time and writes the answer into a VERSION.txt inside the bundle. That
  symmetry is what keeps the nav bar, the Windows .exe resource and the macOS
  Info.plist from disagreeing -- there is one answer per build.
- **Development / Streamlit Cloud**: nothing is stamped, so ask git. Tags are
  the truth in a checkout, and a tagless shallow clone still yields a commit,
  which is enough to identify a build.

APP_VERSION is a *build-time* variable (set job-level in build.yml from the
`release: published` event payload) and does not exist when a user runs the app,
which is exactly why the shipped app reads a file instead.

Every consumer -- the nav bar, the exported analysis script -- calls
get_app_version() rather than reaching for any of these sources itself.

Deliberately stdlib-only: no streamlit import, and functools.lru_cache rather
than st.cache_data. Flim-Playground.spec imports this module during a build, the
tests import it standalone, and src/export_script.py may never inline a
streamlit-dependent helper into the standalone script it generates.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path

# Name of the file the spec writes into the bundle. Kept here so the spec and
# the runtime read cannot drift apart -- the spec imports this constant.
STAMP_NAME = "VERSION.txt"

# Last resort when there is neither a stamp nor a reachable repo.
_FALLBACK = "0.0.0-dev"

_GIT_TIMEOUT_S = 2.0

# The repo this module lives in. git is asked about *this* tree, never about
# whatever directory the server happened to be started from: streamlit's cwd is
# wherever the user typed `streamlit run`, so without an explicit cwd, launching
# the app from a parent directory would describe the wrong repo.
_ROOT = Path(__file__).resolve().parent.parent


def _build_stamp() -> str | None:
    """The version the spec wrote into the bundle, or None when not frozen.

    ``sys._MEIPASS`` is the frozen-vs-dev idiom used at src/config.py:39 and
    main.py:15-18. A bundle missing the stamp is a build-config bug, not a
    reason to crash: return None and let the chain continue so the app starts.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    try:
        return (Path(meipass) / STAMP_NAME).read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _git(*args: str) -> str | None:
    """Read-only git in the repo root, or None for every way it can go wrong.

    Never raises and never hangs: OSError covers git being absent or not
    executable, SubprocessError covers TimeoutExpired on a stalled network
    checkout, and a non-zero exit covers "not a repository" and "no tags".
    stderr is discarded so git's `fatal:` chatter never reaches the user's
    terminal, and stdin is closed so git can never block on a prompt.

    The `.git` check is not a shortcut -- it stops git's *upward* search. git
    walks parent directories looking for a repository, so a stampless app
    unpacked anywhere inside someone's checkout would otherwise answer with
    THAT repository's tag: a wrong-but-plausible version, which is worse than
    an honest 0.0.0-dev and defeats the point of shipping a version at all.
    Tested via `_ROOT = src/` (not a repo, but a repo's child). `.exists()`
    rather than `.is_dir()` because a worktree or submodule makes `.git` a file.
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
    """The bare version string, e.g. ``1.11.2`` or ``1.11.2-4-gaeeaea1-dirty``.

    Cached for the life of the process: streamlit re-runs the whole script on
    every widget interaction, and an uncached call would fork git each time.
    ``functools.lru_cache`` and not ``st.cache_data`` -- this module must stay
    importable with no streamlit present. Tests call ``cache_clear()``.

    The trade-off of caching: a tag created while `streamlit run` is up is not
    picked up until the module is re-imported.

    ``--tags`` is required because this repo's tags are lightweight (bare
    ``MAJOR.MINOR.PATCH``) and plain ``git describe`` only sees annotated ones.
    ``--dirty`` so an uncommitted tree says so. No ``--always``: a bare SHA is
    handled explicitly below, labelled, rather than silently masquerading as a
    describe output.
    """
    return (
        _build_stamp()
        or os.environ.get("APP_VERSION", "").strip()
        or _git("describe", "--tags", "--dirty")
        or _dev_commit()
        or _FALLBACK
    )


def _dev_commit() -> str | None:
    """``dev+<sha>`` for a repo with no tags to describe from.

    The Streamlit Cloud deploy and shallow CI clones land here. A labelled
    commit still identifies a build, which is the whole point; ``dev+`` marks it
    as not-a-release so it can never be mistaken for a version number. No
    spaces or parentheses -- the string ends up in HTML and in file headers.
    """
    sha = _git("rev-parse", "--short", "HEAD")
    return f"dev+{sha}" if sha else None


def get_version_label() -> str:
    """The display form: ``v1.11.2``, but ``dev+aeeaea1`` left bare.

    The ``v`` is presentation and belongs only here; get_app_version() stays
    bare because the export script's header wants the plain string.
    """
    version = get_app_version()
    return f"v{version}" if version[:1].isdigit() else version
