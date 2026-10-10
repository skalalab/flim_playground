"""Update the desktop app in place to the latest GitHub release.

The launcher looks the release up at start, and ``navigation.render_top_menu``
offers it, then runs ``prepare`` and ``finish``. Nothing here imports Streamlit.
The swap happens in ``finish``, the app's last act, so no running code reads a
half-replaced bundle; on Windows the existing installer does it after the exit.
"""
import contextlib
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import requests
from packaging.version import InvalidVersion, Version

from src.version import STAMP_NAME, get_app_version

LATEST_RELEASE_API = "https://api.github.com/repos/skalalab/flim_playground/releases/latest"
RELEASE_NOTES_URL = "https://github.com/skalalab/flim_playground/releases/latest"

# What build.yml's Stage step puts in the Linux folder. Anything else means the
# tarball was unpacked into a shared folder such as ~/Downloads, which must never move.
_LINUX_APP_FILES = {"Flim-Playground", "_internal", "install.sh", "uninstall.sh", "flim-playground.png"}

# Waits for the app to exit, installs silently, records a failure, reopens the app.
# No double quotes: Windows PowerShell 5.1 drops them from -Command arguments.
# Start-Process -Wait waits for the installer's whole process tree, so the .iss
# [Run] entry must keep skipifsilent, or this would wait on the app it launched.
_WINDOWS_HELPER = (
    "Wait-Process -Id $env:FP_PID -ErrorAction SilentlyContinue; "
    "$p = Start-Process $env:FP_SETUP -ArgumentList '/SILENT','/SUPPRESSMSGBOXES','/NORESTART',"
    "'/NOCANCEL','/FORCECLOSEAPPLICATIONS' -Wait -PassThru; "
    "if ($p.ExitCode -ne 0) { Set-Content $env:FP_FAILED ('The installer stopped with exit code ' + $p.ExitCode) }; "
    "$env:PYINSTALLER_RESET_ENVIRONMENT = '1'; Start-Process $env:FP_EXE"
)

_FAILED = "FAILED.txt"

_release = None  # the latest release's JSON, once the launch check succeeds
_failure = None  # why the previous update didn't finish, until it is shown


def asset_name():
    """The release asset this build updates from, as build.yml's Stage step names it."""
    machine = platform.machine()
    if sys.platform == "darwin":
        return "Flim-Playground-mac-intel.tar.gz" if machine == "x86_64" else "Flim-Playground-mac.tar.gz"
    if sys.platform == "win32":
        return "Flim-Playground-Setup.exe"
    if sys.platform.startswith("linux") and machine == "x86_64":
        return "Flim-Playground-linux.tar.gz"
    return None


def pick_update(release, current, name):
    """``(tag, asset)`` when ``release`` is newer than ``current`` and its ``name`` asset
    is fully uploaded with a SHA-256 to check the download against; otherwise None.

    A ``current`` that isn't a version (a local git-describe build) or is a dev release
    (a dispatch build stamped 0.0.0-dev, perhaps newer code than ``latest``) gets none.
    """
    try:
        latest, here = Version(release["tag_name"]), Version(current)
    except (InvalidVersion, KeyError, TypeError):
        return None
    if here.is_devrelease or latest <= here:
        return None
    return next(
        ((release["tag_name"], asset) for asset in release.get("assets", [])
         if asset.get("name") == name and asset.get("state") == "uploaded"
         and str(asset.get("digest")).startswith("sha256:")),
        None,
    )


def available_update():
    """The update the launch check found for this build, or None."""
    return pick_update(_release, get_app_version(), asset_name()) if _release else None


def _app_path():
    """The folder an update replaces: the .app on macOS, the onedir folder on Linux."""
    exe = Path(sys.executable)
    return exe.parents[2] if sys.platform == "darwin" else exe.parent


def _staging():
    """Where an update is unpacked: beside the app, so the swap is a rename; %TEMP% on Windows."""
    if sys.platform == "win32":
        return Path(tempfile.gettempdir()) / "flim-playground-update"
    return _app_path().parent / ".flim-playground-update"


def start_check():
    """In frozen builds, keep the note a failed update left, then remove the previous
    update's files and look up the latest release in the background.

    Never raises: the launcher shuts the app down on any exception.
    """
    global _failure
    if not getattr(sys, "frozen", False):
        return
    with contextlib.suppress(OSError, ValueError):
        _failure = (_staging() / _FAILED).read_text(encoding="utf-8").strip() or None
    threading.Thread(target=_check, daemon=True).start()


def _check():
    global _release
    shutil.rmtree(_staging(), ignore_errors=True)
    # Offline, rate limited or blocked: offer nothing this launch.
    with contextlib.suppress(requests.RequestException, ValueError):
        response = requests.get(LATEST_RELEASE_API, timeout=10)
        response.raise_for_status()
        _release = response.json()


def pop_failure():
    """Why the previous update didn't finish, once; None after that, or when it did."""
    global _failure
    failure, _failure = _failure, None
    return failure


def prepare(tag, asset, on_progress):
    """Download ``asset`` and check it, then unpack it beside the app (macOS, Linux) or
    start the helper that installs it once this process exits (Windows).

    Returns the new app folder for ``finish`` to swap in, or None on Windows.
    ``on_progress(done, total)`` follows the download in bytes. Raises OSError,
    ValueError or SubprocessError, having left the running app untouched.
    """
    app = _app_path()
    if sys.platform == "darwin" and app.suffix != ".app":
        raise ValueError(f"{app} isn't an app bundle")
    if sys.platform.startswith("linux") and (extra := sorted(set(os.listdir(app)) - _LINUX_APP_FILES)):
        raise ValueError(f"{app} also holds {', '.join(extra[:3])}")
    staging = _staging()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    download = staging / asset["name"]
    digest = hashlib.sha256()
    with requests.get(asset["browser_download_url"], stream=True, timeout=30) as response:
        response.raise_for_status()
        with download.open("wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                on_progress(fh.tell(), asset["size"])
    if f"sha256:{digest.hexdigest()}" != asset["digest"]:
        raise ValueError("the download doesn't match the release's checksum")
    if sys.platform == "win32":
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_HELPER],
            env={**os.environ, "FP_PID": str(os.getpid()), "FP_SETUP": str(download),
                 "FP_EXE": sys.executable, "FP_FAILED": str(staging / _FAILED)},
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return None
    new = staging / "new"
    new.mkdir()
    subprocess.run(["tar", "-xzf", str(download), "-C", str(new)], check=True)
    download.unlink()
    new_app = new / "Flim-Playground.app" if sys.platform == "darwin" else new
    stamp = new_app / ("Contents/Resources" if sys.platform == "darwin" else "_internal") / STAMP_NAME
    if stamp.read_text(encoding="utf-8").strip() != tag:
        raise ValueError(f"the download isn't version {tag}")
    return new_app


def _reopen(app):
    """Start ``app`` as a new, independent instance, as a double-click would."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-n", str(app)])
    else:
        subprocess.Popen(
            [str(app / "Flim-Playground")], cwd=app, start_new_session=True,
            env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def finish(new_app):
    """The app's last act: swap ``new_app`` in for the running app and reopen it
    (macOS, Linux), or leave it to the installer helper (Windows). Always exits.
    """
    try:
        if new_app is not None:
            app, staging = _app_path(), _staging()
            try:
                app.rename(staging / "old")
                try:
                    new_app.rename(app)
                except OSError:
                    (staging / "old").rename(app)
                    raise
            except OSError as error:
                (staging / _FAILED).write_text(str(error), encoding="utf-8")
            _reopen(app)
    finally:
        os._exit(0)
