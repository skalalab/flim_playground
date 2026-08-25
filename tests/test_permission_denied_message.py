"""The 'permission denied' folder message must stay general across platforms
and causes, adding an OS-specific hint only when the signal actually matches
(macOS TCC/network-volume, Windows Controlled Folder Access). A plain local
filesystem denial (errno 13, EACCES) must NOT get network-volume advice."""
import errno
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.metadata_widgets import _permission_denied_message

GENERIC = "doesn't have permission to list this folder"
MAC_HINT = "Privacy & Security"
WIN_HINT = "Controlled Folder"


def _err(code):
    e = PermissionError()
    e.errno = code
    return e


def test_generic_sentence_present_on_every_platform(monkeypatch):
    for plat in ("darwin", "win32", "linux"):
        monkeypatch.setattr(sys, "platform", plat)
        msg = _permission_denied_message("/some/path", _err(errno.EACCES))
        assert GENERIC in msg


def test_macos_eperm_shows_privacy_hint(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    msg = _permission_denied_message("/Users/x/data", _err(errno.EPERM))
    assert MAC_HINT in msg
    assert "terminal or IDE" in msg  # dev build names the launching app


def test_macos_volumes_path_shows_privacy_hint_even_on_eacces(monkeypatch):
    # Network shares can surface as EACCES; the /Volumes signal still applies.
    monkeypatch.setattr(sys, "platform", "darwin")
    msg = _permission_denied_message("/Volumes/skala/data", _err(errno.EACCES))
    assert MAC_HINT in msg


def test_macos_local_eacces_is_generic_only(monkeypatch):
    # A mode-000 local folder on mac must not get network-volume advice.
    monkeypatch.setattr(sys, "platform", "darwin")
    msg = _permission_denied_message("/Users/x/private", _err(errno.EACCES))
    assert MAC_HINT not in msg
    assert GENERIC in msg


def test_macos_frozen_names_the_app(monkeypatch):
    # In the packaged .app, FLIM Playground itself is the responsible app.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    msg = _permission_denied_message("/Volumes/skala", _err(errno.EPERM))
    assert "FLIM Playground" in msg
    assert "terminal or IDE" not in msg


def test_windows_shows_controlled_folder_access(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    msg = _permission_denied_message(r"C:\\Users\\x\\Documents", _err(errno.EACCES))
    assert WIN_HINT in msg


def test_linux_is_generic_only(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    msg = _permission_denied_message("/mnt/share/data", _err(errno.EACCES))
    assert MAC_HINT not in msg
    assert WIN_HINT not in msg
    assert GENERIC in msg
