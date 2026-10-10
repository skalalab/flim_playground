"""The in-app update: which release is offered, what is downloaded, how the app is swapped."""
import hashlib
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

from src import updater
from tests.test_quit import _body_rendered, _exits_after, _render

_TOKEN = "t0k3n"


def _release(tag="99.0.0", **asset_fields):
    asset = {"name": "Flim-Playground-mac.tar.gz", "state": "uploaded", "size": 3,
             "digest": "sha256:" + "0" * 64, "browser_download_url": "https://example.invalid/a"}
    asset.update(asset_fields)
    return {"tag_name": tag, "assets": [asset]}


@pytest.mark.parametrize(("release", "current", "offered"), [
    (_release(), "1.14.4", True),
    (_release(tag="1.14.4"), "1.14.4", False),  # already current
    (_release(tag="1.14.3"), "1.14.4", False),  # older
    ({"tag_name": "99.0.0", "assets": []}, "1.14.4", False),  # published while CI still builds
    (_release(state="open"), "1.14.4", False),  # upload unfinished
    (_release(digest=None), "1.14.4", False),  # nothing to verify the download against
    (_release(tag="nightly"), "1.14.4", False),  # not a version
    (_release(), "1.14.4-15-g45dae78-dirty", False),  # a local build reports git describe
    (_release(), "0.0.0-dev", False),  # a dispatch build: `latest` may be older code
])
def test_offers_only_a_newer_release_it_can_verify(release, current, offered):
    update = updater.pick_update(release, current, "Flim-Playground-mac.tar.gz")
    assert update == (("99.0.0", release["assets"][0]) if offered else None)


def _install(tmp_path, monkeypatch):
    """Point sys.executable into a fake installed app and return the app folder."""
    if sys.platform == "darwin":
        app = tmp_path / "Downloads" / "Flim-Playground.app"
        exe = app / "Contents" / "MacOS" / "Flim-Playground"
    else:
        app = tmp_path / "Downloads" / "Flim-Playground-linux"
        exe = app / "Flim-Playground"
    exe.parent.mkdir(parents=True)
    exe.write_text("old")
    monkeypatch.setattr(sys, "executable", str(exe))
    return app


def _stamp(app):
    return app / ("Contents/Resources" if sys.platform == "darwin" else "_internal") / "VERSION.txt"


def _tarball(tag):
    """A release archive laid out as build.yml's Stage step lays out this platform's."""
    member = "Flim-Playground.app/Contents/Resources/VERSION.txt" if sys.platform == "darwin" else "_internal/VERSION.txt"
    data = f"{tag}\n".encode()
    info = tarfile.TarInfo(member)
    info.size = len(data)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))
    return archive.getvalue()


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows hands the download to its installer")
def test_prepare_unpacks_only_a_download_matching_the_release_checksum(tmp_path, monkeypatch):
    app = _install(tmp_path, monkeypatch)
    body = _tarball("99.0.0")
    monkeypatch.setattr(updater.requests, "get", lambda url, **kwargs: _Response(body))
    asset = _release(size=len(body))["assets"][0]

    with pytest.raises(ValueError, match="checksum"):
        updater.prepare("99.0.0", asset, lambda done, total: None)
    assert not (app.parent / ".flim-playground-update" / "new").exists()

    asset["digest"] = "sha256:" + hashlib.sha256(body).hexdigest()
    new_app = updater.prepare("99.0.0", asset, lambda done, total: None)
    assert _stamp(new_app).read_text().strip() == "99.0.0"
    assert app.is_dir(), "preparing must leave the running app in place"


def test_prepare_refuses_a_linux_folder_shared_with_other_files(tmp_path, monkeypatch):
    # A bare `tar xzf` in ~/Downloads makes Downloads itself the app folder; it must never move.
    folder = tmp_path / "Downloads"
    (folder / "_internal").mkdir(parents=True)
    (folder / "Flim-Playground").write_text("old")
    (folder / "notes.txt").write_text("the user's own file")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", str(folder / "Flim-Playground"))
    monkeypatch.setattr(updater.requests, "get", lambda *args, **kwargs: pytest.fail("downloaded before refusing"))

    with pytest.raises(ValueError, match=r"notes\.txt"):
        updater.prepare("99.0.0", _release()["assets"][0], lambda done, total: None)
    assert (folder / "notes.txt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows hands the swap to its installer")
@pytest.mark.parametrize("rename_fails", [False, True])
def test_finish_swaps_in_the_new_app_or_puts_the_old_one_back(tmp_path, monkeypatch, rename_fails):
    app = _install(tmp_path, monkeypatch)
    (app / "which").write_text("old")
    staging = app.parent / ".flim-playground-update"
    new_app = staging / "new" / "Flim-Playground.app" if sys.platform == "darwin" else staging / "new"
    new_app.mkdir(parents=True)
    (new_app / "which").write_text("new")
    reopened, exits = [], []
    monkeypatch.setattr(updater, "_reopen", reopened.append)
    monkeypatch.setattr(os, "_exit", exits.append)
    if rename_fails:
        rename = os.rename

        def refuse_new(src, dst):
            if Path(src) == new_app:
                raise PermissionError(1, "Operation not permitted")
            rename(src, dst)

        monkeypatch.setattr(os, "rename", refuse_new)

    updater.finish(new_app)

    assert exits == [0]
    assert reopened == [app]
    assert (app / "which").read_text() == ("old" if rename_fails else "new")
    if rename_fails:
        assert "Operation not permitted" in (staging / "FAILED.txt").read_text()
    else:
        assert (staging / "old" / "which").read_text() == "old"


def _offer(monkeypatch):
    """Run as this launch, on a release with a newer build for this platform."""
    monkeypatch.setenv("FLIM_PLAYGROUND_QUIT_TOKEN", _TOKEN)
    monkeypatch.setattr(updater, "get_app_version", lambda: "1.14.4")
    monkeypatch.setattr(updater, "_release", _release(name=updater.asset_name()))


def test_update_link_asks_before_updating(monkeypatch):
    _offer(monkeypatch)

    at, exits = _render(monkeypatch, update=_TOKEN)

    assert [button.label for button in at.button] == ["Update now"]
    assert not _body_rendered(at)
    assert not _exits_after(exits, 1.5)


def test_update_now_hands_the_prepared_app_to_finish(monkeypatch):
    _offer(monkeypatch)
    finished = []
    monkeypatch.setattr(updater, "prepare", lambda tag, asset, on_progress: "NEW-APP")
    monkeypatch.setattr(updater, "finish", finished.append)
    at, _exits = _render(monkeypatch, update=_TOKEN)

    at.button[0].click().run(timeout=60)

    assert any("Installing v99.0.0" in info.value for info in at.info)
    deadline = time.monotonic() + 3
    while not finished and time.monotonic() < deadline:
        time.sleep(0.05)
    assert finished == ["NEW-APP"]
