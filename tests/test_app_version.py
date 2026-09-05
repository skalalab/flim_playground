"""The displayed version is nonempty and resolves without raising or hanging.
Resolution supports missing git, tagless checkouts, and unstamped bundles. The module
also imports without Streamlit so PyInstaller can use it at build time.
"""
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import version as version_mod
from src.version import get_app_version, get_version_label

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_version_cache():
    # Clear the process-wide cache between tests and later AppTests.
    get_app_version.cache_clear()
    yield
    get_app_version.cache_clear()


def test_frozen_bundle_reads_the_build_stamp(tmp_path, monkeypatch):
    (tmp_path / version_mod.STAMP_NAME).write_text("1.11.2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    # A bundle stamp takes precedence over a shell APP_VERSION override.
    monkeypatch.setenv("APP_VERSION", "9.9.9-env-must-not-win")
    assert get_app_version() == "1.11.2"


def test_frozen_bundle_without_a_stamp_falls_back_quietly(tmp_path, monkeypatch):
    # An unstamped bundle still resolves a version.
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.setattr(version_mod, "_git", lambda *a: None)
    assert get_app_version() == version_mod._FALLBACK


def test_env_override_wins_in_a_dev_checkout(monkeypatch):
    # The seam Flim-Playground.spec relies on in CI.
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setenv("APP_VERSION", "1.11.3")
    assert get_app_version() == "1.11.3"


def test_dev_checkout_reports_git_describe(monkeypatch):
    described = version_mod._git("describe", "--tags", "--dirty")
    if described is None:
        pytest.skip("no git or no tags in this checkout")
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)
    assert get_app_version() == described
    # A tag, not a bare sha: `--always` is deliberately not passed.
    assert described[:1].isdigit()


@pytest.mark.parametrize("boom", [
    FileNotFoundError("git"),                           # git not installed
    PermissionError("git"),                             # git not executable
    subprocess.TimeoutExpired(cmd="git", timeout=2.0),  # stalled checkout
])
def test_git_failures_fall_back_without_raising(monkeypatch, boom):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)

    def _raise(*a, **k):
        raise boom

    monkeypatch.setattr(version_mod.subprocess, "run", _raise)
    assert get_app_version() == version_mod._FALLBACK


def test_tagless_clone_reports_the_commit(monkeypatch):
    """A tagless checkout falls back from git describe to the commit SHA."""
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delenv("APP_VERSION", raising=False)

    def _fake_run(cmd, **kwargs):
        if "describe" in cmd:
            return types.SimpleNamespace(returncode=128, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="aeeaea1\n")

    monkeypatch.setattr(version_mod.subprocess, "run", _fake_run)
    assert get_app_version() == "dev+aeeaea1"


def test_git_outside_a_repo_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(version_mod, "_ROOT", tmp_path)
    assert version_mod._git("describe", "--tags") is None


def test_git_never_answers_from_an_ancestor_repo(monkeypatch):
    """Block git from borrowing a parent repository's version for an unpacked app."""
    monkeypatch.setattr(version_mod, "_ROOT", _ROOT / "src")
    assert (_ROOT / ".git").exists(), "precondition: the parent really is a repo"
    assert version_mod._git("describe", "--tags", "--dirty") is None


def test_stampless_frozen_app_inside_a_repo_falls_back(tmp_path, monkeypatch):
    """An unstamped bundle inside a repository reports 0.0.0-dev."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)  # no stamp
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.setattr(version_mod, "_ROOT", _ROOT / "src")
    assert get_app_version() == version_mod._FALLBACK


def test_version_is_never_empty_and_carries_no_whitespace():
    # The string is interpolated into HTML and into generated-script headers.
    version = get_app_version()
    assert version
    assert version == version.strip()
    assert " " not in version


def test_label_prefixes_a_real_version_but_not_a_dev_marker(monkeypatch):
    monkeypatch.setattr(version_mod, "get_app_version", lambda: "1.11.2")
    assert get_version_label() == "v1.11.2"
    monkeypatch.setattr(version_mod, "get_app_version", lambda: "dev+aeeaea1")
    assert get_version_label() == "dev+aeeaea1"


def test_module_resolves_a_version_with_no_streamlit_imported():
    """The version module imports at build time and in standalone exports without Streamlit."""
    code = (
        "import sys, src.version as v; "
        "assert 'streamlit' not in sys.modules, sorted(sys.modules); "
        "print(v.get_app_version())"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_ROOT), capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip()
