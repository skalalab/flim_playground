"""Pytest entry point for the visual-encoding checks.

The checks live in ``check_*.py`` rather than ``test_*.py`` so pytest does not collect
their module bodies directly — each runs standalone and exits non-zero on failure, which
is also how they are convenient to run by hand while iterating:

    python tests/check_subcolor.py

Two of the checks are golden-image style: they re-render figures and compare every point
against a stored baseline. Regenerate a baseline only when a change is *meant* to move
points, and say so in the commit:

    python tests/capture_sina.py tests/sina_baseline.json
    python tests/capture_export.py          # writes tests/export_baseline.json

capture_export.py takes its output directory as argv[1] and chdir()s into it, so a
relative path there resolves twice; with no arguments it defaults to this directory.
"""
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


@pytest.mark.parametrize("script", ["check_subcolor.py", "check_interleave.py",
                                    "check_fragment_order.py", "check_sina_scope.py",
                                    "check_encoding_row.py", "check_opacity_ramp.py"])
def test_check_script(script):
    result = subprocess.run([sys.executable, str(HERE / script)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def _load(name):
    return json.loads((HERE / name).read_text())


def test_sina_points_have_not_moved():
    """The sina jitter comes from a KDE fitted per (separate section, colour group) with
    a reseeded rng; any change to the drawing loop must leave every point exactly where
    it was, or the cluster silhouette changes for reasons the reader cannot see.

    check_sina_scope.py asserts the *rule* (no channel may move a point); this asserts
    the *numbers*, so a change to the KDE itself is caught too."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    import capture_sina

    baseline, now = _load("sina_baseline.json"), capture_sina.snapshot()
    for case, expected in baseline.items():
        actual = now[case]
        moved = [c for c in expected["points"] if actual["points"].get(c) != expected["points"][c]]
        assert not moved, f"{case}: {len(moved)} points moved, e.g. {moved[:3]}"
        assert expected["legend"] == actual["legend"], f"{case}: legend changed"


def test_exported_figure_draws_the_same_pixels():
    """scatter_with_encodings is extracted into every method builder, so a change to it
    must leave each point's position, colour, alpha and marker untouched; only the
    grouping of the draw calls may differ."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    import capture_export

    baseline, now = _load("export_baseline.json"), capture_export.snapshot()
    for case, expected in baseline.items():
        if case == "_direct":
            for sub, exp in expected.items():
                assert exp["points"] == now[case][sub]["points"], f"direct/{sub} differs"
        else:
            assert expected["points"] == now[case]["points"], f"{case} differs"
