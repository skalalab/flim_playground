"""Pytest entry points for standalone visual-encoding checks.
check_*.py scripts exit nonzero on failure and can also run directly, for example:

    python tests/check_subcolor.py

Regenerate point baselines only when a change intentionally moves points:

    python tests/capture_sina.py tests/sina_baseline.json
    python tests/capture_export.py

capture_export.py defaults to this directory. If supplying its output directory,
use an absolute path because it changes into that directory.
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
    """Compare sina coordinates with the baseline to catch changes to KDE or jitter.
    check_sina_scope.py separately checks that encoding choices cannot move points.
    """
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
