"""Opening analysis and running PCA should not initialize unused analysis engines."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("action", ["page", "pca"])
def test_analysis_starts_without_unused_engines(tmp_path, action):
    # A new process is essential: other tests may already have imported the engines.
    script = textwrap.dedent("""
        import sys
        from pathlib import Path

        from src import config
        from src.widgets import analysis_config_widgets

        config._CONFIG_PATH = Path(sys.argv[2]) / "config.toml"
        analysis_config_widgets._ANALYSIS_CONFIG_PATH = (
            Path(sys.argv[2]) / "analysis_config.toml")

        if sys.argv[1] == "page":
            from streamlit.testing.v1 import AppTest

            app = AppTest.from_file("pages/data_analysis.py").run(timeout=60)
            assert not app.exception, [e.value for e in app.exception]
            assert len(app.radio) == 2
            assert len(app.get("file_uploader")) == 1
        else:
            import numpy as np
            import pandas as pd
            from src.vis.multivar import dimension_reduction

            frame = pd.DataFrame({"x": [1., 2., 4., 8.], "y": [3., 1., 5., 2.]})
            coordinates, variance = dimension_reduction(frame, method="PCA")
            assert coordinates.shape == (4, 2)
            assert np.isfinite(coordinates.to_numpy()).all()
            assert np.isclose(variance.sum(), 100)

        engines = ("umap", "pynndescent", "numba", "src.classify",
                   "imblearn", "matplotlib.pyplot")
        loaded = [name for name in engines if name in sys.modules]
        assert not loaded, f"Unused analysis engines loaded: {loaded}"
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, action, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba"),
             "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
